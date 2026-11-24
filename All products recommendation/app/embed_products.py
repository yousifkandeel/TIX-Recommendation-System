"""
Embed the Shopify product catalogue with bge-m3 running fully on the GPU
(PyTorch + transformers, fp16). Images are NOT embedded.

Why this is much faster than the Ollama version:
  - the whole model lives on the GPU in fp16 (no CPU offload, no HTTP round trips)
  - texts are tokenised once and SORTED BY LENGTH, so padding is almost zero
  - batches are built from a token budget (many short texts or a few long ones)
  - SDPA / flash attention kernels, inference_mode, no gradients
  - duplicate texts are embedded only once

Output (inside the project folder):
  data/embeddings/parts/<parquet_stem>/   per-file results (meta.parquet + <field>.npy)
  data/embeddings/products_meta.parquet   merged metadata (one row per product)
  data/embeddings/emb_<field>.npy         merged embeddings, float32, L2-normalised

Row i of every emb_<field>.npy corresponds to row i of products_meta.parquet,
so cosine similarity == dot product.

Requirements:  pip install torch transformers pandas pyarrow numpy tqdm
               (torch must be a CUDA build)

Usage:
  python embed_products.py --limit 200   # quick test (writes to data/embeddings_test)
  python embed_products.py               # full run (resumable, safe to stop and re-run)
  python embed_products.py --fields combined title   # only some fields (faster)
  python embed_products.py --token-budget 65536      # bigger batches if you have lots of VRAM
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

BASE = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = "BAAI/bge-m3"
MAX_LEN = 1024            # max tokens per text (3000 chars of description ~ 750 tokens)
TOKEN_BUDGET = 32768      # max (batch_size * padded_length) per forward pass
MAX_BATCH = 512           # hard cap on batch size for very short texts
MAX_DESC_CHARS = 3000     # descriptions longer than this are cut before embedding

# Which embeddings to create. "combined" is the main one for vector search.
# Remove "description" if you want to save time: "combined" already contains it.
FIELDS = ["combined", "title", "description", "category", "brand"]

PARQUET_GLOB = os.path.join(BASE, "data", "product-catalogue-raw", "data", "*.parquet")
TEXT_COLS = [
    "product_title",
    "product_description",
    "potential_product_categories",
    "ground_truth_brand",
    "ground_truth_is_secondhand",
    "ground_truth_category",
]

DEVICE = "cuda"


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def load_model():
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU not available. Install a CUDA build of PyTorch "
            "(https://pytorch.org/get-started/locally/) and check `nvidia-smi`."
        )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    try:
        model = AutoModel.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, attn_implementation="sdpa"
        )
    except Exception:  # older transformers without SDPA for this arch
        model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model = model.to(DEVICE).eval()
    return tokenizer, model


@torch.inference_mode()
def _forward(model, input_ids, attention_mask):
    out = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    cls = out[:, 0].float()                       # bge-m3 dense embedding = [CLS]
    return F.normalize(cls, p=2, dim=1)           # L2-normalise in fp32


def embed_texts(texts, tokenizer, model, desc, token_budget):
    """Embed a list of strings. Returns L2-normalised float32 array in the input order."""
    n = len(texts)
    if n == 0:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)

    texts = [t if t.strip() else "none" for t in texts]
    ids = tokenizer(
        texts, truncation=True, max_length=MAX_LEN, padding=False,
        return_attention_mask=False,
    )["input_ids"]
    lengths = np.fromiter((len(x) for x in ids), dtype=np.int64, count=n)
    order = np.argsort(-lengths, kind="stable")   # longest first -> OOM shows up early

    result = np.empty((n, model.config.hidden_size), dtype=np.float32)
    pad_id = tokenizer.pad_token_id
    pbar = tqdm(total=n, desc=desc, unit="text")

    pos = 0
    budget = token_budget
    while pos < n:
        max_len = int(lengths[order[pos]])
        bs = max(1, min(MAX_BATCH, budget // max_len, n - pos))
        idx = order[pos : pos + bs]

        batch_ids = torch.full((bs, max_len), pad_id, dtype=torch.long)
        batch_mask = torch.zeros((bs, max_len), dtype=torch.long)
        for row, i in enumerate(idx):
            seq = ids[i]
            batch_ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            batch_mask[row, : len(seq)] = 1

        try:
            vec = _forward(
                model,
                batch_ids.to(DEVICE, non_blocking=True),
                batch_mask.to(DEVICE, non_blocking=True),
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if budget <= max_len:
                raise
            budget = max(max_len, budget // 2)
            tqdm.write(f"CUDA OOM -> token budget reduced to {budget}")
            continue

        result[idx] = vec.cpu().numpy()
        pos += bs
        pbar.update(bs)

    pbar.close()
    return result


def embed_unique(values, tokenizer, model, desc, token_budget):
    """Embed only the unique values, then expand back to one vector per row."""
    codes, uniques = pd.factorize(pd.Series(values))
    vectors = embed_texts([str(u) for u in uniques], tokenizer, model, desc, token_budget)
    return vectors[codes]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_meta(path, limit):
    """Build the metadata table (everything except the image bytes)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    df = pd.read_parquet(path, columns=TEXT_COLS)
    if limit:
        df = df.head(limit)
    df = df.reset_index(drop=True)

    category = df["ground_truth_category"].fillna("").astype(str)
    levels = category.str.split(" > ")
    return pd.DataFrame(
        {
            # Stable id: stays valid even if you download more parquet files later
            "product_id": [f"{stem}_{i}" for i in range(len(df))],
            # Use these two to fetch the image from the parquet file when needed
            "source_file": os.path.basename(path),
            "source_row": range(len(df)),
            "title": df["product_title"].fillna("").astype(str),
            "description": df["product_description"].fillna("").astype(str),
            "brand": df["ground_truth_brand"].fillna("").astype(str),
            "category": category,
            "category_level1": levels.str[0],
            "category_level2": levels.str[:2].str.join(" > "),
            "category_depth": levels.str.len(),
            "is_secondhand": df["ground_truth_is_secondhand"].astype(bool),
            "potential_categories": df["potential_product_categories"].apply(list),
        }
    )


def build_texts(meta):
    """The text that is embedded for each field."""
    desc = meta["description"].str.slice(0, MAX_DESC_CHARS)
    combined = (
        "Title: " + meta["title"]
        + "\nBrand: " + meta["brand"]
        + "\nCategory: " + meta["category"]
        + "\nDescription: " + desc
    )
    return {
        "combined": combined.tolist(),
        "title": meta["title"].tolist(),
        "description": desc.tolist(),
        "category": meta["category"].tolist(),
        "brand": meta["brand"].tolist(),
    }


def process_file(path, out_root, limit, fields, tokenizer, model, token_budget):
    stem = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(out_root, "parts", stem)
    os.makedirs(out_dir, exist_ok=True)

    meta = load_meta(path, limit)
    meta.to_parquet(os.path.join(out_dir, "meta.parquet"), index=False)
    texts = build_texts(meta)

    for field in fields:
        out_path = os.path.join(out_dir, f"{field}.npy")
        if os.path.exists(out_path):
            print(f"[skip] {stem} / {field} already done")
            continue
        desc = f"{stem[-14:]} {field}"
        # Deduplicating is cheap and helps every field (many repeated titles/brands/categories)
        vectors = embed_unique(texts[field], tokenizer, model, desc, token_budget)
        tmp_path = out_path + ".tmp.npy"
        np.save(tmp_path, vectors)
        os.replace(tmp_path, out_path)  # file only appears when fully written


def merge(out_root, fields):
    parts = sorted(glob.glob(os.path.join(out_root, "parts", "*")))
    meta = pd.concat(
        [pd.read_parquet(os.path.join(p, "meta.parquet")) for p in parts],
        ignore_index=True,
    )
    meta.to_parquet(os.path.join(out_root, "products_meta.parquet"), index=False)
    for field in fields:
        arr = np.concatenate([np.load(os.path.join(p, f"{field}.npy")) for p in parts])
        assert len(arr) == len(meta), f"{field}: {len(arr)} vectors vs {len(meta)} rows"
        np.save(os.path.join(out_root, f"emb_{field}.npy"), arr)
        print(f"Saved emb_{field}.npy  shape={arr.shape}")
    print(f"Saved products_meta.parquet  rows={len(meta)}")


def demo(out_root):
    """Sanity check: find the 5 most similar products to the first product."""
    meta = pd.read_parquet(os.path.join(out_root, "products_meta.parquet"))
    emb = np.load(os.path.join(out_root, "emb_combined.npy"))
    scores = emb @ emb[0]
    top = np.argsort(-scores)[:6]
    print(f"\nQuery product: {meta.loc[0, 'title']}  [{meta.loc[0, 'category']}]")
    for rank, idx in enumerate(top[1:], start=1):
        print(f"{rank}. {scores[idx]:.3f}  {meta.loc[idx, 'title']}  [{meta.loc[idx, 'category']}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="only first N rows per file (test run)")
    parser.add_argument("--fields", nargs="+", default=FIELDS, choices=FIELDS,
                        help="which embeddings to build (default: all)")
    parser.add_argument("--token-budget", type=int, default=TOKEN_BUDGET,
                        help="max tokens per GPU batch (raise it if VRAM allows)")
    args = parser.parse_args()

    out_root = os.path.join(BASE, "data", "embeddings_test" if args.limit else "embeddings")

    tokenizer, model = load_model()
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  model: {MODEL_NAME} (fp16)  |  "
          f"embedding dimension: {model.config.hidden_size}")

    files = sorted(glob.glob(PARQUET_GLOB))
    if not files:
        raise SystemExit(f"No parquet files found: {PARQUET_GLOB}")
    print(f"Parquet files: {len(files)}  |  fields: {args.fields}  |  output: {out_root}")

    for path in files:
        process_file(path, out_root, args.limit, args.fields, tokenizer, model, args.token_budget)

    merge(out_root, args.fields)
    if "combined" in args.fields:
        demo(out_root)


if __name__ == "__main__":
    main()

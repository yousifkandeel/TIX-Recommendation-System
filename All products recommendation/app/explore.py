import os

BASE = os.path.dirname(os.path.abspath(__file__))

# Keep all caches and temp files inside the project folder
os.environ["HF_HOME"] = os.path.join(BASE, "data", "hf_home")
os.environ["HF_DATASETS_CACHE"] = os.path.join(BASE, "data", "hf_datasets_cache")
os.environ["HF_HUB_CACHE"] = os.path.join(BASE, "data", "hf_hub_cache")
os.environ["TMP"] = os.path.join(BASE, "data", "tmp")
os.environ["TEMP"] = os.environ["TMP"]
os.makedirs(os.environ["TMP"], exist_ok=True)

import glob
import io

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

files = sorted(glob.glob(os.path.join(BASE, "data", "product-catalogue-raw", "data", "*.parquet")))
print(f"Number of files: {len(files)}")

# 1) Schema and row count without loading the data
total = 0
for f in files:
    pf = pq.ParquetFile(f)
    total += pf.metadata.num_rows
print("Total rows:", total)
print("\nColumns:")
print(pq.ParquetFile(files[0]).schema_arrow)

# 2) Identify image columns (structs containing bytes) and text columns
schema = pq.ParquetFile(files[0]).schema_arrow
img_cols = [f.name for f in schema if str(f.type).startswith("struct<bytes")]
text_cols = [n for n in schema.names if n not in img_cols]
print("\nImage columns:", img_cols)
print("Text columns:", text_cols)

# 3) Load only the text columns (very lightweight)
df = pd.concat(
    [pd.read_parquet(f, columns=text_cols) for f in files],
    ignore_index=True,
)
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isna().sum())

# Most frequent values for columns with a limited number of unique values
for c in df.columns:
    try:
        n = df[c].nunique()
    except TypeError:
        continue  # skip list-type columns
    if 1 < n <= 50:
        print(f"\n{c} ({n} unique values):")
        print(df[c].value_counts().head(10))

# 4) Save 10 preview images from the first file only
if img_cols:
    preview_dir = os.path.join(BASE, "preview")
    os.makedirs(preview_dir, exist_ok=True)
    pf = pq.ParquetFile(files[0])
    batch = next(pf.iter_batches(batch_size=10, columns=img_cols)).to_pandas()
    for col in img_cols:
        for i, cell in batch[col].items():
            img = Image.open(io.BytesIO(cell["bytes"])).convert("RGB")
            img.save(os.path.join(preview_dir, f"{col}_{i}.jpg"))
    print("\nImages saved in the preview folder")
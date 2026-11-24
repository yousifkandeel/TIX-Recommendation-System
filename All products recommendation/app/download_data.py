from huggingface_hub import snapshot_download
from datasets import load_dataset

path = snapshot_download(
    repo_id="Shopify/product-catalogue",
    repo_type="dataset",
    local_dir="./data/product-catalogue-raw",
    max_workers=2,   # اتصالات أقل = قطع أقل
)
print("نزلت في:", path)

ds = load_dataset("parquet", data_dir=f"{path}/data")
print(ds)
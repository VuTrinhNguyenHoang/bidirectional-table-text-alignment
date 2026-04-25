import os
from typing import Dict, List

import requests
from datasets import DatasetDict, load_dataset

def get_totto_parquet_files() -> Dict[str, List[str]]:
    api_url = "https://datasets-server.huggingface.co/parquet?dataset=GEM/totto"
    response = requests.get(api_url, timeout=60)
    response.raise_for_status()

    data = response.json()
    parquet_files = data["parquet_files"]

    data_files = {
        "train": [],
        "validation": [],
        "test": []
    }

    for item in parquet_files:
        if item.get("config") != "totto":
            continue

        split = item["split"]
        if split in data_files:
            data_files[split].append(item["url"])

    return {k: v for k, v in data_files.items() if len(v) > 0}

def load_totto_from_parquet() -> DatasetDict:
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    data_files = get_totto_parquet_files()

    print("Resolved ToTTo parquet files:")
    for split, urls in data_files.items():
        print(f"- {split}: {len(urls)} file(s)")
        print(f"  first: {urls[0]}")

    dataset = load_dataset("parquet", data_files=data_files)
    return dataset

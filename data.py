# data.py — Loads AG News from CSV files and prepares PyTorch datasets.
# Downloads train.csv / test.csv on first run, then caches them locally.
# No HuggingFace `datasets` library — uses pandas + urllib only.

# pandas and sklearn must be imported before torch — PyArrow (a pandas dependency)
# causes an access violation on Windows if loaded after CUDA DLLs are already mapped.
import os
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
MAX_LENGTH  = 256          # tokens; 256 covers virtually all AG News headlines+descriptions
DATA_DIR    = "./data"     # local folder where CSVs are stored after download

# AG News CSV hosted on a public GitHub mirror of the Zhang et al. benchmark.
# Format: no header, 3 columns — class_index (1-4), title, description
_URLS = {
    "train": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
    "test":  "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
}

# Lazy tokenizer singleton — created on first call to avoid import-time
# segfaults caused by initialising the Rust tokenizer while other native
# libs are already loaded.
_TOKENIZER: DistilBertTokenizerFast | None = None


def get_tokenizer() -> DistilBertTokenizerFast:
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    return _TOKENIZER


# ------------------------------------------------------------------ #
# Download helpers
# ------------------------------------------------------------------ #
def _download_file(url: str, dest: str) -> None:
    """Download `url` to `dest`, printing a simple progress indicator."""
    print(f"  Downloading {os.path.basename(dest)} ...", end=" ", flush=True)

    def _reporthook(count, block_size, total_size):
        if total_size > 0 and count % 100 == 0:
            pct = min(100, count * block_size * 100 // total_size)
            print(f"\r  Downloading {os.path.basename(dest)} ... {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_reporthook)
    print(" done.")


def download_ag_news(data_dir: str = DATA_DIR) -> tuple[str, str]:
    """
    Ensure train.csv and test.csv exist under `data_dir`, downloading if needed.
    Returns (train_path, test_path).
    """
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "train.csv")
    test_path  = os.path.join(data_dir, "test.csv")

    if not os.path.exists(train_path):
        _download_file(_URLS["train"], train_path)
    else:
        print(f"  Found cached {train_path}")

    if not os.path.exists(test_path):
        _download_file(_URLS["test"], test_path)
    else:
        print(f"  Found cached {test_path}")

    return train_path, test_path


# ------------------------------------------------------------------ #
# CSV loading
# ------------------------------------------------------------------ #
def _load_csv(path: str) -> pd.DataFrame:
    """
    Read an AG News CSV file into a DataFrame with columns [label, text].

    The raw CSV has no header and three columns:
        col 0 — class index, 1-indexed  (1=World, 2=Sports, 3=Business, 4=Sci/Tech)
        col 1 — article title
        col 2 — article description
    We combine title + description into a single 'text' column and convert
    the 1-indexed label to 0-indexed to match PyTorch convention.
    """
    df = pd.read_csv(path, header=None, names=["label", "title", "description"])

    # Combine title and description — gives the model more signal than title alone
    df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")
    df["text"] = df["text"].str.strip()

    # Convert 1-indexed classes (1–4) to 0-indexed (0–3)
    df["label"] = df["label"].astype(int) - 1

    return df[["label", "text"]]


# ------------------------------------------------------------------ #
# PyTorch Dataset
# ------------------------------------------------------------------ #
class AGNewsDataset(Dataset):
    """PyTorch Dataset wrapping tokenized AG News examples."""

    def __init__(self, encodings: dict, labels: list):
        # encodings: plain dict with 'input_ids' and 'attention_mask' (lists of lists)
        # labels:    list/array of integer class indices (0–3)
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Build a dict of tensors that DataLoader and the model both understand
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #
def load_and_split_data(
    val_size: float = 0.1,
    seed: int = 42,
    data_dir: str = DATA_DIR,
) -> tuple["AGNewsDataset", "AGNewsDataset", "AGNewsDataset"]:
    """
    Download (if needed), load, tokenise, and split AG News into
    train / val / test PyTorch Datasets.

    The official train split (120 000 examples) is further divided:
        train  — 90 % of official train  (108 000)
        val    — 10 % of official train  ( 12 000)
        test   — official test set        (  7 600)
    """
    print("Checking for AG News CSV files...")
    train_path, test_path = download_ag_news(data_dir)

    print("Loading CSVs with pandas...")
    train_val_df = _load_csv(train_path)
    test_df      = _load_csv(test_path)

    # Carve out validation set from the training data
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=seed,
        stratify=train_val_df["label"],  # keep class distribution balanced
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)

    print(
        f"Split sizes — train: {len(train_df)}, "
        f"val: {len(val_df)}, test: {len(test_df)}"
    )

    # Tokenise all three splits
    print("Tokenising splits (this may take a minute)...")
    tokenizer = get_tokenizer()

    def tokenize(texts: list[str]) -> dict:
        enc = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        # Return a plain dict so AGNewsDataset can index into it by position
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    train_enc = tokenize(train_df["text"].tolist())
    val_enc   = tokenize(val_df["text"].tolist())
    test_enc  = tokenize(test_df["text"].tolist())

    train_dataset = AGNewsDataset(train_enc, train_df["label"].tolist())
    val_dataset   = AGNewsDataset(val_enc,   val_df["label"].tolist())
    test_dataset  = AGNewsDataset(test_enc,  test_df["label"].tolist())

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(
    batch_size: int = 16,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return DataLoaders for train / val / test splits."""
    train_ds, val_ds, test_ds = load_and_split_data()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    return train_loader, val_loader, test_loader


# ------------------------------------------------------------------ #
# Quick sanity check
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    train_ds, val_ds, test_ds = load_and_split_data()
    sample = train_ds[0]
    print("\nSample keys      :", list(sample.keys()))
    print("input_ids shape  :", sample["input_ids"].shape)
    label_idx = sample["labels"].item()
    print(f"Label            : {label_idx} -> {LABEL_NAMES[label_idx]}")

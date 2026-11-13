import re
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

stop_words_set = set(ENGLISH_STOP_WORDS)


def clean_text(text: str) -> str:
    """
    Standard text cleaning matching EDA_Preprocessing.ipynb:
    - Lowercase conversion
    - Regex strip of non-alphabetic characters and punctuation
    - Whitespace tokenization and English stop word filtering
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words_set]
    return " ".join(words)


class BlogDataset(Dataset):
    """
    PyTorch Dataset wrapper for Blog text samples and category labels.
    """

    def __init__(
        self,
        encodings: Dict[str, torch.Tensor],
        labels: Optional[List[int]] = None,
    ):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_raw_or_cleaned_data(
    data_path: str = "blogs.csv",
    text_column: str = "Data",
    label_column: str = "Labels",
) -> pd.DataFrame:
    """
    Loads CSV data and verifies column existence.
    If text_column is not present but 'Data' is, copies the raw text column.
    Model-specific preprocessing is handled by the calling pipeline.
    """
    df = pd.read_csv(data_path)
    if text_column not in df.columns:
        if "Data" in df.columns:
            df[text_column] = df["Data"]
        else:
            raise ValueError(f"Column '{text_column}' not found in {data_path}. Available: {df.columns.tolist()}")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in {data_path}.")
    return df


def create_stratified_splits(
    df: pd.DataFrame,
    text_column: str = "Data",
    label_column: str = "Labels",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int], Dict[int, str]]:
    """
    Performs stratified 3-way split (train/validation/test) with fixed random seed.
    Returns (train_df, val_df, test_df, label2id, id2label).
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    # Sorted unique labels for deterministic index assignment
    unique_labels = sorted(df[label_column].unique())
    label2id = {lbl: i for i, lbl in enumerate(unique_labels)}
    id2label = {i: lbl for i, lbl in enumerate(unique_labels)}

    # Step 1: Split train and temp (val + test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=seed,
        stratify=df[label_column],
    )

    # Step 2: Split temp into val and test
    relative_test_ratio = test_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        random_state=seed,
        stratify=temp_df[label_column],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df, label2id, id2label


def tokenize_data(
    tokenizer: Any,
    texts: List[str],
    max_length: int = 128,
) -> Dict[str, torch.Tensor]:
    """Tokenizes texts with padding and truncation to PyTorch tensors."""
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def build_pytorch_datasets(
    tokenizer: Any,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_column: str,
    label_column: str,
    label2id: Dict[str, int],
    max_length: int = 128,
) -> Tuple[BlogDataset, BlogDataset, BlogDataset]:
    """Builds tokenized BlogDataset instances for train, val, and test."""
    train_enc = tokenize_data(tokenizer, train_df[text_column].tolist(), max_length=max_length)
    val_enc = tokenize_data(tokenizer, val_df[text_column].tolist(), max_length=max_length)
    test_enc = tokenize_data(tokenizer, test_df[text_column].tolist(), max_length=max_length)

    train_labels = [label2id[lbl] for lbl in train_df[label_column]]
    val_labels = [label2id[lbl] for lbl in val_df[label_column]]
    test_labels = [label2id[lbl] for lbl in test_df[label_column]]

    train_dataset = BlogDataset(train_enc, train_labels)
    val_dataset = BlogDataset(val_enc, val_labels)
    test_dataset = BlogDataset(test_enc, test_labels)

    return train_dataset, val_dataset, test_dataset


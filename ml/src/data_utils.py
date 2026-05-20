from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.src.preprocess import normalize_text


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    required = {"label", "message"}
    if not required.issubset(data.columns):
        raise ValueError("Dataset must contain 'label' and 'message' columns.")

    df = data.copy()
    df["message"] = df["message"].astype(str).map(normalize_text)
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[df["label"].isin(["spam", "ham"])].reset_index(drop=True)
    if df.empty:
        raise ValueError("Dataset has no valid rows after preprocessing.")
    return df


def split_dataset(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    return train_test_split(
        df["message"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )


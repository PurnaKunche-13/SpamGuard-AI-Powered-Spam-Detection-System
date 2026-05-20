from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml.src.data_utils import load_dataset, split_dataset
from ml.src.model import SpamClassifier


ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def train_model(data_path: str | Path, epochs: int = 20, batch_size: int = 8) -> dict[str, float | int]:
    df = load_dataset(data_path)
    x_train, x_test, y_train, y_test = split_dataset(df)

    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    x_train_vec = vectorizer.fit_transform(x_train).toarray()
    x_test_vec = vectorizer.transform(x_test).toarray()

    y_train_tensor = torch.tensor([1 if label == "spam" else 0 for label in y_train], dtype=torch.long)
    y_test_tensor = torch.tensor([1 if label == "spam" else 0 for label in y_test], dtype=torch.long)

    train_dataset = TensorDataset(torch.tensor(x_train_vec, dtype=torch.float32), y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = SpamClassifier(input_dim=x_train_vec.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for _ in range(epochs):
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(x_test_vec, dtype=torch.float32)
        logits = model(test_tensor)
        predictions = torch.argmax(logits, dim=1)

    accuracy = accuracy_score(y_test_tensor.numpy(), predictions.numpy())

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, ARTIFACT_DIR / "vectorizer.joblib")
    torch.save(model.state_dict(), ARTIFACT_DIR / "model.pt")

    return {"accuracy": round(float(accuracy), 4), "samples": int(len(df))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SpamGuard spam classifier")
    parser.add_argument("--data", required=True, help="Path to CSV dataset with label,message columns")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    metrics = train_model(args.data, epochs=args.epochs, batch_size=args.batch_size)
    print(f"Training complete | accuracy={metrics['accuracy']} | samples={metrics['samples']}")


if __name__ == "__main__":
    main()


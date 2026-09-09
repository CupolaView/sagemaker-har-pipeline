"""SageMaker-compatible training script — UCI HAR (raw inertial signals).

Local dry-run:
    python train.py --epochs 1 --train-dir ./data/train --test-dir ./data/test --model-dir ./model_out

On SageMaker the estimator passes hyperparameters as CLI args and the
defaults below pick up the SM_* environment variables automatically.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import HARConvNet

# UCI HAR raw signals: 9 channels x 128 timesteps, 6 activity classes
SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


def load_har_split(split_dir: Path, split: str):
    """Load a data split. Prefers preprocessed .npy (output of preprocess.py);
    falls back to raw UCI txt so local dry-runs against the raw download
    still work. Labels arrive 0-5 in npy, 1-6 in raw."""
    npy = split_dir / "X.npy"
    if npy.exists():
        X = np.load(npy).astype(np.float32)
        y = np.load(split_dir / "y.npy").astype(np.int64)
        print(f"loaded preprocessed npy from {split_dir}")
    else:
        sig_dir = split_dir / "Inertial Signals"
        X = np.stack(
            [np.loadtxt(sig_dir / f"{s}_{split}.txt") for s in SIGNALS], axis=1
        ).astype(np.float32)
        y = np.loadtxt(split_dir / f"y_{split}.txt").astype(np.int64) - 1
        print(f"loaded raw txt from {split_dir}")
    return torch.from_numpy(X), torch.from_numpy(y)


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += yb.numel()
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    # --- hyperparameters (SageMaker passes these as CLI args) ---
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    # --- SageMaker directory contract (env vars exist in the container;
    #     the CLI flags let you override them for the local dry-run) ---
    parser.add_argument("--train-dir", type=str,
                        default=os.environ.get("SM_CHANNEL_TRAIN", "data/train"))
    parser.add_argument("--test-dir", type=str,
                        default=os.environ.get("SM_CHANNEL_TEST", "data/test"))
    parser.add_argument("--model-dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR", "model_out"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    X_train, y_train = load_har_split(Path(args.train_dir), "train")
    X_test, y_test = load_har_split(Path(args.test_dir), "test")
    print(f"train={tuple(X_train.shape)} test={tuple(X_test.shape)}")

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test),
                             batch_size=args.batch_size)

    model = HARConvNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * yb.numel()
        acc = evaluate(model, test_loader, device)
        # NOTE: print metrics in a stable format — SageMaker metric_definitions
        # regex-scrapes stdout, and this feeds CloudWatch later in Phase 1.
        print(f"epoch={epoch + 1} train_loss={running / len(y_train):.4f} "
              f"test_acc={acc:.4f}")

    # --- save to the model dir; SageMaker tars this to s3://.../model.tar.gz ---
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "model.pt")
    (model_dir / "metrics.json").write_text(
        json.dumps({"test_accuracy": evaluate(model, test_loader, device)}))
    print(f"saved model to {model_dir}")


if __name__ == "__main__":
    main()

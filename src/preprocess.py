"""SageMaker Processing script — UCI HAR raw signals -> normalized .npy.

Processing contract (different from training!):
  - inputs are downloaded to paths you choose under /opt/ml/processing/...
  - outputs are uploaded FROM paths you choose under /opt/ml/processing/...
  - no SM_CHANNEL_* env vars, no SM_MODEL_DIR — you define paths in the
    ProcessingInput/ProcessingOutput objects at launch time and mirror
    them here as argparse defaults.

Local dry-run:
    python preprocess.py \
        --input-train "data/UCI HAR Dataset/train" \
        --input-test  "data/UCI HAR Dataset/test" \
        --output-train ./processed/train \
        --output-test  ./processed/test
"""
import argparse
import json
from pathlib import Path

import numpy as np

SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


def load_split(split_dir: Path, split: str):
    sig_dir = split_dir / "Inertial Signals"
    X = np.stack(
        [np.loadtxt(sig_dir / f"{s}_{split}.txt") for s in SIGNALS], axis=1
    ).astype(np.float32)                                   # (N, 9, 128)
    y = np.loadtxt(split_dir / f"y_{split}.txt").astype(np.int64) - 1
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-train", default="/opt/ml/processing/input/train")
    parser.add_argument("--input-test", default="/opt/ml/processing/input/test")
    parser.add_argument("--output-train", default="/opt/ml/processing/output/train")
    parser.add_argument("--output-test", default="/opt/ml/processing/output/test")
    args = parser.parse_args()

    X_train, y_train = load_split(Path(args.input_train), "train")
    X_test, y_test = load_split(Path(args.input_test), "test")

    # Per-channel z-score normalization.
    # Stats computed on TRAIN ONLY, applied to both splits — computing them
    # on test would leak distribution info (interview-grade detail).
    mean = X_train.mean(axis=(0, 2), keepdims=True)        # (1, 9, 1)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    for out_dir, X, y in [
        (Path(args.output_train), X_train, y_train),
        (Path(args.output_test), X_test, y_test),
    ]:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "X.npy", X)
        np.save(out_dir / "y.npy", y)

    # Persist the normalization stats WITH the data: inference must apply
    # the same transform, and Model Monitor will want these as a baseline.
    stats = {
        "mean": mean.squeeze().tolist(),
        "std": std.squeeze().tolist(),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "class_counts_train": np.bincount(y_train, minlength=6).tolist(),
    }
    stats_path = Path(args.output_train) / "preprocess_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))

    print(f"train={X_train.shape} test={X_test.shape} -> outputs written")


if __name__ == "__main__":
    main()

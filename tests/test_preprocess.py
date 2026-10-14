"""Tests for the processing step (preprocess.py), run on a tiny fake dataset
laid out exactly like UCI HAR, so no real data or AWS access is needed."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


def _fake_split(root: Path, split: str, n: int, shift: float):
    sig_dir = root / split / "Inertial Signals"
    sig_dir.mkdir(parents=True)
    rng = np.random.default_rng(len(split))
    for s in SIGNALS:
        np.savetxt(sig_dir / f"{s}_{split}.txt", rng.normal(shift, 1.0, size=(n, 128)))
    np.savetxt(root / split / f"y_{split}.txt", rng.integers(1, 7, size=n), fmt="%d")


def test_preprocess_end_to_end(tmp_path):
    _fake_split(tmp_path, "train", n=20, shift=0.0)
    _fake_split(tmp_path, "test", n=8, shift=5.0)   # deliberately different
    out = tmp_path / "out"

    subprocess.run(
        [sys.executable, str(SRC / "preprocess.py"),
         "--input-train", str(tmp_path / "train"),
         "--input-test", str(tmp_path / "test"),
         "--output-train", str(out / "train"),
         "--output-test", str(out / "test")],
        check=True,
    )

    X_train = np.load(out / "train" / "X.npy")
    X_test = np.load(out / "test" / "X.npy")
    y_train = np.load(out / "train" / "y.npy")
    assert X_train.shape == (20, 9, 128) and X_test.shape == (8, 9, 128)
    assert y_train.min() >= 0 and y_train.max() <= 5   # labels shifted 1-6 -> 0-5

    # Normalization stats come from TRAIN only: train is centered, while the
    # shifted test set is NOT re-centered. This guards against leakage.
    assert np.allclose(X_train.mean(axis=(0, 2)), 0, atol=1e-5)
    assert X_test.mean() > 3

    stats = json.loads((out / "train" / "preprocess_stats.json").read_text())
    assert stats["n_train"] == 20 and len(stats["mean"]) == 9

"""Model Monitor record preprocessor + shared feature definition.

Two roles:
1. Model Monitor uploads this file and calls preprocess_handler() on every
   captured request, turning a raw (9, 128) window into 18 named features:
   the per-channel mean and std. The built-in data-quality analyzer then
   checks those features against the baseline.
2. monitor_baseline.py imports window_features() to compute the SAME
   features on the training set. Because both use one function, the
   baseline and live monitoring can't drift apart in definition.

Pure Python on purpose: the monitoring container isn't guaranteed to
have numpy installed.
"""
import base64
import json
import math

SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]
FEATURE_NAMES = [f"{s}_{stat}" for s in SIGNALS for stat in ("mean", "std")]


def window_features(window):
    """window: 9 channels x 128 floats -> dict of 18 features (ordered)."""
    feats = {}
    for name, channel in zip(SIGNALS, window):
        n = len(channel)
        mean = sum(channel) / n
        var = sum((v - mean) ** 2 for v in channel) / n   # population std
        feats[f"{name}_mean"] = mean
        feats[f"{name}_std"] = math.sqrt(var)
    return feats


def preprocess_handler(inference_record, logger=None):
    """Called by Model Monitor once per captured request.

    Returns a list of feature dicts, one per window in the request. If the
    analyzer rejects list returns, fall back to returning a single dict
    for the first window.
    """
    data = inference_record.endpoint_input.data
    if inference_record.endpoint_input.encoding.upper() == "BASE64":
        data = base64.b64decode(data).decode("utf-8")
    try:
        instances = json.loads(data)["instances"]
    except (ValueError, KeyError, TypeError):
        return []  # malformed request (e.g. the smoke test's bad payload)

    out = []
    for window in instances:
        if len(window) == 9 and all(len(ch) == 128 for ch in window):
            out.append(window_features(window))
    return out

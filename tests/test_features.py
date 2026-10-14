"""Tests for the monitoring feature definition (monitor_preprocessor.py).

window_features() defines what "normal" means for drift detection, for both
the training reference and live traffic. If it silently changed, drift
results would be meaningless, so it gets precise tests.
"""
import math
import types

import numpy as np

from monitor_preprocessor import FEATURE_NAMES, preprocess_handler, window_features


def test_feature_names_are_18_and_ordered():
    assert len(FEATURE_NAMES) == 18
    assert FEATURE_NAMES[:2] == ["body_acc_x_mean", "body_acc_x_std"]


def test_window_features_match_numpy():
    rng = np.random.default_rng(0)
    window = rng.normal(size=(9, 128))
    feats = window_features(window.tolist())
    assert list(feats) == FEATURE_NAMES
    for i, sig in enumerate(FEATURE_NAMES[::2]):
        base = sig[: -len("_mean")]   # (str.removesuffix needs Python 3.9+)
        assert math.isclose(feats[f"{base}_mean"], window[i].mean(), abs_tol=1e-9)
        assert math.isclose(feats[f"{base}_std"], window[i].std(), abs_tol=1e-9)  # population std


def test_offset_and_gain_move_mean_and_std():
    """The sensor-fault model the monitor is built on: offset moves the mean,
    gain scales the std."""
    window = np.random.default_rng(1).normal(size=(9, 128))
    base = window_features(window.tolist())
    drifted = window_features((window * 2.5 + 1.0).tolist())
    assert math.isclose(drifted["body_acc_x_mean"], base["body_acc_x_mean"] * 2.5 + 1.0, abs_tol=1e-9)
    assert math.isclose(drifted["body_acc_x_std"], base["body_acc_x_std"] * 2.5, abs_tol=1e-9)


def _record(payload: str):
    endpoint_input = types.SimpleNamespace(data=payload, encoding="JSON")
    return types.SimpleNamespace(endpoint_input=endpoint_input)


def test_preprocess_handler_parses_valid_request():
    window = np.zeros((9, 128)).tolist()
    out = preprocess_handler(_record('{"instances": %s}' % [window, window]))
    assert len(out) == 2 and set(out[0]) == set(FEATURE_NAMES)


def test_preprocess_handler_skips_malformed_request():
    assert preprocess_handler(_record('{"instances": [[1, 2, 3]]}')) == []
    assert preprocess_handler(_record("not json")) == []

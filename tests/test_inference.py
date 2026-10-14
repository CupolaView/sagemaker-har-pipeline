"""Tests for the serving contract (inference.py) and the shared model.

These run the four SageMaker handler functions locally, the way the serving
container would, without deploying anything.
"""
import json

import pytest
import torch

import inference
from model import ACTIVITY_LABELS, HARConvNet


@pytest.fixture
def model_dir(tmp_path):
    torch.save(HARConvNet().state_dict(), tmp_path / "model.pt")
    return str(tmp_path)


def test_model_output_shape():
    out = HARConvNet()(torch.zeros(4, 9, 128))
    assert out.shape == (4, 6)


def test_full_request_round_trip(model_dir):
    model = inference.model_fn(model_dir)
    body = json.dumps({"instances": torch.zeros(3, 9, 128).tolist()})
    x = inference.input_fn(body, "application/json")
    probs = inference.predict_fn(x, model)
    response = json.loads(inference.output_fn(probs, "application/json"))

    preds = response["predictions"]
    assert len(preds) == 3
    for p in preds:
        assert p["label"] in ACTIVITY_LABELS
        assert p["label"] == ACTIVITY_LABELS[p["class_id"]]
        assert abs(sum(p["probabilities"]) - 1.0) < 1e-5


def test_input_fn_rejects_wrong_shape():
    with pytest.raises(ValueError, match="expected shape"):
        inference.input_fn(json.dumps({"instances": [[1, 2, 3]]}), "application/json")


def test_input_fn_rejects_wrong_content_type():
    with pytest.raises(ValueError, match="unsupported content type"):
        inference.input_fn("1,2,3", "text/csv")

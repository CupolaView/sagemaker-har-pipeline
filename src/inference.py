"""SageMaker inference handler — the serving-side contract.

The PyTorch serving container calls these four functions per request:
    model_fn   (once, at container start): model_dir -> model object
    input_fn   (per request): raw request body -> model input
    predict_fn (per request): model input + model -> raw prediction
    output_fn  (per request): raw prediction -> response body

Request format:
    {"instances": [[[...9x128 floats...]], ...]}   # (N, 9, 128)
Response format:
    {"predictions": [{"label": "WALKING", "class_id": 0,
                      "probabilities": [...6 floats...]}, ...]}
"""
import json

import torch
import torch.nn.functional as F

from model import ACTIVITY_LABELS, HARConvNet


def model_fn(model_dir):
    model = HARConvNet()
    state = torch.load(f"{model_dir}/model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def input_fn(request_body, request_content_type):
    if request_content_type != "application/json":
        raise ValueError(f"unsupported content type: {request_content_type}")
    payload = json.loads(request_body)
    x = torch.tensor(payload["instances"], dtype=torch.float32)
    if x.dim() != 3 or x.shape[1:] != (9, 128):
        raise ValueError(f"expected shape (N, 9, 128), got {tuple(x.shape)}")
    return x


def predict_fn(input_data, model):
    with torch.no_grad():
        logits = model(input_data)
        return F.softmax(logits, dim=1)


def output_fn(prediction, response_content_type):
    probs = prediction.tolist()
    out = [
        {
            "label": ACTIVITY_LABELS[int(max(range(6), key=p.__getitem__))],
            "class_id": int(max(range(6), key=p.__getitem__)),
            "probabilities": p,
        }
        for p in probs
    ]
    return json.dumps({"predictions": out})

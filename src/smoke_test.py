"""Smoke-test the live endpoint with held-out test windows.

Run:  python src/smoke_test.py            (uses ./processed/test by default)
      python src/smoke_test.py --n 25
"""
import argparse
import json
from pathlib import Path

import boto3
import numpy as np

ENDPOINT_NAME = "har-classifier"
REGION = "us-west-2"

parser = argparse.ArgumentParser()
parser.add_argument("--test-dir", default="processed/test")
parser.add_argument("--n", type=int, default=10)
args = parser.parse_args()

X = np.load(Path(args.test_dir) / "X.npy")
y = np.load(Path(args.test_dir) / "y.npy")
idx = np.random.default_rng(42).choice(len(X), size=args.n, replace=False)

runtime = boto3.client("sagemaker-runtime", region_name=REGION)
resp = runtime.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType="application/json",
    Body=json.dumps({"instances": X[idx].tolist()}),
)
preds = json.loads(resp["Body"].read())["predictions"]

correct = 0
for i, p in zip(idx, preds):
    ok = p["class_id"] == int(y[i])
    correct += ok
    print(f"sample {i}: pred={p['label']:<20} actual_class={int(y[i])} {'OK' if ok else 'MISS'}")

print(f"\n{correct}/{args.n} correct")

# malformed-payload check: the endpoint should return a clean 4xx, not hang
try:
    runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps({"instances": [[1, 2, 3]]}),
    )
    print("malformed payload: NOT rejected (unexpected)")
except runtime.exceptions.ModelError as e:
    print(f"malformed payload correctly rejected: {e.response['Error']['Message'][:80]}")

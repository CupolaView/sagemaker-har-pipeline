"""Send test traffic to the endpoint, either normal or deliberately drifted.

Drift simulates a miscalibrated sensor: every value is multiplied by a
gain and shifted by an offset. That moves the per-channel mean/std
features Model Monitor watches.

Run:  python src/send_traffic.py --mode normal --n 200
      python src/send_traffic.py --mode drift  --n 300
"""
import argparse
import json
from pathlib import Path

import boto3
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["normal", "drift"], default="normal")
parser.add_argument("--n", type=int, default=200)
parser.add_argument("--gain", type=float, default=2.5)
parser.add_argument("--offset", type=float, default=1.0)
parser.add_argument("--test-dir", default="processed/test")
parser.add_argument("--endpoint", default="har-classifier")
args = parser.parse_args()

X = np.load(Path(args.test_dir) / "X.npy")
if args.mode == "drift":
    X = X * args.gain + args.offset

rng = np.random.default_rng(7)
runtime = boto3.client("sagemaker-runtime", region_name="us-west-2")
for i in range(args.n):
    window = X[rng.integers(len(X))]
    runtime.invoke_endpoint(
        EndpointName=args.endpoint,
        ContentType="application/json",
        Body=json.dumps({"instances": [window.tolist()]}),
    )
    if (i + 1) % 50 == 0:
        print(f"sent {i + 1}/{args.n} ({args.mode})")
print("done. Wait ~5 minutes for capture files to reach S3 before monitoring.")

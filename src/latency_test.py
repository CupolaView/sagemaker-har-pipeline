"""Latency benchmark for a live endpoint: p50/p95/p99 over N invocations.

Run:  python src/latency_test.py                          # real-time endpoint
      python src/latency_test.py --endpoint har-classifier-serverless
      python src/latency_test.py --batch 16               # payload of 16 windows
"""
import argparse
import json
import time
from pathlib import Path

import boto3
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--endpoint", default="har-classifier")
parser.add_argument("--test-dir", default="processed/test")
parser.add_argument("--n", type=int, default=100)
parser.add_argument("--batch", type=int, default=1, help="windows per request")
args = parser.parse_args()

X = np.load(Path(args.test_dir) / "X.npy")
rng = np.random.default_rng(0)
runtime = boto3.client("sagemaker-runtime", region_name="us-west-2")

latencies = []
for i in range(args.n):
    idx = rng.choice(len(X), size=args.batch, replace=False)
    body = json.dumps({"instances": X[idx].tolist()})
    t0 = time.perf_counter()
    runtime.invoke_endpoint(
        EndpointName=args.endpoint, ContentType="application/json", Body=body)
    latencies.append((time.perf_counter() - t0) * 1000)
    if i == 0:
        print(f"first call (incl. any cold start): {latencies[0]:.0f} ms")

lat = np.array(latencies[1:])  # drop first call from percentiles
print(f"endpoint={args.endpoint} n={args.n} batch={args.batch}")
print(f"p50={np.percentile(lat, 50):.0f} ms  "
      f"p95={np.percentile(lat, 95):.0f} ms  "
      f"p99={np.percentile(lat, 99):.0f} ms  "
      f"mean={lat.mean():.0f} ms")
print("note: measured from client incl. network; CloudWatch ModelLatency "
      "shows server-side only — compare both, the gap is network overhead")

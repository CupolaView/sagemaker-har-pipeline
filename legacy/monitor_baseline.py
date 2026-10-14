"""Build the Model Monitor data-quality baseline from TRAINING data.

Steps:
1. Compute the 18 monitoring features for every training window, using
   the same function the live preprocessor uses.
2. Upload them to S3 as a CSV with a header row.
3. Run a baseline job (suggest_baseline), which writes:
     statistics.json  - per-feature distributions ("what normal looks like")
     constraints.json - rules live traffic must satisfy (types,
                        completeness, and drift thresholds)

Run:  python src/monitor_baseline.py   (needs ./processed/train/X.npy)
Cost: one ml.m5.xlarge processing job, ~5-10 min, a few cents.
"""
import csv
import io

import boto3
import numpy as np
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat

from monitor_preprocessor import FEATURE_NAMES, window_features

BUCKET = "gaurav-mlops-us-west-2"
ROLE = "arn:aws:iam::674224408819:role/service-role/AmazonSageMaker-ExecutionRole-20260609T145595"
INPUT_KEY = "har/monitoring/baseline-input/train_features.csv"
BASELINE_URI = f"s3://{BUCKET}/har/monitoring/baseline"

# --- 1. features for every training window ---
X = np.load("processed/train/X.npy")                      # (7352, 9, 128)
buf = io.StringIO()
writer = csv.writer(buf)
writer.writerow(FEATURE_NAMES)
for window in X.tolist():
    feats = window_features(window)
    writer.writerow([feats[name] for name in FEATURE_NAMES])
print(f"computed features for {len(X)} training windows")

# --- 2. upload ---
boto3.client("s3", region_name="us-west-2").put_object(
    Bucket=BUCKET, Key=INPUT_KEY, Body=buf.getvalue().encode("utf-8"))
print(f"uploaded s3://{BUCKET}/{INPUT_KEY}")

# --- 3. baseline job ---
monitor = DefaultModelMonitor(
    role=ROLE,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    volume_size_in_gb=20,
    max_runtime_in_seconds=1800,
)
monitor.suggest_baseline(
    baseline_dataset=f"s3://{BUCKET}/{INPUT_KEY}",
    dataset_format=DatasetFormat.csv(header=True),
    output_s3_uri=BASELINE_URI,
    wait=True,
    logs=False,
)
print(f"baseline written to {BASELINE_URI}/statistics.json and constraints.json")

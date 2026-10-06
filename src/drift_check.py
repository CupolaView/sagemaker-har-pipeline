"""Data drift check on captured endpoint traffic (replaces SageMaker Model Monitor).

SageMaker Model Monitor is closed to new customers, so this script follows
AWS's recommended replacement: open-source drift detection on data-capture
files, with the result published as a CloudWatch custom metric.

Steps:
  reference = 18 summary features of every TRAINING window
  current   = the same features from recently captured endpoint requests
  test      = two-sample Kolmogorov-Smirnov test per feature; drifted only if
              significant after Bonferroni correction AND KS stat >= min-stat
  verdict   = share of drifted features vs. threshold (default 30%)
  outputs   = console table, optional Evidently HTML report, optional CloudWatch metric

Design choice: the alert number comes from scipy's KS test, not from
Evidently. Evidently's API changes often between versions; the KS test
doesn't. Evidently is used only for the visual report, so if its API
changes, the report breaks but the alert keeps working.

Run:  python src/drift_check.py --minutes 15
      python src/drift_check.py --minutes 15 --publish --report
Deps: pip install scipy pandas "evidently>=0.7"
"""
import argparse
import base64
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from monitor_preprocessor import FEATURE_NAMES, window_features

BUCKET = "gaurav-mlops-us-west-2"
CAPTURE_PREFIX = "har/datacapture/har-classifier/"
REGION = "us-west-2"

parser = argparse.ArgumentParser()
parser.add_argument("--minutes", type=int, default=30,
                    help="look back this far for capture files")
parser.add_argument("--threshold", type=float, default=0.3,
                    help="alert if more than this share of features drift")
parser.add_argument("--alpha", type=float, default=0.05,
                    help="family-wise significance level (Bonferroni-corrected)")
parser.add_argument("--min-stat", type=float, default=0.1,
                    help="minimum KS statistic (effect size) to count as drift")
parser.add_argument("--train-dir", default="processed/train")
parser.add_argument("--publish", action="store_true",
                    help="publish drift share to CloudWatch")
parser.add_argument("--report", action="store_true",
                    help="write an Evidently HTML report to reports/")
args = parser.parse_args()

# --- 1. reference: features of the training set ---
X_train = np.load(Path(args.train_dir) / "X.npy")
reference = pd.DataFrame([window_features(w) for w in X_train.tolist()])[FEATURE_NAMES]
print(f"reference: {len(reference)} training windows")

# --- 2. current: features from recent capture files ---
s3 = boto3.client("s3", region_name=REGION)
cutoff = datetime.now(timezone.utc) - timedelta(minutes=args.minutes)
rows, files = [], 0
for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=CAPTURE_PREFIX):
    for obj in page.get("Contents", []):
        if obj["LastModified"] < cutoff:
            continue
        files += 1
        body = s3.get_object(Bucket=BUCKET, Key=obj["Key"])["Body"].read().decode("utf-8")
        for line in body.splitlines():
            if not line.strip():
                continue
            inp = json.loads(line)["captureData"]["endpointInput"]
            data = inp["data"]
            if inp.get("encoding", "").upper() == "BASE64":
                data = base64.b64decode(data).decode("utf-8")
            try:
                instances = json.loads(data)["instances"]
            except (ValueError, KeyError, TypeError):
                continue  # malformed request; skip
            for w in instances:
                if len(w) == 9 and all(len(ch) == 128 for ch in w):
                    rows.append(window_features(w))

if not rows:
    raise SystemExit(f"no captured windows in the last {args.minutes} min "
                     f"({files} files) - send traffic and wait ~5 min first")
current = pd.DataFrame(rows)[FEATURE_NAMES]
print(f"current:   {len(current)} captured windows from {files} capture file(s)")

# --- 3. per-feature KS test ---
# A feature counts as drifted only if BOTH hold:
#  - statistically significant after a Bonferroni correction (alpha / 18),
#    because running 18 tests at p<0.05 gives ~1 false flag by chance alone
#  - a meaningful effect size (KS statistic >= min-stat), because with large
#    traffic volumes p-values flag even trivial shifts
# Calibrated 2026-10-06: normal test-set traffic had max KS 0.129 (that run
# used raw p<0.05 and flagged 5/18, share 0.28, just under the 0.30 threshold).
p_cut = args.alpha / len(FEATURE_NAMES)
results = []
for f in FEATURE_NAMES:
    stat, p = ks_2samp(reference[f], current[f])
    results.append((f, stat, p, p < p_cut and stat >= args.min_stat))
drifted = [r for r in results if r[3]]
share = len(drifted) / len(FEATURE_NAMES)

print(f"\ndrift rule: p < {p_cut:.2e} (Bonferroni) AND KS stat >= {args.min_stat}")
print(f"{'feature':<22} {'KS stat':>8} {'p-value':>10}  drift?")
for f, stat, p, d in results:
    print(f"{f:<22} {stat:>8.3f} {p:>10.2e}  {'DRIFT' if d else ''}")
verdict = "DRIFT DETECTED" if share > args.threshold else "OK"
print(f"\ndrifted features: {len(drifted)}/{len(FEATURE_NAMES)} "
      f"(share {share:.2f}, threshold {args.threshold:.2f}) -> {verdict}")

# --- 4. CloudWatch metric (the October alarm hangs off this) ---
if args.publish:
    boto3.client("cloudwatch", region_name=REGION).put_metric_data(
        Namespace="HAR/Monitoring",
        MetricData=[
            {"MetricName": "DriftedFeatureShare", "Value": share, "Unit": "None",
             "Dimensions": [{"Name": "Endpoint", "Value": "har-classifier"}]},
            {"MetricName": "WindowsAnalyzed", "Value": len(current), "Unit": "Count",
             "Dimensions": [{"Name": "Endpoint", "Value": "har-classifier"}]},
        ],
    )
    print("published DriftedFeatureShare to CloudWatch (namespace HAR/Monitoring)")

# --- 5. Evidently visual report (optional; failure here doesn't affect the verdict) ---
if args.report:
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
        out = Path("reports") / f"drift_{datetime.now():%Y%m%d_%H%M%S}.html"
        out.parent.mkdir(exist_ok=True)
        Report([DataDriftPreset()]).run(
            current_data=current, reference_data=reference).save_html(str(out))
        print(f"Evidently report: {out}")
    except Exception as e:  # API differences between Evidently versions
        print(f"Evidently report skipped ({type(e).__name__}: {e})")

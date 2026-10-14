"""Run data-quality monitoring against the endpoint's captured traffic.

Modes:
  --mode now     one-time run over the LAST HOUR of captured traffic. Waits
                 for it to finish, then prints any violations. Use this to test.
  --mode hourly  recurring hourly schedule. Only useful while the endpoint
                 is up, so create it only in sessions where you keep it running.

Run:  python src/monitor_run.py --mode now
Cost: each run is one ml.m5.xlarge processing job, ~5-10 min, a few cents.

Teardown: delete the schedule BEFORE deleting the endpoint:
  aws sagemaker delete-monitoring-schedule --monitoring-schedule-name har-data-quality
"""
import argparse
import json
import time

import boto3
from botocore.exceptions import ClientError
from sagemaker.model_monitor import CronExpressionGenerator, DefaultModelMonitor

BUCKET = "gaurav-mlops-us-west-2"
ROLE = "arn:aws:iam::674224408819:role/service-role/AmazonSageMaker-ExecutionRole-20260609T145595"
ENDPOINT = "har-classifier"
SCHEDULE = "har-data-quality"
BASELINE = f"s3://{BUCKET}/har/monitoring/baseline"
REPORTS = f"s3://{BUCKET}/har/monitoring/reports"
TERMINAL = {"Completed", "CompletedWithViolations", "Failed", "Stopped"}

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["now", "hourly"], default="now")
args = parser.parse_args()

sm = boto3.client("sagemaker", region_name="us-west-2")

# --- 1. remove any previous schedule with this name (a one-time schedule
#        must be deleted before another one can be created) ---
try:
    sm.delete_monitoring_schedule(MonitoringScheduleName=SCHEDULE)
    print("deleting previous schedule...")
    while True:
        time.sleep(5)
        sm.describe_monitoring_schedule(MonitoringScheduleName=SCHEDULE)
except ClientError:
    pass  # schedule is gone (or never existed)

# --- 2. create the schedule ---
monitor = DefaultModelMonitor(
    role=ROLE,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    volume_size_in_gb=20,
    max_runtime_in_seconds=1800,
)
kwargs = dict(
    monitor_schedule_name=SCHEDULE,
    endpoint_input=ENDPOINT,
    record_preprocessor_script="src/monitor_preprocessor.py",
    output_s3_uri=REPORTS,
    statistics=f"{BASELINE}/statistics.json",
    constraints=f"{BASELINE}/constraints.json",
    enable_cloudwatch_metrics=True,   # feeds the October alarm work
)
if args.mode == "now":
    kwargs.update(
        # "NOW" is the API's one-time-run expression. Passed as a plain
        # string because CronExpressionGenerator.now() isn't present in
        # every SDK version.
        schedule_cron_expression="NOW",
        data_analysis_start_time="-PT1H",
        data_analysis_end_time="-PT0H",
    )
else:
    kwargs["schedule_cron_expression"] = CronExpressionGenerator.hourly()

monitor.create_monitoring_schedule(**kwargs)
print(f"schedule '{SCHEDULE}' created ({args.mode})")
if args.mode == "hourly":
    raise SystemExit("hourly schedule is live; check results in the console")

# --- 3. wait for the one-time execution to finish ---
print("waiting for monitoring execution (typically 5-10 min)...")
summary = {}
for _ in range(60):
    time.sleep(30)
    desc = sm.describe_monitoring_schedule(MonitoringScheduleName=SCHEDULE)
    summary = desc.get("LastMonitoringExecutionSummary", {})
    status = summary.get("MonitoringExecutionStatus")
    if status:
        print(f"  status: {status}")
    if status in TERMINAL:
        break

status = summary.get("MonitoringExecutionStatus")
if status == "Failed":
    raise SystemExit(f"FAILED: {summary.get('FailureReason')}")

# --- 4. print violations, if any ---
job_name = summary["ProcessingJobArn"].split("/")[-1]
job = sm.describe_processing_job(ProcessingJobName=job_name)
out_uri = job["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
bucket, prefix = out_uri.replace("s3://", "").split("/", 1)
print(f"report location: {out_uri}")

s3 = boto3.client("s3", region_name="us-west-2")
try:
    body = s3.get_object(Bucket=bucket, Key=f"{prefix}/constraint_violations.json")["Body"]
    violations = json.loads(body.read())["violations"]
except ClientError:
    violations = []

if not violations:
    print("RESULT: no violations - traffic matches the baseline")
else:
    print(f"RESULT: {len(violations)} violation(s):")
    for v in violations:
        print(f"  {v['feature_name']:<22} {v['constraint_check_type']:<22} "
              f"{v['description'][:90]}")

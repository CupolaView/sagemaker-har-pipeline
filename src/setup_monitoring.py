"""Create the CloudWatch alerting and dashboard for the HAR endpoint.

Creates (safe to rerun; every call is create-or-update):
  1. SNS topic 'har-alerts' with an email subscription
  2. Three CloudWatch alarms:
       har-drift      DriftedFeatureShare > 0.3   (custom metric from drift_check.py)
       har-latency    ModelLatency p95 > 100 ms   (2 periods in a row)
       har-5xx        any Invocation5XXErrors
  3. Dashboard 'har-classifier' with drift, traffic, latency and error charts

Run:  python src/setup_monitoring.py --email you@example.com
Then confirm the subscription from the email AWS sends, or no alerts arrive.

Cost: within the CloudWatch free tier (10 alarms, 3 dashboards, 10 custom
metrics) for this project; otherwise about $0.10 per alarm per month.
"""
import argparse
import json

import boto3

REGION = "us-west-2"
ENDPOINT = "har-classifier"
TOPIC_NAME = "har-alerts"

parser = argparse.ArgumentParser()
parser.add_argument("--email", required=True, help="where alarm emails go")
args = parser.parse_args()

sns = boto3.client("sns", region_name=REGION)
cw = boto3.client("cloudwatch", region_name=REGION)

# --- 1. SNS topic + email subscription (create_topic is idempotent) ---
topic_arn = sns.create_topic(Name=TOPIC_NAME)["TopicArn"]
subs = sns.list_subscriptions_by_topic(TopicArn=topic_arn)["Subscriptions"]
if any(s["Endpoint"] == args.email for s in subs):
    print(f"email already subscribed to {TOPIC_NAME}")
else:
    sns.subscribe(TopicArn=topic_arn, Protocol="email", Endpoint=args.email)
    print(f"subscription created - CONFIRM IT from the email AWS just sent to {args.email}")

# --- 2. alarms (put_metric_alarm creates or overwrites) ---
common = dict(
    AlarmActions=[topic_arn],
    OKActions=[topic_arn],            # also email when it recovers
    TreatMissingData="notBreaching",  # endpoint is usually torn down: no data is not a problem
    EvaluationPeriods=1,
    Period=300,
)

# Drift: the alarm this whole monitoring block exists for.
cw.put_metric_alarm(
    AlarmName="har-drift",
    AlarmDescription=("More than 30% of input features drifted from the training "
                      "baseline. See docs/monitoring.md."),
    Namespace="HAR/Monitoring",
    MetricName="DriftedFeatureShare",
    Dimensions=[{"Name": "Endpoint", "Value": ENDPOINT}],
    Statistic="Maximum",
    Threshold=0.3,
    ComparisonOperator="GreaterThanThreshold",
    **common,
)

# Latency: measured p95 ModelLatency was 14.3 ms (docs/serving.md);
# 100 ms leaves headroom so the alarm means "something changed".
# Two periods in a row so one slow burst doesn't page anyone.
cw.put_metric_alarm(
    AlarmName="har-latency",
    AlarmDescription="Model p95 latency above 100 ms for 10 minutes.",
    Namespace="AWS/SageMaker",
    MetricName="ModelLatency",                 # reported in microseconds
    Dimensions=[{"Name": "EndpointName", "Value": ENDPOINT},
                {"Name": "VariantName", "Value": "AllTraffic"}],
    ExtendedStatistic="p95",
    Threshold=100_000,                         # 100 ms in microseconds
    ComparisonOperator="GreaterThanThreshold",
    **{**common, "EvaluationPeriods": 2},
)

# Server errors: any 5xx means the model container failed on a request.
cw.put_metric_alarm(
    AlarmName="har-5xx",
    AlarmDescription="Endpoint returned server errors (model container failures).",
    Namespace="AWS/SageMaker",
    MetricName="Invocation5XXErrors",
    Dimensions=[{"Name": "EndpointName", "Value": ENDPOINT},
                {"Name": "VariantName", "Value": "AllTraffic"}],
    Statistic="Sum",
    Threshold=0,
    ComparisonOperator="GreaterThanThreshold",
    **common,
)
print("alarms created: har-drift, har-latency, har-5xx")

# --- 3. dashboard ---
ep = ["EndpointName", ENDPOINT, "VariantName", "AllTraffic"]


def widget(title, metrics, x, y, w=12, h=6, annotations=None, ymax=None):
    props = {"title": title, "region": REGION, "metrics": metrics,
             "view": "timeSeries", "stat": "Sum", "period": 300}
    if annotations:
        props["annotations"] = {"horizontal": annotations}
    if ymax is not None:
        props["yAxis"] = {"left": {"min": 0, "max": ymax}}
    return {"type": "metric", "x": x, "y": y, "width": w, "height": h, "properties": props}


dashboard = {"widgets": [
    widget("Drift: share of features drifted",
           [["HAR/Monitoring", "DriftedFeatureShare", "Endpoint", ENDPOINT,
             {"stat": "Maximum"}]],
           0, 0, annotations=[{"label": "alarm threshold", "value": 0.3}], ymax=1),
    widget("Windows analyzed per drift check",
           [["HAR/Monitoring", "WindowsAnalyzed", "Endpoint", ENDPOINT,
             {"stat": "Maximum"}]],
           12, 0),
    widget("Invocations",
           [["AWS/SageMaker", "Invocations", *ep]],
           0, 6),
    widget("Latency p50 / p95 (microseconds)",
           [["AWS/SageMaker", "ModelLatency", *ep, {"stat": "p50", "label": "model p50"}],
            ["AWS/SageMaker", "ModelLatency", *ep, {"stat": "p95", "label": "model p95"}],
            ["AWS/SageMaker", "OverheadLatency", *ep, {"stat": "p95", "label": "overhead p95"}]],
           12, 6, annotations=[{"label": "alarm (100 ms)", "value": 100000}]),
    widget("Errors (4xx = bad requests, 5xx = model failures)",
           [["AWS/SageMaker", "Invocation4XXErrors", *ep],
            ["AWS/SageMaker", "Invocation5XXErrors", *ep]],
           0, 12, w=24),
]}
cw.put_dashboard(DashboardName="har-classifier", DashboardBody=json.dumps(dashboard))
print("dashboard created: har-classifier")
print(f"view: https://{REGION}.console.aws.amazon.com/cloudwatch/home"
      f"?region={REGION}#dashboards/dashboard/har-classifier")

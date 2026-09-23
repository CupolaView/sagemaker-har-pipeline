"""Deploy the latest APPROVED model package as a real-time endpoint.

The governance loop closes here: this script refuses to deploy anything
that hasn't passed the registry's approval gate. No hardcoded artifacts.

Run:      python src/deploy_endpoint.py
Cost:     ml.m5.large endpoint $0.115/hr (~$84/month) WHILE RUNNING — tear down after
          each session:  aws sagemaker delete-endpoint --endpoint-name har-classifier
"""
import boto3
from sagemaker.model_monitor import DataCaptureConfig
from sagemaker.pytorch import PyTorchModel

GROUP_NAME = "har-classifier"
ENDPOINT_NAME = "har-classifier"
REGION = "us-west-2"
ROLE = "arn:aws:iam::674224408819:role/service-role/AmazonSageMaker-ExecutionRole-20260609T145595"

sm = boto3.client("sagemaker", region_name=REGION)

# --- 1. find the latest Approved package (and only Approved) ---
approved = sm.list_model_packages(
    ModelPackageGroupName=GROUP_NAME,
    ModelApprovalStatus="Approved",
    SortBy="CreationTime",
    SortOrder="Descending",
)["ModelPackageSummaryList"]

if not approved:
    raise SystemExit("no Approved model package in the group — nothing to deploy")

pkg = sm.describe_model_package(ModelPackageName=approved[0]["ModelPackageArn"])
model_data = pkg["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
print(f"deploying package: {approved[0]['ModelPackageArn']}")
print(f"artifact: {model_data}")

# --- 1b. remove stale endpoint-config from a previous deploy/teardown cycle
#         (delete-endpoint leaves the config behind; same-name create collides) ---
try:
    sm.delete_endpoint_config(EndpointConfigName=ENDPOINT_NAME)
    print(f"deleted stale endpoint-config '{ENDPOINT_NAME}'")
except sm.exceptions.ClientError:
    pass  # none existed — normal on first deploy

# --- 2. wrap artifact + inference code and deploy ---
model = PyTorchModel(
    model_data=model_data,
    role=ROLE,
    entry_point="inference.py",
    source_dir="src",
    framework_version="2.6",
    py_version="py312",
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name=ENDPOINT_NAME,
    # capture requests/responses to S3 — Model Monitor's raw material
    data_capture_config=DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=100,
        destination_s3_uri="s3://gaurav-mlops-us-west-2/har/datacapture/",
    ),
)
print(f"endpoint '{ENDPOINT_NAME}' is InService")

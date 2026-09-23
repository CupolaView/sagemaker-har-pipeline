"""Deploy the latest APPROVED package as a SERVERLESS endpoint.

Same governance path as deploy_endpoint.py — only the serving config
differs. Serverless: no instance running when idle, billed per
inference-duration + invocations; cold starts when scaling from zero.

Run:  python src/deploy_serverless.py
Cost: ~zero when idle — safe to leave up, unlike the real-time endpoint.
"""
import boto3
from sagemaker.pytorch import PyTorchModel
from sagemaker.serverless import ServerlessInferenceConfig

GROUP_NAME = "har-classifier"
ENDPOINT_NAME = "har-classifier-serverless"
REGION = "us-west-2"
ROLE = "arn:aws:iam::674224408819:role/service-role/AmazonSageMaker-ExecutionRole-20260609T145595"

sm = boto3.client("sagemaker", region_name=REGION)

approved = sm.list_model_packages(
    ModelPackageGroupName=GROUP_NAME, ModelApprovalStatus="Approved",
    SortBy="CreationTime", SortOrder="Descending",
)["ModelPackageSummaryList"]
if not approved:
    raise SystemExit("no Approved model package — nothing to deploy")

try:
    sm.delete_endpoint_config(EndpointConfigName=ENDPOINT_NAME)
    print(f"deleted stale endpoint-config '{ENDPOINT_NAME}'")
except sm.exceptions.ClientError:
    pass  # none existed

pkg = sm.describe_model_package(ModelPackageName=approved[0]["ModelPackageArn"])
model_data = pkg["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
print(f"deploying package: {approved[0]['ModelPackageArn']}")

model = PyTorchModel(
    model_data=model_data, role=ROLE,
    entry_point="inference.py", source_dir="src",
    framework_version="2.6", py_version="py312",
)

model.deploy(
    serverless_inference_config=ServerlessInferenceConfig(
        memory_size_in_mb=2048,   # 1024-6144; more memory = more cpu = faster
        max_concurrency=5,
    ),
    endpoint_name=ENDPOINT_NAME,
)
print(f"serverless endpoint '{ENDPOINT_NAME}' is InService")

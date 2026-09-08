"""Register the best HPO model in the SageMaker Model Registry.

Creates (idempotently) a model package group, then registers the artifact
of the best tuning-job trial as a new model package version with
approval status PendingManualApproval.

Run:  python src/register_model.py
Then: python src/approve_model.py   (the explicit governance step)
"""
import boto3
from sagemaker.pytorch import PyTorch

BEST_TRAINING_JOB = "pytorch-training-260902-1652-008-3f2d80f5"
GROUP_NAME = "har-classifier"
REGION = "us-west-2"

sm = boto3.client("sagemaker", region_name=REGION)

# --- 1. model package group (idempotent) ---
try:
    sm.create_model_package_group(
        ModelPackageGroupName=GROUP_NAME,
        ModelPackageGroupDescription=(
            "UCI HAR activity classifier — 1D CNN, PyTorch. "
            "Lineage: see docs/hpo.md in the repo."
        ),
    )
    print(f"created model package group '{GROUP_NAME}'")
except sm.exceptions.from_code("ValidationException"):
    print(f"model package group '{GROUP_NAME}' already exists")

# --- 2. attach to the winning training job and register its artifact ---
estimator = PyTorch.attach(BEST_TRAINING_JOB)

package = estimator.register(
    model_package_group_name=GROUP_NAME,
    content_types=["application/json"],
    response_types=["application/json"],
    inference_instances=["ml.m5.large"],
    transform_instances=["ml.m5.large"],
    approval_status="PendingManualApproval",
    customer_metadata_properties={
        "source_tuning_job": "pytorch-training-260902-1652",
        "data_version": "v1-processed",
        "test_accuracy": "0.9355",
        "docs": "docs/hpo.md",
    },
)
print(f"registered: {package.model_package_arn}")
print("status: PendingManualApproval — run approve_model.py to promote")

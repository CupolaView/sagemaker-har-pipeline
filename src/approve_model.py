"""Promote the latest PendingManualApproval package in the group to Approved.

This is the deliberate governance step: separated from registration so that
'a model exists' and 'a model may serve traffic' are different decisions.
Next week's deploy script will refuse anything not Approved.

Run:  python src/approve_model.py
"""
import boto3

GROUP_NAME = "har-classifier"
REGION = "us-west-2"

sm = boto3.client("sagemaker", region_name=REGION)

pending = sm.list_model_packages(
    ModelPackageGroupName=GROUP_NAME,
    ModelApprovalStatus="PendingManualApproval",
    SortBy="CreationTime",
    SortOrder="Descending",
)["ModelPackageSummaryList"]

if not pending:
    print("nothing pending approval")
    raise SystemExit(0)

latest = pending[0]
arn = latest["ModelPackageArn"]
print(f"promoting: {arn}")

sm.update_model_package(
    ModelPackageArn=arn,
    ModelApprovalStatus="Approved",
    ApprovalDescription="test_acc 0.9355 >= 0.90 bar; lineage in docs/hpo.md",
)
print("status: Approved")

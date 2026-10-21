import sagemaker
from sagemaker.pytorch import PyTorch

sess = sagemaker.Session()
# Inside SageMaker Studio you could use sagemaker.get_execution_role() instead.
role = "arn:aws:iam::674224408819:role/service-role/AmazonSageMaker-ExecutionRole-20260609T145595"

estimator = PyTorch(
    entry_point="train.py",
    source_dir="src",
    role=role,
    framework_version="2.6",          # pick a recent one from the docs
    py_version="py312",
    instance_type="ml.m5.large",      # CPU is fine and cheap for this model
    instance_count=1,
    hyperparameters={"epochs": 10, "batch-size": 64, "lr": 1e-3},
    metric_definitions=[
        {"Name": "train_loss", "Regex": r"train_loss=([0-9.]+)"},
        {"Name": "test_acc",   "Regex": r"test_acc=([0-9.]+)"},
    ],
    output_path="s3://gaurav-mlops-us-west-2/har/artifacts/",
    # Where the SDK uploads the zipped src/ before the job starts. Without
    # this it defaults to the root of the output_path bucket; a dedicated
    # prefix keeps the bucket tidy and lets the CI role's S3 permission
    # cover just this prefix (infra/github-cd-policy.json).
    code_location="s3://gaurav-mlops-us-west-2/har/code",
)

estimator.fit({
    "train": "s3://gaurav-mlops-us-west-2/har/data/v1-processed/train/",
    "test":  "s3://gaurav-mlops-us-west-2/har/data/v1-processed/test/",
})
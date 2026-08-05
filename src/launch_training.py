import sagemaker
from sagemaker.pytorch import PyTorch

sess = sagemaker.Session()
role = "arn:aws:iam::674224408819:role/service-role/AmazonSageMaker-ExecutionRole-20260609T145595"#sagemaker.get_execution_role()  # inside Studio; else paste the role ARN

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
)

estimator.fit({
    "train": "s3://gaurav-mlops-us-west-2/har/data/v1/train/",
    "test":  "s3://gaurav-mlops-us-west-2/har/data/v1/test/",
})
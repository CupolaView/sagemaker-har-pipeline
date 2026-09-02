"""SageMaker Hyperparameter Tuning job for the HAR training script.

Reuses the same estimator config as launch_training.py — the tuner just
launches many training jobs with sampled hyperparameters and ranks them
by the objective metric (test_acc, scraped from stdout via regex).

Run:  python src/launch_hpo.py
Cost: 8 jobs x ~10-15 min on ml.m5.large ≈ < $2 total.
"""
from sagemaker.pytorch import PyTorch
from sagemaker.tuner import (
    HyperparameterTuner,
    ContinuousParameter,
    CategoricalParameter,
)

ROLE = "arn:aws:iam::674224408819:role/service-role/AmazonSageMaker-ExecutionRole-20260609T145595"
BUCKET = "gaurav-mlops-us-west-2"

estimator = PyTorch(
    entry_point="train.py",
    source_dir="src",
    role=ROLE,
    framework_version="2.6",
    py_version="py312",
    instance_type="ml.m5.large",
    instance_count=1,
    hyperparameters={"epochs": 10},  # fixed; tuner supplies the rest
    metric_definitions=[
        {"Name": "train_loss", "Regex": r"train_loss=([0-9.]+)"},
        {"Name": "test_acc",   "Regex": r"test_acc=([0-9.]+)"},
    ],
    output_path=f"s3://{BUCKET}/har/artifacts/",
)

tuner = HyperparameterTuner(
    estimator=estimator,
    objective_metric_name="test_acc",
    objective_type="Maximize",
    metric_definitions=estimator.metric_definitions,
    hyperparameter_ranges={
        "lr": ContinuousParameter(1e-4, 1e-2, scaling_type="Logarithmic"),
        "batch-size": CategoricalParameter([32, 64, 128]),
    },
    strategy="Bayesian",       # vs Random/Grid — know the trade-off for the exam
    max_jobs=8,
    max_parallel_jobs=2,       # Bayesian needs sequential info; 2 is a balance
)

tuner.fit({
    "train": f"s3://{BUCKET}/har/data/v1-processed/train/",
    "test":  f"s3://{BUCKET}/har/data/v1-processed/test/",
})

best = tuner.best_training_job()
print(f"best job: {best}")
print(f"analytics:\n{tuner.analytics().dataframe()[['lr', 'batch-size', 'FinalObjectiveValue']]}")

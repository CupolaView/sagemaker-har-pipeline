from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor

role = "arn:aws:iam::674224408819:role/service-role/AmazonSageMaker-ExecutionRole-20260609T145595"

processor = SKLearnProcessor(
    framework_version="1.2-1", role=role,
    instance_type="ml.m5.large", instance_count=1,
)
processor.run(
    code="src/preprocess.py",
    inputs=[
        ProcessingInput(source="s3://gaurav-mlops-us-west-2/har/data/v1/train/",
                        destination="/opt/ml/processing/input/train"),
        ProcessingInput(source="s3://gaurav-mlops-us-west-2/har/data/v1/test/",
                        destination="/opt/ml/processing/input/test"),
    ],
    outputs=[
        ProcessingOutput(source="/opt/ml/processing/output/train",
                         destination="s3://gaurav-mlops-us-west-2/har/data/v1-processed/train/"),
        ProcessingOutput(source="/opt/ml/processing/output/test",
                         destination="s3://gaurav-mlops-us-west-2/har/data/v1-processed/test/"),
    ],
)
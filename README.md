# sagemaker-har-pipeline
A Sagemaker Pipeline for UCL HAR dataset classification

## Pipeline

Raw data (S3, versioned) → Processing job (`src/preprocess.py`) →
Training job (`src/train.py`) → HPO (`src/launch_hpo.py`, see `docs/hpo.md`) →
Model Registry with manual approval gate (`src/register_model.py`,
`src/approve_model.py`) → [endpoint — coming next]

# Data Versioning 

## Layout

```
s3://gaurav-mlops-us-west-2/har/
├── data/
│   ├── v1/                  # raw UCI HAR, as downloaded (immutable)
│   │   ├── train/
│   │   └── test/
│   └── v1-processed/        # output of preprocess.py run against v1
│       ├── train/
│       └── test/
└── artifacts/               # training job outputs (model.tar.gz)
```

## Rules

1. **Versions are immutable.** Nothing under `data/v1/` is ever modified or
   deleted after upload. A change means a new version, never an edit.
2. **A new raw version (`v2/`) is cut when the source data changes:**
   new samples added, labels corrected, or a different upstream dataset
   revision. Never for preprocessing changes.
3. **Processed data is named `<raw-version>-processed/`** and is fully
   reproducible: `preprocess.py` + a raw version ⇒ the processed set.
   If the *preprocessing logic* changes materially, bump a suffix
   (`v1-processed-b/`) and record the git SHA of preprocess.py that made it.
4. **Every training job's lineage is recoverable:** the job's input channel
   URI names the data version, and the processed prefix maps to a
   preprocess.py commit. Given a model.tar.gz, you can name the exact raw
   files and code that produced it.
5. **Why not DVC here:** DVC (used at XSENSOR) solves this with hashes and
   a local cache; this project uses plain S3 prefixes to stay legible to
   anyone reading the repo, and because SageMaker channels address S3 URIs
   directly. Trade-off documented deliberately.

## Current versions

| Version | Created    | Contents                          | Notes            |
|---------|------------|-----------------------------------|------------------|
| v1      | 2026-07-23 | UCI HAR as published (train/test) | initial upload   |
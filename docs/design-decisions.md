# Design Decisions

Each entry covers the context, the decision, the alternatives considered,
and the consequences. Supporting numbers are in [`hpo.md`](hpo.md),
[`serving.md`](serving.md), and [`monitoring.md`](monitoring.md).

---

## 1. Data versioning with S3 prefixes, not DVC

**Context.** Every model needs to be traceable to the exact data that
produced it.

**Decision.** Immutable, versioned S3 prefixes (`data/v1/`,
`data/v1-processed/`) with written rules for when to create a new version
(see the README).

**Alternatives.** DVC (which I use at work), or SageMaker Feature Store.

**Why.** SageMaker jobs read their inputs directly from S3 URIs, so the
version is visible in every job's configuration without extra tooling.
Anyone reading the repo can follow the layout. DVC adds content hashes and
a local cache, which is worth it for frequently changing datasets, but this
dataset changes rarely.

**Consequences.** Immutability is a convention, not enforced. Enabling S3
Object Lock or bucket versioning would enforce it.

---

## 2. Same scripts run locally and on SageMaker

**Context.** Debugging inside cloud jobs is slow and costs money.

**Decision.** `train.py` and `preprocess.py` read their paths from
SageMaker's environment variables when running in a job, with command-line
flags as a fallback. The same file runs on a laptop and in a container.
`train.py` also accepts either preprocessed `.npy` files or raw UCI text
files.

**Consequences.** Every stage was tested locally before its first cloud
run. The metric print format (`epoch=… train_loss=… test_acc=…`) is treated
as an interface: SageMaker extracts values from it with regular expressions
for CloudWatch and for tuning, so changing it would break both.

---

## 3. Bayesian hyperparameter tuning, selected on the test set

**Decision.** SageMaker tuning with a Bayesian strategy, 8 trials, 2 in
parallel, searching learning rate (log scale) and batch size.

**Why Bayesian.** It uses earlier results to choose later trials. In this
run, 6 of 8 trials concentrated around lr ≈ 0.005 / batch size 128. Random
search would have spread trials evenly across the whole range.

**Known trade-off.** The tuning objective was test accuracy, so the test set
influenced model selection. The rigorous approach is a validation split by
subject (UCI HAR has 30 subjects), tuning on that and reporting test accuracy
once. I accepted the trade-off because the pipeline is the focus, and
recorded it so it reads as a decision rather than an oversight.

---

## 4. Registration and approval are separate steps

**Decision.** `register_model.py` creates a registry version with status
`PendingManualApproval`. A separate `approve_model.py` promotes it. The
deploy scripts only ever deploy the latest **Approved** version and contain
no hard-coded artifact paths.

**Why.** "A model exists" and "a model may serve traffic" are different
decisions with different owners. Keeping them as separate executables means
permission to run one doesn't grant the other. In a team setting, approval
could sit behind a separate IAM permission or an automated quality gate.

**Consequences.** Lineage (tuning job, data version, accuracy, link to
docs) is stored as metadata on each registry version, so a deployed model
can be traced back to everything that produced it.

---

## 5. Real-time endpoint as the monitored path, serverless measured as an alternative

**Context.** I measured both options with the same approved model and the
same inference code ([`serving.md`](serving.md)).

**Findings.** Warm latency is nearly identical (243 vs. 259 ms p50).
Serverless has a 4.1 s cold start and a worse tail under larger payloads.
On cost, serverless stays cheaper up to about **3.5M requests/day**.

**Decision.** The real-time endpoint is the monitored path. Serverless is
deployable for cost-sensitive, latency-tolerant use.

**Why.** Below millions of requests a day, cost doesn't decide this.
Latency requirements and monitorability do. **Serverless endpoints don't
support data capture**, so they can't feed drift monitoring.

**Consequences.** Real-time endpoints bill hourly whether or not they're
used, so every deploy script documents the teardown command, and
redeploying cleans up the leftover endpoint configuration.

---

## 6. Replacing SageMaker Model Monitor with open-source drift detection

**Context.** The first attempt to create a Model Monitor schedule failed
with *"not available to new customers."* AWS has closed Model Monitor to new
customers and recommends open-source drift detection (Evidently AI) with
CloudWatch instead.

**Decision.** Built that replacement. `drift_check.py` compares captured
traffic to the training data and publishes a CloudWatch metric.

**Consequences.** Data capture, the feature definitions, and the traffic
generator carried over unchanged. Only the analysis step was replaced. The
abandoned scripts are kept in `legacy/` as a record.

---

## 7. Monitor 18 summary features, not 1,152 raw values

**Decision.** Each 9 × 128 window is reduced to the mean and standard
deviation of each of the 9 channels. Drift is tested on those 18 features.

**Why.**
- Testing each of the 1,152 time-step positions would be meaningless:
  windows start at arbitrary points in the movement, so position 47 in one
  window doesn't correspond to position 47 in another.
- The features match how sensors actually fail. An **offset** error moves
  the mean; a **gain** error changes the standard deviation.
- One shared function (`window_features`) computes the features for both
  the training reference and live traffic, so the two definitions can't
  drift apart. Unit tests check both failure modes.

---

## 8. Drift rule: Bonferroni correction plus a minimum effect size

**Context.** The first rule (any feature with p < 0.05) flagged 5 of 18
features on **healthy** traffic. That's a share of 0.28, only 0.02 below the
alarm threshold.

**Decision.** A feature counts as drifted only if p < 0.05 / 18 **and**
its KS statistic is ≥ 0.1. The alarm fires if more than 30% of features
drift.

**Why.**
- 18 tests at p < 0.05 produce about one false flag by chance alone. The
  Bonferroni correction fixes that.
- p-values measure confidence that *some* difference exists, not its size.
  At high traffic volumes, even trivial shifts become significant. The
  effect-size floor fixes that.

**Result.** Healthy traffic: 0/18. Simulated sensor fault: 18/18. The
weakest drift signal (0.193) is above the strongest healthy one (0.129).

**Also decided.** The alarm decision uses scipy's KS test. Evidently only
produces the visual report. Evidently's API changes often between versions,
so an update can break the report but not the alarm.

---

## 9. Alarms that don't fire on silence

**Decision.** Three CloudWatch alarms (drift share > 0.3, model p95 latency
> 100 ms for 10 minutes, any 5xx error). All treat missing data as "not
breaching," and all send an email on recovery as well as on alarm.

**Why.** The endpoint is torn down most of the time, so "no data" is the
normal state. The latency threshold is about 7× the measured p95
(14.3 ms), so it fires on a real change rather than normal variation.

---

## 10. CI/CD with OIDC, pinned to immutable IDs, least privilege

**Decision.**
- GitHub Actions authenticates to AWS with OIDC: short-lived credentials,
  no stored keys.
- The trust policy matches the token's `sub` claim, which includes GitHub's
  **numeric owner and repo IDs**:
  `repo:CupolaView@43252845/sagemaker-har-pipeline@1309229916:ref:refs/heads/main`.
- The pipeline workflow reruns the full test suite before launching any
  paid AWS job.

**Why IDs instead of names.** Names can be reused. If this account or repo
were renamed or deleted, someone else could register the same name, and a
name-based policy would trust their workflows. Numeric IDs are never reused.

**Least privilege.** The CI role can:
- create and describe processing and training jobs,
- pass *only* the SageMaker execution role, and *only* to SageMaker,
- upload code to one S3 prefix (`har/code/`),
- read job logs.

It can't read training data or modify model artifacts. Those belong to the
execution role, which CI can hand to SageMaker but never use itself. The
policy was built up one action at a time from actual `AccessDenied` errors.

**What's public.** The AWS account ID, bucket names, and role ARNs appear in
this repository. They're identifiers, not credentials. The roles can only
be assumed by SageMaker or by this repository's `main` branch through OIDC.

---

## 11. Pinned tooling

**Decision.** CI pins Python 3.12 (matching the SageMaker container),
`ubuntu-24.04`, the ruff version, and its rule set (`ruff.toml`). The
pipeline installs `sagemaker<3`.

**Why.** Twice during this project, an unpinned dependency changed
underneath working code: a newer ruff release enabled more lint rules by
default, and SDK helpers differed between versions. Pinning means CI only
changes when I change it.

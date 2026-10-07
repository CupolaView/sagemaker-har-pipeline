# Data Drift Monitoring

**Built 2026-10-05 / 06** · Endpoint: `har-classifier` (real-time, data capture on)
· Script: `src/drift_check.py` · Feature definition: `src/monitor_preprocessor.py`

## Why not SageMaker Model Monitor

The original design used SageMaker Model Monitor (`DefaultModelMonitor`). The
first attempt to create a monitoring schedule failed:

> ValidationException: This operation is in maintenance mode and is not
> available to new customers.

AWS has [closed Model Monitor to new customers](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-availability-change.html).
Existing users keep it, but no new features are planned. The replacement AWS now
recommends is open-source drift detection (Evidently AI) on data-capture files,
with CloudWatch metrics and alarms. This project implements that pattern.

The design carried over unchanged. Data capture, the feature definition, and the
traffic generator all stayed the same. Only the analysis engine changed.

## Design

```
endpoint (data capture 100%) ──► S3 har/datacapture/ (JSON lines)
                                        │
training X.npy ──► 18 features ──┐      ▼
  (reference)                    ├──► drift_check.py ──► verdict (console)
captured requests ──► 18 features┘      │            ├─► CloudWatch HAR/Monitoring/DriftedFeatureShare
  (current)                             │            └─► Evidently HTML report
```

### Monitor summary features, not raw signals

Each request carries a 9 × 128 window (1,152 raw values). Testing 1,152 columns
would be noisy and meaningless. Instead, each window is reduced to **18 summary
features**: the mean and standard deviation of each of the 9 sensor channels.
A miscalibrated or failing sensor shows up exactly here, as an offset (mean)
or a gain change (std).

The feature function `window_features()` lives in one place and is used for both
the training reference and live traffic, so the definition of "normal" can't
diverge between the two.

### Detection rule

For each feature, a two-sample Kolmogorov–Smirnov (KS) test compares the
captured traffic to the training reference. A feature counts as **drifted** only
if both conditions hold:

1. **p < α / 18** (Bonferroni correction, α = 0.05 → p < 0.00278)
2. **KS statistic ≥ 0.1** (minimum effect size)

The endpoint is flagged when **more than 30% of features drift**. The 30%
default matches AWS's reference solution.

The alert decision comes from scipy's KS test, not from Evidently. Evidently's
API changes often between versions, so it only produces the visual report. An
Evidently update can break the report but not the alert.

## Calibration: the first rule nearly raised a false alarm

The first version flagged a feature at plain p < 0.05. On **healthy** traffic
(200 windows from the UCI HAR test set), it flagged 5 of 18 features. That's a
share of 0.28, just under the 0.30 threshold, so a routine fluctuation would
have triggered the alarm.

Three causes:

- **Multiple testing.** Running 18 tests at p < 0.05 produces about one false
  flag by chance alone. Most flags sat just under 0.05 (0.041, 0.040, 0.050),
  which is what chance flags look like.
- **Real but harmless differences.** The test set comes from different people
  than the training set. `body_gyro_x_mean` (p = 0.003) likely reflects how
  those subjects move, not sensor drift.
- **p-values track sample size, not importance.** Every flagged feature had a
  KS statistic of 0.13 or less, meaning a small shift. At production traffic
  volumes, p-values flag even trivial shifts, so the rule needs an effect-size
  floor.

The Bonferroni correction handles the first cause and the effect-size floor
handles the third. On the same healthy data, the corrected rule flags 0 of 18.

## Results

### Healthy traffic (200 windows from the test set)

| Rule | Drifted features | Share | Verdict |
|---|---|---|---|
| p < 0.05 (original) | 5 / 18 | 0.28 | OK, but only 0.02 below the threshold |
| Bonferroni + KS ≥ 0.1 | **0 / 18** | **0.00** | **OK** |

Largest KS statistic: 0.129 (`body_gyro_x_mean`, p = 0.00283, just above the
corrected cutoff).

### Simulated sensor miscalibration (300 windows, every value × 2.5 + 1.0)

| Feature group | KS statistic range | Drifted |
|---|---|---|
| Channel means (9) | 0.19 – 1.00 | 9 / 9 |
| Channel standard deviations (9) | 0.34 – 0.46 | 9 / 9 |
| **Total** | | **18 / 18 (share 1.00) → DRIFT DETECTED** |

The weakest drift signal (KS 0.193) is above the largest healthy-traffic signal
(KS 0.129), so the rule separates the two cleanly.

## Observations

1. **Mean features catch an offset strongly**: KS 0.76–1.0 on all six body
   channels, and 0.98–1.0 on the three body-acceleration channels. Shifting every value by +1.0 moves each window's mean, so the
   distributions barely overlap.
2. **Standard-deviation features catch a gain change more weakly**
   (KS 0.34–0.46). Training data already ranges from still (lying) to vigorous
   (walking), so a 2.5× scale still overlaps with normal activity.
3. **A monitor is least sensitive on the features that vary most naturally.**
   `total_acc_y_mean` had the weakest signal (KS 0.193). Total acceleration
   includes gravity, whose direction changes with posture, so the feature
   varies widely even in normal data. Per-feature thresholds should start
   with features like this one.
4. **Serverless endpoints can't be monitored this way.** They don't support data
   capture (see `serving.md`), which is part of why the monitored path in
   this architecture uses a real-time endpoint.

## Limitations and next steps

- **Runs on demand, not on a schedule.** Production would trigger
  `drift_check.py` from an EventBridge schedule (Lambda or a SageMaker
  Processing job), following AWS's reference solutions.
- **Only input drift is monitored.** Model-quality monitoring (accuracy
  against ground-truth labels) needs labels collected after the fact; HAR
  serving has none in this setup.
- **Reads capture files by S3 modification time.** Fine for this traffic
  level. At scale, partition and query the capture data, for example with
  Athena as in AWS's QuickSight solution.
- **Alarm:** the CloudWatch alarm on `DriftedFeatureShare > 0.3` is built in
  the next step (`docs/` dashboard section).

## How to reproduce

```
python src/deploy_endpoint.py
python src/send_traffic.py --mode normal --n 200      # wait ~5 min
python src/drift_check.py --minutes 15 --publish --report
python src/send_traffic.py --mode drift --n 300       # wait ~5 min
python src/drift_check.py --minutes 6 --publish --report
aws sagemaker delete-endpoint --endpoint-name har-classifier
```

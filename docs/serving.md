# Serving Options: Real-Time vs. Serverless

**Measured 2026-09-17**, us-west-2, same model (registry package v1, Approved),
same `inference.py`, client in Calgary (numbers include network + JSON transfer).
100 invocations per cell; first call excluded from percentiles.

## Latency (client-side, ms)

| Config | Batch | p50 | p95 | p99 | First call |
|---|---|---|---|---|---|
| Real-time (ml.m5.large) | 1  | 243 | 270 | 288  | 464 (warm-ish) |
| Real-time (ml.m5.large) | 16 | 400 | 446 | 823  | 528 |
| Serverless (2 GB)       | 1  | 259 | 298 | 354  | **4,091 (cold start)** |
| Serverless (2 GB)       | 16 | 460 | 991 | 1,245 | 654 |

### Where the time goes (real-time, batch 1)

| Layer | p50 (ms) | p95 (ms) | Share of p50 |
|---|---|---|---|
| ModelLatency (inside container: 4 handlers) | 8.97 | 14.3 | ~4% |
| OverheadLatency (SageMaker routing) | 6.74 | 31.0 | ~3% |
| Network + JSON transfer (client − server) | ~227 | ~225 | **~93%** |
| **Client-measured total** | 243 | 270 | 100% |

Source: CloudWatch `AWS/SageMaker` metrics (reported in microseconds),
09-17 test window. The network row is derived (client − model − overhead).
Percentiles don't add exactly, so the p95 remainder is approximate.

**Takeaway:** the server takes ~16 ms. More than 90% of what the client sees
is the Calgary→Oregon round trip plus ~25 KB of JSON per request. Making the
model faster would barely change user-facing latency. Moving the client
closer to the region, or shrinking the payload, would. At p95, overhead
(31 ms) exceeds model time (14.3 ms), so the tail comes from the
platform, not the model.

## Observations

1. **Warm serverless ≈ real-time** (259 vs 243 ms p50): for this model,
   steady-state latency is not the differentiator.
2. **Cold start is ~16x p50** (4.1 s): disqualifying for latency-SLO paths
   that see idle gaps; irrelevant for steady traffic that keeps it warm.
3. **Tail inflates under load on serverless** (batch-16 p95: 991 vs 446 ms).
4. **Payload transfer dominates client latency**: inference itself is
   milliseconds; a batch-16 request is ~400 KB of JSON. Production fix:
   binary content type (npy/protobuf) or gzip — declared content-types in
   the registry are where that contract would change.
5. **Serverless does not support data capture** → no Model Monitor.
   Monitoring capability alone can justify real-time for a governed system.

## Cost model

Rates: us-west-2, SageMaker AI pricing page, checked 2026-09-23.

**Real-time (ml.m5.large, $0.115/hr):** 730 hrs/month → **$83.95/month
whether it serves zero requests or millions.**

**Serverless (2 GB, $0.0000400/sec + $0.016/GB processed in and out):**

| Per-request component (batch 1) | Calculation | Cost |
|---|---|---|
| Compute | ~10 ms billed × $0.00004/s | $0.00000040 |
| Data in | ~25 KB JSON = 2.4e-5 GB × $0.016 | $0.00000038 |
| Data out | ~0.2 KB response | ≈ $0.000000003 |
| **Total** | | **≈ $0.00000078** |

**Break-even:** $83.95 ÷ $0.00000078 ≈ 108M requests/month
≈ **3.5M requests/day** (~41 req/s sustained, 24/7).

Sensitivity: if serverless bills ~50 ms per call instead of 10 ms (cold
starts, slower managed CPU), the per-request cost is ~$0.0000024 and
break-even drops to **~1.2M/day**. That's still far above the traffic a
typical HAR service would see.

**Observations:**
- Below millions of requests per day, cost doesn't favor real-time. The
  choice between the two comes down to latency (no cold start) and
  monitoring (data capture), not price.
- **JSON costs money, not just latency:** data processing is about half the
  serverless per-request cost. Sending the same window as binary float32
  (9×128×4 bytes ≈ 4.6 KB) would cut the data charge about 5×.
- The 10 ms figure assumes serverless compute time matches the real-time
  endpoint's ModelLatency p50 (8.97 ms). Serverless CloudWatch metrics
  would confirm it.

## Recommendation (hypothetical production HAR service)

- **Anything below ~1M req/day with no strict latency SLO:** serverless.
  It costs nothing when idle and pennies per thousand requests, and cold
  starts are acceptable without a tight SLO. Either accept the monitoring
  gap or send a sampled copy of the traffic to a monitored real-time
  endpoint.
- **Steady traffic with latency SLO or drift-monitoring requirement:**
  real-time + autoscaling; data capture on; Model Monitor attached
  (this repo's architecture).
- **Not built here, for completeness:** SageMaker Asynchronous Inference —
  queue-backed, for large payloads / long-running requests; scales to zero
  like serverless but tolerates minutes-scale latency.

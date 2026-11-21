# Phase 2.3 — Selective feedback and independent-seed replication

## Question

Phase 2.2 showed a substantial improvement in acceleration and jerk prediction when the six specialists exchanged embeddings, but its feedback correction **made linear and turning predictions worse**. Can the system learn to reduce feedback *only when appropriate*, and will that behavior replicate across unseen synthetic datasets?

## What is trained

A **feedback-control gate** receives public observation-derived signals from the six nodes: estimated speed, acceleration, jerk, curvature, velocity inconsistency, forecast dispersion, and the expected size of the already-trained peer correction.

The gate controls the fraction of the Phase 2.2 correction actually applied:

```text
prediction = contextual_consensus + gate(scene) * peer_embedding_correction
```

Candidate policies are **off**, **full**, and a learned sigmoid gate constrained to [0, 1]. Gate coefficients are fitted **only on the 512 training scenes**. The best policy is selected on **128 validation scenes**. Test data never enter model selection.

The system can retain full feedback if it is consistently helpful; a selective gate is **not** automatically labeled superior.

The six existing specialist readouts and the Phase 2.2 correction mechanism remain unchanged. This phase trains a small controller, not a large multimodal model, and does not create autonomous neural tissue.

## Independent datasets

The first reproducibility sweep uses five *fresh* seeds, each with:

- 512 training scenes
- 128 validation scenes
- 256 sealed test scenes
- Four balanced synthetic motion regimes: linear, accelerating, jerk, turning

Seeds: `64013, 74129, 85021, 91153, 104729`. Total across runs: **4,480 scenes** (2,560 train / 640 validation / 1,280 held-out test).

These are five independently randomized samples of the same four predefined synthetic distributions. Five runs do **not** imply real-world or out-of-distribution generalization.

## Execution topology

```text
   Five dataset preparation runners (one seed each)
                       |
   Five training runners (no sealed labels present)
       |    each runs the six *logical* specialists and
       |    trains Phase 2.2, then the Phase 2.3 controller
                       |
   Five evaluation runners (test labels only here)
                       |
   One aggregation runner
        | all five seed reports required
                       |
   summary.json + each seed-NNNNN.json
```

**Runner honesty:** Phase 2.3 uses **five parallel seed jobs**, each evaluating six specialists *within that job*. It does **not** allocate 30 simultaneous runners or replace the Phase 2.2 six-physical-runner proof. This bounded topology measures **robustness of the learned algorithm**, not inter-runner communication latency.

All jobs have read-only GitHub permissions. Missing, duplicated, unexpected or mismatched seed reports cause aggregate failure. Training and evaluation are separated by GitHub Actions artifacts: the training job downloads only observations and train/validation labels, while the evaluation job receives the sealed test labels after the model is frozen.

## Per-seed held-out comparisons

- `phase21_static`: global learned weighting, no new selective controller.
- `phase22_context`: scene-aware gate without feedback.
- `phase22_full_feedback`: full Phase 2.2 peer correction.
- `phase23_selective`: validation-selected feedback controller.

Metrics include the mean squared error (MSE) on the 256 unseen scenes for each seed, per-motion-regime MSE and average controller activation, and paired-bootstrap confidence intervals for differences between selective/full and selective/static.

The aggregate records mean test MSE across all five seeds, number of seeds selective beats full and static, mean and range of differences, and frequency of selected gate types. No benchmark automatically certifies improvement: a passing CI run only certifies execution and integrity checks.

## Run and evidence

Workflow: `.github/workflows/phase23-cross-seed.yml`

Its branch-scoped initial push starts one five-seed experiment; later pull request updates run tests without repeated expensive parallel training. Subsequent experiments can be manually started from the default branch after review.

On completion download `phase23-cross-seed-evidence-<run-id>`. The `summary.json` contains all results, and `seed-*.json` retain per-seed details.

## Before considering Phase 2.4

Keep both Phase 2.1 and Phase 2.2 pull requests intact, and stack this phase atop them. Require that results be compared under identical seed sets and label partitions. Treat improvement on five synthetic seeds as preliminary. If selective gating worsens any regime, explicitly retain those failures as counterexamples before expanding to novel motion, missing-node tests, and real sensor data.

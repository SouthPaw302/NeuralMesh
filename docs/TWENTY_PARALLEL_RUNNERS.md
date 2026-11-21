# 20 physical GitHub runners — NeuralMesh execution contract

## Exact objective

Twenty **separate GitHub-hosted Linux job executions run simultaneously** in each of two inference waves, not six jobs with twenty logical tasks per job. Every specialist has its own trained readout, sensor noise, deterministic training stream and numeric feature mask.

- **20 worker identities:** five feature specializations (position, velocity, acceleration, curvature, stabilization) × four unique replicas each.
- **20 concurrent jobs** requested in Wave 1 (`max-parallel: 20`).
- One synaptic barrier combines and verifies **exactly 20** first-round receipts.
- **20 concurrent jobs** requested in Wave 2 (`max-parallel: 20`).
- Every Wave 2 worker reads the first packet and applies feedback from **all 19 other** specialists.
- Final barrier compares consensus against independently sealed future-position ground truth.

There are **40 individual worker job executions** in the two sequential waves, **not 40 concurrent machines**. Additionally, tests/preparation/aggregation/audit jobs run outside the worker waves.

## Proof of *actual* parallel scheduling

The workflow `.github/workflows/neural-mesh-20.yml` includes `audit-parallel`. After both waves finish, it queries the current GitHub workflow run's **actual job start and completion timestamps**, checks there are 20 successful unique worker jobs for each wave, and computes maximum temporal overlap separately for each wave.

The proof is saved as `swarm20-concurrency-evidence-<run_id>/concurrency.json`.

A green `audit-parallel` requires:
- Wave 1 **peak simultaneous jobs = 20**.
- Wave 2 **peak simultaneous jobs = 20**.
- All 40 worker jobs completed successfully.

If account-wide GitHub runner concurrency is exhausted by competing workflows, 20 jobs may queue and the audit will **fail** rather than pretending all 20 ran concurrently. To aid honest measurement, workers intentionally hold their runner slots briefly after inference; this is not computational work, and concurrency is measured from GitHub's job records.

GitHub Actions scheduling and account-level limits, not the YAML `max-parallel` setting alone, decide real capacity. This distributed benchmark is high-latency and unsuitable for continuously firing neural messages.

## Message and model integrity

Every worker writes a JSON receipt containing node ID, model digest, common observation digest, stage, numeric embedding, forecast and (in Wave 2) the previous packet SHA-256. The aggregation stages fail closed on missing or duplicate workers, forged names, nonfinite predictions, mismatched model digests, inconsistent observation hashes, and missing or invalid 19-peer routing.

A sealed target is never downloaded by individual specialist jobs. Only the final evaluator receives it. No checkpoint is auto-promoted, and a successful parallelism test does **not** mean recurrent feedback improved predictive accuracy.

The original six-node code in `neural_mesh/mesh.py` remains intact for historical reproducibility and other phase modules. The 20-node execution lives in `neural_mesh/swarm20.py` to avoid silently changing old experiments.

## Execution

Use **Actions → NeuralMesh 20 Parallel Runners → Run workflow**, seed `314159` by default. The first feature-branch workflow push also runs a complete 20-worker proof. After it is merged, manually dispatch from `main`.

Inspect these artifacts in the run:
- `swarm20-model-evidence-<run_id>`: `metrics.json`, `round1.json`, `round2.json`, candidate checkpoint.
- `swarm20-concurrency-evidence-<run_id>`: precise observed job overlap.
- Wave 1/2 individual receipt artifacts: per-worker vectors and digest provenance.

## Scope

This builds a **20-way parallel distributed numerical inference mesh**. It does not yet distribute the supervised training step across 20 workers; the preparation job fits each unique readout before sending models to separate inference VMs. It is not a continuously running biological neural network, a unified 20-part LLM, or a real-time spatial world model. Those questions should not be conflated with measuring physical job parallelism.

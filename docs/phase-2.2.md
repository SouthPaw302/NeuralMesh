# Phase 2.2 — Context-dependent synapses and peer-embedding feedback

## Research question

On new synthetic motion scenes, can **input-dependent connection weights** and **feedback using peer numerical embeddings** generalize better than Phase 2.1's single global weight vector?

This is a research system based on fitted numerical readouts, **not** a simulation of biological nervous tissue, emergent consciousness, or a real video/AR model.

## What changed

Phase 2.1 fitted a **single** nonnegative six-node weight vector, and selected zero recurrent feedback. Phase 2.2 tests two separately measured mechanisms:

1. **Context-dependent routing.** Each specialist sends its 2D prediction *and* nine-element numerical observation embedding. A softmax gate starts from Phase 2.1's learned global weights, then adjusts each node's attention using five features from the current scene: a bias term, estimated speed, acceleration, forecast disagreement, and local velocity inconsistency. Parameters are learned with regularized full-batch gradient descent; validation selects regularization. This produces an actual per-scene 6-way connection distribution, rather than hard-coded labels for motion regimes.
2. **Recurrent peer-embedding correction.** The first-round embeddings are combined into a shared, 17-element peer-signal vector. A small regularized linear readout is fitted to correct remaining prediction residuals. Six second-round runners each receive the frozen synaptic packet and compute their correction using information sent by the other five nodes. Validation, **not testing**, chooses the ridge coefficient and feedback strength from `0, 0.25, 0.5, 1.0`. If validation finds no value, it can choose zero. This is a test of whether feedback is useful, not a requirement that it be used.

**Important nuance:** this is a distributed, two-stage feedback calculation. It is not continuously running interconnected neurons; GitHub Actions stages exchange cached artifacts with relatively high latency.

## Dataset and experimental boundaries

- **512 train / 128 validation / 256 held-out test** examples.
- Four stratified families: linear, accelerating, jerk, turning.
- Default seed: **2202** (a different scene split from the initial Phase 2.1 benchmark).
- Existing six numerical specialist models are held constant. The changes are learned *connections and residual routing*, not retraining a large language or vision model.
- Training never downloads test labels. The evaluator is the only job to download `test-labels.json`.
- The selected router is frozen before the held-out test is evaluated. It contains its own manifest/digests and does not write to `main`.
- All specialist receipts include versioned model and public input hashes. The coordinator recomputes actual embedding values from the public observations, rejects forged/missing/duplicated workers, and checks recurrent outputs against frozen routing.

## Held-out comparisons

All metrics are mean squared next-position error on the **same** test examples:

| Metric key | Meaning |
|---|---|
| `selected_single` | Best single node selected by Phase 2.1's validation logic |
| `phase1_fixed` | Original hand-designed prior/disagreement routing with 0.28 feedback |
| `phase21_static` | Global learned-weight router fitted on *this seed's* training split |
| `phase22_context` | Phase 2.2 scene-dependent gating, **before** feedback |
| `phase22_feedback` | Scene-dependent gating, **after** validation-selected peer-embedding residual correction |

The evaluation also produces per-regime errors, the selected feedback coefficient, mean and min/max attention per node, and paired-bootstrap 95% intervals for differences against Phase 2.1 and the no-feedback contextual model.

A CI PASS only establishes correct execution. An MSE reduction with confidence interval excluding zero supports a limited within-synthetic-distribution benefit. It does not establish live temporal learning or real-world competence.

## GitHub Actions

Run `.github/workflows/phase22-contextual.yml`. The first branch-specific workflow push runs one full bounded benchmark. Later PR pushes run unit tests only. Once the workflow is merged onto the default branch, manual `workflow_dispatch` can launch more experiments.

Jobs:
- Test and prepare
- First wave: six parallel context observers (no labels)
- Train-context: frozen gate and correction fit (train+validation labels only)
- Second wave: six parallel feedback receivers
- Evaluate: sealed test labels, measurements and artifact upload

Download `phase22-evidence-<run_id>` from Actions. `metrics.json` is the measured result, `router.json` documents the selected frozen weights/feedback, and `manifest.json` records the exact scenario split.

## Reproduce using local jobs

Use Python 3.11+ (stdlib only). Commands below reproduce the same message and evaluation protocol without claiming to simulate independent GitHub runners.

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python -m neural_mesh.phase2 prepare --seed 2202 --train-count 512 --validation-count 128 --test-count 256 --out dataset

mkdir -p observations
for n in position velocity acceleration short_horizon long_horizon stabilizer; do
  python -m neural_mesh.phase22 observe --node "$n" --public dataset/public.json --weights dataset/weights.json --out "observations/node-$n.json"
done

python -m neural_mesh.phase22 fit --public dataset/public.json --weights dataset/weights.json --receipts observations --trainval dataset/trainval-labels.json --out trained

mkdir -p feedback
for n in position velocity acceleration short_horizon long_horizon stabilizer; do
  python -m neural_mesh.phase22 feedback --node "$n" --rows trained/rows.json --router trained/router.json --out "feedback/node-$n.json"
done

python -m neural_mesh.phase22 evaluate --rows trained/rows.json --weights dataset/weights.json --router trained/router.json --round2 feedback --test-labels dataset/test-labels.json --out metrics.json
```

## What this step does not prove

Even a successful contextual network would not show that inter-node recurrent feedback is useful *unless the learned feedback beta is positive and improves held-out MSE relative to contextual routing alone*. Routing weight variation without improvement is not evidence of increased intelligence.

A single seed also cannot establish reproducible results across datasets. Subsequent work should repeat independent seeds, test shifts outside the four synthetic motion regimes, and only then introduce real spatiotemporal features and a durable state store.

# Phase 2.5 repair — first six-dataset results

## Exact execution

- [Independent GitHub workflow 44872941150](https://github.com/SouthPaw302/NeuralMesh/actions/runs/44872941150): **4/4 jobs passed**.
- Evidence artifact: `replay25-evidence-44872941150` (contains parent, candidate, validation decision, held-out results, and frozen promotion gate).
- Immutable parent model SHA-256: `8c147e3d6bad62cd9b6553738cba40dddc4f5120755273f01a366f3e235d22bc`.
- Replay-trained *research candidate* SHA-256: `e66f7fe2188539276b3415feb4f049ca5eace70309e3dd7bab596f925bf78911`.
- Training seed: `26051` (512 training / 128 validation); historical replay seed: `64013` (512 training / 128 validation).
- Training selected **ridge 0.1**, **75% historical replay weighting**, and **25% update interpolation**. Existing six specialist readouts, contextual gate and reliability policy were kept fixed.
- Held-out ID seeds: `26051, 26052, 26053` (768 unseen test scenarios).
- Held-out novel-motion seeds: `97313, 97417, 97523` (384 unseen test scenarios).
- All 1,152 held-out scenarios were scored under clean, missing and grossly corrupted node states: **3,456 measurements**.

## Corrected learning formulation

The previous Phase 2.5 update trained an unconstrained raw peer residual, which did not account for how the actual Phase 2.4 predictor scales that residual for surviving-node count, reliability damping, contextual attention, and learned selective feedback.

The revised learner fits **exactly the residual coefficient matrix used by deployed `phase24.robust_prediction`**, with tests showing algebraic equivalence in all three fault states. Training now includes historical replay examples to protect previous competencies and validation selection across both past and new sets.

### Held-out results

Mean MSE over **three independent sets per population** (lower is better):

| Population / fault | Frozen parent | Updated candidate | Change |
|---|---:|---:|---:|
| Ordinary motion, all fault states | 0.018549526 | **0.018369132** | **−0.97%** |
| Ordinary motion, six healthy nodes | 0.015688581 | **0.015559640** | **−0.82%** |
| Ordinary motion, missing node | 0.019979999 | **0.019773878** | **−1.03%** |
| Ordinary motion, gross corruption | 0.019979999 | **0.019773878** | **−1.03%** |
| Novel motion, all fault states | **0.300680438** | 0.301446192 | **+0.25% worse** |
| Novel motion, six healthy nodes | **0.291029071** | 0.292096684 | **+0.37% worse** |
| Novel motion, missing node | **0.305506121** | 0.306120946 | **+0.20% worse** |
| Novel motion, gross corruption | **0.305506121** | 0.306120946 | **+0.20% worse** |

Per-motion regime:
- Ordinary acceleration: **1.67% lower** error.
- Ordinary jerk: **1.53% lower** error.
- Ordinary linear: **1.80% higher** error.
- Ordinary turning: **0.18% higher** error.
- Novel bounce: **2.68% lower** error.
- Novel late impulse: **0.07% lower** error.
- Novel stop/start: **0.62% higher** error.
- Novel swirl: **0.16% higher** error.

The three ID datasets all improved in aggregate (**3/3 wins**), with a mean per-seed MSE improvement of about **0.945%**. However, all three novel-motion datasets worsened in aggregate by a mean **0.257%**.

### Validation and final decision

The candidate passed training/validation selection:
- New validation MSE: parent **0.016548970** → selected **0.016477642**.
- Historical replay validation MSE: parent **0.020258480** → selected **0.020078941**.

The full independent benchmark returned:

`BLOCKED_GENERALIZATION`

The candidate is therefore **not** a valid replacement for the active neural-memory checkpoint under the predeclared rule that new learning must not worsen novel-motion generalization. No weights were promoted.

The experimental sample is modest and synthetic; a small difference may reflect seed variation. Do **not** reduce the gate after seeing these test labels to make this candidate pass. If changing the architecture again, predeclare new benchmarks with unused seeds.

## Security and integrity changes

The previous single-seed Phase 2.5 promotion code is now revoked for direct approval. The manual Git registry promotion workflow is restricted to successful runs of the repaired workflow and verifies a six-dataset result, source-run provenance, exact parent/child SHA-256 lineage, and digest-linked evidence.

No model-state auto-promotion is authorized. All stored state is plain JSON; no encryption. The earlier model and existing Git `main` remain unchanged.

## Remaining research limitation

This fix repaired the objective and demonstrated bounded continual-learning improvements **within** the original synthetic distribution. The broader research requirement remains unsolved: accumulating skills without regressions on genuinely shifted motion physics. More independent domains and potentially richer representations will be needed before claiming a self-improving distributed brain.

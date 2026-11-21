# Phase 2.4 — Verified robustness and memory results

## Execution evidence

- **OOD / fault-tolerance proof:** [Actions run 44871190473](https://github.com/SouthPaw302/NeuralMesh/actions/runs/44871190473) — **13/13 jobs passed**, with **33 actual regression tests passing**.
- **Independent cross-workflow memory recall:** [Actions run 44871327795](https://github.com/SouthPaw302/NeuralMesh/actions/runs/44871327795) — **3/3 jobs passed**.
- **Primary evidence artifact:** `p24-evidence-44871190473`, with `summary.json` and three per-seed reports.
- **Frozen memory artifact:** `p24-memory-44871190473`, with `checkpoint.json`.
- **Cross-run evidence artifact:** `p24-cross-run-recall-44871327795`, with `recall-metrics.json` and `recall-provenance.json`.

## Data and selected reliability policy

Three novel-motion seeds: **48017, 58129, 68147**, 128 held-out examples per seed (384 scenes). Each was evaluated under clean, missing-node and grossly corrupted-node conditions, giving **1,152 fault-state tests**.

Four regimes not used in the prior training family: bounce, stop-start, swirl and late impulse. The six specialist models, contextual attention, Phase 2.3 selective correction, and newly calibrated fault policy were frozen before novel-motion targets entered evaluation.

The fault policy was selected using **128 in-distribution validation scenes only**:

- Outlier acceptance cutoff **1.2**.
- Retained feedback fraction **0.4**.
- Calibration mean validation MSE **0.020258479583**.

The model memory checkpoint is plain JSON, schema `neuralmesh-memory-v1`, digest:

`8c147e3d6bad62cd9b6553738cba40dddc4f5120755273f01a366f3e235d22bc`

## Three-seed held-out summary

All numbers are synthetic 2D next-position **mean squared error (MSE), lower is better**.

| Condition | Naive previous routing | Fault-aware routing | Relative reduction |
|---|---:|---:|---:|
| No faulty nodes | 0.339378903 | 0.283338230 | **16.51%** |
| One missing node | 0.406174438 | 0.299601575 | **26.24%** |
| One grossly corrupted node | 1.721213957 | 0.299601575 | **82.59%** |

The fault-aware model had lower average MSE on **all three seeds** for each fault condition. It retained all six nodes in healthy examples and five in missing/corrupt examples. Under the deliberately large injected corruption, rejection produced exactly the same forecasts as removing the node.

This demonstrates protection against the **specific severe, isolated errors injected here**. It does not establish adversarial Byzantine security, subtle data-poisoning detection, or recovery when many nodes fail.

## Behavior across unseen motion regimes

Mean held-out errors across three seeds (naive → fault-aware):

| OOD regime | Healthy | Missing node | Corrupted node |
|---|---|---|---|
| Bounce | 0.0382 → **0.0109** | 0.0643 → **0.0205** | 1.4685 → **0.0205** |
| Stop-start | 0.6636 → **0.5036** | 0.8310 → **0.5394** | 1.9567 → **0.5394** |
| Swirl | 0.2952 → **0.2570** | 0.3207 → **0.2739** | 2.0277 → **0.2739** |
| Late impulse | **0.3604** → 0.3618 | 0.4088 → **0.3646** | 1.4320 → **0.3646** |

**Important counterexample:** even with six healthy specialists, robust routing was slightly worse at predicting the unpredictable late impulse. Stop-start and late impulse have relatively high absolute errors despite aggregate improvements. The controller is not solving arbitrary physical surprises; it is limiting the damage of faulty model signals.

## Memory survival verified across two separate runs

In the initial experiment (run **44871190473**), the training job fitted and exported the complete model checkpoint as `p24-memory-44871190473`. Three independent fault-probe runners subsequently restored this state in the same workflow.

A **different workflow execution** (run **44871327795**) then:

1. Generated **64 fresh novel-motion scenes** with seed **79019**.
2. Retrieved the saved model checkpoint from the *earlier* GitHub Actions run, using GitHub's cross-run artifact download mechanism.
3. Restored the checkpoint with its exact stored SHA-256 digest and printed `PERSISTENT_MEMORY_RECALL_PASS`.
4. Ran predictions without retraining.
5. Scored results in a separate evaluation job with held-out targets.

The recall run's measured MSE:

| Condition | Naive | Restored fault-aware memory |
|---|---:|---:|
| Healthy | 0.332219059 | **0.276653902** |
| Missing | 0.388283865 | **0.292372896** |
| Corrupted | 1.824768280 | **0.292372896** |

This is a verified **cross-run model checkpoint**, not an AI that independently remembers conversations or learns continuously. Its state was unchanged between runs. GitHub Actions artifact expiration still limits durability (configured retention **30 days**); a reviewed permanent checkpoint store and a policy for authorized memory updates remain future work.

## Interpretation

**Achieved:** independent OOD generation, frozen model evaluation under missing/corrupt nodes, validation-selected fallback behavior, digest-verifiable model-memory snapshots, exact state recovery across separate GitHub runner lifetimes **and workflow executions**, and measured per-regime failure cases.

**Not achieved:** biological neural-network simulation, a conscious brain, multimodal AR/video perception, subtle adversarial resistance, distributed tensor training, long-term uninterrupted storage, or autonomous online synaptic learning. No checkpoint was auto-merged into `main`, and no other repository was modified.

## Phase 2.5 direction

A sensible next experiment is a **reviewed durable memory registry**, with branching/version lineage and authorized state promotion, plus intermittent faults, multi-node failures, and genuine video/spatial features. Test resource usage and degraded accuracy before scaling beyond the bounded synthetic mesh.

# Phase 2.7 — Six-frame temporal observability results (failed acceptance)

## Verified execution

- [GitHub Actions run 44875873428](https://github.com/SouthPaw302/NeuralMesh/actions/runs/44875873428).
- **66 of 66 unit tests passed.** Preparation, training and independent sealed evaluation jobs completed successfully.
- The separate **scientific acceptance job failed**, correctly reflecting the actual model outcome.
- Evidence artifact: `t31-evidence-44875873428`, containing `training-decision.json`, `temporal.json`, `evaluation.json` and `quality-gate.json`.
- Original parent model: `8c147e3d6bad62cd9b6553738cba40dddc4f5120755273f01a366f3e235d22bc`. Frozen Phase 2.6 physics adapter: `f0223f0d6f3cdc2813d583ffb938c64ed11db46579f381d54c9207908f579a97`.
- No state was promoted or committed into model memory. `main` is unchanged.

## Measured information improvement: six observations help identify motion type

Two **identical nearest-centroid classifier algorithms** were trained and evaluated on the same newly generated six-family synthetic scenes. They differ only in the available temporal observation features.

| Motion-family recognition | Four-frame features | Six-frame features |
|---|---:|---:|
| Overall validation accuracy | **38.19%** | **57.81%** |
| Linear | 14.58% | **59.38%** |
| Accelerating | 25.00% | **46.88%** |
| Jerk | 40.63% | **42.71%** |
| Turning | 55.21% | **65.63%** |
| Bounce | 32.29% | **62.50%** |
| Swirl | 61.46% | **69.79%** |

**This is a substantial information-quality result**: the additional two observed frames improved family identification by 19.62 percentage points on this validation cohort, and increased linear recognition by 44.79 percentage points.

It does **not** prove that the problem is universally solved, nor that these synthetic labels are directly transferable to real video.

## Held-out numerical forecasts

A separate evaluator scored **1,536 fresh synthetic scenes** across three seed cohorts for each of familiar, bounce/swirl and never-trained stop/start/late impulse motion families. Each was evaluated under healthy, one-missing and one-strongly-corrupted node conditions.

The original parent and original physics adapter both used the **same last four observations** to forecast the identical future target (time 6). The temporal router had access to all six observations, but validation did not select an acceptable learned trust policy. It therefore returned the unchanged parent.

| Fresh held-out population | Protected parent MSE | Original adapter MSE | Selected temporal policy MSE |
|---|---:|---:|---:|
| Familiar motion | 0.0259954 | **0.0247956** | 0.0259954 |
| Bounce/swirl | 0.1245194 | **0.0996638** | 0.1245194 |
| Never-trained stop/start and late impulse | 0.2108482 | **0.2074226** | 0.2108482 |

**Result: `NO_VALIDATED_SIX_FRAME_GATE`.** The learned policy did not pass the preregistered validation criterion that old linear/turning skill must not worsen while new-family performance improves. Its absence is explicitly tracked as the parent fallback. The final acceptance check is red by design.

One important dataset-specific detail: in this *new* six-frame test distribution, the original adapter's familiar linear error was **0.008226** versus parent **0.008150** (approximately 0.93% worse), while turning was slightly better. The original Phase 2.6 results had a different regression pattern. This variation emphasizes why any one seed is insufficient to certify non-forgetting.

## What we actually fixed versus what remains

**Implemented and verified:**
- Genuine six-observation temporal data path with a frozen four-observation original predictor using the last four inputs.
- Explicitly separated supervised training/validation labels and sealed test targets, with SHA-256 source model checks.
- Quantified the increased discriminative information available in earlier frames.
- Strict fallback to the immutable parent when no validated improvement exists.
- A separate scientific acceptance job that returns failure even when software tests are all green.
- Reporting of noisy-node abstentions rather than silently deleting failure cases.

**Unresolved:** Reliable task-conditional correction selection that improves unfamiliar physics without degrading earlier linear and turning competence. A simple nearest-centroid class prediction still confuses many trajectories, even with six frames. These data support investigating **better temporal forecasting models and uncertainty calibration**, not repeatedly adding another arbitrary static gate.

This does not establish continuous neural plasticity, a self-improving brain, live video/AR, biological neuron emulation, or real-world generalization.

## Research decision

**Do not promote this candidate.** Retain the protected parent and original Phase 2.6 adapter as separate research snapshots. Future model changes should use richer sequence-conditioned predictors (for example, state-space filtering or sequence learning), properly calibrated uncertainty/abstention, and newly preregistered seeds. Previously exposed held-out labels must not be recycled to tune a fourth classifier.

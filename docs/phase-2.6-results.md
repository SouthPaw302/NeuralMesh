# Phase 2.6 — Held-out cross-domain results

## Verified execution

- [GitHub Actions run 44873535607](https://github.com/SouthPaw302/NeuralMesh/actions/runs/44873535607): **4/4 jobs passed**, including the full 59-test regression suite at initial commit.
- Evidence artifact: `transfer26-evidence-44873535607` with the frozen adapter, validation decision, sealed evaluation, and gate result.
- Parent checkpoint remains: `8c147e3d6bad62cd9b6553738cba40dddc4f5120755273f01a366f3e235d22bc`.
- Adapter proposal hash: `f0223f0d6f3cdc2813d583ffb938c64ed11db46579f381d54c9207908f579a97`.
- **No model memory was overwritten or merged into main.** The adapter is a research sidecar, not a promoted version of the original model.
- Dataset protocol and predeclared acceptance criteria: [phase-2.6.md](phase-2.6.md).

## Intervention

We changed **the predictive representation**, not merely the memory infrastructure. A small learned softmax combines the protected original six-specialist forecast with four physically motivated extrapolators: constant velocity, acceleration, circular rotation, and restart. The gate is computed from observed kinematics and does **not** receive the motion regime label or next-position target at inference.

Training included 1,024 familiar motion scenes (two ID cohorts), 256 newly introduced bounce/swirl scenes, and separate validation cases. Motion physics for stop/start and late impulse were **not present in any training/validation samples**.

Validation selected:
- ID-training relative weight **0.55**
- Softmax regularization **0.0005**
- Interpolation strength **0.25** between parent and learned physical mixture

The parent remained a fall-back option if no candidate satisfied validation preservation.

## Headline held-out result

All figures are mean squared 2D next-position error (MSE; **lower is better**). Each test scenario was scored under healthy, missing-one-node, and strongly corrupted-one-node conditions.

| Domain | Held-out scenarios | Parent MSE | New adapter MSE | Relative error reduction | Seeds improved |
|---|---:|---:|---:|---:|---:|
| Familiar ID motion | 768 | 0.019369557 | **0.018996681** | **1.93%** | 3/3 |
| Bounce/swirl, fresh seeds | 384 | 0.110586612 | **0.088062658** | **20.37%** | 3/3 |
| **Never-trained** stop/start and late impulse | 384 | 0.435666193 | **0.422847674** | **2.94%** | 3/3 |

Total: **1,536 distinct test scenes**, **4,608 scenario/fault evaluations**.

**Result of the originally specified acceptance gate: `TRANSFER_SUPPORTED`.** It required nondegradation of aggregate mean on never-trained regimes, at least 2.5% improvement on newly exposed physics, and protection of aggregate familiar motion plus per-seed fault-state tolerances. The strict threshold was not changed after the held-out answers were observed.

## Per-family results — importantly mixed

The aggregate row conceals differences between motion regimes.

| Family | Parent MSE | Adapter MSE | Relative change |
|---|---:|---:|---:|
| Familiar linear | **0.008407** | 0.008870 | **+5.52% worse** |
| Familiar accelerating | 0.022190 | **0.021270** | −4.14% better |
| Familiar jerk | 0.037039 | **0.035894** | −3.09% better |
| Familiar turning | **0.009843** | 0.009952 | **+1.11% worse** |
| Newly learned bounce | 0.014443 | **0.010392** | −28.05% better |
| Newly learned swirl | 0.206730 | **0.165733** | −19.83% better |
| Never-trained stop/start | 0.505786 | **0.480947** | −4.91% better |
| Never-trained late impulse | 0.365546 | **0.364749** | −0.22% better |

**Scientific qualification:** We have demonstrated aggregate cross-family transfer in one synthetic experimental design, including nine fresh seed-level wins, but **not flawless retention**. Particularly, familiar linear motion degraded by about 5.5% despite the overall familiar-distribution gain. This falls outside the stronger aspiration of preserving every previously learned capability.

Do not conceal the linear regression, retrofit thresholds to the observed answers, or declare the neural system generally self-improving. The next separate experiment should preregister stricter per-regime retention limits on **new sealed seeds**.

## Seed-level evidence

Each row below is the mean over its associated motion class and all three node-fault conditions.

| Test cohort | Parent MSE | Adapter MSE | Relative reduction |
|---|---:|---:|---:|
| Familiar seed 26121 | 0.018294 | 0.017773 | 2.85% |
| Familiar seed 26122 | 0.019513 | 0.019272 | 1.24% |
| Familiar seed 26123 | 0.020302 | 0.019946 | 1.75% |
| Known-new seed 46121 | 0.111260 | 0.086437 | 22.31% |
| Known-new seed 46122 | 0.092321 | 0.073770 | 20.09% |
| Known-new seed 46123 | 0.128179 | 0.103980 | 18.88% |
| Never-trained seed 46131 | 0.468503 | 0.455367 | 2.80% |
| Never-trained seed 46132 | 0.442743 | 0.427608 | 3.42% |
| Never-trained seed 46133 | 0.395752 | 0.385568 | 2.57% |

Seed IDs, raw per-seed/per-regime/fault metrics, and the serialized learned softmax can be recovered from the original run's `transfer26-evidence-44873535607` artifact.

## What is and is not proven

**Demonstrated in this bounded experiment:** A frozen six-node forecasting mesh can gain predictive value from multiple fixed physical extrapolation experts and a learned context-dependent gate, while transferring some of those gains to two **never-trained** synthetic motion families.

**Not proven:** real-world video competence, general intelligence, live biological-style plasticity, real spatial world modeling, robust performance under subtle/coordinated failures, or zero forgetting of every earlier task. The late impulse contains independent randomness introduced only after the last observed frame, so it cannot be fully inferred from the available observations.

**Memory status:** The protected parent remains active. The learned adapter remains an unpromoted research artifact. A future independent confirmation with stricter predeclared per-regime retention is advisable before considering a reviewed adapter memory registry entry.

# Canonical self-contained replication — measured result

**Experiment run:** [44876631145](https://github.com/SouthPaw302/NeuralMesh/actions/runs/44876631145)

## Engineering status

- All source-code regression tests passed.
- The original Phase 2.4 model was **retrained deterministically from the repository**, without downloading an expiring GitHub Actions artifact.
- Its SHA-256 was exactly verified:
  `8c147e3d6bad62cd9b6553738cba40dddc4f5120755273f01a366f3e235d22bc`
- The physics-expert model was retrained without changing its source, hyperparameters or training seeds; it reproduced adapter SHA-256:
  `f0223f0d6f3cdc2813d583ffb938c64ed11db46579f381d54c9207908f579a97`
- A sealed evaluator then tested nine **previously unused** seeds. Its data-preparation, training and evaluation jobs all succeeded.
- **The stricter, separate scientific acceptance job FAILED**, correctly preserving the original model.

## Blind holdout results

All values are 2D next-position mean squared error (MSE, lower better), averaged across three independent seeds and clean/missing/corrupted-node states.

| Population | Parent MSE | Adapter MSE | Change | Winning seeds |
|---|---:|---:|---:|---:|
| Familiar ID motion | 0.019681442 | 0.019162858 | **2.63% better** | 3/3 |
| Bounce + swirl, known training families | 0.118913190 | 0.094286381 | **20.71% better** | 3/3 |
| Stop/start + late impulse, unseen training families | 0.424179135 | 0.414526697 | **2.28% better** | 3/3 |

All nine independent **aggregate seed-level** results improved, but this does **not** guarantee that individual older skills were retained.

## Per-motion behavior

| Motion family | Parent MSE | Adapter MSE | Change |
|---|---:|---:|---:|
| Familiar accelerating | 0.021902 | 0.021072 | **3.79% better** |
| Familiar jerk | 0.038597 | 0.036980 | **4.19% better** |
| **Familiar linear** | **0.008705** | 0.008905 | **2.31% worse** |
| **Familiar turning** | **0.009523** | 0.009694 | **1.80% worse** |
| Bounce | 0.017295 | 0.013703 | **20.76% better** |
| Swirl | 0.220532 | 0.174869 | **20.71% better** |
| Late impulse | **0.387435** | 0.388065 | **0.16% worse** |
| Stop/start | 0.460924 | 0.440989 | **4.33% better** |

These are aggregate scores across the three independent datasets for each population. The late impulse was never a known training family, and small regressions on an intrinsically unpredictable impulse are distinct from forgetting a learned skill.

## Acceptance decision

**SCIENCE: FAIL / STOP.**

The model passed the previous, more permissive `TRANSFER_SUPPORTED` summary but failed the new explicitly preregistered **no linear or turning regression** requirement. The acceptance job's actual error message was:

`AssertionError: STOP: prior linear skill regressed`

This is the exact result that previously launched repeated corrective gates and new phases. Under the new fixed contract, **do not initiate another auto-fix or retune against these observed seeds.**

Preserve the original checkpoint. The adapter remains a research sidecar. Any later experimental design would need a distinct, falsifiable hypothesis and fresh, precommitted holdout seeds, rather than an automatic next numbered phase.

## Durable fix delivered

The repository's trained numerical state now can be **reproduced from source without any old GitHub artifact**. The stack of seven tested PRs has been merged into `main`, and rejected drafts #8–#10 are closed but still accessible. The default system exposes a single canonical experiment and a separate six-runner orchestration proof.

This is a reproducibility and development-process repair. **It is not a claim of solved catastrophic forgetting, general intelligence, real video understanding, or autonomous self-improvement.**

Evidence archive: `transfer26-evidence-44876631145` attached to the Actions run, with per-seed/per-regime reports and checkpoint/decision fingerprints. No automatic model or checkpoint promotion occurred.

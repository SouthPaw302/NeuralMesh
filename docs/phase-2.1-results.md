# Phase 2.1 — First sealed evaluation result

**Run:** https://github.com/SouthPaw302/NeuralMesh/actions/runs/44869721660
**Dataset seed:** `314159`  
**Version:** `neural-mesh-phase-2.1`  
**Splits:** 512 train / 128 validation / 256 held-out test  
**Regimes:** linear, accelerating, jerk, turning (64 test examples each)  
**Workflow result:** 16/16 jobs successful; 13 tests executed and passed.  
**Artifact:** `phase21-evidence-44869721660`

## Held-out results (MSE; lower is better)

| Predictor | MSE |
|---|---:|
| Validation-selected single specialist (`position`) | 0.028426125654 |
| Equal-weight consensus | 0.021373545988 |
| Phase 1 fixed routing, first round | 0.019439032482 |
| Phase 1 fixed routing, feedback round | 0.019527421652 |
| Phase 2.1 learned routing, first round | **0.019121800060** |
| Phase 2.1 learned routing, feedback round | **0.019121800060** |

- Learned vs Phase 1 fixed feedback MSE difference: **-0.000405621592** (about 2.08% lower error).
- Paired bootstrap 95% CI: **[-0.000753731084, -0.000027758820]**.
- Learned vs validation-selected single MSE difference: **-0.009304325594** (about 32.73% lower error).
- Paired bootstrap 95% CI: **[-0.012038990448, -0.006891653404]**.

For this seed and synthetic scenario family, the learned ensemble shows a small measured improvement. Do not extrapolate to unseen real-world data without additional experiments.

## Selected routing, before test labels were available

Validation selected ridge **0.1**, per-example disagreement coefficient **gamma 0**, and peer feedback fraction **alpha 0**.

Node weights (in `mesh.NODES` order):

| Node | Learned weight |
|---|---:|
| position | 0.317582515657 |
| velocity | 0.107684940564 |
| acceleration | 0.095301162927 |
| short_horizon | 0.173204538706 |
| long_horizon | 0.065435633572 |
| stabilizer | 0.240791208574 |

**Important:** despite successful distributed feedback-job execution, the validation-selected policy set `alpha=0`. Thus the second-round predictions were the same as first-round predictions. Phase 2.1 demonstrates **cross-scenario fitting of synaptic aggregation weights** and a holdout benefit—not a demonstrated benefit from recurrent information exchange. The next phase should target learned, scene-dependent routing and meaningful positive recurrent feedback with robust ablations.

## Per-regime held-out MSE (learned vs fixed second round)

| Motion regime | Learned | Phase 1 fixed |
|---|---:|---:|
| linear | 0.005547827141 | 0.005931142847 |
| accelerating | 0.023088720475 | 0.023663324515 |
| jerk | 0.039055665383 | 0.039560327979 |
| turning | 0.008794987242 | 0.008954891268 |

The shared benchmark uses one seed and a predefined synthetic distribution. Confidence intervals summarize scene-level resampling, not uncertainty across entire model-development choices or all possible motion distributions. The test split did not influence hyperparameter choice in the workflow.

## CI correction

The original Phase 1 workflow had an obsolete test file pattern (`test_neural_mesh.py`) even though tests now live in `tests/test_mesh.py`. Phase 2.1 corrects it to `test_*.py`. The runner log confirms **13 actual tests** executed and passed.

## Acceptance boundary

The experiment is technically complete and reproducible. The results do not establish biological neural behavior, consciousness, real-world perception, large-model distributed training, or real video augmentation. No production workflow or external media was modified.

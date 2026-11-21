# Retention repair attempts — failure ledger and research boundary

**Do not interpret passing GitHub Actions unit tests as successful self-improvement.**

The frozen Phase 2.6 physics adapter achieved a measurable synthetic transfer gain, but familiar linear and turning trajectories were worse than the Phase 2.4 parent on its first held-out tests. Three subsequent regression-control prototypes were implemented, tested and evaluated on fresh seeds. **None qualified for model promotion.** All retained the exact original parent fallback.

## Attempt 1 — Smooth logistic trust gate

Implementation `neural_mesh/safety27.py`. Source adapter and parent verified by SHA-256.

- Full experiment: [run 44874638500](https://github.com/SouthPaw302/NeuralMesh/actions/runs/44874638500).
- 67 real regression tests passed after a corrected CLI shadowing issue.
- Validation train/validation seed `27211`, bounce/swirl seeds `47211,47212`.
- No trust-gate candidate passed preregistered old-skill and new-physics validation constraints.
- Returned the unchanged parent, **not** an improved model.

## Attempt 2 — Supervised observed-kinematics competence tree

Implementation `neural_mesh/competence28.py`.

- Full experiment: [run 44874962805](https://github.com/SouthPaw302/NeuralMesh/actions/runs/44874962805).
- 74 real tests passed.
- Independent calibration seed `28211`, known motion training/validation `48211,48212`.
- All 36 decision-tree candidates failed the validation guard. In the recorded validation suite, the frozen Phase 2.6 adapter improved linear by about 0.54%, but worsened turning by approximately 1.03%. The 50%-attenuated adapter still worsened turning by about 0.18%.
- The local tree optimizer selected effectively global trust values instead of separating these cases, so it did not solve forgetting.

## Attempt 3 — Explicit curvature-risk veto

Same competence module plus observed-angle / angle-to-acceleration veto. The predicate never consults true regime labels at inference.

- Full experiment: [run 44875197447](https://github.com/SouthPaw302/NeuralMesh/actions/runs/44875197447).
- 75 real tests passed.
- **Fresh** calibration seed `29211` and known motion training/validation `49211,49212`; no reused sealed answers.
- None of 276 tree/risk policies passed validation.
- The validation parent linear MSE was 0.00656838; many curvature-risk rules worsened it by more than 2%, even while preserving useful known-new motion. Example: best known-physics policy yielded known-new MSE 0.09152771 versus parent 0.11054366, but linear MSE 0.00694743 (5.77% worse).
- An attempted global trust compromise did not eliminate the old-skill regression.

## Root-cause hypothesis and next scientific action

These failures establish that **the current tested gating model classes and observable features have not separated beneficial from harmful corrections well enough**. They do **not** establish mathematical impossibility or universal incompetence. The four observed, noisy 2D motion frames may be insufficient to identify all motion regimes reliably.

The `neural_mesh/diagnose30.py` diagnostic is specifically designed to answer whether:

1. Train-fitted, regime-oracle optimal trust coefficients are stable on a new independent validation sample.
2. An **impossible-to-deploy** regime-aware oracle, with access to truth labels, could meet the desired MSE constraints in principle.
3. The observed kinematic features can classify motion families on independent validation examples without future targets.

If even an oracle fails to retain skills, more information, a fundamentally different physics model, or a revised scientific goal is needed. If the oracle works but observation classification is weak, the next experimental intervention should improve **sensor information and temporal representation**, not add another arbitrary hyperparameter gate.

## Enforced acceptance semantics

`neural_mesh/research_gate.py` rejects any non-`RETENTION_REPAIR_SUPPORTED` status or inconsistent evaluation SHA-256 digest. A dedicated GitHub Actions `acceptance` job is intentionally **red** when the candidate fails scientific acceptance. Successful unit tests and completed data collection remain separately visible. This prevents the project's previous habit of describing scientific failure as a successful phase.

The immutable Phase 2.4 parent remains unchanged and no experimental adapter has been automatically promoted. No GitHub workflow in this repair has permission to merge code or alter model memory.

All methods and data remain synthetic research. The never-trained late impulse includes stochastic changes after the observation window that cannot be deduced from available inputs.

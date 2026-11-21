# Phase 2.5 repair — match deployed inference, retain old knowledge, prove generalization

## Why the original update failed

The first Phase 2.5 updater optimized the raw Phase 2.2 17-feature feedback residual against next-position targets. But the deployed Phase 2.4 predictor **does not add that raw residual directly**. It first computes a robust consensus over surviving nodes, then multiplies the feedback by the Phase 2.2 beta, Phase 2.3 selective activation, Phase 2.4 reliability feedback scale, and proportion of surviving nodes. Fitting the wrong objective allowed validation improvements to disappear (or reverse) once applied to the deployed system.

The old approval code was also too permissive: it could label a candidate `PROMOTION_ELIGIBLE` even if its held-out accuracy *decreased*, provided the degradation was within a tolerance.

## Actual repair

### 1. Learn the exact deployed calculation

The repaired `neural_mesh.replay25` first computes a feature vector for the complete deployed equation:

```text
robust_forecast = consensus_of_consistent_nodes
                + frozen_feedback_beta
                * chosen_selective_feedback_activation
                * reliability_feedback_scale
                * active_node_fraction
                * dot(peer_feature_vector, learned_residual_weights)
```

The model trains only the final residual weights. Outlier rejection, attention, and frozen selection are computed exactly as in the deployed `phase24.robust_prediction` path. Tests assert algebraic equivalence in clean, missing, and corrupted-node conditions to a very small numerical tolerance.

### 2. Rehearse the previous training distribution

Instead of fitting only the new cohort, every update learns from:
- A fresh, independent synthetic learning cohort (`seed=26051`).
- A replay cohort from the previous learned memory's training population (`seed=64013`).

The trainer evaluates a sweep of 5 prior-regularization strengths × 3 replay fractions × 3 update steps; the parent remains an explicit no-change fallback. The regression-safe candidate is selected *only on fresh and replay validation samples*. It must improve fresh validation while preserving historical validation outcomes and individual fault conditions. Any candidate that does not pass leaves the parent unchanged.

This is **replay-based supervised continual learning**, not autonomous unsupervised adaptation.

### 3. Evaluate genuinely unseen datasets

The workflow creates six fresh independently seeded held-out datasets (and never downloads their target coordinates in the fitting job):

| Split | Seeds | Holdout size |
|---|---|---|
| In-distribution motion | 26051, 26052, 26053 | 3 × 256 |
| Novel-motion physics | 97313, 97417, 97523 | 3 × 128 |

All are scored under healthy, one-missing, and one-grossly-corrupted specialist conditions; 3,456 scene/fault cases in total. The new and old models use **exactly the same observations and targets**.

### 4. Fail closed on genuine generalization

The new memory cannot be promoted unless **all** predeclared conditions hold:
- At least two of three fresh ID datasets improve; average ID relative MSE improves by at least **0.5%**.
- Average novel-motion MSE must **not** worsen.
- Average healthy-node error improves on the independent ID datasets.
- None of the fault-state MSEs worsens by more than **0.5%** on any individual dataset.
- No individual motion regime worsens by more than **5% (ID)** or **8% (novel)** on any dataset.

If these conditions are not met, output status is `BLOCKED_GENERALIZATION` or `BLOCKED_VALIDATION`, while the complete research evidence is retained for inspection. This status is not made to look like successful intelligence growth simply because a GitHub job passed.

The former single-seed `phase25.check_promotion` gate is **revoked**. Manual promotion can now accept only a repaired six-dataset evidence bundle with matching parent/candidate SHA-256s, a digest-bound validation decision, checked held-out metrics, and the correct source-workflow provenance.

### 5. Immutable memory remains review-only

The repaired workflow `.github/workflows/repair25-replay.yml` runs one bounded experiment on its feature branch and unit tests on subsequent pull request updates. It never pushes or promotes weights.

`.github/workflows/phase25-promote-reviewed.yml`, after the code reaches the default branch, requires explicit manual dispatch, source run ID, candidate digest and `OPEN_REVIEW_PR_ONLY`. It verifies the **repaired** source workflow and evidence, then opens a separate Git pull request for an immutable memory version. Nothing auto-merges.

## Limitations

- These are still simple synthetic 2D forecasting models, not real video or biological neurons.
- Replaying earlier examples is an established continual-learning technique, but alone does not establish cumulative general intelligence.
- A blocked proposal is a **successful safety decision**, not a successful learning generation.
- This benchmark is only six predeclared seeds, and its held-out labels must not be reused to tune the same experiment repeatedly.
- Durable persistence requires a reviewed and merged registry PR. The parent checkpoint is unchanged.

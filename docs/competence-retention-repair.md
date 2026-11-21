# NeuralMesh — competence-aware routing repair (v2)

## The failure that motivated this design

The Phase 2.6 physics adapter improved many kinds of synthetic motion, but damaged **linear** and **turning** forecasts relative to the protected parent. The first retention repair (`neural_mesh.safety27`) learned one smooth trust function across all motion and failed its validation gate. That first attempt retained the parent and did not promote its rejected gate. Its observations **do not become training data** for this new attempt.

The second attempt models **where the physics adapter is competent** with a small, directly trained decision tree over observed motion features. Each leaf chooses how much of the original adapter correction to trust. The root model, original learned physics adapter, and all six specialist readouts remain frozen.

## Mathematical method

Let `p` be the protected Phase 2.4 prediction, `a` the frozen Phase 2.6 physics-expert forecast, and `x` an 11-element feature vector obtained from four observed frames and the parent–adapter forecast difference. The new model predicts:

```text
prediction(x) = p + trust_tree(x) * (a - p)
```

Each decision-tree leaf fits an optimal constrained trust fraction:

```text
trust_leaf = clamp(
  Σ (weighted dot(a-p, target-p)) /
  (Σ (weighted ||a-p||²) + ridge),
  0, 1
)
```

A deterministic greedy tree chooses feature thresholds that reduce this supervised training loss. The features include observed acceleration, curvature, speed consistency, node disagreement, and correction magnitude. Training uses higher weights for old **linear and turning** examples to explicitly penalize forgetting. The true regime label is used to compute **training weights and evaluation summaries only**, never passed to the prediction function; that behavior is unit-tested.

The tree's candidates are limited to depth `2,3,4`, leaf sizes `90,150`, ridge parameters `0.0001,0.002`, and interpolation factors `0.50,0.75,1.0`. Validation alone selects a candidate, or returns the **exact original parent** when none qualifies.

## Validation acceptance (before sealed testing)

On new, independent in-distribution training/validation data and separately seeded bounce/swirl training and validation, a candidate must:

- Improve overall original-motion MSE by at least **0.1%** versus parent.
- Show **no linear or turning validation regression** versus parent.
- Limit accelerating or jerk degradation to 2%.
- Limit missing/corrupt/healthy fault-state degradation to 1%.
- Improve bounce/swirl validation MSE by at least **10%** compared with parent.

These thresholds are preserved from the first safety attempt, not relaxed after failure.

## Completely new held-out test

**Training/validation seeds:** original-motion `28211`; bounce/swirl training `48211`; bounce/swirl validation `48212`.

**Sealed holdout seeds, generated before training:**

| Dataset family | Seeds | Scenes |
|---|---|---:|
| Familiar linear/accelerating/jerk/turning | `28221, 28222, 28223` | 768 |
| Newly introduced bounce/swirl | `48221, 48222, 48223` | 384 |
| Never-trained stop/start and late impulse | `48231, 48232, 48233` | 384 |

Each scene is scored under clean, one-missing, and one-grossly-corrupted node conditions: **4,608 separate fault-state forecasts**.

The held-out decision `RETENTION_REPAIR_SUPPORTED` requires all of the following:

1. Linear and turning mean MSE **no worse than parent** (only floating-point tolerance), and accelerating/jerk per-regime degradation no higher than 2%.
2. Overall familiar motion improves.
3. The new bounce/swirl group improves at least 15% versus parent, remains within 10% of the original Phase 2.6 adapter's MSE, and improves on at least two of three independent seeds.
4. The never-trained group improves at least 0.5% versus parent and remains within 2% of the original adapter's MSE.
5. Every familiar-domain fault-state MSE remains within 2% of parent on each test seed.

These are **research-acceptance conditions**, not a guarantee against all forgetting or a biological-learning criterion. If the gate rejects the model, no memory is promoted and the failed result is retained.

## GitHub execution

Workflow: `.github/workflows/competence28-proof.yml`. The first branch-specific push runs tests, preparation, competence fitting, and independent sealed evaluation. Subsequent PR updates execute offline tests only.

The fitting job downloads the **exact previously published parent and adapter checkpoint** from Phase 2.6 run `44873535607`. Only calibration targets are accessible to fitting; nine sealed test target bundles are downloaded **only** by the evaluator after the policy is frozen. The final artifact includes the candidate policy, source checksums, validation decision, all nine seed scores, per-motion family regressions and a bounded gate report.

All state is plain JSON and protected by SHA-256 digests. No automatic mutation, checkpoint promotion, hidden encryption, or changes to `main`.

## Important limitations

This remains a synthetic 2D forecasting model with simple supervised physics experts and a small competence tree. It does not learn by experiencing real camera video or prove general artificial intelligence. The held-out novelty families are still synthetic, and stochastic late impulses after the final observation cannot be predicted from independent random information. The benchmark is predeclared and must not be repeatedly tuned using already-revealed sealed targets.

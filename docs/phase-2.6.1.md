# Phase 2.6.1 — Fix old-skill regression without sacrificing physics transfer

## Problem and protected reference

The Phase 2.6 physics-expert adapter improved aggregate performance in three held-out distributions but worsened the **linear** (5.52% higher MSE) and **turning** (1.11% higher MSE) motion families. Passing a global regression threshold did not meet the stronger scientific goal of preserving individual prior competencies.

This repair **does not retrain the Phase 2.6 adapter**, alter the Phase 2.4 parent, or promote either checkpoint. It reads the exact original artifacts from [Phase 2.6 run 44873535607](https://github.com/SouthPaw302/NeuralMesh/actions/runs/44873535607), verifying parent SHA-256 `8c147e3d6bad62cd9b6553738cba40dddc4f5120755273f01a366f3e235d22bc` and adapter SHA-256 `f0223f0d6f3cdc2813d583ffb938c64ed11db46579f381d54c9207908f579a97`.

## Intervention: a learned safety gate

A new small logistic model learns a **trust fraction 0–1** between the parent and the existing adapter. Its input features come only from the four observed frames and the already computed predictions. It sees speed, acceleration, turning angle, speed ratio, directional consistency, node disagreement, a few nonlinear motion interactions, and the magnitude of the parent–adapter correction. It is trained directly on squared next-position error using the **training labels only**.

The architecture is:

```text
Six node observations → robust parent forecast ─────────────────┐
                        ↘ existing frozen physics adapter      |
                                                                  |
Observed kinematics + parent/adapter correction → learned trust  |
                                                                  |
Final forecast = parent + trust(context) × (adapter − parent) ───┘
```

The true `scene["regime"]` label is used **only in training-example weighting, validation safety checks and final reporting**. It is not passed into the runtime trust function; the test suite confirms changing the regime label does not change the forecast.

For old-skill safety, supervised fitting gives additional emphasis to linear and turning training examples. Validation chooses among 24 candidate configurations (three old-skill weights, two regularizers, two starts, two minimum trust floors), using all familiar motion and bounce/swirl validation examples. It requires:

- **No linear or turning MSE regression** relative to the protected parent on ID validation.
- Aggregate ID validation improvement of at least 0.1%.
- No individual healthy/missing/corrupted fault-state regression above 1%.
- No accelerating or jerk regression greater than 2%.
- At least 10% improvement on separately generated bounce/swirl validation.

If nothing qualifies, the model exports **the unchanged parent as the fallback**, rather than inventing a successful new learning generation.

## Fresh sealed benchmark

No holdout seed from Phases 2.1–2.6 is reused here. The first experiment preregisters the following seeds **before test exposure**:

- ID test: `27221, 27222, 27223`, 256 scenes each (768).
- Known-new physics (bounce and swirl): `47221, 47222, 47223`, 128 each (384).
- Completely unknown-to-original-training physics (stop/start and late impulse): `47231, 47232, 47233`, 128 each (384).

The trust gate is trained on new ID training/validation seed `27211`, known bounce/swirl training seed `47211` and validation seed `47212`. No unknown-family examples enter training. The existing Phase 2.6 adapter is preserved.

All **1,536** held-out scenes are separately evaluated under clean, one-missing, and one-strongly-corrupted-node conditions: **4,608 fault-state observations**.

We compare **three** frozen methods on identical input/target pairs: the original protected parent, the original Phase 2.6 adapter, and the new safety-routed adapter.

The test gate reports `RETENTION_REPAIR_SUPPORTED` only if all predeclared conditions pass:

1. Linear and turning **aggregate per-regime MSE not worse** than the protected parent, with accelerating and jerk regressions no greater than 2%.
2. Aggregate familiar ID motion improves relative to the protected parent.
3. Known-new physics reduces MSE by at least 15% relative to parent, remains within 10% of the original adapter's MSE, and wins at least two of three seeds.
4. Never-trained physics improves at least 0.5% relative to parent and stays within 2% of the original adapter's MSE.
5. No individual ID fault-state MSE degrades by more than 2% relative to parent.

A green GitHub job is not evidence of meeting these criteria. Ineligible models are explicitly marked `BLOCKED_OLD_SKILL_OR_TRANSFER_REGRESSION` or `NO_VALIDATED_SAFETY_GATE`. Do not lower the gate after viewing the held-out results.

## Execution and limits

Workflow: `.github/workflows/retention27-proof.yml`. The trainer downloads only new training/validation labels and the earlier frozen model. Only the final evaluation job downloads sealed new test labels. All connections are standard-library Python functions and all model state is saved in plain, digest-verified JSON.

This is still a **synthetic supervised forecasting experiment**, not general artificial intelligence, an autonomous self-improving neural brain, true video/AR, or proof of zero forgetting for all conceivable tasks. A deterministic four-frame observation cannot predict an independent late impulse. No version is auto-promoted into Git memory, and main remains unchanged.

# Phase 2.7 — Temporal observation quality, not another model-only gate

## Background

The Phase 2.6 physics-expert model improved aggregate synthetic accuracy on familiar and never-trained motion types but worsened familiar linear/turning motion. Three subsequent trust-gating experiments failed validation. A new **observability diagnostic** based on six synthetic motion families showed:

- Only **51.17%** validation accuracy for a simple classifier using the existing four-frame motion features.
- Familiar **linear** recognized only **28.12%** of the time.
- Familiar **turning** recognized **75.00%** of the time.
- Turning's optimal train-side adapter strength was approximately **0.943**, but on validation the corresponding optimum was **0**.
- The diagnostic uses 512 ID + 256 bounce/swirl training scenes, 128 ID + 128 bounce/swirl validation scenes, and no sealed test labels. It is a simple centroid separability measurement, **not proof of a universal information-theoretic bound**.

This points to unstable, noisy motion representations as a plausible failure mode. More complex gates over the same four-frame data did not fix old-skill forgetting.

## Actual intervention

**Add two earlier observations to every node's temporal context**, providing six frames instead of four. The original Phase 2.4 six-specialist prediction model still receives only the *last four* frames, so its numerical architecture is unchanged. The protected Phase 2.6 physics adapter also remains unchanged, using the same four-frame window.

The research router compares:

1. A **four-frame feature** classifier computed from the original kinematics.
2. A **six-frame feature** classifier that adds acceleration/turning persistence, velocity variation, jerk consistency, and change-of-heading statistics derived from two earlier observations.

Both classifiers use the same centroid method, trained on the same regime-labeled synthetic *training data*, then measured on independent validation data. The runtime gating function uses only observed positions and class predictions, never the true regime name or future target.

If the inferred motion resembles fragile old **linear or turning** trajectories with sufficient classification margin, the gate can suppress the existing physics correction and use the protected parent. Thresholds and trust levels are chosen using validation data alone.

**This experiment increases the information available to the router.** Even a positive result does not show that a fixed-information neural architecture learned more; it tests whether longer temporal observation can support safer continual learning.

## Fair target comparison

The generator produces trajectories at times 0 through 6. Six noisy node observations are generated at times 0 through 5. The target is always the position at time 6:

```text
six-frame router:  observations t0, t1, t2, t3, t4, t5
four-frame parent:                 t2, t3, t4, t5
both forecast:                                          t6
```

The original four-frame network uses exactly the final four observations of the new six-frame sequence. Both models are evaluated with the identical target and the identical final four noisy observations.

## Preregistered synthetic experiment

Training seed `31211`: 768 scenes balanced across six learned families:
`linear, accelerating, jerk, turning, bounce, swirl`.

Validation seed `31212`: 192 scenes across the same six families. No test target data are available in the training job.

Fresh sealed tests:

| Population | Seeds | Cases |
|---|---|---:|
| Familiar motion | 31221, 31222, 31223 | 3 × 256 |
| Known-new bounce/swirl | 51221, 51222, 51223 | 3 × 128 |
| Never-trained stop/start and late impulse | 51231, 51232, 51233 | 3 × 128 |

Total **1,536 scenes**, each under clean, one-missing, and one-strongly-corrupted node conditions, with abstentions explicitly counted.

Validation selection requires:
- Improvement in aggregate on the known four familiar families.
- No linear or turning validation regression relative to parent.
- No more than 2% acceleration/jerk regression, or 1% healthy/missing/corrupt regression.
- At least 10% improvement on bounce **and** swirl validation.

The final sealed scientific acceptance requires no aggregate linear or turning regression, overall familiar accuracy improvement, at least 15% known-new improvement (while preserving at least 90% of the original adapter's accuracy), positive never-trained transfer, and no familiar fault-state regression above 2%.

A green unit-test run proves implementation integrity, **not** a learned improvement. The dedicated acceptance job fails when these tests are not met. Passing would only qualify the adapter for human review; nothing auto-merges or modifies the parent's memory.

## Evidence, safety, and limitation

Workflow: `.github/workflows/temporal31-proof.yml`. It uses a frozen source parent+adapter from Phase 2.6 run `44873535607`. All state is plain JSON with SHA-256 digests. GitHub runner jobs have read-only repository permissions.

This is **new synthetic 2D temporal-data modeling**, not real camera video, real AR, biological memory, general intelligence, or autonomous background training. Additional observations may improve predictions simply by providing more evidence. The randomly injected final late impulse remains inherently unpredictable.

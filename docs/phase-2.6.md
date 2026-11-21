# Phase 2.6 — Real cross-domain transfer experiment

## Why this is not another infrastructure phase

The Phase 2.5 repair demonstrated modest learning on the original synthetic
motion distribution but lost a little performance on novel physics. Simply
adding memory checkpoints did not solve that. Phase 2.6 instead changes the
**predictive representation itself**, adding physically interpretable experts
and learning to select among them using observations, not ground-truth labels.

The six original trained numerical specialists, robust outlier filter,
and frozen Phase 2.4 checkpoint are preserved intact.

## Model architecture

```text
Six node observations (first four positions only)
           |
Fault detection rejects a missing or grossly corrupted node
           |
Median multi-node trajectory + original neural prediction
           |
Four physics experts: velocity / acceleration / rotation / restart
           |
Trainable 5-way softmax gate from velocity, acceleration,
turning angle, speed ratio, velocity direction and disagreement
           |
Interpolated combination with the protected parent forecast
           |
Unseen next-position prediction
```

The physics gate receives numerical observations and specialist packets,
never a supplied scene regime label or test answer. The dynamic gate has
learnable coefficients, fitted via supervised gradient descent over the
squared next-position error, with a protected original prediction as an
explicit fifth expert and a fallback. All models use Python standard library.

The physics experts implement several inductive biases that the original
17-feature peer readout lacked: extrapolating velocity; extrapolating
acceleration; rotating the observed velocity based on recent curvature;
and a restart heuristic when the most recent observed step is near zero.
These are **fixed interpretable features**, not learned world physics.

## Dataset partitions — predeclared

**Training and validation** receive only:
- Previous ID training seed `64013` plus fresh ID training seed `26111`;
  each contributes 512 train / 128 validation examples.
- A genuinely new set of movement behaviors: **bounce and swirl**.
  The OOD training seed is `46111` (256 scenes); validation seed
  `46112` (128 scenes). The other motion families are physically absent
  from these training artifacts.

**The sealed evaluator** gets nine disjoint, freshly generated cohorts:
- Three ID seeds `26121, 26122, 26123` × 256 examples = 768 familiar scenarios.
- Three exposed-novel-family seeds `46121, 46122, 46123` × 128 =
  384 bounce/swirl scenarios. Their seeds are unseen; the physics families
  were represented during training.
- Three **never-trained-motion-family** seeds `46131, 46132, 46133`
  × 128 = 384 stop/start and late-impulse scenarios.

Each 2D scenario is evaluated under healthy, one-missing, and one-strongly
corrupted node conditions: **1,536 scenes × 3 conditions = 4,608 scored
forecast/fault cases**.

A distinction matters: the stop/start and late-impulse families are *unseen
even by training family*, but they are still synthetic. The results cannot
be projected onto natural video or live spatial perception.

## Protected model selection and independent evaluation

Training tries a bounded, published sweep of parent-preservation weights,
regularization and interpolation strengths. Only training/validation data
choose the candidate.

The validation stage requires at least 2% improvement on exposed
novel-family validation cases, no more than 0.5% aggregate ID validation
regression, no more than 1% ID fault regression, and no more than 3%
degradation for any previous ID motion regime.

The sealed evaluator then reports the original parent and new gate for every
dataset and fault state. To label the new model `TRANSFER_SUPPORTED`, **all**
the following preregistered conditions must be met:

1. ID aggregate MSE must not worsen by more than 0.5%, with no individual
   ID fault state losing more than 1.5%.
2. Exposed novel-family held-out MSE must improve by at least 2.5% and
   at least two of three seeds must win.
3. On the **never-trained** stop/start and late-impulse families,
   average MSE must not worsen, and no individual seed/fault state may
   degrade by more than 2.5%.

If validation rejects every candidate, the evaluated model is exactly the
parent. This is not counted as learned generalization. If any final
criterion fails, the result is `BLOCKED_CROSS_DOMAIN_TRANSFER`.

The evaluation's own test labels are isolated in a separate GitHub Actions
artifact that the trainer never downloads. No checkpoint or registry index
is updated by this experiment.

## Limits and interpretation

- Strong synthetic single-node faults do not establish subtle adversarial
  robustness.
- A physics expert may be useful for some motions and catastrophic for
  others; per-family errors must be inspected regardless of aggregate mean.
- Late impulses injected **after the final observation** are intrinsically
  unpredictable from those four frames. Even a perfect data-fitting system
  cannot infer an independent random impulse; success means calibrated
  uncertainty and controlled error, not impossible foresight.
- A successful GitHub workflow and passing unit tests are distinct from
  improved held-out prediction.
- This is supervised mixture-of-experts routing, not evidence of conscious
  neural processes or a general artificial brain.

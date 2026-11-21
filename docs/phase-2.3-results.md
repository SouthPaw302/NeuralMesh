# Phase 2.3 — First five-seed held-out results

**Workflow:** [Actions run 44870774243](https://github.com/SouthPaw302/NeuralMesh/actions/runs/44870774243)  
**Evidence artifact:** `phase23-cross-seed-evidence-44870774243` (contains `summary.json` and all five `seed-*.json` results)  
**CI result:** 17 of 17 jobs passed; 26 actual unit tests passed.  
**Independent seeds:** `64013, 74129, 85021, 91153, 104729`.  
**Per-seed data:** 512 train / 128 validation / 256 sealed test scenes, four balanced motion regimes.  
**Total:** 4,480 synthetic scenes, including 1,280 held-out test scenes.

## Held-out performance (mean of five independent seeds)

All figures are **mean squared 2D next-position error (MSE), lower is better**.

| Model | Mean test MSE |
|---|---:|
| Phase 2.1 static learned routing | 0.019799029257 |
| Phase 2.2 contextual attention (no feedback) | 0.019628211213 |
| Phase 2.2 full peer-embedding feedback | 0.014194647441 |
| **Phase 2.3 validation-selected selective feedback** | **0.013881872079** |

Across these seeds, Phase 2.3 averaged **2.20% lower MSE** than Phase 2.2 full feedback, and **29.89% lower MSE** than Phase 2.1 static routing.

**Four of five seeds** selected a learned gate and beat full feedback; **one selected full feedback**, producing a tie. No seed was worse after validation selection on this particular sample. This is a descriptive observation over five seeds, **not a guarantee** on future seeds or real data.

## Seed-level results

| Seed | Selected policy | Full feedback MSE | Phase 2.3 MSE | Difference (selected minus full) |
|---|---|---:|---:|---:|
| 64013 | Learned gate | 0.013308284 | 0.012967010 | −0.000341274 |
| 74129 | Learned gate | 0.016019059 | 0.015998157 | −0.000020902 |
| 85021 | Learned gate | 0.014296325 | 0.013729299 | −0.000567026 |
| 91153 | Learned gate | 0.013713400 | 0.013078726 | −0.000634675 |
| 104729 | Full feedback | 0.013636169 | 0.013636169 | 0 |

The per-seed paired bootstrap 95% MSE-difference intervals exclude zero for seeds **64013** and **85021**, but include zero for seeds **74129** and **91153**. Seed **104729** is an exact tie because validation chose the full-feedback baseline. A mean over only five runs is not enough to claim general effectiveness across all possible synthetic distributions.

## Per-motion-regime result — the key tradeoff

Mean held-out MSE across five seeds:

| Motion regime | Full feedback | Phase 2.3 selective | MSE change |
|---|---:|---:|---:|
| Linear | 0.010126950 | **0.008482390** | **−16.24%** |
| Accelerating | **0.013267418** | 0.014196839 | **+7.01% (worse)** |
| Jerk | **0.021448315** | 0.022585033 | **+5.30% (worse)** |
| Turning | 0.011935907 | **0.010263226** | **−14.01%** |

Mean selected feedback strength by motion regime:
- Linear: **0.5885**
- Accelerating: **0.7803**
- Jerk: **0.7788**
- Turning: **0.5653**

The gate **learned to reduce feedback more often on linear and turning trajectories**, which addresses the actual failure cases identified in Phase 2.2. It also reduced correction strength on some accelerating and jerk examples, degrading those families. The aggregate gain therefore conceals meaningful specialized losses.

The connection gate uses only motion signals and peer embeddings—no oracle access to a test regime label. The explicit regime breakdown above is *evaluation-only*.

## What was established

- Selective feedback was implemented as a fitted input-conditioned sigmoid controller with off/full policies available as validation alternatives.
- The controller was fitted on training data and chosen on validation data, with its parameters frozen before the test job downloaded sealed labels.
- All five data splits were independently generated and all five full reports were validated by the final aggregation job.
- Four independently selected learned gates were genuinely scene-dependent and had nonuniform feedback strengths; one seed appropriately retained full feedback.
- Cross-seed results show a small aggregate benefit, accompanied by clear remaining per-regime harms.

## What was **not** established

This does **not** prove real neural emergence, a biological brain simulation, online learning across runner lifetimes, robustness to unseen motion physics, useful real-world video prediction, or that a distributed 30-node interconnect ran. Phase 2.3 used **five concurrent seeded runner jobs**, each executing six logical specialists internally. Phase 2.2 remains the six-physical-runner intercommunication proof.

## Next research gate

Before introducing larger models or a long-lived memory store, test **out-of-distribution motion**, **missing/corrupted specialists**, and a smarter counterfactual/specialist feedback router that can preserve the acceleration/jerk benefit without harming linear/turning trajectories. Evaluate routing complexity and cross-run persistence separately from prediction accuracy. Keep this phase in a stacked reviewable PR until its dependencies are reviewed.

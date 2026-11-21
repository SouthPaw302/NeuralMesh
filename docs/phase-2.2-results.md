# Phase 2.2 — First held-out evaluation

**Source:** [GitHub Actions run 44870216810](https://github.com/SouthPaw302/NeuralMesh/actions/runs/44870216810)  
**Evidence artifact:** `phase22-evidence-44870216810`  
**Data seed:** `2202`  
**Data:** 512 train, 128 validation, 256 held-out synthetic scenes  
**Motion families:** linear, accelerating, jerk, turning (64 test scenes per regime)  
**Execution:** 16/16 GitHub Actions jobs passed; 20 unit tests executed and passed.  
**Scope:** synthetic 2D point forecasting, not real video/AR or neural emergence.

## Headline: peer embeddings help on this held-out set

| Predictor | Test MSE (lower is better) |
|---|---:|
| Validation-selected single node | 0.030056435402 |
| Phase 1 fixed/recurrent routing | 0.021942071000 |
| Phase 2.1 static learned routing, refitted on this split | 0.021416817220 |
| Phase 2.2 context gate, before feedback | 0.021351406158 |
| **Phase 2.2 context + peer embedding feedback** | **0.015303433308** |

Relative to Phase 2.1's global static router, the context+feedback model has **28.55% lower test MSE**. Relative to context alone it has **28.33% lower MSE**. Dynamic gating alone gives a small **0.31% decrease** versus static routing.

Paired, deterministic bootstrap 95% intervals (feedback minus comparator, MSE units):

- Feedback minus Phase 2.1 static: delta **−0.006113383913**, CI **[−0.008649824509, −0.003756887366]**.
- Feedback minus contextual/no-feedback: delta **−0.006047972850**, CI **[−0.008569639035, −0.003729125869]**.

These intervals exclude zero under within-test-scene resampling; this is **evidence of a within-distribution improvement**, not confirmation on independent real-world distributions or repeated random seeds.

## Most important finding: improvements are uneven

| Synthetic motion regime | Phase 2.1 static MSE | Phase 2.2 context+feedback MSE | Direction |
|---|---:|---:|---|
| Linear | 0.006016476 | 0.010609244 | **Worse** |
| Acceleration | 0.021362003 | 0.012246673 | Better |
| Jerk | 0.048646328 | 0.026042515 | Better |
| Turning | 0.009642462 | 0.012315301 | **Worse** |

The new peer-embedding correction is especially useful for accelerating and changing-acceleration motion, while overcorrecting simpler linear motion and turning. A future regime-aware feedback gate needs to learn *when not to correct* instead of applying the full residual everywhere.

## What was actually learned?

- The validation-selected contextual gate regularization was **0.001**.
- Feedback was selected at **beta 1.0** with residual ridge regularization **0.005**.
- Context-only validation MSE was **0.018616064432**.
- Context+feedback validation MSE was **0.013711586034**.
- The second-stage feedback signal consists of 17 learned-input features computed from **the six specialists' nine-dimensional numeric embeddings**, including weighted and uniform statistics. Every stage-two worker independently applies the shared frozen correction readout to peer inputs, rather than becoming a continuously coupled neuron.
- Contextual weights vary across held-out scenes (e.g. position weight **0.3440–0.3744** and short-horizon weight **0.1765–0.2035**); this is actual **scene-conditioned connection adjustment**.
- The six original specialist neural readout models were held fixed. This step learned only the connection gate and residual peer-feature corrector.

## Interpretation and limits

**Proven in this implementation:** six GitHub runner workers exchanged embeddings; a frozen, training-derived model used the peers' representations to correct predictions; a held-out synthetic benchmark measured a beneficial net effect with integrity tests. The initial 2.1 result of zero useful feedback was not forced or erased—the additional **peer embeddings** are what made a feedback model useful here.

**Not proven:** genuinely emergent collective intelligence; online neural plasticity spanning multiple sessions; a biological-style recurrent network; higher accuracy for all scene types; generalization to other distributions or actual videos; massively parallel tensor inference; sustainable low-latency operation on GitHub runners.

The test data were not downloaded into worker or training jobs. The default Phase 2.2 seed differs from the first observed Phase 2.1 test seed, but this remains *one* synthetic draw and must be replicated across independent seeds before promotion.

## Next research priorities

- Test **several fresh independent seeds** and report aggregate means, confidence intervals and worst-case regime degradation.
- Learn a per-scene **feedback suppression gate** to avoid harms on linear and turning motion.
- Add experiments with hidden/unseen regimes and corrupted/missing nodes.
- Evaluate whether observed improvement justifies the second wave's inter-runner communication time and artifact costs.
- Keep state checkpoints versioned and inspectable; no automatic promotion of a single-seed model.

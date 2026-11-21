# Phase 2.5 — First actual memory-evolution experiment

## Provenance

- Experiment: [GitHub Actions run 44872086521](https://github.com/SouthPaw302/NeuralMesh/actions/runs/44872086521).
- Result: **4/4 jobs succeeded**, **40 unit tests passed** in this first run.
- Full artifact: `phase25-review-44872086521`.
- Parent Phase 2.4 checkpoint: `8c147e3d6bad62cd9b6553738cba40dddc4f5120755273f01a366f3e235d22bc`.
- New **version 2** child: `5a9af9011a82ef84ed8a6175041343881b18a1a26d92e48e0625c13b44270d7b`.
- Exact selected regularization: **ridge 0.2**.
- Training cohort: seed **2505**, 512 training + 128 validation + 256 sealed test synthetic scenes.
- Historical regression anchor: seed **64013**, validation data only.
- Held-out OOD: seed **82521**, 128 novel-motion scenes.
- Both test populations scored in healthy, missing-node, and grossly-corrupted-node conditions.

## Training / validation results

| Validation measurement | Parent MSE | New child MSE | Relative change |
|---|---:|---:|---:|
| Fresh-cohort validation | 0.015387390 | **0.015249894** | **−0.894%** |
| Historical anchor validation | **0.017839382** | 0.017951381 | +0.628% |
| Fresh-cohort fault-injected validation | 0.019292790 | **0.018755921** | **−2.783%** |

This satisfies the preregistered validation criteria: >=0.5% improvement on new validation, <=3% degradation on the anchor, and <=5% degradation on fault scenarios. The child is **different from its parent** and has a new immutable SHA-256 identity.

## Held-out results: an important qualification

All metrics are 2D next-position MSE; lower is better.

| Independent test | Parent MSE | Candidate MSE | Relative change |
|---|---:|---:|---:|
| New in-distribution, all six healthy | **0.015268204** | 0.015419196 | **+0.989% worse** |
| New in-distribution, one missing | 0.020778081 | **0.020746629** | **−0.151% better** |
| New in-distribution, one corrupted | 0.020778081 | **0.020746629** | **−0.151% better** |
| Novel motion, all six healthy | **0.304084702** | 0.306600250 | **+0.827% worse** |
| Novel motion, one missing | **0.319304660** | 0.319603282 | **+0.094% worse** |
| Novel motion, one corrupted | **0.319304660** | 0.319603282 | **+0.094% worse** |

The new memory is **not a clear accuracy upgrade**. It learned something on the fresh training cohort and improved fault-injected *validation* performance, but on the unseen scenarios the new state worsened healthy-node accuracy and most OOD conditions slightly.

The software reported `PROMOTION_ELIGIBLE` because it passed the configured **regression tolerance gate** (maximum relative losses permitted), *not* because it outperformed the parent. The guard is deliberately permissive enough to admit bounded tradeoffs; the final decision must be made by a human reviewer based on these exact numbers.

**Recommendation for this candidate: do not promote yet.** Preserve it as versioned research evidence and use fresh independent cohorts to assess whether the improvement is repeatable before replacing the active memory.

## Integrity and actual persistence

- The Phase 2.5 trainer retrieved the **actual frozen model checkpoint from the earlier Phase 2.4 workflow run**, then updated only the learned 17-feature peer-feedback residual readout.
- The parent JSON snapshot remains unchanged. The child has `version=2`, `generation=2`, explicit `parent_sha256`, training-history event, and `adopted_automatically=false`.
- Training never downloaded the sealed new test or OOD test targets. The independent evaluator made no changes to learned weights.
- Promotion evidence contains SHA-256-linked parent, child and evaluation report plus a separate human-review flag.
- **No memory checkpoint has been promoted into the repository's permanent Git registry**. The reviewed Git registry PR workflow is manual-only and becomes available after merging the implementation PR stack into `main`.
- Original GitHub Actions artifact-based memory expires after its configured 30-day retention; no claim of indefinite persistence should be made until a Git registry PR is explicitly merged.

## Next decision / stronger research gate

A later iteration should pre-register stricter expected generalization improvement thresholds **before drawing new sealed test seeds**, repeat enough independent seeds to estimate benefit, and address why validation improvements failed to carry through to clean or novel-motion test data. This Phase 2.5 sealed test is now *seen* and must not be recycled as a new blind confirmation set.

## Scope

This is incremental supervised learning of model coefficients inside a six-specialist numerical routing system, **not** continuous autonomous plasticity, biological neural networks, real video/AR perception or a generic artificial intelligence.

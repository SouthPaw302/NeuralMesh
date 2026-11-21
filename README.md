# NeuralMesh

**Active project contract:** [One reproducible experiment and a fixed stopping rule](docs/ONE_EXPERIMENT.md). The stable tested research foundation is on `main`. [The canonical replication](docs/ONE_EXPERIMENT_RESULT.md) rebuilt the exact model from source and improved all nine aggregate seed scores, but **failed strict no-forgetting acceptance** because familiar linear and turning motion regressed. The parent remains protected; the historical failed trust-gating attempts are closed, not promoted.

**Run one canonical learning experiment:** Actions → **NeuralMesh Canonical Learning Experiment** → Run workflow. The original six-runner distributed proof is a separate orchestration test.


An experimental distributed neural mesh built on GitHub Actions.

## Twenty local learners — sparse cooperative training (experimental)

The new [20-local-learner stack](docs/SPARSE20_LEARNING.md) runs **20 independent training jobs at once**, learns a **five-peer graph** from held-out calibration, and runs a second 20-worker learning wave. Each node maintains its own tiny CPU-based predictive model; peer communication consists of compact, shared-world-coordinate forecast probes. The second wave trains a matched no-peer control alongside the peer-informed candidate.

A complete 20-node state checkpoint is emitted for deliberate **cross-run continuation**. Neither checkpoint nor routing updates are auto-promoted, and a separate scientific gate requires peer learning to beat the independent learners on unseen examples.

This is a functioning small numerical learning experiment, **not** an implementation of Gemma/Qwen/LFM2 language models or multimodal embedded encoders. The previous 20-worker inference proof remains independent.

## Active 20-runner execution

**New:** [20 parallel GitHub-hosted workers, twice](docs/TWENTY_PARALLEL_RUNNERS.md). This experiment uses **20 independently trained specialists** across two consecutive waves of **20 actual runner jobs each**, with `max-parallel: 20` and 19 directed peer messages per worker. The `audit-parallel` job checks the **observed** peak simultaneous runner count in each wave and fails if fewer than 20 overlapped. The new pipeline is **Actions → NeuralMesh 20 Parallel Runners**, not the six-worker legacy workflow.

This verifies distributed execution, not automatic model intelligence improvements. All historical six-node research and its science acceptance rules remain unchanged.

**Verified [first 20-runner results](docs/TWENTY_PARALLEL_RESULTS.md):** Wave 1 and Wave 2 both achieved **20 genuinely overlapping GitHub jobs**, all 40 worker job executions passed, and the final timestamp audit passed. Round-2 feedback modestly improved consensus MSE but did not beat the strongest specialist. No weights were promoted.


**Status: v0 reproducible research prototype.** Six independent learned numerical specialist modules predict motion from different noisy observations. GitHub-hosted workers exchange vector predictions through two rounds of directed, weighted feedback. A final evaluator measures whether the collective improves on the best standalone node.

This is **not** a simulated biological brain, a consciousness model, distributed training of a large neural network, or evidence of superior collective intelligence. GitHub Actions artifacts are a high-latency message bus, not a low-latency tensor transport.

## Architecture

```text
 prepare: train + seal withheld target
           |
  round 1: six independent runners
           |
  synapse 1: adaptive peer routing
           |
  round 2: six independent runners
           |
  synapse 2: aggregate recurrent feedback
           |
 evaluate: compare predictions against withheld target
           |
 metrics.json + candidate-checkpoint.json
```

The historical six-node workflow uses at most six workers in parallel. The **new** 20-node workflow launches twenty per wave (forty separate worker job executions), measures actual runner overlap, and preserves its snapshots as workflow artifacts; no model is automatically promoted.

## Quick start

Python 3.11+, standard library only:

```bash
python -m unittest discover -s tests -v
python -m neural_mesh.mesh prepare --seed 314159 --out /tmp/mesh
python -m neural_mesh.mesh local-proof --state /tmp/mesh --out /tmp/proof
cat /tmp/proof/metrics.json
```

## Running the original six-runner proof (legacy)

Open **Actions → NeuralMesh Distributed Proof → Run workflow**. The legacy six-runner experiment launches **only by manual dispatch**; a normal push or pull request runs tests without allocating the twelve specialist jobs.

After the run, download `mesh-evidence-<run_id>`. Its `metrics.json` records best-individual, first-round, and feedback-round mean squared errors; the remaining files contain both synaptic packets and a candidate adaptive checkpoint.

## First observed result

On the original six-runner test, synthetic seed `314159`:

| Predictor | MSE (lower is better) |
|---|---:|
| Best individual | 0.006927 |
| Round 1 consensus | 0.008579 |
| Round 2 consensus | 0.008628 |

All workers completed, but recurrent feedback **did not improve** this specific prediction. Distributed execution success is distinct from collective accuracy.

## Integrity and scope

- Actions jobs use read-only repository permissions and pinned checkout/artifact actions.
- Independent observations and training are deterministic for a given seed.
- Each worker records a model digest, input digest, node identity, stage, and numerical embedding.
- Missing, duplicated, or cross-scene messages fail the integration checks.
- Held-out ground truth is carried separately and used only by evaluation (not a security boundary against maintainers).
- A candidate state checkpoint is produced, never automatically promoted.
- No external inference API, large model download, real video input, AR scene reconstruction, or production-media write is performed.

## Phase 2.1 — Multi-scenario adaptive routing

The Phase 2.1 implementation adds a **512 training / 128 validation / 256 held-out test** synthetic motion benchmark across four movement regimes. Six GitHub runners predict independently, a separate job learns frozen routing weights from training/validation only, and six second-round workers apply the selected feedback policy. The evaluator measures actual holdout results and paired error intervals.

Read [the experiment protocol](docs/phase-2.1.md) and launch **NeuralMesh Phase 2.1 - Held-out Plasticity Benchmark** from GitHub Actions after review. Its first branch-scoped experiment is recorded in Actions. Existing Phase 1 functionality is retained.

## Phase 2.2 — Contextual synapses

Phase 2.2 tests **scene-dependent node attention** and a learned **peer-embedding feedback residual** over a fresh 512/128/256 synthetic train/validation/test split. It includes independently benchmarked static routing and no-feedback ablations. The fitted router can turn off feedback if validation rejects it.

See [Phase 2.2 experiment protocol](docs/phase-2.2.md). The workflow runs six observation workers, learns connection weights from training/validation only, then runs six feedback workers before independently scoring sealed test data. No result is auto-promoted.

## Phase 2.3 — Selective feedback and cross-seed testing

Phase 2.3 trains a small scene-conditioned **feedback strength controller**, allowing the mesh to suppress or amplify its learned Phase 2.2 peer-embedding correction. A sealed five-seed benchmark measures whether selective feedback actually outperforms the Phase 2.2 full-feedback policy, including per-regime harms.

The bounded experiment uses five independent parallel GitHub training jobs (one per seed), each evaluating six logical specialists, and separate sealed-test evaluation runners. It is not a 30-runner neural interconnect. See [Phase 2.3 research protocol](docs/phase-2.3.md).

## Phase 2.4 — Fault Tolerance and Versioned Neural Memory

Phase 2.4 introduces a SHA-256-verified **memory checkpoint** storing trained specialist readouts, contextual synapses, peer-feedback corrections, and validation-calibrated fault handling. It tests missing and corrupted nodes under four novel synthetic motion families, comparing the fault-aware predictor against the previous naive routing.

Three independent GitHub probe runners restore the same frozen checkpoint; sealed target coordinates remain isolated until evaluation. [Phase 2.4 protocol](docs/phase-2.4.md) documents the architecture and limitations. An independent [cross-run recall](https://github.com/SouthPaw302/NeuralMesh/actions/runs/44871327795) verified that the identical checkpoint survives separate GitHub workflow executions, without retraining; Actions artifacts still expire. See [Phase 2.4 measured results](docs/phase-2.4-results.md).

## Phase 2.5 — Versioned Learning and Manual Memory Promotion

The first evolving memory checkpoint was trained from fresh synthetic data while preserving its original Phase 2.4 ancestor. The [Phase 2.5 architecture](docs/phase-2.5.md) includes SHA-256 lineage, explicit validation/holdout regression gates, immutable version staging in a Git-backed registry, and a **manual-only** workflow that opens (but never merges) a memory promotion PR.

**Scientific result:** the candidate improves fresh validation but slightly worsens healthy-node and novel-motion held-out predictions. The [measured evidence](docs/phase-2.5-results.md) recommends retaining the parent rather than promoting this candidate automatically. The experimental Git memory registry remains uncommitted pending human review.

## Phase 2.5 repair — historical rehearsal and rigorous generalization

The first Phase 2.5 updater optimized a different residual than the **actual deployed fault-aware predictor** uses. [The repaired learner](docs/phase-2.5-repair.md) fits the exact deployed equation under clean/missing/corrupt conditions, rehearses earlier training cohorts, and tests across **three independently seeded ordinary-motion and three novel-motion held-out sets**. A stronger promotion gate now rejects regressions that the old single-seed gate tolerated.

[First measured repair result](docs/phase-2.5-repair-results.md): **0.97% lower error on ordinary motion** across three new seeds, but **0.25% higher error on novel motion**, so **promotion is correctly BLOCKED**. Original memory remains unchanged. Manual registry promotion only accepts the repaired six-set evidence and requires a separate human-reviewed PR.

## Phase 2.6 — Cross-Domain Physics Transfer

The mesh now combines its frozen six-node forecast with four interpretable physics experts, selected by a learned observation-dependent softmax gate. The [Phase 2.6 protocol](docs/phase-2.6.md) introduced bounce and swirl during training, while keeping stop/start and late impulse completely unseen.

The [first nine-seed held-out results](docs/phase-2.6-results.md) improved mean error **1.9% on familiar motion**, **20.4% on newly trained motion families**, and **2.9% on never-trained motion families**. All nine independent test seeds improved in aggregate; **familiar linear motion worsened by 5.5%**, so retention across every capability remains unproven. No model memory was auto-promoted.

## Current scientific decision

Do not add another phase or corrective router in response to a red run. The [canonical experiment](docs/ONE_EXPERIMENT.md) must demonstrate replicated unseen gains **and** retention of familiar linear and turning motion before an adapter may even be considered for promotion. If it fails, retain the parent and report the limitation. The six-runner distributed orchestration and model accuracy are separate claims.

## License

No license has been chosen by the owner. Please select one before soliciting outside contributions.

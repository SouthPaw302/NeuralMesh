# NeuralMesh

An experimental distributed neural mesh built on GitHub Actions.

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

At most six specialist runners are active in parallel, with twelve specialist job executions across two sequential rounds. State is durable within one workflow through artifacts, not yet across workflows.

## Quick start

Python 3.11+, standard library only:

```bash
python -m unittest discover -s tests -v
python -m neural_mesh.mesh prepare --seed 314159 --out /tmp/mesh
python -m neural_mesh.mesh local-proof --state /tmp/mesh --out /tmp/proof
cat /tmp/proof/metrics.json
```

## Running the GitHub swarm

Open **Actions → NeuralMesh Distributed Proof → Run workflow**. The full six-runner experiment launches **only by manual dispatch**; a normal push or pull request runs tests without allocating the twelve specialist jobs.

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

## Direction

The next experiments should measure multi-scene generalization and train routing weights over held-out examples; then introduce real video-derived features, versioned persistent state, and eventually alternative transport for lower-latency neural communication.

## License

No license has been chosen by the owner. Please select one before soliciting outside contributions.

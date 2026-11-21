# Verified 20-runner concurrent execution — first full proof

GitHub Actions: [run 44878036649](https://github.com/SouthPaw302/NeuralMesh/actions/runs/44878036649)

## Physical concurrency — measured, not inferred from YAML

The first new 20-specialist experiment ran two **sequential** waves.

| Job population | Job executions | Peak overlapping jobs | Outcome |
|---|---:|---:|---|
| Wave 1 | 20 | **20** | All succeeded |
| Synaptic barrier 1 | 1 | — | Combined and verified all 20 receipts |
| Wave 2 | 20 | **20** | All succeeded |
| Synaptic barrier 2 | 1 | — | Combined and verified all 20 feedback receipts |
| Independent evaluator | 1 | — | Scored sealed target |
| Actual concurrency audit | 1 | — | **Passed** |

The `audit-parallel` job inspected the GitHub Actions API's real worker-job `started_at` and `completed_at` timestamps for the run, rather than assuming `max-parallel: 20` implies scheduling capacity. Both waves reached 20 concurrent jobs. The proof is recorded as `swarm20-concurrency-evidence-44878036649`.

**Twenty concurrent machines, forty sequentially executed worker jobs total.** This is not forty machines in parallel.

## Model

- 20 unique trained specialists: five role masks × four independently seeded readouts.
- Every specialist supplies a numeric forecast and an 11-feature embedding.
- In the second wave, all 20 specialists receive verified packets from the previous round, with **19 directed peers per recipient**.
- All 40 worker receipts are checked for correct model identity, shared observation digest, stage, peer topology, and numeric integrity.
- The original six-node implementation remains unchanged for historical and Phase 2.x research.

The preparation job currently trains the twenty readouts centrally before distributing them. The 20 GitHub runners perform **parallel inference and feedback computation**, not twenty independently persistent learning processes.

## Measured synthetic forecast

Default scene seed `314159`. Error is squared next-position distance averaged over x/y coordinates; lower is better.

| Method | MSE |
|---|---:|
| Best single specialist | **0.001795916968** |
| Round 1 (20-way consensus) | 0.007832460800 |
| Round 2 (20-way recurrent consensus) | 0.007724661961 |

Feedback reduced consensus MSE by approximately 1.38% compared with round 1, but **the collective remained worse than the strongest individual**. This is one synthetic test scene, not a multi-seed validation of collective learning.

The final evaluator returned `recurrent_feedback_helped: true` and `round2_better_than_best_node: false`.

## Artifacts and safeguards

- `swarm20-model-evidence-44878036649`: both aggregation packets, numeric metrics, non-promoted candidate checkpoint.
- `swarm20-concurrency-evidence-44878036649`: exact requested and observed job overlap and all forty worker job IDs.
- Individual per-worker round artifacts are also available.
- The target coordinates were withheld from all runner jobs and used only by the final evaluator.
- No model checkpoint was automatically committed or promoted.

## Research interpretation

**Engineering claim supported:** A real GitHub-hosted 20-way swarm successfully ran concurrently in both directed message-passing waves.

**Scientific claim not established:** That parallelization alone improves learned predictive competence, continual learning, real-world generalization, or biological-style intelligence. The current runtime still uses high-latency GitHub Actions artifacts between waves rather than real-time peer messaging.

Source: `neural_mesh/swarm20.py` and `.github/workflows/neural-mesh-20.yml`. Full experimental protocol: [TWENTY_PARALLEL_RUNNERS.md](TWENTY_PARALLEL_RUNNERS.md).

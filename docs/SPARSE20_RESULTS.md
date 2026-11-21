# Twenty independent local learners — first measured experiment

## Execution evidence

- [Full independent 20-runner training experiment: run 44880153349](https://github.com/SouthPaw302/NeuralMesh/actions/runs/44880153349)
- Python test suite: **76/76 passed**.
- Preparation and five-peer routing: **passed**.
- Stage 1: **20/20 physically separate local training jobs succeeded**, observed maximum **20 simultaneous jobs**.
- Stage 2: **20/20 separate local training jobs succeeded**, observed maximum **20 simultaneous jobs**.
- Independent evaluator: **passed**, produced candidate checkpoint and per-family/per-seed evidence.
- Timestamp-based concurrency audit: **passed**.
- Scientific acceptance: **FAILED** (the sparse cooperative adaptation did not improve on a paired independent-learning control).
- No candidate model was automatically promoted.

This is the first actual twenty-worker **training** proof. The previous `neural-mesh-20.yml` pipeline proved independent inference, not independent local learning.

## Training and communication

Each specialist trained its own 2-output, 11-feature numerical model from 360 locally generated labelled examples (20 × 360 = **7,200** locally processed labelled scenes). Each learned model has its own readout and independent data/noise stream.

Calibration observations used 150 separate scenes per node to rank peer reliability. The router constructed **20 different five-peer routes** (100 directed peer links, versus 380 directed links in the twenty-node all-to-all inference network).

In the second wave, each worker trained **both** a local control and a peer-distilled candidate from the same initial local weights with 180 *identical* additional labelled scenes. The peer candidate additionally distilled 32 unlabelled public probes, receiving only compact world-coordinate forecasts from its five selected peers.

The two branches have the same number of labelled examples and same supervised SGD algorithm; peer distillation is a deliberate additional update, not a compute-matched zero-cost operation.

## Reproducible evaluation

Experiment generation seed: `810247`. Three preregistered untouched synthetic held-out seeds: `726031`, `726037`, `726043`, 160 scenes each, 480 total.

All evaluations use identical target trajectories and node-specific sensor observations across the before-update, independent-control and five-peer-learning methods.

| Method | Aggregate MSE, lower better | Best individual specialist MSE |
|---|---:|---:|
| After first local training wave | **0.015407104989** | 0.026356169759 |
| After second local training wave, **no communication** | 0.016176511895 | **0.026309238130** |
| After second local training wave, **five-peer distillation** | 0.016253389375 | 0.026317784784 |

Peer learning versus no-peer control: **0.4752% worse** (MSE increased). This regression happened on **all three** held-out seeds.

The additional supervised local update was itself worse than the first-wave checkpoint on aggregate error. The first-wave model was approximately **4.8% lower MSE** than the second-wave independently trained model. That is evidence that the current learning rate/data distribution/update schedule is not safely cumulative on this task.

### Per-regime error

| Regime | Before second wave | Independent control | Five-peer learner |
|---|---:|---:|---:|
| Linear | 0.00241237 | **0.00223820** | 0.00224648 |
| Acceleration | **0.01456696** | 0.01519222 | 0.01525217 |
| Jerk | **0.03882261** | 0.04053799 | 0.04069001 |
| Turning | **0.01059291** | 0.01081041 | 0.01081616 |
| Bounce | **0.01064068** | 0.01210374 | 0.01226213 |

The best individual worker after peer distillation had MSE **0.02631778**, so the overall peer-distilled **ensemble of all twenty** (MSE 0.01625339) **did beat its strongest single participant** on these three seeds. This is an important, separately measured ensemble benefit, but it does **not** mean peer distillation helped: the independent ensemble was better still.

## Checkpoints and acceptance

Candidate checkpoint `sparse20-checkpoint-44880153349` contains all twenty new social model weights, a per-node selected peer graph, a checkpoint generation number, and a digest of the prior state. SHA-256:

`3df9d94ff5e054eb90faa9a934da54af6a8b6505bc5df6fcf6e8191c12671a5c`

This is **research evidence only**. Because the scientific acceptance gate failed, there is **no claim that the learning rule improved the network**, and the candidate remains unpromoted.

The restore workflow `.github/workflows/sparse20-restore-proof.yml` independently downloads the first trial's saved checkpoint from a different GitHub run and verifies that all twenty learned models can be loaded into a generation-two training configuration. That is a durability test, **not an authorization to adopt a scientifically rejected checkpoint**.

## What this experiment proves

**Engineering demonstrated:** Real 20-way parallel *local model training* in both waves; decentralized specialist weights; five-peer routing; compact cross-node messages; paired independent-control research; replayable, digest-bound checkpoint format; no automatic promotion.

**Scientific evidence:** Ensemble diversity is useful on these synthetic motion scenes. Sparse peer distillation as currently defined made generalization modestly worse. More local updates were not automatically better.

**Not demonstrated:** Improved collective learning, preserved skills under distribution shift, production mini-language-model embeddings, real-world vision/audio understanding, continual real-time intelligence, or persistent independently operating agents.

## Next research decision

Do not add another corrective safety layer just because this experiment failed. Preserve its independent controls and model snapshots. A future distinct experiment would need to establish, with a new stream and new untouched holdouts, whether a *trained student-teacher objective with uncertainty calibration* is more appropriate than directly averaging world-coordinate forecasts.

If involving compact pretrained language/vision/embedding models (e.g. MiniLM, Gemma, LFM2), implement a separate compatible encoder interface and representation-alignment experiment. They are **not already deployed** by this numerical testbed.


## Independent cross-run state restoration — verified

A separate [GitHub Actions run 44880486740](https://github.com/SouthPaw302/NeuralMesh/actions/runs/44880486740) **completed successfully**. It downloaded the prior run's serialized checkpoint through GitHub's cross-run artifact API, verified its digest and exactly twenty model identities, generated a new learning-stream configuration `seed=810248`, confirmed `generation=2`, then independently initialized and trained each node from its **actual saved parameters**.

All 20 restored initial weight hashes matched the earlier checkpoint and all 20 local models changed during new training. The evidence is attached as `sparse20-durable-state-proof-44880486740`.

This proves checkpoint reuse across separate Actions runs. It **does not** validate or promote the earlier candidate's learning quality, which remains scientifically rejected.

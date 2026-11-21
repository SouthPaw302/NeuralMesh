# NeuralMesh sparse twenty — locally learning workers

## What this branch builds

This is a **working first modular-learning system**, not twenty copies of a chatbot. It keeps the earlier twenty-host inference proof untouched and introduces:

1. Twenty *independent GitHub runners* that each learn and save a different small predictive model from a nonidentical local stream.
2. Shared world-coordinate forecast messages (two floats per probe) instead of pretending unrelated neural feature dimensions are aligned.
3. A reliability-based **five-peer graph per worker**, with one peer from each of five competence families. Each worker has exactly five incoming peers and never receives itself.
4. A second full twenty-runner training wave: every worker starts from its local readout, trains on additional local ground-truth examples, and uses only the five selected peers' forecast messages to distill a small adjustment on unlabelled shared probes.
5. A paired **independent-learning control**, starting from the *same* stage-1 weights and using the **same** stage-2 supervised examples and SGD hyperparameters. The social and control models are retained separately for genuine communication ablation.
6. An independent evaluator with three preregistered synthetic held-out seeds, comparing the baseline, the locally updated control, the socially updated system, and the best individual specialist.
7. A *versioned checkpoint* containing the twenty learned weights, node identity, parent-state digest, generation number, and learned peer graph. A later workflow dispatch can load this exact checkpoint by source run ID and continue learning.
8. A job-timestamp audit verifies that **20 physical GitHub runners overlapped during each learning wave**. A scientific acceptance job is separately red unless the social update beats the independent control on average and on at least two of three held-out seeds.

**Previous protected experiments are not modified.** No candidate checkpoint automatically becomes the stable one.

## Representation and scale

Each worker is a **tiny CPU-resident online learner**, specifically an 11-feature linear readout with two independently trained prediction outputs. The five families are position, velocity, acceleration, curvature, and stabilization, with four seeded replicas each.

They specialize through **non-identically distributed labelled training experiences** (62% emphasis on the family's motion type) and node-specific sensor noise. They update small numerical weights using local supervised SGD.

This is deliberately below the scale of Gemma, Qwen, LFM2, or a trained language embedding model. It is a minimal, testable substrate for adding those mini-model backends later, **not** an implementation of twenty pretrained language models or embedded multimodal inference.

Each stage-1 message contains a compact list of 32 predictions on identical unlabelled probes, plus model reliability summaries on a disjoint calibration stream. Each route picks the strongest peer from each role, normalizes reliability-based peer weights, and sends only a 5-peer selection and the shared compressed forecast probes to the second wave.

**Alignment:** Predictions are in the same 2D world coordinate system; no claim is made that independently learned hidden-state coordinates from different models are semantically aligned.

**Transport:** Each runner is independent, but GitHub Actions artifacts and barriers introduce high latency. This is a distributed *training experiment*, not a continuous low-latency neuron communication fabric.

## Learning and isolation

- Wave 1: 20 parallel workers, 360 independently generated labelled scenes per worker.
- Route calibration: 150 separate known-answer calibration scenes per worker. Used only to select five peer identities, not to tune against sealed test labels.
- Shared unlabelled probe packet: 32 scenes; only predictions, not future target coordinates, are exchanged.
- Wave 2: 20 parallel workers, 180 additional locally labelled scenes each. Every worker independently trains both a no-peers control and a peer-distilled candidate.
- Test: three held-out seeds `726031, 726037, 726043`, each with 160 scenes balanced across five motion regimes. The evaluator uses the same examples and per-node sensor packets for both strategies.

**Important experimental qualification:** The synthetic generator is installed in the repository, so this is *workflow-isolated evaluation*, not cryptographically unknowable test information. A stronger scientific test needs externally supplied real-world data and untouched evaluation protocols.

## Stable checkpoint and deliberate continuation

First experiment: **Actions → NeuralMesh 20 Local Learners - Sparse Cooperation → Run workflow**, leaving `parent_run_id` blank.

When it completes, GitHub retains `sparse20-checkpoint-<run_id>/checkpoint.json`. That checkpoint is **candidate-only**, not automatically promoted.

For an intentionally separate second learning generation, dispatch the same workflow again with a **new seed** and set `parent_run_id` to the earlier run's numeric identifier. All 20 new worker jobs load the exact checkpoint, verify its SHA-256 digest and 20-node population, and continue local adaptation. A tampered or mismatched checkpoint makes the experiment fail closed. This is a genuinely cross-run persisted state, bounded by GitHub artifact retention.

The code supports a generation chain up to a guarded limit. No external model registry, service credential, encryption, or hidden write permission is required.

## Acceptance and honest reporting

Infrastructure success requires 40 separate training jobs, successful routing, successful evaluation, and a confirmed peak of 20 running simultaneously in *both* waves.

Scientific success is **separate**: peer learning must beat independent local learning on total held-out MSE and on at least two of the three held-out seed cohorts. The report also explicitly measures whether the whole twenty-way ensemble beats the strongest individual specialist. It does not invent a positive result when communication hurts.

A failed scientific gate means **do not promote or tune against those same exposed test seeds**. Retain the baseline, log the candidate as a failed learning attempt, and design a genuinely separate research hypothesis before making further architecture changes.

## Missing capabilities (not disguised)

- These are *tiny numerical specialist models*, not Gemma, Qwen, MiniLM, LFM2, ONNX encoder, multimodal vision models, or a low-rank adapter on a language model.
- Peer messages are next-position predictions, not learned language or image embeddings.
- There is no online asynchronous always-running agent. Both waves are scheduled jobs with a barrier.
- Learning improvements are evaluated in a synthetic 2D task; real-world transfer and no-forgetting on multimodal streams are not established.
- This is a candidate foundation for modular mini-models. Follow-on encoder interfaces and aligned embeddings require independent tests, not claims that this system already has them.

## Files

`neural_mesh/sparse20.py`: deterministic local learning, sparse peer selection, distillation, retained model-state validation and comparison.

`tests/test_sparse20.py`: unit/contract tests for distinct local learning, five selected peers, checksum integrity, matched independent controls, node dropouts and resuming another checkpoint generation.

`.github/workflows/sparse20-learning.yml`: physical twenty-runner learning waves, evidence and candidate memory artifacts, concurrency audit and independent science gate.

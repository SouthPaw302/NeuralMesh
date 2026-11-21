# Phase 2.4 — Robustness, Fault Tolerance, and Neural Memory

## Purpose

Phase 2.3 learned a validation-selected feedback controller; it did **not**
establish how the mesh behaves with unreliable nodes, motion outside its
training distribution, or lost runner sessions. Phase 2.4 tests three separate
engineering questions:

1. Does a **missing** or **grossly corrupted** specialist destabilize the system?
2. Can a fault-aware router improve predictions when the input motion is
   **outside** the four motion regimes used for training?
3. Can we **persist a checksummed learned state** and load it inside separate
   disposable GitHub runners without retraining or silently changing it?

## State / memory model

The memory checkpoint is plain JSON, with no encryption or hidden state.
It is versioned as `neuralmesh-memory-v1`. It includes the frozen six
specialist readouts, Phase 2.2 attention and peer residual readouts, Phase 2.3
selective-feedback model, and Phase 2.4 fault-calibration parameters.

Every checkpoint contains a canonical SHA-256 digest of its complete payload.
Restoration refuses altered state, unsupported format versions, misaligned
model/config identifiers and auto-promoted weights. The checkpoint is uploaded
as an Actions artifact and restored in three *independent runner jobs* later
in the same workflow. This demonstrates persistence across runner lifetimes,
but **not yet** long-lived, multi-run memory; artifacts expire and must be
retrieved by ID. A separate cross-workflow replay should verify that next.

No user memory, private chat data, or other repositories are involved.

## Experiment

**Training:** Fixed Phase 2.3 seed 64013; 512 training and 128 validation
synthetic observations. Test labels from that fixture are not uploaded into
the trainer.

**Validation:** 16 candidate reliability settings, covering prediction
outlier rejection cutoffs and the fraction of retained feedback. Each option
is scored under three synthetic situations using validation targets:
clean, one missing node, and one obviously corrupted node.

**Novel-motion held-out test:** Three independent synthetic seeds
(`48017`, `58129`, `68147`), 128 scenes each (384 total), with four
motion regimes excluded from the Phase 2.1–2.3 training generators:

- `bounce`: a bounded velocity reflection
- `stop_start`: an abrupt pause followed by a restart
- `swirl`: curved trajectories with faster angular velocity
- `late_impulse`: a change immediately before the target frame, deliberately
  difficult to predict from the observations

For each scene the system evaluates:

- `none`: all six healthy nodes
- `missing`: one node omits its prediction and embedding
- `corrupt`: one node transmits deliberately offset prediction/embedding

Failures are deterministic and rotated between specialists. The restored
checkpoint chooses the fault-aware policy **without seeing novel-motion
target coordinates**. We report two policies on exactly the same cases:

- `naive`: previous feedback router with median imputation for missing nodes,
  but no corruption detection.
- `robust`: median/MAD-like agreement screening, zero attention to rejected
  nodes, confidence-renormalized consensus, and validation-selected residual
  damping.

This only tests gross single-node corruption. It does not address subtle,
coordinated Byzantine faults, adversarial security, real sensor failure,
or general fault diagnosis.

## GitHub Actions execution

The branch-scoped first push of
`.github/workflows/phase24-robustness.yml` launches one experiment:

1. Unit tests.
2. Prepare model training/validation data and three sealed novel-motion splits.
3. Train/checksum checkpoint using training and validation data **only**.
4. Restore the checkpoint in three independent probe runners, one per
   novel-motion seed; no test targets are available there.
5. Three evaluation runners receive only predictions and sealed targets.
6. Aggregate exactly three matching seed reports from the same checkpoint.

The experiment uses **logical six-node ensembles** inside each probe runner.
It does **not** allocate 18 distinct model-serving runners. Earlier phases
already established physical six-runner execution; this phase primarily
addresses fault handling and durable state.

The evidence artifact `p24-evidence-<run-id>` contains `summary.json` and
the three seed reports. The separate `p24-memory-<run-id>` artifact contains
the full versioned model checkpoint. Both are ephemeral, with 30-day retention
configured. Repository `main` is not modified and checkpoints are not
automatically promoted.

## Acceptance criteria

- All regression and integrity tests pass.
- The checkpoint loads with the same SHA-256 identity in separate runners.
- Evaluation rejects missing/wrong seeds, mismatched memory versions, and
  altered/mismatched public input digests.
- All novel-motion and fault scenarios produce measured errors; no score is
  interpreted as a successful creative render.
- Report clean-data regression **and** missing/corrupt improvements. Improved
  average error does not erase harms within individual regimes.

## Known limitations

This is **not** a biological brain or an online plasticity system. Its
persisted memory is a **fixed fitted model/checkpoint**, not accumulated
experience or user data. Inter-runner communication is artifact-based and
high latency. The OOD regimes are invented synthetic physics, not real-world
camera or AR measurements. Missing/corrupt faults are intentionally strong;
harder subtle corruption remains open research.

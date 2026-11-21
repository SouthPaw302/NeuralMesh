# NeuralMesh — One active experiment, one stopping rule

**This file replaces the open-ended numbered-phase development loop.**

## What was consolidated

- The deployed `main` branch includes the working six-runner infrastructure and the tested Phase 2.1–2.6 research code, merged in dependency order from PRs **#1–#7**.
- The additional trust-gating attempts **#8–#10 failed their learning criteria**. They have been closed, while their branches, experiment results and review discussions remain in GitHub. They are NOT integrated into the active learning stack.
- The protected model is a versioned numerical checkpoint, **not** a biological brain, general-purpose AI, autonomous online plasticity or proof that recurrent communication always helps.

## Canonical work streams (only two)

1. **Physical distributed runtime:** `.github/workflows/neural-mesh.yml` exercises six model-serving GitHub runners and checks independent communication. It proves orchestration, not improved learning.
2. **Scientific model benchmark:** `.github/workflows/transfer26-domains.yml` regenerates a verified frozen checkpoint **from repository code** and trains the tested physics-expert mixture on fixed training/validation seeds. It produces a separate blind evaluation on nine *new* held-out test cohorts and a separate scientific acceptance job. It no longer needs any **historical** GitHub Actions artifact.

Older workflows remain available for reproducing historical studies; they are not a mandate to continue numbering phases or changing models.

## Fixed architecture

Six specialists ingest noisy 2D observations. A read-only pretrained parent combines specialist predictions using contextual synapses and validation-selected robustness handling. A supervised physics adapter learns to interpolate between this parent and velocity/acceleration/rotation/restart experts. No learned model is automatically promoted.

The current default is still the **Phase 2.4 protected parent**. The physics adapter is a research candidate, not an irrevocably committed active state.

## Locked replication protocol

The new canonical benchmark retains the established training populations (Phase 2.4 parent seed `64013`, fresh familiar seed `26111`, bounce/swirl train `46111`, validation `46112`) to measure **reproducibility**, without changing a single model parameter or hyperparameter rule from the previously measured physics adapter.

The parent is rebuilt using training seed **64013**, 512 training, 128 validation and 256 test examples, then must reproduce its exact fingerprint:

`8c147e3d6bad62cd9b6553738cba40dddc4f5120755273f01a366f3e235d22bc`

If the fingerprint differs, **STOP**. Do not quietly train on a different ancestor.

The evaluator uses the following **previously unused** synthetic seeds:

| Population | Held-out seeds | Scenes |
|---|---|---:|
| Familiar motion | `36121, 36122, 36123` | 3 × 256 = 768 |
| Fresh bounce/swirl | `56121, 56122, 56123` | 3 × 128 = 384 |
| Never-trained stop/start and late impulse | `56131, 56132, 56133` | 3 × 128 = 384 |

Total: **1,536 held-out scenes** and 4,608 healthy/missing/corrupted specialist evaluations. Prediction targets are distributed in an evaluator-only artifact, never in the training runner. The parent and adapter are evaluated on exactly the same input-target pairs.

## Scientific acceptance — all mandatory

- At least two of three seeds improve in **each** population.
- Overall familiar ID mean squared error improves.
- Known-new bounce/swirl mean squared error improves by **at least 10%**.
- Never-trained motion mean squared error does **not worsen**.
- Familiar **linear and turning MSE must each not worsen** relative to the protected parent (aggregated across three ID seeds).

Passing code tests or a green model-training job does **not** override these rules. A separate `scientific-acceptance` job turns **red** if the model fails. The evaluator always publishes its measured metrics first.

## Stop/go contract

**GO (research-only):** If the protected frozen candidate passes *all* predeclared rules on the new held-out seeds, the evaluation is marked **eligible for human review**. A reviewed memory-registry change is still required to activate it.

**STOP:** If any rule fails, **do not** lower a threshold, add another corrective trust layer, rename a failed experiment as a successful phase, or run the same sealed seeds again while tuning the model. Keep the tested parent active, publish the failure and diagnose the limitation. Any materially different future experiment requires a written falsifiable hypothesis and entirely new unused holdouts before coding.

Do not run continuous expensive scheduled jobs. GitHub Actions is a finite high-latency compute platform; it is not a real-time neuron fabric.

## Reproduction

- On GitHub, run **Actions → NeuralMesh Canonical Learning Experiment → Run workflow**. This generates all training and test data within one independently reproducible workflow execution.
- Inspect the `transfer26-evidence-<run_id>` artifact and the **scientific-acceptance** job result.
- Use **Actions → NeuralMesh Distributed Proof** separately when demonstrating physical six-runner job communication, not as evidence of improving prediction.

There is no automatic self-promoting memory or background work. Everything is explicit, versioned and auditable.

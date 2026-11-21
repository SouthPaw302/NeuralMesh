# Phase 2.1 — Multi-scenario synaptic plasticity

Status: **bounded synthetic evaluation**. This is not a claim of emergent intelligence, a biological neural network, or real video/AR capability.

## Research question

Can a routing layer **trained across many independent motion scenarios** improve held-out forecasting relative to a fixed weighted network and a single preselected specialist?

The six specialist numerical models from Phase 1 remain fixed. This phase trains **inter-node combination/routing weights**, not the models' internal neural weights. Every specialist receives different noisy four-frame observations of the same target motion.

## Experimental design

- **512 training** scenes, **128 validation** scenes, **256 sealed test** scenes (default workflow).
- Four deterministic, stratified motion regimes: linear, constant acceleration, acceleration with jerk, and turning.
- Independent random seeds per split and scene; fixed specialists trained separately on the earlier synthetic generator.
- Each specialist is run on a separate GitHub-hosted runner. First-round workers never see target positions.
- After all first-round receipts arrive, a training job uses **only train + validation targets** to choose its connections.
- The objective fits a nonnegative six-node weight vector on the training split with simplex-projected gradient descent and regularization. Validation chooses regularization, an input-conditioned disagreement penalty, and recurrent feedback fraction.
- Six second-round runners apply that frozen routing to their peers' forecasts; the final evaluator uses previously inaccessible test targets.
- Outputs include the full validation-search record and a versioned frozen router, but are never automatically promoted to persistent state.

### Benchmarks

All comparisons use the **same 256 held-out test scenarios** and mean squared 2D next-position error:

1. **Selected single specialist:** best specialist chosen by validation, not test.
2. **Retrospective best specialist:** best overall test specialist reported as an *optimistic reference*, never usable as a model selected at inference.
3. **Uniform ensemble:** equally weighted first-round predictions.
4. **Phase 1 fixed routing:** both first- and second-round results using prior error/disagreement weights and 0.28 feedback.
5. **Phase 2.1 learned routing:** first- and second-round results with frozen, trained routing.

The report includes per-motion-regime errors and deterministic paired-bootstrap 95% intervals of mean error differences. **A green CI run proves protocol execution, not an improvement in accuracy.** An interval spanning zero is inconclusive.

### Information boundaries

```text
Prepare: creates public observations + fixed weights
         /             |                 \
 public inputs     train+validation    TEST TARGETS
       |                 |                 |
  Six blind jobs         |                 |
       |                 |                 |
  round-1 receipts ------+                 |
       |                                   |
 train-synapses (frozen checkpoint)        |
       |                                   |
  Six feedback jobs                        |
       |                                   |
  evaluator <-------------------------------+
       |
 metrics.json (never used for training)
```

The separation is enforced by each job's artifact download paths and by tests rejecting leaked label partitions. GitHub workflow maintainers can still access repository artifacts; this is **experimental isolation, not an access-control security guarantee**.

## Run

Use **Actions → NeuralMesh Phase 2.1 - Held-out Plasticity Benchmark → Run workflow** once the workflow is on the default branch. On the feature branch, the workflow's initial branch-scoped push launches one full experiment. Pull requests only run offline tests to avoid repeated parallel compute usage.

The default workflow runs **12 specialist job executions in two successive waves of up to six concurrent runners**, plus preparation, routing training, evaluation and tests.

After completion, inspect `phase21-evidence-<run_id>`:
- `metrics.json`: all comparative test errors, per-regime errors, confidence intervals and selected router.
- `frozen-router.json`: training/validation-selected parameters and complete candidate search.
- `manifest.json`: scenario counts, dataset seed and label visibility scope.

## Reproduce locally

Python 3.11 or newer, standard library only. This runs the same protocol without simulating GitHub scheduling:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python -m neural_mesh.phase2 prepare --seed 314159 --train-count 512 --validation-count 128 --test-count 256 --out fixture
mkdir -p receipts
for node in position velocity acceleration short_horizon long_horizon stabilizer; do
  python -m neural_mesh.phase2 infer --node "$node" --public fixture/public.json --weights fixture/weights.json --out "receipts/node-$node.json"
done
python -m neural_mesh.phase2 train --public fixture/public.json --weights fixture/weights.json --receipts receipts --trainval fixture/trainval-labels.json --out trained
mkdir -p refined
for node in position velocity acceleration short_horizon long_horizon stabilizer; do
  python -m neural_mesh.phase2 refine --node "$node" --round1 trained/round1.json --router trained/router.json --out "refined/node-$node.json"
done
python -m neural_mesh.phase2 evaluate --round1 trained/round1.json --router trained/router.json --refined refined --weights fixture/weights.json --test-labels fixture/test-labels.json --out evidence
```

## Known limits

- Generated points are not real observed video or spatial imagery.
- Six simple learned readouts plus an adaptive weighting layer are not tensor-parallel layers of a large neural network.
- Output feedback through GitHub artifacts is high latency and unsuitable for token-by-token or frame-by-frame neural inference.
- Using validation to select among 36 combinations may overfit validation; the independent test split measures generalization within this synthetic distribution.
- The test set is drawn from the same four predefined synthetic regimes as training. Out-of-distribution robustness is *not* established.
- Bootstrapped confidence intervals are descriptive finite-sample estimates, not proof of a mechanism.
- No performance improvement should be claimed until the test metrics are inspected.

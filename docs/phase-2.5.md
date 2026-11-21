# Phase 2.5 — Controlled memory evolution and Git-backed registry

## Purpose

The mesh previously restored a frozen checkpoint across separate GitHub
workflow executions. Phase 2.5 tests the next step: **learning new
connection parameters from a new training cohort**, retaining an immutable
parent, checking for regressions, and allowing an explicitly approved
GitHub pull request to add a durable memory version.

This is not continuously running neural tissue, autonomous self-modification,
background learning, or an indefinitely reliable memory service.

## Evolution protocol

The first *bootstrap* generation loads the exact Phase 2.4 checkpoint from
Actions run `44871190473`:

`8c147e3d6bad62cd9b6553738cba40dddc4f5120755273f01a366f3e235d22bc`

The learner then:

1. Generates a **fresh training cohort**, using synthetic seed `2505`, with
   512 train, 128 validation, and 256 sealed test scenes.
2. Recreates a historical nonsealed **regression anchor** from Phase 2.4
   seed `64013`; the earlier model was fitted on that cohort.
3. Loads the previously trained numerical specialists, contextual attention,
   selective feedback, and fault policy; **does not retrain them all**.
4. Fits a 17-feature peer-correction readout from new training labels.
   Regularized normal equations penalize drifting from the parent's readout.
5. Tests candidate regularization values on the new validation split,
   historical anchor validation split, and fault-injected validation
   examples. A candidate must improve new validation by at least 0.5%,
   worsen the anchor by no more than 3%, and worsen fault validation by
   no more than 5%. If no candidate qualifies, parent state remains
   unchanged and the proposal is marked **blocked**.
6. Writes a digest-verified, versioned child checkpoint with explicit
   parent SHA-256 and lineage/history. It never overwrites the parent.
7. Separately evaluates the **unseen** new held-out 256 scenes and
   128 OOD motion scenes from seed `82521`, each in clean, missing-node,
   and grossly corrupted-node conditions.
8. Emits a **promotion gate** recording whether the candidate stays within
   predeclared held-out tolerances: 4% aggregate relative regression on the
   original distribution, 7% on OOD, 15% per original-motion regime,
   and 20% per OOD regime. The gate is a **tolerance check**, not proof
   that the child outperforms the parent.

The original training job **does not download any new test targets**.
The sealed evaluation job never changes the learned checkpoint.

## Workflow safety

### Automatic test / bounded candidate generation

`.github/workflows/phase25-evolution.yml`

- First branch-specific workflow push performs one bounded experiment.
- Pull-request updates run unit tests only, avoiding unnecessary repeated
  compute and repeated viewing of the sealed test.
- Repository permissions are **contents:read, actions:read**.
- It downloads the frozen Phase 2.4 checkpoint from the *earlier run*
  and produces a separate `parent.json` and `candidate.json`.
- Evidence: `phase25-review-<run_id>` includes both snapshots,
  `training-decision.json`, `evaluation.json`, and
  `promotion-gate.json`.
- No workflow stage modifies or merges the main branch.

### Explicit, manual-only Git registry PR

`.github/workflows/phase25-promote-reviewed.yml`

This workflow becomes available for **manual dispatch only after the
implementation is merged onto the default branch**. The reviewer supplies
the experiment run ID, the full expected candidate SHA-256, and the exact
acknowledgment `OPEN_REVIEW_PR_ONLY`.

The action checks that the experiment run succeeded in the same repository,
downloads the original evidence bundle, verifies parent/candidate and
evaluation digests, and refuses a blocked promotion gate. If all checks
pass, it stages two immutable plain-JSON snapshots in
`memory/versions/<sha256>.json`, an append-only index in
`memory/registry/index.json`, and an audit record. It creates a new
`memory/review-*` branch and opens a separate pull request to `main`.

**The PR is never auto-merged.** The memory registry becomes durable in Git
**only if a maintainer reviews and merges that PR**. If GitHub permissions
prevent workflows from creating PRs, that operation will fail closed and
requires repository settings/permissions to be reviewed by the owner.

This two-step process is intentional: the initial experiment's apparently
passing gate does **not** authorize replacing the current learned memory.

## Current limits

- The first implementation is **bootstrap-generation-scoped**, anchored
  to the exact Phase 2.4 SHA-256. A later iteration must explicitly
  load the active memory head from the Git registry before attempting
  a third generation. It must not silently reuse the original parent.
- GitHub Actions artifact retention is 30 days by configuration. The
  proposed Git memory registry is durable only after its review PR merges.
- A tolerance gate can permit small accuracy declines. Review must examine
  **all healthy, missing, corrupt, and OOD test scores**, not only
  the presence of a `PROMOTION_ELIGIBLE` label.
- The synthetic scene families are not real video, spatial perception,
  or general AI intelligence.
- Only one 17-feature feedback readout learns in this iteration. There
  is no online learning loop or hidden user-data collection.

See [the first Phase 2.5 experiment results](phase-2.5-results.md) before
considering the candidate for promotion.

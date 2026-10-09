# Job-tree prototype: brief

Agreed with Pleiades on 2026-10-08. This is the shared input for every agent of the prototype workflow.

## Goal

Prototype the per-tree committer design in `claude/job-tree/design/per-tree-committer.md` (the current version) on the storage side, prove it under the race and restart cases below, benchmark it against `main`, and report.

## Where

- Worktree: `/home/lzh/dev/spider-job-tree-proto`, branch `job-tree-prototype`, created from `278d06f` (upstream `main`). All code changes go here.
- Do not touch `/home/lzh/dev/spider` tracked files (the main checkout). It stays at `278d06f` and is the benchmark baseline.
- Do not commit, push, rebase or create branches. Work in the worktree's working tree; `git diff 278d06f` is the prototype.
- Benchmark harness: `/home/lzh/dev/spider/claude/benchmarks/job-tree-proto/` (a standalone Cargo project with its own `[workspace]`, like `claude/benchmarks/job-tree/`).
- Workflow outputs (plans, reviews, results, report): `/home/lzh/dev/spider/claude/job-tree/prototype/`.

## Scope decisions

- **Storage side only, driven by an execution-manager (EM) harness.** No TDL `create_child_job` and no task-executor or execution-manager transport changes. The harness plays the scheduler and the EM: it polls ready work, registers task instances, and reports success, with the new `children` field, or failure.
- **Design as of the current doc:**
  - a long-running committer per tree, with an unbounded mpsc channel, `recv_many`, a local deferred buffer and a quiescent exit;
  - a tree state transaction is a single DB transaction;
  - the planner orders the events: cancel first, then arrival order, then at most one registering parent per tree state transaction;
  - an overlay;
  - no `Audit` event and no runtime reconciler.
- **In scope:**
  - `BlockingOnChildren`, `parent_id` and `root_id`, with the idempotent upgrade;
  - the session fence;
  - a non-zero exit on the fatal path;
  - `apply_tree_transition`;
  - the committer, the planner and the publisher;
  - spawned handler bodies;
  - registration, with validation, limits and prebuilt child graphs;
  - the completion cascade, failure plus tree abort, and root-only cancel with the `CommitReady` skip;
  - multi-job GC enqueue;
  - restart recovery with repair R1–R5;
  - the `fetch_sub` `AcqRel` ordering.
- **Out of scope:** the TDL API, EM/executor transport, retention, WaitJob, the lazy child-output fetch, the Helm/Compose changes (mention them only), and renaming `apply_tree_transition`.
- **Memory concern to check:** a parked committer future may be large if the tree-state-transaction future is inlined. Box the per-round future (`Box::pin`) unless measurement says otherwise (benchmark B6).
- **Test hooks:** deterministic fault and pause injection points (for example before planning, after COMMIT and before apply, and inside phase A), behind `#[cfg(test)]` or a non-default cargo feature, so the race tests are deterministic, not sleep-based.

## Tests (correctness; deterministic; exact asserts; race tests repeated many times)

- T1: two different tasks fail the same job (same and different tree state transactions).
- T2: a root cancel races leaf job creation (registration).
- T3: a root cancel races a leaf job failure.
- T4: siblings complete at the same time (exactly-once parent advance; outputs visible).
- T5: a straggler fails a task after it succeeded.
- T6: a parent with a commit task (children done → `CommitReady` → commit → climb).
- T7: a cancel while a non-root is `CommitReady` (skip) vs. a root `CommitReady`.
- T8: a storage restart at crash points (before and after a registration's COMMIT, mid-cascade, after a cancel's COMMIT); the tree converges, nothing is registered twice, and a `BlockingOnChildren` parent is recovered and unblocks.
- T9: an RPC future dropped mid-report; the job is not stranded.
- T10: a limit violation (children count or bytes) is a job failure, not a crash.
- T11: a planner property test: random event interleavings against an in-memory row model; invariants hold after every tree state transaction.

## Benchmarks (cost; quiet host; ≥10 trials; an A/A control where it applies)

The kind cluster `clp-rc2-control-plane` was stopped by Pleiades's request for a quiet host. Do not start it.

- B1: plain-job transition latency. Last-task `ReportTaskSuccess` p50/p99, and end-to-end submit → `Succeeded` with no-op tasks, at 1 and 64 concurrent writers. `main` vs. the prototype.
- B2: depth-7 binary job tree (127 jobs, 64 leaves) end to end, with no-op tasks and with 10 ms tasks, vs. (a) a single job whose task graph has the same binary shape (127 tasks) and (b) a single flat job with 64 leaf tasks. Report the per-level overhead.
- B3: the race cases as latencies on the depth-7 tree: cancel acknowledgement, time to quiescence, and a leaf failure → root `Failed` and quiescent.
- B4: sibling fan-in: 1,000 children finishing at once (throughput, events per tree state transaction).
- B5: concurrent trees (16 and 64): cross-tree isolation and connection use.
- B6: memory per live tree, for example 100k live plain jobs: RSS on `main` vs. the prototype, boxed vs. unboxed round future.
- B7: restart recovery time with N live trees: `main` vs. the prototype.

## Report (`claude/job-tree/prototype/report.md`)

- Lines changed, broken down by file and component, excluding tests and the harness, compared with the design's estimate (about 1,770 storage lines).
- Traps the formal implementation must watch for.
- Test results.
- Benchmark results.
- Deviations from the design doc: gaps or wrong assumptions found while implementing.

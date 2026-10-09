# Job Tree Prototype: Report

| | |
|---|---|
| Date | 2026-10-09 |
| Design | `claude/job-tree/design/per-tree-committer.md` (current version) |
| Brief | `brief.md` (this directory): storage side only, driven by an EM harness; tests T1-T11; benchmarks B1-B7 |
| Code | worktree `/home/lzh/dev/spider-job-tree-proto`, branch `job-tree-prototype`, base `278d06f` (upstream `main`), uncommitted. `git diff 278d06f` plus the untracked files under `components/` is the prototype. Its `components/` diff hash is `d4cda1b5a3428faa`, the same tree that the benchmarks measured. |
| Sources | `plan.md`, `plan-review.md`, `slices/` (implementation notes S1-S6), `reviews/` (slice reviews, the style review, the test-phase review), `tests.md`, `bench/results.md` (version 2), `bench/review.md` |
| Roles | architect (plan), plan reviewer, implementer, slice reviewer, style reviewer, test engineer, benchmark engineer, benchmark methodology reviewer |

## 1. Summary

**Status of each phase.**

| Phase | Status | Notes |
|---|---|---|
| Plan | Done | Revision 2 after the plan review, which raised 4 major and 11 minor issues, all resolved in the plan. The majors were: a stale `StartJob` that would poison the service; tests sharing one database with a recovery that loads every row; the in-process restart skipping part of the real recovery path; and B6/B7 overflowing the default task lane. |
| Implementation, 6 slices | Green | S1 schema and fence, S2 size-1 committer, S3 planner, S4 registration, S5 propagation and cancel, S6 recovery. Every slice had an adversarial review: 50 issues, 11 of them serious (S1 1, S2 1, S3 1, S4 3, S5 5, S6 0), all fixed before the next phase. The first S5 review found the slice missing from the worktree. The second attempt implemented it and fixed the four majors that review had found in the S3/S4 code. |
| Style review | Done, mostly applied | 24 items. A fix pass then did the following: the hooks are always compiled, with a zero-sized `TestHooks` when the feature is off; the phantom generic parameters are gone; the old `DbStorage` writers and their 18 `mariadb_test` cases are deleted; `TreeClosed` is renamed `Untracked`; the limits and the property tests moved into their own files; `unboxed-round` became a `cfg` flag; and the nightly clippy error is fixed. That pass has no notes file, and no review covered it (§7). |
| Tests T1-T11 | Green | 4 test groups. About 277,000 hook-driven trees or cases ran in release builds at 1,000 iterations, each test also ran 100 iterations in a debug build, and the property tests planned about 800,000 rounds and repaired 400,000 trees. No implementation defect was found in this phase. 47 deliberate mutations were applied; all but one were caught, and that one is unreachable by design. Full `spider-storage` suite: 283/284 per run. The one failure differs per run and is a pre-existing timing flake that also fails on `main`. |
| Benchmarks B1-B7 | Complete (version 2) | At least 10 trials each on a quiet host, with A/A controls. A methodology review then required 9 fixes, all applied, and B1, B2, B3 and B5 were rerun. Every correctness check passed in every run. |

**Did the design hold up?** On correctness, yes. These parts all behaved as designed under every forced interleaving and crash point:
- the per-tree committer;
- one DB transaction per tree state transaction;
- cancel-first ordering;
- exact from-state guards;
- the session fence;
- repair-based recovery.

The race and restart tests found no implementation defect, and no design-level correctness flaw surfaced.

Where the design did not hold up:
- **Limits.** The limits did not fully bound what a tree state transaction writes. Three gaps would have turned in-limit workloads into a crash loop of the whole service: a failure message larger than `max_allowed_packet`; the union of a job's outputs exceeding the packet; and per-job child limits checked only after decoding.
- **Root-only RPCs.** Root-only enforcement needs care with ordering. A cache-miss `CancelJob` on a child in the registration window caused a fatal shutdown, and a cached child's `CancelJob` cancelled its whole tree.
- **Memory.** Per-tree memory is about 12x the design's "a few hundred bytes": 3.6 KB per live plain job, plus a 10.4 KB pre-sized buffer that is a one-line fix (B6).
- **Wide trees.** "One registering parent per tree state transaction" makes a wide tree's expansion linear in its internal jobs: a depth-7 binary tree takes 7.0-7.1x the time of the same shape inside one job (B2).
- **Job creation.** `RegisterJob` is 23-38 % slower with the fence. No regression was detected on plain-job transitions (B1).
- **Size.** The storage diff is about 3.4x the design's 1,770-line estimate, or 2.4x counting code lines only.

## 2. Lines changed

### 2.1 Method

A script (`count_lines.py` in the session scratchpad) runs over `git diff -U0 278d06f` for modified files, and over whole files for untracked files under `components/`. It classifies every added and removed line:
- **Test.** A file under `tests/`, `state/test_utils.rs` or `cache/tree/property.rs`, or a line inside a `#[cfg(test)]` item of a production file (the attribute plus the gated item, found by brace matching). Removed lines are classified against the base file's own `#[cfg(test)]` ranges.
- **Hook infrastructure.** `src/test_hooks/`, plus items gated on `#[cfg(feature = "test-hooks")]` (the injected-fault error variants).
- **Hook call sites.** Code lines in production files that name a hook (`hooks`, `HookPoint`, `HookContext`, `InjectedFault`, `test_hooks`). A single-line match finds 78. A variant that follows each statement across its continuation lines finds about 115. These lines are counted inside the production totals and also shown separately.
- **Generated.** `spider-proto-rust/src/generated/`.
- **Production.** Everything else, split into code, docstrings/comments (`///`, `//!`, `//`) and blank lines.

The debug-only `assert_invariants` (gated on `any(test, debug_assertions, feature = "test-hooks")`) is counted as production. The benchmark harness lives outside the worktree and is not counted.

### 2.2 Totals

| Scope | Added | Removed | Net |
|---|---|---|---|
| **`spider-storage` production** | **6,061** (4,202 code + 78 hook call sites, 1,429 docstrings/comments, 352 blank) | 1,152 (849 code) | **+4,909** |
| `spider-core` production | 20 | 5 | +15 |
| `spider-proto` + `spider-proto-rust` production | 20 | 1 | +19 |
| `spider-execution-manager` | 1 | 0 | +1 (the `children: Vec::new()` compile fix) |
| Generated protobuf code | 13 | 1 | +12 |
| Hook infrastructure (648 lines in `src/test_hooks/` plus 10 feature-gated error-variant lines) | 658 | 0 | +658 |
| Tests (all components) | 19,556 | 643 | +18,913 |
| of which `spider-storage/tests/` | 14,594 | 518 | +14,076 |
| of which in-source (`#[cfg(test)]` modules, `test_utils.rs`, `property.rs` 1,977) | 4,962 | 125 | +4,837 |

The 648 lines of `src/test_hooks/` split into three files:
- `armed.rs`, 421 lines, compiled only with the feature;
- `mod.rs`, 158 lines, always compiled;
- `disarmed.rs`, 69 lines, the production no-op shim.

The removed tests are the 18 `mariadb_test` cases of the deleted `DbStorage` writers (−416 lines; 53 tests became 35) plus mechanical harness edits.

### 2.3 By component (`spider-storage`, production only)

| Component | + code | + docs/comments | + blank | + hook call sites | + total | − total (− code) | Net |
|---|---|---|---|---|---|---|---|
| `db/` | 678 | 191 | 63 | 0 | 932 | 271 (162) | +661 |
| `cache/` | 1,734 | 695 | 170 | 29 | 2,628 | 494 (340) | +2,134 |
| `state/` | 1,597 | 485 | 95 | 48 | 2,225 | 307 (273) | +1,918 |
| `grpc.rs` | 49 | 16 | 3 | 0 | 68 | 45 (39) | +23 |
| `bin/` | 99 | 17 | 6 | 0 | 122 | 31 (31) | +91 |
| other (`config.rs`, `lib.rs`, `limits.rs`, `job_submission.rs`, `task_instance_pool.rs`, `Cargo.toml`) | 45 | 25 | 15 | 1 | 86 | 4 (4) | +82 |
| **Total** | **4,202** | **1,429** | **352** | **78** | **6,061** | **1,152 (849)** | **+4,909** |

### 2.4 By file (production lines only)

| File | + code | + docs/comments | + blank | + hook call sites | − | Net |
|---|---|---|---|---|---|---|
| `spider-core/src/job.rs` | 19 | 0 | 1 | 0 | 5 | +15 |
| `spider-execution-manager/src/client/grpc/storage.rs` | 1 | 0 | 0 | 0 | 0 | +1 |
| `spider-proto/storage/storage.proto` | 6 | 0 | 1 | 0 | 0 | +7 |
| `spider-proto-rust/src/job.rs` | 2 | 0 | 0 | 0 | 0 | +2 |
| `spider-proto-rust/src/unpack/storage.rs` | 10 | 1 | 0 | 0 | 1 | +10 |
| `spider-storage/Cargo.toml` | 5 | 0 | 2 | 0 | 0 | +7 |
| `spider-storage/src/bin/grpc_server.rs` | 99 | 17 | 6 | 0 | 31 | +91 |
| `spider-storage/src/cache/error.rs` | 24 | 1 | 8 | 0 | 2 | +31 |
| `spider-storage/src/cache/job.rs` | 391 | 137 | 17 | 11 | 410 | +146 |
| `spider-storage/src/cache/mod.rs` | 1 | 0 | 0 | 0 | 0 | +1 |
| `spider-storage/src/cache/task.rs` | 139 | 61 | 14 | 7 | 82 | +139 |
| `spider-storage/src/cache/tree.rs` (new) | 1,179 | 496 | 131 | 11 | 0 | +1,817 |
| `spider-storage/src/config.rs` | 2 | 2 | 1 | 0 | 0 | +5 |
| `spider-storage/src/db/error.rs` | 18 | 4 | 5 | 0 | 3 | +24 |
| `spider-storage/src/db/mariadb.rs` | 609 | 123 | 45 | 0 | 180 | +597 |
| `spider-storage/src/db/mod.rs` | 3 | 0 | 0 | 0 | 0 | +3 |
| `spider-storage/src/db/protocol.rs` | 48 | 64 | 13 | 0 | 88 | +37 |
| `spider-storage/src/grpc.rs` | 49 | 16 | 3 | 0 | 45 | +23 |
| `spider-storage/src/job_submission.rs` | 23 | 12 | 3 | 0 | 0 | +38 |
| `spider-storage/src/lib.rs` | 1 | 0 | 0 | 1 | 0 | +2 |
| `spider-storage/src/limits.rs` (new) | 9 | 11 | 9 | 0 | 0 | +29 |
| `spider-storage/src/state/error.rs` | 39 | 1 | 8 | 0 | 1 | +47 |
| `spider-storage/src/state/job_cache.rs` | 33 | 9 | 1 | 0 | 34 | +9 |
| `spider-storage/src/state/job_cache_gc.rs` | 11 | 0 | 0 | 0 | 14 | −3 |
| `spider-storage/src/state/mod.rs` | 8 | 0 | 0 | 0 | 0 | +8 |
| `spider-storage/src/state/publisher.rs` (new) | 145 | 39 | 12 | 0 | 0 | +196 |
| `spider-storage/src/state/runtime.rs` | 112 | 31 | 5 | 5 | 79 | +74 |
| `spider-storage/src/state/service.rs` | 461 | 141 | 13 | 17 | 179 | +453 |
| `spider-storage/src/state/shutdown.rs` (new) | 30 | 15 | 8 | 0 | 0 | +53 |
| `spider-storage/src/state/tree_committer.rs` (new) | 758 | 249 | 48 | 26 | 0 | +1,081 |
| `spider-storage/src/task_instance_pool.rs` | 5 | 0 | 0 | 0 | 4 | +1 |

### 2.5 Against the design's estimate

The design estimates "about 1,770 handwritten non-test lines" for the storage side (§16.1). The plan raised that to 2,000-2,400, plus about 200 hook lines.

| Basis | Lines | vs 1,770 |
|---|---|---|
| All added production lines | 6,061 | 3.4x |
| Added code lines (no docstrings, comments or blanks), including hook call sites | 4,280 | 2.4x |
| Net code lines (added − removed code) | 3,431 | 1.9x |

The diff deletes the old `DbStorage` writers, which the design keeps until PR 9. A diff shaped like PR 3 would therefore come out higher still.

Where the overshoot comes from:
- **Docstrings.** 24 % of the added lines. The style guide asks for full docstrings with `# Errors`, `# Returns` and `# Type Parameters`.
- **The planner** (`cache/tree.rs`, 1,179 code lines). It holds the overlay, the eight event planners, the round order with deferral and the payload reservation, `check_invariants` (about 200 lines), `plan_repair` and `from_rows`. The design sketches it as a few short functions.
- **The committer** (758 code lines). Its shutdown, poison and requeue paths, `bind_children` and the `fail_culprit` branch are not costed by the design.
- **`db/mariadb.rs`** (+609 code lines). It holds the fence bump with the hardened blocker watcher (§3.5), the upgrade with DDL lock-wait retries, the statement grouping and chunking by ids and bytes, the insert chunking, the bound on serialized outputs and the payload-error classification.
- **Items no design PR budgets:**
  - cleanup of a failed `build_runtime`;
  - the `ShutdownHandle` and the exit-code plumbing in `grpc_server.rs`;
  - typed report rejections with the per-job child reservation;
  - the root-only paths of five RPCs;
  - message truncation.

## 3. Traps the formal implementation must watch for

Each row gives the pitfall, where it bit (slice and review, test, or benchmark), and how the prototype handled it.

### 3.1 Committer, phase A and locking

| Trap | Where it bit | How the prototype handled it |
|---|---|---|
| A stale `Start` reaching the "no ready tasks" check raises the internal `TaskGraphCorrupted`, which poisons the tree and exits the process. A stale `Start` is a duplicate `StartJob` from the client's retry, or a `StartJob` after a committed cancel. | Plan review (M1) | `collect_ready_task_indices` reads the indices and runs the check only when the collected state is `Ready`; otherwise it answers `Stale(JobAlreadyStarted)`. Two tests cover it, and a mutation that removes the check fails both. |
| `collect_inputs` reading the outputs of a job that a cancel already deactivated raises `TaskInputNotReady`, which is fatal. | Plan R3; the T2 mutation "reads a non-`Running` job" caused a fatal shutdown | Outputs and prepared children are taken only when the collected state is `Running`. |
| An event for a job that has already left `handles` was a fatal `JobNotFound`. Example: a straggler `CleanupDone` or `TerminalFailure` of a terminal child in the commit-to-apply window. | S2 review, issue 3 (unreachable at size 1) | `collect_inputs` reads such a job as `NodeState::Absent`, and the event is answered `Untracked`. The JCB maps that to its cached stale error. |
| A requeued round must clear its events' inputs, or the re-plan of a taken input panics. | S2 review, issue 2 | `requeue_all` clears the inputs; prepared children stay with their event. Covered by a hooked unit test. |
| Every parked committer holds its per-round arrival buffer, `Vec::with_capacity(MAX_EVENTS_PER_ROUND)`: 12 KiB per live tree. | B6 (10.4 KB per live job) | **Not fixed** (`state/tree_committer.rs:438`). `Vec::new()` is correct because `recv_many` grows the buffer, and measured 3.6 KB per tree. Its latency cost was not measured. |
| Callers of a crashed round are answered `TreePoisoned` before the committer sets the fatal flag. | T8 group, finding 1 (a racy `is_fatal()` assert) | Harmless through gRPC, because `default_error_handler` marks the same error fatal; the tests wait for the token. Better: poison before dropping the round's reply channels. |
| `submit_and_wait` and `spawn_body` race the reply against the token without `biased`, so a `CancelJob` that the committer already applied can be answered `Stopping`. | S2 review notes | Accepted. The client's retry after the restart sees a stale error. Decide whether a committed reply should win. |
| A committer dropped at runtime teardown logged "exited abnormally" and called `fatal()`. | S2 review, issue 7 | `PoisonOnAbnormalExit::drop` stays quiet once the token is cancelled, unless the thread is panicking. Explicit error exits still poison. |
| The `has_prepared_children` store must come before `fetch_sub(AcqRel)`, or a sibling's outputs or children are lost. | Plan R6 | Implemented as designed and reviewed. **Not testable on x86-64**: a `Relaxed` mutant passes, because a locked RMW is a full barrier and the outputs are read under TCB locks. Pin it with a `loom` model. |
| Handler bodies must be spawned, including `RegisterJob`'s, which the design omits. Otherwise a dropped RPC future strands its job. | S2 review, issue 6; T9 mutation M5 | Every body runs through `spawn_body` in the tracker, `register_job` included. |
| The planner's "never touch a row twice" deferral never fires for real producers. | S3 notes and review, issue 3; T1 mutation M1, whose missing lane check it masks | Kept as a safety net, with a unit test. The style review leaves `debug_assert!` vs deferral to Pleiades. |
| A `CommitterAfter*` pause that ends on the token must let the round finish, as production's shutdown does. | S2 review, issue 4 | `hit_after_commit` treats `Stopping` after COMMIT as "finish the round". |
| A closure captured across an await in a generic future needs `Send + Sync`; otherwise `clippy::future_not_send` (the repo's `-Dclippy::nursery`) rejects the library. | Test phase | `apply_plan` takes `round_context: impl Fn() -> HookContext + Send + Sync`. |

### 3.2 Planner and round shaping

| Trap | Where it bit | How the prototype handled it |
|---|---|---|
| The registration is planned last (for I4). If the 16 MiB payload budget is then enforced in planning order, the registration loses every round while siblings keep completing. | S3 review, issue 1 (major) | The planner reserves the oldest registration's payload from the start of the round and releases it when that registration is planned. An event without payload is never deferred for the budget. T11 asserts liveness after every round: the oldest event other than the first cancel is planned. |
| Per-job deferral keyed on planning order holds a registration back behind a later event of its own job. A deferred extra registration of the root held back a `CancelRoot`. | S3 property test (the cancel case); S3 review fix (the arrival-order case) | Once a job's event is deferred, its later events (by arrival order) are deferred too. An extra registering event does not trigger this. |
| The planner's own 256-event cap is dead code behind `take_events`. If it were reached, it would starve a registration, because it defers by planning position. | T11 finding 1 (the first burst generator hit it) | Not changed. Keep one cap, in `take_events`, or rank the planner's cap by arrival order. |
| Extra registering events are deferred whole even when they can only be stale. After a cancel, N buffered registering events take N empty rounds. | T11 finding 2 (1,659 rounds for one 14-row tree with about 1,000 copies) | Not changed. Answer an extra registering event at once when its job is not `Running` in the overlay. |
| The round-shaping rules (cancel first, the cap, one registration) live in both `take_events` and `RoundOrder`. | Style review, issue 10; the T2 mutation "take_events does not put the cancel first" shows only in round composition | Not changed. Pick one owner. |
| A non-registering `RegularTasksDone` that fails its job aborts the tree, but `jobs_to_collect` collects only the job and its ancestors for that event kind. | S4 review fix 2 (the first, planner-side version panicked T11) | The per-job output bound moved to the DB layer and the committer (§3.3). A registering `RegularTasksDone` collects every job, as a cancel and a failure do. |
| Every abort sorts the whole tree's ids. | S5 review, issue 7 | Not changed; the cost is small next to the DB transaction. Sort once per round. |
| The stale variant depends on the answer path: phase A gives `JobNoLongerRunning` or `JobNoLongerCommitReady`; the planner and the closed-tree path give `JobAlreadyTerminated(state)`; the TCB gives `InvalidTaskInstanceId` or `TaskAlreadyTerminated`. | S5; T1-T7 | All map to FAILED_PRECONDITION, as the design promises, and the tests pin each variant. Clients must not rely on any one variant. |

### 3.3 Limits and payload

| Trap | Where it bit | How the prototype handled it |
|---|---|---|
| The 32 MiB decode limit lets a task failure message exceed `max_allowed_packet`. The guarded UPDATE that stores it then fails deterministically on every restart. | S4 review, issue 1 (major) | `fail_and_abort`, the one place that writes the row, cuts the message to `MAX_ERROR_MESSAGE_CHARS` (64 Ki); ancestors get 1,024 characters. A MariaDB test fails a job with a message of `max_allowed_packet + 1` bytes. |
| A job's stored outputs were bounded only per report. The union of many tasks' outputs can exceed the packet and crash-loop the service. | S4 review, issue 2 (major) | `UpdateStatement::plan` refuses a row whose serialized outputs exceed `max_allowed_packet − 64 KiB` with `OutputsTooLarge`, before BEGIN. The committer's `fail_culprit` turns that event into a job failure and requeues the round. A first version limited raw bytes and regressed the existing 10,000-task `jcb_test`, whose 10 MiB of raw outputs compress to almost nothing. |
| Per-job child limits checked in the planner run only after every report's children have been decoded, built and held. | S4 review, issue 3 (major) | The service reserves each report's `ChildSetSummary` against the job's totals before decoding, under a mutex in the JCB, and releases it on any rejection. The planner check stays as defence in depth and is now unreachable. Two edge effects: the release is inexact after a late error, and a duplicate report's reservation can spuriously reject a third report at the edge of the limit. |
| Raising the task-instance decode limit to 32 MiB removes tonic's 4 MiB default, which was the effective output limit. | S4 deviation 3 | Every success report's decoded outputs get an explicit 4 MiB check. |
| Only a child insert's payload error may be blamed on the registration. | S4 review, issue 4 | Only `insert_children`'s 1153/1406 errors map to `PayloadRejected`. A test confirms that a real 1153 arrives as a database error, not as an I/O error. |
| A rejected success report is answered with the job state (`Running` on a retry, `Failed` at the end), so the EM cannot tell that its success became a failure. | S4 deviations 4 and 8 | The rejection is logged as a failure, but the RPC answer is unchanged. A distinct status is needed. |
| Children on a commit or cleanup report were silently dropped. | S1 review, issue 5 | Such a report gets `INVALID_ARGUMENT` before any cache access. |

### 3.4 External RPCs and trees

| Trap | Where it bit | How the prototype handled it |
|---|---|---|
| `CancelJob` on a cached child submitted `CancelRoot` for the child. That deactivates the whole tree, and can move a non-root `CommitReady` job to `CleanupReady` while its commit runs. | S5 review, issue 2 (major) | A root check on a cache hit, before `jcb.cancel()`; `plan_cancel` also asserts that the job is the root. |
| A cache-miss `CancelJob(child)` in the registration window (rows committed, children not yet cached) hit the "non-terminal but uncached" fatal arm and shut the service down. | S5 review, issue 3 (major) | The cache-miss path reads `get_state_and_parent` and rejects a non-root **before** the fatal arm. A test pauses at `CommitterAfterCommit` of a registration and asserts `BadRequest` with no fatal shutdown. |
| `StartJob`, `GetJobState`, `GetJobOutputs` and `GetJobError` answered for children. | S5 review, issue 4 | All are root-only (`INVALID_ARGUMENT`) on a cache hit and on a miss. Unknown ids keep today's answers, and `StartJob` on an evicted child stays `JobNotFound`. |
| Every external RPC now trusts `parent_id`, but a recovered job was always a root at depth 0. | S4 review, issue 9 | `recover` takes `parent_id` from the row and the depth from the rebuilt structure. A mutation reverting this is caught by the post-COMMIT recovery test. |
| Prepared children of a deactivated job stay alive through the pool's `Arc` of a succeeded TCB until the pool's next GC. | S5 review, issue 6 | `apply_tree_state` drops every TCB's prepared children on deactivation. Covered by a unit test. |
| `root_id` has no index, so a query by tree scans `jobs`. | T2 run time (3,424 s instead of about 1,300 s as rows accumulated) | No production query reads by `root_id` today. Add the index before one does. |

### 3.5 Fence, schema upgrade and exit

| Trap | Where it bit | How the prototype handled it |
|---|---|---|
| MariaDB reports `trx_id = 0` for every transaction that holds only shared locks. The fence watcher's `INNODB_LOCK_WAITS`/`INNODB_TRX` join therefore matched every read-only transaction on the server, and `KILL`ed them all. | S1 self-test (found by tests running in parallel) | The watcher now joins `PROCESSLIST` and keeps only blockers in `DATABASE()` whose transaction started no later than the bump's wait. It skips its own connection and uses `DISTINCT`. Residual risk: an older shared-lock transaction in the same database can still be killed. |
| The test DB user has no `PROCESS` grant (errno 1227/1142). | Plan R2 | The watcher warns once and stops. The bump then relies on the 5 s lock-wait timeout and 5 attempts. |
| The upgrade DDL waits for an orphan's metadata lock, and the pending exclusive MDL stalls every statement on `jobs`. | S1 review, issue 3 | The DDL runs on a detached connection with `lock_wait_timeout = 5`, retried without a cap and logged. The first start can still wait as long as the orphan lives (old `wait_timeout` 28,800 s); the design has the same ordering. The upgrade reads `information_schema` first and runs 0 DDL statements when the schema is up to date. |
| The background actors' fatal exits still exited 0, and `SHUTDOWN_DEADLINE` did not bound `runtime.stop()`. | S1 review, issues 1 (major) and 4 | The pool and the GC take the `ShutdownHandle` and call `fatal()`. `Runtime::stop` returns `Err` on an actor error or panic. One pinned deadline bounds both the server and `stop()`, and `main` calls `shutdown_background()`. |
| One fence row per database makes concurrently running test processes fence each other. | Plan R1 | `DatabaseConfig.fence_id` (`#[serde(skip)]`, 0 in production) with a `BIGINT` id. The style review would keep the design's single row and give tests their own databases. |
| A transaction left open across a test pause keeps a fence S lock and stalls the next `connect`. | Plan R7 | No hook lies between BEGIN and COMMIT; `Pause` races the token; tests call `release_all()` before `stop()`. |
| Regenerating the proto: the system `protoc` 3.12 needs `--experimental_allow_proto3_optional`, and a failed build script deletes `src/generated/`. | S1 | A `PROTOC` wrapper; restore the directory with `git show 278d06f:<path>`. Prost also drops `Eq, Hash` from `ReportTaskSuccessRequest`. |

### 3.6 Recovery

| Trap | Where it bit | How the prototype handled it |
|---|---|---|
| A recovered `CommitReady` job lost its cleanup TCB, so a cancel after a restart skipped the cleanup (#513). | S2 review, issue 1 (major) | `recover_from_commit_ready` builds the cleanup TCB. Covered by a restart-then-cancel test. |
| Committers need the GC, the publisher and the tracker before recovery runs, but `create_runtime` recovered first. | Plan U6 | `build_runtime` runs, in order: connect, the queues, the pool, the GC, `ServiceState`, `recover_jobs`. |
| A failed `create_runtime` leaked what it had already spawned. | Plan R21 | `PartialRuntime::abort` cancels, closes and joins everything. Covered by a leak test. |
| Ancestor walks assume parent ids decrease. A corrupt backward link loops, and a root with a parent hits an assert. | S6 | A pre-check sends such trees straight to R5. |
| Grouping by `root_id` trusts the column. A wrong `root_id` lets R2 commit an unearned success across groups. | S6 review, issue 3 | Not fixed (§6). |
| The check that a `BlockingOnChildren` row has outputs runs after the repair commits. R2 can turn that corruption into a durable `Succeeded` without outputs. | S6 review, issue 4 | Not fixed (§6). |
| `has_failed_child` runs once per `BlockingOnChildren` row, sequentially, with `persistent(false)`. | B7: about 0.4 ms per parent | Not fixed. Batch it, or fold it into the recovery query. |
| Recovery cannot know the depth of a job whose parent row is terminal. | T11 finding 3 | It falls back to depth 1. That is harmless: such jobs never register, which T11 checked on 629,825 rows. |
| Repair cannot see a `Ready` or `Running` job that already has terminal child rows. | T11 finding 4 | A known limit of `check_invariants` (U13). T11 skips and counts such trees. |

### 3.7 Tests, CI and tooling

| Trap | Where it bit | How the prototype handled it |
|---|---|---|
| Recovery, repair and resend act on every row of the database they connect to. | Plan review M2; S2, where 1,200 leftover rows broke `runtime_recovery_test` | A `ScratchDatabase` for each test that creates, recovers or restarts a tree. Its `Drop` kills the database's connections before `DROP DATABASE`; otherwise a panicked test deadlocks. |
| A mock that allocates random job ids breaks I1, which requires a parent's id to be smaller than its child's. | S4 | The mock uses one ascending id counter. |
| `TestEnv`'s execution manager never heartbeats. At the default 60 s cutoff, the pool force-removes its instances at about 90 s. | Test phase | The new test files use a 24 h cutoff. `TestEnv::start` should default to it; three files carry a copy of `start_env`. |
| An `EventSubmitted` observer armed before the first RPC can count the first event. A "same round" case then silently runs as two rounds. | Test phase | The test groups arm the observer after the pause and assert the round composition. |
| Release builds skipped `assert_invariants`; `--all-features` turned on `unboxed-round`. | Plan R8; style review, issue 4 | `assert_invariants` is gated on `any(test, debug_assertions, feature = "test-hooks")`. Unboxing is a `cfg(spider_unboxed_round)` flag that only the benchmark sets. |
| `required-features = ["test-hooks"]` now gates the whole `test_spider_storage` binary, including the pre-existing tests. | Style fix pass | CI's `--all-features` still runs them. A plain `cargo test` now skips them, and says so. |
| `start.py` pulls `mariadb:latest` (13.0.2). Its first start can exceed 120 s, and the timestamp tests flake against it. | T8-T10 and T11 findings | The flakes also fail on `main` (2/30, 2/25). Pin the image. |

### 3.8 Benchmarks and performance

| Trap | Where it bit | How the prototype handled it |
|---|---|---|
| `Com_commit` counts only COMMIT statements. `main`'s autocommit `RegisterJob` INSERT is a durable transaction that it misses. | Methodology review, R2 | The reruns also record fsyncs and disk flushes: both binaries make 3 durable transactions per job. |
| The first two large registrations of a fresh process and database are 5-8x slower than steady state. | B4 diagnostic | Cause unknown. Benchmarks of registration latency need a warm-up. |
| The host has disk-flush slow modes and other slow phases that the idle gate cannot see. | B1, B2 | Counterbalanced process order, per-interval disk and CPU accounting, and ratios paired within a process. |
| A tree workload run against `main` would silently "succeed" as a plain job, because `main` ignores the unknown `children` field. | Plan R18 | The harness refuses tree workloads on `main`, and every tree trial checks its child-row counts. |

## 4. Testing results

### 4.1 T1-T11

Every T1-T10 test is deterministic and hook-driven: there are no sleeps, and timeouts exist only so that a broken implementation fails instead of hanging. Each test runs on its own scratch database with 4 runtime workers. Each iteration checks:
- each RPC's exact reply variant;
- every job's state in the cache and in its row, with `ended_at`;
- the whole tree's error messages;
- the exact lane contents.

Each iteration ends quiescent, with `is_fatal() == false`. All tests ran at 1,000 iterations in release (the default is 50) and at 100 iterations in debug.

| Test | Asserts | Iterations (release) | Result |
|---|---|---|---|
| **T1** two tasks fail one job | Plain root, or leaf J of `R → A → {J, S}`. The second failure lands in the same round, queued behind the first, in the commit-to-apply window, or after the reply. The first reply is `Failed`. The second is exactly `JobAlreadyTerminated(Failed)` or `JobNoLongerRunning`. Exact round composition; 1 DB transaction and 1 apply per job. Messages are exact, and the second message is stored nowhere. S gets one cleanup entry, then becomes `Cancelled`. | 8,000 trees, 213 s | Pass |
| **T2** root cancel vs registration (+ a two-registrations variant) | 10 interleavings × the registrar (leaf or root) × the cleanup modes. Cancel first: the registration is stale, with 0 child rows and 0 child lane entries. Registration first: `BlockingOnChildren`, 3 children inserted and published once, each deactivated to its own target. A row snapshot at the cancel's COMMIT equals the exact post-race states. Exact rounds, transactions, applies and submissions. | 50,000 trees (3,424 s); 9,000 (301 s) | Pass |
| **T3** root cancel vs leaf failure | `R → {A → {L, S}, Q}`, 8 interleavings. Cancel wins: `Ok(target)`, the failure is stale, all 5 jobs are aborted in one DB transaction, and no message is stored. Failure wins: `Ok(Failed)`, the cancel gets `JobAlreadyTerminated(Failed)`, and the chain messages are exact. Always 1 transaction and 1 apply per job. | 24,000 (1,682 s) | Pass |
| **T4** siblings complete together (jobs, and tasks within a job) | k = 2, 8 or 64 children finish in one round, in k rounds, or freely. P is applied exactly once. When P advances, every child is `Succeeded` with its own outputs in the row and the cache. The single apply, the per-round I2 check and `checked_sub` pin "the counter reaches 0 once". Within a job (§10.6 e): n tasks are paused between the TCB flip and `fetch_sub`, then released together; exactly one `RegularTasksDone` follows. | 25,200 trees (1,312 s); 4,000 jobs (96 s) | Pass |
| **T5** straggler fails after a success | 3 shapes × same or other instance × 4 windows × retry 0/1. The exact stale variant, answered in phase A without waiting for the paused success. No `TerminalFailure` event, no message, no re-publication, and the winner's outputs exactly. | 20,000 (505 s) | Pass |
| **T6** parent with a commit task | `R → {P (commit, 3 children), Q}`, with Q completing at 5 points. P stays `BlockingOnChildren`, then becomes `CommitReady` with one commit entry. Its commit sees exactly P's outputs. Then R climbs, through its own commit if it has one. | 10,000 (569 s) | Pass |
| **T7** cancel vs `CommitReady` (non-root, root) | Non-root, 10 interleavings: P stays `CommitReady` (`ended_at` NULL) with no cleanup entry, and its commit succeeds or fails without propagating. Root, 8 interleavings: the cancel gives `CleanupReady`, or `Cancelled` through the new edge; a late commit gets `JobNoLongerCommitReady`; commit first gives `JobAlreadyTerminated(Succeeded)`. | 24,000 (1,467 s); 16,000 (442 s) | Pass |
| **T8** restart at crash points (4 tests) | Crashes cover a registration before or after COMMIT at every committer point, the middle of a cascade, after a cancel's COMMIT and after each apply, plus a zombie runtime. Restarts are direct, or follow a crashed recovery. The caller sees `Stopping` or `TreePoisoned`, and the service shuts down fatally. The rows at the crash are exact. Recovery loads exactly the non-terminal jobs and repairs nothing. The first lanes are exact, with nothing for a blocked parent. The EM's retry is stale. Exactly 3 children result, and the blocked parent unblocks. The zombie gets `Fenced`. | 43,000 crashes and restarts (7,955 s) | Pass |
| **T9** dropped RPC future | 18 drop points: the TCB flip before `fetch_sub`, before submit, at the successor lock, at each committer point, for failure, commit, cleanup, cancel and start, and between instance registration and the pool send (EM alive or dead). The outcome equals that of a completed RPC. | 35,000 (2,533 s) | Pass |
| **T10** limit violation (3 tests) | Over count, over payload, over 16 MiB per report, an undecodable child, outputs over 4 MiB, and an insert rejected by the DB; with retry 0/1, with a bystander. Each attempt but the last gives `Ok(Running)`, the last `Ok(Failed)`. No child row, exact and propagated messages, not fatal. Two reports racing over the per-job limit: exactly one is rejected, and its reservation is released. A depth-8 report fails the 9-job chain. | 2,400 + 6,000 + 100 | Pass |
| **T11** planner property tests (3 tests) | Every event kind on random trees of up to 122 rows, with bursts and payload pressure. After every round: an exact outcome oracle from an independent table; exact failure messages; the registration's child set, depth and freeze; no guard miss, no row changed twice, valid edges; I1-I5 on every row; `check_invariants` agrees with a perturbed model. Every tree in Once mode reaches quiescence. Repair: `check_invariants(plan_repair(rows)) == Ok`, a second repair is empty, and the repaired tree runs on to quiescence. | 800,663 rounds over 67,240 trees; 400,000 repaired trees | Pass |

**Slice-level tests in the suite:**
- fence: every writer fenced, a queued bump, and the watcher killing only the orphan;
- schema upgrade: DDL counts, and MDL waits;
- exit codes: a fenced binary, and the deadline;
- `apply_tree_transition` on MariaDB: guard-miss rollback, three chunks, byte splits, 1,200 child inserts in order, and a real 1153;
- committer lifecycle;
- registration;
- the S5 race tests;
- recovery from forced R1, R2 (with and without a commit task), R4 and R5 states;
- stale starts.

**Full suite.** 283/284 in each of the last three runs, with the property tests included. The one failure is a different `mariadb_test` timing flake each run: a 2 s sleep against a 1 s threshold on a second-precision `TIMESTAMP`. The flaky tests are `test_delete_expired_terminated_jobs`, `test_update_execution_manager_heartbeat_already_dead` and `test_get_dead_execution_managers_atomic`. The first two also fail on unmodified `main` (2/30, 2/25). Clippy (`--all-features` and default features) and nightly rustfmt are clean, with no `#[allow]` added.

**Mutation checks.** The test phase ran 47 mutations: 7 in T1/T4-T6, 14 in T2/T3/T7, 10 in T8-T10, and 8 in T11, each against two test versions and run twice. All were caught except one: a committer that panics on a dropped reply receiver. That one cannot be reached from T9, because the spawned bodies mean a dropped RPC never drops the waiter. The original S3 property test caught 1 of the 8 T11 mutations; the extended oracle caught all 8.

**Bugs found.**
- **Test phase.** No implementation defect. It found test-infrastructure issues (the EM cutoff, the observer-arming window, a racy `is_fatal()`) and needed one clippy fix.
- **Slice reviews.** 11 serious issues, all fixed:
  - S1: the actor fatal exits exited 0.
  - S2: a recovered `CommitReady` job lost its cleanup task.
  - S3: a registration could be starved by the payload budget.
  - S4: three crash-loop limit gaps.
  - S5: the slice was missing; a cancel through a child id cancelled the whole tree; a cache-miss cancel in the registration window was fatal; the other RPCs were not root-only; and no race test ran on the real code.
- **Implementer self-tests.**
  - The fence watcher killed unrelated readers.
  - A deferred registration held back a cancel.
  - A planner-side output limit panicked T11 and regressed `jcb_test`.
  - Leftover test rows broke `runtime_recovery_test`.

**Not covered:**
- **A crash with the transaction open before COMMIT, and a lost COMMIT acknowledgement** (§11.1, gate H17). No hook lies between BEGIN and COMMIT, so the "before COMMIT" tests never run the INSERT. An autocommitted child INSERT would therefore pass every test. A fault-injecting connector is needed.
- **The `AcqRel` ordering**, which is covered by reasoning only.
- **Several gates and error paths outside the brief's T list:**
  - H14's partition in the middle of a transaction;
  - H13 at 1,000 jobs with stalled lanes;
  - H15;
  - H16's 10,000-child and M = 100 registration cases;
  - a real 1406 error;
  - a real 1153 through the service.

## 5. Benchmark results

Every benchmark compares release binaries of `main` (`278d06f`) and the prototype (`d4cda1b5a3428faa`). Each group gets a fresh `mariadb:10.11.16` container, and each storage process a fresh database. `RUST_LOG=warn`, the pool is 64, and the EM harness talks to storage over gRPC.

Statistics: values are `median [95 % CI]` over at least 10 trials, using distribution-free intervals. Ratios are paired by trial. A difference counts only when its CI lies outside the A/A band. The full tables are in `bench/results.md`.

| | Key numbers | Caveats |
|---|---|---|
| **B1** plain jobs, C = 1 and 64, two runs of n = 20 | **Last report p50, proto/main:** 1.030 [0.976, 1.121] and 1.024 [0.981, 1.064] at C = 1; 1.020 [0.965, 1.053] and 1.021 [0.997, 1.091] at C = 64. All are inside the A/A bands, so no regression was detected, but up to +12 % (C = 1) and +9 % (C = 64) is not excluded. **`StartJob`:** 1.01x at C = 1, 1.05x at C = 64. **`RegisterJob`:** 1.245 [1.206, 1.335] and 1.228 at C = 1; 1.383 [1.295, 1.412] and 1.366 at C = 64. That is outside the bands, +0.3 to +1.2 ms. **Lifecycle:** +3.6 to +4.7 % at C = 1 (at the edge of the band) and +10.7 to +11.1 % at C = 64 (outside it). **Throughput:** 0.89-0.94x jobs/s. Both binaries make 3 fsyncs per job; the prototype adds BEGIN, the fence `SELECT … LOCK IN SHARE MODE`, COMMIT, and the committer's creation. At C = 64, MariaDB spends 1.12x the CPU per job and the storage process 1.22x. | H12 holds on its point estimates. The throughput half of H11 is undecided. A disk-flush slow mode (2x for about 50 s every 400 s) hit 13 of 80 processes in the first run and 8 of 80 in the counterbalanced replication. |
| **B2** depth-7 tree (127 jobs) vs the same shape in one job | The tree takes 193.8 [187.5, 198.0] ms with no-op tasks and 245.0 ms with 10-ms tasks. bin127 takes 27.3 and 110.4 ms; flat64 takes 10.1 and 24.0 ms. **Tree / bin127:** 7.14x [6.75, 7.37] (replication 6.96x) with no-op tasks, 2.23x with 10-ms tasks. **Each job-nesting hop costs 27.7-28.3 ms**, against 1.8-2.0 ms per hop in a single-child chain. A tree takes 67-68 DB transactions, 63 of them registrations. Level-5 parents wait 44-68 ms for a registration that itself takes about 2-3 ms. Every single-job ratio is inside its A/A band. | The absolute tree time is 134-207 ms per run, and up to about 400 ms, depending on host phases (partly the disk flush latency, partly unidentified). The probe and B2's preceding workloads were tested and ruled out as causes. |
| **B3** abort of the expanded tree | **An idle abort is one DB transaction** (120 of 120 operations). Cancel acknowledged in 5.30 / 6.35 ms without cleanup tasks and 4.40 / 3.53 ms with them. Leaf failure acknowledged (root `Failed`) in 8.11 / 6.44 ms without cleanup tasks and 5.30 / 4.42 ms with them. Quiescence with cleanup tasks: 16-22 ms. **Racing the expansion** (about 30 registrations queued): the cancel is acknowledged in 5.07 [4.41, 7.37] ms after exactly 2 COMMITs; the 30 queued registrations are answered stale and insert nothing. A leaf failure behaves the same: 5.62 ms after 1-2 COMMITs. | The cleanup quiescence includes the harness's own dispatch. The claim that terminal targets cost more is shown only for failures, and its mechanism is a hypothesis. |
| **B4** 1,000 siblings at once | 64 reporters: 10,815/s, 31.2 events per DB transaction. 1,000 reporters: **21,745/s, 200 events per DB transaction, 1.96x [1.63, 2.11] the rate of 1,000 independent plain jobs on main**. Plain jobs, proto vs main: 1.013 and 0.962. | The plain-job A/A bands are about ±25-35 %, so B4 can only detect large differences. The first two registrations of a fresh process are 5-8x slower than steady state (about 24 ms per 1,000 children), and the cause is open. |
| **B5** K concurrent trees | Per-tree time is 142 ms at K = 1, 352 ms at K = 16 (2.43x) and 793 ms at K = 64 (5.60x). The slowest tree is within 1.02x of the median. Active connections stay ≤ K. A pool of 32 performs like 64, and 64, 128 or 256 workers perform alike. About 5,000 COMMITs/s. | The bound is not identified: CPU 9.7 of 16 cores, the disk 46 % busy, round time 2.1 → 12.6 ms. No noisy-neighbour case was measured. |
| **B6** memory per live job (100,000 jobs) | main: 1,490 B/job (190 MiB). **Proto: 15,490 B/job (1.49 GiB), +14.0 KB.** Unboxed: +735 B over boxed. **10.4 KB of the delta is the arrival buffer**; with `Vec::new()` the delta is +3.6 KB. The committer future is 768 B boxed and 1,592 B unboxed. | These are RSS deltas, so upper bounds. The latency of the `Vec::new()` variant was not measured. |
| **B7** restart recovery | Plain jobs, proto/main: 1.41x [1.30, 1.49] at N = 10,000 and 1.50x [1.38, 1.68] at N = 100,000 (13.2 vs 8.9 µs per job). With 9,090 trees of 1 + 10 jobs: 6.16 s. **Each `BlockingOnChildren` parent adds about 0.4 ms**, which is the sequential `has_failed_child` probe (3.5 of 6.0 s). | The probe is identified by elimination, from unpaired containers. RSS after recovery: 917 vs 212 MiB. |

**Design gates (§15):**
- H12 holds on its point estimates.
- H11: the latency half holds the same way; the throughput half is undecided.
- H13: a cancel waits for at most the round in flight. Measured only with 30 queued registrations, not at 1,000 jobs with stalled lanes.
- A3 group commit passes.
- H15, H16, H17 and H18 were not measured as specified.

## 6. Deviations from the design doc

| Section | Design says or assumes | What the prototype found or chose | Update |
|---|---|---|---|
| §5.4, §13, §7.2 | A live tree costs a few hundred bytes. | It costs 3.6 KB per live tree, or 14.0 KB with the pre-sized buffer. Boxing the round future saves 735 B. | Replace the estimate with the measurement. Specify an unsized arrival buffer and `Box::pin` for the round. |
| §5.4, §14.1 | A plain job's transition costs the same statement count as today. | That holds for transitions. `RegisterJob` went from one autocommit INSERT to BEGIN, fence, INSERT and COMMIT: +23 to +38 % latency. | Add `RegisterJob` to the cost table and to H12. |
| §7.5 rule 3, §14 | One registering parent per round bounds head-of-line delay. | It also serializes sibling registrations: a wide tree takes 7.1x the same shape in one job, 28 ms per hop (B2). | Allow several small registrations per round within the caps, and keep the rule for large ones. |
| §7.5 rules 3 and 5, §13 | The first event of a round is always admitted. | Applied in planning order, this starves the registration. | Reserve the oldest registration's payload, apply per-job deferral in arrival order, and state the liveness property. |
| §7.5 rule 5, §13 | The 256-event cap is enforced by both `take_events` and the planner. | The planner's cap is dead code, and would starve a registration if reached. | One owner for each rule. |
| §7.5 rule 3 | Extra registering parents are deferred. | They are deferred even when they are certainly stale, so N registrations take N empty rounds after an abort. | Answer a registration immediately when its job is not `Running`. |
| §7.5 rule 4 | "Never touch a row twice" is a deferral rule. | Unreachable by real producers. | Describe it as an assertion or as a safety net. |
| §6.2 step 5, §13 | The limits make errno 1153/1406 unreachable, and the 4 MiB output limit is unchanged. | False in three ways: the failure message, the union of a job's outputs, and the decode limit removing tonic's 4 MiB default. | Add `MAX_ERROR_MESSAGE_CHARS`, a per-job bound on serialized outputs that becomes a job failure, and an explicit 4 MiB output check. |
| §6.2 steps 1 and 3 | Per-job child totals are checked in the planner, and the compressed buffers are copied. | The check must come before decoding. The prototype reserves in the service and makes no copies. | Move the check into the service. |
| §6.2, §9.6 | A rejected report becomes a task failure. | The RPC answers with the job state, so the EM cannot tell. | Specify a distinct status. |
| §6 stale table, §10.9 | `Lane::stale_error(Absent)` | The planner cannot answer a cancel of an absent root. The prototype answers `Untracked`, and the JCB maps it to its cached state. The stale variants differ by path. | Document the mapping and the variants. |
| §7.3 step 1, N4 | Cancel and failure rounds collect every job's state. | A registering `RegularTasksDone` must too. | Add it. |
| §6.4 | `truncate(msg)` | The length is unspecified. The prototype uses 1,024 characters for ancestors and 64 Ki for the failing job. | Specify both. |
| §6.5, §9.6 | The cache-miss `cancel_job` reads `parent_id`. | The non-root check must come before the fatal arm, and the cache-hit path needs it too. | Spell out the order. |
| §9.5 | Deactivation runs `cancel_non_terminal` from `Running`, `BlockingOnChildren` and `CommitReady`. | It is also needed from `Ready`, and deactivation must drop the prepared children that TCBs hold. | Add both. |
| §10.10 | The report, start and cancel bodies are spawned. | `RegisterJob` must be spawned too. | Add it. |
| §7.7 | Poison on abnormal exit | Callers are answered before the fatal flag is set. | Poison first. |
| §9.1, §9.2 | One fence row, `TINYINT`. | Concurrent tests fence each other. The prototype uses a `BIGINT` id plus a test-only `fence_id`. | Decide between a test-only id and scratch databases everywhere. |
| §11.3 | Kill the blockers found through `INNODB_LOCK_WAITS`/`INNODB_TRX`. | `trx_id = 0` makes that join match every reader. The behaviour without the `PROCESS` grant is unspecified. | Specify the `PROCESSLIST`/database/start-time filter and the fallback. |
| §9.1, §11.3 | The upgrade runs at connect. | An orphan's MDL stalls it and every statement on `jobs`. The prototype uses a 5 s timeout with uncapped retries, and runs no DDL when the schema is up to date. | Document this, and consider bumping the fence before the DDL. |
| §9.3, §16.1 | The old writers remain as wrappers until PR 9. | Exact guards with the fence first required one transaction per wrapper. The style pass then deleted the writers and their 18 tests. | Decide whether PR 3 keeps them. |
| §11.2 | Group by `root_id`, rebuild, repair R1-R5. | (a) A wrong `root_id` allows an unearned success across groups. (b) The outputs check runs after the repair commits. (c) The probe costs 0.4 ms per parent. (d) Parent links must decrease, so the prototype runs R5 first. (e) The depth fallback is 1. (f) Terminal jobs are not cached. | Validate before repairing, batch the probe, and specify the R5 pre-check and the depth fallback. |
| §11.1, §15, H17 | Test a crash with the transaction open and a lost COMMIT acknowledgement. | Hooks cannot reach either. | Require a fault-injecting connector in PR 7. |
| §8.2, U13 | `check_invariants` checks I2. | It cannot see terminal children. | State the limit. |
| §16.2 | The `job_tree.enabled` flag | Omitted in the prototype. | Still needed. |
| §5.1 | `TreeState` holds JCB references. | A non-generic `TreeStructure` plus a committer-owned handle map keeps the planner pure. | Adopt. |

## 7. Open issues and recommended next steps

**Open issues:**
1. **Arrival buffer.** The pre-sized buffer costs 10.4 KB per live tree. Fix it, then measure the latency of the fixed version.
2. **Registration serialization.** Group small registrations, then rerun B2 and H16.
3. **Recovery probe.** Batch it, and time it directly.
4. **Planner rules.** The round-shaping rules live in two places, and the planner's event cap is dead.
5. **Extra registering events.** Answer them early when their job is not `Running`.
6. **Rejected reports.** A rejected success report needs a distinct RPC status.
7. **Recovery corner cases.** A wrong `root_id` across groups, and the outputs check before repair.
8. **Poison ordering.**
9. **Token race.** The unbiased race in `submit_and_wait` and `spawn_body`.
10. **The unreviewed style fix pass** (about 04:43-05:08). It touched most production files after every slice review. The tests and the benchmarks ran on the result and passed. Still open from the style review:
    - `fence_id`;
    - the touch-twice decision;
    - duplicated test helpers;
    - the 60 s EM cutoff in `TestEnv`.
11. **Test gaps:**
    - a fault-injecting connector;
    - a `loom` model of the `fetch_sub` ordering;
    - a noisy-neighbour case for B5;
    - H13 at 1,000 jobs, H14 and H16 at full size;
    - more statistical power for H11/H12.
12. **Out of scope:**
    - TDL `create_child_job`;
    - EM and executor transport;
    - the scheduler and `BlockingOnChildren`;
    - Helm and Compose;
    - retention by `root_id`, which needs an index;
    - WaitJob;
    - the lazy child-output fetch.

**Recommended next steps:**
1. Fold §6 into the design doc with the measured figures. Decide the round-shaping changes before PR 4/5.
2. Start with PR 0 and PR 1, taken from the prototype:
   - the exit code and the `ShutdownHandle`;
   - the `CommitReady → Cancelled` edge;
   - the recovered cleanup TCB;
   - `cancel_non_terminal` on failure;
   - the resend snapshot;
   - the hardened fence and upgrade.

   Judge H12 with `RegisterJob`'s cost in hand.
3. Carry over the prototype's test infrastructure:
   - per-runtime hooks with a zero-sized disarmed form;
   - `ScratchDatabase`;
   - sentinel lane checks;
   - the T11 oracle.
4. Rebuild PR 3 with the buffer fix and boxing. Rerun B1 and B6 with the counterbalanced harness, and record the disk flush latency.
5. Before PR 7, add the fault-injecting connector and the batched probe, and fix the two §11.2 gaps.

**Host note (the relayed request to kill the kind cluster):** the kind cluster `clp-rc2` has a single node container, `clp-rc2-control-plane`. It was already killed: `Exited (137)`, about 16 hours ago. No container of the cluster is running. It was not started and not deleted; deleting it (`kind delete cluster --name clp-rc2`) cannot be undone and was not asked for.

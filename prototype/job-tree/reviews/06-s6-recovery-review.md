# Review: slice S6 `s6-recovery` (slice reviewer)

Verdict: **ok**. I found no correctness or concurrency defect in the recovery and repair path. The issues below are minor: one docstring regression, two corruption edge cases that the design shares, and four gaps in what the T8 and property tests actually prove. None blocks the benchmarks or the report. Fixing issues 1, 2 and 6 is cheap and worth doing before the report is written.

Reviewed: `git diff 278d06f` in `/home/lzh/dev/spider-job-tree-proto` at 2026-10-09 04:19 EDT, against the slice note `slices/06-s6-recovery.md`, design §11.1-§11.4, and the S2-S5 code that recovery feeds: the committer, the planner, `apply_tree_transition` and the JCB constructors.

Host: the relayed user request asked for the kind cluster to be killed. `clp-rc2-control-plane` was already `Exited (137)` from about 6 hours earlier, and none of its containers is running, so there was nothing to kill. It was neither started nor deleted. This review started no MariaDB container and changed no code.

## What was checked and holds

- **Locks and awaits.** `recover_jobs`, `repair_tree` and `create_recovered_tree` hold no JCB or TCB lock across an await. `insert_many` takes the map write lock once and does not await under it. No lane send happens during recovery: publications from the repair plan are dropped, and `resend_ready_tasks` sends from a snapshot after serving starts.
- **One DB transaction per tree.** Each repair is one `apply_tree_transition` call: the fence is checked, every update is guarded by its exact loaded `from` state, and any error is fatal. Updates that don't match fail as `CorruptedDbState`, so they cannot be skipped silently.
- **Exact from-state guards.**
  - Steps R1, R4 and R2 build their guards from the overlay. Since each job is changed at most once, the overlay state equals the loaded state.
  - When R5 replaces the plan, `plan_tree_failure` builds its guards from the loaded `states`, not from the overlay.
  - The property test asserts every guard against the loaded state.
- **The planner cannot panic on corrupt rows** (`cache/tree.rs:870-964`). By induction over ascending IDs, after R1 every remaining `Ready`/`Running`/`BlockingOnChildren` non-root has a chain of `BlockingOnChildren` ancestors up to a loaded root. That makes the `blocking_ancestors` assert and the `checked_sub` in `climb_after_success` safe. The first R4 hit aborts every abortable job, so no second R4 or R2 can touch a job twice. The pre-check sends a backward parent link (or a root with a parent) to R5 before any ancestor walk.
- **Repaired parents earn their success.** R1 never aborts a child of a `BlockingOnChildren` parent. Any R4 hit leaves no blocked job. So the parents that R2 unblocks have only `Succeeded` terminal children, as long as `root_id` is consistent (see issue 3).
- **The recovered structure matches the DB.** `create_recovered_tree` rebuilds it from the repaired contexts, so its counters equal the number of non-terminal children in the rows. The JCB depth is the structure's depth, so the `MAX_DEPTH` check (`state/service.rs:1244-1258`) uses the true depth for every job that can still register children. The fallback depth of 1 only applies to orphan-free `CommitReady`/`CleanupReady` jobs.
- **Ordering.** Repair finishes before the tree's committer is spawned, and every committer exists before `build_runtime` returns, so before serving starts. Committers that were spawned before a later tree fails to recover exit through the token in `PartialRuntime::abort`. They do not poison the service.
- **Absent roots.**
  - A cancel addressed to a terminal root takes the DB fallback path.
  - Commits of `CommitReady` children stop the climb at the `Absent` parent.
  - `check_closed_tree` is satisfied, because R1 has aborted every active job under that root.

## Issues

### 1 (minor, style): `validate_session` lost its docstring

- `state/service.rs:1164`: `fn validate_session` has no docstring. At `278d06f` it had "Validates that the given `session_id` matches the session ID captured at service creation time." plus an `# Errors` section naming `StaleSession`. The repo grep for that text now returns nothing.
- The slice note says the S5 leftover was "fixed", where `validate_session`'s docstring had been attached to `ensure_uncached_root`. The fix removed that docstring from `ensure_uncached_root` but never restored it on `validate_session`. The style guide requires a docstring on every function.
- Fix: restore the base docstring above `validate_session`.

### 2 (minor, tests): the "before a registration's COMMIT" crash never opens a DB transaction

- `tests/job_tree_recovery_test.rs:404-435` crashes at `CommitterBeforeDbTransaction`. That hook runs before `write_plan` (`state/tree_committer.rs:603-616`), so there is no `BEGIN`, no child `INSERT` and no rollback. The assert `children_of(root) == []` (`:419`) is true by construction.
- Design §11.1 lists a separate crash point: "DB transaction open, before COMMIT (InnoDB rolls back)". In that case the multi-row child `INSERT … RETURNING` has already run on the connection when the process dies. No test covers it, here or in S4: `a_rejected_registration_payload_fails_the_parent` uses `FailDb`, which also replaces the whole call.
- Concrete gap: if a future change commits the child insert in its own autocommit statement, every current test still passes. That change would register the children twice after a restart, which is exactly what T8 is meant to rule out.
- Fix: add a hook inside `apply_tree_transition`, after `insert_children` and before `tx.commit()`, that returns an error or panics. Then assert after the restart that no child rows exist and that the rerun registers exactly N children.

### 3 (minor, design gap shared with §11.2): grouping by `root_id` trusts the column, and R2 can commit a success that was not earned across groups

- `state/service.rs:249` groups by `root_id.unwrap_or(id)`. The planner only sees one group at a time.
- Scenario, with a bad `root_id` from a bug or a manual edit:
  - rows: R=1 `BlockingOnChildren` (root); P=2 `BlockingOnChildren` (parent 1, root 1); C=3 `Running` (parent 2, `root_id` NULL).
  - Tree 1 loads {1, 2}. P has 0 counted children, and `has_failed_child(P)` is false because C is `Running`. R2 moves P to `Succeeded`, and the climb moves R to `Succeeded` (`cache/tree.rs:938-946`).
  - Tree 3 loads {3}. C is its own group's "root" but has a parent, so the pre-check (`cache/tree.rs:877-888`) sends it to R5, and C becomes `Failed`.
  - Result: P and R are `Succeeded`, but P's child is `Failed`. Every row is terminal, so no later start notices. This is the "success is earned" violation that R4 exists to prevent.
- Design §11.4 says the predicates are a safety net "for bugs and manual edits", so a `root_id` that disagrees with the parent chain belongs in scope.
- Cheap guard: when a node's `parent_id` is not among its group's rows, check that the parent is not a non-terminal row of another group; if it is, fail both trees through R5. Alternatively, run the `has_failed_child` probe again on the rows after repair. At minimum, list this in the report as a gap of the design.

### 4 (minor, design gap shared with §11.2): the check that a `BlockingOnChildren` row has outputs runs after its repair is committed

- `repair_tree` (`state/service.rs:1113-1162`) commits R2 before `SharedJobControlBlock::recover` checks for outputs (`cache/job.rs:218-224`).
- Scenario: a corrupt `BlockingOnChildren` row with NULL `serialized_job_outputs` and no active child, so R2 applies.
  - Without a commit task, R2 writes `BlockingOnChildren → Succeeded`. Startup then succeeds, and the job is `Succeeded` with no outputs. The row corruption that §11.2 says "still fails startup" is turned into a durable success. Its consumers then fail on output reads, not at startup.
  - With a commit task, R2 writes `CommitReady`. `recover` then fails on "commit-ready job doesn't have outputs ready" on every start, which matches the design.
- Fix: validate the row invariants that `recover` enforces (outputs for `BlockingOnChildren`/`CommitReady`, a commit or cleanup descriptor where needed) on the loaded rows before `plan_repair`. Or leave it as is and record it in the report as a deviation.

### 5 (minor, tests): the property test does not assert the R2 premise

- `cache/tree.rs:4043-4055` asserts that no repaired `BlockingOnChildren` job has a `Failed`/`Cancelled`/`CleanupReady` child. It does not assert the property R2 relies on: every job that the plan moves from `BlockingOnChildren` to `CommitReady`/`Succeeded` has only `Succeeded` children among all rows.
- Today that property holds only because of the ordering argument above. If R2 ran before R4, or if R1 also aborted children of blocked parents, the test would still pass while R2 committed successes that were not earned.
- Fix: in `apply_repair`/`repair_tree`, for each update `BlockingOnChildren → {CommitReady, Succeeded}`, assert that every row whose parent is that job is `Succeeded`, or `CommitReady` for the `CommitReady` case's non-terminal children.

### 6 (minor, tests): the zombie test does not check why the old runtime failed

- `tests/job_tree_recovery_test.rs:651-662` asserts only `report.is_err()` and `old_runtime.is_fatal()`. Any fatal error passes, for example a pool connection killed by the successor's fence bump, or a panic. The test's claim, that the superseded runtime is fenced off, is not checked.
- `:664` is `let _ = old_runtime.stop().await;`, a bare `let _ =` on a fallible call, which the review guide disallows.
- Fix: assert that the report's error is the fence error (or that the old runtime's log or shutdown cause names `Fenced`). Bind and check the `stop()` result, or `expect_err` it with a reason.

### 7 (minor, tests): "nothing is registered twice" after the post-COMMIT crash is shown only because the harness never reports again

- `a_crash_after_a_registration_commits_recovers_the_blocked_parent` (`tests/job_tree_recovery_test.rs:439-495`) checks that there are still 3 child rows at the end. But nothing after the restart tries to register again.
- In production, a duplicate registration could come from two paths:
  - the EM's re-report of the parent's success on the old session;
  - a new instance of the parent's task.
- Neither is exercised.
- Fix: after the restart, report success for `TaskId::Index(0)` again with children on the new service. Assert that creating the instance is rejected (a `BlockingOnChildren` JCB has no regular tasks), or that the report with the old session ID gets `StaleSession`. Then assert the 3 child rows.

## Nits

- `cache/job.rs:166`: the `recover` docstring says "if `state` is not recoverable", but there is no `state` parameter any more. Say "if the context's state is not recoverable".
- `state/service.rs:282-283`: `#[cfg(not(feature = "test-hooks"))] drop(repaired_job_ids);` exists only to silence an unused binding. Binding the value as `_repaired_job_ids` in the non-hook build, or returning nothing from `repair_tree` when hooks are off, reads better.
- `recover_jobs` probes `has_failed_child` sequentially, once per `BlockingOnChildren` row, including rows that R1 will abort anyway. The slice note already flags this for B7, and it is fine for the prototype.

## Not issues (checked on purpose)

- `CleanupReady → Failed`, `CommitReady → Failed` and `Ready → Cancelled` in R5: `apply_tree_transition` does not check the transition table, and the property test asserts `JobState::is_valid_transition` on every repair edge.
- An R2 unblock to `CommitReady`, or an R1 abort to `CleanupReady`, publishes nothing during recovery. `resend_ready_tasks` sends their commit or cleanup after serving starts, and the T8 lane assertions check this.
- A tree that repair leaves quiescent gets no committer. Its `TreeControlBlock` and receiver are dropped at once, and no JCB keeps a sender to it.
- The ID-order assumption (a parent's ID is smaller than its children's): `AUTO_INCREMENT` allocation is monotonic, the parent row exists before its children's `INSERT`, and the pre-check sends any violation to R5.

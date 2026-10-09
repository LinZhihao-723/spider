# Slice S6 `s6-recovery`: notes

Role: implementer. Worktree: `/home/lzh/dev/spider-job-tree-proto` (`git diff 278d06f`, on top of S1-S5). Nothing was committed.

## What changed (non-test)

- `db/protocol.rs`:
  - `RecoverableJobContext` gains `parent_id: Option<JobId>` and `root_id: Option<JobId>`.
  - The `get_recoverable_jobs` docstring now names `BlockingOnChildren` and the parent and root IDs.
  - New `InternalJobOrchestration::has_failed_child(parent_id) -> Result<bool, DbError>`.
- `db/mariadb.rs`:
  - The recoverable query also loads `BlockingOnChildren` and selects `parent_id` and `root_id`, and the projection carries both.
  - `has_failed_child` runs `SELECT id … WHERE parent_id = ? AND state IN (?, ?) LIMIT 1` with `persistent(false)`. The states are bound, not hard-coded, and the query uses the `job_parent` index.
- `cache/task.rs`: `TaskGraph::recover_blocked(outputs, commit: Option<&_>, cleanup: Option<&_>)` builds no regular tasks, keeps the outputs, and builds the commit and cleanup TCBs. `recover_from_commit_ready` now delegates to it, so the S2 #513 fix (it keeps its cleanup TCB) is unchanged.
- `cache/job.rs`: `SharedJobControlBlock::recover(ctx, depth, sender, pool, tree)`.
  - It takes the loaded `parent_id` from the context and `depth` from the rebuilt structure. This fixes the S4 trap 9, where every recovered job was a root at depth 0.
  - New `BlockingOnChildren` arm through `recover_blocked`. A blocking row without outputs is `InvalidRecoverableJobContext`, so startup fails, as for `CommitReady` today.
  - Docstrings updated.
- `cache/tree.rs` (pure):
  - `RecoveredRow { id, parent_id, state, resource_group_id, has_commit, has_cleanup }`, with `From<&RecoverableJobContext>` (the flags come from the decoded graph).
  - `TreeStructure::from_rows(root_id, rows)`:
    - takes rows in ascending ID order;
    - only non-terminal rows become nodes;
    - depth is the parent's depth plus 1 if the parent is among the rows, else 1;
    - the counter is the number of non-terminal children.
  - `pub(crate) fn plan_repair(structure, states, failed_child) -> Plan`:
    - Pre-check: if a parent link does not point to a smaller ID, or the root has a parent, go straight to R5. Every other step walks ancestor chains and would loop or panic on such links.
    - R1: in ascending ID order, abort every `Ready`/`Running`/`BlockingOnChildren` non-root whose parent is not `BlockingOnChildren` in the overlay. This reuses `deactivate`, so `abort_target` sends a `Ready` row to `Cancelled` (N8). Descendants are aborted in the same pass.
    - R4: every `BlockingOnChildren` job with a `CleanupReady` child (in the overlay) or in `failed_child` goes through `fail_and_abort(…, BlockingOnChildren, "tree repair: a child job failed or was cancelled")`. That is S3's `fail_blocked`: the chain fails, and the rest of the tree is aborted.
    - R2: every remaining `BlockingOnChildren` job with an overlay counter of 0 goes through `unblock`, and the success climbs.
    - R5: `check_invariants` runs on the repaired structure and states. On a violation, the whole plan is replaced by `plan_tree_failure`:
      - `Ready` rows go to `Cancelled`;
      - every other row goes to `Failed` with "tree invariant violation: {violation}" (N8).
    - The repair plan's outcomes are cleared; recovery uses only its updates.
- `state/tree_committer.rs`:
  - `create_recovered_tree(env, root_id, contexts)` recovers a whole tree:
    - it rebuilds the structure from the repaired rows;
    - it recovers one JCB per non-terminal row, all sharing one `TreeControlBlock` rooted at `root_id` (the root may be absent);
    - it inserts them with `insert_many`;
    - it spawns one committer with `CommitterState { structure, handles }` unless the tree is quiescent.
  - `spawn_committer` now takes a `CommitterState`.
- `state/service.rs`:
  - `recover_jobs` sorts by ID, groups by `root_id.unwrap_or(id)` (a `BTreeMap`, so trees are recovered in ascending root order), and then, per tree:
    1. hook `RecoveryBeforeRepair`: context `root_id` plus the loaded job IDs;
    2. `repair_tree`;
    3. hook `RecoveryAfterRepair`: context `root_id` plus the repaired job IDs, empty when nothing was repaired;
    4. `create_recovered_tree`;
    5. the unchanged "Job recovered into cache." log per non-terminal job.
  - New private `repair_tree`:
    - probes `has_failed_child` for each `BlockingOnChildren` row and calls `plan_repair`;
    - if the plan is not empty, logs at `error` "Repairing job tree at startup." with the job IDs, then calls `apply_tree_transition` (fenced; any error fails startup);
    - finally writes the repaired states into the contexts.
  - Also fixed: an S5 leftover where `validate_session`'s docstring had been glued onto `ensure_uncached_root`.
- Mocks: `MockDbConnector::has_failed_child` reads `parents` and `states`, and `NoopDbConnector` returns `false`.

About 420 non-test lines, docstrings included (plan: 350). `cache/tree.rs` accounts for about 230 of them, `state/service.rs` about 95.

## Tests

### Unit and property tests (`cache::tree`)

- Named tests:
  - `from_rows_rebuilds_depths_counters_and_flags`;
  - `a_consistent_tree_needs_no_repair`: a `Ready` root, a `CleanupReady` root, a mixed blocking tree, and `CommitReady`/`CleanupReady` children under an absent root;
  - `an_orphan_is_aborted_with_its_descendants`: an absent root, a parentless non-root, and a `Ready` child of a `Running` root;
  - `a_stuck_parent_is_unblocked_and_its_success_climbs`: the counter decremented, and a three-level climb to a `CommitReady` root with its commit publication;
  - `a_blocking_parent_with_a_failed_child_fails_and_aborts_the_tree`: the probe case with the exact chain messages, and the `CleanupReady`-child case;
  - `a_tree_that_repair_cannot_fix_fails_as_a_whole`: a non-root `Ready` (exact R5 message) and a parent link to a larger ID.
- `property::repaired_trees_keep_the_invariants` (the T11 extension):
  1. It generates a consistent tree: a random root state, depth ≤ 5, fan-out ≤ 4, children consistent with their parent's state, terminal rows included. It asserts that `plan_repair` gives an empty plan for it.
  2. It injects 0-3 corruptions: random states, reparenting (which can point to a larger ID), and cleanup flags.
  3. It loads the tree as recovery does (non-terminal rows only; `failed_child` from the terminal rows) and repairs it.
  4. It asserts:
     - no job is repaired twice, every guard equals the loaded state, and every edge is valid;
     - `check_invariants` is `Ok` on the repaired tree;
     - no repaired `BlockingOnChildren` job has a `Failed`/`Cancelled`/`CleanupReady` child among all rows;
     - a second `plan_repair` on the repaired rows is empty (a violation costs at most one restart).
  - The count is `JOB_TREE_PROPERTY_ROUNDS` trees (3,000 in debug, 20,000 in release), and `JOB_TREE_PROPERTY_SEED` replays one tree. Coverage counters (consistent trees, orphan aborts, unblocked parents, failed blocked parents, tree failures) are each asserted non-zero.
  - Ten debug runs of 20,000 trees each passed. A typical run: about 4,100 consistent trees, 4,500 plans with orphan aborts, 700 unblocks, 420 R4 plans and 520 R5 plans.

### MariaDB tests: `tests/job_tree_recovery_test.rs` (new, feature `test-hooks`, every test on its own `TestEnv`)

T8 crash points. Each crash is a `Crash` hook, then the failing RPC, then `is_fatal()`, then a restart that resends the ready tasks:

| Test | Crash point | Exact asserts after the restart |
|---|---|---|
| `a_crash_before_a_registration_commits_reruns_the_parent` | `CommitterBeforeDbTransaction`, registration round | no repair; the root `Running` with no child rows; its task republished (`[0]`); a rerun registers 3 children; the root stays `BlockingOnChildren` until the last child, then `Succeeded`; exactly the 3 child rows |
| `a_crash_after_a_registration_commits_recovers_the_blocked_parent` | `CommitterAfterCommit`, registration round | no repair; the root `BlockingOnChildren` (cache and row) and the 3 children `Running`; `GetJobState(child)` is `BadRequest` (the recovered child is not a root); the first full poll of all three lanes holds exactly the 3 children's ready entries and nothing for the parent; children succeed → root `CommitReady` → commit → `Succeeded`; still 3 child rows |
| `a_crash_in_the_middle_of_a_cascade_recovers_what_it_committed` | `CommitterAfterApplyJob` for the leaf (first apply of a three-job cascade) | no repair; the root `CommitReady`, parent and leaf `Succeeded`; the root's commit resent (no ready entries); commit → `Succeeded` |
| `a_crash_after_a_cancel_commits_recovers_only_cleanups_and_skipped_commits` | `CommitterAfterCommit`, `CancelRoot` | no repair; root, blocked parent and leaf `CleanupReady`, and the skipped non-root `CommitReady`; the lanes hold exactly the three cleanups and the one commit; everything finishes `Cancelled`/`Succeeded` |
| `a_crash_during_recovery_repairs_the_tree_once` | `RecoveryBeforeRepair` and `RecoveryAfterRepair`, on a forced R2 state | the crashed start fails (`start_failing`); the next start repairs `[root]` (before) or finds nothing to load (after: the crashed start had already committed the repair); the root `Succeeded`; a third start loads nothing |
| `a_superseded_runtime_is_fenced_off_and_its_successor_matches_the_database` | zombie | a successor starts while the old runtime lives; a report on the old service fails and the old runtime `is_fatal()`; the new runtime is not fatal; cache and rows agree (root `BlockingOnChildren`, both children `Running`); both children's tasks are republished and finish the tree |

Forced repair states (rows written by raw SQL on the stopped database, then a restart with `Observe@RecoveryAfterRepair`; the context's job IDs are asserted exactly, then a second restart):

| Test | Forced rows | Repaired IDs | Second restart |
|---|---|---|---|
| `startup_aborts_an_orphan` (R1) | root → `Failed`, child `Running` with cleanup | `[child]` → `CleanupReady` | context empty; cleanup polled and run → `Cancelled` |
| `startup_unblocks_a_stuck_parent_and_climbs` (R2, no commit) | leaf → `Succeeded` under root → parent | `[root, parent]` → both `Succeeded` | nothing loaded |
| `startup_unblocks_a_stuck_parent_with_a_commit_task_to_commit_ready` (R2, commit) | only child → `Succeeded` | `[root]` → `CommitReady` | context empty; commit polled and run → `Succeeded` |
| `startup_fails_a_blocked_parent_with_a_failed_child` (R4) | one child → `Failed`, sibling `Running` with cleanup | `[root, sibling]`: root `Failed` ("tree repair: …"), sibling `CleanupReady` | context empty; cleanup → `Cancelled` |
| `startup_fails_a_tree_that_repair_cannot_fix` (R5) | one child → `Ready` | `[root, ready, running]`: `Failed`, `Cancelled`, `Failed`, message "tree invariant violation: invariant I1 broken at job N: a non-root job is Ready" | nothing loaded |

Infra (`tests/job_tree_infra.rs`):
- `TestEnv::stop() -> StoppedEnv` (expects a clean stop; this also holds after the injected crashes), and `restart()` is now `stop().start(TestHooks::default())`.
- `StoppedEnv::{start(hooks), start_failing(&self, hooks), set_row_state}`. With these, a test arms hooks before recovery and writes rows while no runtime is running.
- `TestEnv::start_successor()` for the zombie case.
- `start_runtime` takes the hooks.

### Mutation checks

Each defect was applied by a script, the targeted tests were run, and the file was restored. The restored files were compared byte for byte with copies.

| Defect | Caught by |
|---|---|
| `BlockingOnChildren` rows not loaded | 6+ recovery tests (the blocked parent is missing and its children are repaired as orphans) |
| recovered JCB is a root at depth 0 (S4 trap 9) | `a_crash_after_a_registration_commits_recovers_the_blocked_parent` (`BadRequest` check) |
| `has_failed_child` always false | `startup_fails_a_blocked_parent_with_a_failed_child` |
| repair not written to the DB | 5+ recovery tests (the second restart repairs again) |
| no R5 replacement | the R5 unit test, the property test, and `startup_fails_a_tree_that_repair_cannot_fix` |
| no R1 | the orphan unit test and `startup_aborts_an_orphan`. The property test passed: R5 then fails those trees, which is consistent, only harsher |
| no R2 | the R2 unit test, the property test, and three MariaDB tests |
| R4 ignores `CleanupReady` children | only the R4 unit test; R5 catches such trees as an I2 violation otherwise |
| no committer for recovered trees | most recovery tests and `job_tree_test::cancel_after_a_restart_of_a_commit_ready_job_runs_its_cleanup` |

## Verification

- `cargo +nightly fmt --all -- --check`: clean.
- `cargo build --workspace` and `cargo build --workspace --all-targets --all-features`: OK.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` and `cargo clippy -p spider-storage --all-targets -- -D warnings`: clean.
- Debug build:
  - `cargo test -p spider-storage --features test-hooks --lib`: 159/159;
  - the 28 `job_tree_*` MariaDB tests: 28/28.
- `cargo nextest run -p spider-storage --all-features --run-ignored all --release --no-fail-fast` on container `jtp-impl-s6` (stopped at the end), three runs:
  - first run: 282/283. The failure was `mariadb_test::test_get_dead_execution_managers_multiple`, the shared-`spider-db` execution-manager liveness flake listed in the S1/S3/S4 notes. It passed when run alone.
  - second and third runs: **283/283**. These include all four `runtime_recovery_test` tests and both property tests.
- No `jt_*` scratch database was left behind.

## Deviations from the plan and the design

1. **`recover(ctx, depth, …)` instead of `recover(ctx, state_after_repair, depth, …)`.** `repair_tree` writes the repaired state into the context's `state`, so the context is the row after repair, and the extra parameter would duplicate it.
2. **R5 also runs before R1-R4** when a parent link points to an equal or larger ID or the root has a parent. Every ancestor walk in the planner (`blocking_ancestors`, the climb, the R1 order) relies on parent IDs decreasing, so a cycle would loop and a root with a parent would hit `blocking_ancestors`' assert. The design's R5 only runs after R1-R4.
3. **Depth of a node whose parent is not loaded is 1** (design: "depth where the parent is loaded"; the fallback is unspecified). After repair, every active job has its whole chain loaded, so the fallback only applies to `CommitReady`/`CleanupReady` jobs under terminal parents, which never register children. The planner uses depth only for the apply order.
4. **The R4 message** is "tree repair: a child job failed or was cancelled", and the ancestors get the usual "descendant job … failed: …" message. The design does not specify either.
5. **The `RecoveryAfterRepair` context carries the repaired job IDs** (as the plan asks), and `RecoveryBeforeRepair` carries the loaded IDs. A tree with no loaded row fires neither hook, so "the second restart logs no repair" shows up either as an empty context or as no hit, depending on whether the repaired tree still has a non-terminal job.
6. **Terminal jobs are not cached by recovery**: a tree that repair makes quiescent gets no JCBs and no committer. `GetJobState` on its root falls back to the DB, as for any terminal job after a restart.

## For the report / later work

- R4's `CleanupReady`-child predicate is effectively shadowed by R5. Without it, such a tree still breaks I2, so R5 fails it as a whole instead of failing the chain. Only the unit test distinguishes the two.
- The repair transaction's `UPDATE`s reuse `apply_tree_transition` unchanged (guarded, fenced). A crash between the repair COMMIT and the end of recovery is covered by `a_crash_during_recovery_repairs_the_tree_once` (after-repair case).
- `recover_jobs` probes `has_failed_child` one query per `BlockingOnChildren` row, sequentially. B7 should time recovery with many blocked parents if that matters.

## Host state

The relayed user request in this run asked for the kind cluster to be killed. It was already stopped: `clp-rc2-control-plane` is `Exited (137)` from about 6 hours earlier, and no container of it is running. It was neither started nor deleted; deleting it cannot be undone, so that is left to Pleiades. This slice's MariaDB container `jtp-impl-s6` was stopped at the end, and no other container is running.

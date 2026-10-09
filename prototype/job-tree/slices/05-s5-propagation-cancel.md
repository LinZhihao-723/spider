# Slice S5 `s5-propagation-cancel`: notes

Author: implementer (last attempt on this slice). Reviewed input: `reviews/05-s5-propagation-cancel-review.md`, which found that S5 was not in the worktree at all. This attempt implements the slice and addresses review issues 1-6. Issue 7 (sorting per abort) is a performance note and is left for B3.

## What changed (non-test)

- `db/protocol.rs`: `ExternalJobOrchestration::get_state_and_parent(job_id) -> Result<(JobState, Option<JobId>), DbError>`.
- `db/mariadb.rs`: one PK read of `state` and `parent_id`; `JobNotFound` for a missing row.
- `state/test_utils.rs`: `MockDbConnector::get_state_and_parent` reads `states` and `parents` (no entry means a root).
- `state/service.rs`:
  - a free function `ensure_root(job_id, parent_id)`: a non-root is `StorageServerError::BadRequest` (gRPC `INVALID_ARGUMENT`, no `grpc.rs` change);
  - a private `ensure_uncached_root(job_id)`: reads `get_state_and_parent`, rejects a non-root, and leaves every other result (including `JobNotFound` and DB errors) to the RPC's own read, so the existing unknown-id answers are unchanged;
  - `StartJob`: root check on a cache hit; a miss is still `JobNotFound` without a DB read (plan S5);
  - `CancelJob`: root check on a cache hit **before** `jcb.cancel()`; on a miss it reads `get_state_and_parent` and rejects a non-root **before** the "non-terminal but not cached" fatal arm. This closes review issue 3: a `CancelJob(child)` in the registration window (children committed, not yet cached) is now `BadRequest`, not a fatal shutdown;
  - `GetJobState`, `GetJobOutputs`, `GetJobError`: root check on a hit (`GetJobError` now looks at the cache only for that check), `ensure_uncached_root` on a miss, then today's read;
  - `#[cfg(feature = "test-hooks")] ServiceState::cached_job_state(job_id)`, a test accessor for any cached job (root or not), which the S4 tests needed once `GetJobState` became root-only.
- `cache/tree.rs`: `plan_cancel` asserts that the cancelled job is the tree's root (review issue 2's defensive assert). It is unreachable from the service now.
- `cache/task.rs`: `TaskGraph::drop_prepared_children()`.
- `cache/job.rs`: `apply_tree_state` drops the prepared children of every TCB of a deactivated job, before `finalize()`, so a TCB that the task-instance pool still holds no longer keeps up to `MAX_REPORT_CHILD_BYTES` of decoded children alive until the pool's next GC (review issue 6). Docstrings of `apply_tree_state`, `SharedJobControlBlock::cancel` and `ServiceState::cancel_job` updated.

About 140 non-test lines, docstrings included (plan: 150).

## Tests

New file `tests/job_tree_race_test.rs` (feature `test-hooks`, included from `test_spider_storage.rs`), all on `TestEnv` (own scratch database). Every race is a loop of `JOB_TREE_RACE_ITERATIONS` (default 50) on fresh trees, each iteration ending with an assert that the tree has no non-terminal row. A generic `race(env, root, point, first, second, second_event)` implements the plan's recipes: pause the first RPC's round at `CommitterAfterReceive` (same round) or `CommitterBeforeDbTransaction` (consecutive rounds), wait for the second RPC's `EventSubmitted`, release.

| Test | Covers | Exact asserts |
|---|---|---|
| `two_tasks_failing_a_leaf_fail_its_tree_once` | T1 in a tree (R → A → {J (2 tasks), S}), same round and consecutive rounds | first reply `Failed`, second `JobAlreadyTerminated(Failed)`; J's error is the first message; R and A `Failed` with "descendant job J failed"; S `Cancelled`; exactly one cache apply per job (`Observe@CommitterAfterApplyJob` per job) |
| `a_root_cancel_racing_a_registration_leaves_no_running_child` | T2 (i) same round, (ii) registration in flight then cancel, (iii) cancel committed then the report | (i) cancel `Cancelled`, registration `JobAlreadyTerminated(Cancelled)`, no child rows; (ii) registration `BlockingOnChildren`, cancel `Cancelled`, all 3 grandchildren rows `Cancelled`; (iii) report `JobNoLongerRunning` (phase A), no child rows |
| `a_root_cancel_racing_a_leaf_failure_is_planned_first` | T3 same round, failure first, cancel first | same round: cancel `Cancelled`, failure `JobAlreadyTerminated(Cancelled)`, no error message stored; failure first: `Failed` and the cancel `JobAlreadyTerminated(Failed)` (the `TreeClosed` path); cancel first: failure `JobAlreadyTerminated(Cancelled)` |
| `siblings_completing_together_advance_their_parent_once` | T4, k = 2, 8, 64; one round, k sequential rounds, free concurrency; parent with and without a commit task | every report `Succeeded`; parent applied exactly once; parent `Succeeded`/`CommitReady`; every child's stored outputs equal the expected bytes; the parent's own stored outputs unchanged. Runs `ceil(iterations / 5)` times over the 18 combinations |
| `a_straggler_failure_after_a_leaf_success_is_stale` | T5 for a leaf, failure in the window (paused before the DB transaction) and after the apply | in the window `TaskAlreadyTerminated(Succeeded)`, after `JobNoLongerRunning`; no error stored; sibling still `Running`, root `BlockingOnChildren`, then `Succeeded` |
| `a_cancel_skips_a_non_root_commit_ready_job_but_not_a_commit_ready_root` | T7: non-root P `CommitReady` with its commit registered; cancel then commit, and cancel and commit in one round; a `CommitReady` root with and without a cleanup task | P stays `CommitReady`, its commit gives `Succeeded`, the root stays `Cancelled`; root case: `Cancelled` or `CleanupReady` via the new edge, late commit `JobNoLongerCommitReady`, cleanup then `Cancelled` |
| `a_limit_violation_in_a_tree_fails_the_tree` | T10 tree variants (a leaf under a root with a sibling): 10,001 children in one report, an undecodable child, outputs over 4 MiB | leaf `Failed` with the matching message, root `Failed`, sibling `Cancelled`, no grandchild rows, not fatal. The depth variant is `registration_test::a_report_beyond_the_maximum_depth_fails_its_task` |
| `external_job_rpcs_accept_only_roots` | the external-RPC test | paused at `CommitterAfterCommit` of a registration: `CancelJob(child)` and `GetJobState(child)` are `BadRequest`, `StartJob(child)` is `JobNotFound`, `is_fatal() == false`; on a cache hit all five RPCs are `BadRequest` on the child; the root's state and outputs work; unknown ids keep today's answers |

Unit tests:
- `state::service::tests::job_rpcs_reject_an_uncached_child_job` (mock, cache miss, child `Running`/`Succeeded`/`Failed`): `CancelJob`, `GetJobState`, `GetJobOutputs`, `GetJobError` are `BadRequest`, `StartJob` is `JobNotFound`. This is the "after eviction" case of the plan's external-RPC test; the existing unknown-id unit tests pass unchanged.
- `cache::task::tests::dropped_prepared_children_leave_a_task_that_is_still_referenced`: a TCB held outside the graph (as the pool holds it) has no prepared children after `drop_prepared_children`.

Existing-test edits (mechanical, for the root-only RPCs; no assertion changed):
- `tests/registration_test.rs::assert_job_state` reads the cached state through `cached_job_state` (falling back to the row when not cached) instead of `GetJobState`, since it is used on children.
- `a_report_beyond_the_maximum_depth_fails_its_task` reads the deepest child's error with the new `TestEnv::row_error` instead of `GetJobError`.

### Stale-error variants (deviation from the plan's T1 table)

The plan's table expected `JobNoLongerRunning` for a failure or registration made stale by a terminal job. The real code answers `JobAlreadyTerminated(state)` when the job is already terminal in the plan or the cache (`TreeEventKind::stale_error`), and `JobNoLongerRunning` only from phase A (job not `Running` in the cache) or for a non-terminal state. The tests assert the real variants exactly; they are all stale (`FAILED_PRECONDITION`) and none is fatal.

### Mutation checks

Each defect was applied, the race tests were run with 5 iterations, and the file was restored and compared byte for byte with a copy.

| Defect | Caught by |
|---|---|
| no cancel-first rule (neither `take_events` nor `RoundOrder`) | `a_root_cancel_racing_a_leaf_failure_is_planned_first` (T2 (i) still passes: a registration is always planned last, so the cancel precedes it anyway) |
| `is_abortable` also aborts `CommitReady` non-roots | `a_cancel_skips_a_non_root_commit_ready_job_but_not_a_commit_ready_root` |
| no root check on the `CancelJob` cache-miss path | `job_rpcs_reject_an_uncached_child_job`, `external_job_rpcs_accept_only_roots` (got the fatal `Internal(JobNotFound)`) |

Not mutated here: "a second registration per round". None of the S5 tests puts two registrations in one round of the real code; the S3 unit test `take_events_takes_the_oldest_cancel_first_and_leaves_later_registrations_in_the_buffer` and the T11 property test cover it.

## Verification

- `cargo +nightly fmt --all -- --check`: clean.
- `cargo build --workspace`: OK. `cargo clippy --workspace --all-targets --all-features -- -D warnings` and `cargo clippy -p spider-storage --all-targets -- -D warnings`: clean.
- Debug build, `cargo test -p spider-storage --features test-hooks --lib`: 152/152 (the committer's `assert_invariants` runs after every round).
- `JOB_TREE_RACE_ITERATIONS=50`: the 8 new tests pass (about 10 s in total).
- Full suite: 265/265, see "Full runs" below.
- Container `jtp-impl-s5` (stopped at the end).

### Full runs

All runs: `cargo nextest run -p spider-storage --all-features --run-ignored all --release --no-fail-fast`, on `jtp-impl-s5`.

- Before the new tests existed (only the service and S4-test edits): 254/255. The one failure was `mariadb_test::test_delete_expired_terminated_jobs`: its job was missing from its own delete list, so another process deleted it first on the shared `spider-db`. Neither the test nor the delete query is touched by this slice, and it passed in every later run. It is listed as a shared-database flake, like the execution-manager liveness tests of the S1-S4 notes.
- `JOB_TREE_RACE_ITERATIONS=500`: **265/265** in 87.5 s (T4 is the slowest test, 87 s; T1-T3, T5 and T7 each take 23-36 s).
- Default iterations (50), twice: **265/265** both times (32 s each).
- No `jt_*` scratch database is left behind by a passing run.

## Host state

The relayed user request in this run asked for the kind cluster to be killed. It was already not running: its only container, `clp-rc2-control-plane`, is `Exited (137)` from about 5 hours earlier, and `docker ps` lists no running container besides this slice's own MariaDB container. The cluster was neither started nor deleted; deleting it cannot be undone, so that is left to Pleiades.

## For the next slice (S6)

- `SharedJobControlBlock::recover` must take the loaded `parent_id` and `depth` (S4 note, item 9); otherwise a recovered child passes the new root checks as a root.
- Every external RPC now relies on `parent_id` being right in the cache and in the row.

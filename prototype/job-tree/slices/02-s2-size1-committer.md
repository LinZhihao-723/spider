# Slice S2 `s2-size1-committer`: notes

Role: implementer. Worktree: `/home/lzh/dev/spider-job-tree-proto` (`git diff 278d06f`, on top of S1).

## What changed

- `cache/tree.rs` (new, not generic):
  - `Lane` (`accepts`, `stale_error`), `TreeEventKind` (`stale_error` shared by the planner and the `TreeClosed` mapping), `TreeEvent`, `TreeOutcome`, `TreeReply`, `Pending`.
  - `TreeControlBlock { root_id, sender, token, hooks }`:
    - `submit` is synchronous. A failed send answers `TreeClosed` at once. A successful send fires `Observe@EventSubmitted`.
    - `submit_and_wait` races the token (`ServiceStopping`). A closed oneshot is `InternalError::TreePoisoned`.
  - `TreeNode` and `TreeStructure` (non-terminal jobs only; `is_quiescent`).
  - The planner:
    - `PlannerInput`, `PlannerEvent { event, input: &mut PlannerInput }`, `Publication`, `EventOutcome`, `JobTransition`.
    - `Plan { updates: Vec<JobStateUpdate>, counters, publications, terminated, outcomes: BTreeMap<usize, _>, deferred, apply_order }`.
    - `plan_events` is `pub(crate)` (clippy `implicit_hasher`). It plans only the first event and defers the rest; a planned event takes its input, and a deferred one keeps it.
    - Per-event planners over a minimal `Overlay` (structure plus collected states), root-level branches only: `plan_start`, `plan_regular_done`, `plan_success` (no climb), `plan_commit_done`, `plan_cleanup_done`, `plan_failure` (lane check, no propagation), `plan_cancel` (`abort_target`; today's stale mapping).
    - `apply_order` is sorted root last, deepest first.
  - Unit tests: 8 planner cases.
- `db/protocol.rs`: `JobStateUpdate` (with `new`), `ChildJobInsert`, `TreeTransition`, and the required `InternalJobOrchestration::apply_tree_transition`.
- `db/mariadb.rs`:
  - `max_statement_payload = min(4 MiB, @@max_allowed_packet / 2)`, read at connect (deferred from S1).
  - `apply_tree_transition`: BEGIN, `check_fence`, then the guarded statements; an explicit ROLLBACK on any error; COMMIT. Non-empty `child_inserts` is `CorruptedDbState("child inserts are not supported yet")`.
  - `UpdateStatement::plan`:
    - groups by `(from, to, has outputs, error_message)`, with the outputs groups first;
    - cuts at 500 ids, and outputs groups where the bound outputs would exceed `max_payload`;
    - every statement keeps at least one row.
  - `UpdateStatement::execute`: `UPDATE jobs FORCE INDEX (PRIMARY) SET state = ? [, serialized_job_outputs = CASE id WHEN ? THEN ? … END] [, error_message = ?] [, ended_at = CURRENT_TIMESTAMP if to is terminal] WHERE id IN (…) AND state = ?`. A matched-row count different from the id count is `CorruptedDbState` naming the ids and the edge.
  - The old `start`/`set_state`/`commit_outputs`/`cancel`/`fail` are wrappers through `transition_one_job` (see deviation 1). `transition_job_state` and `UPDATE_JOB_STATE` are deleted.
  - Unit tests for the grouping and chunking.
- `cache/job.rs`:
  - The JCB gains `parent_id` (`None`), `depth` (0), `has_commit_task`/`has_cleanup_task` (read from the graph at construction), `tree: TreeControlBlock` and `PhantomData<fn() -> D>`.
  - `JobExecutionState<R, T>` and its handle lose the DB connector and the `D` parameter.
  - `create(id, owner, submission, sender, pool, tree)` and `recover(ctx, sender, pool, tree)`.
  - `start` and `cancel` submit `Start` and `CancelRoot`.
  - Success phase A:
    - `fetch_sub(AcqRel)`;
    - ready successors are sent after the guard drop;
    - the last task drops the guard, then submits `RegularTasksDone { registers: false }`;
    - hooks `PhaseAAfterTcbFlip` (under the guard, before `fetch_sub`) and `PhaseABeforeSubmit`.
  - Commit and cleanup success: `read_commit_ready`/`read_cleanup_ready`, then the TCB success, the guard drop and the submit.
  - Failure: the retry send happens after the guard drop; a terminal failure submits `TerminalFailure { lane, message }`. Phase B is gone.
  - `resend_ready_tasks` collects under the guard and sends after it is dropped.
  - `apply_tree_state(from, to)`:
    - plain write lock; a `from` mismatch is `CacheDbDivergence`;
    - `cancel_non_terminal` on deactivation, `finalize` when leaving `Running` or on deactivation (see deviation 3).
  - `collect_state`, `collect_regular_done_outputs` (`None` unless `Running`, trap R3) and `collect_ready_task_indices` (`None` unless `Ready`; the "no ready tasks" `TaskGraphCorrupted` only for a `Ready` job, trap R19/M1).
  - Reply mapping in `submit_event`: `Applied` → `Ok`, `Stale` → `StaleState`, and `TreeClosed` → `ServiceStopping` if the token is cancelled, else `kind.stale_error(cached state)`.
  - Deleted: every write validator, `validate_and_write`, `ensure_ready`/`ensure_cancellable`/`ensure_non_terminated`, and `InternalError::UnexpectedJobTermination` (dead).
  - `ensure_commit_ready`/`ensure_cleanup_ready` answer stale for `BlockingOnChildren`; `ensure_succeeded` answers `StaleStateError::JobNotSucceeded(state)` (#515).
- `cache/task.rs`: `SharedTaskControlBlock::succeed_task_instance` delegates to an impl with a feature-gated hook parameter. `succeed_task_instance_with_hooks` fires `PhaseABeforeSuccessorLock` after `write_outputs` and before the first successor lock. The public signature is unchanged.
- Errors:
  - `CacheError::ServiceStopping`;
  - `InternalError::{CacheDbDivergence, TreePoisoned, HandlerBodyFailed, InjectedFault (feature)}`;
  - `StaleStateError::JobNotSucceeded`;
  - `DbError::Injected { deterministic_payload }` (feature).
  - `StorageServerError` now has a manual `From<CacheError>` that maps `ServiceStopping` to `Stopping`, so every `?` site maps it.
- `state/publisher.rs` (new): `StagingPublisher<Q>` implements `InboundQueueSender`.
  - Sends push into unbounded per-lane queues and always return `Ok`. After shutdown, staged hints are dropped instead of failing the caller (a failure would be an internal error, so fatal, on a graceful stop).
  - One `forward` task per lane in the tracker, each wait raced against the token. A closed lane logs a warning and stops that lane.
  - Unit test: a full lane (capacity 1) delivers all entries once drained, and the lanes exit on cancel.
- `state/tree_committer.rs` (new):
  - `TreeEnv` (clonable), `create_root_tree`, `create_recovered_tree`, `spawn_committer`.
  - `run_committer`:
    - `recv_many(256)` raced against the token; `try_recv` while deferred events wait;
    - `CommitterAfterReceive` hook (drains again after it);
    - the token check before each round; `take_events` takes the oldest event;
    - the round is `Box::pin`ned unless the `unboxed-round` feature is on (added now, for B6);
    - quiescent exit with `close_and_answer(TreeClosed)`; shutdown exit with `close_and_answer(ServiceStopping)`.
  - `PoisonOnAbnormalExit` owns the receiver. Unless disarmed, its `Drop` logs and calls `shutdown.fatal()`.
  - `run_tree_state_transaction`:
    1. collect, reading every node's state when the round has a `CancelRoot` or `TerminalFailure`;
    2. `CommitterBeforePlan`, then plan;
    3. `CommitterBeforeDbTransaction` (`FailDb` → `DbError::Injected`), then `apply_tree_transition` if the plan is not empty;
    4. `CommitterAfterCommit`, then the apply in `apply_order` with `CommitterAfterApplyJob` per job;
    5. drop terminal nodes and handles, then GC-enqueue them;
    6. reply (a stale `TerminalFailure`'s message is logged at info);
    7. publish.
  - A `Stopping` fault at a pre-DB hook requeues the round, which the loop then answers `Stopping`. A fault after COMMIT poisons.
  - Unit tests (mock DB):
    - the committer exits exactly when the root is terminal, and later submits get the stale error;
    - the committer stays while the root is `CleanupReady`;
    - `committer_future_size` prints the parked future's size;
    - feature-gated: a shutdown lets the round in flight finish (`Applied`) and answers the queued event `Stopping`, and a later submit gets `TreeClosed` (rewritten after review 02);
    - feature-gated: a round requeued before the DB transaction collects its inputs again (review 02);
    - a poison guard dropped after the token is cancelled does not poison (review 02);
    - feature-gated: `Crash@CommitterBeforeDbTransaction` and `Panic@CommitterAfterApplyJob` close both waiters' oneshots, set the fatal flag, and later JCB calls get `ServiceStopping`.
- `state/service.rs`:
  - `ServiceStateInner` holds `publisher: StagingPublisher<Q>` instead of the raw sender, plus `tracker` and `hooks` (feature).
  - `pub type ServiceJobCache<Q, D, P> = JobCache<StagingPublisher<Q>, D, P>` is the type of `ServiceStateParams::job_cache`.
  - `new` (default hooks) and `new_with_hooks` (feature) both call `build`, which spawns the publisher and registers the token and tracker in the hooks.
  - New: `tracker()`, `test_hooks()`, `tree_env()`, and `recover_jobs()`, which sorts by id and fires `RecoveryBeforeRepair` and `RecoveryAfterRepair` per tree, then `create_recovered_tree`. The log message "Job recovered into cache." is unchanged.
  - `register_job` calls `create_root_tree`.
  - `start_job`, `cancel_job`, `create_task_instance` and the four report bodies run through `spawn_body` (tracker spawn raced against the token; a join error is `HandlerBodyFailed`).
  - `enqueue_for_gc_if_terminal` is removed.
- `state/runtime.rs`:
  - `create_runtime` and the feature-gated `create_runtime_with_hooks` both just call `build_runtime`.
  - `build_runtime`: connect, the inbound queue, the pool, an empty `JobCache`, the GC, `ServiceState`, then `recover_jobs`.
  - On an error after the pool spawn, `PartialRuntime::abort` cancels the token, closes and awaits the tracker, and awaits the pool and GC joins, all under the 30 s timeout. It logs, then returns the original error.
  - `Runtime` gains `tracker`. `stop` closes it and joins it with the actors. `Runtime::test_hooks()` is added.
- `state/job_cache.rs`: `insert_many`; `resend_ready_tasks` iterates a snapshot of the `Arc`s. `state/job_cache_gc.rs`: `JobCacheGcHandle::new` is now `pub` (a library API change for the report).
- `test_hooks.rs` (new, feature `test-hooks`): every hook point of plan §3 is defined and placed (`CommitterAfterReceive`, `CommitterBeforePlan`, `CommitterBeforeDbTransaction`, `CommitterAfterCommit`, `CommitterAfterApplyJob`, `PhaseAAfterTcbFlip`, `PhaseABeforeSubmit`, `PhaseABeforeSuccessorLock`, `CreateInstanceBeforePoolSend`, `EventSubmitted`, `RecoveryBeforeRepair`, `RecoveryAfterRepair`).
  - Actions:
    - `Pause` (semaphore raced against the registered runtime token, returning `InjectedFault::Stopping`);
    - `Observe`;
    - `Crash`;
    - `FailDb` (asserted to be at `CommitterBeforeDbTransaction` only);
    - `Panic`.
  - Counts `Once`, `Times(n)` and `Always`. A filter matches `root_id`, `job_id` and `EventKindTag`.
  - `observe()` is the synchronous path, and `arm` panics for non-`Observe` at `EventSubmitted`. `ArmedHook::{reached, release, hits, last_context, disarm}`; dropping a hook closes its semaphore. `release_all()`, `runtime_probe()`.
  - `HookContext` carries `task_instance_id` (used by T9(c)). Call sites use inline `#[cfg(feature = "test-hooks")]` statements, not a macro.
- Mock and harness edits (mechanical):
  - `MockDbConnector::apply_tree_transition` is faithful: all-or-nothing guard check, and it records outputs and errors.
  - New test helpers `create_detached_tree` and `create_test_tree_env`; the latter registers the token in the hooks.
  - `NoopDbConnector::apply_tree_transition`, and `run_workload` builds a `TreeEnv` and calls `create_root_tree`.
  - The service tests' hand-built JCB goes through `create_root_tree` with `db.states` seeded, and the now-redundant `job_cache.insert` lines are removed.
  - The job-cache resend test builds its JCB through `create_root_tree`.
  - The `Runtime { .. }` literal gains `tracker`. No assertion was changed.
- New tests:
  - `tests/tree_transition_test.rs` (MariaDB, one `ScratchDatabase` per test, see deviation 5):
    - a guard miss is corruption and rolls back earlier statements;
    - 1,200 updates (three chunks) plus one guard miss write nothing, and alone they all commit;
    - 3 × 1.5 MiB outputs (two statements at 4 MiB) are all read back;
    - a fenced-off connector gets `Fenced` with the exact ids and writes nothing;
    - `ended_at` is set only for terminal `to`.
  - `tests/job_tree_infra.rs` (feature-gated): `TestEnv::{start, restart, finish, register_job, poll_ready_tasks, row_state}` and `build_chain_task_graph`. `start` and `restart` call `resend_ready_tasks` after the runtime is built.
  - `tests/job_tree_test.rs` (feature-gated):
    - stale `StartJob`: a duplicate start after the task is running, a start after a cancel to `CleanupReady`, and a start after a cancel to `Cancelled`. Each is `Stale(JobAlreadyStarted)`, not fatal, with cache and DB in agreement.
    - T9 (a), (b), (c): the drops at `PhaseAAfterTcbFlip`, `PhaseABeforeSuccessorLock` and `CreateInstanceBeforePoolSend`; (c) observes the pool entry through the soft-timeout re-enqueue (review 02).
    - Plain T5: a straggler failure before COMMIT and in the commit-to-apply window is `TaskAlreadyTerminated(Succeeded)`; after the apply it is `JobNoLongerRunning`.
    - A cancel after a restart of a `CommitReady` job with a cleanup task runs the cleanup (review 02, #513).
    - The failed-build leak test.
    - A restart that republishes a recovered `Running` job's task.
  - Service unit tests: `start_job_twice_is_stale_and_not_fatal` and `start_job_after_cancel_job_is_stale_and_not_fatal`.

## Fixes after review 02 (slice reviewer)

1. **Major, fixed: a recovered `CommitReady` job keeps its cleanup task** (design §9.5, #513). `TaskGraph::recover_from_commit_ready` takes the submission's cleanup descriptor and builds the cleanup TCB, so the JCB's `has_cleanup_task` and the tree node's `has_cleanup` match a job that was never restarted. A cancel after a restart now plans `CommitReady → CleanupReady`, publishes the cleanup hint, and `create_cleanup_task_instance` finds the TCB. New test `job_tree_test::cancel_after_a_restart_of_a_commit_ready_job_runs_its_cleanup`: commit and cleanup tasks, `CommitReady`, restart, cancel, then it asserts `CleanupReady` in the cache and the DB, polls the cleanup hint, runs the cleanup instance, and asserts `Cancelled`.
2. **Minor, fixed: a requeue of every event clears the inputs.** Both pre-DB `Stopping` paths (`CommitterBeforePlan`, `CommitterBeforeDbTransaction`) return `requeue_all(events)`, which resets each `input` to `None`, so the next round collects it again. The docstring of `run_tree_state_transaction` says so. New hooked unit test `a_round_requeued_before_the_db_transaction_collects_its_inputs_again`: it runs one round directly, requeues it at a token-raced pause, asserts the input is cleared and the row is still `Ready`, then runs the requeued round and asserts `Applied(Running)` in the reply, the cache and the mock DB. The new `CommitterState::new_root` (used by `spawn_committer`) lets the test build the committer state.
3. **Minor, not fixed (S3 trap):** recorded under "For the next slice (S3)". It cannot happen in a one-node tree.
4. **Minor, fixed: the shutdown test follows production's shutdown path.**
   - The post-commit hook points (`CommitterAfterCommit`, `CommitterAfterApplyJob`) go through `hit_after_commit`. There, a pause that ends because the token fired lets the round finish, as production does; it is no longer an error that poisons.
   - The test is now `shutdown_finishes_the_round_in_flight_and_answers_the_queued_event_with_stopping`. It parks the `Start` at `Pause@CommitterAfterCommit`, queues a `CancelRoot`, and cancels the token. It asserts that the in-flight `Start` gets `Applied(Running)` and the queued cancel gets `ServiceStopping`. The cache and the mock DB both show `Running`, a later submit gets `TreeClosed`, and nothing is fatal.
5. **Minor, fixed: T9 (c) observes the pool entry.** It is renamed `dropped_instance_creation_still_hands_the_instance_to_the_pool`. It runs on a `TestEnv::start_with_task_instance_pool` with `gc_interval_sec: 1` and a chain task whose soft timeout is 100 ms. After the drop and release at `CreateInstanceBeforePoolSend`, it asserts that the pool re-enqueues task 0 once the soft timeout passes, which needs the pool entry, then succeeds the instance. `build_chain_task_graph` now takes an `ExecutionPolicy`, and `start_job` registers, starts and checks the first publication for any graph.
6. **Minor, fixed: `register_job` runs through `spawn_body`.** The fenced INSERT and `create_root_tree` run in the tracker, so a dropped `SubmitJob` future cannot leave a `Ready` row without a JCB and a committer. The request validation stays on the RPC future. `register_job` now also forwards `spawn_body`'s errors (`Stopping` on shutdown, `HandlerBodyFailed` on a panic). Design §10.10 should add registration to the spawned bodies.
7. **Minor, fixed: a committer dropped at teardown does not poison.** `PoisonOnAbnormalExit::drop` stays quiet (debug log) when the token is already cancelled and the thread is not panicking. The committer's explicit error exits (a closed channel, an injected fault, a failed tree state transaction) call `PoisonOnAbnormalExit::poison` directly, so they stay fatal even during a shutdown. New unit test `a_guard_dropped_after_the_token_is_cancelled_does_not_poison` covers both drops.
8. **Minor, fixed: `TestEnv::restart` expects a clean stop**, as `finish` does.
9. **Minor, fixed: T5 covers the commit-to-apply window.** `straggler_failure_after_a_success_is_stale` runs `assert_straggler_in_window_is_stale` at `Pause@CommitterBeforeDbTransaction` and at `Pause@CommitterAfterCommit`. In each window, the straggler's failure is `TaskAlreadyTerminated(Succeeded)`, the cache stays `Running`, and the row stays `Running` before COMMIT or `Succeeded` after it. After the release, both are `Succeeded`. The after-the-apply case is unchanged.
10. **Minor, fixed: docstrings** on `Plan::merge`/`finalize`, the `EventPlan` builders, `Overlay::state`/`node`/`abort_target` and every `plan_*` function, with `# Returns` and `# Panics`.

Verification of the fixes (container `jtp-impl-s2fix`, now stopped):

- `cargo +nightly fmt --all -- --check`, `cargo build --workspace`, and the three clippy runs (`--workspace --all-targets --all-features`, `-p spider-storage --all-targets`, `-p spider-storage --all-targets --features test-hooks`, all `-D warnings`): clean.
- `cargo nextest run -p spider-storage --all-features --run-ignored all --release --no-fail-fast`: **212/212 passed**. No `jt_*` scratch database was left behind. An earlier full run on the same container had one failure, `mariadb_test::test_update_execution_manager_heartbeat_already_dead`. It is the shared-`spider-db` timing flake already listed in the S1 note, it passed when rerun alone, and none of these fixes touch it.
- Mutation checks: all five were applied together, each test failed at the assertion for its own defect, and then the defects were reverted.
  - The recovered `CommitReady` graph without its cleanup TCB: the restart-cancel test fails, because the cancel answers `Cancelled`.
  - `requeue_all` without clearing the inputs: the requeue unit test fails.
  - `hit_after_commit` treating `Stopping` as an error: the shutdown test fails, because the in-flight round is not answered `Applied`.
  - The guard without its teardown branch: the guard unit test fails.
  - Handler bodies awaited inline: the new T9 (c) fails, because the poll times out without a re-enqueue.

Review notes (not defects) for the report: deviation 1's old writers remain until PR 9. `submit_and_wait` and `spawn_body` race the reply against the token without `biased`, so a `CancelJob` that the committer already applied can be answered `Stopping`, and the client's retry after the restart sees a stale error. The commit and cleanup stale errors differ between the windows before and after a cancel's apply (both map to FAILED_PRECONDITION).

## Verification

- `cargo build --workspace` and `cargo build --workspace --all-targets --all-features`: OK.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`, `cargo clippy -p spider-storage --all-targets -- -D warnings` and `cargo clippy -p spider-storage --all-targets --features test-hooks -- -D warnings`: clean. `cargo +nightly fmt --all -- --check`: clean.
- `cargo nextest run -p spider-storage --all-features --run-ignored all --release` on a fresh `spider-db` (container `jtp-impl-s2`, now stopped): **209/209 passed**, and no `jt_*` scratch database was left behind. The 28 new or affected tree, committer, publisher and stale tests passed 5 runs out of 5.
- Mutation checks, each applied, run against its targeted tests, then reverted:
  - Handler bodies awaited inline instead of spawned: T9 (a) and (b) fail. The first T9 (c) still passed; the rewritten one fails (see "Fixes after review 02").
  - The `Ready` check removed from `collect_ready_task_indices`: both stale-start tests fail. The first versions of the tests did not catch it; they were strengthened so that the job is non-`Ready`, has no ready task, and still has a live committer.
  - No matched-row check: the guard-miss and three-chunk tests fail.
  - No cancel and join in `PartialRuntime::abort`: the leak test fails.
  - The poison guard cancels without `fatal()`: both poison tests fail.
- B1 in-process smoke (`SPIDER_TEST_INSTRUMENT_OUTPUT_DIR`, `jcb_test`, release, each checkout on its own DB). The baseline was built into `claude/benchmarks/job-tree-proto/target-main`, with no tracked file touched.
  - The instrumentation is wired only into the `_without_db` workloads (the MariaDB variants pass no sender), so this is a phase-A comparison with a no-op DB. A workload has only one job transition.
  - Flat, success (ok): p50 0.001/0.001 ms, p99 0.069 (proto) vs 0.029 (main).
  - Neural net, success (ok): p99 0.940 vs 0.944.
  - Registration and failure are the same within noise.
  - Not a meaningful DB comparison; the gRPC B1 must provide it.

## Deviations from the plan and the design

1. **The old `DbStorage` writers check the fence before the state.** The plan's wrapper (`get_state`, validate, then `apply_tree_transition`) breaks `fence_test::newer_connector_fences_off_older_connector_with_same_fence_id`: `commit_outputs` on a `Ready` job would return `InvalidJobStateTransition` before reaching the fence. `transition_one_job` keeps today's single transaction instead: BEGIN, `check_fence`, `SELECT … FOR UPDATE`, today's validation, then the same guarded `UpdateStatement` that `apply_tree_transition` uses, then COMMIT. The error contracts are unchanged.
2. **`ServiceState::new_with_hooks`** (feature-gated) is how `create_runtime_with_hooks` hands its hooks over; `ServiceStateParams` is unchanged. `ServiceState::build`, not `build_runtime`, registers the token and the tracker for `runtime_probe()`; they are the runtime's.
3. **`apply_tree_state` also deactivates from `Ready`** (`Ready → Cancelled`): it runs `cancel_non_terminal` and `finalize`, as today's `cancel` does. The plan lists only `Running`, `BlockingOnChildren` and `CommitReady` as the from-states.
4. **The committer awaits `publisher.send_*`** instead of a separate `stage` API. In production the publisher is `StagingPublisher`, whose sends never wait, so this matches the design. Test publishers (`MockInboundQueueSender`, the `jcb_test` channel) are awaited directly.
5. **The `apply_tree_transition` tests use a `ScratchDatabase`**, not `spider-db`. On `spider-db` they left about 1,200 `Running` rows behind. `runtime_recovery_test` recovers and resends them, and its 32-entry poll then misses its own job. That made `restarted_storage_cache_recovers_{running_job_from_start,commit_ready_job}` fail under a concurrent full run on a fresh DB. The baseline passes under the same conditions, so this was caused by the new tests (R20). The plan's rule "tests that never recover stay on `spider-db`" should also require that they leave no non-terminal rows.
6. **Commit, cleanup and cleanup-instance creation have no `CreateInstanceBeforePoolSend` hook**; only the regular instance creation has one, which is the one T9 (c) needs.
7. **The stale outcome of a planned `TerminalFailure` is logged** at info with its message, as design §7.5 asks.
8. Each `TreeClosed` reply re-reads the job state for the stale error, so a submit to a quiescent tree takes one JCB read lock.

## For the next slice (S3)

- Replace `plan_events`'s "plan index 0, defer the rest" loop with the real ordering and deferral rules. `Plan.deferred` and `outcomes: BTreeMap<usize, _>` are already the committer's contract: an event without an outcome is requeued with its input. The committer's `reply()` relies on the outcome map, not on `deferred`.
- `take_events` (in `tree_committer.rs`) returns only the oldest event; S3 widens it.
- `Overlay` in `cache/tree.rs` is the minimal S2 one: it reads the collected states and does not see the plan applied so far. It panics on a missing collected state or node. S3 replaces it with the full overlay and `NodeState::Absent`.
- `plan_success` has no climb, and `plan_failure` no propagation or abort; the planners' signatures are final.
- `collect_inputs` already reads every node's state for rounds with `CancelRoot` or `TerminalFailure`, but only the events' own jobs otherwise (no ancestors yet).
- `PlannerInput::RegularTasksDone { outputs }` has no `children` yet (S4).
- The hook infrastructure is complete. For the "same round" recipe use `Pause@CommitterAfterReceive` plus `Observe@EventSubmitted`; the committer drains the channel again after that hook.
- `create_test_tree_env` (in `state/test_utils.rs`) is the mock-DB `TreeEnv` for committer-level grouping tests. It already registers the token, so `Pause` is token-raced.
- **`collect_inputs` makes an event for a job that is not in `handles` fatal** (review 02, issue 3; not fixed in S2, where it is unreachable). `state.handle(job_id)?` returns `InternalError::JobNotFound`. Once a tree has more than one node, a child removed from `handles` after it became terminal can still receive a straggler `CleanupDone` or `TerminalFailure` in the commit-to-apply window, and the service would exit. S3 must read a job that is missing from `handles` as `NodeState::Absent` (design §7.4) in `collect_inputs`, not only in the overlay.
- A requeue of every event (`requeue_all`) clears the events' inputs, so the next round collects them again. A requeue after a rolled-back DB transaction (S4, design §7.3 step 3) must do the same; the planner takes a planned event's input.

## Line counts (approximate, non-test)

- New files: `cache/tree.rs` 641, `state/tree_committer.rs` 695, `state/publisher.rs` 166. That is 1,502 lines, about half of them docstrings.
- Feature-gated hook code: `test_hooks.rs` 498, plus about 60 `#[cfg]` call-site lines.
- Modified files (net, S2 part):
  - `mariadb.rs` +330 (including about 110 test lines);
  - `service.rs` +200 non-test;
  - `runtime.rs` +140;
  - `task.rs` +66;
  - `protocol.rs` +81;
  - `job.rs` +31 net, with about 400 lines rewritten.
- In total, roughly 2,300 non-test lines against the plan's 950. The overshoot is docstrings, `build_runtime`'s cleanup, the statement planner and the full hook API (the plan budgets about 200 hook lines separately).

# Slice S4 `s4-registration`: notes

Role: implementer. Worktree: `/home/lzh/dev/spider-job-tree-proto` (`git diff 278d06f`, on top of S1-S3). Nothing was committed.

The sections below describe the slice as first implemented. "Review fixes" at the end describes what changed after the slice reviewer's review (`reviews/04-s4-registration-review.md`); where the two disagree, the fixes section is current.

## What changed

- `components/spider-proto-rust/src/unpack/storage.rs`: `ReportTaskSuccessRequest` unpacks to a 6-tuple ending in `Vec<(Vec<u8>, Vec<u8>)>` (graph, inputs); docstring updated.
- `src/job_submission.rs`: `ChildJobSubmission { compressed_serialized_task_graph, compressed_serialized_job_inputs }` with `size()`.
- `src/grpc.rs`, `report_task_success`: passes the children to `ServiceState::succeed_task_instance_with_children`. Non-empty `children` on a commit or cleanup report is `INVALID_ARGUMENT` ("children are only accepted from regular tasks") before any cache access (m10). The old blanket "child jobs are not supported" rejection is gone. The unit test now covers all three lanes (regular: `FAILED_PRECONDITION` from the cache miss; commit and cleanup: `INVALID_ARGUMENT`).
- `src/bin/grpc_server.rs`: `TaskInstanceManagementServiceServer::max_decoding_message_size(MAX_TASK_INSTANCE_MESSAGE_SIZE)` (32 MiB, N18).
- `src/state/limits.rs` (new): `MAX_DEPTH = 8`, `MAX_REPORT_CHILD_BYTES = 16 MiB`, `MAX_REPORT_OUTPUT_BYTES = 4 MiB`, `MAX_TASK_INSTANCE_MESSAGE_SIZE = 32 MiB`, and re-exports of the planner-side `MAX_CHILDREN_PER_JOB`, `MAX_JOB_CHILD_BYTES`, `MAX_EVENTS_PER_ROUND`, `MAX_ROUND_PAYLOAD` from `cache/tree.rs` (the plan's `MAX_TREE_TRANSACTION_PAYLOAD` is S3's `MAX_ROUND_PAYLOAD`; `IN_LIST_MAX_IDS` stays private in `mariadb.rs`).
- `src/db/protocol.rs`: `InternalJobOrchestration::max_statement_payload(&self) -> u64` (MariaDB: the S1 value; mock and noop: 4 MiB). `apply_tree_transition`'s docstring now covers inserts and `PayloadRejected`.
- `src/db/error.rs`: `DbError::PayloadRejected(sqlx::Error)` and `DbError::is_deterministic_payload_error()` (true for `PayloadRejected` and, with test hooks, `Injected { deterministic_payload: true }`).
- `src/db/mariadb.rs`, `apply_tree_transition`: after the UPDATE groups, `INSERT INTO jobs (resource_group_id, parent_id, root_id, state, compressed_serialized_task_graph, compressed_serialized_job_inputs) VALUES (?, ?, ?, ?, ?, ?), … RETURNING id`, with the state bound as `JobState::Running`. `chunk_child_inserts` cuts at 500 rows or where graph + inputs bytes would exceed `max_statement_payload` (at least one row per chunk). Each chunk's returned ID count is checked (`CorruptedDbState` otherwise). Any statement error before COMMIT with errno 1153 or 1406 becomes `PayloadRejected`; the transaction is rolled back as before. *(Review fix 4: only a child insert's error is classified now.)*
- `src/cache/error.rs`: `ChildPreparationError { Submission, TaskGraph, NoReadyTask }`.
- `src/cache/task.rs`:
  - `PreparedChild { graph, num_tasks, has_commit, has_cleanup, ready_task_indices, compressed_graph, compressed_inputs }` with `PreparedChild::create(ChildJobSubmission)` (validate, build the `TaskGraph`, read the ready indices; a graph without a ready task is `NoReadyTask`) and `size()`. *(Review fixes 7 and 10: now `{ graph, num_tasks, ready_task_indices, submission }`, without copies.)*
  - `TaskControlBlock.prepared_children`; `SharedTaskControlBlock::succeed_task_instance_with_children` stores them in the TCB critical section after `base.succeed_task_instance` and `write_outputs` succeed, so a duplicate or stale instance stores nothing. `succeed_task_instance` delegates with an empty vector; `succeed_task_instance_with_hooks` takes the children too.
  - `TaskGraph::take_prepared_children()` in task-index order (and report order within a task).
- `src/cache/job.rs`:
  - `JobExecutionState::new` (constructor, used by all three JCB constructors) and `has_prepared_children: AtomicBool`.
  - `succeed_task_instance_with_children`: the TCB call stores the children; then, still under the read guard and before the hook `PhaseAAfterTcbFlip` and the `AcqRel` `fetch_sub`, `has_prepared_children.store(true, Relaxed)` if this report brought children. The task that takes the counter to 0 loads the flag (`Relaxed`, after its `AcqRel` RMW) into `RegularTasksDone { registers }`. `succeed_task_instance` delegates.
  - `collect_regular_done_inputs()` (was `collect_regular_done_outputs`) returns `RegularDoneInputs { outputs, children }` when `Running`, taking the children out of the TCBs; `None` otherwise.
  - `SharedJobControlBlock::from_prepared_child(id, &parent, prepared, sender, pool)`: infallible, `Running`, parent ID, depth, owner and tree taken from the parent's control block. `from_parts` takes `parent_id` and `depth`.
- `src/state/error.rs`: `ReportRejection { OutputsTooLarge, TooDeep, TooManyChildren, ChildTooLarge, ChildrenTooLarge, InvalidChild }`; its message is the task failure message.
- `src/state/service.rs`:
  - `succeed_task_instance_with_children(session, job, instance, index, serialized_outputs, children)`; `succeed_task_instance` delegates with no children, so `runtime_recovery_test.rs` and every other caller are unchanged.
  - In the spawned body: look up the JCB, then `prepare_report(depth, outputs, children)`: outputs ≤ 4 MiB (sum of the decoded outputs, checked for every report), and, if there are children, depth + 1 ≤ `MAX_DEPTH`, count ≤ `MAX_CHILDREN_PER_JOB`, each child (graph + inputs) ≤ `max_statement_payload`, the report's children ≤ 16 MiB, then `PreparedChild::create` for each child. A rejection logs a warning and calls `jcb.fail_task_instance(instance, TaskId::Index(index), rejection.to_string())`; the RPC answers with the resulting job state (`Running` on a retry, `Failed` at the end).
- `src/state/tree_committer.rs`:
  - `Queued.children: Vec<PreparedChild>`.
  - `collect_inputs`: for `RegularTasksDone { registers }` it appends the taken children to `Queued.children` and sets `PlannerInput::RegularTasksDone.children` to `ChildSetSummary { count, bytes }` when `registers`. If `registers` disagrees with whether the event holds children, it returns `InternalError::TaskGraphCorrupted` (fatal; unreachable by the flag ordering above).
  - The S3 placeholder `assert!(plan.registration.is_none())` is gone. `write_plan` builds `TreeTransition.child_inserts` from the registering event's children (borrowed, no copy). On an error, if the round has a registration and `is_deterministic_payload_error()`, `fail_registration` logs, replaces that event with `TerminalFailure { Regular, "child payload rejected by the database" }` (same reply channel), drops its children, and `requeue_all` returns every event of the round in its original order with its input cleared. Any other error stays fatal.
  - After COMMIT and the apply (`apply_plan`): `CommitterState::bind_children` creates each child JCB with `from_prepared_child`, adds its `TreeNode` (parent, `child_depth`, the parent's resource group, flags from the graph, `active_children = 0`) and handle, and pushes `Publication::Ready` with its prepared `ready_task_indices`; then `JobCache::insert_many`. The parent's counter `k` comes from `plan.counters` as before. Then the resource-group lookup, `update_structure`, GC, replies and publications as in S3.
  - `requeue_all` is no longer test-hooks-only; it keeps already-taken children with their event.
  - New helpers to stay under clippy's 100-line limit: `plan_round`, `write_plan`, `apply_plan`, `child_inserts`, `fail_registration`.
- Mocks: `MockDbConnector` gains `parents` (child → parent) and `next_job_id`; it inserts children as `Running` with ascending IDs and implements `max_statement_payload`. `NoopDbConnector` (`tests/scheduling_infra.rs`) returns one random ID per insert and 4 MiB.

## Tests

New, all passing:

- `grpc::tests::report_task_success_accepts_children_only_from_regular_tasks`.
- `db::mariadb::tests::child_inserts_are_cut_by_rows_and_by_payload` (500-row cut, payload cut, one row minimum, empty input).
- `state::tree_committer::tests` (mock DB, debug build, so the invariant check runs after every round):
  - `a_registration_blocks_the_parent_until_every_child_succeeds`: parent `BlockingOnChildren` with its outputs stored, two child rows with the parent recorded, children cached `Running` at depth 1 in the parent's tree, the first child's success keeps the parent blocked, the second unblocks it to `Succeeded`; GC order `[child 1, child 2, parent]`; not fatal.
  - `hooked::a_rejected_registration_payload_fails_the_parent_in_the_next_round`: `FailDb { deterministic_payload: true }` once; a `Start` forced into the same round (pause at `CommitterAfterReceive`) is answered exactly once (`Stale(JobAlreadyStarted)`); two rounds, the second `[TerminalFailure, Start]` in arrival order; the parent `Failed` with "child payload rejected by the database"; no child rows; not fatal.
  - `hooked::a_payload_error_without_a_registration_poisons_the_tree` and `hooked::a_transient_error_of_a_registration_poisons_the_tree` (the branch needs both conditions).
- `tests/registration_test.rs` (MariaDB, `test-hooks`, each on its own `TestEnv`):
  - `a_registered_parent_commits_after_its_children_succeed` (registration happy path and T6 core): the parent `BlockingOnChildren` in cache and row with outputs in the row; 3 child rows with `root_id`, `Running`, IDs above the parent's and in VALUES order (checked through each row's graph); children cached `Running`; each child's task published exactly once; the parent stays blocked until the last child, then `CommitReady` with one commit publication; the commit's input is exactly the parent's own outputs; commit success gives `Succeeded`.
  - `sibling_tasks_register_the_union_of_their_children_in_task_order`: task 1 reports one child first (`Running`, no rows yet), then task 0 two; one registration with rows ordered `[a0, a1, b]`; all children run, the parent `Succeeded`.
  - `a_duplicate_report_with_children_stores_nothing`: the duplicate gets `InvalidTaskInstanceId`, and the registration holds only the first report's child.
  - `a_report_over_a_limit_fails_its_task` (T10): count `MAX_CHILDREN_PER_JOB + 1`, a child over the statement payload, 5 × 3.5 MiB over the per-report budget, an undecodable child, and outputs over 4 MiB; each with `max_num_retry` 0 (one report gives `Failed`) and 1 (`Running`, republished, then `Failed`); error message checked; no child rows; not fatal.
  - `a_report_beyond_the_maximum_depth_fails_its_task` (T10 depth): a chain registered down to depth 8 (all allowed), the depth-8 job's child is rejected, that job fails, and the failure propagates to every ancestor (all `Failed`); no depth-9 row.
  - `a_rejected_registration_payload_fails_the_parent`: the payload branch on MariaDB.
- `tests/tree_transition_test.rs::child_inserts_return_their_ids_in_order_across_chunks`: 1,200 child inserts in one transition (three chunks) return IDs that match the rows' ID order and VALUES order; the parent moves `Running → BlockingOnChildren` in the same transaction.

Mutation checks (each applied, run, then reverted; the files were compared byte for byte with saved copies afterwards):

| Defect | Caught by |
|---|---|
| prepared children not stored in the TCB | both committer registration tests (internal error on the `registers` mismatch) |
| `registers` always false | both committer registration tests |
| no lane check in `report_task_success` | the gRPC test |
| payload error never requeues | the committer payload test |
| payload error requeues without a registration | `a_payload_error_without_a_registration_poisons_the_tree` |
| children taken in reverse task order | `sibling_tasks_register_the_union_of_their_children_in_task_order` |
| children stored before the instance check | `a_duplicate_report_with_children_stores_nothing` |
| depth check off by one | `a_report_beyond_the_maximum_depth_fails_its_task` |

Verification: see the final section.

## Deviations from the plan and the design

1. **The mock allocates job IDs from one ascending counter** (`register` and child inserts), instead of `JobId::random()` for registrations and a dedicated range for inserts (R13). The debug invariant check (I1: a parent's ID is smaller than its child's) failed with random parent IDs. This matches auto-increment.
2. **`from_prepared_child(id, &parent, prepared, sender, pool)`** instead of `(id, parent_id, depth, owner, prepared, sender, pool, tree)`: the parent's ID, depth + 1, owner and tree are read from the parent's control block (8 parameters would trip clippy, and they can all be derived).
3. **The 4 MiB output limit applies to every success report, children or not**, and counts the decoded outputs' bytes. Raising the task-instance decode limit to 32 MiB removed tonic's 4 MiB default, which was the effective output limit before (design §13 "today's effective limit, unchanged").
4. **A rejected report answers the RPC with the job state after the task failure** (`Running` when retried, `Failed` at the end), not an error. The EM is not told that its success became a failure; the job's error message carries the reason. A formal implementation may want a distinct status.
5. *(Resolved by review fix 7.)* **`PreparedChild::create` clones the two compressed buffers** (as the plan says), because `ValidatedJobSubmission::into_parts` drops them. Up to 16 MiB per report is copied once. The formal implementation can add an accessor that returns them instead.
6. **A `registers`/children mismatch at collect is a fatal internal error.** The flag is set before the `AcqRel` decrement whenever a report stores children, so the last task always sees it; the check is there because a silent mismatch would succeed the parent without its children or let two registrations into one round.
7. **`requeue_all` keeps already-taken children with their event** (inputs are cleared). Only the registering event ever holds children, and it is either replaced (payload branch) or answered `Stopping`, so this only matters if a future path requeues a registration without failing it.
8. *(Answered by review fix 5: a real 1153 arrives as a database error and is classified.)* **`PayloadRejected` only recognizes errors that sqlx reports as database errors with errno 1153 or 1406.** A server that answers an oversized packet by closing the connection surfaces as an I/O error and stays fatal. The limits make both unreachable; the optional real-1153/1406 test was not written.

## For the next slice (S5)

- `MockDbConnector.parents` (child → parent) exists for `get_state_and_parent`; registered roots have no entry.
- Prepared children of a job that a cancel or failure deactivates while `Running` are already dropped: `apply_tree_state` runs `finalize()` when leaving `Running`, which drops the TCBs and their prepared children, and `collect_regular_done_inputs` returns `None` for a non-`Running` job. A separate `drop_prepared_children` (plan S5) may be unnecessary; a test that asserts it is still worthwhile.
- Children are cached with `parent_id`, `depth` and the root's tree, and their rows carry `parent_id`/`root_id`. The external RPCs do not check for roots yet (`get_job_state` on a child works, which the S4 tests use).
- `tests/registration_test.rs` has small helpers (`build_job`, `child`, `run_only_task`, `child_rows`, `poll_one_task_each`) that the race tests can move into `job_tree_infra.rs`; `TreeSpec`/`TestEm` were not built in this slice.
- The depth test already shows a leaf failure propagating up an 8-deep `BlockingOnChildren` chain on the real code.
- Restarting with a tree is still unsupported until S6 (R9); no S4 test restarts.

## Line counts (approximate)

Non-test lines added in this slice, docstrings included: `state/tree_committer.rs` +267 (827 → 1,094), `state/service.rs` about +115, `db/mariadb.rs` about +125, `cache/task.rs` about +105, `cache/job.rs` about +105, `state/error.rs` +27, `db/error.rs` +20, `db/protocol.rs` +12, `job_submission.rs` +18, `cache/error.rs` +13, `state/limits.rs` +18, `grpc.rs` +20, `bin/grpc_server.rs` +4, unpack +15. In total about 840, against the plan's 400. About half is docstrings; the rest is mostly the service validation with its typed rejection enum, the insert chunking, and the helpers split out of `run_tree_state_transaction` for clippy.

## Verification

- `cargo +nightly fmt --all -- --check`: clean.
- `cargo build --workspace` and `cargo build --workspace --all-targets --all-features`: OK, no warnings.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`, and `-p spider-storage --all-targets` without features and with `--features test-hooks`: clean.
- Debug build, `cargo test -p spider-storage --features test-hooks --lib`: 146/146 passed (the committer's `assert_invariants` runs after every round).
- `cargo nextest run -p spider-storage --all-features --run-ignored all --release --no-fail-fast` on container `jtp-impl-s4` (now stopped), five full runs of 246 tests:
  - one run 246/246;
  - four runs 245/246, each failing a single test of the execution-manager liveness family that the S1 and S3 notes already list as flaky (`CURRENT_TIMESTAMP` and 1 s thresholds on the shared `execution_managers` table): `test_update_execution_manager_heartbeat_already_dead` three times (25 s each) and `test_is_execution_manager_alive_em_dead` once. The heartbeat test passed 5/5 when run alone. Running only the 21 execution-manager tests (`execution_manager` filter, so no registration code runs) failed 2 of 4 runs (`…_marks_dead`, `…_multiple`). This slice does not touch that code or those tests.
  - Every registration, tree, committer, transition, fence, upgrade, publisher, stale and T9 test passed in every run. No `jt_*` scratch database was left behind.
- The kind cluster `clp-rc2` was left stopped (`clp-rc2-control-plane` is still `Exited (137)`). It was not started or deleted.

## Review fixes (slice reviewer, `reviews/04-s4-registration-review.md`)

Every issue was fixed; none is rebutted. Issue 2 was fixed in the DB layer instead of the planner, for the reason given there.

### 1 (major): a failure message larger than `max_allowed_packet`

- `cache/tree.rs`: `MAX_ERROR_MESSAGE_CHARS = 64 Ki` characters (re-exported from `state/limits.rs`). `fail_and_abort` stores the failing job's own message cut to that many characters, so every lane and every source (task failures, report rejections, the committer's own failures) is covered in the one place that writes the row. The ancestors' copy is still cut to 1,024 characters; both cuts share a `truncate_chars` helper.
- The 32 MiB decode limit stays on the whole task-instance service (design §13). A stored message is now at most 256 KiB, and `UpdateStatement` gives each distinct message its own statement, so a statement carries one message.
- Tests: `cache::tree::tests::a_failure_message_is_truncated` (multi-byte characters, so a byte cut would be caught) and `registration_test::a_failure_message_over_the_packet_size_is_truncated` (MariaDB: a message of `@@max_allowed_packet + 1` bytes fails the job, the stored error is the first 64 Ki characters, not fatal).

### 2 (major): a job's stored outputs bounded only per report

- **Where the limit lives.** I first put a per-job limit on the raw output bytes into the planner. Two things showed it was wrong:
  - The planner property test (T11) panicked: a non-registering `RegularTasksDone` that fails its job aborts the tree, but `jobs_to_collect` collects only the job and its ancestors for that event kind.
  - `jcb_test::test_flat_success_*` failed. It is the pre-existing 10,000-task flat workload: 10 MiB of raw outputs that zstd compresses to almost nothing, and that `main` stores fine. A raw-bytes limit regresses it.
- **The fix.** The limit is checked on the serialized (compressed) row, which is what has to fit in a packet:
  - `db/mariadb.rs`: `UpdateStatement::plan(updates, max_payload, max_outputs_size)` returns `DbError::OutputsTooLarge { job_id, size, limit }` when one row's serialized outputs exceed `max_outputs_size`. That is `max_allowed_packet - MAX_STATEMENT_OVERHEAD` (64 KiB), read at connect. Planning runs before `BEGIN`, so nothing is sent.
  - Rows within the limit are safe: a multi-row statement is cut at `max_statement_payload` (at most half a packet), and a single-row statement carries one row of at most `max_outputs_size`.
  - `state/tree_committer.rs`: `fail_registration` became `fail_culprit(events, registration, error)`. For `OutputsTooLarge` it finds the round's `RegularTasksDone` of `job_id`, the only event that stores that job's outputs, and turns it into `TerminalFailure { Regular, "job outputs of N serialized bytes exceed the limit of L bytes" }` with the same reply channel. It drops the event's children and requeues the whole round in order, exactly like the payload branch. The next round plans a `TerminalFailure`, which reads every job, so the abort of the rest of the tree is planned against collected states.
  - The registration payload branch is unchanged: `is_deterministic_payload_error()` with a registration. Any other error, or an `OutputsTooLarge` whose job has no `RegularTasksDone` in the round, stays fatal.
  - `MockDbConnector::apply_tree_transition` refuses raw outputs above its 4 MiB statement payload with the same error.
- **Deviation from the design (for the report).** Design §6.2 step 5 says the limits make 1153 unreachable. That was false for the union of a job's outputs. The prototype adds a per-job bound on the serialized outputs. It is enforced where the row is serialized and turned into a job failure by the committer, not by the planner, because:
  - the planner sees raw bytes;
  - a planner-side failure of a non-registering event would need every job's state collected for every completion.
- Tests:
  - `db::mariadb::tests::outputs_over_the_limit_name_their_job`;
  - `state::tree_committer::tests::outputs_over_the_job_limit_fail_the_job_and_abort_its_tree` (mock: a child's outputs over the limit fail the child, fail the root and cancel the sibling; not fatal);
  - `registration_test::outputs_over_the_packet_size_fail_the_job` (MariaDB: 12 tasks each report 2 MiB of incompressible output, within the 4 MiB per-report check; the union is over the packet; the job is `Failed` with the message, has no stored outputs, and the service is not fatal);
  - `tree_transition_test::only_a_child_insert_over_the_packet_size_is_a_payload_rejection` (its second half: an UPDATE with outputs over the packet is refused with `OutputsTooLarge` before it is sent).

### 3 (major): per-job child limits checked only after every child is built

- `cache/job.rs`: `SharedJobControlBlock::reserve_children(ChildSetSummary) -> Result<(), ChildSetSummary>` and `release_children`. They work on a `std::sync::Mutex<ChildSetSummary>` in the JCB, outside the state `RwLock`, with no await while it is held. A reservation fits only if the job's reserved count and bytes stay within `MAX_CHILDREN_PER_JOB` and `MAX_JOB_CHILD_BYTES`; the error carries the totals it would reach. I used a mutex rather than two atomics so that the two limits are checked and taken together, without a rollback window that could reject a concurrent report spuriously.
- `state/service.rs`, `prepare_report(&jcb, outputs, children)`:
  - after the per-report checks and before any decoding, the report's children are reserved;
  - over a limit, the report is rejected with the new `ReportRejection::JobChildrenTooLarge` ("child jobs would bring the job to N child jobs of B bytes, over the limits of … per job"), a task failure like every other rejection;
  - an `InvalidChild` releases the reservation;
  - in the handler, an error from `succeed_task_instance_with_children` (a stale or duplicate instance, a job no longer `Running`) releases it too.
  - So at most one report's children beyond the per-job limits are ever decoded or held at a time per concurrent report, instead of every report's.
- The planner's union check ("child-job limit exceeded") stays as defence in depth; with the reservation it is no longer reachable from the service.
- Deviation (design §6.2 step 3 has the same gap): the per-job totals are reserved in the service layer, before decoding.
- Two traps for the formal implementation:
  - The reservation is released on *any* error of the success call, including one after the TCB stored the children, such as `ServiceStopping` from `submit_event` or a publish error. Then the job is stopping or no longer `Running`, or the planner's union check still catches an over-limit union, so this is safe, but it is not exact.
  - A duplicate report holds its reservation until the TCB rejects it. A third report of the same job that arrives in that window and would fit only without the duplicate is rejected: a spurious task failure at the edge of the limit, retried if retries remain.
- Tests:
  - `registration_test::children_over_the_job_limits_across_reports_fail_the_task_before_decoding`: task 0 stores 2 valid incompressible children (about 2 MiB each). Task 1 reports 4 undecodable 4 MiB children: each is within the per-child and per-report limits, and the union is over 16 MiB. The job fails with the per-job message, not with "is invalid", which shows the rejection comes before decoding. No child rows.
  - `registration_test::rejected_reports_release_their_children_from_the_job_limits`: an `InvalidChild` report, then a valid report, then its duplicate (stale), then a sibling's report sized so that it fits only if both earlier reservations were released. The parent registers all of them, `BlockingOnChildren`.

### 4 (minor): the payload branch blamed the registration for any statement's payload error

- `db/mariadb.rs`: only `insert_children`'s errors are mapped to `PayloadRejected`. With fixes 1 and 2, an UPDATE cannot carry an oversized payload anyway: outputs are refused before sending, and messages are cut.
- `DbError::PayloadRejected`'s message and the protocol docstring now say "a child insert's payload".

### 5 (minor, tests): the real errno classification was never exercised

- New MariaDB test `tree_transition_test::only_a_child_insert_over_the_packet_size_is_a_payload_rejection`. It calls `apply_tree_transition` directly with one child insert of `@@max_allowed_packet + 1` bytes; the service limits are bypassed. It answers deviation 8: **a real 1153 reaches the classifier as a database error with errno 1153, not as an I/O error**, so it becomes `DbError::PayloadRejected`. The transaction is rolled back: the parent is still `Running`, with no child rows. No global variable was changed, so the test is safe to run in parallel.

### 6 (minor, tests): two design-named cases were only unit-tested

- Union over the limit across two reports: `children_over_the_job_limits_across_reports_fail_the_task_before_decoding` (fix 3). Since fix 3, the union is caught at the second report, so "the parent is `Failed`, no child rows" is checked there.
- A co-event in the payload branch: `state::tree_committer::tests::hooked::a_completion_rolled_back_with_a_rejected_registration_is_planned_again`.
  - The setup is a root with two registered children. One child's `RegularTasksDone` (no children) and the other child's registration land in one round: the completion arrives first, held at `CommitterAfterReceive` until the registration's `EventSubmitted` fires.
  - The injected payload error rolls back both planned updates.
  - Round 2 is `[RegularTasksDone, TerminalFailure]`. The completion is collected again and answered once with `Succeeded`, with its outputs stored. The registering child is `Failed` and fails the root. There are no grandchildren, and the service is not fatal.

### 7 (minor): child preparation on the runtime worker, with copies

- `prepare_report` yields to the runtime (`tokio::task::yield_now`) before each child, so one report no longer blocks a worker for all of its children; a single child's decode still runs inline, as a root submission's does.
- No more copies: `ValidatedJobSubmission::into_parts_with_compressed()` returns the decoded parts and the compressed buffers (as a `ChildJobSubmission`). `TaskGraph::create` now forwards to a private `TaskGraph::from_parts(task_graph, task_graph_input)`, which `PreparedChild::create` calls. `PreparedChild` holds `submission: ChildJobSubmission`, and `child_inserts` borrows from it. Deviation 5 is resolved.
- Not done: moving the decode to `spawn_blocking`. Yielding bounds the stall per child, and each child is at most `max_statement_payload` compressed.

### 8 (minor): a rejected success report logged as "Task instance succeeded."

- The rejection path now logs the existing `warn!` and then `info!` "Task instance failed for its rejected success report." with the resulting job state, and returns early. Only an accepted report logs "Task instance succeeded.".
- For the report: the RPC still answers a rejected success with the job state (`Running` on a retry, `Failed` at the end), as in deviation 4. The formal implementation needs a distinct status, so that the EM can tell that its success became a failure.

### 9 (minor, trap for S6): a recovered job is always a root at depth 0

- Not changed in S4 (no S4 path recovers a child). S6 must give `SharedJobControlBlock::recover` the loaded `parent_id` and `depth` and pass them to `from_parts`, instead of `None` and `0`. Otherwise:
  - a recovered depth-8 child can register children down to depth 16 (`prepare_report` reads `jcb.depth()`);
  - its JCB depth disagrees with the structure's `child_depth`.

### 10 (minor, style)

- `PreparedChild::has_commit` and `has_cleanup` are removed (`bind_children` reads the JCB's flags).
- TCB docstrings: the leaf errors (`TaskGraphCorrupted`, `TaskControlBlockBase::succeed_task_instance`, `TaskControlBlock::write_outputs`, `TestHooks::hit`) moved to `succeed_task_instance_impl`, which now has its own summary. `succeed_task_instance`, `succeed_task_instance_with_children` and `succeed_task_instance_with_hooks` forward to it.
- `ReportRejection::TooManyChildren` says "per report".
- `TaskGraph::take_prepared_children` is `pub(crate)`.

### Mutation checks for the fixes

Each defect was applied, the named tests were run, and the file was restored. The restored files were compared byte for byte with copies taken before the run.

| Defect | Caught by |
|---|---|
| the failing job's own message not cut | `a_failure_message_is_truncated`, `a_failure_message_over_the_packet_size_is_truncated` |
| no per-job reservation check | `children_over_the_job_limits_across_reports_fail_the_task_before_decoding` |
| no release after a rejected success call | `rejected_reports_release_their_children_from_the_job_limits` |
| no release after an invalid child | `rejected_reports_release_their_children_from_the_job_limits` |
| `is_payload_rejection` always false | `only_a_child_insert_over_the_packet_size_is_a_payload_rejection` |
| no serialized-outputs check in `UpdateStatement::plan` | `outputs_over_the_limit_name_their_job`, `outputs_over_the_packet_size_fail_the_job`, `only_a_child_insert_over_the_packet_size_is_a_payload_rejection` |
| `OutputsTooLarge` left fatal in `fail_culprit` | `outputs_over_the_job_limit_fail_the_job_and_abort_its_tree`, `outputs_over_the_packet_size_fail_the_job` |
| `fail_culprit` requeues only the failed event | `a_completion_rolled_back_with_a_rejected_registration_is_planned_again`, `a_rejected_registration_payload_fails_the_parent_in_the_next_round` |
| (planner-side version, since replaced) states collected before the output check | `outputs_over_the_job_limit_fail_the_job_and_abort_its_tree` and the T11 property test |

One mutation is no longer observable: classifying every statement's 1153 as `PayloadRejected` (the pre-fix-4 behaviour). An UPDATE can no longer reach the server with an oversized payload, so no test can tell the two apart; fix 4 stays as defence in depth.

### Verification after the fixes

- `cargo +nightly fmt --all -- --check`: clean.
- `cargo build --workspace` and `cargo build --workspace --all-targets --all-features`: OK.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`, and `-p spider-storage --all-targets` without features and with `--features test-hooks`: clean.
- Debug build, `cargo test -p spider-storage --features test-hooks --lib`: 150/150 passed. The committer's `assert_invariants` runs after every round, and the T11 property test caught the first, planner-side version of fix 2.
- `cargo nextest run -p spider-storage --all-features --run-ignored all --release --no-fail-fast` on my own container `jtp-impl-s4fix` (stopped afterwards), 255 tests:
  - the run before the last test edit had 254/255: the payload test's second half still expected the error that an oversized UPDATE got before fix 2, and I updated it to expect `OutputsTooLarge`. After that edit, three full runs had 254/255 each.
  - Each of those three failures was one test of the execution-manager liveness family that the S1, S3 and S4 notes already list as flaky (timestamp thresholds of about 1 s on the shared `execution_managers` table): `test_is_execution_manager_alive_em_dead` twice and `test_get_dead_execution_managers_multiple` once.
  - Run alone 4 times each, they failed once each: `test_is_execution_manager_alive_em_dead` 3/4 passed, `test_get_dead_execution_managers_multiple` 3/4 passed. This slice changes neither those tests nor the execution-manager queries.
  - Every registration, tree, committer, transition, fence, upgrade, publisher, stale, T9 and `jcb_test` test passed in every run, including the 10,000-task flat workload that the planner-side version of fix 2 broke.
- The kind cluster `clp-rc2` was left alone: `clp-rc2-control-plane` was already `Exited (137)`. It was neither started nor deleted.

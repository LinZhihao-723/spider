# Review: S2 `s2-size1-committer` (slice reviewer)

I reviewed the S2 changes in `/home/lzh/dev/spider-job-tree-proto` (`git diff 278d06f`) against:
- the slice note `slices/02-s2-size1-committer.md`;
- design sections 5.2, 6.1, 7.1–7.8, 9.3–9.5 and 10.1–10.10;
- the S1 review;
- the Rust review and style guides.

Paths below are relative to `components/spider-storage/`.

Verdict: **fix_needed**. There is one major issue: a recovered `CommitReady` job loses its cleanup task, so a cancel after a restart skips the cleanup. Everything else is minor. Most of the minor issues are traps that S3 and S4 will walk into, or tests that claim more than they check.

## What holds up

- **Locks.**
  - No path holds two JCB locks.
  - No JCB guard is held across a DB call, a reply wait or a lane send. Phase A, the retry re-enqueue and `resend_ready_tasks` all drop the guard before they send (`cache/job.rs:428-433, 560-566, 321`).
  - The commit and cleanup successes drop the read guard before `submit_event` (`cache/job.rs:476, 507`).
  - The only send still made under a JCB read lock is the pool send in instance creation, as §10.8 allows.
- **The apply.** The committer's collect and apply each take one JCB lock at a time. The apply's write lock excludes every phase-A reader, so `cancel_non_terminal` cannot wait on a TCB lock held by a success that is waiting for a successor.
- **`fetch_sub`.** It is `AcqRel` (`cache/job.rs:419`), and the hook sits before it, under the guard. The send on the channel orders the last task's release sequence before the committer's collect.
- **Exactly one DB transaction per round.**
  - `apply_tree_transition` runs BEGIN, `check_fence`, the guarded statements in plan order, then COMMIT, and rolls back explicitly on any error (`db/mariadb.rs:348-382`).
  - Matched-row counts are reliable: sqlx sets `FOUND_ROWS` (`sqlx-mysql-0.8.6/src/connection/stream.rs:46`). An empty plan skips the DB transaction.
- **Exact guards.**
  - Every planner reads `from` from the collected state and writes one edge.
  - Every edge the planners produce is valid in `is_valid_transition`, including `Ready → Cancelled`, `CommitReady → Cancelled` and `CleanupReady → Failed`.
  - The cache side checks `from` again in `apply_tree_state` (`cache/job.rs:648-655`).
- **Stale mapping.**
  - `plan_cancel` matches today's `ensure_cancellable`.
  - The `TreeClosed` path reuses `TreeEventKind::stale_error`, so a quiescent tree answers what the planner would have answered.
  - A send that fails after a poison sees the cancelled token and maps to `ServiceStopping` (`cache/job.rs:783-788`).
- **Committer lifecycle.**
  - `recv_many` is cancel safe, and the token check after receiving catches both branches of the `select!`.
  - `close_and_answer` closes the receiver before it drains, so no event is lost between the quiescent check and a late `submit`.
  - The poison guard calls `fatal()`, which sets the flag before it cancels the token. The receiver is dropped after that, so every queued oneshot closes.
- **Shutdown and build.**
  - `Runtime::stop` now joins the tracker and fails on an actor error, which fixes S1 issue 1.
  - `PartialRuntime::abort` cancels, closes and joins everything under the stop timeout.
  - The failed-build test does exercise a recovered tree's committer, which was spawned before the crash and stopped by the abort.

## Issues

### 1. Major: a recovered `CommitReady` job has no cleanup task, so a cancel after a restart skips the cleanup

- **Where:**
  - `cache/task.rs:143-159`: `recover_from_commit_ready` sets `cleanup_task: None`.
  - `cache/job.rs:749-750`: `from_parts` derives `has_cleanup_task` from that graph.
  - `state/tree_committer.rs:282-283`: the derived flag becomes `TreeNode.has_cleanup`.
  - `cache/tree.rs:522-528, 633-637`: `abort_target` and `plan_cancel` read it.
- **Scenario:**
  1. A job has a commit task and a cleanup task. Its last regular task succeeds, and the committer writes and applies `Running → CommitReady`.
  2. Storage restarts. `recover` builds the graph with `recover_from_commit_ready`, so the JCB's `has_cleanup_task` is `false`, and so is the tree node's `has_cleanup`.
  3. The client sends `CancelJob`. `plan_cancel` calls `abort_target`: the state is not `Ready` and `has_cleanup` is false, so the target is `Cancelled`.
  4. The committer writes `CommitReady → Cancelled`. The cleanup task never runs.
  5. Without the restart, the same cancel gives `CleanupReady` and the cleanup runs. The outcome depends on whether a restart happened.
- **A plain fix of the flag is not enough.** If only the flag were taken from the submission's graph, the planner would choose `CleanupReady` and publish the cleanup hint. But the recovered graph still has no cleanup TCB, so `create_cleanup_task_instance` would return the internal `UndefinedCleanupTask` (`cache/job.rs:940-943`), which is fatal.
- **Design:** §9.5 requires this: "`recover_from_commit_ready` keeps its cleanup TCB (today `cleanup_task: None`, `cache/task.rs:151`)".
- **Scope:**
  - The baseline has the same bug in `cancel`: it reads `task_graph.has_cleanup_task()`.
  - But S2 is the slice that turned the flag into a planner input fixed at construction.
  - It also blocks T7 and T8, the restart tests of a `CommitReady` root.
- **Fix:** build the cleanup TCB in `recover_from_commit_ready` from the submission's cleanup descriptor. Add a test: register a job with commit and cleanup tasks, reach `CommitReady`, restart, cancel, and assert `CleanupReady` with a published cleanup hint.

### 2. Minor (trap for S4): after planning, a round that returns all its events requeues them without their inputs, so a re-plan panics

- **Where:**
  - `cache/tree.rs:420`: `plan_events` takes the planned event's input with `mem::take`, which leaves `Some(PlannerInput::None)`.
  - `state/tree_committer.rs:467`: the `Stopping` fault at `CommitterBeforeDbTransaction` returns `Ok(events)` after planning.
  - `state/tree_committer.rs:576-579`: `collect_inputs` skips every event whose input `is_some()`.
- **Scenario (S4):** design §7.3 step (3) makes a rolled-back registration requeue every event of the round.
  1. The round `[RegularTasksDone(P) with children]` is planned, which consumes the outputs.
  2. `apply_tree_transition` fails with errno 1153, and the round is requeued.
  3. The next round does not collect the inputs again, because the input is `Some(None)`.
  4. `plan_regular_done` finds P still `Running` and hits `panic!("a running job's regular-tasks-done event should carry its outputs")` (`cache/tree.rs:571`). The committer poisons, and the service exits.
- **Why S2 survives:** in S2 the only path is the hook's `Stopping`, which fires only once the token is cancelled, so the loop answers `Stopping` before it re-plans.
- **Doc:** the docstring at `state/tree_committer.rs:405-407` promises to return "every event" on that path, as if they could be replanned.
- **Fix:** either plan on clones and take the inputs only once the plan has committed, or reset `input = None` on every event that a requeue-all returns.

### 3. Minor (trap for S3): `collect_inputs` makes an event for an absent job fatal

- **Where:** `state/tree_committer.rs:572, 580` (`state.handle(job_id)?`) and `:187-201` (`InternalError::JobNotFound`).
- **Problem:** design §7.4 says that the planner never errors or panics on a missing node, and that such a node reads as `Absent`.
- **Scenario (S3):**
  1. In a multi-node tree, a child C becomes terminal. The committer removes C from `handles` (`:516-519`), but C's JCB stays cached and keeps a sender.
  2. A straggler `CleanupDone(C)` or `TerminalFailure(C)` passes C's JCB validator in the commit-to-apply window and is submitted.
  3. `collect_inputs` returns `JobNotFound`, and the whole service exits.
- **Why S2 cannot reach it:** a one-node tree exits as soon as its root is terminal, so later events get `TreeClosed`.
- **The slice note** lists only the `Overlay` panics as S3 work. It does not list this error path.

### 4. Minor: the shutdown test checks a hook artefact, not production's shutdown path

- **Where:** `state/tree_committer.rs:897-932` (`shutdown_answers_every_waiting_event_with_stopping`).
- **What it does:** the test parks the in-flight `Start` at a token-raced `Pause` before the DB transaction. So the in-flight round itself answers `Stopping`, by way of `return Ok(events)` at `:467`.
- **Why that differs from production:** production has no await between planning and the DB transaction. Design §7.8 says that "the round in flight finishes its DB transaction or fails it" and is answered `Applied`; only the queued events get `Stopping`.
- **What is untested:** the committed round's reply when the token fires mid-transaction, and the fact that the DB and the cache agree afterwards. The test name says it covers "every waiting event", but it only covers hook behaviour. It also leaves the start's input consumed, as in issue 2.
- **Fix:** pause at `CommitterAfterCommit` instead.
  1. Cancel the token while the round is parked there, and queue a second event.
  2. Assert that the in-flight event's oneshot gets `Applied(Running)`.
  3. Assert that the queued event gets `Stopping`, and that the cache and the DB both show `Running`.

### 5. Minor: T9 (c) passes whether or not the handler body is spawned

- **Where:** `tests/job_tree_test.rs:233-281` (`dropped_instance_creation_still_registers_the_instance`).
- **Problem:** the slice note's own mutation check shows that the test still passes when bodies are awaited inline. The TCB registers the instance before the hook, so the later success report succeeds either way. Nothing observes the pool entry, which is what a drop at `CreateInstanceBeforePoolSend` would lose.
- **What the test misses:** a dropped body leaves a TCB instance with no pool entry. If the EM then dies, the pool never re-enqueues the task, and the job is stranded. That is the §10.10 case.
- **Fix:** make the loss observable.
  - One way: register the EM, let it go stale, and assert the pool's re-enqueue of the task (the `poll_ready_tasks` entry) after the GC tick.
  - Another way: expose a test-only pool probe.
  - Until then, record the gap in the report (the note already does) and drop the claim from the test name.

### 6. Minor (pre-existing, design gap): `register_job` is not run through `spawn_body`

- **Where:** `state/service.rs:258-290`.
- **Scenario:**
  1. A client's `SubmitJob` RPC is dropped after the fenced INSERT commits, while `create_root_tree` awaits `TaskGraph::create` or the map write lock.
  2. The row is `Ready` in the DB, but there is no JCB and no committer.
  3. The job is invisible until the next restart. Recovery then loads it as a `Ready` job that nobody will start.
  4. If its id ever reaches `CancelJob`, `cancel_job` finds a non-terminal row that is not cached and returns the internal `JobNotFound`, which is fatal (`:341-352`).
- **Scope:** the baseline has the same shape. Design §10.10 lists the bodies to spawn and omits registration. Wrapping the body in `spawn_body` costs one spawn. Record it in the report's traps if it is not fixed.

### 7. Minor: a committer dropped at runtime teardown logs "exited abnormally" and calls `fatal()`

- **Where:** `state/tree_committer.rs:230-239` and `bin/grpc_server.rs:60` (`runtime.shutdown_background()`).
- **Scenario:**
  1. On the deadline path, `wait_for_shutdown` returns `FAILURE` before `runtime.stop()` finishes, or `run()` returns early after a bind failure.
  2. `shutdown_background` drops every parked committer future.
  3. Each guard's `Drop` runs undisarmed, so it logs an error and calls `fatal()`.
  4. With 100k live trees (B6), that is 100k error lines on an exit that has already been decided. The same noise appears in every `#[tokio::test]` that ends without stopping its runtime.
- **Fix:** have the guard check `shutdown.token().is_cancelled()` before it logs. A drop after the token fired is a teardown, not a poison.

### 8. Minor: `TestEnv::restart` discards the stop result

- **Where:** `tests/job_tree_infra.rs:122` (`let _ = runtime.stop().await;`).
- **Problem:** the review guide forbids silently swallowing errors. `restart` is the T8 entry point. If a committer or a body fails to stop, or an actor exits on an error, the restart test still passes, and the next runtime races the old one's leftovers on the same scratch DB.
- **Fix:** `.expect("the runtime should stop cleanly")`, as `finish` already does.

### 9. Minor: the T5 "before the apply" case pauses before COMMIT, not in the commit-to-apply window

- **Where:** `tests/job_tree_test.rs:291-299` (`Pause@CommitterBeforeDbTransaction`).
- **Problem:** design §10.9's row "between that tree state transaction's COMMIT and its apply" is the window that is new in this design: the DB says `Succeeded` while the cache says `Running`.
- **What the test misses:** the TCB check decides both windows the same way, so the assertion holds. But nothing checks that the DB row and the cache agree once the straggler has been rejected inside the lag window.
- **Fix:** add the same straggler at `Pause@CommitterAfterCommit`.

### 10. Minor (style): the planner's private functions have no docstrings

- **Where:** `cache/tree.rs:437-462` (`Plan::merge`, `Plan::finalize`) and `:506-639` (`Overlay::state`, `Overlay::node`, `plan_event`, `plan_start`, `plan_regular_done`, `plan_success`, `plan_commit_done`, `plan_cleanup_done`, `plan_failure`, `plan_cancel`).
- **Problem:** the review guide says that visibility does not exempt a function from a docstring. These functions return non-trivial values and need at least `# Returns`. `Overlay::state` and `Overlay::node` also need `# Panics`.
- These are the functions that S3 rewrites, so fix them there at the latest.

## Notes for the report (not defects)

- **Deviation 1** keeps `start`, `set_state`, `commit_outputs`, `cancel` and `fail` on `InternalJobOrchestration`. Only `tests/fence_test.rs` and the `mariadb.rs` tests call them now. Design §9.3 deletes them in PR 9.
- **`submit_and_wait` and `spawn_body` race the reply against the token without `biased`.** An event that the committer has already applied can therefore be answered `Stopping`, and the client's retry after the restart then sees a stale error. The design accepts this, but the report should state the user-visible effect for `CancelJob`.
- **The commit and cleanup stale errors depend on the window.**
  - Before the cancel's apply, the planner answers `JobAlreadyTerminated(state)`.
  - After the apply, the validator answers `JobNoLongerCommitReady` or `JobNoLongerCleanupReady`.
  - Both map to FAILED_PRECONDITION, but the two errors for one race differ.

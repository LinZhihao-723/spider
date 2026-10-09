# Review: slice S5 `s5-propagation-cancel` (slice reviewer)

Verdict: **fix_needed**. Slice S5 is not in the worktree. The S4 code that S5 was meant to wire up and prove has a fatal-shutdown window and lets a client cancel a whole tree through a child's ID. Both are reachable today and are listed below so that the S5 coder can fix them.

Reviewed: `git diff 278d06f` in `/home/lzh/dev/spider-job-tree-proto` at 2026-10-09 03:17 EDT, against plan §S5 and §4 (T1-T7, T10), and design §4.6, §6.4, §6.5, §6.6, §7.5 and §9.6.

## What was found

- `slices/` holds only `01`-`04`. There is no `05-s5-propagation-cancel.md`.
- No S5 symbol exists anywhere in `components/spider-storage`:
  - `get_state_and_parent`;
  - `drop_prepared_children`;
  - `JOB_TREE_RACE_ITERATIONS`;
  - `TreeSpec` and `TestEm`;
  - a root check in any external RPC.
- The newest source change (03:11) is S4's review-fix round. The S4 note was last written at 03:15.
- No other worktree has the work, and no cargo build or MariaDB container is running.

So this review covers the propagation, abort and cancel paths that S3 and S4 already built. These are the paths S5 would wire up and prove.

## Issues

### 1 (critical): the slice is not implemented

The plan's S5 scope is all missing:

- the root-only external RPCs (design §9.6, N17);
- `ExternalJobOrchestration::get_state_and_parent` in the trait, MariaDB and the mock (`MockDbConnector.parents` exists but nothing reads it);
- the cancel cache-miss semantics of §6.5;
- the dropping of prepared children of aborted jobs, or a test that it already happens;
- `Absent` parents after GC eviction, end to end;
- T1 (in a tree), T2, T3, T4 (k = 2, 8, 64, in one round and in k rounds), T5 (leaf), T7 and the T10 tree variants;
- the external-RPC test.

Issues 2-5 are the concrete consequences.

### 2 (major): `CancelJob` on a cached child cancels the whole tree, and can abort a job in the middle of its commit

- `ServiceState::cancel_job` (`state/service.rs:350-359`) calls `jcb.cancel()` on any cached job. This submits `CancelRoot` with `job_id` set to the child (`cache/job.rs:751`).
- `plan_event` hands `event.job_id` to `plan_cancel` as `root_id` (`cache/tree.rs:1268`).
- `plan_cancel` (`cache/tree.rs:1537-1555`) then:
  - deactivates the child unconditionally (the `job_id == root_id` branch);
  - deactivates every abortable job of the tree, the real root R included.

Scenario:

1. R is `BlockingOnChildren`. Its child A is `Running`, and its child P is a non-root `CommitReady` job whose commit task is running on an executor.
2. A client sends `CancelJob(P)`. Job IDs are auto-increment, so any client can name P.
3. The plan has three effects:
   - P goes `CommitReady → CleanupReady`, and its cleanup is published while the commit still runs. Design §6.5 (Q2) forbids this, because executors are not preempted.
   - A and R are aborted.
   - The RPC returns `Ok(CleanupReady)` for P.

The planned fix is `jcb.parent_id().is_some()` → `BadRequest` before the submit. The plan for that fix is sound. A defensive `assert!(root_id == structure.root_id())` in `plan_cancel` would make any future misuse loud. That would follow the style guide's "assert preconditions" rule.

### 3 (major): a cache-miss `CancelJob` in the registration window shuts the service down

- `run_tree_state_transaction` commits the children's rows (`write_plan`) before they reach the cache (`state/tree_committer.rs:624-645`). In between it:
  - hits `CommitterAfterCommit`;
  - applies the parent under its JCB write lock, which waits for every phase-A reader of the parent;
  - binds the children;
  - only then calls `env.job_cache.insert_many(children)`.
- During that window, `CancelJob(child_id)` misses the cache. `db.get_state` returns `Running`, so `cancel_job` returns `CacheError::Internal(InternalError::JobNotFound)` (`state/service.rs:360-368`).
- `job_orchestration_service_error_handler` sends that to `default_error_handler`, which calls `shutdown().fatal()` (`grpc.rs:420-427`). One request kills the storage process, and every tree restarts through recovery.

S5's plan reads `get_state_and_parent` and rejects a non-root with `BadRequest`. That closes the window, but only if the non-root check runs **before** the "non-terminal but not cached" fatal arm. That ordering is a trap for the implementation, and the external-RPC test should cover it: pause at `CommitterAfterCommit` with a registration, call `CancelJob(child)`, and assert `BadRequest` and `is_fatal() == false`.

The same window exists for a root between `register_job`'s INSERT and its cache insert. That is pre-existing behaviour from `main`, and the client does not know the ID yet, so it is out of scope.

### 4 (major): `StartJob`, `GetJobState`, `GetJobOutputs` and `GetJobError` are not root-only

The four RPCs are at `state/service.rs:315-437`:

- `GetJobOutputs(child)` on a `Succeeded` child returns the child's outputs from the cache or the DB. Design §9.6 requires `INVALID_ARGUMENT`.
- `GetJobState(child)` and `GetJobError(child)` answer for children. The S4 tests rely on this (S4 note, "For the next slice"), so S5 must move those tests to a raw-SQL row read or a test accessor when it adds the check.
- `StartJob(child)` on a cached child is planned by `plan_start` as stale `JobAlreadyStarted` (FAILED_PRECONDITION) instead of `BadRequest` (INVALID_ARGUMENT).

### 5 (major): no race test runs on the real code

Tree behaviour is covered by planner unit tests over a hand-built `TreeStructure`, for example:

- `a_cancel_is_planned_first_and_makes_a_registration_and_a_completion_stale` (`cache/tree.rs:2049`);
- `a_cancel_and_a_failure_in_one_round_cancel_the_tree_and_drop_the_failure` (`:2086`);
- `siblings_completing_in_one_round_unblock_their_parent_once` (`:2193`);
- `a_cancel_skips_a_non_root_commit_ready_job` (`:2295`).

The committer tests use the mock DB and single-job trees (`state/tree_committer.rs`, `a_cancel_is_planned_before_a_failure_that_arrived_first`).

None of the following runs end to end:

- a cancel on a multi-job tree written by `apply_tree_transition`;
- the skipped non-root `CommitReady` job and its late `CommitDone` against an `Absent` or `CleanupReady` parent;
- the one-observation-per-parent count of T4 under real concurrency;
- the "failure first, then the cancel is `JobAlreadyTerminated`" path, which goes through `TreeOutcome::TreeClosed` and then `kind.stale_error(self.state())` (`cache/job.rs:908-930`) once the failed root has left the structure.

Each must be a hook-driven loop of `JOB_TREE_RACE_ITERATIONS` with exact asserts (plan §4). Each test must also be broken deliberately at least once, per the style guide's "passing tests are not coverage" rule:

- drop the cancel-first rule in `take_events`;
- drop the `CommitReady` skip in `is_abortable`;
- allow a second registration per round.

### 6 (minor): no planned drop of prepared children, and no test that the drop happens

- The S4 note argues that `apply_tree_state` → `finalize()` (`cache/job.rs:765-799`) already drops prepared children. That holds for the task graph's `Vec`. However, the task-instance pool still holds an `Arc` of a succeeded TCB for each straggler instance that is still registered (`task_instance_pool.rs:184-207`). Up to `MAX_REPORT_CHILD_BYTES` of prepared children therefore stay alive until the pool's next GC pass.
- The memory is bounded, so this is not a leak. But the plan item is neither implemented nor asserted.
- At least add the test the S4 note suggests:
  1. use `max_num_instances = 2`;
  2. have i1 succeed with children;
  3. cancel the root before the last task;
  4. assert that `take_prepared_children` on the job yields nothing and that the pool entry is gone after one GC pass.

### 7 (minor, performance): every abort sorts the whole tree

- `abort_rest` and `plan_cancel` each call `TreeStructure::sorted_job_ids()`, which allocates and sorts all N non-terminal IDs (`cache/tree.rs:334-338`, `:1519`, `:1544`).
- A round with a cancel plus k failures, which T1 and T3 produce, sorts N IDs k + 1 times.
- At the design's N = 10,000, this is small next to the 369 ms DB transaction, but B3 should report it. Sorting once per round, or applying in `Plan::finalize`'s depth order only, would remove it.

## Checked and found correct (S3/S4 code on the S5 paths)

- **Lock discipline:**
  - Phase A (`succeed_task_instance_with_children`, `fail_task_instance`) drops the JCB read guard before every lane send and before `submit_event`.
  - The committer takes one JCB lock at a time (`collect_inputs`, `apply_plan`, `assert_invariants`) and holds none across the DB transaction or the publisher.
  - The publisher's sends never wait (`state/publisher.rs`).
- **`fetch_sub(AcqRel)`** with the relaxed `has_prepared_children` store before it and the load after it: the release sequence makes the last decrementer see the flag.
- **Exact from-state guards:** `apply_tree_state` checks `job.state != from` and returns `CacheDbDivergence`. Every plan update carries the overlay's state as `from`, and `Plan::admits` rejects a second change to a job in the same round.
- **Ordering:**
  - `take_events` takes the oldest `CancelRoot` from anywhere in the buffer, and `RoundOrder` plans it first.
  - Only the oldest registration is taken, and later ones keep their place.
  - Duplicate cancels in one round are planned against the overlay and come out stale.
- **Abort semantics:**
  - `is_abortable` (Running or `BlockingOnChildren`) skips non-root `CommitReady` jobs.
  - A skipped job's commit success or failure stops at an `Absent`, `CleanupReady` or `Cancelled` parent (`climb_after_success`, and `fail_and_abort`'s `propagates`).
  - A `CommitReady` root goes to `CleanupReady` or `Cancelled` through the new edge, and its commit TCB is cancelled by `cancel_non_terminal`.
- **Stale vs internal:**
  - A failure that races an abort is stale through `Lane::accepts`.
  - A cancel after the root has failed is `JobAlreadyTerminated(Failed)` through `TreeClosed`.
  - A dropped reply maps to `TreePoisoned`, which happens only after the guard has already poisoned the service.

## Host state

The relayed user request asked for the kind cluster to be killed. It is not running: its only container, `clp-rc2-control-plane`, shows `Exited (137)` from about 5 hours earlier, and `docker ps` lists no running containers. The cluster was not started or deleted. Deleting it cannot be undone, so that is left to Pleiades.

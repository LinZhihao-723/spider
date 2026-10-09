# Review: S3 `s3-planner` (slice reviewer)

I reviewed the S3 changes in `/home/lzh/dev/spider-job-tree-proto` (`git diff 278d06f`, mainly `cache/tree.rs` and `state/tree_committer.rs`) against:
- the slice note `slices/03-s3-planner.md`;
- design sections 3.2–3.3, 6.1–6.6, 7.1–7.5, 8.2–8.4, 13 and 15;
- the plan's S3 section and risks R3, R8 and R12;
- the S2 review;
- the Rust review and style guides.

Paths below are relative to `components/spider-storage/`.

Verdict: **fix_needed**. There is one major issue: the round payload budget can starve a registration indefinitely, because the registration is always planned last. Everything else is minor: two tests that pass for the wrong reason or test an unreachable sequence, weak coverage in the property test, one stale docstring, and style.

## What holds up

- **Exactly one DB transaction per round.**
  - `run_tree_state_transaction` calls `apply_tree_transition` once, with every update of the round (`state/tree_committer.rs:542-551`).
  - An all-stale plan skips it.
  - S1/S2's `UpdateStatement::plan` handles what grouping now produces: several outputs rows, per-row outputs through `CASE id`, and different error messages split by group (`db/mariadb.rs:1081-1143`).
- **Exact from-state guards.**
  - Every planner takes `from` from the overlay: `plan_success`, `deactivate` and `plan_failure` read `job_state`.
  - `unblock` and the failure chain hard-code `BlockingOnChildren`, but only after `is(…, BlockingOnChildren)`, or after the `blocking_ancestors` assertion (`cache/tree.rs:1130-1139, 1319, 1338-1344`).
  - The touch-twice rule keeps every row to one edge per round (`cache/tree.rs:844-850`), and `apply_tree_state` checks `from` again on the cache side.
- **Ordering.**
  - Cancel comes first in `take_events`, from anywhere in the buffer (`state/tree_committer.rs:789-794`), and again in `planning_order` (`cache/tree.rs:1159-1182`).
  - At most one registering event is planned, and it is planned last. Extra registrations are deferred with their inputs intact.
  - The per-job deferral rule (`cache/tree.rs:576-587`) keeps one job's events in arrival order.
  - Requeued events go back to the front in their round order (`state/tree_committer.rs:427-431`).
- **The overlay and `Absent`.**
  - `Overlay::state` reads the planned state first, then `Absent` for a job missing from the structure, then the collected state. Every "blocking?" and "abortable?" test rejects `Absent`.
  - `jobs_to_collect` is shared with the property test. A missing collect panics instead of misplanning, and the property test exercises that path.
  - S2 issue 3 is fixed: an event whose job left the tree gets `PlannerInput::None` without a JCB lookup (`state/tree_committer.rs:650-653`), and the planner answers `Absent`.
- **Stale vs. internal.**
  - A lane mismatch is `Stale`, and `Absent` becomes `TreeClosed`. The JCB maps `TreeClosed` to `kind.stale_error(cached state)`. That state is terminal, because the apply runs before `update_structure` drops the node.
  - Only a guard miss in the DB or the cache, or an invariant panic, is fatal.
  - Deviation 1 is sound. Both answers are FAILED_PRECONDITION, and the JCB, unlike the planner, knows the terminal state of an absent root.
- **Locks.**
  - `collect_inputs` takes one JCB read lock at a time and holds none across `plan_events`.
  - The apply takes one write lock at a time.
  - `assert_invariants` takes one read lock at a time, after the reply.
  - Publication sends go to the staging publisher, which never waits (`state/publisher.rs:105-114`).
  - Only the committer writes a job state, so the per-lock snapshot that `collect_inputs` takes is consistent.
- **Post-apply bookkeeping.**
  - `update_structure` writes the counters before it drops the terminated nodes, so a parent that is decremented and failed in the same round is set, then removed (`state/tree_committer.rs:215-224`). Nothing else sets a counter for an `Absent` job, because the climb stops at a parent that is not `BlockingOnChildren`.
  - The handles and the structure stay in step.
- **`check_invariants`.**
  - Walked I1–I5 against design §8.2.
  - No false positive for skipped `CommitReady` jobs under a `CleanupReady` or absent parent, or for a closed tree whose root is absent.
  - The named corruption test covers every invariant.
- **S2 follow-ups.**
  - Issue 1: `recover_from_commit_ready` now keeps the cleanup TCB (`cache/task.rs:143-160`).
  - Issue 2: `requeue_all` clears the inputs (`state/tree_committer.rs:750-755`).
- **The touch-twice note.** I tried every producer pair and agree that no real event sequence triggers the rule today. Cancel comes first, the registration comes last, and an unblocked ancestor cannot have an active descendant.

## Issues

### 1. Major: the round payload budget can starve a registration indefinitely

- **Where:**
  - `cache/tree.rs:1180`: `planning_order` always appends the registering event last.
  - `cache/tree.rs:851`: `Plan::admits` admits an over-budget event only while `self.updates.is_empty()`.
  - `cache/tree.rs:978, 1270`: a registration's payload is its outputs plus `children.bytes`.
  - `state/tree_committer.rs:427-431, 787-812`: the deferred registration goes back to the front of the buffer, and `take_events` takes it again together with every new arrival.
- **Scenario (pure planner, reproducible with the test helpers of `cache::tree::tests`):**
  1. `two_level_tree()`. Job 3 registers `ChildSetSummary { count: 2, bytes: MAX_JOB_CHILD_BYTES }` with one byte of outputs. That passes the limit check (`>` at `:1260`), and its payload is 16 MiB + 1.
  2. Round 1 is `[regular_done(5, vec![vec![5]]), registration(3, …)]`. Job 5's completion is planned first and writes 1 byte. The registration is planned last: 1 + 16 MiB + 1 > `MAX_ROUND_PAYLOAD`, and the plan is not empty, so it is deferred.
  3. Round 2 is `[registration(3) (requeued, index 0), regular_done(4, …)]`. `planning_order` again plans the completion first and the registration last, and the registration is deferred again.
  4. This repeats as long as every round carries at least one writing event of the tree, for example the completions of a wide sibling fan-in (B4) or a busy tree under 64 concurrent reporters.
  5. Meanwhile, job 3's `ReportTaskSuccess` RPC waits for its reply indefinitely.
- **Not only at the cap:** a registration of B bytes is deferred whenever the round's other writers carry more than 16 MiB − B − outputs. In a fan-in, 256 completions with 64 KiB of outputs each already fill the budget, so even a small registration loses every round.
- **Why it contradicts the design:**
  - §7.5 rule 3: "taking the oldest one keeps registrations from starving".
  - §7.5 rule 5 and §13: "the first event of a tree state transaction is always admitted".
  - A non-registering event that is deferred for payload is planned first in the next round, so it is admitted. A registration never is, because it is planned last. The slice implements "first event" as "first writer", which a registration can never be while anything else writes.
  - Registrations are unreachable until S4 (`children` is always `None`), but this is S3's planner rule, and S4 inherits it unchanged.
- **Tests:** none cover it. `the_payload_budget_defers_events_but_always_admits_the_first` (`cache/tree.rs:2257`) uses plain completions only. The property test keeps child bytes below 4 KiB, except for the over-limit case, and it does not check liveness.
- **Fix:** keep "registration last", which I4 needs, because `abort_rest` and `plan_cancel` cannot see the round's new children. Make a deferred registration win the next round. For example:
  - `take_events` takes only the oldest cancel plus a registering event that is already collected (`input.is_some()`, so it was deferred) when that event is at the front of the buffer; or
  - `plan_events` stops admitting other writers once the round holds a registration that had to be deferred before.
- **Test to add:** the two-round scenario above, asserting that the registration plans in round 2.

### 2. Minor: `take_events` leaving extra registrations in the buffer is untested, and the planner masks the mutation

- **Where:** `state/tree_committer.rs:796-806`.
- **Problem:** the plan requires later registering events to stay in the buffer uncollected. Nothing tests that.
  - The three grouping tests use plain jobs, with no registering event.
  - The property test does not go through `take_events`.
- **Mutation:** if `take_events` took every registering event, `planning_order` (`cache/tree.rs:1170-1175`) would still defer the extras, so every test would pass. That is the "masked by a second layer" case from the style guide.
- **Effect in S4:** each extra registration would have its outputs read and its prepared children taken by `collect_inputs`, then be deferred while holding them. That costs work and memory but gives no wrong result.
- **Fix:** add a pure unit test of `take_events` on a buffer like `[Reg A, X, Reg B, Cancel]`. Assert that the round is `[Cancel, Reg A, X]` and that the buffer keeps `[Reg B]` with `input == None`.

### 3. Minor: the touch-twice tests drive sequences that no producer can generate

- **Where:**
  - `state/tree_committer.rs:1266` (`an_event_that_would_change_its_job_again_waits_for_the_next_round`): a `TerminalFailure` of a `Ready` root, submitted straight into the channel.
  - `cache/tree.rs:2237`: a commit failure and a `CleanupDone` queued behind the `RegularTasksDone` that makes the job `CommitReady`.
- **Problem:** the slice note itself says that the rule cannot fire for real producers. A `TerminalFailure` needs a task instance, and a task instance needs a `Running` JCB.
  - The style guide asks not to test what the protocol makes unreachable, and to say at the guard that it is unreachable instead.
  - The committer test's name reads as if this could happen in production.
- **Fix:** keep the planner unit test as a check of the safety net. Make that explicit in `Plan::admits`'s docstring, or turn the check into `debug_assert!`, plus the deferral in release builds. Drop the committer-level test, or rename it to say that it injects an impossible event. Record the finding in the report under "traps", as the note already plans.

### 4. Minor: the property test checks `check_invariants` in one direction only, and rarely exercises cancels

- **Where:** `cache/tree.rs:2585-2590, 2463`.
- **One direction:** plan T11 says that "`check_invariants` agrees with the model's verdict". The test only asserts `Ok` after the model's own checks pass. A `check_invariants` that always returned `Ok` would pass T11. Only the seven hand-written corruptions in `check_invariants_reports_each_broken_invariant` catch that.
- **Few cancels:** `CancelRoot` is generated with probability 0.01 per round. The note reports about 15 applied cancels per 3,000 rounds. The coverage counters do not count a cancel that shares its round with a failure or a registration, so the T11 claim for "cancel + registration + completion in one round" rests on one named test.
- **Fix:**
  - After the model check, perturb one random row or counter, and assert that `check_invariants` returns `Err` exactly when the model's checks fail.
  - Raise the cancel rate, or add a coverage counter for "applied cancel with another applied or stale event in the same round" and assert that it is non-zero.

### 5. Minor: `submit_event`'s `# Errors` is stale after deviation 1

- **Where:** `cache/job.rs:773-774`.
- **Problem:** the docstring says that `StaleState` comes from a lost race "or if the tree's committer has exited". After S3, `TreeClosed` also answers an event whose job left the tree in an earlier round while the committer is still running. The slice updated `TreeOutcome::TreeClosed`'s docstring (`cache/tree.rs:163-165`) but not this one.
- **Fix:** add "or if its job is no longer in the tree".

### 6. Minor (style)

- **`impl` blocks not right after their types** (review guide, "Module structure"):
  - `struct Violation` is at `cache/tree.rs:601`, but `impl Violation` is at `:644`, after `check_invariants` and a private const.
  - `struct Plan` is at `:500`, but `impl Plan` is at `:829`, after `plan_events`, `Violation`, `check_invariants` and `InvariantChecker`.
  - `pub(crate) fn plan_events` (`:558`) sits among the `pub` items.
  - The private const `MAX_PROPAGATED_MESSAGE_CHARS` (`:642`) sits between `pub` items.
- **Missing docstring:** `Overlay::parent` (`cache/tree.rs:1074`) returns a non-trivial `Option<JobId>` without `# Returns`.
- **Tolerated impossible branch:** `InvariantChecker::check_counters` (`cache/tree.rs:807-809`) iterates `structure.sorted_job_ids()` and then `continue`s if `structure.node(job_id)` is `None`. That branch cannot happen. Use `expect`, as `check_job` does at `:721-724` (style guide: "Assert preconditions instead of tolerating them").

## Notes (no change needed in this slice)

- **The debug invariant check never ran on the integration tests.**
  - `assert_invariants` runs only with `debug_assertions`, and CI builds `--release`.
  - The note's debug run covered `--lib` only. The MariaDB integration tests (`tests/job_tree_test.rs`, `tests/tree_transition_test.rs`) have never run the check.
  - In S3 those tests hold only one-job trees, so little is lost. From S4 on, R8's "one debug-build run of the S3–S6 tests" should include the integration tests.
- **N4 cost.** A registering `RegularTasksDone` reads every job's state (deviation 2). `abort_rest` and `plan_cancel` sort every job id per call. For a 10,000-job tree that is about 0.3 ms of read locks plus an O(N log N) sort per registration round. That is fine for the design's limits, and B4 and B5 should report it.
- **Housekeeping.** Pleiades asked that the kind cluster be killed. I checked, and nothing was left to kill: `clp-rc2-control-plane` was already `Exited (137)`, and nothing was started. I did not delete the cluster.

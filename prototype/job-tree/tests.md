# Job-tree prototype: test results

Each test group appends its own section below.

## T1, T4, T5, T6: completion and failure races (test engineer)

**File:** `components/spider-storage/tests/job_tree_completion_test.rs` (new, included from `tests/test_spider_storage.rs`, feature `test-hooks`). It is self-contained and does not share helpers with `job_tree_race_test.rs`.

**How to run:** `JOB_TREE_RACE_ITERATIONS=1000 MARIADB_PORT=… MARIADB_DATABASE=spider-db SPIDER_STORAGE_DB_USERNAME=spider-user SPIDER_STORAGE_DB_PASSWORD=spider-password MARIADB_ROOT_PASSWORD=spider-password cargo nextest run -p spider-storage --all-features --run-ignored all --release job_tree_completion_test`. The default is 50 iterations, like the other race tests, so CI stays cheap.

**Common to every case:**
- Each test runs on its own `TestEnv` (its own scratch database), on a multi-threaded tokio runtime (4 workers), so that released tasks really run in parallel.
- Every race is made deterministic with the hooks; nothing waits on a sleep. Every hook wait is bounded by 30 s, and every lane drain by 10 s, only so that a broken implementation fails instead of hanging.
- The `race` helper pauses the first RPC's round at a hook point, arms the `EventSubmitted` observer for the second event **after** the pause is reached (so the first event's own submission is not counted), waits for the second event's submission, then releases. Same round = pause at `CommitterAfterReceive`; queued behind = `CommitterBeforeDbTransaction`; commit window = `CommitterAfterCommit`.
- `assert_states` checks each job's state in the cache **and** in its row, that `ended_at` is set exactly when the state is terminal, and that the service has not hit its fatal path (`is_fatal() == false`).
- Every ready, commit and cleanup lane entry the test expects is drained and checked exactly (an unexpected or duplicate entry fails), and each test ends by asserting that all three lanes are empty. A spurious re-publication (for example a straggler's retry) is therefore caught.
- Every iteration ends with no non-terminal row in the tree.
- Outputs are distinct per job and per task (`[tag, index, 0x5a]`), so an output landing in the wrong job or task is detected.

### T1 `two_tasks_failing_one_job_fail_it_once`

Two different tasks (0 and 1) of a two-task job J with `max_num_retry = 0` fail.

| Dimension | Values |
|---|---|
| Job | a plain root job; a leaf J in `R → A → {J, S}` where the sibling S has a cleanup task |
| Second failure | same round; queued behind the first round (before its DB transaction); through phase A in the first round's commit-to-apply window; after the first reply |

Exact asserts:
- The first failure answers `Ok(Failed)`. The second answers `Stale(JobAlreadyTerminated(Failed))` in the three committer cases, and `Stale(JobNoLongerRunning)` (phase A) after the reply.
- Round composition through `Observe@CommitterBeforePlan`: one round holding `[TerminalFailure, TerminalFailure]` for the same-round case; two rounds when the second event is queued in a leaf's tree; one round when it is queued in a root's tree (the root is quiescent after the first round, so its committer exits and answers the queued event without planning it); one round after the reply. `EventSubmitted` saw 2 failure events (1 after the reply, where phase A rejects the second).
- Exactly one DB transaction (`CommitterAfterCommit` hits = 1) and exactly one cache apply per job of the tree (`CommitterAfterApplyJob` per job).
- Error messages of the whole tree, exactly: J has `first failure`; R and A have `descendant job J failed: first failure`; no other row has a message (the second message is stored nowhere).
- States: J, A, R `Failed`; S `CleanupReady` with exactly one cleanup lane entry; S's cleanup then gives `Cancelled`.

Iterations: 1,000 × 2 jobs × 4 interleavings = 8,000 trees. Result: **pass** (1,000 iterations, release, 213 s).

### T4 `sibling_jobs_completing_together_advance_their_parent_once`

`k` one-task children of a parent P complete.

| Dimension | Values |
|---|---|
| k | 2, 8, and 64 (64 on every tenth iteration only: it costs about 30 times as much as 2) |
| Rounds | all last reports in one round; each in its own round; free (all spawned at once) |
| Parent | P is the root; P is the only child of a one-task root R |
| Commit | P without and with a commit task |

Exact asserts:
- Every report answers `Ok(Succeeded)`.
- **Exactly-once advance:** P is applied exactly once (`CommitterAfterApplyJob` on P = 1). One-round case: one round holding `k` `RegularTasksDone` events and one DB transaction; each-round case: `k` DB transactions, and after each non-last report P is still `BlockingOnChildren` in the cache and in its row; free case: between 1 and `k` DB transactions.
- **Counter reaches 0 once:** the counter is not observable from outside, so this rests on three checks that together pin it: the single parent apply above; the committer's `assert_invariants` after every round (compiled in with `test-hooks`), which panics, poisons the tree and sets the fatal flag if a `BlockingOnChildren` parent's counter differs from its number of active children or is 0 (invariant I2); and `climb_after_success`'s `checked_sub`, which panics on a second decrement past 0. Every iteration asserts `is_fatal() == false`.
- **The parent sees every sibling's output:** the committer is paused at `CommitterAfterApplyJob` for P, that is, the moment P advances. At that moment every child is `Succeeded` in its row and in the cache, its stored outputs and its cached outputs equal its own distinct value, and P is in its advanced state in its row and in the cache.
- After: P `Succeeded` (or `CommitReady` with exactly one commit lane entry); R `Succeeded` (applied once) or still `BlockingOnChildren` (not applied) when P has a commit task; P's own stored outputs unchanged. With a commit task, P's commit instance is handed exactly P's own outputs, the commit gives `Succeeded`, R then climbs to `Succeeded`, and P has exactly two applies.

Iterations: 1,000 × (24 combinations for k = 2, 8) + 100 × 12 combinations for k = 64 = 25,200 trees. Result: **pass** (1,000 iterations, release, 1,312 s).

### T4 within a job `sibling_tasks_completing_together_complete_their_job_once`

The design's §10.6 (e) race: the last tasks of one job race on `num_incomplete_tasks`. All `n` tasks (n = 2, 8) of a job J (a root, or the child of a root) are paused at `PhaseAAfterTcbFlip` (after the TCB success, before `fetch_sub`). The test waits until all `n` are paused, asserts that no `RegularTasksDone` was submitted and J is still `Running`, then releases all `n` at once on 4 worker threads.

Exact asserts: exactly one report answers `Succeeded` and `n − 1` answer `Running`; exactly one `RegularTasksDone` submission for J; J's stored and cached outputs are the `n` task outputs in task order; J and R `Succeeded`, R applied once.

Iterations: 1,000 × 2 sizes × 2 positions = 4,000 jobs. Result: **pass** (1,000 iterations, release, 96 s).

### T5 `a_straggler_failure_after_a_success_is_stale`

Task 0 (`max_num_instances = 2`) succeeds from instance `w`, and a straggler then fails task 0.

| Dimension | Values |
|---|---|
| Job | a one-task root (the success completes it); a one-task leaf with a sibling (completes the leaf, not the tree); a two-task root (task 0's success does not complete it) |
| Straggler | the same instance `w`; a second instance of task 0 |
| Window | after the success's TCB flip, before `fetch_sub` (`PhaseAAfterTcbFlip`); the success's round paused before its DB transaction; in the commit-to-apply window (`CommitterAfterCommit`, where the test also asserts that the row is already `Succeeded` while the cache is still `Running`); after the success was answered. A two-task root has no round, so only the first and last windows apply |
| Retries | `max_num_retry` alternates 0 and 1 by iteration, so a straggler that wrongly counted as a failure would show up as a job failure or as a re-published task |

Exact asserts:
- The success answers `Succeeded` (one-task jobs) or `Running` (two-task root).
- The straggler answers exactly: `Stale(JobNoLongerRunning)` after the apply for one-task jobs (phase A); otherwise `Stale(InvalidTaskInstanceId)` for the same instance and `Stale(TaskAlreadyTerminated(Succeeded))` for another instance (the TCB checks).
- In the paused windows the straggler is answered while the success is still paused (bounded by 30 s): it is rejected in phase A and never waits on the committer.
- No `TerminalFailure` event was submitted in the tree; no row has an error message; no task is re-published.
- States in cache and rows: the job `Succeeded`/`Running`, and for the leaf, R `BlockingOnChildren` and the sibling `Running`. The rest of the tree then runs to `Succeeded`, and the job's stored and cached outputs are exactly the winner's output (plus task 1's for the two-task root).

Iterations: 1,000 × 20 cases = 20,000 cases. Result: **pass** (1,000 iterations, release, 505 s).

### T6 `a_parent_with_a_commit_task_commits_then_climbs`

Tree `R → {P, Q}`, P with a commit task and three one-task children, Q a one-task job; R without and with a commit task. Q completes at one of five points: before P's children; in the same round as P's last child; after P is `CommitReady` and before its commit report; in the same round as P's commit success; queued behind P's commit round.

Exact asserts:
- While children complete, P stays `BlockingOnChildren` (cache and row) until the last one.
- After the last child: every child `Succeeded` with its own outputs stored; P `CommitReady` (cache and row, `ended_at` unset) with its own outputs stored; R `BlockingOnChildren`, not applied; P applied exactly once; exactly one commit lane entry for P. For the same-round variant, the last round held exactly `[RegularTasksDone, RegularTasksDone]`.
- P's commit instance's execution context hands exactly P's own outputs (none of its children's).
- The commit answers `Succeeded`, and Q `Succeeded`. Same-round variant: one round `[CommitDone, RegularTasksDone]`; queued-behind variant: two rounds.
- After the commit: P, Q `Succeeded`; R `Succeeded`, or `CommitReady` with exactly one commit lane entry when R has a commit task; P applied exactly twice, Q once, R once. P's cached and stored outputs unchanged by the commit.
- With R's commit task: R's commit context holds exactly R's outputs, the commit gives `Succeeded`, R applied twice. Then `GetJobOutputs(R)` returns exactly R's outputs, and no row of the tree has an error message.

Iterations: 1,000 × 5 timings × 2 root shapes = 10,000 trees. Result: **pass** (1,000 iterations, release, 569 s).

### Other runs

- Debug build (overflow checks and `debug_assertions` on), 100 iterations: all 5 pass.
- The full `spider-storage` suite with `--all-features --run-ignored all --release` (default 50 iterations) on container `jtp-test-t1456` (stopped at the end): **270/270** on the third run. The first two runs each failed one test of the timing-flake family that the S1, S4 and S6 notes already list (`mariadb_test::test_delete_expired_terminated_jobs`: a concurrent test's expiry delete on the shared `spider-db` removed its job first; `mariadb_test::test_is_execution_manager_alive_em_dead`: 1 s thresholds on the shared `execution_managers` table). Both pass when run alone; neither touches code changed here.
- `cargo build --workspace --all-targets --all-features`, `cargo clippy --workspace --all-targets --all-features` and `cargo clippy -p spider-storage --all-targets` (default features): clean. `rustfmt +nightly --check` on the changed files: clean.

### Mutation checks

Each defect was applied in a separate copy of the worktree (never in the shared worktree), the targeted test was run with a few iterations, and the copy was restored. All seven are caught by the final version of the tests.

| Defect | Targets | Caught by |
|---|---|---|
| M1 `plan_failure` skips its lane check | T1 | T1 same round (leaf): two rounds instead of one. The defect is otherwise masked: the second failure is deferred by the "never touch a row twice" rule and then finds J `Absent`, so it is still answered stale. Only the round composition sees it |
| M2 apply order root first instead of deepest first | T4 visibility | T4: at P's apply, a child is still `Running` in the cache |
| M3 the overlay ignores in-round counter decrements | T4 counter | T4 one round: the committer's I2 check panics (`0 active children, counter 1`), the service turns fatal, the advance never comes |
| M4 `fetch_sub` followed by a separate `load` to decide who is last | T4 within a job | two tasks both see 0, the second `RegularTasksDone` answers `Stale(JobAlreadyTerminated(Succeeded))` |
| M5 a TCB failure after success overrides it | T5 | the straggler is not answered while the success is paused (it now waits on the committer instead of being rejected in phase A) |
| M6 a commit success does not climb | T6 | I2 panic and fatal shutdown; R's cached state wrong |
| M7 `unblock` does not publish the commit task | T6 | P's commit lane entry never arrives |

### Bugs found and fixed

No defect was found in the storage logic of T1, T4, T5 or T6.

One implementation fix was needed to keep clippy green, unrelated to these cases' behaviour:
- `state/tree_committer.rs`, `apply_plan`: `round_context: impl Fn() -> HookContext` → `impl Fn() -> HookContext + Send + Sync`. With the repository's `-Dclippy::nursery` (`.cargo/config.toml`), `clippy::future_not_send` rejected the library (`cargo clippy -p spider-storage --all-targets --all-features` failed in the lib before any test was linted).

### Findings for the other test files and the formal implementation

1. **Test environment: the execution manager goes stale after 60 s.** `TestEnv` registers an execution manager that never heartbeats, and the default `TaskInstancePoolConfig` (`execution_manager_stale_cutoff_sec = 60`, GC every 30 s) then declares it dead: from about 90 s into a test, the pool force-removes the test's instances (their reports answer `InvalidTaskInstanceId`) and re-enqueues their tasks. My first 1,000-iteration run failed three tests at exactly 90 s this way. My tests start their environment with a 24 h cutoff. Every other `TestEnv::start` test uses the default; the short ones are unaffected, but `job_tree_race_test.rs` will fail the same way once one of its tests runs past about 90 s (its T4 test took 87 s at 500 iterations, so 1,000 iterations would cross it). Suggested fix: give `TestEnv::start` a long cutoff, or heartbeat from `TestEnv`.
2. **`job_tree_race_test.rs::race` can miss the same-round case of its T1.** It arms the `EventSubmitted` observer before spawning the first RPC, with a filter that the first failure's own event also matches (same job, same kind). `reached(1)` can then return on the first event, and the pause may be released before the second failure is submitted, so the "same round" iteration may run as two rounds. Its asserts hold in both cases, so it passes, but it does not guarantee the coverage it names. Its other callers (cancel, commit) filter a different event and are not affected. My `race` arms the observer after the pause is reached, and T1 here asserts the round composition.
3. **The planner's lane check in `plan_failure` is shadowed** in the same-job case by the twice-touch deferral and the `Absent` handling (M1). It remains the only guard for a failure whose job changed state without becoming terminal in the same round, so it should stay; a planner unit test for that case would pin it directly.
4. **Stale variants depend on the path,** as the S5 notes say: phase A answers `JobNoLongerRunning`, the planner and the closed-tree path answer `JobAlreadyTerminated(state)`, and the TCB answers `InvalidTaskInstanceId` (same instance) or `TaskAlreadyTerminated(Succeeded)` (another instance). All are `FAILED_PRECONDITION`; the tests pin each one exactly.

## T2, T3, T7: root cancel races (test engineer)

**File:** `components/spider-storage/tests/job_tree_cancel_test.rs` (new, included from `tests/test_spider_storage.rs`, feature `test-hooks`). Self-contained; it shares no helpers with `job_tree_race_test.rs` (whose S5 T2/T3/T7 tests stay as they are) or `job_tree_completion_test.rs`.

**How to run:** `JOB_TREE_RACE_ITERATIONS=1000 MARIADB_PORT=… MARIADB_DATABASE=spider-db SPIDER_STORAGE_DB_USERNAME=spider-user SPIDER_STORAGE_DB_PASSWORD=spider-password MARIADB_ROOT_PASSWORD=spider-password cargo nextest run -p spider-storage --all-features --run-ignored all --release job_tree_cancel_test`. Default 50 iterations.

**Common to every case:**
- Own `TestEnv` per test (scratch database), multi-threaded tokio runtime (4 workers), execution-manager stale cutoff 24 h (finding 1 of the T1/T4/T5/T6 section).
- Every interleaving is forced with hooks; no sleeps. A `Race` pauses the first RPC's round at `CommitterAfterReceive` (same round), `CommitterBeforeDbTransaction` (second event queued behind the in-flight round) or `CommitterAfterCommit` (second RPC's phase A runs in the commit-to-apply window), arms the `EventSubmitted` observer for the second event only after the pause is reached, waits for the submission, then releases. Phase-A interleavings use `PhaseAAfterTcbFlip` and `PhaseABeforeSubmit`. Hook waits are bounded by 30 s and lane drains by 10 s, so a broken implementation fails instead of hanging.
- **Commit snapshot:** every case in which the cancel (or, in T3, the winning failure) commits pauses that round at `CommitterAfterCommit`, reads all rows of the tree, then releases. The rows at that instant must equal the exact expected post-race states. This is the "never a live child under a closed tree" check at the moment of the commit, not only at the end.
- **Exact outcomes:** each RPC's result is asserted to the variant (`Ok(state)` or the exact `StaleStateError`, all `FAILED_PRECONDITION`).
- **Exact committer activity:** number of rounds and the event kinds of the last round in planning order (`Observe@CommitterBeforePlan`), number of DB transactions (`CommitterAfterCommit`), cache applies per job and in total for the tree (`CommitterAfterApplyJob`), and the number of submitted events of each racing kind (`EventSubmitted`).
- **Final checks (`finish_tree`):** every job's state in the cache and in its row, `ended_at` set exactly for terminal states, the tree's rows are exactly the expected jobs (no stray inserts), the error messages of the whole tree are exactly the expected ones, `GetJobState(root)` is right, one more `CancelJob` gets the exact stale error (`JobCancellationAlreadyRequested`, `JobAlreadyCancelled` or `JobAlreadyTerminated(state)`), `is_fatal() == false`. Then the cleanup lane must hold exactly the `CleanupReady` jobs (an entry for any other job, such as a skipped `CommitReady` job, fails the drain), each cleanup is run and answers `Cancelled`, and the tree must be quiescent. Each test ends asserting that all three lanes are empty.
- **Cleanup modes** select whether the committer exits at once after the race (`None`: every deactivated job is `Cancelled`, so late events take the closed-tree path), stays alive with the raced job still in the tree (`All`: the raced job is `CleanupReady`, its late event is planned stale), or stays alive while the raced job has left the tree (`BystandersOnly` / `RootOnly`: the late event finds its job `Absent`, `TreeOutcome::Untracked`). The same race therefore goes through all three answer paths.

### T2 `a_root_cancel_racing_a_registration_never_leaves_a_live_child`

The registering job (one task, reporting 3 children; the second child never has a cleanup task) is a leaf `L` of `R → {L, S}`, or the root itself.

| Dimension | Values |
|---|---|
| Registrar, cleanups | leaf × {None, All, BystandersOnly}; root × {None, All} |
| Cancel planned first | same round (registration arrives first); same round (cancel arrives first); registration queued behind the in-flight cancel round; registration's phase A in the cancel's commit-to-apply window; phase A paused after the TCB success holding `L`'s read guard while the cancel commits (its apply must wait); phase A paused after the guard drop, before the submit, while the cancel completes; registration reported after the cancel's reply |
| Registration planned first | cancel queued behind the in-flight registration round; cancel submitted in the registration's commit-to-apply window; cancel after the registration's reply |

Exact asserts:
- The cancel answers `Ok(CleanupReady | Cancelled)` (the root's cleanup flag).
- Cancel first: the registration answers `JobNoLongerRunning` after the cancel's reply (phase A); otherwise `JobAlreadyTerminated(Cancelled)` or `JobNoLongerRunning` by the registering job's post-cancel state. `COUNT(parent_id = L) = 0`; no task-lane entry for any child.
- Registration first: the registration answers `BlockingOnChildren`; exactly 3 child rows; their 3 ready entries are published once; each child is deactivated by the cancel to its own target (`CleanupReady` with a cleanup task, else `Cancelled`), and creating an instance for each published child task answers `JobNoLongerRunning`.
- Inside the windows: in the cancel's commit window the rows are already deactivated while the cache still shows every job in its pre-race state; in the phase-A-held case the rows are deactivated and the cancel RPC has not finished; in the registration's commit window the 3 children's rows exist as `Running` under a `BlockingOnChildren` row, are not cached, and the registering job is still `Running` in the cache.
- Rounds: same round = 1 round `[CancelRoot, RegularTasksDone]`; cancel first otherwise = 1 round `[CancelRoot]` when the tree is quiescent after the cancel, else 2 rounds ending `[RegularTasksDone]`; registration first = 2 rounds ending `[CancelRoot]`. DB transactions: 1 when the cancel is first, 2 otherwise. Applies: 1 per job, the registering job 2 when it registered; total `jobs` or `jobs + 1 + 3`.
- Submitted registrations: 0 after the cancel's reply, 0 when the phase A submits to an exited committer, ≤ 1 in the phase-A-held case on a quiescent tree (the submit races the committer's exit; the answer is the same either way), else 1.

Iterations: 1,000 × 50 trees. Result: **pass** (release, 3,424 s; see the note on run time below).

### T2 variant `a_root_cancel_racing_two_registrations_never_leaves_a_live_child`

Both leaves of `R → {L, S}` register 3 children, with the root's cancel. Cleanups None, All, RootOnly. Interleavings: `L`, `S`, cancel all before one round (the round takes the cancel and `L`'s registration, `S`'s waits in the buffer); `S` and the cancel queued behind `L`'s in-flight registration round; `S`'s phase A and the cancel in `L`'s commit-to-apply window.

Exact asserts: same round: both registrations stale (by the leaves' post-cancel state), no child rows, round `[CancelRoot, RegularTasksDone]` and a second round `[RegularTasksDone]` only when the committer outlives the cancel, 1 DB transaction. `L` first: `L` answers `BlockingOnChildren` with 3 children, then one round `[CancelRoot, RegularTasksDone]` cancels the tree and `L`'s children and makes `S` stale; `S` has no child rows; 2 DB transactions; applies `L` 2, others 1. Plus the common checks.

Iterations: 1,000 × 9 trees. Result: **pass** (release, 301 s).

### T3 `a_root_cancel_racing_a_leaf_failure_lets_the_first_planned_win`

Tree `R → {A → {L, S}, Q}`; `L`'s only task fails with no retry left. Cleanups None, All, BystandersOnly (`S`, `Q`).

| Winner | Interleavings |
|---|---|
| Cancel | same round, failure arrives first; same round, cancel arrives first; failure queued behind the in-flight cancel round; failure's phase A in the cancel's commit window; failure after the cancel's reply |
| Failure | cancel queued behind the in-flight failure round; cancel in the failure's commit window; cancel after the failure's reply |

Exact asserts:
- Cancel wins: cancel `Ok(target(R))`; failure stale (`JobNoLongerRunning` after the reply, else by `L`'s post-cancel state); all 5 jobs at their abort targets in one DB transaction, failing job included; **no error message anywhere in the tree** (the failure is not stored); every cleanup runs.
- Failure wins: failure `Ok(Failed)`; cancel `JobAlreadyTerminated(Failed)` (planned against an `Absent` root, or through the closed tree); `L`, `A`, `R` `Failed`; `S`, `Q` at their abort targets; errors exactly `L: leaf failure`, `A`, `R: descendant job L failed: leaf failure`.
- One DB transaction in every case (the loser writes nothing); 1 apply per job; rounds `[CancelRoot, TerminalFailure]` for the same round, otherwise the loser's event gets a second round only when the committer outlives the winner. Submitted failures 0 only after the cancel's reply; submitted cancels 0 only after the failure's reply on a quiescent tree.

Iterations: 1,000 × 24 trees. Result: **pass** (release, 1,682 s).

### T7 non-root `a_cancel_skips_a_commit_ready_non_root_job_which_then_finishes`

Tree `R → {P → C, Q}`, `P` with a commit task, `C` succeeded so `P` is `CommitReady`; all jobs but `C` with or without a cleanup task. The commit-first races also run without `Q`.

Interleavings: commit instance registered, cancel answered, then commit success / commit failure; cancel answered, then the commit instance registered (it must be accepted) and succeeds; commit success and cancel in one round (both arrival orders); commit failure and cancel in one round; commit queued behind the in-flight cancel round; commit's phase A in the cancel's commit window; cancel queued behind the in-flight commit round; cancel after the commit's reply.

Exact asserts:
- After a cancel reply that precedes the commit: `R` and `Q` deactivated, `P` still `CommitReady` (cache and row, `ended_at` NULL), `C` `Succeeded`.
- Cancel planned first: cancel `Ok(target)`; the commit `Ok(Succeeded)` or `Ok(Failed)`; **no propagation**: `R` keeps the cancel's state and exactly 1 apply, `R` has no error message, the only error in the tree is `P: commit failure` for the failure variants; `P` never gets a cleanup entry, even with a cleanup task. The snapshot at the cancel's commit shows `P` `CommitReady` (or already final in the same-round cases).
- Commit planned first: with `Q`, `P` `Succeeded`, `R` decremented but still `BlockingOnChildren`, then the cancel deactivates `R` and `Q`; without `Q`, `R` climbs to `Succeeded` and the cancel answers `JobAlreadyTerminated(Succeeded)`.
- Rounds and DB transactions: 1 round `[CancelRoot, CommitDone | TerminalFailure]` for the same-round cases, else 2 (the second ending with the commit event, or `[CancelRoot]` when the commit is first), except 1 round when the commit is first and the tree has no `Q`.

Iterations: 1,000 × 24 trees. Result: **pass** (release, 1,467 s).

### T7 root `a_cancel_deactivates_a_commit_ready_root`

A one-job root with a commit task, `CommitReady`, with and without a cleanup task. Interleavings: cancel answered, then the late commit success, the late commit failure, or a new commit instance; commit and cancel in one round (both orders); commit queued behind the in-flight cancel round; cancel queued behind the in-flight commit round; cancel after the commit's reply.

Exact asserts: cancel first: `Ok(CleanupReady)`, or `Ok(Cancelled)` through the `CommitReady → Cancelled` edge (row `Cancelled` with `ended_at`); the late commit, late commit failure and new instance answer `JobNoLongerCommitReady` (phase A); a commit planned in or after the cancel's round answers `JobNoLongerCommitReady` (root `CleanupReady`) or `JobAlreadyTerminated(Cancelled)`; the failure stores no message. Commit first: `Ok(Succeeded)`, the cancel `JobAlreadyTerminated(Succeeded)`, and `GetJobOutputs(root)` returns the root's outputs. Exactly 1 DB transaction and 1 apply; rounds as above.

Iterations: 1,000 × 16 trees. Result: **pass** (release, 442 s).

### Other runs

- Debug build (`debug_assertions`, overflow checks), 100 iterations: all 5 tests pass.
- Full `spider-storage` suite, `--all-features --run-ignored all --release`, default 50 iterations, container `jtp-test-t237` (stopped at the end): **275/275** on two runs. Two other full runs each failed only `mariadb_test::test_is_execution_manager_alive_em_dead`, the known shared-`spider-db` timing flake (1 s staleness threshold); it passes alone, and it touches neither job trees nor anything changed here.
- `cargo build --workspace --all-targets --all-features`, `cargo clippy --workspace --all-targets --all-features -- -D warnings`, `cargo clippy -p spider-storage --all-targets -- -D warnings`: clean. `rustfmt +nightly --check` on the changed files: clean.

### Mutation checks

Each defect was applied to a copy of the worktree with its own target directory (never to the shared worktree), the tests were run with 2-3 iterations, and the file was restored. All 14 defects are caught.

| Defect | Caught by |
|---|---|
| `take_events` does not put the cancel first (the planner's `RoundOrder` still does) | all four original tests: round composition `[X, CancelRoot]` instead of `[CancelRoot, X]`. Outcomes are unchanged because the planner orders the round itself, so only the round assertion sees this layer |
| Neither `take_events` nor `RoundOrder` puts the cancel first | T3 same round: the cancel answers `JobAlreadyTerminated(Failed)`; T7 root: `JobAlreadyTerminated(Succeeded)`; T2, T7 non-root: round composition |
| `RoundOrder` plans the registration before the cancel | T2 same round: registration answers `Ok(BlockingOnChildren)` |
| `is_abortable` also aborts `CommitReady` | T7 non-root: `P` `Cancelled` after the cancel |
| `plan_cancel` deactivates only the root | T2, T3, T7 non-root (an abort later panics the planner: fatal shutdown) |
| `plan_cancel` skips a `CommitReady` root | T7 root: the late commit succeeds |
| `collect_regular_done_inputs` reads a non-`Running` job | T2 (All, cancel in flight): `TaskInputNotReady`, fatal shutdown (trap R3) |
| Committer quiescent when the root alone is terminal | T2, T3, T7 non-root: cleanups and skipped commits of the remaining jobs answered stale |
| A skipped job's failure propagates to any parent | T7 non-root: planner panic, `TreePoisoned` |
| `CancelRoot` stale mapping of `CleanupReady` wrong | the one-more-cancel check in all tests |
| `Untracked` mapped to `JobNoLongerRunning` | all tests (closed-tree and `Absent` paths) |
| No cleanup publication for deactivated non-roots | T2, T3, T7 non-root: cleanup drain times out |
| Registered children not added to `TreeStructure` | T2: the committer's I2 check panics, fatal |
| `take_events` takes every registering event | T2 two registrations: round `[CancelRoot, RTD, RTD]` |
| (control) the unmutated copy | passes |

### Bugs found and fixed

None. No implementation change was needed for T2, T3 or T7; the only worktree edits are the new test file and its `mod` line in `tests/test_spider_storage.rs`.

### Findings

1. **Stale variants depend on the answer path,** as the earlier sections note. For a late regular-lane event after a cancel: `JobNoLongerRunning` from phase A or when the job is still in the tree as `CleanupReady`; `JobAlreadyTerminated(Cancelled)` when the job is `Cancelled` (planned, `Absent`, or closed tree). For a late commit on a cancelled root: phase A answers `JobNoLongerCommitReady` for both `CleanupReady` and `Cancelled`, the planner and the closed-tree path answer `JobAlreadyTerminated(Cancelled)` for `Cancelled`. All are `FAILED_PRECONDITION`, which is what the design promises; the tests pin each one.
2. **A submit to an exited committer is invisible to `EventSubmitted`.** `TreeControlBlock::submit` observes the hook only on a successful send. A test that counts submissions must allow 0 when the tree became quiescent first. In the phase-A-held T2 case on a quiescent tree, whether the late registration reaches the committer before it exits is a real (and harmless) race; both paths give the same answer.
3. **`root_id` has no index.** The tests' tree queries (`WHERE id = ? OR root_id = ?`) scan `jobs`, so a long run that accumulates about 150,000 rows in one scratch database slows down (T2 took 3,424 s instead of the roughly 1,300 s that its first iterations suggested). No production query filters on `root_id` today, but the plan's B3 quiescence probe does: it is fine on a fresh database per trial, and the formal implementation should add an index if anything starts reading a tree by `root_id`.

## T8, T9, T10: restarts at crash points, dropped RPCs, limits (test engineer)

**File:** `components/spider-storage/tests/job_tree_fault_test.rs` (new, 4,028 lines, included from `tests/test_spider_storage.rs`, feature `test-hooks`). Self-contained: it shares no helpers with the other `job_tree_*` files and leaves `job_tree_recovery_test.rs` (the S6 single-run T8 tests), `job_tree_test.rs` (the S2 T9 tests) and `registration_test.rs` (the S4 T10 tests) as they are.

**How to run:** `JOB_TREE_RACE_ITERATIONS=1000 MARIADB_PORT=… MARIADB_DATABASE=spider-db SPIDER_STORAGE_DB_USERNAME=spider-user SPIDER_STORAGE_DB_PASSWORD=spider-password MARIADB_ROOT_PASSWORD=spider-password cargo nextest run -p spider-storage --all-features --run-ignored all --release job_tree_fault_test`. Default 50 iterations. The three tests whose interleavings are all forced and sequential (the limit cases and the depth chain) run a tenth of that, at least 5.

**Common to every case:**
- Own `TestEnv` per test (scratch database), multi-threaded tokio runtime (4 workers), execution-manager stale cutoff 24 h. No sleeps: every interleaving is forced with hooks, and every hook wait (30 s) and lane drain (10 s) is bounded only so that a broken implementation fails instead of hanging.
- **Exact lanes without timing:** `assert_lanes_hold(expected)` registers a sentinel root job with a commit and a cleanup task and walks it through the three lanes (start → task entry; success → commit entry; cancel → cleanup entry; cleanup → `Cancelled`). Each lane is polled up to its sentinel, and the entries before it must be exactly `expected` (multiset), with nothing after it. The publisher and the lanes are FIFO, so this proves the exact lane contents at that moment, not "nothing arrived within a timeout". Every case ends with `assert_lanes_hold(&[])`.
- `assert_tree` checks the tree's rows are exactly the expected jobs (no stray inserts), each in its state; `ended_at` set exactly for terminal states; every non-terminal job cached in its state; after a restart every terminal job **not** cached; and `is_fatal() == false`.
- **A crash** is a `Crash` hook at a committer point. The crashed tree's callers must get exactly `Stopping` or `Internal(TreePoisoned)` (which of the two is a real race: the committer drops the round's reply channels before it cancels the token, and `spawn_body` and `submit_and_wait` both race the token), and the service must shut down **fatally** (the test waits for the token, then checks the flag; see finding 1).
- **A restart** stops the crashed runtime and starts a new one on the same database with `Observe` hooks at `RecoveryBeforeRepair`/`RecoveryAfterRepair`, then resends the ready tasks as the binary does. Restarts take turns between three modes: direct; a first start that crashes at `RecoveryBeforeRepair` for the tree (and must fail) before the real start; the same at `RecoveryAfterRepair`. Every recovery must load **exactly** the tree's non-terminal jobs and repair **nothing** (each tree state transaction is atomic, so a crash never leaves a state that needs repair).

### T8 `a_registration_crashed_at_any_point_registers_its_children_exactly_once`

A job registers 3 children; the committer crashes in that round.

| Dimension | Values |
|---|---|
| Registrar | the root itself; a leaf `L` of `R → {L, S}`; `L` with `S`'s success forced into the same round (same-round recipe) |
| Crash point | before COMMIT: `CommitterAfterReceive` (not with `S`), `CommitterBeforePlan`, `CommitterBeforeDbTransaction`; after COMMIT: `CommitterAfterCommit`, `CommitterAfterApplyJob` after `L`'s apply, after `S`'s apply (with `S`) |
| Root | without / with a commit task (alternating iterations) |
| Restart | direct / crash in recovery before repair / after repair (in turn) |
| Bystander | a separate one-job tree with a registered instance, live across the crash |

Exact asserts:
- Both callers see the crash; the hook fired once; the service shut down fatally.
- Rows at the crash: before COMMIT, `L` `Running`, no child row, `S` `Running`; after COMMIT, `L` `BlockingOnChildren` with exactly 3 `Running` children (and `S` `Succeeded` when it shared the round).
- Recovery loaded exactly the non-terminal jobs, repaired nothing; cache = rows; the recovered blocked parent's stored outputs are its own; every child row has `parent_id = L`, `root_id = R`.
- **First lanes after the restart, exactly:** before COMMIT `L`'s task (+ `S`'s); after COMMIT the 3 children's tasks (+ `S`'s if not done) and **nothing for the blocked parent**; plus the bystander's task. The bystander's pre-crash instance is `InvalidTaskInstanceId`; a new one succeeds.
- **No double registration:** the execution manager's retry of `L`'s original report (same instance, same 3 children) is `StaleSession(new session)` in the old session, and in the new session `InvalidTaskInstanceId` (before COMMIT) or `JobNoLongerRunning` (after COMMIT); `S`'s retried report after a committed round is `JobNotFound(S)` (terminal, not recovered). The child rows are unchanged by the retries.
- Then: before COMMIT, a rerun registers exactly 3 children; the children succeed one by one, `L` stays `BlockingOnChildren` until the last (**the recovered blocked parent unblocks**); the root climbs (`CommitReady` → commit → `Succeeded` with a commit task). End: every job `Succeeded`, exactly 3 children of `L`, no error message, lanes empty, quiescent.

Iterations: 1,000 × 15 trees = 15,000 crashes and restarts (about 10,000 of them with an extra crashed recovery). Result: **pass** (release, 2,969 s).

### T8 `a_cascade_crashed_midway_resumes_from_what_it_committed`

Tree `R → {P → {L}, Q}` (`Q` with a cleanup task). `L`'s only task succeeds (success cascade) or fails with no retry (failure: `L`, `P`, `R` `Failed`, `Q` aborted). `Q` running or already succeeded; `R` without/with a commit task.

Crash points: `CommitterBeforeDbTransaction`, `CommitterAfterCommit`, and `CommitterAfterApplyJob` after each job the round applies (`L`, `P`, plus `R` when the cascade reaches it, plus `Q` when the failure aborts it): 20 trees per iteration.

Exact asserts: rows at the crash = the full post-cascade states once committed (whatever the cache had applied), else unchanged; recovery loads exactly the non-terminal jobs (none when the tree is done; a direct restart then) and repairs nothing; the first lanes are exactly `Q`'s task, `R`'s commit or `Q`'s cleanup as the states call for; `L`'s retried report is `JobNotFound(L)` after COMMIT and `InvalidTaskInstanceId` before it, where a rerun then redoes the cascade with its exact publications. **A recovered `BlockingOnChildren` root (counter rebuilt from the rows) unblocks** when `Q` succeeds. Errors exactly `L: leaf failure`, `P`/`R: descendant job L failed: leaf failure`; final states, lanes empty, quiescent, `GetJobState(R)`.

Iterations: 1,000 × 20 = 20,000. Result: **pass** (release, 3,154 s).

### T8 `a_cancel_crashed_at_any_point_cancels_the_tree_once`

Tree `R → {P → {L}, K, M}`: `R`, `P`, `L` with cleanup tasks, `K` `CommitReady` (skipped by the cancel), `M` plain. Crash points: `CommitterBeforeDbTransaction`, `CommitterAfterCommit`, after the apply of `L`, `P`, `M`, `R`.

Exact asserts: after COMMIT the rows are `R`/`P`/`L` `CleanupReady`, `M` `Cancelled`, `K` `CommitReady`; recovery loads exactly the non-terminal jobs, no repair; the first lanes are exactly the three cleanups and `K`'s commit (before COMMIT: `L`'s and `M`'s tasks and `K`'s commit); the client's retried cancel is `JobCancellationAlreadyRequested` after COMMIT, or `Ok(CleanupReady)` with exactly the three cleanups published before it. `K`'s commit then succeeds **without propagating** (`R` stays `CleanupReady`), the cleanups give `Cancelled`, one more cancel is `JobAlreadyCancelled`, no error message, lanes empty, quiescent.

Iterations: 1,000 × 6 = 6,000. Result: **pass** (release, 1,372 s).

### T8 zombie `a_superseded_runtime_never_registers_children_twice`

`L`'s registration in `R → {L, S}` is paused in the old runtime, a successor runtime starts on the same database (fence bump, recovery, resend), then the old one is released.
- Paused before its DB transaction: the old write is `Fenced` → the caller sees the crash, the old runtime shuts down fatally; the successor has `L` `Running` and **no child row**, publishes exactly `L`'s and `S`'s tasks, and a rerun registers exactly 3 children.
- Paused after its COMMIT: the old caller gets `Ok(BlockingOnChildren)` (committed), the old runtime is not fatal until it writes again; `S`'s report on the old runtime then crashes it (fenced); the successor recovered `L` blocked with exactly the 3 children and publishes exactly their tasks and `S`'s.
- Both: the tree finishes on the successor with exactly 3 children of `L`, lanes empty, quiescent.

Iterations: 1,000 × 2. Result: **pass** (release, 460 s).

### T9 `a_dropped_rpc_never_strands_its_job`

The RPC's task is aborted (`JoinError::is_cancelled`) while its work is paused at a hook; the hook is then released and nothing repeats the RPC. `J` alone, or a leaf of `R → {J, S}` (`S` with a cleanup task). 18 cases × 2 positions (start: root only) = 35 trees per iteration:

| Case | Drop point | Exact asserts beyond the common end state |
|---|---|---|
| last report | `PhaseAAfterTcbFlip` (between the TCB success and `fetch_sub`), `PhaseABeforeSubmit` | no submit while paused, `J` still `Running`; then `J` applied once, `Succeeded`, outputs stored |
| registering report | `PhaseAAfterTcbFlip` | no child row while paused; then `J` `BlockingOnChildren` with exactly 3 children, their tasks published once; `J` unblocks |
| task 0 of 2, task 1 reports in full meanwhile | `PhaseAAfterTcbFlip` | task 1 answers `Running`, no `RegularTasksDone` submitted; after the release the **dropped** report is the last decrement and submits: `J` `Succeeded` with outputs `[t0, t1]` |
| both tasks of 2 | `PhaseAAfterTcbFlip` × 2 | exactly one `RegularTasksDone` submitted, `J` `Succeeded` |
| last report vs. a root cancel | `PhaseAAfterTcbFlip` holding `J`'s guard while the cancel commits (its apply must wait) | the cancel answers `Ok(Cancelled/CleanupReady)` only after the release; the late completion is submitted once and planned (stale), `J` applied once, no outputs stored; cleanups run |
| first task of a chain | `PhaseABeforeSuccessorLock` | the successor published exactly once; outputs `[t1]` |
| last report, event in the committer | `CommitterAfterReceive`, `CommitterBeforePlan`, `CommitterBeforeDbTransaction`, `CommitterAfterCommit` | `J` applied, not fatal (the reply goes to a waiter of the spawned body) |
| terminal failure / commit success / cleanup success / root cancel / root start, event in the committer | `CommitterBeforeDbTransaction` | failure: `J` and `R` `Failed`, exact errors, `S` aborted and cleaned up; commit → `Succeeded`; cleanup → `Cancelled`; cancel → tree cancelled; start → `Running` and its task published |
| instance creation | `CreateInstanceBeforePoolSend` | the dropped creation's instance reports successfully |
| instance creation for a **dead** execution manager | `CreateInstanceBeforePoolSend` | proves the pool send happened despite the drop, without timing: the pool finds the EM dead, removes the instance and republishes the task exactly once; the removed instance's report is `InvalidTaskInstanceId`; a new instance succeeds |

Every case: final states of the whole tree (rows, cache, `ended_at`), `J`'s stored outputs exactly, error messages exactly, the number of `RegularTasksDone` submissions of `J` exactly (1, or 0 where `J`'s task never ran), lanes empty, quiescent, `GetJobState(root)`, not fatal.

Iterations: 1,000 × 35 = 35,000. Result: **pass** (release, 2,533 s).

### T10 `a_report_over_a_limit_fails_its_job_and_tree_but_not_the_service`

`J`'s only task reports over a limit, on every allowed attempt (`max_num_retry` 0 or 1), alone or as a leaf of `R → {J, S}`, with a bystander root running alongside. Cases, each with its exact message: 10,001 children (`10001 child jobs exceed the limit of 10000 per report`); a child one byte over the statement payload (`child job 1 of … bytes exceeds the limit of … bytes`); children over 16 MiB per report; an undecodable child (`child job 1 is invalid: …`, prefix checked, then the exact stored text propagated); outputs over 4 MiB (`task outputs of … bytes exceed the limit of 4194304 bytes`); the child insert rejected by the database for its payload (`FailDb { deterministic_payload: true }`: `child payload rejected by the database`, a job failure on the first attempt regardless of retries).

Exact asserts: every attempt but the last answers `Ok(Running)` and republishes the task exactly once; the last answers `Ok(Failed)` (a failure, not an RPC error); no child row after any attempt; `J` `Failed` with exactly that message, `R` `Failed` with `descendant job J failed: <message>`, `S` `CleanupReady` then `Cancelled`; no outputs stored; the tree has no other row; the bystander then succeeds; lanes empty, quiescent, not fatal.

Iterations: 100 × 24 = 2,400 (sequential cases). Result: **pass** (release, 280 s).

### T10 `reports_racing_over_the_job_child_limit_fail_one_task_only`

The two tasks of one job each report 3 valid children of about 3.4 MiB (each report within 16 MiB, together over the 16 MiB per job). Races: free (both at once on 4 threads); the first held at `PhaseAAfterTcbFlip` (reserved, holding the guard) while the second is rejected and, with no retry, its failure commits and waits to apply; sequential. `max_num_retry` 0 and 1.

Exact asserts: exactly one report is rejected (the loser): with no retry it answers `Ok(Failed)` and the winner `Ok(Running)` (or, in the free race only, `JobNoLongerRunning` when the failure was applied first); the job's error is exactly `child jobs would bring the job to 6 child jobs of N bytes, over the limits of 10000 child jobs and 16777216 bytes per job`; no child row. With a retry the loser answers `Ok(Running)` and its task is republished once; its rerun without children registers **exactly the winner's 3 children**, which shows the rejected report's reservation was not kept; the job then succeeds with outputs `[t0, t1]`.

Iterations: 1,000 × 6 = 6,000. Result: **pass** (release, 997 s).

### T10 depth `a_report_beyond_the_maximum_depth_fails_the_whole_chain`

A chain registered down to depth 8 with a sibling `S` (cleanup) at depth 1; the depth-8 job reports one child. Exact asserts: the report answers `Ok(Failed)`; no child row; the depth-8 job's error is `child jobs at depth 9 exceed the maximum depth 8`, each of the other 8 chain jobs has `descendant job D8 failed: …`; all 9 `Failed`, `S` cleaned up to `Cancelled`; lanes empty, quiescent.

Iterations: 100. Result: **pass** (release, 12 s).

### Other runs

- Debug build (`debug_assertions`, overflow checks), 100 iterations: all 8 tests pass (527 s).
- The full `spider-storage` suite, `--all-features --run-ignored all --release`, default 50 iterations, container `jtp-test-t8910`: **283/283**.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`, `cargo clippy -p spider-storage --all-targets -- -D warnings` (default features) and `rustfmt +nightly --check` on the changed files: clean, with no `#[allow]` in the new file.

### Mutation checks

Each defect was applied to a copy of the worktree with its own target directory (never to the shared worktree), all 8 tests were run with 2 iterations, and the file was restored.

| Defect | Caught by |
|---|---|
| M1 recovery does not load `BlockingOnChildren` rows | registration, cascade, cancel, zombie (recovery loads the wrong jobs, or repairs the children as orphans) |
| M2 rebuilt parent counter one too high | registration, cascade, cancel, zombie (R5 fails consistent trees: "repaired nothing" broken) |
| M3 resend skips recovered non-root `Running` jobs | registration, cascade, cancel, zombie (the sentinel lane check: the children's tasks are missing) |
| M5 handler bodies awaited inline instead of spawned | T9 (the dropped work never applies: the job is stranded) |
| M6 the committer's reply to a dropped waiter is a panic | **not caught**; see finding 2 |
| M7 depth limit off by one | depth (the depth-9 child is registered) |
| M8 child reservation counts the report only, not the job's earlier reports | reservation race (both reports are accepted; the planner's backstop then fails the job with `child-job limit exceeded`) |
| M9 a deterministic payload error of a registration poisons instead of failing the parent | limit (`PayloadRejected`: `Stopping`) |
| M10 no fence check in `apply_tree_transition` | zombie (the superseded runtime registers its children after the successor started) |
| M11 a report over a limit is an RPC error instead of a task failure | limit, depth, reservation race |

### Bugs found and fixed

No implementation defect was found for T8, T9 or T10; the implementation is unchanged. The only worktree edits are the new test file and its `mod` line in `tests/test_spider_storage.rs`.

One test bug was found and fixed while validating the tests: asserting `is_fatal()` right after a crashed RPC returned is racy (finding 1). M5 exposed it (the cascade test failed on it with inline bodies). The tests now wait for the service's token, then check the flag.

### Findings

1. **A crashed tree's callers are answered before the service is marked fatal.** `run_tree_state_transaction` returns `Err`, which drops the round's reply channels (callers wake with `TreePoisoned`), and only then does `run_committer` call `guard.poison()`, which sets the fatal flag and cancels the token. Through gRPC this is harmless: `default_error_handler` marks the shutdown fatal on the same `INTERNAL` error. An in-process caller that checks `is_fatal()` at once can see `false`. The formal implementation could poison before dropping the round, so that a caller never observes the error before the flag.
2. **The committer's tolerance of a dropped reply receiver is not reachable by T9.** Handler bodies are spawned, so dropping an RPC never drops the waiter: the reply always has a receiver (M6 is not caught). The tolerance matters only where `submit_and_wait` gives up on the token during a shutdown; the S2 shutdown unit tests are the place to pin it.
3. **The per-job child limit has two layers.** The JCB reservation rejects the report before decoding (the exact `child jobs would bring the job …` message, a task failure); the planner's own check fires only if the reservation is broken (M8), and fails the whole job with a generic `child-job limit exceeded`. Only the first layer is reachable today.
4. **What an execution manager sees after a restart, for a report it retries:** `StaleSession` in the old session (it must re-fetch the session), then in the new one `InvalidTaskInstanceId` (the instance was lost with the old runtime, the task is republished), `JobNoLongerRunning` (the job blocked on children before the crash), or `JobNotFound` (the job became terminal before the crash and is not recovered; gRPC `NOT_FOUND`, which the EM must treat as stale). The caller of a crashed RPC sees `UNAVAILABLE` (`Stopping`) or `INTERNAL` (`TreePoisoned`), nondeterministically.
5. **Known gap, as the plan says:** no hook lies between BEGIN and COMMIT, so a crash with the transaction open and a lost COMMIT acknowledgement are not exercised on the real code.
6. **Host:** `start.py` pulls `mariadb:latest`, which is now 13.0.2; its first start after the pull timed out at the default 120 s readiness wait (a second start was ready in 10 s). The run used container `jtp-test-t8910`, stopped at the end. The relayed request to kill the kind cluster: `clp-rc2-control-plane` was already stopped (`Exited (137)` about 10 hours before); it was neither started nor deleted.

## T11: planner property tests (test engineer)

**File:** `components/spider-storage/src/cache/tree/property.rs` (`#[cfg(test)] mod property` of `cache/tree.rs`). S3 and S6 wrote its first version (1,165 lines, two tests); this group extended it to 1,950 lines (+915/−130) and added a third test. No other worktree file changed.

**How to run:** `cargo test -p spider-storage --features test-hooks --lib --release -- cache::tree::property --nocapture` (CI's `cargo nextest run --all --all-features --run-ignored all --release` runs it too). `JOB_TREE_PROPERTY_ROUNDS` sets the planned rounds per raising mode and the number of repaired trees (default 20,000 in release, 3,000 in debug). `JOB_TREE_PROPERTY_SEED` replays the one tree whose seed a failure prints. Cost on this host, release, each test alone: 24 s, 29 s and 83 s (the repaired-tree continuations dominate); the three run in parallel.

**Why there are no hooks and no sleeps:** `plan_events` and `plan_repair` are pure and synchronous. The test plays the committer and the database itself: an in-memory row model is the DB, a `TreeStructure` is the cache. An "interleaving" is the arrival order of events in the committer's buffer, the round boundaries, deferral and requeue, and every one is drawn from a seeded `StdRng`. A run is deterministic for its seed. **The fatal-path analogue:** `plan_events` panics on a broken invariant, which in the service poisons the tree and exits non-zero. No planned round in any run panicked.

### The model

- **Rows (the DB):** parent, depth, state, `has_commit`, `has_cleanup`, stored outputs, stored error. **Structure (the cache):** a `TreeStructure` with counters, updated from each plan as the committer does.
- **Tree shapes:** a root with random commit and cleanup tasks. A registering `RegularTasksDone` registers 1-4 children with random flags, down to depth 5. Each tree draws a profile:
  - registration probability 0.3, 0.6 or 0.95;
  - failure probability 0.15 or 0.01;
  - cancel probability per round 0.02 or 0.001.
  The low-failure profile grows trees of up to 122 rows.
- **Every event kind:**
  - `Start`;
  - `RegularTasksDone` with and without children. Child sets are 5 % over 16 MiB (a limit failure), 5 % exactly 16 MiB, 10 % between 8 and 16 MiB, and small otherwise. Outputs are 6 MiB in 5 % of reports, to hit the 16 MiB round budget;
  - `CommitDone` and `CleanupDone`;
  - `TerminalFailure` on each lane;
  - `CancelRoot`.

  At the tree's failure rate, a second `TerminalFailure` of the same job and lane follows (another task instance failing: the T1 and T5 shapes at planner level).
- **Arrivals per round:**
  - each job's event with probability 0.6;
  - 0-2 copies of earlier events (stale re-issues);
  - in 0.5 % of rounds, a burst of 257-319 copies, after which the round is a full 256 events.
- **Rounds:**
  - a random prefix of the buffer of at most `MAX_EVENTS_PER_ROUND` (256) events, as `take_events` takes;
  - inputs collected as the committer does (`jobs_to_collect`), and deferred events keep theirs;
  - deferred events go back to the front of the buffer in order.
- **Two raising modes:**
  - **Repeated** (the original test): a job may raise its state's event again in every round.
  - **Once** (new): each job raises the event of each of its states once, as the real producers do (phase A's `fetch_sub`, one commit and one cleanup task). An event that is wrongly answered stale, or never planned, then leaves its job stuck, and the test sees it.

### `planned_rounds_keep_the_tree_consistent` (Repeated) and `events_raised_once_are_each_answered_and_quiesce_the_tree` (Once, new)

Asserted after **every** round, that is, every tree state transaction ("new" marks what this group added):

1. **Answered once.** Every event of the round is planned or deferred, exactly once. At most 256 are planned (new). Deferred events keep their inputs, and planned ones take theirs.
2. **Liveness.** The first `CancelRoot` is planned, and so is the oldest event other than it.
3. **Exact outcome of every event (new).** Each planned event is replanned with only the events that the planner planned before it, which gives the state it found its job in. That replan must defer nothing and give the earlier events identical outcomes, so a later event never changes an earlier one's plan.
   - The outcome must be **exactly** `Untracked` if the job is not in the tree.
   - Else it is `Applied(s)` if the event acts on the state it found:
     - `Start` on `Ready`;
     - `RegularTasksDone` and a regular-lane failure on `Running`;
     - `CommitDone` and a commit-lane failure on `CommitReady`;
     - `CleanupDone` and a cleanup-lane failure on `CleanupReady`;
     - `CancelRoot` on any state other than `CleanupReady` and the terminal states.
   - Else it is `Stale(e)`, with `e` exactly: `JobAlreadyStarted`; `JobNoLongerRunning`, `JobNoLongerCommitReady` or `JobNoLongerCleanupReady`; `JobAlreadyTerminated(s)`; `JobCancellationAlreadyRequested`; or `JobAlreadyCancelled`.
   - The expected `s` comes from the design's event table, written independently in the test:
     - `Start` → `Running`;
     - `RegularTasksDone` → `BlockingOnChildren` (children within the limits), `Failed` (over a limit), `CommitReady` (with a commit task) or `Succeeded`;
     - `CommitDone` → `Succeeded`, and `CleanupDone` → `Cancelled`;
     - a failure → `Failed`;
     - a cancel → `Cancelled` (a `Ready` root, or no cleanup task) or `CleanupReady`.
   - An applied event's update of its job goes from the round's starting state to `s`.
4. **Failures (new).** The failed job's stored error is exactly the reported message, or `child-job limit exceeded` for a limit failure.
   - If the event found its parent `BlockingOnChildren` and it was not a cleanup failure, every ancestor is `Failed` with exactly `descendant job J failed: <message>`.
   - Otherwise nothing propagates: a skipped `CommitReady` job's commit failure under an aborted parent fails only that job.
5. **Registration (new).** At most one per round, and its event is `Applied(BlockingOnChildren)`. The child set equals the event's input and is within the limits; the depth is the parent's depth + 1 and the root is the tree's root. The parent had no child row before (the child set is frozen).
6. **The writes.**
   - No guard miss: every update's `from` equals the model row.
   - No row is updated twice, and every edge is valid (`is_valid_transition`).
   - A plan writes iff some event applies, and an all-stale round gives an empty plan.
   - The payload is within 16 MiB unless a single event carries it.
   - Publications, `terminated` and the apply order (deepest first, root last) follow from the updates.
7. **After the apply.**
   - Each `Applied(s)` matches the row, with the event's outputs stored.
   - I1-I5 hold on all rows, including terminal children: the full I2, "success is earned", the I3 active chain and the `CommitReady` rule, I4 closed tree, and I5 (the structure holds exactly the non-terminal rows, node fields equal the rows, counters equal the active children).
   - `check_invariants` returns `Ok`.
   - One random node's state or counter is perturbed, and `check_invariants` must fail iff the model, restricted to what the structure sees, finds a violation.
8. **Quiescence (Once only, new).** Every tree becomes quiescent.
   - Copies of earlier events stop after 400 rounds. The tree must then be quiescent within the buffer's backlog plus 400 rounds: each round plans the oldest event, so only a lost event can exceed that.
   - The events still buffered are then planned with no arrivals, and every one must be `Untracked`.
9. **Coverage.** Every coverage counter is asserted non-zero, and so is every (event kind × {Applied, Stale}) pair in Repeated mode and every (event kind × {Applied, Stale, Untracked}) pair in Once mode, over the nine kinds (`RegularTasksDone` split by registering, `TerminalFailure` by lane).

### `repaired_trees_keep_the_invariants` (extended)

Unchanged from S6:
- Generate a consistent tree (any root state, depth ≤ 5, fan-out ≤ 4, terminal rows included); `plan_repair` must be empty for it.
- Inject 0-3 corruptions: a random state, a reparent (including to a larger ID or to itself), or a flipped cleanup flag.
- Load it as recovery does (non-terminal rows, and `failed_child` probed from the terminal rows), then repair it.
- Assert the guards against the loaded states, valid edges and at most one update per job; `check_invariants(repaired) == Ok`; no blocking job with a `Failed`, `Cancelled` or `CleanupReady` child (on all rows); and a second `plan_repair` is empty.

New:
- **Rebuilt depths.** A job whose ancestors were all loaded gets its true depth from `TreeStructure::from_rows`. Any other job, whose depth the rebuild cannot know, must not be `Ready`, `Running` or `BlockingOnChildren`, so it can never register children with a wrong depth (finding 3).
- **Continuation.** Unless it is quiescent, the repaired tree becomes the model and runs in Once mode until quiescent, with every per-round check above. This shows that what recovery leaves is plannable to the end.
  - Corruptions of *terminal* rows are invisible to recovery and cannot be repaired: a parent link to a larger ID, or success that is no longer earned. A tree with such rows is checked with those two rules relaxed for terminal rows ("lenient").
  - A tree in which a `Ready` or `Running` row already has child rows is skipped and counted (finding 4).

### Iterations and results

| Run | Planned rounds / trees | Result |
|---|---|---|
| Release, 20 runs of the final code (1,991 s) | Repeated: 400,254 rounds over 35,569 trees; Once: 400,409 rounds over 31,671 trees, **every tree quiescent** (79 needed the backlog phase); repair: 400,000 trees | **pass**, 20/20 |
| Debug (`debug_assertions`, overflow checks), 3 runs of the final code plus 6 runs of the same checks before the refactor into methods | 9 × 3,000 rounds per mode, 9 × 3,000 repaired trees | **pass**, 9/9 |

Repair totals over the 20 release runs:
- 83,920 consistent trees (empty plan);
- 89,193 plans with orphan aborts (R1);
- 14,159 with unblocks (R2);
- 8,551 R4 plans and 10,527 R5 plans;
- 305,886 repaired trees run on to quiescence, 131,642 of them lenient;
- 5,880 skipped (finding 4);
- 629,825 rows whose rebuilt depth differs from the true one, none of them able to register.

A typical release run, Once mode:
- trees up to 100 rows;
- 5,599 grouped rounds, 1.33 M deferrals, 36,051 payload-budget deferrals and 114 full 256-event rounds;
- 2,748 registrations and 148 limit failures;
- 846 propagated failures and 113 applied cancels;
- 129 skipped commits and 843 unblocked parents.

Each event kind is applied, stale and untracked from a few dozen to a few thousand times per run. The rarest are commit-lane failures: 58 applied, 57 stale and 46 untracked in one Repeated run.

### Mutation checks

Each defect was applied to the planner or the repair in a copy of the worktree (`rsync` without `target/`, own target directory), never to the shared worktree. The original S3/S6 test file and the extended one were each run once at the release default (20,000), then the copy was restored.

| Defect | Original test | Extended test |
|---|---|---|
| N1 a cancel of a `Cancelled` root answers `JobAlreadyTerminated(Cancelled)` instead of `JobAlreadyCancelled` | not caught | caught (exact outcome) |
| N2 a skipped `CommitReady` job's `CommitDone` is stale (a lost event; the tree never quiesces) | caught only indirectly (the "skipped commits" coverage counter stays 0) | caught (exact outcome, all three tests) |
| N3 the chain error names the nearest ancestor instead of the failed job | not caught | caught (failure check) |
| N7 a child set of exactly 16 MiB fails its job (`>=` for `>`) | not caught | caught (exact outcome) |
| N13 a cancel of a `CommitReady` root is stale (a lost cancel) | not caught | caught (exact outcome) |
| N15 registered children get their parent's depth (so the depth limit is off by one per level) | not caught | caught (registration check) |
| N16 `from_rows` gives a recovered child its parent's depth | not caught | caught (rebuilt depths) |
| N17 `from_rows` drops recovered jobs' cleanup flags (a cancel after a restart would skip cleanup tasks) | not caught | caught (continuation, I5 node check) |

### Other runs

- The full `spider-storage` suite, `cargo nextest run -p spider-storage --all-features --run-ignored all --release`, container `jtp-test-t11` (port 54705, MariaDB 13.0.2), three runs: each **283/284**. All three property tests pass inside it (31 s, 34 s, 114 s under the suite's load). The one failure differs per run and is a pre-existing timing flake of `mariadb_test` that this change cannot reach (finding 6): `test_delete_expired_terminated_jobs`, then `test_update_execution_manager_heartbeat_already_dead`, then `test_get_dead_execution_managers_atomic`.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` and `cargo clippy -p spider-storage --all-targets -- -D warnings`: clean, no `#[allow]` added. `rustfmt +nightly --edition 2024 --check` on the file: clean.

### Bugs found and fixed

No implementation defect was found, and the implementation is unchanged.

My own new assertions were wrong at first in four places, all fixed in the test:
1. The failure chain was expected from the parent's state at the start of the round. A skipped `CommitReady` job whose parent an earlier event of the same round deactivated does not propagate, so the expectation must use the state that the event found.
2. Bursts first fed the planner more than 256 events, an input that `take_events` never produces (finding 1).
3. The quiescence bound first ignored a backlog of registering copies (finding 2).
4. Rare stale cases (commit-lane and cleanup-lane failures) were not reached in 3,000 debug rounds until the second failure was raised on every lane.

### Findings

1. **The planner's own event cap is dead code, and it would starve a registration if it were reached.** `take_events` already takes at most 256 events, so `plan_events` never sees more. Given more, it defers by planning position, and the registering event is planned last, so the oldest event of the round can be deferred if it registers. That breaks the "the oldest event other than the first cancel is planned" liveness that the S3 notes claim for the pure planner. The first version of the bursts hit exactly this. The formal implementation should keep a single cap (in `take_events`), or rank the planner's cap by arrival order.
2. **Registering events are answered one per round, even when they can only be stale or untracked.** Extra registering events are deferred whole, whatever their job's state. After a cancel or a failure, a buffer holding N registering events of the tree takes N rounds to answer them. Each of those rounds has an empty plan, so it writes nothing to the DB, but each still collects inputs, and each caller waits its turn. One test tree (14 rows, about 1,000 buffered copies) took 1,659 rounds. In production N is at most the number of jobs whose last report registered children, which a wide fan-out (B4) can make large. The planner could answer an extra registering event at once when its job is not `Running` in the overlay.
3. **`TreeStructure::from_rows` cannot know the depth of a job whose parent row is terminal,** because recovery loads only non-terminal rows: a skipped `CommitReady` job under a cancelled parent, or a `CleanupReady` orphan. It gives such a job depth 1, as its docstring says. This is harmless today: about 31,000 such rows per 20,000 trees, and none could register children. Only the deepest-first apply order among such jobs is affected, and the root is still applied last. If depth is ever used for more, recovery should load it, or compute it from all rows.
4. **Repair cannot see a `Ready` or `Running` job that already has child rows when all of them are terminal.** This breaks I1 (the child set is frozen at registration), but recovery loads no terminal rows and `has_failed_child` probes only `BlockingOnChildren` jobs. Only row corruption reaches it, through a backwards state edge (1.5 % of the generated trees here). If such a job registers again, its child set mixes old and new children, and if an old child `Failed`, the next restart's R4 fails the parent. It is a known limit of `check_invariants` (U13), so the test skips and counts these trees rather than fail.
5. **Kind cluster (the relayed request to kill it):** `clp-rc2-control-plane` was already stopped (`Exited (137)` about 11 hours before this run). It was neither started nor deleted.
6. **Pre-existing timing flakes in `mariadb_test`, not caused by this group.**
   - `test_delete_expired_terminated_jobs`, `test_update_execution_manager_heartbeat_already_dead` and `test_get_dead_execution_managers_atomic` all sleep 2 s, then expect a row older than 1 s by `NOW() - INTERVAL ? SECOND` on a second-precision `TIMESTAMP`. Against MariaDB 13.0.2 (what `start.py` now pulls as `latest`), each occasionally finds nothing (`got []`).
   - The same two tests, run in isolation from the unmodified `main` checkout (`/home/lzh/dev/spider`, built into its own untracked `target/`, no tracked file touched) against the same container, fail at the same rate: `test_delete_expired_terminated_jobs` 2 of 30, and `test_update_execution_manager_heartbeat_already_dead` 2 of 25. The prototype worktree gave 1 of 15 for the first.
   - This group's only change is a `#[cfg(test)]` module of the library, which is not compiled into the `test_spider_storage` binary that holds these tests. Nothing was changed to hide the flakes.

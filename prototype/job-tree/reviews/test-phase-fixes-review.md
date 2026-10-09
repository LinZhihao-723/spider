# Review: test-phase fixes for T1, T4, T5, T6 (slice reviewer)

**Scope:** the edits that the T1/T4/T5/T6 test group (test engineer) made in `/home/lzh/dev/spider-job-tree-proto`, and the claims in its `tests.md` section:
- `components/spider-storage/src/state/tree_committer.rs` `apply_plan`: `round_context: impl Fn() -> HookContext` became `impl Fn() -> HookContext + Send + Sync`;
- `components/spider-storage/tests/test_spider_storage.rs`: `mod job_tree_completion_test;`;
- the new `components/spider-storage/tests/job_tree_completion_test.rs` (2,247 lines);
- the findings on the stale execution manager and on `job_tree_race_test.rs::race`.

**Verdict: ok.** Both edits are correct and change no behavior. The tests pin what the brief asks for T1, T4, T5 and T6, with exact asserts. I found no false pass. I found two minor weaknesses in the test harness. Each can only produce a false failure or miss the very last round of a run, and neither needs a fix before the benchmarks. The rest of this file is notes for the report and for the formal implementation.

## What I checked

- `cargo clippy -p spider-storage --all-targets --all-features` in the worktree, with the repository's `.cargo/config.toml` (`-Dclippy::all -Dclippy::nursery -Dclippy::pedantic`): **clean** (exit 0; the only warning is the existing future-incompatibility note for the `proc-macro-error2` dependency). I did not run the tests again; the run counts below come from test engineer's notes.
- File timestamps. The only edits after the S1–S6 and style reviews that this group made are its test file (05:46), its `mod` line, and `tree_committer.rs` (05:45). `TestEnv::start_with_task_instance_pool` already existed from S2 (`slices/02-s2-size1-committer.md` §5), so test engineer changed no shared infrastructure.
- Every claim in the T1/T4/T5/T6 section, checked against the code: the hook firing order, `TreeControlBlock::submit`, `ArmedHook`, `run_committer`, `run_tree_state_transaction`, `climb_after_success`, the JCB's `fetch_sub`, `Runtime::stop`, `TaskInstancePoolConfig::default`, and `job_tree_race_test.rs::race` with its callers.

## 1. The `apply_plan` bound (correct, no behavior change)

- `apply_plan` has one caller (`tree_committer.rs:604`). Its closure captures a `JobId` and a `&Vec<Queued>`, so it is `Send + Sync` at the call site. The bound adds no constraint that the code did not already meet. It only lets `clippy::future_not_send` prove that the generic `apply_plan` future is `Send`: the inner closure holds `&round_context` across `hit_after_commit(..).await`, which needs `Sync`.
- The bound does not depend on the `spider_unboxed_round` cfg (B6) or on the `test-hooks` feature: with the disarmed hooks the closure is never called, but it is the same type.
- The docstring is still accurate. No comment was added.
- **Where it came from.** style reviewer's run at 04:35 found clippy clean apart from the `fetch_update` deprecation (style review §9). The src files touched between 05:00 and 05:08 (`cache/job.rs`, `cache/task.rs`, `cache/tree.rs`, `state/service.rs`, `test_hooks/mod.rs`) postdate every review in `reviews/`. That pass is the likely origin of the `round_context` closure, and so of the lint failure. No review in `reviews/` covers that pass. This is a process note for the report, not a defect of this fix.

## 2. Minor: a narrow window in the `race` helper (false failure only)

`race` (`job_tree_completion_test.rs:772-795`) arms the `EventSubmitted` observer only after the first RPC's round is paused, "so that the first event's own submission is not counted" (`tests.md`, and the comment at line 786). This narrows the problem that finding 2 reports for `job_tree_race_test.rs`, but it does not remove it:
- `TreeControlBlock::submit` (`cache/tree.rs:232-237`) observes `EventSubmitted` **after** `sender.send` returns, and `TestHooks::observe` matches the hooks that are armed when it runs (`test_hooks/armed.rs:289-296`).
- So the following order is possible: the first handler sends; the committer receives the event and pauses at `CommitterAfterReceive`; the test sees the pause and arms the observer; only then does the first handler call `observe`. The observer counts the first event, `reached(&submitted, 1)` returns, and the pause is released before the second RPC submits.
- In T1 the filter (`event_filter(root, J, TerminalFailure)`) matches both events. The same pattern is in `report_sibling_successes` (`SiblingRounds::One`, lines 1231-1249), whose filter is any `RegularTasksDone` of the root.
- **Effect: a false failure, never a false pass.** T1 asserts the round composition, the failure-event count and the exact stale variant, and T4 `One` asserts one round of `k` events. A second event that misses the round fails those asserts.
- **How likely:** rare. The wake from `send` usually lands in the sending worker's LIFO slot, and tokio does not steal from that slot, so the committer usually runs only after the handler yields, which is after `observe`. The 1,000-iteration runs passed. But this is a scheduler heuristic, not a guarantee: when the LIFO slot is already taken, or has been used too often in this tick, the woken task goes to the stealable local queue.
- **Fix (when convenient):** before spawning the first RPC, arm an observer for the first event; wait for both it and the pause; then arm the second event's observer. The first `observe` has then run, so the second observer can only count later submissions. Also make the comment at line 786 accurate.

## 3. Minor: the last round of each test is never checked for the fatal path

- `run_tree_state_transaction` runs `assert_invariants` (the I2 counter check that T4's "counter reaches 0 once" claim relies on) **after** `reply` and the publications (`tree_committer.rs:629-636`). A violation in the final round therefore panics after the caller has its answer.
- `assert_states` checks `is_fatal()` right after the replies, so it can run before that panic. Within a run this does not matter: the next iteration's RPCs fail on the cancelled token.
- The last round of each test function is not covered, however. Neither `TestEnv::finish` nor `Runtime::stop` reports a fatal shutdown or a panicked tracker task: `stop` checks only the task instance pool and the GC join results, and `tracker.wait()` swallows panics.
- **Fix:** add `assert!(!env.is_fatal())` after the final `assert_lanes_empty` (which already waits at least 300 ms) in each of the five tests. This is the same ordering as the T8–T10 group's finding 1 (callers are answered before the fatal flag is set).

## 4. The findings, checked

1. **The stale execution manager: confirmed.**
   - `TaskInstancePoolConfig::default()` has `execution_manager_stale_cutoff_sec = 60` and `gc_interval_sec = 30` (`task_instance_pool.rs:89-90`). `TestEnv` registers an execution manager and never heartbeats it. With GC ticks at 30, 60 and 90 s, the first death verdict at 90 s matches the observed failures.
   - The later test groups adopted the long cutoff: `job_tree_cancel_test.rs:321` and `job_tree_fault_test.rs:251` copy the same `start_env`. `job_tree_race_test.rs` (8 uses), `job_tree_recovery_test.rs` (11), `job_tree_test.rs` (7) and `registration_test.rs` (10) still use the default `TestEnv::start`. At the default 50 iterations they stay well under 90 s.
   - Three files now carry an identical `start_env`. The single fix is the one test engineer suggests: make `TestEnv::start` use the long cutoff, and drop the copies.
2. **The `job_tree_race_test.rs::race` weakness: confirmed, and probably worse than stated.**
   - Its observer is armed before the first spawn (`job_tree_race_test.rs:503-512`). Its T1 caller (line 577) filters on the leaf's `TerminalFailure`, which the first failure also matches.
   - The test runs on `#[tokio::test]`, a current-thread runtime. There, `reached(1)` returns without yielding, so the pause is released before the second RPC has even started. The `CommitterAfterReceive` iteration therefore most likely always runs as "queued behind", never "same round". I did not run it.
   - Its assert of `JobAlreadyTerminated(Failed)` holds only because the second RPC's phase A precedes the first round's apply. test engineer's T1 `AfterReply` case shows that the answer would be `JobNoLongerRunning` otherwise.
   - The cancel and commit callers filter on a different event, as test engineer says, and are unaffected. The completion test's own T1 covers the same-round case with an exact round assert, so coverage of T1 as a whole is intact.
3. **M1's masking ties into an open style decision.** style reviewer §24 asks Pleiades whether the "changes a job twice" deferral (`Plan::admits`) should become a `debug_assert!`, since "only a broken producer can trigger it". M1 is exactly such a broken producer: without `plan_failure`'s lane check, the second same-round failure is caught only by that deferral. Under a `debug_assert!`, M1 would panic instead of passing silently, which supports the assert. Either way, a planner unit test for the lane check, as test engineer suggests, should come first.
4. **The stale variants by path:** consistent with the S5 review and the T2/T3/T7 findings.

## 5. Notes for the report

- **The `fetch_sub` `AcqRel` ordering (brief, in scope) is not tested by T4-within-a-job, and cannot be on this host.** The test pins that exactly one decrementer submits `RegularTasksDone` (M4). A `Relaxed` mutant would also pass, for two reasons. On x86-64 (TSO), a `Relaxed` read-modify-write is still a locked instruction. And the outputs that the last decrementer reads are written and read under the TCB locks (`collect_regular_done_inputs`), which already order them. The report should say that the ordering is covered by reasoning, not by a test. A `loom` model of `succeed_task_instance` would be the way to pin it.
- **T6 `WithLastChild` checks only the last round's composition (`[RegularTasksDone, RegularTasksDone]`), not the number of rounds.** This is sufficient: a split round would have made the last round a single event.
- **The T5 retry dimension (`max_num_retry` alternating 0 and 1):** a wrongly re-published task is caught by the next iteration's `drain_lane` ("unexpected entry"), or by the final `assert_lanes_empty`. The claim in `tests.md` holds.
- **The inline comments in the test** (lines 786, 965-967, 1116, 1648-1649) explain *why* a value or a skip is chosen, which the style guide allows. Line 786 overstates its guarantee (see 2).

## Kind cluster

The relayed request was to kill the kind cluster. `clp-rc2-control-plane` is already stopped (`docker ps -a`: `Exited (137) 13 hours ago`), so there is nothing left to kill. I did not start it, and I did not delete it (`kind delete cluster --name clp-rc2`), because deleting is irreversible and was not asked for explicitly.

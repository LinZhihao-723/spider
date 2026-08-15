# Prototype Implementation Stage Report

This report records the state of the resource-group-aware scheduler prototype at the end of the correctness-validation stage. It documents what was built, how the implementation corresponds to the correctness argument in [design.md](design.md), every test that exists and what it actually asserts, and — explicitly — what is *not* covered.

Performance evaluation is a separate stage and is deliberately out of scope. No number in this report should be read as a performance measurement.

### Terminology

**Publish vs. dispatch.** The core *publishes* an assignment: `publish` in `scheduling_unit.rs` stamps a fresh, globally unique `TaskAssignmentId` from `TaskAssignmentIdIssuer` and sends the assignment into its resource group's queue. Publication is the core committing to a scheduling decision. *Dispatch* is the later, separate event of an execution manager pulling that assignment over gRPC. The two are distinct in time and the tests depend on the distinction.

**Publication order.** The order assignments were published in, recovered by sorting on assignment ID. Because the core is single-threaded and the issuer is a single global counter, publication order is a total order on the core's decisions. It is observable from workers' records alone, even though workers receive assignments concurrently and out of order — which is why the integration tests assert on publication order rather than on absolute elapsed time, and why they stay meaningful on a loaded machine.

**Pinned / general execution manager.** A *pinned* manager names a resource group and may receive assignments only from it. A *general* manager names none and may receive any assignment. The distinction is the reason the hint channel exists.

**Symbols**, carried over from [design.md](design.md): `B` is the dispatch buffer capacity (`dispatch_queue_capacity`); `F` the buffer's current free space; `S` a group's dispatch queue size; `q` the assignments physically in that queue; `H` a group's `living_hint` counter; `N` the number of backlogged groups.

## 1. Status

| | |
|---|---|
| Crate | `components/spider-scheduler-new` |
| Size | 28 files, ~6 300 lines including tests |
| Tests | 41 unit + 7 integration, **48 passing, 0 failing** |
| Lint | `task lint:fix-rust` clean; nightly clippy clean under `-Dclippy::all -Dclippy::nursery -Dclippy::pedantic` |
| Workspace impact | 3 lines in the root `Cargo.toml`; nothing else outside the crate |

The prototype does not depend on `spider-scheduler`. It shares only `spider-core` types, so it is a parallel implementation rather than a refactor, and the existing scheduler is untouched.

## 2. What was built

### 2.1 The core

| Module | Responsibility |
|---|---|
| `src/core.rs` | The five-step tick, the run loop, and the dedicated-thread launcher |
| `src/scheduling_unit.rs` | `RgSchedulingUnit` and `try_make_assignment` — the per-group decision |
| `src/job_registry.rs` | `JobEntry`, `SharedJobEntry`, `JobRegistry` |
| `src/resource_group.rs` | `ResourceGroupTable`, `RgDispatchQueueReader`, the per-group queues |
| `src/dispatch_queue.rs` | The hint channel and the pinned/general dispatch paths |
| `src/config.rs`, `src/error.rs`, `src/session.rs`, `src/storage_client.rs`, `src/types.rs` | Supporting definitions |

**The core future is `!Send`.** It holds `Rc<RefCell<RgSchedulingUnit>>` and `Rc<SharedJobEntryInner>` across await points, which is a direct consequence of the design's decision to co-own scheduling units by `Rc` rather than look them up by ID. It therefore cannot be handed to `tokio::spawn`; it runs under a `LocalSet` on a dedicated OS thread with a current-thread runtime. Inbound polls are still issued with `tokio::task::spawn`, because those futures capture only the cloned storage client and are `Send`.

One consequence worth recording: `run_core_on_dedicated_thread` takes a `FnOnce() -> Core<…> + Send` factory rather than a `Core` value, because the value itself cannot cross the thread boundary.

### 2.2 The harness

The harness is a first-class module (`src/harness/`), not test-gated, so the performance stage can drive it from a benchmark binary rather than only from `#[cfg(test)]`.

| Module | Responsibility |
|---|---|
| `src/harness/fake_storage.rs` | `FakeStorage` — plays the inbound queue, generating ready tasks and finalizations |
| `src/harness/grpc_service.rs` | `HarnessServer` — a tonic service in front of the real `DispatchService` |
| `src/harness/fake_worker.rs` | `FakeWorker` — gRPC execution managers, pinned or general, with configurable task duration |
| `src/harness/metrics.rs` | `LatencySamples`, `DispatchRecord` — client-side latency collection |
| `src/harness/mod.rs` | `Harness` — bootstraps the stack and runs it to completion |

The crate carries its own protobuf schema (`proto/prototype_scheduler.proto`) because the production `scheduler.proto` has no way to express a pinned execution manager. `NextTaskRequest` adds an `optional resource_group_id`; absent means general. Generated into `OUT_DIR`, so nothing is committed.

**The harness is built to be a trustworthy measuring instrument.** Each worker accumulates latency samples into its own pre-sized `Vec`; nothing on the measured path takes a lock, touches shared state, or allocates per sample. Merging happens only after the run. The timing window closes around the RPC alone, with the simulated execution sleep outside it, and the gRPC client is created once per worker so connection setup is never inside a measured window.

## 3. Correspondence between the proof and the code

The correctness argument in design §8 is a mathematical one, and the implementation's job is to match it. Each requirement below fails *silently* if violated — no test would necessarily catch it — so the correspondence is recorded explicitly here.

| Design requirement | Code site | How it is realized |
|---|---|---|
| §8.2 req. 1 — the assignment is in the queue **before** `H` is compared against `S` | `scheduling_unit.rs:360` then `:368-370` | `try_send` precedes the `dispatch_queue_size()` / `living_hint.load()` pair, in that order, in a single non-async function |
| §8.2 req. 2 — neither send may block or fail | `resource_group.rs:163`, `dispatch_queue.rs:37` | Both channels are `async_channel::unbounded()`; see §5.1 below |
| §8.2 req. 3 — no await point between the hint receive and `H -= 1` | `dispatch_queue.rs:146-161`, `resource_group.rs:78` | `consume_hint_and_try_recv` is a **synchronous `fn`**. A future can only be dropped at an await, so making the decrement unreachable from an await point is a type-system guarantee, not a convention |
| §8.2 req. 4 — a stale-generation hint is discarded **without touching any counter** | `dispatch_queue.rs:152` | The session comparison precedes `consume_hint_and_try_recv`, and `continue`s past it |
| §8.1 memory ordering — `Acquire` load, `Release` increment, no compare-exchange | `scheduling_unit.rs:370`, `:374` | `living_hint.load(Acquire)` and `fetch_add(1, Release)`. A decrement landing between them only lowers `H`, which strengthens the condition that triggered publication |
| §6.2 — the admission threshold `S < F` with α = 1 | `scheduling_unit.rs:160` | `if self.dispatch_queue_size() >= free { return Err(DispatchQueueFull) }` |
| §6.4 — `F` recomputed **by decrement** on every decision | `core.rs:451` | `free -= 1` inside the round-robin loop, not once per tick |
| §6.4 — admission interleaved at quantum 1 | `core.rs:453` | The arm advances by exactly one group per successful assignment |
| §5.4b — the rotation arm persists across ticks | `core.rs:412-431` | A single pass over `active_rg_list` sums occupancy, builds `rg_rr_list`, and locates `last_served_rg` to seed `arm` |
| §5.4d — `swap_remove` must **not** advance the arm | `core.rs:459-464` | `swap_remove(arm)` followed by `if arm == len { arm = 0 }`, with no increment — the tail element moved into that slot and incrementing would skip it |
| §5.4e — deactivation requires an empty dispatch queue **and** no schedulable tasks | `core.rs:483` | `deactivate_exhausted_units` checks both. `F` is summed over `active_rg_list` only, so a deactivated group still holding assignments would hide its occupancy and let the core over-admit |
| §5.1 — steps 4 and 5 run even when polling results are not ready | `core.rs:185` | Only steps 2 and 3 are skipped |
| §9 — the session bump clears the dedup set and the finalized job table | `core.rs:224` | `apply_session_bump` clears all five structures and drains the hint channel |
| §3.3 — `finalized` outside the `RefCell` | `job_registry.rs:273` | `Cell<bool>`, so `finalize()` and `is_finalized()` mutate and read through `&self` and cannot panic while a sibling borrow is live |

## 4. Test inventory

### 4.1 Unit tests — 41

These drive the core and its data structures directly, with no gRPC and no workers.

**`src/tests/admission.rs` — 4.** The dynamic threshold (design §6).

| Test | Asserts |
|---|---|
| `one_tick_leaves_every_backlogged_group_at_the_dynamic_threshold` | Five backlogged groups against `B = 256` each land within ±6 of `B/(N+1) = 42`, with ≈42 free |
| `no_group_is_batch_filled_while_another_waits` | max − min ≤ 2 across groups, max < `B/2`, min > 0 — the anti-staircase assertion, using the design's own worked numbers |
| `a_newly_active_group_is_admitted_against_a_backlogged_incumbent` | A newcomer facing a 100-deep incumbent reaches ≥ `B/8`, the incumbent's lead shrinks, ≥ `B/8` stays free |
| `a_lone_group_takes_no_more_than_half_the_dispatch_buffer` | `B/4 ≤ occupancy ≤ B/2` — §6.3's accepted "no per-group ceiling" |

**`src/tests/core.rs` — 7.** The tick (design §5, §9).

| Test | Asserts |
|---|---|
| `the_rotation_arm_persists_across_ticks` | §5.4b arm seeding from `last_served_rg` across two ticks |
| `dropping_an_exhausted_group_does_not_skip_the_group_moved_into_its_slot` | §5.4d — the `swap_remove` mechanic |
| `an_exhausted_group_stays_active_until_its_dispatch_queue_drains` | §5.4e — the empty-queue half of the deactivation condition |
| `dispatching_and_retirement_run_while_a_storage_poll_is_in_flight` | §5.1 / §4 — steps 4 and 5 still run, with retirement observed via `job_registry.len()` |
| `a_session_bump_clears_the_dedup_set_and_the_finalized_job_table` | §9 steps 3–4, via sentinel entries |
| `a_session_bump_readmits_the_tasks_storage_replays` | §9 — the reason the dedup set must be cleared |
| `a_rescheduled_assignment_is_readmitted` | §3.1 / §3.6 — the reschedule lane's happy path |

**`src/tests/dispatch_queue.rs` — 7.** The hint scheme (design §7, §8.1).

| Test | Asserts |
|---|---|
| `a_hint_is_published_only_while_the_hint_count_trails_the_queue_size` | The `H < S` publication rule |
| `a_pinned_pop_leaves_the_hint_count_untouched` | The pinned path never reads or writes `H` |
| `a_general_pop_consumes_one_hint` | `H` decrements by exactly one |
| `a_stale_hint_on_an_empty_group_consumes_the_hint_and_yields_nothing` | A stale hint is consumed, not requeued |
| `next_task_general_discards_a_stale_hint_without_touching_the_hint_count` | §8.2 req. 4 — asserts `H` is *still* 1 afterwards |
| `next_task_general_drops_an_assignment_published_in_a_stale_session` | §7.3 on the general path |
| `next_task_pinned_drops_an_assignment_published_in_a_stale_session` | §7.3 on the pinned path |

**`src/tests/job_registry.rs` — 7.** Design §3.3: upsert of a new job vs. an existing one, scheduling position preserved on append, `finalize_and_remove`, `get_next_task` returning `Err` on a finalized entry, `downgrade_counter` restored on insert, `take_ready_tasks` draining without finalizing, and `remove`/`clear`.

**`src/tests/scheduling_unit.rs` — 10.** Design §5.4 and §5.5.

| Test | Asserts |
|---|---|
| `an_empty_unit_reports_no_task` | Step 1 |
| `the_admission_threshold_binds_at_the_free_space_boundary` | Step 2, exactly at `S >= F` |
| `finalization_tasks_are_dispatched_ahead_of_regular_ones` | §5.4 "Finalization priority" |
| `an_exhausted_active_job_is_replaced_by_a_pending_one` | Promotion |
| `a_finalized_active_job_is_swapped_out_for_a_pending_one` | Step 4's finalized-job swap |
| `promotion_discards_a_finalized_pending_job` | §5.5 — a finalized job found during the promotion scan |
| `a_job_that_stops_producing_tasks_is_downgraded_and_then_retired` | §5.5 through `DOWNGRADE_LIVES = 1` |
| `an_arriving_task_restores_a_downgraded_job` | §5.5 — the counter reset |
| `a_rejected_publication_returns_the_finalization_it_took` | A closed queue returns the task rather than losing it |
| `a_rejected_publication_returns_the_regular_task_it_took` | Same, for a regular task, and stable across retries |

The last two exist because a task is taken out of its buffering structure *before* publication, so a rejected publish must put it back: the core removes it from the dedup set only on success, and a task in neither place could never be re-admitted.

**`src/tests/metrics.rs` — 6.** `LatencySamples` merge and sort, percentile interpolation, clamping of out-of-range and `NaN` percentiles, a single sample, an empty set, and `mean`. Harness instrumentation only — no design property.

### 4.2 Integration tests — 7

Full stack: the real core on its dedicated thread, the real gRPC server, real gRPC workers.

| Test | Shape | Asserts |
|---|---|---|
| `general_workers_alone_drain_every_resource_group` | 4 RG × 4 jobs × 16 tasks, 4 general workers | Exactly-once over the merged multiset; all 4 groups served; drained. With no pinned worker in existence, every one of the 256 assignments had to be reached by a hint — this is the §8.1 coverage invariant observed end to end |
| `pinned_workers_receive_only_their_own_resource_group` | 4 RG, one pinned worker each | Exactly-once; **zero** foreign-group assignments per worker; each worker's count is exactly its group's workload |
| `unpinned_resource_groups_drain_through_general_workers` | 6 RG, 3 pinned + 3 general | Exactly-once; pinned isolation; all 3 unpinned groups appear in general workers' records — §1.1's "a group whose dedicated managers are absent is still fully served" |
| `a_slow_resource_group_neither_blocks_nor_takes_over_the_buffer` | 4 RG × 2 jobs × 128 tasks (1 024 total), `B = 32`; run **twice** — a baseline with all four workers at 1 ms/task, then a contended run with RG0's worker at 10 ms/task | **Non-interference**: each fast group's mean job completion time in the contended run ≤ 2.0× its *own* baseline figure. **Cross-group fairness**: the three fast groups within 1.8× of each other. Plus exactly-once, pinned isolation, full workload per worker, and the publication-run occupancy proxy ≤ `B/2` |
| `the_full_scale_workload_is_dispatched_exactly_once` | **16 RG × 32 jobs × 128 tasks = 65 536**, 16 pinned + 16 general workers | Exactly-once over all 65 536; pinned isolation; all 16 groups served; drained |
| `every_finalization_task_follows_its_job_and_is_dispatched_once` | 4 RG × 4 × 16, commit **and** cleanup armed | Each of the 16 finalizations dispatched exactly once with the armed kind; every finalization's assignment ID strictly greater than every regular assignment ID of the same job; no unarmed finalization; the cleanup lane is non-empty |
| `a_mid_run_session_bump_replays_every_unfinished_task` | 4 RG × 4 × 64, bump at 200 ms | The bump landed before drain; new session ID; coverage of every task; duplicates bounded; no task dispatched three times; drained |

The exactly-once assertion is one shared helper. On failure it prints which tasks were missing and which were duplicated, because a bare count mismatch is not diagnosable at 65 536 tasks.

## 5. Design corrections made during implementation

Two errors in the specification were found by building against it. Both are corrected in `design.md`.

### 5.1 The hint channel cannot be statically bounded

Design §3.5 and §8.1 originally claimed outstanding hints are bounded by `dispatch_queue_capacity`. **That is false.** `H` is decremented only by a *general* execution manager, so a pinned execution manager draining a group lowers `S` while leaving `H` at that group's old peak. The real bound is `Σ_r peak(S_r)`, and those peaks are not simultaneous: a group backlogged alone reaches `B/2` under α = 1, so `N` groups going backlogged in turn — each drained by its own pinned execution managers — can leave up to `N·B/2` hints outstanding with the buffer empty.

A bounded channel would have rejected a send and broken the coverage invariant silently, with no way to detect or repair it.

### 5.2 No channel in this design is bounded by its type

Following from the above, and applied to the per-group queues as well: the **algorithm** is responsible for bounding these structures, not the channel type. The admission threshold `S < F` is the sole limit on a group's occupancy and hence on the buffer's; the publishing rule `H < S` is the sole limit on a group's hints.

A per-group channel capacity would be a redundant second bound measured against the wrong quantity — a per-group limit against a buffer-wide budget — whose only reachable effect is to reject a send the proof requires to succeed. Both the per-group queues and the hint channel are therefore unbounded, and `ResourceGroupTable::new` takes no capacity argument.

## 6. Known weaknesses in the current tests

Recorded honestly, because a passing suite that overstates its coverage is worse than a smaller one.

**`a_slow_resource_group_neither_blocks_nor_takes_over_the_buffer` now measures non-interference directly, but the measurement sits on a noisy base.**

The scenario was rewritten after an earlier version was found not to measure the property it guards. That version ran 48 tasks per group, so the fast groups finished in ~48 ms while the slow group ran for 2.4 s: for roughly 98% of the run only **one** group was backlogged, which is exactly the regime in which its `B/2` bound — the lone-group ceiling from §6.3 — is trivially satisfied. It also asserted only that each worker *received* its workload, which is liveness, not throughput. §1.1's non-interference guarantee was not measured at all.

The current version raises the workload to 256 tasks per group and measures per-group **mean job completion time**, timestamped by the workers after execution against one run-start origin. Because the fast groups now run for ~650 ms while the slow group runs for ~2.9 s, the entire measurement falls inside a window where all four groups are backlogged.

What remains weak:

- **The absolute numbers are dominated by harness overhead, not by scheduling.** A fast group's 256 tasks represent 256 ms of simulated execution but complete in ~650 ms, so roughly 1.5 ms per dispatch is gRPC and runtime cost in a debug build. The baseline-relative comparison is unaffected — both runs pay the same overhead — but the assertion's sensitivity is: catching a 2.0× slowdown on ~650 ms means catching starvation of more than twice the intrinsic work time, not a subtle one.
- **The cross-group spread bound of 1.8× is coarse by necessity.** In the *baseline* runs, where all four groups are identical and no slow group exists, the three groups still spread by 1.1–1.6×. A tighter bound would measure harness noise rather than scheduler fairness. Part of that variance is structural: with only two jobs per group, the metric swings by ~33% depending on whether the core drains a group's two jobs concurrently or one after the other, which is itself sensitive to whether a job transiently runs dry and is downgraded. Averaging over more, smaller jobs (8 × 32 instead of 2 × 128, same 256 tasks) was measured to cut the tail from 1.35 to 1.23.
- **The occupancy half is still a proxy.** The publication-run bound infers queue depth from the order assignments were published in rather than reading occupancy, because the integration suite is deliberately black-box through gRPC. It is now secondary evidence rather than the scenario's primary argument, but it is still checked against `B/2` where four backlogged groups predict `B/(N+1) ≈ 6`.

**Other gaps in strength:**

- The session-bump scenario replaces exactly-once with coverage plus a duplicate ceiling. That is the correct call — exactly-once is not a property of a session bump, which legitimately replays work — but the ceiling is loose relative to the scenario's 1 024 tasks.
- No full-scale run has finalization enabled, so the finalize queue's interaction with the admission threshold at depth is untested.
- Group-coverage assertions fire on "served at least once", never on proportional service. §11 makes rate an explicit non-goal, so this matches the design, but no test would catch a regression that served one group far less than another while still draining it.

## 7. Design properties with no test coverage

**Concurrency and the coverage proof (§8.1, §8.2).** The invariant `C = H + m ≥ q` is not instrumented. Every hint test is a single-threaded publish/pop sequence, and `m` — execution managers in flight between the decrement and the pop attempt — has no representation in any test. The integration scenarios observe only the consequence (nothing was lost) under a workload that never contends hard enough to expose a lost-wakeup window.

This is a deliberate position rather than an oversight: the mathematical proof is the primary artifact, and §3 above records the code site realizing each of its steps so the correspondence can be audited by reading. Note in particular that §8.2 req. 1's ordering is enforced structurally but asserted nowhere — reversing those two lines would still pass all 48 tests under light load.

**Proportional hint steering (§11, §1.1).** Nothing tests that general capacity flows *toward* a backlogged group. `H` tracks `S`, so a group holding 100 assignments should attract ~100× the hint representation of one holding a single assignment — the longest-queue-first behaviour the design names as what makes the "pinned execution managers went offline" case self-correcting without a health signal. The two halves of that claim are covered separately and never together: `unpinned_resource_groups_drain_through_general_workers` has general workers but no backlog asymmetry, since every group runs at the same speed, so it proves reachability rather than proportionality; `a_slow_resource_group_neither_blocks_nor_takes_over_the_buffer` has the asymmetry but no general workers to steer. The missing scenario is a slow or absent pinned worker on one group **plus** general workers, asserting the general workers' dispatches skew toward the backlogged group rather than splitting evenly.

**Core filtering paths.**

- §3.2's "a regular task belonging to a finalized job is dropped" is unreached. `FakeStorage` arms a finalization only after every regular task of that job is reported complete, so a regular task for a finalized job never arrives. This is the filter §3.2 says exists *specifically* because storage and the reschedule queue can deliver one.
- Duplicate finalization entries being idempotent is likewise unreached.
- §5.2's "finalizations processed before regular tasks, so a same-batch regular task is discarded" — the ordering exists in code, but no test constructs a batch containing both.
- §3.6's reschedule-queue filters: neither "a rescheduled assignment carrying a stale session is dropped" nor "a rescheduled regular task for a finalized job is dropped" is asserted.

**Session bump.** Step 5, draining the hint channel, is never asserted; step 2's `rg_table.clear()` is only implied. §7.1's "the reader is re-fetched per request, so a bump that replaces `rg_table` is picked up on the next call" is exercised incidentally but not asserted.

**Step 1.** Per-lane fetch counts derived from remaining buffer capacity — no test asserts the core asks storage for fewer items when its buffer is full. This is the back-pressure path and it matters directly for the performance stage.

**Configuration.** `tick_interval_ms` is never asserted to govern anything, and `active_job_list_capacity` is exercised only indirectly through observed promotion behaviour — no test asserts `active_jobs.len() ≤ capacity` as an invariant.

**Accepted non-properties**, listed so they are not mistaken for gaps: §6.3's absence of push-out (a group that stops draining entirely is never reclaimed) and all four of §11's non-goals are untested by design.

## 8. Follow-ups

- `Core::finalized_jobs` grows one `JobId` per finalized job for the lifetime of a session and is cleared only on a session bump. Design §3.2/§9 specify this, but it needs an eviction path before the prototype folds into `spider-scheduler`.
- `RgSchedulingUnit::restore` returns a regular task to the head of its job entry without rewinding `rr_arm`, so a rejected publication perturbs the intra-group rotation by one slot. Harmless — the task is not lost — and the path is unreachable while the queue is open.
- The `Harness` no longer exposes a reschedule-queue writer. An execution-manager-loss replay scenario would need it back.

## 9. Not measured

No performance claim is made or implied here. The full-scale scenario's wall-clock time is a debug build in which workers sleep 1 ms per task; it measures the sleep, not the scheduler. Metrics, workload shape, and methodology for the performance stage are still to be decided.

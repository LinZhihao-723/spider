# Prototype Scheduler Performance Evaluation — Run 003

This is the report required by [benchmark-contract.md](../../benchmark-contract.md) §4.2, for run 003 of the five cases A–E of §2. It is self-contained: everything needed to read it is here, and no part of it depends on [run-002's report](../run-002/report.md) or [run-001's](../run-001/report.md). Those runs are referenced only where the comparison is the point, which in this run is most of §4.

Run 003 exists to answer one question: **did the ownership refactor — replacing the core's `Rc<RefCell<…>>` scheduling state with arenas the core owns outright — change what the scheduler costs?** The refactor was made for architecture, not for speed (§1), so the expected answer is "no", and the report's job is to test that rather than to assume it.

Every run-003 number below was recomputed from the raw JSON in this directory: percentiles over the per-tick and per-job series from the raw samples by nearest rank; dispatch-latency percentiles from the harness's stored histogram percentiles, which use the same nearest-rank convention reported at the containing bucket's upper bound. **Run 001's and run 002's raw JSON are not on disk in this worktree** — those directories contain only their reports — so every run-001/002 figure quoted here is read out of those reports rather than recomputed. That is a real limitation on the comparison and §8.7 says what it costs.

The headline, stated up front:

- **The refactor is performance-neutral.** `fill` — the step that lost the refcount traffic, the borrow-flag checks and the pointer chasing, and so the step where a change should show — moved by −7.2%, −2.7%, −2.1% and +8.1% per assignment in the four cases that are not supply-confounded. Three of those four are inside the ±1–3% noise band the two prior runs showed between themselves at unchanged code paths, or favour run 003; the fourth (case B) sits inside a case-wide shift that also moved the loopback transport, which the refactor cannot touch.
- **The whole tick loop costs 421–605 ns per assignment published**, against run 002's 428–620 ns for the same four cases. Total per assignment: A −0.6%, C −1.7%, E −2.4%, B +10.1%.
- **Dispatch latency did not move materially**: server-side immediate median 0.343–0.631 µs and p99 1.17–2.08 µs, against run 002's 0.327–0.711 µs and 1.14–2.18 µs.
- **The one step that rose consistently in all five cases is `retire`, and it is not the refactor.** Its rise is a *floor* shift: 93%–97% of ticks retire nothing at all and cost only a clock-read pair there, and it is exactly those ticks whose cost went up (p50 53–72 ns → 90–98 ns). §4.4.
- **Case D's `fill` is 27.8% dearer per assignment, and that is supply, not code.** Run 003's harness delivered inbound tasks ~8% slower than run 002's (30 200–31 500 tasks/s against 31 900–33 400), case D is the one case pinned against that ceiling, its waited-request share rose from 78.3% to 84.3%, and run 002 §8.3 already established that a larger parked-receiver population makes `fill`'s publishes dearer. Case E — same 8 resource groups, same 1 048 576 tasks, not starved — got 7.2% *cheaper*, which is what rules out an 8-group scaling problem in the new code. §8.1.
- **Correctness is perfect in all five cases**: every one of 2 944 jobs completed exactly 1024 tasks, four independent counts agree exactly per case, zero empty responses, zero `WARN` and zero `ERROR` lines in all ten process logs.
- **The whole measurement session ran ~4–8% slower than run 002's**, on evidence that has nothing to do with the scheduler: the timer-driven tick cadence widened by a near-identical +4.2% to +4.4% in all five cases, and the harness's inbound delivery rate fell ~5–8%. §4.5 defines that common-mode band and §4.6 uses it.

## 1. What changed since run 002

Run 002 measured the prototype whose core **co-owned** its scheduling state. Job entries were `Rc<SharedJobEntryInner>` values with a `RefCell` interior and a `Cell<bool>` `finalized` flag; resource-group scheduling units were `Rc<RefCell<RgSchedulingUnit>>` values held in a `HashMap` and cloned into an active list. Because those `Rc`s were held across await points, the core's future was `!Send`, which the code acknowledged with two `#[allow(clippy::future_not_send)]` attributes and a `LocalSet` on a dedicated thread. The existing `spider-scheduler` runtime cannot spawn a `!Send` future, so the prototype could not be adopted as written.

Run 003 measures the same prototype after that ownership model was replaced:

| | Run 002 | Run 003 |
|---|---|---|
| Job entries | `Rc<SharedJobEntryInner>` with `RefCell` interior, co-owned by the registry and one scheduling position | A `slotmap::SlotMap<JobKey, JobEntry>` arena owned solely by the registry; scheduling positions hold a generational `JobKey` |
| Job removal | `finalize_and_remove` drops the registry's `Rc`; positions keep theirs alive | `remove_by_job_id` / `remove(job_key)` frees the slot; a position's key then fails to resolve, which is how it discovers the job is gone |
| "Job is finalized" | A `Cell<bool>` flag on the shared entry, checked on every `get_next_task`, plus a `JobEntryError::Finalized` variant | Gone. Finalization removes the entry, so the stale-key resolution failure carries the same information |
| Resource-group units | `HashMap<ResourceGroupId, Rc<RefCell<RgSchedulingUnit>>>`, active list of `Rc` clones | `Vec<RgSchedulingUnit>` (append-only within a session, per design §3.4) plus a `HashMap<ResourceGroupId, usize>` index; active list and round-robin list hold `usize` |
| `fill`'s inner loop | `Rc::clone` per group per pass, `borrow_mut()` per assignment, `Rc::ptr_eq` scan on deactivation | Index into a contiguous `Vec`, one `SlotMap` lookup per assignment, `usize` equality on deactivation |
| Core's future | `!Send`; `LocalSet`; two `#[allow(clippy::future_not_send)]` | `Send`; no `LocalSet`; both allows removed |

Two things follow, and they set what the reader should expect from the rest of the report.

**This was a correctness and architecture change, not a performance change.** Its purpose was to make the core spawnable on the existing runtime and to state two design invariants in the type system — that a job entry's identity is generational, so a stale reference to a removed job cannot silently alias a new one, and that resource-group units are append-only within a session, so their indices are stable. Nothing about it was undertaken to make the scheduler faster, and no number in this report should be read as a claim that it did.

**A change should nonetheless be *visible* in `fill` if anywhere.** Step 4 is the only step that touches the changed data structures once per assignment: it lost one `Rc` clone per group per pass, one `RefCell` borrow-flag check per assignment, and one `finalized` flag read per task, and gained one `SlotMap` lookup per assignment; its traversal of the active group list became contiguous indexing instead of pointer chasing. §4.2 and §4.3 are that measurement. Step 5 (`retire`) is the only other step whose work changed shape — a slot free instead of an `Rc` drop — and §4.4 is about why its numbers moved for a different reason than that.

The harness, the configuration, the binaries' build profile, the measurement instrumentation and the host are otherwise identical to run 002. In particular the core still runs on a dedicated OS thread under a current-thread runtime — that is now a measurement choice rather than a requirement, but it was kept, so the two runs' cores execute in the same environment.

The refactor also had to leave the crate's existing unit and integration tests passing; that gate is outside this report, which measures only the five benchmark cases and validates them through §6.

## 2. Terminology

### 2.1 The three request classes

Every `NextTask` request falls into exactly one class, and the class determines whether and where it is timed ([benchmark-contract.md](../../benchmark-contract.md) §1.2):

| Class | Meaning | Treatment |
|---|---|---|
| **Immediate** | An assignment was available when the handler was entered; the handler returned it without ever awaiting | **Timed.** This is dispatch cost |
| **Waited** | The handler found the queue empty, awaited, and an assignment arrived before the wait expired | **Timed separately.** Dominated by supply, not by the scheduler |
| **Empty** | No assignment arrived before the wait expired | Counted only, never timed. Zero in all five cases |

**Only the immediate class answers "is the server side under control".** A waited request's duration is set by when the next task is published into the queue it is parked on, which is a property of how fast work is arriving. Pooling the two makes the server-side mean a function of the harness's supply rate — the defect run 002 existed to remove.

**"Waited" means the handler entered an await, not that it waited long.** The flag is set the moment the non-blocking pop misses, so a request whose await resolves in half a microsecond is still a waited one; case D's server-side pinned waited series has a minimum of 0.551 µs and case B's of 41.0 µs. The class is a clean partition by control flow, which is what makes the immediate series interpretable.

### 2.2 Client side and server side

- **Server-side** is measured in the service handler around the `DispatchService::next_task_*` call. It excludes transport and serialization, and it is the trustworthy measure of what the scheduler costs per request.
- **Client-side** is measured in the worker coroutine around the gRPC call, immediately before the request to immediately after the response. It includes transport, serialization and the server time, so it is always the larger of the two.

`client − server` **over the immediate class only** is the transport and framing overhead; on this single-node run it is loopback cost, recorded so a later multi-node run has a baseline. Taking the difference over a pooled series would subtract two different mixtures of two populations.

Both are reported separately for **dedicated (pinned)** and **shared (general)** execution managers, because the paths differ structurally: a pinned request touches one resource group's queue and nothing else, while a general request pops the hint channel and may traverse several stale hints before it finds work.

### 2.3 Tick step timings

The core's tick has the five steps of [design.md](../../design.md) §5, and every tick is timed per step and in total:

| Key | Step | Covers |
|---|---|---|
| `collect` | 1 | Draining the inbound poll result and the reschedule queue |
| `process` | 2 | Deduplicating and grouping the polled entries into `rg_updates` |
| `apply` | 3 | Creating scheduling units, placing new jobs, and starting the next inbound poll |
| `fill` | 4 | The round-robin admission loop — the scheduling policy itself |
| `retire` | 5 | Removing retired jobs from the job registry |
| `total` | — | The whole tick |

Three accounting details are baked into the data and must be read with it, unchanged from runs 001 and 002:

- `start_inbound_poll` is billed to `apply` on a tick that drained a poll result, and to `collect` on a tick that had no poll in flight.
- Steps 2 and 3 are skipped entirely, and record zero, on any tick where the inbound poll is still in flight. That is why `process` and `apply` have a median of 0 in every case. Their means over all ticks are still the right per-tick cost; their medians are not.
- A poll-draining tick takes seven `Instant::now()` readings rather than six, so `total` exceeds the sum of the five steps by about one clock read: a mean residual of 65.8–84.7 ns per case in this run (run 002: 56–86 ns), never negative on any of the 70 929 ticks recorded across run 003's five cases, with a per-case minimum of 23–43 ns.

`per assignment` divides a step's total time over the whole run by the run's total assignments published, and is the figure to compare across cases and across runs; per-tick figures are not comparable between cases because assignments per tick differ by an order of magnitude, and they are sensitive to the tick cadence (§8.4). `busy mean` restricts to ticks that published at least one assignment.

### 2.4 Per-job end-to-end time

**First seen** is when the fake inbound queue first emits a task belonging to the job; **last completed** is when the dispatch service received the last completion report for a task of that job; **E2E** is the difference.

**A job's E2E includes one extra round trip.** A worker reports a task's completion on its *next* `NextTask` request, so the last completion of a job is observed one request cycle after that task actually finished. Every E2E figure in §7 is inflated by one client-side round trip. It is also why the run does not end when the last task is dispatched: it ends when the last completion is received, which is what the harness's 500 ms drain linger flushes.

E2E is **not** a per-task service-time measure. It spans a job's whole emission-to-completion life, including queueing behind the backlog the core has buffered, so its distribution is shaped mostly by the batch release rule (§3) — see §7.1. The completion count per job doubles as the run's validity gate (§6).

### 2.5 Histogram resolution

Dispatch latency is stored as a fixed-bucket atomic histogram of 1984 buckets: the first 64 ns resolved to the nanosecond, every octave above that split into 64 buckets, giving ≤1.6% relative error above 64 ns. Percentiles use nearest rank reported at the containing bucket's upper bound, so a quoted percentile is never understated; count, mean, min and max are exact, maintained separately from the buckets.

One consequence to expect in the small waited series: a quoted percentile can exceed the recorded maximum, because the percentile is a bucket upper bound while the maximum is exact. Case A's pinned waited p99 is 13 107.199 µs against a max of 13 018.814 µs. This is the convention, not a data error. A second consequence, visible in case C, is that two small series can land on an identical bucket boundary at the coarse high end and report the same percentile by coincidence (§8.6).

## 3. Configuration

Constant across every case, unchanged from run 002:

| Parameter | Value |
|---|---|
| Jobs per resource group | 128, released in **4 batches of 32** |
| Tasks per job | 1024 |
| Task execution time | 5 ms, simulated by `tokio::time::sleep` in the worker |
| Inbound wave size | 256 tasks per poll response |
| `active_job_list_capacity` | 16 |
| `tick_interval_ms` | 1 |
| `storage_poll_timeout_ms` | 5 |
| `ready_task_capacity` | the case's total task count, so the core buffers everything and no task waits in the inbound queue |
| Dedicated workers per group | 32 (0 in case E) |
| Shared workers | 8 × #RG (128 in case E) |
| `channel_pool_size` | **the case's worker count — one gRPC channel per worker coroutine**, verified per case from the worker result file |

The five cases:

| Case | #RG | Total tasks | Shared | Dedicated | Total workers | Reserve `R` | `B` |
|---|---|---|---|---|---|---|---|
| A | 1 | 131 072 | 8 | 32 | 40 | 40 | 80 |
| B | 2 | 262 144 | 16 | 64 | 80 | 40 | 120 |
| C | 4 | 524 288 | 32 | 128 | 160 | 40 | 200 |
| D | 8 | 1 048 576 | 64 | 256 | 320 | 40 | 360 |
| E | 8 | 1 048 576 | 128 | 0 | 128 | 16 | 144 |

**Deriving `B`, the dispatch queue capacity.** At least `R` slots must stay free when every resource group is active, where `R = (shared workers / #RG) + (dedicated workers per group)`. Under the admission policy of [design.md](../../design.md) §6.2 with α = 1 and `N` backlogged groups, free space settles at `F = B / (N + 1)`, so `F ≥ R` gives `B = R × (N + 1)`, which is the `B` column. Case E has no dedicated workers, so `R = 128 / 8 = 16` and `B = 16 × 9 = 144`. The rule puts equilibrium occupancy at roughly one execution-manager round with no margin, which run 002 found runs out at 320 execution managers; run 003 reproduces that and slightly worse (§8.1).

**Batch release rule.** Batch `n+1` is released once every job in batch `n` has had all of its tasks reported complete. This is an *interpretation* of "jobs are created in batches" rather than something the design specifies, and it is recorded because it materially shapes the results: it is the direct cause of the E2E distribution shapes in §7. With 32 jobs per batch against an active list of 16, there is always a pending queue.

**Expected steady-state duration** is `total_tasks × 5 ms / total_workers`: 16.384 s for A–D and 40.960 s for E. A case whose measured duration greatly exceeds this is worker-starved or harness-bound, and its wall clock is not a scheduler measurement.

### 3.1 What was run

| Case | #RG | Total tasks | Workers (ded. + shared) | `B` | Pool | Wall clock (s) | Expected (s) | Ratio | Throughput (tasks/s) | Ticks | Busy ticks | Core duty cycle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 1 | 131 072 | 32 + 8 = 40 | 80 | 40 | 21.617 | 16.384 | 1.32× | 6 063 | 9 983 | 5 719 | 0.476% |
| B | 2 | 262 144 | 64 + 16 = 80 | 120 | 80 | 21.674 | 16.384 | 1.32× | 12 095 | 10 035 | 6 565 | 0.672% |
| C | 4 | 524 288 | 128 + 32 = 160 | 200 | 160 | 22.071 | 16.384 | 1.35× | 23 755 | 10 216 | 7 840 | 0.999% |
| D | 8 | 1 048 576 | 256 + 64 = 320 | 360 | 320 | 34.500 | 16.384 | **2.11×** | 30 393 | 15 790 | 4 147 | 1.722% |
| E | 8 | 1 048 576 | 0 + 128 = 128 | 144 | 128 | 54.598 | 40.960 | 1.33× | 19 205 | 24 905 | 18 663 | 1.161% |

Wall clock is the scheduler-internal measure of §2.4 — first inbound emission to last completion received, plus the 500 ms drain linger — as in run 002's equivalent table, so the two are directly comparable. The script-level wall clock, which brackets worker-process launch and teardown instead, was 22.030 / 20.132 / 20.548 / 32.913 / 52.995 s and is recorded in `<case>-run.json`; **the two clocks are not nested** (§8.9). Core duty cycle is the sum of every tick's `total_nanos` over the scheduler-internal wall clock.

Against run 002:

| Case | Wall 002 (s) | Wall 003 (s) | Δ | Ratio 002 | Ratio 003 | tasks/s 002 | tasks/s 003 | Δ | Duty 002 | Duty 003 |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 21.465 | 21.617 | +0.7% | 1.31× | 1.32× | 6 106 | 6 063 | −0.7% | 0.483% | 0.476% |
| B | 21.327 | 21.674 | +1.6% | 1.30× | 1.32× | 12 292 | 12 095 | −1.6% | 0.621% | 0.672% |
| C | 21.722 | 22.071 | +1.6% | 1.33× | 1.35× | 24 137 | 23 755 | −1.6% | 1.033% | 0.999% |
| D | 32.584 | 34.500 | +5.9% | 1.99× | **2.11×** | 32 181 | 30 393 | −5.6% | 1.585% | 1.722% |
| E | 53.411 | 54.598 | +2.2% | 1.30× | 1.33× | 19 632 | 19 205 | −2.2% | 1.217% | 1.161% |

No case is a scheduler measurement at the wall-clock level: A, B, C and E overshoot by 1.32×–1.35× for the reason run 002 established (the worker's own request cycle, a ~6 ms effective sleep plus a round trip), and case D is inbound-supply-bound (§8.1). The 0.7%–2.2% slowdown in A, B, C and E is the common-mode drift of §4.5, not a cost the core imposed: the core spent 0.48%–1.72% of one core inside `tick()`, and 103–634 ms of CPU across runs lasting 22–55 s.

Environment: single host, Linux 6.6.87.2 under WSL2, 32 cores, `--release` build, scheduler and workers as two processes on the same machine over loopback `http://127.0.0.1:50151`, one worker process hosting all worker coroutines. Cases ran strictly sequentially, one run each, no retries, each gated on an idle-machine check immediately before launch (§8.8). Neither process log contains a single `WARN`, `ERROR` or panic line in any case.

Files: `<case>-scheduler.json` (config, every tick sample, every per-job record, the four server-side histograms), `<case>-workers-0.json` (config, channel pool size, the four client-side histograms, request tallies), `<case>-run.json` (script-level wall clock), `logs/`.

## 4. Tick step timings

All times in nanoseconds. Each case's table gives run 003's full step profile, with run 002's `mean`, `p99` and `per assignment` beside it; `Δ/asg` is the change in per-assignment cost, which is the cadence-independent comparator.

### Case A — 9 983 ticks, 5 719 busy, 131 072 assignments, 2.165 ms mean spacing, 22.9 assignments per busy tick

| Step | mean 003 | mean 002 | p50 003 | p99 003 | p99 002 | max 003 | busy mean 003 | /asg 003 | /asg 002 | Δ/asg | share 003 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `collect` | 1 024 | 973 | 362 | 5 110 | 5 368 | 95 131 | 1 052 | 78.0 | 76.8 | +1.6% | 9.9% |
| `process` | 1 585 | 1 436 | 0 | 31 296 | 26 551 | 1 041 748 | 1 538 | 120.7 | 113.4 | +6.4% | 15.4% |
| `apply` | 650 | 645 | 0 | 3 983 | 4 318 | 19 094 | 647 | 49.5 | 50.9 | −2.7% | 6.3% |
| `fill` | 6 849 | 6 789 | 2 931 | 25 538 | 25 768 | 104 621 | 11 446 | **521.7** | **536.0** | **−2.7%** | 66.4% |
| `retire` | 122 | 83 | 96 | 416 | 315 | 23 800 | 132 | 9.3 | 6.6 | +41% | 1.2% |
| `total` | 10 312 | 10 012 | 6 734 | 45 481 | 41 546 | 1 053 397 | 14 896 | **785.4** | **790.5** | **−0.6%** | 100% |

### Case B — 10 035 ticks, 6 565 busy, 262 144 assignments, 2.160 ms mean spacing, 39.9 assignments per busy tick

| Step | mean 003 | mean 002 | p50 003 | p99 003 | p99 002 | max 003 | busy mean 003 | /asg 003 | /asg 002 | Δ/asg | share 003 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `collect` | 896 | 789 | 312 | 4 642 | 4 659 | 22 090 | 895 | 34.3 | 31.0 | +10.6% | 6.2% |
| `process` | 2 690 | 2 265 | 0 | 32 468 | 27 775 | 1 291 953 | 2 816 | 103.0 | 89.0 | +15.7% | 18.5% |
| `apply` | 611 | 528 | 0 | 3 659 | 3 491 | 18 704 | 602 | 23.4 | 20.8 | +12.5% | 4.2% |
| `fill` | 10 125 | 9 127 | 5 788 | 38 289 | 34 801 | 126 229 | 15 126 | **387.6** | **358.7** | **+8.1%** | 69.7% |
| `retire` | 117 | 74 | 94 | 676 | 487 | 11 333 | 121 | 4.5 | 2.9 | +55% | 0.8% |
| `total` | 14 523 | 12 845 | 9 933 | 55 420 | 48 586 | 1 307 345 | 19 635 | **555.9** | **504.9** | **+10.1%** | 100% |

### Case C — 10 216 ticks, 7 840 busy, 524 288 assignments, 2.160 ms mean spacing, 66.9 assignments per busy tick

| Step | mean 003 | mean 002 | p50 003 | p99 003 | p99 002 | max 003 | busy mean 003 | /asg 003 | /asg 002 | Δ/asg | share 003 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `collect` | 867 | 838 | 265 | 5 235 | 5 188 | 40 100 | 851 | 16.9 | 16.7 | +1.2% | 4.0% |
| `process` | 4 563 | 4 553 | 0 | 37 803 | 39 397 | 1 627 167 | 4 732 | 88.9 | 91.0 | −2.3% | 21.1% |
| `apply` | 549 | 527 | 0 | 3 502 | 3 613 | 18 643 | 542 | 10.7 | 10.5 | +1.9% | 2.5% |
| `fill` | 15 418 | 15 359 | 12 074 | 62 207 | 60 784 | 195 039 | 19 877 | **300.4** | **307.0** | **−2.1%** | 71.4% |
| `retire` | 122 | 84 | 90 | 939 | 993 | 7 951 | 118 | 2.4 | 1.7 | +41% | 0.6% |
| `total` | 21 585 | 21 417 | 18 798 | 81 912 | 79 462 | 1 655 572 | 26 184 | **420.6** | **428.0** | **−1.7%** | 100% |

### Case D — 15 790 ticks, 4 147 busy, 1 048 576 assignments, 2.185 ms mean spacing, 252.9 assignments per busy tick

| Step | mean 003 | mean 002 | p50 003 | p99 003 | p99 002 | max 003 | busy mean 003 | /asg 003 | /asg 002 | Δ/asg | share 003 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `collect` | 1 041 | 956 | 301 | 7 886 | 5 673 | 52 107 | 2 952 | 15.7 | 14.2 | +10.6% | 2.8% |
| `process` | 5 460 | 7 367 (5 265) | 0 | 46 794 | 37 042 | 222 322 | 20 558 | 82.2 | 109.3 (78.1) | +5.2% vs excl. | 14.5% |
| `apply` | 792 | 787 | 0 | 6 527 | 5 012 | 33 640 | 2 928 | 11.9 | 11.7 | +1.7% | 2.1% |
| `fill` | 29 750 | 23 618 | 1 893 | 250 750 | 134 406 | 828 275 | 109 937 | **448.0** | **350.5** | **+27.8%** | 79.1% |
| `retire` | 507 | 401 | 98 | 4 602 | 3 215 | 60 275 | 112 | 7.6 | 5.9 | +29% | 1.3% |
| `total` | 37 629 | 33 191 (31 091) | 3 343 | 295 319 | 174 856 | 920 736 | 136 564 | **566.6** | **492.6** (461.4) | +15.0% (+22.8%) | 100% |

Run 002's case D contained a single 32.7 ms `process` outlier that contributed 28% of its `process` mean, so its report gave both raw and outlier-excluded figures; the parenthesized run-002 values above are the excluded ones, and they are the fair comparator. **That outlier did not recur** — run 003's largest `process` tick in case D is 222 µs — so run 003's case-D figures need no excluded variant. Case D is supply-confounded throughout (§8.1) and its `fill` row is the subject of §4.3.

### Case E — 24 905 ticks, 18 663 busy, 1 048 576 assignments, 2.192 ms mean spacing, 56.2 assignments per busy tick

| Step | mean 003 | mean 002 | p50 003 | p99 003 | p99 002 | max 003 | busy mean 003 | /asg 003 | /asg 002 | Δ/asg | share 003 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `collect` | 873 | 767 | 281 | 4 570 | 4 570 | 62 795 | 879 | 20.7 | 18.6 | +11.3% | 3.4% |
| `process` | 5 583 | 4 820 | 0 | 49 246 | 45 163 | 3 878 389 | 5 878 | 132.6 | 116.9 | +13.4% | 21.9% |
| `apply` | 532 | 534 | 0 | 3 075 | 3 396 | 72 638 | 533 | 12.6 | 12.9 | −2.3% | 2.1% |
| `fill` | 18 287 | 19 294 | 13 422 | 65 856 | 67 728 | 257 326 | 24 118 | **434.3** | **468.0** | **−7.2%** | 71.8% |
| `retire` | 108 | 73 | 91 | 469 | 353 | 58 235 | 111 | 2.6 | 1.8 | +44% | 0.4% |
| `total` | 25 455 | 25 549 | 21 525 | 92 877 | 89 222 | 3 920 828 | 31 589 | **604.6** | **619.7** | **−2.4%** | 100% |

### 4.1 The scaling trend across #RG

| Quantity | A (1 RG) | B (2 RG) | C (4 RG) | D (8 RG) | E (8 RG, no dedicated) |
|---|---|---|---|---|---|
| `fill` share of tick | 66.4% | 69.7% | 71.4% | 79.1% | 71.8% |
| `retire` share of tick | 1.2% | 0.8% | 0.6% | 1.3% | 0.4% |
| `fill` per assignment (ns) | 521.7 | 387.6 | 300.4 | 448.0 | 434.3 |
| `process` per inbound entry (ns) | 120.7 | 103.0 | 88.9 | 82.2 | 132.6 |
| `collect` per assignment (ns) | 78.0 | 34.3 | 16.9 | 15.7 | 20.7 |
| `apply` per assignment (ns) | 49.5 | 23.4 | 10.7 | 11.9 | 12.6 |
| `retire` per assignment (ns) | 9.3 | 4.5 | 2.4 | 7.6 | 2.6 |
| `total` per assignment (ns) | 785.4 | 555.9 | 420.6 | 566.6 | 604.6 |
| Assignments per busy tick (mean) | 22.9 | 39.9 | 66.9 | 252.9 | 56.2 |
| Mean active resource groups | 0.975 / 1 | 1.951 / 2 | 3.900 / 4 | **2.158 / 8** | 7.921 / 8 |
| Total core time in `tick()` (ms) | 102.9 | 145.7 | 220.5 | 594.2 | 634.0 |

**Step 4 dominates and step 5 is near zero, exactly as the contract predicts, and the refactor did not change that shape.** `fill` is 66.4%–79.1% of tick time; `retire` never exceeds 1.3% and costs 2.4–9.3 ns per assignment. Run 002's equivalents were 67.8%–75.5% and 0.3%–1.2%.

**Nothing scales super-linearly in #RG.** From 1 to 8 resource groups with 8× the tasks and 8× the workers, the whole tick loop's cost per assignment fell from 785 ns to 567–605 ns, and total core time grew 5.8× for 8× the work. Per-assignment cost falls because a per-tick fixed cost is amortized over more assignments.

A per-case least-squares fit of `fill_nanos` against `assignments_published` over every tick separates the fixed and marginal parts, and it is the cleanest way to compare `fill` across runs because it is immune to the assignments-per-tick differences that the cadence drift introduced (§4.5):

| Case | Fixed 003 (ns/tick) | Fixed 002 | Marginal 003 (ns/asg) | Marginal 002 | R² 003 | R² 002 | Marginal 003, excl. 6 largest |
|---|---|---|---|---|---|---|---|
| A | 992 | 1 022 | **446** | **455** | 0.85 | 0.12 (0.85 excl.) | 444 |
| B | 1 088 | 1 190 | **346** | **312** | 0.87 | 0.86 | 345 |
| C | 1 179 | 1 593 | **277** | **275** | 0.80 | 0.79 | 275 |
| D | 1 145 | 1 067 | 431 | 335 | 0.65 | 0.80 | 427 |
| E | 1 505 | 1 863 | **399** | **423** | 0.88 | 0.86 | 397 |

Case D's fit should not be trusted for its fixed/marginal split, in either run: D's `assignments_published` is exactly 256 from p10 through p99, so the regression has almost no leverage, and run 003's R² of 0.65 is the lowest in the set. The robust equivalent for D is `fill` busy mean over assignments per busy tick, 109 937 / 252.9 = **435 ns per assignment** (run 002: 85 455 / 252.5 = 338 ns).

Run 002's case-A fit had an R² of 0.12 because of a single 1.81 ms `fill` tick; run 003's case A has no such tick — its `fill` maximum is 104.6 µs — so its unfiltered R² is already 0.85 and dropping the six largest ticks moves the coefficients by under 1%.

### 4.2 Did `fill` move? — the question the run exists to answer

`fill` is where the refactor's cost change, if any, must appear. Four independent readings of it, three of which agree:

| Case | `fill` /asg 002 → 003 | `fill` marginal (fit) 002 → 003 | `fill` busy mean ÷ asg-per-busy-tick 002 → 003 | `fill` share of tick 002 → 003 |
|---|---|---|---|---|
| A | 536.0 → 521.7 (**−2.7%**) | 455 → 446 (−2.0%) | 512 → 500 (−2.3%) | 67.8% → 66.4% |
| B | 358.7 → 387.6 (**+8.1%**) | 312 → 346 (+10.9%) | 350 → 379 (+8.3%) | 71.1% → 69.7% |
| C | 307.0 → 300.4 (**−2.1%**) | 275 → 277 (+0.7%) | 303 → 297 (−2.0%) | 71.7% → 71.4% |
| D | 350.5 → 448.0 (**+27.8%**) | no leverage | 338 → 435 (+28.6%) | 71.2% → 79.1% |
| E | 468.0 → 434.3 (**−7.2%**) | 423 → 399 (−5.7%) | 463 → 429 (−7.3%) | 75.5% → 71.8% |

**Cases A and C are flat.** −2.7% and −2.1% per assignment, −2.0% and +0.7% on the marginal fit. Runs 001 and 002 differed by −14% to +3% on this same statistic at code paths neither of their fixes touched, and by ±1–3% on most of the tick-step figures; a 2% move is inside that band and means "unchanged".

**Case E is 5.7%–7.3% cheaper**, on 1 048 576 assignments and the best-conditioned fit in the set (R² 0.88). This is the largest unstarved case, at 8 resource groups — the configuration where contiguous indexing over `rg_units` should help most if it helps at all. It is a favourable result, but it is one case, it is barely outside the noise band, and it moves in the opposite direction to the session's common-mode drift (§4.5), so the honest statement is that `fill` did not get dearer and may have got slightly cheaper.

**Case B is 8.1%–10.9% dearer, and case B is the case where everything moved.** Its `collect` is +10.6%, `apply` +12.5%, `process` +15.7% and `total` +10.1% per assignment — and its *client-side loopback median* moved +12.8% (352.3 → 397.3 µs) and its server-side pinned immediate p50 +17%. The refactor touches none of the transport path, none of the gRPC handler, and none of the dispatch queue's pop; a change confined to the core's arenas cannot move the worker's round-trip median. A uniform double-digit shift across steps *and* transport is the signature of host conditions for that case, so case B's `fill` figure should be read as "this whole case ran ~10% slow", not as a `fill` regression.

**Case D is +27.8%, and §4.3 is why that is supply.**

The conclusion: **no case shows a `fill` regression that survives its own controls.** Three of the five are flat or better; one is confounded case-wide; one is supply-confounded.

### 4.3 Case D's `fill`, and why it is not the refactor

Case D's `fill` cost 448 ns per assignment against run 002's 350 ns. Three things say this is the harness's supply rate rather than the code:

- **Case D is pinned against the harness's inbound ceiling, and that ceiling was lower in run 003.** The implied inbound delivery rate, measured from how fast each batch's tasks were first seen, was 30 251–30 597 tasks/s in case D of this run against 33 000–33 200 in run 002 — 8.2% lower — and the same ~5–8% deficit appears in all five cases (§4.5). Case D's measured throughput of 30 393 tasks/s is **99.9% of its own inbound ceiling**; run 002's case D was at 97.3% of its. D is not slow, it is empty.
- **The parked-receiver population grew.** Case D's waited-request share rose from 78.29% to 84.26%. Run 002 §8.3 established the mechanism, when case D's `fill` rose 41% between run 001 and run 002 *with no change to the core at all*: each group's assignment queue is an `async_channel`, and publishing into a queue that has a receiver parked on it must wake that receiver, which a publish into an unwaited queue does not. Across three runs, case D's `fill` per assignment tracks its waited share monotonically and tracks nothing else: 241 ns at ~0% (run 001, where head-of-line queueing throttled worker demand), 338 ns at 78.3% (run 002), 435 ns at 84.3% (run 003).
- **Case E, the control, went the other way.** E has the same 8 resource groups, the same 1 048 576 tasks and the same core code as D, and it is *not* starved (0.037% waited). Its `fill` got 7.2% cheaper. If the new `rg_units` `Vec` or the `JobKey` resolution had made the round-robin loop dearer at 8 groups, E would show it; E shows the opposite.

What this run cannot do is *prove* the negative, because case D's supply rate is not held fixed between runs and this run did not vary it deliberately. The evidence is three consistent signatures and one control case, not a controlled experiment. The decisive experiment is the one run 002 already asked for and neither run has done: raise `inbound_wave_size` or issue overlapping polls so case D stops being supply-bound, then re-measure. Until then case D's wall clock, throughput, waited series and client-side series are harness measurements, and only its immediate-class dispatch latency and its per-assignment tick costs survive — the latter with this caveat attached.

### 4.4 `retire` rose in all five cases, and it is a floor shift, not a removal cost

This is the one step-level figure that moved consistently, and it is the one where the refactor's fingerprint would be most expected — step 5 is exactly where a `SlotMap` slot free replaced an `Rc` drop:

| Case | mean 002 | mean 003 | p50 002 | p50 003 | /asg 002 | /asg 003 | Δ/asg |
|---|---|---|---|---|---|---|---|
| A | 83 | 122 | 72 | 96 | 6.6 | 9.3 | +41% |
| B | 74 | 117 | 53 | 94 | 2.9 | 4.5 | +55% |
| C | 84 | 122 | 53 | 90 | 1.7 | 2.4 | +41% |
| D | 401 | 507 | 66 | 98 | 5.9 | 7.6 | +29% |
| E | 73 | 108 | 53 | 91 | 1.8 | 2.6 | +44% |

**It is not the removal cost, because the ticks that got dearer are the ticks that remove nothing.** Retirements are rare events — a job retires only when it exhausts its downgrade budget — and the distribution shows it: 93.1%–97.2% of ticks record a `retire` under 200 ns, with per-case minima of 19–37 ns and p10 of 54–56 ns. On such a tick, step 5 iterates an empty `Vec` and does nothing else, in both versions of the code; its measured cost is one `Instant::now()` pair. Yet it is precisely that population whose median moved, from 53–72 ns to 90–98 ns, a uniform +24 to +41 ns in all five cases. A data-structure change cannot make a no-op loop dearer.

The same floor shift is visible in the other cheap statistics. `collect`'s median rose in all five cases (342 → 362, 251 → 312, 244 → 265, 277 → 301, 243 → 281 ns), and `collect` was not touched by the refactor either. What both share is that they are near the resolution of a clock-read pair, which is the quantity §4.5 shows moved for this whole session.

In absolute terms the argument barely matters: `retire` totals 1.2 ms of CPU across case A's 21.6 s run, is 0.4%–1.3% of the tick everywhere, and costs 2.4–9.3 ns per assignment. Even taken at face value as a refactor cost it would be covered several times over by `fill`'s movement in the opposite direction. It is recorded here so that run 004 can check whether the floor returns to run 002's level, which is the observation that would settle it.

### 4.5 The common-mode band — how much of this session was the host

Three quantities in this run cannot be affected by the refactor, and all three moved together against run 002:

| Indicator | What it measures | Run 002 | Run 003 | Δ |
|---|---|---|---|---|
| Mean tick spacing, A/B/C/D/E | The sleep-driven tick cadence, with the core 0.5%–1.7% busy | 2.074 / 2.070 / 2.073 / 2.094 / 2.100 ms | 2.165 / 2.160 / 2.160 / 2.185 / 2.192 ms | **+4.4 / +4.3 / +4.2 / +4.3 / +4.4%** |
| Implied inbound delivery rate | The fake inbound queue's own throughput, a harness property | 31 900–33 400 tasks/s | 30 200–31 500 tasks/s | −5% to −8% |
| `client − server`, immediate p50 | Loopback round trip outside the handler | 347.7 / 351.9 / 446.1 / 802.5 / 409.3 µs | 347.7 / 396.9 / 466.6 / 835.2 / 442.0 µs | 0% / +12.8% / +4.6% / +4.1% / +8.0% |

The tick-cadence figure is the most telling: a timer-driven sleep, requested at 1 ms, landed 4.2%–4.4% wider in five independently launched cases. That is a property of the host's timer behaviour under WSL2, and it is the same drift direction and roughly the same magnitude as the inbound-rate deficit and the transport shift.

**So run 003's measurement session ran on the order of 4%–8% slower than run 002's, before any code is considered.** Every comparison in §4 should be read against that: a step that came out flat or cheaper did so against a headwind, and a step that came out 8%–13% dearer (case B's, and `collect`/`process` generally) is consistent with the headwind alone.

### 4.6 What §4 establishes

- The refactor did not make the scheduling loop dearer. `fill` per assignment is flat in A and C, better in E, confounded case-wide in B, and supply-confounded in D.
- It did not change the shape of the tick either: `fill` still dominates at 66%–79%, `retire` is still under 1.3%, and nothing scales super-linearly in #RG.
- The whole tick loop still costs 421–605 ns per assignment in the four unconfounded cases (run 002: 428–620 ns) and 0.48%–1.72% of one core.
- The only consistent movement, `retire`'s, is a measurement floor shift shared with untouched code (§4.4).
- Between 4% and 8% of every unfavourable comparison in this section is attributable to the session, not the code (§4.5).

## 5. Dispatch latency

All values in **microseconds**. There were no empty responses in any case, on either side. These series were not expected to move: the refactor is entirely inside the core's tick, and a dispatch request never touches the job arena or the resource-group units — it pops an assignment that the core published earlier.

### 5.1 The class mix

| Case | pinned immediate | pinned waited | general immediate | general waited | empty | waited share 003 | waited share 002 | pinned share of requests |
|---|---|---|---|---|---|---|---|---|
| A | 104 726 | 128 | 26 186 | 32 | 0 | 0.122% | 0.092% | 80.00% |
| B | 209 452 | 257 | 52 387 | 48 | 0 | 0.116% | 0.104% | 80.00% |
| C | 418 669 | 719 | 104 769 | 131 | 0 | 0.162% | 0.150% | 79.99% |
| D | 127 532 | 671 827 | 37 509 | 211 708 | 0 | **84.26%** | 78.29% | 76.23% |
| E | 0 | 0 | 1 048 192 | 384 | 0 | 0.037% | 0.037% | 0% |

**Client and server counts agree exactly, series by series, in every case** — delta 0 on all four series in all five cases, which means both sides classified every single request identically.

Cases A, B, C and E were essentially never starved (0.037%–0.162% blocked), matching run 002 to within 0.03 percentage points. **Case D blocked on 84.3% of its requests**, up from 78.3%, which §4.3 and §8.1 attribute to the harness's inbound supply rate. Case D's immediate-class figures therefore rest on 15.7% of its requests, against 21.7% in run 002.

Case E has no dedicated workers by construction, so both of its pinned series are empty by design rather than by omission.

### 5.2 The immediate class — what a dispatch costs

| Case | Path | Side | n | min | p50 | p90 | p99 | p99.9 | max | mean | p50 in 002 | p99 in 002 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | dedicated | **server** | 104 726 | 0.114 | **0.447** | 1.007 | **1.759** | 4.735 | 192.2 | 0.570 | 0.503 | 1.807 |
| A | dedicated | client | 104 726 | 48.131 | 348.159 | 573.439 | 819.199 | 1 032.19 | 42 718.4 | 372.726 | 348.159 | 851.967 |
| A | shared | **server** | 26 186 | 0.090 | **0.631** | 1.263 | **2.079** | 4.671 | 198.2 | 0.740 | 0.711 | 2.175 |
| A | shared | client | 26 186 | 73.746 | 348.159 | 581.631 | 827.391 | 1 064.96 | 2 167.8 | 371.742 | 352.255 | 860.159 |
| B | dedicated | **server** | 209 452 | 0.111 | **0.383** | 0.815 | **1.423** | 2.719 | 279.5 | 0.474 | 0.327 | 1.247 |
| B | dedicated | client | 209 452 | 47.200 | 397.311 | 655.359 | 925.695 | 1 163.26 | 42 850.6 | 421.092 | 352.255 | 835.583 |
| B | shared | **server** | 52 387 | 0.095 | **0.495** | 0.999 | **1.679** | 3.135 | 25.6 | 0.583 | 0.483 | 1.455 |
| B | shared | client | 52 387 | 46.859 | 397.311 | 655.359 | 925.695 | 1 146.88 | 1 606.7 | 421.137 | 348.159 | 843.775 |
| C | dedicated | **server** | 418 669 | 0.111 | **0.343** | 0.759 | **1.167** | 2.495 | 84.2 | 0.424 | 0.331 | 1.135 |
| C | dedicated | client | 418 669 | 44.378 | 466.943 | 745.471 | 1 015.807 | 1 277.95 | 3 413.8 | 486.047 | 446.463 | 999.423 |
| C | shared | **server** | 104 769 | 0.092 | **0.419** | 0.871 | **1.311** | 3.231 | 86.0 | 0.509 | 0.427 | 1.327 |
| C | shared | client | 104 769 | 57.014 | 471.039 | 745.471 | 1 015.807 | 1 277.95 | 3 409.9 | 487.844 | 446.463 | 991.231 |
| D | dedicated | **server** | 127 532 | 0.114 | **0.407** | 0.815 | **1.247** | 2.079 | 68.8 | 0.489 | 0.351 | 1.167 |
| D | dedicated | client | 127 532 | 45.203 | 835.583 | 1 196.031 | 1 490.943 | 1 818.62 | 3 057.9 | 824.529 | 802.815 | 1 490.943 |
| D | shared | **server** | 37 509 | 0.093 | **0.503** | 0.967 | **1.567** | 3.167 | 135.1 | 0.580 | 0.467 | 1.439 |
| D | shared | client | 37 509 | 42.418 | 884.735 | 1 228.799 | 1 507.327 | 1 785.86 | 4 031.7 | 866.944 | 843.775 | 1 523.711 |
| E | dedicated | server / client | 0 | — | — | — | — | — | — | — | — | — |
| E | shared | **server** | 1 048 192 | 0.091 | **0.347** | 0.767 | **1.311** | 2.591 | 285.1 | 0.435 | 0.327 | 1.231 |
| E | shared | client | 1 048 192 | 56.387 | 442.367 | 712.703 | 966.655 | 1 212.42 | 2 544.7 | 462.276 | 409.599 | 909.311 |

**The scheduler's dispatch cost is 0.343–0.631 µs at the median and 1.17–2.08 µs at p99, on both paths, in all five cases** — run 002's figures were 0.327–0.711 µs and 1.14–2.18 µs. The refactor did not move dispatch latency, which is the expected result: the request path pops from a queue the core filled earlier and never touches the arenas.

Movement at the median is +3.6% to +17% in cases B, C, D and E and −11% in case A, i.e. no consistent direction and no case outside 60 ns of absolute change. The largest single move, case B's dedicated p50 at +17% (56 ns), belongs to the case whose whole run was ~10% slow including its transport (§4.2).

**The server-side mean sits below its own p99 on every immediate series in every case** (0.424–0.740 µs mean against 1.167–2.079 µs p99), so run 002's separation of the immediate and waited classes is still holding and the mean remains quotable.

The server-side maxima (25.6–285.1 µs) are single samples: p99.9 is 2.1–4.7 µs everywhere, so they sit two to three orders of magnitude above the body of the distribution and move no mean. They are consistent with OS preemption of the handler. Run 002's largest was 1 167 µs, so run 003's tails are milder on this series, not worse.

### 5.3 Dedicated versus shared

The shared (general) path carries the same structural premium over the dedicated (pinned) path at the median, immediate class, server side:

| Case | dedicated p50 | shared p50 | premium | absolute | premium in 002 | dedicated p99 | shared p99 | premium |
|---|---|---|---|---|---|---|---|---|
| A | 0.447 µs | 0.631 µs | +41% | +184 ns | +41% | 1.759 µs | 2.079 µs | +18% |
| B | 0.383 µs | 0.495 µs | +29% | +112 ns | +48% | 1.423 µs | 1.679 µs | +18% |
| C | 0.343 µs | 0.419 µs | +22% | +76 ns | +29% | 1.167 µs | 1.311 µs | +12% |
| D | 0.407 µs | 0.503 µs | +24% | +96 ns | +33% | 1.247 µs | 1.567 µs | +26% |
| E | — | — | — | — | — | — | — | — |

A pinned request touches one group's queue; a general request pops the hint channel and may traverse stale hints. The premium is 76–184 ns absolutely (run 002: 96–208 ns) and narrows by p99, exactly as before.

### 5.4 The waited class

Reported separately, and not to be read as dispatch cost. These are the requests that found their queue empty and parked; their duration measures when work next arrived.

| Case | Path | Side | n | min | p50 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| A | dedicated | server | 128 | 1 847.6 | 5 636.1 | 11 927.6 | 13 107.2 | 13 018.8 | 6 810.0 |
| A | dedicated | client | 128 | 2 112.9 | 6 029.3 | 12 320.8 | 13 369.3 | 13 333.4 | 7 098.6 |
| A | shared | server | 32 | 1 830.8 | 4 849.7 | 11 796.5 | 13 107.2 | 12 993.6 | 6 425.1 |
| A | shared | client | 32 | 2 155.7 | 5 111.8 | 12 189.7 | 13 369.3 | 13 294.3 | 6 725.6 |
| B | dedicated | server | 257 | 41.0 | 7 209.0 | 12 976.1 | 13 238.3 | 13 211.7 | 6 787.9 |
| B | dedicated | client | 257 | 288.3 | 7 536.6 | 13 500.4 | 13 631.5 | 13 684.8 | 7 228.7 |
| B | shared | server | 48 | 3 015.3 | 8 781.8 | 12 976.1 | 12 976.1 | 12 914.2 | 8 488.7 |
| B | shared | client | 48 | 3 563.1 | 9 044.0 | 13 500.4 | 13 500.4 | 13 420.5 | 8 899.9 |
| C | dedicated | server | 719 | 17.4 | 10 092.5 | 18 350.1 | 22 020.1 | 21 949.4 | 8 040.3 |
| C | dedicated | client | 719 | 163.6 | 10 616.8 | 19 398.7 | 22 282.2 | 22 603.6 | 8 726.2 |
| C | shared | server | 131 | 119.6 | 10 092.5 | 14 811.1 | 14 942.2 | 14 902.4 | 8 396.2 |
| C | shared | client | 131 | 887.2 | 10 485.8 | 15 335.4 | 15 597.6 | 15 670.0 | 9 028.0 |
| D | dedicated | server | 671 827 | 0.6 | 2 719.7 | 10 616.8 | 11 534.3 | 29 657.1 | 4 732.3 |
| D | dedicated | client | 671 827 | 251.6 | 3 407.9 | 11 403.3 | 12 451.8 | 31 125.3 | 5 529.0 |
| D | shared | server | 211 708 | 1.1 | 2 097.2 | 2 949.1 | 8 519.7 | 19 792.6 | 2 191.9 |
| D | shared | client | 211 708 | 309.4 | 2 850.8 | 3 604.5 | 9 699.3 | 20 667.7 | 2 943.5 |
| E | shared | server | 384 | 550.1 | 7 143.4 | 10 747.9 | 10 747.9 | 10 753.3 | 7 101.9 |
| E | shared | client | 384 | 1 894.8 | 7 667.7 | 11 010.0 | 11 141.1 | 11 205.4 | 7 580.9 |

In A, B, C and E these are 160, 305, 850 and 384 samples — run-in and batch-barrier moments where a worker outran supply, waiting 4.8–10.6 ms at the median, the same order as run 002's 7.5–12.3 ms. Case D's 883 535 waited requests are a supply measurement in their entirety.

The client − server difference over the waited class is 262–655 µs at p50 across A, B, C and E, the same order as the immediate class's transport cost, which confirms both sides describe one population. As in run 002, the waited class must not be read as a path comparison — case C's two paths report an identical p50 here for a bucketing reason (§8.6).

### 5.5 `client − server` on loopback — the multi-node baseline

Taken over the **immediate class only**, per path, as §2.2 requires:

| Case | Path | Client p50 | Server p50 | **`client − server` p50** | Same in 002 | Client min | Server min |
|---|---|---|---|---|---|---|---|
| A | dedicated | 348.159 | 0.447 | **347.712 µs** | 347.656 | 48.131 | 0.114 |
| A | shared | 348.159 | 0.631 | **347.528 µs** | 351.544 | 73.746 | 0.090 |
| B | dedicated | 397.311 | 0.383 | **396.928 µs** | 351.928 | 47.200 | 0.111 |
| B | shared | 397.311 | 0.495 | **396.816 µs** | 347.676 | 46.859 | 0.095 |
| C | dedicated | 466.943 | 0.343 | **466.600 µs** | 446.132 | 44.378 | 0.111 |
| C | shared | 471.039 | 0.419 | **470.620 µs** | 446.036 | 57.014 | 0.092 |
| D | dedicated | 835.583 | 0.407 | **835.176 µs** | 802.464 | 45.203 | 0.114 |
| D | shared | 884.735 | 0.503 | **884.232 µs** | 843.308 | 42.418 | 0.093 |
| E | shared | 442.367 | 0.347 | **442.020 µs** | 409.272 | 56.387 | 0.087 |

The two paths agree to within 0.9% within every case, which is what a single shared transport population should look like. The floor — the smallest round trip observed anywhere — is 42.4–73.7 µs against a server-side minimum of 0.087–0.114 µs, so a loopback round trip costs tens of microseconds at best while the server contributes under 1 µs of it.

Against run 002 this baseline moved 0% (A), +12.8% (B), +4.6% (C), +4.1% (D) and +8.0% (E), which is part of the common-mode evidence of §4.5 and, since the refactor cannot touch transport, is one of the strongest indications that the session — not the code — is what moved. As in run 002, **348–885 µs is not a wire cost** (§8.5); it is dominated by worker-process coroutine scheduling and is a bound, not a baseline.

## 6. Correctness

The completion count per job is the run's validity gate: it must equal `tasks_per_job` = 1024 for every job, or the timings mean nothing.

| Case | Jobs expected | Jobs present | Jobs with exactly 1024 completions | Sum of completions | Total tasks | Sum of `assignments_published` | Worker-side assignments (pinned + general) | Client samples | Server samples | RGs represented | Empty responses (scheduler pinned / general, worker) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 128 | 128 | **128** | 131 072 | 131 072 | 131 072 | 104 854 + 26 218 = 131 072 | 131 072 | 131 072 | 1 of 1 | 0 / 0 / 0 |
| B | 256 | 256 | **256** | 262 144 | 262 144 | 262 144 | 209 709 + 52 435 = 262 144 | 262 144 | 262 144 | 2 of 2 | 0 / 0 / 0 |
| C | 512 | 512 | **512** | 524 288 | 524 288 | 524 288 | 419 388 + 104 900 = 524 288 | 524 288 | 524 288 | 4 of 4 | 0 / 0 / 0 |
| D | 1024 | 1024 | **1024** | 1 048 576 | 1 048 576 | 1 048 576 | 799 359 + 249 217 = 1 048 576 | 1 048 576 | 1 048 576 | 8 of 8 | 0 / 0 / 0 |
| E | 1024 | 1024 | **1024** | 1 048 576 | 1 048 576 | 1 048 576 | 0 + 1 048 576 = 1 048 576 | 1 048 576 | 1 048 576 | 8 of 8 | 0 / 0 / 0 |

**In every case, every job completed exactly its task count.** No job over- or under-completed; not one of the 2 944 jobs deviated; no completion was lost or double-counted. Five independent counts agree exactly in each case: the sum of per-job completion counts, the sum of `assignments_published` over all ticks, the worker-side assignment tallies, the client-side sample total and the server-side sample total. The per-class client-versus-server counts also agree exactly — all four series, delta 0, in all five cases.

No job carried the `UNKNOWN_RESOURCE_GROUP_ID` fallback, and all per-job E2E values are positive and finite. Zero `WARN`, zero `ERROR` and zero panic lines appear in any of the ten process logs.

This matters more for run 003 than for its predecessors: the refactor replaced the mechanism by which a scheduling position learns that its job is gone — a shared `Rc` plus a `finalized` flag, now a generational key that fails to resolve. A defect there would show up as a lost or duplicated task, and none of the 2 944 000 tasks dispatched across the five cases was lost or duplicated.

## 7. Per-job end-to-end distributions

All values in milliseconds. Every value includes the extra round trip described in §2.4.

| Case | jobs | min | p10 | p25 | p50 | p75 | p90 | p99 | max | mean | p50 in 002 | max in 002 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 128 | 1 342 | 2 336 | 2 586 | 2 700 | 4 477 | 4 514 | 4 625 | 4 628 | 3 440 | 2 704 | 4 628 |
| B | 256 | 400 | 1 826 | 2 452 | 2 701 | 3 513 | 3 603 | 3 749 | 3 750 | 2 809 | 2 663 | 3 687 |
| C | 512 | 207 | 555 | 973 | 1 561 | 1 931 | 2 088 | 2 226 | 2 238 | 1 439 | 1 668 | 2 293 |
| D | 1024 | 258 | 268 | 272 | 278 | 282 | 286 | 291 | 414 | 278 | 261 | 390 |
| E | 1024 | 582 | 1 967 | 4 227 | 5 841 | 6 354 | 6 495 | 6 615 | 6 621 | 5 040 | 5 949 | 6 530 |

The distributions are within a few percent of run 002's in every case (p50: A −0.1%, B +1.4%, C −6.4%, D +6.5%, E −1.8%), and their shapes are unchanged. Case D's +6.5% tracks its 5.9% longer wall clock, which is the supply deficit of §4.3; case C's −6.4% tracks nothing in particular and is inside the run-to-run variation these distributions showed before.

### 7.1 Reading these shapes

**The distributions are shaped by the batch release rule, not by per-job service variability.** Within a batch the inbound queue emits jobs' tasks in job order, so the first job of a batch is first-seen at the batch's start while the last is first-seen only after the earlier jobs' tasks have been emitted — yet all of them finish at roughly the same moment, when the batch's barrier is reached. A job's E2E is therefore mostly a function of its position within its batch, which is why the distributions are multi-modal with a heavy mode at the top of the range and a thin left tail.

**Case D's tight distribution is a symptom, not an achievement.** Its 1024 jobs all landed within 258–414 ms with a p50 of 278 ms, an order of magnitude tighter and shorter than any other case, because D never built a backlog: tasks were executed about as fast as the harness could emit them (§8.1), so a job's E2E is essentially its own emission span plus one round trip. Where a backlog exists — A, B, C, E — E2E includes queueing behind it and runs to seconds.

**No resource group was starved relative to another.** Per-group E2E medians agree closely in every multi-group case: B 2 699.6 / 2 701.1 ms (0.06% apart); C 1 554.1 / 1 561.4 / 1 561.5 / 1 555.5 ms (0.5% spread); D 278.3–278.4 ms across all eight; E 5 841.3–5 841.5 ms across all eight. This is the same non-interference run 002 reported, and it survived the refactor — which matters, because the round-robin arm and the active-group list are exactly the structures that changed from `Rc` clones to indices.

## 8. Anomalies and threats to validity

### 8.1 Case D is more supply-starved than in run 002, and that is what its numbers are measuring

The fake inbound queue returns at most `inbound_wave_size` = 256 tasks per poll and the core runs one poll at a time. That ceiling is measurable from how fast each batch's tasks were first seen, and it is a harness property independent of load:

| Case | Batch 0 | Batch 1 | Batch 2 | Batch 3 | Run 003 range | Run 002 range |
|---|---|---|---|---|---|---|
| A | 31 339 | 31 457 | 30 955 | 31 318 | 31 000–31 500 | 31 900–32 200 |
| B | 30 922 | 31 131 | 31 129 | 30 946 | 30 900–31 100 | 32 300–32 700 |
| C | 30 804 | 31 233 | 30 669 | 31 246 | 30 700–31 200 | 32 900–33 400 |
| D | 30 251 | 30 363 | 30 438 | 30 597 | 30 300–30 600 | 33 000–33 200 |
| E | 30 879 | 30 415 | 30 365 | 30 206 | 30 200–30 900 | 33 200–33 300 |

Case D is the only case whose worker demand exceeds that ceiling: 320 workers at one task per ~6 ms effective sleep is ~53 000 tasks/s against a supply of ~30 400. Its measured 30 393 tasks/s is **99.9% of the ceiling**. Three independent signatures confirm it ran starved rather than slow: publication exactly tracked arrival (assignments per busy tick is exactly 256 at p10, p50, p90 *and* p99 over 4 147 busy ticks); 84.26% of its requests blocked waiting for work, against 0.037%–0.162% everywhere else; and its mean active resource-group count was 2.158 of 8, while A, B, C and E sat at 0.975/1, 1.951/2, 3.900/4 and 7.921/8.

So case D's wall clock, throughput, ratio, waited series and client-side series are harness measurements, and its `fill` figure is confounded by the parked-receiver mechanism of §4.3. Its immediate-class dispatch latency and its per-assignment tick costs survive, the former over 15.7% of its requests. The remedy is unchanged from run 002: raise `inbound_wave_size` or issue overlapping polls, then re-measure. It has become more urgent, because case D is now the *only* case whose comparison against run 002 is unreadable, and it is also the case a `Send` core would be deployed at.

### 8.2 The whole session ran 4%–8% slower than run 002's, and it was not the code

§4.5 documents this in full: the timer-driven tick cadence widened +4.2%–+4.4% in all five cases, the harness's own inbound delivery rate fell 5%–8%, and the loopback transport median rose 0%–12.8%. None of those three is reachable from the core's data structures.

I cannot say *what* the host was doing differently — nothing was running at launch (§8.8), and no per-case CPU trace was captured beyond the load averages. The practical consequence is that this run can support the statement "the refactor did not make the scheduler dearer" but cannot support a claim finer than a few percent in either direction. **A repeat of run 002's binary on this session's host would settle it, and is the single most useful thing run 004 could do**: it would separate the session from the code exactly, which three-way inference from indicators cannot.

### 8.3 Case B moved uniformly and case B alone

Case B's per-assignment costs rose 8%–16% on every step, its transport median rose 12.8% and its server-side handler median rose 17%, while its wall clock rose only 1.6%. A refactor confined to the core cannot move transport, so §4.2 reads case B as a case-level host effect. What is unexplained is why case B and not its neighbours: A and C ran three minutes either side of it, under the same idle gate, and moved by −0.6% and −1.7% on the same statistic. I have no mechanism for that, only the observation that the moved quantities include ones the code cannot reach.

### 8.4 The tick loop ran at half its configured cadence, and slightly worse than run 002

Configured `tick_interval_ms = 1`; observed mean spacing 2.160–2.192 ms against run 002's 2.070–2.100 ms and run 001's 2.040–2.068 ms. With the core 0.48%–1.72% busy this is not CPU starvation but WSL2 timer granularity, the same effect that makes a 5 ms worker sleep behave like ~6 ms. It does not affect per-step timings, since each tick is timed individually, but it changes the number of ticks per second and therefore the assignments published per tick — run 003 published 22.9/39.9/66.9/252.9/56.2 assignments per busy tick against run 002's 20.3/37.1/63.3/252.5/53.8. **Any per-tick figure in §4 must be paired with its per-assignment counterpart**, which is cadence-independent, and the fitted marginal costs in §4.1 are the comparator that removes the amortization effect entirely.

### 8.5 The transport figure is not a wire cost

`client − server` over the immediate class is 348–885 µs, against a floor of 42.4–73.7 µs and a server-side minimum of 0.087–0.114 µs. The gap between the median and the floor is client-side runtime cost — one worker process hosting up to 320 coroutines and 320 loopback connections on a 32-core host — not the loopback. It is a bound on the loopback baseline rather than the baseline, and its rise with worker count (348 → 397 → 442 → 467 → 835 µs from 40 to 320 workers) is the same monotone artifact run 002 documented at one channel per worker.

### 8.6 Single-sample tails and bucketing coincidences

Each of these is a single sample or a convention artifact sitting orders of magnitude from the body of its distribution, moving no mean or percentile of interest:

| Series | Observation | Context |
|---|---|---|
| A client dedicated immediate | max 42 718 µs | p99.9 is 1 032 µs; the server side's max on the same series is 192 µs, so the 42.7 ms was spent entirely outside the handler. Run 001 had a 47 359 µs equivalent; run 002 had none |
| B client dedicated immediate | max 42 851 µs | Same shape, same series, 1 of 209 452 |
| E tick `process` | max 3 878 µs at tick 9 602 of 24 905 | 42× the next-largest `process` tick. Contributes 2.8% of `process`'s mean; excluding it, `process` is 5 427 ns and case E's total is 600.8 ns per assignment instead of 604.6 |
| A / B / C tick `process` | max 1 042 / 1 292 / 1 627 µs | One tick each, at a 256-task inbound wave drain on a batch-arming boundary. The same feature appeared in runs 001 and 002 |
| D tick `retire` | max 60 275 ns | A single tick; the next largest is 24 619 ns, and `retire`'s median for the case is 98 ns |
| C server waited, both paths | identical p50 of 10 092.543 µs | A bucket-boundary coincidence at the coarse high end over 719 and 131 samples; their p90/p99 differ normally (18 350/22 020 against 14 811/14 942). Run 002 got distinct values here, so it is not a fixed artifact of the bucket layout |
| A/B/C/E waited series | p99 above the recorded max | The nearest-rank-at-bucket-upper-bound convention of §2.5, not a data error |

Run 002's own single largest anomaly — case D's 32.7 ms `process` tick — **did not recur**, which is why run 003's case D needs no outlier-excluded variant of its figures.

### 8.7 Runs 001 and 002 cannot be re-analysed, only re-read

`perf/run-001/` and `perf/run-002/` contain only `report.md`; their raw JSON is not in this worktree. Every run-001/002 number in this report is therefore quoted from those reports rather than recomputed, which has three consequences worth stating.

Only the statistics those reports chose to publish are available for comparison: per-step `mean`, `p50`, `p99`, `max`, `busy mean`, `per assignment` and `share`, plus the fitted `fill` coefficients and the dispatch histogram percentiles. Statistics they did not publish — `retire`'s p10 and minimum, for instance, which §4.4's floor-shift argument would be sharper with — cannot be recovered. And no run-002 figure here can be re-derived under a different convention, so the comparisons rely on run 002 having used the conventions it documented, which §2 restates and this run followed.

Nothing in §4 depends on a single quoted number, and the load-bearing comparisons (`fill` per assignment, the fitted marginal cost, the class mix, the dispatch percentiles) are all published in run 002's tables.

### 8.8 Machine contention and the other worktree

The scheduler under test lives in this worktree, `/home/lzh/dev/spider-arena`; a second worktree at `/home/lzh/dev/spider` hosts unrelated work on a different branch, and it shares the host. Each case was launched only after an idle gate — a check that no `cargo`, `rustc` or `bench` process was running, plus the load average — and each gate passed on the first attempt:

| Case | Started | 1-min load at start | Notes |
|---|---|---|---|
| A | 22:11 | **0.91** | The highest of the set. Decaying load from the release build that finished at 22:08; fell to 0.72 by the case's end. No `cargo`/`rustc`/`bench` process running |
| B | 22:14 | 0.10 | — |
| C | 22:17 | 0.05 | 0.39 at exit |
| D | 22:20 | 0.09 | 5-/15-min averages 0.55 / 1.07, i.e. the decayed tail of the 22:08 build; nothing running at launch |
| E | 22:23 | 0.13 | Peak CPU during the run 122% of 3 200% available |

**No case ran under measurable contention from the other worktree, and no case's result is explained by its load average.** The two highest starting loads, A's 0.91 and D's 1.07 at 15 minutes, belong to the case with the *best* comparison against run 002 (A: `fill` −2.7%, `total` −0.6%) and to the case whose deficit is fully accounted for by inbound supply (D). The case that moved most in the unfavourable direction, B, started at 0.10. Load average therefore does not order the results, and the 4%–8% common-mode drift of §4.5 is not attributable to a busy machine at any case's launch — which leaves it unexplained (§8.2).

The release binaries were built once, before the sequence, and each case ran with the build step skipped, so no compilation overlapped any measurement. Cases ran strictly sequentially, one attempt each, no retries, no source file modified during the sequence.

### 8.9 The two wall clocks are not nested

The scheduler-internal wall clock exceeds the script-level one in four of five cases: B 21.674 vs 20.132 s, C 22.071 vs 20.548 s, D 34.500 vs 32.913 s, E 54.598 vs 52.995 s; only A is the other way round (21.617 vs 22.030 s). The scheduler's clock starts when the fake inbound queue first emits a task — before the script takes its own start timestamp — and ends after the 500 ms drain linger; the script's clock brackets worker-process launch and teardown instead. Neither contains the other, and §3.1 uses the scheduler-internal figure throughout, matching run 002's convention. Quoting the script clock for one run and the internal clock for the other would manufacture differences of up to 1.6 s.

### 8.10 Instrumentation cost, unchanged

Tick timing takes seven `Instant::now()` readings per poll-draining tick and five otherwise; the residual between `total` and the sum of the five steps averages 65.8–84.7 ns per tick and was never negative on any of the 70 929 ticks in this run, with per-case minima of 23–43 ns. Dispatch latency takes two readings and a few relaxed atomic increments per request with no lock; per-job progress uses a `DashMap` with atomic fields; client-side samples stay in per-coroutine owned `Vec`s and merge after the run. `fill`'s 297–500 ns per assignment is two to three orders of magnitude above the per-request instrumentation cost. Nothing about the instrumentation changed between runs 002 and 003.

Worth noting for §4.4: the instrumentation floor is itself the quantity that moved. A step that does no work costs one clock-read pair, and that pair went from 53–72 ns to 90–98 ns between the two sessions.

### 8.11 What these numbers still do not establish

- **They do not establish that the refactor is free at a finer resolution than a few percent.** The session drifted 4%–8% against run 002 on quantities the code cannot reach (§4.5), which sets the floor on any claim made here. A same-session A/B of the two binaries is what would establish it.
- **They do not exercise the invariant the generational keys exist to protect.** Every case runs one storage session with no bump, no reschedule, no worker or scheduler failure and no storage error. A stale `JobKey` surviving into a later session, and the append-only `rg_units` `Vec` whose indices are *not* generation-guarded (the core's own comment on `apply_session_bump` says nothing in the type system checks this), are untested by this benchmark. Their correctness rests on the crate's unit and integration tests, not on anything measured here.
- **They do not establish a network dispatch cost.** Everything ran on one host over loopback from one worker process; `client − server` is a bound dominated by worker-process coroutine scheduling (§8.5).
- **They do not establish the scheduler's throughput ceiling.** The highest duty cycle observed was 1.72% of one core. The maximum sustainable assignment rate is above everything measured here and unknown, because the harness topped out at ~30 500 tasks/s (§8.1).
- **They do not characterize the scheduler at 320 workers.** Case D is the only case at that scale and it ran harder-starved than in run 002 (§8.1).
- **They do not establish behaviour under contention, unequal load or failure.** Every case ran on an otherwise idle 32-core host with 1–8 groups whose loads are identical by construction. Interference between groups of unequal weight, session bumps, task rescheduling and error paths are all unexercised — the reschedule queue was never touched, and there were zero empty responses, zero warnings and zero errors in all ten process logs.
- **They do not establish the effect of the admission policy's parameters.** `B` was derived per case from `B = R × (N + 1)` with α fixed at 1; no case varied either.
- **They do not establish steady-state behaviour over long runs.** The longest case ran 55 s across four batches.

## 9. What could not be explained

Three things, recorded so run 004 can aim at them:

**Why the session ran 4%–8% slower than run 002's.** The drift is coherent and measurable in three independent indicators (§4.5), and it is definitely not the refactor, since two of the three indicators — the timer-driven tick cadence and the harness's inbound rate — never touch the core's data structures. But its cause is unknown: every case passed an idle gate, the highest starting load belongs to the best-performing case, and no per-case CPU or thermal trace was captured. Only a same-host re-run of run 002's binary can separate it from the code.

**Why case B moved uniformly when A and C, three minutes either side, did not** (§8.3).

**Why `retire`'s no-op floor rose by a uniform 24–41 ns.** §4.4 shows it cannot be the removal cost, because the ticks that got dearer are the ones that remove nothing, and `collect`'s median moved the same way in the same cases at untouched code. That places it with the session drift rather than the refactor, but the mechanism — clock-read cost, timer resolution, or something else in how the two sessions' processes were scheduled — is not identified, and run 002's raw samples, which would show whether its own floor was stable across cases, are not on disk (§8.7).
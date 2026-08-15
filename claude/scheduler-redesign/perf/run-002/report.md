# Prototype Scheduler Performance Evaluation — Run 002

This is the report required by [benchmark-contract.md](../../benchmark-contract.md) §4.2, for run 002 of the five cases A–E of §2. It is self-contained: everything needed to read it is here, and no part of it depends on [run-001's report](../run-001/report.md). Run 001 is referenced only in §1 and §8, where the comparison is the point.

Every number below was recomputed from the raw JSON files in this directory rather than copied from a run log. Percentiles over the dispatch-latency histograms were recomputed from the stored bucket arrays using the harness's own nearest-rank convention; percentiles over the per-tick and per-job series were computed from the raw samples.

The headline, stated up front so nothing below is read as a claim it does not support:

- **The scheduler's cost to hand out one task is 0.33–0.71 µs at the median and 1.1–2.2 µs at p99, in every case, on both paths.** That figure is now measured over the immediate class only, which is the only class that answers the question. It did not degrade from 1 to 8 resource groups.
- **The scheduler core was never the bottleneck.** It spent 0.48%–1.59% of one core inside `tick()` on a 32-core host.
- **Four of the five cases now land within 1.30×–1.33× of their expected steady-state duration**, down from 1.63×–1.83× in run 001, and the residual is fully accounted for by the worker's own request cycle. **Case D is unchanged at 1.99×** because its binding constraint is the harness's inbound delivery rate, which this run did not touch.
- **The loopback transport baseline the multi-node runs will be compared against is 348–446 µs at the median for cases A–C and E**, with an observed floor of 41–55 µs. It is still not a pure wire cost — §9.4.

## 1. What changed since run 001, and why

Run 001 produced valid tick timings and passed every correctness gate, but two measurement defects made its dispatch-latency numbers unusable for their stated purpose. Both are fixed in this run; the contract was amended (§1.2) to make both mandatory.

### 1.1 Defect 1 — the client side measured a shared gRPC channel pool, not the transport

Run 001 ran every case with `channel_pool_size = 16`, the binary's default, left unset rather than chosen. With 40, 80, 160, 320 and 128 worker coroutines that is 2.5, 5, 10, 20 and 8 coroutines multiplexed onto each HTTP/2 connection. The resulting head-of-line queueing happens inside our own worker process, and the measurement cannot distinguish it from transport cost.

The evidence that identified it is that the client-side median tracked workers-per-channel and nothing else, while the server side stayed flat:

| Run 001 case | Workers | Workers per channel | Client pinned p50 | Server pinned p50 | Client p90 |
|---|---|---|---|---|---|
| A | 40 | 2.5 | 327.68 µs | 0.70 µs | 6 750.21 µs |
| B | 80 | 5 | 1 441.79 µs | 0.50 µs | 6 881.28 µs |
| C | 160 | 10 | 2 719.74 µs | 0.50 µs | 6 750.21 µs |
| D | 320 | 20 | 3 899.39 µs | 1.15 µs | 7 208.96 µs |
| E | 128 | 8 | 2 080.77 µs (shared) | 0.44 µs | 6 750.21 µs |

The client median rose 8.3× across A→C while the server median did not move at all, and a p90 mode at 6.75–7.21 ms appeared in all five cases despite loads differing by 8×. That mode is the signature of queueing behind a shared connection, not of anything load-dependent.

**What was done.** `channel_pool_size` now defaults to the shard's worker count, and a worker takes channel `worker_index % pool_size`, so the default is exactly one channel per worker coroutine. Pooling is still available as an explicit opt-in and the pool size is recorded in every worker result file. Every case in this run recorded `channel_pool_size` equal to its worker count: 40, 80, 160, 320, 128.

### 1.2 Defect 2 — the server side billed long-poll waits as dispatch cost

Run 001 excluded only requests that returned nothing. A request that **blocked inside** `next_task_*` waiting for an assignment to exist, and then got one, billed its entire wait as dispatch latency.

The evidence that identified it is the mean/p99 inversion. Case C's server-side pinned series in run 001:

| Run 001 case C, pinned, server side | Value |
|---|---|
| p50 | 0.50 µs |
| p99 | 1.79 µs |
| **mean** | **21.44 µs** |
| max | 46 932 µs |

A mean twelve times its own p99 is not a latency distribution; it is two populations pooled. The mean was measuring how long a worker waited for work to exist. The same inversion appeared in every case (A 11.04 µs mean against a 2.94 µs p99, B 14.34 against 2.21, E 6.74 against 1.74) and case D was worse in kind, with a mean of 1 140 µs.

**What was done.** `next_task_pinned_classified` and `next_task_general_classified` now report which class the request fell into, and there are four server-side and four client-side histograms instead of two and two. The classification is an observation of the path the handler already takes, not a second path: a non-blocking pop precedes every wait, so the wait is entered only once the queue has been seen empty, and neither the wait budget nor the return value changes.

Run 002's case C, same case, same measurement point, now split:

| Run 002 case C, pinned, server side | Immediate | Waited |
|---|---|---|
| n | 418 788 (99.84%) | 651 (0.16%) |
| p50 | 0.331 µs | 7 536.6 µs |
| p99 | 1.135 µs | 30 408.7 µs |
| mean | 0.413 µs | 8 510.2 µs |

The immediate mean now sits **below** its own p99, where a latency mean belongs. 0.16% of the requests were producing a 52× overstatement of the reported figure.

## 2. Terminology

### 2.1 The three request classes

Every `NextTask` request falls into exactly one class, and the class determines whether and where it is timed ([benchmark-contract.md](../../benchmark-contract.md) §1.2):

| Class | Meaning | Treatment |
|---|---|---|
| **Immediate** | An assignment was available when the handler was entered; the handler returned it without ever awaiting | **Timed.** This is dispatch cost |
| **Waited** | The handler found the queue empty, awaited, and an assignment arrived before the wait expired | **Timed separately.** Dominated by supply, not by the scheduler |
| **Empty** | No assignment arrived before the wait expired | Counted only, never timed. Zero in all five cases |

**Only the immediate class answers "is the server side under control".** A waited request's duration is set by when the next task is published into the queue it is parked on — that is a property of how fast work is arriving, not of how long the scheduler takes to hand work over. Pooling the two makes the server-side mean a function of the harness's supply rate, which is what run 001's numbers were.

**"Waited" means the handler entered an await, not that it waited long.** The flag is set the moment the non-blocking pop misses, so a request whose await resolves in half a microsecond is still a waited one. Case D's server-side pinned waited series has a minimum of 0.519 µs and case C's has 7.854 µs, which is that effect and not a mismeasurement. The class is therefore a clean partition by control flow, which is exactly what makes the immediate series interpretable.

### 2.2 Client side and server side

- **Server-side** is measured in the service handler around the `DispatchService::next_task_*` call. It excludes transport and serialization, and it is the trustworthy measure of what the scheduler costs per request.
- **Client-side** is measured in the worker coroutine around the gRPC call, immediately before the request to immediately after the response. It includes transport, serialization, and the server time, so it is always the larger of the two.

`client − server` **over the immediate class only** is the transport and framing overhead. On this single-node run it is loopback cost, recorded so a later multi-node run has a baseline to subtract. Taking the difference over a pooled series would subtract two different mixtures of the two populations and mean nothing.

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

Three accounting details are baked into the data and must be read with it, unchanged from run 001:

- `start_inbound_poll` is billed to `apply` on a tick that drained a poll result, and to `collect` on a tick that had no poll in flight, rather than to `collect` unconditionally as the contract's table describes.
- Steps 2 and 3 are skipped entirely, and record zero, on any tick where the inbound poll is still in flight. That is why `process` and `apply` have a median of 0 in every case. Their means over all ticks are still the right per-tick cost; their medians are not.
- A poll-draining tick takes seven `Instant::now()` readings (`TickTimer::start`, five `finish_step` calls, and `TickTimer::finish`) rather than six, which is why `total` exceeds the sum of the five steps by one clock read on every tick: a mean residual of 56–86 ns per case, and never negative on any of the 72 123 ticks recorded across this run's five cases.

### 2.4 Per-job end-to-end time

**First seen** is when the fake inbound queue first emits a task belonging to the job; **last completed** is when the dispatch service received the last completion report for a task of that job; **E2E** is the difference.

**A job's E2E includes one extra round trip.** A worker reports a task's completion on its *next* `NextTask` request, not when it finishes executing, so the last completion of a job is observed one request cycle after that task actually finished. Every E2E figure in §7 is inflated by one client-side round trip relative to the true execution span. This is also why the run does not end when the last task is dispatched: it ends when the last completion is received, which is what the harness's 500 ms drain linger exists to flush.

E2E is **not** a per-task service-time measure. It spans a job's whole emission-to-completion life, including all queueing behind the backlog the core has buffered, so its distribution is shaped mostly by the batch release rule (§3) — see §7.1.

The completion count per job doubles as the run's validity gate (§6).

### 2.5 Histogram resolution

Dispatch latency is stored as a fixed-bucket atomic histogram of 1984 buckets: the first 64 ns resolved to the nanosecond, and every octave above that split into 64 buckets, giving ≤1.6% relative error anywhere above 64 ns. Percentiles use the nearest-rank method reported at the containing bucket's upper bound, so a quoted percentile is never understated. Count, mean, min and max are exact, being maintained separately from the buckets.

One consequence to expect in the small waited series: a quoted percentile can exceed the recorded maximum, because the percentile is a bucket upper bound while the maximum is exact. Case A's pinned waited series shows p90 = 15 466.495 µs against a max of 15 361.640 µs. This is the convention, not a data error.

## 3. Configuration

Constant across every case:

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
| `channel_pool_size` | **the case's worker count — one gRPC channel per worker coroutine** |

The five cases:

| Case | #RG | Total tasks | Shared | Dedicated | Total workers | Reserve `R` | `B` |
|---|---|---|---|---|---|---|---|
| A | 1 | 131 072 | 8 | 32 | 40 | 40 | 80 |
| B | 2 | 262 144 | 16 | 64 | 80 | 40 | 120 |
| C | 4 | 524 288 | 32 | 128 | 160 | 40 | 200 |
| D | 8 | 1 048 576 | 64 | 256 | 320 | 40 | 360 |
| E | 8 | 1 048 576 | 128 | 0 | 128 | 16 | 144 |

**Deriving `B`, the dispatch queue capacity.** The requirement is that at least `R` slots stay free when every resource group is active, where `R = (shared workers / #RG) + (dedicated workers per group)`. Under the admission policy of [design.md](../../design.md) §6.2 with α = 1 and `N` backlogged groups, free space settles at `F = B / (N + 1)`, so requiring `F ≥ R` gives `B = R × (N + 1)`, which is the `B` column above. Case E has no dedicated workers, so `R = 128 / 8 = 16` and `B = 16 × 9 = 144`.

**Batch release rule.** Batch `n+1` is released once every job in batch `n` has had all of its tasks reported complete. This is an *interpretation* of "jobs are created in batches" rather than something the design specifies, and it is recorded here because it materially shapes the results: it is the direct cause of the E2E distribution shapes in §7 and of the staircase in job arrival visible in every case. With 32 jobs per batch against an active list of 16, there is always a pending queue.

**Expected steady-state duration** is `total_tasks × 5 ms / total_workers`: 16.384 s for A–D and 40.96 s for E. A case whose measured duration greatly exceeds this is worker-starved or harness-bound, and its wall clock is not a scheduler measurement.

### 3.1 What was run

| Case | #RG | Total tasks | Workers (ded. + shared) | `B` | Pool | Wall clock (s) | Expected (s) | Ratio | Throughput (tasks/s) | Ticks | Core duty cycle |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 1 | 131 072 | 32 + 8 = 40 | 80 | 40 | 21.465 | 16.384 | 1.31× | 6 106 | 10 349 | 0.483% |
| B | 2 | 262 144 | 64 + 16 = 80 | 120 | 80 | 21.327 | 16.384 | 1.30× | 12 292 | 10 303 | 0.621% |
| C | 4 | 524 288 | 128 + 32 = 160 | 200 | 160 | 21.722 | 16.384 | 1.33× | 24 137 | 10 478 | 1.033% |
| D | 8 | 1 048 576 | 256 + 64 = 320 | 360 | 320 | 32.584 | 16.384 | **1.99×** | 32 181 | 15 561 | 1.585% |
| E | 8 | 1 048 576 | 0 + 128 = 128 | 144 | 128 | 53.411 | 40.960 | 1.30× | 19 632 | 25 432 | 1.217% |

Wall clock is the scheduler-internal measure of §2.4 — first inbound emission to last completion received, plus the 500 ms drain linger. The script-level wall clock, which brackets worker process launch and teardown instead, was 21.323 / 21.735 / 21.452 / 32.412 / 52.632 s and is recorded in `<case>-run.json`. **The two clocks are not nested** — see §9.10. Core duty cycle is the sum of every tick's `total_nanos` divided by the scheduler-internal wall clock.

Environment: single host, Linux 6.6.87.2 under WSL2, 32 cores, `--release` build, scheduler and workers as two processes on the same machine over loopback `http://127.0.0.1:50151`, one worker process hosting all worker coroutines. Cases ran strictly sequentially with the machine otherwise idle, one run each, no retries. Neither process log contains a single `WARN` or `ERROR` line in any case.

Files: `<case>-scheduler.json` (config, every tick sample, every per-job record, the four server-side histograms), `<case>-workers-0.json` (config, channel pool size, the four client-side histograms, request tallies), `<case>-run.json` (script-level wall clock), `logs/`.

## 4. Tick step timings

`per assignment` divides a step's total time over the whole run by the run's total assignments published, and is the figure to compare across cases; per-tick figures are not comparable between cases because assignments per tick differ by an order of magnitude. `busy mean` restricts to ticks that published at least one assignment. All times in nanoseconds.

### Case A — 10 349 ticks, 6 464 busy, 131 072 assignments, 2.074 ms mean spacing

| Step | mean | p50 | p99 | max | busy mean | per assignment | share of tick |
|---|---|---|---|---|---|---|---|
| `collect` | 973 | 342 | 5 368 | 61 598 | 967 | 76.8 | 9.7% |
| `process` | 1 436 | 0 | 26 551 | 1 196 589 | 1 592 | 113.4 | 14.3% |
| `apply` | 645 | 0 | 4 318 | 23 298 | 633 | 50.9 | 6.4% |
| `fill` | 6 789 | 3 574 | 25 768 | 1 810 881 | 10 374 | 536.0 | 67.8% |
| `retire` | 83 | 72 | 315 | 4 902 | 83 | 6.6 | 0.8% |
| `total` | 10 012 | 6 631 | 41 546 | 1 811 353 | 13 731 | 790.5 | 100% |

### Case B — 10 303 ticks, 7 063 busy, 262 144 assignments, 2.070 ms mean spacing

| Step | mean | p50 | p99 | max | busy mean | per assignment | share of tick |
|---|---|---|---|---|---|---|---|
| `collect` | 789 | 251 | 4 659 | 32 529 | 784 | 31.0 | 6.1% |
| `process` | 2 265 | 0 | 27 775 | 936 051 | 2 402 | 89.0 | 17.6% |
| `apply` | 528 | 0 | 3 491 | 14 696 | 519 | 20.8 | 4.1% |
| `fill` | 9 127 | 6 133 | 34 801 | 75 432 | 12 981 | 358.7 | 71.1% |
| `retire` | 74 | 53 | 487 | 26 593 | 76 | 2.9 | 0.6% |
| `total` | 12 845 | 9 617 | 48 586 | 970 119 | 16 821 | 504.9 | 100% |

### Case C — 10 478 ticks, 8 278 busy, 524 288 assignments, 2.073 ms mean spacing

| Step | mean | p50 | p99 | max | busy mean | per assignment | share of tick |
|---|---|---|---|---|---|---|---|
| `collect` | 838 | 244 | 5 188 | 36 615 | 850 | 16.7 | 3.9% |
| `process` | 4 553 | 0 | 39 397 | 1 497 741 | 4 743 | 91.0 | 21.3% |
| `apply` | 527 | 0 | 3 613 | 23 015 | 536 | 10.5 | 2.5% |
| `fill` | 15 359 | 12 006 | 60 784 | 195 669 | 19 211 | 307.0 | 71.7% |
| `retire` | 84 | 53 | 993 | 13 083 | 81 | 1.7 | 0.4% |
| `total` | 21 417 | 18 472 | 79 462 | 1 526 408 | 25 476 | 428.0 | 100% |

### Case D — 15 561 ticks, 4 152 busy, 1 048 576 assignments, 2.094 ms mean spacing

| Step | mean | p50 | p99 | max | busy mean | per assignment | share of tick |
|---|---|---|---|---|---|---|---|
| `collect` | 956 | 277 | 5 673 | 19 206 | 2 773 | 14.2 | 2.9% |
| `process` | 7 367 | 0 | 37 042 | **32 706 681** | 19 653 | 109.3 | 22.2% |
| `apply` | 787 | 0 | 5 012 | 22 491 | 2 891 | 11.7 | 2.4% |
| `fill` | 23 618 | 2 038 | 134 406 | 588 112 | 85 455 | 350.5 | 71.2% |
| `retire` | 401 | 66 | 3 215 | 16 739 | 78 | 5.9 | 1.2% |
| `total` | 33 191 | 3 319 | 174 856 | 32 713 861 | 110 913 | 492.6 | 100% |

Case D contains a single 32.7 ms `process` outlier at tick 62 of 15 561, about 0.13 s into the run, on a tick that published no assignments (§9.3). It alone contributes 28% of the `process` mean and 6.3% of the case's total core time. **Excluding that one tick**, `process` mean is 5 265 ns and 78.1 ns per assignment, `total` mean is 31 091 ns and **461.4 ns per assignment**, total core time is 483.8 ms and the duty cycle is 1.485%. Both figures are given wherever case D's per-assignment cost is compared.

### Case E — 25 432 ticks, 19 481 busy, 1 048 576 assignments, 2.100 ms mean spacing

| Step | mean | p50 | p99 | max | busy mean | per assignment | share of tick |
|---|---|---|---|---|---|---|---|
| `collect` | 767 | 243 | 4 570 | 74 463 | 773 | 18.6 | 3.0% |
| `process` | 4 820 | 0 | 45 163 | 2 471 603 | 5 108 | 116.9 | 18.9% |
| `apply` | 534 | 0 | 3 396 | 175 068 | 542 | 12.9 | 2.1% |
| `fill` | 19 294 | 14 412 | 67 728 | 338 127 | 24 905 | 468.0 | 75.5% |
| `retire` | 73 | 53 | 353 | 13 217 | 72 | 1.8 | 0.3% |
| `total` | 25 549 | 22 254 | 89 222 | 2 518 244 | 31 460 | 619.7 | 100% |

### 4.1 The scaling trend across #RG

| Quantity | A (1 RG) | B (2 RG) | C (4 RG) | D (8 RG) | E (8 RG, no dedicated) |
|---|---|---|---|---|---|
| `fill` share of tick | 67.8% | 71.1% | 71.7% | 71.2% | 75.5% |
| `retire` share of tick | 0.8% | 0.6% | 0.4% | 1.2% | 0.3% |
| `fill` per assignment (ns) | 536 | 359 | 307 | 350 | 468 |
| `process` per inbound entry (ns) | 113 | 89 | 91 | 109 (78 excl. tick 62) | 117 |
| `collect` per assignment (ns) | 76.8 | 31.0 | 16.7 | 14.2 | 18.6 |
| `apply` per assignment (ns) | 50.9 | 20.8 | 10.5 | 11.7 | 12.9 |
| `retire` per assignment (ns) | 6.6 | 2.9 | 1.7 | 5.9 | 1.8 |
| `total` per assignment (ns) | 790 | 505 | 428 | 493 (461) | 620 |
| Assignments per busy tick (mean) | 20.3 | 37.1 | 63.3 | 252.5 | 53.8 |
| Mean active resource groups | 0.975 / 1 | 1.949 / 2 | 3.898 / 4 | **2.177 / 8** | 7.919 / 8 |
| Total core time in `tick()` (ms) | 103.6 | 132.3 | 224.4 | 516.5 (483.8) | 649.8 |

**Step 4 dominates and step 5 is near zero, exactly as the contract predicts.** `fill` is 67.8%–75.5% of tick time in every case; `retire` never exceeds 1.2% and costs 1.7–6.6 ns per assignment.

**Nothing scales super-linearly in #RG.** Going from 1 to 8 resource groups with 8× the tasks and 8× the workers, the cost of the whole tick loop per assignment published went **down**, from 790 ns to 461–493 ns, and total core time spent scheduling grew 4.7× for 8× the work. Per-assignment cost falls because a per-tick fixed cost is amortized over more assignments, not because any step got intrinsically cheaper.

A per-case least-squares fit of `fill_nanos` against `assignments_published` over every tick separates the fixed and marginal parts:

| Case | #RG | Fixed cost per tick (ns) | Marginal cost per assignment (ns) | R² | R² excluding the 6 largest `fill` ticks |
|---|---|---|---|---|---|
| A | 1 | 1 022 | 455 | 0.12 | 0.85 (fixed 1 108, marginal 433) |
| B | 2 | 1 190 | 312 | 0.86 | 0.87 |
| C | 4 | 1 593 | 275 | 0.79 | 0.81 |
| D | 8 | 1 067 | 335 | 0.80 | 0.83 |
| E | 8 | 1 863 | 423 | 0.86 | 0.88 |

Case A's R² of 0.12 is one tick: tick 6140 recorded a 1.81 ms `fill` for 39 assignments, against a case p99 of 25.8 µs. Dropping the six largest `fill` ticks per case restores R² to 0.81–0.88 everywhere and moves no other case's coefficients by more than 3%.

Case D's fit should not be trusted for its fixed/marginal split: D's `assignments_published` is exactly 256 from p10 through p99 (§9.1), so the regression has almost no leverage. The robust equivalent for D is `fill` busy mean divided by assignments per busy tick, 85 455 / 252.5 = **338 ns per assignment**, which is the figure used in §8.

The fixed cost of one round-robin pass grows with the number of resource groups but sub-linearly, from about 1.02–1.11 µs at one group to 1.59–1.86 µs at four and eight. The fitted marginal cost per assignment is 275–455 ns across the set. Neither term blows up at 8 resource groups.

`process` costs 78–117 ns per inbound entry and is the second-largest step from case B onward, reaching 17.6%–22.2% of tick time. It is also the step with the heaviest tail: its maxima (1.20 ms in A, 0.94 ms in B, 1.50 ms in C, 32.71 ms in D, 2.47 ms in E) are the only per-step outliers large enough to drive `total`'s maximum, and they land on ticks that drain a 256-task inbound wave at a batch-arming boundary. These are single ticks out of tens of thousands; p99 for the step stays between 26.6 µs and 45.2 µs.

The absolute cost bears restating: the entire scheduling loop consumed 104–650 ms of CPU across runs lasting 21–53 s, i.e. **0.48%–1.59% of a single core** on a 32-core host. No case came close to saturating the core.

**Did these move relative to run 001?** They were not expected to, since neither fix touches the core, and four of the five cases confirm that. Case D does not, and §8.3 is about why.

## 5. Dispatch latency

This is the heart of this run. All values in **microseconds**. There were no empty responses in any case, on either side.

### 5.1 The class mix

| Case | pinned immediate | pinned waited | general immediate | general waited | empty | waited share | pinned share of requests |
|---|---|---|---|---|---|---|---|
| A | 104 773 | 96 | 26 179 | 24 | 0 | 0.092% | 80.01% |
| B | 209 481 | 225 | 52 390 | 48 | 0 | 0.104% | 80.00% |
| C | 418 788 | 651 | 104 716 | 133 | 0 | 0.150% | 80.00% |
| D | 173 752 | 630 650 | 53 947 | 190 227 | 0 | **78.29%** | 76.71% |
| E | 0 | 0 | 1 048 192 | 384 | 0 | 0.037% | 0% |

The pinned share tracks the dedicated-worker share exactly in A–C (80% dedicated workers, 80.0% pinned requests) and at 76.7% against 80% in D, where the pinned path is the one that starved harder. Case E has no dedicated workers by construction, so both pinned series are empty by design.

**Client and server counts agree exactly, series by series, in every case** — delta 0 on all four series in all five cases. That is a stronger check than the totals agreeing, because it means both sides classified every single request identically.

**Cases A, B, C and E were essentially never starved:** 0.037%–0.150% of requests blocked for work. **Case D blocked on 78.3% of its requests**, which §9.1 attributes to the harness's inbound supply rate and not to the scheduler. Case D's immediate-class figures therefore rest on 21.7% of its requests, and its waited series is a supply measurement.

### 5.2 The immediate class — what a dispatch costs

| Case | Path | Side | n | min | p50 | p90 | p99 | p99.9 | max | mean | < 10 µs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | dedicated | **server** | 104 773 | 0.112 | **0.503** | 1.071 | **1.807** | 5.439 | 528.1 | 0.668 | 99.936% |
| A | dedicated | client | 104 773 | 47.520 | 348.159 | 606.207 | 851.967 | 1 064.96 | 1 926.7 | 376.949 | 0% |
| A | shared | **server** | 26 179 | 0.090 | **0.711** | 1.391 | **2.175** | 7.167 | 167.7 | 0.831 | 99.924% |
| A | shared | client | 26 179 | 54.579 | 352.255 | 606.207 | 860.159 | 1 097.73 | 44 817.1 | 381.494 | 0% |
| B | dedicated | **server** | 209 481 | 0.110 | **0.327** | 0.751 | **1.247** | 2.879 | 915.8 | 0.483 | 99.964% |
| B | dedicated | client | 209 481 | 46.785 | 352.255 | 565.247 | 835.583 | 1 146.88 | 2 052.4 | 370.311 | 0% |
| B | shared | **server** | 52 390 | 0.088 | **0.483** | 0.935 | **1.455** | 3.455 | 864.5 | 0.592 | 99.969% |
| B | shared | client | 52 390 | 45.970 | 348.159 | 565.247 | 843.775 | 1 146.88 | 1 855.1 | 369.679 | 0% |
| C | dedicated | **server** | 418 788 | 0.107 | **0.331** | 0.711 | **1.135** | 2.431 | 237.3 | 0.413 | 99.968% |
| C | dedicated | client | 418 788 | 41.186 | 446.463 | 729.087 | 999.423 | 1 359.87 | 40 794.1 | 466.539 | 0% |
| C | shared | **server** | 104 716 | 0.089 | **0.427** | 0.863 | **1.327** | 3.391 | 468.9 | 0.510 | 99.958% |
| C | shared | client | 104 716 | 52.026 | 446.463 | 729.087 | 991.231 | 1 310.72 | 2 611.4 | 467.616 | 0% |
| D | dedicated | **server** | 173 752 | 0.108 | **0.351** | 0.735 | **1.167** | 1.855 | 298.9 | 0.438 | 99.985% |
| D | dedicated | client | 173 752 | 54.219 | 802.815 | 1 146.879 | 1 490.943 | 1 949.70 | 3 143.8 | 814.821 | 0% |
| D | shared | **server** | 53 947 | 0.088 | **0.467** | 0.903 | **1.439** | 2.655 | 45.0 | 0.534 | 99.987% |
| D | shared | client | 53 947 | 73.278 | 843.775 | 1 179.647 | 1 523.711 | 2 129.92 | 3 099.5 | 849.199 | 0% |
| E | dedicated | server / client | 0 | — | — | — | — | — | — | — | — |
| E | shared | **server** | 1 048 192 | 0.087 | **0.327** | 0.711 | **1.231** | 2.463 | 1 167.2 | 0.411 | 99.970% |
| E | shared | client | 1 048 192 | 43.840 | 409.599 | 663.551 | 909.311 | 1 163.26 | 48 488.0 | 428.077 | 0% |

**The scheduler's dispatch cost is 0.33–0.71 µs at the median and 1.1–2.2 µs at p99, on both paths, in all five cases**, and 99.92%–99.99% of immediate requests complete in under 10 µs server-side. It did not degrade as the case grew: case E's median of 0.327 µs over 1 048 192 requests and case B's 0.327 µs are the fastest in the set, and case D's 0.351 µs — measured while 78% of its requests were blocking — is indistinguishable from them.

**The server-side mean now sits below its own p99 in every case and on both paths** (0.411–0.831 µs mean against 1.135–2.175 µs p99). That is the specific inversion defect 2 existed to remove, and it is gone. The server-side mean is now safe to quote.

The server-side maxima (45–1 167 µs) are single samples: p99.9 is 1.9–7.2 µs everywhere, so these sit four orders of magnitude above the body of the distribution and move no mean. They are consistent with OS preemption of the handler.

### 5.3 Dedicated versus shared

The shared (general) path carries a consistent structural premium over the dedicated (pinned) path at the median, immediate class, server side:

| Case | dedicated p50 | shared p50 | premium | absolute | dedicated p99 | shared p99 | premium |
|---|---|---|---|---|---|---|---|
| A | 0.503 µs | 0.711 µs | +41% | +208 ns | 1.807 µs | 2.175 µs | +20% |
| B | 0.327 µs | 0.483 µs | +48% | +156 ns | 1.247 µs | 1.455 µs | +17% |
| C | 0.331 µs | 0.427 µs | +29% | +96 ns | 1.135 µs | 1.327 µs | +17% |
| D | 0.351 µs | 0.467 µs | +33% | +116 ns | 1.167 µs | 1.439 µs | +23% |
| E | — | — | — | — | — | — | — |

This is the structural difference the contract predicts: a pinned request touches one group's queue, a general request pops the hint channel and may traverse stale hints. In absolute terms the premium is **96–208 ns**, and it narrows to +17%–+23% by p99. The pinned path is faster in every case; the difference is real but small against the 348–843 µs the client sees.

### 5.4 The waited class

Reported separately, and not to be read as dispatch cost. These are the requests that found their queue empty and parked.

| Case | Path | Side | n | min | p50 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| A | dedicated | server | 96 | 6 924.1 | 11 665.4 | 15 466.5 | 15 466.5 | 15 361.6 | 12 170.3 |
| A | dedicated | client | 96 | 7 401.2 | 12 320.8 | 15 859.7 | 15 859.7 | 15 828.7 | 12 552.0 |
| A | shared | server | 24 | 9 226.1 | 11 665.4 | 15 466.5 | 15 466.5 | 15 349.9 | 12 191.2 |
| A | shared | client | 24 | 9 396.9 | 12 189.7 | 15 728.6 | 15 728.6 | 15 721.3 | 12 585.0 |
| B | dedicated | server | 225 | 23.9 | 11 010.0 | 11 534.3 | 17 301.5 | 17 285.6 | 8 692.0 |
| B | dedicated | client | 225 | 339.6 | 11 403.3 | 11 796.5 | 17 825.8 | 17 922.3 | 9 104.9 |
| B | shared | server | 48 | 4 298.3 | 9 961.5 | 11 403.3 | 15 597.6 | 15 540.4 | 8 905.2 |
| B | shared | client | 48 | 4 783.3 | 10 354.7 | 11 665.4 | 16 252.9 | 16 162.4 | 9 269.9 |
| C | dedicated | server | 651 | 7.9 | 7 536.6 | 20 185.1 | 30 408.7 | 30 321.0 | 8 510.2 |
| C | dedicated | client | 651 | 174.9 | 7 995.4 | 20 709.4 | 31 195.1 | 31 474.8 | 9 087.5 |
| C | shared | server | 133 | 653.2 | 8 323.1 | 13 631.5 | 13 762.6 | 13 697.0 | 7 779.5 |
| C | shared | client | 133 | 1 239.2 | 8 912.9 | 14 155.8 | 14 286.8 | 14 250.7 | 8 317.4 |
| D | dedicated | server | 630 650 | 0.5 | 2 326.5 | 9 437.2 | 10 747.9 | 26 599.6 | 4 252.5 |
| D | dedicated | client | 630 650 | 304.8 | 3 047.4 | 10 223.6 | 11 534.3 | 26 993.4 | 5 069.2 |
| D | shared | server | 190 227 | 0.7 | 1 474.6 | 2 588.7 | 8 257.5 | 16 630.2 | 1 939.9 |
| D | shared | client | 190 227 | 245.5 | 2 228.2 | 3 375.1 | 9 437.2 | 17 873.4 | 2 711.5 |
| E | shared | server | 384 | 3 149.5 | 9 568.3 | 11 927.6 | 12 058.6 | 12 089.6 | 9 004.8 |
| E | shared | client | 384 | 4 197.3 | 9 961.5 | 12 976.1 | 13 107.2 | 13 133.4 | 9 565.1 |

In A, B, C and E these are 120, 273, 784 and 384 samples — run-in and batch-barrier moments where a worker outran supply, waiting 7.5–12.3 ms at the median. **This is the population that inflated run 001's server-side means**: 0.15% of case C's requests, at ~8 ms each, produced its 21.4 µs pooled mean.

Two observations that keep this table from being over-read. First, the client−server difference over the waited class is 393–1 000 µs at p50 across A, B, C and E — the same order as the immediate class's transport cost, which confirms both sides are describing one population rather than two different events. Second, in case C the pinned waited p50 (7.54 ms) is *below* the general waited p50 (8.32 ms) while its p90 and p99 (20.2 / 30.4 ms) are far *above* general's (13.6 / 13.8 ms); with 651 and 133 samples those tails are 65 and 13 observations. The waited class is supply-driven by construction and must not be read as a path comparison.

### 5.5 `client − server` on loopback — the multi-node baseline

Taken over the **immediate class only**, per path, as §2.2 requires. This is the number a multi-node run will be compared against.

| Case | Path | Client p50 | Server p50 | **`client − server` p50** | Client min | Server min |
|---|---|---|---|---|---|---|
| A | dedicated | 348.159 | 0.503 | **347.656 µs** | 47.520 | 0.112 |
| A | shared | 352.255 | 0.711 | **351.544 µs** | 54.579 | 0.090 |
| B | dedicated | 352.255 | 0.327 | **351.928 µs** | 46.785 | 0.110 |
| B | shared | 348.159 | 0.483 | **347.676 µs** | 45.970 | 0.088 |
| C | dedicated | 446.463 | 0.331 | **446.132 µs** | 41.186 | 0.107 |
| C | shared | 446.463 | 0.427 | **446.036 µs** | 52.026 | 0.089 |
| D | dedicated | 802.815 | 0.351 | **802.464 µs** | 54.219 | 0.108 |
| D | shared | 843.775 | 0.467 | **843.308 µs** | 73.278 | 0.088 |
| E | shared | 409.599 | 0.327 | **409.272 µs** | 43.840 | 0.087 |

The two paths agree to within 1.2% within every case (A 347.7 vs 351.5, B 351.9 vs 347.7, C 446.1 vs 446.0, D 802.5 vs 843.3), which is what a single shared transport population should look like and is itself a check that the immediate-class split is coherent.

Two things this table does establish. The **floor** — the smallest round trip observed anywhere is 41.2–73.3 µs against a server-side minimum of 0.087–0.112 µs, so loopback transport plus framing plus worker-side scheduling of the response costs on the order of tens of microseconds at best. And the **median**, 348–446 µs for A, B, C and E, which is now free of the shared-channel artifact that made run 001's equivalent figure 328–2 720 µs.

One thing it does not establish: that 348–446 µs is transport. It is not — see §9.4. The client-side median still rises with the number of coroutines in the worker process, and that residual belongs to the harness, not to the wire.

## 6. Correctness

The completion count per job is the run's validity gate: it must equal `tasks_per_job` = 1024 for every job, or the timings mean nothing.

| Case | Jobs expected | Jobs present | Jobs with exactly 1024 completions | Sum of completions | Total tasks | Sum of `assignments_published` | Client samples | Server samples | RGs represented | Empty responses |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 128 | 128 | **128** | 131 072 | 131 072 | 131 072 | 131 072 | 131 072 | 1 of 1 | 0 |
| B | 256 | 256 | **256** | 262 144 | 262 144 | 262 144 | 262 144 | 262 144 | 2 of 2 | 0 |
| C | 512 | 512 | **512** | 524 288 | 524 288 | 524 288 | 524 288 | 524 288 | 4 of 4 | 0 |
| D | 1024 | 1024 | **1024** | 1 048 576 | 1 048 576 | 1 048 576 | 1 048 576 | 1 048 576 | 8 of 8 | 0 |
| E | 1024 | 1024 | **1024** | 1 048 576 | 1 048 576 | 1 048 576 | 1 048 576 | 1 048 576 | 8 of 8 | 0 |

In every case, every job completed exactly its task count; no job over- or under-completed; no completion was lost or double-counted; not one job deviated. Four independent counts agree exactly in each case: the sum of per-job completion counts, the sum of `assignments_published` over all ticks, the client-side sample total and the server-side sample total. This run adds a fifth check that run 001 could not make — the **per-class** client and server counts agree exactly, all four series, delta 0, in all five cases.

No job carried the `UNKNOWN_RESOURCE_GROUP_ID` fallback, and all per-job E2E values are positive and finite. Zero `WARN` and zero `ERROR` lines appear in any of the ten process logs.

## 7. Per-job end-to-end distributions

All values in milliseconds. Every value includes the extra round trip described in §2.4.

| Case | jobs | min | p10 | p25 | p50 | p75 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 128 | 1 334 | 2 333 | 2 582 | 2 704 | 4 447 | 4 511 | 4 623 | 4 628 | 3 424 |
| B | 256 | 402 | 1 806 | 2 424 | 2 663 | 3 484 | 3 580 | 3 684 | 3 687 | 2 784 |
| C | 512 | 205 | 551 | 1 027 | 1 668 | 2 012 | 2 157 | 2 281 | 2 293 | 1 511 |
| D | 1024 | 254 | 257 | 259 | 261 | 264 | 266 | 270 | 390 | 263 |
| E | 1024 | 604 | 2 131 | 4 505 | 5 949 | 6 328 | 6 450 | 6 523 | 6 530 | 5 143 |

```
case A (n=128, bin width 274 ms)
    1334 -     1608 ms | ##                                                3
    1608 -     1883 ms |                                                   0
    1883 -     2157 ms | ###                                               4
    2157 -     2432 ms | ############                                     13
    2432 -     2706 ms | #########################################        44
    2706 -     2981 ms |                                                   0
    2981 -     3255 ms |                                                   0
    3255 -     3530 ms | ##                                                3
    3530 -     3805 ms |                                                   0
    3805 -     4079 ms | ##                                                3
    4079 -     4354 ms | ########                                          9
    4354 -     4628 ms | ##############################################   49
```

```
case B (n=256, bin width 274 ms)
     402 -      676 ms | ###                                               6
     676 -      949 ms |                                                   0
     949 -     1223 ms | ###                                               6
    1223 -     1497 ms | ##                                                4
    1497 -     1771 ms | ####                                              8
    1771 -     2044 ms | #######                                          14
    2044 -     2318 ms | ########                                         16
    2318 -     2592 ms | #####################                            40
    2592 -     2865 ms | #########################                        46
    2865 -     3139 ms | ####                                              8
    3139 -     3413 ms | #############                                    24
    3413 -     3687 ms | ##############################################   84
```

```
case C (n=512, bin width 174 ms)
     205 -      379 ms | ###########                                      24
     379 -      553 ms | #############                                    28
     553 -      727 ms | ###########                                      24
     727 -      901 ms | ##############                                   31
     901 -     1075 ms | ##########                                       22
    1075 -     1249 ms | ##############                                   31
    1249 -     1423 ms | ##############                                   31
    1423 -     1597 ms | ####################                             45
    1597 -     1771 ms | ########################                         52
    1771 -     1945 ms | ############################                     62
    1945 -     2119 ms | ##############################################   99
    2119 -     2293 ms | #############################                    63
```

```
case D (n=1024, bin width 11 ms)
     254 -      266 ms | ##############################################  856
     266 -      277 ms | ########                                        160
     277 -      288 ms |                                                   0
     288 -      300 ms |                                                   0
     300 -      311 ms |                                                   0
     311 -      322 ms |                                                   0
     322 -      334 ms |                                                   0
     334 -      345 ms |                                                   0
     345 -      356 ms |                                                   0
     356 -      368 ms |                                                   0
     368 -      379 ms |                                                   0
     379 -      390 ms | #                                                 8
```

```
case E (n=1024, bin width 494 ms)
     604 -     1098 ms | ###                                              32
    1098 -     1592 ms | ###                                              40
    1592 -     2085 ms | ##                                               24
    2085 -     2579 ms | ###                                              32
    2579 -     3073 ms | ###                                              32
    3073 -     3567 ms | ###                                              32
    3567 -     4061 ms | ###                                              32
    4061 -     4555 ms | ###                                              32
    4555 -     5048 ms | ####                                             48
    5048 -     5542 ms | #######                                          80
    5542 -     6036 ms | #############                                   144
    6036 -     6530 ms | ##############################################  488
```

### 7.1 Reading these shapes

**The distributions are shaped by the batch release rule, not by per-job service variability.** Within a batch, the inbound queue emits jobs' tasks in job order, so the first job of a batch is first-seen at the batch's start while the last is first-seen only after the earlier jobs' tasks have all been emitted — yet all of them finish at roughly the same moment, when the batch's barrier is reached. A job's E2E is therefore mostly a function of its position within its batch, which is why the histograms are multi-modal with a heavy mode at the top of the range and a thin left tail: the jobs emitted last in each batch have the shortest E2E.

**Case D's tight distribution is a symptom, not an achievement.** Its 1024 jobs all landed within 254–390 ms with a p50 of 261 ms, an order of magnitude tighter and shorter than any other case, because D never built a backlog: tasks were executed about as fast as the harness could emit them (§9.1), so a job's E2E is essentially its own emission span (1024 tasks at the ~4.1k tasks/s per group the inbound queue sustained ≈ 248 ms) plus one round trip. Where a backlog exists — A, B, C, E — E2E includes queueing behind it and runs to seconds.

**No resource group was starved relative to another.** Per-group E2E medians agree closely in every multi-group case: B 2 664.8 / 2 665.8 ms (0.04% apart); C 1 667.2 / 1 676.2 / 1 676.3 / 1 682.3 ms (0.9% spread); D 261.3–261.4 ms across all eight; E 5 947.3–5 950.6 ms across all eight.

## 8. Comparison against run 001

Run 001's raw files and report are in `../run-001/`. Its five cases used identical configuration except `channel_pool_size = 16` in all five, and it excluded only the empty class from the server-side series.

### 8.1 Wall clocks and throughput

| Case | Run 001 wall (s) | Run 002 wall (s) | Change | Run 001 ratio | Run 002 ratio | Run 001 tasks/s | Run 002 tasks/s | Change |
|---|---|---|---|---|---|---|---|---|
| A | 26.635 | 21.465 | −19.4% | 1.63× | **1.31×** | 4 921 | 6 106 | +24.1% |
| B | 29.150 | 21.327 | −26.8% | 1.78× | **1.30×** | 8 993 | 12 292 | +36.7% |
| C | 30.031 | 21.722 | −27.7% | 1.83× | **1.33×** | 17 458 | 24 137 | +38.3% |
| D | 33.047 | 32.584 | −1.4% | 2.02× | **1.99×** | 31 730 | 32 181 | +1.4% |
| E | 73.978 | 53.411 | −27.8% | 1.81× | **1.30×** | 14 174 | 19 632 | +38.5% |

Four cases improved by 19%–28% and their overshoot fell from 1.63×–1.83× to 1.30×–1.33×. Case D did not move, because its binding constraint is elsewhere (§9.1).

The residual 1.30×–1.33× is fully accounted for by the worker's own request cycle, with nothing left over to attribute to the scheduler. A worker's per-task cycle is its simulated 5 ms sleep plus one client-side round trip; WSL2 timer granularity inflates short sleeps (the tick loop, asking for 1 ms, ran at 2.07–2.10 ms), so the 5 ms sleep is realistically ~6 ms. Modelling throughput as `workers / (6 ms + client-side mean over all requests)`:

| Case | Workers | Client mean, all requests (ms) | Predicted (tasks/s) | Measured (tasks/s) | Error |
|---|---|---|---|---|---|
| A | 40 | 0.389 | 6 261 | 6 106 | +2.5% |
| B | 80 | 0.379 | 12 541 | 12 292 | +2.0% |
| C | 160 | 0.479 | 24 693 | 24 137 | +2.3% |
| E | 128 | 0.431 | 19 902 | 19 632 | +1.4% |

Case D is excluded because the arithmetic is circular there: its client-side mean of 3.719 ms is dominated by the wait described in §9.1.

### 8.2 What the two defects had cost

**Defect 1 (shared channel pool), client-side series:**

| Case | Run 001 client p50 | Run 002 client p50 | Overstatement | Run 001 client p90 | Run 002 client p90 |
|---|---|---|---|---|---|
| A (40 workers) | 327.68 µs | 348.159 µs | −6% | 6 750.21 µs | 606.207 µs |
| B (80 workers) | 1 441.79 µs | 352.255 µs | **4.1×** | 6 881.28 µs | 565.247 µs |
| C (160 workers) | 2 719.74 µs | 446.463 µs | **6.1×** | 6 750.21 µs | 729.087 µs |
| D (320 workers) | 3 899.39 µs | 802.815 µs | **4.9×** | 7 208.96 µs | 1 146.879 µs |
| E (128 workers) | 2 080.77 µs | 409.599 µs | **5.1×** | 6 750.21 µs | 663.551 µs |

Case A is the exception, and it is expected rather than a regression: at 2.5 workers per channel A had the mildest pooling of the five cases, so its *median* was barely contaminated and the artifact showed only in the tail. What the fix removed at case A is a multi-millisecond tail mode, not a median offset — p90 fell 11.1× and max fell from 47 359 µs to 1 927 µs, while p50 rose 6%. The 6.75–7.21 ms client p90 mode that appeared identically in all five of run 001's cases despite 8× differences in load is gone from all five of run 002's.

The consequence for the multi-node runs is that run 001's `client − server` figure of 2 719 µs at case C was overstating the loopback baseline **6.1×**. The corrected baseline is 446 µs (§5.5).

**Defect 2 (pooled long-poll waits), server-side series:**

| Case | Run 001 server pinned mean | Run 002 immediate mean | Overstatement | Run 001 server pinned p50 | Run 002 immediate p50 | Waited samples responsible |
|---|---|---|---|---|---|---|
| A | 11.04 µs | 0.668 µs | **17×** | 0.70 µs | 0.503 µs | 96 of 104 869 (0.09%) |
| B | 14.34 µs | 0.483 µs | **30×** | 0.50 µs | 0.327 µs | 225 of 209 706 (0.11%) |
| C | 21.44 µs | 0.413 µs | **52×** | 0.50 µs | 0.331 µs | 651 of 419 439 (0.16%) |
| D | 1 140.46 µs | 0.438 µs | **2 600×** | 1.15 µs | 0.351 µs | 630 650 of 804 402 (78%) |
| E (shared) | 6.74 µs | 0.411 µs | **16×** | 0.44 µs | 0.327 µs | 384 of 1 048 576 (0.04%) |

In A, B, C and E, between 0.04% and 0.16% of the samples were producing a 16×–52× overstatement of the reported mean. Run 001 had to tell its readers not to quote the server-side mean at all; run 002's is quotable. The medians also fell 12%–39%, because even the median of a pooled series is dragged by a right-heavy contaminant when the body of the distribution is sub-microsecond.

### 8.3 Tick step timings — four cases unmoved, case D 41% dearer

Neither fix touches the core, so these were not expected to move.

| Quantity | A 001 → 002 | B 001 → 002 | C 001 → 002 | D 001 → 002 | E 001 → 002 |
|---|---|---|---|---|---|
| `total` per assignment (ns) | 779 → 790 | 591 → 505 | 449 → 428 | 353 → 493 (461 excl. tick 62) | 644 → 620 |
| `fill` per assignment, busy ticks (ns) | 497 → 512 | 409 → 350 | 322 → 303 | **241 → 338** | 465 → 463 |
| Assignments per busy tick | 15.0 → 20.3 | 25.6 → 37.1 | 44.8 → 63.3 | 235.3 → 252.5 | 36.8 → 53.8 |
| `fill` share of tick | 67.0% → 67.8% | 71.2% → 71.1% | 72.7% → 71.7% | 73.4% → 71.2% | 73.1% → 75.5% |

For A, B, C and E, `fill` per assignment on busy ticks moved by −14% to +3%, and the residual movement tracks assignments per busy tick — a higher fill rate amortizes `fill`'s per-tick constant over more assignments, which is why B and C got cheaper while A, whose assignments per busy tick is smallest, did not. `fill`'s share of tick time is within 2.3 points of run 001 in every case. These are the same numbers.

**Case D is the exception: `fill` cost 41% more per assignment than in run 001, at essentially the same assignments per busy tick (252.5 vs 235.3).** This is real and it needs an explanation. Two mechanisms are consistent with the data and this run cannot separate them:

- **More parked receivers.** Each resource group's assignment queue is an `async_channel`; publishing into a queue that has a receiver parked on it must wake that receiver, which a publish into a queue with no waiter does not. In run 001, 57.6% of case D's pinned requests completed in under 10 µs server-side; in run 002 only 21.6% are in the immediate class. **The fix made case D more starved, not less**: run 001's workers were themselves slowed by channel head-of-line queueing (client mean 3.9 ms), so their demand was closer to what the harness could supply, whereas run 002's workers issue requests four times faster against an unchanged supply and therefore park far more often. More parked receivers means more of `fill`'s publishes pay a wakeup.
- **More connections in the scheduler process.** Case D went from 16 HTTP/2 connections to 320, so the scheduler process's I/O work competes with the tick loop for CPU and cache.

The first fits better. Case C also went from 16 to 160 connections — a 10× rise — and its `fill` per assignment fell 6%; case E went from 16 to 128 and did not move. Only case D, the only case with a large parked population, got dearer. The connection-count mechanism would have to be strongly non-linear between 160 and 320 connections to explain the pattern, whereas the parked-receiver count jumps from near zero to ~250 exactly at case D. This is an inference from four cases, not a controlled result.

Either way it is a small absolute number: case D's whole tick loop cost 461–493 ns per assignment and 1.59% of one core.

### 8.4 Per-job E2E

| Case | Run 001 p50 | Run 002 p50 | Run 001 max | Run 002 max |
|---|---|---|---|---|
| A | 3 469 ms | 2 704 ms | 5 935 ms | 4 628 ms |
| B | 3 676 ms | 2 663 ms | 5 668 ms | 3 687 ms |
| C | 3 584 ms | 1 668 ms | 4 103 ms | 2 293 ms |
| D | 271 ms | 261 ms | 449 ms | 390 ms |
| E | 9 093 ms | 5 949 ms | 12 018 ms | 6 530 ms |

Uniformly shorter, in proportion to the shorter wall clocks. Case D barely moved, as its wall clock barely moved.

## 9. Anomalies and threats to validity

### 9.1 Case D is still bound by the harness's inbound delivery rate, and this run did not address it

Case D overshot its expected duration by 1.99×, against 2.02× in run 001. That is the one number that did not improve, and the cause is a ceiling neither fix touches.

The fake inbound queue returns at most `inbound_wave_size` = 256 tasks per poll, and the core runs one poll at a time. That ceiling is measurable directly from how fast each batch's tasks were first seen, and it is a fixed harness property independent of load:

| Case | Batch 0 | Batch 1 | Batch 2 | Batch 3 | Implied inbound rate (tasks/s) |
|---|---|---|---|---|---|
| A | 32 219 | 32 108 | 31 927 | 32 030 | 31 900 – 32 200 |
| B | 32 333 | 32 652 | 32 681 | 32 609 | 32 300 – 32 700 |
| C | 33 001 | 33 352 | 32 938 | 33 028 | 32 900 – 33 400 |
| D | 32 983 | 33 212 | 33 179 | 33 161 | 33 000 – 33 200 |
| E | 33 324 | 33 251 | 33 266 | 33 209 | 33 200 – 33 300 |

Case D is the only case whose worker demand exceeds it: 320 workers at one task per ~6 ms effective sleep is ~53 000 tasks/s (64 000 at the nominal 5 ms) against a supply of ~33 000. Its measured 32 181 tasks/s is **97.3% of the inbound ceiling**. Three independent signatures confirm it ran starved rather than merely slow:

- **Publication exactly tracked arrival.** D's assignments per busy tick is 256 at p10, p25, p50, p75, p90 *and* p99 — precisely one inbound wave — over 4 152 busy ticks. The core published what had just arrived and had nothing buffered. A, B, C and E have medians of 20, 34, 59 and 49 against maxima of 40, 81, 162 and 129, which is what having a backlog looks like.
- **Workers blocked waiting for work.** 78.3% of D's requests fell into the waited class, against 0.037%–0.150% in every other case.
- **Resource groups sat idle.** D's mean active resource-group count was 2.18 of 8, while A, B, C and E sat at 0.975/1, 1.949/2, 3.898/4 and 7.919/8 — essentially always at maximum. Three quarters of D's groups had nothing to schedule at any given moment.

So case D's 1.99× overshoot is a harness inbound-supply limit. Its per-tick timing profile reflects bursty wave-driven publication, its wall clock and throughput are harness figures, and its waited series is a supply measurement. Its per-assignment tick costs and its immediate-class dispatch latency remain valid — but the latter now rests on 21.7% of its requests. Raising `inbound_wave_size` or issuing overlapping polls is what would give case D an unstarved measurement, and until that is done case D cannot be used to characterize the scheduler at 320 workers.

### 9.2 The client-side median still rises with worker count, and it is no longer head-of-line queueing

At a strict one-channel-per-worker ratio the client-side immediate p50 is still monotone in coroutine count while the server side is flat:

| Case | Workers | Channels | Client immediate p50 | Server immediate p50 |
|---|---|---|---|---|
| A | 40 | 40 | 348 µs | 0.50 µs |
| B | 80 | 80 | 352 µs | 0.33 µs |
| E | 128 | 128 | 410 µs | 0.33 µs |
| C | 160 | 160 | 446 µs | 0.33 µs |
| D | 320 | 320 | 803 µs | 0.35 µs |

With one channel per worker this cannot be HTTP/2 head-of-line queueing on a shared connection. What remains is one worker process hosting N tokio coroutines and N loopback connections on a 32-core host: runtime scheduling of the response, socket wakeups, and framing. The scaling is far milder than run 001's — case C is 1.28× case A instead of 8.3× — but it is not zero, and case D's 803 µs is additionally confounded because its immediate requests share a process with 630 650 parked ones.

The practical consequence for the multi-node runs: **803 µs should be read as an upper bound on client-side cost at 320 coroutines per process, not as a transport figure**, and a multi-node run should either shard workers across processes or record per-process coroutine density alongside the channel count, so the network cost is not confounded with this the way run 001's was confounded with pooling.

### 9.3 Case D's 32.7 ms `process` tick

Tick 62 of case D's 15 561 recorded a `process` step of 32 706 681 ns, against a case p99 of 37 042 ns — three orders of magnitude out. It is a run-in tick: about 0.13 s into the run, publishing no assignments, at the moment 320 worker connections are being established against the scheduler process. It contributes 28% of D's reported `process` mean, 2.1 µs of D's 33.2 µs `total` mean and 6.3% of D's total core time, so §4 reports D's per-assignment figures both with and without it. It sets `total`'s maximum for the case. No other case has a `process` sample above 2.5 ms.

### 9.4 The transport figure is not a wire cost

`client − server` over the immediate class is 348–446 µs for A, B, C and E, against a floor — the smallest round trip observed anywhere — of 41–55 µs and a server-side minimum of 0.087–0.112 µs. The gap between the median and the floor is client-side runtime cost (§9.2), not the loopback. Quoting 348–446 µs as "loopback transport" would repeat run 001's error in a smaller form. What this run establishes is that on this host, with one channel per worker and one worker process, a full round trip costs **tens of microseconds at best and a few hundred at the median**, and that the server contributes under 1 µs of it.

### 9.5 The tick loop ran at half its configured cadence, unchanged from run 001

Configured `tick_interval_ms = 1`; observed mean spacing 2.074 ms (A), 2.070 ms (B), 2.073 ms (C), 2.094 ms (D), 2.100 ms (E) — remarkably constant across five very different loads, and matching run 001's 2.040–2.068 ms. With the core only 0.48%–1.59% busy this is not CPU starvation; it is timer granularity or coalescing under WSL2, the same effect that makes a 5 ms worker sleep behave like ~6 ms (§8.1). It does not affect per-step timings, since each tick is timed individually, but it halves the number of ticks per second and therefore doubles the assignments published per tick relative to a true 1 ms cadence. **Any per-tick figure in §4 should be paired with its per-assignment counterpart**, which is cadence-independent.

### 9.6 Single-sample tails

Every one of these is a single sample sitting orders of magnitude above its own p99.9, moving no mean or percentile of interest, and each is recorded so it is not mistaken for a distributional feature:

| Series | max | p99.9 | ratio |
|---|---|---|---|
| A client general immediate | 44 817 µs | 1 098 µs | 41× |
| C client dedicated immediate | 40 794 µs | 1 360 µs | 30× |
| E client shared immediate | 48 488 µs | 1 163 µs | 42× |
| A server dedicated immediate | 528 µs | 5.4 µs | 97× |
| B server dedicated immediate | 916 µs | 2.9 µs | 318× |
| E server shared immediate | 1 167 µs | 2.5 µs | 474× |
| A tick `fill` | 1 811 µs | p99 25.8 µs | 70× |
| D tick `process` | 32 707 µs | p99 37.0 µs | 883× |

The server-side ones are consistent with OS preemption of the handler; the client-side ones with the same on the worker side. None is explained beyond that, and none is reproducible from this data.

### 9.7 The batch release rule cost little idle time

Exactly two stretches of 20 or more consecutive idle ticks occurred in each case, at the very start and the very end of the run (the initial ramp and the 500 ms drain linger), totalling 0.593 s (A), 0.602 s (B), 0.609 s (C), 0.634 s (D) and 0.607 s (E). The barrier between batches is invisible as idle time because the core has already buffered the whole batch and keeps publishing from its backlog while the previous batch's tail completes. It is very visible in the E2E distributions (§7.1) — it is what makes them multi-modal — but it is not a material contributor to the wall-clock overshoot.

### 9.8 Instrumentation cost

Tick timing takes seven `Instant::now()` readings per poll-draining tick and five otherwise; the residual between `total` and the sum of the five steps averages 56–86 ns per tick, i.e. about one clock read, and was never negative on any of the 72 123 ticks in this run. Dispatch latency takes two readings and a handful of relaxed atomic increments per request, with no lock on the request path; per-job progress uses a `DashMap` with atomic fields; client-side samples stay in per-coroutine owned `Vec`s and are merged only after the run. `fill`'s cost of 303–512 ns per assignment is two to three orders of magnitude above the per-request instrumentation cost.

This run's fixes did not increase instrumentation cost measurably: the four histograms per side replace two, but a request records into exactly one of them, so the per-request work is unchanged. What did change is that the histograms no longer mix populations, which was the point.

### 9.9 A residual asymmetry in the results schema

The worker result file records a single pooled `num_empty_responses`, while the scheduler result file splits it into `num_pinned_empty_responses` and `num_general_empty_responses`. All three are zero in all five cases, so nothing is lost here, but a future run that produces a non-zero empty count will not be able to attribute it to a path from the client side. Worth aligning before a run that expects empty responses.

### 9.10 The two wall clocks are not nested

The scheduler-internal wall clock exceeds the script-level one in four of five cases: A 21.465 vs 21.323 s, C 21.722 vs 21.452 s, D 32.584 vs 32.412 s, E 53.411 vs 52.632 s; only B is the other way round (21.327 vs 21.735 s). The scheduler's clock starts when the fake inbound queue first emits a task, which happens as soon as the core is ready and therefore before the script takes its own start timestamp, and it ends after the 500 ms drain linger; the script's clock brackets worker-process launch and teardown instead. Neither contains the other. The differences are within ±0.8 s in every case and §3.1 uses the scheduler-internal figure throughout.

### 9.11 What these numbers still do not establish

- **They do not establish a network dispatch cost.** Everything ran on one host over loopback, from one worker process. `client − server` is now free of the shared-channel artifact but is still dominated by worker-process coroutine scheduling (§9.2, §9.4), so it is a *bound* on the loopback baseline rather than the baseline itself.
- **They do not establish the scheduler's throughput ceiling.** The highest duty cycle observed was 1.59% of one core. The scheduler's maximum sustainable assignment rate is above everything measured here, and by how much is unknown, because the harness could not offer more load — inbound delivery topped out at ~33k tasks/s (§9.1).
- **They do not characterize the scheduler at 320 workers.** Case D is the only case at that scale and it ran supply-starved (§9.1), so its wall clock, throughput, waited series and client-side series are harness measurements. Only its per-assignment tick costs and its immediate-class dispatch latency survive, and the latter over 21.7% of its requests.
- **They do not establish behaviour under contention or failure.** Every case ran on an idle 32-core host with 1–8 groups whose loads were identical by construction. Nothing here measures interference between groups of unequal weight or demand, worker or scheduler failure, storage-session bumps, or task rescheduling — the reschedule queue was never exercised, and there were zero empty responses, zero warnings and zero errors in all ten process logs.
- **They do not establish the effect of the admission policy's parameters.** `B` was derived per case from `B = R × (N + 1)` and α was fixed at 1; no case varied either, so the policy's sensitivity to them is unmeasured. The fitted `fill` marginal costs in §4.1 vary with case in ways confounded by `B`, by assignments per tick and by the pinned/shared mix, and should not be read as a clean per-group scaling law.
- **They do not establish steady-state behaviour over long runs.** The longest case ran 53 s across four batches. There is no evidence of drift within that window — case D's throughput held at 159k, 162k, 161k, 162k, 161k and 161k tasks per successive 5 s log interval — but hours-long behaviour, memory growth and registry churn are untested.
- **They do not explain case D's 41% dearer `fill`.** §8.3 gives the mechanism that best fits four cases' worth of evidence; separating it from the connection-count alternative needs a run that varies the parked-receiver count and the channel count independently.

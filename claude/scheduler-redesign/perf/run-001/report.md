# Prototype Scheduler Performance Evaluation

This is the report required by [benchmark-contract.md](../../benchmark-contract.md) §4.2. It reports the five cases A–E of §2, measured with the instrumentation of §1 and the harness of §3, from the raw results in this directory.

Every number below was recomputed from the raw JSON files rather than copied from a run log. Percentiles over the dispatch-latency histograms were recomputed from the stored bucket arrays with the same nearest-rank convention the harness uses; percentiles over the per-tick and per-job series were computed from the raw samples.

The headline is stated up front so nothing below is read as a claim it does not support: **the scheduler core was never the bottleneck in any of the five cases** — it spent between 0.38% and 1.12% of one core inside `tick()` — and **every case overshot its expected steady-state duration by 1.63×–2.02× for reasons that lie in the harness, not in the scheduler**. What the tick-step and server-side dispatch numbers measure is therefore the scheduler's cost; what the wall clocks measure is the harness's throughput ceiling.

## 1. What was run

| Case | #RG | Total tasks | Workers (ded. + shared) | `B` | Wall clock (s) | Expected (s) | Ratio | Throughput (tasks/s) | Ticks | Core duty cycle |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 1 | 131 072 | 32 + 8 = 40 | 80 | 26.635 | 16.384 | 1.63× | 4 921 | 13 053 | 0.38% |
| B | 2 | 262 144 | 64 + 16 = 80 | 120 | 29.150 | 16.384 | 1.78× | 8 993 | 14 236 | 0.53% |
| C | 4 | 524 288 | 128 + 32 = 160 | 200 | 30.031 | 16.384 | 1.83× | 17 458 | 14 682 | 0.78% |
| D | 8 | 1 048 576 | 256 + 64 = 320 | 360 | 33.047 | 16.384 | 2.02× | 31 730 | 16 074 | 1.12% |
| E | 8 | 1 048 576 | 0 + 128 = 128 | 144 | 73.978 | 40.960 | 1.81× | 14 174 | 35 773 | 0.91% |

Wall clock is the scheduler-internal measure of §1.3 — first inbound emission to last completion received, plus the harness's 500 ms drain linger. The script-level wall clock, which additionally covers worker process start and teardown, was 27.056 s / 29.570 s / 30.442 s / 33.463 s / 74.398 s; it is recorded in `<case>-run.json`. Core duty cycle is the sum of every tick's `total_nanos` divided by the wall clock, i.e. the fraction of one core the scheduling loop consumed.

Environment: single host, Linux 6.6.87.2 under WSL2, 32 cores, `--release` build, scheduler and workers as two processes on the same machine talking over loopback `http://127.0.0.1:50151`, one worker process hosting all worker coroutines, gRPC `channel_pool_size = 16` (the binary default) in every case. Cases ran strictly sequentially with the machine otherwise idle, one run each, no retries. Neither process log contains a single `WARN` or `ERROR` line in any case.

Files: `run-001/<case>-scheduler.json` (config, every tick sample, every per-job record, the two server-side histograms), `run-001/<case>-workers-0.json` (config, channel pool size, the two client-side histograms, request tallies), `run-001/<case>-run.json` (script-level wall clock), `run-001/logs/`.

## 2. Terminology

### 2.1 Tick step timings

The core's tick has the five steps of [design.md](../../design.md) §5, and every tick is timed per step and in total ([benchmark-contract.md](../../benchmark-contract.md) §1.1):

| Key | Step | Covers |
|---|---|---|
| `collect` | 1 | Draining the inbound poll result and the reschedule queue |
| `process` | 2 | Deduplicating and grouping the polled entries into `rg_updates` |
| `apply` | 3 | Creating scheduling units, placing new jobs, and starting the next inbound poll |
| `fill` | 4 | The round-robin admission loop — the scheduling policy itself |
| `retire` | 5 | Removing retired jobs from the job registry |
| `total` | — | The whole tick |

Two deviations from the contract's table are baked into the data and must be read with it. First, `start_inbound_poll` is billed to `apply` on a tick that drained a poll result and to `collect` on a tick that had no poll in flight, rather than to `collect` unconditionally as the contract's table describes. Second, a tick that drained a poll result takes seven `Instant::now()` readings rather than the contract's six — `TickTimer::start`, five `finish_step` calls, and `TickTimer::finish` closing `total_nanos` — which is why `total` exceeds the sum of the five steps by one clock read's worth on every tick: a mean residual of 58–80 ns per case, and never negative on any of the 93 818 ticks recorded across the five cases.

Steps 2 and 3 are skipped entirely, and therefore record zero, on any tick where the inbound poll is still in flight. That is why `process` and `apply` have a median of 0 in most cases; their means over all ticks are still the right per-tick cost, but their medians are not.

### 2.2 Dispatch latency

Two durations are recorded for every request that **returns an assignment**, and only for those; a request that returned nothing is counted (`num_empty_responses`, zero in all five cases) but excluded, because including it would measure the long-poll wait rather than dispatch cost.

**Server-side** is measured in the service handler around the `DispatchService::next_task_*` call. It excludes transport and serialization, and it is the trustworthy measure of what the scheduler costs per request.

**Client-side** is measured in the worker coroutine around the gRPC call, from immediately before the request to immediately after the response. It includes transport, serialization, and the server time — so it is always the larger of the two, and it contains everything that happens between the two processes.

`client − server` is therefore the transport and framing overhead. On this single-node run it is loopback cost, and it is recorded precisely so that a later multi-node run has a baseline to subtract from. **The caveat that dominates this report is that in these five runs `client − server` is not a clean measure of loopback cost** — see §8.1.

Both are reported separately for **dedicated (pinned)** and **shared (general)** execution managers, because the paths differ structurally: a pinned request touches one group's queue and nothing else, while a general request pops the hint channel and may traverse several stale hints before it finds work.

A note on the server-side series that applies to every case, and severely to case D: the timer sits around the `next_task_*` call as the contract specifies, so a request that *blocked inside that call* waiting for work and then received an assignment bills its entire wait as dispatch latency. Nothing was moved; the effect is visible as a bimodal distribution, and where it matters the sub-population below 10 µs is reported separately.

### 2.3 Per-job end-to-end time

**First seen** is when the fake inbound queue first emits a task belonging to the job; **last completed** is when the dispatch service received the last completion report for a task of that job; **E2E** is the difference.

**A job's E2E includes one extra round trip.** A worker reports a task's completion on its *next* `NextTask` request, not when it finishes executing. So the last completion of a job is observed one request cycle after that task actually finished, and every E2E figure below is inflated by one client-side round trip (§6) relative to the true execution span. This is also why the run does not end when the last task is dispatched: it ends when the last completion is received, which is what the harness's drain linger exists to flush.

E2E is *not* a per-task service-time measure. It spans the job's whole emission-to-completion life, including all queueing behind the backlog the core has buffered, so its distribution is shaped mostly by the batch release rule (§3) — see §7.

### 2.4 Histogram resolution

Dispatch latency is stored as a fixed-bucket atomic histogram of 1984 buckets: the first 64 ns resolved to the nanosecond, and every octave above that split into 64 buckets, giving ≤1.6% relative error anywhere above 64 ns. Percentiles use the nearest-rank method reported at the containing bucket's upper bound, so a quoted percentile is never understated. Count, mean, min and max are exact, being maintained separately from the buckets.

## 3. Configuration

Constant across every case:

| Parameter | Value |
|---|---|
| Jobs per resource group | 128, released in 4 batches of 32 |
| Tasks per job | 1024 |
| Task execution time | 5 ms, simulated by `tokio::time::sleep` in the worker |
| Inbound wave size | 256 tasks per poll response |
| `active_job_list_capacity` | 16 |
| `tick_interval_ms` | 1 |
| `storage_poll_timeout_ms` | 5 |
| `ready_task_capacity` | the case's total task count |
| Dedicated workers per group | 32 (0 in case E) |
| Shared workers | 8 × #RG (128 in case E) |

The five cases:

| Case | #RG | Total tasks | Shared | Dedicated | Total workers | Reserve `R` | `B` |
|---|---|---|---|---|---|---|---|
| A | 1 | 131 072 | 8 | 32 | 40 | 40 | 80 |
| B | 2 | 262 144 | 16 | 64 | 80 | 40 | 120 |
| C | 4 | 524 288 | 32 | 128 | 160 | 40 | 200 |
| D | 8 | 1 048 576 | 64 | 256 | 320 | 40 | 360 |
| E | 8 | 1 048 576 | 128 | 0 | 128 | 16 | 144 |

**Deriving `B`.** The requirement is that at least `R` slots of the dispatch queue stay free when every resource group is active, with `R = (shared workers / #RG) + (dedicated workers per group)`. Under the admission policy of [design.md](../../design.md) §6.2 with α = 1 and `N` backlogged groups, free space settles at `F = B / (N + 1)`, so requiring `F ≥ R` gives `B = R × (N + 1)`, which is the `B` column above. Case E has no dedicated workers, so `R = 128 / 8 = 16` and `B = 16 × 9 = 144`.

**Batch release rule.** Batch `n+1` is released once every job in batch `n` has had all of its tasks reported complete. This is an *interpretation* of "jobs are created in batches" rather than something the design specifies, and it is recorded here because it materially shapes the results: it is the direct cause of the E2E distribution shapes in §7 and of the staircase in job arrival visible in every case. With 32 jobs per batch against an active list of 16, there is always a pending queue.

## 4. Correctness

The completion count per job is the run's validity gate: it must equal `tasks_per_job` = 1024 for every job, or the timings mean nothing.

| Case | Jobs expected | Jobs present | Jobs with exactly 1024 completions | Sum of completions | Total tasks | RGs represented | Empty responses | Client samples | Server samples |
|---|---|---|---|---|---|---|---|---|---|
| A | 128 | 128 | **128** | 131 072 | 131 072 | 1 of 1 | 0 | 131 072 | 131 072 |
| B | 256 | 256 | **256** | 262 144 | 262 144 | 2 of 2 | 0 | 262 144 | 262 144 |
| C | 512 | 512 | **512** | 524 288 | 524 288 | 4 of 4 | 0 | 524 288 | 524 288 |
| D | 1024 | 1024 | **1024** | 1 048 576 | 1 048 576 | 8 of 8 | 0 | 1 048 576 | 1 048 576 |
| E | 1024 | 1024 | **1024** | 1 048 576 | 1 048 576 | 8 of 8 | 0 | 1 048 576 | 1 048 576 |

In every case, every job completed exactly its task count; no job over- or under-completed; no completion was lost or double-counted. Three independent counts agree exactly in each case: the sum of per-job completion counts, the sum of `assignments_published` over all ticks, and the client-side and server-side latency sample counts. No job carried the `UNKNOWN_RESOURCE_GROUP_ID` fallback, and all per-job E2E values are positive and finite.

## 5. Tick step timings

Per-step figures follow, per case. `per assignment` divides the step's total time over the whole run by the run's total assignments published, which is the figure to compare across cases — per-tick figures are not comparable between cases because the number of assignments per tick differs by an order of magnitude (§8.2). `busy-tick mean` restricts to ticks that published at least one assignment.

### Case A — 13 053 ticks, 8 764 with assignments, 131 072 assignments

| Step | mean (ns) | p50 (ns) | p99 (ns) | max (ns) | busy-tick mean (ns) | per assignment (ns) | share of tick |
|---|---|---|---|---|---|---|---|
| `collect` | 828 | 292 | 3 933 | 154 238 | 810 | 82.5 | 10.6% |
| `process` | 1 040 | 0 | 23 877 | 1 161 181 | 946 | 103.5 | 13.3% |
| `apply` | 554 | 0 | 3 470 | 62 556 | 536 | 55.1 | 7.1% |
| `fill` | 5 242 | 3 754 | 19 246 | 73 235 | 7 454 | 522.1 | 67.0% |
| `retire` | 81 | 67 | 191 | 54 902 | 82 | 8.1 | 1.0% |
| `total` | 7 824 | 5 992 | 35 385 | 1 175 514 | 9 902 | 779.2 | 100% |

### Case B — 14 236 ticks, 10 259 with assignments, 262 144 assignments

| Step | mean (ns) | p50 (ns) | p99 (ns) | max (ns) | busy-tick mean (ns) | per assignment (ns) | share of tick |
|---|---|---|---|---|---|---|---|
| `collect` | 697 | 236 | 3 463 | 30 606 | 688 | 37.9 | 6.4% |
| `process` | 1 842 | 0 | 27 926 | 1 502 003 | 1 910 | 100.0 | 16.9% |
| `apply` | 455 | 0 | 2 913 | 17 468 | 443 | 24.7 | 4.2% |
| `fill` | 7 742 | 5 941 | 28 586 | 112 270 | 10 478 | 420.4 | 71.2% |
| `retire` | 71 | 53 | 354 | 22 713 | 71 | 3.9 | 0.7% |
| `total` | 10 874 | 8 357 | 42 857 | 1 514 134 | 13 651 | 590.5 | 100% |

### Case C — 14 682 ticks, 11 704 with assignments, 524 288 assignments

| Step | mean (ns) | p50 (ns) | p99 (ns) | max (ns) | busy-tick mean (ns) | per assignment (ns) | share of tick |
|---|---|---|---|---|---|---|---|
| `collect` | 692 | 251 | 3 339 | 42 710 | 697 | 19.4 | 4.3% |
| `process` | 3 086 | 0 | 31 013 | 1 472 432 | 3 214 | 86.4 | 19.3% |
| `apply` | 454 | 0 | 2 688 | 18 195 | 458 | 12.7 | 2.8% |
| `fill` | 11 653 | 10 214 | 39 787 | 118 211 | 14 432 | 326.3 | 72.7% |
| `retire` | 78 | 55 | 681 | 17 485 | 78 | 2.2 | 0.5% |
| `total` | 16 030 | 13 998 | 56 119 | 1 495 542 | 18 945 | 448.9 | 100% |

### Case D — 16 074 ticks, 4 456 with assignments, 1 048 576 assignments

| Step | mean (ns) | p50 (ns) | p99 (ns) | max (ns) | busy-tick mean (ns) | per assignment (ns) | share of tick |
|---|---|---|---|---|---|---|---|
| `collect` | 781 | 244 | 4 071 | 66 969 | 2 088 | 12.0 | 3.4% |
| `process` | 4 327 | 0 | 28 891 | 497 237 | 15 521 | 66.3 | 18.8% |
| `apply` | 621 | 0 | 3 396 | 16 693 | 2 207 | 9.5 | 2.7% |
| `fill` | 16 914 | 1 904 | 91 922 | 2 040 696 | 56 604 | 259.3 | 73.4% |
| `retire` | 325 | 59 | 2 105 | 32 231 | 71 | 5.0 | 1.4% |
| `total` | 23 030 | 2 902 | 122 776 | 2 070 172 | 76 553 | 353.0 | 100% |

### Case E — 35 773 ticks, 28 470 with assignments, 1 048 576 assignments

| Step | mean (ns) | p50 (ns) | p99 (ns) | max (ns) | busy-tick mean (ns) | per assignment (ns) | share of tick |
|---|---|---|---|---|---|---|---|
| `collect` | 641 | 222 | 2 921 | 53 276 | 646 | 21.9 | 3.4% |
| `process` | 3 864 | 0 | 41 840 | 3 397 029 | 4 041 | 131.8 | 20.5% |
| `apply` | 450 | 0 | 2 623 | 41 058 | 453 | 15.3 | 2.4% |
| `fill` | 13 789 | 11 988 | 45 401 | 581 708 | 17 103 | 470.4 | 73.1% |
| `retire` | 67 | 52 | 148 | 18 726 | 66 | 2.3 | 0.4% |
| `total` | 18 868 | 15 817 | 68 483 | 3 426 196 | 22 364 | 643.7 | 100% |

### 5.1 The scaling trend across #RG

This is the question §1.1 exists to answer, so it is stated as a trend table rather than left in the per-case tables.

| Quantity | A (1 RG) | B (2 RG) | C (4 RG) | D (8 RG) | E (8 RG, no dedicated) |
|---|---|---|---|---|---|
| `fill` share of tick | 67.0% | 71.2% | 72.7% | 73.4% | 73.1% |
| `retire` share of tick | 1.0% | 0.7% | 0.5% | 1.4% | 0.4% |
| `fill` per assignment (ns) | 522 | 420 | 326 | 259 | 470 |
| `process` per inbound entry (ns) | 104 | 100 | 86 | 66 | 132 |
| `collect` per assignment (ns) | 83 | 38 | 19 | 12 | 22 |
| `apply` per assignment (ns) | 55 | 25 | 13 | 9.5 | 15 |
| `retire` per assignment (ns) | 8.1 | 3.9 | 2.2 | 5.0 | 2.3 |
| `total` per assignment (ns) | 779 | 591 | 449 | 353 | 644 |
| Assignments per busy tick (mean) | 15.0 | 25.6 | 44.8 | 235.3 | 36.8 |
| Total core time in `tick()` (ms) | 102 | 155 | 235 | 370 | 675 |

**Step 4 dominates and step 5 is near zero, exactly as the contract predicts.** `fill` is 67–73% of tick time in every case and its share grows slightly with #RG; `retire` never exceeds 1.4% and costs 2–8 ns per assignment.

**Nothing scales super-linearly in #RG.** Going from 1 to 8 resource groups with 8× the tasks and 8× the workers, the cost of the whole tick loop *per assignment published* went **down**, from 779 ns to 353 ns, and the total core time spent scheduling grew 3.6× for 8× the work. Per-assignment cost falls because a per-tick fixed cost is amortized over more assignments as the load grows, not because any step got cheaper intrinsically.

To separate the fixed and marginal parts of `fill`, a per-case least-squares fit of `fill_nanos` against `assignments_published` over every tick gives:

| Case | #RG | Fixed cost per tick (ns) | Marginal cost per assignment (ns) | R² |
|---|---|---|---|---|
| A | 1 | 1 067 | 416 | 0.80 |
| B | 2 | 1 118 | 360 | 0.83 |
| C | 4 | 1 367 | 288 | 0.85 |
| D | 8 | 1 599 | 235 | 0.65 |
| E | 8 | 1 497 | 419 | 0.79 |

The fixed cost of one round-robin pass grows with the number of resource groups but sub-linearly: +530 ns going from 1 to 8 groups, roughly 76 ns per additional group, against a per-tick baseline of about 1.07 µs. The fitted marginal cost per assignment falls from 416 ns to 235 ns across A→D; case E, which has the same 8 groups but no dedicated workers and a third of D's dispatch queue capacity, sits at 419 ns. The A→D decline should not be read as "admission gets cheaper with more groups" — the fits are per-case and confounded with how many assignments a tick publishes (which varies 15× across the cases) and with cache behaviour at those batch sizes. What the fits do establish is that `fill` is well described as a modest per-tick constant plus a few hundred nanoseconds per assignment, and that neither term blows up at 8 resource groups.

`process` costs 66–132 ns per inbound entry and is the second-largest step from case B onward, reaching 18.8–20.5% of tick time at 8 groups. It is also the step with the heaviest tail: its maxima (1.16 ms in A, 1.50 ms in B, 1.47 ms in C, 0.50 ms in D, 3.40 ms in E) are the only per-step outliers large enough to drive `total`'s maximum, and they land on the ticks that drain a 256-task inbound wave right at a batch-arming boundary. These are single ticks out of tens of thousands, and p99 for the step stays between 24 µs and 42 µs.

Case D's per-tick columns look different from the rest — a busy-tick `total` mean of 76.6 µs against C's 18.9 µs — purely because D published 235 assignments per busy tick where C published 45. Per assignment, D is the *cheapest* case in the set. The reason D's publication is so lumpy is in §8.2.

The absolute cost is small enough that it bears restating: the entire scheduling loop consumed 102–675 ms of CPU across runs lasting 27–74 s, i.e. 0.38%–1.12% of a single core, on a 32-core host. No case came close to saturating the core.

## 6. Dispatch latency

All values in **microseconds**. `n` counts only assignment-returning requests; there were no empty responses in any case. Dedicated = pinned path, shared = general path.

| Case | Path | Side | n | min | p50 | p90 | p99 | p99.9 | max | mean | % under 10 µs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | dedicated | server | 105 621 | 0.15 | 0.70 | 1.70 | 2.94 | 44.03 | 17 697 | 11.04 | 99.83% |
| A | dedicated | client | 105 621 | 45.99 | 327.68 | 6 750.21 | 7 405.57 | 21 233.66 | 47 359 | 1 947.15 | 0% |
| A | shared | server | 25 451 | 0.13 | 0.96 | 1.85 | 3.13 | 25.86 | 17 777 | 12.08 | 99.83% |
| A | shared | client | 25 451 | 50.94 | 360.45 | 6 815.74 | 7 405.57 | 14 811.14 | 46 274 | 2 245.99 | 0% |
| B | dedicated | server | 209 827 | 0.15 | 0.50 | 1.17 | 2.21 | 39.42 | 25 267 | 14.34 | 99.86% |
| B | dedicated | client | 209 827 | 45.94 | 1 441.79 | 6 881.28 | 7 471.10 | 19 922.94 | 47 398 | 2 766.54 | 0% |
| B | shared | server | 52 317 | 0.13 | 0.66 | 1.28 | 2.30 | 24.83 | 19 889 | 12.68 | 99.87% |
| B | shared | client | 52 317 | 45.09 | 1 507.33 | 6 881.28 | 7 471.10 | 13 369.34 | 47 357 | 2 791.52 | 0% |
| C | dedicated | server | 419 412 | 0.15 | 0.50 | 1.15 | 1.79 | 23.04 | 46 932 | 21.44 | 99.87% |
| C | dedicated | client | 419 412 | 41.65 | 2 719.74 | 6 750.21 | 7 471.10 | 28 311.55 | 53 559 | 3 046.03 | 0% |
| C | shared | server | 104 876 | 0.13 | 0.65 | 1.22 | 1.87 | 45.57 | 19 808 | 12.70 | 99.87% |
| C | shared | client | 104 876 | 64.78 | 2 719.74 | 6 750.21 | 7 471.10 | 19 398.65 | 45 914 | 3 045.78 | 0% |
| D | dedicated | server | 830 369 | 0.15 | 1.15 | 3 964.93 | 6 422.53 | 8 781.82 | 25 883 | 1 140.46 | 57.55% |
| D | dedicated | client | 830 369 | 63.39 | 3 899.39 | 7 208.96 | 11 534.33 | 15 073.28 | 49 794 | 4 068.58 | 0% |
| D | shared | server | 218 207 | 0.13 | 1.31 | 3 276.80 | 5 570.56 | 7 602.18 | 24 004 | 1 025.63 | 54.78% |
| D | shared | client | 218 207 | 86.46 | 3 342.34 | 6 946.81 | 10 485.76 | 14 417.92 | 48 674 | 3 580.31 | 0% |
| E | dedicated | server | 0 | — | — | — | — | — | — | — | — |
| E | dedicated | client | 0 | — | — | — | — | — | — | — | — |
| E | shared | server | 1 048 576 | 0.13 | 0.44 | 1.05 | 1.74 | 4.80 | 31 072 | 6.74 | 99.93% |
| E | shared | client | 1 048 576 | 44.04 | 2 080.77 | 6 750.21 | 7 471.10 | 8 388.61 | 48 289 | 2 931.30 | 0% |

Case E has no dedicated workers by construction, so its pinned series are empty by design and it contributes no dedicated-versus-shared comparison. Its shared row covers all 1 048 576 requests.

### 6.1 Server-side: what a dispatch costs

For A, B, C and E — the four cases where workers were not starved — **99.83% or more of requests completed in under 10 µs on the server side, with a median between 0.44 µs and 0.96 µs and a p99 between 1.7 µs and 3.1 µs.** That is the scheduler's cost to hand out one task, and it did not degrade as the case grew: case E's median of 0.44 µs over a million requests is the *fastest* of the set.

Server-side means (6.7–21.4 µs) are an order of magnitude above the corresponding p99s in these four cases, and the maxima reach 17–47 ms. That entire discrepancy is long-poll wait folded into the measured window (§2.2), affecting under 0.2% of samples. **Do not quote the server-side mean, p99.9 or max as dispatch cost.** The p50/p90/p99 columns are the ones that describe it.

Case D is different in kind, not degree: only 57.6% (dedicated) and 54.8% (shared) of its requests finished in under 10 µs. In D the workers spent nearly half their requests blocked inside `next_task_*` waiting for work to exist, and that wait is billed as latency. Restricting to the sub-10 µs sub-population — the requests that found work already waiting — D's dispatch cost is entirely ordinary:

| Case D, sub-10 µs sub-population | n | p50 | p90 | p99 |
|---|---|---|---|---|
| dedicated | 477 916 | 0.64 µs | 1.20 µs | 1.55 µs |
| shared | 119 535 | 0.78 µs | 1.28 µs | 2.02 µs |

Those figures are in line with A, B, C and E. **Case D's server-side p90 of ~4 ms is a starvation measurement, not a cost measurement**, and §8.2 identifies what starved it.

### 6.2 Dedicated versus shared

The shared (general) path is consistently more expensive than the dedicated (pinned) path at the median, by 22%–37%:

| Case | dedicated p50 | shared p50 | shared premium | dedicated p99 | shared p99 |
|---|---|---|---|---|---|
| A | 0.70 µs | 0.96 µs | +37% | 2.94 µs | 3.13 µs |
| B | 0.50 µs | 0.66 µs | +33% | 2.21 µs | 2.30 µs |
| C | 0.50 µs | 0.65 µs | +30% | 1.79 µs | 1.87 µs |
| D (sub-10 µs) | 0.64 µs | 0.78 µs | +22% | 1.55 µs | 2.02 µs |

This is the structural difference §1.2 predicts: a pinned request touches one group's queue, a general request pops the hint channel and may traverse stale hints. In absolute terms the premium is 140–260 ns, and the two paths converge by p99, where the gap is under 0.5 µs in every case. The request mix followed the worker split exactly — 80.6% / 80.0% / 80.0% / 79.2% of requests took the pinned path in A–D, against an 80% dedicated-worker share, and 0% in E.

### 6.3 `client − server` on loopback

This is the number a multi-node run would be compared against, so it is stated with its caveat attached rather than in isolation.

| Case | dedicated p50 client − server | shared p50 client − server | smallest observed round trip (client min) | server min |
|---|---|---|---|---|
| A | 327.0 µs | 359.5 µs | 45.99 µs | 0.15 µs |
| B | 1 441.3 µs | 1 506.7 µs | 45.09 µs | 0.13 µs |
| C | 2 719.2 µs | 2 719.1 µs | 41.65 µs | 0.13 µs |
| D | 3 898.2 µs | 3 341.0 µs | 63.39 µs | 0.13 µs |
| E | — | 2 080.3 µs | 44.04 µs | 0.13 µs |

**These median differences are not loopback transport cost.** They rise monotonically with the number of worker coroutines (40 → 80 → 160 → 320 workers gives 0.33 → 1.44 → 2.72 → 3.90 ms) while the server-side cost stays flat at well under 2 µs, and every case ran with `channel_pool_size = 16`, i.e. 2.5, 5, 10, 20 and 8 worker coroutines sharing each HTTP/2 connection. What the difference measures is head-of-line queueing inside the worker process's shared channels — a client-side artifact of the harness, not a property of the scheduler or of the transport. The distinctive client p90 of 6.75–7.21 ms, nearly identical across four cases with completely different loads, is the signature of that queueing rather than of anything load-dependent.

The defensible statement about loopback cost from this data is the **floor**: the smallest round trip observed was 41.65–86.46 µs against a server-side minimum of 0.13 µs, so loopback transport plus framing plus worker-side scheduling of the response costs on the order of **tens of microseconds** here, and the unqueued median cannot be smaller than that floor. Obtaining a clean number requires re-running with `BENCH_CHANNEL_POOL_SIZE` set to the case's worker count; that was not done, and it is the single change that would most improve this benchmark (§8.1).

## 7. Per-job end-to-end distributions

All values in milliseconds. Every value includes the extra round trip described in §2.3.

| Case | jobs | min | p10 | p25 | p50 | p75 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 128 | 2 075 | 3 004 | 3 268 | 3 469 | 5 725 | 5 846 | 5 932 | 5 935 | 4 425 |
| B | 256 | 994 | 2 870 | 3 463 | 3 676 | 5 444 | 5 570 | 5 656 | 5 668 | 4 254 |
| C | 512 | 402 | 1 376 | 2 776 | 3 584 | 3 765 | 3 922 | 4 071 | 4 103 | 3 155 |
| D | 1024 | 262 | 266 | 269 | 271 | 275 | 279 | 388 | 449 | 275 |
| E | 1024 | 1 191 | 5 467 | 7 977 | 9 093 | 11 011 | 11 535 | 11 983 | 12 018 | 8 953 |

```
case A (n=128, bin width 322 ms)
    2075 -     2396 ms | ###                                               3
    2396 -     2718 ms | ###                                               3
    2718 -     3040 ms | #######                                           8
    3040 -     3361 ms | ###################################              41
    3361 -     3683 ms | ########                                          9
    3683 -     4005 ms |                                                   0
    4005 -     4326 ms |                                                   0
    4326 -     4648 ms |                                                   0
    4648 -     4970 ms | ###                                               3
    4970 -     5291 ms |                                                   0
    5291 -     5613 ms | ######                                            7
    5613 -     5935 ms | ##############################################   54
```

```
case B (n=256, bin width 389 ms)
     994 -     1384 ms | ###                                               6
    1384 -     1773 ms |                                                   0
    1773 -     2163 ms | ##                                                4
    2163 -     2552 ms | ###                                               7
    2552 -     2942 ms | ######                                           13
    2942 -     3331 ms | #########                                        20
    3331 -     3721 ms | ######################################           82
    3721 -     4110 ms | #                                                 2
    4110 -     4500 ms |                                                   0
    4500 -     4889 ms | ###                                               6
    4889 -     5279 ms | ########                                         17
    5279 -     5668 ms | ##############################################   99
```

```
case C (n=512, bin width 308 ms)
     402 -      710 ms | ###                                              12
     710 -     1019 ms | #####                                            20
    1019 -     1327 ms | ####                                             16
    1327 -     1636 ms | ###                                              12
    1636 -     1944 ms | ###                                              11
    1944 -     2252 ms | ###                                              13
    2252 -     2561 ms | ######                                           23
    2561 -     2869 ms | #####                                            21
    2869 -     3178 ms | ########                                         33
    3178 -     3486 ms | ##############                                   56
    3486 -     3794 ms | ##############################################  183
    3794 -     4103 ms | ############################                    112
```

```
case D (n=1024, bin width 16 ms)
     262 -      277 ms | ##############################################  861
     277 -      293 ms | #######                                         123
     293 -      308 ms | #                                                17
     308 -      324 ms |                                                   5
     324 -      340 ms |                                                   2
     340 -      355 ms |                                                   0
     355 -      371 ms |                                                   0
     371 -      387 ms |                                                   5
     387 -      402 ms |                                                   3
     402 -      418 ms |                                                   0
     418 -      433 ms |                                                   0
     433 -      449 ms |                                                   8
```

```
case E (n=1024, bin width 902 ms)
    1191 -     2094 ms | ######                                           32
    2094 -     2996 ms | #                                                 8
    2996 -     3898 ms | ###                                              24
    3898 -     4800 ms | ######                                           32
    4800 -     5702 ms | #                                                 8
    5702 -     6604 ms | ########                                         48
    6604 -     7507 ms | ##########                                       56
    7507 -     8409 ms | #################                                96
    8409 -     9311 ms | ######################################          264
    9311 -    10213 ms | ########                                         48
   10213 -    11115 ms | ############################                    160
   11115 -    12018 ms | ###########################################     248
```

### 7.1 Reading these shapes

**The distributions are shaped by the batch release rule, not by per-job service variability.** Within a batch, the inbound queue emits jobs' tasks in job order, so the first job of a batch is first-seen at the batch's start while the last is first-seen only after the earlier jobs' tasks have all been emitted — yet all of them finish at roughly the same moment, when the batch's barrier is reached. A job's E2E is therefore mostly a function of its position in its batch, which is why the histograms are multi-modal with a heavy mode at the top of the range and a thin left tail: the jobs emitted last in each batch have the shortest E2E.

Case A, resource group 0, batch by batch, makes the structure explicit:

| Batch | first seen (s) | last completed (s) | E2E range (ms) |
|---|---|---|---|
| 0 | 0.00 – 0.99 | 2.94 – 6.69 | 2 943 – 5 860 |
| 1 | 6.69 – 7.68 | 8.77 – 13.24 | 2 075 – 5 712 |
| 2 | 13.25 – 14.23 | 15.34 – 20.01 | 2 095 – 5 935 |
| 3 | 20.02 – 21.01 | 22.10 – 26.63 | 2 082 – 5 769 |

Each batch is a near-exact repetition of the previous one — the pattern repeats four times per case in all five cases — so there is no drift or degradation as jobs accumulate over a run.

**Case D's tight distribution is a symptom, not an achievement.** Its 1024 jobs all landed within 262–449 ms with a p50 of 271 ms, an order of magnitude tighter and shorter than any other case. That is because D never built a backlog: tasks were executed about as fast as the harness could emit them (§8.2), so a job's E2E is essentially its own emission span (1024 tasks at the ~4.1k tasks/s per group the inbound queue sustained ≈ 248 ms) plus one round trip. Where a backlog exists — A, B, C, E — E2E includes queueing behind it and runs to seconds.

**No resource group was starved relative to another.** Per-group E2E medians agree to within 0.3% in every multi-group case: B 3 674 / 3 688 ms; C 3 584 / 3 586 / 3 592 / 3 587 ms; D 271.0–271.2 ms across all eight; E 9 093–9 101 ms across all eight.

## 8. Anomalies and threats to validity

**Every case overshot its expected steady-state duration**, by 1.63× (A), 1.78× (B), 1.83× (C), 2.02× (D) and 1.81× (E). Per the contract's rule, that means these runs were worker-starved or harness-bound and their wall clocks must not be presented as scheduler cost. Two distinct mechanisms are responsible, and they apply to different cases.

### 8.1 The gRPC channel pool bound cases A, B, C and E

All five cases ran with `channel_pool_size = 16`, the binary's default, which was left unset rather than chosen per case. That puts 2.5 (A), 5 (B), 10 (C), 20 (D) and 8 (E) worker coroutines on each HTTP/2 connection, and the resulting head-of-line queueing is what the client-side series measures (§6.3).

The evidence that it is the binding constraint in A, B, C and E is arithmetic. A worker's per-task cycle is its simulated 5 ms sleep plus one client-side round trip. Timer granularity on this host inflates short sleeps — the tick loop, asking for 1 ms, actually ran at 2.05 ms in every case — so the 5 ms sleep is realistically ~6 ms. Modelling throughput as `workers / (6 ms + client-side mean latency)` gives:

| Case | Workers | Client mean (ms) | Predicted (tasks/s) | Measured (tasks/s) | Error |
|---|---|---|---|---|---|
| A | 40 | 1.95 | 5 030 | 4 921 | −2.2% |
| B | 80 | 2.77 | 9 122 | 8 993 | −1.4% |
| C | 160 | 3.05 | 17 680 | 17 458 | −1.3% |
| D | 320 | 3.97 | 32 106 | 31 730 | −1.2% |
| E | 128 | 2.93 | 14 334 | 14 174 | −1.1% |

Every case is explained to within 2.2% by the worker cycle alone, leaving no residual to attribute to the scheduler. In A, B, C and E the workers were never made to wait for work — 99.83%+ of their requests were served in under 10 µs — so they were rate-limited by their own request cycle, and the dominant term in that cycle after the 5 ms sleep is client-side channel queueing. (For case D the same arithmetic is circular, because its client-side latency itself contains the wait described in §8.2.)

The consequence for this report is narrow but important: the wall clocks, the throughput figures and the client-side latency series characterize *the harness at pool size 16*, not the scheduler. The tick-step timings and the server-side latency series are unaffected, because they are measured inside the scheduler process.

Fixing this requires re-running with `BENCH_CHANNEL_POOL_SIZE` set per case (40, 80, 160, 320, 128). That would also change the wall clocks and E2E distributions, so cases run at a different pool size are not comparable to these; a re-run must cover all five. Note that the contract (§3.2) explicitly warns against one channel per coroutine as an alternative failure mode, so the pool size deserves an explicit decision rather than a default.

### 8.2 Case D was additionally bound by the harness's inbound delivery rate

Case D has a second, independent ceiling that the other cases do not hit.

The fake inbound queue returns at most `inbound_wave_size` = 256 tasks per poll, and the core runs one poll at a time. Across all five cases the poll turnaround was a near-constant 3.84–3.86 tick intervals, i.e. about 7.9 ms at the observed 2.05 ms cadence, which caps inbound delivery at roughly 256 / 7.9 ms ≈ 32 000 tasks/s. That ceiling can be measured directly and independently, from how fast each batch's tasks were first seen:

| Case | Tasks per batch | Batch 0 emission span (s) | Batch 1 | Batch 2 | Batch 3 | Implied inbound rate (tasks/s) |
|---|---|---|---|---|---|---|
| A | 32 768 | 0.99 | 0.99 | 0.98 | 0.99 | 33 000 – 33 400 |
| B | 65 536 | 1.99 | 1.98 | 1.98 | 1.99 | 32 900 – 33 200 |
| C | 131 072 | 3.97 | 3.98 | 4.00 | 3.98 | 32 800 – 33 100 |
| D | 262 144 | 7.94 | 8.00 | 8.08 | 7.95 | 32 500 – 33 000 |
| E | 262 144 | 7.98 | 7.96 | 7.98 | 7.95 | 32 800 – 33 000 |

The inbound queue delivered 32.5k–33.4k tasks/s in every case and every batch, regardless of load — a fixed harness property. Case D is the only case whose worker demand (320 workers × 1 task per 5 ms = 64 000 tasks/s) exceeds it, and D's measured throughput of 31 730 tasks/s is 96% of that ceiling.

Three independent signatures confirm D ran starved rather than merely slow:

- **Publication exactly tracked arrival.** D published a median *and* p90 of 256 assignments per busy tick — precisely one inbound wave — on 4 456 busy ticks against 4 165 ticks that drained a poll and 4 096 waves' worth of tasks. The core published what had just arrived and had nothing buffered. In A, B, C and E busy ticks outnumber poll-draining ticks by 2.6×–3.1×, which is what having a backlog looks like.
- **Workers blocked waiting for work.** 42.5% of D's dedicated requests and 45.2% of its shared requests took longer than 10 µs server-side, versus under 0.2% in every other case; that time is long-poll wait (§2.2).
- **Resource groups sat idle.** D's mean active resource-group count was 4.86 of 8, while A, B, C and E sat at 0.98/1, 1.96/2, 3.92/4 and 7.94/8 — essentially always at maximum. Half of D's groups had nothing to schedule at any given moment.

So case D's 2.02× overshoot is a harness inbound-supply limit, and its per-tick timing profile (235 assignments per busy tick, 56.6 µs of `fill` per busy tick) reflects bursty wave-driven publication rather than a steady rate. Its *per-assignment* costs remain valid and are the cheapest of the set. Raising `inbound_wave_size` or issuing overlapping polls would be needed to give case D an unstarved measurement.

### 8.3 The tick loop ran at half its configured cadence

Configured `tick_interval_ms = 1`, but the observed mean spacing was 2.040 ms (A), 2.048 ms (B), 2.045 ms (C), 2.056 ms (D) and 2.068 ms (E) — remarkably constant across five very different loads. With the core only 0.38%–1.12% busy this is not CPU starvation; it is timer granularity or coalescing under WSL2, the same effect that makes a 5 ms worker sleep behave like ~6 ms (§8.1). It does not affect per-step timings, since each tick is timed individually, but it halves the number of ticks per second and therefore doubles the assignments published per tick relative to a true 1 ms cadence. **Any per-tick figure in §5 should be paired with its per-assignment counterpart**, which is cadence-independent.

### 8.4 The batch release rule cost less idle time than expected

The barrier between batches was expected to drain workers idle while stragglers finished. It did not, measurably: in every case exactly two stretches of 20 or more consecutive idle ticks occurred, at the very start and the very end of the run (the initial ramp and the 500 ms drain linger), totalling 0.60–0.63 s. The barrier is invisible as idle time because the core has already buffered the whole batch and keeps publishing from its backlog while the previous batch's tail completes. The barrier is very visible in the E2E distributions (§7.1) — it is what makes them multi-modal — but it is not a material contributor to the wall-clock overshoot.

### 8.5 Instrumentation cost

Tick timing takes seven `Instant::now()` readings per poll-draining tick and five otherwise (§2.1); at the observed cadence that is ~3 400 readings per second, and the residual between `total` and the sum of the five steps averages 58–80 ns per tick, i.e. the cost of about one reading. Dispatch latency takes two readings and a handful of relaxed atomic increments per request, with no lock on the request path, as §3.2 requires; per-job progress uses a `DashMap` with atomic fields; client-side samples stay in per-coroutine owned `Vec`s and are merged only after the run. The `fill` step's cost of 235–522 ns per assignment is 2–3 orders of magnitude above the per-request instrumentation cost, so the measurement does not meaningfully perturb what it measures. What instrumentation *does* distort is the server-side tail, by design (§2.2).

### 8.6 What these numbers do not establish

- **They do not establish a network dispatch cost.** Everything ran on one host over loopback, from one worker process. `client − server` here is contaminated by client-side channel queueing (§6.3), so it is not yet the loopback baseline the contract wants a multi-node run compared against.
- **They do not establish the scheduler's throughput ceiling.** No case saturated the core; the highest duty cycle observed was 1.12% of one core. The scheduler's maximum sustainable assignment rate is above everything measured here, and by how much is unknown, because the harness could not offer more load — it topped out at ~32k tasks/s inbound (§8.2) and at a worker cycle bounded by the channel pool (§8.1).
- **They do not establish behaviour under contention or failure.** Every case ran on an idle 32-core host with 1–8 groups whose loads were identical by construction. Nothing here measures interference between groups of unequal weight or demand, worker or scheduler failure, storage-session bumps, or task rescheduling — the reschedule queue was never exercised, and there were zero empty responses, zero warnings and zero errors in all ten process logs.
- **They do not establish the effect of the admission policy's parameters.** `B` was derived per case from `B = R × (N + 1)` and α was fixed at 1; no case varied either, so the policy's sensitivity to them is unmeasured. Relatedly, the fitted `fill` marginal costs in §5.1 vary with case in ways confounded by `B`, by assignments per tick and by the pinned/shared mix, and should not be read as a clean per-group scaling law.
- **They do not establish steady-state behaviour over long runs.** The longest case ran 74 s across four batches. There is no evidence of drift within that window — batch-over-batch timings repeat closely (§7.1) and case D's throughput held at 31.4k, 31.4k and 31.9k tasks/s across successive 5 s log intervals — but hours-long behaviour, memory growth and registry churn are untested.

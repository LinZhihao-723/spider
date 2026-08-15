# Scaling Behaviour of the Resource-Group-Aware Scheduler Core

This document records what the performance evaluation established about how the prototype scales, and — as importantly — what it did not establish. It is derived from two benchmark runs whose reports are in [`perf/run-001/report.md`](perf/run-001/report.md) and [`perf/run-002/report.md`](perf/run-002/report.md). The methodology is [benchmark-contract.md](benchmark-contract.md); the design under test is [design.md](design.md).

Run 002 supersedes run 001 for all latency figures. Run 001's client-side series was distorted by a shared gRPC channel pool and its server-side series pooled long-poll waits into dispatch cost; both are corrected in run 002. Run 001's tick timings and correctness results remain valid.

## 1. Summary

The scheduler core scales cleanly across every dimension tested — 1 to 8 resource groups, 40 to 320 execution managers, 131 072 to 1 048 576 tasks per case. Server-side dispatch cost stays sub-microsecond and does not degrade; per-tick cost grows sub-linearly in resource groups; per-assignment cost *improves* with scale.

The core was never the bottleneck in any case. Its duty cycle stayed between 0.48% and 1.59% of a single core, which is also why no saturation point was found: the evaluation establishes an absence of degradation over the range tested, not a ceiling.

One genuine limit was found, and it is a configuration interaction rather than a property of the design: at 320 execution managers the dispatch buffer, sized by the reserve rule to exactly one worker-round of slack, empties between ticks and 78% of requests wait.

## 2. What was measured

Five configurations, each 128 jobs per resource group of 1024 tasks, 5 ms simulated execution, jobs released in 4 batches of 32, `active_job_list_capacity` 16, `tick_interval_ms` 1, release build, 32-core host.

| Case | Resource groups | Tasks | Dedicated EMs | Shared EMs | Total EMs | `B` |
|---|---|---|---|---|---|---|
| A | 1 | 131 072 | 32 | 8 | 40 | 80 |
| B | 2 | 262 144 | 64 | 16 | 80 | 120 |
| C | 4 | 524 288 | 128 | 32 | 160 | 200 |
| D | 8 | 1 048 576 | 256 | 64 | 320 | 360 |
| E | 8 | 1 048 576 | 0 | 128 | 128 | 144 |

`B = R × (N + 1)` where `R = shared/N + dedicated_per_group`, derived from the admission policy's equilibrium free space `F = B/(N+1)` at α = 1 ([design.md](design.md) §6.2).

## 3. Tick cost

Mean nanoseconds per step across all ticks of the run.

| Case | RGs | collect | process | apply | fill | retire | **total** | ns/assignment | duty cycle |
|---|---|---|---|---|---|---|---|---|---|
| A | 1 | 973 | 1 436 | 645 | 6 789 | 83 | **10 012** | 791 | 0.48% |
| B | 2 | 789 | 2 265 | 528 | 9 127 | 74 | **12 845** | 505 | 0.62% |
| C | 4 | 838 | 4 553 | 527 | 15 359 | 84 | **21 417** | 428 | 1.03% |
| D | 8 | 956 | 7 367 | 787 | 23 618 | 401 | **33 191** | 493 | 1.59% |
| E | 8 | 767 | 4 820 | 534 | 19 294 | 73 | **25 549** | 620 | 1.22% |

**Step 4 (`fill`) dominates and step 5 (`retire`) is negligible**, exactly as the design predicts. `fill` is 68–75% of the tick; `retire` is under 1.2% in every case and never exceeds 401 ns.

**Per-tick cost grows sub-linearly in resource groups.** Eight times the resource groups (A → D) costs 3.3× the tick. The growth is concentrated in `fill` and `process`; `collect` and `apply` are essentially flat at 0.8–1.0 µs and 0.5–0.8 µs regardless of scale.

**Per-assignment cost improves with scale**, 791 → 428 ns from A to C. A fatter tick amortizes the fixed per-tick work — the pass over `active_rg_list`, the arm seeding, the promotion pass — over more decisions. This is the scale-free figure, and it is the one to quote: the per-tick figure is inflated by the cadence effect in §6.

**`process` is the step that scales with arrival volume**, 1.4 → 7.4 µs from A to D, and it carries the heaviest tail. It is skipped entirely on ticks where an inbound poll is still in flight, so its cost concentrates into the ticks that drain a 256-task wave.

## 4. Dispatch cost

Server-side, immediate class only — requests where an assignment was available and the handler never awaited on an empty channel. This is the quantity that answers "is the server side under control irrespective of how long a worker waited for work to exist".

| Case | Dedicated p50 / mean / p99 | Shared p50 / mean / p99 |
|---|---|---|
| A | 0.503 / 0.668 / 1.807 µs | 0.711 / 0.831 / 2.175 µs |
| B | 0.327 / 0.483 / 1.247 µs | 0.483 / 0.592 / 1.455 µs |
| C | 0.331 / 0.413 / 1.135 µs | 0.427 / 0.510 / 1.327 µs |
| D | 0.351 / 0.438 / 1.167 µs | 0.467 / 0.534 / 1.439 µs |
| E | — | 0.327 / 0.411 / 1.231 µs |

**Sub-microsecond, and flat across scale.** Case E — 8 resource groups, 1 048 192 requests, every one served through the hint channel — ties case B for the fastest median. Case A, the smallest configuration, is the *slowest*. There is no degradation trend in resource groups or in execution manager count.

The mean sits below p99 in every series, which is the signature of the immediate/waited split working. Pooling waits into the same series, as run 001 did, put the mean an order of magnitude above p99 — case C's server-side pinned mean was 21.44 µs against a p99 of 1.79 µs, an artifact produced by 0.04–0.16% of samples.

**The shared path costs 29–48% more than the dedicated path** — 96 to 208 ns in absolute terms, narrowing to 17–23% at p99. This is the hint-channel traversal of [design.md](design.md) §7.2, and it is the first clean measurement of what the general-execution-manager path costs relative to the pinned one. It is a small constant, not a scaling term.

## 5. The one limit found: supply starvation at 320 execution managers

| Case | Immediate | Waited | % waited |
|---|---|---|---|
| A | 130 952 | 120 | 0.1% |
| B | 261 871 | 273 | 0.1% |
| C | 523 504 | 784 | 0.1% |
| **D** | **227 699** | **820 877** | **78.3%** |
| E | 1 048 192 | 384 | 0.0% |

Case D is the only configuration in which execution managers routinely found nothing to take. The mechanism is an interaction between two settings, not a defect in the design:

- The reserve rule sizes `B` so that equilibrium free space equals `R`, which also puts each group's equilibrium occupancy at `R` — about 40 buffered assignments against about 40 execution managers per group. That is exactly one worker-round of slack, with no margin.
- The tick loop ran at a measured **2.07 ms**, not the configured 1 ms (§6).

At 320 execution managers consuming a 5 ms task each, a group's 40 buffered assignments are drained well within one tick period, and everyone blocks until the next tick's burst. D publishes 252 assignments per busy tick but only 4 152 of its 15 561 ticks are busy, and the waited p50 of 2.33 ms is one tick period.

The reserve rule therefore holds to 160 execution managers and runs out at 320. Two independent remedies: shorten the tick interval, or size `B` for more than one worker-round when the execution manager count is high.

**The design degraded gracefully under this pressure.** It made execution managers wait; it did not drop, duplicate, or misroute work. Every job in every case completed exactly 1024 tasks, with client-side and server-side counts agreeing exactly.

## 6. Threats to validity

**The scheduler was never saturated.** At a duty cycle of 0.48–1.59% of one core, this evaluation establishes that no degradation appears over the range tested. It does not locate a ceiling, and any headroom estimate from this data is extrapolation.

**The tick loop ran at roughly half its configured rate** — a measured 2.07 ms cadence against `tick_interval_ms = 1`, consistently in all five cases and independent of load. The core is ~1% busy, so this is not CPU starvation; the likely cause is timer granularity under WSL2 or coalescing in the runtime. Two consequences: the core was never asked to run at the rate we specified, and per-tick figures carry roughly twice the work a true 1 ms cadence would give them. This is why §3 quotes per-assignment cost as the scale-free number, and it is a direct contributor to §5.

**The bottleneck in these runs was the load generator, not the network and not the scheduler.** Client-side latency has a floor of 41–54 µs that is flat across an 8× range of execution manager counts — that floor is the loopback transport cost. The client-side *median* is 348–803 µs and rises with coroutine count, so 87–93% of client-side time is queueing inside the single worker process, not on the wire.

| Case | EMs | Client min | Client p50 | Floor as % of p50 |
|---|---|---|---|---|
| A | 40 | 47.5 µs | 348 µs | 13.6% |
| B | 80 | 46.8 µs | 352 µs | 13.3% |
| E | 128 | 43.8 µs | 410 µs | 10.7% |
| C | 160 | 41.2 µs | 446 µs | 9.2% |
| D | 320 | 54.2 µs | 803 µs | 6.8% |

Ranked, what limits throughput today: worker-process CPU contention first, the buffer-sizing-versus-cadence interaction of §5 second, and the actual network a distant third at ~45 µs.

**Single-node only.** Everything here is loopback. The harness is built for multi-node — separate `bench-scheduler` and `bench-workers` binaries with worker sharding — but no cross-host run has been made.

**Simulated execution.** Tasks are a 5 ms sleep. No real work, no data movement, no storage interaction.

## 7. Implications for the multi-node runs

**Shard execution managers across processes and hosts.** One process hosting 320 coroutines is the dominant cost in the current numbers and would be indistinguishable from network latency in a cross-host result — the same trap run 001 fell into with channel pooling, one level down.

**Use ~45 µs as the loopback baseline**, not the median. The median is a property of our load generator; the floor is the property of the transport.

**Expect case D to remain supply-bound** until either the tick interval or the buffer sizing is revisited. Comparing a multi-node case D against its single-node counterpart will otherwise measure §5's starvation rather than the network.

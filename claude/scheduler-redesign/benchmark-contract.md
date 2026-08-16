# Prototype Benchmark Contract

The shared contract for the performance evaluation of `spider-scheduler-new`. [design.md](design.md) is
normative for behaviour and [implementation-stage-report.md](implementation-stage-report.md) records what
correctness testing established. This document is normative for the benchmark: what is measured, how, at
what configuration, and what the output looks like.

## 1. What is measured

### 1.1 Tick step timing

The core's tick has five steps ([design.md](design.md) §5). Every tick is timed, per step and in total:

| Key | Step | Covers |
|---|---|---|
| `collect` | 1 | Draining the inbound poll result and the reschedule queue, and starting the next poll |
| `process` | 2 | Deduplicating and grouping the polled entries into `rg_updates` |
| `apply` | 3 | Creating scheduling units and placing new jobs |
| `fill` | 4 | The round-robin admission loop — the scheduling policy itself |
| `retire` | 5 | Removing retired jobs from the job registry |
| `total` | — | The whole tick, which may exceed the sum by a small bookkeeping remainder |

Each tick also records the number of assignments published and the number of active resource groups, so
per-step cost can be read per assignment rather than only per tick.

The question this answers is how each step scales as resource groups and jobs grow. Step 4 is expected to
dominate and to scale with assignments published; steps 2 and 3 scale with inbound arrivals; step 5 is
expected to be near zero.

### 1.2 Dispatch latency

Every request falls into exactly one of three classes, and the class determines whether it is timed:

| Class | Meaning | Treatment |
|---|---|---|
| **Immediate** | An assignment was available when the handler was entered; it returned without awaiting | **Timed.** This is dispatch cost |
| **Waited** | The handler blocked until an assignment arrived, then returned it | **Timed separately.** Dominated by supply, not by the scheduler |
| **Empty** | No assignment arrived before the wait expired | Counted only, never timed |

For immediate and waited requests, two durations:

- **Client-side** — measured in the worker around the gRPC call, from immediately before the request to
  immediately after the response. Includes transport, serialization, and server time.
- **Server-side** — measured in the service handler around the `DispatchService::next_task_*` call.
  Excludes transport and serialization.

`client − server`, over the **immediate** class only, is the transport and framing overhead. On a single
node it is loopback cost; across nodes it becomes the network cost, which is why both are recorded.

Both are reported **separately for dedicated (pinned) and shared (general)** execution managers, because
they take structurally different paths: a pinned request touches one group's queue and nothing else, while
a general request pops the hint channel and may traverse several stale hints.

**The immediate/waited split is not optional, and pooling the two invalidates the server-side figure.**
Run 001 excluded only the empty class, which left long-poll waits billed as dispatch cost: case C's
server-side pinned mean came out at 21.4 µs against a p99 of 1.79 µs and a max of 46.9 ms. The mean was
measuring how long a worker waited for work to exist, not how long the scheduler took to hand it over. The
question the server-side series exists to answer — is the server side under control irrespective of
network conditions — can only be read off the immediate class.

**Client-side measurement requires one gRPC channel per worker coroutine.** Sharing a channel across
coroutines introduces HTTP/2 head-of-line queueing that the measurement cannot distinguish from transport
cost. Run 001 used a pool of 16 and measured client-side p50 climbing 328 µs → 1.44 ms → 2.72 ms purely as
a function of workers-per-channel (2.5 → 5 → 10), while server-side stayed flat at ~0.5 µs throughout. The
default must therefore be one channel per worker; any pooling must be an explicit opt-in and recorded in
the results. This matters most for the planned multi-node runs, where the client-side series is the
measurement of interest and an artifact of this size would swamp the real network cost.

### 1.3 Per-job end-to-end time

- **First seen** — the timestamp at which the fake inbound queue first emits a task belonging to a job.
- **Last completed** — the timestamp at which the dispatch service last received a completion report for a
  task of that job, taken from the `completed` field of the following `NextTask` request.
- **E2E** = last completed − first seen.

The completion count per job doubles as a correctness check: it must equal `tasks_per_job` for every job,
or the run is invalid regardless of its timings.

**Known skew, to be stated in the report:** a task's completion is reported on the worker's *next* request,
so each job's E2E includes one extra round trip beyond its final task's execution. The run therefore must
not end when the last task is dispatched — it ends when the last completion is *received*.

## 2. Configurations

Constant across every case:

| Parameter | Value |
|---|---|
| Jobs per resource group | 128, released in **4 batches of 32** |
| Tasks per job | 1024 |
| Task execution time | 5 ms (simulated by `tokio::time::sleep` in the worker) |
| Inbound wave size | 256 tasks per poll response |
| `active_job_list_capacity` | 16 |
| `tick_interval_ms` | 1 |
| `storage_poll_timeout_ms` | 5 |
| `ready_task_capacity` | the case's total task count — the core buffers everything, so no task waits in the inbound queue |
| Dedicated workers per group | 32 (except case E) |
| Shared workers | `8 × #RG` (except case E) |

**Batch release rule.** Batch `n+1` is released once every job in batch `n` has had all of its tasks
reported complete. This is an interpretation of "jobs are created in batches" and it is what makes the
batching observable: with 32 jobs per batch against an active list of 16, there is always a pending queue.
It is recorded here because it materially shapes the results.

### The five cases

| Case | #RG | Total tasks | Shared | Dedicated | Total workers | Reserve `R` | `B` |
|---|---|---|---|---|---|---|---|
| A | 1 | 131 072 | 8 | 32 | 40 | 40 | 80 |
| B | 2 | 262 144 | 16 | 64 | 80 | 40 | 120 |
| C | 4 | 524 288 | 32 | 128 | 160 | 40 | 200 |
| D | 8 | 1 048 576 | 64 | 256 | 320 | 40 | 360 |
| E | 8 | 1 048 576 | 128 | 0 | 128 | 16 | 144 |

### Deriving `B`, the dispatch queue capacity

The requirement is that at least `R` slots stay free when every resource group is active, where

```
R = (shared workers / #RG) + (dedicated workers per group)
```

Under the admission policy ([design.md](design.md) §6.2) with α = 1 and `N` backlogged groups, free space
settles at `F = B / (N + 1)`. Requiring `F ≥ R` gives

```
B = R × (N + 1)
```

which is the `B` column above. Case E has no dedicated workers, so `R = 128/8 = 16` and `B = 16 × 9 = 144`.

### Expected steady-state duration

`total_tasks × 5 ms / total_workers`, i.e. **≈16.4 s** for A–D and **≈41 s** for E. A case whose measured
duration greatly exceeds this is worker-starved, and the report must say so rather than presenting the
timings as scheduler cost.

### Known limitation of the reserve rule, established by run 002

`B = R × (N + 1)` puts equilibrium free space at `R`, which also puts each group's equilibrium occupancy at
`R` — roughly one execution-manager round of slack, with no margin. Run 002 found this holds to 160
execution managers and runs out at 320: case D drained its buffers between ticks and **78% of its requests
waited**, against ≤0.1% in every other case. The measured tick cadence of ~2.07 ms against a configured
1 ms is the other half of the interaction.

A future run that reuses this rule at high execution-manager counts will measure that starvation rather
than the scheduler. Either shorten the tick interval or size `B` for more than one round. See
[scaling-behavior.md](scaling-behavior.md) §5.

## 3. Harness changes required

### 3.1 Multi-node capability

Two binaries, so the scheduler and the workers can run on different machines. Single-node runs use the same
two binaries on one host; nothing about the topology is special-cased.

- **`bench-scheduler`** — runs the core, the fake inbound queue, and the gRPC dispatch service. Binds a
  configurable address. On drain, writes its results JSON and exits.
- **`bench-workers`** — connects to a scheduler endpoint, runs the configured pinned and general workers as
  coroutines, and writes its results JSON on completion. One process may host any number of workers; do not
  spawn one process per worker.

Coordination is via a `RunStatus` RPC on the bench service: workers stop once it reports the run drained.
Workers must keep polling until then, which is what flushes the final completion reports (§1.3).

Each side writes its own JSON, so a multi-node run produces one scheduler file and *n* worker files.

### 3.2 Instrumentation cost must not distort the measurement

- Tick timing: six `Instant::now()` calls per tick at 1 ms cadence — negligible.
- Server-side dispatch timing: two `Instant::now()` per request. At case D's ~64 000 requests/s this must
  not take a lock. Record into a **fixed-bucket atomic histogram** (`fetch_add` on an `AtomicU64` bucket),
  never a `Mutex<Vec<_>>`.
- Per-job completion tracking runs on the same hot path. Use a `DashMap<JobId, JobProgress>` whose fields
  are atomics, so ≤1024 jobs shard cleanly and no global lock appears at 64 000 updates/s.
- Client-side latency stays per-worker-coroutine in an owned `Vec`, merged only after the run, exactly as
  the correctness harness already does.
- Worker gRPC channels are pooled and shared across coroutines rather than one per coroutine. HTTP/2
  multiplexes, but a single channel for 320 coroutines would introduce head-of-line queueing that the
  measurement would wrongly attribute to the scheduler. Make the pool size configurable and record it in
  the results.

## 4. Output

### 4.1 Raw results are the deliverable, not just the report

Runs are sequenced. Each lives in its own directory under `claude/scheduler-redesign/perf/` and is
self-contained — its raw files and its report together, so a run can be read without reference to any
other:

| Run | Directory | What it measured |
|---|---|---|
| 001 | `perf/run-001/` | First pass. Tick timings and correctness valid. Client-side series invalidated by a shared channel pool; server-side series inflated by pooled long-poll waits |
| 002 | `perf/run-002/` | One channel per worker, and the immediate/waited/empty split of §1.2 |

Within a run directory:

- `<case>-scheduler.json`
- `<case>-workers-<index>.json`

Storage policy — raw where the volume allows, histograms where it does not:

| Series | Volume at case D | Stored as |
|---|---|---|
| Per-tick step timings | ~16 000 ticks × 6 | **every sample**, raw |
| Per-job E2E and completion counts | ≤1024 jobs | **every sample**, raw |
| Dispatch latency (4 series) | ~1 000 000 samples each | fixed-bucket histogram, plus count/mean/min/max and p50/p90/p99/p99.9 |

Histogram buckets must be fine enough to reconstruct any percentile the report might later want: use
sub-microsecond resolution at the low end. Every JSON also carries the full configuration it was produced
from, the binary's build profile, and the host's core count, so a result file is self-describing.

### 4.2 Report

`claude/scheduler-redesign/perf/report.md`, containing:

- The terminology of §1 — what each measured quantity means and, for dispatch latency, why client and
  server are both present.
- The configuration table of §2, including the `B` derivation and the batch release rule.
- Tick step timings per case, with the scaling trend across #RG called out.
- Dispatch latency per case, dedicated and shared reported separately, client and server side by side.
- Per-job E2E distributions.
- The correctness check: every job completed exactly `tasks_per_job` tasks, on every case.
- Any case that failed to reach its expected steady-state duration, and why.

## 5. Execution rules

- **Release build.** `cargo build --release`. A debug build measures the compiler, not the design.
- **One configuration at a time.** Cases must never overlap; the machine must be otherwise idle. This is
  why the run phase is strictly sequential.
- Record the wall-clock duration of each case and compare it against §2's expectation.
- Before the five cases, run one **smoke case** at reduced dimensions to prove the pipeline end to end.
  Its results are not part of the report.

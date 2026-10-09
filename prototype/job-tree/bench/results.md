# Job-tree prototype: benchmark results (B1-B7)

| | |
|---|---|
| Author | benchmark engineer (benchmark engineer) |
| Date | 2026-10-09 |
| Version | 2: the consolidated results after the methodology review `review.md` (methodology reviewer). Its required fixes R1-R9 are applied, and B1, B2, B3 and B5 were rerun (§11) |
| Inputs | `../brief.md` (B1-B7), `../plan.md` §5-6 (the EM harness, boxing), `../../04-bench/review.md` (methodology), `review.md` (this document's review) |
| Harness | `claude/benchmarks/job-tree-proto/` (`README.md` has every run's command; `run-followups.sh` and `run-rerun.sh` replay them) |
| Raw data | under `claude/benchmarks/job-tree-proto/results/`, one directory per run (§1); `tables.md` in a run directory is the output of `analyze.py`, and `b1-combined/tables.md` merges the first B1 run's two halves |

**Summary.**
- **Plain jobs:** no regression was detected on the job transitions that now go through the size-1 committer. The last report's p50 is 1.02-1.03x main in two runs of 20 trials, inside the A/A bands. The intervals do not exclude up to +12 % at C = 1 or +9 % at C = 64. `RegisterJob` is 23-38 % slower: an explicit BEGIN, the fence round trip, a COMMIT statement and the committer's creation; it makes the same 3 fsyncs per job as main. The job lifecycle is 4-5 % slower at C = 1 (at the edge of the A/A band) and 11 % slower at C = 64 (outside it). The closed-loop job throughput is 5-11 % lower. At C = 64 MariaDB spends 1.12x and the storage process 1.22x the CPU per job (B1). Gate H12 holds on its point estimates; H11's throughput half is undecided.
- **Memory:** 14.0 KB more per live plain job than main: 1.49 GiB instead of 190 MiB at 100,000 jobs, an RSS delta and so an upper bound. 10.4 KB of it is the committer's pre-sized arrival buffer, a one-line fix. Without it the cost is 3.6 KB, still about 12x the design's "a few hundred bytes". Boxing the round future saves 735 B per tree (B6).
- **Trees:** a depth-7 binary tree (127 jobs) takes 7.0-7.1x the time of the same shape inside one job, 28 ms per job-nesting hop, while a hop with a single child costs 2 ms. Its 63 sibling registrations are 63 consecutive DB transactions (one registering parent per tree state transaction) (B2). Its absolute latency depends on a host phase: 134-207 ms per run, and up to about 400 ms (B2, `b2gap`). 64 concurrent trees are fair to each other. Per-tree latency grows 5.6x at K = 64 whatever the pool size or the EM's worker count. About 5,000 COMMITs/s was achieved, and the bound is not identified (B5).
- **Fan-in, abort:** 1,000 siblings finish at 21,700/s with 200 events per DB transaction, 2x independent plain jobs (B4). A whole-tree abort is one DB transaction, acknowledged in 3.5-8.1 ms. An abort that races the expansion waits for at most the one registration in flight: 5.1-5.6 ms with about 30 registrations queued (B3).
- **Recovery:** 1.4-1.5x main per plain job. Trees add about 0.4 ms per `BlockingOnChildren` parent, attributed by elimination to a sequential per-parent repair query (B7).
- **Correctness:** every run passed every correctness check.
- **Host caveat:** this host has slow phases that the idle gate cannot see. One is a periodic disk-flush slow mode, which doubles B1's C = 1 latency for about 50 s every 400 s. Others slow the depth-7 tree 1.5-3x for minutes. Ratios paired within a process or a trial are robust; absolute latencies are not.

## 1. Setup

**Host.** 16 vCPUs, 23 GiB, WSL2 (Linux 6.18.33.2). The kind cluster `clp-rc2-control-plane` stayed stopped (`Exited (137)`) through every run. The idle gate passed before every storage process: no `cargo`/`rustc`/linker process, and fewer than 0.75 busy cores outside the harness, the storage process and the MariaDB container, over three consecutive 1-s probes. The gate runs only at trial boundaries and cannot see I/O stalls; nothing watched the host during a measured interval in the first five runs. The reruns record CPU, iowait and disk counters over every measured interval (below).

**Database.** `mariadb:10.11.16` (the deployment pin) in Docker with default server settings, a fresh container per benchmark group and a fresh database per storage process. The reruns record the settings in `env.txt`: `innodb_flush_log_at_trx_commit = 1` (every commit fsyncs the redo log), `innodb_flush_method = O_DIRECT`, `innodb_doublewrite = ON`, `log_bin = OFF`, a 128 MiB buffer pool and a 96 MiB redo log. The storage process and the harness reach it through the container's bridge IP, with no docker-proxy hop. `tools/scripts/mariadb/start.py` hard-codes `mariadb:latest`, so `db-up.sh` replicates its `docker run` with the pinned image; the container is always stopped with the worktree's `tools/scripts/mariadb/stop.py`. In the reruns each container is first warmed up by one discarded storage process (300 plain jobs at C = 1, 64 × 30 at C = 64, and one depth-7 tree).

**Counting DB transactions.** MariaDB's `Com_commit` counts COMMIT statements only. An autocommit statement is a durable transaction (it fsyncs under `innodb_flush_log_at_trx_commit = 1`) but does not increment `Com_commit`. This matters for `main`, whose `RegisterJob` is one autocommit `INSERT` (`components/spider-storage/src/db/mariadb.rs:108-139` on `main`): on `main`, `Com_commit` undercounts DB transactions by one per job. On the prototype every runtime `jobs` write runs in an explicit transaction, so there `Com_commit` equals its DB transactions, and the per-tree counts and events per DB transaction of B2-B5 stand. Where `main` appears, this document says "COMMIT statements". The reruns also record `Innodb_data_fsyncs` (in 10.11 it counts every InnoDB fsync, the redo log's included), `Handler_commit`, and the block device's flush requests and their service time (`/proc/diskstats` of `sdd`, the disk under `/var/lib/docker`).

**Binaries under test** (release builds, copied into `bin/` with provenance; rebuilt before the reruns, with identical hashes: `670e995f…` main, `ae449d78…` proto):

| Name | Source |
|---|---|
| `main` | `/home/lzh/dev/spider` at `278d06f` (no tracked changes) |
| `proto` | the worktree at `278d06f` + the prototype diff (`components/` diff hash `d4cda1b5a3428faa`); the per-round future is `Box::pin`ned |
| `proto-unboxed` | the same with `RUSTFLAGS="--cfg spider_unboxed_round"`, the prototype's switch for an inlined per-round future (B6 only) |
| `proto-smallbuf` | diagnostic, not the prototype: a copy of the worktree whose committer starts its per-round arrival buffer empty (`Vec::new()`) instead of `Vec::with_capacity(MAX_EVENTS_PER_ROUND)` (B6 only) |

**Runs.** All on 2026-10-09, in this order. The harness changed between runs; within a run every comparison uses one harness build.

| Run | Harness build | EM reports | Used for |
|---|---|---|---|
| `full-20261009` | `8283d6fb` | awaited by the worker | B1 trials 0-9, B3 (first run), B4, B6, B7; B2 and B5 sensitivity |
| `b4diag-20261009` | `32287980` | awaited | the B4 registration diagnostic |
| `em-async-20261009` | `32287980` | spawned | B2 and B5 |
| `b1-extra-20261009` | `32287980` | (B1 has no EM workers) | B1 trials 10-19 |
| `b7-fanout-20261009` | `32287980` | (drain only) | where the trees' recovery time goes |
| `b3-rerun-20261009` | `42bbf085` | spawned | B3: the abort's DB transactions, the race variants |
| `b2gap-20261009` | `42bbf085` | spawned | the B2/B5 gap of the depth-7 tree |
| `b5w-20261009` | `42bbf085` | spawned | B5 with 64, 128 and 256 EM workers, with CPU and disk accounting |
| `b1-williams-20261009` | `42bbf085` | (B1 has no EM workers) | B1 replication: counterbalanced order, per-phase counters |
| `b2-rerun-20261009` | `3582da77` | spawned | B2 replication |
| `b2-protoonly-20261009` | `3582da77` | spawned | B2 with only the proto process per trial |
| `b2-treeonly-20261009`, `b2-trees-20261009` | `29799e53` | spawned | B2 with only its tree workloads (bisection of the B2/B5 gap) |
| `b2-treemeter-20261009` | `2e42903c` | spawned | the tree-only B2 with disk and CPU accounting per tree |
| `b2gap-again-20261009` | `2e42903c` | spawned | the `b2gap` control, right after the B2 reruns |

Build `42bbf085` adds the new experiments and the interval accounting. It leaves the timed RPCs of B1, B2 and B5 unchanged. It adds one `SHOW GLOBAL STATUS` before and after each B1 phase and each B5 interval, outside the timed RPCs, and one `Com_commit` read between B3's expansion and its abort. Builds `3582da77`, `29799e53` and `2e42903c` add only B2 knobs (`JT_B2_VARIANTS`, `JT_B2_WORKLOADS`, `JT_B2_METER`); with the knobs unset, B2 runs as in `42bbf085`. The commands of every run are in the harness's `README.md`, `run-followups.sh` and `run-rerun.sh`.

**Storage configuration**, identical for every binary within an experiment: `max_connections = 64` (B5 also 32), `execution_manager_stale_cutoff_sec = 3600`, `RUST_LOG=warn`, the default inbound-queue capacities for B1-B5 and `task_capacity = 1,048,576` (the deployed value) for B6 and B7.

**Harness.** It plays the scheduler and the EM over gRPC (plan §5): one poller per lane (`max_items = 256`, `wait_ms = 1`), a router, and W workers (64 unless stated) that register task instances and report success (with pre-compressed children for tree nodes), failure, or hold the instance until the experiment releases it. "Spawned" reports leave the worker free as soon as its task body ends, as the real EM does (`spider-execution-manager/src/runtime.rs:421-433`); "awaited" reports keep the worker until the reply returns, as the plan's harness did. It uses the generated tonic clients over 16 connections, without `spider-client`'s retry wrapper. Against `main` it sends only plain-job requests (the `children` field stays empty and is encoded as nothing). Every timestamp is taken in the harness; a report's latency is its RPC round trip.

**Methodology** (`04-bench/review.md`): at least 10 trials per configuration (B1: 20 in each of its two runs; b2gap: 20; the two diagnostics: 3 and 5); `median [95 % CI]` across trials, with the distribution-free order-statistic interval of the median ([x(2), x(9)] at n = 10, [x(6), x(15)] at n = 20); ratios paired by trial and judged against an A/A control where one exists (B1: `main-aa`, `proto-aa`; B2: `main-aa`; B4: `proto-aa`). A ratio is "outside the noise" only when its CI lies outside the A/A band (the A/A CI mirrored around 1). This rule detects differences; it cannot establish that two things are equal, so "no difference detected" is reported with the CI's bounds. Tails are labelled p99 only when every trial has at least 200 samples. Each storage process first runs one unrecorded warm-up iteration (B2-B5). Every trial ends with correctness checks: the DB state of every job the trial created, `GetJobState` of the roots, the expected number of resent entries (B6, B7), and exit code 0 after SIGTERM. No check failed in any run, and every storage process exited 0 after SIGTERM.

## 2. B1: plain-job transition latency

Each writer loops over jobs: `RegisterJob`, `StartJob`, wait for its own routed task entry, `RegisterTaskInstance`, then a timed `ReportTaskSuccess` of the job's only task. That report is the job's last task, so it is the job transition (`Running → Succeeded`), and its reply follows the apply: end to end (`RegisterJob` sent → that reply) is submit-to-`Succeeded`. `StartJob` (`Ready → Running`) is a second job transition per job. C = 1: 2,000 measured jobs after 200 warm-up jobs per trial; C = 64: 64 writers × 300 measured jobs after 30 warm-up jobs each. Each trial runs `main`, `proto`, `main-aa` and `proto-aa` as four storage processes. Every job ended `Succeeded` (checked in the DB per process). Latencies in ms; every p99 is over at least 2,000 samples per trial.

Two runs of 20 trials each:

- **First run** (trials 0-9 of `full-20261009` and 10-19 of `b1-extra-20261009`, merged in `b1-combined/tables.md`). The order alternated between main, proto, main-aa, proto-aa and proto, main, proto-aa, main-aa, so the A/A processes always ran third and fourth. The 10 extra trials were added after looking at the first ten, whose C = 64 A/A band was too wide to decide anything. The two halves come from different harness builds (`8283d6fb`, `32287980`). The change between them is the EM's report mode, which B1 does not use: it has no EM workers.
- **Replication** (`b1-williams-20261009`, one harness build `42bbf085`, a warmed-up container). The process order of trial t is row t % 4 of a 4 × 4 Williams square, so every variant takes every position, and follows every other variant, five times. Per phase it also records the server's `Com_commit`, `Com_select`, `Innodb_data_fsyncs` and `Handler_commit` deltas, the CPU of each component, and the disk's flush requests with their service time.

**First run, C = 1**

| Variant | Last report p50 | Last report p99 | End to end p50 | End to end p99 | `StartJob` p50 | `RegisterJob` p50 | Jobs/s |
|---|---|---|---|---|---|---|---|
| main | 1.81 [1.67, 1.95] | 3.90 [3.37, 4.22] | 6.95 [6.64, 7.19] | 11.97 [11.33, 13.46] | 1.73 [1.58, 1.86] | 1.41 [1.32, 1.50] | 140 [132, 147] |
| main-aa | 1.87 [1.78, 1.96] | 3.79 [3.45, 4.05] | 7.03 [6.76, 7.18] | 11.82 [11.13, 12.63] | 1.78 [1.67, 1.86] | 1.46 [1.36, 1.49] | 136 [133, 144] |
| proto | 1.93 [1.75, 2.09] | 4.29 [3.59, 4.63] | 7.35 [7.06, 7.90] | 13.24 [12.24, 14.78] | 1.78 [1.68, 1.93] | 1.81 [1.70, 1.98] | 128 [120, 135] |
| proto-aa | 1.95 [1.72, 2.11] | 4.24 [3.39, 5.73] | 7.38 [6.93, 7.92] | 13.40 [11.75, 17.94] | 1.81 [1.63, 1.94] | 1.84 [1.67, 1.97] | 128 [120, 139] |

**First run, C = 64**

| Variant | Last report p50 | Last report p99 | End to end p50 | End to end p99 | `StartJob` p50 | `RegisterJob` p50 | Jobs/s |
|---|---|---|---|---|---|---|---|
| main | 4.27 [4.03, 4.45] | 9.50 [8.86, 9.80] | 13.76 [13.00, 14.33] | 21.37 [20.40, 23.49] | 4.19 [3.97, 4.34] | 3.13 [3.00, 3.27] | 4,478 [4,182, 4,747] |
| main-aa | 4.23 [4.03, 4.40] | 9.15 [8.68, 9.98] | 13.67 [12.99, 14.23] | 21.25 [20.11, 23.98] | 4.16 [3.96, 4.35] | 3.13 [2.98, 3.23] | 4,506 [4,332, 4,752] |
| proto | 4.20 [4.01, 4.53] | 8.55 [8.20, 9.22] | 14.75 [14.12, 15.89] | 22.35 [21.57, 23.97] | 4.25 [4.06, 4.59] | 4.19 [4.02, 4.50] | 4,191 [3,894, 4,380] |
| proto-aa | 4.30 [4.14, 4.59] | 9.10 [8.40, 9.60] | 15.18 [14.56, 16.22] | 23.06 [21.72, 24.24] | 4.35 [4.16, 4.68] | 4.28 [4.15, 4.64] | 4,079 [3,822, 4,245] |

**Replication, C = 1**

| Variant | Last report p50 | Last report p99 | End to end p50 | End to end p99 | `StartJob` p50 | `RegisterJob` p50 | Jobs/s |
|---|---|---|---|---|---|---|---|
| main | 1.66 [1.61, 1.72] | 3.23 [3.16, 3.46] | 6.68 [6.60, 6.80] | 10.82 [10.64, 11.29] | 1.59 [1.54, 1.67] | 1.33 [1.30, 1.36] | 147 [143, 150] |
| main-aa | 1.67 [1.61, 1.71] | 3.23 [3.12, 3.35] | 6.69 [6.58, 6.74] | 10.79 [10.66, 10.95] | 1.59 [1.55, 1.63] | 1.34 [1.31, 1.35] | 146 [145, 149] |
| proto | 1.72 [1.64, 1.82] | 3.38 [3.28, 4.23] | 6.92 [6.80, 7.08] | 11.67 [11.37, 15.11] | 1.60 [1.56, 1.69] | 1.65 [1.59, 1.71] | 139 [125, 142] |
| proto-aa | 1.72 [1.67, 2.05] | 3.38 [3.21, 4.20] | 6.93 [6.81, 7.86] | 11.33 [11.08, 14.13] | 1.61 [1.58, 1.90] | 1.65 [1.61, 1.92] | 139 [120, 142] |

**Replication, C = 64**

| Variant | Last report p50 | Last report p99 | End to end p50 | End to end p99 | `StartJob` p50 | `RegisterJob` p50 | Jobs/s |
|---|---|---|---|---|---|---|---|
| main | 3.92 [3.82, 4.06] | 8.60 [8.26, 8.98] | 12.70 [12.30, 13.08] | 19.88 [18.99, 21.28] | 3.85 [3.74, 4.00] | 2.90 [2.80, 3.01] | 4,842 [4,699, 5,017] |
| main-aa | 3.90 [3.87, 4.01] | 8.62 [8.28, 12.46] | 12.65 [12.50, 12.92] | 19.81 [19.18, 24.09] | 3.85 [3.80, 3.92] | 2.90 [2.87, 2.95] | 4,863 [4,720, 4,908] |
| proto | 4.08 [4.00, 4.51] | 8.36 [8.16, 9.81] | 14.34 [13.99, 15.94] | 21.75 [21.19, 24.98] | 4.12 [4.04, 4.63] | 4.08 [3.99, 4.54] | 4,304 [3,861, 4,439] |
| proto-aa | 4.00 [3.89, 4.21] | 8.22 [7.87, 9.51] | 14.17 [13.67, 14.96] | 21.30 [20.50, 24.82] | 4.08 [3.94, 4.31] | 4.03 [3.87, 4.22] | 4,350 [4,144, 4,532] |

**proto / main, paired by trial, against the A/A ratio main' / main.** Bold: the ratio's CI lies outside the A/A band (the A/A CI mirrored around 1). The prototype's own A/A ratios (proto' / proto) are 0.94-1.03 at their medians in the first run and 0.96-1.02 in the replication.

| C = 1 | First run: proto / main | First run: A/A | Replication: proto / main | Replication: A/A |
|---|---|---|---|---|
| Last report p50 | 1.030 [0.976, 1.121] | 0.996 [0.974, 1.043] | 1.024 [0.981, 1.064] | 0.987 [0.954, 1.045] |
| Last report p99 | 1.074 [1.008, 1.155] | 0.988 [0.874, 1.082] | 1.019 [0.983, 1.068] | 0.995 [0.958, 1.030] |
| `StartJob` p50 | 1.009 [0.963, 1.097] | 1.004 [0.968, 1.056] | 1.013 [0.981, 1.039] | 0.994 [0.947, 1.039] |
| `RegisterJob` p50 | **1.245 [1.206, 1.335]** | 1.002 [0.971, 1.039] | **1.228 [1.190, 1.263]** | 0.997 [0.978, 1.041] |
| End to end p50 | 1.047 [1.025, 1.093] | 1.003 [0.983, 1.030] | 1.036 [1.022, 1.061] | 0.995 [0.980, 1.024] |
| End to end p99 | 1.107 [1.009, 1.139] | 0.985 [0.877, 1.029] | 1.070 [1.031, 1.107] | 1.001 [0.973, 1.034] |
| Jobs/s | 0.939 [0.897, 0.966] | 1.010 [0.961, 1.034] | **0.943 [0.930, 0.961]** | 1.005 [0.965, 1.031] |

| C = 64 | First run: proto / main | First run: A/A | Replication: proto / main | Replication: A/A |
|---|---|---|---|---|
| Last report p50 | 1.020 [0.965, 1.053] | 0.998 [0.964, 1.049] | 1.021 [0.997, 1.091] | 1.012 [0.936, 1.040] |
| Last report p99 | 0.946 [0.928, 0.985] | 0.999 [0.922, 1.051] | 0.973 [0.923, 1.034] | 1.031 [0.944, 1.086] |
| `StartJob` p50 | 1.052 [0.990, 1.078] | 1.003 [0.966, 1.047] | 1.056 [1.037, 1.113] | 1.010 [0.936, 1.034] |
| `RegisterJob` p50 | **1.383 [1.295, 1.412]** | 1.002 [0.962, 1.041] | **1.366 [1.346, 1.450]** | 1.010 [0.939, 1.041] |
| End to end p50 | **1.107 [1.048, 1.137]** | 0.996 [0.968, 1.046] | **1.111 [1.088, 1.173]** | 1.012 [0.935, 1.036] |
| End to end p99 | 1.065 [1.013, 1.118] | 1.007 [0.862, 1.032] | 1.105 [1.021, 1.221] | 1.029 [0.907, 1.157] |
| Jobs/s | 0.907 [0.883, 0.959] | 0.999 [0.962, 1.042] | 0.894 [0.852, 0.926] | 0.985 [0.962, 1.074] |

**Server counters and CPU per job** (replication; per phase, warm-up jobs included; medians over 20 trials, every CI within ±0.01 of the median for the counters):

| Phase | Variant | `Com_commit` / job | `Com_select` / job | `Innodb_data_fsyncs` / job | Disk flushes / job | Storage CPU / job | MariaDB CPU / job |
|---|---|---|---|---|---|---|---|
| C = 1 | main | 2.00 | 2.00 | 3.04 | 3.07 | 1.88 ms | 1.02 ms |
| C = 1 | proto | 3.00 | 3.00 | 3.05 | 3.07 | 2.12 ms | 1.14 ms |
| C = 64 | main | 2.00 | 2.00 | 0.21 | 0.21 | 0.73 ms | 0.91 ms |
| C = 64 | proto | 3.00 | 3.00 | 0.23 | 0.23 | 0.89 ms | 1.03 ms |

CPU per job, proto / main, paired: storage 1.119 [1.073, 1.181] at C = 1 and 1.220 [1.165, 1.275] at C = 64 (A/A 1.010 and 1.033); MariaDB 1.126 [1.097, 1.188] and 1.122 [1.094, 1.165] (A/A 1.019 and 1.030). At C = 1 the storage figure is dominated by the 1-ms polling of an idle lane, so the C = 64 figure is the one that describes work per job. At C = 64 the fsyncs per job drop to about 0.2 because MariaDB groups the commits of concurrent connections into one fsync.

- **Transitions through the size-1 committer: no regression detected, but a few percent are not excluded.** The last report's p50 (the `Running → Succeeded` round) is 1.030 and 1.024 of main's at C = 1 in the two runs, and 1.020 and 1.021 at C = 64. The point estimates are inside the A/A bands. The upper CI bounds allow up to +12 % and +6 % at C = 1 and +5 % and +9 % at C = 64. `StartJob`, the other committer round of every job, is 1.009 and 1.013 at C = 1 and 1.052 and 1.056 at C = 64. That is inside the bands, but about 5 % slower at C = 64 in both runs, with upper bounds of +8 % and +11 %. The p99 of the last report is inside the bands too.
- **The design's gates.** H12 reads "Fence cost: the plain-job p50 stays within the A1 A/A band at C = 1 and C = 64". Its point estimates are inside the band in both runs, but the CIs do not exclude +12 % (first run) or +6 % (replication) at C = 1, nor +5 % or +9 % at C = 64. H11 reads "Plain jobs through size-1 committers: the A1 p50 and the A3 throughput of today's path stay within the A/A band of today's measurements". Its p50 half is answered as for H12. **Its throughput half is undecided.**
  - The A3 analogue here, B4's 1,000 plain jobs released together, shows no difference, but inside A/A bands about ±25-35 % wide (§5).
  - B1's closed-loop job throughput includes `RegisterJob`. In the first run it is 0.939 [0.897, 0.966] and 0.907 [0.883, 0.959]: both CIs lie wholly below 1, and their upper bounds sit within 0.005 of the band edges.
  - In the replication it is 0.943 [0.930, 0.961] at C = 1, outside the band [0.965, 1.035], and 0.894 [0.852, 0.926] at C = 64, whose upper bound touches the band's edge (0.926).
  - So the job lifecycle's throughput is 5-11 % lower; B1 and B4 cannot tell whether the transitions' own throughput is.
- **`RegisterJob` is the RPC that got slower:** +25 % and +23 % at C = 1 (+0.40 and +0.32 ms), +38 % and +37 % at C = 64 (+1.06 and +1.18 ms), outside the A/A band in both runs. It is not an extra fsync. Both binaries make the same number of durable transactions per job: 3.04 and 3.05 `Innodb_data_fsyncs` per job at C = 1, and 3.07 disk flushes per job on both. `Com_commit` counts 2 against 3 only because main's `RegisterJob` is an autocommit `INSERT` (§1). The prototype's `RegisterJob` adds an explicit BEGIN, the fence `SELECT … LOCK IN SHARE MODE` round trip (`Com_select` 3 against 2 per job), the COMMIT statement, and the creation of the tree's channel and committer task. With the fence statement in every transaction, MariaDB spends about 12 % more CPU per job, and at C = 64 the storage process about 22 % more.
- **The job lifecycle** (end to end, submit to `Succeeded`) is +4.7 % (first run) and +3.6 % (replication) at C = 1. Both sit at the edge of the A/A band, so neither is established: [1.025, 1.093] against the band [0.970, 1.030], and [1.022, 1.061] against [0.976, 1.024]. At C = 64 it is +10.7 % and +11.1 %, outside the band in both runs. B6's load phase (100,000 `RegisterJob` + `StartJob` calls from 64 clients) is consistent with this: 1.17x [0.85, 1.28] main's time, a CI that includes 1.
- **Noise: a periodic slow mode of the disk.** The two runs show the same 2x slow mode; the replication identifies it.
  - **First run.** 13 of 80 processes ran at about 2x at C = 1: last report p50 2.85-3.88 ms, against 1.56-2.12 ms for the others. They were main 3.56/3.12 (t0/t1); proto 3.43/2.85 (t0/t1); main-aa 3.50/3.65/3.88/3.47 (t0/t11/t15/t19); proto-aa 3.79/3.88/3.70/3.58/3.78 (t0/t1/t11/t15/t19). `RegisterJob` slowed with it, from about 1.4-1.8 to 2.3-3.8 ms.
  - **It aliased onto the A/A processes.** In the extra half the slow mode came about every 400 s (every 4 trials) and lasted about 50 s. It fell exactly on the third and fourth processes of trials 11, 15 and 19, which are the two A/A processes. So the A/A CI of trials 10-19 alone is [0.991, 2.085] at C = 1, while proto / main is clean there. With n = 20 the order statistics exclude these outliers, so the merged conclusions stand. Per half, the last report p50 is 0.991 and 1.066, and `RegisterJob` 1.242 and 1.283.
  - **Replication.** 8 of 80 processes were slow (3.49-3.87 ms), again about every 4 trials (trials 2, 6, 10, 14 and 17). With the Williams order they spread over all variants and positions: main 2, main-aa 1, proto 2, proto-aa 3, and 1, 2, 2 and 3 at positions 0-3.
  - **It is the disk's flush latency.** In every slow process the disk's mean flush time was 1.23-1.33 ms, against 0.54 [0.535, 0.542] ms in the other 72. The device sits outside the guest, so the idle gate cannot see it.
- **The two runs agree on every ratio within their intervals**, although the replication's absolute latencies are lower (main at C = 1: 1.66 vs 1.81 ms last report p50); the runs were 2 hours apart.

## 3. B2: a depth-7 tree vs one job of the same shape

Workloads, each started by `RegisterJob` and ended by the return of the last task report (the cascade is applied before that reply), validated by `GetJobState(root) = Succeeded` and the DB rows of every job:

- **tree**: a depth-7 binary job tree, 127 one-task jobs; each internal job's task reports 2 children, the 64 leaves do nothing (prototype only);
- **bin127** (the brief's (a)): one job whose 127 tasks form the same binary shape, each task consuming its parent task's output;
- **flat64** (the brief's (b)): one job with 64 independent tasks;
- **chain** and **chain7job** (added to separate the cost of one level from the cost of siblings): a chain of 7 one-task jobs, each registering one child, vs one job with a 7-task chain.

No-op tasks and 10-ms tasks; 64 workers; 5 repetitions per trial after one warm-up. Each trial runs `proto`, `main` and `main-aa` as three storage processes on the same MariaDB container. On main only the single-job workloads run.

Primary data: run `em-async-20261009`, in which the harness EM sends every report from a spawned task, as the real EM does. The first run (`full-20261009`), whose workers awaited their own reports, gave the same results within the intervals (tree 199.0 [186.4, 220.0] ms no-op, 247.1 [229.5, 266.3] ms with 10-ms tasks).

| Workload | Tasks | main | main-aa | proto | COMMIT statements (main / proto) |
|---|---|---|---|---|---|
| tree, 127 jobs | no-op | — | — | 193.8 [187.5, 198.0] | — / 67 [67, 68] |
| bin127, one job | no-op | 31.44 [29.76, 39.91] | 31.52 [26.75, 32.84] | 27.26 [26.87, 28.84] | 2 / 3 |
| flat64, one job | no-op | 11.21 [8.84, 15.19] | 10.57 [9.89, 12.23] | 10.11 [8.96, 11.45] | 2 / 3 |
| chain, 7 jobs | no-op | — | — | 36.83 [36.01, 38.29] | — / 9 |
| chain7job, one job | no-op | 25.09 [23.57, 28.72] | 25.65 [24.55, 26.88] | 25.65 [24.94, 27.37] | 2 / 3 |
| tree, 127 jobs | 10 ms | — | — | 245.0 [237.1, 246.7] | — / 72 [70, 72] |
| bin127, one job | 10 ms | 113.0 [109.8, 117.2] | 112.7 [106.7, 114.3] | 110.4 [108.0, 111.8] | 2 / 3 |
| flat64, one job | 10 ms | 23.46 [22.93, 28.04] | 24.37 [22.58, 25.53] | 23.95 [23.27, 24.80] | 2 / 3 |
| chain, 7 jobs | 10 ms | — | — | 126.8 [125.1, 131.4] | — / 9 |
| chain7job, one job | 10 ms | 109.3 [105.5, 115.4] | 107.3 [105.3, 112.9] | 108.2 [105.7, 112.5] | 2 / 3 |

End to end in ms. The COMMIT statements of a single job are 2 on main and 3 on the prototype, but both binaries run 3 durable DB transactions per job: main's `RegisterJob` is an autocommit `INSERT` that `Com_commit` does not count (§1; B1's fsync counts confirm it, §2).

**Overhead of job nesting**, paired by trial. The depth-7 tree has 7 levels and 6 job-nesting hops, like the 7-job chain. The primary basis is per hop; the per-level figure of the first version of this document (divided by 7) is kept for reference.

| Tasks | (tree - bin127) / 6 hops, both on proto | (tree on proto - bin127 on main) / 6 | (chain - chain7job) / 6 hops, both on proto | (chain on proto - chain7job on main) / 6 | For reference: (tree - bin127) / 7 levels, proto |
|---|---|---|---|---|---|
| no-op | 27.70 [26.60, 28.53] | 26.89 [26.15, 28.05] | 1.82 [1.77, 2.22] | 2.09 [1.60, 2.24] | 23.74 [22.80, 24.45] |
| 10 ms | 22.54 [21.09, 22.99] | 22.05 [20.66, 22.80] | 3.25 [2.90, 3.55] | 3.20 [2.46, 3.73] | 19.32 [18.07, 19.70] |

Tree / bin127 on proto, paired by trial: 7.14 [6.75, 7.37] (no-op), 2.23 [2.14, 2.29] (10 ms).

The same single-job workload on proto vs main (end to end, paired): bin127 0.870 [0.756, 0.931] (A/A 0.977 [0.815, 1.102]) no-op and 0.979 [0.944, 0.994] (A/A 0.993 [0.943, 1.040]) with 10-ms tasks; flat64 0.899 [0.774, 1.280] (A/A 0.952 [0.862, 1.209]) and 1.006 [0.878, 1.061] (A/A 0.999 [0.924, 1.113]); chain7job 1.019 [0.968, 1.095] (A/A 1.020 [0.975, 1.095]) and 1.002 [0.941, 1.034] (A/A 0.988 [0.957, 1.018]). All are inside the A/A bands.

Latency of a registering parent's report in the tree, by level (median over the level's parents; level d has 2^d parents, each registering 2 children):

| Tasks | L0 (1) | L1 (2) | L2 (4) | L3 (8) | L4 (16) | L5 (32) | Leaf report p50 |
|---|---|---|---|---|---|---|---|
| no-op | 3.04 [2.42, 3.66] | 2.81 [2.32, 3.99] | 5.36 [4.95, 7.75] | 11.78 [10.46, 14.56] | 25.23 [24.72, 27.88] | 66.04 [62.28, 69.74] | 5.32 [5.06, 5.75] |
| 10 ms | 3.61 [3.23, 4.53] | 4.20 [3.77, 4.58] | 6.45 [6.07, 7.94] | 7.68 [6.19, 9.78] | 16.02 [12.62, 18.49] | 52.14 [48.09, 58.10] | 5.11 [4.86, 6.15] |

**Replication.** B2 rerun on the current harness build (`b2-rerun-20261009`, 10 trials, the same three processes per trial) gives the same structure: tree / bin127 6.96x [6.45, 7.42] (no-op) and 2.29x [2.20, 2.44] (10 ms), 28.30 [27.31, 30.09] and 23.38 [22.03, 26.58] ms per hop for the tree, 2.02 [1.59, 2.38] and 3.03 [2.35, 4.04] ms per hop for the chain, 67-68 and 71-72 DB transactions per tree, and every single-job ratio inside its A/A band (bin127 no-op 0.918 [0.797, 1.024]).

**The absolute tree latency depends on when it is measured.** The identical depth-7 no-op tree (same binary, 64 workers, spawned reports unless noted, pool 64) measured, in the order of the runs:

| Run (start time), experiment | Depth-7 no-op tree, end to end (ms) |
|---|---|
| `full` (11:36), B2, awaited reports | 199.0 [186.4, 220.0] |
| `full` (11:42), B5 K = 1, awaited reports | 166.5 [148.6, 247.4] (pool 64), 157.9 [142.0, 200.6] (pool 32) |
| `em-async` (12:18), B2 | 193.8 [187.5, 198.0] |
| `em-async` (12:22), B5 K = 1 | 142.0 [139.1, 159.6] (pool 64), 145.6 [133.3, 150.3] (pool 32) |
| `b2gap` (13:16), all four cells | 133.8-147.0; per repetition about 145-270 in trials 0-4 and mostly 110-170 in trials 5-19 (one 660) |
| `b5w` (13:20), B5 K = 1 | 138.2-153.0; 226-307 per repetition in trials 3-5 |
| `b2-rerun` (13:56), B2 | 199.4 [191.1, 208.7] |
| `b2-protoonly` (13:59), B2 with only the proto process per trial | 203.2 [190.5, 215.7] |
| `b2-treeonly` (14:02), B2 with only the no-op tree workload | 192.7 [180.4, 221.6] |
| `b2-treemeter` (14:06), the same with disk accounting | 206.8 (median of the 95 of 100 repetitions whose mean disk flush time was under 0.8 ms) |
| `b2gap-again` (14:08), all four cells | 165.5-172.4; per trial about 200 (trials 0, 3, 4), 370-520 (trials 1-2) and 135-165 (trials 5-9) |

The review proposed two causes for the gap between B2 (194 ms) and B5 at K = 1 (142 ms) in the same run: B5's 5-ms connection probe, which keeps MariaDB and the vCPUs busy, and B2's other workloads interleaved in the same process. `b2gap-20261009` tests both in a 2 × 2 design on one proto process per trial (20 trials, 3 repetitions per cell, cells rotated). The measured tree runs with the probe on or off, right after one unmeasured no-op tree (B5's history) or right after B2's five 10-ms workloads (B2's history). `b2gap` runs the tree with the same calls as B2.

| Cell | Tree end to end (ms) | Level-5 registration report p50 (ms) |
|---|---|---|
| probe off, after a tree | 135.5 [127.6, 147.3] | 44.32 [42.27, 50.17] |
| probe on, after a tree (B5's condition) | 133.8 [126.9, 144.7] | 44.71 [43.26, 52.45] |
| probe off, after B2's workloads (B2's condition) | 147.0 [140.3, 157.8] | 46.68 [43.83, 51.07] |
| probe on, after B2's workloads | 146.0 [137.5, 164.4] | 45.60 [43.82, 56.67] |

| Paired by trial, ratio of end to end | `b2gap` (20 trials) | `b2gap-again` (10 trials) |
|---|---|---|
| probe on / off, after a tree | 1.016 [0.969, 1.071] | 1.007 [0.917, 1.094] |
| probe on / off, after B2's workloads | 1.013 [0.950, 1.046] | 1.001 [0.964, 1.047] |
| after B2's workloads / after a tree, probe off | 1.068 [1.019, 1.160] | 0.989 [0.950, 1.088] |
| after B2's workloads / after a tree, probe on | 1.071 [1.037, 1.124] | 1.010 [0.983, 1.083] |

What the reruns show:

- **Not the probe** (on / off 1.00-1.02 in both runs), **not B2's preceding workloads** (+7 % in one run, nothing in the other, and in any case far from +37 %), **not the main processes** that alternate with proto in B2 (203 ms without them), and **not B2's code path**: `b2gap`, which runs the tree with the same calls, gave about 200 ms and even 370-520 ms in some trials and 135 ms in others.
- **The tree's latency follows a host state that changes over minutes and that the idle gate cannot see.** Background load at every gate was at most 0.33 cores. Part of it is the disk's flush latency. In `b5w`, every K = 1 tree slower than 220 ms ran while the disk's mean flush time was 1.1-1.6 ms, and every tree under 170 ms ran at 0.45-0.6 ms (B1's slow mode, §2). Part of it is not: in `b2-treemeter`, trees took 171-277 ms at normal flush times (0.53-0.65 ms). The cause of that part is not identified. That every B2 run fell into a slow phase is consistent with this, but not proven to be chance.
- **The robust numbers are the ratios paired within a process and the counts.** Tree / bin127 is 7.14x and 6.96x in the two B2 runs, the per-hop overheads agree (27.7 and 28.3 ms), and every tree takes 67-68 DB transactions. The absolute depth-7 latency is 134-207 ms depending on the run, and up to about 400 ms in the slowest phase seen. The per-hop overhead in ms carries the same dependence.

**Findings.**

- **No single-job workload is slower on the prototype.** Every proto / main ratio is inside its A/A band. bin127 no-op is in fact faster on proto in 10 of 10 trials (0.870 [0.756, 0.931]), but main's own bin127 spread (28.7-40.3 ms) makes its A/A band wide ([0.815, 1.185] mirrored). The two binaries also see different histories in the process: proto's bin127 always runs right after a tree, while main's runs after the previous repetition's 10-ms chain.
- **The depth-7 tree takes 7.1x [6.75, 7.37] the time of the same shape inside one job** (no-op; 193.8 vs 27.3 ms in em-async), **27.7 ms per job-nesting hop.** One hop by itself is cheap: the chain pays 1.8 ms (no-op) to 3.3 ms (10-ms tasks) per hop over a task level, about one registration DB transaction.
- **The rest is the serialization of sibling registrations.** Design §7.5 rule 3 admits one registering parent per tree state transaction, so a level with 2^d registering parents is written by 2^d consecutive DB transactions. The tree needs 67 DB transactions: `RegisterJob`, `StartJob`, 63 registrations, and 2 rounds for the 64 leaf completions and the cascade. The median parent of level 5 waits 66 ms (em-async B2; 44-47 ms in `b2gap`'s faster phase) for a 2-child registration that alone takes about 2-3 ms. The rule guards completions against queues of large registrations (gate H16). For small registrations it makes a wide tree's expansion linear in its number of internal jobs. Grouping several registering parents into one tree state transaction, within the existing per-round caps (16 MiB, 256 events), would bring the depth-7 tree to about 6 registration rounds. The chain's per-hop cost suggests roughly 40-50 ms end to end; that is an extrapolation, not a measurement.

## 4. B3: cancel and leaf failure on the depth-7 tree

Two runs. Both expand the depth-7 tree before each operation and then abort it. The first, `full-20261009`, used awaited EM reports. The rerun, `b3-rerun-20261009`, used spawned reports, counted the COMMIT statements between the abort's send and its acknowledgement, and added the race variants below.

**The idle tree.** The tree is fully expanded: 63 jobs `BlockingOnChildren` and 64 leaf jobs `Running`, with their instances registered and held (checked from the DB). Then either `CancelJob(root)` or a `ReportTaskFailure` of one leaf (`max_num_retry = 0`). In the cleanup variants every job of the tree defines a cleanup task, which the harness runs as soon as it is published. The acknowledgement is the RPC latency. Quiescence is taken from the harness's own replies. Without cleanup tasks it is the acknowledgement itself. With cleanup tasks it is the return of the last expected cleanup report: 127 after a cancel, 120 after a failure (every non-terminal job except the failed leaf and its 6 `Failed` ancestors). One DB probe after the measured interval checks that no job of the tree is non-terminal and that the counts are exactly `127 Cancelled` (cancel) or `120 Cancelled + 7 Failed` (failure). The root's final state is checked with `GetJobState`. 3 repetitions per trial after one warm-up, 10 trials per run.

| Case | Ack, first run (ms) | Ack, rerun (ms) | Quiescent, first run (ms) | Quiescent, rerun (ms) | COMMIT statements, send → ack (rerun) | Final rows |
|---|---|---|---|---|---|---|
| Cancel, no cleanup tasks | 5.30 [4.82, 6.24] | 6.35 [4.58, 7.07] | = ack | = ack | 1 in 30 of 30 | 127 `Cancelled` |
| Cancel, a cleanup task on every job | 4.40 [3.57, 5.71] (reply `CleanupReady`) | 3.53 [3.20, 4.06] | 21.09 [19.61, 24.07] | 16.51 [13.90, 18.43] | 1 in 30 of 30 | 127 `Cancelled` |
| Leaf failure, no cleanup tasks | 8.11 [5.87, 8.70] (root `Failed` when the reply returns) | 6.44 [5.45, 8.15] | = ack | = ack | 1 in 30 of 30 | 7 `Failed`, 120 `Cancelled` |
| Leaf failure, a cleanup task on every job | 5.30 [3.80, 6.32] | 4.42 [3.99, 5.43] | 21.82 [20.12, 25.82] | 15.91 [15.51, 18.04] | 1 in 30 of 30 | 7 `Failed`, 120 `Cancelled` |

The quiescence probe and the row counts were exact in all 240 measured idle operations of the two runs (and in their 80 warm-ups).

| Paired by trial: ack without cleanup - ack with cleanup (ms) | First run | Rerun |
|---|---|---|
| Cancel | 1.26 [-0.79, 2.11], > 0 in 8 of 10 trials | 2.42 [1.37, 4.21], > 0 in 10 of 10 |
| Leaf failure | 2.58 [0.40, 4.48], > 0 in 10 of 10 | 1.97 [0.92, 3.36], > 0 in 9 of 10 |

**The race variants** (rerun only). The same tree, leaves held, is aborted while it is still expanding. A root cancel is sent as soon as the first level-5 registration report returns. A leaf failure is reported as soon as the first leaf instance is held. At that moment the harness has 23-31 level-5 registration reports in flight, which wait at the committer behind one registration per tree state transaction. After the acknowledgement the harness waits until no record arrives for 300 ms. Then it checks that every row of the tree is terminal and that the tree has exactly 1 + 2 × (registration reports answered OK) rows, so no registration answered with an error inserted anything. A second probe 200 ms later must find the same rows, and the root must be `Cancelled` or `Failed`. Every reply after the abort is either OK, a stale `FAILED_PRECONDITION`, or a held leaf. 3 repetitions per trial after one warm-up, 10 trials.

| Case | Ack (ms) | Last in-flight reply (ms after the send) | Registrations in flight at the send | COMMIT statements, send → ack | Registrations answered OK after the send | Stale replies | Tree rows |
|---|---|---|---|---|---|---|---|
| Cancel during expansion | 5.07 [4.41, 7.37] (max 8.16 of 30) | 5.17 [4.46, 7.57] | 31 [29, 31] | 2 in 30 of 30 | 1 in 30 of 30 | 30 [30, 30] | 67 [67, 67] |
| Leaf failure during expansion | 5.62 [4.97, 6.82] (max 11.46 of 30) | 5.68 [5.06, 6.96] | 30 [29, 31] | 1 in 11, 2 in 19 of 30 | 1 in 30 of 30 | 29 [29, 29] | 68 [67, 69] |

Every check held in all 60 race operations (and their 20 warm-ups).

- **A whole-tree abort of the idle tree is one DB transaction**, acknowledged in 3.5-8.1 ms for 127 jobs. In all 120 measured idle aborts of the rerun, exactly one COMMIT statement ran between the send and the acknowledgement, with nothing else writing (the leaves are held). The rows were all terminal and exact at the DB probe that follows the acknowledgement, in every operation of both runs. That the abort is visible in the DB at the moment of its commit, not only by the next probe, is what the race tests T2, T3 and T7 check (`job_tree_cancel_test.rs`, `tests.md`), not this benchmark. The design's emulation measured the root `Failed` 3.43 ms after the failure at depth 6 (A5); the real code adds the RPC, the collection of 127 JCB states and the apply of 127 jobs.
- **An abort that races the expansion waits for at most the one registration in flight.** A cancel sent while about 30 registrations were queued was acknowledged in 5.07 ms, with exactly two COMMIT statements before the acknowledgement in all 30 operations: the round in flight, which carried one registration and answered it OK, and the abort round. The other 30 queued registrations were answered `FAILED_PRECONDITION` (stale) and inserted nothing. This is the bound the design claims (cancel first, one registering parent per tree state transaction, §7.5). A leaf failure is not moved to the front like a cancel, but `take_events` takes non-registering events past the queued registrations, so it too waited for at most the round in flight: 5.62 ms, one or two COMMIT statements. The worst of 30 was 8.2 ms (cancel) and 11.5 ms (failure).
- **Terminal targets cost more than `CleanupReady` targets in the failure case** (2.58 [0.40, 4.48] and 1.97 [0.92, 3.36] ms in the two runs). For the cancel the first run cannot tell (1.26 [-0.79, 2.11] ms), and the rerun shows 2.42 [1.37, 4.21] ms. A hypothesis for the mechanism, not tested: jobs that end terminal also get `cancel_non_terminal`/`finalize`, leave the tree structure and are GC-enqueued inside the same round.
- **With cleanup tasks, quiescence is the harness EM's dispatch and reports plus the `CleanupDone` rounds.** The cleanup tasks themselves do nothing. Changing only the EM's report mode, from awaited (first run) to spawned (rerun), moved it from about 21 ms to about 16 ms, so the harness is part of this figure.

## 5. B4: 1,000 siblings finishing at once

Run `full-20261009`. A root whose task reports 1,000 one-task children; all 1,000 child instances are registered and held; then their success reports are released together by 64 or by 1,000 concurrent reporters. Reference: 1,000 independent plain jobs held and released the same way, on `main` and on the prototype. Completions/s = 1,000 / (last reply - first send). COMMIT statements are the `Com_commit` delta over the release; on the prototype every runtime `jobs` write is an explicit transaction, so there this count equals its DB transactions (§1).

| Variant | Shape, concurrent reporters | Completions/s | Report p50 (ms) | Report p99 (ms) | COMMIT statements | Events per DB transaction |
|---|---|---|---|---|---|---|
| proto | 1,000 siblings, 64 | 10,815 [7,411, 12,131] | 5.57 [5.03, 8.22] | 7.93 [6.44, 11.20] | 32 [32, 32] | 31.2 [31.2, 31.2] |
| proto-aa | 1,000 siblings, 64 | 10,781 [9,185, 11,201] | 5.68 [5.35, 6.70] | 8.38 [7.22, 11.58] | 32 [32, 32] | 31.2 [31.2, 31.2] |
| proto | 1,000 siblings, 1,000 | 21,745 [15,386, 22,914] | 27.12 [24.93, 30.16] | 45.20 [42.58, 64.16] | 5 [5, 6] | 200 [167, 200] |
| proto-aa | 1,000 siblings, 1,000 | 21,475 [17,753, 22,956] | 24.59 [23.58, 28.31] | 45.71 [42.66, 55.56] | 5 [5, 5] | 200 [200, 200] |
| main | 1,000 plain jobs, 64 | 9,119 [7,408, 9,710] | 6.28 [5.96, 7.47] | 16.03 [12.95, 20.39] | 1,000 | 1.0 |
| proto | 1,000 plain jobs, 64 | 9,514 [8,204, 10,964] | 5.79 [5.39, 7.27] | 14.14 [11.78, 21.79] | 1,000 | 1.0 |
| main | 1,000 plain jobs, 1,000 | 10,724 [8,754, 12,135] | 47.39 [42.65, 65.12] | 91.50 [79.70, 113.3] | 1,000 | 1.0 |
| proto | 1,000 plain jobs, 1,000 | 10,860 [8,613, 12,399] | 48.83 [38.30, 57.36] | 90.50 [78.61, 115.0] | 1,000 | 1.0 |

| Completions/s ratio | Ratio | A/A (proto' / proto) |
|---|---|---|
| 1,000 plain jobs, 64 reporters: proto / main | 1.013 [0.926, 1.361] | 0.962 [0.803, 1.275] |
| 1,000 plain jobs, 1,000 reporters: proto / main | 0.962 [0.890, 1.225] | 0.978 [0.657, 1.192] |
| 1,000 siblings on proto / 1,000 plain jobs on main, 64 reporters | 1.177 [0.870, 1.448] | 1.011 [0.796, 1.476] |
| 1,000 siblings on proto / 1,000 plain jobs on main, 1,000 reporters | 1.963 [1.633, 2.111] | 1.006 [0.875, 1.408] |

Every release ended with the root `Succeeded` and all 1,001 rows `Succeeded`.

- **Group commit works on the real code, and the grouping is bounded by the reporters, not the committer.** With 64 closed-loop reporters (each sends its next report when the previous one returns), a tree state transaction carries 31.2 events: 1,000 completions in 32 DB transactions, about 2.9 ms per round. The reporters split into two cohorts, one in the round in flight and one queued for the next, so a round holds C / 2 events and the throughput is about (C / 2) / (round time) ≈ 11,000/s. With 1,000 reporters a round carries 200 events (5 DB transactions of about 9 ms each; the cap is 256) and the tree completes 21,700/s, 1.96x [1.63, 2.11] the rate of 1,000 independent plain jobs on main.
- **Context from a different harness:** the design's emulation reported 62.7 events per round at C = 64 (A3, 23,186/s). That emulation had its own reporter model and did not run this code, so the comparison is context, not a measurement of the same thing.
- **At 64 reporters the grouped tree is not distinguishable from independent plain jobs** (1.18x [0.87, 1.45] main): both are bounded by the reporters' round trips here, and a plain job's single-row DB transaction runs in parallel with the other 63.
- **Plain jobs released together: no difference detected between proto and main, with little power.** The ratios are 1.013 and 0.962, but the A/A bands are [0.803, 1.275] (64 reporters) and [0.657, 1.192] (1,000 reporters): a difference smaller than about 25-35 % could not show here.
- **The 1,000-child registration report** (the root's `ReportTaskSuccess` carrying 1,000 children, `registration_report_ms` in the raw data) measured 85.9 [74.1, 100.4] ms in the first recorded phase, which is the second registration of the storage process, and 20.6 [16.7, 27.1] ms in the second. A diagnostic (`b4diag-20261009`, 3 trials of 12 consecutive registration-release cycles on one storage process) shows that early registrations are slow and later ones are not: on a fresh storage process and database the first two took 123-133 ms and 196-201 ms, and every later one 24.3 ms median (18.9-64.5 ms over 30 cycles), whichever concurrency released the previous tree. **The diagnostic does not reproduce B4's number:** its second registration took 196-201 ms in all three trials, more than twice B4's 85.9 ms, so the cause of the slow early registrations stays open (the fresh tablespace growing and the process heap's first growth are untested candidates). The later registrations match the design's A6 emulation (19.8 ms for 1,000 children of 1 KiB), again across harnesses.

## 6. B5: concurrent trees

K = 1, 16 and 64 depth-7 no-op trees submitted at once, sharing the harness's W EM workers, with the storage pool at 64 and at 32 connections. Per-tree end to end runs from that tree's `RegisterJob` to its last report. Active connections are sampled every 5 ms from `information_schema.PROCESSLIST` (storage connections not in `Sleep`). These samples miss sub-5-ms transactions, so they are a lower bound next to the `Max_used_connections` peak (which includes idle pool connections, the harness's root connection and its probe).

Primary data: run `em-async-20261009` (W = 64, spawned reports).

| Pool | K | Per-tree e2e p50 (ms) | Per-tree e2e max (ms) | p50 / (K = 1) | Active conns mean / max | `Threads_running` max | `Max_used_connections` | DB transactions per tree |
|---|---|---|---|---|---|---|---|---|
| 64 | 1 | 142.0 [139.1, 159.6] | = p50 | 1 | 0.54 / 1 | 1 | 5 | 67.0 |
| 64 | 16 | 352.5 [333.2, 404.5] | 359.7 [341.8, 411.7] | 2.43 [2.30, 2.85] | 11.1 / 16 | 16 | 33 | 67.1 |
| 64 | 64 | 792.8 [749.9, 863.8] | 807.0 [769.6, 885.8] | 5.60 [4.77, 5.77] | 38.3 / 64 | 64 | 67 | 67.1 |
| 32 | 1 | 145.6 [133.3, 150.3] | = p50 | 1 | 0.53 / 1 | 1 | 5 | 67.0 |
| 32 | 16 | 347.4 [344.3, 406.9] | 355.6 [351.0, 411.1] | 2.57 [2.30, 2.80] | 11.0 / 16 | 16 | 33 | 67.1 |
| 32 | 64 | 818.1 [785.3, 1,017] | 826.0 [792.4, 1,028] | 5.86 [5.15, 6.97] | 19.2 / 32 | 32 | 35 | 67.0 |

Medians of the per-trial values with their 95 % CIs; the connection columns are medians (their CIs are within ±1 of the median for the maxima). Every root ended `Succeeded` and every row `Succeeded` (127 K per K).

**The EM worker count and the host's load** (rerun `b5w-20261009`: pool 64, W = 64, 128 and 256 workers, spawned reports, 10 trials; CPU from `/proc`, the MariaDB container's cgroup and the storage process, and disk flushes from `/proc/diskstats`, over each measured interval):

| W | K | Per-tree e2e p50 (ms) | p50 / (K = 1) | COMMITs/s | Disk flushes/s (mean ms) | Storage cores | MariaDB cores | Harness cores | Host busy cores (of 16) | Harness dispatch p50 / p99 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 1 | 143.0 [132.3, 256.7] | 1 | 472 | 519 (0.53) | 0.42 | 0.37 | 0.36 | 1.47 | 0.30 / 0.50 |
| 64 | 16 | 355.7 [332.0, 674.2] | 2.49 [2.44, 2.63] | 2,952 | 990 (0.65) | 2.25 | 2.70 | 1.23 | 6.21 | 0.66 / 1.81 |
| 64 | 64 | 842.9 [740.0, 921.6] | 5.45 [3.39, 6.01] | 4,889 [4,515, 5,665] | 700 (0.65) | 3.05 | 5.12 | 1.67 | 9.69 | 1.26 / 4.71 |
| 128 | 64 | 803.9 [767.2, 919.1] | 5.62 [3.28, 6.10] | 5,221 [4,545, 5,454] | 717 (0.65) | 3.06 | 5.01 | 1.66 | 9.63 | 1.34 / 5.06 |
| 256 | 64 | 821.9 [739.1, 890.6] | 5.29 [3.19, 5.82] | 5,072 [4,701, 5,621] | 721 (0.63) | 3.12 | 5.18 | 1.70 | 9.79 | 1.26 / 4.54 |

COMMITs/s is the `Com_commit` delta over the makespan; harness dispatch is the time from a task's poll to its report's send (the router queue plus `RegisterTaskInstance`). W = 128 and 256 at K = 1 and 16 match W = 64 within their intervals (`b5w-20261009/tables.md`). Every root and row ended `Succeeded`.

Sensitivity: in the first run, whose workers awaited their reports, K = 64 took 1,110 [913, 1,640] ms per tree with a pool of 64 and 940 [857, 1,076] ms with 32, and a tree needed 109 DB transactions: workers parked behind slow registration reports spread the leaf completions over more rounds.

- **Trees are isolated from each other in the fairness sense.** No tree starves: the slowest of 64 concurrent trees finishes within 1.02x of the median one (807 vs 793 ms). Each committer keeps at most one DB connection busy, so active connections never exceed K (16 at K = 16) and reach the pool size at K = 64. Isolation was measured only as fairness among identical trees submitted together; a noisy neighbour (one tree with a large registration next to a small tree) was not measured.
- **A tree's latency grows with the number of concurrent trees** (2.4x at K = 16, 5.6x at K = 64), and neither the pool size nor the harness's worker count changes it: pool 32 vs 64 gives 818 vs 793 ms, and W = 64, 128 and 256 give 843, 804 and 822 ms, with overlapping intervals.
- **About 5,000 COMMITs/s is the rate this setup achieved at K = 64, not a measured capacity of the DB.** em-async achieved 5,321 [4,840, 5,577]/s with pool 64 (about 64 trees × 67.1 / 0.81 s) and 5,190/s with pool 32, and b5w 4,900-5,200/s. The same server committed 6,608 [5,564, 6,977]/s at K = 64 with pool 32 in the first run (a different mix, 109 smaller transactions per tree) and about 10,700/s for main's 1,000 plain jobs in B4. The bound is not identified, but the rerun narrows it:
  - not the harness's EM workers: quadrupling them changes nothing, and dispatch stays at 1.3 ms;
  - not the host's CPU: 9.7 of 16 cores are busy (MariaDB 5.1, storage 3.1, harness 1.7);
  - not the disk's flush rate: flushes keep the device busy about 46 % of the interval (about 700/s at 0.65 ms, about 7 commits per flush, so group commit is at work).

  What remains is the latency of a tree state transaction under 64 concurrent ones. The per-tree time at K = 64 is about 12.6 ms per round, against 2.1 ms at K = 1. That round time spans MariaDB's statement latency with 64 statements running (for example the shared lock every transaction takes on the one fence row, or the redo log), and the storage process's own per-round path. Neither was measured.

## 7. B6: memory per live tree

Run `full-20261009`. 100,000 plain jobs registered and started by 64 clients and never given instances; the drain-only poller takes their task entries, so every lane and the prototype's staging queue are empty when RSS is sampled. RSS is sampled once the poller has drained exactly N entries and three consecutive 0.5-s samples agree within 0.2 %; bytes per job = (RSS after - RSS before the first job) / N. This is an RSS delta, an upper bound on live memory: VmHWM equals VmRSS at every sample, so whatever the allocator retains from the load phase is included.

| Variant | Bytes per live job | RSS before (KiB) | RSS with 100,000 live jobs (KiB) |
|---|---|---|---|
| main | 1,490 [1,479, 1,510] | 48,686 | 194,170 (190 MiB) |
| proto (boxed round future) | 15,490 [15,450, 15,558] | 49,376 | 1,562,036 (1.49 GiB) |
| proto-unboxed | 16,237 [16,202, 16,277] | 49,354 | 1,634,992 (1.56 GiB) |
| proto-smallbuf (diagnostic) | 5,073 [5,053, 5,099] | 49,532 | 544,974 (532 MiB) |

| Paired difference | Bytes per live job |
|---|---|
| proto - main | 13,999 [13,957, 14,053] |
| proto-unboxed - proto | 735 [708, 782] |
| proto - proto-smallbuf (the arrival buffer) | 10,426 [10,388, 10,478] |
| proto-smallbuf - main (everything else) | 3,578 [3,562, 3,609] |

Type sizes (`sizes/`, a probe against the prototype crate, and the prototype's own `committer_future_size` test with mock connector types): `size_of::<Pending>()` = 48 B; the `run_committer` future = 768 B boxed, 1,592 B unboxed.

- **The design's "a few hundred bytes" per live tree (§5.4, §13) is too low by about 12x for the design itself and about 50x for this prototype.** Every live plain job costs 14.0 KB more than on `main` (10.4x main's per-job memory: 1.49 GiB instead of 190 MiB for 100,000 live jobs), which is about 50x the estimate, but 10.4 KB of that is an implementation defect (next bullet). Without it the extra cost is 3.6 KB per live job, about 12x the estimate. Both are RSS deltas, so upper bounds.
- **10.4 KB is the committer's arrival buffer.** `run_committer` (`state/tree_committer.rs:438`) creates `Vec::with_capacity(MAX_EVENTS_PER_ROUND)` before each `recv_many`, so every parked committer holds 256 × 48 B = 12,288 B for its whole life (10.4 KB of it resident here). tokio's `recv_many` pushes into the vector and grows it as needed, so starting from `Vec::new()` is correct; the diagnostic build with that one change cuts the extra cost from 14.0 KB to 3.6 KB per live job. Its latency cost (a few small reallocations per round) was not measured: `proto-smallbuf` ran only in B6.
- **The remaining 3.6 KB per tree** is the committer task (the 768-B future plus the task header), the unbounded channel with its first block of 32 slots (32 × 48 B = 1,536 B plus the header), the first hash-table allocations of the tree structure and of the handle map, and the new JCB fields. It is not broken down further.
- **Keep `Box::pin` on the round future.** Inlining it doubles the parked future (768 → 1,592 B) and costs 735 [708, 782] B more per live tree (4.8 % over the boxed prototype); boxing costs one allocation per round, which B1 does not show.

## 8. B7: restart recovery

Run `full-20261009`. N live plain jobs (`Running`, no instances) on `main` and on the prototype, plus a prototype variant with N plain jobs and ⌊N/11⌋ trees of one `BlockingOnChildren` parent with 10 `Running` children (the plan's workload). SIGKILL, then a new process on the same database with `RUST_LOG=warn`. Measured from the spawn: the first successful `GetSession` (recovery runs inside `create_runtime`, before `serve`, on both binaries) and the moment the drain-only poller, attached as soon as `GetSession` succeeds, has received every expected resent entry (N, or N + 10 ⌊N/11⌋ with trees: the `BlockingOnChildren` parents publish nothing). The checks: exactly the expected number of entries within 1 s after the last one, and the DB state counts before the crash equal those after the restart.

| N | Variant | Live jobs | Spawn → first `GetSession` (s) | Spawn → every entry resent (s) | Resent entries | RSS after recovery |
|---|---|---|---|---|---|---|
| 10,000 | main | 10,000 | 0.115 [0.108, 0.120] | 0.129 [0.124, 0.134] | 10,000 | 66 MiB |
| 10,000 | proto | 10,000 | 0.159 [0.152, 0.166] | 0.174 [0.166, 0.183] | 10,000 | 135 MiB |
| 10,000 | proto + 909 trees | 19,999 | 0.675 [0.634, 0.715] | 0.702 [0.658, 0.740] | 19,090 | 161 MiB |
| 100,000 | main | 100,000 | 0.891 [0.879, 0.909] | 1.004 [0.988, 1.025] | 100,000 | 212 MiB |
| 100,000 | proto | 100,000 | 1.321 [1.254, 1.467] | 1.459 [1.378, 1.572] | 100,000 | 917 MiB |
| 100,000 | proto + 9,090 trees | 199,990 | 6.163 [6.008, 6.337] | 6.364 [6.233, 6.570] | 190,900 | 1,170 MiB |

Same number of live jobs, proto / main (paired by trial): spawn → `GetSession` 1.414 [1.296, 1.486] at N = 10,000 and 1.502 [1.379, 1.679] at N = 100,000; spawn → every entry resent 1.369 [1.272, 1.464] and 1.460 [1.326, 1.616].

The tree variant has about twice as many live jobs as the plain variants (2N - 1 against N), so its ratio to main is not a like-for-like comparison. Per live job, spawn → `GetSession` takes 11.5 µs (10,000) and 8.9 µs (100,000) on main, 15.9 and 13.2 µs on the plain prototype, and 33.8 and 30.8 µs on the prototype with trees.

Every restart resent exactly the expected entries and left the DB state counts unchanged (`Running` N + 10 ⌊N/11⌋, `BlockingOnChildren` ⌊N/11⌋).

**Where the trees' recovery time goes** (follow-up `b7-fanout-20261009`): the same 100,000 plain jobs plus 99,990 tree jobs, arranged as 9,090 trees of 10 children or as 990 trees of 100 children. The two shapes ran in two containers one after the other (host ports 40667 for 100 children, then 51989 for 10 children), 5 trials each, so the table gives medians with [min, max] and the two shapes are compared by their medians, not paired.

| Trees | `BlockingOnChildren` parents | Spawn → first `GetSession` (s) | Spawn → every entry resent (s) |
|---|---|---|---|
| 9,090 × (1 + 10) | 9,090 | 6.033 [5.772, 6.079] | 6.245 [6.006, 6.295] |
| 990 × (1 + 100) | 990 | 2.742 [2.667, 2.877] | 3.022 [2.890, 3.085] |

The difference of the medians is 3.29 s for 8,100 more `BlockingOnChildren` parents: 0.406 ms per parent (pairing trial i of one container with trial i of the other gives 0.367-0.416, but those pairs share nothing). `repair_tree` (`state/service.rs:1059-1072`) calls `has_failed_child` once per `BlockingOnChildren` row, one after another, and that query runs with `.persistent(false)` (`db/mariadb.rs:471-486`), so every parent costs a DB round trip plus a statement prepare: about 3.5 s of the 6.0 s here. This attribution holds by elimination, not by a direct measurement: `repair_tree` makes no other per-parent DB call. A cross-run estimate agrees with it: the 990-tree shape takes (2.742 - 1.321) s = 1.42 s more than the plain-job run of the table above for its 99,990 tree jobs, about 14 µs per tree job, or about 10 µs once 990 × 0.406 ms of parent probes are taken out, close to a plain prototype job's 13.2 µs. That estimate subtracts times from two different runs.

- **Plain jobs recover 1.41-1.50x slower than on `main`**: 13.2 vs 8.9 µs per live job at N = 100,000 (1.32 vs 0.89 s to the first `GetSession`). Each recovered job now also gets a tree channel and a spawned committer. RSS after recovery is 917 MiB against main's 212 MiB, consistent with B6 (it is lower than B6's 1.49 GiB for the same N, which suggests that fewer of the 12 KiB arrival buffers' pages are touched when the committers are created back to back; not investigated).
- **Trees recover more slowly than their job count suggests because of the per-parent repair probe** (by elimination, above): about 0.4 ms per `BlockingOnChildren` parent, so a deployment with 10,000 blocked parents would add about 4 s to every restart. One batched query per chunk of parents, or the same information folded into the recovery query, removes it.

## 9. Conclusions

1. **No regression was detected on plain-job transitions; job creation got slower** (B1, two runs of n = 20).
   - **Transitions.** The last report's p50 is 1.02-1.03x main at C = 1 and C = 64, and `StartJob` is 1.01x at C = 1 and 1.05x at C = 64. All are inside the A/A bands, but the intervals allow up to +12 % (C = 1) and +9 % (C = 64) on the last report. Gate H12 holds on its point estimates.
   - **Gate H11.** It also asks for "the A3 throughput" within the A/A band, and that half is undecided. B4's plain-job comparison has A/A bands of [0.803, 1.275] and [0.657, 1.192]. B1's lifecycle throughput, which includes `RegisterJob`, is 5-11 % lower, outside the band at C = 1 in the replication.
   - **Job creation.** `RegisterJob` is 23-25 % (C = 1) to 37-38 % (C = 64) slower. The root insert became an explicit transaction with the fence statement, and the tree's channel and committer are created with it. Both binaries make 3 durable transactions (3 fsyncs) per job, so the cost is round trips and CPU, not durability: MariaDB spends 1.12x the CPU per job, and the storage process 1.22x at C = 64.
   - **The whole lifecycle** is 3.6-4.7 % slower at C = 1, at the edge of the A/A band, and 10.7-11.1 % slower at C = 64, outside it.
   - **Single jobs.** Single-job workflows of 7-127 tasks showed no slowdown in B2 (every ratio inside its A/A band). 1,000 plain jobs finishing together in B4 showed no difference, but inside A/A bands of about ±25-35 %, too wide to detect anything smaller.
2. **Memory per live job is 10.4x main's, and the design's estimate is too low** (B6). The prototype costs 14.0 KB more per live plain job than main: 1.49 GiB instead of 190 MiB for 100,000 live jobs, an RSS delta and so an upper bound. Against the design's "a few hundred bytes" that is about 50x. Of the 14.0 KB, 10.4 KB is one line: the committer pre-sizes its arrival buffer to `MAX_EVENTS_PER_ROUND` × 48 B before every `recv_many`, so every parked committer holds 12 KiB. Starting from `Vec::new()` brings the overhead down to 3.6 KB per live job, still about 12x the estimate; the latency cost of that change was not measured. Keep `Box::pin` on the round future: the unboxed variant costs 735 B more per live tree.
3. **Wide trees are slow to expand because sibling registrations are serialized** (B2, B5).
   - **The cost.** A depth-7 binary tree takes 7.0-7.1x the time of the same shape inside one job: 27.7-28.3 ms per job-nesting hop, while a hop with a single child costs 1.8-2.0 ms (the chain, per hop on the same basis). The difference is the rule "one registering parent per tree state transaction" (design §7.5 rule 3). The tree needs 67 DB transactions, 63 of them two-child registrations, and level 5's parents wait 44-68 ms at the median. In absolute terms the tree took 134-207 ms depending on the run, and up to about 400 ms in a slow host phase.
   - **Under concurrency.** With 64 concurrent trees, per-tree latency grows 5.6x. A pool of 32 connections does as well as 64, and 256 EM workers as well as 64. The bound is not identified: the host's CPU (9.7 of 16 cores busy) and the disk's flushes (46 % busy) are not saturated, and about 5,000 COMMITs/s is what this setup achieved, not a capacity.
   - **The lever.** Admit several registering parents per tree state transaction, within the existing payload and event caps. The chain suggests about 40-50 ms for the depth-7 tree, an extrapolation.
4. **Group commit works on the real code** (B4). 1,000 sibling completions commit in 32 DB transactions at 64 reporters (31 events each) and in 5 at 1,000 reporters (200 events each). That reaches 21,700 completions/s, 1.96x [1.63, 2.11] the rate of 1,000 independent plain jobs. With closed-loop reporters a round holds about half of them.
5. **An abort is one DB transaction, and a racing abort waits for at most one registration** (B3).
   - **The idle tree.** A cancel or a leaf failure on the expanded 127-job tree committed exactly one transaction before its acknowledgement in all 120 measured operations. It was acknowledged in 3.5-8.1 ms, and the tree was all terminal by the probe that follows.
   - **The race.** Sent while about 30 registrations were queued, a cancel was acknowledged in 5.1 ms after exactly two transactions: the registration in flight and the abort. The queued registrations were answered stale and inserted nothing. A leaf failure behaved the same way (5.6 ms).
   - **Cleanup tasks.** With them, the tree is quiescent 16-22 ms later. The harness's own dispatch and reports are part of that figure.
6. **Restart recovery is 1.4-1.5x main's for plain jobs, and a per-parent probe dominates it for trees** (B7). Each `BlockingOnChildren` parent adds about 0.4 ms; the cause, one sequential `has_failed_child` query each, is attributed by elimination. That is about 3.5 of the 6.0 s it takes to recover 9,090 trees (99,990 jobs) next to 100,000 plain jobs.

**Traps for the formal implementation** (from the measurements):
- Do not pre-size the committer's arrival buffer (`Vec::with_capacity(MAX_EVENTS_PER_ROUND)` before `recv_many`). Every parked committer holds it (B6, 10.4 KB per live tree). Re-measure the per-tree fixed cost (3.6 KB remaining) against the design's estimate, and measure the latency of the `Vec::new()` variant.
- One registering parent per tree state transaction makes registration latency linear in the number of siblings registering at once (B2: 44-68 ms at 32 siblings; 63 DB transactions for a 127-job tree). Group small registrations, and keep the guard for large ones.
- Batch the startup repair probe (`has_failed_child`) instead of one non-persistent query per `BlockingOnChildren` parent (B7, about 0.4 ms per parent), and time it directly.
- `RegisterJob`'s fenced insert transaction costs +0.3 to +1.2 ms per job (B1). It is the only clearly measurable plain-job latency regression, and with the fence in every transaction MariaDB spends about 12 % more CPU per job. A plain job's other transitions were not shown slower, but a few percent are not excluded; H11 and H12 need a run with more power, or a decision on what the bands mean, before PR 3.
- The first two large registrations of a fresh storage process and database were 5-8x slower than the steady state (B4 diagnostic), and the cause is open. A benchmark or gate on registration latency needs a warm-up.
- For whoever benchmarks next on this host: record the disk's flush latency (or `Innodb_data_fsyncs` and its timing) with every measurement, counterbalance the process order, and compare only within a process or a trial. Slow phases of the host's disk, and others not identified, change absolute latencies by 1.5-3x for seconds to minutes, and the idle gate cannot see them.

## 10. Deviations from the plan

- **No `committer-stats` feature.** The prototype does not have it; the harness counts COMMIT statements with `Com_commit` instead (exact counts on the prototype, no per-round histogram, no per-transaction durations; §1 for what it does not count on main).
- **The unboxed variant is a `cfg` flag**, `spider_unboxed_round`, not the `unboxed-round` feature: the prototype moved it out of `--all-features` after the style review.
- **`RUST_LOG=warn` for every experiment** (the plan had `info` for B2-B5): the storage logs every RPC at `info`, which would put JSON logging on every measured path. Both binaries get the same filter.
- **The EM's reports come from spawned tasks in the B2, B3 (rerun) and B5 data** (`JT_REPORT_SPAWN=1`), as the real EM sends them. The plan's worker awaited each report before taking the next assignment, which the first run did; its results are kept as sensitivity results (§3, §4, §6). The report mode reaches every experiment whose EM workers run tasks: B2, B3's cleanup quiescence (§4) and B5. B1 has no EM workers; B4 and the B3 acknowledgements time their own reports.
- **Generated tonic clients instead of `SpiderClient`**, so no retry wrapper and no per-call graph compression is inside a measured interval. **No EM heartbeat:** the stale cutoff is 3,600 s and no run lasts that long.
- **Added workloads and variants:** the chain workloads in B2; a prototype plain-job reference and an A/A variant in B4; the `proto-smallbuf` diagnostic in B6; a like-for-like plain-job prototype variant in B7 next to the plan's tree variant; an unrecorded warm-up per storage process (B2-B5) and, in the reruns, per container; the B4 registration diagnostic (§5); and, after the methodology review, the counterbalanced B1 replication, the B3 race variants, the B2/B5 gap experiment and the B5 worker sweep.
- **The in-process `jcb_test` supplement for B1 was not run.** Its MariaDB workloads have one job transition per run, so it would give 10 samples per checkout; B1's `StartJob` latency is already a pure job transition (no task work) measured on both binaries.
- **MariaDB 10.11.16 through a replica of `start.py`'s `docker run`**, as in the previous harness, because `start.py` hard-codes `mariadb:latest`.
- **Not done:** a noisy-neighbour isolation case for B5 (a large registration in one tree with a small tree's latency measured next to it), a direct timing of B7's per-parent repair probe, and the `smallbuf` variant on B1/B4 latency.

## 11. Changes after the methodology review

`review.md` (methodology reviewer) required nine fixes. This version applies them and reruns B1, B2, B3 and B5 (§1 lists the runs).

| Fix | Where | What changed |
|---|---|---|
| R1 | §2, Summary, §9.1 | "No regression detected" with the CI upper bounds instead of "costs what today's path costs"; H11 quoted in full; its throughput half reported as undecided. |
| R2 | §1, §2, §3, §9.1 | `Com_commit` explained (COMMIT statements only); main's `RegisterJob` named as an uncounted autocommit transaction; "3 DB transactions per job instead of 2" corrected (both binaries run 3, measured by `Innodb_data_fsyncs` in the B1 replication); the `RegisterJob` regression explained without an extra fsync. |
| R3 | §6, §9.3 | 5,000-5,300 COMMITs/s called the achieved rate; the bound left unidentified, narrowed by the new W sweep and CPU/disk accounting. |
| R4 | §3, §9.3 | Per-hop basis for both tree and chain (per-level kept for reference, labelled). The absolute latency's dependence disclosed: 134-207 ms per run, up to about 400 ms. The two proposed causes tested and rejected (`b2gap`). B2 replicated, bisected and metered: the gap follows host phases, partly the disk's flush latency and partly unidentified, not the code path. |
| R5 | §4, §9.5 | The terminal-vs-`CleanupReady` claim restricted to the failure case (the cancel holds only in the rerun), the mechanism labelled a hypothesis; one DB transaction per abort measured; "applied" worded as seen by the following probe, with T2/T3/T7 cited; the report mode's effect on cleanup quiescence measured; the race case added. |
| R6 | §2 | The slow mode described (count, period, aliasing onto the A/A processes); the extra trials disclosed as added after looking; the two harness builds disclosed; the C = 1 lifecycle figure qualified; a counterbalanced replication added. |
| R7 | §5, §9.1 | A/A band widths beside "indistinguishable"; `b4diag` disclosed as not reproducing B4's number; the emulation comparison labelled cross-harness. |
| R8 | §8 | Two containers, unpaired; the difference of medians with ranges; the 14-µs figure labelled cross-run; the probe attribution labelled "by elimination". |
| R9 | §2, §3, §7, §8 | The wording fixes listed in the review. |
| S1, S2, S4, S5, S7 | §1, §2, §4, §6 | Williams-square order in the B1 replication; CPU, iowait, fsync and disk-flush accounting per interval; server settings in `env.txt`; the B3 race variant; a discarded warm-up process per container. |

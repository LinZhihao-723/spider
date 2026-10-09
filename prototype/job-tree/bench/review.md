# Job-tree prototype: benchmark methodology review (B1-B7)

| | |
|---|---|
| Reviewer | benchmark methodology reviewer |
| Date | 2026-10-09 |
| Scope | `bench/results.md` (benchmark engineer), the harness `claude/benchmarks/job-tree-proto/` (`src/*.rs`, `analyze.py`, `run-all.sh`, `run-followups.sh`, `build.sh`, `db-up.sh`), the raw data in `results/` (all five runs), against `../brief.md` B1-B7 and `../plan.md` §5-6 |
| Method | Nothing was run against storage or MariaDB. The per-trial values below were recomputed from the CSVs with a read-only script; harness paths are relative to `claude/benchmarks/job-tree-proto/` |

## 0. Verdict

**Not sound as written, but no rerun is needed.** The data are trustworthy: identical binaries by hash, identical DB setup and durability, an idle gate before every storage process, A/A controls, distribution-free intervals, and correctness checks that ran and passed everywhere. The large structural results stand and can be used now:

- sibling registrations are serialized: 67 DB transactions for a 127-job tree, and the depth-7 tree takes 7.1x [6.75, 7.37] the time of the same shape in one job (B2);
- the pre-sized arrival buffer costs 10.4 KB per live tree, and boxing saves 735 B (B6);
- group commit works: 31.2 and 200 events per DB transaction at 64 and 1,000 reporters (B4);
- a whole-tree abort is acknowledged in 4-8 ms (B3);
- recovery is 1.4-1.5x main per plain job, plus about 0.4 ms per `BlockingOnChildren` parent (B7);
- `RegisterJob` is 25-38 % slower (B1).

Several statements go beyond what the data show, and one rests on a misread counter (§2). Each fix is a rewording or a recomputation from the existing CSVs. Two fixes can optionally be settled by a short extra run instead of a disclosure: the B2/B5 gap (R4) and the B3 race case (R5).

## 1. Checklist

| Item | Status | Evidence |
|---|---|---|
| Binaries and provenance | OK | `bin/*.provenance`, and the same four binary hashes in every run's `env.txt` (`670e995f…` main, `ae449d78…` proto). main is built from `278d06f` with 0 dirty tracked files. The proto variants share the `components/` diff hash `d4cda1b5a3428faa`. |
| Same harness within each comparison | OK, disclose | Every A/B pair runs inside one trial of one harness build. But the harness changed between runs: `full-20261009` used build `8283d6fb`, and the four follow-ups used `32287980` (the current source). `main.rs`, `em.rs` and `exp.rs` were edited at 11:56, 12:04 and 12:17, during and after the full run (11:17-12:16). The merged B1 data (trials 0-9 and 10-19) therefore span two builds. B1 uses no EM workers, so the `JT_REPORT_SPAWN` change does not reach it, but results.md does not say so (R6). |
| Same DB setup and durability | OK | `mariadb:10.11.16` (same image id in every `env.txt`), default settings, a fresh container per group, a fresh database per storage process, the bridge IP with no docker-proxy hop (`run-all.sh`, `db.rs:create_db`). Both binaries get the same YAML (`proc.rs:yaml`), `RUST_LOG=warn`, and `task_capacity = 1,048,576` in B6/B7. The durability settings are defaults and are not recorded (recommendation S4). |
| Idle gate | OK as specified, limited | Recomputed from the GATE rows: 333 gates, 0 failed, longest wait 22 s (`b6` proto-unboxed t8), background load at most 0.38 cores, matching §1. Limits: the gate excludes the MariaDB container's CPU from `bg_cores` (`host.rs:117`), cannot see I/O stalls, and runs only at the trial boundary (`host.rs:174`). Nothing is monitored during a measured interval. §2 R6 shows a periodic 2x slow mode that the gate did not catch. |
| kind cluster | OK | `Exited (137)` in every `env.txt`. It was still `Exited (137)` (15 hours) at review time. |
| A/A controls | OK, placement biased | B1 (`main-aa`, `proto-aa`), B2 (`main-aa`), B4 (`proto-aa`). The A/A variant always runs after the variants it controls: third and fourth of four processes in B1, third of three in B2 and B4 (`exp.rs:243-247`, `404-408`, `641-645`). Position-dependent noise therefore lands on the A/A ratio but not on proto/main, and in B1 it did (R6). |
| Statistics | OK | The order-statistic CI ranks are correct (`analyze.py:ci_ranks`): [x(2), x(9)] at n = 10 has 97.9 % coverage, [x(6), x(15)] at n = 20 has 95.9 %. Ratios are paired by trial. A p99 is reported only with at least 200 samples per trial. The verdict rule (ratio CI outside the mirrored A/A band) is a difference test. It cannot establish "the same" (R1). |
| Correctness assertions ran | OK | `assertion_failure` count is 0 in every `*.jsonl` of all five runs; no `stray` records (B3). All 263 SIGTERM-stopped processes exited 0. The 70 restarted B7 processes are checked by `out.check` (`exp.rs:1092`), and `b7.jsonl` is empty in both B7 runs. Tree trials assert per-tree row counts (`exp.rs:474`, `569`, `610`, `760`). main never receives `children`: tree and chain workloads are skipped on main (`exp.rs:428`), and B3, B5 and B7's tree variant run on proto only. |
| CSV integrity | OK | The only duplicate keys are B7's GATE rows: the gate key `(variant, trial)` repeats across N, and the two fan-out invocations append to one CSV. All of those gates passed, so `analyze.py` dropped nothing. |

## 2. Required fixes

### R1. B1 does not show that the plain-job path "costs what today's path costs"; H11 is misquoted

- results.md §2 and Conclusion 1 call the last-report p50 "inside the A/A band" and answer the design's gates H11/H12 with "yes". The point estimates are inside the band, but the paired CIs are not:
  - C = 1: 1.030 [0.976, **1.121**] against the band [0.957, 1.043];
  - C = 64: 1.020 [0.965, **1.053**] against [0.951, 1.049].
- The data therefore show no detectable regression; they do not exclude one of up to +12 % at C = 1 or +5 % at C = 64. The verdict rule detects differences; it cannot certify equality.
- §2 paraphrases H11 as "the plain-job p50 within the A/A band". The design's H11 (design §15) is "the A1 p50 **and the A3 throughput**". That throughput half is undecided:
  - B1 jobs/s is 0.939 [0.897, 0.966] and 0.907 [0.883, 0.959]; both upper bounds sit within 0.005 of the band edge, and both CIs lie wholly below 1;
  - the A3 analogue, B4's 1,000 plain jobs released together, is inside A/A bands about ±25-35 % wide, which cannot detect anything smaller (R7).
- **Fix:**
  - State H12 as "point estimate inside the band; the CI does not exclude +12 % at C = 1".
  - Quote H11 in full, and report its throughput half as undecided.
  - Replace "as fast as" / "costs what today's path costs" with "no regression detected" plus the CI upper bounds.

### R2. "DB commits" counts COMMIT statements, not DB transactions; main's `RegisterJob` is an uncounted autocommit transaction

- MariaDB's `Com_commit` counts COMMIT statements; an autocommit statement does not increment it. main's `register` is a single autocommit `INSERT` (`components/spider-storage/src/db/mariadb.rs:108-139`), which is still a durable transaction with an fsync under `innodb_flush_log_at_trx_commit = 1`.
- So the B2 column "DB commits (main / proto) 2 / 3" and the B2 bullet "the prototype commits 3 DB transactions per job instead of 2" are wrong: both binaries commit 3 durable transactions per job.
- What the prototype adds to `RegisterJob` is an explicit BEGIN, the fence `SELECT … LOCK IN SHARE MODE` round trip, the COMMIT statement, and the channel and committer creation. It does not add an fsync. §2's explanation of the `RegisterJob` regression should say this.
- For the prototype alone, `Com_commit` does equal its DB transactions: every runtime `jobs` write in the worktree runs in an explicit transaction. So the B4 and B5 per-tree counts and events per transaction stand.
- **Fix:**
  - Rename the metric "COMMIT statements" wherever main appears.
  - Correct the 3-vs-2 sentence in §3 and Conclusion 1.
  - Say in §1 that `Com_commit` does not count autocommit statements. Use `Handler_commit` or `Innodb_os_log_fsyncs` if durable commits ever need counting on main.

### R3. B5: the bottleneck attribution is contradicted by the data

- §6 and Conclusion 3 say per-tree latency grows 5.6x at K = 64 "because all trees share the DB's commit throughput", at "about 5,300 DB transactions/s". They also conclude that the number of DB transactions per tree, not the pool, decides how many trees can run at once.
- The DB is not shown to be saturated at 5,300/s:
  - the same server sustained 6,608 [5,564, 6,977] commits/s at K = 64 with pool 32 in the full run (`full-20261009/b5.csv`, 109 transactions per tree);
  - it sustained about 10,700/s for main's 1,000 plain jobs in B4 (1,000 commits in 93.3 ms).
- The harness is a confounder: 64 fixed EM workers serve 8,128 tasks at K = 64. Changing only the EM's report mode moved the K = 64 per-tree p50 from 1,110 ms to 793 ms (§6 sensitivity).
- Pool invariance (32 vs 64) is consistent with a DB bound, a harness bound and a storage-CPU bound alike. No storage, DB or harness CPU, fsync rate or worker utilization was recorded during B5.
- **Fix:** Call 5,300/s the achieved rate, not the DB's capacity. Say the bound is not identified (the DB, the harness EM's 64 workers or storage CPU). Keep "per-tree latency scales with K and is pool-independent" and "fair across trees". Optionally (S2), rerun K = 64 with W = 128/256 and record `host::usage` per interval.

### R4. B2: the absolute depth-7 numbers are not stable across experiments, and the per-level overhead mixes two definitions

- **Unexplained gap.** The identical workload gives two different numbers in the same `em-async-20261009` run:
  - the workload: a depth-7 no-op tree from `tree_plan(…, 7, …, leaf_noop)`, 64 workers, spawned reports, pool 64, the same proto binary and harness build;
  - B2 measures 193.8 [187.5, 198.0] ms (trials 1-9: 187-198);
  - B5 at K = 1 measures 142.0 [139.1, 159.6] ms with pool 64 and 145.6 ms with pool 32;
  - both have 67 DB transactions per tree, so B5 spends about 2.1 ms per transaction and B2 about 2.9 ms;
  - the full run shows the same direction: about 199 vs 158-167 ms.
- **Candidate causes**, neither tested:
  - B5 runs its 5-ms `PROCESSLIST`/`SHOW GLOBAL STATUS` probe during the interval (`exp.rs:812`, `db.rs:94`). A periodic poller keeps MariaDB and the vCPUs awake and can shorten commit latency.
  - B2 interleaves nine other workloads in the same process, while B5 runs consecutive trees.
- The ratio to bin127 is paired within one process and is less exposed. But the headline "194 vs 27 ms" and the per-level overhead in ms carry roughly ±25 % of this unexplained environment dependence.
- **Normalization.** The tree's per-level overhead divides by 7 (`analyze.py:199-200`, as plan §5 specified); the chain divides by 6 (`analyze.py:214-215`). §3 and Conclusion 3 then compare "23.7 ms per level" with "1.8-3.3 ms per level". The depth-7 tree has 6 job-nesting hops, like the 7-job chain. On the per-hop basis the tree costs 27.70 [26.60, 28.53] ms (no-op) and 22.54 [21.09, 22.99] ms (10 ms) per hop, recomputed from `em-async-20261009/b2.csv`.
- **Fix:**
  - Report both bases or switch to per hop, and label which one each number uses.
  - Disclose the B2/B5 gap next to the B2 headline, and give the depth-7 tree as 142-194 ms depending on the experiment.
  - Optionally (S3), settle it with a B2 tree run with the probe on (or B5 K = 1 with it off).

### R5. B3: one comparison is unsupported, one claim is not measured, one claim about the harness is wrong, and the race case is absent

- **The cancel comparison is unsupported.** "Terminal targets cost more than `CleanupReady` targets (cancel 5.30 vs 4.40 ms …)". The paired per-trial difference for cancel is 1.26 [−0.79, 2.11] ms and is positive in 8 of 10 trials: the CI includes 0. For failure it holds: 2.58 [0.40, 4.48] ms, positive in 10 of 10. The mechanism offered (`cancel_non_terminal`/`finalize`, GC enqueue) is a hypothesis. Restrict the claim to the failure case, and label the mechanism as a hypothesis.
- **One transaction is not measured.** "A whole-tree cancel or failure abort is one tree state transaction" is stated as a finding, but B3 reads no `Com_commit` (`exp.rs:526-628`). Cite the test that verifies it, or say it comes from the design. A `Com_commit` delta around the operation would measure it.
- **"Applied at the acknowledgement" is inferred.** "Durable and applied at the acknowledgement" comes from the design (§7.2): the DB probe and `GetJobState` run after the reply. The data show it was applied by the probe that follows the ack. Word it that way, or cite T2/T3/T7.
- **The report mode does affect B3.** §10 says "B1, B3 and B4 time their reports explicitly and are unaffected" by the EM's report mode. B3's quiescence with cleanup tasks is the return of the last of 127 cleanup reports sent by the 64 workers. In the full run those workers awaited each report (`em.rs:402-411`, `JT_REPORT_SPAWN` unset), so the 21 ms includes the harness's per-worker serialization. Correct §10, and attribute the 21 ms to "the harness EM's dispatch and reports plus the `CleanupDone` rounds", not to "the cleanup tasks' own work".
- **The race case is absent.** The brief asks for "the race cases as latencies". B3 aborts a fully expanded, idle tree, so no registration or completion is in flight when the cancel or failure arrives. The latency a user sees in the T2/T3 race is the case the design bounds by one in-flight registration (cancel first, one registering parent per round), and it is not measured. Either disclose this in §4, or add a variant that cancels during expansion, for example at the first level-4 registration (S5).

### R6. B1: the noise narrative is wrong, and the C = 1 lifecycle claim exceeds the decision rule

- **The slow processes are not "one slow early trial".** §2 says the extra trials were run "because one slow early trial made the C = 64 A/A band … too wide". The per-process C = 1 last-report p50 shows a bimodal slow mode at about 2x, in 13 of 80 processes:
  - main 3.56/3.12 (t0/t1); proto 3.43/2.85 (t0/t1);
  - main-aa 3.50/3.65/3.88/3.47 (t0/t11/t15/t19);
  - proto-aa 3.79/3.88/3.70/3.58/3.78 (t0/t1/t11/t15/t19);
  - all the others are 1.56-2.12.
- **It is periodic and lined up with the trial structure.** In the extra run it hit exactly the third and fourth processes of trials 11, 15 and 19, which are the two A/A processes (`b1-extra-20261009/b1.log`). That is one burst of about 50 s every 4 trials (about 400 s). `RegisterJob` doubles with it (1.3 → 3.0-3.2 ms), which points at fsync latency: host I/O or InnoDB checkpointing. CPU noise would have tripped the gate.
- **Consequences:**
  - the A/A CI of trials 10-19 is [0.991, 2.085] at C = 1, while proto/main is clean in those trials;
  - with n = 20 the order statistics exclude these outliers, so the merged conclusions stand. The same mode explains the slow first trials of B2 and B5 in `em-async` (tree 378 ms in t0).
- **Lifecycle wording.** The Summary ("4.7-10.7 % slower") and Conclusion 1 state the C = 1 lifecycle slowdown as established. By the report's own rule it is "within noise": 1.047 [1.025, 1.093] against the band [0.970, 1.030].
- **Fix:**
  - Describe the slow mode (counts, periodicity, aliasing onto the A/A positions) instead of "one slow early trial".
  - Say the extra trials were added after looking at the first ten.
  - State that the two halves come from different harness builds, and that B1's code path does not use the changed report mode.
  - Qualify the C = 1 lifecycle figure as "+4.7 %, at the band edge".
  - Optionally, show the per-half ratios: last report p50 0.991 and 1.066, `RegisterJob` 1.242 and 1.283 at C = 1.

### R7. B4: "indistinguishable" has little power, and the registration diagnostic does not reproduce B4's number

- **Little power.** "Plain jobs released together: proto and main are indistinguishable" and Conclusion 1's "1,000 plain jobs finishing together (B4) are indistinguishable from main" rest on A/A bands of [0.803, 1.275] (64 reporters) and [0.657, 1.192] (1,000 reporters). Differences under about 25-35 % cannot show. State the band width next to the claim.
- **The diagnostic does not reproduce B4.** The 1,000-child registration "position effect" uses `b4diag` to explain B4's first recorded registration (85.9 [74.1, 100.4] ms, the second burst of the process). But the second burst in `b4diag` took 196-201 ms in all three trials, more than twice as long. The diagnostic confirms that early registrations are slow; it does not reproduce B4's magnitude, so the cause is open. Say so.
- **The emulation is a different harness.** The comparison with the emulation's 62.7 events per round (A3) crosses harnesses and reporter models. Present it as context, not as "the real code reaches half of that grouping".

### R8. B7 fan-out: the description is wrong, and the pairing is nominal

- **Two containers, not one.** §8 says "5 trials per shape in one container". `run-followups.sh` runs two `run-all.sh` invocations, and `b7-fanout-20261009/env.txt` shows two containers (host ports 40667 for fan-out 100, then 51989 for fan-out 10), run one after the other.
- **The pairing is nominal.** "Paired by trial" pairs trial i of one container with trial i of the other, run about 2 minutes apart, so the pairs share nothing. The per-parent cost is robust either way: unpaired medians give 0.406 ms, and the paired values are 0.367-0.416. Correct the description, and present the result as a difference of medians with the 5 + 5 ranges.
- **A cross-run figure.** "A tree job costs about 14 µs ((2.742 − 1.321) s / 99,990)" subtracts a time from a different run (`full-20261009`), and it leaves in the 990 × 0.4 ms of parent probes (about 10 µs without them). Label it as cross-run, or drop it.
- **Attribution.** The attribution of the 0.4 ms to `has_failed_child` holds by elimination: `repair_tree` makes no other per-parent DB call, and per-tree `create_recovered_tree` costs about 13 µs, like a plain job. Say it was not measured directly (S6).

### R9. Minor wording

- **B2: a single job is not "the same" on both binaries.** "A single job costs the same on both binaries": bin127 no-op is faster on proto in 10 of 10 trials (0.870 [0.756, 0.931]). It is within main's wide A/A band only because main's own bin127 spread is 26-40 ms. Say "not slower; bin127 no-op 0.87x, within main's wide A/A band". Note also that proto's bin127 always runs right after a tree in the same process, while main's runs after the previous repetition's 10-ms chain, so the two processes have different histories.
- **B2: the "DB commits" column needs the R2 caveat.**
- **B1: "B6's load phase agrees" is too strong.** The load-phase ratio is 1.17x [0.85, 1.28], and its CI includes 1. Write "is consistent with".
- **B6: "starting from `Vec::new()` loses nothing" is untested.** `proto-smallbuf` was measured only for RSS, never for B1 or B4 latency. Say "not measured for latency".
- **B6: the "50x" mixes an implementation defect into a design estimate.** It compares 14 KB (with the buffer defect) with the design's "a few hundred bytes". Also give the defect-free figure: 3.6 KB, about 12x the estimate. This is an RSS delta, an upper bound on live memory, because VmHWM equals VmRSS at the sample, so allocator retention is included.
- **B7: the "proto + trees / main" ratio compares populations of different sizes.** The ratio column (5.4-6.9x) compares 2N live jobs with N; this is the "different live sets" trap of `04-bench/review.md` F7. Present per-live-job times, or keep the ratio only with that label beside it.

## 3. Brief compliance, B1-B7

| | The brief asks | Measured | Gap |
|---|---|---|---|
| B1 | Last-report p50/p99 and end-to-end submit → `Succeeded`, no-op, C = 1 and 64, main vs proto | Yes, n = 20, with A/A on both binaries | Equivalence wording and H11 (R1); noise narrative (R6). The in-process `jcb_test` supplement of plan §5 was skipped, and that is disclosed. |
| B2 | Depth-7 tree (127 jobs, 64 leaves) vs (a) one job with the same binary shape and (b) one flat 64-task job; no-op and 10 ms; per-level overhead | Yes, plus chain workloads to separate nesting from siblings | Per-level normalization and the B2/B5 gap (R4); COMMIT counting (R2) |
| B3 | Race cases as latencies on the depth-7 tree: cancel ack, quiescence, leaf failure → root `Failed` and quiescent | Ack and quiescence on an idle, fully expanded tree | No racing variant (R5); the cancel comparison and the one-transaction claim (R5) |
| B4 | 1,000 children finishing at once: throughput, events per tree state transaction | Yes, at 64 and 1,000 reporters; events per transaction from `Com_commit` (a mean; the `committer-stats` histogram does not exist, as disclosed) | Low-power plain-job comparison and the diagnostic mismatch (R7) |
| B5 | 16 and 64 concurrent trees: cross-tree isolation and connection use | Connection use: yes. Isolation: only as fairness among identical trees submitted together | Bottleneck attribution (R3). No heterogeneous noisy-neighbour case, such as a large registration in one tree with a small tree's latency measured next to it (S2). The plan defines isolation as fairness, so this is a limitation to state, not a failure. |
| B6 | Memory per live tree (100k plain jobs), main vs proto, boxed vs unboxed | Yes, plus the `smallbuf` diagnostic, which isolates the 10.4 KB | ΔRSS is an upper bound on live memory (R9) |
| B7 | Restart recovery with N live trees, main vs proto | Yes: plain N on both, plus the plan's tree variant on proto | Fan-out description and the population-ratio column (R8, R9) |

## 4. The depth-7 tree against the single-job baselines

What is like for like:

- The tree and bin127 run in the same proto process, in the same repetition, and are paired by trial. bin127 on main is paired within the trial.
- Both have 127 tasks, 7 task levels and 64 leaves, the same per-task sleep (0 or 10 ms, internal nodes included: `NodeSpec::children(sleep_ms, …)`), and the same 64 workers. Both use pre-compressed graphs (no compression inside the interval).
- Both end at the last report's return, with the root `Succeeded` checked right after. The tree's per-job rows are asserted.
- bin127's tasks consume their parent task's output; the tree's children get their own 4-byte input at registration. Output flow differs, but each node stores one output in both, and the payloads are small. This does not bias the comparison.
- flat64, the brief's (b), is measured and reported. No conclusion depends on it.

What is not:

- The per-level divisor: 7 for the tree, 6 for the chain (R4).
- The absolute tree latency depends on the experiment, 142 vs 194 ms for the identical tree (R4).
- The two binaries see different process histories before bin127: proto's bin127 always follows a tree, main's follows a 10-ms chain (R9). This affects the bin127 proto/main ratio, not tree vs bin127.

None of these changes the structural conclusion. The 67 DB transactions per tree (63 single-parent registrations) and the level-5 registration latency (66 ms at the median, against 2-3 ms at L0/L1) are counts and within-run latencies. They are independent of both issues.

## 5. Recommendations (not blocking)

- **S1. Counterbalance positions.** Rotate all variants through all positions with a Latin square over 4 trials, so that position and periodic noise cannot alias onto the A/A variants (R6).
- **S2. Record load during measurement.** Record `host::usage` (storage, DB, harness and background cores) over each measured interval, plus DB fsync activity (`Innodb_os_log_fsyncs`, `Innodb_data_fsyncs` deltas). Drop or flag intervals with I/O stalls. B5 needs this to name its bound; vary W (64/128/256) as well.
- **S3. Rerun B2's tree with the probe.** Run B2's tree workload with the 5-ms probe on, or B5 K = 1 with it off, to explain the 142/194 ms gap (R4).
- **S4. Snapshot the server settings.** Append `SHOW GLOBAL VARIABLES` (at least `innodb_flush_log_at_trx_commit`, `innodb_flush_method`, `innodb_log_file_size`, `sync_binlog`, `log_bin`) to each group's `env.txt`.
- **S5. Add a racing B3 variant.** Cancel the root, or fail a leaf, while level-4/5 registrations are queued. Measure the ack against the in-flight registration's DB transaction, and read `Com_commit` around the abort to count its DB transactions.
- **S6. Measure the B7 probe directly.** Confirm the per-parent cost with a `Com_select`/`Com_stmt_prepare` delta or a timing log around `repair_tree`. Measure the `smallbuf` variant on B1 and B4 before recommending `Vec::new()` as free.
- **S7. Warm up each container.** Run one discarded storage process per fresh container before trial 0, or report trial 0 separately. The first processes on a new container or after a build were slow in B1 (`full`), B2 and B5 (`em-async`).

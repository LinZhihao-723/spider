# Job-tree prototype: adversarial review of `plan.md`

| | |
|---|---|
| Reviewer | plan reviewer (adversarial plan reviewer) |
| Date | 2026-10-08 |
| Inputs | `plan.md`, `brief.md`, `../design/per-tree-committer.md`, the worktree `/home/lzh/dev/spider-job-tree-proto` at `278d06f` |
| Verdict | **fix_needed**: 4 major issues and 11 minor ones. None of them forces a redesign. Each one can be fixed by editing the plan. |

The plan follows the design where it matters:
- a long-running committer per tree, with `recv_many`, a local buffer and a quiescent exit;
- one DB transaction per tree state transaction, including the payload-error requeue;
- cancel first, then arrival order, then one registering parent last;
- an overlay;
- no `Audit` event and no reconciler;
- `AcqRel` on `num_incomplete_tasks`.

It covers every in-scope item of the brief and maps T1-T11 and B1-B7. The problems are in the details of the collect step, test isolation in the shared `spider-db`, the restart helper and the harness setup for B6/B7.

---

## Major

### M1. A `Start` that is no longer valid poisons the service (S2, `collect_inputs`)

**Plan text.** S2, committer round step 1: "for `Start`, the ready task indices, keeping today's 'no ready tasks' `TaskGraphCorrupted` check".

**The problem.** The check runs at collect time, before `plan_start` can answer `Stale`. Today the check is only reached after `write_ready` has passed (`cache/job.rs:247-256`). Two cases reach it with a job that is no longer `Ready`:
- a `StartJob` after a committed cancel: `finalize` cleared the TCB list, so `get_all_ready_task_indices` returns an empty vector;
- a retried or duplicate `StartJob` after the first one applied: the only root task is `Running` or `Succeeded`, so no task is `Ready`.

In both cases collect returns the internal `TaskGraphCorrupted`, the round returns `Err`, the guard poisons the tree and the process exits. The client's `call_with_retry` resends UNAVAILABLE requests, so a duplicate `StartJob` is a realistic input.

**Fix.** Mirror trap R3: read the ready indices only when the collected state is `Ready`, and run the "no ready tasks" check only then. Add a test: `StartJob` twice, and `CancelJob` then `StartJob`. Both must be stale (`JobAlreadyStarted`), and `is_fatal()` must stay false.

### M2. The new tests share one database with a recovery that loads every row of it (sections 0.6, 4, S6)

**What the existing suite already does.**
- `get_recoverable_jobs` loads every non-terminal row of the shared `spider-db`.
- `resend_ready_tasks` republishes all of those rows.
- For this reason the existing suite marks 20 of its MariaDB tests `#[serial_test::file_serial]`: 10 in `mariadb_test.rs`, 6 in `jcb_test.rs` and 4 in `runtime_recovery_test.rs`. The plan says that "`mariadb_test.rs` is not serialized", which is only partly true.

**What the plan omits.** It never mentions `file_serial` or any other isolation for the new MariaDB tests. With the prototype, each restart in the shared DB also does the following:
- it repairs any tree it considers inconsistent. A forced-repair test (R1/R2/R4/R5 rows written by raw SQL) can be repaired by a concurrent T8, zombie or `runtime_recovery_test` restart in another nextest process before its own restart runs. That breaks "one restart converges and logs a repair".
- it spawns committers for, and resends the tasks of, every leftover non-terminal row:
  - other tests' live trees;
  - T2/T3/T7 iterations that leave `CleanupReady` or held `CommitReady` jobs. At 50-500 iterations these accumulate.

  `TestEm::run_until_quiescent` polls the lanes without a filter. It will receive foreign entries whose labels it cannot resolve.

**Fix.** Give every job-tree test that uses MariaDB its own scratch database. `MARIADB_ROOT_PASSWORD` is already used for the upgrade test, and the fence id then needs no special handling. Failing that, mark them `file_serial` together with `runtime_recovery_test`. In both cases `TestEm` must ignore lane entries that are not in its tree.

### M3. The in-process restart skips part of the real recovery path (section 3, "Simulating a storage restart")

The plan never runs these three parts of the real path:

1. **No `resend_ready_tasks`.** The binary calls it after serving (`bin/grpc_server.rs:98-109`), and `runtime_recovery_test.rs` calls it explicitly after every restart (lines 76, 102, 160). Steps 1-4 of the plan's restart helper omit it. As a result:
   - recovered `Running` children never publish their tasks, so `run_until_quiescent` hits its iteration cap;
   - the T8 assertion "a recovered `BlockingOnChildren` parent publishes nothing" is vacuous.
2. **`create_runtime_with_hooks` is not tied to `create_runtime`.** The plan does not say that the production `create_runtime` delegates to it. Unless it does, T8 tests a sibling of the production path.
3. **Crash during recovery leaks the old runtime.** T8 "Crash at `RecoveryBeforeRepair`/`RecoveryAfterRepair`" makes `create_runtime_with_hooks` return `Err`. By then, under the S2 reordering, these are already running:
   - the task-instance pool;
   - the GC;
   - the publisher lane tasks;
   - the committers of the trees recovered so far.

   No `Runtime` exists to stop them, so they leak across iterations, with a live token, holding connector clones.

**Fix.**
- Make `create_runtime` call `create_runtime_with_hooks(config, TestHooks::default())`, with the hooks feature-gated.
- On any error after `connect`, cancel the token and join the spawned tasks before returning.
- Make `TestEnv::restart` call `service_state.resend_ready_tasks()` after the runtime is created, in the order that `main` uses.

### M4. B6 and B7 overflow the default task lane, so main hangs and proto's RSS includes the overflow (section 5)

**The arithmetic.**
- The default `task_capacity` is 65,536 (`inbound_queue.rs:323`).
- B6 starts 100,000 plain jobs with no pollers.
- B7 needs 100,000 `Running` jobs, and its proto variant adds trees on top.

**What happens on each binary.**
- On main, `start` awaits `send_task_ready` under the JCB write lock (`cache/job.rs:258-265`). The 65,537th `StartJob` therefore blocks forever, and the trial never finishes.
- On the prototype the overflow sits in the unbounded staging queue, which inflates the per-job RSS that B6 measures.

The plan writes "the default inbound-queue capacities, or the deployed ones where stated", and B6 and B7 state none.

**Fix.** Set `task_capacity` to at least N (the deployed 1,048,576) for B6 and B7, identically on both binaries. Alternatively, have the harness run one drain-only poller that discards entries. In either case, record the lane occupancy at the time of the RSS sample.

---

## Minor

### m1. The `Pause` hook does not race the token (section 3)

`Pause` blocks "until released". `Runtime::stop` now waits for the tracker, which holds the spawned handler bodies. Two cases leave a body paused:
- a paused phase-A body, which also holds the JCB read lock and blocks the committer's apply;
- any pause left armed when a `Crash` fires.

Either one makes every restart wait for the 30 s stop timeout, and at 50-500 iterations per test that is prohibitive.

**Fix.** Make `Pause` select on the runtime token, and release every armed pause when `TestHooks` (or the `ArmedHook`) is dropped.

### m2. The upgrade issues an `ALTER` on every connect, which contradicts its own test (S1)

`upgrade_jobs_schema` runs `ALTER TABLE jobs ADD COLUMN IF NOT EXISTS …, ADD INDEX IF NOT EXISTS …` unconditionally. Two problems follow:
- the upgrade test then asserts that "the second connect issues no `ALTER`", which cannot hold;
- every connect of every concurrently running test process runs DDL against the shared `jobs` table, and that DDL can queue on metadata locks behind other processes' open transactions.

**Fix.** Read `information_schema.COLUMNS` and `STATISTICS` first, as the plan already does for the enum, and alter only what is missing.

### m3. Existing mock-based tests will fail, but the plan does not list the edits (S2, S5)

The plan's compatibility notes miss three groups of failures:
- **Hand-built JCB ids are not in the mock's state map (S2).** `cancel_job_transitions_to_terminal_state` and `cancel_job_enqueues_terminal_job_for_cache_gc` insert a hand-built JCB with `JobId::random()`. That id is not in `MockDbConnector.states`, so the faithful `apply_tree_transition` mock gets a guard miss, returns `CorruptedDbState`, and the tree is poisoned. These tests need `db.states.insert(job_id, Ready)`.
- **Two DB-fallback tests seed no state (S5).** `get_job_outputs_returns_outputs_from_db` seeds only `outputs`, and `get_job_error_returns_error_message_from_db` seeds only `errors`. The new root check calls `get_state_and_parent`, which returns `JobNotFound`. These tests need a seeded state, or the mock needs a fallback.
- **`JobCacheGcHandle::new` is `pub(crate)` (S2).** `tests/scheduling_infra.rs`, an external crate, cannot call it to build its `TreeEnv`. Making it public is a visibility change in the library, not a harness edit.

### m4. The shared root check is stricter than the design (S5)

`ensure_root_cached_or_db` applies "a non-terminal root on a cache miss is fatal" to `start_job`, `get_job_state`, `get_job_outputs` and `get_job_error`. Today only `cancel_job` has that fatal branch:
- `start_job` returns `JobNotFound` on a miss without reading the DB;
- the getters answer from the DB.

The design (§6.5, §9.6) requires only the non-root rejection for these RPCs.

**Fix.** Keep each RPC's existing miss behaviour after the parent check.

### m5. `create_runtime_with_hooks` arrives in S6, but S2 and S5 need it

S6 introduces `create_runtime_with_hooks`, but section 4's `TestEnv::start` uses it, and so do T9 (S2) and T1-T7 (S5).

**Fix.** Introduce it in S2 together with the hook infrastructure.

### m6. The `EventSubmitted` hook cannot be reached from a synchronous `submit` (section 3)

`submit` is synchronous, and design §7.2 requires that. `TestHooks::hit` is `async`, so `EventSubmitted` cannot call it.

**Fix.** Provide a synchronous, observe-only path, such as an atomic counter plus `Notify`.

### m7. T9 tests only one of the drop points that design §15 lists

T9 covers only `PhaseAAfterTcbFlip`. Design §15 asks for a dropped handler future at each await of the report path. That includes:
- inside the TCB success, at `write_outputs` and at the successor mutexes;
- between the TCB registration of an instance and its send to the pool.

**Fix.** Add a hook, or reuse an existing one, at a successor-lock await and at the pool send.

### m8. Two crash points of the design are not exercised (T8)

There is deliberately no hook between BEGIN and COMMIT (R7). As a result two crash points of the design are not exercised on the real code:
- a crash with an open DB transaction (design §11.1, the "DB transaction open, before COMMIT" row);
- a lost COMMIT acknowledgement (H17).

The S2 guard-miss rollback test covers the SQL side only. State this as a known gap in the report.

### m9. The DB probes in the harness disturb what they measure (section 5)

- **Quiescence probe.** It polls every 1 ms with `(id = ? OR root_id = ?)`, and the prototype has no `job_root` index. The probe competes with the committer's transactions during the B3 time-to-quiescence measurement.
- **Connection probe.** It reads `PROCESSLIST` every 5 ms as `spider-user`:
  - it counts its own connection;
  - it misses transactions that last about 1 ms.

**Fix.**
- Probe at a coarser interval, or derive quiescence from the harness's own replies (the last cleanup or commit report).
- Count connections as root, excluding the probe's own connection.
- Take the peak from `Max_used_connections`, or from per-trial `Threads_running` sampling, alongside the 5 ms samples.

### m10. Children on a commit or cleanup success are silently dropped (S4)

`grpc.rs` routes a success report by its task id. A non-empty `children` field on a commit or cleanup success would be dropped without an error.

**Fix.** Reject it with INVALID_ARGUMENT.

### m11. B7 recovery time includes per-job logging

Recovery logs `"Job recovered into cache."` at `info` level for every job (`state/runtime.rs:238-242`). At N = 100,000 that logging is a large part of the recovery span.

**Fix.** Run B7 with `RUST_LOG=warn` on both binaries, or keep the per-job log identical in the prototype and note it.

---

## Checked and found consistent

- **Design invariants.**
  - A long-running committer per tree, with a quiescent exit, poison on abnormal exit and token-raced waits (§7.2, §7.7, §7.8).
  - A single fenced DB transaction per tree state transaction, with UPDATE groups before INSERT chunks. The deterministic payload error is handled by requeue, not by a second DB transaction.
  - Cancel first, arrival order, and at most one registering parent, planned last. The `registers` flag (U1) keeps other registering parents uncollected, which is a sound refinement.
  - No `Audit` event and no reconciler. `AcqRel` with the `has_prepared_children` store placed before the read-modify-write.
- **Race recipes.** The "same round", "consecutive rounds" and "phase A against a committed cancel" recipes are deterministic as written. The third matches design §10.6 (b) 2, because the cancel's collect needs only a shared read lock.
- **Restart through `Crash`.** It drives the real poison and token path. A crash at `CommitterBeforeDbTransaction` or at `CommitterAfterCommit` for a registration does test the "nothing registered twice" property.
- **S1 surface.** The proto and enum changes compile with one EM struct-literal fix. No other crate matches `JobState` exhaustively, and `tests/huntsman/e2e` uses `_ =>`.
- **Harness.** B1-B5 measure what the brief asks for, with A/A controls. The protobuf compatibility analysis is right: main ignores field 7, and the harness refuses tree workloads against main.

---

Environment note: Pleiades asked for the kind cluster to be killed. When this review ran, `clp-rc2-control-plane` was already stopped (Exited 137), with no running containers. plan reviewer did not delete the cluster and did not start anything.

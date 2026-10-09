# Review: S1 `s1-schema-fence` (slice reviewer)

Reviewed the S1 changes in `/home/lzh/dev/spider-job-tree-proto` (`git diff 278d06f`), measured against the slice note `slices/01-s1-schema-fence.md`, design sections 3.4, 7.7, 7.8, 9.1, 9.2, 9.6 and 11.1–11.3, and the Rust review and style guides. This is the first slice, so no earlier slices interact with it.

Verdict: **fix_needed**. One major issue: the non-zero fatal exit is incomplete. The fence, the transition table, the proto and the upgrade are sound.

## What holds up

- `is_valid_transition` matches design §3.4. I counted the 19 edges by hand, and the 64-pair test enforces them.
- Every statement that writes `jobs` (`register`, `start`, `set_state`, `commit_outputs`, `cancel`, `fail`, `delete_expired_terminated_jobs`) runs `check_fence` as its first statement in the same DB transaction. No other `jobs` writer exists. An early `?` drops the transaction, which rolls it back.
- Fence lock order: the fence S lock is always taken before any `jobs` row lock, and the bump touches only the fence row. So the bump cannot form a wait-for cycle with old-session transactions.
- `GREATEST` keeps the fence monotone. The bump runs after the session INSERT and before `recover_job_cache` (`connect` at `db/mariadb.rs:133`).
- `Fenced` is fatal on every gRPC path. A direct DB error hits the new `Db(Fenced)` arm (`grpc.rs:439`). A cache-path DB error arrives as `Cache(Internal(Db(Fenced)))` and hits the existing cache-internal arm (fatal, though the log message says "cache layer"). `Stopping` is handled first and is never fatal, also in strict mode.
- `ShutdownHandle::fatal` stores the flag with Release before it cancels the token. `main` reads the flag with Acquire after the server task joins, and the token's internal synchronization orders the two.
- The bump's `select!` keeps the pinned UPDATE future alive across watcher ticks, so the request is never dropped mid-flight. The detached connection is closed, so `innodb_lock_wait_timeout = 5` cannot leak back into the pool.

## Issues

### 1. Major: the fatal exits of the background actors still exit 0

- **Where:**
  - `src/task_instance_pool.rs:535` and `src/state/job_cache_gc.rs:128` call the plain `cancellation_token.cancel()`, not `ShutdownHandle::fatal()`.
  - `src/state/runtime.rs:86-90` and `:100-104` (`Runtime::stop`) only log an actor's `Ok(Err(_))` or panic join result, then return `Ok(())`.
- **Scenario:**
  1. MariaDB blips.
  2. `TaskInstancePool::handle_message` gets a `DbError` from `is_execution_manager_alive` (`task_instance_pool.rs:317` via `?`). The actor returns `Err`, logs "Cancelling the service." and cancels the plain token.
  3. tonic shuts down gracefully, `runtime.is_fatal()` is false, and `stop()` logs "Task instance pool exited on error." and returns `Ok`.
  4. `wait_for_shutdown` returns `ExitCode::SUCCESS`. Compose's default `restart: on-failure:3` (`compose.yaml:12`) leaves storage down.
- **Same class as the one S1 fixed:** the slice note's deviation 4 converted the startup `resend_ready_tasks` failure from exit 0. These two sites have the same problem and were missed. The brief's "non-zero exit on the fatal path" and design §7.7's rule ("any exit … other than a quiescent exit or a shutdown … exits non-zero") are not met.
- **Fix:** pass the `ShutdownHandle` (or its fatal flag) into both actor factories and call `.fatal()` on their error exits. Have `Runtime::stop` return an error, or report failure, when an actor joined with `Err` or panicked. Add a test: an actor error leads to a non-zero exit.

### 2. Minor: `fence_id` is production configuration, and a mismatch silently disables the fence

- **Where:** `src/config.rs:52`, `db/mariadb.rs:119-125`, `166-189`.
- **Problem:** the design has one fence row (`id = 0`) per database. Making the id a deserialized config field, documented as "services with different fence IDs are independent", lets two storage processes that point at the same DB with different ids both pass their fence checks.
- **Scenario:** a rollout whose new ConfigMap sets `fence_id: 1` while the old pod runs with the default 0.
  1. Under RollingUpdate, both processes serve.
  2. Neither is ever `Fenced`.
  3. Both recover and write the same `jobs` rows, which is the stuck-cache divergence that §11.3 exists to prevent.
- **Fix:** keep the per-test isolation out of the production surface, for example `#[cfg(feature = "test-hooks")]` on the field, `#[serde(skip)]` with a test-only setter, or a scratch DB per fence test. Or at least document that the value must never change across a deployment's lifetime.

### 3. Minor: the upgrade DDL runs before the fence bump and its kill watcher, so an orphan can stall the first start for hours

- **Where:** `db/mariadb.rs:107-109` (the call in `connect`) and `upgrade_jobs_schema` at `:1129-1197`.
- **Problem:** `ALTER TABLE jobs ADD …` and `MODIFY state` need an exclusive MDL on `jobs`.
- **Scenario:** on the first start of the new binary, a predecessor (old binary, no fence, server `wait_timeout` 28,800 s) died or was partitioned inside a transaction that touched `jobs`.
  - Its orphaned connection holds a shared MDL on `jobs`.
  - The new process's ALTER waits on it for up to `lock_wait_timeout` (MariaDB default 86,400 s). The 1 s kill watcher, which exists only inside `bump_fence`, never runs.
  - If the old process is still alive, its new statements on `jobs` queue behind the pending exclusive MDL, so it stalls too.
- **Impact:** this breaks §11.3's "the new process starts within seconds" for the upgrade start. It happens once per installation, and the design shares the ordering, so record it as a trap for the report.
- **Fix:** set a session `lock_wait_timeout` for the DDL and retry, or run the upgrade under the same watcher, keyed on MDL waits (`information_schema.METADATA_LOCK_INFO` / `PROCESSLIST`).

### 4. Minor: `SHUTDOWN_DEADLINE` does not bound `runtime.stop()`

- **Where:** `src/bin/grpc_server.rs:166-196`.
- **Problem:** the deadline races only `server`. Once the server finishes at, say, 9.9 s, `runtime.stop()` can take its full 30 s `stop_timeout`, for example with an actor stuck in a DB call. A fatal exit can therefore take about 40 s.
- **Design gap:** §7.7's rule and the §11.1 fatal row promise an exit within `SHUTDOWN_DEADLINE`.
- **Fix:** race `stop()` against the remaining deadline too, or record the 40 s bound as a deviation.

### 5. Minor: a non-empty `children` field is silently dropped

- **Where:** `src/grpc.rs:665-670`.
- **Problem:** `report_task_success` unpacks the request without looking at the new `children` field (proto field 7).
- **Scenario:** a client that sets `children` today gets its parent job `Succeeded` with no children, which is silent data loss.
- **Design gap:** §9.6 says to reject a non-empty `children` with `INVALID_ARGUMENT` while the feature is off. Until registration lands (S3/S4), reject it.

### 6. Minor (tests): the fence checks for four of the seven writers are unguarded

- **Where:** `tests/fence_test.rs:38-63`.
- **Problem:** only `register`, `start` and `fail` are asserted. The suite still passes after removing `check_fence` from `set_state` (`mariadb.rs:334`), `commit_outputs` (`:365`), `cancel` (`:402`) or `delete_expired_terminated_jobs` (`:459`). The slice's mutation check covered only `start`.
- **Fix:** extend the loop to all seven writers.

### 7. Minor (tests): the tests do not exercise the properties they rely on

- `connect_waits_for_transactions_that_passed_the_fence` (`fence_test.rs:83-106`):
  - It asserts a negative with a 500 ms timer. A slow `connect` (shared `spider-db`, CI load) passes it with no lock wait at all.
  - It does not check the property the design's correctness argument rests on: an old-session transaction that *starts* while the X request waits must queue behind it and then fail `Fenced`, so that a busy zombie cannot starve the bump.
  - Fix: start a second `older.start(..)` after the bump is queued, then assert that it returns `Fenced` once the holder commits.
- `fenced_storage_server_exits_with_failure` (`exit_code_test.rs:134-169`):
  - The only client is a temporary that is dropped before the exit wait, so tonic's graceful shutdown finishes at once.
  - The `SHUTDOWN_DEADLINE` arm and `shutdown_background()` are never exercised.
  - Fix: keep a client connection open, for example with a pending streaming call or an idle HTTP/2 channel, and assert the exit within 11 s.

### 8. Minor (tests): `ScratchDatabase` leaks its database on failure

- **Where:** `tests/mariadb_infra.rs:13-94`.
- **Problem:** `drop_database` is an explicit async call with no `Drop` guard. Any `?` or `assert!` failure before it leaks a `jt_*` database, for example `exit_code_test.rs:147` and `:156-164`, and `schema_upgrade_test.rs:116-141`. The leaks accumulate in the shared container across runs, which is the kind of row and DB build-up already seen to cause R20-style flakiness.
- **Fix:** use a guard that drops the database on a blocking runtime, or have tests clean up stale `jt_*` databases.

### 9. Minor (style)

- `db/mariadb.rs:1285`: `let _ = connection.close().await;` discards a fallible result with no justification. The review guide's rule is "No bare `let _ = fallible().await`".
- `state/service.rs:136` `ServiceState::shutdown()` and `state/shutdown.rs:51` `ShutdownHandle::token()` have no `# Returns` docstring, although their siblings (`session_id`, `is_fatal`, `is_cancelled`) have one.
- `ShutdownHandle::token()` and `is_cancelled()` have no caller in S1. They are dead API until S2 or later uses them.

## Suggested PR title (for the eventual formal PR)

`feat(spider-storage)!: Add the BlockingOnChildren job state, the storage session fence, and a non-zero exit on fatal shutdown.`

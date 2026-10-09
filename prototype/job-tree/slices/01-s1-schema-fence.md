# Slice S1 `s1-schema-fence`: notes

Role: implementer. Worktree: `/home/lzh/dev/spider-job-tree-proto` (`git diff 278d06f`). The worktree was empty before this slice.

## What changed

- `spider-core/src/job.rs`: `JobState::BlockingOnChildren` is now the last variant. `is_valid_transition` follows design §3.4, including the new `CommitReady → Cancelled` edge. The unit tests check all 64 `(from, to)` pairs against an explicit list of the 19 valid edges.
- `storage.proto`: `BLOCKING_ON_CHILDREN = 8`, `message ChildJobRequest`, and `repeated ChildJobRequest children = 7` on `ReportTaskSuccessRequest`. `src/generated/storage.rs` was regenerated. Its diff contains only these additions, plus one side effect: prost drops `Eq, Hash` from the `ReportTaskSuccessRequest` derive, because the message now has a repeated message field. Nothing used them.
- `spider-proto-rust/src/job.rs`: adds the arms for both conversions and a round-trip test.
- EM `client/grpc/storage.rs`: `children: Vec::new()`. This is a compile fix only.
- `DatabaseConfig.fence_id: u64` (`#[serde(skip)]` after review: configuration files cannot set it, so every deployed service uses 0; only tests set it), and `DbError::Fenced { mine, current }`.
- `db/mariadb.rs`:
  - The pool's `after_connect` sets `wait_timeout = 900`.
  - `jobs` DDL adds `parent_id`, `root_id` and `INDEX job_parent`. The columns are appended after `num_retries`, so new installs and upgraded tables end up with the same column order.
  - `upgrade_jobs_schema` reads `information_schema.COLUMNS` and `STATISTICS` first, then runs at most one `ALTER … ADD … IF NOT EXISTS` with only the missing clauses, and at most one `MODIFY state`. It returns the number of statements it ran.
  - `storage_fence` (`id BIGINT UNSIGNED PK`, `session_id`) plus `INSERT IGNORE (fence_id, 0)`.
  - `bump_fence` runs on a dedicated, detached pool connection. The connection is closed afterwards, so `innodb_lock_wait_timeout = 5` never leaks back into the pool. It runs a `GREATEST` UPDATE, retries up to 5 times on 1205, and uses a 1 s ticker watcher. The watcher joins `INNODB_LOCK_WAITS`/`INNODB_TRX` and runs `KILL CONNECTION` (through `raw_sql`, as text protocol) on each blocker. On 1227/1142 it warns once and stops.
  - `check_fence` (`LOCK IN SHARE MODE`) is a method on the connector. A missing row is `CorruptedDbState`.
  - `register` is now BEGIN, fence, INSERT, COMMIT. `start`, `set_state`, `commit_outputs`, `cancel`, `fail` and `delete_expired_terminated_jobs` run the fence check as their first statement.
  - `#[cfg(feature = "test-hooks")] schema_ddl_statements_at_connect()`.
- `Cargo.toml`: `[features] test-hooks = []`. S2 adds the hook module to this feature.
- `state/shutdown.rs`: `ShutdownHandle { token, fatal }` with `fatal()` (store with Release, then cancel), `is_fatal()` and `token()`. After review, `create_runtime` builds one handle and passes it to `ServiceStateParams.shutdown` (replacing `cancellation_token`), `create_task_instance_pool` and `create_job_cache_gc`. `ServiceState::shutdown()` returns it. The scheduler-registration resend failure and both actors' error exits call `.fatal()`. Also added: `Runtime::is_fatal()`.
- `grpc.rs`:
  - The fatal arms call `self.inner.shutdown().fatal()`.
  - A new arm makes `Db(Fenced)` fatal in every handler. The non-strict handlers reach it too, because no specific arm matches `Fenced`.
  - `Stopping` maps to `UNAVAILABLE` and is never fatal. It is handled at the top of `default_error_handler`, which every handler falls through to. Every handler's docstring lists `UNAVAILABLE`.
- `bin/grpc_server.rs`:
  - `main() -> ExitCode` uses a hand-built multi-thread runtime and calls `shutdown_background()` after `block_on`, so an expired deadline exits at once and does not wait for blocking-pool threads.
  - `run()` handles startup. `wait_for_shutdown()` races `server` against `token.cancelled()` followed by `SHUTDOWN_DEADLINE` (10 s).
  - The exit code is FAILURE if the deadline expires, the server errors or panics, `runtime.stop()` fails, or `runtime.is_fatal()`. `is_fatal()` is read just before `stop()` consumes the runtime.
- Tests (`tests/`):
  - `mariadb_infra.rs`: `create_mariadb_config` sets `fence_id: rand::random()`. New `ScratchDatabase::{create, config, drop_database}`, which connects as root with `MARIADB_ROOT_PASSWORD`, falling back to `SPIDER_STORAGE_DB_PASSWORD`. New `mariadb_root_password()` and `create_raw_pool(&config)`.
  - `fence_test.rs` (shared `spider-db`, no `file_serial`):
    - A newer connector with the same id fences `register`, `start` and `fail` of the older one, with the exact `mine` and `current`.
    - Connectors with different ids are independent.
    - A raw S-lock holder delays `connect` (not done after 500 ms; done within 5 s after COMMIT).
    - Extra: with root credentials, which have `PROCESS`, `connect` kills an orphaned S-lock holder within 4 s, and the orphan's COMMIT fails. This checks the watcher SQL for real.
  - `schema_upgrade_test.rs` (`#[cfg(feature = "test-hooks")]`, scratch DB):
    - The pre-tree DDL with one row, then two connects. The test checks the columns, the index, the exact enum type, the fence row (the first session, then the second), the old row's NULL `parent_id`/`root_id`, and the DDL count 2, then 0.
    - Extra: a connect to a new DB runs 0 upgrade DDL statements.
  - `exit_code_test.rs` (scratch DB): runs two `spider-storage` binaries with the same config, then sends `RegisterJob` to the first. The reply is `INTERNAL`, and the first binary exits non-zero within 11 s.

## Verification

- `cargo build --workspace --all-targets --all-features`: OK.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`: clean.
- `cargo clippy -p spider-storage --all-targets -- -D warnings` (feature off): clean.
- `cargo +nightly fmt --all -- --check`: clean.
- `cargo nextest run -p spider-storage -p spider-core -p spider-proto-rust --all-features --run-ignored all --release` on a fresh MariaDB container: **275/275 passed**.
- Mutation checks, each one applied, run, then reverted:
  - Removing the `Fenced` fatal call makes the exit-code test fail ("the fenced server should exit").
  - A `fatal()` that does not set the flag makes it fail (exit status 0).
  - Removing the fence check from `start` makes `newer_connector_fences_off_…` fail ("expected Fenced, got Ok(())").
- Pre-existing flakiness, also seen on the baseline checkout (`/home/lzh/dev/spider`, 278d06f) against the same container. It is not caused by this slice:
  - The timing-based `mariadb_test` tests `test_get_dead_execution_managers_*`, `test_is_execution_manager_alive_em_dead` and `test_delete_expired_terminated_jobs` fail in about 1 run in 4 on both trees. They use `CURRENT_TIMESTAMP` with a 1 s threshold.
  - `runtime_recovery_test::restarted_storage_cache_recovers_{running_job_from_start,commit_ready_job}` fail on both trees once `spider-db` has collected rows from earlier runs (R20). They pass on a fresh database.

## Deviations from the plan and the design

1. **`max_statement_payload` is deferred to S2.** Nothing reads it in S1, so it would be a dead field (a warning). S2 should add it to `MariaDbStorageConnector` (`min(4 MiB, @@max_allowed_packet / 2)`, read once in `connect`).
2. **`GrpcServiceState::new` no longer takes a `CancellationToken`.** Every fatal path goes through `ServiceState::shutdown()`, so the field was dead. Its only caller is `bin/grpc_server.rs`.
3. **`storage_fence.id` is `BIGINT UNSIGNED`**, not the design's `TINYINT`, to hold a random `fence_id` (plan §0 item 6).
4. **The `resend_ready_tasks` failure at startup in `main` now calls `fatal()`**, not a plain cancel. It was a silent exit 0 before.
5. The watcher treats lookup errors other than 1227/1142 as transient: it warns and retries on the next tick. A failed `KILL` (for example 1094, the blocker is already gone) is logged and skipped.
6. The extra fence test and the extra upgrade test go beyond the plan's list.

## For the next slice

- `ServiceState::shutdown() -> &ShutdownHandle` is the fatal path to use: the committer guard, `TestEnv`, and `Runtime::is_fatal()` for the M1 stale-start test.
- `ScratchDatabase::create().await`, then `.config()` (random `fence_id`, `max_connections: 5`), then `.drop_database().await`. Two runtimes built from one `config()` share a fence id and fence each other, as the restart tests need.
- `check_fence(&self, &mut MySqlConnection)` is the statement that `apply_tree_transition` should start with. In S2 it replaces `fetch_job_state_for_update` in the wrappers' write transaction.
- The `test-hooks` feature exists but is empty except for `schema_ddl_statements_at_connect`. `schema_upgrade_test` is gated on it in `tests/test_spider_storage.rs`.
- Regenerating the proto with the system `protoc` 3.12.4 needs `--experimental_allow_proto3_optional`. Use a wrapper through `PROTOC=<wrapper>`, for example `/tmp/claude-1000/-home-lzh-dev-spider/d243e5f9-8757-4d79-9a35-b2d494133cd2/scratchpad/bin/protoc`. Running the build script without it deletes `src/generated/` before it fails; restore it with `git show 278d06f:<path>`.
- `.rustfmt.toml` was copied into the worktree root from the main checkout so that `cargo +nightly fmt` uses the project config. It is gitignored, and the `yscope-dev-utils` submodule is not checked out in the worktree.
- Non-test lines: about 640 added, against a rough estimate of 300. Most of the excess is docstrings and multi-line SQL. The bulk is in `mariadb.rs` (+378/−4: the fence, the watcher and the upgrade) and `grpc_server.rs` (+107/−28).

## Review fixes (slice review round 1, `reviews/01-s1-schema-fence-review.md`)

All nine issues are fixed. Fixing issue 7 also exposed a new bug in the fence watcher, listed first.

0. **New, found by the issue 7 tests: the fence watcher killed unrelated connections.**
   - Cause: MariaDB (13.0.2 in the test container) reports `trx_id = 0` for every transaction that holds only shared locks. When the bump waits on such a holder (the design's orphaned `check_fence` case), `b.trx_id = w.blocking_trx_id` matched every read-only transaction on the server, in any database, and `KILL CONNECTION` hit them all. It showed up when the privileged watcher test ran in parallel with the new tests: their lock holders were killed.
   - Fix: `kill_fence_blockers` now joins `PROCESSLIST` and keeps only blockers whose connection uses `DATABASE()` and whose transaction started no later than the bump's `trx_wait_started`. It also skips its own connection and uses `DISTINCT`. The bump UPDATE always has a real trx ID while it waits, so the requester side of the join is unique. Residual risk, documented in the docstring: a shared-lock-only transaction in the same database that started before the wait can still be killed. In practice that is a transaction of the session being fenced off.
   - Test: `connect_kills_an_orphaned_fence_holder_when_privileged` now runs on a `ScratchDatabase`. It also holds an unrelated `LOCK IN SHARE MODE` transaction in `spider-db` and asserts that this transaction still commits. Mutation check: with the old query, the test fails ("expected to read 4 bytes, got 0 bytes at EOF": the unrelated connection was killed).
1. **Major, actor fatal exits.** `create_task_instance_pool` and `create_job_cache_gc` take a `ShutdownHandle` and call `.fatal()` when their actor returns `Err`. `Runtime::stop` returns `StorageServerError::Stopping` when an actor joined with `Err` or panicked, as well as on timeout. `wait_for_shutdown` now reads `shutdown.is_fatal()` after `stop()`, so a fatal call made during the stop also counts. Tests:
   - `spawned_pool_shuts_the_service_down_fatally_on_error`: a liveness store that returns `DbError` makes the pool return `Err` and sets the fatal flag.
   - `stop_runtime_on_task_error` now expects `Err(Stopping)`. The new `stop_runtime_on_task_panic` covers the panic arm.
   - Either result makes the binary exit with FAILURE. The binary's mapping of `is_fatal` to FAILURE is already covered by `fenced_storage_server_exits_with_failure`.
2. **`fence_id` in production config.** It is now `#[serde(skip)]`, and the docstring says that deployed services always use 0. `exit_code_test` no longer writes it to the YAML. The deadline test pins `fence_id: 0` so that its raw lock hits the binary's row.
3. **Upgrade DDL behind an orphan.** `upgrade_jobs_schema` runs its DDL on a detached connection with `SET SESSION lock_wait_timeout = 5`. `execute_ddl_with_lock_wait_retries` retries on 1205 with no cap and logs a warning on each attempt. The upgrade still waits for the orphan to go away, because killing MDL holders would need `METADATA_LOCK_INFO`, which is a plugin. But other statements on `jobs` no longer queue behind the pending exclusive MDL for more than 5 s at a time, and the stall shows up in the log.
   - Test: `upgrade_waits_for_an_open_transaction_without_blocking_the_table`. An open transaction reads `jobs`, and connect is pending. A third `SELECT` on `jobs` completes within 7 s, and connect finishes after the commit with 2 DDL statements.
   - Mutation check: with `lock_wait_timeout = 86400`, the test fails.
   - **Trap for the report:** the first start of the new binary can still wait as long as the orphan lives (old binary: `wait_timeout` 28,800 s). The design shares the ordering.
4. **`SHUTDOWN_DEADLINE` now bounds `stop()` too.** One pinned deadline future (token cancelled, then 10 s) is raced first against the server and then against `runtime.stop()`.
5. **Non-empty `children`.** `report_task_success` returns `INVALID_ARGUMENT` ("child jobs are not supported") before it unpacks. Unit test `grpc::tests::report_task_success_rejects_children` uses the mock service state. `state::test_utils` is now `pub(crate)` so that `grpc.rs` can use it.
6. **All seven writers are now guarded.** `newer_connector_fences_off_…` asserts `Fenced` for `register`, `start`, `set_state`, `commit_outputs`, `cancel`, `fail` and `delete_expired_terminated_jobs`. Mutation check: removing the fence check from `set_state` fails the test.
7. **Tests now exercise the properties they rely on.**
   - `connect_waits_for_transactions_that_passed_the_fence`:
     - The 500 ms negative timer is gone. `wait_for_queued_fence_bump` probes the fence row with `LOCK IN SHARE MODE NOWAIT` until the probe gets 1205, which means an X request is queued.
     - Then the test starts `older.start(job_id)`, commits the holder, and asserts that the late `start` returns `Fenced` with the exact session IDs. So a transaction that starts while the bump waits queues behind the bump.
     - The first version used a second timer, and it failed once under full-suite load. The probe made it stable: 5 of 5 runs passed in a row.
   - New `storage_server_exits_with_failure_when_a_request_outlives_the_shutdown_deadline` (instead of reworking the fenced test):
     - A raw transaction holds `FOR UPDATE` on fence row 0, so an in-flight `RegisterJob` blocks in `check_fence`. The test then sends SIGTERM.
     - The binary must exit non-zero, no earlier than 9 s and within 13 s.
     - This covers the deadline arm and `shutdown_background()`. It also checks that a SIGTERM shutdown exits non-zero only because of the deadline.
8. **`ScratchDatabase` leak.**
   - It now stores `MySqlConnectOptions` and an `is_dropped` flag. `drop_database()` stays explicit.
   - `Drop` runs on a separate thread with its own current-thread runtime. It first kills every connection whose `PROCESSLIST.DB` is the scratch DB, then runs `DROP DATABASE IF EXISTS`. Failures go to stderr.
   - The kill step is needed. Without it, a panicked test deadlocked: the DROP waited on the MDL of an open transaction, and only the test's own runtime could roll that transaction back, but that runtime was blocked in `Drop`.
   - After the final full run, the container had no leftover `jt_*` databases.
9. **Style.** `let _ = connection.close().await` is now `close_detached_connection`, which logs a warning on failure. The bump and the upgrade both use it. `ServiceState::shutdown()` and `ShutdownHandle::token()` have `# Returns` docstrings. `is_cancelled()` is removed, and its two unit-test uses call `token().is_cancelled()`. `token()` is now used by both actor factories and by `wait_for_shutdown`.

### Verification after the fixes

- `cargo clippy --workspace --all-targets --all-features -- -D warnings` and `cargo clippy -p spider-storage --all-targets -- -D warnings`: clean. `cargo +nightly fmt --all -- --check`: clean.
- `cargo nextest run -p spider-storage -p spider-core -p spider-proto-rust --all-features --run-ignored all --release` on a fresh MariaDB container: **280/280 passed**.
- Runs on a container that already had rows from earlier runs hit only the known pre-existing flakes: `restarted_storage_cache_recovers_running_job_from_start` (R20), and the shared-`spider-db` timing tests `test_delete_expired_terminated_jobs` and `test_update_execution_manager_heartbeat_already_dead`.
- The container was stopped after the runs.

### Notes for the next slice (updates)

- `ServiceStateParams` takes `shutdown: ShutdownHandle`, not `cancellation_token`. Tests build it with `ShutdownHandle::new(CancellationToken::new())`.
- `Runtime::stop()` is now `Err` when an actor fails. A test that expects a clean stop must not let an actor return `Err`.
- `ScratchDatabase` cleans up on panic. Tests should still call `.drop_database().await` on success.
- `Runtime::is_fatal()` has no caller in S1. It is kept for S2's M1 stale-start test.


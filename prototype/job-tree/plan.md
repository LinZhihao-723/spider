# Job-tree prototype: implementation plan

| | |
|---|---|
| Author | prototype architect |
| Date | 2026-10-08 |
| Inputs | `brief.md` (this directory); `../design/per-tree-committer.md` (the current version); the worktree `/home/lzh/dev/spider-job-tree-proto` at `278d06f` |
| Output of | the planning step of the prototype workflow |
| Consumers | the implementation agents (one slice at a time), the adversarial reviewers, the test and benchmark agents |
| Revision | 2, after `plan-review.md` (plan reviewer): all 4 major and 11 minor issues resolved; see section 10 for where each one is handled |

Paths are relative to `components/spider-storage/` unless they start with `components/`, `tests/` (repository root) or an absolute path. "Design §x" means a section of `../design/per-tree-committer.md`. Line numbers are at `278d06f`.

---

## 0. How the prototype fits into today's code

These decisions shape every slice. Each was chosen to keep the existing callers and tests compiling and passing with as few edits as possible.

1. **The JCB methods keep their names and return types, and submit internally.** `SharedJobControlBlock::{start, succeed_task_instance, succeed_commit_task_instance, succeed_cleanup_task_instance, fail_task_instance, cancel}` still return `Result<JobState, CacheError>` (or `()` for `start`). Inside, they run phase A as today. At the point where today's code takes the write lock and calls the DB, they drop the guard, call `submit(&self.tree, event)`, and await the reply, raced against the service token. `tests/scheduling_infra.rs` (the `jcb_test` workloads with the no-op and the MariaDB connector) then needs no call-site changes, only a different JCB construction. The spawned handler bodies of design §10.10 wrap these calls in `ServiceState`.
2. **`TreeControlBlock` is not generic.** An event carries only ids, a lane and a message (`TreeEvent`), and the reply is `Result<TreeOutcome, CacheError>`. So `cache/tree.rs` needs no JCB type parameters and `cache/` still imports nothing from `state/` (design §5.1).
3. **The tree's structure and its JCB handles are split.** `TreeStructure` (parents, depths, resource groups, commit and cleanup flags, `active_children`; no JCBs, no states) lives in `cache/tree.rs` and is what the pure planner and the property tests use. The committer owns it plus a `HashMap<JobId, SharedJobControlBlock<..>>` of the non-terminal JCBs (the design's "strong reference per node", design §9.5), so the generic JCB type never reaches the planner.
4. **The publisher is an `InboundQueueSender`.** `state/publisher.rs` adds `StagingPublisher<InboundQueueSenderType>`, whose `send_*` methods push into unbounded per-lane staging queues and return at once; one task per lane forwards into the wrapped bounded sender, raced against the token (design §7.6). `ServiceState::new` wraps the raw `inbound_queue_sender` it is given, so `ServiceStateParams`, `Runtime<InboundQueueSenderHandle, ..>` and `GrpcServiceState<InboundQueueSenderHandle, ..>` keep their type parameters (`tests/runtime_recovery_test.rs` spells the concrete `Runtime` type three times). The `JobCache` inside `ServiceState` becomes `JobCache<StagingPublisher<Q>, D, P>`; tests that write `job_cache: JobCache::new()` still infer.
5. **`DbConnectorType` stays a type parameter of the JCB** but the JCB no longer holds a connector (it would be dead). `JobControlBlock` gets a `PhantomData<fn() -> DbConnectorType>` so `JobCache<Q, D, P>`, `ServiceState<Q, D, P>` and every alias keep their shape. Removing the parameter is a cleanup for the formal implementation (report it as a trap, not done here).
6. **The session fence is scoped by a fence id.** Today's CI and the workflow's test runs share one MariaDB database across concurrently running test processes (nextest runs every test in its own process). Only 20 MariaDB tests are `#[serial_test::file_serial]` (10 in `mariadb_test.rs`, 6 in `jcb_test.rs`, 4 in `runtime_recovery_test.rs`, all on the default key); the rest of `mariadb_test.rs` and `jcb_test.rs` run concurrently with them and with each other. A single fence row would make every `connect` fence off every other live test's connector. `DatabaseConfig` gains `#[serde(default)] pub fence_id: u64` (production: 0, so the design's single row is the production behaviour), and `tests/mariadb_infra.rs::create_mariadb_config` draws a random id per config. Two runtimes created from the same config (the restart tests) share an id and do fence each other. This is a prototype deviation; record it in the report.
7. **Test hooks are a non-default cargo feature, `test-hooks`.** Integration tests in `tests/` link the library without `cfg(test)`, so `#[cfg(test)]` cannot reach them. CI runs `cargo nextest run --all --all-features --run-ignored all --release` (`taskfiles/test.yaml:322`), so feature-gated tests run in CI. Hooks are per runtime instance, never global (section 3).
8. **The old `DbStorage` write methods stay, as wrappers.** `start`, `set_state`, `commit_outputs`, `cancel` and `fail` keep their error contracts, which `tests/mariadb_test.rs` asserts (`JobNotFound`, `UnexpectedJobState`, `InvalidJobStateTransition { from, to }`). Each wrapper reads the state (`get_state`), validates as today, then calls `apply_tree_transition` with that state as the exact guard. They have no production caller after slice S2.
9. **No `job_tree.enabled` flag** (design §16.2). The prototype is never deployed, and every test would enable it. Mention it in the report as omitted.
10. **Every new test that recovers, restarts or creates a tree gets its own scratch database.** Recovery loads, repairs and resends every non-terminal row of the database it connects to, so a shared `spider-db` would let one test's restart repair or republish another test's rows. `tests/mariadb_infra.rs` gains `ScratchDatabase::create() -> ScratchDatabase` (S1):
    - It connects as `root` with `MARIADB_ROOT_PASSWORD`, falling back to `SPIDER_STORAGE_DB_PASSWORD`. `tools/scripts/mariadb/start.py:105` sets the root password to the user password, so CI (`taskfiles/test.yaml`, which exports no root password) works unchanged. A failed root connect panics with a message naming both variables.
    - It runs ``CREATE DATABASE `jt_<random u64 hex>` `` and ``GRANT ALL PRIVILEGES ON `jt_…`.* TO 'spider-user'@'%'``.
    - `config()` returns a `DatabaseConfig` for it (random `fence_id`, `max_connections: 5`). `drop_database(self).await` runs `DROP DATABASE` at the end of a passing test. A panicking test leaks its uniquely named database until the container stops, which is harmless.

    Users: the S1 upgrade and exit-code tests, every `TestEnv` (S2 on), and so every T1-T10 test, the S6 forced-repair tests and the external-RPC test. Tests that never recover and never create a tree (`fence_test.rs`, the `apply_tree_transition` tests) stay on `spider-db`. By this rule `spider-db` never holds a tree, so the unchanged `runtime_recovery_test.rs` restarts never repair anything, and the new tests need no `file_serial`.

Line budget: the design estimates about 1,770 handwritten non-test storage lines (design §16.1). The rough estimate for this plan is 2,000-2,400 non-test lines, plus about 200 feature-gated hook lines that the report should list separately. The overshoot comes mainly from the fence's blocker watcher, the wrappers, the publisher and `build_runtime`'s cleanup on error.

---

## 1. Slice overview

Each slice must leave the worktree building (`cargo build --workspace --all-targets --all-features`, the default members plus the EM), clippy-clean, and the existing `spider-storage` tests passing (unit tests and the `#[ignore = "requires MariaDB"]` integration tests). "Existing tests" means existing assertions. Mechanical harness edits are allowed and listed per slice: constructors, a new required trait method on a mock, a spelled-out type. Assertion changes are not allowed. Behaviour changes that make an existing assertion wrong are called out explicitly; there is one, in S2.

| Key | Title | Depends on | Rough non-test lines |
|---|---|---|---|
| `s1-schema-fence` | Schema, `BlockingOnChildren`, proto, idempotent upgrade, session fence, non-zero fatal exit, `ScratchDatabase` test helper | — | 300 |
| `s2-size1-committer` | Committer for trees of size 1: TrCB, long-running committer, `apply_tree_transition` (UPDATE part), apply, publisher, spawned bodies, `AcqRel`, wrappers, test hooks, `create_runtime_with_hooks` with cleanup on error, `TestEnv` | S1 | 950 |
| `s3-planner` | The complete pure planner and overlay, grouped tree state transactions, deferral, caps, `check_invariants`, property tests (T11) | S2 | 550 |
| `s4-registration` | Registration: `children` field, service validation and limits, prebuilt child graphs, prepared children in TCBs, multi-row `INSERT … RETURNING`, child JCBs, payload-error branch | S3 | 400 |
| `s5-propagation-cancel` | Propagation, abort and root-only cancel, end to end: root-only external RPCs, `get_state_and_parent`, the abort apply, the race tests T1-T7 and T10 | S4 | 150 |
| `s6-recovery` | Recovery: `BlockingOnChildren` rows, tree rebuild, repair R1-R5, a committer per recovered tree, the crash-point tests T8 | S5 | 350 |

S3 deliberately deviates from the brief's suggested order. The whole planner, including the cascade, the abort and the `CommitReady` skip, is pure code over `TreeStructure` and can be written and property-tested before any I/O exists. That lets S4 and S5 become wiring plus tests. Until S4, no tree with more than one job can exist, so the multi-job planner paths are unreachable at runtime and cannot break existing tests.

Window to know about: between S4 and S6, a restart would recover child rows as roots and would not load `BlockingOnChildren` parents (the old recovery query). No existing test restarts with a tree, and no new test may do so before S6.

---

## 2. Slices

### S1 `s1-schema-fence`: schema, job state, proto, upgrade, fence, fatal exit

**Scope.** The new job state and its transition table, the proto additions (generated once, for the whole prototype), the schema columns and the idempotent upgrade, the fenced session protocol with the connect-time X bump, and the non-zero exit on the fatal path with `SHUTDOWN_DEADLINE`. Nothing writes `parent_id` or `root_id` yet.

**Files and functions.**

- `components/spider-core/src/job.rs`: append `BlockingOnChildren` as the last `JobState` variant. Replace `is_valid_transition` with design §3.4, including the new `CommitReady → Cancelled` edge. Add unit tests for the full table (every `(from, to)` pair, asserted against an explicit list).
- `components/spider-proto/storage/storage.proto`: `enum JobState { … BLOCKING_ON_CHILDREN = 8; }`, `message ChildJobRequest { bytes compressed_serialized_task_graph = 1; bytes compressed_serialized_task_graph_input = 2; }`, and `repeated ChildJobRequest children = 7;` on `ReportTaskSuccessRequest`. Regenerate the committed code with `SPIDER_PROTO_RUST_GENERATE_FROM_SOURCE=ON cargo build -p spider-proto-rust` (`protoc` is `/usr/bin/protoc`), then check that `git diff` of `src/generated/` contains only these additions.
- `components/spider-proto-rust/src/job.rs`: add the arms in both conversions, plus a round-trip test for the new value.
- `components/spider-execution-manager/src/client/grpc/storage.rs:100`: add `children: Vec::new()` to the struct literal. This is the only change outside storage: a compile fix, not an EM transport change.
- `src/config.rs`: `DatabaseConfig` gains `#[serde(default)] pub fence_id: u64`, with a docstring.
- `src/db/error.rs`: `DbError::Fenced { mine: SessionId, current: SessionId }`.
- `src/db/mariadb.rs`:
  - `MariaDbStorageConnector` gains `fence_id: u64` and `max_statement_payload: u64`. The latter is `min(4 MiB, @@max_allowed_packet / 2)`, read once at connect and used from S2 on.
  - `connect`:
    - The pool gets an `after_connect` hook that runs `SET SESSION wait_timeout = 900`.
    - Then: create the tables (`jobs_creation_query` gains `parent_id`, `root_id` and `INDEX job_parent (parent_id)`), `upgrade_jobs_schema`, `create_fence_table`, bump the session as today, then `bump_fence`.
  - `upgrade_jobs_schema(pool)` reads before it alters, so a connect to an up-to-date schema issues no DDL and takes no metadata lock on `jobs`:
    1. One read of `information_schema.COLUMNS` (`TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'jobs'`) for the `COLUMN_NAME`s and the `state` `COLUMN_TYPE`, and one read of `information_schema.STATISTICS` for `INDEX_NAME = 'job_parent'`.
    2. If `parent_id`, `root_id` or `job_parent` is missing, one `ALTER TABLE jobs` with only the missing `ADD COLUMN IF NOT EXISTS`/`ADD INDEX IF NOT EXISTS` clauses (the `IF NOT EXISTS` makes two racing upgraders safe).
    3. If the `state` type lacks `'BlockingOnChildren'`, `ALTER TABLE jobs MODIFY state {JobState::as_mysql_enum_decl()} NOT NULL DEFAULT 'Ready'` (design §9.1).
  - `create_fence_table`:
    - `CREATE TABLE IF NOT EXISTS storage_fence (id BIGINT UNSIGNED PRIMARY KEY, session_id BIGINT UNSIGNED NOT NULL)`.
    - Then `INSERT IGNORE … VALUES (fence_id, 0)`.
  - `bump_fence(pool, fence_id, session_id)` follows design §11.3:
    - Dedicated connection A: `SET SESSION innodb_lock_wait_timeout = 5`, `SELECT CONNECTION_ID()`, then the `GREATEST` update on `id = fence_id`.
    - Concurrently, a watcher on connection B starts 1 s in and runs every 1 s until A finishes. It joins `INNODB_LOCK_WAITS`/`INNODB_TRX` and runs `KILL CONNECTION` on each blocker.
    - On error 1205, A retries, at most 5 attempts.
    - If the watcher's lookup fails for lack of privilege (errno 1227 or 1142; the test DB user has no `PROCESS`), it logs one warning and stops. The bump then relies on the 5 s timeout and the retries.
  - `check_fence(conn) -> Result<(), DbError>`: `SELECT session_id FROM storage_fence WHERE id = ? LOCK IN SHARE MODE`. A mismatch is `Fenced`.
  - `register`: BEGIN, `check_fence`, INSERT, COMMIT.
  - The old write methods (`start`, `set_state`, `commit_outputs`, `cancel`, `fail`, `delete_expired_terminated_jobs`): add `check_fence` as the first statement. S2 turns the first five into wrappers.
- `src/state/shutdown.rs` (new, re-exported from `state`): `ShutdownHandle { token: CancellationToken, fatal: Arc<AtomicBool> }` with `fatal()` (sets the flag, then cancels the token), `is_fatal()`, `is_cancelled()`, `token()`. `ServiceState::new` builds one from `params.cancellation_token`, so the params are unchanged. `ServiceState::shutdown()` exposes it, and the scheduler-registration resend failure (`state/service.rs:770`) calls `.fatal()`.
- `src/grpc.rs`, `default_error_handler`:
  - Every fatal arm calls `self.inner.shutdown().fatal()` instead of `self.cancellation_token.cancel()`.
  - New arm: `StorageServerError::Db(DbError::Fenced { .. })` is always fatal. That covers the non-strict job-orchestration handler too (design N13).
  - `StorageServerError::Stopping(_)` maps to `Status::unavailable` in every handler.
- `src/state/runtime.rs`: `Runtime::is_fatal()`.
- `src/bin/grpc_server.rs`:
  - `main` returns `std::process::ExitCode`.
  - `server.await` is raced against `token.cancelled()` followed by `SHUTDOWN_DEADLINE = 10 s`; on expiry it exits non-zero at once.
  - The exit code is `FAILURE` if `runtime.is_fatal()`, if the deadline expired, if the server returned an error, or if `runtime.stop()` failed (design §7.8).
- `tests/mariadb_infra.rs` (harness edits): `create_mariadb_config` sets `fence_id: rand::random()`, and `ScratchDatabase` is added (section 0, item 10).

**Compatibility.**
- `mariadb_test.rs` gets one fence row per connector.
- `runtime_recovery_test.rs` reuses one config per test, so its second runtime fences the first, which is already stopped.
- `jcb_test.rs` registers through its own connector.
- The only enum consumer outside storage, `tests/huntsman/e2e/src/test_driver.rs:252`, matches with `_ =>`.

**Verification.**
1. Build the workspace with all targets and features. Run `cargo nextest run -p spider-core -p spider-proto-rust` and the full `spider-storage` suite with MariaDB, including `--run-ignored all`.
2. New `tests/fence_test.rs` (MariaDB):
   - Two connectors with the same `fence_id`: after B connects, A's `register` returns `Fenced` and B's succeeds.
   - Different ids are independent.
   - A raw connection that holds `SELECT … LOCK IN SHARE MODE` on the fence row in an open transaction delays `connect` until it commits. Assert that `connect` has not returned after 500 ms, commit, then assert that it returns.
3. New upgrade test, on a `ScratchDatabase`: create the `278d06f` `jobs` DDL (7-value enum, no new columns), then `connect` twice. Assert the columns, the index, the enum value and the fence row. Assert that the first connect ran 2 DDL statements and the second ran none, through `MariaDbStorageConnector::schema_ddl_statements_at_connect() -> usize` (`#[cfg(feature = "test-hooks")]`; `upgrade_jobs_schema` returns the count it issued). Global status counters are not used: other test processes run DDL concurrently.
4. New exit-code test, on a `ScratchDatabase`, in `tests/` using `env!("CARGO_BIN_EXE_spider-storage")`: start the binary, start a second one on the same DB and `fence_id`, then send `RegisterJob` to the first. It must exit with a non-zero status within 11 s.

### S2 `s2-size1-committer`: committer for trees of size 1

**Scope.**
- Every plain-job transition goes through a long-running per-tree committer: one event per tree state transaction in this slice, with grouping in S3.
- The pieces this slice adds:
  - the `TreeControlBlock` with its unbounded channel;
  - `recv_many` with a local buffer and a quiescent exit;
  - poison on abnormal exit, token-raced waits and the `TaskTracker`;
  - `apply_tree_transition` (fence, exact guards, `FORCE INDEX (PRIMARY)`, 500-id and byte chunks, matched-row checks);
  - the apply step (`apply_tree_state`);
  - the per-lane publisher;
  - spawned handler bodies;
  - the `AcqRel` decrement;
  - the old DB methods as wrappers;
  - multi-job GC enqueue from the committer;
  - the `resend_ready_tasks` snapshot;
  - the test-hook infrastructure.

**New types and signatures** (abbreviated, so docstrings are omitted here but required in code):

```rust
// cache/tree.rs (new; not generic)
pub enum Lane { Regular, Commit, Cleanup }
pub enum TreeEventKind {
    Start,
    RegularTasksDone { registers: bool },        // `registers` is always false until S4
    CommitDone,
    CleanupDone,
    TerminalFailure { lane: Lane, message: String },
    CancelRoot,
}
pub struct TreeEvent { pub job_id: JobId, pub kind: TreeEventKind }
pub enum TreeOutcome { Applied(JobState), Stale(StaleStateError), TreeClosed }
pub type TreeReply = Result<TreeOutcome, CacheError>;
pub struct Pending { pub event: TreeEvent, pub reply: oneshot::Sender<TreeReply> }

#[derive(Clone)]
pub struct TreeControlBlock {
    root_id: JobId,
    sender: mpsc::UnboundedSender<Pending>,
    token: CancellationToken,                    // for the token-raced reply wait inside JCB methods
    #[cfg(feature = "test-hooks")]
    hooks: TestHooks,
}
impl TreeControlBlock {
    pub fn new(root_id: JobId, token: CancellationToken, ..) -> (Self, mpsc::UnboundedReceiver<Pending>);
    pub fn submit(&self, event: TreeEvent) -> oneshot::Receiver<TreeReply>;   // synchronous, design §7.2
    pub async fn submit_and_wait(&self, event: TreeEvent) -> TreeReply;       // races the token
}
impl TreeEventKind { pub fn stale_error(&self, state: JobState) -> StaleStateError; }  // shared with the planner

pub struct TreeNode { pub parent: Option<JobId>, pub depth: u32, pub resource_group_id: ResourceGroupId,
                      pub has_commit: bool, pub has_cleanup: bool, pub active_children: u32 }
pub struct TreeStructure { root_id: JobId, nodes: HashMap<JobId, TreeNode> }    // non-terminal jobs only

pub struct PlannedUpdate { pub job_id: JobId, pub from: JobState, pub to: JobState,
                           pub outputs: Option<Vec<TaskOutput>>, pub error_message: Option<String> }
pub enum Publication { Ready { job_id: JobId, task_indices: Vec<TaskIndex> }, Commit(JobId), Cleanup(JobId) }
pub enum EventOutcome { Applied(JobState), Stale(StaleStateError) }
pub struct Plan { pub updates: Vec<PlannedUpdate>, pub counters: HashMap<JobId, u32>,
                  pub publications: Vec<Publication>, pub terminated: Vec<JobId>,
                  pub outcomes: Vec<EventOutcome>, pub apply_order: Vec<JobId> }   // grows in S3/S4
```

In S2 the per-event planners have their final signatures over an `Overlay` (design §6), but only their root-level branches are reachable: `plan_start`, `plan_regular_done` without children, `plan_success` (the climb is a no-op for a root), `plan_commit_done`, `plan_cleanup_done`, `plan_failure` (a root has no chain) and `plan_cancel`. The committer plans one event per round with `plan_events(&TreeStructure, &collected, events) -> (Plan, deferred)`, restricted to the first event.

```rust
// db/protocol.rs
pub struct JobStateUpdate { pub job_id: JobId, pub from: JobState, pub to: JobState,
                            pub outputs: Option<Vec<TaskOutput>>, pub error_message: Option<String> }
pub struct ChildJobInsert<'a> { pub parent_id: JobId, pub root_id: JobId, pub resource_group_id: ResourceGroupId,
                                pub compressed_task_graph: &'a [u8], pub compressed_job_inputs: &'a [u8] }
pub struct TreeTransition<'a> { pub updates: Vec<JobStateUpdate>, pub child_inserts: Vec<ChildJobInsert<'a>> }
// InternalJobOrchestration gains (required):
async fn apply_tree_transition(&self, transition: TreeTransition<'_>) -> Result<Vec<JobId>, DbError>;
```

`mariadb.rs::apply_tree_transition` builds the SQL of design §9.4:
- One transaction (REPEATABLE READ): BEGIN, then `check_fence`.
- Updates are grouped by `(from, to, outputs.is_some(), error_message)`, with the outputs groups first.
- Each group is chunked to at most 500 ids, and outputs groups are additionally cut where their bound payload would exceed `max_statement_payload` (every chunk carries at least one row).
- Every UPDATE is `UPDATE jobs FORCE INDEX (PRIMARY) SET state = ?, [serialized_job_outputs = CASE id WHEN ? THEN ? … END,] [error_message = ?,] [ended_at = CURRENT_TIMESTAMP when `to` is terminal] WHERE id IN (…) AND state = ?`.
- Each statement's `rows_affected()` must equal its id count; otherwise it returns `CorruptedDbState` naming the ids and the expected state. sqlx sets `CLIENT_FOUND_ROWS`, so the count is matched rows.
- Non-empty `child_inserts` return `CorruptedDbState("child inserts are not supported yet")` until S4.

`MockDbConnector::apply_tree_transition` (`state/test_utils.rs`) is a faithful in-memory version: it checks the guards against `states`, records outputs and errors, allocates ids for inserts (from S4 on), and returns `CorruptedDbState` on a guard miss. `NoopDbConnector` (`tests/scheduling_infra.rs`) accepts everything and returns fresh ids.

```rust
// state/tree_committer.rs (new)
pub struct TreeEnv<Q: InboundQueueSender, D: InternalJobOrchestration, P: TaskInstancePoolConnector> {
    pub db: D, pub job_cache: JobCache<Q, D, P>, pub gc: JobCacheGcHandle, pub publisher: Q,
    pub task_instance_pool: P, pub shutdown: ShutdownHandle, pub tracker: TaskTracker,
    #[cfg(feature = "test-hooks")] pub hooks: TestHooks,
}
pub async fn create_root_tree<Q, D, P>(env: &TreeEnv<Q, D, P>, job_id: JobId, owner: ResourceGroupId,
    submission: ValidatedJobSubmission) -> Result<SharedJobControlBlock<Q, D, P>, StorageServerError>;
    // builds the TrCB and the channel, the JCB, a one-node TreeStructure, inserts into JobCache, spawns run_committer
async fn run_committer<Q, D, P>(structure: TreeStructure, handles: HashMap<JobId, SharedJobControlBlock<Q, D, P>>,
    rx: mpsc::UnboundedReceiver<Pending>, env: TreeEnv<Q, D, P>);
async fn run_tree_state_transaction<..>(state: &mut CommitterState<..>, events: Vec<Queued>, env: &TreeEnv<..>)
    -> Result<Vec<Queued>, StorageServerError>;   // boxed per round, see section 6
struct Queued { pending: Pending, inputs: Option<EventInputs> }
struct PoisonOnAbnormalExit { rx: mpsc::UnboundedReceiver<Pending>, shutdown: ShutdownHandle, disarmed: bool }
fn close_and_answer(rx: &mut mpsc::UnboundedReceiver<Pending>, buffer: &mut VecDeque<Queued>, reply: impl Fn() -> TreeReply);
```

The committer loop is design §7.2 verbatim, except that in S2 `take_events` returns only the oldest buffered event. Its round runs these steps:
1. `collect_inputs` (one JCB read lock at a time):
   - the state of the event's job;
   - for `RegularTasksDone`, the outputs, **only if the state is `Running`** (see trap R3);
   - for `Start`, the ready task indices, **only if the state is `Ready`** (trap R19). Only then does today's "no ready tasks" `TaskGraphCorrupted` check run (today it is reached only after `write_ready` passed, `cache/job.rs:247-256`). In any other state the input is `PlannerInput::None` and `plan_start` answers `Stale(JobAlreadyStarted)`, today's error from `ensure_ready`. A duplicate `StartJob` (the client's `call_with_retry` resends on UNAVAILABLE) and a `StartJob` after a committed cancel therefore never reach the internal error.
2. Hook `CommitterBeforePlan`, then plan.
3. Hook `CommitterBeforeDbTransaction`, then `apply_tree_transition` unless the plan is empty.
4. Hook `CommitterAfterCommit`, then the apply: per job in `apply_order`, `jcb.apply_tree_state(from, to, mode)` under the write lock, with hook `CommitterAfterApplyJob` after each job. Then update `TreeStructure`, dropping terminal nodes and their handles.
5. GC-enqueue `plan.terminated`.
6. Reply to every event.
7. Stage `plan.publications` through `env.publisher`.
8. In debug builds only, `assert_invariants` from S3 on.

Any `Err` returns, and the guard poisons the tree.

**JCB changes** (`cache/job.rs`):
- `JobControlBlock` gains `parent_id: Option<JobId>`, `depth: u32`, `tree: TreeControlBlock` and `_db: PhantomData<fn() -> DbConnectorType>`.
- `JobExecutionState` loses `db_connector`.
- Constructors:
  - `create(id, owner, submission, sender, pool, tree)` is used by `create_root_tree`.
  - `recover(ctx, sender, pool, tree)`: S6 adds the parent and depth.
- New accessors: `parent_id()`, `depth()`, `tree()`, `has_commit_task()` and `has_cleanup_task()` (read once at creation into the `TreeNode`).
- Methods:
  - `start`: submits `Start`.
  - `succeed_task_instance`:
    - Phase A as today, but `fetch_sub(1, Ordering::AcqRel)`.
    - Ready successors: clone the sender out of the guard, `drop(job)`, then send.
    - Last task: `drop(job)`, then `submit_and_wait(RegularTasksDone)` with no await in between.
    - Hook `PhaseAAfterTcbFlip` sits between the TCB success and `fetch_sub`, and `PhaseABeforeSubmit` sits between the guard drop and the submit.
  - `succeed_commit_task_instance` and `succeed_cleanup_task_instance` use `read_commit_ready`/`read_cleanup_ready`, then the termination TCB success, the guard drop and the submit of `CommitDone`/`CleanupDone`.
  - `fail_task_instance`:
    - Retry path: clone the sender, drop the guard, send.
    - Terminal path: drop the guard and submit `TerminalFailure { lane, message }`.
    - The `write_non_terminated` phase B is deleted.
  - `cancel` submits `CancelRoot`.
  - `resend_ready_tasks` collects under the guard, drops it, then sends.
  - `apply_tree_state(&self, from: JobState, to: JobState) -> Result<(), CacheError>`:
    - Plain write lock, no validator.
    - A `from` mismatch is a new `InternalError::CacheDbDivergence { job_id, expected, actual }`.
    - It sets the state.
    - When `to` is `Failed`, `Cancelled` or `CleanupReady` coming from `Running`, `BlockingOnChildren` or `CommitReady`, it runs `task_graph.cancel_non_terminal()` (this fixes #517 on failure).
    - It runs `finalize()` when leaving `Running` or when deactivated.
  - Committer-only accessors, all `pub(crate)` except where `scheduling_infra` needs them: `collect_state()`, `collect_regular_done_outputs() -> Result<Option<Vec<TaskOutput>>, CacheError>` (returns `None` when not `Running`) and `collect_ready_task_indices()`.
- Reply mapping inside the JCB methods:
  - `Applied(s)` becomes `Ok(s)`.
  - `Stale(e)` becomes `Err(StaleState(e))`.
  - `TreeClosed` becomes the stale error that `kind.stale_error(self.state().await)` gives, or `CacheError::ServiceStopping` when the token is cancelled.
  - A closed oneshot becomes `InternalError::TreePoisoned`.
- `CacheError` gains `ServiceStopping`, which `ServiceState` maps to `StorageServerError::Stopping`.
- The write validators (`write_running`, `write_commit_ready`, `write_cleanup_ready`, `write_non_terminated`, `write_ready`, `write_cancellable`) are deleted as dead code. `ensure_commit_ready`/`ensure_cleanup_ready` answer stale for `BlockingOnChildren`. `ensure_succeeded` answers a new `StaleStateError::JobNotSucceeded(JobState)` for any non-`Succeeded` state (#515).

**Service and runtime** (`state/service.rs`, `state/runtime.rs`, `state/publisher.rs`):
- `ServiceStateInner` gains:
  - `shutdown: ShutdownHandle` (from S1);
  - `tracker: TaskTracker`;
  - `publisher: StagingPublisher<Q>`, spawned in `new` into the tracker (lane tasks select on the token);
  - `#[cfg(feature = "test-hooks")] hooks`.
  - Its `job_cache` becomes `JobCache<StagingPublisher<Q>, D, P>`.
  - Add `fn tree_env(&self) -> TreeEnv<StagingPublisher<Q>, D, P>`.
- The bodies of `start_job`, `cancel_job`, `create_task_instance`, `succeed_task_instance`, `succeed_commit_task_instance`, `succeed_cleanup_task_instance` and `fail_task_instance` run through `fn spawn_body<T>(&self, fut) -> impl Future<Output = Result<T, StorageServerError>>`. It runs `tracker.spawn(fut)` and selects on the token: a join error becomes internal, and the token becomes `Stopping`.
- `enqueue_for_gc_if_terminal` and its six calls are removed; the committer enqueues.
- `register_job`: fenced `register`, then `create_root_tree`.
- `recover_job_cache` moves to `ServiceState::recover_jobs(&self)`. In S2 each recovered row becomes its own one-node tree with its own committer. `create_runtime` is reordered:
  1. connect, the inbound queue, the pool;
  2. an empty `JobCache`, then the GC;
  3. `ServiceState::new`;
  4. `recover_jobs`;
  5. return.
- One body for the production and the test path. A private `build_runtime(config, #[cfg(feature = "test-hooks")] hooks: TestHooks)` holds the steps above. `create_runtime(config)` calls it (with `TestHooks::default()` when the feature is on), and the feature-gated `create_runtime_with_hooks(config, hooks)` calls it with the caller's hooks. Neither public function has any other logic, so every test through `create_runtime_with_hooks` exercises the production path.
- Cleanup on a failed build. Once anything is spawned (the pool, the GC, the publisher lanes, recovered committers), an error from a later step must not leak it. `build_runtime` collects what it spawned into a `PartialRuntime` guard. On any `Err` after `connect` it:
  1. calls `shutdown.token().cancel()`;
  2. closes the tracker and awaits it;
  3. awaits the pool and GC join handles, all under the existing 30 s stop timeout;
  4. drops the connector clones;
  5. then returns the original error, with any cleanup failure logged at `error`.
  The guard is disarmed when `build_runtime` returns `Ok(Runtime)`.
- `Runtime` gains the tracker. `Runtime::stop` cancels the token, closes the tracker, and waits for it together with the pool and GC joins under the existing 30 s timeout.
- Ready-task resend stays outside `create_runtime`, as on `main`: `bin/grpc_server.rs:98-109` calls `resend_ready_tasks` after `serve` starts, and every restart in the tests calls it at the same point (section 3).
- `JobCache::resend_ready_tasks` copies the `Arc`s under the read lock, releases it, then iterates. `JobCache::insert_many` is added now (used in S4).

**Compatibility and mechanical test edits.**
- Library visibility change: `JobCacheGcHandle::new` becomes `pub` (it is `pub(crate)` today). `tests/scheduling_infra.rs` is an external crate and must build its `TreeEnv`. List it in the report with the other API changes.
- `tests/scheduling_infra.rs`:
  - `NoopDbConnector` gains `apply_tree_transition`.
  - `run_workload` builds a `TreeEnv` (a fresh `JobCache`, `JobCacheGcHandle::new(unbounded_channel().0)`, its own `MockInboundQueueSender` as the publisher, a `ShutdownHandle`, a `TaskTracker`) and calls `create_root_tree` instead of `SharedJobControlBlock::create`.
  - Assertions are unchanged. The `jcb_test` cancel workloads still see exactly one cleanup and no commit.
- `state/service.rs` tests:
  - The tests that inserted a hand-built JCB (`create_test_jcb`: `cancel_job_transitions_to_terminal_state`, `cancel_job_enqueues_terminal_job_for_cache_gc`) build it through `create_root_tree` on the service's `tree_env()`. Their `JobId::random()` is not in `MockDbConnector.states`, so the faithful mock would answer the first `apply_tree_transition` with a guard miss and poison the tree. Each test therefore seeds `db.states.insert(job_id, <the JCB's state>)` before the call (mock data, not an assertion change).
  - The GC tests keep `try_recv() == Ok(job_id)`, because the committer enqueues before it replies.
  - `get_job_outputs_returns_error_when_job_not_succeeded` still gets an error (now stale).
- `state/job_cache.rs` tests and the `state/runtime.rs` `Runtime { .. }` literal get the same mechanical edits.
- `mariadb_test.rs` is unchanged: the wrappers keep the error contracts.
- `runtime_recovery_test.rs` is unchanged.

**Verification.**
1. Rerun the full existing suite. With MariaDB, `jcb_test` with the MariaDB connector is the strongest end-to-end check: it puts 10,000 tasks and the commit and cleanup paths through the committer.
2. New committer lifecycle unit tests on `MockDbConnector`:
   - The committer exits exactly when the root becomes terminal; a `submit` after that is answered at once with the job's stale error.
   - A shutdown answers every buffered and queued event `Stopping`.
   - A poison (hook `Crash` at `CommitterBeforeDbTransaction`) wakes every queued waiter with `TreePoisoned` and sets the fatal flag.
   - A panic (hook `Panic` at `CommitterAfterApplyJob`) does the same.
3. New `apply_tree_transition` tests (MariaDB):
   - A guard miss is `CorruptedDbState` and rolls back every earlier statement.
   - 1,200 updates span three 500-id chunks in one transaction.
   - An outputs group exceeding `max_statement_payload` is split.
   - A wrong fence is `Fenced`.
   - `ended_at` is set only for terminal `to` states.
   - Optional: assert through `EXPLAIN` that every guarded statement uses `PRIMARY`.
4. Stale `StartJob` (trap R19), on `MockDbConnector` and on a `TestEnv`:
   - `StartJob` twice on one job: the second reply is `Stale(JobAlreadyStarted)`.
   - `CancelJob` and then `StartJob`: `Stale(JobAlreadyStarted)`.
   - In both, `is_fatal()` stays false and the tree is not poisoned (a later `GetJobState` answers).
5. T9 (dropped RPC future, section 4, all three drop points) and the plain-job form of T5. This slice adds `tests/job_tree_infra.rs` with `TestEnv` (section 4) and `create_runtime_with_hooks`, so these tests and every later slice use them.
6. A failed build leaks nothing. On a `ScratchDatabase` holding two `Running` jobs, arm `Crash@RecoveryBeforeRepair` filtered to the `root_id` of the job with the larger id, so that it fires on the second recovered tree (recovery goes in ascending id order) (the hook points exist in `recover_jobs` from S2 on, per tree, with the S6 repair placed between them), after the first tree's committer is spawned. `create_runtime_with_hooks` returns `Err`. Then assert through `TestHooks::runtime_probe()` (feature-gated; `build_runtime` registers its token and tracker there) that the token is cancelled and the tracker is closed and empty, and that a following `create_runtime_with_hooks` on the same config returns within 2 s and recovers both jobs.
7. In-process B1 smoke: `SPIDER_TEST_INSTRUMENT_OUTPUT_DIR=… cargo nextest run -p spider-storage jcb_test --run-ignored all --release` on the worktree and on `/home/lzh/dev/spider` gives a cheap per-call latency comparison before the gRPC benchmarks.

### S3 `s3-planner`: the pure planner, overlay and grouped rounds

**Scope.**
- The complete planner of design §6 and §7.4-§7.5 as pure, synchronous code, including the tree cases:
  - `climb_after_success`, `unblock`, `fail_and_abort`, `fail_blocked` and `abort_rest`;
  - `plan_cancel` with the non-root `CommitReady` skip;
  - `plan_regular_done` with a child set;
  - the `Absent` node semantics.
- The overlay. The event-ordering rules: cancel first, then arrival order, then the oldest registering parent last. The deferral rules: never touch a row twice; the 16 MiB payload budget with the first event always admitted; the 256-event cap.
- `check_invariants` (I1-I4).
- The committer switches to grouped rounds, with deferred events keeping their inputs (design N3).
- The T11 property tests.

**Files and signatures** (`cache/tree.rs`, plus `state/tree_committer.rs` for the loop):

```rust
pub enum NodeState { Present(JobState), Absent }
pub struct Overlay<'a> {                      // design §7.4
    structure: &'a TreeStructure, collected: &'a HashMap<JobId, JobState>,
    planned: HashMap<JobId, JobState>, counters: HashMap<JobId, u32>, touched: HashSet<JobId>,
}
pub struct ChildSetSummary { pub count: usize, pub bytes: u64 }   // what the planner needs from prepared children
pub enum PlannerInput {                        // the planner never sees TaskGraphs or JCBs
    Start { ready_task_indices: Vec<TaskIndex> },
    RegularTasksDone { outputs: Option<Vec<TaskOutput>>, children: Option<ChildSetSummary> },
    None,
}
pub struct EventPlan { /* updates, counter changes, child registration, publications, terminated, outcome, payload_bytes */ }
pub fn plan_events(structure: &TreeStructure, collected: &HashMap<JobId, JobState>,
                   events: &[(TreeEvent, PlannerInput)]) -> (Plan, Vec<usize> /* deferred event indices */);
pub fn check_invariants(structure: &TreeStructure, states: &HashMap<JobId, JobState>) -> Result<(), Violation>;
```

- `Plan` gains `registration: Option<PlannedRegistration { parent_id, root_id, resource_group_id, child_depth, count, bytes, event_index }>` and `payload_bytes`.
- `apply_order` sorts the touched jobs by depth, descending and stable, with the root last.
- `terminated` lists every update whose `to` is terminal.
- Publications: a commit for every `→ CommitReady`, a cleanup for every `→ CleanupReady`, the ready tasks for `Start`, and from S4 the children's ready tasks.

The committer side:
- `take_events` puts the oldest `CancelRoot` first (later cancels follow it and plan stale), then the remaining events in arrival order with registrations removed, then the oldest `RegularTasksDone { registers: true }`. Any other registering event stays in the buffer uncollected for the next round.
- `collect_inputs` reads the states of every event's job and its ancestors (`TreeStructure` parents), or of every node when the round holds a `CancelRoot` or a `TerminalFailure` (design N4). A deferred `Queued` keeps its inputs.
- After the apply, the counters from `plan.counters` are written into `TreeStructure`.
- In debug builds, `check_invariants` runs on the post-state, reading the states back from the JCBs.

**Compatibility.** All size-1 plans are identical to S2's, except that several events of one plain job can now share a round. For example, two `TerminalFailure`s of one job in one round give `Failed`, then `Stale`. No existing assertion depends on round boundaries.

**Verification.**
- T11 (section 4), in `cache/tree.rs` `#[cfg(test)] mod property`, with no I/O.
- Named unit cases (design §15):
  - two siblings failing in one round;
  - cancel, registration and completion in one round;
  - cancel and failure in one round (the outcome table of design §7.5);
  - two registering parents in one round;
  - a child set over `MAX_JOB_CHILD_BYTES` across two reports, failed by the planner;
  - a skipped `CommitReady` job whose aborted parent is `Absent`;
  - an all-stale round that gives an empty plan.
- Committer-level grouping tests on `MockDbConnector` with the `CommitterAfterReceive` pause, to force two events into one round.
- Rerun the full existing suite.

### S4 `s4-registration`: registration

**Scope.**
- The storage side of child creation:
  - the `children` field end to end;
  - service-level decoding, validation and limits before any cache mutation;
  - prebuilt child graphs;
  - prepared children stored in the TCB critical section;
  - the `registers` flag;
  - `plan_regular_done` with children (already planned in S3);
  - `INSERT … RETURNING id` chunks inside the same transaction;
  - child JCB binding and `JobCache::insert_many`;
  - staging of the children's ready tasks;
  - the deterministic-payload-error branch (design §6.2 step 5);
  - the 32 MiB decode limit.
- With S3's planner and collect, children's completions now cascade, so trees succeed end to end.

**Files and signatures.**
- `components/spider-proto-rust/src/unpack/storage.rs`: `ReportTaskSuccessRequest` unpacks to a 6-tuple ending in `Vec<(Vec<u8>, Vec<u8>)>` (graph, inputs). Update the docstring.
- `src/grpc.rs`, `report_task_success`: passes the children to `succeed_task_instance_with_children`. When the task id routes the report to `succeed_commit_task_instance` or `succeed_cleanup_task_instance`, a non-empty `children` is rejected with `INVALID_ARGUMENT` ("children are only accepted from regular tasks") before any cache access, instead of being dropped silently. Add a unit test for both lanes.
- `src/bin/grpc_server.rs`: `TaskInstanceManagementServiceServer::new(..).max_decoding_message_size(32 * 1024 * 1024)` (design N18).
- Limits: `src/state/limits.rs` (new; constants):
  - `MAX_DEPTH = 8`, `MAX_CHILDREN_PER_JOB = 10_000`;
  - `MAX_REPORT_CHILD_BYTES = 16 MiB`, `MAX_REPORT_OUTPUT_BYTES = 4 MiB`;
  - `MAX_JOB_CHILD_BYTES = 16 MiB`, `MAX_TREE_TRANSACTION_PAYLOAD = 16 MiB`;
  - `MAX_EVENTS_PER_ROUND = 256`, `IN_LIST_MAX_IDS = 500`.
  - The planner-side constants live in `cache/tree.rs`, because `cache/` must not import `state/`.
- `src/db/protocol.rs`: `InternalJobOrchestration::max_statement_payload(&self) -> u64` (sync; the MariaDB value from S1, 4 MiB in the mocks), which gives `max_child_bytes`.
- `src/state/service.rs`:
  - `succeed_task_instance_with_children(session, job, instance, index, outputs, children: Vec<ChildJobSubmission>)`. The existing `succeed_task_instance` delegates with an empty vector, so `runtime_recovery_test.rs` is unchanged.
  - The spawned body runs, in order:
    1. Decode the outputs and check `MAX_REPORT_OUTPUT_BYTES`.
    2. Look up the JCB, then check `jcb.depth() + 1 <= MAX_DEPTH`, the count, the per-child bytes and the per-report bytes.
    3. For each child: clone the two compressed buffers, `ValidatedJobSubmission::create`, read `num_tasks`, `has_commit` and `has_cleanup` from the submitted graph, `TaskGraph::create`, and `get_all_ready_task_indices`.
  - Any violation or decode or build error calls `jcb.fail_task_instance(instance, TaskId::Index(index), message)` instead (a task failure).
- `src/cache/task.rs`:
  - `pub struct PreparedChild { pub graph: TaskGraph, pub num_tasks: usize, pub has_commit: bool, pub has_cleanup: bool, pub ready_task_indices: Vec<TaskIndex>, pub compressed_graph: Vec<u8>, pub compressed_inputs: Vec<u8> }`.
  - `TaskControlBlock` gains `prepared_children: Vec<PreparedChild>`.
  - `SharedTaskControlBlock::succeed_task_instance_with_children(id, outputs, prepared)` stores them after the existing checks. The old method delegates, so the TCB unit tests are unchanged.
  - `take_prepared_children(&self) -> Vec<PreparedChild>`.
- `src/cache/job.rs`:
  - `JobExecutionState` gains `has_prepared_children: AtomicBool`, set in the TCB critical section's caller before the `AcqRel` `fetch_sub`. The last task reads it after the decrement into `RegularTasksDone { registers }`.
  - `succeed_task_instance_with_children` is added; the old method delegates.
  - `collect_regular_done_inputs()` returns the outputs plus every TCB's prepared children in task-index order (design §6.2 step 3), taking them out of the TCBs. It takes and drops them when the job is no longer `Running`.
  - `SharedJobControlBlock::from_prepared_child(id, parent_id, depth, owner, prepared, sender, pool, tree) -> Self` is infallible (state `Running`).
- `src/db/mariadb.rs`:
  - `apply_tree_transition` runs the INSERT chunks after the UPDATE groups: `INSERT INTO jobs (resource_group_id, parent_id, root_id, state, compressed_serialized_task_graph, compressed_serialized_job_inputs) VALUES (?, ?, ?, 'Running', ?, ?), … RETURNING id`. Chunks are cut by rows (500) and by bound bytes. It returns the ids in VALUES order.
  - A statement before COMMIT that fails with errno 1153 or 1406 maps to a new `DbError::PayloadRejected(sqlx::Error)`. Add `DbError::is_deterministic_payload_error()`.
- `src/state/tree_committer.rs`:
  - `Queued` carries `Option<Vec<PreparedChild>>` next to the planner's `ChildSetSummary`.
  - On `PayloadRejected` in a round with a registration, the committer replaces that event with `TerminalFailure { Regular, "child payload rejected by the database" }` (keeping the reply sender), drops the children, and returns every event of the round for requeue (design §7.3).
  - After COMMIT:
    1. bind child JCBs through `from_prepared_child` with the returned ids;
    2. add `TreeNode`s (depth, flags, `active_children = 0`, the parent's counter `k`) and handles;
    3. `job_cache.insert_many` after every guard is dropped;
    4. stage each child's `ready_task_indices`.
- `MockDbConnector`: inserts allocate ids from a counter that starts above any `JobId::random()` collision risk, or a dedicated range.

**Compatibility.**
- The unpack tuple change touches only `grpc.rs`.
- The old success paths are untouched when `children` is empty, which it always is for existing tests and for the EM (`Vec::new()`).
- The EM never sends children.

**Verification** (MariaDB, `test-hooks` where noted):
- Registration happy path:
  - the parent becomes `BlockingOnChildren` with its outputs in the row;
  - `k` rows have `parent_id`/`root_id` set, with ids ascending in task-index order;
  - the children are cached `Running` and their tasks are published;
  - the cascade unblocks the parent (to `Succeeded`, or to `CommitReady` and then `Succeeded` after the commit task). This is the core of T6.
- Two sibling tasks of P that both carry children: one registration with the union.
- A duplicate success report of a task that carries children is rejected before storing anything.
- T10 limit violations (count, per-child bytes, per-report bytes, depth, outputs): with `max_num_retry = 0` the job fails and nothing crashes; with retries, every retry fails the same way and the job then fails.
- The payload-error branch: a `FailDb { deterministic_payload: true }` hook at `CommitterBeforeDbTransaction` makes the parent fail in the next round with that message, and every other event of the round is still answered once.
- Optional real 1406/1153 test: a root connection lowers `max_allowed_packet` for a scratch database.
- Rerun the full existing suite.

### S5 `s5-propagation-cancel`: propagation, abort and root-only cancel, end to end

**Scope.**
- The external-RPC rules for trees, and the remaining apply-side behaviour of aborts:
  - root-only `StartJob`, `CancelJob`, `GetJobState`, `GetJobOutputs` and `GetJobError` (design §9.6, N17);
  - `get_state_and_parent` for the cache-miss path;
  - the cancel cache-miss semantics of design §6.5;
  - dropping prepared children of aborted jobs;
  - cleanup publications for aborted jobs;
  - `Absent` parents after GC eviction.
- Then the race tests T1-T7 and T10 on the real code.
- The planner logic itself is S3's; this slice proves it under real concurrency.

**Files and signatures.**
- `src/db/protocol.rs`: `ExternalJobOrchestration::get_state_and_parent(&self, job_id) -> Result<(JobState, Option<JobId>), DbError>`, implemented in MariaDB (one PK read) and in `MockDbConnector` (parents default to `None`).
- `src/state/service.rs`: the root check is added in front of each RPC's existing behaviour, which it does not change (design §6.5, §9.6 require only the non-root rejection):
  - On a cache hit: `jcb.parent_id().is_some()` is `BadRequest`; otherwise the RPC proceeds as today.
  - On a cache miss, `cancel_job` reads `get_state_and_parent` instead of `get_state`. A non-root is `BadRequest`. Otherwise today's arms are unchanged: a terminal root returns its state, a non-terminal root is the fatal `InternalError::JobNotFound`, and `JobNotFound` stays `JobNotFound`.
  - On a cache miss, `get_job_state`, `get_job_outputs` and `get_job_error` call `get_state_and_parent` first. A non-root is `BadRequest`. Any other result (including `JobNotFound`) falls through to today's DB read, whose result and errors are returned as today.
  - On a cache miss, `start_job` returns `JobNotFound` without reading the DB, as today. A child is never `Ready`, so a `StartJob` on an evicted child id cannot do harm, and its answer stays `JobNotFound`.
- `src/cache/job.rs`, `apply_tree_state` in deactivation mode: also clears every TCB's `prepared_children`. `TaskGraph` gets `drop_prepared_children()` for this.
- `src/grpc.rs`: `BadRequest` is already `INVALID_ARGUMENT` in the job-orchestration handler; no change.

**Compatibility.**
- Every existing job is a root, so the root checks pass.
- Existing tests that read unknown ids (`get_job_state_returns_error_for_unknown_job`, `cancel_job_returns_job_not_found_if_not_exist`) still get their errors: the mock's `get_state_and_parent` returns `JobNotFound` for unknown ids, `cancel_job` keeps today's mapping, and the getters fall through to today's DB read.
- `get_job_outputs_returns_outputs_from_db` seeds only `outputs`, and `get_job_error_returns_error_message_from_db` seeds only `errors`. Because the getters fall through on `JobNotFound`, they pass unchanged. The mock's `get_state_and_parent` still reads `states` and `parents` only; no fallback is added.

**Verification.**
- T1-T7 and T10 (section 4), each a deterministic hook-driven test repeated `JOB_TREE_RACE_ITERATIONS` times (default 50, set to 500 for the final test run).
- An external-RPC test: `CancelJob`, `GetJobState`, `GetJobOutputs` and `GetJobError` on a child id get `BadRequest` on a cache hit and after eviction (a GC with retention 0, or `job_cache.remove`). `StartJob` gets `BadRequest` on a hit and `JobNotFound` after eviction. Unknown ids keep today's answer for every RPC.
- Rerun the full existing suite.

### S6 `s6-recovery`: recovery and repair

**Scope.**
- The storage-session-restart algorithm of design §11.2:
  - load `BlockingOnChildren` rows with `parent_id`/`root_id`;
  - group by tree in ascending id order;
  - rebuild `TreeStructure` (flags from the decoded graphs, depth where the parent is loaded, counters from the loaded rows);
  - probe `has_failed_child` with `persistent(false)`;
  - plan the repair R1-R5 purely and apply it in one fenced transaction per tree;
  - recover each JCB:
    - `BlockingOnChildren` through `TaskGraph::recover_blocked`;
    - `CommitReady` keeps its cleanup TCB (fixes #513);
  - spawn one committer per non-quiescent tree before serving.
- The crash-point tests T8 and the forced repair states.

**Files and signatures.**
- `src/db/protocol.rs`:
  - `RecoverableJobContext` gains `parent_id: Option<JobId>` and `root_id: Option<JobId>`.
  - `get_recoverable_jobs` also loads `BlockingOnChildren`, and its docstring is updated.
  - `InternalJobOrchestration::has_failed_child(&self, parent: JobId) -> Result<bool, DbError>` (`EXISTS … WHERE parent_id = ? AND state IN ('Failed','Cancelled')`, `persistent(false)`).
- `src/db/mariadb.rs`: the projection and the query.
- `src/cache/task.rs`:
  - `TaskGraph::recover_blocked(outputs, commit_descriptor: Option<&TerminationTaskDescriptor>, cleanup_descriptor: Option<&…>)`.
  - `recover_from_commit_ready(commit, cleanup: Option<&…>, outputs)` keeps the cleanup TCB.
- `src/cache/job.rs`: `recover(ctx, state_after_repair, depth, sender, pool, tree)`. The `BlockingOnChildren` arm needs outputs, and a row without outputs is a row corruption that fails startup, as for `CommitReady` today.
- `src/cache/tree.rs`:
  - `TreeStructure::from_rows(root_id, rows: &[RecoveredRow])`, where `RecoveredRow { id, parent_id, state, resource_group_id, has_commit, has_cleanup }`.
  - `plan_repair(structure, states, failed_child: &HashSet<JobId>) -> Plan`, pure, using the S3 building blocks. R5 replaces the R1-R4 plan (design N8).
- `src/state/service.rs`, `recover_jobs`:
  1. sort by id;
  2. group by `root_id.unwrap_or(id)`;
  3. per tree: probes, `plan_repair`, and, when the plan is non-empty, an error log with the job ids followed by `apply_tree_transition`;
  4. build the TrCB, the JCBs and the `TreeStructure`, insert into `JobCache`, and spawn the committer unless quiescent.
  - The probes and the repair go between the hooks `RecoveryBeforeRepair` and `RecoveryAfterRepair`, which S2 already placed per tree. A `Crash` there makes `build_runtime` return `Err` after its S2 cleanup, which is the in-process form of a storage process dying during recovery.
- `src/state/runtime.rs`: no structural change beyond S2's ordering.

**Compatibility.** `runtime_recovery_test.rs` covers plain `Ready`, `Running`, `CommitReady` and `CleanupReady` jobs: each is a one-row tree, and `plan_repair` is empty for them. The `CommitReady` test now also gets the cleanup TCB, which it does not observe.

**Verification.**
- T8 (section 4).
- Forced repair states (design §15), each on its own `ScratchDatabase` and written by raw SQL before a restart:
  - R1 orphan;
  - R2 with and without a commit task;
  - R4 (one `Failed` and one `Running` child under a `BlockingOnChildren` parent);
  - R5.
- One restart converges, and a second restart logs no repair. The repair is observed through an `Observe@RecoveryAfterRepair` hook whose context carries the tree's repaired job ids, not by scanning the log, so the assertion names exactly the forced rows.
- `check_invariants(plan_repair(rows)) == Ok` over random injected row sets, extending T11.
- Rerun the full existing suite, `runtime_recovery_test` in particular.

---

## 3. Test-hook design

**Feature.** `[features] test-hooks = []` in `components/spider-storage/Cargo.toml`. Code lives in `src/test_hooks.rs` (`#[cfg(feature = "test-hooks")] pub mod test_hooks;` in `lib.rs`). Call sites use a macro that expands to nothing without the feature:

```rust
hook!(handle, HookPoint::CommitterBeforeDbTransaction, HookContext::round(root_id, &events))?;
```

**Points** (the first four are the brief's minimum):

| Point | Where | Typical use |
|---|---|---|
| `CommitterBeforePlan` | `run_tree_state_transaction`, after `collect_inputs` | Change the tree under a collected round; crash before planning |
| `CommitterAfterCommit` | after `apply_tree_transition` returns `Ok`, before the apply | Crash after COMMIT (T8); observe the DB ahead of the cache (I5) |
| `PhaseAAfterTcbFlip` | `SharedJobControlBlock::succeed_task_instance`, between the TCB success and `fetch_sub` | Drop the RPC future (T9); sibling races (T4); straggler timing (T5) |
| `CommitterBeforeDbTransaction` | after planning, before `apply_tree_transition` | Hold a round in flight while other events queue; crash before COMMIT; inject `FailDb` |
| `CommitterAfterReceive` | after `recv_many`/`try_recv` fills the buffer, before `take_events`; after a pause it drains the channel again with `try_recv` | Force several events into one round deterministically |
| `CommitterAfterApplyJob` | after each job's `apply_tree_state` | Crash mid-cascade; `Panic` in the apply (H17) |
| `PhaseABeforeSubmit` | after the guard drop, before `submit` | Interleave a last-task submit with a cancel |
| `PhaseABeforeSuccessorLock` | `succeed_task_instance`, inside the TCB success, after `write_outputs` and before the first successor TCB lock is awaited | Drop the RPC future at a successor-lock await (T9, design §15) |
| `CreateInstanceBeforePoolSend` | `create_task_instance`, after the TCB registered the instance and before it is sent to the task-instance pool | Drop the RPC future between registration and the pool send (T9, design §15) |
| `EventSubmitted` | inside the synchronous `submit`, after a successful send | `Observe` only, through the synchronous path below: wait until a racing event is in the channel |
| `RecoveryBeforeRepair`, `RecoveryAfterRepair` | `recover_jobs`, per tree (from S2; the S6 probes and repair sit between them) | Crash during recovery; `Observe` with the repaired job ids in the context |

**Actions.**
- `Pause`: block until released, raced against the runtime's token. If the token is cancelled first (a `Crash` elsewhere, `Runtime::stop`), the hit returns `InjectedFault::Stopping`: a committer point ends its round as on shutdown, and a phase-A point returns `CacheError::ServiceStopping` to its caller. A paused phase-A body therefore never holds its JCB read guard past a stop, and a restart never waits for the 30 s stop timeout. In addition, dropping an `ArmedHook`, and dropping or calling `TestHooks::release_all()`, closes its semaphore, which lets every waiter of that hook continue at once. `TestEnv::restart` calls `release_all()` before `stop()`.
- `Observe`: count and notify, never block. At `EventSubmitted` it is the only valid action: `arm` panics for any other action there, because `submit` is synchronous (design §7.2) and cannot await `hit`. That point uses `TestHooks::observe(point, &ctx)`, a synchronous method that increments the matching hooks' `AtomicUsize` hit counters and calls `Notify::notify_waiters` (plus a stored permit, so a waiter that arrives later still sees the count). `ArmedHook::reached(n)` waits on that `Notify` until the counter reaches `n`, the same way for both paths.
- `Crash`: committer points return an injected internal error, so the round ends abnormally, the guard poisons the tree and the process path is the real fatal path. Phase-A points return it to the caller.
- `FailDb { deterministic_payload: bool }`: valid at `CommitterBeforeDbTransaction` only. The round sees a feature-gated `DbError::Injected { deterministic_payload }` in place of the DB call's result.
- `Panic`.

Every action has a filter (`root_id`, `job_id`, event-kind tag, each optional) and a count (`Once`, `Times(n)`, `Always`).

**API.**

```rust
#[derive(Clone, Default)] pub struct TestHooks { inner: Arc<Mutex<Vec<Arc<ArmedHookState>>>> }
impl TestHooks {
    pub fn arm(&self, point: HookPoint, filter: HookFilter, action: HookAction, count: HookCount) -> ArmedHook;
    pub fn release_all(&self);                // closes every armed hook's semaphore
    pub fn runtime_probe(&self) -> Option<(CancellationToken, TaskTracker)>;   // registered by build_runtime
    pub(crate) async fn hit(&self, point: HookPoint, ctx: &HookContext) -> Result<(), InjectedFault>;  // races the token
    pub(crate) fn observe(&self, point: HookPoint, ctx: &HookContext);           // synchronous, Observe only
}
impl ArmedHook {
    pub async fn reached(&self, n: usize);   // waits until the n-th matching hit
    pub fn release(&self);                    // one permit; a release before the hit is not lost (Semaphore)
    pub fn hits(&self) -> usize;
    pub fn disarm(self);                      // same as drop: closes the semaphore
}
impl Drop for ArmedHook { /* close the semaphore, remove the hook from TestHooks */ }
```

**Plumbing.** Hooks are per runtime instance, never global. In-process tests with `cargo test` share a process, and parallel tests must not see each other's hooks.
- `ServiceState::new` creates a `TestHooks` (feature only), unless a test supplies one through `create_runtime_with_hooks`.
- `TreeEnv` and every `TreeControlBlock` hold a clone, which is how the JCB phase-A points reach it.
- `ServiceState::test_hooks()` and `Runtime::test_hooks()` expose it.

**Determinism recipes.** No sleeps; every wait is on `reached`, `Observe` counts or RPC completion.
- *Same round*:
  1. Arm `Pause@CommitterAfterReceive` for the root.
  2. Spawn RPC 1 and `reached(1)`.
  3. Spawn RPC 2 and wait for `Observe@EventSubmitted` for its event.
  4. `release`. The drain after the pause puts both events into one round.
- *Consecutive rounds, the second queued during the first*:
  1. Arm `Pause@CommitterBeforeDbTransaction`.
  2. Spawn RPC 1 and `reached(1)`.
  3. Spawn RPC 2 and wait for its `EventSubmitted`.
  4. `release`.
- *Phase A against a committed cancel* (design §10.6 (b) 2):
  1. Arm `Pause@PhaseAAfterTcbFlip` for J and `Observe@CommitterAfterCommit` filtered to `CancelRoot`.
  2. Spawn the report and `reached(1)`.
  3. Spawn the cancel and wait for its `CommitterAfterCommit` observation. Do not await the cancel RPC: its apply waits for phase A's read guard.
  4. `release`, then await both and assert that the report's event was answered stale.

**Simulating a storage restart.** This follows what `runtime_recovery_test.rs` does with `create_runtime`/`stop`/`resend_ready_tasks`, and the order of `bin/grpc_server.rs`.
1. `Crash` at a committer point drives the real fatal path: the poison sets the flag and cancels the token.
2. The test awaits the failing RPC (`TreePoisoned` or `Stopping`).
3. It calls `hooks.release_all()`, then `runtime.stop()` and ignores the result, then drops the runtime. The connector's pool closes when the last clone is dropped; the test also drops its `ServiceState` clones.
4. It calls `create_runtime_with_hooks(&same_config, fresh_hooks)`: a new session, the fence bump, then recovery from the DB. If a `Recovery*` crash is armed, this returns `Err` after the S2 cleanup, and the helper calls it again with fresh hooks (the second recovery of the same rows).
5. It calls `service_state.resend_ready_tasks().await` on the new runtime, the point where the binary calls it after `serve`, and only then lets `TestEm` poll. Without this step recovered `Running` jobs publish nothing.

Zombie variant:
1. Keep the old runtime alive (no crash) and create the new one from the same config.
2. Drive an RPC on the old `ServiceState`.
3. Its committer's write must fail with `Fenced`, the old runtime must report `is_fatal()`, and the new runtime's cache must match the DB.

**Trap for restart tests.** A transaction must never be open across a pause (no hook lies between BEGIN and COMMIT). A future dropped inside a transaction leaves a connection whose rollback sqlx performs lazily, and its fence S lock would make the next `connect` wait for the 5 s lock-wait timeout and retry.

**Known gap, stated in the report.** Because no hook lies between BEGIN and COMMIT, two crash points of design §11.1 are not exercised on the real code: a crash with the DB transaction open before COMMIT, and a lost COMMIT acknowledgement (H17). The S2 guard-miss rollback test covers only the SQL side of the first. The report lists both under "Test results" as not covered, with the reason, so the formal implementation can add a fault-injecting connector for them.

---

## 4. In-process test infrastructure and the T1-T11 mapping

**Infrastructure.** `tests/job_tree_infra.rs` and `tests/job_tree_test.rs` are included from `tests/test_spider_storage.rs` behind `#[cfg(feature = "test-hooks")]`, so the existing binary is unchanged without the feature. `job_tree_infra.rs` and `TestEnv` arrive in S2. Later slices extend `TreeSpec` and `TestEm`.
- `TestEnv::start() -> TestEnv`:
  - a fresh `ScratchDatabase` (section 0, item 10), so this test's recovery, repair and resend only ever see its own rows;
  - a `RuntimeConfig` from `scratch.config()`;
  - `create_runtime_with_hooks`, then `resend_ready_tasks` (a no-op on an empty database, kept for symmetry with `restart`);
  - the `ServiceState`, the hooks, and a raw `sqlx::MySqlPool` on the same DB for row assertions (`SELECT id, parent_id, root_id, state FROM jobs WHERE id = ? OR root_id = ?`).
- `TestEnv::finish(self)` stops the runtime and drops the scratch database. Each test calls it at the end.
- `TestEnv::restart(self)` follows section 3, including the resend. Variants are `crash_at(point, filter)` and `zombie()`.
- `TreeSpec`: a builder of job graphs whose tasks carry the node label in `tdl_context.task_func` (for example `"n17"`), with optional commit and cleanup tasks, `max_num_retry`, `max_num_instances`, and per-node children (each child is a `TreeSpec`). `TreeSpec::compress()` gives the `ChildJobSubmission` bytes.
- `TestEm` drives `ServiceState` directly, with no gRPC:
  - `poll_all(wait)` covers the three lanes.
  - `register(job, task_id) -> ExecutionContext`.
  - `succeed(job, index, instance)` looks up the node's children through the label and calls `succeed_task_instance_with_children`.
  - `fail(job, task_id, instance, msg)`.
  - `run_until_quiescent(root)` loops poll, register and report until the SQL count of non-terminal rows of the tree is 0, bounded by an iteration cap, not a timeout.
  - `TestEm` holds the set of job ids of the tree it currently drives (the root plus every child id read from the DB after each round of reports). A polled lane entry for any other job is ignored and counted in `TestEm::foreign_entries()`. Such entries are expected: a cancelled iteration leaves stale task entries in the lanes that nobody polled, and the next iteration on the same `TestEnv` receives them. A foreign entry is never registered or reported.
- Every race test is a loop of `JOB_TREE_RACE_ITERATIONS` iterations on fresh trees, and each iteration asserts exact final states, row counts and reply variants. Each iteration ends by driving its tree to quiescence: it releases held commit tasks (T7), runs the cleanup tasks of `CleanupReady` jobs (T2, T3, T7) and asserts that the scratch database has no non-terminal row. Nothing accumulates across iterations, and a T8 restart recovers only the tree under test.

**Mapping.**

| Test | Slice | Shape and interleaving | Exact asserts |
|---|---|---|---|
| T1 two tasks fail the same job | S3 (plain), S5 (in a tree) | Two tasks of J with `max_num_retry = 0` fail. Same round: the *same round* recipe. Different rounds: the *consecutive rounds* recipe. In a tree, J is a leaf under a `BlockingOnChildren` chain. | One reply `Failed`, the other stale (`JobNoLongerRunning`); `error_message` is the first failure's; ancestors `Failed` with "descendant job … failed"; every other abortable job `CleanupReady`/`Cancelled`; exactly one row update per job (the `EventSubmitted`/apply observation counts) |
| T2 root cancel vs leaf registration | S5 | Leaf L's last task reports children while the root is cancelled: (i) same round, (ii) registration in flight then cancel, (iii) cancel committed then the report | (i) `COUNT(parent_id = L) = 0`, L deactivated, the registration reply stale; (ii) k children inserted and then all deactivated in the cancel's round; (iii) phase A stale or the event stale; in all cases I4 holds and no child is `Running` |
| T3 root cancel vs leaf failure | S5 | Same round and both orders across rounds | Same round: root `CleanupReady`/`Cancelled`, the failure reply stale, the failure message not stored. Failure first: root `Failed` and the cancel stale (`JobAlreadyTerminated`) |
| T4 siblings complete together | S4/S5 | k = 2, 8, 64 leaves under P; last reports forced into one round, and into k rounds | P advances exactly once (one apply observation for P); the counter reaches 0 once; P `Succeeded` or `CommitReady`; every child's outputs are in its row and P's own outputs unchanged |
| T5 straggler fails after success | S2 (plain), S5 (leaf) | `max_num_instances = 2`; instance i1 succeeds; i2's failure arrives before the apply (pause `CommitterBeforeDbTransaction`) and after it | Failure reply `TaskAlreadyTerminated` or `JobNoLongerRunning` (stale); job state unaffected; no fatal flag |
| T6 parent with a commit task | S4 | P with a commit task and 3 children | P `BlockingOnChildren` → `CommitReady` (one commit publication) → commit run → `Succeeded` → root climbs; P's commit sees only P's outputs |
| T7 cancel vs `CommitReady` | S5 | Non-root P `CommitReady` (its commit registered and held), then cancel the root; and a `CommitReady` root with and without a cleanup task | P stays `CommitReady`, its commit then gives `Succeeded` with no propagation; the root `CleanupReady`/`Cancelled`; root case: `CleanupReady` or `Cancelled` via the new edge, and a late commit success stale |
| T8 restart at crash points | S6 | `Crash` at `CommitterBeforeDbTransaction`/`CommitterAfterCommit` for a registration; at `CommitterAfterApplyJob` mid-cascade; at `CommitterAfterCommit` for a cancel; at `RecoveryBeforeRepair`/`RecoveryAfterRepair` (forced repair state); plus the zombie | The tree converges after `TestEm::run_until_quiescent`; `COUNT(parent_id = P) = k` exactly (nothing registered twice); after the restart's `resend_ready_tasks`, the recovered `Running` children's tasks are polled (so a missing resend fails the test by its iteration cap), while the first full poll of all three lanes contains no entry for the recovered `BlockingOnChildren` parent; that parent has no regular tasks and unblocks when its children finish; the cancel case recovers only `CleanupReady` and skipped jobs; the second restart logs no repair (the `Observe@RecoveryAfterRepair` context is empty) |
| T9 dropped RPC future | S2 | Three drop points, each its own case: (a) `Pause@PhaseAAfterTcbFlip` (after the TCB success, before `fetch_sub`); (b) `Pause@PhaseABeforeSuccessorLock` (inside the TCB success, at the successor-lock await) on a job with two dependent tasks; (c) `Pause@CreateInstanceBeforePoolSend` (between the TCB registration of an instance and its pool send). The caller's task is `abort()`ed while paused; `release` | The handler bodies are spawned into the tracker (S2), so the abort must not cut any of them short: in every case the effect equals that of an RPC that completed. (a) the job reaches `Succeeded`/`CommitReady` with no further RPC; (b) the outputs are stored, the successor is published exactly once (one lane entry), and the job finishes when the successor reports; (c) the instance is in the task-instance pool and its later report succeeds. In every case the DB equals the cache and `is_fatal()` is false |
| T10 limit violation | S4 | Over-count, over-bytes, over-depth children, outputs over 4 MiB | Job `Failed` (or after the retries); `is_fatal() == false`; no child rows |
| T11 planner property | S3, extended in S4 and S6 | In-memory row model (`HashMap<JobId, Row { parent, root, state, has_commit, has_cleanup }>`); random tree shapes (depth ≤ 5, fan-out ≤ 4, random flags); a forward simulation to reach random consistent states; random batches of events that were valid at some earlier snapshot (so they may be stale now), including registrations with `ChildSetSummary`; random round boundaries; seeded `rand` (dev-dependency), seeds printed on failure, `JOB_TREE_PROPERTY_ROUNDS` (default 20,000 in release) | After every round: every update's `from` equals the model row (no guard miss); no row twice; at most one registration; all-stale gives an empty plan; every event either applied, stale or deferred; I1-I4 on the model (the full I2 needs every row, including terminal children); `check_invariants` agrees with the model's verdict; S6 adds `check_invariants(plan_repair(rows)) == Ok` for injected sets |

CI runs tests with `--release`, so `debug_assertions` are off. The property tests call `check_invariants` explicitly, and the test agent should also run them once in a debug build.

---

## 5. EM harness for the benchmarks

**Location and build.**
- `/home/lzh/dev/spider/claude/benchmarks/job-tree-proto/` is a standalone Cargo project with its own `[workspace]`, `Cargo.lock` and `target/`, like `claude/benchmarks/job-tree/`.
- It depends by path on the **worktree's** `spider-core`, `spider-proto-rust` and `spider-client`, plus `tonic`, `prost`, `tokio`, `sqlx = "=0.8.6"` (`mysql`, `runtime-tokio`), `zstd`, `serde`/`serde_json`, `clap` and `rand`. All versions are pinned to the ones in the repo's `Cargo.lock` so `--offline` works.
- It copies `stats.rs` (percentiles, distribution-free median CIs) and `host.rs` (the idle gate and environment capture) from `claude/benchmarks/job-tree/src/`.

**Binaries under test.** Each is built into its own target directory so nothing in either checkout changes. `build.sh`:
- **main**: `cd /home/lzh/dev/spider && CARGO_TARGET_DIR=<harness>/target-main cargo build --release -p spider-storage`. Only `target-main/` is written; no tracked file changes.
- **proto**: `cd /home/lzh/dev/spider-job-tree-proto && cargo build --release -p spider-storage`, in the worktree's own target.
- **proto-unboxed**: the same with `--features unboxed-round`, into `<harness>/target-proto-unboxed`.
- **proto-stats** (B4, B5 only): `--features committer-stats`, into `<harness>/target-proto-stats`.

**Database.**
- `db-up.sh`/`db-down.sh` follow the old harness: a container `job-tree-proto-bench-mariadb` on a port from `tools/scripts/get_free_port.py`, image `mariadb:10.11.16` (the deployment pin; `start.py` hard-codes `mariadb:latest`), always stopped by `stop.py` from an EXIT/INT/TERM trap.
- Each trial gets a fresh database: `CREATE DATABASE bench_<variant>_<trial>` through the root account, then `GRANT ALL` to `spider-user`. main and the prototype never share a schema: the prototype's upgrade alters `jobs`.
- Variants alternate within a group (A, B, A', B', …) to cancel drift.

**Storage process control** (`src/storage_proc.rs`):
- It writes a `ServerConfig` YAML, identical for both binaries within an experiment: host and port, the DB, `max_connections` (64 unless the experiment sets it), `task_instance_pool.execution_manager_stale_cutoff_sec = 3600`, and the inbound-queue capacities. B1-B5 use the defaults (`task_capacity` 65,536 is far above their live entries). B6 and B7 use the deployed `task_capacity = 1,048,576` (`tools/deployment/spider-helm/values.yaml:81`, `spider-compose/.env.example:26`), because with 100,000 started jobs the default lane overflows: on main, `start` awaits `send_task_ready` under the JCB write lock (`cache/job.rs:258-265`), so the 65,537th `StartJob` would block forever, and on the prototype the overflow would sit in the unbounded staging queue and inflate the RSS that B6 measures. Every run's YAML is copied into the results directory.
- It spawns the binary with `RUST_LOG=info` for B2-B5, and `RUST_LOG=warn` for B1 (latency), B6 and B7 (see B7), on both binaries. The stderr JSON log goes to a file.
- It waits for readiness by retrying `GetSession`, which succeeds only after recovery and `serve`.
- It samples `VmRSS`/`VmHWM` from `/proc/<pid>/status`.
- It stops with SIGTERM (graceful) or SIGKILL (B7 crash).

**The scheduler and EM emulator** (`src/em.rs`):
- One `RegisterExecutionManager` per harness EM identity, with a heartbeat task every 10 s.
- **Pollers**: one task per lane loops `PollReadyTasks`/`PollReadyCommitTasks`/`PollReadyCleanupTasks` with `max_items = 256` and `wait = 1 ms`. `recv_batch` waits the whole interval unless full (`inbound_queue.rs:351-376`), so a short wait bounds the added latency, and it is identical for both binaries. Entries go to a dispatcher.
- **Dispatcher**: routes each `(job_id, task_id)` to a worker pool of size `W`, or to a per-job oneshot when a benchmark waits for a specific job (B1).
- **Worker**:
  1. `RegisterTaskInstance(session, job, task, em)`.
  2. Decode the node label from `tdl_context.task_func`.
  3. Look up the node's behaviour in the run's `Plan`: no-op, sleep 10 ms, fail with `max_num_retry = 0`, report children (each child's graph and inputs are pre-compressed per node label), or hold until a barrier (B3, B4).
  4. Send `ReportTaskSuccess { serialized_outputs, children }` or `ReportTaskFailure`, timing the RPC.
  - A `FAILED_PRECONDITION`/`NOT_FOUND` reply is counted, not fatal: stale assignments after a cancel are expected.
- **Job RPCs**: `spider-client`'s `SpiderClient` for `RegisterJob`, `StartJob`, `CancelJob` and `GetJobState`, with its retry wrapper disabled or noted. Task graphs come from `spider-core` (one bytes input and one bytes output per task, small payloads) and are compressed with `spider_core::compression`.
- **Drain-only poller** (B6, B7): one task per lane that polls with `max_items = 256` and discards the entries, counting them. It never registers an instance, so the jobs stay live (`Running`, no instances).
- **Quiescence is derived from the harness's own replies, not from a DB poll during the measurement.** Every abort round answers its event only after the apply (design §7.2), so:
  - with no cleanup tasks, the tree is quiescent when the `CancelJob` reply (or the failing `ReportTaskFailure` reply) returns;
  - with cleanup tasks, it is quiescent when the last of the expected cleanup reports returns. The harness knows the expected set: every non-terminal job of the tree at the abort, since all of them have a cleanup task in that variant.
  
  After the measured interval ends, one DB probe validates the claim: `SELECT COUNT(*) FROM jobs WHERE (id = ? OR root_id = ?) AND state NOT IN ('Succeeded','Failed','Cancelled')` must be 0. A non-zero count marks the trial invalid, and the harness records it. No query runs against `jobs` while a measured interval is open.
- **Connection probes**, on a dedicated root connection:
  - every 5 ms: `SELECT COUNT(*) FROM information_schema.PROCESSLIST WHERE USER = 'spider-user' AND DB = ? AND COMMAND <> 'Sleep' AND ID <> CONNECTION_ID()`;
  - per trial: `FLUSH STATUS` before the trial and `SHOW GLOBAL STATUS LIKE 'Max_used_connections'` after it (the peak number of open connections, including idle pool connections);
  - every 5 ms: `Threads_running` from `SHOW GLOBAL STATUS`, minus 1 for the probe.
  
  The 5 ms samples miss transactions of about 1 ms, so the report presents them as a lower bound on active connections, next to the peak.
- **Child counts** are read once per trial after quiescence.

**Protobuf compatibility.**
- The harness is compiled once against the prototype's `spider-proto-rust`: field 7 `children` and enum value 8.
- With main, plain-job runs leave `children` empty. proto3 encodes an empty repeated field as nothing, so main's decoder sees byte-identical requests. main never sends `BLOCKING_ON_CHILDREN`.
- The hazard runs the other way: main would silently *ignore* a non-empty `children` (an unknown field), so a tree workload against main would "succeed" as a plain job and produce a wrong number. The harness refuses any plan with children when `--target main`, and every tree trial asserts the final child-row count from the DB.

**Experiments** (≥ 10 trials each on a quiet host, with the idle gate before every trial; the kind cluster stays stopped):

| | Variants | Workload | Metrics |
|---|---|---|---|
| B1 | main, proto; A/A main-main and proto-proto | Plain 1-task jobs, no-op. `C = 1`: 2,000 jobs after 200 warm-up. `C = 64` writers: 64 × 300 jobs. Each writer: `RegisterJob`, `StartJob`, await its routed entry, `RegisterTaskInstance`, timed `ReportTaskSuccess` (the last task, so the job transition), done | Last-report latency p50/p99; end-to-end `RegisterJob` start → report return (the reply follows the apply, so the job is `Succeeded` in the cache); ratio proto/main with a CI against the A/A band; throughput at `C = 64` |
| B2 | proto (trees); main and proto for (a) and (b) | (t) depth-7 binary tree, 127 jobs (internal nodes: 1 task reporting 2 children; leaves: 1 task); (a) one job with a 127-task binary-shaped graph (each task depends on its parent task); (b) one flat job with 64 tasks; no-op and 10 ms tasks; `W = 64` | End-to-end `RegisterJob` → the return of the final leaf report (the cascade is applied before that reply), validated by a final `GetJobState = Succeeded`; per-level overhead `(t − a) / 7`; registration report latency per level |
| B3 | proto | The depth-7 tree fully expanded: all 64 leaf instances registered and held. (i) `CancelJob(root)`; (ii) a leaf failure with `max_num_retry = 0`; variants with no cleanup tasks and with cleanup tasks on every job (cleanups run by the harness) | Cancel acknowledgement = `CancelJob` RPC latency; time to quiescence (from the harness's replies, validated by one DB probe afterwards); failure: the report latency, which is the root-`Failed` acknowledgement, plus the time to quiescence |
| B4 | proto-stats; main reference: 1,000 plain jobs | The root's task reports 1,000 children (registration latency measured); all 1,000 child instances registered and held; the success reports released together, with concurrency 64 and 1,000 | Completions/s = 1,000 / (last reply − first send); report p50/p99; events per tree state transaction (histogram from `committer-stats`); main reference: 1,000 independent plain jobs' last reports released together |
| B5 | proto-stats | K = 1, 16, 64 concurrent depth-7 trees, no-op; `max_connections` 64, and 32 to exceed the pool at K = 64 | Per-tree end-to-end distribution vs K = 1 (isolation); max and mean active DB connections; rounds and events per tree |
| B6 | main, proto, proto-unboxed | 100,000 plain jobs registered and started by 64 concurrent clients, never registered as instances; `task_capacity = 1,048,576`; the drain-only poller runs throughout | RSS sampled once the poller has drained exactly N entries (lane occupancy 0 on both binaries, staging queue empty on the prototype) and RSS is stable, minus RSS before, per job; the drained count and the sample time are recorded with each RSS sample; plus `size_of_val` of the committer future (section 6) and the per-tree channel's first block, to explain the delta |
| B7 | main, proto | N = 10,000 and 100,000 live plain jobs (`Running`); proto also N/11 trees of 1 `BlockingOnChildren` parent with 10 `Running` children; `task_capacity = 1,048,576`; `RUST_LOG=warn` on both binaries; SIGKILL, then restart with the drain-only poller attached as soon as `GetSession` succeeds | Time from spawn to the first `GetSession` success (recovery runs inside `create_runtime` on both binaries, before `serve`); time from spawn until the drain-only poller has received every expected resent entry (N on main; N + 10 × ⌊N/11⌋ on the prototype: the plain jobs plus the trees' `Running` children, while the `BlockingOnChildren` parents publish nothing); the drained count is recorded. The per-job `"Job recovered into cache."` `info` log (`state/runtime.rs:238-242`) is kept identical in the prototype, and `warn` suppresses it on both, so logging does not dominate the span |

**In-process supplement for B1.** The existing `jcb_test` instrumentation (`SPIDER_TEST_INSTRUMENT_OUTPUT_DIR`) on both checkouts gives per-call JCB latencies with the MariaDB connector. It is cheap, and it isolates the committer hop from gRPC.

**`committer-stats` feature** (non-default, prototype only): two `AtomicU64` counters (rounds, events), a 9-bucket histogram of events per round (1, 2-3, 4-7, …, 256) and DB-transaction durations. It logs `"Committer statistics."` every 10 s and at `Runtime::stop`. The harness parses the JSON log. Count these lines separately in the report.

Outputs go to `<harness>/results/<run-id>/` (CSV per experiment plus `env.txt`). The analysis script prints the tables the report needs.

---

## 6. Boxing the per-round future

**Where it matters.**
- A committer is a tokio task that lives as long as its tree. For a plain job, it is parked in `recv_many` for most of the job's life.
- The task's allocation is the `run_committer` future, whose size is the maximum over its await points of the live locals plus any inlined child future.
- If `run_tree_state_transaction(..).await` is inlined, every parked committer carries the storage for the round's deepest state machine: `collect_inputs`, the JCB `RwLock` acquire futures, the per-job `apply_tree_state` futures (which inline `TaskGraph::cancel_non_terminal`'s TCB lock futures), `JobCache::insert_many`, and the hook futures.
- At 100,000 live plain jobs, every KiB of that is about 100 MB of RSS.

**What is already boxed.**
- `apply_tree_transition` is an `#[async_trait]` method, so its future is a `Pin<Box<dyn Future>>` and the SQL state machine is never inline.
- The spawned handler bodies are tracker tasks (boxed by tokio) that live for one RPC.
- The planner is synchronous.

So the inline excess is the cache-side futures only, probably a few hundred bytes to a few KiB. B6 measures it instead of guessing.

**Code.**

```rust
#[cfg(not(feature = "unboxed-round"))]
let result = Box::pin(run_tree_state_transaction(&mut state, events, &env)).await;
#[cfg(feature = "unboxed-round")]
let result = run_tree_state_transaction(&mut state, events, &env).await;
```

`unboxed-round` is a non-default feature of `spider-storage`. CI's `--all-features` builds and tests it too, which is harmless because behaviour is identical.

**Measurement.**
- A unit test, `committer_future_size` in `state/tree_committer.rs`, prints `std::mem::size_of_val(&run_committer(..))` for the configured variant. Run it with `-- --nocapture` once per feature set.
- B6 compares RSS per live job for main, proto and proto-unboxed.
- Also record the fixed per-tree costs that the design's "a few hundred bytes" (design §7.2, §13) omits:
  - tokio's unbounded mpsc allocates its first block of 32 `Pending` slots when the channel is created;
  - the `TreeStructure` `HashMap` and the handles map;
  - the extra JCB fields.
- If boxing does not measurably reduce RSS, the report says so, and the formal implementation may inline.

---

## 7. Risks and traps

| # | Risk or trap | Mitigation in this plan |
|---|---|---|
| R1 | A single fence row would make concurrently running test processes fence each other (nextest, a shared `spider-db`) | Per-config `fence_id` (section 0, item 6); the restart tests share one config |
| R2 | The test DB user lacks `PROCESS`, and `start.py` uses `mariadb:latest` | The watcher degrades to the lock-wait-timeout path; the benchmarks pin `mariadb:10.11.16`; FORCE INDEX and deadlock results are only claimed for 10.11.16 |
| R3 | `collect_inputs` reading the outputs of a job that a cancel already deactivated would raise `TaskInputNotReady` (internal, fatal) | Read the outputs and take the prepared children only when the collected state is `Running`; otherwise drop the children and let the planner answer stale |
| R4 | A JCB method that awaits its reply while still holding the guard deadlocks against the committer's apply | Every submit follows an explicit `drop(guard)`; the reviewer checks each of the six call sites |
| R5 | Sends moved after the guard drop now run against a JCB that a cancel may apply in between | Accepted by the design: stale hints are rejected at instance creation (design §10.6 (d)) |
| R6 | `Relaxed` instead of `AcqRel` on `num_incomplete_tasks`, or the `has_prepared_children` store placed after the `fetch_sub`, would lose a sibling's outputs or children | The store goes in the TCB section, before the RMW; the reviewer checks the ordering; T4 exercises it but cannot prove it |
| R7 | An open transaction across a hook pause, or a dropped runtime with a lazily rolled-back connection, keeps a fence S lock and stalls the next `connect` | No hook between BEGIN and COMMIT; the restart helper releases every pause and drops every `ServiceState` clone before reconnecting |
| R8 | Without `debug_assertions`, `check_invariants` never runs in CI (release builds) | The property tests call it directly; one debug-build run of the S3-S6 tests |
| R9 | Between S4 and S6 a restart mis-recovers trees | No tree restart test before S6 (section 1) |
| R10 | The planner needs to know whether a `RegularTasksDone` registers *before* collecting | The `registers` flag set in phase A (S4); see U1 |
| R11 | Per-tree memory exceeds the design's estimate (the mpsc first block, the maps) | Measured in B6 and reported |
| R12 | A grouped `TerminalFailure` touches many rows (`abort_rest`), so the twice-touch rule defers it behind a completion that touched the parent | Correct by arrival order; it costs one extra round; T1, T4 and B3 observe it |
| R13 | `MockDbConnector::register` returns `JobId::random()`, and its insert ids could collide | Mock inserts use a dedicated id range |
| R14 | The publisher's lane tasks need a tokio context in `ServiceState::new` | Every caller is inside a runtime; documented in the docstring |
| R15 | The `DbConnectorType` parameter of the JCB is now phantom | Left in place (section 0, item 5); listed as a cleanup for the formal implementation |
| R16 | An end-to-end measurement through gRPC includes the 1 ms lane-poll granularity | Identical for both binaries; B1 also reports the report RPC latency alone, plus the in-process `jcb_test` supplement |
| R17 | The spider-scheduler's storage client and round-robin code never saw `BlockingOnChildren` | Out of scope (no scheduler in the benchmarks); listed for the formal implementation |
| R18 | The harness running a tree workload against main would silently ignore `children` | The harness refuses it, and every tree trial checks child rows |
| R19 | `collect_inputs` running the "no ready tasks" check for a `Start` whose job is no longer `Ready` (a duplicate or retried `StartJob`, or one after a committed cancel) raises the internal `TaskGraphCorrupted`, which poisons the tree and exits the process | Read the ready indices and run the check only when the collected state is `Ready`; otherwise `plan_start` answers `Stale(JobAlreadyStarted)` (S2, with a test) |
| R20 | Recovery, repair and resend act on every row of the database they connect to, so tests that share `spider-db` would repair or republish each other's trees | Every test that recovers, restarts or creates a tree uses its own `ScratchDatabase` (section 0, item 10); `TestEm` ignores foreign lane entries; each race iteration ends quiescent |
| R21 | A failed `create_runtime` after the pool, GC, publisher or committers were spawned leaks them with a live token and connector clones | `build_runtime` cancels, closes and joins everything it spawned before returning `Err` (S2, with a test) |
| R22 | A benchmark that starts more jobs than `task_capacity` blocks `StartJob` forever on main and inflates the staging queue on the prototype | B6 and B7 use the deployed 1,048,576 on both binaries, and a drain-only poller, with the drained count recorded |
| R23 | A `Pause` left armed when the token is cancelled would hold its body, and a phase-A read guard, until the 30 s stop timeout | `Pause` races the token; dropping the hook or `release_all()` frees every waiter; `TestEnv::restart` calls `release_all()` first |

---

## 8. Underspecified design sections, and the choices this prototype makes

| # | Design section | Gap | Prototype choice (report as a deviation if it changes behaviour) |
|---|---|---|---|
| U1 | §7.5 rule 3, §7.2 `take_events` | Whether a `RegularTasksDone` is a registration is only known after `collect_inputs`, but `take_events` must order it last and admit only one | Phase A sets `has_prepared_children` on the JCB, and the last task's event carries `registers`; a second registering parent stays in the buffer uncollected |
| U2 | §6.1, §7.2, §7.8, N14 | The full mapping of `TreeClosed`, `Stopping` and a closed oneshot to `CacheError`, `StorageServerError` and gRPC codes, given that `cache/` cannot name `StorageServerError` | `CacheError::ServiceStopping` → `StorageServerError::Stopping` → UNAVAILABLE; `InternalError::TreePoisoned` → INTERNAL (fatal, idempotent); `TreeClosed` → `TreeEventKind::stale_error(cached_state)` |
| U3 | §5.1, §9.5 | `TreeState` in `cache/` holds generic JCB references; the planner's purity and the property tests need a non-generic model | `TreeStructure` (non-generic, in `cache/tree.rs`) plus the committer-owned handles map |
| U4 | §5.4, §9.5 | "The JCB methods lose their DB calls", but the JCB is generic over `DbConnectorType` | Phantom parameter (R15) |
| U5 | §7.3 step (1) | Collect for `RegularTasksDone` of a job that is no longer `Running` | Skip the outputs and drop the children (R3) |
| U6 | §11.2 vs today's `create_runtime` | Committers need the GC, the publisher and the tracker before recovery, but today recovery runs before the GC is created | Reorder `create_runtime` (S2) |
| U7 | §11.3, Q11 | Behaviour without the `PROCESS` grant is not specified | Warn once and fall back to the 5 s timeout with 5 retries |
| U8 | §9.2 | One fence row assumes one storage deployment per database; tests share one | `fence_id` (section 0, item 6); production 0 |
| U9 | §16.1 PR 3 | "Old `DbStorage` methods remain as wrappers" does not say how their error contracts survive exact guards | Read the state first and keep today's error mapping (section 0, item 8) |
| U10 | §6.2 step 1, §13 | What exactly the per-child and per-report byte limits count | The compressed graph plus the compressed inputs, as received |
| U11 | §6.2 step 6 | Where a child's initial ready task indices come from (today `start` reads them under the write lock) | Computed at prebuild time into `PreparedChild.ready_task_indices` |
| U12 | §9.4 | Byte-chunking the outputs CASE UPDATE is stated, but not where the cut happens relative to the 500-id cut | Cut at whichever limit binds first; at least one row per statement |
| U13 | §8.2 I2, §7.4 `check_invariants` | `check_invariants(&TreeState, states)` cannot verify "every child `Succeeded` or active", because terminal children are not in `TreeState` | `check_invariants` checks what the structure can see; the full I2 is checked in the property tests' row model, and at recovery via `has_failed_child` |
| U14 | §6.5 | `CommitReady → CleanupReady` for a *recovered* `CommitReady` root needs its cleanup TCB (#513), which the design puts in PR 0 | Fixed in S6, together with `recover_blocked`; before S6 a recovered `CommitReady` root keeps today's behaviour |
| U15 | §16.2 | The `job_tree.enabled` flag | Omitted (section 0, item 9) |
| U16 | §15, §11.1 | How to emulate a crash in-process | Hook `Crash` → the real poison path → rebuild the runtime from the DB (section 3) |

---

## 9. Working rules for the implementation agents

- Work only in `/home/lzh/dev/spider-job-tree-proto`, and never commit, stash, reset or switch branches. `git diff 278d06f` is the deliverable.
- Follow Pleiades's Rust style guide:
  - `expect` with a message, never `unwrap`, including in tests;
  - full docstrings with `# Errors` naming each forwarded function;
  - no visual-divider comments;
  - exact dependency versions, sorted;
  - log messages capitalized with a period, error messages not.
- Update docstrings that a change makes wrong (for example `get_recoverable_jobs`, `succeed_task_instance`'s "commits the job outputs", `finalize`), but do not add comments that merely narrate new behaviour.
- After each slice:
  - `cargo build --workspace --all-targets --all-features`;
  - `task lint:fix-rust` in the worktree (or `cargo clippy --workspace --all-targets --all-features -- -D warnings` plus nightly fmt);
  - the slice's verification list.
- MariaDB for tests:
  1. `port=$(tools/scripts/get_free_port.py)`.
  2. `tools/scripts/mariadb/start.py --name jtp-<agent>-<slice> --port $port --database spider-db --username spider-user --password spider-password`.
  3. Run with `MARIADB_PORT=$port MARIADB_DATABASE=spider-db SPIDER_STORAGE_DB_USERNAME=spider-user SPIDER_STORAGE_DB_PASSWORD=spider-password MARIADB_ROOT_PASSWORD=spider-password cargo nextest run -p spider-storage --all-features --run-ignored all --release`, in the background, polling.
  4. Always `tools/scripts/mariadb/stop.py --name jtp-<agent>-<slice>`, also on failure.
- Do not start the kind cluster.
- Record per-slice notes (what deviated, test results) in `/home/lzh/dev/spider/claude/job-tree/prototype/slices/<key>.md` for the report.

---

## 10. Resolution of `plan-review.md`

Every issue is resolved by a change to this plan; none is rebutted.

| Issue | Resolution | Where |
|---|---|---|
| M1 a stale `Start` poisons the service | Ready indices and the "no ready tasks" check only when the collected state is `Ready`; otherwise `Stale(JobAlreadyStarted)`. Tests: `StartJob` twice, `CancelJob` then `StartJob`, `is_fatal()` false | S2 round step 1, S2 verification 4, R19 |
| M2 tests share one database with a recovery that loads every row | `ScratchDatabase` per `TestEnv` and for every test that recovers, restarts or creates a tree, with a root-password fallback that works in CI; `TestEm` ignores and counts foreign lane entries; each race iteration ends quiescent; the `file_serial` facts corrected | section 0 items 6 and 10, S1, section 4, R20 |
| M3 the in-process restart skips part of the real recovery path | (1) `TestEnv::start`/`restart` call `resend_ready_tasks` after the runtime is built, in `main`'s order, and T8 relies on it; (2) `create_runtime` and `create_runtime_with_hooks` both delegate to one `build_runtime`; (3) `build_runtime` cancels, closes and joins everything it spawned on an error, with a test | S2 service and runtime, S2 verification 6, section 3 restart steps 4-5, T8 row, R21 |
| M4 B6/B7 overflow the default task lane | `task_capacity = 1,048,576` on both binaries for B6 and B7, a drain-only poller, and the drained count recorded with each RSS sample | section 5 storage process control, drain-only poller, B6 and B7 rows, R22 |
| m1 `Pause` does not race the token | `Pause` races the token; drop and `release_all()` close the semaphores; `restart` calls `release_all()` first | section 3 actions and API, R23 |
| m2 the upgrade alters on every connect | Read `COLUMNS` and `STATISTICS` first and alter only what is missing; the test counts DDL statements through a feature-gated accessor | S1 `upgrade_jobs_schema`, S1 verification 3 |
| m3 mock-based tests will fail | Seed `db.states` in the two hand-built-JCB tests; the getter fall-through keeps the two DB-fallback tests unchanged; `JobCacheGcHandle::new` becomes `pub`, listed as a library change | S2 compatibility, S5 compatibility |
| m4 the root check is stricter than the design | Only the non-root rejection is added; each RPC keeps its own miss behaviour (`start_job` stays cache-only, the getters read the DB, `cancel_job` keeps its fatal arm) | S5 `service.rs`, S5 verification |
| m5 `create_runtime_with_hooks` arrives too late | Introduced in S2 together with `TestEnv` | slice table, S2 |
| m6 `EventSubmitted` cannot await `hit` | A synchronous `observe` path with atomic counters and `Notify`; only `Observe` may be armed there | section 3 points and actions |
| m7 T9 covers one drop point | New points `PhaseABeforeSuccessorLock` and `CreateInstanceBeforePoolSend`; T9 has three cases with exact asserts | section 3 points, T9 row |
| m8 two crash points are not exercised | Stated as a known gap that the report lists | section 3 "Known gap" |
| m9 the DB probes disturb what they measure | Quiescence from the harness's replies, validated by one probe after the interval; connection probes on a root connection, excluding the probe, plus `Max_used_connections` and `Threads_running` | section 5 probes, B3 row |
| m10 children on a commit or cleanup success are dropped | Rejected with `INVALID_ARGUMENT`, with a test | S4 `grpc.rs` |
| m11 B7 includes per-job logging | `RUST_LOG=warn` on both binaries for B7; the per-job log is kept identical | section 5 storage process control, B7 row |

Final slice list (unchanged keys and order; S2 grows by the runtime cleanup, `TestEnv` and the stale-start fix):

1. `s1-schema-fence`
2. `s2-size1-committer`
3. `s3-planner`
4. `s4-registration`
5. `s5-propagation-cancel`
6. `s6-recovery`

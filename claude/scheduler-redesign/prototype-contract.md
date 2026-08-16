# `spider-scheduler-new` Prototype Contract

This document is the **shared contract** for the prototype implementation of the resource-group-aware
scheduler core described in [design.md](design.md). Every agent working on the prototype builds
against the module layout, type signatures, and file ownership recorded here.

The design doc is normative for *behaviour*. This document is normative for *structure*: where code
lives, who owns which file, and what the cross-module signatures are.

## 1. Crate location and wiring

- Path: `components/spider-scheduler-new`
- Package name: `spider-scheduler-new`, lib name `spider_scheduler_new`
- Added to `Cargo.toml` `[workspace] members` **and** `default-members`, both alphabetically sorted
  (after `spider-scheduler`).
- Added to `[workspace.dependencies]` as `spider-scheduler-new = { path = "components/spider-scheduler-new" }`.

The prototype **does not** depend on `spider-scheduler`. It reuses `spider-core` types only. It is a
parallel implementation, not a refactor of the existing crate.

### Dependencies

```toml
[dependencies]
async-channel, async-trait, dashmap, prost, serde, spider-core, spider-utils, thiserror,
tokio (macros, net, rt, rt-multi-thread, sync, time), tokio-util, tonic, tonic-prost, tracing

[build-dependencies]
tonic-prost-build

[dev-dependencies]
anyhow, rand, tokio (macros, rt-multi-thread)
```

All entries use `{ workspace = true }` and stay alphabetically sorted.

## 2. Module layout and file ownership

Each file has exactly one owning agent. **Do not edit a file you do not own.** If you need a change
in someone else's file, report it in your final summary instead of making it.

| File | Owner | Contents |
|---|---|---|
| `Cargo.toml`, `build.rs`, `proto/prototype_scheduler.proto` | Scaffold | Crate wiring and protobuf schema |
| `src/lib.rs` | Scaffold | Module declarations and re-exports |
| `src/config.rs` | Scaffold | `CoreConfig` |
| `src/error.rs` | Scaffold | All error enums |
| `src/types.rs` | Scaffold | `InboundEntry`, `FinalizeKind`, re-exports |
| `src/session.rs` | Scaffold | `SessionManager` |
| `src/storage_client.rs` | Scaffold | `SchedulerStorageClient` trait |
| `src/proto.rs` | Scaffold | Generated-code include and wire conversions |
| `src/resource_group.rs` | Core | `RgDispatchQueueReader`, `RgDispatchQueueEndpoints`, `ResourceGroupTable` |
| `src/dispatch_queue.rs` | Core | `GlobalDispatchQueue`, `DispatchService` |
| `src/job_registry.rs` | Core | `JobEntry`, `SharedJobEntry`, `JobRegistry` |
| `src/scheduling_unit.rs` | Core | `RgSchedulingUnit`, `try_make_assignment` |
| `src/core.rs` | Core | `Core`, the five-step tick, `run_core_on_dedicated_thread` |
| `src/harness/mod.rs` | Harness integrator | Harness module declarations, `Harness` bootstrap |
| `src/harness/fake_storage.rs` | Fake-storage agent | `FakeStorage`, `FakeStorageConfig` |
| `src/harness/grpc_service.rs` | gRPC agent | tonic service impl and server bootstrap |
| `src/harness/fake_worker.rs` | Worker agent | `FakeWorker`, `FakeWorkerConfig`, `WorkerReport` |
| `src/harness/metrics.rs` | Worker agent | `LatencySamples`, `DispatchRecord` |
| `src/tests/*.rs` | Unit-test agent | Crate-level unit tests |
| `tests/integration.rs` | Integration agent | Scaled integration scenarios |

Crate internals that unit tests must inspect are declared `pub(crate)`, mirroring the `pub(super)`
convention already used by `components/spider-scheduler/src/core_impl/round_robin`.

## 3. Scaffold-owned signatures

These are fixed. Implementers fill in bodies; they do not change signatures without reporting it.

### `src/types.rs`

```rust
pub use spider_core::types::id::JobId;
pub use spider_core::types::id::ResourceGroupId;
pub use spider_core::types::id::SessionId;
pub use spider_core::types::id::TaskId;
pub use spider_core::types::scheduler::TaskAssignment;

pub struct InboundEntry {
    pub resource_group_id: ResourceGroupId,
    pub job_id: JobId,
    pub task_id: TaskId,
}

pub enum FinalizeKind {
    Commit,
    Cleanup,
}
```

`FinalizeKind` maps to `TaskId::Commit` / `TaskId::Cleanup` when an assignment is built.

### `src/config.rs`

```rust
pub struct CoreConfig {
    pub dispatch_queue_capacity: NonZeroUsize,
    pub active_job_list_capacity: NonZeroUsize,
    pub ready_task_capacity: NonZeroUsize,
    pub commit_ready_task_capacity: NonZeroUsize,
    pub cleanup_ready_task_capacity: NonZeroUsize,
    pub storage_poll_timeout_ms: u64,
    pub tick_interval_ms: NonZeroU64,
}
```

`DOWNGRADE_LIVES: u32 = 1` is a crate-level constant in `src/core.rs`. α is fixed at 1 and does not
appear in the config.

### `src/storage_client.rs`

```rust
#[async_trait]
pub trait SchedulerStorageClient: Send + Sync + Clone + 'static {
    async fn poll_ready(&self, max_items: usize, wait: Duration)
        -> Result<(SessionId, Vec<InboundEntry>), StorageClientError>;

    async fn poll_commit_ready(&self, max_items: usize, wait: Duration)
        -> Result<(SessionId, Vec<InboundEntry>), StorageClientError>;

    async fn poll_cleanup_ready(&self, max_items: usize, wait: Duration)
        -> Result<(SessionId, Vec<InboundEntry>), StorageClientError>;
}
```

The commit/cleanup lanes return entries whose `task_id` is `TaskId::Commit` / `TaskId::Cleanup`.

### `src/session.rs`

```rust
#[derive(Clone, Default)]
pub struct SessionManager { /* Arc<AtomicU64> */ }

impl SessionManager {
    pub fn current(&self) -> SessionId;
    pub fn bump(&self, new_session_id: SessionId);
}
```

### `src/error.rs`

```rust
pub enum StorageClientError { Server(String), Transport(String) }

pub enum CoreError {
    Storage(#[from] StorageClientError),
    DispatchQueueClosed,
    InvalidSessionId(SessionId),
    Internal(String),
}

/// Returned by `RgSchedulingUnit::try_make_assignment`.
pub enum MakeAssignmentError {
    NoTask,
    DispatchQueueFull,
    DispatchQueueClosed,
}

pub enum JobEntryError { Finalized }

pub enum HarnessError { /* transport, bind, config */ }
```

### `proto/prototype_scheduler.proto`

```proto
syntax = "proto3";
package prototype_scheduler;

service PrototypeSchedulerService {
  rpc NextTask(NextTaskRequest) returns (NextTaskResponse);
}

message Void {}

message TaskId {
  oneof kind {
    uint64 index = 1;
    Void commit = 2;
    Void cleanup = 3;
  }
}

message CompletedAssignment {
  uint64 job_id = 1;
  TaskId task_id = 2;
}

message NextTaskRequest {
  uint64 execution_manager_id = 1;
  // Absent for a general execution manager; present for one pinned to a resource group.
  optional uint64 resource_group_id = 2;
  uint64 wait_time_ms = 3;
  optional CompletedAssignment completed = 4;
}

message Assignment {
  uint64 id = 1;
  uint64 resource_group_id = 2;
  uint64 job_id = 3;
  TaskId task_id = 4;
  uint64 session_id = 5;
}

message NextTaskResponse {
  oneof result {
    Assignment assignment = 1;
    Void no_task = 2;
  }
}
```

`build.rs` compiles this with `tonic_prost_build` into `OUT_DIR` (build client and server). `src/proto.rs`
does the `include!` and provides `TaskAssignment` ⇄ `Assignment` and `TaskId` ⇄ wire conversions.

## 4. Core-owned signatures

### `src/resource_group.rs`

```rust
#[derive(Clone)]
pub struct RgDispatchQueueReader(Arc<RgDispatchQueueReaderInner>);

struct RgDispatchQueueReaderInner {
    receiver: async_channel::Receiver<TaskAssignment>,
    living_hint: Arc<AtomicUsize>,
    rg_id: ResourceGroupId,
    session_id: SessionId,
}

impl RgDispatchQueueReader {
    pub(crate) fn rg_id(&self) -> ResourceGroupId;
    pub(crate) fn session_id(&self) -> SessionId;
    pub(crate) fn len(&self) -> usize;

    /// Blocks until an assignment arrives or `wait_time` expires. Used by a pinned execution
    /// manager. Does not touch `living_hint`.
    pub(crate) async fn recv_pinned(&self, wait_time: Duration) -> Option<TaskAssignment>;

    /// Decrements `living_hint` and then attempts one non-blocking pop. There must be no await
    /// point between the caller's hint receive and this call's decrement (design §8.2 req. 3).
    pub(crate) fn consume_hint_and_try_recv(&self) -> Option<TaskAssignment>;
}

#[derive(Clone)]
pub struct ResourceGroupTable(Arc<DashMap<ResourceGroupId, RgDispatchQueueEndpoints>>);

impl ResourceGroupTable {
    pub(crate) fn get_dispatch_queue_reader(&self, rg_id: ResourceGroupId, session_id: SessionId)
        -> RgDispatchQueueReader;
    pub(crate) fn get_or_create(&self, rg_id: ResourceGroupId, session_id: SessionId)
        -> RgDispatchQueueEndpoints;
    pub(crate) fn clear(&self);
}
```

Per-group queues are **unbounded**, as is the hint channel. No channel in this design is bounded by
its type: the dynamic threshold limits per-group occupancy and the publishing rule limits hints, so a
channel capacity would be a redundant second bound whose only reachable effect is to reject a send
the coverage proof requires to succeed.

### `src/dispatch_queue.rs`

```rust
/// The hint channel. Unbounded — the publishing rule caps each group's hints at that group's peak
/// occupancy, but those peaks are not simultaneous, so the total is not bounded by
/// `dispatch_queue_capacity` (design §8.1).
#[derive(Clone)]
pub struct GlobalDispatchQueue { /* async_channel Sender + Receiver of RgDispatchQueueReader */ }

/// The execution-manager-facing dispatch path.
#[derive(Clone)]
pub struct DispatchService { /* ResourceGroupTable, GlobalDispatchQueue, SessionManager */ }

impl DispatchService {
    pub async fn next_task_pinned(&self, rg_id: ResourceGroupId, wait_time: Duration)
        -> Option<TaskAssignment>;
    pub async fn next_task_general(&self, wait_time: Duration) -> Option<TaskAssignment>;
}
```

Both paths drop assignments whose `session_id` does not match `SessionManager::current()` and retry
within the remaining wait time.

### `src/scheduling_unit.rs`

```rust
pub(crate) struct RgSchedulingUnit { /* per design §3.4 */ }

impl RgSchedulingUnit {
    /// Publishes at most one assignment for this group. `free` is the tick's remaining free space,
    /// read but not modified here — the caller decrements it on `Ok`.
    pub(crate) fn try_make_assignment(
        &mut self,
        free: usize,
        session_id: SessionId,
        id_issuer: &TaskAssignmentIdIssuer,
        global_queue: &GlobalDispatchQueue,
        jobs_to_retire: &mut Vec<JobId>,
    ) -> Result<(JobId, TaskId), MakeAssignmentError>;
}
```

### `src/core.rs`

```rust
pub struct Core<StorageClientType: SchedulerStorageClient> { /* … */ }

impl<StorageClientType: SchedulerStorageClient> Core<StorageClientType> {
    pub fn new(
        config: CoreConfig,
        storage_client: StorageClientType,
        rg_table: ResourceGroupTable,
        global_queue: GlobalDispatchQueue,
        session_manager: SessionManager,
        reschedule_queue_reader: tokio::sync::mpsc::UnboundedReceiver<TaskAssignment>,
        cancellation_token: CancellationToken,
    ) -> Self;

    pub(crate) async fn tick(&mut self) -> Result<(), CoreError>;
    pub async fn run(self) -> Result<(), CoreError>;
}

/// Spawns `core` on a dedicated OS thread running a current-thread runtime and a `LocalSet`.
pub fn run_core_on_dedicated_thread<StorageClientType: SchedulerStorageClient>(
    core: Core<StorageClientType>,
) -> std::thread::JoinHandle<Result<(), CoreError>>;
```

**The core future is `!Send`.** It owns `Rc<RefCell<RgSchedulingUnit>>` and `Rc<SharedJobEntryInner>`
across await points, so it cannot be handed to `tokio::spawn`. It runs under a `LocalSet`. Inbound
polls are still issued with `tokio::task::spawn`, because those futures capture only the cloned
storage client and are `Send`.

## 5. Harness-owned signatures

### `src/harness/fake_storage.rs`

```rust
#[derive(Clone, Debug)]
pub struct FakeStorageConfig {
    pub num_resource_groups: usize,
    pub num_jobs_per_group: usize,
    pub num_tasks_per_job: usize,
    /// Emits a commit-ready entry for a job once all of its regular tasks have been reported
    /// complete.
    pub emit_commit_ready: bool,
}

#[derive(Clone)]
pub struct FakeStorage { /* Arc<Mutex<…>> */ }

impl FakeStorage {
    pub fn new(config: FakeStorageConfig) -> Self;

    /// The full set of regular tasks the configuration will ever emit.
    pub fn expected_regular_tasks(&self) -> HashSet<(JobId, TaskId)>;

    /// Reports a task as executed, so the commit-ready lane can advance.
    pub fn complete_task(&self, job_id: JobId, task_id: TaskId);

    /// Bumps the session and re-arms every task for replay, mirroring a storage restart.
    pub fn bump_session(&self) -> SessionId;

    pub fn total_regular_tasks(&self) -> usize;
}
```

`FakeStorage` implements `SchedulerStorageClient`. Job IDs are dense and deterministic:
`job_id = rg_index * num_jobs_per_group + job_index`, `resource_group_id = rg_index`, task IDs are
`TaskId::Index(0..num_tasks_per_job)`. A poll returns at most `max_items` entries and blocks up to
`wait` only when it has nothing to give.

### `src/harness/grpc_service.rs`

```rust
pub struct HarnessServer { /* … */ }

impl HarnessServer {
    /// Binds an ephemeral local port and starts serving.
    pub async fn start(dispatch_service: DispatchService, storage: FakeStorage)
        -> Result<Self, HarnessError>;

    pub fn endpoint(&self) -> String;
    pub async fn shutdown(self);
}
```

`NextTask` routes to `DispatchService::next_task_pinned` when `resource_group_id` is present and
`next_task_general` otherwise, and forwards `completed` to `FakeStorage::complete_task`.

### `src/harness/fake_worker.rs` and `src/harness/metrics.rs`

```rust
#[derive(Clone, Debug)]
pub struct FakeWorkerConfig {
    pub execution_manager_id: u64,
    /// `None` for a general execution manager.
    pub resource_group_id: Option<ResourceGroupId>,
    pub task_duration_ms: u64,
    pub next_task_wait_ms: u64,
}

#[derive(Debug)]
pub struct DispatchRecord {
    pub assignment_id: TaskAssignmentId,
    pub resource_group_id: ResourceGroupId,
    pub job_id: JobId,
    pub task_id: TaskId,
    /// Client-side latency: request send to response receipt.
    pub latency: Duration,
}

#[derive(Debug, Default)]
pub struct WorkerReport {
    pub execution_manager_id: u64,
    pub resource_group_id: Option<ResourceGroupId>,
    pub dispatches: Vec<DispatchRecord>,
    /// Responses that carried no assignment.
    pub num_empty_responses: usize,
}

pub struct LatencySamples(Vec<Duration>);

impl LatencySamples {
    pub fn from_reports(reports: &[WorkerReport]) -> Self;
    pub fn count(&self) -> usize;
    pub fn mean(&self) -> Duration;
    pub fn percentile(&self, percentile: f64) -> Duration;
}

impl FakeWorker {
    pub async fn run(
        config: FakeWorkerConfig,
        endpoint: String,
        cancellation_token: CancellationToken,
    ) -> Result<WorkerReport, HarnessError>;
}
```

The worker loop is: record `Instant`, issue `NextTask` (carrying the previous assignment as
`completed`), record the latency on response, `sleep(task_duration_ms)` if an assignment came back,
repeat until cancelled. Latency is recorded per request whether or not an assignment is returned, but
only requests that return an assignment produce a `DispatchRecord`.

Every worker keeps its samples in its own `Vec` and they are merged only after the run, so latency
collection adds no shared-state contention on the measured path. This is what makes the same harness
reusable for the performance evaluation.

### `src/harness/mod.rs`

```rust
pub struct HarnessConfig {
    pub core: CoreConfig,
    pub storage: FakeStorageConfig,
    pub workers: Vec<FakeWorkerConfig>,
}

pub struct Harness { /* … */ }

impl Harness {
    pub async fn start(config: HarnessConfig) -> Result<Self, HarnessError>;

    /// Runs every configured worker until the expected task set is drained or `timeout` expires.
    pub async fn run_until_drained(self, timeout: Duration)
        -> Result<HarnessOutcome, HarnessError>;
}

pub struct HarnessOutcome {
    pub reports: Vec<WorkerReport>,
    pub storage: FakeStorage,
}
```

## 6. Test plan

### 6.1 Crate-level unit tests (`src/tests/`)

One file per area. These drive `Core::tick` directly — no gRPC, no workers — using `FakeStorage` and
draining the per-group queues by hand.

- `job_registry.rs` — upsert new vs. existing, finalize-and-remove, `get_next_task` on a finalized
  entry returns `Err`, `downgrade_counter` reset on insert.
- `dispatch_queue.rs` — hint publication rule (`H < S`), pinned pop leaves `living_hint` untouched,
  general pop decrements it, stale hint on an empty group yields `None` and consumes the hint, a
  stale-session assignment is dropped by both paths.
- `admission.rs` — with `N` backlogged groups and `B` capacity, one tick leaves each group at
  ≈`B/(N+1)` and free space ≈`B/(N+1)`; a batch-filled staircase (one group at `B/2`, the rest at 0)
  must **not** occur.
- `scheduling_unit.rs` — finalization tasks precede regular ones; `NoTask` when everything is empty;
  `DispatchQueueFull` at the threshold; promotion, downgrade through `DOWNGRADE_LIVES`, retirement.
- `core.rs` — arm persistence across ticks via `last_served_rg`; `swap_remove` does not skip the
  swapped-in group; deactivation requires an empty dispatch queue as well as no schedulable tasks;
  steps 4 and 5 still run when polling results are not ready; session bump clears the global task set
  and the finalized job table so replayed tasks are re-admitted.

### 6.2 Integration tests (`tests/integration.rs`)

Full stack: core on its dedicated thread, gRPC harness server, gRPC workers.

Scale target for the largest case: **16 resource groups × 32 jobs × 128 tasks = 65 536 tasks.**
Smaller scenarios use reduced dimensions so the suite stays fast; at least one test runs at full scale.

Scenarios and their assertions:

1. **General workers only** — every task dispatched exactly once, every group served. This is the
   coverage invariant end-to-end.
2. **Pinned workers only, one per group** — exactly-once again, and every worker receives only tasks
   from its own group.
3. **Mixed** — pinned workers for a subset of groups plus general workers; groups with no pinned
   worker still fully drain.
4. **Asymmetric drain** — one group's pinned worker is very slow (large `task_duration_ms`); assert
   the slow group cannot occupy more than roughly `B/2` of the buffer, and that the other groups'
   throughput is not degraded. This is the fairness property the whole design exists for.
5. **Full scale** — 16 × 32 × 128 with a mix of pinned and general workers; exactly-once and
   completion within the timeout.
6. **Commit-ready** — with `emit_commit_ready`, every job's commit task is dispatched after its
   regular tasks, exactly once.
7. **Session bump mid-run** — assignments published before the bump are rejected, replayed tasks are
   re-admitted, and the run still drains to completion.

The exactly-once assertion is the same in every scenario: merge every `WorkerReport`'s dispatches,
assert the multiset of `(job_id, task_id)` equals `FakeStorage::expected_regular_tasks()` with no
duplicates.

## 7. Rules for every agent

- Read `claude/scheduler-redesign/design.md` first. It is the specification.
- Read the Rust coding style guide before writing code:
  `https://github.com/LinZhihao-723/claude-instruction/blob/main/rust/coding-style.md`
- `.cargo/config.toml` sets `-Dclippy::all -Dclippy::nursery -Dclippy::pedantic`. Warnings are
  errors. Fix clippy findings rather than adding `#[allow(...)]`.
- No `unwrap` anywhere, including tests. Use `expect` with a lowercase, period-free message.
- Docstrings on every function, public and private, per the style guide's template. `# Parameters`
  only in trait declarations.
- Do not add comments explaining what new code does. Comments justify *why* — a non-obvious ordering
  constraint, a borrow hazard — and nothing else.
- Do not edit files you do not own. Report needed changes instead.
- Build with `cargo build -p spider-scheduler-new` and `cargo clippy -p spider-scheduler-new
  --all-targets`. Do not run `task lint:fix-rust` yourself; a final agent does that once.

## 8. Deviations recorded after implementation

The prototype landed with the following departures from §3–§6 above. They are the implementation's
behaviour, not open items.

1. **No channel is bounded by its type.** Both the hint channel and the per-group dispatch queues are
   unbounded; design.md §3.5 / §8.1 / §8.2 have been corrected to match. Two separate reasons:
   - The original claim that outstanding hints are bounded by `dispatch_queue_capacity` is false.
     `H` is decremented only by a general execution manager, so a pinned execution manager draining a
     group leaves `H` at that group's peak while `S` falls. The bound is `Σ_r peak(S_r)`, whose terms
     are not simultaneous.
   - A per-group channel capacity is a redundant second bound measured against the wrong quantity —
     a per-group limit against a buffer-wide budget. The admission threshold `S < F` is what bounds
     occupancy, so the capacity's only reachable effect is to reject a send the proof requires to
     succeed. `ResourceGroupTable::new` therefore takes no capacity argument.
2. `run_core_on_dedicated_thread` takes a `CoreFactoryType: FnOnce() -> Core<…> + Send + 'static`
   rather than a `Core` value. Forced by the core being `!Send` — the value cannot cross the thread
   boundary, only a factory can.
3. `SharedJobEntry` gained `take_ready_tasks`, used to drain a finalized job's buffered task indices
   back out of the global task set.
4. `JobRegistry::upsert` takes `rg_id` alongside `(job_id, task_indices)`.
5. `WorkerReport` gained `latencies: Vec<Duration>`, which `LatencySamples::from_reports` consumes.
6. `FakeStorageConfig` gained `emit_cleanup_ready`. A job finalizes exactly once, so the two flags
   alternate jobs between the lanes (even index commits, odd index cleans up) rather than emitting
   both for one job. `FakeStorage::expected_finalization_tasks()` exposes the resulting expectation.
7. §6.2's blanket exactly-once assertion does not hold for scenario 7. A session bump legitimately
   replays work, so that scenario asserts coverage, a duplicate ceiling, and no triple dispatch
   instead.
8. `Harness` exposes neither `reschedule_queue_writer()` nor `endpoint()`. The reschedule lane is
   covered by a unit test instead; an execution-manager-loss replay scenario would need the accessor
   back.

### Known follow-ups

- `Core::finalized_jobs` grows one `JobId` per finalized job for the lifetime of a session and is
  cleared only on a session bump. Design §3.2/§9 specify this, but it needs an eviction path before
  the prototype folds into `spider-scheduler`.
- `RgSchedulingUnit::restore` returns a regular task to the head of its job entry without rewinding
  `rr_arm`, so a rejected publication perturbs the intra-group rotation by one slot. Harmless — the
  task is not lost — and the error path is unreachable while the channel capacity invariant holds.

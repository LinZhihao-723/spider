//! A prototype of the resource-group-aware Spider scheduler.
//!
//! The crate is a parallel implementation of the scheduler core specified in
//! `claude/scheduler-redesign/design.md`, built to validate that design before it is folded into
//! `spider-scheduler`. It reuses `spider-core`'s types and shares no code with the production
//! scheduler.
//!
//! The core is a single-threaded, tick-based loop. It drains ready tasks from the storage-owned
//! inbound queue, decides which of them to dispatch, and publishes the resulting assignments into
//! per-resource-group queues. Execution managers pull those assignments either pinned to one
//! resource group or generally, steered by an unbounded channel of payload-free hints:
//!
//! ```text
//!   storage  ── inbound queue (ready / commit-ready / cleanup-ready lanes)
//!         │
//!         │  SchedulerStorageClient
//!         ▼
//!   ┌───────────────────┐
//!   │       Core        │  tick: collect → process → apply → fill → retire
//!   └───────────────────┘
//!         │                              │
//!         │  assignments                 │  hints
//!         ▼                              ▼
//!   ┌───────────────────┐        ┌───────────────────┐
//!   │  per-RG queues    │        │  global queue     │
//!   └───────────────────┘        └───────────────────┘
//!         │                              │
//!         │  pinned                      │  general
//!         ▼                              ▼
//!   ┌─────────────────────────────────────────────────┐
//!   │                DispatchService                  │ ──▶ execution managers
//!   └─────────────────────────────────────────────────┘
//! ```
//!
//! Buffer space is accounted per resource group against a threshold derived from the buffer's live
//! free space, so no single group can take over the dispatch buffer and a newly arrived group
//! always finds room to publish.

pub mod bench;
pub mod config;
pub mod core;
pub mod dispatch_queue;
pub mod error;
pub mod harness;
pub mod proto;
pub mod resource_group;
pub mod session;
pub mod storage_client;
pub mod types;

pub(crate) mod job_registry;
pub(crate) mod scheduling_unit;

#[cfg(test)]
mod tests;

pub use crate::bench::BenchCase;
pub use crate::bench::JobE2eRecord;
pub use crate::bench::JobProgress;
pub use crate::bench::JobProgressMap;
pub use crate::bench::LatencyHistogram;
pub use crate::bench::LatencySeries;
pub use crate::bench::SchedulerResults;
pub use crate::bench::TickSample;
pub use crate::bench::TickStep;
pub use crate::bench::TickTimer;
pub use crate::bench::WorkerResults;
pub use crate::config::CoreConfig;
pub use crate::error::CoreError;
pub use crate::error::HarnessError;
pub use crate::error::JobEntryError;
pub use crate::error::MakeAssignmentError;
pub use crate::error::StorageClientError;
pub use crate::session::SessionManager;
pub use crate::storage_client::SchedulerStorageClient;
pub use crate::types::FinalizeKind;
pub use crate::types::InboundEntry;
pub use crate::types::TaskAssignment;

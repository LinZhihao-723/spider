//! Unit tests for the benchmark worker pool's configuration.

use crate::bench::cases::BenchCase;
use crate::harness::fake_worker::DEFAULT_NEXT_TASK_WAIT_MS;
use crate::harness::fake_worker::DEFAULT_RUN_STATUS_POLL_INTERVAL_MS;
use crate::harness::fake_worker::WorkerPoolConfig;
use crate::types::ResourceGroupId;

/// # Returns
///
/// A pool over `num_resource_groups` groups with `dedicated_workers_per_group` pinned workers each
/// and `shared_workers` general ones, identified from `execution_manager_id_base`.
fn pool_config(
    num_resource_groups: usize,
    dedicated_workers_per_group: usize,
    shared_workers: usize,
    execution_manager_id_base: u64,
) -> WorkerPoolConfig {
    WorkerPoolConfig {
        endpoint: "http://127.0.0.1:1".to_owned(),
        num_resource_groups,
        dedicated_workers_per_group,
        shared_workers,
        task_duration_ms: 5,
        next_task_wait_ms: DEFAULT_NEXT_TASK_WAIT_MS,
        channel_pool_size: shared_workers + num_resource_groups * dedicated_workers_per_group,
        execution_manager_id_base,
        sample_capacity_hint: 16,
        run_status_poll_interval_ms: DEFAULT_RUN_STATUS_POLL_INTERVAL_MS,
        run_timeout_ms: 0,
    }
}

#[test]
fn worker_configs_group_the_pinned_workers_and_place_the_general_ones_last() {
    let config = pool_config(2, 3, 4, 0);
    let worker_configs = config.worker_configs();

    assert_eq!(worker_configs.len(), config.num_workers());
    let resource_group_ids: Vec<Option<ResourceGroupId>> = worker_configs
        .iter()
        .map(|worker_config| worker_config.resource_group_id)
        .collect();
    let group_0 = Some(ResourceGroupId::from(0));
    let group_1 = Some(ResourceGroupId::from(1));
    assert_eq!(
        resource_group_ids,
        vec![
            group_0, group_0, group_0, group_1, group_1, group_1, None, None, None, None,
        ]
    );
}

#[test]
fn worker_configs_take_a_contiguous_identity_block_from_the_base() {
    let config = pool_config(2, 3, 4, 100);
    let execution_manager_ids: Vec<u64> = config
        .worker_configs()
        .iter()
        .map(|worker_config| worker_config.execution_manager_id)
        .collect();

    assert_eq!(execution_manager_ids, (100..110).collect::<Vec<u64>>());
}

#[test]
fn a_pool_without_dedicated_workers_hosts_only_general_ones() {
    let config = pool_config(8, 0, 5, 0);
    let worker_configs = config.worker_configs();

    assert_eq!(worker_configs.len(), 5);
    assert_eq!(
        worker_configs
            .iter()
            .filter(|worker_config| worker_config.resource_group_id.is_some())
            .count(),
        0
    );
}

#[test]
fn from_case_takes_the_whole_case_and_separates_the_processes() {
    let case = BenchCase::from_name("D").expect("case D is in the table");
    let config = WorkerPoolConfig::from_case(&case, "127.0.0.1:9000".to_owned(), 2);

    assert_eq!(config.num_resource_groups, case.num_resource_groups);
    assert_eq!(
        config.dedicated_workers_per_group,
        case.dedicated_workers_per_group
    );
    assert_eq!(config.shared_workers, case.shared_workers);
    assert_eq!(config.task_duration_ms, case.task_duration_ms);
    assert_eq!(config.num_workers(), case.total_workers());
    assert_eq!(
        config.execution_manager_id_base,
        u64::try_from(2 * case.total_workers()).expect("the identity base fits in a u64")
    );
}

#[test]
fn from_case_gives_every_worker_a_channel_of_its_own() {
    for name in ["A", "B", "C", "D", "E"] {
        let case = BenchCase::from_name(name).expect("the case is in the table");
        let config = WorkerPoolConfig::from_case(&case, "127.0.0.1:9000".to_owned(), 0);

        assert_eq!(config.channel_pool_size, config.num_workers());
    }
}

#[test]
fn from_case_reserves_room_for_a_workers_share_of_the_workload() {
    let case = BenchCase::from_name("D").expect("case D is in the table");
    let config = WorkerPoolConfig::from_case(&case, "127.0.0.1:9000".to_owned(), 0);

    assert!(config.sample_capacity_hint > case.total_tasks() / case.total_workers());
}

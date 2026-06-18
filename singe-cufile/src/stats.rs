use crate::{
    error::{Error, Result},
    sys, try_ffi,
    types::StatisticsLevel,
};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OperationCounter {
    pub ok: u64,
    pub err: u64,
}

impl From<sys::CUfileOpCounter_t> for OperationCounter {
    fn from(value: sys::CUfileOpCounter_t) -> Self {
        Self {
            ok: value.ok,
            err: value.err,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StatisticsLevel1 {
    pub read_ops: OperationCounter,
    pub write_ops: OperationCounter,
    pub handle_register_ops: OperationCounter,
    pub handle_deregister_ops: OperationCounter,
    pub buffer_register_ops: OperationCounter,
    pub buffer_deregister_ops: OperationCounter,
    pub read_bytes: u64,
    pub write_bytes: u64,
    pub read_bandwidth_bytes_per_second: u64,
    pub write_bandwidth_bytes_per_second: u64,
    pub read_latency_average_microseconds: u64,
    pub write_latency_average_microseconds: u64,
    pub read_ops_per_second: u64,
    pub write_ops_per_second: u64,
    pub read_latency_sum_microseconds: u64,
    pub write_latency_sum_microseconds: u64,
    pub batch_submit_ops: OperationCounter,
    pub batch_complete_ops: OperationCounter,
    pub batch_setup_ops: OperationCounter,
    pub batch_cancel_ops: OperationCounter,
    pub batch_destroy_ops: OperationCounter,
    pub batch_enqueued_ops: OperationCounter,
    pub batch_posix_enqueued_ops: OperationCounter,
    pub batch_processed_ops: OperationCounter,
    pub batch_posix_processed_ops: OperationCounter,
    pub batch_nvfs_submit_ops: OperationCounter,
    pub batch_p2p_submit_ops: OperationCounter,
    pub batch_aio_submit_ops: OperationCounter,
    pub batch_iouring_submit_ops: OperationCounter,
    pub batch_mixed_io_submit_ops: OperationCounter,
    pub batch_total_submit_ops: OperationCounter,
    pub batch_read_bytes: u64,
    pub batch_write_bytes: u64,
    pub batch_read_bandwidth_bytes: u64,
    pub batch_write_bandwidth_bytes: u64,
    pub batch_submit_latency_average_microseconds: u64,
    pub batch_completion_latency_average_microseconds: u64,
    pub batch_submit_ops_per_second: u64,
    pub batch_complete_ops_per_second: u64,
    pub batch_submit_latency_sum_microseconds: u64,
    pub batch_completion_latency_sum_microseconds: u64,
    pub last_batch_read_bytes: u64,
    pub last_batch_write_bytes: u64,
}

impl From<sys::CUfileStatsLevel1_t> for StatisticsLevel1 {
    fn from(value: sys::CUfileStatsLevel1_t) -> Self {
        Self {
            read_ops: value.read_ops.into(),
            write_ops: value.write_ops.into(),
            handle_register_ops: value.hdl_register_ops.into(),
            handle_deregister_ops: value.hdl_deregister_ops.into(),
            buffer_register_ops: value.buf_register_ops.into(),
            buffer_deregister_ops: value.buf_deregister_ops.into(),
            read_bytes: value.read_bytes,
            write_bytes: value.write_bytes,
            read_bandwidth_bytes_per_second: value.read_bw_bytes_per_sec,
            write_bandwidth_bytes_per_second: value.write_bw_bytes_per_sec,
            read_latency_average_microseconds: value.read_lat_avg_us,
            write_latency_average_microseconds: value.write_lat_avg_us,
            read_ops_per_second: value.read_ops_per_sec,
            write_ops_per_second: value.write_ops_per_sec,
            read_latency_sum_microseconds: value.read_lat_sum_us,
            write_latency_sum_microseconds: value.write_lat_sum_us,
            batch_submit_ops: value.batch_submit_ops.into(),
            batch_complete_ops: value.batch_complete_ops.into(),
            batch_setup_ops: value.batch_setup_ops.into(),
            batch_cancel_ops: value.batch_cancel_ops.into(),
            batch_destroy_ops: value.batch_destroy_ops.into(),
            batch_enqueued_ops: value.batch_enqueued_ops.into(),
            batch_posix_enqueued_ops: value.batch_posix_enqueued_ops.into(),
            batch_processed_ops: value.batch_processed_ops.into(),
            batch_posix_processed_ops: value.batch_posix_processed_ops.into(),
            batch_nvfs_submit_ops: value.batch_nvfs_submit_ops.into(),
            batch_p2p_submit_ops: value.batch_p2p_submit_ops.into(),
            batch_aio_submit_ops: value.batch_aio_submit_ops.into(),
            batch_iouring_submit_ops: value.batch_iouring_submit_ops.into(),
            batch_mixed_io_submit_ops: value.batch_mixed_io_submit_ops.into(),
            batch_total_submit_ops: value.batch_total_submit_ops.into(),
            batch_read_bytes: value.batch_read_bytes,
            batch_write_bytes: value.batch_write_bytes,
            batch_read_bandwidth_bytes: value.batch_read_bw_bytes,
            batch_write_bandwidth_bytes: value.batch_write_bw_bytes,
            batch_submit_latency_average_microseconds: value.batch_submit_lat_avg_us,
            batch_completion_latency_average_microseconds: value.batch_completion_lat_avg_us,
            batch_submit_ops_per_second: value.batch_submit_ops_per_sec,
            batch_complete_ops_per_second: value.batch_complete_ops_per_sec,
            batch_submit_latency_sum_microseconds: value.batch_submit_lat_sum_us,
            batch_completion_latency_sum_microseconds: value.batch_completion_lat_sum_us,
            last_batch_read_bytes: value.last_batch_read_bytes,
            last_batch_write_bytes: value.last_batch_write_bytes,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StatisticsLevel2 {
    pub basic: StatisticsLevel1,
    pub read_size_kilobyte_histogram: [u64; 32],
    pub write_size_kilobyte_histogram: [u64; 32],
}

impl From<sys::CUfileStatsLevel2_t> for StatisticsLevel2 {
    fn from(value: sys::CUfileStatsLevel2_t) -> Self {
        Self {
            basic: value.basic.into(),
            read_size_kilobyte_histogram: value.read_size_kb_hist,
            write_size_kilobyte_histogram: value.write_size_kb_hist,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PerGpuStatistics {
    pub uuid: [u8; 16],
    pub read_bytes: u64,
    pub read_bandwidth_bytes_per_second: u64,
    pub read_utilization: u64,
    pub read_duration_microseconds: u64,
    pub total_reads: u64,
    pub p2p_reads: u64,
    pub nvfs_reads: u64,
    pub posix_reads: u64,
    pub unaligned_reads: u64,
    pub direct_reads: u64,
    pub sparse_regions: u64,
    pub inline_regions: u64,
    pub read_errors: u64,
    pub write_bytes: u64,
    pub write_bandwidth_bytes_per_second: u64,
    pub write_utilization: u64,
    pub write_duration_microseconds: u64,
    pub total_writes: u64,
    pub p2p_writes: u64,
    pub nvfs_writes: u64,
    pub posix_writes: u64,
    pub unaligned_writes: u64,
    pub direct_writes: u64,
    pub write_errors: u64,
    pub mmap: u64,
    pub mmap_ok: u64,
    pub mmap_errors: u64,
    pub mmap_free: u64,
    pub registered_bytes: u64,
}

impl From<sys::CUfilePerGpuStats_t> for PerGpuStatistics {
    fn from(value: sys::CUfilePerGpuStats_t) -> Self {
        Self {
            uuid: value.uuid.map(|byte| byte as u8),
            read_bytes: value.read_bytes,
            read_bandwidth_bytes_per_second: value.read_bw_bytes_per_sec,
            read_utilization: value.read_utilization,
            read_duration_microseconds: value.read_duration_us,
            total_reads: value.n_total_reads,
            p2p_reads: value.n_p2p_reads,
            nvfs_reads: value.n_nvfs_reads,
            posix_reads: value.n_posix_reads,
            unaligned_reads: value.n_unaligned_reads,
            direct_reads: value.n_dr_reads,
            sparse_regions: value.n_sparse_regions,
            inline_regions: value.n_inline_regions,
            read_errors: value.n_reads_err,
            write_bytes: value.writes_bytes,
            write_bandwidth_bytes_per_second: value.write_bw_bytes_per_sec,
            write_utilization: value.write_utilization,
            write_duration_microseconds: value.write_duration_us,
            total_writes: value.n_total_writes,
            p2p_writes: value.n_p2p_writes,
            nvfs_writes: value.n_nvfs_writes,
            posix_writes: value.n_posix_writes,
            unaligned_writes: value.n_unaligned_writes,
            direct_writes: value.n_dr_writes,
            write_errors: value.n_writes_err,
            mmap: value.n_mmap,
            mmap_ok: value.n_mmap_ok,
            mmap_errors: value.n_mmap_err,
            mmap_free: value.n_mmap_free,
            registered_bytes: value.reg_bytes,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StatisticsLevel3 {
    pub detailed: StatisticsLevel2,
    pub per_gpu_statistics: Vec<PerGpuStatistics>,
}

impl From<sys::CUfileStatsLevel3_t> for StatisticsLevel3 {
    fn from(value: sys::CUfileStatsLevel3_t) -> Self {
        let count = (value.num_gpus as usize).min(value.per_gpu_stats.len());
        Self {
            detailed: value.detailed.into(),
            per_gpu_statistics: value.per_gpu_stats[..count]
                .iter()
                .copied()
                .map(PerGpuStatistics::from)
                .collect(),
        }
    }
}

pub fn set_level(level: StatisticsLevel) -> Result<()> {
    unsafe {
        try_ffi!(sys::cuFileSetStatsLevel(level as i32))?;
    }
    Ok(())
}

pub fn level() -> Result<StatisticsLevel> {
    let mut level = 0;
    unsafe {
        try_ffi!(sys::cuFileGetStatsLevel(&raw mut level))?;
    }
    StatisticsLevel::try_from(level).map_err(|_| Error::OutOfRange {
        name: "level".into(),
    })
}

pub fn start() -> Result<()> {
    unsafe {
        try_ffi!(sys::cuFileStatsStart())?;
    }
    Ok(())
}

pub fn stop() -> Result<()> {
    unsafe {
        try_ffi!(sys::cuFileStatsStop())?;
    }
    Ok(())
}

pub fn reset() -> Result<()> {
    unsafe {
        try_ffi!(sys::cuFileStatsReset())?;
    }
    Ok(())
}

pub fn level1() -> Result<StatisticsLevel1> {
    let mut stats = sys::CUfileStatsLevel1_t::default();
    unsafe {
        try_ffi!(sys::cuFileGetStatsL1(&raw mut stats))?;
    }
    Ok(stats.into())
}

pub fn level2() -> Result<StatisticsLevel2> {
    let mut stats = sys::CUfileStatsLevel2_t::default();
    unsafe {
        try_ffi!(sys::cuFileGetStatsL2(&raw mut stats))?;
    }
    Ok(stats.into())
}

pub fn level3() -> Result<StatisticsLevel3> {
    let mut stats = sys::CUfileStatsLevel3_t::default();
    unsafe {
        try_ffi!(sys::cuFileGetStatsL3(&raw mut stats))?;
    }
    Ok(stats.into())
}

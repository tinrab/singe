use crate::{
    error::{Error, Result},
    sys, try_ffi,
    types::StatisticsLevel,
};

/// Successful and failed operation counts.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OperationCounter {
    /// Number of successful operations.
    pub ok: u64,
    /// Number of failed operations.
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

/// Basic cuFile I/O counters, bandwidth, and latency statistics.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StatisticsLevel1 {
    /// Vectored read operation counts.
    pub readv_ops: OperationCounter,
    /// Vectored read bytes.
    pub readv_bytes: u64,
    /// Vectored read bandwidth in bytes per second.
    pub readv_bandwidth_bytes_per_second: u64,
    /// Vectored read average latency in microseconds.
    pub readv_latency_average_microseconds: u64,
    /// Vectored read total latency in microseconds.
    pub readv_latency_sum_microseconds: u64,
    /// Vectored read operations per second.
    pub readv_ops_per_second: u64,
    /// Vectored write operation counts.
    pub writev_ops: OperationCounter,
    /// Vectored write bytes.
    pub writev_bytes: u64,
    /// Vectored write bandwidth in bytes per second.
    pub writev_bandwidth_bytes_per_second: u64,
    /// Vectored write average latency in microseconds.
    pub writev_latency_average_microseconds: u64,
    /// Vectored write total latency in microseconds.
    pub writev_latency_sum_microseconds: u64,
    /// Vectored write operations per second.
    pub writev_ops_per_second: u64,
    /// Read operations.
    pub read_ops: OperationCounter,
    /// Write operations.
    pub write_ops: OperationCounter,
    /// Handle register operations.
    pub handle_register_ops: OperationCounter,
    /// Handle deregister operations.
    pub handle_deregister_ops: OperationCounter,
    /// Buffer register operations.
    pub buffer_register_ops: OperationCounter,
    /// Buffer deregister operations.
    pub buffer_deregister_ops: OperationCounter,
    /// Total bytes read.
    pub read_bytes: u64,
    /// Total bytes written.
    pub write_bytes: u64,
    /// Read bandwidth (bytes/sec).
    pub read_bandwidth_bytes_per_second: u64,
    /// Write bandwidth (bytes/sec).
    pub write_bandwidth_bytes_per_second: u64,
    /// Average read latency (microseconds).
    pub read_latency_average_microseconds: u64,
    /// Average write latency (microseconds).
    pub write_latency_average_microseconds: u64,
    /// Read operations per second.
    pub read_ops_per_second: u64,
    /// Write operations per second.
    pub write_ops_per_second: u64,
    /// Sum of read latencies, in microseconds.
    pub read_latency_sum_microseconds: u64,
    /// Sum of write latencies, in microseconds.
    pub write_latency_sum_microseconds: u64,
    /// Batch submit operations.
    pub batch_submit_ops: OperationCounter,
    /// Batch complete operations.
    pub batch_complete_ops: OperationCounter,
    /// Batch setup operations.
    pub batch_setup_ops: OperationCounter,
    /// Batch cancel operations.
    pub batch_cancel_ops: OperationCounter,
    /// Batch destroy operations.
    pub batch_destroy_ops: OperationCounter,
    /// Batch enqueue operations.
    pub batch_enqueued_ops: OperationCounter,
    /// POSIX batch enqueue operations.
    pub batch_posix_enqueued_ops: OperationCounter,
    /// Batch process operations.
    pub batch_processed_ops: OperationCounter,
    /// POSIX batch process operations.
    pub batch_posix_processed_ops: OperationCounter,
    /// NVFS batch submit operations.
    pub batch_nvfs_submit_ops: OperationCounter,
    /// P2P batch submit operations.
    pub batch_p2p_submit_ops: OperationCounter,
    /// AIO batch submit operations.
    pub batch_aio_submit_ops: OperationCounter,
    /// `io_uring` batch submit operations.
    pub batch_iouring_submit_ops: OperationCounter,
    /// Mixed IO batch submit operations.
    pub batch_mixed_io_submit_ops: OperationCounter,
    /// Total batch submit operations.
    pub batch_total_submit_ops: OperationCounter,
    /// Total batch read bytes.
    pub batch_read_bytes: u64,
    /// Total batch write bytes.
    pub batch_write_bytes: u64,
    /// Batch read bandwidth.
    pub batch_read_bandwidth_bytes: u64,
    /// Batch write bandwidth.
    pub batch_write_bandwidth_bytes: u64,
    /// Avg batch submit latency, in microseconds.
    pub batch_submit_latency_average_microseconds: u64,
    /// Avg batch completion latency, in microseconds.
    pub batch_completion_latency_average_microseconds: u64,
    /// Batch submit operations per second.
    pub batch_submit_ops_per_second: u64,
    /// Batch complete operations per second.
    pub batch_complete_ops_per_second: u64,
    /// Sum of batch submit latencies, in microseconds.
    pub batch_submit_latency_sum_microseconds: u64,
    /// Sum of batch completion latencies, in microseconds.
    pub batch_completion_latency_sum_microseconds: u64,
    /// Last batch read bytes.
    pub last_batch_read_bytes: u64,
    /// Last batch write bytes.
    pub last_batch_write_bytes: u64,
}

impl From<sys::CUfileStatsLevel1_t> for StatisticsLevel1 {
    fn from(value: sys::CUfileStatsLevel1_t) -> Self {
        Self {
            readv_ops: value.readv_ops.into(),
            readv_bytes: value.readv_bytes,
            readv_bandwidth_bytes_per_second: value.readv_bw_bytes_per_sec,
            readv_latency_average_microseconds: value.readv_lat_avg_us,
            readv_latency_sum_microseconds: value.readv_lat_sum_us,
            readv_ops_per_second: value.readv_ops_per_sec,
            writev_ops: value.writev_ops.into(),
            writev_bytes: value.writev_bytes,
            writev_bandwidth_bytes_per_second: value.writev_bw_bytes_per_sec,
            writev_latency_average_microseconds: value.writev_lat_avg_us,
            writev_latency_sum_microseconds: value.writev_lat_sum_us,
            writev_ops_per_second: value.writev_ops_per_sec,
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

/// Basic statistics with read and write size histograms.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StatisticsLevel2 {
    /// Basic statistics (Level 1).
    pub basic: StatisticsLevel1,
    /// Histogram of read sizes measured in KB.
    pub read_size_kilobyte_histogram: [u64; 32],
    /// Histogram of write sizes measured in KB.
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

/// I/O paths, bandwidth, utilization, and buffer registrations for one GPU.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PerGpuStatistics {
    /// The GPU UUID as 16 bytes.
    pub uuid: [u8; 16],
    /// Total bytes read.
    pub read_bytes: u64,
    /// Read bandwidth in bytes per second.
    pub read_bandwidth_bytes_per_second: u64,
    /// Read utilization percentage.
    pub read_utilization: u64,
    /// Read operation duration, in microseconds.
    pub read_duration_microseconds: u64,
    /// Total number of reads.
    pub total_reads: u64,
    /// Number of `PCIe` `P2PDMA` reads.
    pub p2p_reads: u64,
    /// Number of nvidia-fs reads.
    pub nvfs_reads: u64,
    /// Number of POSIX reads.
    pub posix_reads: u64,
    /// Number of unaligned reads.
    pub unaligned_reads: u64,
    /// Number of reads using dynamic routing.
    pub direct_reads: u64,
    /// Number of sparse regions.
    pub sparse_regions: u64,
    /// Number of inline regions.
    pub inline_regions: u64,
    /// Number of read errors.
    pub read_errors: u64,
    /// Total bytes written.
    pub write_bytes: u64,
    /// Write bandwidth in bytes per second.
    pub write_bandwidth_bytes_per_second: u64,
    /// Write utilization percentage.
    pub write_utilization: u64,
    /// Write operation duration, in microseconds.
    pub write_duration_microseconds: u64,
    /// Total number of writes.
    pub total_writes: u64,
    /// Number of `PCIe` `P2PDMA` writes.
    pub p2p_writes: u64,
    /// Number of nvidia-fs writes.
    pub nvfs_writes: u64,
    /// Number of POSIX writes.
    pub posix_writes: u64,
    /// Number of unaligned writes.
    pub unaligned_writes: u64,
    /// Number of writes using dynamic routing.
    pub direct_writes: u64,
    /// Number of write errors.
    pub write_errors: u64,
    /// Number of buffer registrations.
    pub mmap: u64,
    /// Successful registrations.
    pub mmap_ok: u64,
    /// Failed registrations.
    pub mmap_errors: u64,
    /// Number of buffer deregistrations.
    pub mmap_free: u64,
    /// Total bytes registered.
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

/// Detailed cuFile statistics with one entry per reported GPU.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StatisticsLevel3 {
    /// Detailed statistics (Level 2).
    pub detailed: StatisticsLevel2,
    /// Statistics for the GPUs reported by cuFile, bounded by the native array capacity.
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

/// Selects the amount of cuFile statistics to collect.
///
/// # Errors
///
/// Returns an error if cuFile rejects the statistics level.
pub fn set_level(level: StatisticsLevel) -> Result<()> {
    // SAFETY: `cuFileSetStatsLevel` receives only plain values.
    unsafe {
        try_ffi!(sys::cuFileSetStatsLevel(level as i32))?;
    }
    Ok(())
}

/// Returns the current statistics collection level.
///
/// # Errors
///
/// Returns an error if cuFile cannot report the collection level.
/// Also returns an error if the native level is not a recognized [`StatisticsLevel`].
pub fn level() -> Result<StatisticsLevel> {
    let mut level = 0;
    // SAFETY: `cuFileGetStatsLevel` receives valid output locations.
    unsafe {
        try_ffi!(sys::cuFileGetStatsLevel(&raw mut level))?;
    }
    StatisticsLevel::try_from(level).map_err(|_| Error::out_of_range("level"))
}

/// Starts cuFile statistics collection.
///
/// # Errors
///
/// Returns an error if cuFile cannot start statistics collection.
pub fn start() -> Result<()> {
    // SAFETY: `cuFileStatsStart` takes no arguments.
    unsafe {
        try_ffi!(sys::cuFileStatsStart())?;
    }
    Ok(())
}

/// Stops cuFile statistics collection.
///
/// # Errors
///
/// Returns an error if cuFile cannot stop statistics collection.
pub fn stop() -> Result<()> {
    // SAFETY: `cuFileStatsStop` takes no arguments.
    unsafe {
        try_ffi!(sys::cuFileStatsStop())?;
    }
    Ok(())
}

/// Resets the collected cuFile statistics.
///
/// # Errors
///
/// Returns an error if cuFile cannot reset the collected statistics.
pub fn reset() -> Result<()> {
    // SAFETY: `cuFileStatsReset` takes no arguments.
    unsafe {
        try_ffi!(sys::cuFileStatsReset())?;
    }
    Ok(())
}

/// Returns the current basic I/O statistics.
///
/// # Errors
///
/// Returns an error if cuFile cannot report level 1 statistics.
/// The driver must be initialized and the requested statistics level must be enabled.
pub fn level1() -> Result<StatisticsLevel1> {
    let mut stats = sys::CUfileStatsLevel1_t::default();
    // SAFETY: `cuFileGetStatsL1` receives valid output locations.
    unsafe {
        try_ffi!(sys::cuFileGetStatsL1(&raw mut stats))?;
    }
    Ok(stats.into())
}

/// Returns the current detailed I/O statistics, including size histograms.
///
/// # Errors
///
/// Returns an error if cuFile cannot report level 2 statistics.
/// The driver must be initialized and the requested statistics level must be enabled.
pub fn level2() -> Result<StatisticsLevel2> {
    let mut stats = sys::CUfileStatsLevel2_t::default();
    // SAFETY: `cuFileGetStatsL2` receives valid output locations.
    unsafe {
        try_ffi!(sys::cuFileGetStatsL2(&raw mut stats))?;
    }
    Ok(stats.into())
}

/// Returns detailed statistics and per-GPU counters.
///
/// # Errors
///
/// Returns an error if cuFile cannot report level 3 statistics.
/// The driver must be initialized and the requested statistics level must be enabled.
pub fn level3() -> Result<StatisticsLevel3> {
    let mut stats = sys::CUfileStatsLevel3_t::default();
    // SAFETY: `cuFileGetStatsL3` receives valid output locations.
    unsafe {
        try_ffi!(sys::cuFileGetStatsL3(&raw mut stats))?;
    }
    Ok(stats.into())
}

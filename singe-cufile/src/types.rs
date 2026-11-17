use std::fmt::{self, Display, Formatter};

use bitflags::parser;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::impl_enum_display;
use singe_cufile_sys as sys;

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    /// Supported filesystems and block-device types reported by the cuFile driver.
    pub struct DriverStatusFlags: u32 {
        /// Support for DDN Lustre.
        const LUSTRE_SUPPORTED = 1 << (sys::CUfileDriverStatusFlags::CU_FILE_LUSTRE_SUPPORTED.0);
        /// Support for WekaFS.
        const WEKAFS_SUPPORTED = 1 << (sys::CUfileDriverStatusFlags::CU_FILE_WEKAFS_SUPPORTED.0);
        /// Support for NFS.
        const NFS_SUPPORTED = 1 << (sys::CUfileDriverStatusFlags::CU_FILE_NFS_SUPPORTED.0);
        /// Support for IBM Spectrum Scale (GPFS).
        const GPFS_SUPPORTED = 1 << (sys::CUfileDriverStatusFlags::CU_FILE_GPFS_SUPPORTED.0);
        /// Support for `NVMe`.
        const NVME_SUPPORTED = 1 << (sys::CUfileDriverStatusFlags::CU_FILE_NVME_SUPPORTED.0);
        /// Support for `NVMe` over Fabrics.
        const NVMEOF_SUPPORTED = 1 << (sys::CUfileDriverStatusFlags::CU_FILE_NVMEOF_SUPPORTED.0);
        /// Support for SCSI.
        const SCSI_SUPPORTED = 1 << (sys::CUfileDriverStatusFlags::CU_FILE_SCSI_SUPPORTED.0);
        /// Support for ScaleFlux computational storage devices.
        const SCALEFLUX_CSD_SUPPORTED = 1 << (sys::CUfileDriverStatusFlags::CU_FILE_SCALEFLUX_CSD_SUPPORTED.0);
        /// Support for `NVMesh` block devices.
        const NVMESH_SUPPORTED = 1 << (sys::CUfileDriverStatusFlags::CU_FILE_NVMESH_SUPPORTED.0);
        /// Support for `BeeGFS`.
        const BEEGFS_SUPPORTED = 1 << (sys::CUfileDriverStatusFlags::CU_FILE_BEEGFS_SUPPORTED.0);
        /// Deprecated native support bit; NVIDIA directs callers not to use it.
        const NVME_P2P_SUPPORTED = 1 << (sys::CUfileDriverStatusFlags::CU_FILE_NVME_P2P_SUPPORTED.0);
        /// Support for `ScateFS`.
        const SCATEFS_SUPPORTED = 1 << (sys::CUfileDriverStatusFlags::CU_FILE_SCATEFS_SUPPORTED.0);
        /// Support for PanFS.
        const PANFS_SUPPORTED = 1 << (sys::CUfileDriverStatusFlags::CU_FILE_PANFS_SUPPORTED.0);
        /// Support for `VirtioFS`.
        const VIRTIOFS_SUPPORTED = 1 << (sys::CUfileDriverStatusFlags::CU_FILE_VIRTIOFS_SUPPORTED.0);
    }
}

impl Display for DriverStatusFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

singe_core::native_enum! {
    /// driver properties
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum DriverStatus: sys::CUfileDriverStatusFlags {
        ///< Support for DDN LUSTRE
        LustreSupported = CU_FILE_LUSTRE_SUPPORTED,
        ///< Support for WEKAFS
        WekaFsSupported = CU_FILE_WEKAFS_SUPPORTED,
        ///< Support for NFS
        NfsSupported = CU_FILE_NFS_SUPPORTED,
        /// Bit position indicating GPFS support in the driver status flags.
        GpfsSupported = CU_FILE_GPFS_SUPPORTED,
        ///< Support for `NVMe`
        NvmeSupported = CU_FILE_NVME_SUPPORTED,
        ///< Support for `NVMeOF`
        NvmeOfSupported = CU_FILE_NVMEOF_SUPPORTED,
        ///< Support for SCSI
        ScsiSupported = CU_FILE_SCSI_SUPPORTED,
        ///< Support for Scaleflux CSD
        ScalefluxCsdSupported = CU_FILE_SCALEFLUX_CSD_SUPPORTED,
        ///< Support for `NVMesh` Block Dev
        NvmeshSupported = CU_FILE_NVMESH_SUPPORTED,
        ///< Support for `BeeGFS`
        BeeGfsSupported = CU_FILE_BEEGFS_SUPPORTED,
        ///< Do not use this macro. This is deprecated now
        NvmeP2pSupported = CU_FILE_NVME_P2P_SUPPORTED,
        ///< Support for `ScateFS`
        ScateFsSupported = CU_FILE_SCATEFS_SUPPORTED,
        ///<Support for `VirtioFS`
        VirtioFsSupported = CU_FILE_VIRTIOFS_SUPPORTED,
        /// Support for `PanFS`.
        PanFsSupported = CU_FILE_PANFS_SUPPORTED,
        ///<Maximum FS supported
        MaxTargetTypes = CU_FILE_MAX_TARGET_TYPES,
    }
}

impl_enum_display!(DriverStatus, {
    Self::LustreSupported => "CU_FILE_LUSTRE_SUPPORTED",
    Self::WekaFsSupported => "CU_FILE_WEKAFS_SUPPORTED",
    Self::NfsSupported => "CU_FILE_NFS_SUPPORTED",
    Self::GpfsSupported => "CU_FILE_GPFS_SUPPORTED",
    Self::NvmeSupported => "CU_FILE_NVME_SUPPORTED",
    Self::NvmeOfSupported => "CU_FILE_NVMEOF_SUPPORTED",
    Self::ScsiSupported => "CU_FILE_SCSI_SUPPORTED",
    Self::ScalefluxCsdSupported => "CU_FILE_SCALEFLUX_CSD_SUPPORTED",
    Self::NvmeshSupported => "CU_FILE_NVMESH_SUPPORTED",
    Self::BeeGfsSupported => "CU_FILE_BEEGFS_SUPPORTED",
    Self::NvmeP2pSupported => "CU_FILE_NVME_P2P_SUPPORTED",
    Self::ScateFsSupported => "CU_FILE_SCATEFS_SUPPORTED",
    Self::VirtioFsSupported => "CU_FILE_VIRTIOFS_SUPPORTED",
    Self::PanFsSupported => "CU_FILE_PANFS_SUPPORTED",
    Self::MaxTargetTypes => "CU_FILE_MAX_TARGET_TYPES",
});

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    /// Enabled cuFile I/O controls.
    pub struct DriverControlFlags: u32 {
        /// Polling mode is enabled.
        const USE_POLL_MODE = 1 << (sys::CUfileDriverControlFlags::CU_FILE_USE_POLL_MODE.0);
        /// Compatibility mode is allowed.
        const ALLOW_COMPAT_MODE = 1 << (sys::CUfileDriverControlFlags::CU_FILE_ALLOW_COMPAT_MODE.0);
        /// Vanilla POSIX I/O mode is enabled.
        const POSIX_IO_MODE = 1 << (sys::CUfileDriverControlFlags::CU_FILE_VANILLA_POSIX_IO_MODE.0);
        /// Fallback I/O mode is enabled.
        const FALLBACK_IO_MODE = 1 << (sys::CUfileDriverControlFlags::CU_FILE_FALLBACK_IO_MODE.0);
    }
}

impl Display for DriverControlFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    /// Capabilities supported by the cuFile library.
    pub struct FeatureFlags: u32 {
        /// Dynamic routing across `PCIe` bridges is supported.
        const DYN_ROUTING_SUPPORTED = 1 << (sys::CUfileFeatureFlags::CU_FILE_DYN_ROUTING_SUPPORTED.0);
        /// Batch I/O is supported.
        const BATCH_IO_SUPPORTED = 1 << (sys::CUfileFeatureFlags::CU_FILE_BATCH_IO_SUPPORTED.0);
        /// Stream-based asynchronous I/O is supported.
        const STREAMS_SUPPORTED = 1 << (sys::CUfileFeatureFlags::CU_FILE_STREAMS_SUPPORTED.0);
        /// Parallel I/O is supported.
        const PARALLEL_IO_SUPPORTED = 1 << (sys::CUfileFeatureFlags::CU_FILE_PARALLEL_IO_SUPPORTED.0);
        /// PCI peer-to-peer DMA is supported.
        const P2P_SUPPORTED = 1 << (sys::CUfileFeatureFlags::CU_FILE_P2P_SUPPORTED.0);
    }
}

impl Display for FeatureFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    /// Options passed to cuFile buffer registration.
    pub struct BufferRegisterFlags: i32 {
        /// Requests RDMA registration.
        const RDMA_REGISTER = sys::CU_FILE_RDMA_REGISTER.cast_signed();
        /// Selects relaxed ordering for RDMA registration.
        const RDMA_RELAXED_ORDERING = sys::CU_FILE_RDMA_RELAXED_ORDERING.cast_signed();
    }
}

impl Display for BufferRegisterFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    /// Properties declared when registering a stream with cuFile.
    pub struct StreamRegisterFlags: u32 {
        /// Stream-registration flag declaring that the buffer offset is set at submission time.
        const FIXED_BUFFER_OFFSET = sys::CU_FILE_STREAM_FIXED_BUF_OFFSET;
        /// Stream-registration flag declaring that the file offset is set at submission time.
        const FIXED_FILE_OFFSET = sys::CU_FILE_STREAM_FIXED_FILE_OFFSET;
        /// Stream-registration flag declaring that the I/O size is set at submission time.
        const FIXED_FILE_SIZE = sys::CU_FILE_STREAM_FIXED_FILE_SIZE;
        /// Stream-registration flag declaring that the I/O size, file offset, and buffer offset are all 4 KiB aligned.
        const PAGE_ALIGNED_INPUTS = sys::CU_FILE_STREAM_PAGE_ALIGNED_INPUTS;
    }
}

impl Display for StreamRegisterFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    /// Supported or enabled peer-to-peer I/O mechanisms.
    pub struct P2PFlags: u32 {
        /// PCI peer-to-peer DMA support.
        const PCI_P2PDMA = 1 << sys::CUfileP2PFlags::CUFILE_P2PDMA.0;
        /// Support through the nvidia-fs driver.
        const NVFS = 1 << sys::CUfileP2PFlags::CUFILE_NVFS.0;
        /// DMA-buffer support.
        const DMABUF = 1 << sys::CUfileP2PFlags::CUFILE_DMABUF.0;
        /// Chip-to-chip support on Grace-based systems.
        const C2C = 1 << sys::CUfileP2PFlags::CUFILE_C2C.0;
        /// NVIDIA peer-memory support for IBM Spectrum Scale and WekaFS.
        const NVIDIA_PEERMEM = 1 << sys::CUfileP2PFlags::CUFILE_NVIDIA_PEERMEM.0;
    }
}

impl Display for P2PFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

singe_core::native_enum! {
    /// Direction of a cuFile batch I/O request.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum Operation: sys::CUfileOpcode {
        /// Reads from the registered file into the request buffer.
        Read = CUFILE_READ,
        /// Writes from the request buffer to the registered file.
        Write = CUFILE_WRITE,
    }
}

impl_enum_display!(Operation, {
    Self::Read => "CUFILE_READ",
    Self::Write => "CUFILE_WRITE",
});

singe_core::native_enum! {
    ///Experimental Batch API's
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum BatchStatus: sys::CUFILEStatus_enum {
        /// required value prior to submission
        Waiting = CUFILE_WAITING,
        /// once enqueued
        Pending = CUFILE_PENDING,
        /// request was ill-formed or could not be enqueued
        Invalid = CUFILE_INVALID,
        /// request successfully canceled
        Canceled = CUFILE_CANCELED,
        /// request successfully completed
        Complete = CUFILE_COMPLETE,
        /// request timed out
        Timeout = CUFILE_TIMEOUT,
        /// unable to complete
        Failed = CUFILE_FAILED,
    }
}

impl_enum_display!(BatchStatus, {
    Self::Waiting => "CUFILE_WAITING",
    Self::Pending => "CUFILE_PENDING",
    Self::Invalid => "CUFILE_INVALID",
    Self::Canceled => "CUFILE_CANCELED",
    Self::Complete => "CUFILE_COMPLETE",
    Self::Timeout => "CUFILE_TIMEOUT",
    Self::Failed => "CUFILE_FAILED",
});

singe_core::native_enum! {
    /// Selectors for numeric configuration values read by [`size_value`](crate::config::size_value) and set by [`set_size_value`](crate::config::set_size_value).
    /// Size parameters with a `_KB` suffix are expressed in KB.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum SizeConfigParameter: sys::CUFileSizeTConfigParameter_t {
        /// Selects the profiling statistics level.
        ProfileStats = CUFILE_PARAM_PROFILE_STATS,
        /// Selects the maximum queued I/O request count.
        ExecutionMaxIoQueueDepth = CUFILE_PARAM_EXECUTION_MAX_IO_QUEUE_DEPTH,
        /// Selects the maximum number of I/O worker threads.
        ExecutionMaxIoThreads = CUFILE_PARAM_EXECUTION_MAX_IO_THREADS,
        /// Selects the minimum I/O size threshold for execution, in KB.
        ExecutionMinIoThresholdSizeKb = CUFILE_PARAM_EXECUTION_MIN_IO_THRESHOLD_SIZE_KB,
        /// Selects the maximum parallelism for an I/O request.
        ExecutionMaxRequestParallelism = CUFILE_PARAM_EXECUTION_MAX_REQUEST_PARALLELISM,
        /// Selects the maximum direct I/O transfer size sent to the filesystem, in KB.
        PropertiesMaxDirectIoSizeKb = CUFILE_PARAM_PROPERTIES_MAX_DIRECT_IO_SIZE_KB,
        /// Selects the maximum internal GPU cache space per device, in KB.
        PropertiesMaxDeviceCacheSizeKb = CUFILE_PARAM_PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB,
        /// Selects the size of an internal bounce buffer, in KB.
        PropertiesPerBufferCacheSizeKb = CUFILE_PARAM_PROPERTIES_PER_BUFFER_CACHE_SIZE_KB,
        /// Selects the maximum pinned device-memory space, in KB.
        PropertiesMaxDevicePinnedMemorySizeKb = CUFILE_PARAM_PROPERTIES_MAX_DEVICE_PINNED_MEM_SIZE_KB,
        /// Selects the batch I/O request count.
        PropertiesIoBatchSize = CUFILE_PARAM_PROPERTIES_IO_BATCHSIZE,
        /// Selects the maximum I/O size handled in polling mode, in KB.
        PollThresholdSizeKb = CUFILE_PARAM_POLLTHRESHOLD_SIZE_KB,
        /// Selects the batch I/O timeout, in milliseconds.
        PropertiesBatchIoTimeoutMilliseconds = CUFILE_PARAM_PROPERTIES_BATCH_IO_TIMEOUT_MS,
    }
}

impl_enum_display!(SizeConfigParameter, {
    Self::ProfileStats => "CUFILE_PARAM_PROFILE_STATS",
    Self::ExecutionMaxIoQueueDepth => "CUFILE_PARAM_EXECUTION_MAX_IO_QUEUE_DEPTH",
    Self::ExecutionMaxIoThreads => "CUFILE_PARAM_EXECUTION_MAX_IO_THREADS",
    Self::ExecutionMinIoThresholdSizeKb => "CUFILE_PARAM_EXECUTION_MIN_IO_THRESHOLD_SIZE_KB",
    Self::ExecutionMaxRequestParallelism => "CUFILE_PARAM_EXECUTION_MAX_REQUEST_PARALLELISM",
    Self::PropertiesMaxDirectIoSizeKb => "CUFILE_PARAM_PROPERTIES_MAX_DIRECT_IO_SIZE_KB",
    Self::PropertiesMaxDeviceCacheSizeKb => "CUFILE_PARAM_PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB",
    Self::PropertiesPerBufferCacheSizeKb => "CUFILE_PARAM_PROPERTIES_PER_BUFFER_CACHE_SIZE_KB",
    Self::PropertiesMaxDevicePinnedMemorySizeKb => "CUFILE_PARAM_PROPERTIES_MAX_DEVICE_PINNED_MEM_SIZE_KB",
    Self::PropertiesIoBatchSize => "CUFILE_PARAM_PROPERTIES_IO_BATCHSIZE",
    Self::PollThresholdSizeKb => "CUFILE_PARAM_POLLTHRESHOLD_SIZE_KB",
    Self::PropertiesBatchIoTimeoutMilliseconds => "CUFILE_PARAM_PROPERTIES_BATCH_IO_TIMEOUT_MS",
});

singe_core::native_enum! {
    /// Selectors for boolean configuration values read by [`bool_value`](crate::config::bool_value) and set by [`set_bool_value`](crate::config::set_bool_value).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum BoolConfigParameter: sys::CUFileBoolConfigParameter_t {
        /// Configure GDS fallback IO.
        GdsFallbackIo = CUFILE_PARAM_GDS_FALLBACK_IO,
        /// Configure vanilla POSIX IO mode.
        PropertiesPosixIoMode = CUFILE_PARAM_PROPERTIES_POSIX_IO_MODE,
        /// Selects whether polling mode is enabled.
        PropertiesUsePollMode = CUFILE_PARAM_PROPERTIES_USE_POLL_MODE,
        /// Selects whether compatibility I/O is allowed when GDS is unavailable.
        PropertiesAllowCompatMode = CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE,
        /// Selects whether compatibility I/O is forced.
        ForceCompatMode = CUFILE_PARAM_FORCE_COMPAT_MODE,
        /// Selects whether aggressive filesystem API checking is enabled.
        FileSystemMiscApiCheckAggressive = CUFILE_PARAM_FS_MISC_API_CHECK_AGGRESSIVE,
        /// Selects whether parallel I/O execution is enabled.
        ExecutionParallelIo = CUFILE_PARAM_EXECUTION_PARALLEL_IO,
        /// Do not use this macro.
        /// This is deprecated now.
        ProfileNvtx = CUFILE_PARAM_PROFILE_NVTX,
        /// Selects whether system-memory buffers are allowed for I/O.
        PropertiesAllowSystemMemory = CUFILE_PARAM_PROPERTIES_ALLOW_SYSTEM_MEMORY,
        /// Selects whether PCI peer-to-peer DMA is enabled.
        UsePciP2pDma = CUFILE_PARAM_USE_PCIP2PDMA,
        /// Selects whether the `io_uring` I/O backend is preferred.
        PreferIoUring = CUFILE_PARAM_PREFER_IO_URING,
        /// Selects whether direct I/O mode is forced.
        ForceODirectMode = CUFILE_PARAM_FORCE_ODIRECT_MODE,
        /// Selects whether storage and GPU topology detection is skipped.
        SkipTopologyDetection = CUFILE_PARAM_SKIP_TOPOLOGY_DETECTION,
        /// Selects whether stream memory operations are bypassed.
        StreamMemopsBypass = CUFILE_PARAM_STREAM_MEMOPS_BYPASS,
    }
}

impl_enum_display!(BoolConfigParameter, {
    Self::GdsFallbackIo => "CUFILE_PARAM_GDS_FALLBACK_IO",
    Self::PropertiesPosixIoMode => "CUFILE_PARAM_PROPERTIES_POSIX_IO_MODE",
    Self::PropertiesUsePollMode => "CUFILE_PARAM_PROPERTIES_USE_POLL_MODE",
    Self::PropertiesAllowCompatMode => "CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE",
    Self::ForceCompatMode => "CUFILE_PARAM_FORCE_COMPAT_MODE",
    Self::FileSystemMiscApiCheckAggressive => "CUFILE_PARAM_FS_MISC_API_CHECK_AGGRESSIVE",
    Self::ExecutionParallelIo => "CUFILE_PARAM_EXECUTION_PARALLEL_IO",
    Self::ProfileNvtx => "CUFILE_PARAM_PROFILE_NVTX",
    Self::PropertiesAllowSystemMemory => "CUFILE_PARAM_PROPERTIES_ALLOW_SYSTEM_MEMORY",
    Self::UsePciP2pDma => "CUFILE_PARAM_USE_PCIP2PDMA",
    Self::PreferIoUring => "CUFILE_PARAM_PREFER_IO_URING",
    Self::ForceODirectMode => "CUFILE_PARAM_FORCE_ODIRECT_MODE",
    Self::SkipTopologyDetection => "CUFILE_PARAM_SKIP_TOPOLOGY_DETECTION",
    Self::StreamMemopsBypass => "CUFILE_PARAM_STREAM_MEMOPS_BYPASS",
});

singe_core::native_enum! {
    /// Selectors for string configuration values read by [`string_value`](crate::config::string_value) and set by [`set_string_value`](crate::config::set_string_value).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum StringConfigParameter: sys::CUFileStringConfigParameter_t {
        /// Configure the RDMA transport.
        RdmaTransport = CUFILE_PARAM_RDMA_TRANSPORT,
        /// Selects the logging verbosity string.
        LoggingLevel = CUFILE_PARAM_LOGGING_LEVEL,
        /// Selects the log file path string.
        EnvironmentLogFilePath = CUFILE_PARAM_ENV_LOGFILE_PATH,
        /// Selects the directory used for log files.
        LogDirectory = CUFILE_PARAM_LOG_DIR,
    }
}

impl_enum_display!(StringConfigParameter, {
    Self::RdmaTransport => "CUFILE_PARAM_RDMA_TRANSPORT",
    Self::LoggingLevel => "CUFILE_PARAM_LOGGING_LEVEL",
    Self::EnvironmentLogFilePath => "CUFILE_PARAM_ENV_LOGFILE_PATH",
    Self::LogDirectory => "CUFILE_PARAM_LOG_DIR",
});

/// The amount of runtime statistics collected by cuFile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(i32)]
#[non_exhaustive]
pub enum StatisticsLevel {
    /// Disables statistics collection.
    Disabled = 0,
    /// Collects basic I/O counters and latency statistics.
    Basic = 1,
    /// Adds I/O size histograms to the basic statistics.
    Detailed = 2,
    /// Adds per-GPU statistics to the detailed statistics.
    PerGpu = 3,
}

impl_enum_display!(StatisticsLevel, {
    Self::Disabled => "CUFILE_STATS_LEVEL_DISABLED",
    Self::Basic => "CUFILE_STATS_LEVEL_BASIC",
    Self::Detailed => "CUFILE_STATS_LEVEL_DETAILED",
    Self::PerGpu => "CUFILE_STATS_LEVEL_PER_GPU",
});

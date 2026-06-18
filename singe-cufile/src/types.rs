use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cufile_sys as sys;

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct DriverStatusFlags: u32 {
        const LUSTRE_SUPPORTED = 1 << (sys::CUfileDriverStatusFlags::CU_FILE_LUSTRE_SUPPORTED as u32);
        const WEKAFS_SUPPORTED = 1 << (sys::CUfileDriverStatusFlags::CU_FILE_WEKAFS_SUPPORTED as u32);
        const NFS_SUPPORTED = 1 << (sys::CUfileDriverStatusFlags::CU_FILE_NFS_SUPPORTED as u32);
        const GPFS_SUPPORTED = 1 << (sys::CUfileDriverStatusFlags::CU_FILE_GPFS_SUPPORTED as u32);
        const NVME_SUPPORTED = 1 << (sys::CUfileDriverStatusFlags::CU_FILE_NVME_SUPPORTED as u32);
        const NVMEOF_SUPPORTED = 1 << (sys::CUfileDriverStatusFlags::CU_FILE_NVMEOF_SUPPORTED as u32);
        const SCSI_SUPPORTED = 1 << (sys::CUfileDriverStatusFlags::CU_FILE_SCSI_SUPPORTED as u32);
        const SCALEFLUX_CSD_SUPPORTED = 1 << (sys::CUfileDriverStatusFlags::CU_FILE_SCALEFLUX_CSD_SUPPORTED as u32);
        const NVMESH_SUPPORTED = 1 << (sys::CUfileDriverStatusFlags::CU_FILE_NVMESH_SUPPORTED as u32);
        const BEEGFS_SUPPORTED = 1 << (sys::CUfileDriverStatusFlags::CU_FILE_BEEGFS_SUPPORTED as u32);
        const NVME_P2P_SUPPORTED = 1 << (sys::CUfileDriverStatusFlags::CU_FILE_NVME_P2P_SUPPORTED as u32);
        const SCATEFS_SUPPORTED = 1 << (sys::CUfileDriverStatusFlags::CU_FILE_SCATEFS_SUPPORTED as u32);
        const VIRTIOFS_SUPPORTED = 1 << (sys::CUfileDriverStatusFlags::CU_FILE_VIRTIOFS_SUPPORTED as u32);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum DriverStatus {
    LustreSupported = sys::CUfileDriverStatusFlags::CU_FILE_LUSTRE_SUPPORTED as _,
    WekaFsSupported = sys::CUfileDriverStatusFlags::CU_FILE_WEKAFS_SUPPORTED as _,
    NfsSupported = sys::CUfileDriverStatusFlags::CU_FILE_NFS_SUPPORTED as _,
    GpfsSupported = sys::CUfileDriverStatusFlags::CU_FILE_GPFS_SUPPORTED as _,
    NvmeSupported = sys::CUfileDriverStatusFlags::CU_FILE_NVME_SUPPORTED as _,
    NvmeOfSupported = sys::CUfileDriverStatusFlags::CU_FILE_NVMEOF_SUPPORTED as _,
    ScsiSupported = sys::CUfileDriverStatusFlags::CU_FILE_SCSI_SUPPORTED as _,
    ScalefluxCsdSupported = sys::CUfileDriverStatusFlags::CU_FILE_SCALEFLUX_CSD_SUPPORTED as _,
    NvmeshSupported = sys::CUfileDriverStatusFlags::CU_FILE_NVMESH_SUPPORTED as _,
    BeeGfsSupported = sys::CUfileDriverStatusFlags::CU_FILE_BEEGFS_SUPPORTED as _,
    NvmeP2pSupported = sys::CUfileDriverStatusFlags::CU_FILE_NVME_P2P_SUPPORTED as _,
    ScateFsSupported = sys::CUfileDriverStatusFlags::CU_FILE_SCATEFS_SUPPORTED as _,
    VirtioFsSupported = sys::CUfileDriverStatusFlags::CU_FILE_VIRTIOFS_SUPPORTED as _,
    MaxTargetTypes = sys::CUfileDriverStatusFlags::CU_FILE_MAX_TARGET_TYPES as _,
}

impl_enum_conversion!(sys::CUfileDriverStatusFlags, DriverStatus);

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
    Self::MaxTargetTypes => "CU_FILE_MAX_TARGET_TYPES",
});

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct DriverControlFlags: u32 {
        const USE_POLL_MODE = 1 << (sys::CUfileDriverControlFlags::CU_FILE_USE_POLL_MODE as u32);
        const ALLOW_COMPAT_MODE = 1 << (sys::CUfileDriverControlFlags::CU_FILE_ALLOW_COMPAT_MODE as u32);
        const POSIX_IO_MODE = 1 << (sys::CUfileDriverControlFlags::CU_FILE_POSIX_IO_MODE as u32);
        const FALLBACK_IO_MODE = 1 << (sys::CUfileDriverControlFlags::CU_FILE_FALLBACK_IO_MODE as u32);
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct FeatureFlags: u32 {
        const DYN_ROUTING_SUPPORTED = 1 << (sys::CUfileFeatureFlags::CU_FILE_DYN_ROUTING_SUPPORTED as u32);
        const BATCH_IO_SUPPORTED = 1 << (sys::CUfileFeatureFlags::CU_FILE_BATCH_IO_SUPPORTED as u32);
        const STREAMS_SUPPORTED = 1 << (sys::CUfileFeatureFlags::CU_FILE_STREAMS_SUPPORTED as u32);
        const PARALLEL_IO_SUPPORTED = 1 << (sys::CUfileFeatureFlags::CU_FILE_PARALLEL_IO_SUPPORTED as u32);
        const P2P_SUPPORTED = 1 << (sys::CUfileFeatureFlags::CU_FILE_P2P_SUPPORTED as u32);
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct BufferRegisterFlags: i32 {
        const RDMA_REGISTER = sys::CU_FILE_RDMA_REGISTER as i32;
        const RDMA_RELAXED_ORDERING = sys::CU_FILE_RDMA_RELAXED_ORDERING as i32;
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct StreamRegisterFlags: u32 {
        const FIXED_BUFFER_OFFSET = sys::CU_FILE_STREAM_FIXED_BUF_OFFSET;
        const FIXED_FILE_OFFSET = sys::CU_FILE_STREAM_FIXED_FILE_OFFSET;
        const FIXED_FILE_SIZE = sys::CU_FILE_STREAM_FIXED_FILE_SIZE;
        const PAGE_ALIGNED_INPUTS = sys::CU_FILE_STREAM_PAGE_ALIGNED_INPUTS;
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct P2PFlags: u32 {
        const PCI_P2PDMA = 1 << sys::CUfileP2PFlags::CUFILE_P2PDMA.0;
        const NVFS = 1 << sys::CUfileP2PFlags::CUFILE_NVFS.0;
        const DMABUF = 1 << sys::CUfileP2PFlags::CUFILE_DMABUF.0;
        const C2C = 1 << sys::CUfileP2PFlags::CUFILE_C2C.0;
        const NVIDIA_PEERMEM = 1 << sys::CUfileP2PFlags::CUFILE_NVIDIA_PEERMEM.0;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Operation {
    Read = sys::CUfileOpcode::CUFILE_READ as _,
    Write = sys::CUfileOpcode::CUFILE_WRITE as _,
}

impl_enum_conversion!(sys::CUfileOpcode, Operation);

impl_enum_display!(Operation, {
    Self::Read => "CUFILE_READ",
    Self::Write => "CUFILE_WRITE",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum BatchStatus {
    Waiting = sys::CUFILEStatus_enum::CUFILE_WAITING as _,
    Pending = sys::CUFILEStatus_enum::CUFILE_PENDING as _,
    Invalid = sys::CUFILEStatus_enum::CUFILE_INVALID as _,
    Canceled = sys::CUFILEStatus_enum::CUFILE_CANCELED as _,
    Complete = sys::CUFILEStatus_enum::CUFILE_COMPLETE as _,
    Timeout = sys::CUFILEStatus_enum::CUFILE_TIMEOUT as _,
    Failed = sys::CUFILEStatus_enum::CUFILE_FAILED as _,
}

impl_enum_conversion!(sys::CUFILEStatus_enum, BatchStatus);

impl_enum_display!(BatchStatus, {
    Self::Waiting => "CUFILE_WAITING",
    Self::Pending => "CUFILE_PENDING",
    Self::Invalid => "CUFILE_INVALID",
    Self::Canceled => "CUFILE_CANCELED",
    Self::Complete => "CUFILE_COMPLETE",
    Self::Timeout => "CUFILE_TIMEOUT",
    Self::Failed => "CUFILE_FAILED",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum SizeConfigParameter {
    ProfileStats = sys::CUFileSizeTConfigParameter_t::CUFILE_PARAM_PROFILE_STATS as _,
    ExecutionMaxIoQueueDepth =
        sys::CUFileSizeTConfigParameter_t::CUFILE_PARAM_EXECUTION_MAX_IO_QUEUE_DEPTH as _,
    ExecutionMaxIoThreads =
        sys::CUFileSizeTConfigParameter_t::CUFILE_PARAM_EXECUTION_MAX_IO_THREADS as _,
    ExecutionMinIoThresholdSizeKb =
        sys::CUFileSizeTConfigParameter_t::CUFILE_PARAM_EXECUTION_MIN_IO_THRESHOLD_SIZE_KB as _,
    ExecutionMaxRequestParallelism =
        sys::CUFileSizeTConfigParameter_t::CUFILE_PARAM_EXECUTION_MAX_REQUEST_PARALLELISM as _,
    PropertiesMaxDirectIoSizeKb =
        sys::CUFileSizeTConfigParameter_t::CUFILE_PARAM_PROPERTIES_MAX_DIRECT_IO_SIZE_KB as _,
    PropertiesMaxDeviceCacheSizeKb =
        sys::CUFileSizeTConfigParameter_t::CUFILE_PARAM_PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB as _,
    PropertiesPerBufferCacheSizeKb =
        sys::CUFileSizeTConfigParameter_t::CUFILE_PARAM_PROPERTIES_PER_BUFFER_CACHE_SIZE_KB as _,
    PropertiesMaxDevicePinnedMemorySizeKb =
        sys::CUFileSizeTConfigParameter_t::CUFILE_PARAM_PROPERTIES_MAX_DEVICE_PINNED_MEM_SIZE_KB
            as _,
    PropertiesIoBatchSize =
        sys::CUFileSizeTConfigParameter_t::CUFILE_PARAM_PROPERTIES_IO_BATCHSIZE as _,
    PollThresholdSizeKb =
        sys::CUFileSizeTConfigParameter_t::CUFILE_PARAM_POLLTHRESHOLD_SIZE_KB as _,
    PropertiesBatchIoTimeoutMilliseconds =
        sys::CUFileSizeTConfigParameter_t::CUFILE_PARAM_PROPERTIES_BATCH_IO_TIMEOUT_MS as _,
}

impl_enum_conversion!(sys::CUFileSizeTConfigParameter_t, SizeConfigParameter);

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum BoolConfigParameter {
    PropertiesUsePollMode =
        sys::CUFileBoolConfigParameter_t::CUFILE_PARAM_PROPERTIES_USE_POLL_MODE as _,
    PropertiesAllowCompatMode =
        sys::CUFileBoolConfigParameter_t::CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE as _,
    ForceCompatMode = sys::CUFileBoolConfigParameter_t::CUFILE_PARAM_FORCE_COMPAT_MODE as _,
    FileSystemMiscApiCheckAggressive =
        sys::CUFileBoolConfigParameter_t::CUFILE_PARAM_FS_MISC_API_CHECK_AGGRESSIVE as _,
    ExecutionParallelIo = sys::CUFileBoolConfigParameter_t::CUFILE_PARAM_EXECUTION_PARALLEL_IO as _,
    ProfileNvtx = sys::CUFileBoolConfigParameter_t::CUFILE_PARAM_PROFILE_NVTX as _,
    PropertiesAllowSystemMemory =
        sys::CUFileBoolConfigParameter_t::CUFILE_PARAM_PROPERTIES_ALLOW_SYSTEM_MEMORY as _,
    UsePciP2pDma = sys::CUFileBoolConfigParameter_t::CUFILE_PARAM_USE_PCIP2PDMA as _,
    PreferIoUring = sys::CUFileBoolConfigParameter_t::CUFILE_PARAM_PREFER_IO_URING as _,
    ForceODirectMode = sys::CUFileBoolConfigParameter_t::CUFILE_PARAM_FORCE_ODIRECT_MODE as _,
    SkipTopologyDetection =
        sys::CUFileBoolConfigParameter_t::CUFILE_PARAM_SKIP_TOPOLOGY_DETECTION as _,
    StreamMemopsBypass = sys::CUFileBoolConfigParameter_t::CUFILE_PARAM_STREAM_MEMOPS_BYPASS as _,
}

impl_enum_conversion!(sys::CUFileBoolConfigParameter_t, BoolConfigParameter);

impl_enum_display!(BoolConfigParameter, {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum StringConfigParameter {
    LoggingLevel = sys::CUFileStringConfigParameter_t::CUFILE_PARAM_LOGGING_LEVEL as _,
    EnvironmentLogFilePath = sys::CUFileStringConfigParameter_t::CUFILE_PARAM_ENV_LOGFILE_PATH as _,
    LogDirectory = sys::CUFileStringConfigParameter_t::CUFILE_PARAM_LOG_DIR as _,
}

impl_enum_conversion!(sys::CUFileStringConfigParameter_t, StringConfigParameter);

impl_enum_display!(StringConfigParameter, {
    Self::LoggingLevel => "CUFILE_PARAM_LOGGING_LEVEL",
    Self::EnvironmentLogFilePath => "CUFILE_PARAM_ENV_LOGFILE_PATH",
    Self::LogDirectory => "CUFILE_PARAM_LOG_DIR",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(i32)]
#[non_exhaustive]
pub enum StatisticsLevel {
    Disabled = 0,
    Basic = 1,
    Detailed = 2,
    PerGpu = 3,
}

impl_enum_display!(StatisticsLevel, {
    Self::Disabled => "CUFILE_STATS_LEVEL_DISABLED",
    Self::Basic => "CUFILE_STATS_LEVEL_BASIC",
    Self::Detailed => "CUFILE_STATS_LEVEL_DETAILED",
    Self::PerGpu => "CUFILE_STATS_LEVEL_PER_GPU",
});

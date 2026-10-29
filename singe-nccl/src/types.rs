use std::{
    ffi::CStr,
    fmt::{self, Display, Formatter},
    mem::{self, size_of},
    ptr,
};

use bitflags::parser;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cuda::{
    types::{bf16, f8e4m3, f8e5m2, f16},
    view::{AnyBitPatternDeviceRepr, NoUninitDeviceRepr},
};
use singe_nccl_sys as sys;

const CONFIG_UNDEF_INT: i32 = i32::MIN;
const UNIQUE_ID_BYTE_LEN: usize = sys::NCCL_UNIQUE_ID_BYTES as usize;

/// An opaque NCCL communicator-clique identifier that can be distributed as bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UniqueId {
    bytes: [u8; UNIQUE_ID_BYTE_LEN],
}

impl UniqueId {
    /// Reconstructs a unique identifier from its native byte representation.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; UNIQUE_ID_BYTE_LEN]) -> Self {
        Self { bytes }
    }

    /// Returns the native bytes for distributing this identifier to other ranks.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; UNIQUE_ID_BYTE_LEN] {
        &self.bytes
    }

    pub(crate) fn into_raw(self) -> sys::ncclUniqueId {
        let mut raw = sys::ncclUniqueId::default();
        for (dst, src) in raw.internal.iter_mut().zip(self.bytes) {
            *dst = src.cast_signed();
        }
        raw
    }
}

impl From<sys::ncclUniqueId> for UniqueId {
    fn from(value: sys::ncclUniqueId) -> Self {
        let mut bytes = [0_u8; UNIQUE_ID_BYTE_LEN];
        for (dst, src) in bytes.iter_mut().zip(value.internal) {
            *dst = src as u8;
        }
        Self { bytes }
    }
}

/// The lifecycle state tracked by the communicator wrapper.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CommunicatorState {
    /// Initialization is in progress. Poll the communicator asynchronous status before use.
    Initializing,
    /// The communicator is initialized and can enqueue operations.
    Active,
    /// Communicator resources are suspended and collective operations are unavailable.
    Suspended,
    /// Finalization has begun; poll asynchronous status before destroying the communicator.
    Finalizing,
    /// The communicator has been revoked and cannot enqueue new operations.
    Revoked,
    /// The communicator has been aborted and its native resources released.
    Aborted,
    /// The communicator has been destroyed.
    Destroyed,
}

impl CommunicatorState {
    pub(crate) const fn into_raw(self) -> u8 {
        match self {
            Self::Initializing => 6,
            Self::Active => 0,
            Self::Suspended => 5,
            Self::Finalizing => 1,
            Self::Revoked => 2,
            Self::Aborted => 3,
            Self::Destroyed => 4,
        }
    }

    pub(crate) const fn from_raw(value: u8) -> Self {
        match value {
            6 => Self::Initializing,
            0 => Self::Active,
            5 => Self::Suspended,
            1 => Self::Finalizing,
            2 => Self::Revoked,
            3 => Self::Aborted,
            _ => Self::Destroyed,
        }
    }

    /// Returns the lowercase name of this state.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Initializing => "initializing",
            Self::Active => "active",
            Self::Suspended => "suspended",
            Self::Finalizing => "finalizing",
            Self::Revoked => "revoked",
            Self::Aborted => "aborted",
            Self::Destroyed => "destroyed",
        }
    }
}

/// Defines the reduction operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ReductionOperator {
    /// Perform a sum (+) operation.
    Sum = sys::ncclRedOp_t::ncclSum as _,
    /// Perform a product (\*) operation.
    Product = sys::ncclRedOp_t::ncclProd as _,
    /// Perform a max operation.
    Maximum = sys::ncclRedOp_t::ncclMax as _,
    /// Perform a min operation.
    Minimum = sys::ncclRedOp_t::ncclMin as _,
    /// Perform an average operation: a sum across all ranks divided by the number of ranks.
    Average = sys::ncclRedOp_t::ncclAvg as _,
}

impl_enum_conversion!(sys::ncclRedOp_t, ReductionOperator);

/// Participation and group selection for communicator splitting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SplitColor {
    /// Excludes this rank from the split communicator.
    Undefined,
    /// Places this rank in the group identified by the given color.
    Value(i32),
}

impl SplitColor {
    /// Returns the native split color, including the no-color sentinel.
    #[must_use]
    pub const fn into_raw(self) -> i32 {
        match self {
            Self::Undefined => sys::NCCL_SPLIT_NOCOLOR,
            Self::Value(value) => value,
        }
    }
}

/// Indicates where scalar arguments reside and when they can be dereferenced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ScalarResidence {
    /// The scalar resides in device-visible memory and is dereferenced when needed.
    Device = sys::ncclScalarResidence_t::ncclScalarDevice as _,
    /// The scalar resides in host memory and is dereferenced immediately.
    HostImmediate = sys::ncclScalarResidence_t::ncclScalarHostImmediate as _,
}

impl_enum_conversion!(sys::ncclScalarResidence_t, ScalarResidence);

/// The policy used to balance communication thread blocks against other GPU work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum CtaPolicy {
    /// Use the default CTA policy for NCCL communicator. In this policy, NCCL will automatically adjust resource usage and achieve
    /// maximal performance. This policy is suitable for most applications.
    Default = sys::NCCL_CTA_POLICY_DEFAULT,
    /// Use the CTA efficiency policy for NCCL communicator. In this policy, NCCL will optimize CTA usage and use minimal
    /// number of CTAs to achieve the decent performance when possible. This policy is suitable for applications which require
    /// better compute and communication overlap.
    Efficiency = sys::NCCL_CTA_POLICY_EFFICIENCY,
    /// Use the Zero-CTA policy for NCCL communicator. In this policy, NCCL will use zero CTA whenever it can, even when that choice
    /// may sacrifice some performance. Select this mode when your application must preserve the maximum number of CTAs for compute kernels.
    Zero = sys::NCCL_CTA_POLICY_ZERO,
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    /// Options controlling communicator shrinking and recovery from pending operations.
    pub struct ShrinkFlags: u32 {
        /// These flags modify the behavior of the `ncclCommShrink` operation.
        ///
        /// Default behavior. Shrink the parent communicator without affecting ongoing operations.
        /// Value: `0x00`.
        const DEFAULT = sys::NCCL_SHRINK_DEFAULT;
        /// First, terminate ongoing parent communicator operations, and then proceed with shrinking the communicator.
        /// This is used for error recovery scenarios where the parent communicator might be in a hung state.
        /// Resources of parent comm are still not freed, users should decide whether to call ncclCommAbort on the parent communicator after shrink.
        /// Value: `0x01`.
        const ABORT = sys::NCCL_SHRINK_ABORT;
    }
}

impl Display for ShrinkFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    /// Options controlling registration and access to an NCCL communication window.
    pub struct WindowFlags: u32 {
        /// Register buffer into NCCL window with default behavior. The default behavior allows users to
        /// pass any offset to the buffer head address as the input of NCCL collective operations. However,
        /// this behavior can cause suboptimal performance in NCCL due to the asymmetric buffer usage.
        const DEFAULT = sys::NCCL_WIN_DEFAULT;
        /// Register buffer into NCCL window, and users need to guarantee the offset to the buffer head address
        /// from all ranks must be equal when calling NCCL collective operations. It allows NCCL to operate
        /// buffer in a symmetric way and provide the best performance.
        const COLLECTIVE_SYMMETRIC = sys::NCCL_WIN_COLL_SYMMETRIC;
        /// Register buffer into NCCL window while ensuring strict ordering for window operations using the IB Verbs transport.
        /// This flag is mostly intended for buffers used for GIN VA Signals (see [Signals and Counters](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2323/user-guide/docs/api/device_gin.html#devapi-signals)).
        const STRICT_ORDERING = sys::NCCL_WIN_STRICT_ORDERING;
        /// Register buffer into NCCL window for GIN use only.
        const GIN_ONLY = sys::NCCL_WIN_GIN_ONLY;
        /// Register buffer into NCCL window with CFT counted support.
        /// Registered buffers must be aligned to 256 bytes and are used in counted operations (see [CFT counted operations](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2323/user-guide/docs/api/device_cft.html#cft-counted-operations)).
        /// The user cannot exceed the 256GB limit of registered memory for counted windows.
        const CFT_COUNTED = sys::NCCL_WIN_CFT_COUNTED;
    }
}

impl Display for WindowFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

/// NCCL scalar data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum DataType {
    /// Signed 8-bit integer.
    I8 = sys::ncclDataType_t::ncclInt8 as _,
    /// Unsigned 8-bit integer.
    U8 = sys::ncclDataType_t::ncclUint8 as _,
    /// Signed 32-bit integer.
    I32 = sys::ncclDataType_t::ncclInt32 as _,
    /// Unsigned 32-bit integer.
    U32 = sys::ncclDataType_t::ncclUint32 as _,
    /// Signed 64-bit integer.
    I64 = sys::ncclDataType_t::ncclInt64 as _,
    /// Unsigned 64-bit integer.
    U64 = sys::ncclDataType_t::ncclUint64 as _,
    /// 16-bit floating-point number (half precision).
    F16 = sys::ncclDataType_t::ncclFloat16 as _,
    /// 32-bit floating-point number (single precision).
    F32 = sys::ncclDataType_t::ncclFloat32 as _,
    /// 64-bit floating-point number (double precision).
    F64 = sys::ncclDataType_t::ncclFloat64 as _,
    /// 16-bit floating-point number with truncated bfloat16 precision (CUDA 11 or later).
    Bf16 = sys::ncclDataType_t::ncclBfloat16 as _,
    /// 8-bit floating-point number with 4 exponent bits and 3 mantissa bits (CUDA &gt;= 11.8 and SM &gt;= 90).
    F8E4M3 = sys::ncclDataType_t::ncclFloat8e4m3 as _,
    /// 8-bit floating-point number with 5 exponent bits and 2 mantissa bits (CUDA &gt;= 11.8 and SM &gt;= 90).
    F8E5M2 = sys::ncclDataType_t::ncclFloat8e5m2 as _,
}

impl_enum_conversion!(sys::ncclDataType_t, DataType);

impl DataType {
    /// Returns the storage size of one element, in bytes.
    #[must_use]
    pub const fn byte_len(self) -> usize {
        match self {
            Self::I8 | Self::U8 | Self::F8E4M3 | Self::F8E5M2 => 1,
            Self::F16 | Self::Bf16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
        }
    }
}

mod sealed {
    pub trait DataType {}
}

/// A supported NCCL element with an exact native ABI mapping.
pub trait DataTypeLike:
    sealed::DataType + AnyBitPatternDeviceRepr + NoUninitDeviceRepr + fmt::Debug
{
    /// The NCCL data type with the same representation as this Rust element type.
    const DATA_TYPE: DataType;
}

macro_rules! impl_nccl_data_type {
    ($ty:ty, $data_type:ident) => {
        impl sealed::DataType for $ty {}
        impl DataTypeLike for $ty {
            const DATA_TYPE: DataType = DataType::$data_type;
        }
    };
}

impl_nccl_data_type!(i8, I8);

impl_nccl_data_type!(u8, U8);

impl_nccl_data_type!(i32, I32);

impl_nccl_data_type!(u32, U32);

impl_nccl_data_type!(i64, I64);

impl_nccl_data_type!(u64, U64);

impl_nccl_data_type!(f16, F16);

impl_nccl_data_type!(bf16, Bf16);

impl_nccl_data_type!(f32, F32);

impl_nccl_data_type!(f64, F64);

impl_nccl_data_type!(f8e4m3, F8E4M3);

impl_nccl_data_type!(f8e5m2, F8E5M2);

/// Values for the [`Config::host_cft_mode`] communicator configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(i32)]
#[non_exhaustive]
pub enum HostCftMode {
    /// Use the version-specific default.
    Default = sys::ncclHostCftMode_t::ncclHostCftDefault as _,
    /// Enable host-side CFT support.
    Enable = sys::ncclHostCftMode_t::ncclHostCftEnable as _,
    /// Disable host-side CFT support.
    Disable = sys::ncclHostCftMode_t::ncclHostCftDisable as _,
    /// Try to create CFT logical endpoints.
    /// In case of an error, host-side CFT will be disabled.
    Fallback = sys::ncclHostCftMode_t::ncclHostCftFallback as _,
}

impl_enum_conversion!(i32, sys::ncclHostCftMode_t, HostCftMode);

impl_enum_display!(HostCftMode, {
    Self::Default => "ncclHostCftDefault",
    Self::Enable => "ncclHostCftEnable",
    Self::Disable => "ncclHostCftDisable",
    Self::Fallback => "ncclHostCftFallback",
});

bitflags::bitflags! {
    /// Host NVLS components to disable; empty flags enable all components.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    pub struct NvlsHostFlags: i32 {
        /// Disable the host NVLS transport.
        const DISABLE_TRANSPORT = sys::ncclNvlsHostMode_t::ncclNvlsHostModeDisableTransport as _;
        /// Disable host NVLS symmetric multimemory support.
        const DISABLE_SYMMETRIC_MULTIMEM = sys::ncclNvlsHostMode_t::ncclNvlsHostModeDisableSymmetricMultimem as _;
        /// Disable host NVLS support.
        const DISABLE = sys::ncclNvlsHostMode_t::ncclNvlsHostModeDisable as _;
    }
}

impl Display for NvlsHostFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

/// Optional overrides used when creating an NCCL communicator.
///
/// `None` leaves a field at the native undefined value so NCCL can apply its defaults or environment settings.
#[derive(Debug, Clone, Copy, Default)]
pub struct Config<'a> {
    /// Whether communicator operations use blocking behavior; the native default is blocking.
    pub blocking: Option<bool>,
    /// Cooperative Group Array size for NCCL kernels, from 0 through 8.
    ///
    /// The native default is 4 on SM90 and newer architectures and 0 on older devices.
    pub cga_cluster_size: Option<i32>,
    /// Minimum communication thread blocks per kernel; the documented range is 1 through 32.
    pub min_ctas: Option<i32>,
    /// Maximum communication thread blocks per kernel; the documented range is 1 through 32.
    pub max_ctas: Option<i32>,
    /// Network module name, matched without case sensitivity; for example, `IB` or `Socket`.
    pub net_name: Option<&'a CStr>,
    /// Whether child communicators may share parent resources during split, encoded as 0 or 1.
    ///
    /// Sharing resources can make an abort affect other communicators in the same family.
    pub split_share: Option<i32>,
    /// Traffic class for network operations.
    pub traffic_class: Option<i32>,
    /// A user-defined communicator name.
    pub comm_name: Option<&'a CStr>,
    /// Whether to enable InfiniBand SHARP collective offload; disabled by default.
    pub collnet_enable: Option<bool>,
    /// Policy for balancing communication thread blocks against computation.
    pub cta_policy: Option<CtaPolicy>,
    /// Whether shrinking may share resources with the parent communicator, encoded as 0 or 1.
    pub shrink_share: Option<i32>,
    /// Number of thread blocks used by `NVLink` SHARP kernels.
    pub nvls_ctas: Option<i32>,
    /// Network channels per communicating peer, rounded up to a power of two.
    ///
    /// The value must be positive.
    pub channel_count_per_net_peer: Option<i32>,
    /// Selection of the native NVLink-centric scheduling mode.
    pub nvlink_centric_sched: Option<i32>,
    /// Graph usage: 0 for no graphs, 1 for one graph, or 2 for multiple graphs or mixed graph and direct calls.
    ///
    /// Mode 2 requires capture-time stream ordering; do not combine it with `graph_stream_ordering = Some(false)`.
    pub graph_usage_mode: Option<i32>,
    /// Number of one-sided remote-memory communication contexts; the native default is 1.
    pub num_rma_contexts: Option<i32>,
    /// Maximum peers a rank communicates with concurrently through point-to-point operations.
    ///
    /// Values below 1 or above the rank count select all ranks.
    pub max_p2p_peers: Option<i32>,
    /// Whether NCCL serializes communication kernels during graph capture.
    ///
    /// When disabled, the application must establish correct ordering.
    pub graph_stream_ordering: Option<bool>,
    /// Whether to enable NCCL implicit launch ordering for this communicator.
    pub launch_order_implicit: Option<bool>,
    /// Number of one-sided signal indices per communication context; the native default is 1.
    pub num_rma_signals: Option<i32>,
    /// Whether to initialize one-sided signaling during communicator creation.
    ///
    /// Otherwise initialization occurs collectively during the first window registration.
    pub rma_eager_init: Option<bool>,
    /// Support mode for host-side Compute Fabric Transport queries.
    pub host_cft_mode: Option<HostCftMode>,
    /// Host-side `NVLink` SHARP components to disable.
    ///
    /// Explicit device-API multimem requests remain available.
    pub nvls_host_flags: Option<NvlsHostFlags>,
}

impl Config<'_> {
    /// Builds an initialized native configuration with these overrides.
    ///
    /// String pointers borrow the original C strings and must not outlive them.
    pub fn into_raw(self) -> sys::ncclConfig_t {
        // SAFETY: `ncclConfig_t` is a plain C struct for which all-zero bytes are a valid value.
        let mut raw: sys::ncclConfig_t = unsafe { mem::zeroed() };
        raw.size = size_of::<sys::ncclConfig_t>() as _;
        raw.magic = 0xcafe_beef;
        raw.version = sys::NCCL_VERSION_CODE;
        raw.blocking = self.blocking.map_or(CONFIG_UNDEF_INT, i32::from);
        raw.cgaClusterSize = self.cga_cluster_size.unwrap_or(CONFIG_UNDEF_INT);
        raw.minCTAs = self.min_ctas.unwrap_or(CONFIG_UNDEF_INT);
        raw.maxCTAs = self.max_ctas.unwrap_or(CONFIG_UNDEF_INT);
        raw.netName = self.net_name.map_or(ptr::null(), CStr::as_ptr);
        raw.splitShare = self.split_share.unwrap_or(CONFIG_UNDEF_INT);
        raw.trafficClass = self.traffic_class.unwrap_or(CONFIG_UNDEF_INT);
        raw.commName = self.comm_name.map_or(ptr::null(), CStr::as_ptr);
        raw.collnetEnable = self.collnet_enable.map_or(CONFIG_UNDEF_INT, i32::from);
        raw.CTAPolicy = self
            .cta_policy
            .map_or(CONFIG_UNDEF_INT, |policy| policy as i32);
        raw.shrinkShare = self.shrink_share.unwrap_or(CONFIG_UNDEF_INT);
        raw.nvlsCTAs = self.nvls_ctas.unwrap_or(CONFIG_UNDEF_INT);
        raw.nChannelsPerNetPeer = self.channel_count_per_net_peer.unwrap_or(CONFIG_UNDEF_INT);
        raw.nvlinkCentricSched = self.nvlink_centric_sched.unwrap_or(CONFIG_UNDEF_INT);
        raw.graphUsageMode = self.graph_usage_mode.unwrap_or(CONFIG_UNDEF_INT);
        raw.numRmaCtx = self.num_rma_contexts.unwrap_or(CONFIG_UNDEF_INT);
        raw.maxP2pPeers = self.max_p2p_peers.unwrap_or(CONFIG_UNDEF_INT);
        raw.graphStreamOrdering = self
            .graph_stream_ordering
            .map_or(CONFIG_UNDEF_INT, i32::from);
        raw.launchOrderImplicit = self
            .launch_order_implicit
            .map_or(CONFIG_UNDEF_INT, i32::from);
        raw.numRmaSig = self.num_rma_signals.unwrap_or(CONFIG_UNDEF_INT);
        raw.rmaEagerInit = self.rma_eager_init.map_or(CONFIG_UNDEF_INT, i32::from);
        raw.hostCftMode = self.host_cft_mode.map_or(CONFIG_UNDEF_INT, i32::from);
        raw.nvlsHostMode = self
            .nvls_host_flags
            .map_or(CONFIG_UNDEF_INT, |flags| flags.bits());
        raw
    }
}

/// Estimated execution metadata returned by NCCL group simulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimulationInfo {
    /// The estimated execution time reported by NCCL for the grouped operations.
    pub estimated_time: f32,
}

impl SimulationInfo {
    /// Builds an initialized native simulation-info structure.
    #[must_use]
    pub fn into_raw(self) -> sys::ncclSimInfo_t {
        sys::ncclSimInfo_t {
            size: size_of::<sys::ncclSimInfo_t>() as _,
            magic: 0x7468_5283,
            version: sys::NCCL_VERSION_CODE,
            estimatedTime: self.estimated_time,
        }
    }
}

impl Default for SimulationInfo {
    fn default() -> Self {
        Self {
            estimated_time: -1.0,
        }
    }
}

impl From<sys::ncclSimInfo_t> for SimulationInfo {
    fn from(value: sys::ncclSimInfo_t) -> Self {
        Self {
            estimated_time: value.estimatedTime,
        }
    }
}

impl_enum_display!(ReductionOperator, {
    ReductionOperator::Sum => "ncclSum",
    ReductionOperator::Product => "ncclProd",
    ReductionOperator::Maximum => "ncclMax",
    ReductionOperator::Minimum => "ncclMin",
    ReductionOperator::Average => "ncclAvg",
});

impl Display for CommunicatorState {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl_enum_display!(DataType, {
    DataType::I8 => "ncclInt8",
    DataType::U8 => "ncclUint8",
    DataType::I32 => "ncclInt32",
    DataType::U32 => "ncclUint32",
    DataType::I64 => "ncclInt64",
    DataType::U64 => "ncclUint64",
    DataType::F16 => "ncclFloat16",
    DataType::F32 => "ncclFloat32",
    DataType::F64 => "ncclFloat64",
    DataType::Bf16 => "ncclBfloat16",
    DataType::F8E4M3 => "ncclFloat8e4m3",
    DataType::F8E5M2 => "ncclFloat8e5m2",
});

impl_enum_display!(ScalarResidence, {
    ScalarResidence::Device => "ncclScalarDevice",
    ScalarResidence::HostImmediate => "ncclScalarHostImmediate",
});

impl_enum_display!(CtaPolicy, {
    CtaPolicy::Default => "NCCL_CTA_POLICY_DEFAULT",
    CtaPolicy::Efficiency => "NCCL_CTA_POLICY_EFFICIENCY",
    CtaPolicy::Zero => "NCCL_CTA_POLICY_ZERO",
});

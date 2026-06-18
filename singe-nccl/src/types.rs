use std::{
    ffi::CStr,
    fmt::{self, Display, Formatter},
    mem::{self, size_of},
};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cuda::types::{bf16, f8e4m3, f8e5m2, f16};
use singe_nccl_sys as sys;

const CONFIG_UNDEF_INT: i32 = i32::MIN;
const UNIQUE_ID_BYTE_LEN: usize = sys::NCCL_UNIQUE_ID_BYTES as usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UniqueId {
    bytes: [u8; UNIQUE_ID_BYTE_LEN],
}

impl UniqueId {
    pub const fn from_bytes(bytes: [u8; UNIQUE_ID_BYTE_LEN]) -> Self {
        Self { bytes }
    }

    pub const fn as_bytes(&self) -> &[u8; UNIQUE_ID_BYTE_LEN] {
        &self.bytes
    }

    pub(crate) fn into_raw(self) -> sys::ncclUniqueId {
        let mut raw = sys::ncclUniqueId::default();
        for (dst, src) in raw.internal.iter_mut().zip(self.bytes) {
            *dst = src as i8;
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

/// Status codes returned by NCCL operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Status {
    /// Operation succeeded.
    Success,
    /// A call to a CUDA function failed.
    UnhandledCudaError,
    /// A call to the system failed.
    SystemError,
    /// An internal check failed.
    /// Usually caused by an NCCL bug or memory corruption.
    InternalError,
    /// An argument has an invalid value.
    InvalidArgument,
    /// The call to NCCL is incorrect.
    /// This usually reflects a programming error.
    InvalidUsage,
    /// A call failed possibly due to a network error or a remote process exiting prematurely.
    RemoteError,
    /// An NCCL operation on the communicator is being enqueued and progressed in the background.
    ///
    /// Whenever an operation returns an error (neither [`Status::Success`] nor [`Status::InProgress`]),
    /// NCCL may print a more detailed message when the `NCCL_DEBUG` environment variable is set to `WARN`.
    InProgress,
    /// NCCL returned a status code not known to this wrapper.
    Unknown(u32),
}

impl Status {
    pub const fn raw(self) -> u32 {
        match self {
            Self::Success => sys::ncclResult_t::ncclSuccess as _,
            Self::UnhandledCudaError => sys::ncclResult_t::ncclUnhandledCudaError as _,
            Self::SystemError => sys::ncclResult_t::ncclSystemError as _,
            Self::InternalError => sys::ncclResult_t::ncclInternalError as _,
            Self::InvalidArgument => sys::ncclResult_t::ncclInvalidArgument as _,
            Self::InvalidUsage => sys::ncclResult_t::ncclInvalidUsage as _,
            Self::RemoteError => sys::ncclResult_t::ncclRemoteError as _,
            Self::InProgress => sys::ncclResult_t::ncclInProgress as _,
            Self::Unknown(code) => code,
        }
    }

    pub fn description(self) -> String {
        let Ok(raw) = sys::ncclResult_t::try_from(self) else {
            return String::from("unknown nccl error");
        };

        unsafe {
            let ptr = sys::ncclGetErrorString(raw);
            if ptr.is_null() {
                String::from("unknown nccl error")
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        }
    }
}

impl From<sys::ncclResult_t> for Status {
    fn from(status: sys::ncclResult_t) -> Self {
        Self::from(status as u32)
    }
}

impl From<u32> for Status {
    fn from(code: u32) -> Self {
        match code {
            x if x == sys::ncclResult_t::ncclSuccess as u32 => Self::Success,
            x if x == sys::ncclResult_t::ncclUnhandledCudaError as u32 => Self::UnhandledCudaError,
            x if x == sys::ncclResult_t::ncclSystemError as u32 => Self::SystemError,
            x if x == sys::ncclResult_t::ncclInternalError as u32 => Self::InternalError,
            x if x == sys::ncclResult_t::ncclInvalidArgument as u32 => Self::InvalidArgument,
            x if x == sys::ncclResult_t::ncclInvalidUsage as u32 => Self::InvalidUsage,
            x if x == sys::ncclResult_t::ncclRemoteError as u32 => Self::RemoteError,
            x if x == sys::ncclResult_t::ncclInProgress as u32 => Self::InProgress,
            code => Self::Unknown(code),
        }
    }
}

impl TryFrom<Status> for sys::ncclResult_t {
    type Error = Status;

    fn try_from(status: Status) -> Result<Self, Self::Error> {
        match status {
            Status::Success => Ok(Self::ncclSuccess),
            Status::UnhandledCudaError => Ok(Self::ncclUnhandledCudaError),
            Status::SystemError => Ok(Self::ncclSystemError),
            Status::InternalError => Ok(Self::ncclInternalError),
            Status::InvalidArgument => Ok(Self::ncclInvalidArgument),
            Status::InvalidUsage => Ok(Self::ncclInvalidUsage),
            Status::RemoteError => Ok(Self::ncclRemoteError),
            Status::InProgress => Ok(Self::ncclInProgress),
            Status::Unknown(_) => Err(status),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CommunicatorState {
    Active,
    Finalizing,
    Revoked,
    Aborted,
    Destroyed,
}

impl CommunicatorState {
    pub(crate) const fn into_raw(self) -> u8 {
        match self {
            Self::Active => 0,
            Self::Finalizing => 1,
            Self::Revoked => 2,
            Self::Aborted => 3,
            Self::Destroyed => 4,
        }
    }

    pub(crate) const fn from_raw(value: u8) -> Self {
        match value {
            0 => Self::Active,
            1 => Self::Finalizing,
            2 => Self::Revoked,
            3 => Self::Aborted,
            4 => Self::Destroyed,
            _ => Self::Destroyed,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Active => "active",
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SplitColor {
    Undefined,
    Value(i32),
}

impl SplitColor {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum CtaPolicy {
    Default = sys::NCCL_CTA_POLICY_DEFAULT,
    Efficiency = sys::NCCL_CTA_POLICY_EFFICIENCY,
    Zero = sys::NCCL_CTA_POLICY_ZERO,
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ShrinkFlags: u32 {
        const DEFAULT = sys::NCCL_SHRINK_DEFAULT;
        const ABORT = sys::NCCL_SHRINK_ABORT;
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct WindowFlags: u32 {
        const DEFAULT = sys::NCCL_WIN_DEFAULT;
        const COLLECTIVE_SYMMETRIC = sys::NCCL_WIN_COLL_SYMMETRIC;
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
    pub const fn byte_len(self) -> usize {
        match self {
            Self::I8 | Self::U8 | Self::F8E4M3 | Self::F8E5M2 => 1,
            Self::F16 | Self::Bf16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
        }
    }
}

pub trait DataTypeLike: Clone + Copy + fmt::Debug + 'static {
    fn nccl_data_type() -> DataType;
}

macro_rules! impl_nccl_data_type {
    ($ty:ty, $data_type:ident) => {
        impl DataTypeLike for $ty {
            fn nccl_data_type() -> DataType {
                DataType::$data_type
            }
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

#[derive(Debug, Clone, Copy, Default)]
pub struct Config<'a> {
    pub blocking: Option<bool>,
    pub cga_cluster_size: Option<i32>,
    pub min_ctas: Option<i32>,
    pub max_ctas: Option<i32>,
    pub net_name: Option<&'a CStr>,
    pub split_share: Option<i32>,
    pub traffic_class: Option<i32>,
    pub comm_name: Option<&'a CStr>,
    pub collnet_enable: Option<bool>,
    pub cta_policy: Option<CtaPolicy>,
    pub shrink_share: Option<i32>,
    pub nvls_ctas: Option<i32>,
    pub channel_count_per_net_peer: Option<i32>,
    pub nvlink_centric_sched: Option<i32>,
}

impl Config<'_> {
    pub fn into_raw(self) -> sys::ncclConfig_t {
        let mut raw: sys::ncclConfig_t = unsafe { mem::zeroed() };
        raw.size = size_of::<sys::ncclConfig_t>() as _;
        raw.magic = 0xcafebeef;
        raw.version = sys::NCCL_VERSION_CODE;
        raw.blocking = self.blocking.map_or(CONFIG_UNDEF_INT, i32::from);
        raw.cgaClusterSize = self.cga_cluster_size.unwrap_or(CONFIG_UNDEF_INT);
        raw.minCTAs = self.min_ctas.unwrap_or(CONFIG_UNDEF_INT);
        raw.maxCTAs = self.max_ctas.unwrap_or(CONFIG_UNDEF_INT);
        raw.netName = self.net_name.map_or(std::ptr::null(), CStr::as_ptr);
        raw.splitShare = self.split_share.unwrap_or(CONFIG_UNDEF_INT);
        raw.trafficClass = self.traffic_class.unwrap_or(CONFIG_UNDEF_INT);
        raw.commName = self.comm_name.map_or(std::ptr::null(), CStr::as_ptr);
        raw.collnetEnable = self.collnet_enable.map_or(CONFIG_UNDEF_INT, i32::from);
        raw.CTAPolicy = self
            .cta_policy
            .map_or(CONFIG_UNDEF_INT, |policy| policy as i32);
        raw.shrinkShare = self.shrink_share.unwrap_or(CONFIG_UNDEF_INT);
        raw.nvlsCTAs = self.nvls_ctas.unwrap_or(CONFIG_UNDEF_INT);
        raw.nChannelsPerNetPeer = self.channel_count_per_net_peer.unwrap_or(CONFIG_UNDEF_INT);
        raw.nvlinkCentricSched = self.nvlink_centric_sched.unwrap_or(CONFIG_UNDEF_INT);
        raw
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimulationInfo {
    pub estimated_time: f32,
}

impl SimulationInfo {
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

impl Display for Status {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Success => f.write_str("ncclSuccess"),
            Self::UnhandledCudaError => f.write_str("ncclUnhandledCudaError"),
            Self::SystemError => f.write_str("ncclSystemError"),
            Self::InternalError => f.write_str("ncclInternalError"),
            Self::InvalidArgument => f.write_str("ncclInvalidArgument"),
            Self::InvalidUsage => f.write_str("ncclInvalidUsage"),
            Self::RemoteError => f.write_str("ncclRemoteError"),
            Self::InProgress => f.write_str("ncclInProgress"),
            Self::Unknown(code) => write!(f, "unknown nccl status ({code})"),
        }
    }
}

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

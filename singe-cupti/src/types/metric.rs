use std::fmt::{self, Display, Formatter};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::impl_enum_display;
use singe_cupti_sys as sys;

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// Storage kind of a CUPTI metric value.
    pub enum MetricValueKind: sys::CUpti_MetricValueKind {
        /// The metric value is a 64-bit double.
        Double = CUPTI_METRIC_VALUE_KIND_DOUBLE,
        /// The metric value is a 64-bit unsigned integer.
        Uint64 = CUPTI_METRIC_VALUE_KIND_UINT64,
        /// The metric value is a percentage represented by a 64-bit double.
        ///
        /// For example, 57.5% is represented by the value 57.5.
        Percent = CUPTI_METRIC_VALUE_KIND_PERCENT,
        /// The metric value is a throughput represented by a 64-bit integer.
        ///
        /// The unit for throughput values is bytes/second.
        Throughput = CUPTI_METRIC_VALUE_KIND_THROUGHPUT,
        /// The metric value is a 64-bit signed integer.
        Int64 = CUPTI_METRIC_VALUE_KIND_INT64,
        /// The metric value is a [`MetricValueUtilizationLevel`].
        UtilizationLevel = CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL,
        /// Metric value encoded as NVTX Extended Payload.
        NvtxExtendedPayload = CUPTI_METRIC_VALUE_KIND_NVTX_EXTENDED_PAYLOAD,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_METRIC_VALUE_KIND_FORCE_INT,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// Utilization level used by utilization-valued metrics.
    pub enum MetricValueUtilizationLevel: sys::CUpti_MetricValueUtilizationLevel {
        /// Idle utilization level.
        Idle = CUPTI_METRIC_VALUE_UTILIZATION_IDLE,
        /// Low utilization level.
        Low = CUPTI_METRIC_VALUE_UTILIZATION_LOW,
        /// Mid utilization level.
        Mid = CUPTI_METRIC_VALUE_UTILIZATION_MID,
        /// High utilization level.
        High = CUPTI_METRIC_VALUE_UTILIZATION_HIGH,
        /// Max utilization level.
        Max = CUPTI_METRIC_VALUE_UTILIZATION_MAX,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_METRIC_VALUE_UTILIZATION_FORCE_INT,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
/// Decoded CUPTI metric value.
pub enum MetricValue {
    /// Metric value encoded as Double.
    Double(f64),
    /// Metric value encoded as Uint64.
    Uint64(u64),
    /// Metric value encoded as Percent.
    Percent(f64),
    /// Metric value encoded as Throughput.
    Throughput(u64),
    /// Metric value encoded as Int64.
    Int64(i64),
    /// Metric value encoded as Utilization Level.
    UtilizationLevel(MetricValueUtilizationLevel),
    /// Metric value encoded as NVTX Extended Payload.
    NvtxExtendedPayload(u64),
}

impl_enum_display!(MetricValueKind, {
    Self::Double => "CUPTI_METRIC_VALUE_KIND_DOUBLE",
    Self::Uint64 => "CUPTI_METRIC_VALUE_KIND_UINT64",
    Self::Percent => "CUPTI_METRIC_VALUE_KIND_PERCENT",
    Self::Throughput => "CUPTI_METRIC_VALUE_KIND_THROUGHPUT",
    Self::Int64 => "CUPTI_METRIC_VALUE_KIND_INT64",
    Self::UtilizationLevel => "CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL",
    Self::NvtxExtendedPayload => "CUPTI_METRIC_VALUE_KIND_NVTX_EXTENDED_PAYLOAD",
    Self::ForceInt => "CUPTI_METRIC_VALUE_KIND_FORCE_INT",
});

impl_enum_display!(MetricValueUtilizationLevel, {
    Self::Idle => "CUPTI_METRIC_VALUE_UTILIZATION_IDLE",
    Self::Low => "CUPTI_METRIC_VALUE_UTILIZATION_LOW",
    Self::Mid => "CUPTI_METRIC_VALUE_UTILIZATION_MID",
    Self::High => "CUPTI_METRIC_VALUE_UTILIZATION_HIGH",
    Self::Max => "CUPTI_METRIC_VALUE_UTILIZATION_MAX",
    Self::ForceInt => "CUPTI_METRIC_VALUE_UTILIZATION_FORCE_INT",
});

impl Display for MetricValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Double(value) => write!(f, "Double({value})"),
            Self::Uint64(value) => write!(f, "Uint64({value})"),
            Self::Percent(value) => write!(f, "Percent({value})"),
            Self::Throughput(value) => write!(f, "Throughput({value})"),
            Self::Int64(value) => write!(f, "Int64({value})"),
            Self::UtilizationLevel(value) => write!(f, "UtilizationLevel({value})"),
            Self::NvtxExtendedPayload(value) => {
                write!(f, "NvtxExtendedPayload({value})")
            }
        }
    }
}

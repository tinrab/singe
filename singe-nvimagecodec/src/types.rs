use std::fmt::{self, Display, Formatter};

use bitflags::bitflags;

use num_enum::{IntoPrimitive, TryFromPrimitive};

use singe_core::{impl_enum_conversion, impl_enum_display};

use singe_nvimagecodec_sys as sys;

/// Where a backend runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum BackendKind {
    /// On the CPU only.
    CpuOnly = sys::nvimgcodecBackendKind_t::NVIMGCODEC_BACKEND_KIND_CPU_ONLY as _,
    /// On the GPU only.
    GpuOnly = sys::nvimgcodecBackendKind_t::NVIMGCODEC_BACKEND_KIND_GPU_ONLY as _,
    /// On both the CPU and the GPU.
    HybridCpuGpu = sys::nvimgcodecBackendKind_t::NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU as _,
    /// On a dedicated hardware engine in the GPU.
    HardwareGpuOnly = sys::nvimgcodecBackendKind_t::NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY as _,
}

impl_enum_conversion!(sys::nvimgcodecBackendKind_t, BackendKind);

impl_enum_display!(BackendKind, {
    Self::CpuOnly => "NVIMGCODEC_BACKEND_KIND_CPU_ONLY",
    Self::GpuOnly => "NVIMGCODEC_BACKEND_KIND_GPU_ONLY",
    Self::HybridCpuGpu => "NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU",
    Self::HardwareGpuOnly => "NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY",
});

/// Size of the chroma planes relative to the luma plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(i32)]
#[non_exhaustive]
pub enum ChromaSubsampling {
    /// No subsampling; chroma and luma have the same size (4:4:4).
    None = sys::nvimgcodecChromaSubsampling_t::NVIMGCODEC_SAMPLING_NONE as _,
    /// Chroma has half the luma width and the same height.
    Subsampling422 = sys::nvimgcodecChromaSubsampling_t::NVIMGCODEC_SAMPLING_422 as _,
    /// Chroma has half the luma width and half its height.
    Subsampling420 = sys::nvimgcodecChromaSubsampling_t::NVIMGCODEC_SAMPLING_420 as _,
    /// Chroma has the luma width and half its height.
    Subsampling440 = sys::nvimgcodecChromaSubsampling_t::NVIMGCODEC_SAMPLING_440 as _,
    /// Chroma has a quarter of the luma width and the same height.
    Subsampling411 = sys::nvimgcodecChromaSubsampling_t::NVIMGCODEC_SAMPLING_411 as _,
    /// Chroma has a quarter of the luma width and half its height.
    Subsampling410 = sys::nvimgcodecChromaSubsampling_t::NVIMGCODEC_SAMPLING_410 as _,
    /// Only luma is present.
    Gray = sys::nvimgcodecChromaSubsampling_t::NVIMGCODEC_SAMPLING_GRAY as _,
    /// Vertical variant of 4:1:0 subsampling.
    Subsampling410V = sys::nvimgcodecChromaSubsampling_t::NVIMGCODEC_SAMPLING_410V as _,
    /// Subsampling that nvImageCodec does not support.
    Unsupported = sys::nvimgcodecChromaSubsampling_t::NVIMGCODEC_SAMPLING_UNSUPPORTED as _,
}

impl ChromaSubsampling {
    /// Same as [`None`](Self::None): chroma and luma have the same size.
    pub const SUBSAMPLING_444: Self = Self::None;
}

impl_enum_conversion!(i32, sys::nvimgcodecChromaSubsampling_t, ChromaSubsampling);

impl_enum_display!(ChromaSubsampling, {
    Self::None => "NVIMGCODEC_SAMPLING_NONE",
    Self::Subsampling422 => "NVIMGCODEC_SAMPLING_422",
    Self::Subsampling420 => "NVIMGCODEC_SAMPLING_420",
    Self::Subsampling440 => "NVIMGCODEC_SAMPLING_440",
    Self::Subsampling411 => "NVIMGCODEC_SAMPLING_411",
    Self::Subsampling410 => "NVIMGCODEC_SAMPLING_410",
    Self::Gray => "NVIMGCODEC_SAMPLING_GRAY",
    Self::Subsampling410V => "NVIMGCODEC_SAMPLING_410V",
    Self::Unsupported => "NVIMGCODEC_SAMPLING_UNSUPPORTED",
});

/// How the codec interprets the color channels of an image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(i32)]
#[non_exhaustive]
pub enum ColorSpec {
    /// Unknown or unspecified.
    Unknown = sys::nvimgcodecColorSpec_t::NVIMGCODEC_COLORSPEC_UNKNOWN as _,
    /// sRGB.
    Srgb = sys::nvimgcodecColorSpec_t::NVIMGCODEC_COLORSPEC_SRGB as _,
    /// Grayscale, with a single channel.
    Gray = sys::nvimgcodecColorSpec_t::NVIMGCODEC_COLORSPEC_GRAY as _,
    /// sYCC: YCbCr with sRGB primaries.
    Sycc = sys::nvimgcodecColorSpec_t::NVIMGCODEC_COLORSPEC_SYCC as _,
    /// CMYK: cyan, magenta, yellow, and black.
    Cmyk = sys::nvimgcodecColorSpec_t::NVIMGCODEC_COLORSPEC_CMYK as _,
    /// YCCK: YCbCr plus a black channel.
    Ycck = sys::nvimgcodecColorSpec_t::NVIMGCODEC_COLORSPEC_YCCK as _,
    /// Indices into a color palette.
    Palette = sys::nvimgcodecColorSpec_t::NVIMGCODEC_COLORSPEC_PALETTE as _,
}

impl ColorSpec {
    /// Same as [`Unknown`](Self::Unknown); asks the codec to keep the color space of the encoded image where it can.
    pub const UNCHANGED: Self = Self::Unknown;
}

impl_enum_conversion!(i32, sys::nvimgcodecColorSpec_t, ColorSpec);

impl_enum_display!(ColorSpec, {
    Self::Unknown => "NVIMGCODEC_COLORSPEC_UNKNOWN",
    Self::Srgb => "NVIMGCODEC_COLORSPEC_SRGB",
    Self::Gray => "NVIMGCODEC_COLORSPEC_GRAY",
    Self::Sycc => "NVIMGCODEC_COLORSPEC_SYCC",
    Self::Cmyk => "NVIMGCODEC_COLORSPEC_CMYK",
    Self::Ycck => "NVIMGCODEC_COLORSPEC_YCCK",
    Self::Palette => "NVIMGCODEC_COLORSPEC_PALETTE",
});

/// Kind of memory that holds an image's pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(i32)]
#[non_exhaustive]
pub enum ImageBufferKind {
    /// Unknown.
    Unknown = sys::nvimgcodecImageBufferKind_t::NVIMGCODEC_IMAGE_BUFFER_KIND_UNKNOWN as _,
    /// Device memory, with each plane stored row by row with a stride.
    StridedDevice =
        sys::nvimgcodecImageBufferKind_t::NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE as _,
    /// Host memory, with each plane stored row by row with a stride.
    StridedHost = sys::nvimgcodecImageBufferKind_t::NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST as _,
    /// Memory that nvImageCodec does not support.
    Unsupported = sys::nvimgcodecImageBufferKind_t::NVIMGCODEC_IMAGE_BUFFER_KIND_UNSUPPORTED as _,
}

impl_enum_conversion!(i32, sys::nvimgcodecImageBufferKind_t, ImageBufferKind);

impl_enum_display!(ImageBufferKind, {
    Self::Unknown => "NVIMGCODEC_IMAGE_BUFFER_KIND_UNKNOWN",
    Self::StridedDevice => "NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE",
    Self::StridedHost => "NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST",
    Self::Unsupported => "NVIMGCODEC_IMAGE_BUFFER_KIND_UNSUPPORTED",
});

/// How color components are ordered in channels, and how channels are split into planes.
///
/// Planar formats store each channel in its own plane; interleaved formats store all channels of a pixel together in one plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(i32)]
#[non_exhaustive]
pub enum SampleFormat {
    /// Unknown.
    Unknown = sys::nvimgcodecSampleFormat_t::NVIMGCODEC_SAMPLEFORMAT_UNKNOWN as _,
    /// Planar, with the components of the encoded image unchanged.
    PlanarUnchanged = sys::nvimgcodecSampleFormat_t::NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED as _,
    /// Interleaved, with the components of the encoded image unchanged.
    InterleavedUnchanged = sys::nvimgcodecSampleFormat_t::NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED as _,
    /// Luma only, as one plane of shape (1, height, width).
    PlanarY = sys::nvimgcodecSampleFormat_t::NVIMGCODEC_SAMPLEFORMAT_P_Y as _,
    /// Luma only, as one plane of shape (height, width, 1).
    InterleavedY = sys::nvimgcodecSampleFormat_t::NVIMGCODEC_SAMPLEFORMAT_I_Y as _,
    /// Planar luma and alpha.
    PlanarYa = sys::nvimgcodecSampleFormat_t::NVIMGCODEC_SAMPLEFORMAT_P_YA as _,
    /// Interleaved luma and alpha.
    InterleavedYa = sys::nvimgcodecSampleFormat_t::NVIMGCODEC_SAMPLEFORMAT_I_YA as _,
    /// Planar RGB.
    PlanarRgb = sys::nvimgcodecSampleFormat_t::NVIMGCODEC_SAMPLEFORMAT_P_RGB as _,
    /// Interleaved RGB.
    InterleavedRgb = sys::nvimgcodecSampleFormat_t::NVIMGCODEC_SAMPLEFORMAT_I_RGB as _,
    /// Planar BGR.
    PlanarBgr = sys::nvimgcodecSampleFormat_t::NVIMGCODEC_SAMPLEFORMAT_P_BGR as _,
    /// Interleaved BGR.
    InterleavedBgr = sys::nvimgcodecSampleFormat_t::NVIMGCODEC_SAMPLEFORMAT_I_BGR as _,
    /// Planar YUV.
    PlanarYuv = sys::nvimgcodecSampleFormat_t::NVIMGCODEC_SAMPLEFORMAT_P_YUV as _,
    /// Interleaved YUV.
    InterleavedYuv = sys::nvimgcodecSampleFormat_t::NVIMGCODEC_SAMPLEFORMAT_I_YUV as _,
    /// Planar RGBA.
    PlanarRgba = sys::nvimgcodecSampleFormat_t::NVIMGCODEC_SAMPLEFORMAT_P_RGBA as _,
    /// Interleaved RGBA.
    InterleavedRgba = sys::nvimgcodecSampleFormat_t::NVIMGCODEC_SAMPLEFORMAT_I_RGBA as _,
    /// Planar YCCK.
    PlanarYcck = sys::nvimgcodecSampleFormat_t::NVIMGCODEC_SAMPLEFORMAT_P_YCCK as _,
    /// Interleaved YCCK.
    InterleavedYcck = sys::nvimgcodecSampleFormat_t::NVIMGCODEC_SAMPLEFORMAT_I_YCCK as _,
    /// Planar CMYK.
    PlanarCmyk = sys::nvimgcodecSampleFormat_t::NVIMGCODEC_SAMPLEFORMAT_P_CMYK as _,
    /// Interleaved CMYK.
    InterleavedCmyk = sys::nvimgcodecSampleFormat_t::NVIMGCODEC_SAMPLEFORMAT_I_CMYK as _,
    /// A format that nvImageCodec does not support.
    Unsupported = sys::nvimgcodecSampleFormat_t::NVIMGCODEC_SAMPLEFORMAT_UNSUPPORTED as _,
}

impl SampleFormat {
    /// Same as [`PlanarYuv`](Self::PlanarYuv): planar Y, Cb, and Cr.
    pub const PLANAR_YCC: Self = Self::PlanarYuv;

    /// Same as [`InterleavedYuv`](Self::InterleavedYuv): interleaved Y, Cb, and Cr.
    pub const INTERLEAVED_YCC: Self = Self::InterleavedYuv;
}

impl_enum_conversion!(i32, sys::nvimgcodecSampleFormat_t, SampleFormat);

impl_enum_display!(SampleFormat, {
    Self::Unknown => "NVIMGCODEC_SAMPLEFORMAT_UNKNOWN",
    Self::PlanarUnchanged => "NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED",
    Self::InterleavedUnchanged => "NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED",
    Self::PlanarY => "NVIMGCODEC_SAMPLEFORMAT_P_Y",
    Self::InterleavedY => "NVIMGCODEC_SAMPLEFORMAT_I_Y",
    Self::PlanarYa => "NVIMGCODEC_SAMPLEFORMAT_P_YA",
    Self::InterleavedYa => "NVIMGCODEC_SAMPLEFORMAT_I_YA",
    Self::PlanarRgb => "NVIMGCODEC_SAMPLEFORMAT_P_RGB",
    Self::InterleavedRgb => "NVIMGCODEC_SAMPLEFORMAT_I_RGB",
    Self::PlanarBgr => "NVIMGCODEC_SAMPLEFORMAT_P_BGR",
    Self::InterleavedBgr => "NVIMGCODEC_SAMPLEFORMAT_I_BGR",
    Self::PlanarYuv => "NVIMGCODEC_SAMPLEFORMAT_P_YUV",
    Self::InterleavedYuv => "NVIMGCODEC_SAMPLEFORMAT_I_YUV",
    Self::PlanarRgba => "NVIMGCODEC_SAMPLEFORMAT_P_RGBA",
    Self::InterleavedRgba => "NVIMGCODEC_SAMPLEFORMAT_I_RGBA",
    Self::PlanarYcck => "NVIMGCODEC_SAMPLEFORMAT_P_YCCK",
    Self::InterleavedYcck => "NVIMGCODEC_SAMPLEFORMAT_I_YCCK",
    Self::PlanarCmyk => "NVIMGCODEC_SAMPLEFORMAT_P_CMYK",
    Self::InterleavedCmyk => "NVIMGCODEC_SAMPLEFORMAT_I_CMYK",
    Self::Unsupported => "NVIMGCODEC_SAMPLEFORMAT_UNSUPPORTED",
});

/// Type of each sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(i32)]
#[non_exhaustive]
pub enum SampleDataType {
    /// Unknown.
    Unknown = sys::nvimgcodecSampleDataType_t::NVIMGCODEC_SAMPLE_DATA_TYPE_UNKNOWN as _,
    /// 8-bit signed integer.
    I8 = sys::nvimgcodecSampleDataType_t::NVIMGCODEC_SAMPLE_DATA_TYPE_INT8 as _,
    /// 8-bit unsigned integer.
    U8 = sys::nvimgcodecSampleDataType_t::NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8 as _,
    /// 16-bit signed integer.
    I16 = sys::nvimgcodecSampleDataType_t::NVIMGCODEC_SAMPLE_DATA_TYPE_INT16 as _,
    /// 16-bit unsigned integer.
    U16 = sys::nvimgcodecSampleDataType_t::NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16 as _,
    /// 32-bit signed integer.
    I32 = sys::nvimgcodecSampleDataType_t::NVIMGCODEC_SAMPLE_DATA_TYPE_INT32 as _,
    /// 32-bit unsigned integer.
    U32 = sys::nvimgcodecSampleDataType_t::NVIMGCODEC_SAMPLE_DATA_TYPE_UINT32 as _,
    /// 64-bit signed integer.
    I64 = sys::nvimgcodecSampleDataType_t::NVIMGCODEC_SAMPLE_DATA_TYPE_INT64 as _,
    /// 64-bit unsigned integer.
    U64 = sys::nvimgcodecSampleDataType_t::NVIMGCODEC_SAMPLE_DATA_TYPE_UINT64 as _,
    /// 16-bit floating-point number.
    F16 = sys::nvimgcodecSampleDataType_t::NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT16 as _,
    /// 32-bit floating-point number.
    F32 = sys::nvimgcodecSampleDataType_t::NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32 as _,
    /// 64-bit floating-point number.
    F64 = sys::nvimgcodecSampleDataType_t::NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT64 as _,
    /// A type that nvImageCodec does not support.
    Unsupported = sys::nvimgcodecSampleDataType_t::NVIMGCODEC_SAMPLE_DATA_TYPE_UNSUPPORTED as _,
}

impl SampleDataType {
    /// Returns the size of one sample in bytes, or `None` for [`Unknown`](Self::Unknown) and [`Unsupported`](Self::Unsupported).
    #[must_use]
    pub const fn byte_size(self) -> Option<usize> {
        match self {
            Self::I8 | Self::U8 => Some(1),
            Self::I16 | Self::U16 | Self::F16 => Some(2),
            Self::I32 | Self::U32 | Self::F32 => Some(4),
            Self::I64 | Self::U64 | Self::F64 => Some(8),
            Self::Unknown | Self::Unsupported => None,
        }
    }
}

impl_enum_conversion!(i32, sys::nvimgcodecSampleDataType_t, SampleDataType);

impl_enum_display!(SampleDataType, {
    Self::Unknown => "NVIMGCODEC_SAMPLE_DATA_TYPE_UNKNOWN",
    Self::I8 => "NVIMGCODEC_SAMPLE_DATA_TYPE_INT8",
    Self::U8 => "NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8",
    Self::I16 => "NVIMGCODEC_SAMPLE_DATA_TYPE_INT16",
    Self::U16 => "NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16",
    Self::I32 => "NVIMGCODEC_SAMPLE_DATA_TYPE_INT32",
    Self::U32 => "NVIMGCODEC_SAMPLE_DATA_TYPE_UINT32",
    Self::I64 => "NVIMGCODEC_SAMPLE_DATA_TYPE_INT64",
    Self::U64 => "NVIMGCODEC_SAMPLE_DATA_TYPE_UINT64",
    Self::F16 => "NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT16",
    Self::F32 => "NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32",
    Self::F64 => "NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT64",
    Self::Unsupported => "NVIMGCODEC_SAMPLE_DATA_TYPE_UNSUPPORTED",
});

/// How a backend uses its load hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum LoadHintPolicy {
    /// Ignores the load hint.
    Ignore = sys::nvimgcodecLoadHintPolicy_t::NVIMGCODEC_LOAD_HINT_POLICY_IGNORE as _,
    /// Uses the load hint once to set the backend's share of each batch.
    Fixed = sys::nvimgcodecLoadHintPolicy_t::NVIMGCODEC_LOAD_HINT_POLICY_FIXED as _,
    /// Starts from the load hint and adjusts it on every iteration to reduce idle threads.
    AdaptiveMinimizeIdleTime =
        sys::nvimgcodecLoadHintPolicy_t::NVIMGCODEC_LOAD_HINT_POLICY_ADAPTIVE_MINIMIZE_IDLE_TIME
            as _,
}

impl_enum_conversion!(sys::nvimgcodecLoadHintPolicy_t, LoadHintPolicy);

impl_enum_display!(LoadHintPolicy, {
    Self::Ignore => "NVIMGCODEC_LOAD_HINT_POLICY_IGNORE",
    Self::Fixed => "NVIMGCODEC_LOAD_HINT_POLICY_FIXED",
    Self::AdaptiveMinimizeIdleTime => "NVIMGCODEC_LOAD_HINT_POLICY_ADAPTIVE_MINIMIZE_IDLE_TIME",
});

/// How an encoder interprets the quality value set with [`EncodeParams::set_quality`](crate::encode::EncodeParams::set_quality).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum QualityType {
    /// Each codec picks its own quality settings and ignores the value.
    Default = sys::nvimgcodecQualityType_t::NVIMGCODEC_QUALITY_TYPE_DEFAULT as _,
    /// Lossless encoding, which keeps the original image exactly.
    ///
    /// The value is ignored, except by the CUDA TIFF encoder, where 0 means no compression and 1 means LZW, and the CPU PNG encoder, where it is the compression level from 0 (fastest, largest) to 9 (slowest, smallest).
    Lossless = sys::nvimgcodecQualityType_t::NVIMGCODEC_QUALITY_TYPE_LOSSLESS as _,
    /// JPEG-style quality from 1 (worst) to 100 (best).
    Quality = sys::nvimgcodecQualityType_t::NVIMGCODEC_QUALITY_TYPE_QUALITY as _,
    /// Quantization step, the amount that pixel data is divided by; higher values give lower quality.
    QuantizationStep = sys::nvimgcodecQualityType_t::NVIMGCODEC_QUALITY_TYPE_QUANTIZATION_STEP as _,
    /// Target peak signal-to-noise ratio, a positive value; higher values give higher quality.
    Psnr = sys::nvimgcodecQualityType_t::NVIMGCODEC_QUALITY_TYPE_PSNR as _,
    /// Target size as a fraction of the original, between 0.0 and 1.0 exclusive; 0.1 targets 10% of the original size.
    SizeRatio = sys::nvimgcodecQualityType_t::NVIMGCODEC_QUALITY_TYPE_SIZE_RATIO as _,
}

impl_enum_conversion!(sys::nvimgcodecQualityType_t, QualityType);

impl_enum_display!(QualityType, {
    Self::Default => "NVIMGCODEC_QUALITY_TYPE_DEFAULT",
    Self::Lossless => "NVIMGCODEC_QUALITY_TYPE_LOSSLESS",
    Self::Quality => "NVIMGCODEC_QUALITY_TYPE_QUALITY",
    Self::QuantizationStep => "NVIMGCODEC_QUALITY_TYPE_QUANTIZATION_STEP",
    Self::Psnr => "NVIMGCODEC_QUALITY_TYPE_PSNR",
    Self::SizeRatio => "NVIMGCODEC_QUALITY_TYPE_SIZE_RATIO",
});

bitflags! {
    /// Severities of debug messages that a messenger receives.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct DebugMessageSeverity: u32 {
        /// The most detailed diagnostic messages, for developers.
        const TRACE = sys::nvimgcodecDebugMessageSeverity_t::NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_TRACE as u32;
        /// Diagnostic messages for developers, less detailed than [`TRACE`](Self::TRACE).
        const DEBUG = sys::nvimgcodecDebugMessageSeverity_t::NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_DEBUG as u32;
        /// Informational messages, such as the creation of a resource.
        const INFO = sys::nvimgcodecDebugMessageSeverity_t::NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_INFO as u32;
        /// Behavior that is not necessarily an error but is very likely a bug in the application.
        const WARNING = sys::nvimgcodecDebugMessageSeverity_t::NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_WARNING as u32;
        /// Invalid behavior that can make an operation fail or give a wrong result, such as a file that cannot be opened, without crashing the application.
        const ERROR = sys::nvimgcodecDebugMessageSeverity_t::NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_ERROR as u32;
        /// Invalid behavior that can crash the application or force it to shut down.
        const FATAL = sys::nvimgcodecDebugMessageSeverity_t::NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_FATAL as u32;
        /// Warnings, errors, and fatal errors.
        const DEFAULT = sys::nvimgcodecDebugMessageSeverity_t::NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_DEFAULT as u32;
        /// Every severity, including ones added in later versions.
        const ALL = sys::nvimgcodecDebugMessageSeverity_t::NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_ALL as u32;
    }
}

bitflags! {
    /// Categories of debug messages that a messenger receives.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct DebugMessageCategory: u32 {
        /// Events unrelated to the specification or performance.
        const GENERAL = sys::nvimgcodecDebugMessageCategory_t::NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_GENERAL as u32;
        /// Events that indicate a possible mistake.
        const VALIDATION = sys::nvimgcodecDebugMessageCategory_t::NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_VALIDATION as u32;
        /// Possibly inefficient use of the library.
        const PERFORMANCE = sys::nvimgcodecDebugMessageCategory_t::NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_PERFORMANCE as u32;
        /// Every category, including ones added in later versions.
        const ALL = sys::nvimgcodecDebugMessageCategory_t::NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_ALL as u32;
    }
}

/// Priority of a decoder or encoder within its codec.
///
/// nvImageCodec tries decoders and encoders from highest to lowest priority, falling back to the next when one cannot handle the image; ties go in registration order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Priority {
    /// Highest priority.
    Highest = sys::nvimgcodecPriority_t::NVIMGCODEC_PRIORITY_HIGHEST as _,
    /// Very high priority.
    VeryHigh = sys::nvimgcodecPriority_t::NVIMGCODEC_PRIORITY_VERY_HIGH as _,
    /// High priority.
    High = sys::nvimgcodecPriority_t::NVIMGCODEC_PRIORITY_HIGH as _,
    /// Normal priority.
    Normal = sys::nvimgcodecPriority_t::NVIMGCODEC_PRIORITY_NORMAL as _,
    /// Low priority.
    Low = sys::nvimgcodecPriority_t::NVIMGCODEC_PRIORITY_LOW as _,
    /// Very low priority.
    VeryLow = sys::nvimgcodecPriority_t::NVIMGCODEC_PRIORITY_VERY_LOW as _,
    /// Lowest priority.
    Lowest = sys::nvimgcodecPriority_t::NVIMGCODEC_PRIORITY_LOWEST as _,
}

impl_enum_conversion!(sys::nvimgcodecPriority_t, Priority);

impl_enum_display!(Priority, {
    Self::Highest => "NVIMGCODEC_PRIORITY_HIGHEST",
    Self::VeryHigh => "NVIMGCODEC_PRIORITY_VERY_HIGH",
    Self::High => "NVIMGCODEC_PRIORITY_HIGH",
    Self::Normal => "NVIMGCODEC_PRIORITY_NORMAL",
    Self::Low => "NVIMGCODEC_PRIORITY_LOW",
    Self::VeryLow => "NVIMGCODEC_PRIORITY_VERY_LOW",
    Self::Lowest => "NVIMGCODEC_PRIORITY_LOWEST",
});

/// Order of packets in a JPEG 2000 codestream, as defined by the standard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Jpeg2kProgressionOrder {
    /// Layer, resolution, component, position.
    Lrcp = sys::nvimgcodecJpeg2kProgOrder_t::NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP as _,
    /// Resolution, layer, component, position.
    Rlcp = sys::nvimgcodecJpeg2kProgOrder_t::NVIMGCODEC_JPEG2K_PROG_ORDER_RLCP as _,
    /// Resolution, position, component, layer.
    Rpcl = sys::nvimgcodecJpeg2kProgOrder_t::NVIMGCODEC_JPEG2K_PROG_ORDER_RPCL as _,
    /// Position, component, resolution, layer.
    Pcrl = sys::nvimgcodecJpeg2kProgOrder_t::NVIMGCODEC_JPEG2K_PROG_ORDER_PCRL as _,
    /// Component, position, resolution, layer.
    Cprl = sys::nvimgcodecJpeg2kProgOrder_t::NVIMGCODEC_JPEG2K_PROG_ORDER_CPRL as _,
}

impl_enum_conversion!(sys::nvimgcodecJpeg2kProgOrder_t, Jpeg2kProgressionOrder);

impl_enum_display!(Jpeg2kProgressionOrder, {
    Self::Lrcp => "NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP",
    Self::Rlcp => "NVIMGCODEC_JPEG2K_PROG_ORDER_RLCP",
    Self::Rpcl => "NVIMGCODEC_JPEG2K_PROG_ORDER_RPCL",
    Self::Pcrl => "NVIMGCODEC_JPEG2K_PROG_ORDER_PCRL",
    Self::Cprl => "NVIMGCODEC_JPEG2K_PROG_ORDER_CPRL",
});

/// Container for JPEG 2000 data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Jpeg2kBitstreamType {
    /// A bare JPEG 2000 codestream.
    J2k = sys::nvimgcodecJpeg2kBitstreamType_t::NVIMGCODEC_JPEG2K_STREAM_J2K as _,
    /// A JP2 file.
    Jp2 = sys::nvimgcodecJpeg2kBitstreamType_t::NVIMGCODEC_JPEG2K_STREAM_JP2 as _,
}

impl_enum_conversion!(sys::nvimgcodecJpeg2kBitstreamType_t, Jpeg2kBitstreamType);

impl_enum_display!(Jpeg2kBitstreamType, {
    Self::J2k => "NVIMGCODEC_JPEG2K_STREAM_J2K",
    Self::Jp2 => "NVIMGCODEC_JPEG2K_STREAM_JP2",
});

/// JPEG coding process, as given by the start-of-frame marker.
///
/// The values follow table B.1 of the [JPEG standard](https://www.w3.org/Graphics/JPEG/itu-t81.pdf).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum JpegEncoding {
    /// The JPEG encoding process is unknown.
    Unknown = sys::nvimgcodecJpegEncoding_t::NVIMGCODEC_JPEG_ENCODING_UNKNOWN as _,
    /// Baseline sequential DCT with Huffman entropy coding.
    BaselineDct = sys::nvimgcodecJpegEncoding_t::NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT as _,
    /// Extended sequential DCT with Huffman entropy coding.
    ExtendedSequentialDctHuffman =
        sys::nvimgcodecJpegEncoding_t::NVIMGCODEC_JPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_HUFFMAN
            as _,
    /// Progressive DCT with Huffman entropy coding.
    ProgressiveDctHuffman =
        sys::nvimgcodecJpegEncoding_t::NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN as _,
    /// Lossless predictive JPEG with Huffman entropy coding.
    LosslessHuffman =
        sys::nvimgcodecJpegEncoding_t::NVIMGCODEC_JPEG_ENCODING_LOSSLESS_HUFFMAN as _,
    /// Differential sequential DCT with Huffman entropy coding.
    DifferentialSequentialDctHuffman =
        sys::nvimgcodecJpegEncoding_t::NVIMGCODEC_JPEG_ENCODING_DIFFERENTIAL_SEQUENTIAL_DCT_HUFFMAN
            as _,
    /// Differential progressive DCT with Huffman entropy coding.
    DifferentialProgressiveDctHuffman =
        sys::nvimgcodecJpegEncoding_t::NVIMGCODEC_JPEG_ENCODING_DIFFERENTIAL_PROGRESSIVE_DCT_HUFFMAN
            as _,
    /// Differential lossless JPEG with Huffman entropy coding.
    DifferentialLosslessHuffman =
        sys::nvimgcodecJpegEncoding_t::NVIMGCODEC_JPEG_ENCODING_DIFFERENTIAL_LOSSLESS_HUFFMAN
            as _,
    /// JPEG start-of-frame marker reserved for JPEG extensions.
    ReservedForJpegExtensions =
        sys::nvimgcodecJpegEncoding_t::NVIMGCODEC_JPEG_ENCODING_RESERVED_FOR_JPEG_EXTENSIONS as _,
    /// Extended sequential DCT with arithmetic entropy coding.
    ExtendedSequentialDctArithmetic =
        sys::nvimgcodecJpegEncoding_t::NVIMGCODEC_JPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_ARITHMETIC
            as _,
    /// Progressive DCT with arithmetic entropy coding.
    ProgressiveDctArithmetic =
        sys::nvimgcodecJpegEncoding_t::NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_ARITHMETIC as _,
    /// Lossless predictive JPEG with arithmetic entropy coding.
    LosslessArithmetic =
        sys::nvimgcodecJpegEncoding_t::NVIMGCODEC_JPEG_ENCODING_LOSSLESS_ARITHMETIC as _,
    /// Differential sequential DCT with arithmetic entropy coding.
    DifferentialSequentialDctArithmetic =
        sys::nvimgcodecJpegEncoding_t::NVIMGCODEC_JPEG_ENCODING_DIFFERENTIAL_SEQUENTIAL_DCT_ARITHMETIC
            as _,
    /// Differential progressive DCT with arithmetic entropy coding.
    DifferentialProgressiveDctArithmetic =
        sys::nvimgcodecJpegEncoding_t::NVIMGCODEC_JPEG_ENCODING_DIFFERENTIAL_PROGRESSIVE_DCT_ARITHMETIC
            as _,
    /// Differential lossless JPEG with arithmetic entropy coding.
    DifferentialLosslessArithmetic =
        sys::nvimgcodecJpegEncoding_t::NVIMGCODEC_JPEG_ENCODING_DIFFERENTIAL_LOSSLESS_ARITHMETIC
            as _,
}

impl_enum_conversion!(sys::nvimgcodecJpegEncoding_t, JpegEncoding);

impl_enum_display!(JpegEncoding, {
    Self::Unknown => "NVIMGCODEC_JPEG_ENCODING_UNKNOWN",
    Self::BaselineDct => "NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT",
    Self::ExtendedSequentialDctHuffman => "NVIMGCODEC_JPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_HUFFMAN",
    Self::ProgressiveDctHuffman => "NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN",
    Self::LosslessHuffman => "NVIMGCODEC_JPEG_ENCODING_LOSSLESS_HUFFMAN",
    Self::DifferentialSequentialDctHuffman => "NVIMGCODEC_JPEG_ENCODING_DIFFERENTIAL_SEQUENTIAL_DCT_HUFFMAN",
    Self::DifferentialProgressiveDctHuffman => "NVIMGCODEC_JPEG_ENCODING_DIFFERENTIAL_PROGRESSIVE_DCT_HUFFMAN",
    Self::DifferentialLosslessHuffman => "NVIMGCODEC_JPEG_ENCODING_DIFFERENTIAL_LOSSLESS_HUFFMAN",
    Self::ReservedForJpegExtensions => "NVIMGCODEC_JPEG_ENCODING_RESERVED_FOR_JPEG_EXTENSIONS",
    Self::ExtendedSequentialDctArithmetic => "NVIMGCODEC_JPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_ARITHMETIC",
    Self::ProgressiveDctArithmetic => "NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_ARITHMETIC",
    Self::LosslessArithmetic => "NVIMGCODEC_JPEG_ENCODING_LOSSLESS_ARITHMETIC",
    Self::DifferentialSequentialDctArithmetic => "NVIMGCODEC_JPEG_ENCODING_DIFFERENTIAL_SEQUENTIAL_DCT_ARITHMETIC",
    Self::DifferentialProgressiveDctArithmetic => "NVIMGCODEC_JPEG_ENCODING_DIFFERENTIAL_PROGRESSIVE_DCT_ARITHMETIC",
    Self::DifferentialLosslessArithmetic => "NVIMGCODEC_JPEG_ENCODING_DIFFERENTIAL_LOSSLESS_ARITHMETIC",
});

/// Kind of metadata stored in an image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum MetadataKind {
    /// Unknown.
    Unknown = sys::nvimgcodecMetadataKind_t::NVIMGCODEC_METADATA_KIND_UNKNOWN as _,
    /// A TIFF tag.
    TiffTag = sys::nvimgcodecMetadataKind_t::NVIMGCODEC_METADATA_KIND_TIFF_TAG as _,
    /// An ICC color profile.
    IccProfile = sys::nvimgcodecMetadataKind_t::NVIMGCODEC_METADATA_KIND_ICC_PROFILE as _,
    /// EXIF data; reserved for future use.
    Exif = sys::nvimgcodecMetadataKind_t::NVIMGCODEC_METADATA_KIND_EXIF as _,
    /// Geographic data.
    Geo = sys::nvimgcodecMetadataKind_t::NVIMGCODEC_METADATA_KIND_GEO as _,
    /// Medical imaging data in Aperio format.
    MedicalAperio = sys::nvimgcodecMetadataKind_t::NVIMGCODEC_METADATA_KIND_MED_APERIO as _,
    /// Medical imaging data in Philips format.
    MedicalPhilips = sys::nvimgcodecMetadataKind_t::NVIMGCODEC_METADATA_KIND_MED_PHILIPS as _,
    /// Medical imaging data in Ventana format.
    MedicalVentana = sys::nvimgcodecMetadataKind_t::NVIMGCODEC_METADATA_KIND_MED_VENTANA as _,
    /// Medical imaging data in Leica format.
    MedicalLeica = sys::nvimgcodecMetadataKind_t::NVIMGCODEC_METADATA_KIND_MED_LEICA as _,
    /// Medical imaging data in Trestle format.
    MedicalTrestle = sys::nvimgcodecMetadataKind_t::NVIMGCODEC_METADATA_KIND_MED_TRESTLE as _,
    /// List of the TIFF tag IDs in the image.
    TiffTagList = sys::nvimgcodecMetadataKind_t::NVIMGCODEC_METADATA_KIND_TIFF_TAG_LIST as _,
}

impl_enum_conversion!(sys::nvimgcodecMetadataKind_t, MetadataKind);

impl_enum_display!(MetadataKind, {
    Self::Unknown => "NVIMGCODEC_METADATA_KIND_UNKNOWN",
    Self::TiffTag => "NVIMGCODEC_METADATA_KIND_TIFF_TAG",
    Self::IccProfile => "NVIMGCODEC_METADATA_KIND_ICC_PROFILE",
    Self::Exif => "NVIMGCODEC_METADATA_KIND_EXIF",
    Self::Geo => "NVIMGCODEC_METADATA_KIND_GEO",
    Self::MedicalAperio => "NVIMGCODEC_METADATA_KIND_MED_APERIO",
    Self::MedicalPhilips => "NVIMGCODEC_METADATA_KIND_MED_PHILIPS",
    Self::MedicalVentana => "NVIMGCODEC_METADATA_KIND_MED_VENTANA",
    Self::MedicalLeica => "NVIMGCODEC_METADATA_KIND_MED_LEICA",
    Self::MedicalTrestle => "NVIMGCODEC_METADATA_KIND_MED_TRESTLE",
    Self::TiffTagList => "NVIMGCODEC_METADATA_KIND_TIFF_TAG_LIST",
});

/// Format of metadata bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum MetadataFormat {
    /// Unknown.
    Unknown = sys::nvimgcodecMetadataFormat_t::NVIMGCODEC_METADATA_FORMAT_UNKNOWN as _,
    /// Raw binary data.
    Raw = sys::nvimgcodecMetadataFormat_t::NVIMGCODEC_METADATA_FORMAT_RAW as _,
    /// JSON.
    Json = sys::nvimgcodecMetadataFormat_t::NVIMGCODEC_METADATA_FORMAT_JSON as _,
    /// XML.
    Xml = sys::nvimgcodecMetadataFormat_t::NVIMGCODEC_METADATA_FORMAT_XML as _,
    /// XMP.
    Xmp = sys::nvimgcodecMetadataFormat_t::NVIMGCODEC_METADATA_FORMAT_XMP as _,
}

impl_enum_conversion!(sys::nvimgcodecMetadataFormat_t, MetadataFormat);

impl_enum_display!(MetadataFormat, {
    Self::Unknown => "NVIMGCODEC_METADATA_FORMAT_UNKNOWN",
    Self::Raw => "NVIMGCODEC_METADATA_FORMAT_RAW",
    Self::Json => "NVIMGCODEC_METADATA_FORMAT_JSON",
    Self::Xml => "NVIMGCODEC_METADATA_FORMAT_XML",
    Self::Xmp => "NVIMGCODEC_METADATA_FORMAT_XMP",
});

/// Type of metadata values, as defined for TIFF tags by the TIFF specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum MetadataValueType {
    /// Unknown.
    Unknown = sys::nvimgcodecMetadataValueType_t::NVIMGCODEC_METADATA_VALUE_TYPE_UNKNOWN as _,
    /// 8-bit unsigned integer.
    Byte = sys::nvimgcodecMetadataValueType_t::NVIMGCODEC_METADATA_VALUE_TYPE_BYTE as _,
    /// 7-bit ASCII character in a byte; the last byte is NUL.
    Ascii = sys::nvimgcodecMetadataValueType_t::NVIMGCODEC_METADATA_VALUE_TYPE_ASCII as _,
    /// 16-bit unsigned integer.
    Short = sys::nvimgcodecMetadataValueType_t::NVIMGCODEC_METADATA_VALUE_TYPE_SHORT as _,
    /// 32-bit unsigned integer.
    Long = sys::nvimgcodecMetadataValueType_t::NVIMGCODEC_METADATA_VALUE_TYPE_LONG as _,
    /// Two [`Long`](Self::Long) values: numerator and denominator.
    Rational = sys::nvimgcodecMetadataValueType_t::NVIMGCODEC_METADATA_VALUE_TYPE_RATIONAL as _,
    /// 8-bit signed integer.
    SignedByte = sys::nvimgcodecMetadataValueType_t::NVIMGCODEC_METADATA_VALUE_TYPE_SBYTE as _,
    /// Byte whose meaning depends on the field.
    Undefined = sys::nvimgcodecMetadataValueType_t::NVIMGCODEC_METADATA_VALUE_TYPE_UNDEFINED as _,
    /// 16-bit signed integer.
    SignedShort = sys::nvimgcodecMetadataValueType_t::NVIMGCODEC_METADATA_VALUE_TYPE_SSHORT as _,
    /// 32-bit signed integer.
    SignedLong = sys::nvimgcodecMetadataValueType_t::NVIMGCODEC_METADATA_VALUE_TYPE_SLONG as _,
    /// Two [`SignedLong`](Self::SignedLong) values: numerator and denominator.
    SignedRational =
        sys::nvimgcodecMetadataValueType_t::NVIMGCODEC_METADATA_VALUE_TYPE_SRATIONAL as _,
    /// 32-bit IEEE floating-point number.
    Float = sys::nvimgcodecMetadataValueType_t::NVIMGCODEC_METADATA_VALUE_TYPE_FLOAT as _,
    /// 64-bit IEEE floating-point number.
    Double = sys::nvimgcodecMetadataValueType_t::NVIMGCODEC_METADATA_VALUE_TYPE_DOUBLE as _,
    /// 32-bit unsigned IFD offset.
    Ifd = sys::nvimgcodecMetadataValueType_t::NVIMGCODEC_METADATA_VALUE_TYPE_IFD as _,
    /// 64-bit unsigned integer (`BigTIFF`).
    Long8 = sys::nvimgcodecMetadataValueType_t::NVIMGCODEC_METADATA_VALUE_TYPE_LONG8 as _,
    /// 64-bit signed integer (`BigTIFF`).
    SignedLong8 = sys::nvimgcodecMetadataValueType_t::NVIMGCODEC_METADATA_VALUE_TYPE_SLONG8 as _,
    /// 64-bit unsigned IFD offset (`BigTIFF`).
    Ifd8 = sys::nvimgcodecMetadataValueType_t::NVIMGCODEC_METADATA_VALUE_TYPE_IFD8 as _,
}

impl_enum_conversion!(sys::nvimgcodecMetadataValueType_t, MetadataValueType);

impl_enum_display!(MetadataValueType, {
    Self::Unknown => "NVIMGCODEC_METADATA_VALUE_TYPE_UNKNOWN",
    Self::Byte => "NVIMGCODEC_METADATA_VALUE_TYPE_BYTE",
    Self::Ascii => "NVIMGCODEC_METADATA_VALUE_TYPE_ASCII",
    Self::Short => "NVIMGCODEC_METADATA_VALUE_TYPE_SHORT",
    Self::Long => "NVIMGCODEC_METADATA_VALUE_TYPE_LONG",
    Self::Rational => "NVIMGCODEC_METADATA_VALUE_TYPE_RATIONAL",
    Self::SignedByte => "NVIMGCODEC_METADATA_VALUE_TYPE_SBYTE",
    Self::Undefined => "NVIMGCODEC_METADATA_VALUE_TYPE_UNDEFINED",
    Self::SignedShort => "NVIMGCODEC_METADATA_VALUE_TYPE_SSHORT",
    Self::SignedLong => "NVIMGCODEC_METADATA_VALUE_TYPE_SLONG",
    Self::SignedRational => "NVIMGCODEC_METADATA_VALUE_TYPE_SRATIONAL",
    Self::Float => "NVIMGCODEC_METADATA_VALUE_TYPE_FLOAT",
    Self::Double => "NVIMGCODEC_METADATA_VALUE_TYPE_DOUBLE",
    Self::Ifd => "NVIMGCODEC_METADATA_VALUE_TYPE_IFD",
    Self::Long8 => "NVIMGCODEC_METADATA_VALUE_TYPE_LONG8",
    Self::SignedLong8 => "NVIMGCODEC_METADATA_VALUE_TYPE_SLONG8",
    Self::Ifd8 => "NVIMGCODEC_METADATA_VALUE_TYPE_IFD8",
});

/// How pixels outside the image are filled when a region extends past it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum OutOfBoundsPolicy {
    /// Fills them with the constant values set on the [`Region`](crate::code_stream::Region).
    Constant = sys::nvimgcodecOutOfBoundsPolicy_t::NVIMGCODEC_OUT_OF_BOUNDS_POLICY_CONSTANT as _,
}

impl_enum_conversion!(sys::nvimgcodecOutOfBoundsPolicy_t, OutOfBoundsPolicy);

impl_enum_display!(OutOfBoundsPolicy, {
    Self::Constant => "NVIMGCODEC_OUT_OF_BOUNDS_POLICY_CONSTANT",
});

/// Overall result of processing one image, from the low two bits of a [`ProcessingStatus`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProcessingState {
    /// The result is not known.
    Unknown,
    /// The backend can process the image, possibly after converting its format.
    Supported,
    /// The backend could process the image but has reached its load limit.
    Saturated,
    /// Processing failed.
    Failed,
}

bitflags! {
    /// Reasons a backend reported [`ProcessingState::Supported`] but needs the image or settings changed.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct UnsupportedFeatures: u32 {
        /// The color specification is not supported.
        const COLOR_SPEC_UNSUPPORTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED as u32 & !3;
        /// Applying the orientation was requested but is not supported.
        const ORIENTATION_UNSUPPORTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED as u32 & !3;
        /// Decoding a region of the image is not supported.
        const ROI_UNSUPPORTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED as u32 & !3;
        /// The chroma subsampling is not supported.
        const SAMPLING_UNSUPPORTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED as u32 & !3;
        /// The sample type is not supported.
        const SAMPLE_TYPE_UNSUPPORTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED as u32 & !3;
        /// The sample format is not supported.
        const SAMPLE_FORMAT_UNSUPPORTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED as u32 & !3;
        /// The number of images is not supported.
        const NUM_IMAGES_UNSUPPORTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_NUM_IMAGES_UNSUPPORTED as u32 & !3;
        /// The number of planes is not supported.
        const NUM_PLANES_UNSUPPORTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED as u32 & !3;
        /// The number of channels is not supported.
        const NUM_CHANNELS_UNSUPPORTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED as u32 & !3;
        /// The encode quality type is not supported.
        const QUALITY_TYPE_UNSUPPORTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_QUALITY_TYPE_UNSUPPORTED as u32 & !3;
        /// The encode quality value is not supported.
        const QUALITY_VALUE_UNSUPPORTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_QUALITY_VALUE_UNSUPPORTED as u32 & !3;
        /// The encode tiling is not supported.
        const TILING_UNSUPPORTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_TILING_UNSUPPORTED as u32 & !3;
    }
}

bitflags! {
    /// Reasons for [`ProcessingState::Failed`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ProcessingFailures: u32 {
        /// The compressed data is corrupt.
        const IMAGE_CORRUPTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED as u32 & !3;
        /// The codec is not supported.
        const CODEC_UNSUPPORTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED as u32 & !3;
        /// None of the allowed backends supports the image.
        const BACKEND_UNSUPPORTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_BACKEND_UNSUPPORTED as u32 & !3;
        /// The codec's encoding process is not supported.
        const ENCODING_UNSUPPORTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_ENCODING_UNSUPPORTED as u32 & !3;
        /// The image resolution is not supported.
        const RESOLUTION_UNSUPPORTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_RESOLUTION_UNSUPPORTED as u32 & !3;
        /// A feature of the compressed data is not supported.
        const CODESTREAM_UNSUPPORTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED as u32 & !3;
    }
}

/// Result of processing one image: a [`ProcessingState`] in the low two bits, and reason flags above them.
///
/// The flags mean different things for the supported and failed states, so read them with [`unsupported_features`](Self::unsupported_features) or [`failures`](Self::failures).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProcessingStatus(u32);

impl ProcessingStatus {
    /// The result is not known.
    pub const UNKNOWN: Self =
        Self(sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_UNKNOWN as u32);

    /// Processing succeeded.
    pub const SUCCESS: Self =
        Self(sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_SUCCESS as u32);

    /// The decoder or encoder could process the image but has reached its load limit; see [`Backend::set_load_hint`](crate::codec::Backend::set_load_hint).
    pub const SATURATED: Self =
        Self(sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_SATURATED as u32);

    /// Processing failed for an unknown reason.
    pub const FAIL: Self =
        Self(sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_FAIL as u32);

    /// Failed because the compressed data is corrupt.
    pub const IMAGE_CORRUPTED: Self =
        Self(sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED as u32);

    /// Failed because the codec is not supported.
    pub const CODEC_UNSUPPORTED: Self = Self(
        sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED as u32,
    );

    /// Failed because none of the allowed backends supports the image.
    pub const BACKEND_UNSUPPORTED: Self = Self(
        sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_BACKEND_UNSUPPORTED as u32,
    );

    /// Failed because the codec's encoding process is not supported.
    pub const ENCODING_UNSUPPORTED: Self = Self(
        sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_ENCODING_UNSUPPORTED as u32,
    );

    /// Failed because the image resolution is not supported.
    pub const RESOLUTION_UNSUPPORTED: Self = Self(
        sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_RESOLUTION_UNSUPPORTED as u32,
    );

    /// Failed because a feature of the compressed data is not supported.
    pub const CODESTREAM_UNSUPPORTED: Self = Self(
        sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED as u32,
    );

    /// The backend requires a different color specification.
    pub const COLOR_SPEC_UNSUPPORTED: Self = Self(
        sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED as u32,
    );

    /// The backend cannot apply the requested orientation.
    pub const ORIENTATION_UNSUPPORTED: Self = Self(
        sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED
            as u32,
    );

    /// The backend cannot decode a region of the image.
    pub const ROI_UNSUPPORTED: Self =
        Self(sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED as u32);

    /// The backend requires different chroma subsampling.
    pub const SAMPLING_UNSUPPORTED: Self = Self(
        sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED as u32,
    );

    /// The backend requires a different sample data type.
    pub const SAMPLE_TYPE_UNSUPPORTED: Self = Self(
        sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED
            as u32,
    );

    /// The backend requires a different channel or plane layout.
    pub const SAMPLE_FORMAT_UNSUPPORTED: Self = Self(
        sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED
            as u32,
    );

    /// The backend does not support the requested number of images.
    pub const NUM_IMAGES_UNSUPPORTED: Self = Self(
        sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_NUM_IMAGES_UNSUPPORTED as u32,
    );

    /// The backend does not support the requested number of planes.
    pub const NUM_PLANES_UNSUPPORTED: Self = Self(
        sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED as u32,
    );

    /// The backend does not support the requested number of channels.
    pub const NUM_CHANNELS_UNSUPPORTED: Self = Self(
        sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED
            as u32,
    );

    /// The backend does not support the requested encode quality metric.
    pub const QUALITY_TYPE_UNSUPPORTED: Self = Self(
        sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_QUALITY_TYPE_UNSUPPORTED
            as u32,
    );

    /// The backend does not support the requested encode quality value.
    pub const QUALITY_VALUE_UNSUPPORTED: Self = Self(
        sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_QUALITY_VALUE_UNSUPPORTED
            as u32,
    );

    /// The backend does not support the requested encode tiling.
    pub const TILING_UNSUPPORTED: Self = Self(
        sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_TILING_UNSUPPORTED as u32,
    );

    /// Wraps a native status value, keeping any bits this crate does not know.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Returns the native status value.
    #[must_use]
    pub const fn as_raw(self) -> u32 {
        self.0
    }

    /// Returns whether this is exactly [`SUCCESS`](Self::SUCCESS).
    ///
    /// A supported state with feature flags set is not success.
    #[must_use]
    pub const fn is_success(self) -> bool {
        self.0 == Self::SUCCESS.0
    }

    /// Returns the overall result from the low two bits.
    #[must_use]
    pub const fn state(self) -> ProcessingState {
        match self.0 & 3 {
            0 => ProcessingState::Unknown,
            1 => ProcessingState::Supported,
            2 => ProcessingState::Saturated,
            _ => ProcessingState::Failed,
        }
    }

    /// Returns the feature flags for a [`ProcessingState::Supported`] status, or `None` for other states.
    ///
    /// Flag bits this crate does not know are kept.
    #[must_use]
    pub fn unsupported_features(self) -> Option<UnsupportedFeatures> {
        (self.state() == ProcessingState::Supported)
            .then(|| UnsupportedFeatures::from_bits_retain(self.0 & !3))
    }

    /// Returns the failure flags for a [`ProcessingState::Failed`] status, or `None` for other states.
    ///
    /// Flag bits this crate does not know are kept.
    #[must_use]
    pub fn failures(self) -> Option<ProcessingFailures> {
        (self.state() == ProcessingState::Failed)
            .then(|| ProcessingFailures::from_bits_retain(self.0 & !3))
    }
}

impl fmt::Debug for ProcessingStatus {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl Display for ProcessingStatus {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match *self {
            Self::UNKNOWN => f.write_str("NVIMGCODEC_PROCESSING_STATUS_UNKNOWN"),
            Self::SUCCESS => f.write_str("NVIMGCODEC_PROCESSING_STATUS_SUCCESS"),
            Self::SATURATED => f.write_str("NVIMGCODEC_PROCESSING_STATUS_SATURATED"),
            Self::FAIL => f.write_str("NVIMGCODEC_PROCESSING_STATUS_FAIL"),
            Self::IMAGE_CORRUPTED => f.write_str("NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED"),
            Self::CODEC_UNSUPPORTED => {
                f.write_str("NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED")
            }
            Self::BACKEND_UNSUPPORTED => {
                f.write_str("NVIMGCODEC_PROCESSING_STATUS_BACKEND_UNSUPPORTED")
            }
            Self::ENCODING_UNSUPPORTED => {
                f.write_str("NVIMGCODEC_PROCESSING_STATUS_ENCODING_UNSUPPORTED")
            }
            Self::RESOLUTION_UNSUPPORTED => {
                f.write_str("NVIMGCODEC_PROCESSING_STATUS_RESOLUTION_UNSUPPORTED")
            }
            Self::CODESTREAM_UNSUPPORTED => {
                f.write_str("NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED")
            }
            Self::COLOR_SPEC_UNSUPPORTED => {
                f.write_str("NVIMGCODEC_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED")
            }
            Self::ORIENTATION_UNSUPPORTED => {
                f.write_str("NVIMGCODEC_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED")
            }
            Self::ROI_UNSUPPORTED => f.write_str("NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED"),
            Self::SAMPLING_UNSUPPORTED => {
                f.write_str("NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED")
            }
            Self::SAMPLE_TYPE_UNSUPPORTED => {
                f.write_str("NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED")
            }
            Self::SAMPLE_FORMAT_UNSUPPORTED => {
                f.write_str("NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED")
            }
            Self::NUM_IMAGES_UNSUPPORTED => {
                f.write_str("NVIMGCODEC_PROCESSING_STATUS_NUM_IMAGES_UNSUPPORTED")
            }
            Self::NUM_PLANES_UNSUPPORTED => {
                f.write_str("NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED")
            }
            Self::NUM_CHANNELS_UNSUPPORTED => {
                f.write_str("NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED")
            }
            Self::QUALITY_TYPE_UNSUPPORTED => {
                f.write_str("NVIMGCODEC_PROCESSING_STATUS_QUALITY_TYPE_UNSUPPORTED")
            }
            Self::QUALITY_VALUE_UNSUPPORTED => {
                f.write_str("NVIMGCODEC_PROCESSING_STATUS_QUALITY_VALUE_UNSUPPORTED")
            }
            Self::TILING_UNSUPPORTED => {
                f.write_str("NVIMGCODEC_PROCESSING_STATUS_TILING_UNSUPPORTED")
            }
            value => match value.state() {
                ProcessingState::Supported => write!(
                    f,
                    "NVIMGCODEC_PROCESSING_STATUS_FORMAT_UNSUPPORTED({:?})",
                    value
                        .unsupported_features()
                        .expect("supported-state statuses carry unsupported feature flags")
                ),
                ProcessingState::Failed => write!(
                    f,
                    "NVIMGCODEC_PROCESSING_STATUS_FAIL({:?})",
                    value
                        .failures()
                        .expect("failed-state statuses carry failure flags")
                ),
                _ => write!(f, "UNKNOWN_NVIMGCODEC_PROCESSING_STATUS({:#x})", value.0),
            },
        }
    }
}

impl From<sys::nvimgcodecProcessingStatus_t> for ProcessingStatus {
    fn from(status: sys::nvimgcodecProcessingStatus_t) -> Self {
        Self(status)
    }
}

/// Version information of the loaded nvImageCodec library.
#[derive(Debug, Clone, Copy)]
pub struct Properties {
    raw: sys::nvimgcodecProperties_t,
}

impl Properties {
    pub(crate) const fn from_raw(raw: sys::nvimgcodecProperties_t) -> Self {
        Self { raw }
    }

    /// Returns the library version as `major * 10000 + minor * 100 + patch`.
    #[must_use]
    pub const fn version(&self) -> u32 {
        self.raw.version
    }

    /// Returns the CUDA runtime version nvImageCodec was built with, as `major * 1000 + minor * 10`.
    #[must_use]
    pub const fn cudart_version(&self) -> u32 {
        self.raw.cudart_version
    }
}

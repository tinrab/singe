use std::{
    fmt::{self, Display, Formatter},
    ptr,
};

use bitflags::bitflags;

use num_enum::{IntoPrimitive, TryFromPrimitive};

use singe_core::impl_enum_display;

use singe_nvimagecodec_sys as sys;

singe_core::native_enum! {
    /// Where a backend runs.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum BackendKind: sys::nvimgcodecBackendKind_t {
        /// On the CPU only.
        CpuOnly = NVIMGCODEC_BACKEND_KIND_CPU_ONLY,
        /// On the GPU only.
        GpuOnly = NVIMGCODEC_BACKEND_KIND_GPU_ONLY,
        /// On both the CPU and the GPU.
        HybridCpuGpu = NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU,
        /// On a dedicated hardware engine in the GPU.
        HardwareGpuOnly = NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY,
    }
}

impl_enum_display!(BackendKind, {
    Self::CpuOnly => "NVIMGCODEC_BACKEND_KIND_CPU_ONLY",
    Self::GpuOnly => "NVIMGCODEC_BACKEND_KIND_GPU_ONLY",
    Self::HybridCpuGpu => "NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU",
    Self::HardwareGpuOnly => "NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY",
});

singe_core::native_enum! {
    /// Size of the chroma planes relative to the luma plane.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(i32)]
    #[non_exhaustive]
    pub enum ChromaSubsampling: sys::nvimgcodecChromaSubsampling_t {
        /// No subsampling; chroma and luma have the same size (4:4:4).
        None = NVIMGCODEC_SAMPLING_NONE,
        /// Chroma has half the luma width and the same height.
        Subsampling422 = NVIMGCODEC_SAMPLING_422,
        /// Chroma has half the luma width and half its height.
        Subsampling420 = NVIMGCODEC_SAMPLING_420,
        /// Chroma has the luma width and half its height.
        Subsampling440 = NVIMGCODEC_SAMPLING_440,
        /// Chroma has a quarter of the luma width and the same height.
        Subsampling411 = NVIMGCODEC_SAMPLING_411,
        /// Chroma has a quarter of the luma width and half its height.
        Subsampling410 = NVIMGCODEC_SAMPLING_410,
        /// Only luma is present.
        Gray = NVIMGCODEC_SAMPLING_GRAY,
        /// Vertical variant of 4:1:0 subsampling.
        Subsampling410V = NVIMGCODEC_SAMPLING_410V,
        /// Subsampling that nvImageCodec does not support.
        Unsupported = NVIMGCODEC_SAMPLING_UNSUPPORTED,
    }
}

impl ChromaSubsampling {
    /// Same as [`None`](Self::None): chroma and luma have the same size.
    pub const SUBSAMPLING_444: Self = Self::None;
}

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

singe_core::native_enum! {
    /// How the codec interprets the color channels of an image.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(i32)]
    #[non_exhaustive]
    pub enum ColorSpec: sys::nvimgcodecColorSpec_t {
        /// Unknown or unspecified.
        Unknown = NVIMGCODEC_COLORSPEC_UNKNOWN,
        /// sRGB.
        Srgb = NVIMGCODEC_COLORSPEC_SRGB,
        /// Grayscale, with a single channel.
        Gray = NVIMGCODEC_COLORSPEC_GRAY,
        /// sYCC: YCbCr with sRGB primaries.
        Sycc = NVIMGCODEC_COLORSPEC_SYCC,
        /// CMYK: cyan, magenta, yellow, and black.
        Cmyk = NVIMGCODEC_COLORSPEC_CMYK,
        /// YCCK: YCbCr plus a black channel.
        Ycck = NVIMGCODEC_COLORSPEC_YCCK,
        /// Indices into a color palette.
        Palette = NVIMGCODEC_COLORSPEC_PALETTE,
    }
}

impl ColorSpec {
    /// Same as [`Unknown`](Self::Unknown); asks the codec to keep the color space of the encoded image where it can.
    pub const UNCHANGED: Self = Self::Unknown;
}

impl_enum_display!(ColorSpec, {
    Self::Unknown => "NVIMGCODEC_COLORSPEC_UNKNOWN",
    Self::Srgb => "NVIMGCODEC_COLORSPEC_SRGB",
    Self::Gray => "NVIMGCODEC_COLORSPEC_GRAY",
    Self::Sycc => "NVIMGCODEC_COLORSPEC_SYCC",
    Self::Cmyk => "NVIMGCODEC_COLORSPEC_CMYK",
    Self::Ycck => "NVIMGCODEC_COLORSPEC_YCCK",
    Self::Palette => "NVIMGCODEC_COLORSPEC_PALETTE",
});

singe_core::native_enum! {
    /// Kind of memory that holds an image's pixels.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(i32)]
    #[non_exhaustive]
    pub enum ImageBufferKind: sys::nvimgcodecImageBufferKind_t {
        /// Unknown.
        Unknown = NVIMGCODEC_IMAGE_BUFFER_KIND_UNKNOWN,
        /// Device memory, with each plane stored row by row with a stride.
        StridedDevice = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE,
        /// Host memory, with each plane stored row by row with a stride.
        StridedHost = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST,
        /// Memory that nvImageCodec does not support.
        Unsupported = NVIMGCODEC_IMAGE_BUFFER_KIND_UNSUPPORTED,
    }
}

impl_enum_display!(ImageBufferKind, {
    Self::Unknown => "NVIMGCODEC_IMAGE_BUFFER_KIND_UNKNOWN",
    Self::StridedDevice => "NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE",
    Self::StridedHost => "NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST",
    Self::Unsupported => "NVIMGCODEC_IMAGE_BUFFER_KIND_UNSUPPORTED",
});

singe_core::native_enum! {
    /// How color components are ordered in channels, and how channels are split into planes.
    ///
    /// Planar formats store each channel in its own plane; interleaved formats store all channels of a pixel together in one plane.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(i32)]
    #[non_exhaustive]
    pub enum SampleFormat: sys::nvimgcodecSampleFormat_t {
        /// Unknown.
        Unknown = NVIMGCODEC_SAMPLEFORMAT_UNKNOWN,
        /// Planar, with the components of the encoded image unchanged.
        PlanarUnchanged = NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED,
        /// Interleaved, with the components of the encoded image unchanged.
        InterleavedUnchanged = NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED,
        /// Luma only, as one plane of shape (1, height, width).
        PlanarY = NVIMGCODEC_SAMPLEFORMAT_P_Y,
        /// Luma only, as one plane of shape (height, width, 1).
        InterleavedY = NVIMGCODEC_SAMPLEFORMAT_I_Y,
        /// Planar luma and alpha.
        PlanarYa = NVIMGCODEC_SAMPLEFORMAT_P_YA,
        /// Interleaved luma and alpha.
        InterleavedYa = NVIMGCODEC_SAMPLEFORMAT_I_YA,
        /// Planar RGB.
        PlanarRgb = NVIMGCODEC_SAMPLEFORMAT_P_RGB,
        /// Interleaved RGB.
        InterleavedRgb = NVIMGCODEC_SAMPLEFORMAT_I_RGB,
        /// Planar BGR.
        PlanarBgr = NVIMGCODEC_SAMPLEFORMAT_P_BGR,
        /// Interleaved BGR.
        InterleavedBgr = NVIMGCODEC_SAMPLEFORMAT_I_BGR,
        /// Planar YUV.
        PlanarYuv = NVIMGCODEC_SAMPLEFORMAT_P_YUV,
        /// Interleaved YUV.
        InterleavedYuv = NVIMGCODEC_SAMPLEFORMAT_I_YUV,
        /// Planar RGBA.
        PlanarRgba = NVIMGCODEC_SAMPLEFORMAT_P_RGBA,
        /// Interleaved RGBA.
        InterleavedRgba = NVIMGCODEC_SAMPLEFORMAT_I_RGBA,
        /// Planar YCCK.
        PlanarYcck = NVIMGCODEC_SAMPLEFORMAT_P_YCCK,
        /// Interleaved YCCK.
        InterleavedYcck = NVIMGCODEC_SAMPLEFORMAT_I_YCCK,
        /// Planar CMYK.
        PlanarCmyk = NVIMGCODEC_SAMPLEFORMAT_P_CMYK,
        /// Interleaved CMYK.
        InterleavedCmyk = NVIMGCODEC_SAMPLEFORMAT_I_CMYK,
        /// A format that nvImageCodec does not support.
        Unsupported = NVIMGCODEC_SAMPLEFORMAT_UNSUPPORTED,
    }
}

impl SampleFormat {
    /// Same as [`PlanarYuv`](Self::PlanarYuv): planar Y, Cb, and Cr.
    pub const PLANAR_YCC: Self = Self::PlanarYuv;

    /// Same as [`InterleavedYuv`](Self::InterleavedYuv): interleaved Y, Cb, and Cr.
    pub const INTERLEAVED_YCC: Self = Self::InterleavedYuv;
}

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

singe_core::native_enum! {
    /// Type of each sample.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(i32)]
    #[non_exhaustive]
    pub enum SampleDataType: sys::nvimgcodecSampleDataType_t {
        /// Unknown.
        Unknown = NVIMGCODEC_SAMPLE_DATA_TYPE_UNKNOWN,
        /// 8-bit signed integer.
        I8 = NVIMGCODEC_SAMPLE_DATA_TYPE_INT8,
        /// 8-bit unsigned integer.
        U8 = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8,
        /// 16-bit signed integer.
        I16 = NVIMGCODEC_SAMPLE_DATA_TYPE_INT16,
        /// 16-bit unsigned integer.
        U16 = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16,
        /// 32-bit signed integer.
        I32 = NVIMGCODEC_SAMPLE_DATA_TYPE_INT32,
        /// 32-bit unsigned integer.
        U32 = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT32,
        /// 64-bit signed integer.
        I64 = NVIMGCODEC_SAMPLE_DATA_TYPE_INT64,
        /// 64-bit unsigned integer.
        U64 = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT64,
        /// 16-bit floating-point number.
        F16 = NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT16,
        /// 32-bit floating-point number.
        F32 = NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32,
        /// 64-bit floating-point number.
        F64 = NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT64,
        /// A type that nvImageCodec does not support.
        Unsupported = NVIMGCODEC_SAMPLE_DATA_TYPE_UNSUPPORTED,
    }
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

singe_core::native_enum! {
    /// How a backend uses its load hint.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum LoadHintPolicy: sys::nvimgcodecLoadHintPolicy_t {
        /// Ignores the load hint.
        Ignore = NVIMGCODEC_LOAD_HINT_POLICY_IGNORE,
        /// Uses the load hint once to set the backend's share of each batch.
        Fixed = NVIMGCODEC_LOAD_HINT_POLICY_FIXED,
        /// Starts from the load hint and adjusts it on every iteration to reduce idle threads.
        AdaptiveMinimizeIdleTime = NVIMGCODEC_LOAD_HINT_POLICY_ADAPTIVE_MINIMIZE_IDLE_TIME,
    }
}

impl_enum_display!(LoadHintPolicy, {
    Self::Ignore => "NVIMGCODEC_LOAD_HINT_POLICY_IGNORE",
    Self::Fixed => "NVIMGCODEC_LOAD_HINT_POLICY_FIXED",
    Self::AdaptiveMinimizeIdleTime => "NVIMGCODEC_LOAD_HINT_POLICY_ADAPTIVE_MINIMIZE_IDLE_TIME",
});

singe_core::native_enum! {
    /// How an encoder interprets the quality value set with [`EncodeParams::set_quality`](crate::encode::EncodeParams::set_quality).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum QualityType: sys::nvimgcodecQualityType_t {
        /// Each codec picks its own quality settings and ignores the value.
        Default = NVIMGCODEC_QUALITY_TYPE_DEFAULT,
        /// Lossless encoding, which keeps the original image exactly.
        ///
        /// The value is ignored, except by the CUDA TIFF encoder, where 0 means no compression and 1 means LZW, and the CPU PNG encoder, where it is the compression level from 0 (fastest, largest) to 9 (slowest, smallest).
        Lossless = NVIMGCODEC_QUALITY_TYPE_LOSSLESS,
        /// JPEG-style quality from 1 (worst) to 100 (best).
        Quality = NVIMGCODEC_QUALITY_TYPE_QUALITY,
        /// Quantization step, the amount that pixel data is divided by; higher values give lower quality.
        QuantizationStep = NVIMGCODEC_QUALITY_TYPE_QUANTIZATION_STEP,
        /// Target peak signal-to-noise ratio, a positive value; higher values give higher quality.
        Psnr = NVIMGCODEC_QUALITY_TYPE_PSNR,
        /// Target size as a fraction of the original, between 0.0 and 1.0 exclusive; 0.1 targets 10% of the original size.
        SizeRatio = NVIMGCODEC_QUALITY_TYPE_SIZE_RATIO,
    }
}

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
        const TRACE = sys::nvimgcodecDebugMessageSeverity_t::NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_TRACE.0;
        /// Diagnostic messages for developers, less detailed than [`TRACE`](Self::TRACE).
        const DEBUG = sys::nvimgcodecDebugMessageSeverity_t::NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_DEBUG.0;
        /// Informational messages, such as the creation of a resource.
        const INFO = sys::nvimgcodecDebugMessageSeverity_t::NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_INFO.0;
        /// Behavior that is not necessarily an error but is very likely a bug in the application.
        const WARNING = sys::nvimgcodecDebugMessageSeverity_t::NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_WARNING.0;
        /// Invalid behavior that can make an operation fail or give a wrong result, such as a file that cannot be opened, without crashing the application.
        const ERROR = sys::nvimgcodecDebugMessageSeverity_t::NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_ERROR.0;
        /// Invalid behavior that can crash the application or force it to shut down.
        const FATAL = sys::nvimgcodecDebugMessageSeverity_t::NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_FATAL.0;
        /// Warnings, errors, and fatal errors.
        const DEFAULT = sys::nvimgcodecDebugMessageSeverity_t::NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_DEFAULT.0;
        /// Every severity, including ones added in later versions.
        const ALL = sys::nvimgcodecDebugMessageSeverity_t::NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_ALL.0;
    }
}

bitflags! {
    /// Categories of debug messages that a messenger receives.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct DebugMessageCategory: u32 {
        /// Events unrelated to the specification or performance.
        const GENERAL = sys::nvimgcodecDebugMessageCategory_t::NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_GENERAL.0;
        /// Events that indicate a possible mistake.
        const VALIDATION = sys::nvimgcodecDebugMessageCategory_t::NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_VALIDATION.0;
        /// Possibly inefficient use of the library.
        const PERFORMANCE = sys::nvimgcodecDebugMessageCategory_t::NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_PERFORMANCE.0;
        /// Every category, including ones added in later versions.
        const ALL = sys::nvimgcodecDebugMessageCategory_t::NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_ALL.0;
    }
}

singe_core::native_enum! {
    /// Priority of a decoder or encoder within its codec.
    ///
    /// nvImageCodec tries decoders and encoders from highest to lowest priority, falling back to the next when one cannot handle the image; ties go in registration order.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum Priority: sys::nvimgcodecPriority_t {
        /// Highest priority.
        Highest = NVIMGCODEC_PRIORITY_HIGHEST,
        /// Very high priority.
        VeryHigh = NVIMGCODEC_PRIORITY_VERY_HIGH,
        /// High priority.
        High = NVIMGCODEC_PRIORITY_HIGH,
        /// Normal priority.
        Normal = NVIMGCODEC_PRIORITY_NORMAL,
        /// Low priority.
        Low = NVIMGCODEC_PRIORITY_LOW,
        /// Very low priority.
        VeryLow = NVIMGCODEC_PRIORITY_VERY_LOW,
        /// Lowest priority.
        Lowest = NVIMGCODEC_PRIORITY_LOWEST,
    }
}

impl_enum_display!(Priority, {
    Self::Highest => "NVIMGCODEC_PRIORITY_HIGHEST",
    Self::VeryHigh => "NVIMGCODEC_PRIORITY_VERY_HIGH",
    Self::High => "NVIMGCODEC_PRIORITY_HIGH",
    Self::Normal => "NVIMGCODEC_PRIORITY_NORMAL",
    Self::Low => "NVIMGCODEC_PRIORITY_LOW",
    Self::VeryLow => "NVIMGCODEC_PRIORITY_VERY_LOW",
    Self::Lowest => "NVIMGCODEC_PRIORITY_LOWEST",
});

singe_core::native_enum! {
    /// Order of packets in a JPEG 2000 codestream, as defined by the standard.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum Jpeg2kProgressionOrder: sys::nvimgcodecJpeg2kProgOrder_t {
        /// Layer, resolution, component, position.
        Lrcp = NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP,
        /// Resolution, layer, component, position.
        Rlcp = NVIMGCODEC_JPEG2K_PROG_ORDER_RLCP,
        /// Resolution, position, component, layer.
        Rpcl = NVIMGCODEC_JPEG2K_PROG_ORDER_RPCL,
        /// Position, component, resolution, layer.
        Pcrl = NVIMGCODEC_JPEG2K_PROG_ORDER_PCRL,
        /// Component, position, resolution, layer.
        Cprl = NVIMGCODEC_JPEG2K_PROG_ORDER_CPRL,
    }
}

impl_enum_display!(Jpeg2kProgressionOrder, {
    Self::Lrcp => "NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP",
    Self::Rlcp => "NVIMGCODEC_JPEG2K_PROG_ORDER_RLCP",
    Self::Rpcl => "NVIMGCODEC_JPEG2K_PROG_ORDER_RPCL",
    Self::Pcrl => "NVIMGCODEC_JPEG2K_PROG_ORDER_PCRL",
    Self::Cprl => "NVIMGCODEC_JPEG2K_PROG_ORDER_CPRL",
});

singe_core::native_enum! {
    /// Container for JPEG 2000 data.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum Jpeg2kBitstreamType: sys::nvimgcodecJpeg2kBitstreamType_t {
        /// A bare JPEG 2000 codestream.
        J2k = NVIMGCODEC_JPEG2K_STREAM_J2K,
        /// A JP2 file.
        Jp2 = NVIMGCODEC_JPEG2K_STREAM_JP2,
    }
}

impl_enum_display!(Jpeg2kBitstreamType, {
    Self::J2k => "NVIMGCODEC_JPEG2K_STREAM_J2K",
    Self::Jp2 => "NVIMGCODEC_JPEG2K_STREAM_JP2",
});

singe_core::native_enum! {
    /// JPEG coding process, as given by the start-of-frame marker.
    ///
    /// The values follow table B.1 of the [JPEG standard](https://www.w3.org/Graphics/JPEG/itu-t81.pdf).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum JpegEncoding: sys::nvimgcodecJpegEncoding_t {
        /// The JPEG encoding process is unknown.
        Unknown = NVIMGCODEC_JPEG_ENCODING_UNKNOWN,
        /// Baseline sequential DCT with Huffman entropy coding.
        BaselineDct = NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT,
        /// Extended sequential DCT with Huffman entropy coding.
        ExtendedSequentialDctHuffman = NVIMGCODEC_JPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_HUFFMAN,
        /// Progressive DCT with Huffman entropy coding.
        ProgressiveDctHuffman = NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN,
        /// Lossless predictive JPEG with Huffman entropy coding.
        LosslessHuffman = NVIMGCODEC_JPEG_ENCODING_LOSSLESS_HUFFMAN,
        /// Differential sequential DCT with Huffman entropy coding.
        DifferentialSequentialDctHuffman = NVIMGCODEC_JPEG_ENCODING_DIFFERENTIAL_SEQUENTIAL_DCT_HUFFMAN,
        /// Differential progressive DCT with Huffman entropy coding.
        DifferentialProgressiveDctHuffman = NVIMGCODEC_JPEG_ENCODING_DIFFERENTIAL_PROGRESSIVE_DCT_HUFFMAN,
        /// Differential lossless JPEG with Huffman entropy coding.
        DifferentialLosslessHuffman = NVIMGCODEC_JPEG_ENCODING_DIFFERENTIAL_LOSSLESS_HUFFMAN,
        /// JPEG start-of-frame marker reserved for JPEG extensions.
        ReservedForJpegExtensions = NVIMGCODEC_JPEG_ENCODING_RESERVED_FOR_JPEG_EXTENSIONS,
        /// Extended sequential DCT with arithmetic entropy coding.
        ExtendedSequentialDctArithmetic = NVIMGCODEC_JPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_ARITHMETIC,
        /// Progressive DCT with arithmetic entropy coding.
        ProgressiveDctArithmetic = NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_ARITHMETIC,
        /// Lossless predictive JPEG with arithmetic entropy coding.
        LosslessArithmetic = NVIMGCODEC_JPEG_ENCODING_LOSSLESS_ARITHMETIC,
        /// Differential sequential DCT with arithmetic entropy coding.
        DifferentialSequentialDctArithmetic = NVIMGCODEC_JPEG_ENCODING_DIFFERENTIAL_SEQUENTIAL_DCT_ARITHMETIC,
        /// Differential progressive DCT with arithmetic entropy coding.
        DifferentialProgressiveDctArithmetic = NVIMGCODEC_JPEG_ENCODING_DIFFERENTIAL_PROGRESSIVE_DCT_ARITHMETIC,
        /// Differential lossless JPEG with arithmetic entropy coding.
        DifferentialLosslessArithmetic = NVIMGCODEC_JPEG_ENCODING_DIFFERENTIAL_LOSSLESS_ARITHMETIC,
    }
}

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

singe_core::native_enum! {
    /// Kind of metadata stored in an image.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum MetadataKind: sys::nvimgcodecMetadataKind_t {
        /// Unknown.
        Unknown = NVIMGCODEC_METADATA_KIND_UNKNOWN,
        /// A TIFF tag.
        TiffTag = NVIMGCODEC_METADATA_KIND_TIFF_TAG,
        /// An ICC color profile.
        IccProfile = NVIMGCODEC_METADATA_KIND_ICC_PROFILE,
        /// EXIF data; reserved for future use.
        Exif = NVIMGCODEC_METADATA_KIND_EXIF,
        /// Geographic data.
        Geo = NVIMGCODEC_METADATA_KIND_GEO,
        /// Medical imaging data in Aperio format.
        MedicalAperio = NVIMGCODEC_METADATA_KIND_MED_APERIO,
        /// Medical imaging data in Philips format.
        MedicalPhilips = NVIMGCODEC_METADATA_KIND_MED_PHILIPS,
        /// Medical imaging data in Ventana format.
        MedicalVentana = NVIMGCODEC_METADATA_KIND_MED_VENTANA,
        /// Medical imaging data in Leica format.
        MedicalLeica = NVIMGCODEC_METADATA_KIND_MED_LEICA,
        /// Medical imaging data in Trestle format.
        MedicalTrestle = NVIMGCODEC_METADATA_KIND_MED_TRESTLE,
        /// List of the TIFF tag IDs in the image.
        TiffTagList = NVIMGCODEC_METADATA_KIND_TIFF_TAG_LIST,
    }
}

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

singe_core::native_enum! {
    /// Format of metadata bytes.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum MetadataFormat: sys::nvimgcodecMetadataFormat_t {
        /// Unknown.
        Unknown = NVIMGCODEC_METADATA_FORMAT_UNKNOWN,
        /// Raw binary data.
        Raw = NVIMGCODEC_METADATA_FORMAT_RAW,
        /// JSON.
        Json = NVIMGCODEC_METADATA_FORMAT_JSON,
        /// XML.
        Xml = NVIMGCODEC_METADATA_FORMAT_XML,
        /// XMP.
        Xmp = NVIMGCODEC_METADATA_FORMAT_XMP,
    }
}

impl_enum_display!(MetadataFormat, {
    Self::Unknown => "NVIMGCODEC_METADATA_FORMAT_UNKNOWN",
    Self::Raw => "NVIMGCODEC_METADATA_FORMAT_RAW",
    Self::Json => "NVIMGCODEC_METADATA_FORMAT_JSON",
    Self::Xml => "NVIMGCODEC_METADATA_FORMAT_XML",
    Self::Xmp => "NVIMGCODEC_METADATA_FORMAT_XMP",
});

singe_core::native_enum! {
    /// Type of metadata values, as defined for TIFF tags by the TIFF specification.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum MetadataValueType: sys::nvimgcodecMetadataValueType_t {
        /// Unknown.
        Unknown = NVIMGCODEC_METADATA_VALUE_TYPE_UNKNOWN,
        /// 8-bit unsigned integer.
        Byte = NVIMGCODEC_METADATA_VALUE_TYPE_BYTE,
        /// 7-bit ASCII character in a byte; the last byte is NUL.
        Ascii = NVIMGCODEC_METADATA_VALUE_TYPE_ASCII,
        /// 16-bit unsigned integer.
        Short = NVIMGCODEC_METADATA_VALUE_TYPE_SHORT,
        /// 32-bit unsigned integer.
        Long = NVIMGCODEC_METADATA_VALUE_TYPE_LONG,
        /// Two [`Long`](Self::Long) values: numerator and denominator.
        Rational = NVIMGCODEC_METADATA_VALUE_TYPE_RATIONAL,
        /// 8-bit signed integer.
        SignedByte = NVIMGCODEC_METADATA_VALUE_TYPE_SBYTE,
        /// Byte whose meaning depends on the field.
        Undefined = NVIMGCODEC_METADATA_VALUE_TYPE_UNDEFINED,
        /// 16-bit signed integer.
        SignedShort = NVIMGCODEC_METADATA_VALUE_TYPE_SSHORT,
        /// 32-bit signed integer.
        SignedLong = NVIMGCODEC_METADATA_VALUE_TYPE_SLONG,
        /// Two [`SignedLong`](Self::SignedLong) values: numerator and denominator.
        SignedRational = NVIMGCODEC_METADATA_VALUE_TYPE_SRATIONAL,
        /// 32-bit IEEE floating-point number.
        Float = NVIMGCODEC_METADATA_VALUE_TYPE_FLOAT,
        /// 64-bit IEEE floating-point number.
        Double = NVIMGCODEC_METADATA_VALUE_TYPE_DOUBLE,
        /// 32-bit unsigned IFD offset.
        Ifd = NVIMGCODEC_METADATA_VALUE_TYPE_IFD,
        /// 64-bit unsigned integer (`BigTIFF`).
        Long8 = NVIMGCODEC_METADATA_VALUE_TYPE_LONG8,
        /// 64-bit signed integer (`BigTIFF`).
        SignedLong8 = NVIMGCODEC_METADATA_VALUE_TYPE_SLONG8,
        /// 64-bit unsigned IFD offset (`BigTIFF`).
        Ifd8 = NVIMGCODEC_METADATA_VALUE_TYPE_IFD8,
    }
}

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

singe_core::native_enum! {
    /// How pixels outside the image are filled when a region extends past it.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum OutOfBoundsPolicy: sys::nvimgcodecOutOfBoundsPolicy_t {
        /// Fills them with the constant values set on the [`Region`](crate::code_stream::Region).
        Constant = NVIMGCODEC_OUT_OF_BOUNDS_POLICY_CONSTANT,
    }
}

impl_enum_display!(OutOfBoundsPolicy, {
    Self::Constant => "NVIMGCODEC_OUT_OF_BOUNDS_POLICY_CONSTANT",
});

/// Which codecs a capability check, such as [`Decoder::can_decode`](crate::decode::Decoder::can_decode), examines.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum CodecFallback {
    /// Stops at the preferred codec when it can process the input but not the requested format or parameters, and reports in the status flags what to change to avoid falling back.
    #[default]
    StopAtPreferred,
    /// Falls back to lower-priority codecs when the preferred one does not support the requested format, and checks them too.
    FallBack,
}

impl CodecFallback {
    pub(crate) const fn force_format(self) -> i32 {
        match self {
            Self::StopAtPreferred => 0,
            Self::FallBack => 1,
        }
    }
}

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
        const COLOR_SPEC_UNSUPPORTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED.0 & !3;
        /// Applying the orientation was requested but is not supported.
        const ORIENTATION_UNSUPPORTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED.0 & !3;
        /// Decoding a region of the image is not supported.
        const ROI_UNSUPPORTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED.0 & !3;
        /// The chroma subsampling is not supported.
        const SAMPLING_UNSUPPORTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED.0 & !3;
        /// The sample type is not supported.
        const SAMPLE_TYPE_UNSUPPORTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED.0 & !3;
        /// The sample format is not supported.
        const SAMPLE_FORMAT_UNSUPPORTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED.0 & !3;
        /// The number of images is not supported.
        const NUM_IMAGES_UNSUPPORTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_NUM_IMAGES_UNSUPPORTED.0 & !3;
        /// The number of planes is not supported.
        const NUM_PLANES_UNSUPPORTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED.0 & !3;
        /// The number of channels is not supported.
        const NUM_CHANNELS_UNSUPPORTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED.0 & !3;
        /// The encode quality type is not supported.
        const QUALITY_TYPE_UNSUPPORTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_QUALITY_TYPE_UNSUPPORTED.0 & !3;
        /// The encode quality value is not supported.
        const QUALITY_VALUE_UNSUPPORTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_QUALITY_VALUE_UNSUPPORTED.0 & !3;
        /// The encode tiling is not supported.
        const TILING_UNSUPPORTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_TILING_UNSUPPORTED.0 & !3;
    }
}

bitflags! {
    /// Reasons for [`ProcessingState::Failed`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ProcessingFailures: u32 {
        /// The compressed data is corrupt.
        const IMAGE_CORRUPTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED.0 & !3;
        /// The codec is not supported.
        const CODEC_UNSUPPORTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED.0 & !3;
        /// None of the allowed backends supports the image.
        const BACKEND_UNSUPPORTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_BACKEND_UNSUPPORTED.0 & !3;
        /// The codec's encoding process is not supported.
        const ENCODING_UNSUPPORTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_ENCODING_UNSUPPORTED.0 & !3;
        /// The image resolution is not supported.
        const RESOLUTION_UNSUPPORTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_RESOLUTION_UNSUPPORTED.0 & !3;
        /// A feature of the compressed data is not supported.
        const CODESTREAM_UNSUPPORTED = sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED.0 & !3;
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
        Self(sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_UNKNOWN.0);

    /// Processing succeeded.
    pub const SUCCESS: Self =
        Self(sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_SUCCESS.0);

    /// The decoder or encoder could process the image but has reached its load limit; see [`Backend::set_load_hint`](crate::codec::Backend::set_load_hint).
    pub const SATURATED: Self =
        Self(sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_SATURATED.0);

    /// Processing failed for an unknown reason.
    pub const FAIL: Self =
        Self(sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_FAIL.0);

    /// Failed because the compressed data is corrupt.
    pub const IMAGE_CORRUPTED: Self =
        Self(sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED.0);

    /// Failed because the codec is not supported.
    pub const CODEC_UNSUPPORTED: Self =
        Self(sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED.0);

    /// Failed because none of the allowed backends supports the image.
    pub const BACKEND_UNSUPPORTED: Self =
        Self(sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_BACKEND_UNSUPPORTED.0);

    /// Failed because the codec's encoding process is not supported.
    pub const ENCODING_UNSUPPORTED: Self =
        Self(sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_ENCODING_UNSUPPORTED.0);

    /// Failed because the image resolution is not supported.
    pub const RESOLUTION_UNSUPPORTED: Self = Self(
        sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_RESOLUTION_UNSUPPORTED.0,
    );

    /// Failed because a feature of the compressed data is not supported.
    pub const CODESTREAM_UNSUPPORTED: Self = Self(
        sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED.0,
    );

    /// The backend requires a different color specification.
    pub const COLOR_SPEC_UNSUPPORTED: Self = Self(
        sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED.0,
    );

    /// The backend cannot apply the requested orientation.
    pub const ORIENTATION_UNSUPPORTED: Self = Self(
        sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED.0,
    );

    /// The backend cannot decode a region of the image.
    pub const ROI_UNSUPPORTED: Self =
        Self(sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED.0);

    /// The backend requires different chroma subsampling.
    pub const SAMPLING_UNSUPPORTED: Self =
        Self(sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED.0);

    /// The backend requires a different sample data type.
    pub const SAMPLE_TYPE_UNSUPPORTED: Self = Self(
        sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED.0,
    );

    /// The backend requires a different channel or plane layout.
    pub const SAMPLE_FORMAT_UNSUPPORTED: Self = Self(
        sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED.0,
    );

    /// The backend does not support the requested number of images.
    pub const NUM_IMAGES_UNSUPPORTED: Self = Self(
        sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_NUM_IMAGES_UNSUPPORTED.0,
    );

    /// The backend does not support the requested number of planes.
    pub const NUM_PLANES_UNSUPPORTED: Self = Self(
        sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED.0,
    );

    /// The backend does not support the requested number of channels.
    pub const NUM_CHANNELS_UNSUPPORTED: Self = Self(
        sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED.0,
    );

    /// The backend does not support the requested encode quality metric.
    pub const QUALITY_TYPE_UNSUPPORTED: Self = Self(
        sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_QUALITY_TYPE_UNSUPPORTED.0,
    );

    /// The backend does not support the requested encode quality value.
    pub const QUALITY_VALUE_UNSUPPORTED: Self = Self(
        sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_QUALITY_VALUE_UNSUPPORTED.0,
    );

    /// The backend does not support the requested encode tiling.
    pub const TILING_UNSUPPORTED: Self =
        Self(sys::nvimgcodecProcessingStatus::NVIMGCODEC_PROCESSING_STATUS_TILING_UNSUPPORTED.0);

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

    /// Returns the library version as `major * 1000 + minor * 100 + patch`.
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

/// Native structure with the `struct_type`, `struct_size`, and `struct_next` header that nvImageCodec checks.
pub(crate) trait NativeStruct {
    /// Returns the structure with its header filled in and the other fields at their defaults.
    fn init() -> Self;
}

macro_rules! impl_native_struct {
    ($($ty:ident => $kind:ident $({ $($field:ident: $value:expr),* $(,)? })?;)*) => {
        $(
            impl NativeStruct for sys::$ty {
                fn init() -> Self {
                    Self {
                        struct_type: sys::nvimgcodecStructureType_t::$kind,
                        struct_size: size_of::<Self>() as sys::size_t,
                        struct_next: ptr::null_mut(),
                        $($($field: $value,)*)?
                        ..Default::default()
                    }
                }
            }
        )*
    };
}

impl_native_struct! {
    nvimgcodecProperties_t => NVIMGCODEC_STRUCTURE_TYPE_PROPERTIES;
    nvimgcodecInstanceCreateInfo_t => NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO { load_builtin_modules: 1, load_extension_modules: 1 };
    nvimgcodecExecutionParams_t => NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS { device_id: sys::NVIMGCODEC_DEVICE_CURRENT };
    nvimgcodecDecodeParams_t => NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS;
    nvimgcodecEncodeParams_t => NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS { quality_type: sys::nvimgcodecQualityType_t::NVIMGCODEC_QUALITY_TYPE_DEFAULT };
    nvimgcodecJpegEncodeParams_t => NVIMGCODEC_STRUCTURE_TYPE_JPEG_ENCODE_PARAMS;
    nvimgcodecJpeg2kEncodeParams_t => NVIMGCODEC_STRUCTURE_TYPE_JPEG2K_ENCODE_PARAMS;
    nvimgcodecOrientation_t => NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION;
    nvimgcodecImagePlaneInfo_t => NVIMGCODEC_STRUCTURE_TYPE_IMAGE_PLANE_INFO;
    nvimgcodecImageInfo_t => NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO { orientation: sys::nvimgcodecOrientation_t::init() };
    nvimgcodecJpegImageInfo_t => NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO;
    nvimgcodecRegion_t => NVIMGCODEC_STRUCTURE_TYPE_REGION { out_of_bounds_policy: sys::nvimgcodecOutOfBoundsPolicy_t::NVIMGCODEC_OUT_OF_BOUNDS_POLICY_CONSTANT };
    nvimgcodecCodeStreamView_t => NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_VIEW { region: sys::nvimgcodecRegion_t::init() };
    nvimgcodecCodeStreamInfo_t => NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO;
    nvimgcodecMetadata_t => NVIMGCODEC_STRUCTURE_TYPE_METADATA;
    nvimgcodecDebugMessengerDesc_t => NVIMGCODEC_STRUCTURE_TYPE_DEBUG_MESSENGER_DESC;
}

//! Formats, backends, flags, and other types shared by decoding and encoding.

use std::fmt::{self, Display, Formatter};

use bitflags::parser;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_nvjpeg_sys as sys;

use crate::error::{Error, Result};

bitflags::bitflags! {
    /// Flags for creating a [`Context`](crate::context::Context) with [`Context::create_with_flags`](crate::context::Context::create_with_flags).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct CreateFlags: u32 {
        /// Default behavior.
        const DEFAULT = sys::NVJPEG_FLAGS_DEFAULT;
        /// Stops [`Backend::Hardware`] from buffering extra images in batched decoding, which it does by default for speed.
        ///
        /// Other backends ignore it.
        const HARDWARE_DECODE_NO_PIPELINE = sys::NVJPEG_FLAGS_HW_DECODE_NO_PIPELINE;
        /// Deprecated; ignored since CUDA 11.1.
        const ENABLE_MEMORY_POOLS = sys::NVJPEG_FLAGS_ENABLE_MEMORY_POOLS;
        /// Fails on JPEG data that does not strictly follow the JPEG specification, instead of trying to decode it.
        const BITSTREAM_STRICT = sys::NVJPEG_FLAGS_BITSTREAM_STRICT;
        /// Uses less memory where possible with [`Backend::Hybrid`] and [`Backend::GpuHybrid`].
        const REDUCED_MEMORY_DECODE = sys::NVJPEG_FLAGS_REDUCED_MEMORY_DECODE;
        /// Uses zero-copy memory where the platform supports it.
        const REDUCED_MEMORY_DECODE_ZERO_COPY =
            sys::NVJPEG_FLAGS_REDUCED_MEMORY_DECODE_ZERO_COPY;
        /// Interpolates when upsampling chroma during the YCbCr to RGB conversion.
        const UPSAMPLING_WITH_INTERPOLATION =
            sys::NVJPEG_FLAGS_UPSAMPLING_WITH_INTERPOLATION;
    }
}

impl Display for CreateFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

/// Decoding implementation: on the CPU, the GPU, or the hardware decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Backend {
    /// nvJPEG chooses the backend.
    Default = sys::nvjpegBackend_t::NVJPEG_BACKEND_DEFAULT as _,
    /// Decodes Huffman data on the CPU.
    Hybrid = sys::nvjpegBackend_t::NVJPEG_BACKEND_HYBRID as _,
    /// Decodes Huffman data on the GPU.
    ///
    /// Batched decoding uses the GPU for baseline JPEG with interleaved scans when the batch has more than 50 images; staged decoding uses GPU-assisted Huffman decoding.
    GpuHybrid = sys::nvjpegBackend_t::NVJPEG_BACKEND_GPU_HYBRID as _,
    /// Uses the hardware decoder.
    ///
    /// Supports baseline JPEG with a single scan and one or three channels, except 4:1:0 and 4:1:1 subsampling.
    Hardware = sys::nvjpegBackend_t::NVJPEG_BACKEND_HARDWARE as _,
    /// Reads JPEG data from device memory, for batched decoding of baseline JPEG without restart intervals.
    GpuHybridDevice = sys::nvjpegBackend_t::NVJPEG_BACKEND_GPU_HYBRID_DEVICE as _,
    /// Uses the hardware decoder and reads JPEG data from device memory, for batched decoding only.
    ///
    /// Supports baseline JPEG with a single scan and one or three channels, except 4:1:0 and 4:1:1 subsampling.
    HardwareDevice = sys::nvjpegBackend_t::NVJPEG_BACKEND_HARDWARE_DEVICE as _,
    /// Decodes lossless JPEG as defined in the JPEG-92 standard, with up to two channels and prediction mode 1.
    LosslessJpeg = sys::nvjpegBackend_t::NVJPEG_BACKEND_LOSSLESS_JPEG as _,
}

impl_enum_conversion!(sys::nvjpegBackend_t, Backend);

impl_enum_display!(Backend, {
    Self::Default => "NVJPEG_BACKEND_DEFAULT",
    Self::Hybrid => "NVJPEG_BACKEND_HYBRID",
    Self::GpuHybrid => "NVJPEG_BACKEND_GPU_HYBRID",
    Self::Hardware => "NVJPEG_BACKEND_HARDWARE",
    Self::GpuHybridDevice => "NVJPEG_BACKEND_GPU_HYBRID_DEVICE",
    Self::HardwareDevice => "NVJPEG_BACKEND_HARDWARE_DEVICE",
    Self::LosslessJpeg => "NVJPEG_BACKEND_LOSSLESS_JPEG",
});

/// Encoding implementation: CUDA or the hardware encoder.
///
/// The hardware encoder is used only if it is available, the backend is [`Default`](Self::Default) or [`Hardware`](Self::Hardware), and the image meets all of these:
///
/// - input and output chroma subsampling are both [`ChromaSubsampling::Css420`];
/// - optimized Huffman tables are off;
/// - the encoding is baseline;
/// - width and height are even and between 16 and 16384.
///
/// Otherwise [`Default`](Self::Default) falls back to CUDA, and [`Hardware`](Self::Hardware) fails with [`Status::InvalidParameter`](crate::error::Status::InvalidParameter).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum EncoderBackend {
    /// Uses the hardware encoder when possible and CUDA otherwise.
    Default = sys::nvjpegEncBackend_t::NVJPEG_ENC_BACKEND_DEFAULT as _,
    /// Always uses CUDA.
    Gpu = sys::nvjpegEncBackend_t::NVJPEG_ENC_BACKEND_GPU as _,
    /// Always uses the hardware encoder, without falling back to CUDA.
    Hardware = sys::nvjpegEncBackend_t::NVJPEG_ENC_BACKEND_HARDWARE as _,
}

impl_enum_conversion!(sys::nvjpegEncBackend_t, EncoderBackend);

/// Pixel format of decoded images.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum OutputFormat {
    /// The encoded components, each in its own plane.
    Unchanged = sys::nvjpegOutputFormat_t::NVJPEG_OUTPUT_UNCHANGED as _,
    /// Planar YUV.
    Yuv = sys::nvjpegOutputFormat_t::NVJPEG_OUTPUT_YUV as _,
    /// Luma only.
    Y = sys::nvjpegOutputFormat_t::NVJPEG_OUTPUT_Y as _,
    /// Planar RGB.
    Rgb = sys::nvjpegOutputFormat_t::NVJPEG_OUTPUT_RGB as _,
    /// Planar BGR.
    Bgr = sys::nvjpegOutputFormat_t::NVJPEG_OUTPUT_BGR as _,
    /// Interleaved RGB.
    RgbInterleaved = sys::nvjpegOutputFormat_t::NVJPEG_OUTPUT_RGBI as _,
    /// Interleaved BGR.
    BgrInterleaved = sys::nvjpegOutputFormat_t::NVJPEG_OUTPUT_BGRI as _,
    /// The encoded components, interleaved as 16-bit unsigned integers.
    UnchangedInterleavedU16 = sys::nvjpegOutputFormat_t::NVJPEG_OUTPUT_UNCHANGEDI_U16 as _,
    /// NV12: a luma plane and an interleaved chroma plane.
    ///
    /// Requires 4:2:0 chroma subsampling.
    Nv12 = sys::nvjpegOutputFormat_t::NVJPEG_OUTPUT_NV12 as _,
    /// YUY2: interleaved YUYV.
    ///
    /// Requires 4:2:2 chroma subsampling.
    Yuy2 = sys::nvjpegOutputFormat_t::NVJPEG_OUTPUT_YUY2 as _,
    /// The encoded components, interleaved as 8-bit unsigned integers.
    UnchangedInterleaved = sys::nvjpegOutputFormat_t::NVJPEG_OUTPUT_UNCHANGEDI as _,
}

impl_enum_conversion!(sys::nvjpegOutputFormat_t, OutputFormat);

/// Pixel format of images to encode.
///
/// With [`Nv12`](Self::Nv12), the input subsampling and the output subsampling must both be [`ChromaSubsampling::Css420`], or encoding fails with [`Status::InvalidParameter`](crate::error::Status::InvalidParameter).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum InputFormat {
    /// Planar YUV.
    Yuv = sys::nvjpegInputFormat_t::NVJPEG_INPUT_YUV as _,
    /// Planar RGB.
    Rgb = sys::nvjpegInputFormat_t::NVJPEG_INPUT_RGB as _,
    /// Planar BGR.
    Bgr = sys::nvjpegInputFormat_t::NVJPEG_INPUT_BGR as _,
    /// Interleaved RGB.
    RgbInterleaved = sys::nvjpegInputFormat_t::NVJPEG_INPUT_RGBI as _,
    /// Interleaved BGR.
    BgrInterleaved = sys::nvjpegInputFormat_t::NVJPEG_INPUT_BGRI as _,
    /// NV12: a luma plane and an interleaved chroma plane.
    Nv12 = sys::nvjpegInputFormat_t::NVJPEG_INPUT_NV12 as _,
}

impl_enum_conversion!(sys::nvjpegInputFormat_t, InputFormat);

/// Size of the chroma planes relative to the luma plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(i32)]
#[non_exhaustive]
pub enum ChromaSubsampling {
    /// No chroma subsampling; chroma and luma have the same dimensions.
    Css444 = sys::nvjpegChromaSubsampling_t::NVJPEG_CSS_444 as _,
    /// Chroma has half the luma width and the same height.
    Css422 = sys::nvjpegChromaSubsampling_t::NVJPEG_CSS_422 as _,
    /// Chroma has half the luma width and half its height.
    Css420 = sys::nvjpegChromaSubsampling_t::NVJPEG_CSS_420 as _,
    /// Chroma has the luma width and half its height.
    Css440 = sys::nvjpegChromaSubsampling_t::NVJPEG_CSS_440 as _,
    /// Chroma has one quarter of the luma width and the same height.
    Css411 = sys::nvjpegChromaSubsampling_t::NVJPEG_CSS_411 as _,
    /// Chroma has one quarter of the luma width and half its height.
    Css410 = sys::nvjpegChromaSubsampling_t::NVJPEG_CSS_410 as _,
    /// Only luma is encoded; there are no chroma planes.
    Gray = sys::nvjpegChromaSubsampling_t::NVJPEG_CSS_GRAY as _,
    /// Vertical variant of 4:1:0 chroma subsampling.
    Css410V = sys::nvjpegChromaSubsampling_t::NVJPEG_CSS_410V as _,
    /// Unknown or not supported.
    Unknown = sys::nvjpegChromaSubsampling_t::NVJPEG_CSS_UNKNOWN as _,
}

impl_enum_conversion!(i32, sys::nvjpegChromaSubsampling_t, ChromaSubsampling);

/// JPEG encoding process, as given by the start-of-frame marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum JpegEncoding {
    /// A process nvJPEG does not support.
    Unknown = sys::nvjpegJpegEncoding_t::NVJPEG_ENCODING_UNKNOWN as _,
    /// Baseline DCT (marker 0xC0).
    BaselineDct = sys::nvjpegJpegEncoding_t::NVJPEG_ENCODING_BASELINE_DCT as _,
    /// Extended sequential DCT with Huffman coding (marker 0xC1).
    ExtendedSequentialDctHuffman =
        sys::nvjpegJpegEncoding_t::NVJPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_HUFFMAN as _,
    /// Progressive DCT with Huffman coding (marker 0xC2).
    ProgressiveDctHuffman = sys::nvjpegJpegEncoding_t::NVJPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN as _,
    /// Lossless with Huffman coding (marker 0xC3).
    LosslessHuffman = sys::nvjpegJpegEncoding_t::NVJPEG_ENCODING_LOSSLESS_HUFFMAN as _,
}

impl_enum_conversion!(sys::nvjpegJpegEncoding_t, JpegEncoding);

/// How much the hardware decoder scales the output down.
///
/// Only [`Backend::Hardware`] supports scaling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ScaleFactor {
    /// No scaling.
    None = sys::nvjpegScaleFactor_t::NVJPEG_SCALE_NONE as _,
    /// Half the width and height.
    OneByTwo = sys::nvjpegScaleFactor_t::NVJPEG_SCALE_1_BY_2 as _,
    /// A quarter of the width and height.
    OneByFour = sys::nvjpegScaleFactor_t::NVJPEG_SCALE_1_BY_4 as _,
    /// An eighth of the width and height.
    OneByEight = sys::nvjpegScaleFactor_t::NVJPEG_SCALE_1_BY_8 as _,
}

impl_enum_conversion!(sys::nvjpegScaleFactor_t, ScaleFactor);

/// EXIF orientation of an image, and how decoding applies it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ExifOrientation {
    /// The image has no EXIF orientation.
    Unknown = sys::nvjpegExifOrientation_t::NVJPEG_ORIENTATION_UNKNOWN as _,
    /// The output is not changed.
    Normal = sys::nvjpegExifOrientation_t::NVJPEG_ORIENTATION_NORMAL as _,
    /// The output is flipped horizontally.
    FlipHorizontal = sys::nvjpegExifOrientation_t::NVJPEG_ORIENTATION_FLIP_HORIZONTAL as _,
    /// The output is rotated 180 degrees.
    Rotate180 = sys::nvjpegExifOrientation_t::NVJPEG_ORIENTATION_ROTATE_180 as _,
    /// The output is flipped vertically.
    FlipVertical = sys::nvjpegExifOrientation_t::NVJPEG_ORIENTATION_FLIP_VERTICAL as _,
    /// The output is flipped horizontally, then rotated 90 degrees counterclockwise.
    Transpose = sys::nvjpegExifOrientation_t::NVJPEG_ORIENTATION_TRANSPOSE as _,
    /// The output is rotated 90 degrees counterclockwise.
    Rotate90 = sys::nvjpegExifOrientation_t::NVJPEG_ORIENTATION_ROTATE_90 as _,
    /// The output is flipped horizontally, then rotated 270 degrees counterclockwise.
    Transverse = sys::nvjpegExifOrientation_t::NVJPEG_ORIENTATION_TRANSVERSE as _,
    /// The output is rotated 270 degrees counterclockwise.
    Rotate270 = sys::nvjpegExifOrientation_t::NVJPEG_ORIENTATION_ROTATE_270 as _,
}

impl_enum_conversion!(sys::nvjpegExifOrientation_t, ExifOrientation);

/// Chroma subsampling and component sizes of a JPEG, from its header.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct JpegInfo {
    subsampling: ChromaSubsampling,
    components: Vec<JpegComponentInfo>,
}

/// Size of one JPEG component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JpegComponentInfo {
    width: u32,
    height: u32,
}

impl JpegInfo {
    /// Creates JPEG metadata from raw nvJPEG header fields.
    pub(crate) fn from_raw(
        component_count: usize,
        subsampling: ChromaSubsampling,
        widths: [i32; sys::NVJPEG_MAX_COMPONENT as usize],
        heights: [i32; sys::NVJPEG_MAX_COMPONENT as usize],
    ) -> Result<Self> {
        if component_count == 0 {
            return Err(Error::Empty {
                name: "components".into(),
            });
        }

        if component_count > sys::NVJPEG_MAX_COMPONENT as usize {
            return Err(Error::OutOfRange {
                name: "components".into(),
            });
        }

        let components = (0..component_count)
            .map(|index| {
                if widths[index] <= 0 {
                    return Err(Error::OutOfRange {
                        name: "component width".into(),
                    });
                }

                if heights[index] <= 0 {
                    return Err(Error::OutOfRange {
                        name: "component height".into(),
                    });
                }
                Ok(JpegComponentInfo {
                    width: widths[index] as u32,
                    height: heights[index] as u32,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            subsampling,
            components,
        })
    }

    /// Returns the chroma subsampling.
    #[must_use]
    pub const fn subsampling(&self) -> ChromaSubsampling {
        self.subsampling
    }

    /// Returns the number of components.
    #[must_use]
    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    /// Returns each component's size.
    #[must_use]
    pub fn components(&self) -> &[JpegComponentInfo] {
        &self.components
    }

    /// Returns the first component, if there is one.
    #[must_use]
    pub fn first_component(&self) -> Option<JpegComponentInfo> {
        self.components.first().copied()
    }
}

impl JpegComponentInfo {
    /// Returns the width in pixels.
    #[must_use]
    pub const fn width(self) -> u32 {
        self.width
    }

    /// Returns the height in pixels.
    #[must_use]
    pub const fn height(self) -> u32 {
        self.height
    }
}

/// Number of hardware decoder engines and their cores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HardwareDecoderInfo {
    engines: u32,
    cores_per_engine: u32,
}

/// Number of hardware encoder engines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HardwareEncoderInfo {
    engines: u32,
}

impl HardwareDecoderInfo {
    pub(crate) const fn new(engines: u32, cores_per_engine: u32) -> Self {
        Self {
            engines,
            cores_per_engine,
        }
    }

    /// Returns the number of hardware decoder engines.
    #[must_use]
    pub const fn engines(self) -> u32 {
        self.engines
    }

    /// Returns the number of cores in each engine.
    #[must_use]
    pub const fn cores_per_engine(self) -> u32 {
        self.cores_per_engine
    }
}

impl HardwareEncoderInfo {
    pub(crate) const fn new(engines: u32) -> Self {
        Self { engines }
    }

    /// Returns the number of hardware encoder engines.
    #[must_use]
    pub const fn engines(self) -> u32 {
        self.engines
    }
}

impl_enum_display!(EncoderBackend, {
    EncoderBackend::Default => "NVJPEG_ENC_BACKEND_DEFAULT",
    EncoderBackend::Gpu => "NVJPEG_ENC_BACKEND_GPU",
    EncoderBackend::Hardware => "NVJPEG_ENC_BACKEND_HARDWARE",
});

impl_enum_display!(OutputFormat, {
    OutputFormat::Unchanged => "NVJPEG_OUTPUT_UNCHANGED",
    OutputFormat::Yuv => "NVJPEG_OUTPUT_YUV",
    OutputFormat::Y => "NVJPEG_OUTPUT_Y",
    OutputFormat::Rgb => "NVJPEG_OUTPUT_RGB",
    OutputFormat::Bgr => "NVJPEG_OUTPUT_BGR",
    OutputFormat::RgbInterleaved => "NVJPEG_OUTPUT_RGBI",
    OutputFormat::BgrInterleaved => "NVJPEG_OUTPUT_BGRI",
    OutputFormat::UnchangedInterleavedU16 => "NVJPEG_OUTPUT_UNCHANGEDI_U16",
    OutputFormat::Nv12 => "NVJPEG_OUTPUT_NV12",
    OutputFormat::Yuy2 => "NVJPEG_OUTPUT_YUY2",
    OutputFormat::UnchangedInterleaved => "NVJPEG_OUTPUT_UNCHANGEDI",
});

impl_enum_display!(InputFormat, {
    InputFormat::Yuv => "NVJPEG_INPUT_YUV",
    InputFormat::Rgb => "NVJPEG_INPUT_RGB",
    InputFormat::Bgr => "NVJPEG_INPUT_BGR",
    InputFormat::RgbInterleaved => "NVJPEG_INPUT_RGBI",
    InputFormat::BgrInterleaved => "NVJPEG_INPUT_BGRI",
    InputFormat::Nv12 => "NVJPEG_INPUT_NV12",
});

impl_enum_display!(ChromaSubsampling, {
    ChromaSubsampling::Css444 => "NVJPEG_CSS_444",
    ChromaSubsampling::Css422 => "NVJPEG_CSS_422",
    ChromaSubsampling::Css420 => "NVJPEG_CSS_420",
    ChromaSubsampling::Css440 => "NVJPEG_CSS_440",
    ChromaSubsampling::Css411 => "NVJPEG_CSS_411",
    ChromaSubsampling::Css410 => "NVJPEG_CSS_410",
    ChromaSubsampling::Gray => "NVJPEG_CSS_GRAY",
    ChromaSubsampling::Css410V => "NVJPEG_CSS_410V",
    ChromaSubsampling::Unknown => "NVJPEG_CSS_UNKNOWN",
});

impl_enum_display!(JpegEncoding, {
    JpegEncoding::Unknown => "NVJPEG_ENCODING_UNKNOWN",
    JpegEncoding::BaselineDct => "NVJPEG_ENCODING_BASELINE_DCT",
    JpegEncoding::ExtendedSequentialDctHuffman => "NVJPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_HUFFMAN",
    JpegEncoding::ProgressiveDctHuffman => "NVJPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN",
    JpegEncoding::LosslessHuffman => "NVJPEG_ENCODING_LOSSLESS_HUFFMAN",
});

impl_enum_display!(ScaleFactor, {
    ScaleFactor::None => "NVJPEG_SCALE_NONE",
    ScaleFactor::OneByTwo => "NVJPEG_SCALE_1_BY_2",
    ScaleFactor::OneByFour => "NVJPEG_SCALE_1_BY_4",
    ScaleFactor::OneByEight => "NVJPEG_SCALE_1_BY_8",
});

impl_enum_display!(ExifOrientation, {
    ExifOrientation::Unknown => "NVJPEG_ORIENTATION_UNKNOWN",
    ExifOrientation::Normal => "NVJPEG_ORIENTATION_NORMAL",
    ExifOrientation::FlipHorizontal => "NVJPEG_ORIENTATION_FLIP_HORIZONTAL",
    ExifOrientation::Rotate180 => "NVJPEG_ORIENTATION_ROTATE_180",
    ExifOrientation::FlipVertical => "NVJPEG_ORIENTATION_FLIP_VERTICAL",
    ExifOrientation::Transpose => "NVJPEG_ORIENTATION_TRANSPOSE",
    ExifOrientation::Rotate90 => "NVJPEG_ORIENTATION_ROTATE_90",
    ExifOrientation::Transverse => "NVJPEG_ORIENTATION_TRANSVERSE",
    ExifOrientation::Rotate270 => "NVJPEG_ORIENTATION_ROTATE_270",
});

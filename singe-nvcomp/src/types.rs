use num_enum::{IntoPrimitive, TryFromPrimitive};

use singe_core::{CudaRuntimeVersion, LibraryVersion, impl_enum_conversion, impl_enum_display};

use singe_nvcomp_sys as sys;

use crate::{
    error::{Error, Result},
    utility::to_usize,
};

/// Properties of the loaded nvCOMP library.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Properties {
    /// nvCOMP library version.
    pub version: LibraryVersion,
    /// CUDA runtime version that nvCOMP was built with.
    pub cudart_version: CudaRuntimeVersion,
}

impl From<sys::nvcompProperties_t> for Properties {
    fn from(value: sys::nvcompProperties_t) -> Self {
        let version = u64::from(value.version);
        Self {
            version: LibraryVersion::new(version / 1000, (version % 1000) / 100, version % 100),
            cudart_version: CudaRuntimeVersion::from(value.cudart_version),
        }
    }
}

/// Minimum buffer alignments, in bytes, that a codec requires.
///
/// Buffers aligned beyond these minimums, such as to 16 or 32 bytes, can be faster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AlignmentRequirements {
    /// Minimum alignment of each input buffer.
    pub input: usize,
    /// Minimum alignment of each output buffer.
    pub output: usize,
    /// Minimum alignment of the temporary workspace.
    ///
    /// Codecs that use no temporary storage report 1.
    pub temp: usize,
}

impl TryFrom<sys::nvcompAlignmentRequirements_t> for AlignmentRequirements {
    type Error = Error;

    fn try_from(value: sys::nvcompAlignmentRequirements_t) -> Result<Self> {
        Ok(Self {
            input: to_usize(value.input, "input alignment")?,
            output: to_usize(value.output, "output alignment")?,
            temp: to_usize(value.temp, "temporary alignment")?,
        })
    }
}

impl From<AlignmentRequirements> for sys::nvcompAlignmentRequirements_t {
    fn from(value: AlignmentRequirements) -> Self {
        Self {
            input: value.input as _,
            output: value.output as _,
            temp: value.temp as _,
        }
    }
}

/// Element type that a codec interprets the input as.
///
/// Multi-byte types are little-endian.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum DataType {
    /// 8-bit signed integer.
    I8 = sys::nvcompType_t::NVCOMP_TYPE_CHAR as _,
    /// 8-bit unsigned integer.
    U8 = sys::nvcompType_t::NVCOMP_TYPE_UCHAR as _,
    /// 16-bit signed integer.
    I16 = sys::nvcompType_t::NVCOMP_TYPE_SHORT as _,
    /// 16-bit unsigned integer.
    U16 = sys::nvcompType_t::NVCOMP_TYPE_USHORT as _,
    /// 32-bit signed integer.
    I32 = sys::nvcompType_t::NVCOMP_TYPE_INT as _,
    /// 32-bit unsigned integer.
    U32 = sys::nvcompType_t::NVCOMP_TYPE_UINT as _,
    /// 64-bit signed integer.
    I64 = sys::nvcompType_t::NVCOMP_TYPE_LONGLONG as _,
    /// 64-bit unsigned integer.
    U64 = sys::nvcompType_t::NVCOMP_TYPE_ULONGLONG as _,
    /// 16-bit floating-point number.
    F16 = sys::nvcompType_t::NVCOMP_TYPE_FLOAT16 as _,
    /// IEEE 754 single-precision floating-point number.
    F32 = sys::nvcompType_t::NVCOMP_TYPE_FLOAT32 as _,
    /// IEEE 754 double-precision floating-point number.
    F64 = sys::nvcompType_t::NVCOMP_TYPE_FLOAT64 as _,
    /// 8-bit floating-point number in E4M3 layout: 1 sign bit, 4 exponent bits, and 3 mantissa bits.
    F8E4M3 = sys::nvcompType_t::NVCOMP_TYPE_FLOAT8_E4M3 as _,
    /// Untyped bit stream.
    Bits = sys::nvcompType_t::NVCOMP_TYPE_BITS as _,
}

impl DataType {
    /// Returns the size of one element in bytes, or 1 for [`Bits`](Self::Bits).
    #[must_use]
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::I8 | Self::U8 | Self::F8E4M3 | Self::Bits => 1,
            Self::I16 | Self::U16 | Self::F16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
        }
    }
}

impl_enum_conversion!(sys::nvcompType_t, DataType);

impl_enum_display!(DataType, {
    Self::I8 => "NVCOMP_TYPE_CHAR",
    Self::U8 => "NVCOMP_TYPE_UCHAR",
    Self::I16 => "NVCOMP_TYPE_SHORT",
    Self::U16 => "NVCOMP_TYPE_USHORT",
    Self::I32 => "NVCOMP_TYPE_INT",
    Self::U32 => "NVCOMP_TYPE_UINT",
    Self::I64 => "NVCOMP_TYPE_LONGLONG",
    Self::U64 => "NVCOMP_TYPE_ULONGLONG",
    Self::F16 => "NVCOMP_TYPE_FLOAT16",
    Self::F32 => "NVCOMP_TYPE_FLOAT32",
    Self::F64 => "NVCOMP_TYPE_FLOAT64",
    Self::F8E4M3 => "NVCOMP_TYPE_FLOAT8_E4M3",
    Self::Bits => "NVCOMP_TYPE_BITS",
});

/// Engine that runs decompression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum DecompressBackend {
    /// Lets nvCOMP choose between the hardware engine and a CUDA implementation.
    Default = sys::nvcompDecompressBackend_t::NVCOMP_DECOMPRESS_BACKEND_DEFAULT as _,
    /// Uses the dedicated hardware decompression engine.
    ///
    /// nvCOMP does not check buffer sizes against the engine's limit.
    /// A buffer over the limit makes decompression fail asynchronously.
    /// The limit is currently 4 MB and may change; query it with `Device::attribute` and `DeviceAttribute::MemDecompressMaximumLength` from `singe-cuda`.
    Hardware = sys::nvcompDecompressBackend_t::NVCOMP_DECOMPRESS_BACKEND_HARDWARE as _,
    /// Uses the CUDA implementation.
    Cuda = sys::nvcompDecompressBackend_t::NVCOMP_DECOMPRESS_BACKEND_CUDA as _,
}

impl_enum_conversion!(sys::nvcompDecompressBackend_t, DecompressBackend);

impl_enum_display!(DecompressBackend, {
    Self::Default => "NVCOMP_DECOMPRESS_BACKEND_DEFAULT",
    Self::Hardware => "NVCOMP_DECOMPRESS_BACKEND_HARDWARE",
    Self::Cuda => "NVCOMP_DECOMPRESS_BACKEND_CUDA",
});

/// ANS coding variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum AnsType {
    /// Uses range asymmetric numeral systems (rANS) coding.
    Rans = sys::nvcompANSType_t::nvcomp_rANS as _,
}

impl_enum_conversion!(sys::nvcompANSType_t, AnsType);

impl_enum_display!(AnsType, {
    Self::Rans => "nvcomp_rANS",
});

/// Bitcomp compression algorithm.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(i32)]
#[non_exhaustive]
pub enum BitcompAlgorithm {
    /// Default algorithm, which usually gives the best compression ratio.
    #[default]
    Default = 0,
    /// Algorithm for sparse data with many zeroes, usually faster than [`Default`](Self::Default).
    Sparse = 1,
}

impl_enum_display!(BitcompAlgorithm, {
    Self::Default => "Default",
    Self::Sparse => "Sparse",
});

/// Whether Bitcomp compresses losslessly or quantizes floating-point input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum BitcompMode {
    /// Preserves the input exactly.
    Lossless = sys::bitcompMode_t::BITCOMP_LOSSLESS as _,
    /// Lossy compression of floating-point data quantized to signed integers.
    LossyFloatToSigned = sys::bitcompMode_t::BITCOMP_LOSSY_FP_TO_SIGNED as _,
    /// Lossy compression of floating-point data quantized to unsigned integers.
    LossyFloatToUnsigned = sys::bitcompMode_t::BITCOMP_LOSSY_FP_TO_UNSIGNED as _,
}

impl_enum_conversion!(sys::bitcompMode_t, BitcompMode);

impl_enum_display!(BitcompMode, {
    Self::Lossless => "BITCOMP_LOSSLESS",
    Self::LossyFloatToSigned => "BITCOMP_LOSSY_FP_TO_SIGNED",
    Self::LossyFloatToUnsigned => "BITCOMP_LOSSY_FP_TO_UNSIGNED",
});

/// Speed and compression-ratio tradeoff for Deflate, `GDeflate`, and Gzip compression.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(i32)]
#[non_exhaustive]
pub enum CompressionAlgorithm {
    /// Highest throughput and lowest compression ratio, using entropy coding only.
    ///
    /// Use it when compression and decompression should run at similar speeds.
    HighestThroughput = 0,
    /// High throughput and low compression ratio.
    #[default]
    HighThroughput = 1,
    /// Medium throughput, with a better compression ratio than zlib level 1.
    Medium = 2,
    /// Reserved for a future level; nvCOMP currently treats it as [`Medium`](Self::Medium).
    ReservedMedium = 3,
    /// Lower throughput, with a better compression ratio than zlib level 6.
    HighCompression = 4,
    /// Lowest throughput and highest compression ratio.
    HighestCompression = 5,
}

impl_enum_display!(CompressionAlgorithm, {
    Self::HighestThroughput => "HighestThroughput",
    Self::HighThroughput => "HighThroughput",
    Self::Medium => "Medium",
    Self::ReservedMedium => "ReservedMedium",
    Self::HighCompression => "HighCompression",
    Self::HighestCompression => "HighestCompression",
});

/// Bit-transposition step applied before compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum BitshuffleMode {
    /// Does not shuffle bits.
    None = sys::nvcompBitshuffleMode_t::NVCOMP_BITSHUFFLE_NONE as _,
    /// Shuffles bits, writing the most significant bits first.
    MsbFirst = sys::nvcompBitshuffleMode_t::NVCOMP_BITSHUFFLE_MSB_FIRST as _,
    /// Shuffles bits, writing the least significant bits first.
    LsbFirst = sys::nvcompBitshuffleMode_t::NVCOMP_BITSHUFFLE_LSB_FIRST as _,
}

impl_enum_conversion!(sys::nvcompBitshuffleMode_t, BitshuffleMode);

impl_enum_display!(BitshuffleMode, {
    Self::None => "NVCOMP_BITSHUFFLE_NONE",
    Self::MsbFirst => "NVCOMP_BITSHUFFLE_MSB_FIRST",
    Self::LsbFirst => "NVCOMP_BITSHUFFLE_LSB_FIRST",
});

/// CUDA algorithm for Gzip decompression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum GzipDecompressAlgorithm {
    /// Uses the baseline algorithm.
    Naive =
        sys::nvcompBatchedGzipDecompressAlgorithm_t::NVCOMP_GZIP_DECOMPRESS_ALGORITHM_NAIVE as _,
    /// Uses the lookahead algorithm.
    Lookahead =
        sys::nvcompBatchedGzipDecompressAlgorithm_t::NVCOMP_GZIP_DECOMPRESS_ALGORITHM_LOOKAHEAD
            as _,
}

impl_enum_conversion!(
    sys::nvcompBatchedGzipDecompressAlgorithm_t,
    GzipDecompressAlgorithm
);

impl_enum_display!(GzipDecompressAlgorithm, {
    Self::Naive => "NVCOMP_GZIP_DECOMPRESS_ALGORITHM_NAIVE",
    Self::Lookahead => "NVCOMP_GZIP_DECOMPRESS_ALGORITHM_LOOKAHEAD",
});

/// How the CRC32 kernel divides work among GPU threads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Crc32KernelKind {
    /// Each warp processes its own chunk.
    Warp = sys::nvcompCRC32KernelKind_t::nvcompCRC32WarpKernel as _,
    /// One or more thread blocks process each chunk.
    Block = sys::nvcompCRC32KernelKind_t::nvcompCRC32BlockKernel as _,
}

impl_enum_conversion!(sys::nvcompCRC32KernelKind_t, Crc32KernelKind);

impl_enum_display!(Crc32KernelKind, {
    Self::Warp => "nvcompCRC32WarpKernel",
    Self::Block => "nvcompCRC32BlockKernel",
});

/// Position of a segment within a message when computing CRC32 over several calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Crc32SegmentKind {
    /// The segment is the complete message.
    Only = sys::nvcompCRC32SegmentKind_t::nvcompCRC32OnlySegment as _,
    /// First segment of a message, possibly followed by more.
    First = sys::nvcompCRC32SegmentKind_t::nvcompCRC32FirstSegment as _,
    /// Segment after the first, possibly followed by more.
    Mid = sys::nvcompCRC32SegmentKind_t::nvcompCRC32MidSegment as _,
    /// Last segment of a message.
    ///
    /// Use [`Only`](Self::Only) when the segment is also the first one.
    /// `Last` can also mark an already processed segment as the end of its message; see [`compute_raw_async`](crate::crc32::compute_raw_async).
    Last = sys::nvcompCRC32SegmentKind_t::nvcompCRC32LastSegment as _,
}

impl_enum_conversion!(sys::nvcompCRC32SegmentKind_t, Crc32SegmentKind);

impl_enum_display!(Crc32SegmentKind, {
    Self::Only => "nvcompCRC32OnlySegment",
    Self::First => "nvcompCRC32FirstSegment",
    Self::Mid => "nvcompCRC32MidSegment",
    Self::Last => "nvcompCRC32LastSegment",
});

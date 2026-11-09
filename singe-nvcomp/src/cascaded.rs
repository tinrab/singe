use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_nvcomp_sys as sys;

use crate::{
    codec::impl_nvcomp_batched_codec_module,
    types::{DataType, DecompressBackend},
    utility::validate_reserved,
};

/// Cascaded compression scheme.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Mode {
    /// No mode; valid only for decompression.
    ///
    /// Decompression then accepts data from either mode, at some cost in speed.
    #[default]
    Unspecified = sys::nvcompCascadedMode_t::NVCOMP_CASCADED_MODE_UNSPECIFIED as _,
    /// Adaptive asymmetric compression.
    Asymmetric = sys::nvcompCascadedMode_t::NVCOMP_CASCADED_MODE_ASYMMETRIC as _,
    /// Symmetric compression, with decompression that mirrors it.
    ///
    /// Supports signed and unsigned 32-bit and 64-bit integers.
    Symmetric = sys::nvcompCascadedMode_t::NVCOMP_CASCADED_MODE_SYMMETRIC as _,
}

impl_enum_conversion!(sys::nvcompCascadedMode_t, Mode);

impl_enum_display!(Mode, {
    Self::Unspecified => "NVCOMP_CASCADED_MODE_UNSPECIFIED",
    Self::Asymmetric => "NVCOMP_CASCADED_MODE_ASYMMETRIC",
    Self::Symmetric => "NVCOMP_CASCADED_MODE_SYMMETRIC",
});

bitflags::bitflags! {
    /// Terminal codecs that compression may use and decompression must support.
    ///
    /// Compression considers every enabled codec but may not use all of them.
    /// An empty set is invalid unless the mode is [`Mode::Unspecified`], which ignores it.
    ///
    /// For decompression, the set lists the codecs the input may have been compressed with.
    /// An input that used a codec outside the set fails with [`Status::CannotDecompress`](crate::error::Status::CannotDecompress).
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    pub struct TerminalCodecs: u64 {
        /// Bit packing.
        const BITPACK = sys::nvcompCascadedTerminalCodec_t::NVCOMP_CASCADED_TERMINAL_CODEC_CASCADED_BITPACK as _;
    }

    /// Encodings that compression may apply before [`TerminalCodecs::BITPACK`].
    ///
    /// Asymmetric mode requires [`FRAME_OF_REFERENCE`](Self::FRAME_OF_REFERENCE); symmetric mode does not.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    pub struct FineGrainedEncodings: u64 {
        /// Delta encoding.
        const DELTA = sys::nvcompCascadedFineGrainedEncoding_t::NVCOMP_CASCADED_FINE_GRAINED_ENCODING_DELTA as _;
        /// Run-length encoding.
        const RLE = sys::nvcompCascadedFineGrainedEncoding_t::NVCOMP_CASCADED_FINE_GRAINED_ENCODING_RLE as _;
        /// Frame-of-reference encoding.
        const FRAME_OF_REFERENCE = sys::nvcompCascadedFineGrainedEncoding_t::NVCOMP_CASCADED_FINE_GRAINED_ENCODING_FOR as _;
    }
}

/// Cascaded settings that compression requires and decompression can use.
///
/// Passing the compression settings to decompression makes it faster, because nvCOMP launches only the implementations the input needs.
/// When [`mode`](Self::mode) is [`Mode::Unspecified`], nvCOMP ignores the other fields and decompresses any Cascaded input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CommonOptions {
    /// Data type of the uncompressed input.
    ///
    /// Asymmetric mode supports signed and unsigned 8-bit to 64-bit integers, and symmetric mode supports signed and unsigned 32-bit and 64-bit integers.
    pub data_type: DataType,
    /// Compression scheme.
    pub mode: Mode,
    /// Terminal codecs that compression may use and decompression must support.
    pub terminal_codecs: TerminalCodecs,
}

impl Default for CommonOptions {
    fn default() -> Self {
        Self {
            data_type: DataType::Bits,
            mode: Mode::Unspecified,
            terminal_codecs: TerminalCodecs::empty(),
        }
    }
}

impl TryFrom<sys::nvcompCascadedCommonOpts_t> for CommonOptions {
    type Error = Error;

    fn try_from(value: sys::nvcompCascadedCommonOpts_t) -> Result<Self> {
        if value.coarse_grained_encoding_flags != 0 {
            return Err(Error::InvalidValue {
                name: "coarse_grained_encoding_flags".into(),
            });
        }
        Ok(Self {
            data_type: value.data_type.into(),
            mode: value.mode.into(),
            terminal_codecs: TerminalCodecs::from_bits(value.terminal_codec_flags).ok_or_else(
                || Error::InvalidValue {
                    name: "terminal_codec_flags".into(),
                },
            )?,
        })
    }
}

impl From<CommonOptions> for sys::nvcompCascadedCommonOpts_t {
    fn from(value: CommonOptions) -> Self {
        Self {
            data_type: value.data_type.into(),
            mode: value.mode.into(),
            terminal_codec_flags: value.terminal_codecs.bits(),
            coarse_grained_encoding_flags: 0,
        }
    }
}

/// Cascaded compression options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CompressOptions {
    /// Settings that decompression can also use to run faster.
    pub common: CommonOptions,
    /// Encodings that compression may apply.
    pub fine_grained_encodings: FineGrainedEncodings,
    /// Speed and compression-ratio tradeoff, 3 by default.
    ///
    /// In asymmetric mode, higher levels try more encoding stages.
    /// Levels above 7, or above 3 when only [`FineGrainedEncodings::DELTA`] is enabled, behave like the highest useful level.
    /// Symmetric mode ignores this setting.
    pub compression_level: u8,
}

impl Default for CompressOptions {
    fn default() -> Self {
        Self::new(DataType::U32)
    }
}

impl CompressOptions {
    /// Creates asymmetric compression options for `data_type` with bit packing, every fine-grained encoding, and level 3.
    #[must_use]
    pub fn new(data_type: DataType) -> Self {
        Self {
            common: CommonOptions {
                data_type,
                mode: Mode::Asymmetric,
                terminal_codecs: TerminalCodecs::BITPACK,
            },
            fine_grained_encodings: FineGrainedEncodings::all(),
            compression_level: 3,
        }
    }
}

impl TryFrom<sys::nvcompBatchedCascadedCompressOpts_t> for CompressOptions {
    type Error = Error;

    fn try_from(value: sys::nvcompBatchedCascadedCompressOpts_t) -> Result<Self> {
        validate_reserved(&value.reserved)?;
        Ok(Self {
            common: value.common_opts.try_into()?,
            fine_grained_encodings: FineGrainedEncodings::from_bits(
                value.fine_grained_encoding_flags,
            )
            .ok_or_else(|| Error::InvalidValue {
                name: "fine_grained_encoding_flags".into(),
            })?,
            compression_level: value.compression_level,
        })
    }
}

impl From<CompressOptions> for sys::nvcompBatchedCascadedCompressOpts_t {
    fn from(value: CompressOptions) -> Self {
        Self {
            common_opts: value.common.into(),
            fine_grained_encoding_flags: value.fine_grained_encodings.bits(),
            compression_level: value.compression_level,
            reserved: [0; 31],
        }
    }
}

/// Cascaded decompression options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DecompressOptions {
    /// Engine that runs decompression.
    pub backend: DecompressBackend,
    /// Settings used for compression, which let decompression run faster.
    ///
    /// Leave [`CommonOptions::mode`] as [`Mode::Unspecified`] to accept any Cascaded input.
    pub common: CommonOptions,
}

impl Default for DecompressOptions {
    fn default() -> Self {
        Self::new(DecompressBackend::Default)
    }
}

impl DecompressOptions {
    /// Creates options that decompress any Cascaded input with `backend`.
    #[must_use]
    pub fn new(backend: DecompressBackend) -> Self {
        Self {
            backend,
            common: CommonOptions::default(),
        }
    }
}

impl TryFrom<sys::nvcompBatchedCascadedDecompressOpts_t> for DecompressOptions {
    type Error = Error;

    fn try_from(value: sys::nvcompBatchedCascadedDecompressOpts_t) -> Result<Self> {
        validate_reserved(&value.reserved)?;
        Ok(Self {
            backend: value.backend.into(),
            common: value.common_opts.try_into()?,
        })
    }
}

impl From<DecompressOptions> for sys::nvcompBatchedCascadedDecompressOpts_t {
    fn from(value: DecompressOptions) -> Self {
        Self {
            backend: value.backend.into(),
            common_opts: value.common.into(),
            reserved: [0; 32],
        }
    }
}

impl_nvcomp_batched_codec_module! {
    "Cascaded",
    compress_options = nvcompBatchedCascadedCompressOpts_t,
    decompress_options = nvcompBatchedCascadedDecompressOpts_t,
    compression_max_allowed_chunk_size = nvcompCascadedCompressionMaxAllowedChunkSize,
    decompression_max_allowed_chunk_size = nvcompCascadedDecompressionMaxAllowedChunkSize,
    required_compression_alignment = nvcompCascadedRequiredCompressionAlignment,
    required_decompression_alignment = nvcompCascadedRequiredDecompressionAlignment,
    compress_get_required_alignments = nvcompBatchedCascadedCompressGetRequiredAlignments,
    decompress_get_required_alignments = nvcompBatchedCascadedDecompressGetRequiredAlignments,
    compress_get_temporary_size = nvcompBatchedCascadedCompressGetTempSize,
    compress_get_temporary_size_sync = nvcompBatchedCascadedCompressGetTempSizeSync,
    decompress_get_temporary_size = nvcompBatchedCascadedDecompressGetTempSize,
    decompress_get_temporary_size_sync = nvcompBatchedCascadedDecompressGetTempSizeSync,
    compress_get_max_output_chunk_size = nvcompBatchedCascadedCompressGetMaxOutputChunkSize,
    compress_raw_async = nvcompBatchedCascadedCompressAsync,
    decompress_size_raw_async = nvcompBatchedCascadedGetDecompressSizeAsync,
    decompress_raw_async = nvcompBatchedCascadedDecompressAsync,
}

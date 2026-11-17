use singe_nvcomp_sys as sys;

use crate::{
    codec::impl_nvcomp_batched_codec_module,
    error::{Error, Result},
    types::{BitshuffleMode, DataType, DecompressBackend},
    utility::{bool_from_i32, validate_reserved},
};

/// LZ4 compression options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CompressOptions {
    /// Data type of the uncompressed input.
    ///
    /// Each chunk size must be a multiple of this type's size.
    pub data_type: DataType,
    /// Bit shuffle applied before compression.
    ///
    /// nvCOMP shuffles 8 KiB subchunks.
    /// In a smaller subchunk, it shuffles the largest multiple of 8 elements and leaves the rest unshuffled.
    pub bitshuffle_mode: BitshuffleMode,
}

impl Default for CompressOptions {
    fn default() -> Self {
        Self {
            data_type: DataType::I8,
            bitshuffle_mode: BitshuffleMode::None,
        }
    }
}

impl TryFrom<sys::nvcompBatchedLZ4CompressOpts_t> for CompressOptions {
    type Error = Error;

    fn try_from(value: sys::nvcompBatchedLZ4CompressOpts_t) -> Result<Self> {
        validate_reserved(&value.reserved)?;
        Ok(Self {
            data_type: value.data_type.try_into()?,
            bitshuffle_mode: value.bitshuffle_mode.try_into()?,
        })
    }
}

impl From<CompressOptions> for sys::nvcompBatchedLZ4CompressOpts_t {
    fn from(value: CompressOptions) -> Self {
        Self {
            data_type: value.data_type.into(),
            bitshuffle_mode: value.bitshuffle_mode.into(),
            reserved: [0; 56],
        }
    }
}

/// LZ4 decompression options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DecompressOptions {
    /// Engine that runs decompression.
    pub backend: DecompressBackend,
    /// Whether to sort chunks before hardware decompression to balance the load.
    ///
    /// Used only with [`DecompressBackend::Hardware`].
    pub sort_before_hw_decompress: bool,
    /// Data type used for compression, which decompression needs to reverse the bit shuffle.
    ///
    /// It must match the compression setting.
    /// nvCOMP ignores it when bit shuffling is off.
    pub data_type: DataType,
    /// Bit shuffle used for compression, which decompression reverses.
    pub bitshuffle_mode: BitshuffleMode,
}

impl Default for DecompressOptions {
    fn default() -> Self {
        Self {
            backend: DecompressBackend::Default,
            sort_before_hw_decompress: false,
            data_type: DataType::I8,
            bitshuffle_mode: BitshuffleMode::None,
        }
    }
}

impl TryFrom<sys::nvcompBatchedLZ4DecompressOpts_t> for DecompressOptions {
    type Error = Error;

    fn try_from(value: sys::nvcompBatchedLZ4DecompressOpts_t) -> Result<Self> {
        validate_reserved(&value.reserved)?;
        Ok(Self {
            backend: value.backend.try_into()?,
            sort_before_hw_decompress: bool_from_i32(
                value.sort_before_hw_decompress,
                "sort_before_hw_decompress",
            )?,
            data_type: value.data_type.try_into()?,
            bitshuffle_mode: value.bitshuffle_mode.try_into()?,
        })
    }
}

impl From<DecompressOptions> for sys::nvcompBatchedLZ4DecompressOpts_t {
    fn from(value: DecompressOptions) -> Self {
        Self {
            backend: value.backend.into(),
            sort_before_hw_decompress: i32::from(value.sort_before_hw_decompress),
            data_type: value.data_type.into(),
            bitshuffle_mode: value.bitshuffle_mode.into(),
            reserved: [0; 48],
        }
    }
}

impl_nvcomp_batched_codec_module! {
    "LZ4",
    compress_options = nvcompBatchedLZ4CompressOpts_t,
    decompress_options = nvcompBatchedLZ4DecompressOpts_t,
    compression_max_allowed_chunk_size = nvcompLZ4CompressionMaxAllowedChunkSize,
    decompression_max_allowed_chunk_size = nvcompLZ4DecompressionMaxAllowedChunkSize,
    required_compression_alignment = nvcompLZ4RequiredCompressionAlignment,
    required_decompression_alignment = nvcompLZ4RequiredDecompressionAlignment,
    compress_get_required_alignments = nvcompBatchedLZ4CompressGetRequiredAlignments,
    decompress_get_required_alignments = nvcompBatchedLZ4DecompressGetRequiredAlignments,
    compress_get_temp_size = nvcompBatchedLZ4CompressGetTempSize,
    compress_get_temp_size_sync = nvcompBatchedLZ4CompressGetTempSizeSync,
    decompress_get_temp_size = nvcompBatchedLZ4DecompressGetTempSize,
    decompress_get_temp_size_sync = nvcompBatchedLZ4DecompressGetTempSizeSync,
    compress_get_max_output_chunk_size = nvcompBatchedLZ4CompressGetMaxOutputChunkSize,
    compress_async = nvcompBatchedLZ4CompressAsync,
    get_decompress_size_async = nvcompBatchedLZ4GetDecompressSizeAsync,
    decompress_async = nvcompBatchedLZ4DecompressAsync,
}

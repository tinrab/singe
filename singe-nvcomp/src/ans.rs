use singe_nvcomp_sys as sys;

use crate::{
    codec::impl_nvcomp_batched_codec_module,
    types::{AnsType, DataType, DecompressBackend},
    utility::{bool_from_i32, validate_reserved},
};

/// ANS compression options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CompressOptions {
    /// ANS coding variant.
    pub type_: AnsType,
    /// Data type of the uncompressed input.
    ///
    /// Use [`DataType::I8`] or [`DataType::U8`] for generic bytes, and [`DataType::F16`] for any half-precision format.
    pub data_type: DataType,
    /// Maximum number of subchunks per chunk, or 0 for the default of 8.
    ///
    /// A nonzero value must be a power of two from 4 to 64. Leave it at 0 unless you have measured its effect on your application.
    pub max_sub_chunk_count: u8,
    /// Number of interleaved rANS streams per lane: 1, 2, or 0 to choose automatically.
    ///
    /// The automatic choice uses two streams for every data type.
    /// Two streams are faster but add 128 bytes per subchunk, which lowers the compression ratio more as subchunks get smaller.
    pub states_per_lane: u8,
    /// Base-2 logarithm of how much input to skip when building the histogram.
    ///
    /// A value of `n` samples `1 / 2^n` of the input, up to `n` = 8; 0 builds an exact histogram.
    /// Sampling speeds up compression when every chunk follows the same distribution.
    /// Chunks under 4096 ANS symbols always get an exact histogram.
    pub histogram_reduction_log2: u8,
}

impl Default for CompressOptions {
    fn default() -> Self {
        Self {
            type_: AnsType::Rans,
            data_type: DataType::I8,
            max_sub_chunk_count: 0,
            states_per_lane: 0,
            histogram_reduction_log2: 0,
        }
    }
}

impl TryFrom<sys::nvcompBatchedANSCompressOpts_t> for CompressOptions {
    type Error = Error;

    fn try_from(value: sys::nvcompBatchedANSCompressOpts_t) -> Result<Self> {
        validate_reserved(&value.reserved)?;
        Ok(Self {
            type_: value.type_.into(),
            data_type: value.data_type.into(),
            max_sub_chunk_count: value.max_sub_chunk_count,
            states_per_lane: value.states_per_lane,
            histogram_reduction_log2: value.histogram_reduction_log2,
        })
    }
}

impl From<CompressOptions> for sys::nvcompBatchedANSCompressOpts_t {
    fn from(value: CompressOptions) -> Self {
        Self {
            type_: value.type_.into(),
            data_type: value.data_type.into(),
            max_sub_chunk_count: value.max_sub_chunk_count,
            states_per_lane: value.states_per_lane,
            histogram_reduction_log2: value.histogram_reduction_log2,
            reserved: [0; 53],
        }
    }
}

/// ANS decompression options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DecompressOptions {
    /// Engine that runs decompression.
    pub backend: DecompressBackend,
    /// Data type used for compression.
    ///
    /// The default, [`DataType::Bits`], reads the type from the compressed data, so any ANS stream decodes.
    /// Naming a type requires it to match the compressed data.
    pub data_type: DataType,
    /// Maximum number of subchunks per chunk, or 0 to read it from the compressed data.
    ///
    /// A nonzero value must be a power of two from 4 to 64. Decompression fails if it is smaller than the count used for compression.
    pub max_sub_chunk_count: u8,
    /// Number of interleaved rANS streams per lane used for compression, or 0 to read it from the compressed data.
    ///
    /// A nonzero value must match the compressed data.
    /// Together with a named [`data_type`](Self::data_type), it lets nvCOMP launch a more specialized kernel.
    pub states_per_lane: u8,
    /// Whether the decompression kernel skips validating each chunk.
    pub skip_validate: bool,
}

impl Default for DecompressOptions {
    fn default() -> Self {
        Self {
            backend: DecompressBackend::Default,
            data_type: DataType::Bits,
            max_sub_chunk_count: 0,
            states_per_lane: 0,
            skip_validate: false,
        }
    }
}

impl TryFrom<sys::nvcompBatchedANSDecompressOpts_t> for DecompressOptions {
    type Error = Error;

    fn try_from(value: sys::nvcompBatchedANSDecompressOpts_t) -> Result<Self> {
        validate_reserved(&value.reserved)?;
        Ok(Self {
            backend: value.backend.into(),
            data_type: value.data_type.into(),
            max_sub_chunk_count: value.max_sub_chunk_count,
            states_per_lane: value.states_per_lane,
            skip_validate: bool_from_i32(value.skip_validate.into(), "skip_validate")?,
        })
    }
}

impl From<DecompressOptions> for sys::nvcompBatchedANSDecompressOpts_t {
    fn from(value: DecompressOptions) -> Self {
        Self {
            backend: value.backend.into(),
            data_type: value.data_type.into(),
            max_sub_chunk_count: value.max_sub_chunk_count,
            states_per_lane: value.states_per_lane,
            skip_validate: value.skip_validate.into(),
            reserved: [0; 53],
        }
    }
}

impl_nvcomp_batched_codec_module! {
    "ANS",
    compress_options = nvcompBatchedANSCompressOpts_t,
    decompress_options = nvcompBatchedANSDecompressOpts_t,
    compression_max_allowed_chunk_size = nvcompANSCompressionMaxAllowedChunkSize,
    decompression_max_allowed_chunk_size = nvcompANSDecompressionMaxAllowedChunkSize,
    required_compression_alignment = nvcompANSRequiredCompressionAlignment,
    required_decompression_alignment = nvcompANSRequiredDecompressionAlignment,
    compress_get_required_alignments = nvcompBatchedANSCompressGetRequiredAlignments,
    decompress_get_required_alignments = nvcompBatchedANSDecompressGetRequiredAlignments,
    compress_get_temporary_size = nvcompBatchedANSCompressGetTempSize,
    compress_get_temporary_size_sync = nvcompBatchedANSCompressGetTempSizeSync,
    decompress_get_temporary_size = nvcompBatchedANSDecompressGetTempSize,
    decompress_get_temporary_size_sync = nvcompBatchedANSDecompressGetTempSizeSync,
    compress_get_max_output_chunk_size = nvcompBatchedANSCompressGetMaxOutputChunkSize,
    compress_raw_async = nvcompBatchedANSCompressAsync,
    decompress_size_raw_async = nvcompBatchedANSGetDecompressSizeAsync,
    decompress_raw_async = nvcompBatchedANSDecompressAsync,
}

impl CompressOptions {
    /// Creates ANS compression options with the given settings.
    #[must_use]
    pub fn new(type_: AnsType, data_type: DataType) -> Self {
        Self {
            type_,
            data_type,
            ..Self::default()
        }
    }
}

impl DecompressOptions {
    /// Creates ANS decompression options with the given settings.
    #[must_use]
    pub fn new(backend: DecompressBackend) -> Self {
        Self {
            backend,
            ..Self::default()
        }
    }
}

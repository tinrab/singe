use singe_nvcomp_sys as sys;

use crate::{
    codec::impl_nvcomp_batched_codec_module,
    error::{Error, Result},
    types::{CompressionAlgorithm, DecompressBackend, GzipDecompressAlgorithm},
    utility::{bool_from_i32, codec_size, validate_reserved},
};

/// Largest uncompressed chunk size, in bytes, that compression accepts.
///
/// nvCOMP balances the work internally, so compression speed depends little on the chunk size.
pub const COMPRESSION_MAX_ALLOWED_CHUNK_SIZE: usize =
    codec_size(sys::nvcompGzipCompressionMaxAllowedChunkSize);

/// Alignment, in bytes, that meets every minimum alignment for compression buffers.
///
/// Typed buffers must also be aligned to their element size, such as 4 bytes for an `i32`.
pub const REQUIRED_COMPRESSION_ALIGNMENT: usize =
    codec_size(sys::nvcompGzipRequiredCompressionAlignment);

/// Gzip compression options.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct CompressOptions {
    /// Speed and compression-ratio tradeoff.
    pub algorithm: CompressionAlgorithm,
}

impl TryFrom<sys::nvcompBatchedGzipCompressOpts_t> for CompressOptions {
    type Error = Error;

    fn try_from(value: sys::nvcompBatchedGzipCompressOpts_t) -> Result<Self> {
        validate_reserved(&value.reserved)?;
        Ok(Self {
            algorithm: value
                .algorithm
                .try_into()
                .map_err(|_| Error::UnknownEnumValue {
                    name: "algorithm".into(),
                    value: i64::from(value.algorithm),
                })?,
        })
    }
}

impl From<CompressOptions> for sys::nvcompBatchedGzipCompressOpts_t {
    fn from(value: CompressOptions) -> Self {
        Self {
            algorithm: value.algorithm.into(),
            reserved: [0; 60],
        }
    }
}

/// Largest compressed or decompressed chunk size, in bytes, that the naive decompression algorithm accepts.
///
/// Smaller chunks, such as 64 KiB, or the lookahead algorithm decompress faster.
pub const NAIVE_DECOMPRESSION_MAX_ALLOWED_CHUNK_SIZE: usize =
    codec_size(sys::nvcompGzipNaiveDecompressionMaxAllowedChunkSize);

/// Largest compressed or decompressed chunk size, in bytes, that the lookahead decompression algorithm accepts.
///
/// The hardware decompression engine may have a lower limit; see [`DecompressBackend::Hardware`].
pub const LOOKAHEAD_DECOMPRESSION_MAX_ALLOWED_CHUNK_SIZE: usize =
    codec_size(sys::nvcompGzipLookaheadDecompressionMaxAllowedChunkSize);

/// Alignment, in bytes, that meets every minimum alignment for decompression buffers.
///
/// Typed buffers must also be aligned to their element size, such as 4 bytes for an `i32`.
pub const REQUIRED_DECOMPRESSION_ALIGNMENT: usize =
    codec_size(sys::nvcompGzipRequiredDecompressionAlignment);

/// Gzip decompression options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DecompressOptions {
    /// Engine that runs decompression.
    pub backend: DecompressBackend,
    /// CUDA algorithm for decompression.
    pub algorithm: GzipDecompressAlgorithm,
    /// Whether to sort chunks before hardware decompression to balance the load.
    ///
    /// Used only with [`DecompressBackend::Hardware`].
    pub sort_before_hw_decompress: bool,
}

impl Default for DecompressOptions {
    fn default() -> Self {
        Self {
            backend: DecompressBackend::Default,
            algorithm: GzipDecompressAlgorithm::Naive,
            sort_before_hw_decompress: false,
        }
    }
}

impl TryFrom<sys::nvcompBatchedGzipDecompressOpts_t> for DecompressOptions {
    type Error = Error;

    fn try_from(value: sys::nvcompBatchedGzipDecompressOpts_t) -> Result<Self> {
        validate_reserved(&value.reserved)?;
        Ok(Self {
            backend: value.backend.try_into()?,
            algorithm: value.algorithm.try_into()?,
            sort_before_hw_decompress: bool_from_i32(
                value.sort_before_hw_decompress,
                "sort_before_hw_decompress",
            )?,
        })
    }
}

impl From<DecompressOptions> for sys::nvcompBatchedGzipDecompressOpts_t {
    fn from(value: DecompressOptions) -> Self {
        Self {
            backend: value.backend.into(),
            algorithm: value.algorithm.into(),
            sort_before_hw_decompress: i32::from(value.sort_before_hw_decompress),
            reserved: [0; 52],
        }
    }
}

const fn decompression_chunk_limit(options: DecompressOptions) -> usize {
    match options.algorithm {
        GzipDecompressAlgorithm::Naive => NAIVE_DECOMPRESSION_MAX_ALLOWED_CHUNK_SIZE,
        GzipDecompressAlgorithm::Lookahead => LOOKAHEAD_DECOMPRESSION_MAX_ALLOWED_CHUNK_SIZE,
    }
}

impl_nvcomp_batched_codec_module! {
    @operations
    compress_options = nvcompBatchedGzipCompressOpts_t,
    decompress_options = nvcompBatchedGzipDecompressOpts_t,
    compress_get_required_alignments = nvcompBatchedGzipCompressGetRequiredAlignments,
    decompress_get_required_alignments = nvcompBatchedGzipDecompressGetRequiredAlignments,
    compress_get_temp_size = nvcompBatchedGzipCompressGetTempSize,
    compress_get_temp_size_sync = nvcompBatchedGzipCompressGetTempSizeSync,
    decompress_get_temp_size = nvcompBatchedGzipDecompressGetTempSize,
    decompress_get_temp_size_sync = nvcompBatchedGzipDecompressGetTempSizeSync,
    compress_get_max_output_chunk_size = nvcompBatchedGzipCompressGetMaxOutputChunkSize,
    compress_async = nvcompBatchedGzipCompressAsync,
    get_decompress_size_async = nvcompBatchedGzipGetDecompressSizeAsync,
    decompress_async = nvcompBatchedGzipDecompressAsync,
}

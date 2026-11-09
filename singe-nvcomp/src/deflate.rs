use singe_nvcomp_sys as sys;

use crate::{
    codec::impl_nvcomp_batched_codec_module,
    types::{CompressionAlgorithm, DecompressBackend},
    utility::{bool_from_i32, validate_reserved},
};

/// Deflate compression options.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct CompressOptions {
    /// Speed and compression-ratio tradeoff.
    pub algorithm: CompressionAlgorithm,
}

impl TryFrom<sys::nvcompBatchedDeflateCompressOpts_t> for CompressOptions {
    type Error = Error;

    fn try_from(value: sys::nvcompBatchedDeflateCompressOpts_t) -> Result<Self> {
        validate_reserved(&value.reserved)?;
        Ok(Self {
            algorithm: value
                .algorithm
                .try_into()
                .map_err(|_| Error::InvalidValue {
                    name: "algorithm".into(),
                })?,
        })
    }
}

impl From<CompressOptions> for sys::nvcompBatchedDeflateCompressOpts_t {
    fn from(value: CompressOptions) -> Self {
        Self {
            algorithm: value.algorithm.into(),
            reserved: [0; 60],
        }
    }
}

/// Deflate decompression options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DecompressOptions {
    /// Engine that runs decompression.
    pub backend: DecompressBackend,
    /// Whether to sort chunks before hardware decompression to balance the load.
    ///
    /// Used only with [`DecompressBackend::Hardware`].
    pub sort_before_hw_decompress: bool,
}

impl Default for DecompressOptions {
    fn default() -> Self {
        Self {
            backend: DecompressBackend::Default,
            sort_before_hw_decompress: false,
        }
    }
}

impl TryFrom<sys::nvcompBatchedDeflateDecompressOpts_t> for DecompressOptions {
    type Error = Error;

    fn try_from(value: sys::nvcompBatchedDeflateDecompressOpts_t) -> Result<Self> {
        validate_reserved(&value.reserved)?;
        Ok(Self {
            backend: value.backend.into(),
            sort_before_hw_decompress: bool_from_i32(
                value.sort_before_hw_decompress,
                "sort_before_hw_decompress",
            )?,
        })
    }
}

impl From<DecompressOptions> for sys::nvcompBatchedDeflateDecompressOpts_t {
    fn from(value: DecompressOptions) -> Self {
        Self {
            backend: value.backend.into(),
            sort_before_hw_decompress: i32::from(value.sort_before_hw_decompress),
            reserved: [0; 56],
        }
    }
}

impl_nvcomp_batched_codec_module! {
    "Deflate",
    compress_options = nvcompBatchedDeflateCompressOpts_t,
    decompress_options = nvcompBatchedDeflateDecompressOpts_t,
    compression_max_allowed_chunk_size = nvcompDeflateCompressionMaxAllowedChunkSize,
    decompression_max_allowed_chunk_size = nvcompDeflateDecompressionMaxAllowedChunkSize,
    required_compression_alignment = nvcompDeflateRequiredCompressionAlignment,
    required_decompression_alignment = nvcompDeflateRequiredDecompressionAlignment,
    compress_get_required_alignments = nvcompBatchedDeflateCompressGetRequiredAlignments,
    decompress_get_required_alignments = nvcompBatchedDeflateDecompressGetRequiredAlignments,
    compress_get_temporary_size = nvcompBatchedDeflateCompressGetTempSize,
    compress_get_temporary_size_sync = nvcompBatchedDeflateCompressGetTempSizeSync,
    decompress_get_temporary_size = nvcompBatchedDeflateDecompressGetTempSize,
    decompress_get_temporary_size_sync = nvcompBatchedDeflateDecompressGetTempSizeSync,
    compress_get_max_output_chunk_size = nvcompBatchedDeflateCompressGetMaxOutputChunkSize,
    compress_raw_async = nvcompBatchedDeflateCompressAsync,
    decompress_size_raw_async = nvcompBatchedDeflateGetDecompressSizeAsync,
    decompress_raw_async = nvcompBatchedDeflateDecompressAsync,
}

impl CompressOptions {
    /// Creates Deflate compression options with the given settings.
    #[must_use]
    pub fn new(algorithm: CompressionAlgorithm) -> Self {
        Self { algorithm }
    }
}

impl DecompressOptions {
    /// Creates Deflate decompression options with the given settings.
    #[must_use]
    pub fn new(backend: DecompressBackend, sort_before_hw_decompress: bool) -> Self {
        Self {
            backend,
            sort_before_hw_decompress,
        }
    }
}

use singe_nvcomp_sys as sys;

use crate::{
    codec::impl_nvcomp_batched_codec_module,
    types::{CompressionAlgorithm, DecompressBackend},
    utility::validate_reserved,
};

/// `GDeflate` compression options.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct CompressOptions {
    /// Speed and compression-ratio tradeoff.
    pub algorithm: CompressionAlgorithm,
}

impl TryFrom<sys::nvcompBatchedGdeflateCompressOpts_t> for CompressOptions {
    type Error = Error;

    fn try_from(value: sys::nvcompBatchedGdeflateCompressOpts_t) -> Result<Self> {
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

impl From<CompressOptions> for sys::nvcompBatchedGdeflateCompressOpts_t {
    fn from(value: CompressOptions) -> Self {
        Self {
            algorithm: value.algorithm.into(),
            reserved: [0; 60],
        }
    }
}

/// `GDeflate` decompression options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DecompressOptions {
    /// Engine that runs decompression.
    pub backend: DecompressBackend,
}

impl Default for DecompressOptions {
    fn default() -> Self {
        Self {
            backend: DecompressBackend::Default,
        }
    }
}

impl TryFrom<sys::nvcompBatchedGdeflateDecompressOpts_t> for DecompressOptions {
    type Error = Error;

    fn try_from(value: sys::nvcompBatchedGdeflateDecompressOpts_t) -> Result<Self> {
        validate_reserved(&value.reserved)?;
        Ok(Self {
            backend: value.backend.into(),
        })
    }
}

impl From<DecompressOptions> for sys::nvcompBatchedGdeflateDecompressOpts_t {
    fn from(value: DecompressOptions) -> Self {
        Self {
            backend: value.backend.into(),
            reserved: [0; 60],
        }
    }
}

impl_nvcomp_batched_codec_module! {
    "Gdeflate",
    compress_options = nvcompBatchedGdeflateCompressOpts_t,
    decompress_options = nvcompBatchedGdeflateDecompressOpts_t,
    compression_max_allowed_chunk_size = nvcompGdeflateCompressionMaxAllowedChunkSize,
    decompression_max_allowed_chunk_size = nvcompGdeflateDecompressionMaxAllowedChunkSize,
    required_compression_alignment = nvcompGdeflateRequiredCompressionAlignment,
    required_decompression_alignment = nvcompGdeflateRequiredDecompressionAlignment,
    compress_get_required_alignments = nvcompBatchedGdeflateCompressGetRequiredAlignments,
    decompress_get_required_alignments = nvcompBatchedGdeflateDecompressGetRequiredAlignments,
    compress_get_temporary_size = nvcompBatchedGdeflateCompressGetTempSize,
    compress_get_temporary_size_sync = nvcompBatchedGdeflateCompressGetTempSizeSync,
    decompress_get_temporary_size = nvcompBatchedGdeflateDecompressGetTempSize,
    decompress_get_temporary_size_sync = nvcompBatchedGdeflateDecompressGetTempSizeSync,
    compress_get_max_output_chunk_size = nvcompBatchedGdeflateCompressGetMaxOutputChunkSize,
    compress_raw_async = nvcompBatchedGdeflateCompressAsync,
    decompress_size_raw_async = nvcompBatchedGdeflateGetDecompressSizeAsync,
    decompress_raw_async = nvcompBatchedGdeflateDecompressAsync,
}

impl CompressOptions {
    /// Creates `GDeflate` compression options with the given settings.
    #[must_use]
    pub fn new(algorithm: CompressionAlgorithm) -> Self {
        Self { algorithm }
    }
}

impl DecompressOptions {
    /// Creates `GDeflate` decompression options with the given settings.
    #[must_use]
    pub fn new(backend: DecompressBackend) -> Self {
        Self { backend }
    }
}

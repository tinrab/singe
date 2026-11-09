use singe_nvcomp_sys as sys;

use crate::{
    codec::impl_nvcomp_batched_codec_module, types::DecompressBackend, utility::validate_reserved,
};

/// Zstd compression options.
///
/// Zstd compression has no settings in nvCOMP 6.0.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct CompressOptions;

impl TryFrom<sys::nvcompBatchedZstdCompressOpts_t> for CompressOptions {
    type Error = Error;

    fn try_from(value: sys::nvcompBatchedZstdCompressOpts_t) -> Result<Self> {
        validate_reserved(&value.reserved)?;
        Ok(Self)
    }
}

impl From<CompressOptions> for sys::nvcompBatchedZstdCompressOpts_t {
    fn from(_: CompressOptions) -> Self {
        Self { reserved: [0; 64] }
    }
}

/// Zstd decompression options.
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

impl TryFrom<sys::nvcompBatchedZstdDecompressOpts_t> for DecompressOptions {
    type Error = Error;

    fn try_from(value: sys::nvcompBatchedZstdDecompressOpts_t) -> Result<Self> {
        validate_reserved(&value.reserved)?;
        Ok(Self {
            backend: value.backend.into(),
        })
    }
}

impl From<DecompressOptions> for sys::nvcompBatchedZstdDecompressOpts_t {
    fn from(value: DecompressOptions) -> Self {
        Self {
            backend: value.backend.into(),
            reserved: [0; 60],
        }
    }
}

impl_nvcomp_batched_codec_module! {
    "Zstd",
    compress_options = nvcompBatchedZstdCompressOpts_t,
    decompress_options = nvcompBatchedZstdDecompressOpts_t,
    compression_max_allowed_chunk_size = nvcompZstdCompressionMaxAllowedChunkSize,
    decompression_max_allowed_chunk_size = nvcompZstdDecompressionMaxAllowedChunkSize,
    required_compression_alignment = nvcompZstdRequiredCompressionAlignment,
    required_decompression_alignment = nvcompZstdRequiredDecompressionAlignment,
    compress_get_required_alignments = nvcompBatchedZstdCompressGetRequiredAlignments,
    decompress_get_required_alignments = nvcompBatchedZstdDecompressGetRequiredAlignments,
    compress_get_temporary_size = nvcompBatchedZstdCompressGetTempSize,
    compress_get_temporary_size_sync = nvcompBatchedZstdCompressGetTempSizeSync,
    decompress_get_temporary_size = nvcompBatchedZstdDecompressGetTempSize,
    decompress_get_temporary_size_sync = nvcompBatchedZstdDecompressGetTempSizeSync,
    compress_get_max_output_chunk_size = nvcompBatchedZstdCompressGetMaxOutputChunkSize,
    compress_raw_async = nvcompBatchedZstdCompressAsync,
    decompress_size_raw_async = nvcompBatchedZstdGetDecompressSizeAsync,
    decompress_raw_async = nvcompBatchedZstdDecompressAsync,
}

impl CompressOptions {
    /// Creates Zstd compression options.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl DecompressOptions {
    /// Creates Zstd decompression options with the given settings.
    #[must_use]
    pub fn new(backend: DecompressBackend) -> Self {
        Self { backend }
    }
}

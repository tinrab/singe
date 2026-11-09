use singe_nvcomp_sys as sys;

use crate::{
    codec::impl_nvcomp_batched_codec_module,
    types::DecompressBackend,
    utility::{bool_from_i32, validate_reserved},
};

/// Snappy compression options.
///
/// Snappy compression has no settings in nvCOMP 6.0.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct CompressOptions;

impl TryFrom<sys::nvcompBatchedSnappyCompressOpts_t> for CompressOptions {
    type Error = Error;

    fn try_from(value: sys::nvcompBatchedSnappyCompressOpts_t) -> Result<Self> {
        validate_reserved(&value.reserved)?;
        Ok(Self)
    }
}

impl From<CompressOptions> for sys::nvcompBatchedSnappyCompressOpts_t {
    fn from(_: CompressOptions) -> Self {
        Self { reserved: [0; 64] }
    }
}

/// Snappy decompression options.
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

impl TryFrom<sys::nvcompBatchedSnappyDecompressOpts_t> for DecompressOptions {
    type Error = Error;

    fn try_from(value: sys::nvcompBatchedSnappyDecompressOpts_t) -> Result<Self> {
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

impl From<DecompressOptions> for sys::nvcompBatchedSnappyDecompressOpts_t {
    fn from(value: DecompressOptions) -> Self {
        Self {
            backend: value.backend.into(),
            sort_before_hw_decompress: i32::from(value.sort_before_hw_decompress),
            reserved: [0; 56],
        }
    }
}

impl_nvcomp_batched_codec_module! {
    "Snappy",
    compress_options = nvcompBatchedSnappyCompressOpts_t,
    decompress_options = nvcompBatchedSnappyDecompressOpts_t,
    compression_max_allowed_chunk_size = nvcompSnappyCompressionMaxAllowedChunkSize,
    decompression_max_allowed_chunk_size = nvcompSnappyDecompressionMaxAllowedChunkSize,
    required_compression_alignment = nvcompSnappyRequiredCompressionAlignment,
    required_decompression_alignment = nvcompSnappyRequiredDecompressionAlignment,
    compress_get_required_alignments = nvcompBatchedSnappyCompressGetRequiredAlignments,
    decompress_get_required_alignments = nvcompBatchedSnappyDecompressGetRequiredAlignments,
    compress_get_temporary_size = nvcompBatchedSnappyCompressGetTempSize,
    compress_get_temporary_size_sync = nvcompBatchedSnappyCompressGetTempSizeSync,
    decompress_get_temporary_size = nvcompBatchedSnappyDecompressGetTempSize,
    decompress_get_temporary_size_sync = nvcompBatchedSnappyDecompressGetTempSizeSync,
    compress_get_max_output_chunk_size = nvcompBatchedSnappyCompressGetMaxOutputChunkSize,
    compress_raw_async = nvcompBatchedSnappyCompressAsync,
    decompress_size_raw_async = nvcompBatchedSnappyGetDecompressSizeAsync,
    decompress_raw_async = nvcompBatchedSnappyDecompressAsync,
}

impl CompressOptions {
    /// Creates Snappy compression options.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl DecompressOptions {
    /// Creates Snappy decompression options with the given settings.
    #[must_use]
    pub fn new(backend: DecompressBackend, sort_before_hw_decompress: bool) -> Self {
        Self {
            backend,
            sort_before_hw_decompress,
        }
    }
}

//! Batched Bitcomp compression and decompression.
//!
//! Batched Bitcomp decompression is not fully asynchronous: nvCOMP synchronizes the stream to read the compressed data, then launches decompression.
//! As with other codecs, synchronize again and check [`DeviceStatuses`] before reading the results.
use singe_nvcomp_sys as sys;

use crate::{
    codec::impl_nvcomp_batched_codec_module,
    types::{BitcompAlgorithm, BitcompMode, DataType, DecompressBackend},
    utility::validate_reserved,
};

/// Bitcomp compression options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompressOptions {
    /// Compression algorithm.
    pub algorithm: BitcompAlgorithm,
    /// Data type of the uncompressed input.
    ///
    /// Lossless compression supports the integer types and [`DataType::F16`], [`DataType::F32`], and [`DataType::F64`].
    /// Lossy compression supports only those three floating-point types.
    pub data_type: DataType,
    /// Quantization step for lossy compression.
    ///
    /// Decompressed values differ from the originals by at most `delta / 2`.
    /// nvCOMP rounds `delta` down to a power of two.
    /// It must stay finite, positive, and normal after conversion to [`data_type`](Self::data_type).
    /// Lossless mode ignores it.
    pub delta: f64,
    /// Lossless or lossy compression.
    pub mode: BitcompMode,
}

impl Default for CompressOptions {
    fn default() -> Self {
        Self {
            algorithm: BitcompAlgorithm::Default,
            data_type: DataType::U8,
            delta: 0.0,
            mode: BitcompMode::Lossless,
        }
    }
}

impl TryFrom<sys::nvcompBatchedBitcompCompressOpts_t> for CompressOptions {
    type Error = Error;

    fn try_from(value: sys::nvcompBatchedBitcompCompressOpts_t) -> Result<Self> {
        validate_reserved(&value.reserved)?;
        let algorithm = value
            .algorithm
            .try_into()
            .map_err(|_| Error::InvalidValue {
                name: "algorithm".into(),
            })?;
        Ok(Self {
            algorithm,
            data_type: value.data_type.into(),
            delta: value.delta,
            mode: value.mode.into(),
        })
    }
}

impl From<CompressOptions> for sys::nvcompBatchedBitcompCompressOpts_t {
    fn from(value: CompressOptions) -> Self {
        Self {
            algorithm: value.algorithm.into(),
            data_type: value.data_type.into(),
            delta: value.delta,
            mode: value.mode.into(),
            reserved: [0; 44],
        }
    }
}

/// Bitcomp decompression options.
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

impl TryFrom<sys::nvcompBatchedBitcompDecompressOpts_t> for DecompressOptions {
    type Error = Error;

    fn try_from(value: sys::nvcompBatchedBitcompDecompressOpts_t) -> Result<Self> {
        validate_reserved(&value.reserved)?;
        Ok(Self {
            backend: value.backend.into(),
        })
    }
}

impl From<DecompressOptions> for sys::nvcompBatchedBitcompDecompressOpts_t {
    fn from(value: DecompressOptions) -> Self {
        Self {
            backend: value.backend.into(),
            reserved: [0; 60],
        }
    }
}

impl_nvcomp_batched_codec_module! {
    "Bitcomp",
    compress_options = nvcompBatchedBitcompCompressOpts_t,
    decompress_options = nvcompBatchedBitcompDecompressOpts_t,
    compression_max_allowed_chunk_size = nvcompBitcompCompressionMaxAllowedChunkSize,
    decompression_max_allowed_chunk_size = nvcompBitcompDecompressionMaxAllowedChunkSize,
    required_compression_alignment = nvcompBitcompRequiredCompressionAlignment,
    required_decompression_alignment = nvcompBitcompRequiredDecompressionAlignment,
    compress_get_required_alignments = nvcompBatchedBitcompCompressGetRequiredAlignments,
    decompress_get_required_alignments = nvcompBatchedBitcompDecompressGetRequiredAlignments,
    compress_get_temporary_size = nvcompBatchedBitcompCompressGetTempSize,
    compress_get_temporary_size_sync = nvcompBatchedBitcompCompressGetTempSizeSync,
    decompress_get_temporary_size = nvcompBatchedBitcompDecompressGetTempSize,
    decompress_get_temporary_size_sync = nvcompBatchedBitcompDecompressGetTempSizeSync,
    compress_get_max_output_chunk_size = nvcompBatchedBitcompCompressGetMaxOutputChunkSize,
    compress_raw_async = nvcompBatchedBitcompCompressAsync,
    decompress_size_raw_async = nvcompBatchedBitcompGetDecompressSizeAsync,
    decompress_raw_async = nvcompBatchedBitcompDecompressAsync,
}

impl CompressOptions {
    /// Creates Bitcomp compression options with the given settings.
    #[must_use]
    pub fn new(algorithm: BitcompAlgorithm, data_type: DataType) -> Self {
        Self {
            algorithm,
            data_type,
            ..Self::default()
        }
    }
}

impl DecompressOptions {
    /// Creates Bitcomp decompression options with the given settings.
    #[must_use]
    pub fn new(backend: DecompressBackend) -> Self {
        Self { backend }
    }
}

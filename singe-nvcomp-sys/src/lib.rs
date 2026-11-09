//! Raw FFI bindings for nvCOMP.
//!
//! Prefer the safe `singe-nvcomp` crate unless direct NVIDIA ABI access is required.

#![allow(
    unpredictable_function_pointer_comparisons,
    non_upper_case_globals,
    deprecated,
    dead_code,
    non_camel_case_types,
    non_snake_case,
    unnecessary_transmutes,
    unused_imports,
    unused_qualifications,
    clippy::correctness,
    clippy::suspicious,
    clippy::complexity,
    clippy::perf,
    clippy::style,
    clippy::absolute_paths,
    clippy::pedantic,
    clippy::nursery,
    reason = "generated bindings preserve upstream ABI conventions and documentation"
)]

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_cuda_sys::runtime::cudaStream_t;

#[cfg(feature = "nvcomp_6_0")]
include!("sys_6000.rs");

/// Default ANS compression options
#[cfg(feature = "nvcomp_6_0")]
pub const nvcompBatchedANSCompressDefaultOpts: nvcompBatchedANSCompressOpts_t =
    nvcompBatchedANSCompressOpts_t {
        type_: nvcompANSType_t::nvcomp_rANS,
        data_type: nvcompType_t::NVCOMP_TYPE_CHAR,
        max_sub_chunk_count: 0,
        states_per_lane: 0,
        histogram_reduction_log2: 0,
        reserved: [0; 53],
    };

/// Default ANS decompression options
#[cfg(feature = "nvcomp_6_0")]
pub const nvcompBatchedANSDecompressDefaultOpts: nvcompBatchedANSDecompressOpts_t =
    nvcompBatchedANSDecompressOpts_t {
        backend: nvcompDecompressBackend_t::NVCOMP_DECOMPRESS_BACKEND_DEFAULT,
        data_type: nvcompType_t::NVCOMP_TYPE_BITS,
        max_sub_chunk_count: 0,
        states_per_lane: 0,
        skip_validate: 0,
        reserved: [0; 53],
    };

/// Default Bitcomp compression options
#[cfg(feature = "nvcomp_6_0")]
pub const nvcompBatchedBitcompCompressDefaultOpts: nvcompBatchedBitcompCompressOpts_t =
    nvcompBatchedBitcompCompressOpts_t {
        algorithm: 0,
        data_type: nvcompType_t::NVCOMP_TYPE_UCHAR,
        delta: 0.0,
        mode: bitcompMode_t::BITCOMP_LOSSLESS,
        reserved: [0; 44],
    };

/// Default Bitcomp decompression options
#[cfg(feature = "nvcomp_6_0")]
pub const nvcompBatchedBitcompDecompressDefaultOpts: nvcompBatchedBitcompDecompressOpts_t =
    nvcompBatchedBitcompDecompressOpts_t {
        backend: nvcompDecompressBackend_t::NVCOMP_DECOMPRESS_BACKEND_DEFAULT,
        reserved: [0; 60],
    };

/// Default Cascaded compression options
#[cfg(feature = "nvcomp_6_0")]
pub const nvcompBatchedCascadedCompressDefaultOpts: nvcompBatchedCascadedCompressOpts_t =
    nvcompBatchedCascadedCompressOpts_t {
        common_opts: nvcompCascadedCommonOpts_t {
            data_type: nvcompType_t::NVCOMP_TYPE_UINT,
            mode: nvcompCascadedMode_t::NVCOMP_CASCADED_MODE_ASYMMETRIC,
            terminal_codec_flags:
                nvcompCascadedTerminalCodec_t::NVCOMP_CASCADED_TERMINAL_CODEC_CASCADED_BITPACK as _,
            coarse_grained_encoding_flags: 0,
        },
        fine_grained_encoding_flags:
            nvcompCascadedFineGrainedEncoding_t::NVCOMP_CASCADED_FINE_GRAINED_ENCODING_ALL as _,
        compression_level: 3,
        reserved: [0; 31],
    };

/// Default Cascaded decompression options
#[cfg(feature = "nvcomp_6_0")]
pub const nvcompBatchedCascadedDecompressDefaultOpts: nvcompBatchedCascadedDecompressOpts_t =
    nvcompBatchedCascadedDecompressOpts_t {
        backend: nvcompDecompressBackend_t::NVCOMP_DECOMPRESS_BACKEND_DEFAULT,
        common_opts: nvcompCascadedCommonOpts_t {
            data_type: nvcompType_t::NVCOMP_TYPE_BITS,
            mode: nvcompCascadedMode_t::NVCOMP_CASCADED_MODE_UNSPECIFIED,
            terminal_codec_flags: 0,
            coarse_grained_encoding_flags: 0,
        },
        reserved: [0; 32],
    };

/// Default Deflate compression options
#[cfg(feature = "nvcomp_6_0")]
pub const nvcompBatchedDeflateCompressDefaultOpts: nvcompBatchedDeflateCompressOpts_t =
    nvcompBatchedDeflateCompressOpts_t {
        algorithm: 1,
        reserved: [0; 60],
    };

/// Default Deflate decompression options
#[cfg(feature = "nvcomp_6_0")]
pub const nvcompBatchedDeflateDecompressDefaultOpts: nvcompBatchedDeflateDecompressOpts_t =
    nvcompBatchedDeflateDecompressOpts_t {
        backend: nvcompDecompressBackend_t::NVCOMP_DECOMPRESS_BACKEND_DEFAULT,
        sort_before_hw_decompress: 0,
        reserved: [0; 56],
    };

/// Default Gdeflate compression options
#[cfg(feature = "nvcomp_6_0")]
pub const nvcompBatchedGdeflateCompressDefaultOpts: nvcompBatchedGdeflateCompressOpts_t =
    nvcompBatchedGdeflateCompressOpts_t {
        algorithm: 1,
        reserved: [0; 60],
    };

/// Default Gdeflate decompression options
#[cfg(feature = "nvcomp_6_0")]
pub const nvcompBatchedGdeflateDecompressDefaultOpts: nvcompBatchedGdeflateDecompressOpts_t =
    nvcompBatchedGdeflateDecompressOpts_t {
        backend: nvcompDecompressBackend_t::NVCOMP_DECOMPRESS_BACKEND_DEFAULT,
        reserved: [0; 60],
    };

/// Default Gzip compression options
#[cfg(feature = "nvcomp_6_0")]
pub const nvcompBatchedGzipCompressDefaultOpts: nvcompBatchedGzipCompressOpts_t =
    nvcompBatchedGzipCompressOpts_t {
        algorithm: 1,
        reserved: [0; 60],
    };

/// Default Gzip decompression options
#[cfg(feature = "nvcomp_6_0")]
pub const nvcompBatchedGzipDecompressDefaultOpts: nvcompBatchedGzipDecompressOpts_t =
    nvcompBatchedGzipDecompressOpts_t {
        backend: nvcompDecompressBackend_t::NVCOMP_DECOMPRESS_BACKEND_DEFAULT,
        algorithm: nvcompBatchedGzipDecompressAlgorithm_t::NVCOMP_GZIP_DECOMPRESS_ALGORITHM_NAIVE,
        sort_before_hw_decompress: 0,
        reserved: [0; 52],
    };

/// Default LZ4 compression options
#[cfg(feature = "nvcomp_6_0")]
pub const nvcompBatchedLZ4CompressDefaultOpts: nvcompBatchedLZ4CompressOpts_t =
    nvcompBatchedLZ4CompressOpts_t {
        data_type: nvcompType_t::NVCOMP_TYPE_CHAR,
        bitshuffle_mode: nvcompBitshuffleMode_t::NVCOMP_BITSHUFFLE_NONE,
        reserved: [0; 56],
    };

/// Default LZ4 decompression options
#[cfg(feature = "nvcomp_6_0")]
pub const nvcompBatchedLZ4DecompressDefaultOpts: nvcompBatchedLZ4DecompressOpts_t =
    nvcompBatchedLZ4DecompressOpts_t {
        backend: nvcompDecompressBackend_t::NVCOMP_DECOMPRESS_BACKEND_DEFAULT,
        sort_before_hw_decompress: 0,
        data_type: nvcompType_t::NVCOMP_TYPE_CHAR,
        bitshuffle_mode: nvcompBitshuffleMode_t::NVCOMP_BITSHUFFLE_NONE,
        reserved: [0; 48],
    };

/// Default Snappy compression options
#[cfg(feature = "nvcomp_6_0")]
pub const nvcompBatchedSnappyCompressDefaultOpts: nvcompBatchedSnappyCompressOpts_t =
    nvcompBatchedSnappyCompressOpts_t { reserved: [0; 64] };

/// Default Snappy decompression options
#[cfg(feature = "nvcomp_6_0")]
pub const nvcompBatchedSnappyDecompressDefaultOpts: nvcompBatchedSnappyDecompressOpts_t =
    nvcompBatchedSnappyDecompressOpts_t {
        backend: nvcompDecompressBackend_t::NVCOMP_DECOMPRESS_BACKEND_DEFAULT,
        sort_before_hw_decompress: 0,
        reserved: [0; 56],
    };

/// Default Zstd compression options
#[cfg(feature = "nvcomp_6_0")]
pub const nvcompBatchedZstdCompressDefaultOpts: nvcompBatchedZstdCompressOpts_t =
    nvcompBatchedZstdCompressOpts_t { reserved: [0; 64] };

/// Default Zstd decompression options
#[cfg(feature = "nvcomp_6_0")]
pub const nvcompBatchedZstdDecompressDefaultOpts: nvcompBatchedZstdDecompressOpts_t =
    nvcompBatchedZstdDecompressOpts_t {
        backend: nvcompDecompressBackend_t::NVCOMP_DECOMPRESS_BACKEND_DEFAULT,
        reserved: [0; 60],
    };

/// Standard CRC32 (aka CRC-32/PKZIP) model preset.
#[cfg(feature = "nvcomp_6_0")]
pub const nvcompCRC32: nvcompCRC32Spec_t = nvcompCRC32Spec_t {
    poly: 0x04C11DB7,
    init: 0xFFFFFFFF,
    ref_in: true,
    ref_out: true,
    xorout: 0xFFFFFFFF,
    reserved: [0; 16],
};

/// CRC32-C (aka CRC-32/ISCSI) model preset.
#[cfg(feature = "nvcomp_6_0")]
pub const nvcompCRC32_C: nvcompCRC32Spec_t = nvcompCRC32Spec_t {
    poly: 0x1EDC6F41,
    init: 0xFFFFFFFF,
    ref_in: true,
    ref_out: true,
    xorout: 0xFFFFFFFF,
    reserved: [0; 16],
};

/// CRC32-D (aka CRC-32/BASE91-D) model preset.
#[cfg(feature = "nvcomp_6_0")]
pub const nvcompCRC32_D: nvcompCRC32Spec_t = nvcompCRC32Spec_t {
    poly: 0xA833982B,
    init: 0xFFFFFFFF,
    ref_in: true,
    ref_out: true,
    xorout: 0xFFFFFFFF,
    reserved: [0; 16],
};

/// CRC32-Q (aka CRC-32/AIXM) model preset.
#[cfg(feature = "nvcomp_6_0")]
pub const nvcompCRC32_Q: nvcompCRC32Spec_t = nvcompCRC32Spec_t {
    poly: 0x814141AB,
    init: 0,
    ref_in: false,
    ref_out: false,
    xorout: 0,
    reserved: [0; 16],
};

/// CRC-32/MEF model preset.
#[cfg(feature = "nvcomp_6_0")]
pub const nvcompCRC32_MEF: nvcompCRC32Spec_t = nvcompCRC32Spec_t {
    poly: 0x741B8CD7,
    init: 0xFFFFFFFF,
    ref_in: true,
    ref_out: true,
    xorout: 0,
    reserved: [0; 16],
};

/// CRC-32/XFER model preset.
#[cfg(feature = "nvcomp_6_0")]
pub const nvcompCRC32_XFER: nvcompCRC32Spec_t = nvcompCRC32Spec_t {
    poly: 0x000000AF,
    init: 0,
    ref_in: false,
    ref_out: false,
    xorout: 0,
    reserved: [0; 16],
};

/// CRC-32/BZIP2 (aka CRC-32/AAL-5) model preset.
#[cfg(feature = "nvcomp_6_0")]
pub const nvcompCRC32_BZIP2: nvcompCRC32Spec_t = nvcompCRC32Spec_t {
    poly: 0x04C11DB7,
    init: 0xFFFFFFFF,
    ref_in: false,
    ref_out: false,
    xorout: 0xFFFFFFFF,
    reserved: [0; 16],
};

/// CRC-32/POSIX (aka CRC-32/CKSUM) model preset.
#[cfg(feature = "nvcomp_6_0")]
pub const nvcompCRC32_POSIX: nvcompCRC32Spec_t = nvcompCRC32Spec_t {
    poly: 0x04C11DB7,
    init: 0,
    ref_in: false,
    ref_out: false,
    xorout: 0xFFFFFFFF,
    reserved: [0; 16],
};

/// CRC-32/JAMCRC model preset.
#[cfg(feature = "nvcomp_6_0")]
pub const nvcompCRC32_JAMCRC: nvcompCRC32Spec_t = nvcompCRC32Spec_t {
    poly: 0x04C11DB7,
    init: 0xFFFFFFFF,
    ref_in: true,
    ref_out: true,
    xorout: 0,
    reserved: [0; 16],
};

/// CRC-32/MPEG-2 model preset.
#[cfg(feature = "nvcomp_6_0")]
pub const nvcompCRC32_MPEG_2: nvcompCRC32Spec_t = nvcompCRC32Spec_t {
    poly: 0x04C11DB7,
    init: 0xFFFFFFFF,
    ref_in: false,
    ref_out: false,
    xorout: 0,
    reserved: [0; 16],
};

/// CRC-32/AUTOSAR model preset.
#[cfg(feature = "nvcomp_6_0")]
pub const nvcompCRC32_AUTOSAR: nvcompCRC32Spec_t = nvcompCRC32Spec_t {
    poly: 0xF4ACFB13,
    init: 0xFFFFFFFF,
    ref_in: true,
    ref_out: true,
    xorout: 0xFFFFFFFF,
    reserved: [0; 16],
};

/// CRC-32/CD-ROM-EDC model preset.
#[cfg(feature = "nvcomp_6_0")]
pub const nvcompCRC32_CD_ROM_EDC: nvcompCRC32Spec_t = nvcompCRC32Spec_t {
    poly: 0x8001801B,
    init: 0,
    ref_in: true,
    ref_out: true,
    xorout: 0,
    reserved: [0; 16],
};

/// Value to pass as `device_input_chunk_bytes` to `nvcompBatchedCRC32GetHeuristicConf` when specifying the maximum input chunk size in `max_input_chunk_bytes`.
/// Equal to a null pointer.
#[cfg(feature = "nvcomp_6_0")]
pub const nvcompCRC32IgnoredInputChunkBytes: *const size_t = std::ptr::null();

/// C ABI bridge for the nvCOMP C++ managers and streaming interface.
#[cfg(feature = "nvcomp_6_0")]
pub mod manager {
    include!("manager_sys.rs");
}

pub use singe_loader::LoadError;

static LIBRARY: singe_loader::Library = singe_loader::Library::nvcomp();

// Loads nvCOMP, which each of its functions otherwise does on first call.
pub fn load() -> Result<(), LoadError> {
    LIBRARY.load()
}

#[cfg(feature = "nvcomp_6_0")]
singe_loader::missing_status!(
    nvcompStatus_t,
    library: nvcompStatus_t::nvcompErrorInternal,
    symbol: nvcompStatus_t::nvcompErrorNotSupported,
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it_works() {
        let mut properties = nvcompProperties_t::default();
        // SAFETY: the version query writes only into the local output locations.
        unsafe {
            assert_eq!(
                nvcompGetProperties(&raw mut properties),
                nvcompStatus_t::nvcompSuccess,
            );
        }
        println!("nvCOMP version: {}", properties.version);
        assert_ne!(properties.version, 0);
    }
}

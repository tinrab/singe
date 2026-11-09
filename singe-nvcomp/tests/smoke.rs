use singe_cuda::{
    context::Context,
    memory::DeviceMemory,
    stream::Stream,
    view::{DeviceSlice, DeviceSliceMut},
};
use singe_nvcomp::{
    ans,
    batch::{DeviceChunkTable, DeviceChunkTableMut, DeviceStatuses},
    bitcomp, cascaded,
    compressed::{CompressedBatch, CompressedBatchParts, CompressedChunk, CompressedChunkParts},
    crc32::{self, Options, Spec},
    deflate,
    error::{DeviceStatus, Error, Result, Status},
    format::{
        Ans, BatchedCodec, Bitcomp, Cascaded, Codec, Deflate, Gdeflate, Gzip, Lz4, Snappy, Zstd,
    },
    gdeflate, gzip, lz4, snappy,
    testing::cpu_crc32,
    types::*,
    workspace::TemporaryBuffer,
    zstd,
};
use singe_nvcomp_sys as sys;

#[test]
fn test_generated_option_defaults_match_raw_defaults() {
    assert_eq!(
        ans::CompressOptions::default().as_raw(),
        sys::nvcompBatchedANSCompressDefaultOpts
    );
    assert_eq!(
        ans::DecompressOptions::default().as_raw(),
        sys::nvcompBatchedANSDecompressDefaultOpts
    );
    assert_eq!(
        bitcomp::CompressOptions::default().as_raw(),
        sys::nvcompBatchedBitcompCompressDefaultOpts
    );
    assert_eq!(
        bitcomp::DecompressOptions::default().as_raw(),
        sys::nvcompBatchedBitcompDecompressDefaultOpts
    );
    assert_eq!(
        cascaded::CompressOptions::default().as_raw(),
        sys::nvcompBatchedCascadedCompressDefaultOpts
    );
    assert_eq!(
        cascaded::DecompressOptions::default().as_raw(),
        sys::nvcompBatchedCascadedDecompressDefaultOpts
    );
    assert_eq!(
        deflate::CompressOptions::default().as_raw(),
        sys::nvcompBatchedDeflateCompressDefaultOpts
    );
    assert_eq!(
        deflate::DecompressOptions::default().as_raw(),
        sys::nvcompBatchedDeflateDecompressDefaultOpts
    );
    assert_eq!(
        gdeflate::CompressOptions::default().as_raw(),
        sys::nvcompBatchedGdeflateCompressDefaultOpts
    );
    assert_eq!(
        gdeflate::DecompressOptions::default().as_raw(),
        sys::nvcompBatchedGdeflateDecompressDefaultOpts
    );
    let gzip_default: sys::nvcompBatchedGzipDecompressOpts_t =
        gzip::DecompressOptions::default().into();
    assert_eq!(gzip_default, sys::nvcompBatchedGzipDecompressDefaultOpts);
    let gzip_compress_default: sys::nvcompBatchedGzipCompressOpts_t =
        gzip::CompressOptions::default().into();
    assert_eq!(
        gzip_compress_default,
        sys::nvcompBatchedGzipCompressDefaultOpts
    );
    assert_eq!(
        snappy::CompressOptions.as_raw(),
        sys::nvcompBatchedSnappyCompressDefaultOpts
    );
    assert_eq!(
        snappy::DecompressOptions::default().as_raw(),
        sys::nvcompBatchedSnappyDecompressDefaultOpts
    );
    assert_eq!(
        zstd::CompressOptions.as_raw(),
        sys::nvcompBatchedZstdCompressDefaultOpts
    );
    assert_eq!(
        zstd::DecompressOptions::default().as_raw(),
        sys::nvcompBatchedZstdDecompressDefaultOpts
    );
}

#[test]
fn test_typed_option_setters_preserve_reserved_zeroes() {
    let ans = ans::CompressOptions::new(AnsType::Rans, DataType::F16).as_raw();
    assert_eq!(ans.type_, sys::nvcompANSType_t::nvcomp_rANS);
    assert_eq!(ans.data_type, sys::nvcompType_t::NVCOMP_TYPE_FLOAT16);
    assert_eq!(ans.max_sub_chunk_count, 0);
    assert_eq!(ans.reserved, [0; 53]);

    let bitcomp = bitcomp::CompressOptions::new(BitcompAlgorithm::Sparse, DataType::U32).as_raw();
    assert_eq!(bitcomp.algorithm, 1);
    assert_eq!(bitcomp.data_type, sys::nvcompType_t::NVCOMP_TYPE_UINT);
    assert_eq!(bitcomp.reserved, [0; 44]);

    let cascaded = cascaded::CompressOptions::new(DataType::I64).as_raw();
    assert_eq!(
        cascaded.common_opts.data_type,
        sys::nvcompType_t::NVCOMP_TYPE_LONGLONG
    );
    assert_eq!(
        cascaded.common_opts.mode,
        sys::nvcompCascadedMode_t::NVCOMP_CASCADED_MODE_ASYMMETRIC
    );
    assert_eq!(cascaded.compression_level, 3);
    assert_eq!(cascaded.reserved, [0; 31]);

    let deflate = deflate::CompressOptions::new(CompressionAlgorithm::HighCompression).as_raw();
    assert_eq!(deflate.algorithm, 4);
    assert_eq!(deflate.reserved, [0; 60]);

    let gdeflate =
        gdeflate::CompressOptions::new(CompressionAlgorithm::HighestCompression).as_raw();
    assert_eq!(gdeflate.algorithm, 5);
    assert_eq!(gdeflate.reserved, [0; 60]);

    let snappy = snappy::DecompressOptions::new(DecompressBackend::Hardware, true).as_raw();
    assert_eq!(
        snappy.backend,
        sys::nvcompDecompressBackend_t::NVCOMP_DECOMPRESS_BACKEND_HARDWARE
    );
    assert_eq!(snappy.sort_before_hw_decompress, 1);
    assert_eq!(snappy.reserved, [0; 56]);

    let gzip = sys::nvcompBatchedGzipDecompressOpts_t::from(gzip::DecompressOptions {
        backend: DecompressBackend::Hardware,
        algorithm: GzipDecompressAlgorithm::Lookahead,
        sort_before_hw_decompress: true,
    });
    assert_eq!(
        gzip.backend,
        sys::nvcompDecompressBackend_t::NVCOMP_DECOMPRESS_BACKEND_HARDWARE
    );
    assert_eq!(
        gzip.algorithm,
        sys::nvcompBatchedGzipDecompressAlgorithm_t::NVCOMP_GZIP_DECOMPRESS_ALGORITHM_LOOKAHEAD
    );
    assert_eq!(gzip.sort_before_hw_decompress, 1);
    assert_eq!(gzip.reserved, [0; 52]);

    let zstd = zstd::DecompressOptions::new(DecompressBackend::Cuda).as_raw();
    assert_eq!(
        zstd.backend,
        sys::nvcompDecompressBackend_t::NVCOMP_DECOMPRESS_BACKEND_CUDA
    );
    assert_eq!(zstd.reserved, [0; 60]);
}

#[test]
fn test_raw_option_imports_reject_invalid_values() {
    let mut invalid_algorithm = sys::nvcompBatchedBitcompCompressDefaultOpts;
    invalid_algorithm.algorithm = 99;
    assert!(matches!(
        bitcomp::CompressOptions::try_from(invalid_algorithm),
        Err(Error::InvalidValue { name }) if name == "algorithm"
    ));

    let mut nonzero_reserved = sys::nvcompBatchedANSCompressDefaultOpts;
    nonzero_reserved.reserved[0] = 1;
    assert!(matches!(
        ans::CompressOptions::try_from(nonzero_reserved),
        Err(Error::InvalidValue { name }) if name == "reserved option bytes"
    ));
}

#[test]
fn test_typed_enums_display_c_constant_names() {
    assert_eq!(DataType::I8.to_string(), "NVCOMP_TYPE_CHAR");
    assert_eq!(DataType::U64.to_string(), "NVCOMP_TYPE_ULONGLONG");
    assert_eq!(DataType::F8E4M3.to_string(), "NVCOMP_TYPE_FLOAT8_E4M3");
    assert_eq!(
        sys::nvcompType_t::from(DataType::U32),
        sys::nvcompType_t::NVCOMP_TYPE_UINT
    );
}

#[test]
fn test_crc32_kernel_conf_checked_constructor_preserves_raw_layout() -> Result<()> {
    let config = crc32::KernelConf::create(Crc32KernelKind::Warp, 16, 2)?;
    let raw = config.as_raw();

    assert_eq!(config.kernel_kind(), Crc32KernelKind::Warp);
    assert_eq!(config.bytes_per_read(), 16);
    assert_eq!(config.blocks_per_msg(), 2);
    assert_eq!(
        raw.kernel_kind,
        sys::nvcompCRC32KernelKind_t::nvcompCRC32WarpKernel
    );
    assert_eq!(raw.bytes_per_read, 16);
    assert_eq!(raw.blocks_per_msg, 2);
    assert_eq!(raw.reserved, [0; 20]);

    assert!(matches!(
        crc32::KernelConf::create(Crc32KernelKind::Block, i32::MAX as usize + 1, 1),
        Err(Error::OutOfRange { .. })
    ));

    Ok(())
}

#[test]
fn test_device_status_exposes_fallible_raw_and_safe_status() {
    let success = DeviceStatus::from_raw(Status::Success.raw());

    assert_eq!(success.status(), Status::Success);
    assert_eq!(
        success.try_as_raw(),
        Some(sys::nvcompStatus_t::nvcompSuccess)
    );

    let unknown = DeviceStatus::from_raw(u32::MAX);
    assert_eq!(unknown.status(), Status::Unknown(u32::MAX));
    assert_eq!(unknown.try_as_raw(), None);
}

#[test]
fn test_codec_markers_expose_names() {
    assert_eq!(Lz4::NAME, "LZ4");
    assert_eq!(Gzip::NAME, "Gzip");
}

#[test]
fn test_codec_handle_owns_options_and_delegates_validation() {
    let compress_options = lz4::CompressOptions {
        data_type: DataType::U8,
        bitshuffle_mode: BitshuffleMode::MsbFirst,
    };
    let decompress_options = lz4::DecompressOptions {
        data_type: DataType::U8,
        bitshuffle_mode: BitshuffleMode::MsbFirst,
        ..Default::default()
    };
    let codec = Codec::<Lz4>::new(compress_options, decompress_options);

    assert_eq!(codec.compress_options(), compress_options);
    assert_eq!(codec.decompress_options(), decompress_options);

    let actual = one_past(lz4::COMPRESSION_MAX_ALLOWED_CHUNK_SIZE);
    assert_size_limit(
        codec.max_output_chunk_size(actual).unwrap_err(),
        lz4::COMPRESSION_MAX_ALLOWED_CHUNK_SIZE,
        actual,
    );
}

#[test]
fn test_codec_derives_format_metadata_for_decoding() {
    let compress = lz4::CompressOptions {
        data_type: DataType::U32,
        bitshuffle_mode: BitshuffleMode::MsbFirst,
    };
    let configured = lz4::DecompressOptions {
        backend: DecompressBackend::Cuda,
        data_type: DataType::I8,
        bitshuffle_mode: BitshuffleMode::None,
        ..Default::default()
    };

    let decoding = <Lz4 as BatchedCodec>::decoding_options(compress, configured);
    assert_eq!(decoding.backend, DecompressBackend::Cuda);
    assert_eq!(decoding.data_type, compress.data_type);
    assert_eq!(decoding.bitshuffle_mode, compress.bitshuffle_mode);
}

#[test]
fn test_compressed_chunk_reports_lengths() -> Result<()> {
    let allocation_context = Context::create()?;
    let memory = DeviceMemory::<u8>::zeroed(&allocation_context, 64)?;
    // SAFETY: the chunk is only inspected, never decompressed, so the zeroed bytes need not be valid LZ4 data.
    let chunk = unsafe {
        CompressedChunk::<Lz4>::create(
            memory.into_uninit(),
            17,
            42,
            lz4::DecompressOptions::default(),
        )?
    };

    assert_eq!(chunk.len(), 17);
    assert!(!chunk.is_empty());
    assert_eq!(chunk.uncompressed_len(), 42);
    assert_eq!(chunk.capacity(), 64);
    assert_eq!(chunk.memory().len(), 17);

    Ok(())
}

#[test]
fn test_compressed_chunk_rejects_oversized_valid_length() -> Result<()> {
    let allocation_context = Context::create()?;
    let memory = DeviceMemory::<u8>::zeroed(&allocation_context, 64)?;
    // SAFETY: construction rejects the oversized length before adopting the memory.
    let error = unsafe {
        CompressedChunk::<Lz4>::create(
            memory.into_uninit(),
            65,
            42,
            lz4::DecompressOptions::default(),
        )
        .unwrap_err()
    };

    assert_length_mismatch(error, "compressed chunk bytes", 65, 64);

    Ok(())
}

#[test]
fn test_compressed_batch_reports_metadata() -> Result<()> {
    let allocation_context = Context::create()?;
    let memory = DeviceMemory::<u8>::zeroed(&allocation_context, 32)?;
    // SAFETY: the batch is only inspected, never decompressed, so the zeroed bytes need not be valid LZ4 data.
    let batch = unsafe {
        CompressedBatch::<Lz4>::create(CompressedBatchParts {
            memory: memory.into_uninit(),
            stride_bytes: 8,
            chunk_bytes: vec![3, 5, 1],
            uncompressed_chunk_bytes: vec![4, 4, 2],
            decoding_options: lz4::DecompressOptions::default(),
        })?
    };

    assert_eq!(batch.len(), 3);
    assert!(!batch.is_empty());
    assert_eq!(batch.stride_bytes(), 8);
    assert_eq!(batch.chunk_bytes(), [3, 5, 1]);
    assert_eq!(batch.uncompressed_chunk_bytes(), [4, 4, 2]);
    assert_eq!(batch.total_compressed_len(), 9);
    assert_eq!(batch.total_uncompressed_len(), 10);
    assert_eq!(batch.memory().len(), 32);

    Ok(())
}

#[test]
fn test_compressed_batch_rejects_mismatched_metadata_lengths() -> Result<()> {
    let allocation_context = Context::create()?;
    let memory = DeviceMemory::<u8>::zeroed(&allocation_context, 32)?;
    // SAFETY: construction rejects the mismatched length lists before adopting the memory.
    let error = unsafe {
        CompressedBatch::<Lz4>::create(CompressedBatchParts {
            memory: memory.into_uninit(),
            stride_bytes: 8,
            chunk_bytes: vec![3, 5],
            uncompressed_chunk_bytes: vec![4],
            decoding_options: lz4::DecompressOptions::default(),
        })
        .unwrap_err()
    };

    assert_length_mismatch(error, "uncompressed chunk bytes", 2, 1);

    Ok(())
}

#[test]
fn test_compressed_batch_rejects_nonempty_zero_stride() -> Result<()> {
    let allocation_context = Context::create()?;
    let memory = DeviceMemory::<u8>::zeroed(&allocation_context, 0)?;
    // SAFETY: construction rejects the zero stride before adopting the memory.
    let error = unsafe {
        CompressedBatch::<Lz4>::create(CompressedBatchParts {
            memory: memory.into_uninit(),
            stride_bytes: 0,
            chunk_bytes: vec![0],
            uncompressed_chunk_bytes: vec![0],
            decoding_options: lz4::DecompressOptions::default(),
        })
        .unwrap_err()
    };

    assert!(matches!(
        error,
        Error::InvalidValue { name } if name == "stride bytes"
    ));

    Ok(())
}

#[test]
fn test_compressed_batch_rejects_chunk_larger_than_stride() -> Result<()> {
    let allocation_context = Context::create()?;
    let memory = DeviceMemory::<u8>::zeroed(&allocation_context, 16)?;
    // SAFETY: construction rejects the chunk that exceeds its stride before adopting the memory.
    let error = unsafe {
        CompressedBatch::<Lz4>::create(CompressedBatchParts {
            memory: memory.into_uninit(),
            stride_bytes: 8,
            chunk_bytes: vec![9],
            uncompressed_chunk_bytes: vec![4],
            decoding_options: lz4::DecompressOptions::default(),
        })
        .unwrap_err()
    };

    assert_length_mismatch(error, "compressed chunk bytes", 8, 9);

    Ok(())
}

#[test]
fn test_oversize_chunk_queries_fail_before_nvcomp() {
    let cuda = Context::create().unwrap();
    let stream = cuda.create_stream().unwrap();
    let lz4_actual = one_past(lz4::COMPRESSION_MAX_ALLOWED_CHUNK_SIZE);
    assert_size_limit(
        Lz4::max_output_chunk_size(lz4_actual, lz4::CompressOptions::default()).unwrap_err(),
        lz4::COMPRESSION_MAX_ALLOWED_CHUNK_SIZE,
        lz4_actual,
    );

    let zstd_actual = one_past(zstd::COMPRESSION_MAX_ALLOWED_CHUNK_SIZE);
    assert_size_limit(
        Zstd::max_output_chunk_size(zstd_actual, zstd::CompressOptions).unwrap_err(),
        zstd::COMPRESSION_MAX_ALLOWED_CHUNK_SIZE,
        zstd_actual,
    );

    let gzip_naive_actual = one_past(gzip::NAIVE_DECOMPRESSION_MAX_ALLOWED_CHUNK_SIZE);
    assert_size_limit(
        Gzip::decompression_temporary_size(
            1,
            gzip_naive_actual,
            gzip::DecompressOptions::default(),
            gzip_naive_actual,
            &stream,
        )
        .unwrap_err(),
        gzip::NAIVE_DECOMPRESSION_MAX_ALLOWED_CHUNK_SIZE,
        gzip_naive_actual,
    );
}

#[test]
fn test_checked_chunk_apis_reject_mismatched_lengths_before_nvcomp() -> Result<()> {
    let cuda = Context::create()?;
    let stream = cuda.create_stream()?;

    let input_buffers: [&dyn DeviceSlice<u8>; 0] = [];
    let input_chunks = DeviceChunkTable::from_device_slices(&cuda, &input_buffers)?;
    let mut output_memory = DeviceMemory::<u8>::create_uninit(stream.context(), 16)?;
    let mut output_chunks = DeviceChunkTableMut::from_contiguous(&cuda, &mut output_memory)?;
    let mut temp = TemporaryBuffer::create(&cuda, 0)?;
    let mut statuses = DeviceStatuses::create(&cuda, input_chunks.len())?;

    assert_length_mismatch(
        // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
        unsafe {
            Lz4::compress_chunks_async(
                &input_chunks,
                &mut output_chunks,
                &mut temp,
                &mut statuses,
                lz4::CompressOptions::default(),
                &stream,
            )
        }
        .unwrap_err(),
        "output chunks",
        0,
        1,
    );

    assert_length_mismatch(
        // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
        unsafe {
            Zstd::compress_chunks_async(
                &input_chunks,
                &mut output_chunks,
                &mut temp,
                &mut statuses,
                zstd::CompressOptions,
                &stream,
            )
        }
        .unwrap_err(),
        "output chunks",
        0,
        1,
    );

    let mut queried_decompressed_sizes = DeviceMemory::<usize>::zeroed(stream.context(), 1)?;
    assert_length_mismatch(
        stream
            .sync_scope(|_| {
                // SAFETY: the enclosing draining scope keeps the chunk table and output alive until the work completes.
                Ok(unsafe {
                    Lz4::decompress_size_chunks_async(
                        input_chunks.view(),
                        &mut queried_decompressed_sizes,
                        &stream,
                    )
                })
            })?
            .unwrap_err(),
        "uncompressed chunk bytes",
        0,
        1,
    );

    assert_length_mismatch(
        // SAFETY: validation rejects the mismatched lengths before any work is enqueued.
        unsafe {
            Gzip::decompress_chunks_async(
                input_chunks.view(),
                &mut output_chunks,
                &mut temp,
                &mut statuses,
                gzip::DecompressOptions::default(),
                &stream,
            )
        }
        .unwrap_err(),
        "output chunks",
        0,
        1,
    );

    Ok(())
}

#[test]
fn test_checked_compression_rejects_smallest_output_chunk_capacity() -> Result<()> {
    let cuda = Context::create()?;
    let stream = cuda.create_stream()?;

    let input_a = DeviceMemory::<u8>::zeroed(stream.context(), 16)?;
    let input_b = DeviceMemory::<u8>::zeroed(stream.context(), 16)?;
    let input_buffers: [&dyn DeviceSlice<u8>; 2] = [&input_a, &input_b];
    let input_chunks = DeviceChunkTable::from_device_slices(&cuda, &input_buffers)?;
    let required_output_bytes =
        Zstd::max_output_chunk_size(input_chunks.max_chunk_bytes(), zstd::CompressOptions)?;

    let mut small_output = DeviceMemory::<u8>::zeroed(stream.context(), required_output_bytes - 1)?;
    let mut large_output = DeviceMemory::<u8>::zeroed(stream.context(), required_output_bytes)?;
    let mut output_buffers: [&mut dyn DeviceSliceMut<u8>; 2] =
        [&mut small_output, &mut large_output];
    let mut output_chunks = DeviceChunkTableMut::from_device_slices(&cuda, &mut output_buffers)?;
    let mut temp = TemporaryBuffer::create(&cuda, 0)?;
    let mut statuses = DeviceStatuses::create(&cuda, input_chunks.len())?;

    assert_length_mismatch(
        // SAFETY: validation rejects the mismatched lengths before any work is enqueued.
        unsafe {
            Zstd::compress_chunks_async(
                &input_chunks,
                &mut output_chunks,
                &mut temp,
                &mut statuses,
                zstd::CompressOptions,
                &stream,
            )
        }
        .unwrap_err(),
        "output chunk capacity",
        required_output_bytes,
        required_output_bytes - 1,
    );

    Ok(())
}

#[test]
fn test_standard_codecs_roundtrip_checked_batched_api() -> Result<()> {
    let cuda = Context::create()?;
    let stream = cuda.create_stream()?;
    let input = b"nvCOMP checked batched roundtrip through Singe tests. ".repeat(256);

    roundtrip_codec::<Ans>(&stream, &input)?;
    roundtrip_codec::<Bitcomp>(&stream, &input)?;
    roundtrip_codec::<Cascaded>(&stream, &input)?;
    roundtrip_codec::<Deflate>(&stream, &input)?;
    roundtrip_codec::<Gdeflate>(&stream, &input)?;
    roundtrip_codec::<Gzip>(&stream, &input)?;
    roundtrip_codec::<Lz4>(&stream, &input)?;
    roundtrip_codec::<Snappy>(&stream, &input)?;
    roundtrip_codec::<Zstd>(&stream, &input)?;

    Ok(())
}

#[test]
fn test_gzip_decompression_algorithms_decode_known_fixture() -> Result<()> {
    let cuda = Context::create()?;
    let stream = cuda.create_stream()?;

    gzip_decompress_fixture(
        &stream,
        gzip::DecompressOptions {
            algorithm: GzipDecompressAlgorithm::Naive,
            ..Default::default()
        },
    )?;
    gzip_decompress_fixture(
        &stream,
        gzip::DecompressOptions {
            algorithm: GzipDecompressAlgorithm::Lookahead,
            ..Default::default()
        },
    )?;

    Ok(())
}

#[test]
fn test_codec_decompresses_rewrapped_compressed_chunk() -> Result<()> {
    let cuda = Context::create()?;
    let stream = cuda.create_stream()?;
    let input = b"nvCOMP rewrapped compressed chunk through Singe tests. ".repeat(256);
    let device_input = DeviceMemory::from_slice(stream.context(), &input)?;
    let codec = Codec::<Lz4>::default();

    let compressed = codec.compress_one(&device_input, &stream)?;
    let CompressedChunkParts {
        memory,
        uncompressed_bytes,
        decoding_options: options,
    } = compressed.into_parts();
    let compressed_bytes = memory.len();
    // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
    let compressed = unsafe {
        CompressedChunk::<Lz4>::create(
            memory.into_uninit(),
            compressed_bytes,
            uncompressed_bytes,
            options,
        )?
    };

    let mut output = DeviceMemory::<u8>::create_uninit(stream.context(), input.len())?;
    let output_bytes = codec.decompress_one(&compressed, &mut output, &stream)?;
    assert_eq!(output_bytes, input.len());
    // SAFETY: the completed helper produced exactly the reported prefix.
    let output = unsafe { output.assume_init_prefix(output_bytes)? };
    assert_eq!(output.copy_to_host_vec()?, input);

    Ok(())
}

#[test]
fn test_codec_decompresses_strided_compressed_chunk_view() -> Result<()> {
    let cuda = Context::create()?;
    let stream = cuda.create_stream()?;
    let input = b"nvCOMP strided compressed chunk through Singe tests. ".repeat(256);
    let device_input = DeviceMemory::from_slice(stream.context(), &input)?;
    let codec = Codec::<Lz4>::default();

    let compressed = codec.compress_one(&device_input, &stream)?;
    let compressed_bytes = compressed.len();
    let compressed_stride = compressed.capacity();
    let compressed_chunks = DeviceChunkTable::from_contiguous_strided_chunks(
        &cuda,
        compressed.memory(),
        compressed_stride,
        &[compressed_bytes],
    )?;
    let mut output = DeviceMemory::<u8>::create_uninit(stream.context(), input.len())?;
    let mut output_chunks = DeviceChunkTableMut::from_contiguous(&cuda, &mut output)?;

    // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        codec.decompress_chunks_alloc_async(
            compressed_chunks.view(),
            &mut output_chunks,
            &stream,
        )?
    }
    .check()?;

    // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
    let output_sizes = unsafe { output_chunks.copy_written_bytes(&stream) }?;
    assert_eq!(output_sizes[0], input.len());
    drop(output_chunks);
    // SAFETY: the checked raw producer completed this reported prefix.
    let output = unsafe { output.assume_init_prefix(output_sizes[0])? };
    assert_eq!(output.copy_to_host_vec()?, input);

    Ok(())
}

#[test]
fn test_codec_roundtrips_chunked_contiguous_batch() -> Result<()> {
    let cuda = Context::create()?;
    let stream = cuda.create_stream()?;
    let input = b"nvCOMP chunked contiguous batch through Singe tests. ".repeat(512);
    let device_input = DeviceMemory::from_slice(stream.context(), &input)?;
    let codec = Codec::<Lz4>::default();

    let compressed = codec.compress_contiguous(&device_input, 1024, &stream)?;
    assert!(compressed.len() > 1);
    assert_eq!(compressed.total_uncompressed_len(), input.len());

    let mut output = DeviceMemory::<u8>::create_uninit(stream.context(), input.len())?;
    let output_bytes = codec.decompress_contiguous(&compressed, &mut output, &stream)?;
    assert_eq!(output_bytes, input.len());
    // SAFETY: the completed helper produced exactly the reported prefix.
    let output = unsafe { output.assume_init_prefix(output_bytes)? };
    assert_eq!(output.copy_to_host_vec()?, input);

    Ok(())
}

#[test]
fn test_codec_reports_chunked_batch_wrong_uncompressed_chunk_length() -> Result<()> {
    let cuda = Context::create()?;
    let stream = cuda.create_stream()?;
    let input = b"nvCOMP chunked batch wrong metadata through Singe tests. ".repeat(512);
    let device_input = DeviceMemory::from_slice(stream.context(), &input)?;
    let codec = Codec::<Lz4>::default();

    let compressed = codec.compress_contiguous(&device_input, 1024, &stream)?;
    let mut parts = compressed.into_parts();
    let expected = parts.uncompressed_chunk_bytes[1] + 1;
    parts.uncompressed_chunk_bytes[1] = expected;
    // SAFETY: the bytes come from a completed compression; the uncompressed size is wrong on purpose, to check that decompression reports the mismatch.
    let compressed = unsafe { CompressedBatch::<Lz4>::create(parts)? };

    let mut output =
        DeviceMemory::<u8>::create_uninit(stream.context(), compressed.total_uncompressed_len())?;
    let error = codec
        .decompress_contiguous(&compressed, &mut output, &stream)
        .unwrap_err();

    assert!(matches!(
        error,
        Error::ChunkLengthMismatch {
            index: 1,
            name,
            expected: actual_expected,
            actual,
        } if name == "decompressed chunk bytes"
            && actual_expected == expected
            && actual + 1 == expected
    ));

    Ok(())
}

#[test]
fn test_codec_rejects_rewrapped_chunk_with_wrong_uncompressed_length() -> Result<()> {
    let cuda = Context::create()?;
    let stream = cuda.create_stream()?;
    let input = b"nvCOMP wrong chunk metadata through Singe tests. ".repeat(256);
    let device_input = DeviceMemory::from_slice(stream.context(), &input)?;
    let codec = Codec::<Lz4>::default();

    let compressed = codec.compress_one(&device_input, &stream)?;
    let CompressedChunkParts {
        memory,
        decoding_options: options,
        ..
    } = compressed.into_parts();
    let compressed_bytes = memory.len();
    let wrong_uncompressed_bytes = input.len() + 1;
    // SAFETY: the bytes come from a completed compression; the uncompressed size is wrong on purpose, to check that decompression reports the mismatch.
    let compressed = unsafe {
        CompressedChunk::<Lz4>::create(
            memory.into_uninit(),
            compressed_bytes,
            wrong_uncompressed_bytes,
            options,
        )?
    };

    let mut output = DeviceMemory::<u8>::create_uninit(stream.context(), wrong_uncompressed_bytes)?;
    let error = codec
        .decompress_one(&compressed, &mut output, &stream)
        .unwrap_err();
    assert!(matches!(
        error,
        Error::LengthMismatch {
            name,
            expected,
            actual,
        } if name == "decompressed chunk bytes"
            && expected == wrong_uncompressed_bytes
            && actual == input.len()
    ));

    Ok(())
}

#[test]
fn test_codec_one_shot_empty_chunk_is_noop() -> Result<()> {
    let cuda = Context::create()?;
    let stream = cuda.create_stream()?;
    let input = DeviceMemory::<u8>::zeroed(stream.context(), 0)?;
    let codec = Codec::<Lz4>::default();

    let compressed = codec.compress_one(&input, &stream)?;
    assert!(compressed.is_empty());
    assert_eq!(compressed.uncompressed_len(), 0);
    assert_eq!(compressed.capacity(), 0);

    let mut output = DeviceMemory::<u8>::create_uninit(stream.context(), 0)?;
    let output_bytes = codec.decompress_one(&compressed, &mut output, &stream)?;
    assert_eq!(output_bytes, 0);

    Ok(())
}

#[test]
fn test_codec_rejects_empty_compressed_chunk_with_nonzero_output_metadata() -> Result<()> {
    let cuda = Context::create()?;
    let stream = cuda.create_stream()?;
    let codec = Codec::<Lz4>::default();
    // SAFETY: an empty chunk has no bytes to initialize.
    let compressed = unsafe {
        CompressedChunk::<Lz4>::create(
            DeviceMemory::<u8>::create_uninit(stream.context(), 0)?,
            0,
            1,
            lz4::DecompressOptions::default(),
        )?
    };
    let mut output = DeviceMemory::<u8>::create_uninit(stream.context(), 1)?;

    let error = codec
        .decompress_one(&compressed, &mut output, &stream)
        .unwrap_err();
    assert!(matches!(
        error,
        Error::InvalidValue { name } if name == "compressed chunk bytes"
    ));

    Ok(())
}

#[test]
fn test_checked_zstd_compress_rejects_misaligned_input() -> Result<()> {
    let cuda = Context::create()?;
    let stream = cuda.create_stream()?;
    let input = DeviceMemory::<u8>::zeroed(stream.context(), 256)?;
    let misaligned_input = input.view().slice(1..)?;
    let input_chunks = DeviceChunkTable::from_contiguous(&cuda, &misaligned_input)?;
    let mut output = DeviceMemory::<u8>::create_uninit(stream.context(), 1024)?;
    let mut output_chunks = DeviceChunkTableMut::from_contiguous(&cuda, &mut output)?;
    let mut temp = TemporaryBuffer::create(&cuda, 0)?;
    let mut statuses = DeviceStatuses::create(&cuda, input_chunks.len())?;
    let options = zstd::CompressOptions;

    // SAFETY: validation rejects the arguments before any work is enqueued.
    let error = unsafe {
        Zstd::compress_chunks_async(
            &input_chunks,
            &mut output_chunks,
            &mut temp,
            &mut statuses,
            options,
            &stream,
        )
    }
    .unwrap_err();

    let expected_alignment = Zstd::compression_alignments(options)?.input;
    assert!(expected_alignment > 1);
    assert!(matches!(
        error,
        Error::AlignmentMismatch { name, alignment }
            if name == "input chunks" && alignment == expected_alignment
    ));

    Ok(())
}

#[test]
fn test_codec_temp_size_queries_accept_sync_and_async_paths() -> Result<()> {
    let cuda = Context::create()?;
    let stream = cuda.create_stream()?;
    let input = b"nvCOMP temp-size query coverage through Singe tests. ".repeat(256);
    let device_input = DeviceMemory::from_slice(stream.context(), &input)?;
    let input_chunks = DeviceChunkTable::from_contiguous(&cuda, &device_input)?;
    let codec = Codec::<Lz4>::default();

    let async_compress_temp_bytes =
        codec.compression_temporary_size(input_chunks.len(), input.len(), input.len(), &stream)?;
    // SAFETY: the synchronous size query completes before returning, while the test keeps the chunk table alive.
    let sync_compress_temp_bytes =
        unsafe { codec.compression_temporary_size_sync(&input_chunks, &stream) }?;
    TemporaryBuffer::create(&cuda, async_compress_temp_bytes)?
        .require(async_compress_temp_bytes)?;
    TemporaryBuffer::create(&cuda, sync_compress_temp_bytes)?.require(sync_compress_temp_bytes)?;

    let compressed = codec.compress_one(&device_input, &stream)?;
    let compressed_chunks =
        DeviceChunkTable::from_contiguous_prefix(&cuda, compressed.memory(), compressed.len())?;
    let mut output = DeviceMemory::<u8>::create_uninit(stream.context(), input.len())?;
    let output_chunks = DeviceChunkTableMut::from_contiguous(&cuda, &mut output)?;
    let mut statuses = DeviceStatuses::create(&cuda, compressed_chunks.len())?;

    let async_decompress_temp_bytes = codec.decompression_temporary_size(
        compressed_chunks.len(),
        input.len(),
        input.len(),
        &stream,
    )?;
    // SAFETY: the size query only reads chunk metadata, and the test keeps the chunks alive past the stream work.
    let sync_decompress_temp_bytes = unsafe {
        codec.decompression_temporary_size_for_output_sync(
            compressed_chunks.view(),
            &output_chunks,
            &mut statuses,
            &stream,
        )
    }?;
    TemporaryBuffer::create(&cuda, async_decompress_temp_bytes)?
        .require(async_decompress_temp_bytes)?;
    TemporaryBuffer::create(&cuda, sync_decompress_temp_bytes)?
        .require(sync_decompress_temp_bytes)?;

    Ok(())
}

fn roundtrip_codec<C>(stream: &Stream, input: &[u8]) -> Result<()>
where
    C: BatchedCodec,
{
    let output = roundtrip_with_codec::<C>(stream, input)?;
    assert_eq!(output, input);
    Ok(())
}

fn roundtrip_with_codec<C>(stream: &Stream, input: &[u8]) -> Result<Vec<u8>>
where
    C: BatchedCodec,
{
    let device_input = DeviceMemory::from_slice(stream.context(), input)?;
    let input_chunks = DeviceChunkTable::from_contiguous(stream.context(), &device_input)?;
    let codec = Codec::<C>::default();

    let max_compressed_chunk_bytes = codec.max_output_chunk_size(input.len())?;

    let mut compressed =
        DeviceMemory::<u8>::create_uninit(stream.context(), max_compressed_chunk_bytes)?;
    let mut compressed_chunks =
        DeviceChunkTableMut::from_contiguous(stream.context(), &mut compressed)?;

    // SAFETY: initialized input, codec-sized output slots, and all table owners
    // remain live through check(); the owning result drains native errors too.
    unsafe { codec.compress_chunks_alloc_async(&input_chunks, &mut compressed_chunks, stream)? }
        .check()?;

    let mut queried_decompressed_sizes =
        DeviceMemory::<usize>::zeroed(stream.context(), input_chunks.len())?;
    stream.sync_scope(|_| {
        // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
        Ok(unsafe {
            codec.decompress_size_chunks_async(
                compressed_chunks.written_chunks(),
                &mut queried_decompressed_sizes,
                stream,
            )
        })
    })??;

    let mut output = DeviceMemory::<u8>::create_uninit(stream.context(), input.len())?;
    let mut output_chunks = DeviceChunkTableMut::from_contiguous(stream.context(), &mut output)?;

    // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        codec.decompress_chunks_alloc_async(
            compressed_chunks.written_chunks(),
            &mut output_chunks,
            stream,
        )?
    }
    .check()?;

    let queried_decompressed_sizes = queried_decompressed_sizes.copy_to_host_vec()?;
    // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
    let output_sizes = unsafe { output_chunks.copy_written_bytes(stream) }?;
    assert_eq!(queried_decompressed_sizes[0], input.len());
    assert_eq!(output_sizes[0], input.len());

    drop(output_chunks);

    // SAFETY: checked decompression completed the prefix reported above.
    let output = unsafe { output.assume_init_prefix(output_sizes[0])? };
    output.copy_to_host_vec().map_err(Into::into)
}

fn gzip_decompress_fixture(stream: &Stream, options: gzip::DecompressOptions) -> Result<()> {
    let expected = gzip_fixture_bytes();
    let compressed = DeviceMemory::from_slice(stream.context(), GZIP_COMPRESSED_FIXTURE)?;
    let input_chunks = DeviceChunkTable::from_contiguous(stream.context(), &compressed)?;

    let mut queried_decompressed_sizes =
        DeviceMemory::<usize>::zeroed(stream.context(), input_chunks.len())?;
    stream.sync_scope(|_| {
        // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
        Ok(unsafe {
            Gzip::decompress_size_chunks_async(
                input_chunks.view(),
                &mut queried_decompressed_sizes,
                stream,
            )
        })
    })??;

    let mut output = DeviceMemory::<u8>::create_uninit(stream.context(), expected.len())?;
    let mut output_chunks = DeviceChunkTableMut::from_contiguous(stream.context(), &mut output)?;

    // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        gzip::decompress_chunks_alloc_async(
            input_chunks.view(),
            &mut output_chunks,
            options,
            stream,
        )?
    }
    .check()?;

    let queried_decompressed_sizes = queried_decompressed_sizes.copy_to_host_vec()?;
    // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
    let output_sizes = unsafe { output_chunks.copy_written_bytes(stream) }?;
    assert_eq!(queried_decompressed_sizes[0], expected.len());
    assert_eq!(output_sizes[0], expected.len());

    drop(output_chunks);

    // SAFETY: checked decompression completed the reported logical output.
    let output = unsafe { output.assume_init_prefix(output_sizes[0])? };
    assert_eq!(output.copy_to_host_vec()?, expected);

    Ok(())
}

#[test]
fn test_crc32_single_segment_matches_cpu() -> Result<()> {
    let cuda = Context::create()?;
    let stream = cuda.create_stream()?;
    let input = b"nvCOMP CRC32 single segment through Singe tests. ".repeat(512);
    let spec = Spec::CRC32;

    let actual = device_crc32_segment(&stream, spec, Crc32SegmentKind::Only, &input)?;
    assert_eq!(actual, cpu_crc32(spec, &input));

    Ok(())
}

#[test]
fn test_crc32_heuristic_conf_from_device_sizes_matches_cpu() -> Result<()> {
    let cuda = Context::create()?;
    let stream = cuda.create_stream()?;
    let input = b"nvCOMP CRC32 device-size heuristic through Singe tests. ".repeat(512);
    let spec = Spec::CRC32_C;

    let device_input = DeviceMemory::from_slice(stream.context(), &input)?;
    let input_chunks = DeviceChunkTable::from_contiguous(&cuda, &device_input)?;
    let mut crc32_values = DeviceMemory::<u32>::zeroed(stream.context(), input_chunks.len())?;
    let mut statuses = DeviceStatuses::create(&cuda, input_chunks.len())?;
    // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
    let kernel_conf = unsafe { crc32::heuristic_conf_from_sizes(input_chunks.view(), &stream) }?;
    let options = Options::new(spec, kernel_conf);

    stream
        .sync_scope(|_| {
            // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
            Ok(unsafe {
                crc32::compute_chunks_async(
                    input_chunks.view(),
                    &mut crc32_values,
                    &mut statuses,
                    options,
                    Crc32SegmentKind::Only,
                    &stream,
                )
            })
        })??
        .check()?;

    let actual = crc32_values.copy_to_host_vec()?[0];
    assert_eq!(actual, cpu_crc32(spec, &input));

    Ok(())
}

#[test]
fn test_crc32_accepts_misaligned_input_chunk() -> Result<()> {
    let cuda = Context::create()?;
    let stream = cuda.create_stream()?;
    let input = b"nvCOMP CRC32 misaligned input through Singe tests. ".repeat(512);
    let spec = Spec::CRC32_C;
    let mut padded = Vec::with_capacity(input.len() + 1);
    padded.push(0);
    padded.extend_from_slice(&input);

    let device_input = DeviceMemory::from_slice(stream.context(), &padded)?;
    let misaligned_input = device_input.view().slice(1..)?;
    let input_chunks = DeviceChunkTable::from_contiguous(&cuda, &misaligned_input)?;
    assert!(!input_chunks.is_aligned_to(2));

    let mut crc32_values = DeviceMemory::<u32>::zeroed(stream.context(), input_chunks.len())?;
    let mut statuses = DeviceStatuses::create(&cuda, input_chunks.len())?;
    let kernel_conf = crc32::heuristic_conf_for_chunks(&input_chunks, &stream)?;
    let options = Options::new(spec, kernel_conf);

    stream
        .sync_scope(|_| {
            // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
            Ok(unsafe {
                crc32::compute_chunks_async(
                    input_chunks.view(),
                    &mut crc32_values,
                    &mut statuses,
                    options,
                    Crc32SegmentKind::Only,
                    &stream,
                )
            })
        })??
        .check()?;

    let actual = crc32_values.copy_to_host_vec()?[0];
    assert_eq!(actual, cpu_crc32(spec, &input));

    Ok(())
}

#[test]
fn test_crc32_streaming_segments_match_cpu() -> Result<()> {
    let cuda = Context::create()?;
    let stream = cuda.create_stream()?;
    let first = b"nvCOMP CRC32 first streaming segment. ".repeat(256);
    let last = b"nvCOMP CRC32 last streaming segment. ".repeat(256);
    let mut input = first.clone();
    input.extend_from_slice(&last);
    let spec = Spec::CRC32;

    let first_device = DeviceMemory::from_slice(stream.context(), &first)?;
    let first_chunks = DeviceChunkTable::from_contiguous(&cuda, &first_device)?;
    let mut crc32_values =
        DeviceMemory::<u32>::create_uninit(stream.context(), first_chunks.len())?;
    let mut statuses = DeviceStatuses::create(&cuda, first_chunks.len())?;
    let kernel_conf = crc32::heuristic_conf_for_max_chunk_size(
        first_chunks.len(),
        first.len().max(last.len()),
        &stream,
    )?;
    let options = Options::new(spec, kernel_conf);

    stream
        .sync_scope(|_| {
            // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
            Ok(unsafe {
                crc32::compute_chunks_async(
                    first_chunks.view(),
                    &mut crc32_values,
                    &mut statuses,
                    options,
                    Crc32SegmentKind::First,
                    &stream,
                )
            })
        })??
        .check()?;

    let last_device = DeviceMemory::from_slice(stream.context(), &last)?;
    let last_chunks = DeviceChunkTable::from_contiguous(&cuda, &last_device)?;
    stream
        .sync_scope(|_| {
            // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
            Ok(unsafe {
                crc32::compute_chunks_async(
                    last_chunks.view(),
                    &mut crc32_values,
                    &mut statuses,
                    options,
                    Crc32SegmentKind::Last,
                    &stream,
                )
            })
        })??
        .check()?;

    // SAFETY: First produced the pending checksums, and Last completed their update.
    let crc32_values = unsafe { crc32_values.assume_init_prefix(first_chunks.len())? };
    let actual = crc32_values.copy_to_host_vec()?[0];
    assert_eq!(actual, cpu_crc32(spec, &input));

    Ok(())
}

fn device_crc32_segment(
    stream: &Stream,
    spec: Spec,
    segment_kind: Crc32SegmentKind,
    input: &[u8],
) -> Result<u32> {
    let device_input = DeviceMemory::from_slice(stream.context(), input)?;
    let input_chunks = DeviceChunkTable::from_contiguous(stream.context(), &device_input)?;
    let mut crc32_values = DeviceMemory::<u32>::zeroed(stream.context(), input_chunks.len())?;
    let mut statuses = DeviceStatuses::create(stream.context(), input_chunks.len())?;
    let kernel_conf = crc32::heuristic_conf_for_chunks(&input_chunks, stream)?;
    let options = Options::new(spec, kernel_conf);

    stream
        .sync_scope(|_| {
            // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
            Ok(unsafe {
                crc32::compute_chunks_async(
                    input_chunks.view(),
                    &mut crc32_values,
                    &mut statuses,
                    options,
                    segment_kind,
                    stream,
                )
            })
        })??
        .check()?;

    Ok(crc32_values.copy_to_host_vec()?[0])
}

fn gzip_fixture_bytes() -> Vec<u8> {
    b"nvCOMP gzip decompression fixture through Singe tests. ".repeat(64)
}

const GZIP_COMPRESSED_FIXTURE: &[u8] = &[
    31, 139, 8, 0, 0, 0, 0, 0, 0, 255, 237, 203, 177, 13, 128, 64, 8, 5, 208, 85, 152, 192, 37,
    172, 141, 38, 142, 160, 200, 81, 8, 151, 131, 51, 198, 233, 117, 10, 171, 95, 190, 226, 217,
    53, 206, 211, 66, 242, 104, 165, 157, 55, 63, 107, 227, 8, 117, 163, 67, 239, 236, 141, 41, 75,
    243, 46, 133, 86, 53, 249, 196, 145, 49, 144, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161,
    161, 161, 161, 253, 219, 94, 103, 236, 16, 29, 192, 13, 0, 0,
];

fn one_past(max: usize) -> usize {
    max.checked_add(1)
        .expect("nvCOMP chunk size limit should fit test increment")
}

fn assert_size_limit(error: Error, expected_max: usize, expected_actual: usize) {
    match error {
        Error::SizeLimitExceeded { max, actual, .. } => {
            assert_eq!(max, expected_max);
            assert_eq!(actual, expected_actual);
        }
        other => panic!("expected size-limit error, got {other:?}"),
    }
}

fn assert_length_mismatch(error: Error, expected_name: &str, expected: usize, actual: usize) {
    match error {
        Error::LengthMismatch {
            name,
            expected: error_expected,
            actual: error_actual,
        } => {
            assert_eq!(name, expected_name);
            assert_eq!(error_expected, expected);
            assert_eq!(error_actual, actual);
        }
        other => panic!("expected length-mismatch error, got {other:?}"),
    }
}

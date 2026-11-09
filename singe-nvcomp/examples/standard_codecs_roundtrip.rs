use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory, stream::Stream};
use singe_nvcomp::{
    batch::{DeviceChunkTable, DeviceChunkTableMut},
    error::Result,
    format::{Ans, BatchedCodec, Bitcomp, Cascaded, Codec, Deflate, Gdeflate, Lz4, Snappy, Zstd},
};

fn roundtrip_codec<C: BatchedCodec>(stream: &Stream, input: &[u8]) -> Result<()> {
    let output = roundtrip_with_codec::<C>(stream, input)?;
    assert_eq!(output.as_slice(), input);

    println!("{} roundtrip passed", C::NAME);
    Ok(())
}

fn roundtrip_with_codec<C: BatchedCodec>(stream: &Stream, input: &[u8]) -> Result<Vec<u8>> {
    let chunk_bytes = input.len();

    let device_input = DeviceMemory::from_slice(stream.context(), input)?;
    let input_chunks = DeviceChunkTable::from_contiguous(stream.context(), &device_input)?;
    let codec = Codec::<C>::default();

    let max_compressed_chunk_bytes = codec.max_output_chunk_size(chunk_bytes)?;

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
        // SAFETY: the example keeps these buffers alive and synchronizes before reading the results.
        Ok(unsafe {
            codec.decompress_size_chunks_async(
                compressed_chunks.written_chunks(),
                &mut queried_decompressed_sizes,
                stream,
            )
        })
    })??;

    let mut output = DeviceMemory::<u8>::create_uninit(stream.context(), chunk_bytes)?;
    let mut output_chunks = DeviceChunkTableMut::from_contiguous(stream.context(), &mut output)?;

    // SAFETY: `compressed_chunks` was produced immediately above by the same codec.
    unsafe {
        codec.decompress_chunks_alloc_async(
            compressed_chunks.written_chunks(),
            &mut output_chunks,
            stream,
        )?
    }
    .check()?;

    let queried_decompressed_sizes = queried_decompressed_sizes.copy_to_host_vec()?;
    // SAFETY: the example keeps these buffers alive and synchronizes before reading the results.
    let output_sizes = unsafe { output_chunks.copy_written_bytes(stream) }?;
    assert_eq!(queried_decompressed_sizes[0], chunk_bytes);
    assert_eq!(output_sizes[0], chunk_bytes);

    drop(output_chunks);

    // SAFETY: checked decompression completed and reported the initialized prefix.
    let output = unsafe { output.assume_init_prefix(output_sizes[0])? };
    Ok(output.copy_to_host_vec()?)
}

fn main() -> Result<()> {
    let cuda = CudaContext::create()?;
    let stream = cuda.create_stream()?;

    let input = b"nvCOMP standard codec roundtrip through Singe wrappers. ".repeat(1024);

    roundtrip_codec::<Ans>(&stream, &input)?;
    roundtrip_codec::<Bitcomp>(&stream, &input)?;
    roundtrip_codec::<Cascaded>(&stream, &input)?;
    roundtrip_codec::<Deflate>(&stream, &input)?;
    roundtrip_codec::<Gdeflate>(&stream, &input)?;
    roundtrip_codec::<Lz4>(&stream, &input)?;
    roundtrip_codec::<Snappy>(&stream, &input)?;
    roundtrip_codec::<Zstd>(&stream, &input)?;

    Ok(())
}

use singe_cuda::{
    context::Context as CudaContext,
    memory::DeviceMemory,
    stream::{Stream, StreamOptions},
};
use singe_nvcomp::{
    error::Result,
    format::{Ans, BatchedCodec, Bitcomp, Cascaded, Codec, Deflate, Gdeflate, Lz4, Snappy, Zstd},
};

const CHUNK_BYTES: usize = 16 * 1024;

fn roundtrip_codec<C: BatchedCodec>(stream: &Stream, input: &[u8]) -> Result<()> {
    let device_input = DeviceMemory::from_slice(stream.context(), input)?;
    let codec = Codec::<C>::default();
    let compressed = codec.compress_contiguous(&device_input, CHUNK_BYTES, stream)?;

    let mut output = DeviceMemory::<u8>::create_uninit(stream.context(), input.len())?;
    let output_bytes = codec.decompress_contiguous(&compressed, &mut output, stream)?;
    // SAFETY: decompression completed and reported this prefix length.
    let output = unsafe { output.assume_init_prefix(output_bytes)? }.copy_to_host_vec()?;
    assert_eq!(output.as_slice(), input);

    println!(
        "{} roundtrip passed: {} chunks, {} compressed bytes",
        C::NAME,
        compressed.len(),
        compressed.chunk_bytes().iter().sum::<usize>()
    );
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let cuda = CudaContext::create()?;
    let stream = cuda.create_stream(StreamOptions::default())?;

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

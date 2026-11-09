use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory};

use singe_nvcomp::{
    error::Result,
    format::{Codec, Lz4},
};

fn main() -> Result<()> {
    let cuda = CudaContext::create()?;
    let stream = cuda.create_stream()?;

    let input = b"nvCOMP LZ4 roundtrip through Singe wrappers. ".repeat(1024);
    let device_input = DeviceMemory::from_slice(&cuda, &input)?;
    let codec = Codec::<Lz4>::default();
    let compressed = codec.compress_one(&device_input, &stream)?;

    let mut output = DeviceMemory::<u8>::create_uninit(&cuda, input.len())?;
    let output_bytes = codec.decompress_one(&compressed, &mut output, &stream)?;
    assert_eq!(output_bytes, input.len());
    // SAFETY: decompression completed and reported this prefix length.
    let output = unsafe { output.assume_init_prefix(output_bytes)? }.copy_to_host_vec()?;
    assert_eq!(output, input);

    println!(
        "LZ4 roundtrip passed: {} input bytes, {} compressed bytes",
        input.len(),
        compressed.len()
    );

    Ok(())
}

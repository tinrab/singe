use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory};
use singe_nvcomp::{
    batch::{DeviceChunkTable, DeviceStatuses},
    crc32,
    crc32::{Options, Spec},
    error::Result,
    testing::cpu_crc32,
    types::Crc32SegmentKind,
};

fn main() -> Result<()> {
    let cuda = CudaContext::create()?;
    let stream = cuda.create_stream()?;

    let input = b"nvCOMP CRC32 through Singe wrappers. ".repeat(4096);
    let chunk_bytes = input.len();

    let device_input = DeviceMemory::from_slice(&cuda, &input)?;
    let input_chunks = DeviceChunkTable::from_contiguous(&cuda, &device_input)?;
    let mut crc32_values = DeviceMemory::<u32>::create_uninit(&cuda, input_chunks.len())?;
    let mut statuses = DeviceStatuses::create(&cuda, input_chunks.len())?;

    let spec = Spec::CRC32;
    let kernel_conf = crc32::heuristic_conf_for_chunks(&input_chunks, &stream)?;
    let options = Options::new(spec, kernel_conf);

    stream
        .sync_scope(|_| {
            // SAFETY: the example keeps these buffers alive and synchronizes before reading the results.
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

    // SAFETY: Only completed every checksum; unused capacity is not promoted.
    let crc32_values =
        unsafe { crc32_values.assume_init_prefix(input_chunks.len())? }.copy_to_host_vec()?;
    let expected = cpu_crc32(spec, &input);
    assert_eq!(crc32_values[0], expected);

    println!("CRC32 passed: {chunk_bytes} input bytes, checksum {expected:#010x}");

    Ok(())
}

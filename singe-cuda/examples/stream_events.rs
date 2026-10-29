//! Rust take on the CUDA "simpleStreams" idea: split work across several streams,
//! launch a kernel on each, and time the whole submission with events.

use std::error::Error;

use singe_cuda::{
    context::Context, cuda_module, event::EventRecordFlags, memory::DeviceMemory,
    module::LaunchConfig,
};

const STREAM_COUNT: usize = 4;
const CHUNK: usize = 1 << 20;
const BLOCK_SIZE: usize = 256;

cuda_module! {
    pub mod scale_kernel {
        source: r#"
            extern "C" __global__ void scale(float* values, float factor, int len) {
                int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
                if (i < len) {
                    values[i] *= factor;
                }
            }
        "#,
        exports: {
            scale {
                parameters: { values: write },
            },
        },
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = Context::create()?;
    let module = scale_kernel::Module::create(&context)?;
    let config = LaunchConfig::for_1d_grid(CHUNK, BLOCK_SIZE);

    let streams = (0..STREAM_COUNT)
        .map(|_| context.create_stream())
        .collect::<Result<Vec<_>, _>>()?;
    let mut chunks = (0..STREAM_COUNT)
        .map(|index| DeviceMemory::from_slice(&context, &vec![index as f32 + 1.0; CHUNK]))
        .collect::<Result<Vec<_>, _>>()?;

    let start = context.create_event()?;
    let stop = context.create_event()?;
    start.record(&streams[0], EventRecordFlags::DEFAULT)?;
    // Every stream waits for the start marker so the timing covers all launches.
    for stream in &streams[1..] {
        stream.wait_event(&start)?;
    }
    for (stream, chunk) in streams.iter().zip(&mut chunks) {
        // SAFETY: each chunk holds `CHUNK` elements, is written by exactly one stream,
        // and stays alive until the streams are synchronized below.
        unsafe {
            module.scale_on(&config, stream, chunk, 2.0, CHUNK as i32)?;
        }
    }
    // The first stream records the stop marker after waiting for all other streams.
    for stream in &streams[1..] {
        let done = context.create_event()?;
        done.record(stream, EventRecordFlags::DEFAULT)?;
        streams[0].wait_event(&done)?;
    }
    stop.record(&streams[0], EventRecordFlags::DEFAULT)?;
    stop.synchronize()?;

    let elapsed = stop.elapsed_time_since(&start)?;
    println!("scaled {STREAM_COUNT} chunks of {CHUNK} elements in {elapsed:.3} ms");

    for (index, chunk) in chunks.iter().enumerate() {
        let values = chunk.copy_to_host_vec()?;
        let expected = (index as f32 + 1.0) * 2.0;
        if values.iter().any(|value| *value != expected) {
            return Err(format!("stream {index} produced unexpected values").into());
        }
    }
    println!("all streams produced the expected values");
    Ok(())
}

//! Capture a kernel once and replay it against the same device allocation.

#![expect(
    clippy::significant_drop_tightening,
    reason = "cuda_module! generates the kernel cache lock scope"
)]

use std::error::Error;

use singe_cuda::{
    context::Context, cuda_module, memory::DeviceMemory, module::LaunchConfig,
    stream::StreamCaptureMode,
};

const REPLAYS: u32 = 5;

cuda_module! {
    pub mod kernels {
        source: r#"
            extern "C" __global__ void increment(unsigned int* values, int len) {
                int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
                if (i < len) values[i] += 1;
            }
        "#,
        exports: {
            increment { parameters: { values: read_write } },
        },
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = Context::create()?;
    let stream = context.create_stream()?;
    // Allocate and compile before capture; capture records work without executing it.
    let mut values = DeviceMemory::<u32>::zeroed(&context, 1024)?;
    let module = kernels::Module::create(&context)?;
    let config = LaunchConfig::for_1d_grid(values.len(), 128);
    let len = i32::try_from(values.len())?;
    let graph = stream.capture(StreamCaptureMode::ThreadLocal, |scope| {
        // SAFETY: the kernel checks bounds and has exclusive access to initialized
        // values. The allocation stays alive through every replay and synchronization.
        unsafe { module.increment_record(scope, &config, &mut values, len) }
    })?;
    let executable = graph.instantiate()?;

    for _ in 0..REPLAYS {
        // SAFETY: all launches use one stream, with no concurrent buffer access.
        // The allocation, module and executable outlive completion below.
        unsafe { executable.launch(&stream)? };
    }
    stream.synchronize()?;
    assert_eq!(values.copy_to_host_vec()?, vec![REPLAYS; values.len()]);
    println!("replayed the increment graph {REPLAYS} times");
    Ok(())
}

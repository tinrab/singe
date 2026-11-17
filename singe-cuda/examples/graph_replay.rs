//! Capture a kernel once and replay it against the same device allocation.

use singe_cuda::{
    context::Context,
    cuda_module,
    graph::GraphInstantiateFlags,
    memory::DeviceMemory,
    module::LaunchConfig,
    stream::{StreamCaptureMode, StreamOptions},
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

fn main() -> anyhow::Result<()> {
    let context = Context::create()?;
    let stream = context.create_stream(StreamOptions::default())?;
    // Allocate and compile before capture; capture records work without executing it.
    let mut values = DeviceMemory::<u32>::zeroed(&context, 1024)?;
    let module = kernels::Module::create(&context)?;
    let config = LaunchConfig::for_1d_grid(values.len(), 128);
    let len = i32::try_from(values.len())?;
    let graph = stream.capture(StreamCaptureMode::ThreadLocal, |scope| {
        // SAFETY: the kernel checks `i < len` and has exclusive access to the initialized `values`, which stay alive through every replay and the final synchronization.
        unsafe { module.increment_record(&mut values, len, &config, scope) }
    })?;
    let executable = graph.instantiate(GraphInstantiateFlags::empty())?;

    for _ in 0..REPLAYS {
        // SAFETY: every launch runs on one stream with no concurrent access to `values`, and the allocation, module, and executable outlive the synchronization below.
        unsafe { executable.launch_async(&stream)? };
    }

    stream.synchronize()?;
    assert_eq!(values.copy_to_host_vec()?, vec![REPLAYS; values.len()]);
    println!("replayed the increment graph {REPLAYS} times");
    Ok(())
}

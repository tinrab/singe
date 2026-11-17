//! Rust take on the CUDA "vectorAdd" sample: compile a kernel with NVRTC through `cuda_module!`, add two vectors on the device, and verify the result on the host.

use singe_cuda::{
    context::Context, cuda_module, memory::DeviceMemory, module::LaunchConfig,
    stream::StreamOptions,
};

const COUNT: usize = 50_000;
const BLOCK_SIZE: usize = 256;

cuda_module! {
    pub mod vector_add_kernel {
        source: r#"
            extern "C" __global__ void vector_add(
                const float* a,
                const float* b,
                float* c,
                int len
            ) {
                int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
                if (i < len) {
                    c[i] = a[i] + b[i];
                }
            }
        "#,
        exports: {
            vector_add {
                parameters: { a: read, b: read, c: write },
            },
        },
    }
}

fn main() -> anyhow::Result<()> {
    let context = Context::create()?;

    let a = (0..COUNT).map(|i| i as f32 * 0.5).collect::<Vec<_>>();
    let b = (0..COUNT).map(|i| (COUNT - i) as f32).collect::<Vec<_>>();

    let a_device = DeviceMemory::from_slice(&context, &a)?;
    let b_device = DeviceMemory::from_slice(&context, &b)?;
    let mut c_device = DeviceMemory::<f32>::zeroed(&context, COUNT)?;

    let stream = context.create_stream(StreamOptions::default())?;
    let module = vector_add_kernel::Module::create(&context)?;
    let config = LaunchConfig::for_1d_grid(COUNT, BLOCK_SIZE);
    println!(
        "launching vector_add with {} blocks of {BLOCK_SIZE} threads",
        config.grid_dim().x
    );
    // SAFETY: all buffers hold `COUNT` elements and stay alive until the stream is synchronized below.
    unsafe {
        module.vector_add_async(
            &a_device,
            &b_device,
            &mut c_device,
            i32::try_from(COUNT)?,
            &config,
            &stream,
        )?;
    }

    stream.synchronize()?;

    let c = c_device.copy_to_host_vec()?;
    let mismatch = a
        .iter()
        .zip(&b)
        .zip(&c)
        .position(|((a, b), c)| (a + b - c).abs() > 1e-5);
    if let Some(index) = mismatch {
        anyhow::bail!("result verification failed at element {index}");
    }

    println!("vector addition of {COUNT} elements passed");
    Ok(())
}

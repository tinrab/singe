//! Run a PTX kernel through `singe-cuda` on emulated GPUs, with no GPU or CUDA installed.
//!
//! Pass the number of GPUs as the first argument; the default is 2.

use std::env;

use singe_cuda::{
    context::Context,
    device::Device,
    dim::Dim3,
    jit::JitOptions,
    memory::DeviceMemory,
    module::{KernelParameters, LaunchConfig, ModuleImage},
    stream::StreamOptions,
};
use singe_ptx::emulator::{EmulatorOptions, Topology};

const PTX: &str = r"
.version 9.4
.target sm_80
.address_size 64

.visible .entry add(.param .u64 a, .param .u64 b, .param .u64 out, .param .u32 n)
{
    .reg .pred %p;
    .reg .b32 %r<6>;
    .reg .b64 %rd<7>;

    mov.u32 %r0, %tid.x;
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mad.lo.u32 %r0, %r4, %r5, %r0;
    ld.param.u32 %r1, [n];
    setp.ge.u32 %p, %r0, %r1;
    @%p bra done;
    ld.param.u64 %rd0, [a];
    ld.param.u64 %rd1, [b];
    ld.param.u64 %rd2, [out];
    mul.wide.u32 %rd3, %r0, 4;
    add.u64 %rd4, %rd0, %rd3;
    add.u64 %rd5, %rd1, %rd3;
    add.u64 %rd6, %rd2, %rd3;
    ld.global.f32 %r2, [%rd4];
    ld.global.f32 %r3, [%rd5];
    add.f32 %r2, %r2, %r3;
    st.global.f32 [%rd6], %r2;
done:
    ret;
}
";

const COUNT: usize = 200;
const THREADS: u32 = 64;

fn main() -> anyhow::Result<()> {
    run().map_err(|error| match singe_cuda_emulator::last_fault() {
        Some(fault) => error.context(format!("emulator fault: {fault}")),
        None => error,
    })
}

fn run() -> anyhow::Result<()> {
    let device_count: usize = env::args().nth(1).map_or(Ok(2), |count| count.parse())?;
    singe_cuda_emulator::install(EmulatorOptions {
        topology: Topology::node(device_count),
        ..Default::default()
    })?;
    let count = u32::try_from(COUNT)?;

    for id in 0..Device::count()? {
        let device = Device::new(id);
        device.set_current()?;
        let context = Context::create()?;
        let a: Vec<f32> = (0..COUNT).map(|index| index as f32).collect();
        let b: Vec<f32> = (0..COUNT)
            .map(|index| (id as f32).mul_add(1000.0, 0.5 * index as f32))
            .collect();
        let a_device = DeviceMemory::from_slice(&context, &a)?;
        let b_device = DeviceMemory::from_slice(&context, &b)?;
        let mut sums = DeviceMemory::<f32>::zeroed(&context, COUNT)?;

        let module = context.load_module(
            &ModuleImage::from_ptx(PTX.as_bytes())?,
            JitOptions::default(),
        )?;
        let stream = context.create_stream(StreamOptions::default())?;
        let config = LaunchConfig::create(
            Dim3::new(count.div_ceil(THREADS), 1, 1),
            Dim3::new(THREADS, 1, 1),
            0,
        )?;
        let mut parameters = KernelParameters::default();
        parameters
            .device_input(&a_device)
            .device_input(&b_device)
            .device_slice_mut(&mut sums)
            .owned_arg(count);
        // SAFETY: the buffers hold `COUNT` floats and outlive the synchronization below.
        unsafe {
            module
                .function("add")?
                .launch_async(&config, &mut parameters, &stream)?;
        }

        stream.synchronize()?;

        let sums = sums.copy_to_host_vec()?;
        let wrong = (0..COUNT).find(|&index| sums[index] != a[index] + b[index]);
        anyhow::ensure!(wrong.is_none(), "device {id} has a wrong sum at {wrong:?}");
    }

    println!("add of {COUNT} floats succeeded on {device_count} emulated GPUs");
    Ok(())
}

use singe_cuda::{
    context::Context as CudaContext,
    device::Device,
    memory::DeviceMemory,
    stream::{StreamFlags, StreamOptions},
    types::Complex32,
};
use singe_cufft::{
    plan::{ManyLayout, Plan},
    types::{Direction, TransformType},
};

fn main() -> anyhow::Result<()> {
    let context = CudaContext::create_for_device(Device::new(0))?;
    let stream = context.create_stream(StreamOptions {
        flags: StreamFlags::NON_BLOCKING,
        ..StreamOptions::default()
    })?;

    let n: i32 = 2;
    let batch_size: i32 = 2;
    let fft_size = usize::try_from(n * n * n)?;
    let element_count = fft_size * usize::try_from(batch_size)?;

    let input = (0..element_count)
        .map(|index| Complex32::new(index as f32, -(index as f32)))
        .collect::<Vec<_>>();
    let mut device_data = DeviceMemory::from_slice(&context, &input)?;

    let mut plan = Plan::create(&context)?;
    plan.plan_many(&ManyLayout {
        n: &[n, n, n],
        inembed: None,
        istride: 1,
        idist: 0,
        onembed: None,
        ostride: 1,
        odist: 0,
        kind: TransformType::ComplexToComplex,
        batch: batch_size,
    })?;

    plan.execute_in_place(&mut device_data, Direction::Forward, &stream)?;
    plan.execute_in_place(&mut device_data, Direction::Inverse, &stream)?;

    let output = device_data.copy_to_host_vec()?;
    for (actual, expected) in output.iter().zip(input.iter()) {
        singe_core::assert_complex_close!(*actual, *expected * fft_size as f32, 1.0e-3);
    }

    println!("3d c2c round trip result: {output:?}");
    Ok(())
}

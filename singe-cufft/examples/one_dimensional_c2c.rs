use singe_cuda::{
    context::Context as CudaContext,
    device::Device,
    memory::DeviceMemory,
    stream::{StreamFlags, StreamOptions},
    types::Complex32,
};
use singe_cufft::{
    plan::Plan,
    types::{Direction, TransformType},
};

fn main() -> anyhow::Result<()> {
    let context = CudaContext::create_for_device(Device::new(0))?;
    let stream = context.create_stream(StreamOptions {
        flags: StreamFlags::NON_BLOCKING,
        ..StreamOptions::default()
    })?;

    let fft_size = 8;
    let batch_size = 2;
    let element_count = fft_size * batch_size;

    let input = (0..element_count)
        .map(|index| Complex32::new(index as f32, -(index as f32)))
        .collect::<Vec<_>>();
    let mut device_data = DeviceMemory::from_slice(&context, &input)?;

    let mut plan = Plan::create(&context)?;
    plan.plan_1d(fft_size, TransformType::ComplexToComplex, batch_size)?;

    // Transforming in place matches cuFFT's C API when the input and output pointers are equal.
    plan.execute_in_place(&mut device_data, Direction::Forward, &stream)?;
    plan.execute_in_place(&mut device_data, Direction::Inverse, &stream)?;

    // cuFFT does not normalize inverse transforms.
    // A forward transform followed by an inverse transform returns the original data scaled by fft_size.
    let output = device_data.copy_to_host_vec()?;
    for (actual, expected) in output.iter().zip(input.iter()) {
        singe_core::assert_complex_close!(*actual, *expected * fft_size as f32, 1.0e-3);
    }

    println!("1d c2c round trip result: {output:?}");
    Ok(())
}

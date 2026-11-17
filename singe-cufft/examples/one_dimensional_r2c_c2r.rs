use singe_cuda::{
    context::Context as CudaContext,
    device::Device,
    memory::DeviceMemory,
    stream::{StreamFlags, StreamOptions},
    types::Complex32,
    view::DeviceOutputView,
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
    let spectrum_count = (fft_size / 2 + 1) * batch_size;

    let input = (0..element_count)
        .map(|index| index as f32)
        .collect::<Vec<_>>();
    let mut device_input = DeviceMemory::from_slice(&context, &input)?;
    let mut device_spectrum = DeviceMemory::<Complex32>::create_uninit(&context, spectrum_count)?;

    let mut plan_r2c = Plan::create(&context)?;
    plan_r2c.plan_1d(fft_size, TransformType::RealToComplex, batch_size)?;
    let mut plan_c2r = Plan::create(&context)?;
    plan_c2r.plan_1d(fft_size, TransformType::ComplexToReal, batch_size)?;

    plan_r2c.execute(
        &mut device_input,
        device_spectrum.output(),
        Direction::Forward,
        &stream,
    )?;
    // SAFETY: the forward transform wrote the whole half spectrum of every batch.
    let mut device_spectrum = unsafe { device_spectrum.assume_init() };
    // The inverse may overwrite the spectrum, which it borrows exclusively.
    plan_c2r.execute(
        &mut device_spectrum,
        DeviceOutputView::from_output(&mut device_input),
        Direction::Inverse,
        &stream,
    )?;

    // R2C stores only the nonredundant half spectrum.
    // The inverse C2R result is scaled by fft_size because cuFFT leaves normalization to the caller.
    let output = device_input.copy_to_host_vec()?;
    for (actual, expected) in output.iter().zip(input.iter()) {
        assert!((actual - expected * fft_size as f32).abs() < 1e-3);
    }

    println!("1d r2c/c2r round trip result: {output:?}");
    Ok(())
}

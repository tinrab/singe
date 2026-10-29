use std::error::Error;

use singe_cuda::{
    context::Context as CudaContext,
    device::Device,
    memory::DeviceMemory,
    stream::{StreamBinding, StreamFlags},
    types::Complex32,
    view::DeviceOutputView,
};
use singe_cufft::{error::Result as CufftResult, plan::PlanBuilder, types::TransformType};

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::create_for_device(Device::new(0))?;
    let stream = context.create_stream_with_flags(StreamFlags::NON_BLOCKING)?;

    let fft_size = 8;
    let batch_size = 2;
    let element_count = fft_size * batch_size;
    let spectrum_count = (fft_size / 2 + 1) * batch_size;

    let input = (0..element_count)
        .map(|index| index as f32)
        .collect::<Vec<_>>();
    let mut device_input = DeviceMemory::from_slice(&context, &input)?;
    let mut device_spectrum = DeviceMemory::<Complex32>::create_uninit(&context, spectrum_count)?;

    let plan_r2c = PlanBuilder::new(&context)
        .one_dimensional(fft_size, TransformType::RealToComplex)
        .batch(batch_size)
        .create()?;
    let plan_c2r = PlanBuilder::new(&context)
        .one_dimensional(fft_size, TransformType::ComplexToReal)
        .batch(batch_size)
        .create()?;

    // The half-spectrum capacity matches both configured plans.
    // The inverse reads the spectrum after the forward producer on the same stream and may overwrite it.
    StreamBinding::Borrowed(stream.to_borrowed()).sync_scope(|| -> CufftResult<()> {
        // SAFETY: the example keeps these buffers alive and synchronizes before reading the results.
        unsafe {
            plan_r2c.execute_r2c_f32_async(
                &device_input,
                device_spectrum.output(),
                Some(&stream),
            )?;
            plan_c2r.execute_c2r_f32_async(
                device_spectrum.output(),
                DeviceOutputView::from_output(&mut device_input),
                Some(&stream),
            )?;
        }
        Ok(())
    })?;

    // R2C stores only the non-redundant half spectrum. The inverse C2R result is
    // scaled by fft_size because cuFFT leaves normalization to the caller.
    let output = device_input.copy_to_host_vec()?;
    for (actual, expected) in output.iter().zip(input.iter()) {
        assert!((actual - expected * fft_size as f32).abs() < 1e-3);
    }

    println!("1d r2c/c2r round trip result: {output:?}");
    Ok(())
}

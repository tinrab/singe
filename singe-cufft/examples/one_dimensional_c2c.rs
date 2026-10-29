use std::error::Error;

use singe_cuda::{
    context::Context as CudaContext,
    device::Device,
    memory::DeviceMemory,
    stream::{StreamBinding, StreamFlags},
    types::Complex32,
    view::DeviceOutputView,
};
use singe_cufft::{
    error::Result as CufftResult,
    plan::PlanBuilder,
    types::{Direction, TransformType},
};

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::create_for_device(Device::new(0))?;
    let stream = context.create_stream_with_flags(StreamFlags::NON_BLOCKING)?;

    let fft_size = 8;
    let batch_size = 2;
    let element_count = fft_size * batch_size;

    let input = (0..element_count)
        .map(|index| Complex32::new(index as f32, -(index as f32)))
        .collect::<Vec<_>>();
    let mut device_data = DeviceMemory::from_slice(&context, &input)?;

    let plan = PlanBuilder::new(&context)
        .one_dimensional(fft_size, TransformType::ComplexToComplex)
        .batch(batch_size)
        .create()?;

    // Passing the same device allocation as input and output performs an
    // in-place transform, matching cuFFT's C API behavior.
    // The configured transform covers this complete initialized buffer.
    // Both queued accesses remain ordered on this stream and borrowed until the scope drains.
    StreamBinding::Borrowed(stream.to_borrowed()).sync_scope(|| -> CufftResult<()> {
        // SAFETY: the example keeps these buffers alive and synchronizes before reading the results.
        unsafe {
            plan.execute_c2c_f32_in_place_async(
                DeviceOutputView::from_output(&mut device_data),
                Direction::Forward,
                Some(&stream),
            )?;
            plan.execute_c2c_f32_in_place_async(
                DeviceOutputView::from_output(&mut device_data),
                Direction::Inverse,
                Some(&stream),
            )?;
        }
        Ok(())
    })?;

    // cuFFT does not normalize inverse transforms. A forward transform followed
    // by an inverse transform returns the original data scaled by fft_size.
    let output = device_data.copy_to_host_vec()?;
    for (actual, expected) in output.iter().zip(input.iter()) {
        singe_core::assert_complex_close!(*actual, *expected * fft_size as f32, 1.0e-3);
    }

    println!("1d c2c round trip result: {output:?}");
    Ok(())
}

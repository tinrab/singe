use std::error::Error;

use singe_cuda::{
    context::Context as CudaContext, device::Device, memory::DeviceMemory, stream::StreamFlags,
    types::Complex32,
};
use singe_cufft::{
    plan::{ManyPlanConfig, ManyPlanConfigI32, PlanBuilder},
    types::TransformType,
};

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::create_for_device(Device::new(0))?;
    let stream = context.create_stream_with_flags(StreamFlags::NON_BLOCKING)?;

    let rows = 2;
    let cols = 4;
    let batch_size = 2;
    let real_count = rows * cols * batch_size;
    let spectrum_count = rows * (cols / 2 + 1) * batch_size;

    let input_spectrum = (0..spectrum_count)
        .map(|index| Complex32::new(index as f32, 0.0))
        .collect::<Vec<_>>();
    let mut device_spectrum = DeviceMemory::from_slice(&input_spectrum)?;
    let mut device_real = DeviceMemory::<f32>::zeroes(real_count)?;

    let plan_c2r = PlanBuilder::new(&context)
        .many(ManyPlanConfig::I32(ManyPlanConfigI32 {
            n: &[rows, cols],
            inembed: None,
            istride: 1,
            idist: 0,
            onembed: None,
            ostride: 1,
            odist: 0,
            kind: TransformType::ComplexToReal,
            batch: batch_size,
        }))
        .create()?;
    let plan_r2c = PlanBuilder::new(&context)
        .many(ManyPlanConfig::I32(ManyPlanConfigI32 {
            n: &[rows, cols],
            inembed: None,
            istride: 1,
            idist: 0,
            onembed: None,
            ostride: 1,
            odist: 0,
            kind: TransformType::RealToComplex,
            batch: batch_size,
        }))
        .create()?;

    plan_c2r.execute_c2r_f32(&device_spectrum, &mut device_real, Some(&stream))?;
    plan_r2c.execute_r2c_f32(&device_real, &mut device_spectrum, Some(&stream))?;
    stream.synchronize()?;

    // For 2D transforms, the round-trip scale is rows * cols for each batch.
    let output_spectrum = device_spectrum.copy_to_host_vec()?;
    for (actual, expected) in output_spectrum.iter().zip(input_spectrum.iter()) {
        singe_core::assert_complex_close!(*actual, *expected * (rows * cols) as f32, 1.0e-3);
    }

    println!("2d c2r/r2c round trip result: {output_spectrum:?}");
    Ok(())
}

use std::error::Error;

use singe_cuda::{
    context::Context as CudaContext, device::Device, memory::DeviceMemory, stream::StreamFlags,
    types::Complex32,
};
use singe_cufft::{
    plan::{ManyPlanConfig, ManyPlanConfigI32, PlanBuilder},
    types::{Direction, TransformType},
};

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::create_for_device(Device::new(0))?;
    let stream = context.create_stream_with_flags(StreamFlags::NON_BLOCKING)?;

    let n = 2;
    let batch_size = 2;
    let fft_size = n * n * n;
    let element_count = fft_size * batch_size;

    let input = (0..element_count)
        .map(|index| Complex32::new(index as f32, -(index as f32)))
        .collect::<Vec<_>>();
    let mut device_data = DeviceMemory::from_slice(&input)?;

    let plan = PlanBuilder::new(&context)
        .many(ManyPlanConfig::I32(ManyPlanConfigI32 {
            n: &[n, n, n],
            inembed: None,
            istride: 1,
            idist: 0,
            onembed: None,
            ostride: 1,
            odist: 0,
            kind: TransformType::ComplexToComplex,
            batch: batch_size,
        }))
        .create()?;

    plan.execute_c2c_f32_in_place(&mut device_data, Direction::Forward, Some(&stream))?;
    plan.execute_c2c_f32_in_place(&mut device_data, Direction::Inverse, Some(&stream))?;
    stream.synchronize()?;

    let output = device_data.copy_to_host_vec()?;
    for (actual, expected) in output.iter().zip(input.iter()) {
        assert_complex_close(*actual, *expected * fft_size as f32);
    }

    println!("3d c2c round trip result: {output:?}");
    Ok(())
}

fn assert_complex_close(actual: Complex32, expected: Complex32) {
    assert!((actual.re - expected.re).abs() < 1e-3);
    assert!((actual.im - expected.im).abs() < 1e-3);
}

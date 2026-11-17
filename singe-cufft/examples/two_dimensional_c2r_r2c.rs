use singe_cuda::{
    context::Context as CudaContext,
    device::Device,
    memory::DeviceMemory,
    stream::{StreamFlags, StreamOptions},
    types::Complex32,
    view::DeviceOutputView,
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

    let rows: i32 = 2;
    let cols: i32 = 4;
    let batch_size: i32 = 2;
    let real_count = usize::try_from(rows * cols * batch_size)?;
    let spectrum_count = usize::try_from(rows * (cols / 2 + 1) * batch_size)?;

    let input_spectrum = (0..spectrum_count)
        .map(|index| Complex32::new(index as f32, 0.0))
        .collect::<Vec<_>>();
    let mut device_spectrum = DeviceMemory::from_slice(&context, &input_spectrum)?;
    let mut device_real = DeviceMemory::<f32>::create_uninit(&context, real_count)?;

    let dimensions = [rows, cols];
    let layout = |kind: TransformType| ManyLayout {
        n: &dimensions,
        inembed: None,
        istride: 1,
        idist: 0,
        onembed: None,
        ostride: 1,
        odist: 0,
        kind,
        batch: batch_size,
    };
    let mut plan_c2r = Plan::create(&context)?;
    plan_c2r.plan_many(&layout(TransformType::ComplexToReal))?;
    let mut plan_r2c = Plan::create(&context)?;
    plan_r2c.plan_many(&layout(TransformType::RealToComplex))?;

    // The inverse may overwrite its input spectrum, which it borrows exclusively.
    plan_c2r.execute(
        &mut device_spectrum,
        device_real.output(),
        Direction::Inverse,
        &stream,
    )?;
    // SAFETY: the inverse transform wrote every real sample of every batch.
    let mut device_real = unsafe { device_real.assume_init() };
    plan_r2c.execute(
        &mut device_real,
        DeviceOutputView::from_output(&mut device_spectrum),
        Direction::Forward,
        &stream,
    )?;

    // For 2D transforms, the round-trip scale is rows * cols for each batch.
    let output_spectrum = device_spectrum.copy_to_host_vec()?;
    for (actual, expected) in output_spectrum.iter().zip(input_spectrum.iter()) {
        singe_core::assert_complex_close!(*actual, *expected * (rows * cols) as f32, 1.0e-3);
    }

    println!("2d c2r/r2c round trip result: {output_spectrum:?}");
    Ok(())
}

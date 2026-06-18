use super::*;

#[test]
fn fluent_signal_pipeline_runs_scaled_binary_arithmetic_steps() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let u8_source_host = [8u8, 16, 32, 64];
    let u8_other_host = [2u8, 4, 8, 16];
    let u8_source_memory = DeviceMemory::from_slice(&u8_source_host)?;
    let u8_other_memory = DeviceMemory::from_slice(&u8_other_host)?;
    let u8_source = SignalView::from_memory(&u8_source_memory, u8_source_host.len())?;
    let u8_other = SignalView::from_memory(&u8_other_memory, u8_other_host.len())?;
    let mut u8_workspace = Workspace::create();

    let u8_signal = SignalPipeline::from_view(&stream_context, &mut u8_workspace, u8_source)
        .add_scaled(&u8_other, 0)?
        .subtract_scaled(&u8_other, 0)?
        .multiply_scaled(&u8_other, 0)?
        .divide_scaled(&u8_other, 0)?
        .finish()?;

    let u16_source_host = [8u16, 16, 32, 64];
    let u16_other_host = [2u16, 4, 8, 16];
    let u16_source_memory = DeviceMemory::from_slice(&u16_source_host)?;
    let u16_other_memory = DeviceMemory::from_slice(&u16_other_host)?;
    let u16_source = SignalView::from_memory(&u16_source_memory, u16_source_host.len())?;
    let u16_other = SignalView::from_memory(&u16_other_memory, u16_other_host.len())?;
    let mut u16_workspace = Workspace::create();

    let u16_is_owned = matches!(
        SignalPipeline::from_view(&stream_context, &mut u16_workspace, u16_source)
            .add_scaled(&u16_other, 0)?
            .subtract_scaled(&u16_other, 0)?
            .multiply_scaled(&u16_other, 0)?
            .divide_scaled(&u16_other, 0)?
            .into_backing(),
        SignalBacking::Owned(signal) if signal.len() == u16_source_host.len()
    );

    let i16_source_host = [8i16, 16, 32, 64];
    let i16_other_host = [2i16, 4, 8, 16];
    let i16_source_memory = DeviceMemory::from_slice(&i16_source_host)?;
    let i16_other_memory = DeviceMemory::from_slice(&i16_other_host)?;
    let i16_source = SignalView::from_memory(&i16_source_memory, i16_source_host.len())?;
    let i16_other = SignalView::from_memory(&i16_other_memory, i16_other_host.len())?;
    let mut i16_workspace = Workspace::create();

    let i16_signal = SignalPipeline::from_view(&stream_context, &mut i16_workspace, i16_source)
        .add_scaled(&i16_other, 0)?
        .subtract_scaled(&i16_other, 0)?
        .multiply_scaled(&i16_other, 0)?
        .divide_scaled(&i16_other, 0)?
        .finish()?;

    let i32_source_host = [8i32, 16, 32, 64];
    let i32_other_host = [2i32, 4, 8, 16];
    let i32_source_memory = DeviceMemory::from_slice(&i32_source_host)?;
    let i32_other_memory = DeviceMemory::from_slice(&i32_other_host)?;
    let i32_source = SignalView::from_memory(&i32_source_memory, i32_source_host.len())?;
    let i32_other = SignalView::from_memory(&i32_other_memory, i32_other_host.len())?;
    let mut i32_workspace = Workspace::create();

    let i32_signal = SignalPipeline::from_view(&stream_context, &mut i32_workspace, i32_source)
        .add_scaled(&i32_other, 0)?
        .subtract_scaled(&i32_other, 0)?
        .multiply_scaled(&i32_other, 0)?
        .divide_scaled(&i32_other, 0)?
        .finish()?;

    stream.synchronize()?;

    assert_eq!(u8_signal.len(), u8_source_host.len());
    assert!(u16_is_owned);
    assert_eq!(i16_signal.len(), i16_source_host.len());
    assert_eq!(i32_signal.len(), i32_source_host.len());

    Ok(())
}

#[test]
fn fluent_signal_pipeline_runs_scaled_constant_steps() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let u8_host = [8u8, 16, 32, 64];
    let u8_memory = DeviceMemory::from_slice(&u8_host)?;
    let u8_source = SignalView::from_memory(&u8_memory, u8_host.len())?;
    let mut u8_workspace = Workspace::create();

    let u8_signal = SignalPipeline::from_view(&stream_context, &mut u8_workspace, u8_source)
        .add_constant_scaled(2, 0)?
        .subtract_constant_scaled(1, 0)?
        .subtract_from_constant_scaled(96, 0)?
        .multiply_constant_scaled(1, 0)?
        .divide_constant_scaled(1, 0)?
        .finish()?;

    let u16_host = [8u16, 16, 32, 64];
    let u16_memory = DeviceMemory::from_slice(&u16_host)?;
    let u16_source = SignalView::from_memory(&u16_memory, u16_host.len())?;
    let mut u16_workspace = Workspace::create();

    let u16_is_owned = matches!(
        SignalPipeline::from_view(&stream_context, &mut u16_workspace, u16_source)
            .add_constant_scaled(2, 0)?
            .subtract_constant_scaled(1, 0)?
            .subtract_from_constant_scaled(96, 0)?
            .multiply_constant_scaled(1, 0)?
            .divide_constant_scaled(1, 0)?
            .into_backing(),
        SignalBacking::Owned(signal) if signal.len() == u16_host.len()
    );

    let i16_host = [8i16, 16, 32, 64];
    let i16_memory = DeviceMemory::from_slice(&i16_host)?;
    let i16_source = SignalView::from_memory(&i16_memory, i16_host.len())?;
    let mut i16_workspace = Workspace::create();

    let i16_signal = SignalPipeline::from_view(&stream_context, &mut i16_workspace, i16_source)
        .add_constant_scaled(2, 0)?
        .subtract_constant_scaled(1, 0)?
        .subtract_from_constant_scaled(96, 0)?
        .multiply_constant_scaled(1, 0)?
        .divide_constant_scaled(1, 0)?
        .finish()?;

    let i32_host = [8i32, 16, 32, 64];
    let i32_memory = DeviceMemory::from_slice(&i32_host)?;
    let i32_source = SignalView::from_memory(&i32_memory, i32_host.len())?;
    let mut i32_workspace = Workspace::create();

    let i32_signal = SignalPipeline::from_view(&stream_context, &mut i32_workspace, i32_source)
        .add_constant_scaled(2, 0)?
        .subtract_constant_scaled(1, 0)?
        .subtract_from_constant_scaled(96, 0)?
        .multiply_constant_scaled(1, 0)?
        .finish()?;

    stream.synchronize()?;

    assert_eq!(u8_signal.len(), u8_host.len());
    assert!(u16_is_owned);
    assert_eq!(i16_signal.len(), i16_host.len());
    assert_eq!(i32_signal.len(), i32_host.len());

    Ok(())
}

#[test]
fn fluent_signal_pipeline_runs_constant_bitwise_and_shift_steps() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let u8_host = [0b0011_0011u8, 0b1111_0000, 0b0101_0101, 0b1010_1010];
    let u8_memory = DeviceMemory::from_slice(&u8_host)?;
    let u8_source = SignalView::from_memory(&u8_memory, u8_host.len())?;
    let mut u8_workspace = Workspace::create();

    let u8_signal = SignalPipeline::from_view(&stream_context, &mut u8_workspace, u8_source)
        .bit_and_constant(0b1111_0000)?
        .bit_or_constant(0b0000_1111)?
        .bit_xor_constant(0b0101_0101)?
        .left_shift_constant(1)?
        .right_shift_constant(1)?
        .finish()?;

    let u16_host = [0x00ffu16, 0xff00, 0x0f0f, 0xf0f0];
    let u16_memory = DeviceMemory::from_slice(&u16_host)?;
    let u16_source = SignalView::from_memory(&u16_memory, u16_host.len())?;
    let mut u16_workspace = Workspace::create();

    let u16_is_owned = matches!(
        SignalPipeline::from_view(&stream_context, &mut u16_workspace, u16_source)
            .divide_into_constant(0xffff)?
            .bit_and_constant(0xff00)?
            .bit_or_constant(0x00ff)?
            .bit_xor_constant(0x5555)?
            .left_shift_constant(1)?
            .right_shift_constant(1)?
            .into_backing(),
        SignalBacking::Owned(signal) if signal.len() == u16_host.len()
    );

    let u32_host = [0x0000_ffffu32, 0xffff_0000, 0x0f0f_0f0f, 0xf0f0_f0f0];
    let u32_memory = DeviceMemory::from_slice(&u32_host)?;
    let u32_source = SignalView::from_memory(&u32_memory, u32_host.len())?;
    let mut u32_workspace = Workspace::create();

    let u32_is_owned = matches!(
        SignalPipeline::from_view(&stream_context, &mut u32_workspace, u32_source)
            .bit_and_constant(0xffff_0000)?
            .bit_or_constant(0x0000_ffff)?
            .bit_xor_constant(0x5555_5555)?
            .left_shift_constant(1)?
            .right_shift_constant(1)?
            .into_backing(),
        SignalBacking::Owned(signal) if signal.len() == u32_host.len()
    );

    let i16_host = [8i16, 16, 32, 64];
    let i16_memory = DeviceMemory::from_slice(&i16_host)?;
    let i16_source = SignalView::from_memory(&i16_memory, i16_host.len())?;
    let mut i16_workspace = Workspace::create();

    let i16_signal = SignalPipeline::from_view(&stream_context, &mut i16_workspace, i16_source)
        .left_shift_constant(1)?
        .right_shift_constant(1)?
        .finish()?;

    let i32_host = [8i32, 16, 32, 64];
    let i32_memory = DeviceMemory::from_slice(&i32_host)?;
    let i32_source = SignalView::from_memory(&i32_memory, i32_host.len())?;
    let mut i32_workspace = Workspace::create();

    let i32_signal = SignalPipeline::from_view(&stream_context, &mut i32_workspace, i32_source)
        .left_shift_constant(1)?
        .right_shift_constant(1)?
        .finish()?;

    stream.synchronize()?;

    assert_eq!(u8_signal.len(), u8_host.len());
    assert!(u16_is_owned);
    assert!(u32_is_owned);
    assert_eq!(i16_signal.len(), i16_host.len());
    assert_eq!(i32_signal.len(), i32_host.len());

    Ok(())
}

#[test]
fn fluent_signal_pipeline_runs_bitwise_steps() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [0b0011_0011u8, 0b1111_0000, 0b0101_0101, 0b1010_1010];
    let other_host = [0b0000_1111u8, 0b0011_0011, 0b1111_0000, 0b0101_0101];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let other_memory = DeviceMemory::from_slice(&other_host)?;
    let source = SignalView::from_memory(&source_memory, source_host.len())?;
    let other = SignalView::from_memory(&other_memory, other_host.len())?;
    let mut workspace = Workspace::create();

    let signal = SignalPipeline::from_view(&stream_context, &mut workspace, source)
        .bit_and(&other)?
        .bit_or(&other)?
        .bit_xor(&other)?
        .finish()?;

    let u16_source_host = [0x00ffu16, 0xff00, 0x0f0f, 0xf0f0];
    let u16_other_host = [0x0ff0u16, 0xf00f, 0x3333, 0x5555];
    let u16_source_memory = DeviceMemory::from_slice(&u16_source_host)?;
    let u16_other_memory = DeviceMemory::from_slice(&u16_other_host)?;
    let u16_source = SignalView::from_memory(&u16_source_memory, u16_source_host.len())?;
    let u16_other = SignalView::from_memory(&u16_other_memory, u16_other_host.len())?;
    let mut u16_workspace = Workspace::create();

    let u16_is_owned = matches!(
        SignalPipeline::from_view(&stream_context, &mut u16_workspace, u16_source)
            .bit_and(&u16_other)?
            .bit_or(&u16_other)?
            .bit_xor(&u16_other)?
            .into_backing(),
        SignalBacking::Owned(signal) if signal.len() == u16_source_host.len()
    );

    let u32_source_host = [0x0000_ffffu32, 0xffff_0000, 0x0f0f_0f0f, 0xf0f0_f0f0];
    let u32_other_host = [0x00ff_00ffu32, 0xff00_ff00, 0x3333_3333, 0x5555_5555];
    let u32_source_memory = DeviceMemory::from_slice(&u32_source_host)?;
    let u32_other_memory = DeviceMemory::from_slice(&u32_other_host)?;
    let u32_source = SignalView::from_memory(&u32_source_memory, u32_source_host.len())?;
    let u32_other = SignalView::from_memory(&u32_other_memory, u32_other_host.len())?;
    let mut u32_workspace = Workspace::create();

    let u32_is_owned = matches!(
        SignalPipeline::from_view(&stream_context, &mut u32_workspace, u32_source)
            .bit_and(&u32_other)?
            .bit_or(&u32_other)?
            .bit_xor(&u32_other)?
            .into_backing(),
        SignalBacking::Owned(signal) if signal.len() == u32_source_host.len()
    );

    stream.synchronize()?;

    assert_eq!(signal.len(), source_host.len());
    assert!(u16_is_owned);
    assert!(u32_is_owned);

    Ok(())
}

#[test]
fn fluent_signal_pipeline_runs_integer_and_f64_threshold_variants() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let i16_host = [-4i16, 2, 7, -1];
    let i16_memory = DeviceMemory::from_slice(&i16_host)?;
    let i16_source = SignalView::from_memory(&i16_memory, i16_host.len())?;
    let f64_host = [1.0f64, 2.0, 4.0, 8.0];
    let f64_memory = DeviceMemory::from_slice(&f64_host)?;
    let f64_source = SignalView::from_memory(&f64_memory, f64_host.len())?;
    let mut i16_workspace = Workspace::create();

    let i16_signal = SignalPipeline::from_view(&stream_context, &mut i16_workspace, i16_source)
        .threshold(0, ComparisonOperation::Greater)?
        .threshold_less(1)?
        .threshold_greater(6)?
        .threshold_less_value(0, 0)?
        .threshold_greater_value(5, 5)?
        .finish()?;

    let mut f64_workspace = Workspace::create();
    let f64_is_owned = matches!(
        SignalPipeline::from_view(&stream_context, &mut f64_workspace, f64_source)
            .threshold(2.0, ComparisonOperation::Greater)?
            .threshold_less(3.0)?
            .threshold_greater(1.0)?
            .threshold_less_value(2.0, 2.0)?
            .threshold_greater_value(3.0, 3.0)?
            .into_backing(),
        SignalBacking::Owned(signal) if signal.len() == f64_host.len()
    );

    stream.synchronize()?;

    assert_eq!(i16_signal.len(), i16_host.len());
    assert!(f64_is_owned);

    Ok(())
}

#[test]
fn fluent_signal_pipeline_runs_every_statistics() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [1.0f32, 4.0, 9.0, 16.0];
    let other_host = [2.0f32, 3.0, 10.0, 12.0];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let other_memory = DeviceMemory::from_slice(&other_host)?;
    let source = SignalView::from_memory(&source_memory, source_host.len())?;
    let other = SignalView::from_memory(&other_memory, other_host.len())?;
    let mut workspace = Workspace::create();

    let signal = SignalPipeline::from_view(&stream_context, &mut workspace, source)
        .min_every(&other)?
        .max_every(&other)?
        .finish()?;

    stream.synchronize()?;

    assert_eq!(signal.len(), source_host.len());
    assert!(!signal.is_empty());

    Ok(())
}

#[test]
fn fluent_signal_pipeline_runs_integral_filter() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [1i32, 2, 3, 4];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let source = SignalView::from_memory(&source_memory, source_host.len())?;
    let mut workspace = Workspace::create();

    let signal = SignalPipeline::from_view(&stream_context, &mut workspace, source)
        .set(2)?
        .integral()?
        .finish()?;

    stream.synchronize()?;

    assert_eq!(signal.len(), source_host.len() + 1);
    assert!(!signal.is_empty());
    assert!(workspace.scratch_buffers() >= 1);

    Ok(())
}

#[test]
fn fluent_signal_pipeline_runs_scalar_statistics() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [1.0f32, 2.0, 4.0, 8.0];
    let other_host = [0.5f32, 2.5, 3.0, 10.0];
    let complex_other_host = [
        Complex32::new(1.0, 0.5),
        Complex32::new(2.0, 1.0),
        Complex32::new(4.0, 2.0),
        Complex32::new(8.0, 4.0),
    ];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let other_memory = DeviceMemory::from_slice(&other_host)?;
    let complex_other_memory = DeviceMemory::from_slice(&complex_other_host)?;
    let source = SignalView::from_memory(&source_memory, source_host.len())?;
    let other = SignalView::from_memory(&other_memory, other_host.len())?;
    let complex_other = SignalView::from_memory(&complex_other_memory, complex_other_host.len())?;
    let mut workspace = Workspace::create();

    let sum = SignalPipeline::from_view(&stream_context, &mut workspace, source)
        .sum()?
        .finish()?;
    let mean = SignalPipeline::from_view(&stream_context, &mut workspace, source)
        .mean()?
        .finish()?;
    let standard_deviation = SignalPipeline::from_view(&stream_context, &mut workspace, source)
        .standard_deviation()?
        .finish()?;
    let mean_standard_deviation =
        SignalPipeline::from_view(&stream_context, &mut workspace, source)
            .mean_standard_deviation()?;
    let norm_inf = SignalPipeline::from_view(&stream_context, &mut workspace, source)
        .norm_inf()?
        .finish()?;
    let norm_l1 = SignalPipeline::from_view(&stream_context, &mut workspace, source)
        .norm_l1()?
        .finish()?;
    let norm_l2 = SignalPipeline::from_view(&stream_context, &mut workspace, source)
        .norm_l2()?
        .finish()?;
    let max = SignalPipeline::from_view(&stream_context, &mut workspace, source)
        .max()?
        .finish()?;
    let min = SignalPipeline::from_view(&stream_context, &mut workspace, source)
        .min()?
        .finish()?;
    let sum_natural_logarithm = SignalPipeline::from_view(&stream_context, &mut workspace, source)
        .sum_natural_logarithm()?
        .finish()?;
    let norm_diff_inf = SignalPipeline::from_view(&stream_context, &mut workspace, source)
        .norm_diff_inf(&other)?
        .finish()?;
    let norm_diff_l1 = SignalPipeline::from_view(&stream_context, &mut workspace, source)
        .norm_diff_l1(&other)?
        .finish()?;
    let norm_diff_l2 = SignalPipeline::from_view(&stream_context, &mut workspace, source)
        .norm_diff_l2(&other)?
        .finish()?;
    let dot_product = SignalPipeline::from_view(&stream_context, &mut workspace, source)
        .dot_product(&other)?
        .finish()?;
    let dot_product_f64 = SignalPipeline::from_view(&stream_context, &mut workspace, source)
        .dot_product_to_f64(&other)?
        .finish()?;
    let dot_product_complex = SignalPipeline::from_view(&stream_context, &mut workspace, source)
        .dot_product_with_complex(&complex_other)?
        .finish()?;
    let dot_product_f64_complex =
        SignalPipeline::from_view(&stream_context, &mut workspace, source)
            .dot_product_with_complex_to_f64_complex(&complex_other)?
            .finish()?;

    stream.synchronize()?;

    assert_eq!(sum.len(), 1);
    assert_eq!(mean.len(), 1);
    assert_eq!(standard_deviation.len(), 1);
    assert_eq!(mean_standard_deviation.mean.len(), 1);
    assert_eq!(mean_standard_deviation.standard_deviation.len(), 1);
    assert_eq!(norm_inf.len(), 1);
    assert_eq!(norm_l1.len(), 1);
    assert_eq!(norm_l2.len(), 1);
    assert_eq!(max.len(), 1);
    assert_eq!(min.len(), 1);
    assert_eq!(sum_natural_logarithm.len(), 1);
    assert_eq!(norm_diff_inf.len(), 1);
    assert_eq!(norm_diff_l1.len(), 1);
    assert_eq!(norm_diff_l2.len(), 1);
    assert_eq!(dot_product.len(), 1);
    assert_eq!(dot_product_f64.len(), 1);
    assert_eq!(dot_product_complex.len(), 1);
    assert_eq!(dot_product_f64_complex.len(), 1);
    assert!(workspace.scratch_buffers() >= 1);

    Ok(())
}

#[test]
fn fluent_signal_pipeline_runs_error_metric_statistics() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [1.0f32, 2.0, 4.0, 8.0];
    let other_host = [2.0f32, 3.0, 2.0, 4.0];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let other_memory = DeviceMemory::from_slice(&other_host)?;
    let source = SignalView::from_memory(&source_memory, source_host.len())?;
    let other = SignalView::from_memory(&other_memory, other_host.len())?;
    let mut workspace = Workspace::create();

    let maximum_error =
        SignalPipeline::from_view(&stream_context, &mut workspace, source).maximum_error(&other)?;
    let average_error =
        SignalPipeline::from_view(&stream_context, &mut workspace, source).average_error(&other)?;
    let maximum_relative_error = SignalPipeline::from_view(&stream_context, &mut workspace, source)
        .maximum_relative_error(&other)?;
    let average_relative_error = SignalPipeline::from_view(&stream_context, &mut workspace, source)
        .average_relative_error(&other)?;

    stream.synchronize()?;

    assert_eq!(maximum_error.len(), 1);
    assert_eq!(average_error.len(), 1);
    assert_eq!(maximum_relative_error.len(), 1);
    assert_eq!(average_relative_error.len(), 1);
    assert!(workspace.scratch_buffers() >= 1);

    Ok(())
}

#[test]
fn fluent_signal_pipeline_runs_zero_crossing_statistics() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let i16_source_host = [-2i16, -1, 0, 3, -4, 5];
    let f32_source_host = [-2.0f32, -1.0, 0.0, 3.0, -4.0, 5.0];
    let i16_source_memory = DeviceMemory::from_slice(&i16_source_host)?;
    let f32_source_memory = DeviceMemory::from_slice(&f32_source_host)?;
    let i16_source = SignalView::from_memory(&i16_source_memory, i16_source_host.len())?;
    let f32_source = SignalView::from_memory(&f32_source_memory, f32_source_host.len())?;
    let mut i16_workspace = Workspace::create();
    let mut f32_workspace = Workspace::create();

    let i16_zero_crossing =
        SignalPipeline::from_view(&stream_context, &mut i16_workspace, i16_source)
            .zero_crossing(ZeroCrossingType::SignChange)?;
    let f32_zero_crossing =
        SignalPipeline::from_view(&stream_context, &mut f32_workspace, f32_source)
            .zero_crossing(ZeroCrossingType::SignChangeCountZero)?;

    stream.synchronize()?;

    assert_eq!(i16_zero_crossing.len(), 1);
    assert_eq!(f32_zero_crossing.len(), 1);
    assert!(i16_workspace.scratch_buffers() >= 1);
    assert!(f32_workspace.scratch_buffers() >= 1);

    Ok(())
}

#[test]
fn fluent_signal_pipeline_runs_integer_extremum_statistics() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [-4i16, 2, 7, -1];
    let other_host = [-2i16, 3, 4, -8];
    let count_source_host = [-4i32, 2, 7, -1];
    let complex_other_host = [
        ComplexI16::new(1, 0),
        ComplexI16::new(2, 1),
        ComplexI16::new(3, 1),
        ComplexI16::new(4, 2),
    ];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let other_memory = DeviceMemory::from_slice(&other_host)?;
    let count_source_memory = DeviceMemory::from_slice(&count_source_host)?;
    let complex_other_memory = DeviceMemory::from_slice(&complex_other_host)?;
    let source = SignalView::from_memory(&source_memory, source_host.len())?;
    let other = SignalView::from_memory(&other_memory, other_host.len())?;
    let count_source = SignalView::from_memory(&count_source_memory, count_source_host.len())?;
    let complex_other = SignalView::from_memory(&complex_other_memory, complex_other_host.len())?;
    let mut workspace = Workspace::create();

    let max = SignalPipeline::from_view(&stream_context, &mut workspace, source)
        .max()?
        .finish()?;
    let min = SignalPipeline::from_view(&stream_context, &mut workspace, source)
        .min()?
        .finish()?;
    let max_absolute = SignalPipeline::from_view(&stream_context, &mut workspace, source)
        .max_absolute()?
        .finish()?;
    let min_absolute = SignalPipeline::from_view(&stream_context, &mut workspace, source)
        .min_absolute()?
        .finish()?;
    let min_max = SignalPipeline::from_view(&stream_context, &mut workspace, source).min_max()?;
    let max_index =
        SignalPipeline::from_view(&stream_context, &mut workspace, source).max_index()?;
    let min_index =
        SignalPipeline::from_view(&stream_context, &mut workspace, source).min_index()?;
    let max_absolute_index =
        SignalPipeline::from_view(&stream_context, &mut workspace, source).max_absolute_index()?;
    let min_absolute_index =
        SignalPipeline::from_view(&stream_context, &mut workspace, source).min_absolute_index()?;
    let count_in_range = SignalPipeline::from_view(&stream_context, &mut workspace, count_source)
        .count_in_range(-1, 7)?
        .finish()?;
    let norm_diff_inf = SignalPipeline::from_view(&stream_context, &mut workspace, source)
        .norm_diff_inf_to_f32(&other)?
        .finish()?;
    let norm_diff_l1 = SignalPipeline::from_view(&stream_context, &mut workspace, source)
        .norm_diff_l1_to_f32(&other)?
        .finish()?;
    let norm_diff_l2 = SignalPipeline::from_view(&stream_context, &mut workspace, source)
        .norm_diff_l2_to_f32(&other)?
        .finish()?;
    let dot_product_i64 = SignalPipeline::from_view(&stream_context, &mut workspace, source)
        .dot_product_to_i64(&other)?
        .finish()?;
    let dot_product_f32 = SignalPipeline::from_view(&stream_context, &mut workspace, source)
        .dot_product_to_f32(&other)?
        .finish()?;
    let dot_product_complex_i64 =
        SignalPipeline::from_view(&stream_context, &mut workspace, source)
            .dot_product_with_complex_to_i64_complex(&complex_other)?
            .finish()?;
    let dot_product_complex_f32 =
        SignalPipeline::from_view(&stream_context, &mut workspace, source)
            .dot_product_with_complex_to_f32_complex(&complex_other)?
            .finish()?;

    stream.synchronize()?;

    assert_eq!(max.len(), 1);
    assert_eq!(min.len(), 1);
    assert_eq!(max_absolute.len(), 1);
    assert_eq!(min_absolute.len(), 1);
    assert_eq!(min_max.min.len(), 1);
    assert_eq!(min_max.max.len(), 1);
    assert_eq!(max_index.value.len(), 1);
    assert_eq!(max_index.index.len(), 1);
    assert_eq!(min_index.value.len(), 1);
    assert_eq!(min_index.index.len(), 1);
    assert_eq!(max_absolute_index.value.len(), 1);
    assert_eq!(max_absolute_index.index.len(), 1);
    assert_eq!(min_absolute_index.value.len(), 1);
    assert_eq!(min_absolute_index.index.len(), 1);
    assert_eq!(count_in_range.len(), 1);
    assert_eq!(norm_diff_inf.len(), 1);
    assert_eq!(norm_diff_l1.len(), 1);
    assert_eq!(norm_diff_l2.len(), 1);
    assert_eq!(dot_product_i64.len(), 1);
    assert_eq!(dot_product_f32.len(), 1);
    assert_eq!(dot_product_complex_i64.len(), 1);
    assert_eq!(dot_product_complex_f32.len(), 1);
    assert!(workspace.scratch_buffers() >= 1);

    Ok(())
}

#[test]
fn fluent_signal_pipeline_runs_f64_scalar_statistics() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [1.0f64, 2.0, 4.0, 8.0];
    let other_host = [0.5f64, 2.5, 3.0, 10.0];
    let complex_other_host = [
        Complex64::new(1.0, 0.5),
        Complex64::new(2.0, 1.0),
        Complex64::new(4.0, 2.0),
        Complex64::new(8.0, 4.0),
    ];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let other_memory = DeviceMemory::from_slice(&other_host)?;
    let complex_other_memory = DeviceMemory::from_slice(&complex_other_host)?;
    let source = SignalView::from_memory(&source_memory, source_host.len())?;
    let other = SignalView::from_memory(&other_memory, other_host.len())?;
    let complex_other = SignalView::from_memory(&complex_other_memory, complex_other_host.len())?;
    let mut workspace = Workspace::create();

    let mean_is_scalar = matches!(
        SignalPipeline::from_view(&stream_context, &mut workspace, source)
            .mean()?
            .into_backing(),
        SignalBacking::Owned(signal) if signal.len() == 1
    );
    let max_is_scalar = matches!(
        SignalPipeline::from_view(&stream_context, &mut workspace, source)
            .max()?
            .into_backing(),
        SignalBacking::Owned(signal) if signal.len() == 1
    );
    let min_is_scalar = matches!(
        SignalPipeline::from_view(&stream_context, &mut workspace, source)
            .min()?
            .into_backing(),
        SignalBacking::Owned(signal) if signal.len() == 1
    );
    let min_max = SignalPipeline::from_view(&stream_context, &mut workspace, source).min_max()?;
    let mean_standard_deviation =
        SignalPipeline::from_view(&stream_context, &mut workspace, source)
            .mean_standard_deviation()?;
    let max_index =
        SignalPipeline::from_view(&stream_context, &mut workspace, source).max_index()?;
    let min_index =
        SignalPipeline::from_view(&stream_context, &mut workspace, source).min_index()?;
    let norm_diff_l2_is_scalar = matches!(
        SignalPipeline::from_view(&stream_context, &mut workspace, source)
            .norm_diff_l2(&other)?
            .into_backing(),
        SignalBacking::Owned(signal) if signal.len() == 1
    );
    let dot_product_is_scalar = matches!(
        SignalPipeline::from_view(&stream_context, &mut workspace, source)
            .dot_product(&other)?
            .into_backing(),
        SignalBacking::Owned(signal) if signal.len() == 1
    );
    let dot_product_complex_is_scalar = matches!(
        SignalPipeline::from_view(&stream_context, &mut workspace, source)
            .dot_product_with_complex(&complex_other)?
            .into_backing(),
        SignalBacking::Owned(signal) if signal.len() == 1
    );

    stream.synchronize()?;

    assert!(mean_is_scalar);
    assert!(max_is_scalar);
    assert!(min_is_scalar);
    assert_eq!(min_max.min.len(), 1);
    assert_eq!(min_max.max.len(), 1);
    assert_eq!(mean_standard_deviation.mean.len(), 1);
    assert_eq!(mean_standard_deviation.standard_deviation.len(), 1);
    assert_eq!(max_index.value.len(), 1);
    assert_eq!(max_index.index.len(), 1);
    assert_eq!(min_index.value.len(), 1);
    assert_eq!(min_index.index.len(), 1);
    assert!(norm_diff_l2_is_scalar);
    assert!(dot_product_is_scalar);
    assert!(dot_product_complex_is_scalar);
    assert!(workspace.scratch_buffers() >= 1);

    Ok(())
}

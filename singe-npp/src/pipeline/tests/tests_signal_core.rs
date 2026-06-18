use super::*;

#[test]
fn fluent_signal_pipeline_converts_and_runs_in_place_steps() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [1u8, 2, 4, 8];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let source = SignalView::from_memory(&source_memory, source_host.len())?;
    let mut workspace = Workspace::create();

    let signal = SignalPipeline::from_view(&stream_context, &mut workspace, source)
        .convert_to_f32()?
        .zero()?
        .set(2.0)?
        .multiply_constant(2.5)?
        .natural_logarithm()?
        .finish()?;

    stream.synchronize()?;

    assert_eq!(signal.len(), source_host.len());
    assert!(!signal.is_empty());

    Ok(())
}

#[test]
fn fluent_signal_pipeline_runs_complex_fill_steps() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let i64_complex_host = [
        ComplexI64::new(1, -1),
        ComplexI64::new(2, -2),
        ComplexI64::new(3, -3),
        ComplexI64::new(4, -4),
    ];
    let i64_complex_memory = DeviceMemory::from_slice(&i64_complex_host)?;
    let i64_complex_source = SignalView::from_memory(&i64_complex_memory, i64_complex_host.len())?;
    let f32_complex_host = [
        Complex32::new(1.0, -1.0),
        Complex32::new(2.0, -2.0),
        Complex32::new(3.0, -3.0),
        Complex32::new(4.0, -4.0),
    ];
    let f32_complex_memory = DeviceMemory::from_slice(&f32_complex_host)?;
    let f32_complex_source = SignalView::from_memory(&f32_complex_memory, f32_complex_host.len())?;

    let mut i64_complex_workspace = Workspace::create();
    let i64_complex = SignalPipeline::from_view(
        &stream_context,
        &mut i64_complex_workspace,
        i64_complex_source,
    )
    .zero()?
    .set(ComplexI64::new(7, -7))?
    .finish()?;

    let mut f32_complex_copy_workspace = Workspace::create();
    let f32_complex_copy = SignalPipeline::from_view(
        &stream_context,
        &mut f32_complex_copy_workspace,
        f32_complex_source,
    )
    .finish()?;

    let mut f32_complex_fill_workspace = Workspace::create();
    let f32_complex_filled = SignalPipeline::from_view(
        &stream_context,
        &mut f32_complex_fill_workspace,
        f32_complex_source,
    )
    .zero()?
    .set(Complex32::new(0.5, -0.5))?
    .finish()?;

    stream.synchronize()?;

    assert_eq!(i64_complex.len(), i64_complex_host.len());
    assert_eq!(f32_complex_copy.len(), f32_complex_host.len());
    assert_eq!(f32_complex_filled.len(), f32_complex_host.len());

    Ok(())
}

#[test]
fn fluent_signal_pipeline_runs_typed_conversions() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let i8_host = [-4i8, 2, 7, -1];
    let i8_memory = DeviceMemory::from_slice(&i8_host)?;
    let i8_source = SignalView::from_memory(&i8_memory, i8_host.len())?;
    let i16_host = [-4i16, 2, 7, -1];
    let i16_memory = DeviceMemory::from_slice(&i16_host)?;
    let i16_source = SignalView::from_memory(&i16_memory, i16_host.len())?;
    let u16_host = [4u16, 2, 7, 1];
    let u16_memory = DeviceMemory::from_slice(&u16_host)?;
    let u16_source = SignalView::from_memory(&u16_memory, u16_host.len())?;
    let i32_host = [-4i32, 2, 7, -1];
    let i32_memory = DeviceMemory::from_slice(&i32_host)?;
    let i32_source = SignalView::from_memory(&i32_memory, i32_host.len())?;
    let i64_host = [-4i64, 2, 7, -1];
    let i64_memory = DeviceMemory::from_slice(&i64_host)?;
    let i64_source = SignalView::from_memory(&i64_memory, i64_host.len())?;
    let f32_host = [1.0f32, 2.0, 4.0, 8.0];
    let f32_memory = DeviceMemory::from_slice(&f32_host)?;
    let f32_source = SignalView::from_memory(&f32_memory, f32_host.len())?;
    let mut i8_workspace = Workspace::create();

    let i8_signal = SignalPipeline::from_view(&stream_context, &mut i8_workspace, i8_source)
        .convert_to_i16()?
        .convert_to_f32()?
        .finish()?;

    let i8_signal_direct = SignalPipeline::from_view(&stream_context, &mut i8_workspace, i8_source)
        .convert_to_f32()?
        .finish()?;
    let mut i16_workspace = Workspace::create();

    let i16_signal = SignalPipeline::from_view(&stream_context, &mut i16_workspace, i16_source)
        .convert_to_i32()?
        .convert_to_f32_scaled(0)?
        .convert_to_i8_scaled(RoundMode::Near, 0)?
        .convert_to_i16()?
        .finish()?;

    let mut u16_workspace = Workspace::create();
    let u16_signal = SignalPipeline::from_view(&stream_context, &mut u16_workspace, u16_source)
        .convert_to_f32()?
        .finish()?;

    let mut i32_workspace = Workspace::create();
    let i32_signal = SignalPipeline::from_view(&stream_context, &mut i32_workspace, i32_source)
        .convert_to_i16_scaled(0)?
        .convert_to_f32()?
        .convert_to_i32_scaled(RoundMode::Near, 0)?
        .convert_to_f64()?
        .convert_to_i32_scaled(RoundMode::Near, 0)?
        .convert_to_f64_scaled(0)?
        .convert_to_f32()?
        .finish()?;

    let mut i64_workspace = Workspace::create();
    let i64_signal = SignalPipeline::from_view(&stream_context, &mut i64_workspace, i64_source)
        .convert_to_f64()?
        .convert_to_f32()?
        .finish()?;

    let mut f32_workspace = Workspace::create();
    let f32_signal = SignalPipeline::from_view(&stream_context, &mut f32_workspace, f32_source)
        .convert_to_f64()?
        .convert_to_i16_scaled(RoundMode::Near, 0)?
        .convert_to_f64_scaled(0)?
        .convert_to_i32_scaled(RoundMode::Near, 0)?
        .convert_to_f32_scaled(0)?
        .convert_to_i8_scaled(RoundMode::Near, 0)?
        .convert_to_f32()?
        .convert_to_u8_scaled(RoundMode::Near, 0)?
        .convert_to_f32()?
        .finish()?;

    stream.synchronize()?;

    assert_eq!(i8_signal.len(), i8_host.len());
    assert_eq!(i8_signal_direct.len(), i8_host.len());
    assert_eq!(i16_signal.len(), i16_host.len());
    assert_eq!(u16_signal.len(), u16_host.len());
    assert_eq!(i32_signal.len(), i32_host.len());
    assert_eq!(i64_signal.len(), i64_host.len());
    assert_eq!(f32_signal.len(), f32_host.len());

    Ok(())
}

#[test]
fn fluent_signal_pipeline_runs_constant_unary_and_threshold_steps() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [1.0f32, 4.0, 9.0, 16.0];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let source = SignalView::from_memory(&source_memory, source_host.len())?;
    let mut workspace = Workspace::create();

    let signal = SignalPipeline::from_view(&stream_context, &mut workspace, source)
        .add_constant(3.0)?
        .square_root()?
        .square()?
        .subtract_constant(1.0)?
        .divide_constant(2.0)?
        .subtract_from_constant(10.0)?
        .divide_into_constant(100.0)?
        .exponent()?
        .natural_logarithm()?
        .threshold(2.0, ComparisonOperation::Greater)?
        .threshold_less(3.0)?
        .threshold_greater(1.0)?
        .threshold_less_value(2.0, 2.0)?
        .threshold_greater_value(3.0, 3.0)?
        .finish()?;

    stream.synchronize()?;

    assert_eq!(signal.len(), source_host.len());
    assert!(!signal.is_empty());

    Ok(())
}

#[test]
fn fluent_signal_pipeline_runs_additional_unary_steps() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let i32_host = [-8i32, -4, 4, 8];
    let i32_memory = DeviceMemory::from_slice(&i32_host)?;
    let i32_source = SignalView::from_memory(&i32_memory, i32_host.len())?;
    let mut i32_workspace = Workspace::create();

    let i32_signal = SignalPipeline::from_view(&stream_context, &mut i32_workspace, i32_source)
        .absolute()?
        .finish()?;

    let f32_host = [-1.0f32, -0.5, 0.5, 1.0];
    let f32_memory = DeviceMemory::from_slice(&f32_host)?;
    let f32_source = SignalView::from_memory(&f32_memory, f32_host.len())?;
    let mut f32_workspace = Workspace::create();

    let f32_signal = SignalPipeline::from_view(&stream_context, &mut f32_workspace, f32_source)
        .absolute()?
        .arctangent()?
        .cube_root()?
        .cauchy(1.0)?
        .cauchy_derivative(1.0)?
        .exponent_to_f64()?
        .convert_to_f32()?
        .finish()?;

    let f64_host = [1.0f64, 2.0, 4.0, 8.0];
    let f64_memory = DeviceMemory::from_slice(&f64_host)?;
    let f64_source = SignalView::from_memory(&f64_memory, f64_host.len())?;
    let mut f64_workspace = Workspace::create();

    let f64_is_owned = matches!(
        SignalPipeline::from_view(&stream_context, &mut f64_workspace, f64_source)
            .absolute()?
            .arctangent()?
            .natural_logarithm_to_f32()?
            .into_backing(),
        SignalBacking::Owned(signal) if signal.len() == f64_host.len()
    );

    let u8_host = [0b0011_0011u8, 0b1111_0000, 0b0101_0101, 0b1010_1010];
    let u8_memory = DeviceMemory::from_slice(&u8_host)?;
    let u8_source = SignalView::from_memory(&u8_memory, u8_host.len())?;
    let mut u8_workspace = Workspace::create();

    let u8_signal = SignalPipeline::from_view(&stream_context, &mut u8_workspace, u8_source)
        .bit_not()?
        .finish()?;

    let u32_host = [0x0000_ffffu32, 0xffff_0000, 0x0f0f_0f0f, 0xf0f0_f0f0];
    let u32_memory = DeviceMemory::from_slice(&u32_host)?;
    let u32_source = SignalView::from_memory(&u32_memory, u32_host.len())?;
    let mut u32_workspace = Workspace::create();

    let u32_is_owned = matches!(
        SignalPipeline::from_view(&stream_context, &mut u32_workspace, u32_source).bit_not()?.into_backing(),
        SignalBacking::Owned(signal) if signal.len() == u32_host.len()
    );

    stream.synchronize()?;

    assert_eq!(i32_signal.len(), i32_host.len());
    assert_eq!(f32_signal.len(), f32_host.len());
    assert!(f64_is_owned);
    assert_eq!(u8_signal.len(), u8_host.len());
    assert!(u32_is_owned);

    Ok(())
}

#[test]
fn fluent_signal_pipeline_runs_scaled_unary_steps() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let u8_host = [1u8, 4, 9, 16];
    let u8_memory = DeviceMemory::from_slice(&u8_host)?;
    let u8_source = SignalView::from_memory(&u8_memory, u8_host.len())?;
    let mut u8_workspace = Workspace::create();

    let u8_signal = SignalPipeline::from_view(&stream_context, &mut u8_workspace, u8_source)
        .square_scaled(0)?
        .square_root_scaled(0)?
        .finish()?;

    let u16_host = [1u16, 4, 9, 16];
    let u16_memory = DeviceMemory::from_slice(&u16_host)?;
    let u16_source = SignalView::from_memory(&u16_memory, u16_host.len())?;
    let mut u16_workspace = Workspace::create();

    let u16_is_owned = matches!(
        SignalPipeline::from_view(&stream_context, &mut u16_workspace, u16_source)
            .square_scaled(0)?
            .square_root_scaled(0)?
            .into_backing(),
        SignalBacking::Owned(signal) if signal.len() == u16_host.len()
    );

    let i16_host = [1i16, 4, 9, 16];
    let i16_memory = DeviceMemory::from_slice(&i16_host)?;
    let i16_source = SignalView::from_memory(&i16_memory, i16_host.len())?;
    let mut i16_workspace = Workspace::create();

    let i16_signal = SignalPipeline::from_view(&stream_context, &mut i16_workspace, i16_source)
        .exponent_scaled(0)?
        .natural_logarithm_scaled(0)?
        .square_scaled(0)?
        .square_root_scaled(0)?
        .finish()?;

    let i32_host = [1i32, 4, 9, 16];
    let i32_memory = DeviceMemory::from_slice(&i32_host)?;
    let i32_source = SignalView::from_memory(&i32_memory, i32_host.len())?;
    let mut i32_workspace = Workspace::create();

    let i32_signal = SignalPipeline::from_view(&stream_context, &mut i32_workspace, i32_source)
        .exponent_scaled(0)?
        .natural_logarithm_scaled(0)?
        .ten_times_log10_scaled(0)?
        .finish()?;

    let mut i32_mixed_workspace = Workspace::create();
    let i32_mixed_signal =
        SignalPipeline::from_view(&stream_context, &mut i32_mixed_workspace, i32_source)
            .natural_logarithm_to_i16_scaled(0)?
            .convert_to_i32()?
            .cube_root_to_i16_scaled(0)?
            .finish()?;

    let mut i32_square_root_workspace = Workspace::create();
    let i32_square_root_signal =
        SignalPipeline::from_view(&stream_context, &mut i32_square_root_workspace, i32_source)
            .square_root_to_i16_scaled(0)?
            .finish()?;

    let i64_host = [1i64, 4, 9, 16];
    let i64_memory = DeviceMemory::from_slice(&i64_host)?;
    let i64_source = SignalView::from_memory(&i64_memory, i64_host.len())?;
    let mut i64_workspace = Workspace::create();

    let i64_signal = SignalPipeline::from_view(&stream_context, &mut i64_workspace, i64_source)
        .exponent_scaled(0)?
        .square_root_scaled(0)?
        .finish()?;

    let mut i64_mixed_workspace = Workspace::create();
    let i64_mixed_signal =
        SignalPipeline::from_view(&stream_context, &mut i64_mixed_workspace, i64_source)
            .square_root_to_i16_scaled(0)?
            .finish()?;

    stream.synchronize()?;

    assert_eq!(u8_signal.len(), u8_host.len());
    assert!(u16_is_owned);
    assert_eq!(i16_signal.len(), i16_host.len());
    assert_eq!(i32_signal.len(), i32_host.len());
    assert_eq!(i32_mixed_signal.len(), i32_host.len());
    assert_eq!(i32_square_root_signal.len(), i32_host.len());
    assert_eq!(i64_signal.len(), i64_host.len());
    assert_eq!(i64_mixed_signal.len(), i64_host.len());

    Ok(())
}

#[test]
fn fluent_signal_pipeline_runs_normalize_steps() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let f32_host = [1.0f32, 2.0, 4.0, 8.0];
    let f32_memory = DeviceMemory::from_slice(&f32_host)?;
    let f32_source = SignalView::from_memory(&f32_memory, f32_host.len())?;
    let mut f32_workspace = Workspace::create();

    let f32_signal = SignalPipeline::from_view(&stream_context, &mut f32_workspace, f32_source)
        .normalize(1.0, 2.0)?
        .absolute()?
        .finish()?;

    let f64_host = [1.0f64, 2.0, 4.0, 8.0];
    let f64_memory = DeviceMemory::from_slice(&f64_host)?;
    let f64_source = SignalView::from_memory(&f64_memory, f64_host.len())?;
    let mut f64_workspace = Workspace::create();

    let f64_is_owned = matches!(
        SignalPipeline::from_view(&stream_context, &mut f64_workspace, f64_source)
            .normalize(1.0, 2.0)?
            .absolute()?
            .into_backing(),
        SignalBacking::Owned(signal) if signal.len() == f64_host.len()
    );

    let i16_host = [1i16, 2, 4, 8];
    let i16_memory = DeviceMemory::from_slice(&i16_host)?;
    let i16_source = SignalView::from_memory(&i16_memory, i16_host.len())?;
    let mut i16_workspace = Workspace::create();

    let i16_signal = SignalPipeline::from_view(&stream_context, &mut i16_workspace, i16_source)
        .normalize_scaled(1, 2, 0)?
        .absolute()?
        .finish()?;

    let f32_complex_host = [
        Complex32::new(1.0, 0.5),
        Complex32::new(2.0, 1.0),
        Complex32::new(4.0, 2.0),
        Complex32::new(8.0, 4.0),
    ];
    let f32_complex_memory = DeviceMemory::from_slice(&f32_complex_host)?;
    let f32_complex_source = SignalView::from_memory(&f32_complex_memory, f32_complex_host.len())?;
    let mut f32_complex_workspace = Workspace::create();

    let f32_complex_signal = SignalPipeline::from_view(
        &stream_context,
        &mut f32_complex_workspace,
        f32_complex_source,
    )
    .normalize_complex(Complex32::new(1.0, 0.0), 2.0)?
    .finish()?;

    let f64_complex_host = [
        Complex64::new(1.0, 0.5),
        Complex64::new(2.0, 1.0),
        Complex64::new(4.0, 2.0),
        Complex64::new(8.0, 4.0),
    ];
    let f64_complex_memory = DeviceMemory::from_slice(&f64_complex_host)?;
    let f64_complex_source = SignalView::from_memory(&f64_complex_memory, f64_complex_host.len())?;
    let mut f64_complex_workspace = Workspace::create();

    let f64_complex_signal = SignalPipeline::from_view(
        &stream_context,
        &mut f64_complex_workspace,
        f64_complex_source,
    )
    .normalize_complex(Complex64::new(1.0, 0.0), 2.0)?
    .finish()?;

    let i16_complex_host = [
        ComplexI16::new(2, 1),
        ComplexI16::new(4, 2),
        ComplexI16::new(8, 4),
        ComplexI16::new(16, 8),
    ];
    let i16_complex_memory = DeviceMemory::from_slice(&i16_complex_host)?;
    let i16_complex_source = SignalView::from_memory(&i16_complex_memory, i16_complex_host.len())?;
    let mut i16_complex_workspace = Workspace::create();

    let i16_complex_signal = SignalPipeline::from_view(
        &stream_context,
        &mut i16_complex_workspace,
        i16_complex_source,
    )
    .normalize_scaled(ComplexI16::new(1, 0), 2, 0)?
    .finish()?;

    stream.synchronize()?;

    assert_eq!(f32_signal.len(), f32_host.len());
    assert!(f64_is_owned);
    assert_eq!(i16_signal.len(), i16_host.len());
    assert_eq!(f32_complex_signal.len(), f32_complex_host.len());
    assert_eq!(f64_complex_signal.len(), f64_complex_host.len());
    assert_eq!(i16_complex_signal.len(), i16_complex_host.len());

    Ok(())
}

#[test]
fn fluent_signal_pipeline_runs_binary_arithmetic_steps() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [2.0f32, 4.0, 8.0, 16.0];
    let other_host = [1.0f32, 2.0, 4.0, 8.0];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let other_memory = DeviceMemory::from_slice(&other_host)?;
    let source = SignalView::from_memory(&source_memory, source_host.len())?;
    let other = SignalView::from_memory(&other_memory, other_host.len())?;
    let mut workspace = Workspace::create();

    let signal = SignalPipeline::from_view(&stream_context, &mut workspace, source)
        .add_signal(&other)?
        .subtract(&other)?
        .multiply(&other)?
        .divide(&other)?
        .finish()?;

    let i16_source_host = [2i16, 4, 8, 16];
    let i16_other_host = [1i16, 2, 4, 8];
    let i16_source_memory = DeviceMemory::from_slice(&i16_source_host)?;
    let i16_other_memory = DeviceMemory::from_slice(&i16_other_host)?;
    let i16_source = SignalView::from_memory(&i16_source_memory, i16_source_host.len())?;
    let i16_other = SignalView::from_memory(&i16_other_memory, i16_other_host.len())?;
    let mut i16_workspace = Workspace::create();

    let i16_signal = SignalPipeline::from_view(&stream_context, &mut i16_workspace, i16_source)
        .add_signal(&i16_other)?
        .subtract(&i16_other)?
        .multiply(&i16_other)?
        .finish()?;

    let f64_source_host = [2.0f64, 4.0, 8.0, 16.0];
    let f64_other_host = [1.0f64, 2.0, 4.0, 8.0];
    let f64_source_memory = DeviceMemory::from_slice(&f64_source_host)?;
    let f64_other_memory = DeviceMemory::from_slice(&f64_other_host)?;
    let f64_source = SignalView::from_memory(&f64_source_memory, f64_source_host.len())?;
    let f64_other = SignalView::from_memory(&f64_other_memory, f64_other_host.len())?;
    let mut f64_workspace = Workspace::create();

    let f64_is_owned = matches!(
        SignalPipeline::from_view(&stream_context, &mut f64_workspace, f64_source)
            .add_signal(&f64_other)?
            .subtract(&f64_other)?
            .multiply(&f64_other)?
            .divide(&f64_other)?
            .into_backing(),
        SignalBacking::Owned(signal) if signal.len() == f64_source_host.len()
    );

    stream.synchronize()?;

    assert_eq!(signal.len(), source_host.len());
    assert_eq!(i16_signal.len(), i16_source_host.len());
    assert!(f64_is_owned);

    Ok(())
}

#[test]
fn fluent_signal_pipeline_runs_complex_arithmetic_steps() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [
        Complex32::new(2.0, 1.0),
        Complex32::new(4.0, 2.0),
        Complex32::new(8.0, 4.0),
        Complex32::new(16.0, 8.0),
    ];
    let other_host = [
        Complex32::new(1.0, 0.0),
        Complex32::new(2.0, 0.0),
        Complex32::new(4.0, 0.0),
        Complex32::new(8.0, 0.0),
    ];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let other_memory = DeviceMemory::from_slice(&other_host)?;
    let source = SignalView::from_memory(&source_memory, source_host.len())?;
    let other = SignalView::from_memory(&other_memory, other_host.len())?;
    let mut workspace = Workspace::create();

    let signal = SignalPipeline::from_view(&stream_context, &mut workspace, source)
        .add_constant(Complex32::new(1.0, 0.0))?
        .subtract_constant(Complex32::new(1.0, 0.0))?
        .multiply_constant(Complex32::new(1.0, 0.0))?
        .divide_constant(Complex32::new(1.0, 0.0))?
        .threshold_complex(2.0, ComparisonOperation::Greater)?
        .threshold_less_complex(1.0)?
        .threshold_greater_complex(32.0)?
        .threshold_less_value_complex(1.0, Complex32::new(1.0, 0.0))?
        .threshold_greater_value_complex(32.0, Complex32::new(32.0, 0.0))?
        .add_signal(&other)?
        .subtract(&other)?
        .multiply(&other)?
        .divide(&other)?
        .finish()?;

    let mut f32_complex_stats_workspace = Workspace::create();
    let f32_complex_sum =
        SignalPipeline::from_view(&stream_context, &mut f32_complex_stats_workspace, source)
            .sum_complex()?
            .finish()?;
    let f32_complex_mean =
        SignalPipeline::from_view(&stream_context, &mut f32_complex_stats_workspace, source)
            .mean_complex()?
            .finish()?;
    let f32_complex_norm_inf =
        SignalPipeline::from_view(&stream_context, &mut f32_complex_stats_workspace, source)
            .norm_inf_to_f32()?
            .finish()?;
    let f32_complex_norm_l1 =
        SignalPipeline::from_view(&stream_context, &mut f32_complex_stats_workspace, source)
            .norm_l1_to_f64()?
            .finish()?;
    let f32_complex_norm_l2 =
        SignalPipeline::from_view(&stream_context, &mut f32_complex_stats_workspace, source)
            .norm_l2_to_f64()?
            .finish()?;
    let f32_complex_norm_diff_inf =
        SignalPipeline::from_view(&stream_context, &mut f32_complex_stats_workspace, source)
            .norm_diff_inf_to_f32(&other)?
            .finish()?;
    let f32_complex_norm_diff_l1 =
        SignalPipeline::from_view(&stream_context, &mut f32_complex_stats_workspace, source)
            .norm_diff_l1_to_f64(&other)?
            .finish()?;
    let f32_complex_norm_diff_l2 =
        SignalPipeline::from_view(&stream_context, &mut f32_complex_stats_workspace, source)
            .norm_diff_l2_to_f64(&other)?
            .finish()?;
    let f32_complex_dot_product =
        SignalPipeline::from_view(&stream_context, &mut f32_complex_stats_workspace, source)
            .dot_product_complex(&other)?
            .finish()?;
    let f32_complex_dot_product_f64 =
        SignalPipeline::from_view(&stream_context, &mut f32_complex_stats_workspace, source)
            .dot_product_to_f64_complex(&other)?
            .finish()?;

    let f64_source_host = [
        Complex64::new(2.0, 1.0),
        Complex64::new(4.0, 2.0),
        Complex64::new(8.0, 4.0),
        Complex64::new(16.0, 8.0),
    ];
    let f64_other_host = [
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(4.0, 0.0),
        Complex64::new(8.0, 0.0),
    ];
    let f64_source_memory = DeviceMemory::from_slice(&f64_source_host)?;
    let f64_other_memory = DeviceMemory::from_slice(&f64_other_host)?;
    let f64_source = SignalView::from_memory(&f64_source_memory, f64_source_host.len())?;
    let f64_other = SignalView::from_memory(&f64_other_memory, f64_other_host.len())?;
    let mut f64_workspace = Workspace::create();

    let f64_signal = SignalPipeline::from_view(&stream_context, &mut f64_workspace, f64_source)
        .add_constant(Complex64::new(1.0, 0.0))?
        .subtract_constant(Complex64::new(1.0, 0.0))?
        .multiply_constant(Complex64::new(1.0, 0.0))?
        .divide_constant(Complex64::new(1.0, 0.0))?
        .threshold_complex(2.0, ComparisonOperation::Greater)?
        .threshold_less_complex(1.0)?
        .threshold_greater_complex(32.0)?
        .threshold_less_value_complex(1.0, Complex64::new(1.0, 0.0))?
        .threshold_greater_value_complex(32.0, Complex64::new(32.0, 0.0))?
        .add_signal(&f64_other)?
        .subtract(&f64_other)?
        .multiply(&f64_other)?
        .divide(&f64_other)?
        .finish()?;

    let mut f64_complex_stats_workspace = Workspace::create();
    let f64_complex_sum = SignalPipeline::from_view(
        &stream_context,
        &mut f64_complex_stats_workspace,
        f64_source,
    )
    .sum_complex()?
    .finish()?;
    let f64_complex_mean = SignalPipeline::from_view(
        &stream_context,
        &mut f64_complex_stats_workspace,
        f64_source,
    )
    .mean_complex()?
    .finish()?;
    let f64_complex_norm_inf = SignalPipeline::from_view(
        &stream_context,
        &mut f64_complex_stats_workspace,
        f64_source,
    )
    .norm_inf_to_f64()?
    .finish()?;
    let f64_complex_norm_l1 = SignalPipeline::from_view(
        &stream_context,
        &mut f64_complex_stats_workspace,
        f64_source,
    )
    .norm_l1_to_f64()?
    .finish()?;
    let f64_complex_norm_l2 = SignalPipeline::from_view(
        &stream_context,
        &mut f64_complex_stats_workspace,
        f64_source,
    )
    .norm_l2_to_f64()?
    .finish()?;
    let f64_complex_norm_diff_inf = SignalPipeline::from_view(
        &stream_context,
        &mut f64_complex_stats_workspace,
        f64_source,
    )
    .norm_diff_inf_to_f64(&f64_other)?
    .finish()?;
    let f64_complex_norm_diff_l1 = SignalPipeline::from_view(
        &stream_context,
        &mut f64_complex_stats_workspace,
        f64_source,
    )
    .norm_diff_l1_to_f64(&f64_other)?
    .finish()?;
    let f64_complex_norm_diff_l2 = SignalPipeline::from_view(
        &stream_context,
        &mut f64_complex_stats_workspace,
        f64_source,
    )
    .norm_diff_l2_to_f64(&f64_other)?
    .finish()?;
    let f64_complex_dot_product = SignalPipeline::from_view(
        &stream_context,
        &mut f64_complex_stats_workspace,
        f64_source,
    )
    .dot_product_complex(&f64_other)?
    .finish()?;

    stream.synchronize()?;

    assert_eq!(signal.len(), source_host.len());
    assert_eq!(f32_complex_sum.len(), 1);
    assert_eq!(f32_complex_mean.len(), 1);
    assert_eq!(f32_complex_norm_inf.len(), 1);
    assert_eq!(f32_complex_norm_l1.len(), 1);
    assert_eq!(f32_complex_norm_l2.len(), 1);
    assert_eq!(f32_complex_norm_diff_inf.len(), 1);
    assert_eq!(f32_complex_norm_diff_l1.len(), 1);
    assert_eq!(f32_complex_norm_diff_l2.len(), 1);
    assert_eq!(f32_complex_dot_product.len(), 1);
    assert_eq!(f32_complex_dot_product_f64.len(), 1);
    assert_eq!(f64_signal.len(), f64_source_host.len());
    assert_eq!(f64_complex_sum.len(), 1);
    assert_eq!(f64_complex_mean.len(), 1);
    assert_eq!(f64_complex_norm_inf.len(), 1);
    assert_eq!(f64_complex_norm_l1.len(), 1);
    assert_eq!(f64_complex_norm_l2.len(), 1);
    assert_eq!(f64_complex_norm_diff_inf.len(), 1);
    assert_eq!(f64_complex_norm_diff_l1.len(), 1);
    assert_eq!(f64_complex_norm_diff_l2.len(), 1);
    assert_eq!(f64_complex_dot_product.len(), 1);

    Ok(())
}

#[test]
fn fluent_signal_pipeline_runs_integer_complex_scaled_steps() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [
        ComplexI16::new(2, 1),
        ComplexI16::new(4, 2),
        ComplexI16::new(8, 4),
        ComplexI16::new(16, 8),
    ];
    let other_host = [
        ComplexI16::new(1, 0),
        ComplexI16::new(2, 0),
        ComplexI16::new(4, 0),
        ComplexI16::new(8, 0),
    ];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let other_memory = DeviceMemory::from_slice(&other_host)?;
    let source = SignalView::from_memory(&source_memory, source_host.len())?;
    let other = SignalView::from_memory(&other_memory, other_host.len())?;
    let mut workspace = Workspace::create();

    let signal = SignalPipeline::from_view(&stream_context, &mut workspace, source)
        .add_constant_scaled(ComplexI16::new(1, 0), 0)?
        .subtract_constant_scaled(ComplexI16::new(1, 0), 0)?
        .subtract_from_constant_scaled(ComplexI16::new(32, 0), 0)?
        .multiply_constant_scaled(ComplexI16::new(1, 0), 0)?
        .divide_constant_scaled(ComplexI16::new(1, 0), 0)?
        .threshold_complex(2, ComparisonOperation::Greater)?
        .threshold_less_complex(1)?
        .threshold_greater_complex(32)?
        .threshold_less_value_complex(1, ComplexI16::new(1, 0))?
        .threshold_greater_value_complex(32, ComplexI16::new(32, 0))?
        .add_scaled(&other, 0)?
        .subtract_scaled(&other, 0)?
        .multiply_scaled(&other, 0)?
        .divide_scaled(&other, 0)?
        .square_scaled(0)?
        .square_root_scaled(0)?
        .finish()?;

    let i32_source_host = [
        ComplexI32::new(2, 1),
        ComplexI32::new(4, 2),
        ComplexI32::new(8, 4),
        ComplexI32::new(16, 8),
    ];
    let i32_other_host = [
        ComplexI32::new(1, 0),
        ComplexI32::new(2, 0),
        ComplexI32::new(4, 0),
        ComplexI32::new(8, 0),
    ];
    let i32_source_memory = DeviceMemory::from_slice(&i32_source_host)?;
    let i32_other_memory = DeviceMemory::from_slice(&i32_other_host)?;
    let i32_source = SignalView::from_memory(&i32_source_memory, i32_source_host.len())?;
    let i32_other = SignalView::from_memory(&i32_other_memory, i32_other_host.len())?;
    let mut i32_workspace = Workspace::create();

    let i32_signal = SignalPipeline::from_view(&stream_context, &mut i32_workspace, i32_source)
        .add_constant_scaled(ComplexI32::new(1, 0), 0)?
        .subtract_constant_scaled(ComplexI32::new(1, 0), 0)?
        .subtract_from_constant_scaled(ComplexI32::new(32, 0), 0)?
        .multiply_constant_scaled(ComplexI32::new(1, 0), 0)?
        .add_scaled(&i32_other, 0)?
        .subtract_scaled(&i32_other, 0)?
        .multiply_scaled(&i32_other, 0)?
        .finish()?;

    stream.synchronize()?;

    assert_eq!(signal.len(), source_host.len());
    assert_eq!(i32_signal.len(), i32_source_host.len());

    Ok(())
}

#[test]
fn fluent_signal_pipeline_runs_mixed_binary_arithmetic_steps() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let u8_source_host = [2u8, 4, 8, 16];
    let u8_other_host = [1u8, 2, 4, 8];
    let u8_source_memory = DeviceMemory::from_slice(&u8_source_host)?;
    let u8_other_memory = DeviceMemory::from_slice(&u8_other_host)?;
    let u8_source = SignalView::from_memory(&u8_source_memory, u8_source_host.len())?;
    let u8_other = SignalView::from_memory(&u8_other_memory, u8_other_host.len())?;
    let mut u8_add_workspace = Workspace::create();
    let mut u8_multiply_workspace = Workspace::create();

    let u16_add_is_owned = matches!(
        SignalPipeline::from_view(&stream_context, &mut u8_add_workspace, u8_source)
            .add_to_u16(&u8_other)?
            .into_backing(),
        SignalBacking::Owned(signal) if signal.len() == u8_source_host.len()
    );
    let u16_multiply_is_owned = matches!(
        SignalPipeline::from_view(&stream_context, &mut u8_multiply_workspace, u8_source)
            .multiply_to_u16(&u8_other)?
            .into_backing(),
        SignalBacking::Owned(signal) if signal.len() == u8_source_host.len()
    );

    let i16_source_host = [2i16, 4, 8, 16];
    let i16_other_host = [1i16, 2, 4, 8];
    let i16_source_memory = DeviceMemory::from_slice(&i16_source_host)?;
    let i16_other_memory = DeviceMemory::from_slice(&i16_other_host)?;
    let i16_source = SignalView::from_memory(&i16_source_memory, i16_source_host.len())?;
    let i16_other = SignalView::from_memory(&i16_other_memory, i16_other_host.len())?;
    let mut i16_add_workspace = Workspace::create();
    let mut i16_subtract_workspace = Workspace::create();
    let mut i16_multiply_workspace = Workspace::create();
    let mut i16_scaled_workspace = Workspace::create();

    let f32_add_signal =
        SignalPipeline::from_view(&stream_context, &mut i16_add_workspace, i16_source)
            .add_to_f32(&i16_other)?
            .finish()?;
    let f32_subtract_signal =
        SignalPipeline::from_view(&stream_context, &mut i16_subtract_workspace, i16_source)
            .subtract_to_f32(&i16_other)?
            .finish()?;
    let f32_multiply_signal =
        SignalPipeline::from_view(&stream_context, &mut i16_multiply_workspace, i16_source)
            .multiply_to_f32(&i16_other)?
            .finish()?;
    let i32_scaled_signal =
        SignalPipeline::from_view(&stream_context, &mut i16_scaled_workspace, i16_source)
            .multiply_to_i32_scaled(&i16_other, 0)?
            .finish()?;

    let u16_source_host = [2u16, 4, 8, 16];
    let u16_source_memory = DeviceMemory::from_slice(&u16_source_host)?;
    let u16_source = SignalView::from_memory(&u16_source_memory, u16_source_host.len())?;
    let mut u16_workspace = Workspace::create();

    let i16_from_u16_signal =
        SignalPipeline::from_view(&stream_context, &mut u16_workspace, u16_source)
            .multiply_with_i16_scaled(&i16_other, 0)?
            .finish()?;

    stream.synchronize()?;

    assert!(u16_add_is_owned);
    assert!(u16_multiply_is_owned);
    assert_eq!(f32_add_signal.len(), i16_source_host.len());
    assert_eq!(f32_subtract_signal.len(), i16_source_host.len());
    assert_eq!(f32_multiply_signal.len(), i16_source_host.len());
    assert_eq!(i32_scaled_signal.len(), i16_source_host.len());
    assert_eq!(i16_from_u16_signal.len(), u16_source_host.len());

    Ok(())
}

#[test]
fn fluent_signal_pipeline_runs_add_product_steps() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let left_host = [1.0f32, 2.0, 3.0, 4.0];
    let right_host = [2.0f32, 3.0, 4.0, 5.0];
    let destination_host = [10.0f32, 20.0, 30.0, 40.0];
    let left_memory = DeviceMemory::from_slice(&left_host)?;
    let right_memory = DeviceMemory::from_slice(&right_host)?;
    let destination_memory = DeviceMemory::from_slice(&destination_host)?;
    let left = SignalView::from_memory(&left_memory, left_host.len())?;
    let right = SignalView::from_memory(&right_memory, right_host.len())?;
    let destination = SignalView::from_memory(&destination_memory, destination_host.len())?;
    let mut workspace = Workspace::create();

    let signal = SignalPipeline::from_view(&stream_context, &mut workspace, destination)
        .add_product(&left, &right)?
        .finish()?;

    let i16_left_host = [1i16, 2, 3, 4];
    let i16_right_host = [2i16, 3, 4, 5];
    let i16_destination_host = [10i16, 20, 30, 40];
    let i16_left_memory = DeviceMemory::from_slice(&i16_left_host)?;
    let i16_right_memory = DeviceMemory::from_slice(&i16_right_host)?;
    let i16_destination_memory = DeviceMemory::from_slice(&i16_destination_host)?;
    let i16_left = SignalView::from_memory(&i16_left_memory, i16_left_host.len())?;
    let i16_right = SignalView::from_memory(&i16_right_memory, i16_right_host.len())?;
    let i16_destination =
        SignalView::from_memory(&i16_destination_memory, i16_destination_host.len())?;
    let mut i16_workspace = Workspace::create();

    let i16_signal =
        SignalPipeline::from_view(&stream_context, &mut i16_workspace, i16_destination)
            .add_product_scaled(&i16_left, &i16_right, 0)?
            .finish()?;

    let i32_left_host = [1i32, 2, 3, 4];
    let i32_right_host = [2i32, 3, 4, 5];
    let i32_destination_host = [10i32, 20, 30, 40];
    let i32_left_memory = DeviceMemory::from_slice(&i32_left_host)?;
    let i32_right_memory = DeviceMemory::from_slice(&i32_right_host)?;
    let i32_destination_memory = DeviceMemory::from_slice(&i32_destination_host)?;
    let i32_left = SignalView::from_memory(&i32_left_memory, i32_left_host.len())?;
    let i32_right = SignalView::from_memory(&i32_right_memory, i32_right_host.len())?;
    let i32_destination =
        SignalView::from_memory(&i32_destination_memory, i32_destination_host.len())?;
    let mut i32_workspace = Workspace::create();

    let i32_signal =
        SignalPipeline::from_view(&stream_context, &mut i32_workspace, i32_destination)
            .add_product_scaled(&i32_left, &i32_right, 0)?
            .finish()?;

    let i32_mixed_destination_host = [10i32, 20, 30, 40];
    let i32_mixed_destination_memory = DeviceMemory::from_slice(&i32_mixed_destination_host)?;
    let i32_mixed_destination = SignalView::from_memory(
        &i32_mixed_destination_memory,
        i32_mixed_destination_host.len(),
    )?;
    let mut i32_mixed_workspace = Workspace::create();

    let i32_mixed_signal = SignalPipeline::from_view(
        &stream_context,
        &mut i32_mixed_workspace,
        i32_mixed_destination,
    )
    .add_product_i16_scaled(&i16_left, &i16_right, 0)?
    .finish()?;

    stream.synchronize()?;

    assert_eq!(signal.len(), destination_host.len());
    assert_eq!(i16_signal.len(), i16_destination_host.len());
    assert_eq!(i32_signal.len(), i32_destination_host.len());
    assert_eq!(i32_mixed_signal.len(), i32_mixed_destination_host.len());

    Ok(())
}

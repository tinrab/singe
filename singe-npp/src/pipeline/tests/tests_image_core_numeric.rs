use super::*;

#[test]
fn fluent_image_pipeline_converts_u8_images_to_f32() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [1u8, 2, 3, 4];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(2, 2))?;
    let mut workspace = Workspace::create();

    let image = ImagePipeline::from_view(&stream_context, &mut workspace, source)
        .scale_to_f32(0.0, 255.0)?
        .scale_to_u8(0.0, 255.0)?
        .scale_to_u16()?
        .threshold(1, ComparisonOperation::Greater)?
        .scale_to_u8(HintAlgorithm::Accurate)?
        .scale_to_i16()?
        .threshold(1, ComparisonOperation::Greater)?
        .scale_to_u8(HintAlgorithm::Accurate)?
        .scale_to_i32()?
        .scale_to_u8(HintAlgorithm::Accurate)?
        .convert_to_f32()?
        .threshold(1.0, ComparisonOperation::Greater)?
        .resize(
            Size {
                width: 3,
                height: 3,
            },
            InterpolationMode::Linear,
        )?
        .duplicate_to_c3()?
        .threshold_channels([1.0, 1.0, 1.0], ComparisonOperation::Greater)?
        .set_channel(3.0, 1)?
        .copy_channel(1, 2)?
        .extract_channel(2)?
        .duplicate_to_c4()?
        .add_constant([1.0, 1.0, 1.0, 1.0], 0)?
        .multiply_constant([2.0, 2.0, 2.0, 2.0], 0)?
        .subtract_constant([1.0, 1.0, 1.0, 1.0], 0)?
        .divide_constant([2.0, 2.0, 2.0, 2.0], 0)?
        .finish()?;

    stream.synchronize()?;

    assert_eq!(
        image.size(),
        Size {
            width: 3,
            height: 3,
        }
    );
    assert!(image.step() >= 36);

    Ok(())
}

#[test]
fn fluent_image_pipeline_runs_complex_linear_steps() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [
        Complex32::new(3.0, 4.0),
        Complex32::new(5.0, 12.0),
        Complex32::new(8.0, 15.0),
        Complex32::new(7.0, 24.0),
    ];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(2, 2))?;
    let mut magnitude_workspace = Workspace::create();

    let magnitude = ImagePipeline::from_view(&stream_context, &mut magnitude_workspace, source)
        .magnitude()?
        .threshold(5.0, ComparisonOperation::Greater)?
        .finish()?;

    let mut squared_workspace = Workspace::create();
    let squared = ImagePipeline::from_view(&stream_context, &mut squared_workspace, source)
        .magnitude_squared()?
        .scale_to_u8(0.0, 625.0)?
        .finish()?;

    stream.synchronize()?;

    assert_eq!(
        magnitude.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert_eq!(
        squared.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert!(magnitude.step() >= 8);
    assert!(squared.step() >= 2);

    Ok(())
}

#[test]
fn fluent_image_pipeline_runs_complex_binary_arithmetic_steps() -> Result<()> {
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
    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(2, 2))?;
    let other = ImageView::<_, C1>::from_memory(&other_memory, Size::new(2, 2))?;
    let mut workspace = Workspace::create();

    let image = ImagePipeline::from_view(&stream_context, &mut workspace, source)
        .add(&other, 0)?
        .subtract(&other, 0)?
        .multiply(&other, 0)?
        .divide(&other, 0)?
        .finish()?;

    stream.synchronize()?;

    assert_eq!(
        image.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert!(image.step() >= (2 * size_of::<Complex32>()) as i32);

    Ok(())
}

#[test]
fn fluent_image_pipeline_runs_integer_complex_binary_arithmetic_steps() -> Result<()> {
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
    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(2, 2))?;
    let other = ImageView::<_, C1>::from_memory(&other_memory, Size::new(2, 2))?;
    let mut workspace = Workspace::create();

    let image = ImagePipeline::from_view(&stream_context, &mut workspace, source)
        .add(&other, 0)?
        .subtract(&other, 0)?
        .multiply(&other, 0)?
        .divide(&other, 0)?
        .finish()?;

    stream.synchronize()?;

    assert_eq!(
        image.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert!(image.step() >= (2 * size_of::<ComplexI16>()) as i32);

    Ok(())
}

#[test]
fn fluent_image_pipeline_runs_template_matching_steps() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let template_host = [1.0f32, 2.0, 4.0, 5.0];
    let source_c3_host: Vec<f32> = source_host
        .iter()
        .flat_map(|value| [*value, *value + 1.0, *value + 2.0])
        .collect();
    let template_c3_host: Vec<f32> = template_host
        .iter()
        .flat_map(|value| [*value, *value + 1.0, *value + 2.0])
        .collect();
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let template_memory = DeviceMemory::from_slice(&template_host)?;
    let source_u8_host = [1u8, 2, 3, 4, 5, 6, 7, 8, 9];
    let template_u8_host = [1u8, 2, 4, 5];
    let source_i8_host = [-4i8, -3, -2, -1, 0, 1, 2, 3, 4];
    let template_i8_host = [-4i8, -3, -1, 0];
    let source_u16_host = [1u16, 2, 3, 4, 5, 6, 7, 8, 9];
    let template_u16_host = [1u16, 2, 4, 5];
    let source_u8_memory = DeviceMemory::from_slice(&source_u8_host)?;
    let template_u8_memory = DeviceMemory::from_slice(&template_u8_host)?;
    let source_i8_memory = DeviceMemory::from_slice(&source_i8_host)?;
    let template_i8_memory = DeviceMemory::from_slice(&template_i8_host)?;
    let source_u16_memory = DeviceMemory::from_slice(&source_u16_host)?;
    let template_u16_memory = DeviceMemory::from_slice(&template_u16_host)?;
    let source_c3_memory = DeviceMemory::from_slice(&source_c3_host)?;
    let template_c3_memory = DeviceMemory::from_slice(&template_c3_host)?;
    let source_u8_c3_host: Vec<u8> = source_u8_host
        .iter()
        .flat_map(|value| [*value, value.saturating_add(1), value.saturating_add(2)])
        .collect();
    let template_u8_c3_host: Vec<u8> = template_u8_host
        .iter()
        .flat_map(|value| [*value, value.saturating_add(1), value.saturating_add(2)])
        .collect();
    let source_i8_c3_host: Vec<i8> = source_i8_host
        .iter()
        .flat_map(|value| [*value, value.saturating_add(1), value.saturating_add(2)])
        .collect();
    let template_i8_c3_host: Vec<i8> = template_i8_host
        .iter()
        .flat_map(|value| [*value, value.saturating_add(1), value.saturating_add(2)])
        .collect();
    let source_u8_c3_memory = DeviceMemory::from_slice(&source_u8_c3_host)?;
    let template_u8_c3_memory = DeviceMemory::from_slice(&template_u8_c3_host)?;
    let source_i8_c3_memory = DeviceMemory::from_slice(&source_i8_c3_host)?;
    let template_i8_c3_memory = DeviceMemory::from_slice(&template_i8_c3_host)?;
    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(3, 3))?;
    let template = ImageView::<_, C1>::from_memory(&template_memory, Size::new(2, 2))?;
    let source_u8 = ImageView::<_, C1>::from_memory(&source_u8_memory, Size::new(3, 3))?;
    let template_u8 = ImageView::<_, C1>::from_memory(&template_u8_memory, Size::new(2, 2))?;
    let source_i8 = ImageView::<_, C1>::from_memory(&source_i8_memory, Size::new(3, 3))?;
    let template_i8 = ImageView::<_, C1>::from_memory(&template_i8_memory, Size::new(2, 2))?;
    let source_u16 = ImageView::<_, C1>::from_memory(&source_u16_memory, Size::new(3, 3))?;
    let template_u16 = ImageView::<_, C1>::from_memory(&template_u16_memory, Size::new(2, 2))?;
    let source_c3 = ImageView::<_, C3>::from_memory(&source_c3_memory, Size::new(3, 3))?;
    let template_c3 = ImageView::<_, C3>::from_memory(&template_c3_memory, Size::new(2, 2))?;
    let source_u8_c3 = ImageView::<_, C3>::from_memory(&source_u8_c3_memory, Size::new(3, 3))?;
    let template_u8_c3 = ImageView::<_, C3>::from_memory(&template_u8_c3_memory, Size::new(2, 2))?;
    let source_i8_c3 = ImageView::<_, C3>::from_memory(&source_i8_c3_memory, Size::new(3, 3))?;
    let template_i8_c3 = ImageView::<_, C3>::from_memory(&template_i8_c3_memory, Size::new(2, 2))?;
    let mut full_workspace = Workspace::create();
    let mut same_workspace = Workspace::create();
    let mut valid_workspace = Workspace::create();
    let mut c3_workspace = Workspace::create();
    let mut u8_full_workspace = Workspace::create();
    let mut u8_same_workspace = Workspace::create();
    let mut u8_valid_workspace = Workspace::create();
    let mut u8_c3_workspace = Workspace::create();
    let mut i8_workspace = Workspace::create();
    let mut i8_c3_workspace = Workspace::create();
    let mut u8_scaled_full_workspace = Workspace::create();
    let mut u8_scaled_same_workspace = Workspace::create();
    let mut u8_scaled_valid_workspace = Workspace::create();
    let mut u8_scaled_c3_workspace = Workspace::create();
    let mut correlation_full_workspace = Workspace::create();
    let mut correlation_same_workspace = Workspace::create();
    let mut correlation_valid_workspace = Workspace::create();
    let mut correlation_u8_workspace = Workspace::create();
    let mut correlation_i8_workspace = Workspace::create();
    let mut correlation_u16_workspace = Workspace::create();
    let mut correlation_scaled_full_workspace = Workspace::create();
    let mut correlation_scaled_same_workspace = Workspace::create();
    let mut correlation_scaled_valid_workspace = Workspace::create();
    let mut correlation_scaled_c3_workspace = Workspace::create();

    let full = ImagePipeline::from_view(&stream_context, &mut full_workspace, source)
        .square_distance_full_norm(&template)?
        .finish()?;
    let same = ImagePipeline::from_view(&stream_context, &mut same_workspace, source)
        .square_distance_same_norm(&template)?
        .finish()?;
    let valid = ImagePipeline::from_view(&stream_context, &mut valid_workspace, source)
        .square_distance_valid_norm(&template)?
        .finish()?;
    let c3_valid = ImagePipeline::from_view(&stream_context, &mut c3_workspace, source_c3)
        .square_distance_valid_norm(&template_c3)?
        .finish()?;
    let u8_full = ImagePipeline::from_view(&stream_context, &mut u8_full_workspace, source_u8)
        .square_distance_full_norm_to_f32(&template_u8)?
        .finish()?;
    let u8_same = ImagePipeline::from_view(&stream_context, &mut u8_same_workspace, source_u8)
        .square_distance_same_norm_to_f32(&template_u8)?
        .finish()?;
    let u8_valid = ImagePipeline::from_view(&stream_context, &mut u8_valid_workspace, source_u8)
        .square_distance_valid_norm_to_f32(&template_u8)?
        .finish()?;
    let u8_c3_valid = ImagePipeline::from_view(&stream_context, &mut u8_c3_workspace, source_u8_c3)
        .square_distance_valid_norm_to_f32(&template_u8_c3)?
        .finish()?;
    let i8_valid = ImagePipeline::from_view(&stream_context, &mut i8_workspace, source_i8)
        .square_distance_valid_norm_to_f32(&template_i8)?
        .finish()?;
    let i8_c3_valid = ImagePipeline::from_view(&stream_context, &mut i8_c3_workspace, source_i8_c3)
        .square_distance_valid_norm_to_f32(&template_i8_c3)?
        .finish()?;
    let u8_scaled_full =
        ImagePipeline::from_view(&stream_context, &mut u8_scaled_full_workspace, source_u8)
            .square_distance_full_norm_scaled(&template_u8, 0)?
            .finish()?;
    let u8_scaled_same =
        ImagePipeline::from_view(&stream_context, &mut u8_scaled_same_workspace, source_u8)
            .square_distance_same_norm_scaled(&template_u8, 0)?
            .finish()?;
    let u8_scaled_valid =
        ImagePipeline::from_view(&stream_context, &mut u8_scaled_valid_workspace, source_u8)
            .square_distance_valid_norm_scaled(&template_u8, 0)?
            .finish()?;
    let u8_scaled_c3_valid =
        ImagePipeline::from_view(&stream_context, &mut u8_scaled_c3_workspace, source_u8_c3)
            .square_distance_valid_norm_scaled(&template_u8_c3, 0)?
            .finish()?;
    let correlation_full =
        ImagePipeline::from_view(&stream_context, &mut correlation_full_workspace, source)
            .cross_correlation_full_norm(&template)?
            .finish()?;
    let correlation_same =
        ImagePipeline::from_view(&stream_context, &mut correlation_same_workspace, source)
            .cross_correlation_same_norm(&template)?
            .finish()?;
    let correlation_valid =
        ImagePipeline::from_view(&stream_context, &mut correlation_valid_workspace, source)
            .cross_correlation_valid_norm(&template)?
            .finish()?;
    let correlation_u8 =
        ImagePipeline::from_view(&stream_context, &mut correlation_u8_workspace, source_u8)
            .cross_correlation_valid_norm_to_f32(&template_u8)?
            .finish()?;
    let correlation_i8 =
        ImagePipeline::from_view(&stream_context, &mut correlation_i8_workspace, source_i8)
            .cross_correlation_valid_norm_to_f32(&template_i8)?
            .finish()?;
    let correlation_u16 =
        ImagePipeline::from_view(&stream_context, &mut correlation_u16_workspace, source_u16)
            .cross_correlation_valid_norm_to_f32(&template_u16)?
            .finish()?;
    let correlation_scaled_full = ImagePipeline::from_view(
        &stream_context,
        &mut correlation_scaled_full_workspace,
        source_u8,
    )
    .cross_correlation_full_norm_scaled(&template_u8, 0)?
    .finish()?;
    let correlation_scaled_same = ImagePipeline::from_view(
        &stream_context,
        &mut correlation_scaled_same_workspace,
        source_u8,
    )
    .cross_correlation_same_norm_scaled(&template_u8, 0)?
    .finish()?;
    let correlation_scaled_valid = ImagePipeline::from_view(
        &stream_context,
        &mut correlation_scaled_valid_workspace,
        source_u8,
    )
    .cross_correlation_valid_norm_scaled(&template_u8, 0)?
    .finish()?;
    let correlation_scaled_c3_valid = ImagePipeline::from_view(
        &stream_context,
        &mut correlation_scaled_c3_workspace,
        source_u8_c3,
    )
    .cross_correlation_valid_norm_scaled(&template_u8_c3, 0)?
    .finish()?;

    stream.synchronize()?;

    assert_eq!(
        full.size(),
        Size {
            width: 4,
            height: 4,
        }
    );
    assert_eq!(
        same.size(),
        Size {
            width: 3,
            height: 3,
        }
    );
    assert_eq!(
        valid.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert_eq!(
        c3_valid.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert_eq!(u8_full.size(), full.size());
    assert_eq!(u8_same.size(), same.size());
    assert_eq!(u8_valid.size(), valid.size());
    assert_eq!(u8_c3_valid.size(), c3_valid.size());
    assert_eq!(i8_valid.size(), valid.size());
    assert_eq!(i8_c3_valid.size(), c3_valid.size());
    assert_eq!(u8_scaled_full.size(), full.size());
    assert_eq!(u8_scaled_same.size(), same.size());
    assert_eq!(u8_scaled_valid.size(), valid.size());
    assert_eq!(u8_scaled_c3_valid.size(), c3_valid.size());
    assert_eq!(correlation_full.size(), full.size());
    assert_eq!(correlation_same.size(), same.size());
    assert_eq!(correlation_valid.size(), valid.size());
    assert_eq!(correlation_u8.size(), valid.size());
    assert_eq!(correlation_i8.size(), valid.size());
    assert_eq!(correlation_u16.size(), valid.size());
    assert_eq!(correlation_scaled_full.size(), full.size());
    assert_eq!(correlation_scaled_same.size(), same.size());
    assert_eq!(correlation_scaled_valid.size(), valid.size());
    assert_eq!(correlation_scaled_c3_valid.size(), c3_valid.size());
    assert!(full.step() >= 16);
    assert!(same.step() >= 12);
    assert!(valid.step() >= 8);
    assert!(c3_valid.step() >= 24);
    assert!(u8_full.step() >= 16);
    assert!(u8_same.step() >= 12);
    assert!(u8_valid.step() >= 8);
    assert!(u8_c3_valid.step() >= 24);
    assert!(i8_valid.step() >= 8);
    assert!(i8_c3_valid.step() >= 24);
    assert!(u8_scaled_full.step() >= 4);
    assert!(u8_scaled_same.step() >= 3);
    assert!(u8_scaled_valid.step() >= 2);
    assert!(u8_scaled_c3_valid.step() >= 6);
    assert!(correlation_full.step() >= 16);
    assert!(correlation_same.step() >= 12);
    assert!(correlation_valid.step() >= 8);
    assert!(correlation_u8.step() >= 8);
    assert!(correlation_i8.step() >= 8);
    assert!(correlation_u16.step() >= 8);
    assert!(correlation_scaled_full.step() >= 4);
    assert!(correlation_scaled_same.step() >= 3);
    assert!(correlation_scaled_valid.step() >= 2);
    assert!(correlation_scaled_c3_valid.step() >= 6);

    Ok(())
}

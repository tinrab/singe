use super::*;

#[path = "tests_image_core_arithmetic.rs"]
mod image_core_arithmetic;

#[path = "tests_image_core_planar.rs"]
mod image_core_planar;

#[path = "tests_image_core_numeric.rs"]
mod image_core_numeric;

#[test]
fn fluent_image_pipeline_resizes_converts_and_thresholds() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [
        255u8, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 16, 32, 48, 64, 80, 96, 112, 128, 144,
        160, 176, 192, 208, 224, 240, 8, 16, 24, 40, 48, 56, 72, 80, 88, 104, 112, 120, 136, 144,
        152, 168, 176, 184, 200, 208, 216,
    ];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let source = ImageView::<_, C3>::from_memory(&source_memory, Size::new(4, 4))?;
    let mut workspace = Workspace::create();

    let image = ImagePipeline::from_view(&stream_context, &mut workspace, source)
        .resize(
            Size {
                width: 2,
                height: 2,
            },
            InterpolationMode::Linear,
        )?
        .rgb_to_gray()?
        .threshold(128, ComparisonOperation::Greater)?
        .threshold_greater(64)?
        .threshold_less(192)?
        .threshold_value(96, 0, ComparisonOperation::Greater)?
        .threshold_greater_value(80, 255)?
        .threshold_less_value(16, 0)?
        .threshold_less_greater_value(32, 0, 224, 255)?
        .finish()?;

    let source = ImageView::<_, C3>::from_memory(&source_memory, Size::new(4, 4))?;
    let mut channel_workspace = Workspace::create();
    let channel_image = ImagePipeline::from_view(&stream_context, &mut channel_workspace, source)
        .threshold_channels([16, 32, 48], ComparisonOperation::Greater)?
        .threshold_channels_greater([32, 64, 96])?
        .threshold_channels_less([224, 224, 224])?
        .threshold_channels_value([64, 96, 128], [0, 0, 0], ComparisonOperation::Greater)?
        .threshold_channels_greater_value([96, 128, 160], [255, 255, 255])?
        .threshold_channels_less_value([16, 16, 16], [0, 0, 0])?
        .threshold_channels_less_greater_value(
            [16, 16, 16],
            [0, 0, 0],
            [224, 224, 224],
            [255, 255, 255],
        )?
        .finish()?;

    stream.synchronize()?;

    assert_eq!(
        image.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert!(image.step() >= 2);
    assert_eq!(
        channel_image.size(),
        Size {
            width: 4,
            height: 4,
        }
    );
    assert!(channel_image.step() >= 12);

    Ok(())
}

#[test]
fn npp_image_download_compacts_pitched_rows() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let size = Size::new(385, 3);
    let source_host = (0..(size.width * size.height))
        .map(|value| (value % 251) as u8)
        .collect::<Vec<_>>();
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let source = ImageView::<_, C1>::from_memory(&source_memory, size)?;
    let mut image = Image::<u8, C1>::create(size)?;

    {
        let mut destination = image.view_mut()?;
        ImagePipeline::<u8, C1>::copy_into(&stream_context, &source, &mut destination)?;
    }
    stream.synchronize()?;

    assert_eq!(image.copy_to_host_vec()?, source_host);
    assert_eq!(
        image.copy_to_device_memory()?.copy_to_host_vec()?,
        source_host
    );

    Ok(())
}

#[test]
fn fluent_image_pipeline_runs_fused_absolute_difference_threshold() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [1u8, 8, 16, 32];
    let other_host = [0u8, 4, 18, 64];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let other_memory = DeviceMemory::from_slice(&other_host)?;
    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(2, 2))?;
    let other = ImageView::<_, C1>::from_memory(&other_memory, Size::new(2, 2))?;
    let mut workspace = Workspace::create();

    let image = ImagePipeline::from_view(&stream_context, &mut workspace, source)
        .fused_absolute_difference_threshold_greater_value(&other, 3, 255)?
        .fused_absolute_difference_threshold_greater_value(&other, 3, 128)?
        .finish()?;

    stream.synchronize()?;

    assert_eq!(
        image.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert!(image.step() >= 2);

    Ok(())
}

#[test]
fn fluent_image_pipeline_runs_compare_steps() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [1u8, 8, 16, 32];
    let other_host = [0u8, 8, 24, 16];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let other_memory = DeviceMemory::from_slice(&other_host)?;
    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(2, 2))?;
    let other = ImageView::<_, C1>::from_memory(&other_memory, Size::new(2, 2))?;
    let c3_source_host = [1u8, 2, 3, 8, 9, 10, 16, 17, 18, 32, 33, 34];
    let c3_source_memory = DeviceMemory::from_slice(&c3_source_host)?;
    let c3_source = ImageView::<_, C3>::from_memory(&c3_source_memory, Size::new(2, 2))?;
    let f32_source_host = [1.0f32, 1.1, 2.0, 2.2];
    let f32_other_host = [1.05f32, 1.2, 1.75, 2.35];
    let f32_source_memory = DeviceMemory::from_slice(&f32_source_host)?;
    let f32_other_memory = DeviceMemory::from_slice(&f32_other_host)?;
    let f32_source = ImageView::<_, C1>::from_memory(&f32_source_memory, Size::new(2, 2))?;
    let f32_other = ImageView::<_, C1>::from_memory(&f32_other_memory, Size::new(2, 2))?;
    let mut compare_workspace = Workspace::create();

    let compare_mask = ImagePipeline::from_view(&stream_context, &mut compare_workspace, source)
        .compare(&other, ComparisonOperation::Greater)?
        .compare_constant(0, ComparisonOperation::Greater)?
        .finish()?;

    let mut constant_workspace = Workspace::create();
    let constant_mask =
        ImagePipeline::from_view(&stream_context, &mut constant_workspace, c3_source)
            .compare_constant([4, 4, 4], ComparisonOperation::Greater)?
            .finish()?;

    let mut epsilon_workspace = Workspace::create();
    let epsilon_mask =
        ImagePipeline::from_view(&stream_context, &mut epsilon_workspace, f32_source)
            .compare_equal_epsilon(&f32_other, 0.15)?
            .compare_constant(0, ComparisonOperation::Greater)?
            .finish()?;

    let mut epsilon_constant_workspace = Workspace::create();
    let epsilon_constant_mask =
        ImagePipeline::from_view(&stream_context, &mut epsilon_constant_workspace, c3_source)
            .convert_to_f32()?
            .compare_equal_epsilon_constant([1.0, 2.0, 3.0], 0.25)?
            .finish()?;

    stream.synchronize()?;

    assert_eq!(
        compare_mask.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert_eq!(
        constant_mask.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert_eq!(
        epsilon_mask.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert_eq!(
        epsilon_constant_mask.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert!(compare_mask.step() >= 2);
    assert!(constant_mask.step() >= 2);
    assert!(epsilon_mask.step() >= 2);
    assert!(epsilon_constant_mask.step() >= 2);

    Ok(())
}

#[test]
fn fluent_image_pipeline_can_demosaic_cfa_to_rgb_and_rgba() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [
        16u8, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255,
    ];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(4, 4))?;
    let source_roi = Rectangle {
        x: 0,
        y: 0,
        width: 4,
        height: 4,
    };

    let mut rgb_workspace = Workspace::create();
    let rgb_image = ImagePipeline::from_view(&stream_context, &mut rgb_workspace, source)
        .cfa_to_rgb(source_roi, BayerGridPosition::Bggr)?
        .finish()?;

    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(4, 4))?;
    let mut rgba_workspace = Workspace::create();
    let rgba_image = ImagePipeline::from_view(&stream_context, &mut rgba_workspace, source)
        .cfa_to_rgba(source_roi, BayerGridPosition::Rggb, 255)?
        .finish()?;

    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(4, 4))?;
    let mut destination = Image::<u8, C3>::create(Size::new(4, 4))?;
    {
        let mut destination_view = destination.view_mut()?;
        ImagePipeline::<u8, C1>::cfa_to_rgb_into(
            &stream_context,
            &source,
            source_roi,
            &mut destination_view,
            BayerGridPosition::Grbg,
        )?;
    }

    stream.synchronize()?;

    assert_eq!(rgb_image.size(), source_roi.size());
    assert_eq!(rgba_image.size(), source_roi.size());
    assert_eq!(destination.size(), source_roi.size());
    assert!(rgb_image.step() >= 12);
    assert!(rgba_image.step() >= 16);
    assert!(destination.step() >= 12);

    Ok(())
}

#[test]
fn fluent_image_pipeline_can_round_trip_packed_uyvp_10u() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [
        16u8, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255, 240, 224, 208,
        192, 176, 160, 144, 128,
    ];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let source = ImageView::<_, C3>::from_memory(&source_memory, Size::new(4, 2))?;
    let source_offset = Point { x: 0, y: 0 };
    let mut workspace = Workspace::create();

    let rgb_image = ImagePipeline::from_view(&stream_context, &mut workspace, source)
        .rgb_to_uyvp_10u(source_offset, ColorSpace::Bt601Jpeg)?
        .uyvp_10u_to_rgb_u8(source_offset, ColorSpace::Bt601Jpeg)?
        .finish()?;

    let source = ImageView::<_, C3>::from_memory(&source_memory, Size::new(4, 2))?;
    let mut destination = Image::<u8, C3>::create(Size::new(4, 2))?;
    {
        let mut destination_view = destination.view_mut()?;
        ImagePipeline::<u8, C3>::rgb_to_uyvp_10u_into(
            &stream_context,
            &source,
            source_offset,
            &mut destination_view,
            ColorSpace::Bt709Hdtv,
        )?;
    }

    stream.synchronize()?;

    assert_eq!(rgb_image.size(), source.size());
    assert_eq!(destination.size(), source.size());
    assert!(rgb_image.step() >= 12);
    assert!(destination.step() >= 12);

    Ok(())
}

#[test]
fn fluent_image_pipeline_chains_unsigned_conversions() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [1u8, 8, 16, 32, 64, 128, 192, 255];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(4, 2))?;
    let mut workspace = Workspace::create();

    let image = ImagePipeline::from_view(&stream_context, &mut workspace, source)
        .convert_to_u16()?
        .convert_to_i32()?
        .convert_to_f32()?
        .finish()?;

    stream.synchronize()?;

    assert_eq!(
        image.size(),
        Size {
            width: 4,
            height: 2,
        }
    );
    assert!(image.step() >= 16);

    Ok(())
}

#[test]
fn fluent_image_pipeline_chains_signed_conversions() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [-128i16, -16, 0, 16, 64, 128, 512, 1024];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(4, 2))?;
    let mut workspace = Workspace::create();

    let image = ImagePipeline::from_view(&stream_context, &mut workspace, source)
        .convert_to_i32()?
        .convert_to_u8()?
        .finish()?;

    stream.synchronize()?;

    assert_eq!(
        image.size(),
        Size {
            width: 4,
            height: 2,
        }
    );
    assert!(image.step() >= 4);

    Ok(())
}

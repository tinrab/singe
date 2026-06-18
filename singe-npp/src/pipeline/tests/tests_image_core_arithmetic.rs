use super::*;

#[test]
fn fluent_image_pipeline_runs_binary_arithmetic_steps() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [8u8, 16, 32, 64];
    let other_host = [1u8, 2, 4, 8];
    let transcendental_host = [1u8, 1, 1, 1];
    let i16_host = [-8i16, 16, -32, 64];
    let f32_other_host = [1.0f32, 2.0, 4.0, 8.0];
    let f32_transcendental_host = [1.0f32, 1.1, 1.2, 1.3];
    let u8_ac4_host = [8u8, 16, 32, 128, 64, 32, 16, 192];
    let u16_ac4_host = [8u16, 16, 32, 128, 64, 32, 16, 192];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let other_memory = DeviceMemory::from_slice(&other_host)?;
    let transcendental_memory = DeviceMemory::from_slice(&transcendental_host)?;
    let i16_memory = DeviceMemory::from_slice(&i16_host)?;
    let f32_other_memory = DeviceMemory::from_slice(&f32_other_host)?;
    let f32_transcendental_memory = DeviceMemory::from_slice(&f32_transcendental_host)?;
    let u8_ac4_memory = DeviceMemory::from_slice(&u8_ac4_host)?;
    let u16_ac4_memory = DeviceMemory::from_slice(&u16_ac4_host)?;
    let u8_absolute_difference_constant = DeviceMemory::from_slice(&[8u8])?;
    let f32_absolute_difference_constant = DeviceMemory::from_slice(&[2.0f32])?;
    let u8_device_constant = DeviceMemory::from_slice(&[1u8])?;
    let u8_ac4_device_constant = DeviceMemory::from_slice(&[1u8, 1, 1])?;
    let u16_ac4_device_constant = DeviceMemory::from_slice(&[1u16, 1, 1])?;
    let i16_device_constant = DeviceMemory::from_slice(&[1i16])?;
    let f32_device_constant = DeviceMemory::from_slice(&[1.0f32])?;
    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(2, 2))?;
    let other = ImageView::<_, C1>::from_memory(&other_memory, Size::new(2, 2))?;
    let transcendental_source =
        ImageView::<_, C1>::from_memory(&transcendental_memory, Size::new(2, 2))?;
    let i16_source = ImageView::<_, C1>::from_memory(&i16_memory, Size::new(2, 2))?;
    let f32_other = ImageView::<_, C1>::from_memory(&f32_other_memory, Size::new(2, 2))?;
    let f32_transcendental_source =
        ImageView::<_, C1>::from_memory(&f32_transcendental_memory, Size::new(2, 2))?;
    let u8_ac4_source = ImageView::<_, AC4>::from_memory(&u8_ac4_memory, Size::new(2, 1))?;
    let u8_ac4_other = ImageView::<_, AC4>::from_memory(&u8_ac4_memory, Size::new(2, 1))?;
    let u16_ac4_source = ImageView::<_, AC4>::from_memory(&u16_ac4_memory, Size::new(2, 1))?;
    let u16_ac4_other = ImageView::<_, AC4>::from_memory(&u16_ac4_memory, Size::new(2, 1))?;
    let mut workspace = Workspace::create();

    let image = ImagePipeline::from_view(&stream_context, &mut workspace, source)
        .add(&other, 0)?
        .subtract(&other, 0)?
        .multiply(&other, 0)?
        .divide(&other, 0)?
        .absolute_difference(&other)?
        .absolute_difference_constant(16)?
        .absolute_difference_device_constant(&u8_absolute_difference_constant)?
        .add_device_constant(&u8_device_constant, 0)?
        .subtract_device_constant(&u8_device_constant, 0)?
        .multiply_device_constant(&u8_device_constant, 0)?
        .divide_device_constant(&u8_device_constant, 0)?
        .square(0)?
        .square_root(0)?
        .logical_not()?
        .logical_and(&other)?
        .logical_or(&other)?
        .logical_xor(&other)?
        .comp_color_key(&other, 8)?
        .alpha_comp_constant(128, &other, 64, AlphaOperation::Plus)?
        .alpha_premultiply_constant(128)?
        .convert_to_f32()?
        .add_constant(1.0, 0)?
        .add(&f32_other, 0)?
        .subtract(&f32_other, 0)?
        .multiply(&f32_other, 0)?
        .divide(&f32_other, 0)?
        .absolute_difference(&f32_other)?
        .absolute_difference_constant(1.0)?
        .absolute_difference_device_constant(&f32_absolute_difference_constant)?
        .add_device_constant(&f32_device_constant, 0)?
        .subtract_device_constant(&f32_device_constant, 0)?
        .multiply_device_constant(&f32_device_constant, 0)?
        .divide_device_constant(&f32_device_constant, 0)?
        .alpha_comp_constant(0.5, &f32_other, 0.25, AlphaOperation::Plus)?
        .absolute()?
        .square(0)?
        .square_root(0)?
        .finish()?;

    let mut absolute_workspace = Workspace::create();
    let absolute_image =
        ImagePipeline::from_view(&stream_context, &mut absolute_workspace, i16_source)
            .absolute()?
            .add_device_constant(&i16_device_constant, 0)?
            .subtract_device_constant(&i16_device_constant, 0)?
            .multiply_device_constant(&i16_device_constant, 0)?
            .divide_device_constant(&i16_device_constant, 0)?
            .square(0)?
            .square_root(0)?
            .finish()?;

    let mut transcendental_workspace = Workspace::create();
    let transcendental_image = ImagePipeline::from_view(
        &stream_context,
        &mut transcendental_workspace,
        transcendental_source,
    )
    .natural_logarithm(0)?
    .exponential(0)?
    .finish()?;

    let mut f32_transcendental_workspace = Workspace::create();
    let f32_transcendental_image = ImagePipeline::from_view(
        &stream_context,
        &mut f32_transcendental_workspace,
        f32_transcendental_source,
    )
    .natural_logarithm(0)?
    .exponential(0)?
    .finish()?;

    let mut u8_ac4_workspace = Workspace::create();
    let u8_ac4_image =
        ImagePipeline::from_view(&stream_context, &mut u8_ac4_workspace, u8_ac4_source)
            .add_device_constant(&u8_ac4_device_constant, 0)?
            .subtract_device_constant(&u8_ac4_device_constant, 0)?
            .multiply_device_constant(&u8_ac4_device_constant, 0)?
            .divide_device_constant(&u8_ac4_device_constant, 0)?
            .alpha_comp(&u8_ac4_other, AlphaOperation::Over)?
            .alpha_comp_color_key(128, &u8_ac4_other, 64, [0, 0, 0, 0], AlphaOperation::Plus)?
            .alpha_premultiply()?
            .alpha_premultiply_constant(128)?
            .finish()?;

    let mut u16_ac4_workspace = Workspace::create();
    let u16_ac4_image =
        ImagePipeline::from_view(&stream_context, &mut u16_ac4_workspace, u16_ac4_source)
            .add_device_constant(&u16_ac4_device_constant, 0)?
            .subtract_device_constant(&u16_ac4_device_constant, 0)?
            .multiply_device_constant(&u16_ac4_device_constant, 0)?
            .divide_device_constant(&u16_ac4_device_constant, 0)?
            .alpha_comp(&u16_ac4_other, AlphaOperation::Over)?
            .alpha_premultiply()?
            .alpha_premultiply_constant(128)?
            .finish()?;

    stream.synchronize()?;

    assert_eq!(
        image.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert!(image.step() >= 8);
    assert_eq!(
        absolute_image.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert!(absolute_image.step() >= 4);
    assert_eq!(
        transcendental_image.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert_eq!(
        f32_transcendental_image.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert_eq!(
        u8_ac4_image.size(),
        Size {
            width: 2,
            height: 1,
        }
    );
    assert_eq!(
        u16_ac4_image.size(),
        Size {
            width: 2,
            height: 1,
        }
    );
    assert!(transcendental_image.step() >= 2);
    assert!(f32_transcendental_image.step() >= 8);
    assert!(u8_ac4_image.step() >= 8);
    assert!(u16_ac4_image.step() >= 16);

    Ok(())
}

#[test]
fn fluent_image_pipeline_extracts_duplicates_and_swaps_channels() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let source = ImageView::<_, C3>::from_memory(&source_memory, Size::new(2, 2))?;
    let insert_host = [1u8, 2, 3, 4];
    let insert_memory = DeviceMemory::from_slice(&insert_host)?;
    let insert = ImageView::<_, C1>::from_memory(&insert_memory, Size::new(2, 2))?;
    let c4_other_host = [1u8; 16];
    let c4_other_memory = DeviceMemory::from_slice(&c4_other_host)?;
    let c4_other = ImageView::<_, C4>::from_memory(&c4_other_memory, Size::new(2, 2))?;
    let c4_device_constant = DeviceMemory::from_slice(&[1u8, 1, 1, 1])?;
    let mut workspace = Workspace::create();

    let image = ImagePipeline::from_view(&stream_context, &mut workspace, source)
        .swap_channels([2, 1, 0])?
        .extract_channel(0)?
        .duplicate_to_c4()?
        .insert_channel(&insert, 3)?
        .set_channel(9, 2)?
        .copy_channel(2, 1)?
        .comp_color_key(&c4_other, [0, 0, 0, 0])?
        .alpha_comp_constant(128, &c4_other, 64, AlphaOperation::Plus)?
        .add_device_constant(&c4_device_constant, 0)?
        .subtract_device_constant(&c4_device_constant, 0)?
        .multiply_device_constant(&c4_device_constant, 0)?
        .divide_device_constant(&c4_device_constant, 0)?
        .alpha_premultiply_constant(128)?
        .add_constant([1, 2, 3, 4], 0)?
        .multiply_constant([2, 2, 2, 2], 0)?
        .subtract_constant([1, 1, 1, 1], 0)?
        .divide_constant([1, 1, 1, 1], 0)?
        .finish()?;

    stream.synchronize()?;

    assert_eq!(
        image.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert!(image.step() >= 8);

    Ok(())
}

#[test]
fn fluent_image_pipeline_runs_integer_channel_steps() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [10u16, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let source = ImageView::<_, C3>::from_memory(&source_memory, Size::new(2, 2))?;
    let other = ImageView::<_, C3>::from_memory(&source_memory, Size::new(2, 2))?;
    let insert_host = [1u16, 2, 3, 4];
    let insert_memory = DeviceMemory::from_slice(&insert_host)?;
    let insert = ImageView::<_, C1>::from_memory(&insert_memory, Size::new(2, 2))?;
    let u16_c3_device_constant = DeviceMemory::from_slice(&[1u16, 1, 1])?;
    let mut workspace = Workspace::create();

    let image = ImagePipeline::from_view(&stream_context, &mut workspace, source)
        .extract_channel(0)?
        .duplicate_to_c3()?
        .insert_channel(&insert, 2)?
        .set_channel(9, 1)?
        .copy_channel(1, 0)?
        .add_device_constant(&u16_c3_device_constant, 0)?
        .subtract_device_constant(&u16_c3_device_constant, 0)?
        .multiply_device_constant(&u16_c3_device_constant, 0)?
        .divide_device_constant(&u16_c3_device_constant, 0)?
        .alpha_comp_constant(128, &other, 64, AlphaOperation::Plus)?
        .alpha_premultiply_constant(128)?
        .finish()?;

    let source = ImageView::<_, C3>::from_memory(&source_memory, Size::new(2, 2))?;
    let mut gray_workspace = Workspace::create();
    let gray = ImagePipeline::from_view(&stream_context, &mut gray_workspace, source)
        .rgb_to_gray()?
        .finish()?;

    let source = ImageView::<_, C3>::from_memory(&source_memory, Size::new(2, 2))?;
    let mut gradient_gray_workspace = Workspace::create();
    let gradient_gray =
        ImagePipeline::from_view(&stream_context, &mut gradient_gray_workspace, source)
            .gradient_color_to_gray(ImageNormalization::L2)?
            .finish()?;

    let c4_source_host = [
        10u16, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
    ];
    let c4_source_memory = DeviceMemory::from_slice(&c4_source_host)?;
    let c4_source = ImageView::<_, C4>::from_memory(&c4_source_memory, Size::new(2, 2))?;
    let mut weighted_gray_workspace = Workspace::create();
    let weighted_gray =
        ImagePipeline::from_view(&stream_context, &mut weighted_gray_workspace, c4_source)
            .color_to_gray([0.25, 0.25, 0.25, 0.25])?
            .finish()?;

    let i32_host = [1i32, 2, 3, 4];
    let i32_memory = DeviceMemory::from_slice(&i32_host)?;
    let i32_source = ImageView::<_, C1>::from_memory(&i32_memory, Size::new(2, 2))?;
    let i32_c3_host = [1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let i32_c3_memory = DeviceMemory::from_slice(&i32_c3_host)?;
    let i32_c3_source = ImageView::<_, C3>::from_memory(&i32_c3_memory, Size::new(2, 2))?;
    let i32_c3_device_constant = DeviceMemory::from_slice(&[1i32, 1, 1])?;
    let mut i32_workspace = Workspace::create();
    let i32_image = ImagePipeline::from_view(&stream_context, &mut i32_workspace, i32_source)
        .duplicate_to_c4()?
        .swap_channels([3, 2, 1, 0])?
        .set_channel(5, 3)?
        .copy_channel(3, 0)?
        .swap_channels_to_c3([2, 1, 0])?
        .swap_channels_to_c4([2, 1, 0, 3], 7)?
        .finish()?;

    let mut i32_c3_workspace = Workspace::create();
    let i32_c3_image =
        ImagePipeline::from_view(&stream_context, &mut i32_c3_workspace, i32_c3_source)
            .add_device_constant(&i32_c3_device_constant, 0)?
            .subtract_device_constant(&i32_c3_device_constant, 0)?
            .multiply_device_constant(&i32_c3_device_constant, 0)?
            .divide_device_constant(&i32_c3_device_constant, 0)?
            .finish()?;

    stream.synchronize()?;

    assert_eq!(
        image.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert_eq!(
        i32_image.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert_eq!(
        i32_c3_image.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert!(image.step() >= 12);
    assert_eq!(
        gray.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert!(gray.step() >= 4);
    assert_eq!(
        gradient_gray.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert!(gradient_gray.step() >= 4);
    assert_eq!(
        weighted_gray.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert!(weighted_gray.step() >= 4);
    assert!(i32_image.step() >= 32);
    assert!(i32_c3_image.step() >= 24);

    Ok(())
}

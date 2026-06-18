use super::*;

#[test]
fn fluent_image_pipeline_runs_lookup_tables() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [0u8, 1, 2, 3];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(2, 2))?;
    let u8_values = DeviceMemory::from_slice(&[0i32, 64, 128, 255])?;
    let u8_levels = DeviceMemory::from_slice(&[0i32, 1, 2, 3])?;
    let f32_values = DeviceMemory::from_slice(&[0.0f32, 10.0, 20.0, 30.0])?;
    let f32_levels = DeviceMemory::from_slice(&[0.0f32, 1.0, 2.0, 3.0])?;
    let u16_source_host = [0u16, 1, 2, 3];
    let u16_source_memory = DeviceMemory::from_slice(&u16_source_host)?;
    let u16_source = ImageView::<_, C1>::from_memory(&u16_source_memory, Size::new(2, 2))?;
    let i16_source_host = [0i16, 1, 2, 3];
    let i16_source_memory = DeviceMemory::from_slice(&i16_source_host)?;
    let i16_source = ImageView::<_, C1>::from_memory(&i16_source_memory, Size::new(2, 2))?;
    let c3_source_host = [0u8, 1, 2, 3, 2, 1, 2, 3, 0, 1, 0, 3];
    let c3_source_memory = DeviceMemory::from_slice(&c3_source_host)?;
    let c3_source = ImageView::<_, C3>::from_memory(&c3_source_memory, Size::new(2, 2))?;
    let c4_source_host = [
        0u8, 0, 0, 255, 255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255,
    ];
    let c4_source_memory = DeviceMemory::from_slice(&c4_source_host)?;
    let c4_source = ImageView::<_, C4>::from_memory(&c4_source_memory, Size::new(2, 2))?;
    let u8_channel_values = [&u8_values, &u8_values, &u8_values];
    let u8_channel_levels = [&u8_levels, &u8_levels, &u8_levels];
    let f32_channel_values = [&f32_values, &f32_values, &f32_values];
    let f32_channel_levels = [&f32_levels, &f32_levels, &f32_levels];
    let trilinear_values = DeviceMemory::from_slice(&[0u32, 32, 64, 96, 128, 160, 192, 255])?;
    let trilinear_levels = [&[0u8, 255][..], &[0u8, 255][..], &[0u8, 255][..]];
    let palette_u8 = DeviceMemory::from_slice(&[0u8, 64, 128, 255])?;
    let palette_u8_c3 =
        DeviceMemory::from_slice(&[0u8, 0, 0, 64, 32, 16, 128, 96, 64, 255, 255, 255])?;
    let palette_u32 = DeviceMemory::from_slice(&[0u32, 10, 20, 30])?;
    let mut u8_workspace = Workspace::create();

    let u8_image = ImagePipeline::from_view(&stream_context, &mut u8_workspace, source)
        .lookup_table(&u8_values, &u8_levels)?
        .lookup_table_linear(&u8_values, &u8_levels)?
        .lookup_table_cubic(&u8_values, &u8_levels)?
        .finish()?;

    let mut f32_workspace = Workspace::create();
    let f32_image = ImagePipeline::from_view(&stream_context, &mut f32_workspace, source)
        .convert_to_f32()?
        .lookup_table(&f32_values, &f32_levels)?
        .lookup_table_linear(&f32_values, &f32_levels)?
        .lookup_table_cubic(&f32_values, &f32_levels)?
        .finish()?;

    let mut u16_workspace = Workspace::create();
    let u16_image = ImagePipeline::from_view(&stream_context, &mut u16_workspace, u16_source)
        .lookup_table(&u8_values, &u8_levels)?
        .lookup_table_linear(&u8_values, &u8_levels)?
        .lookup_table_cubic(&u8_values, &u8_levels)?
        .finish()?;

    let mut i16_workspace = Workspace::create();
    let i16_image = ImagePipeline::from_view(&stream_context, &mut i16_workspace, i16_source)
        .lookup_table(&u8_values, &u8_levels)?
        .lookup_table_linear(&u8_values, &u8_levels)?
        .lookup_table_cubic(&u8_values, &u8_levels)?
        .finish()?;

    let mut u8_c3_workspace = Workspace::create();
    let u8_c3_image = ImagePipeline::from_view(&stream_context, &mut u8_c3_workspace, c3_source)
        .gamma_forward()?
        .gamma_inverse()?
        .lookup_table_channels(&u8_channel_values, &u8_channel_levels)?
        .lookup_table_channels_linear(&u8_channel_values, &u8_channel_levels)?
        .lookup_table_channels_cubic(&u8_channel_values, &u8_channel_levels)?
        .finish()?;

    let mut f32_c3_workspace = Workspace::create();
    let f32_c3_image = ImagePipeline::from_view(&stream_context, &mut f32_c3_workspace, c3_source)
        .convert_to_f32()?
        .lookup_table_channels(&f32_channel_values, &f32_channel_levels)?
        .lookup_table_channels_linear(&f32_channel_values, &f32_channel_levels)?
        .lookup_table_channels_cubic(&f32_channel_values, &f32_channel_levels)?
        .finish()?;

    let mut trilinear_workspace = Workspace::create();
    let trilinear_image =
        ImagePipeline::from_view(&stream_context, &mut trilinear_workspace, c4_source)
            .lookup_table_trilinear(&trilinear_values, trilinear_levels)?
            .finish()?;

    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(2, 2))?;
    let mut palette_workspace = Workspace::create();
    let palette_image = ImagePipeline::from_view(&stream_context, &mut palette_workspace, source)
        .lookup_table_palette(&palette_u8, 2)?
        .finish()?;

    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(2, 2))?;
    let mut palette_c3_workspace = Workspace::create();
    let palette_c3_image =
        ImagePipeline::from_view(&stream_context, &mut palette_c3_workspace, source)
            .lookup_table_palette_to::<u8, C3>(&palette_u8_c3, 2)?
            .finish()?;

    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(2, 2))?;
    let mut palette_u32_workspace = Workspace::create();
    let palette_u32_image =
        ImagePipeline::from_view(&stream_context, &mut palette_u32_workspace, source)
            .lookup_table_palette_to::<u32, C1>(&palette_u32, 2)?
            .finish()?;

    let c3_source = ImageView::<_, C3>::from_memory(&c3_source_memory, Size::new(2, 2))?;
    let mut palette_channels_workspace = Workspace::create();
    let palette_channels_image =
        ImagePipeline::from_view(&stream_context, &mut palette_channels_workspace, c3_source)
            .lookup_table_palette_channels::<3>(&[&palette_u8, &palette_u8, &palette_u8], 2)?
            .finish()?;

    let c3_source = ImageView::<_, C3>::from_memory(&c3_source_memory, Size::new(2, 2))?;
    let mut palette_swap_workspace = Workspace::create();
    let palette_swap_image =
        ImagePipeline::from_view(&stream_context, &mut palette_swap_workspace, c3_source)
            .lookup_table_palette_swap_to_c4(255, &[&palette_u8, &palette_u8, &palette_u8], 2)?
            .finish()?;

    stream.synchronize()?;

    assert_eq!(
        u8_image.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert_eq!(
        f32_image.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert!(u8_image.step() >= 2);
    assert!(f32_image.step() >= 8);
    assert_eq!(
        u16_image.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert_eq!(
        i16_image.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert!(u16_image.step() >= 4);
    assert!(i16_image.step() >= 4);
    assert_eq!(
        u8_c3_image.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert_eq!(
        f32_c3_image.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert_eq!(
        trilinear_image.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert_eq!(palette_image.size(), u8_image.size());
    assert_eq!(palette_c3_image.size(), u8_c3_image.size());
    assert_eq!(palette_u32_image.size(), u8_image.size());
    assert_eq!(palette_channels_image.size(), u8_c3_image.size());
    assert_eq!(palette_swap_image.size(), trilinear_image.size());
    assert!(u8_c3_image.step() >= 6);
    assert!(f32_c3_image.step() >= 24);
    assert!(trilinear_image.step() >= 8);
    assert!(palette_image.step() >= 2);
    assert!(palette_c3_image.step() >= 6);
    assert!(palette_u32_image.step() >= 8);
    assert!(palette_channels_image.step() >= 6);
    assert!(palette_swap_image.step() >= 8);

    Ok(())
}

#[test]
fn fluent_image_pipeline_runs_color_space_conversions() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [12u8, 32, 64, 80, 120, 160, 24, 48, 96, 200, 180, 140];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let source = ImageView::<_, C3>::from_memory(&source_memory, Size::new(2, 2))?;
    let identity_twist = [
        [1.0f32, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ];
    let identity_twist4 = [
        [1.0f32, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let zero_constants4 = [0.0f32; 4];
    let mut workspace = Workspace::create();

    let image = ImagePipeline::from_view(&stream_context, &mut workspace, source)
        .color_twist(identity_twist)?
        .rgb_to_xyz()?
        .xyz_to_rgb()?
        .rgb_to_luv()?
        .luv_to_rgb()?
        .rgb_to_hsv()?
        .hsv_to_rgb()?
        .rgb_to_hls()?
        .hls_to_rgb()?
        .rgb_to_yuv()?
        .yuv_to_rgb()?
        .rgb_to_ycbcr()?
        .ycbcr_to_rgb()?
        .finish()?;

    let source = ImageView::<_, C3>::from_memory(&source_memory, Size::new(2, 2))?;
    let mut lab_workspace = Workspace::create();
    let lab_image = ImagePipeline::from_view(&stream_context, &mut lab_workspace, source)
        .bgr_to_lab()?
        .lab_to_bgr()?
        .color_twist(identity_twist)?
        .finish()?;

    let source = ImageView::<_, C3>::from_memory(&source_memory, Size::new(2, 2))?;
    let mut f32_twist_workspace = Workspace::create();
    let f32_twist_image =
        ImagePipeline::from_view(&stream_context, &mut f32_twist_workspace, source)
            .convert_to_f32()?
            .color_twist(identity_twist)?
            .finish()?;

    let c4_source_host = [
        12u8, 32, 64, 255, 80, 120, 160, 255, 24, 48, 96, 255, 200, 180, 140, 255,
    ];
    let c4_source_memory = DeviceMemory::from_slice(&c4_source_host)?;
    let c4_source = ImageView::<_, C4>::from_memory(&c4_source_memory, Size::new(2, 2))?;
    let mut c4_twist_workspace = Workspace::create();
    let c4_twist_image =
        ImagePipeline::from_view(&stream_context, &mut c4_twist_workspace, c4_source)
            .color_twist(identity_twist)?
            .color_twist_with_constants(identity_twist4, zero_constants4)?
            .finish()?;

    let c4_source = ImageView::<_, C4>::from_memory(&c4_source_memory, Size::new(2, 2))?;
    let mut f32_c4_twist_workspace = Workspace::create();
    let f32_c4_twist_image =
        ImagePipeline::from_view(&stream_context, &mut f32_c4_twist_workspace, c4_source)
            .convert_to_f32()?
            .color_twist(identity_twist)?
            .color_twist_with_constants(identity_twist4, zero_constants4)?
            .finish()?;

    let source = ImageView::<_, C3>::from_memory(&source_memory, Size::new(2, 2))?;
    let mut yuv_workspace = Workspace::create();
    let yuv_image = ImagePipeline::from_view(&stream_context, &mut yuv_workspace, source)
        .rgb_to_yuv422()?
        .yuv422_to_rgb()?
        .finish()?;

    let source = ImageView::<_, C3>::from_memory(&source_memory, Size::new(2, 2))?;
    let mut ycbcr_workspace = Workspace::create();
    let ycbcr_image = ImagePipeline::from_view(&stream_context, &mut ycbcr_workspace, source)
        .rgb_to_ycbcr422()?
        .ycbcr422_to_ycrcb422()?
        .ycrcb422_to_rgb()?
        .finish()?;

    let source = ImageView::<_, C3>::from_memory(&source_memory, Size::new(2, 2))?;
    let mut cbycr_workspace = Workspace::create();
    let cbycr_image = ImagePipeline::from_view(&stream_context, &mut cbycr_workspace, source)
        .rgb_to_cbycr422()?
        .cbycr422_to_ycbcr422()?
        .ycbcr422_to_cbycr422()?
        .cbycr422_to_rgb()?
        .finish()?;

    let source = ImageView::<_, C3>::from_memory(&source_memory, Size::new(2, 2))?;
    let mut gamma_workspace = Workspace::create();
    let gamma_image = ImagePipeline::from_view(&stream_context, &mut gamma_workspace, source)
        .rgb_to_cbycr422_gamma()?
        .cbycr422_to_rgb()?
        .finish()?;

    let source = ImageView::<_, C3>::from_memory(&source_memory, Size::new(2, 2))?;
    let mut ycrcb_workspace = Workspace::create();
    let ycrcb_image = ImagePipeline::from_view(&stream_context, &mut ycrcb_workspace, source)
        .rgb_to_ycrcb422()?
        .ycrcb422_to_rgb()?
        .finish()?;

    let source = ImageView::<_, C3>::from_memory(&source_memory, Size::new(2, 2))?;
    let mut bgr_ycbcr_workspace = Workspace::create();
    let bgr_ycbcr_image =
        ImagePipeline::from_view(&stream_context, &mut bgr_ycbcr_workspace, source)
            .bgr_to_ycbcr422()?
            .ycbcr422_to_bgr()?
            .finish()?;

    let source = ImageView::<_, C3>::from_memory(&source_memory, Size::new(2, 2))?;
    let mut bgr_ycbcr_c4_workspace = Workspace::create();
    let bgr_ycbcr_c4_image =
        ImagePipeline::from_view(&stream_context, &mut bgr_ycbcr_c4_workspace, source)
            .bgr_to_ycbcr422()?
            .ycbcr422_to_bgr_c4(255)?
            .finish()?;

    let source = ImageView::<_, C3>::from_memory(&source_memory, Size::new(2, 2))?;
    let mut bgr_cbycr_workspace = Workspace::create();
    let bgr_cbycr_image =
        ImagePipeline::from_view(&stream_context, &mut bgr_cbycr_workspace, source)
            .bgr_to_cbycr422_709hdtv()?
            .cbycr422_to_bgr_709hdtv()?
            .finish()?;

    let source = ImageView::<_, C3>::from_memory(&source_memory, Size::new(2, 2))?;
    let mut cbycr_c4_workspace = Workspace::create();
    let cbycr_c4_image = ImagePipeline::from_view(&stream_context, &mut cbycr_c4_workspace, source)
        .rgb_to_cbycr422()?
        .cbycr422_to_bgr_c4(200)?
        .finish()?;

    let source = ImageView::<_, C3>::from_memory(&source_memory, Size::new(2, 2))?;
    let mut cbycr_709hdtv_c4_workspace = Workspace::create();
    let cbycr_709hdtv_c4_image =
        ImagePipeline::from_view(&stream_context, &mut cbycr_709hdtv_c4_workspace, source)
            .bgr_to_cbycr422_709hdtv()?
            .cbycr422_to_bgr_709hdtv_c4(180)?
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
        lab_image.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert_eq!(
        f32_twist_image.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert_eq!(
        c4_twist_image.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert_eq!(
        f32_c4_twist_image.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert_eq!(
        yuv_image.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert_eq!(
        ycbcr_image.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert_eq!(
        cbycr_image.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert_eq!(
        gamma_image.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert_eq!(
        ycrcb_image.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert_eq!(
        bgr_ycbcr_image.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert_eq!(
        bgr_ycbcr_c4_image.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert_eq!(
        bgr_cbycr_image.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert_eq!(
        cbycr_c4_image.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert_eq!(
        cbycr_709hdtv_c4_image.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert!(image.step() >= 6);
    assert!(lab_image.step() >= 6);
    assert!(f32_twist_image.step() >= 24);
    assert!(c4_twist_image.step() >= 8);
    assert!(f32_c4_twist_image.step() >= 32);
    assert!(yuv_image.step() >= 6);
    assert!(ycbcr_image.step() >= 6);
    assert!(cbycr_image.step() >= 6);
    assert!(gamma_image.step() >= 6);
    assert!(ycrcb_image.step() >= 6);
    assert!(bgr_ycbcr_image.step() >= 6);
    assert!(bgr_ycbcr_c4_image.step() >= 8);
    assert!(bgr_cbycr_image.step() >= 6);
    assert!(cbycr_c4_image.step() >= 8);
    assert!(cbycr_709hdtv_c4_image.step() >= 8);

    Ok(())
}

#[test]
fn fluent_image_pipeline_sets_mirrors_and_transposes() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [1u8, 2, 3, 4];
    let overlay_host = [9u8, 10, 11, 12];
    let mask_host = [255u8, 0, 255, 0];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let overlay_memory = DeviceMemory::from_slice(&overlay_host)?;
    let mask_memory = DeviceMemory::from_slice(&mask_host)?;
    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(2, 2))?;
    let overlay = ImageView::<_, C1>::from_memory(&overlay_memory, Size::new(2, 2))?;
    let mask = ImageView::<_, C1>::from_memory(&mask_memory, Size::new(2, 2))?;
    let mut workspace = Workspace::create();

    let image = ImagePipeline::from_view(&stream_context, &mut workspace, source)
        .set(7)?
        .set_masked(3, &mask)?
        .copy_masked_from(&overlay, &mask)?
        .mirror(Axis::Horizontal)?
        .transpose()?
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
fn fluent_image_pipeline_copies_subpixel_roi() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [1u8, 2, 3, 4];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(2, 2))?;
    let mut workspace = Workspace::create();

    let image = ImagePipeline::from_view(&stream_context, &mut workspace, source)
        .copy_subpixel(
            Size {
                width: 1,
                height: 1,
            },
            0.25,
            0.25,
        )?
        .finish()?;

    stream.synchronize()?;

    assert_eq!(
        image.size(),
        Size {
            width: 1,
            height: 1,
        }
    );
    assert!(image.step() >= 1);

    Ok(())
}

#[test]
fn fluent_image_pipeline_runs_geometric_warps() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [1u8, 2, 3, 4];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(2, 2))?;
    let x_map_memory = DeviceMemory::from_slice(&[0.0f32, 1.0, 0.0, 1.0])?;
    let y_map_memory = DeviceMemory::from_slice(&[0.0f32, 0.0, 1.0, 1.0])?;
    let x_map = ImageView::<_, C1>::from_memory(&x_map_memory, Size::new(2, 2))?;
    let y_map = ImageView::<_, C1>::from_memory(&y_map_memory, Size::new(2, 2))?;
    let roi = Rectangle {
        x: 0,
        y: 0,
        width: 2,
        height: 2,
    };
    let affine = AffineCoefficients::from([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    let perspective =
        PerspectiveCoefficients::from([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let quadrangle = QuadrangleF64::from([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]);
    let mut workspace = Workspace::create();

    let image = ImagePipeline::from_view(&stream_context, &mut workspace, source)
        .rotate(geometry::Rotate {
            source_roi: roi,
            destination_roi: roi,
            angle: 0.0,
            x_shift: 0.0,
            y_shift: 0.0,
            interpolation: InterpolationMode::Nearest,
        })?
        .resize_sqr_pixel(geometry::ResizeSqrPixel {
            source_roi: roi,
            destination_roi: roi,
            x_factor: 1.0,
            y_factor: 1.0,
            x_shift: 0.0,
            y_shift: 0.0,
            interpolation: InterpolationMode::Nearest,
        })?
        .resize_sqr_pixel_advanced(geometry::ResizeSqrPixelAdvanced {
            source_roi: roi,
            destination_roi: roi,
            x_factor: 1.0,
            y_factor: 1.0,
            interpolation: InterpolationMode::Lanczos3Advanced,
        })?
        .remap(
            geometry::Remap {
                source_roi: roi,
                interpolation: InterpolationMode::Nearest,
            },
            &x_map,
            &y_map,
        )?
        .warp_affine(geometry::WarpAffine {
            source_roi: roi,
            destination_roi: roi,
            coefficients: affine,
            interpolation: InterpolationMode::Nearest,
        })?
        .warp_affine_back(geometry::WarpAffine {
            source_roi: roi,
            destination_roi: roi,
            coefficients: affine,
            interpolation: InterpolationMode::Nearest,
        })?
        .warp_affine_quad(geometry::WarpQuad {
            source_roi: roi,
            source_quadrangle: quadrangle,
            destination_roi: roi,
            destination_quadrangle: quadrangle,
            interpolation: InterpolationMode::Nearest,
        })?
        .warp_perspective(geometry::WarpPerspective {
            source_roi: roi,
            destination_roi: roi,
            coefficients: perspective,
            interpolation: InterpolationMode::Nearest,
        })?
        .warp_perspective_back(geometry::WarpPerspective {
            source_roi: roi,
            destination_roi: roi,
            coefficients: perspective,
            interpolation: InterpolationMode::Nearest,
        })?
        .warp_perspective_quad(geometry::WarpQuad {
            source_roi: roi,
            source_quadrangle: quadrangle,
            destination_roi: roi,
            destination_quadrangle: quadrangle,
            interpolation: InterpolationMode::Nearest,
        })?
        .finish()?;

    stream.synchronize()?;

    assert_eq!(image.size(), roi.size());
    assert!(image.step() >= 2);

    Ok(())
}

#[test]
fn fluent_image_pipeline_adds_borders() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [1u8, 2, 3, 4];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(2, 2))?;
    let mut workspace = Workspace::create();

    let image = ImagePipeline::from_view(&stream_context, &mut workspace, source)
        .constant_border(
            Size {
                width: 4,
                height: 4,
            },
            1,
            1,
            0,
        )?
        .replicate_border(
            Size {
                width: 6,
                height: 6,
            },
            1,
            1,
        )?
        .wrap_border(
            Size {
                width: 8,
                height: 8,
            },
            1,
            1,
        )?
        .finish()?;

    stream.synchronize()?;

    assert_eq!(
        image.size(),
        Size {
            width: 8,
            height: 8,
        }
    );
    assert!(image.step() >= 8);

    Ok(())
}

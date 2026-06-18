use super::*;

#[test]
fn fluent_image_pipeline_copies_between_packed_and_planar() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [1u8, 2, 3, 4, 5, 6];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let source = ImageView::<_, C3>::from_memory(&source_memory, Size::new(2, 1))?;
    let mut planar_workspace = Workspace::create();

    let planar = ImagePipeline::from_view(&stream_context, &mut planar_workspace, source)
        .copy_to_planar::<3>()?;

    let planar_view = planar.view()?;
    let source = ImageView::<_, C3>::from_memory(&source_memory, Size::new(2, 1))?;
    let mut packed_workspace = Workspace::create();
    let packed = ImagePipeline::from_view(&stream_context, &mut packed_workspace, source)
        .copy_from_planar::<3>(&planar_view)?
        .finish()?;

    let color_source_host = [12u8, 32, 64, 80, 120, 160, 24, 48, 96, 200, 180, 140];
    let color_source_memory = DeviceMemory::from_slice(&color_source_host)?;
    let identity_twist = [
        [1.0f32, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ];
    let color_source = ImageView::<_, C3>::from_memory(&color_source_memory, Size::new(2, 2))?;
    let mut yuv_planar_workspace = Workspace::create();
    let yuv_planar =
        ImagePipeline::from_view(&stream_context, &mut yuv_planar_workspace, color_source)
            .rgb_to_yuv_planar()?;

    let yuv_planar_view = yuv_planar.view()?;
    let color_source = ImageView::<_, C3>::from_memory(&color_source_memory, Size::new(2, 2))?;
    let mut yuv_packed_workspace = Workspace::create();
    let yuv_packed =
        ImagePipeline::from_view(&stream_context, &mut yuv_packed_workspace, color_source)
            .yuv_planar_to_rgb(&yuv_planar_view)?
            .finish()?;

    let color_source = ImageView::<_, C3>::from_memory(&color_source_memory, Size::new(2, 2))?;
    let mut ycbcr_planar_workspace = Workspace::create();
    let ycbcr_planar =
        ImagePipeline::from_view(&stream_context, &mut ycbcr_planar_workspace, color_source)
            .rgb_to_ycbcr_planar()?;

    let ycbcr_planar_view = ycbcr_planar.view()?;
    let color_source = ImageView::<_, C3>::from_memory(&color_source_memory, Size::new(2, 2))?;
    let mut ycbcr_packed_workspace = Workspace::create();
    let ycbcr_packed =
        ImagePipeline::from_view(&stream_context, &mut ycbcr_packed_workspace, color_source)
            .ycbcr_planar_to_rgb(&ycbcr_planar_view)?
            .finish()?;

    let color_source = ImageView::<_, C3>::from_memory(&color_source_memory, Size::new(2, 2))?;
    let mut bgr_yuv_planar_workspace = Workspace::create();
    let bgr_yuv_planar =
        ImagePipeline::from_view(&stream_context, &mut bgr_yuv_planar_workspace, color_source)
            .bgr_to_yuv_planar()?;

    let bgr_yuv_planar_view = bgr_yuv_planar.view()?;
    let color_source = ImageView::<_, C3>::from_memory(&color_source_memory, Size::new(2, 2))?;
    let mut bgr_yuv_packed_workspace = Workspace::create();
    let bgr_yuv_packed =
        ImagePipeline::from_view(&stream_context, &mut bgr_yuv_packed_workspace, color_source)
            .yuv_planar_to_bgr(&bgr_yuv_planar_view)?
            .finish()?;

    let color_source = ImageView::<_, C3>::from_memory(&color_source_memory, Size::new(2, 2))?;
    let mut bgr_ycbcr_planar_workspace = Workspace::create();
    let bgr_ycbcr_planar = ImagePipeline::from_view(
        &stream_context,
        &mut bgr_ycbcr_planar_workspace,
        color_source,
    )
    .bgr_to_ycbcr_planar()?;

    let bgr_ycbcr_planar_view = bgr_ycbcr_planar.view()?;
    let color_source = ImageView::<_, C3>::from_memory(&color_source_memory, Size::new(2, 2))?;
    let mut bgr_ycbcr_packed_workspace = Workspace::create();
    let bgr_ycbcr_packed = ImagePipeline::from_view(
        &stream_context,
        &mut bgr_ycbcr_packed_workspace,
        color_source,
    )
    .ycbcr_planar_to_bgr(&bgr_ycbcr_planar_view)?
    .finish()?;

    let color_source = ImageView::<_, C3>::from_memory(&color_source_memory, Size::new(2, 2))?;
    let mut rgb_planar_workspace = Workspace::create();
    let rgb_yuv_planar =
        ImagePipeline::from_view(&stream_context, &mut rgb_planar_workspace, color_source)
            .copy_to_planar::<3>()?
            .rgb_to_yuv(&stream_context)?
            .yuv_to_rgb(&stream_context)?;

    let color_source = ImageView::<_, C3>::from_memory(&color_source_memory, Size::new(2, 2))?;
    let mut rgb_ycbcr_roundtrip_workspace = Workspace::create();
    let rgb_ycbcr_planar = ImagePipeline::from_view(
        &stream_context,
        &mut rgb_ycbcr_roundtrip_workspace,
        color_source,
    )
    .copy_to_planar::<3>()?
    .rgb_to_ycbcr(&stream_context)?
    .ycbcr_to_rgb(&stream_context)?;

    let color_source = ImageView::<_, C3>::from_memory(&color_source_memory, Size::new(2, 2))?;
    let mut bgr_yuv_roundtrip_workspace = Workspace::create();
    let bgr_yuv_roundtrip_planar = ImagePipeline::from_view(
        &stream_context,
        &mut bgr_yuv_roundtrip_workspace,
        color_source,
    )
    .copy_to_planar::<3>()?
    .bgr_to_yuv(&stream_context)?
    .yuv_to_bgr(&stream_context)?;

    let color_source = ImageView::<_, C3>::from_memory(&color_source_memory, Size::new(2, 2))?;
    let mut bgr_hls_workspace = Workspace::create();
    let bgr_hls_planar =
        ImagePipeline::from_view(&stream_context, &mut bgr_hls_workspace, color_source)
            .copy_to_planar::<3>()?
            .bgr_to_hls(&stream_context)?
            .hls_to_bgr(&stream_context)?;

    let color_source = ImageView::<_, C3>::from_memory(&color_source_memory, Size::new(2, 2))?;
    let mut rgb_ycbcr422_planar_workspace = Workspace::create();
    let rgb_ycbcr422_packed = ImagePipeline::from_view(
        &stream_context,
        &mut rgb_ycbcr422_planar_workspace,
        color_source,
    )
    .copy_to_planar::<3>()?
    .rgb_to_ycbcr422(&stream_context)?;

    let rgb_ycbcr422_packed_view = rgb_ycbcr422_packed.view()?;
    let mut ycbcr422_rgb_planar_workspace = Workspace::create();
    let ycbcr422_rgb_planar = ImagePipeline::from_view(
        &stream_context,
        &mut ycbcr422_rgb_planar_workspace,
        rgb_ycbcr422_packed_view,
    )
    .ycbcr422_to_rgb_planar()?;

    let color_source = ImageView::<_, C3>::from_memory(&color_source_memory, Size::new(2, 2))?;
    let mut rgb_ycrcb422_planar_workspace = Workspace::create();
    let rgb_ycrcb422_packed = ImagePipeline::from_view(
        &stream_context,
        &mut rgb_ycrcb422_planar_workspace,
        color_source,
    )
    .copy_to_planar::<3>()?
    .rgb_to_ycrcb422(&stream_context)?;

    let rgb_ycrcb422_packed_view = rgb_ycrcb422_packed.view()?;
    let mut ycrcb422_rgb_planar_workspace = Workspace::create();
    let ycrcb422_rgb_planar = ImagePipeline::from_view(
        &stream_context,
        &mut ycrcb422_rgb_planar_workspace,
        rgb_ycrcb422_packed_view,
    )
    .ycrcb422_to_rgb_planar()?;

    let color_source = ImageView::<_, C3>::from_memory(&color_source_memory, Size::new(2, 2))?;
    let mut gamma_planar_workspace = Workspace::create();
    let gamma_planar =
        ImagePipeline::from_view(&stream_context, &mut gamma_planar_workspace, color_source)
            .copy_to_planar::<3>()?
            .gamma_forward(&stream_context)?
            .gamma_inverse(&stream_context)?
            .color_twist(&stream_context, identity_twist)?;

    let color_source = ImageView::<_, C3>::from_memory(&color_source_memory, Size::new(2, 2))?;
    let mut f32_planar_workspace = Workspace::create();
    let f32_planar =
        ImagePipeline::from_view(&stream_context, &mut f32_planar_workspace, color_source)
            .convert_to_f32()?
            .copy_to_planar::<3>()?
            .color_twist(&stream_context, identity_twist)?;

    let u16_source_host = [12u16, 32, 64, 80, 120, 160, 24, 48, 96, 200, 180, 140];
    let u16_source_memory = DeviceMemory::from_slice(&u16_source_host)?;
    let u16_source = ImageView::<_, C3>::from_memory(&u16_source_memory, Size::new(2, 2))?;
    let mut u16_planar_workspace = Workspace::create();
    let u16_planar =
        ImagePipeline::from_view(&stream_context, &mut u16_planar_workspace, u16_source)
            .copy_to_planar::<3>()?
            .color_twist(&stream_context, identity_twist)?;

    let i16_source_host = [12i16, 32, 64, 80, 120, 160, 24, 48, 96, 200, 180, 140];
    let i16_source_memory = DeviceMemory::from_slice(&i16_source_host)?;
    let i16_source = ImageView::<_, C3>::from_memory(&i16_source_memory, Size::new(2, 2))?;
    let mut i16_planar_workspace = Workspace::create();
    let i16_planar =
        ImagePipeline::from_view(&stream_context, &mut i16_planar_workspace, i16_source)
            .copy_to_planar::<3>()?
            .color_twist(&stream_context, identity_twist)?;

    stream.synchronize()?;

    assert_eq!(
        packed.size(),
        Size {
            width: 2,
            height: 1,
        }
    );
    assert!(packed.step() >= 6);
    for plane in planar.planes() {
        assert_eq!(
            plane.size(),
            Size {
                width: 2,
                height: 1,
            }
        );
        assert!(plane.step() >= 2);
    }
    assert_eq!(
        yuv_packed.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert_eq!(
        ycbcr_packed.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert_eq!(
        bgr_yuv_packed.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert_eq!(
        bgr_ycbcr_packed.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert!(yuv_packed.step() >= 6);
    assert!(ycbcr_packed.step() >= 6);
    assert!(bgr_yuv_packed.step() >= 6);
    assert!(bgr_ycbcr_packed.step() >= 6);
    assert_eq!(
        rgb_ycbcr422_packed.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert_eq!(
        rgb_ycrcb422_packed.size(),
        Size {
            width: 2,
            height: 2,
        }
    );
    assert!(rgb_ycbcr422_packed.step() >= 4);
    assert!(rgb_ycrcb422_packed.step() >= 4);
    for planar in [
        &rgb_yuv_planar,
        &rgb_ycbcr_planar,
        &bgr_yuv_roundtrip_planar,
        &bgr_hls_planar,
        &ycbcr422_rgb_planar,
        &ycrcb422_rgb_planar,
        &gamma_planar,
    ] {
        for plane in planar.planes() {
            assert_eq!(
                plane.size(),
                Size {
                    width: 2,
                    height: 2,
                }
            );
            assert!(plane.step() >= 2);
        }
    }
    for plane in f32_planar.planes() {
        assert_eq!(
            plane.size(),
            Size {
                width: 2,
                height: 2,
            }
        );
        assert!(plane.step() >= 8);
    }
    for plane in u16_planar.planes() {
        assert_eq!(
            plane.size(),
            Size {
                width: 2,
                height: 2,
            }
        );
        assert!(plane.step() >= 4);
    }
    for plane in i16_planar.planes() {
        assert_eq!(
            plane.size(),
            Size {
                width: 2,
                height: 2,
            }
        );
        assert!(plane.step() >= 4);
    }

    Ok(())
}

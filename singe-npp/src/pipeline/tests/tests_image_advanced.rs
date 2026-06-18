use super::*;

#[test]
fn fluent_image_pipeline_runs_hough_line_filter() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [255u8, 0, 0, 0, 0, 255, 0, 0, 0, 0, 255, 0, 0, 0, 0, 255];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(4, 4))?;
    let mut workspace = Workspace::create();
    let delta = PointPolar {
        rho: 1.0,
        theta: 0.5,
    };

    let lines = ImagePipeline::from_view(&stream_context, &mut workspace, source)
        .filter_hough_lines(delta, 1, 8)?;

    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(4, 4))?;
    let mut region_workspace = Workspace::create();
    let region_lines = ImagePipeline::from_view(&stream_context, &mut region_workspace, source)
        .filter_hough_lines_region(
            delta,
            1,
            [
                PointPolar {
                    rho: 0.0,
                    theta: 0.3,
                },
                PointPolar {
                    rho: 8.0,
                    theta: 2.5,
                },
            ],
            8,
        )?;

    stream.synchronize()?;

    assert_eq!(lines.lines.len(), 8);
    assert_eq!(lines.line_count.len(), 1);
    assert_eq!(region_lines.lines.len(), 8);
    assert_eq!(region_lines.line_count.len(), 1);
    assert!(lines.line_count.copy_to_host_vec()?[0] <= 8);
    assert!(region_lines.line_count.copy_to_host_vec()?[0] <= 8);

    Ok(())
}

#[test]
fn fluent_image_pipeline_runs_watershed_segmentation() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [
        0u8, 8, 16, 24, 8, 32, 64, 32, 16, 64, 128, 64, 24, 32, 64, 255,
    ];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(4, 4))?;
    let mut workspace = Workspace::create();

    let image = ImagePipeline::from_view(&stream_context, &mut workspace, source)
        .segment_watershed(
            None,
            ImageNormalization::L1,
            WatershedSegmentBoundaryType::None,
        )?
        .finish()?;

    stream.synchronize()?;

    assert_eq!(
        image.size(),
        Size {
            width: 4,
            height: 4,
        }
    );
    assert!(image.step() >= 4);

    Ok(())
}

#[test]
fn fluent_image_pipeline_runs_label_markers() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [
        255u8, 255, 0, 0, 255, 255, 0, 255, 0, 0, 255, 255, 0, 0, 255, 255,
    ];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(4, 4))?;
    let mut workspace = Workspace::create();

    let labels = ImagePipeline::from_view(&stream_context, &mut workspace, source)
        .label_markers_uf(ImageNormalization::L1)?
        .finish()?;

    stream.synchronize()?;

    assert_eq!(
        labels.size(),
        Size {
            width: 4,
            height: 4,
        }
    );
    assert!(labels.step() >= 16);

    Ok(())
}

#[test]
fn fluent_image_pipeline_runs_gradient_vector_filters() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [1u8, 2, 3, 4, 5, 7, 9, 11, 13, 16, 19, 22, 25, 29, 33, 37];
    let source_c3_host: Vec<u8> = source_host
        .iter()
        .flat_map(|value| [*value, value.saturating_add(1), value.saturating_add(2)])
        .collect();
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let source_c3_memory = DeviceMemory::from_slice(&source_c3_host)?;
    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(4, 4))?;
    let source_c3 = ImageView::<_, C3>::from_memory(&source_c3_memory, Size::new(4, 4))?;
    let mut sobel_workspace = Workspace::create();
    let mut prewitt_workspace = Workspace::create();
    let mut scharr_c3_workspace = Workspace::create();

    let sobel = ImagePipeline::from_view(&stream_context, &mut sobel_workspace, source)
        .gradient_vector_sobel_border_to::<i16>(
            Point { x: 0, y: 0 },
            MaskSize::Mask3x3,
            ImageNormalization::L2,
            BorderType::Replicate,
        )?;
    let prewitt = ImagePipeline::from_view(&stream_context, &mut prewitt_workspace, source)
        .gradient_vector_prewitt_border_to::<i16>(
            Point { x: 0, y: 0 },
            MaskSize::Mask3x3,
            ImageNormalization::L1,
            BorderType::Replicate,
        )?;
    let scharr_c3 = ImagePipeline::from_view(&stream_context, &mut scharr_c3_workspace, source_c3)
        .gradient_vector_scharr_border_to::<i16>(
            Point { x: 0, y: 0 },
            MaskSize::Mask3x3,
            ImageNormalization::L2,
            BorderType::Replicate,
        )?;

    stream.synchronize()?;

    for image in [
        &sobel.x,
        &sobel.y,
        &sobel.magnitude,
        &prewitt.x,
        &prewitt.y,
        &prewitt.magnitude,
        &scharr_c3.x,
        &scharr_c3.y,
        &scharr_c3.magnitude,
    ] {
        assert_eq!(
            image.size(),
            Size {
                width: 4,
                height: 4,
            }
        );
        assert!(image.step() >= 8);
    }
    for image in [&sobel.angle, &prewitt.angle, &scharr_c3.angle] {
        assert_eq!(
            image.size(),
            Size {
                width: 4,
                height: 4,
            }
        );
        assert!(image.step() >= 16);
    }

    Ok(())
}

#[test]
fn fluent_image_pipeline_runs_histogram_of_gradients() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source = Image::<u8, C1>::create(Size::new(64, 128))?;
    let mut workspace = Workspace::create();
    let config = HistogramOfGradientsConfig {
        cell_size: 8,
        histogram_block_size: 16,
        histogram_bins: 9,
        detection_window_size: Size {
            width: 64,
            height: 128,
        },
    };

    let histogram = ImagePipeline::from_owned(&stream_context, &mut workspace, source)
        .set(16)?
        .histogram_of_gradients_border(
            Point { x: 0, y: 0 },
            &[Point { x: 0, y: 0 }],
            Size {
                width: 64,
                height: 128,
            },
            config,
            BorderType::Replicate,
        )?;

    stream.synchronize()?;

    assert!(histogram.descriptor_bytes > 0);
    assert!(histogram.descriptors.byte_len() >= histogram.descriptor_bytes);

    Ok(())
}

#[test]
fn fluent_image_pipeline_runs_flood_fill_steps() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [1u8, 1, 9, 9, 1, 1, 9, 9, 8, 8, 2, 2, 8, 8, 2, 2];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(4, 4))?;
    let mut region = ConnectedRegion::default();
    let mut workspace = Workspace::create();

    let image = ImagePipeline::from_view(&stream_context, &mut workspace, source)
        .flood_fill(
            Point { x: 0, y: 0 },
            3,
            ImageNormalization::L1,
            Some(&mut region),
        )?
        .flood_fill_boundary(Point { x: 3, y: 0 }, 4, 3, ImageNormalization::L1, None)?
        .flood_fill_range(Point { x: 0, y: 2 }, 0, 1, 5, ImageNormalization::L1, None)?
        .flood_fill_range_boundary(
            Point { x: 2, y: 2 },
            0,
            1,
            6,
            5,
            ImageNormalization::L1,
            None,
        )?
        .flood_fill_gradient(Point { x: 1, y: 1 }, 0, 4, 7, ImageNormalization::L1, None)?
        .flood_fill_gradient_boundary(
            Point { x: 3, y: 3 },
            0,
            4,
            8,
            7,
            ImageNormalization::L1,
            None,
        )?
        .finish()?;

    let source_c3_host: Vec<u8> = source_host
        .iter()
        .flat_map(|value| [*value, *value, *value])
        .collect();
    let source_c3_memory = DeviceMemory::from_slice(&source_c3_host)?;
    let source_c3 = ImageView::<_, C3>::from_memory(&source_c3_memory, Size::new(4, 4))?;
    let mut c3_workspace = Workspace::create();
    let c3_image = ImagePipeline::from_view(&stream_context, &mut c3_workspace, source_c3)
        .flood_fill(
            Point { x: 0, y: 0 },
            [3, 3, 3],
            ImageNormalization::L1,
            None,
        )?
        .finish()?;

    stream.synchronize()?;

    assert_eq!(
        image.size(),
        Size {
            width: 4,
            height: 4,
        }
    );
    assert_eq!(c3_image.size(), image.size());
    assert!(image.step() >= 4);
    assert!(c3_image.step() >= 12);
    assert!(region.connected_pixel_count > 0);

    Ok(())
}

#[test]
fn fluent_image_pipeline_runs_planar_color_and_geometry_steps() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [
        255u8, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 32, 64, 96, 96, 64, 32, 16, 32, 48, 48,
        32, 16, 128, 160, 192, 192, 160, 128, 64, 96, 128, 128, 96, 64, 224, 208, 192, 192, 208,
        224, 80, 112, 144, 144, 112, 80,
    ];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let source = ImageView::<_, C3>::from_memory(&source_memory, Size::new(4, 4))?;
    let mut workspace = Workspace::create();

    let planar = ImagePipeline::from_view(&stream_context, &mut workspace, source)
        .rgb_to_ycbcr444_jpeg_planar()?
        .resize(
            &stream_context,
            geometry::Resize {
                source_roi: Rectangle {
                    x: 0,
                    y: 0,
                    width: 4,
                    height: 4,
                },
                destination_roi: Rectangle {
                    x: 0,
                    y: 0,
                    width: 2,
                    height: 2,
                },
                interpolation: InterpolationMode::Linear,
            },
        )?
        .ycbcr444_to_rgb_jpeg(&stream_context)?;

    stream.synchronize()?;

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

    Ok(())
}

#[test]
fn fluent_image_pipeline_runs_subsampled_planar_jpeg_round_trip() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [
        255u8, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 16, 32, 48, 48, 32, 16, 64, 96, 128, 128,
        96, 64, 192, 160, 128, 128, 160, 192, 224, 208, 192, 192, 208, 224, 8, 24, 40, 40, 24, 8,
        72, 88, 104, 104, 88, 72,
    ];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let source = ImageView::<_, C3>::from_memory(&source_memory, Size::new(4, 4))?;
    let mut workspace = Workspace::create();

    let planar = ImagePipeline::from_view(&stream_context, &mut workspace, source)
        .rgb_to_ycbcr420_jpeg_planar()?;

    assert_eq!(
        planar.size(),
        Size {
            width: 4,
            height: 4,
        }
    );
    assert_eq!(
        planar.chroma_size(),
        Size {
            width: 2,
            height: 2,
        }
    );

    let image = planar.ycbcr420_to_rgb_jpeg_c3(&stream_context)?;

    stream.synchronize()?;

    assert_eq!(
        image.size(),
        Size {
            width: 4,
            height: 4,
        }
    );
    assert!(image.step() >= 12);

    Ok(())
}

use super::*;

#[test]
fn fluent_image_pipeline_runs_filtering_steps() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(4, 4))?;
    let mut workspace = Workspace::create();

    let image = ImagePipeline::from_view(&stream_context, &mut workspace, source)
        .filter_box(
            Size {
                width: 3,
                height: 3,
            },
            Point { x: 1, y: 1 },
        )?
        .filter_box_border(
            Point { x: 0, y: 0 },
            Size {
                width: 3,
                height: 3,
            },
            Point { x: 1, y: 1 },
            BorderType::Replicate,
        )?
        .filter_threshold_adaptive_box_border(
            Point { x: 0, y: 0 },
            Size {
                width: 3,
                height: 3,
            },
            0.0,
            255,
            0,
            BorderType::Replicate,
        )?
        .filter_max(
            Size {
                width: 3,
                height: 3,
            },
            Point { x: 1, y: 1 },
        )?
        .filter_min(
            Size {
                width: 3,
                height: 3,
            },
            Point { x: 1, y: 1 },
        )?
        .filter_max_border(
            Point { x: 0, y: 0 },
            Size {
                width: 3,
                height: 3,
            },
            Point { x: 1, y: 1 },
            BorderType::Replicate,
        )?
        .filter_min_border(
            Point { x: 0, y: 0 },
            Size {
                width: 3,
                height: 3,
            },
            Point { x: 1, y: 1 },
            BorderType::Replicate,
        )?
        .filter_median(
            Size {
                width: 3,
                height: 3,
            },
            Point { x: 1, y: 1 },
        )?
        .filter_median_border(
            Point { x: 0, y: 0 },
            Size {
                width: 3,
                height: 3,
            },
            Point { x: 1, y: 1 },
            BorderType::Replicate,
        )?
        .filter_kernel(
            &[0, 0, 0, 0, 1, 0, 0, 0, 0],
            Size {
                width: 3,
                height: 3,
            },
            Point { x: 1, y: 1 },
            1,
        )?
        .filter_kernel_border(
            Point { x: 0, y: 0 },
            &[0, 0, 0, 0, 1, 0, 0, 0, 0],
            Size {
                width: 3,
                height: 3,
            },
            Point { x: 1, y: 1 },
            1,
            BorderType::Replicate,
        )?
        .filter_kernel32f(
            &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            Size {
                width: 3,
                height: 3,
            },
            Point { x: 1, y: 1 },
        )?
        .filter_kernel32f_border(
            Point { x: 0, y: 0 },
            &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            Size {
                width: 3,
                height: 3,
            },
            Point { x: 1, y: 1 },
            BorderType::Replicate,
        )?
        .filter_gauss(MaskSize::Mask3x3)?
        .filter_low_pass(MaskSize::Mask3x3)?
        .filter_high_pass(MaskSize::Mask3x3)?
        .filter_laplace(MaskSize::Mask3x3)?
        .filter_sharpen()?
        .filter_high_pass_border(
            Point { x: 0, y: 0 },
            MaskSize::Mask3x3,
            BorderType::Replicate,
        )?
        .filter_low_pass_border(
            Point { x: 0, y: 0 },
            MaskSize::Mask3x3,
            BorderType::Replicate,
        )?
        .filter_gauss_border(
            Point { x: 0, y: 0 },
            MaskSize::Mask3x3,
            BorderType::Replicate,
        )?
        .filter_laplace_border(
            Point { x: 0, y: 0 },
            MaskSize::Mask3x3,
            BorderType::Replicate,
        )?
        .filter_sharpen_border(Point { x: 0, y: 0 }, BorderType::Replicate)?
        .filter_unsharp_border(
            Point { x: 0, y: 0 },
            1.0,
            1.0,
            0.5,
            0.0,
            BorderType::Replicate,
        )?
        .filter_gauss_advanced(&[0.25, 0.5, 0.25])?
        .filter_gauss_advanced_border(
            Point { x: 0, y: 0 },
            &[0.25, 0.5, 0.25],
            BorderType::Replicate,
        )?
        .filter_bilateral_gauss_border(Point { x: 0, y: 0 }, 1, 1, 1.0, 1.0, BorderType::Replicate)?
        .filter_column(&[1, 1, 1], 1, 3)?
        .filter_row(&[1, 1, 1], 1, 3)?
        .filter_column32f(&[0.25, 0.5, 0.25], 1)?
        .filter_row32f(&[0.25, 0.5, 0.25], 1)?
        .filter_column_border(
            Point { x: 0, y: 0 },
            &[1, 1, 1],
            1,
            3,
            BorderType::Replicate,
        )?
        .filter_row_border(
            Point { x: 0, y: 0 },
            &[1, 1, 1],
            1,
            3,
            BorderType::Replicate,
        )?
        .filter_column_border32f(
            Point { x: 0, y: 0 },
            &[0.25, 0.5, 0.25],
            1,
            BorderType::Replicate,
        )?
        .filter_row_border32f(
            Point { x: 0, y: 0 },
            &[0.25, 0.5, 0.25],
            1,
            BorderType::Replicate,
        )?
        .filter_prewitt_horizontal()?
        .filter_prewitt_vertical()?
        .filter_roberts_down()?
        .filter_roberts_up()?
        .filter_sobel_horizontal()?
        .filter_sobel_vertical()?
        .filter_prewitt_horizontal_border(Point { x: 0, y: 0 }, BorderType::Replicate)?
        .filter_prewitt_vertical_border(Point { x: 0, y: 0 }, BorderType::Replicate)?
        .filter_roberts_down_border(Point { x: 0, y: 0 }, BorderType::Replicate)?
        .filter_roberts_up_border(Point { x: 0, y: 0 }, BorderType::Replicate)?
        .filter_sobel_horizontal_border(Point { x: 0, y: 0 }, BorderType::Replicate)?
        .filter_sobel_vertical_border(Point { x: 0, y: 0 }, BorderType::Replicate)?
        .filter_canny_border(
            Point { x: 0, y: 0 },
            DifferentialKernel::Sobel,
            MaskSize::Mask3x3,
            0,
            16,
            ImageNormalization::L2,
            BorderType::Replicate,
        )?
        .filter_harris_corners_border(
            Point { x: 0, y: 0 },
            DifferentialKernel::Sobel,
            MaskSize::Mask3x3,
            MaskSize::Mask3x3,
            0.04,
            1.0,
            BorderType::Replicate,
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
    assert!(image.step() >= 16);

    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(4, 4))?;
    let mut sobel_horizontal_workspace = Workspace::create();
    let sobel_horizontal =
        ImagePipeline::from_view(&stream_context, &mut sobel_horizontal_workspace, source)
            .filter_sobel_horizontal_to::<i16, C1>(MaskSize::Mask3x3)?
            .finish()?;

    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(4, 4))?;
    let mut sobel_vertical_workspace = Workspace::create();
    let sobel_vertical =
        ImagePipeline::from_view(&stream_context, &mut sobel_vertical_workspace, source)
            .filter_sobel_vertical_border_to::<i16, C1>(
                Point { x: 0, y: 0 },
                MaskSize::Mask3x3,
                BorderType::Replicate,
            )?
            .finish()?;

    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(4, 4))?;
    let mut scharr_horizontal_workspace = Workspace::create();
    let scharr_horizontal =
        ImagePipeline::from_view(&stream_context, &mut scharr_horizontal_workspace, source)
            .filter_scharr_horizontal_to::<i16, C1>()?
            .finish()?;

    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(4, 4))?;
    let mut scharr_vertical_workspace = Workspace::create();
    let scharr_vertical =
        ImagePipeline::from_view(&stream_context, &mut scharr_vertical_workspace, source)
            .filter_scharr_vertical_border_to::<i16, C1>(
                Point { x: 0, y: 0 },
                BorderType::Replicate,
            )?
            .finish()?;

    stream.synchronize()?;

    for image in [
        &sobel_horizontal,
        &sobel_vertical,
        &scharr_horizontal,
        &scharr_vertical,
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

    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(4, 4))?;
    let mut sum_column_workspace = Workspace::create();
    let sum_column = ImagePipeline::from_view(&stream_context, &mut sum_column_workspace, source)
        .sum_window_column(3, 1)?
        .finish()?;

    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(4, 4))?;
    let mut sum_row_workspace = Workspace::create();
    let sum_row = ImagePipeline::from_view(&stream_context, &mut sum_row_workspace, source)
        .sum_window_row(3, 1)?
        .finish()?;

    stream.synchronize()?;

    assert_eq!(
        sum_column.size(),
        Size {
            width: 4,
            height: 4,
        }
    );
    assert!(sum_column.step() >= 16);
    assert_eq!(
        sum_row.size(),
        Size {
            width: 4,
            height: 4,
        }
    );
    assert!(sum_row.step() >= 16);

    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(4, 4))?;
    let mut sum_column_border_workspace = Workspace::create();
    let sum_column_border =
        ImagePipeline::from_view(&stream_context, &mut sum_column_border_workspace, source)
            .sum_window_column_border(Point { x: 0, y: 0 }, 3, 1, BorderType::Replicate)?
            .finish()?;

    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(4, 4))?;
    let mut sum_row_border_workspace = Workspace::create();
    let sum_row_border =
        ImagePipeline::from_view(&stream_context, &mut sum_row_border_workspace, source)
            .sum_window_row_border(Point { x: 0, y: 0 }, 3, 1, BorderType::Replicate)?
            .finish()?;

    stream.synchronize()?;

    assert_eq!(
        sum_column_border.size(),
        Size {
            width: 4,
            height: 4,
        }
    );
    assert!(sum_column_border.step() >= 16);
    assert_eq!(
        sum_row_border.size(),
        Size {
            width: 4,
            height: 4,
        }
    );
    assert!(sum_row_border.step() >= 16);

    let distance_source_host = vec![255u8; 64 * 64];
    let distance_source_memory = DeviceMemory::from_slice(&distance_source_host)?;
    let source = ImageView::<_, C1>::from_memory(&distance_source_memory, Size::new(64, 64))?;
    let mut distance_workspace = Workspace::create();
    let distance = ImagePipeline::from_view(&stream_context, &mut distance_workspace, source)
        .distance_transform_pba::<u16>(0, 0)?
        .distance_transform_abs_pba::<f32>(0, 0)?
        .finish()?;

    stream.synchronize()?;

    assert_eq!(
        distance.size(),
        Size {
            width: 64,
            height: 64,
        }
    );
    assert!(distance.step() >= 256);

    Ok(())
}

use super::*;

#[test]
fn fluent_image_pipeline_runs_extended_f32_edge_filters() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [
        1.0f32, 2.0, 4.0, 8.0, 3.0, 6.0, 9.0, 12.0, 5.0, 10.0, 15.0, 20.0, 7.0, 14.0, 21.0, 28.0,
    ];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(4, 4))?;
    let mut workspace = Workspace::create();

    let image = ImagePipeline::from_view(&stream_context, &mut workspace, source)
        .filter_scharr_horizontal()?
        .filter_scharr_vertical_border(Point { x: 0, y: 0 }, BorderType::Replicate)?
        .filter_sobel_horizontal_mask(MaskSize::Mask3x3)?
        .filter_sobel_vertical_second(MaskSize::Mask3x3)?
        .filter_sobel_cross_border(
            Point { x: 0, y: 0 },
            MaskSize::Mask3x3,
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

    Ok(())
}

#[test]
fn fluent_image_pipeline_runs_extended_typed_edge_filters() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [1u8, 2, 4, 8, 3, 6, 9, 12, 5, 10, 15, 20, 7, 14, 21, 28];
    let source_memory = DeviceMemory::from_slice(&source_host)?;

    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(4, 4))?;
    let mut second_workspace = Workspace::create();
    let second = ImagePipeline::from_view(&stream_context, &mut second_workspace, source)
        .filter_sobel_horizontal_second_to::<i16, C1>(MaskSize::Mask3x3)?
        .filter_laplace(MaskSize::Mask3x3)?
        .finish()?;

    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(4, 4))?;
    let mut border_workspace = Workspace::create();
    let border = ImagePipeline::from_view(&stream_context, &mut border_workspace, source)
        .filter_sobel_cross_border_to::<i16, C1>(
            Point { x: 0, y: 0 },
            MaskSize::Mask3x3,
            BorderType::Replicate,
        )?
        .filter_laplace_border(
            Point { x: 0, y: 0 },
            MaskSize::Mask3x3,
            BorderType::Replicate,
        )?
        .finish()?;

    stream.synchronize()?;

    for image in [&second, &border] {
        assert_eq!(
            image.size(),
            Size {
                width: 4,
                height: 4,
            }
        );
        assert!(image.step() >= 8);
    }

    Ok(())
}

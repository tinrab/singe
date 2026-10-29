use crate::pipeline::tests::tests_image_statistics_filtering::*;

#[test]
fn test_fluent_image_pipeline_runs_extended_f32_edge_filters() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = (0..100).map(|value| value as f32).collect::<Vec<_>>();
    let source_memory = DeviceMemory::from_slice(stream.context(), &source_host)?;
    let source = ImageView::<_, C1>::from_buffer(&source_memory, Size::new(10, 10))?;
    let mut workspace = Workspace::create();

    let image = ImagePipeline::from_view(&stream_context, &mut workspace, source)?
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
fn test_fluent_image_pipeline_runs_extended_typed_edge_filters() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = (0..36).map(|value| value as u8).collect::<Vec<_>>();
    let source_memory = DeviceMemory::from_slice(stream.context(), &source_host)?;

    let source = ImageView::<_, C1>::from_buffer(&source_memory, Size::new(6, 6))?;
    let mut second_workspace = Workspace::create();
    let second = ImagePipeline::from_view(&stream_context, &mut second_workspace, source)?
        .filter_sobel_horizontal_second_to::<i16, C1>(MaskSize::Mask3x3)?
        .filter_laplace(MaskSize::Mask3x3)?
        .finish()?;

    let source = ImageView::<_, C1>::from_buffer(&source_memory, Size::new(4, 4))?;
    let mut border_workspace = Workspace::create();
    let border = ImagePipeline::from_view(&stream_context, &mut border_workspace, source)?
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

    for (image, expected) in [(&second, Size::new(2, 2)), (&border, Size::new(4, 4))] {
        assert_eq!(image.size(), expected);
        assert!(image.step() >= 8);
    }

    Ok(())
}

use crate::pipeline::tests::tests_image_statistics_filtering::*;

#[test]
fn test_fluent_image_pipeline_runs_morphology_steps() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = (0..144).map(|value| value as u8).collect::<Vec<_>>();
    let source_memory = DeviceMemory::from_slice(stream.context(), &source_host)?;
    let source = ImageView::<_, C1>::from_buffer(&source_memory, Size::new(12, 12))?;
    let mut workspace = Workspace::create();
    let mask = [1u8; 9];
    let mask_size = Size {
        width: 3,
        height: 3,
    };
    let anchor = Point { x: 1, y: 1 };

    let image = ImagePipeline::from_view(&stream_context, &mut workspace, source)?
        .dilate(&mask, mask_size, anchor)?
        .erode(&mask, mask_size, anchor)?
        .dilate_3x3()?
        .erode_3x3()?
        .dilate_border(
            Point { x: 0, y: 0 },
            &mask,
            mask_size,
            anchor,
            BorderType::Replicate,
        )?
        .erode_border(
            Point { x: 0, y: 0 },
            &mask,
            mask_size,
            anchor,
            BorderType::Replicate,
        )?
        .dilate_3x3_border(Point { x: 0, y: 0 }, BorderType::Replicate)?
        .erode_3x3_border(Point { x: 0, y: 0 }, BorderType::Replicate)?
        .morph_close_border(
            Point { x: 0, y: 0 },
            &mask,
            mask_size,
            anchor,
            BorderType::Replicate,
        )?
        .morph_open_border(
            Point { x: 0, y: 0 },
            &mask,
            mask_size,
            anchor,
            BorderType::Replicate,
        )?
        .morph_top_hat_border(
            Point { x: 0, y: 0 },
            &mask,
            mask_size,
            anchor,
            BorderType::Replicate,
        )?
        .morph_black_hat_border(
            Point { x: 0, y: 0 },
            &mask,
            mask_size,
            anchor,
            BorderType::Replicate,
        )?
        .morph_gradient_border(
            Point { x: 0, y: 0 },
            &mask,
            mask_size,
            anchor,
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
    assert!(image.step() >= 4);

    Ok(())
}

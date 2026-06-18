use super::*;

#[test]
fn fluent_image_pipeline_runs_morphology_steps() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(4, 4))?;
    let mut workspace = Workspace::create();
    let mask = [1u8; 9];
    let mask_size = Size {
        width: 3,
        height: 3,
    };
    let anchor = Point { x: 1, y: 1 };

    let image = ImagePipeline::from_view(&stream_context, &mut workspace, source)
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

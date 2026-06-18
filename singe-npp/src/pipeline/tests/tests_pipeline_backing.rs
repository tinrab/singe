use super::*;

#[test]
fn fluent_pipelines_expose_owned_backing_after_first_allocating_operation() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_image_memory = DeviceMemory::from_slice(&[0u8; 12])?;
    let source_image = ImageView::<_, C3>::from_memory(&source_image_memory, Size::new(2, 2))?;
    let source_c4_image_memory = DeviceMemory::from_slice(&[0u8; 16])?;
    let source_c4_image =
        ImageView::<_, C4>::from_memory(&source_c4_image_memory, Size::new(2, 2))?;
    let source_signal_memory = DeviceMemory::from_slice(&[1.0f32, 2.0, 4.0])?;
    let source_signal = SignalView::from_memory(&source_signal_memory, source_signal_memory.len())?;

    let mut image_workspace = Workspace::create();
    let image_pipeline =
        ImagePipeline::from_view(&stream_context, &mut image_workspace, source_image)
            .threshold_channels([1, 2, 3], ComparisonOperation::Greater)?;

    assert!(image_pipeline.is_owned());
    stream.synchronize()?;
    assert!(matches!(
        image_pipeline.into_backing(),
        ImageBacking::Owned(_)
    ));

    let mut c4_image_workspace = Workspace::create();
    let c4_image_pipeline =
        ImagePipeline::from_view(&stream_context, &mut c4_image_workspace, source_c4_image)
            .extract_channel(3)?;

    assert!(c4_image_pipeline.is_owned());
    stream.synchronize()?;
    assert!(matches!(
        c4_image_pipeline.into_backing(),
        ImageBacking::Owned(_)
    ));

    let mut signal_workspace = Workspace::create();
    let signal_pipeline =
        SignalPipeline::from_view(&stream_context, &mut signal_workspace, source_signal)
            .multiply_constant(3.0)?;

    assert!(signal_pipeline.is_owned());
    stream.synchronize()?;
    assert!(matches!(
        signal_pipeline.into_backing(),
        SignalBacking::Owned(_)
    ));

    Ok(())
}

#[test]
fn fluent_image_pipeline_can_start_from_allocation_and_resize_with_roi() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let mut workspace = Workspace::create();

    let image = ImagePipeline::<u8, C1>::create(
        &stream_context,
        &mut workspace,
        Size {
            width: 4,
            height: 4,
        },
    )?
    .set(7)?
    .resize_with(geometry::Resize {
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
        interpolation: InterpolationMode::Nearest,
    })?
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
fn fluent_signal_pipeline_can_start_from_memory_and_write_integral_into() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [1i32, 2, 3];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let mut workspace = Workspace::create();

    let signal = SignalPipeline::<i32>::from_memory(
        &stream_context,
        &mut workspace,
        &source_memory,
        source_host.len(),
    )?
    .integral()?
    .finish()?;

    let source = SignalView::from_memory(&source_memory, source_host.len())?;
    let mut destination = Signal::<i32>::create(source_host.len() + 1)?;
    {
        let mut destination_view = destination.view_mut()?;
        SignalPipeline::<i32>::integral_into(&stream_context, &source, &mut destination_view)?;
    }

    stream.synchronize()?;

    assert_eq!(signal.len(), source_host.len() + 1);
    assert_eq!(destination.len(), source_host.len() + 1);

    Ok(())
}

#[test]
fn signal_pipeline_static_helpers_can_update_float_signal_in_place() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let mut signal = Signal::<f32>::create(4)?;

    {
        let mut signal_view = signal.view_mut()?;
        SignalPipeline::<f32>::set_into(&stream_context, 2.0, &mut signal_view)?;
        SignalPipeline::<f32>::add_constant_in_place(&stream_context, &mut signal_view, 1.0)?;
        SignalPipeline::<f32>::multiply_constant_in_place(&stream_context, &mut signal_view, 3.0)?;
        SignalPipeline::<f32>::square_root_in_place(&stream_context, &mut signal_view)?;
        SignalPipeline::<f32>::threshold_greater_value_in_place(
            &stream_context,
            &mut signal_view,
            2.0,
            2.0,
        )?;
    }

    stream.synchronize()?;

    assert_eq!(signal.len(), 4);

    Ok(())
}

#[test]
fn signal_pipeline_static_helpers_can_update_integer_signal_in_place() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let mut signal = Signal::<u8>::create(8)?;

    {
        let mut signal_view = signal.view_mut()?;
        SignalPipeline::<u8>::set_into(&stream_context, 4, &mut signal_view)?;
        SignalPipeline::<u8>::add_constant_scaled_in_place(
            &stream_context,
            &mut signal_view,
            8,
            1,
        )?;
        SignalPipeline::<u8>::bit_xor_constant_in_place(&stream_context, &mut signal_view, 3)?;
        SignalPipeline::<u8>::left_shift_constant_in_place(&stream_context, &mut signal_view, 1)?;
        SignalPipeline::<u8>::divide_constant_scaled_in_place(
            &stream_context,
            &mut signal_view,
            2,
            0,
        )?;
    }

    stream.synchronize()?;

    assert_eq!(signal.len(), 8);

    Ok(())
}

use super::*;

#[test]
fn fluent_image_pipeline_runs_integral_statistics() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [1u8, 2, 3, 4];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(2, 2))?;
    let mut i32_workspace = Workspace::create();

    let i32_image = ImagePipeline::from_view(&stream_context, &mut i32_workspace, source)
        .integral_to_i32(0)?
        .finish()?;

    let mut f32_workspace = Workspace::create();
    let f32_image = ImagePipeline::from_view(&stream_context, &mut f32_workspace, source)
        .integral_to_f32(0.0)?
        .finish()?;
    let mut squared_i32_workspace = Workspace::create();
    let squared_i32 = ImagePipeline::from_view(&stream_context, &mut squared_i32_workspace, source)
        .squared_integral_to_i32_i32(0, 0)?;
    let mut squared_f64_workspace = Workspace::create();
    let squared_f64 = ImagePipeline::from_view(&stream_context, &mut squared_f64_workspace, source)
        .squared_integral_to_i32_f64(0, 0.0)?;
    let mut squared_f32_workspace = Workspace::create();
    let squared_f32 = ImagePipeline::from_view(&stream_context, &mut squared_f32_workspace, source)
        .squared_integral_to_f32_f64(0.0, 0.0)?;
    let rectangle = Rectangle {
        x: 0,
        y: 0,
        width: 2,
        height: 2,
    };
    let mut rect_i32_workspace = Workspace::create();
    let rect_i32 = ImagePipeline::from_view(
        &stream_context,
        &mut rect_i32_workspace,
        squared_i32.integral.view()?,
    )
    .rect_standard_deviation_scaled(&squared_i32.squared.view()?, rectangle, 0)?
    .finish()?;
    let mut rect_i32_to_f32_workspace = Workspace::create();
    let rect_i32_to_f32 = ImagePipeline::from_view(
        &stream_context,
        &mut rect_i32_to_f32_workspace,
        squared_f64.integral.view()?,
    )
    .rect_standard_deviation_to_f32(&squared_f64.squared.view()?, rectangle)?
    .finish()?;
    let mut rect_f32_workspace = Workspace::create();
    let rect_f32 = ImagePipeline::from_view(
        &stream_context,
        &mut rect_f32_workspace,
        squared_f32.integral.view()?,
    )
    .rect_standard_deviation(&squared_f32.squared.view()?, rectangle)?
    .finish()?;

    stream.synchronize()?;

    assert_eq!(
        i32_image.size(),
        Size {
            width: 3,
            height: 3,
        }
    );
    assert_eq!(
        f32_image.size(),
        Size {
            width: 3,
            height: 3,
        }
    );
    assert!(i32_image.step() >= 12);
    assert!(f32_image.step() >= 12);
    assert_eq!(
        squared_i32.integral.size(),
        Size {
            width: 3,
            height: 3,
        }
    );
    assert_eq!(squared_i32.squared.size(), squared_i32.integral.size());
    assert_eq!(squared_f64.squared.size(), squared_f64.integral.size());
    assert_eq!(squared_f32.squared.size(), squared_f32.integral.size());
    assert_eq!(
        squared_i32.integral.values().copy_to_host_vec()?,
        [0, 0, 0, 0, 1, 3, 0, 4, 10]
    );
    assert_eq!(
        squared_i32.squared.values().copy_to_host_vec()?,
        [0, 0, 0, 0, 1, 5, 0, 10, 30]
    );
    assert_eq!(
        squared_f64.squared.values().copy_to_host_vec()?,
        [0.0, 0.0, 0.0, 0.0, 1.0, 5.0, 0.0, 10.0, 30.0]
    );
    assert_eq!(
        squared_f32.integral.values().copy_to_host_vec()?,
        [0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 4.0, 10.0]
    );
    assert_eq!(
        rect_i32.size(),
        Size {
            width: 1,
            height: 1,
        }
    );
    assert_eq!(
        rect_i32_to_f32.size(),
        Size {
            width: 1,
            height: 1,
        }
    );
    assert_eq!(
        rect_f32.size(),
        Size {
            width: 1,
            height: 1,
        }
    );

    Ok(())
}

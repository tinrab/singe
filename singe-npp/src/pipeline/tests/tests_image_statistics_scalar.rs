use super::*;

#[test]
fn fluent_image_pipeline_runs_scalar_statistics() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [1u8, 2, 3, 4];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(2, 2))?;
    let mut workspace = Workspace::create();

    let sum = ImagePipeline::from_view(&stream_context, &mut workspace, source).sum()?;
    let mean = ImagePipeline::from_view(&stream_context, &mut workspace, source).mean()?;
    let norm_inf = ImagePipeline::from_view(&stream_context, &mut workspace, source).norm_inf()?;
    let norm_l1 = ImagePipeline::from_view(&stream_context, &mut workspace, source).norm_l1()?;
    let norm_l2 = ImagePipeline::from_view(&stream_context, &mut workspace, source).norm_l2()?;

    stream.synchronize()?;

    assert_eq!(sum.len(), 1);
    assert_eq!(mean.len(), 1);
    assert_eq!(norm_inf.len(), 1);
    assert_eq!(norm_l1.len(), 1);
    assert_eq!(norm_l2.len(), 1);
    assert_eq!(sum.values().copy_to_host_vec()?, [10.0]);
    assert_eq!(mean.values().copy_to_host_vec()?, [2.5]);
    assert_eq!(norm_inf.values().copy_to_host_vec()?, [4.0]);
    assert_eq!(norm_l1.values().copy_to_host_vec()?, [10.0]);
    assert!((norm_l2.values().copy_to_host_vec()?[0] - 30.0f64.sqrt()).abs() < 1e-9);

    Ok(())
}

#[test]
fn fluent_image_pipeline_runs_per_channel_scalar_statistics() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_c3_host = [1u8, 2, 3, 4, 5, 6];
    let source_c4_host = [1u8, 2, 3, 4, 5, 6, 7, 8];
    let source_c3_memory = DeviceMemory::from_slice(&source_c3_host)?;
    let source_c4_memory = DeviceMemory::from_slice(&source_c4_host)?;
    let source_c3 = ImageView::<_, C3>::from_memory(&source_c3_memory, Size::new(2, 1))?;
    let source_c4 = ImageView::<_, C4>::from_memory(&source_c4_memory, Size::new(2, 1))?;
    let mut workspace = Workspace::create();

    let c3_sum = ImagePipeline::from_view(&stream_context, &mut workspace, source_c3).sum()?;
    let c3_mean = ImagePipeline::from_view(&stream_context, &mut workspace, source_c3).mean()?;
    let c3_norm_inf =
        ImagePipeline::from_view(&stream_context, &mut workspace, source_c3).norm_inf()?;
    let c4_sum = ImagePipeline::from_view(&stream_context, &mut workspace, source_c4).sum()?;
    let c4_norm_l1 =
        ImagePipeline::from_view(&stream_context, &mut workspace, source_c4).norm_l1()?;
    let c4_norm_l2 =
        ImagePipeline::from_view(&stream_context, &mut workspace, source_c4).norm_l2()?;

    stream.synchronize()?;

    assert_eq!(c3_sum.len(), 3);
    assert_eq!(c4_sum.len(), 4);
    assert_eq!(c3_sum.values().copy_to_host_vec()?, [5.0, 7.0, 9.0]);
    assert_eq!(c3_mean.values().copy_to_host_vec()?, [2.5, 3.5, 4.5]);
    assert_eq!(c3_norm_inf.values().copy_to_host_vec()?, [4.0, 5.0, 6.0]);
    assert_eq!(c4_sum.values().copy_to_host_vec()?, [6.0, 8.0, 10.0, 12.0]);
    assert_eq!(
        c4_norm_l1.values().copy_to_host_vec()?,
        [6.0, 8.0, 10.0, 12.0]
    );

    let c4_norm_l2_values = c4_norm_l2.values().copy_to_host_vec()?;
    assert!((c4_norm_l2_values[0] - 26.0f64.sqrt()).abs() < 1e-9);
    assert!((c4_norm_l2_values[1] - 40.0f64.sqrt()).abs() < 1e-9);
    assert!((c4_norm_l2_values[2] - 58.0f64.sqrt()).abs() < 1e-9);
    assert!((c4_norm_l2_values[3] - 80.0f64.sqrt()).abs() < 1e-9);

    Ok(())
}

#[test]
fn fluent_image_pipeline_runs_extremum_statistics() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_c1_host = [3u8, 1, 4, 2];
    let source_c3_host = [9u8, 2, 7, 4, 5, 6];
    let source_c4_host = [8u8, 2, 7, 4, 1, 6, 3, 5];
    let source_c1_memory = DeviceMemory::from_slice(&source_c1_host)?;
    let source_c3_memory = DeviceMemory::from_slice(&source_c3_host)?;
    let source_c4_memory = DeviceMemory::from_slice(&source_c4_host)?;
    let source_c1 = ImageView::<_, C1>::from_memory(&source_c1_memory, Size::new(2, 2))?;
    let source_c3 = ImageView::<_, C3>::from_memory(&source_c3_memory, Size::new(2, 1))?;
    let source_c4 = ImageView::<_, C4>::from_memory(&source_c4_memory, Size::new(2, 1))?;
    let mut workspace = Workspace::create();

    let c1_min = ImagePipeline::from_view(&stream_context, &mut workspace, source_c1).min()?;
    let c1_max = ImagePipeline::from_view(&stream_context, &mut workspace, source_c1).max()?;
    let c1_min_max =
        ImagePipeline::from_view(&stream_context, &mut workspace, source_c1).min_max()?;
    let c3_min_max =
        ImagePipeline::from_view(&stream_context, &mut workspace, source_c3).min_max()?;
    let c4_min = ImagePipeline::from_view(&stream_context, &mut workspace, source_c4).min()?;
    let c4_max = ImagePipeline::from_view(&stream_context, &mut workspace, source_c4).max()?;

    stream.synchronize()?;

    assert_eq!(c1_min.values().copy_to_host_vec()?, [1]);
    assert_eq!(c1_max.values().copy_to_host_vec()?, [4]);
    assert_eq!(c1_min_max.min.values().copy_to_host_vec()?, [1]);
    assert_eq!(c1_min_max.max.values().copy_to_host_vec()?, [4]);
    assert_eq!(c3_min_max.min.values().copy_to_host_vec()?, [4, 2, 6]);
    assert_eq!(c3_min_max.max.values().copy_to_host_vec()?, [9, 5, 7]);
    assert_eq!(c4_min.values().copy_to_host_vec()?, [1, 2, 3, 4]);
    assert_eq!(c4_max.values().copy_to_host_vec()?, [8, 6, 7, 5]);

    Ok(())
}

#[test]
fn fluent_image_pipeline_runs_every_statistics() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [1u8, 9, 3, 7];
    let other_host = [2u8, 4, 6, 8];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let other_memory = DeviceMemory::from_slice(&other_host)?;
    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(2, 2))?;
    let other = ImageView::<_, C1>::from_memory(&other_memory, Size::new(2, 2))?;
    let mut workspace = Workspace::create();

    let max_sum = ImagePipeline::from_view(&stream_context, &mut workspace, source)
        .max_every(&other)?
        .sum()?;
    let min_sum = ImagePipeline::from_view(&stream_context, &mut workspace, source)
        .min_every(&other)?
        .sum()?;

    stream.synchronize()?;

    assert_eq!(max_sum.values().copy_to_host_vec()?, [25.0]);
    assert_eq!(min_sum.values().copy_to_host_vec()?, [15.0]);

    Ok(())
}

#[test]
fn fluent_image_pipeline_runs_indexed_extremum_statistics() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_c1_host = [3u8, 1, 4, 2];
    let source_c3_host = [9u8, 2, 7, 4, 5, 6];
    let source_c1_memory = DeviceMemory::from_slice(&source_c1_host)?;
    let source_c3_memory = DeviceMemory::from_slice(&source_c3_host)?;
    let source_c1 = ImageView::<_, C1>::from_memory(&source_c1_memory, Size::new(2, 2))?;
    let source_c3 = ImageView::<_, C3>::from_memory(&source_c3_memory, Size::new(2, 1))?;
    let mut workspace = Workspace::create();

    let c1_min =
        ImagePipeline::from_view(&stream_context, &mut workspace, source_c1).min_index()?;
    let c1_max =
        ImagePipeline::from_view(&stream_context, &mut workspace, source_c1).max_index()?;
    let c3_min =
        ImagePipeline::from_view(&stream_context, &mut workspace, source_c3).min_index()?;
    let c3_max =
        ImagePipeline::from_view(&stream_context, &mut workspace, source_c3).max_index()?;

    stream.synchronize()?;

    assert_eq!(c1_min.value.values().copy_to_host_vec()?, [1]);
    assert_eq!(c1_min.index_x.values().copy_to_host_vec()?, [1]);
    assert_eq!(c1_min.index_y.values().copy_to_host_vec()?, [0]);
    assert_eq!(c1_max.value.values().copy_to_host_vec()?, [4]);
    assert_eq!(c1_max.index_x.values().copy_to_host_vec()?, [0]);
    assert_eq!(c1_max.index_y.values().copy_to_host_vec()?, [1]);
    assert_eq!(c3_min.value.values().copy_to_host_vec()?, [4, 2, 6]);
    assert_eq!(c3_min.index_x.values().copy_to_host_vec()?, [1, 0, 1]);
    assert_eq!(c3_min.index_y.values().copy_to_host_vec()?, [0, 0, 0]);
    assert_eq!(c3_max.value.values().copy_to_host_vec()?, [9, 5, 7]);
    assert_eq!(c3_max.index_x.values().copy_to_host_vec()?, [0, 1, 0]);
    assert_eq!(c3_max.index_y.values().copy_to_host_vec()?, [0, 0, 0]);

    Ok(())
}

#[test]
fn fluent_image_pipeline_runs_paired_indexed_extremum_statistics() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_c1_host = [3u8, 1, 4, 2];
    let source_c3_host = [9u8, 2, 7, 4, 5, 6];
    let source_c1_memory = DeviceMemory::from_slice(&source_c1_host)?;
    let source_c3_memory = DeviceMemory::from_slice(&source_c3_host)?;
    let source_c1 = ImageView::<_, C1>::from_memory(&source_c1_memory, Size::new(2, 2))?;
    let source_c3 = ImageView::<_, C3>::from_memory(&source_c3_memory, Size::new(2, 1))?;
    let mut workspace = Workspace::create();

    let c1 =
        ImagePipeline::from_view(&stream_context, &mut workspace, source_c1).min_max_index()?;
    let c3_channel_2 = ImagePipeline::from_view(&stream_context, &mut workspace, source_c3)
        .min_max_index_channel(2)?;

    stream.synchronize()?;

    assert_eq!(c1.min.value.values().copy_to_host_vec()?, [1]);
    assert_eq!(c1.min.index_x.values().copy_to_host_vec()?, [1]);
    assert_eq!(c1.min.index_y.values().copy_to_host_vec()?, [0]);
    assert_eq!(c1.max.value.values().copy_to_host_vec()?, [4]);
    assert_eq!(c1.max.index_x.values().copy_to_host_vec()?, [0]);
    assert_eq!(c1.max.index_y.values().copy_to_host_vec()?, [1]);
    assert_eq!(c3_channel_2.min.value.values().copy_to_host_vec()?, [6]);
    assert_eq!(c3_channel_2.min.index_x.values().copy_to_host_vec()?, [1]);
    assert_eq!(c3_channel_2.min.index_y.values().copy_to_host_vec()?, [0]);
    assert_eq!(c3_channel_2.max.value.values().copy_to_host_vec()?, [7]);
    assert_eq!(c3_channel_2.max.index_x.values().copy_to_host_vec()?, [0]);
    assert_eq!(c3_channel_2.max.index_y.values().copy_to_host_vec()?, [0]);

    Ok(())
}

#[test]
fn fluent_image_pipeline_runs_count_in_range_statistics() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_c1_host = [1.0f32, 2.0, 3.0, 4.0];
    let source_c3_host = [1u8, 2, 3, 4, 5, 6];
    let source_ac4_host = [1u8, 2, 3, 99, 4, 5, 6, 99];
    let source_c1_memory = DeviceMemory::from_slice(&source_c1_host)?;
    let source_c3_memory = DeviceMemory::from_slice(&source_c3_host)?;
    let source_ac4_memory = DeviceMemory::from_slice(&source_ac4_host)?;
    let source_c1 = ImageView::<_, C1>::from_memory(&source_c1_memory, Size::new(2, 2))?;
    let source_c3 = ImageView::<_, C3>::from_memory(&source_c3_memory, Size::new(2, 1))?;
    let source_ac4 = ImageView::<_, AC4>::from_memory(&source_ac4_memory, Size::new(2, 1))?;
    let mut workspace = Workspace::create();

    let c1_count = ImagePipeline::from_view(&stream_context, &mut workspace, source_c1)
        .count_in_range(1.5, 3.5)?;
    let c3_count = ImagePipeline::from_view(&stream_context, &mut workspace, source_c3)
        .count_in_range([0, 0, 0], [10, 10, 10])?;
    let ac4_count = ImagePipeline::from_view(&stream_context, &mut workspace, source_ac4)
        .count_in_range([0, 0, 0], [10, 10, 10])?;

    stream.synchronize()?;

    assert_eq!(c1_count.values().copy_to_host_vec()?, [2]);
    assert_eq!(c3_count.values().copy_to_host_vec()?, [2, 2, 2]);
    assert_eq!(ac4_count.values().copy_to_host_vec()?, [2, 2, 2]);

    Ok(())
}

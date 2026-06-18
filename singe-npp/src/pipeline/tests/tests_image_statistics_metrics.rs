use super::*;

#[test]
fn fluent_image_pipeline_runs_histogram_even_statistics() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [0u8, 1, 2, 3];
    let source_c3_host = [0u8, 1, 2, 2, 3, 0];
    let source_ac4_host = [0u8, 1, 2, 99, 2, 3, 0, 99];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let source_c3_memory = DeviceMemory::from_slice(&source_c3_host)?;
    let source_ac4_memory = DeviceMemory::from_slice(&source_ac4_host)?;
    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(2, 2))?;
    let source_c3 = ImageView::<_, C3>::from_memory(&source_c3_memory, Size::new(2, 1))?;
    let source_ac4 = ImageView::<_, AC4>::from_memory(&source_ac4_memory, Size::new(2, 1))?;
    let mut workspace = Workspace::create();

    let histogram = ImagePipeline::from_view(&stream_context, &mut workspace, source)
        .histogram_even(3, 0, 4)?;
    let c3_histograms = ImagePipeline::from_view(&stream_context, &mut workspace, source_c3)
        .histogram_even_channels([3, 3, 3], [0, 0, 0], [4, 4, 4])?;
    let ac4_histograms = ImagePipeline::from_view(&stream_context, &mut workspace, source_ac4)
        .histogram_even_channels([3, 3, 3], [0, 0, 0], [4, 4, 4])?;

    stream.synchronize()?;

    assert_eq!(histogram.values().copy_to_host_vec()?, [2, 2]);
    assert_eq!(c3_histograms.len(), 3);
    assert_eq!(
        c3_histograms.channels[0].values().copy_to_host_vec()?,
        [1, 1]
    );
    assert_eq!(
        c3_histograms.channels[1].values().copy_to_host_vec()?,
        [1, 1]
    );
    assert_eq!(
        c3_histograms.channels[2].values().copy_to_host_vec()?,
        [1, 1]
    );
    assert_eq!(ac4_histograms.len(), 3);
    assert_eq!(
        ac4_histograms.channels[0].values().copy_to_host_vec()?,
        [1, 1]
    );
    assert_eq!(
        ac4_histograms.channels[1].values().copy_to_host_vec()?,
        [1, 1]
    );
    assert_eq!(
        ac4_histograms.channels[2].values().copy_to_host_vec()?,
        [1, 1]
    );

    Ok(())
}

#[test]
fn fluent_image_pipeline_runs_histogram_range_statistics() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [0u8, 1, 2, 3];
    let source_c3_host = [0u8, 1, 2, 2, 3, 0];
    let source_ac4_host = [0u8, 1, 2, 99, 2, 3, 0, 99];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let source_c3_memory = DeviceMemory::from_slice(&source_c3_host)?;
    let source_ac4_memory = DeviceMemory::from_slice(&source_ac4_host)?;
    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(2, 2))?;
    let source_c3 = ImageView::<_, C3>::from_memory(&source_c3_memory, Size::new(2, 1))?;
    let source_ac4 = ImageView::<_, AC4>::from_memory(&source_ac4_memory, Size::new(2, 1))?;
    let mut workspace = Workspace::create();
    let levels = [0, 2, 4];

    let histogram = ImagePipeline::from_view(&stream_context, &mut workspace, source)
        .histogram_range(&levels)?;
    let c3_histograms = ImagePipeline::from_view(&stream_context, &mut workspace, source_c3)
        .histogram_range_channels([&levels, &levels, &levels])?;
    let ac4_histograms = ImagePipeline::from_view(&stream_context, &mut workspace, source_ac4)
        .histogram_range_channels([&levels, &levels, &levels])?;

    stream.synchronize()?;

    assert_eq!(histogram.values().copy_to_host_vec()?, [2, 2]);
    assert_eq!(c3_histograms.len(), 3);
    assert_eq!(
        c3_histograms.channels[0].values().copy_to_host_vec()?,
        [1, 1]
    );
    assert_eq!(
        c3_histograms.channels[1].values().copy_to_host_vec()?,
        [1, 1]
    );
    assert_eq!(
        c3_histograms.channels[2].values().copy_to_host_vec()?,
        [1, 1]
    );
    assert_eq!(ac4_histograms.len(), 3);
    assert_eq!(
        ac4_histograms.channels[0].values().copy_to_host_vec()?,
        [1, 1]
    );
    assert_eq!(
        ac4_histograms.channels[1].values().copy_to_host_vec()?,
        [1, 1]
    );
    assert_eq!(
        ac4_histograms.channels[2].values().copy_to_host_vec()?,
        [1, 1]
    );

    Ok(())
}

#[test]
fn fluent_image_pipeline_runs_dot_product_statistics() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_c1_host = [1u8, 2, 3, 4];
    let other_c1_host = [5u8, 6, 7, 8];
    let source_c3_host = [1u8, 2, 3, 4, 5, 6];
    let other_c3_host = [1u8, 1, 1, 2, 2, 2];
    let source_c1_memory = DeviceMemory::from_slice(&source_c1_host)?;
    let other_c1_memory = DeviceMemory::from_slice(&other_c1_host)?;
    let source_c3_memory = DeviceMemory::from_slice(&source_c3_host)?;
    let other_c3_memory = DeviceMemory::from_slice(&other_c3_host)?;
    let source_c1 = ImageView::<_, C1>::from_memory(&source_c1_memory, Size::new(2, 2))?;
    let other_c1 = ImageView::<_, C1>::from_memory(&other_c1_memory, Size::new(2, 2))?;
    let source_c3 = ImageView::<_, C3>::from_memory(&source_c3_memory, Size::new(2, 1))?;
    let other_c3 = ImageView::<_, C3>::from_memory(&other_c3_memory, Size::new(2, 1))?;
    let mut workspace = Workspace::create();

    let c1_dot = ImagePipeline::from_view(&stream_context, &mut workspace, source_c1)
        .dot_product(&other_c1)?;
    let c3_dot = ImagePipeline::from_view(&stream_context, &mut workspace, source_c3)
        .dot_product(&other_c3)?;

    stream.synchronize()?;

    assert_eq!(c1_dot.values().copy_to_host_vec()?, [70.0]);
    assert_eq!(c3_dot.values().copy_to_host_vec()?, [9.0, 12.0, 15.0]);

    Ok(())
}

#[test]
fn fluent_image_pipeline_runs_error_metric_statistics() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_c1_host = [1u8, 2, 3, 4];
    let other_c1_host = [1u8, 4, 2, 8];
    let source_c3_host = [1u8, 2, 3, 4, 5, 6];
    let other_c3_host = [1u8, 4, 2, 8, 3, 6];
    let source_c1_memory = DeviceMemory::from_slice(&source_c1_host)?;
    let other_c1_memory = DeviceMemory::from_slice(&other_c1_host)?;
    let source_c3_memory = DeviceMemory::from_slice(&source_c3_host)?;
    let other_c3_memory = DeviceMemory::from_slice(&other_c3_host)?;
    let source_c1 = ImageView::<_, C1>::from_memory(&source_c1_memory, Size::new(2, 2))?;
    let other_c1 = ImageView::<_, C1>::from_memory(&other_c1_memory, Size::new(2, 2))?;
    let source_c3 = ImageView::<_, C3>::from_memory(&source_c3_memory, Size::new(2, 1))?;
    let other_c3 = ImageView::<_, C3>::from_memory(&other_c3_memory, Size::new(2, 1))?;
    let mut workspace = Workspace::create();

    let c1_maximum = ImagePipeline::from_view(&stream_context, &mut workspace, source_c1)
        .maximum_error(&other_c1)?;
    let c1_average = ImagePipeline::from_view(&stream_context, &mut workspace, source_c1)
        .average_error(&other_c1)?;
    let c3_maximum = ImagePipeline::from_view(&stream_context, &mut workspace, source_c3)
        .maximum_error(&other_c3)?;
    let c3_average = ImagePipeline::from_view(&stream_context, &mut workspace, source_c3)
        .average_error(&other_c3)?;

    stream.synchronize()?;

    assert_eq!(c1_maximum.values().copy_to_host_vec()?, [4.0]);
    assert_eq!(c1_average.values().copy_to_host_vec()?, [1.75]);
    assert_eq!(c3_maximum.values().copy_to_host_vec()?, [4.0]);
    assert_eq!(c3_average.values().copy_to_host_vec()?, [1.5]);

    Ok(())
}

#[test]
fn fluent_image_pipeline_runs_quality_metric_statistics() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_c1_host = [0u8, 10, 20, 30];
    let other_c1_host = [0u8, 20, 10, 50];
    let source_c3_host = [0u8, 10, 20, 30, 40, 50];
    let other_c3_host = [0u8, 20, 40, 40, 20, 50];
    let source_c1_memory = DeviceMemory::from_slice(&source_c1_host)?;
    let other_c1_memory = DeviceMemory::from_slice(&other_c1_host)?;
    let source_c3_memory = DeviceMemory::from_slice(&source_c3_host)?;
    let other_c3_memory = DeviceMemory::from_slice(&other_c3_host)?;
    let source_c1 = ImageView::<_, C1>::from_memory(&source_c1_memory, Size::new(2, 2))?;
    let other_c1 = ImageView::<_, C1>::from_memory(&other_c1_memory, Size::new(2, 2))?;
    let source_c3 = ImageView::<_, C3>::from_memory(&source_c3_memory, Size::new(2, 1))?;
    let other_c3 = ImageView::<_, C3>::from_memory(&other_c3_memory, Size::new(2, 1))?;
    let mut workspace = Workspace::create();

    let c1_mse = ImagePipeline::from_view(&stream_context, &mut workspace, source_c1)
        .mean_squared_error(&other_c1)?;
    let c1_psnr = ImagePipeline::from_view(&stream_context, &mut workspace, source_c1)
        .peak_signal_to_noise_ratio(&other_c1)?;
    let c3_mse = ImagePipeline::from_view(&stream_context, &mut workspace, source_c3)
        .mean_squared_error(&other_c3)?;
    let c3_psnr = ImagePipeline::from_view(&stream_context, &mut workspace, source_c3)
        .peak_signal_to_noise_ratio(&other_c3)?;

    stream.synchronize()?;

    assert_eq!(c1_mse.values().copy_to_host_vec()?, [150.0]);
    assert_eq!(c3_mse.values().copy_to_host_vec()?, [50.0, 250.0, 200.0]);
    assert_eq!(c1_psnr.len(), 1);
    assert_eq!(c3_psnr.len(), 3);
    assert!(c1_psnr.values().copy_to_host_vec()?[0].is_finite());
    assert!(
        c3_psnr
            .values()
            .copy_to_host_vec()?
            .into_iter()
            .all(f32::is_finite)
    );

    Ok(())
}

#[test]
fn fluent_image_pipeline_runs_structural_similarity_statistics() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let width = 256;
    let height = 256;
    let source_c1_host: Vec<u8> = (0..width * height)
        .map(|index| (index % 251) as u8)
        .collect();
    let other_c1_host: Vec<u8> = source_c1_host
        .iter()
        .enumerate()
        .map(|(index, value)| value.saturating_add((index % 7 == 0) as u8))
        .collect();
    let source_c3_host: Vec<u8> = (0..width * height * 3)
        .map(|index| (index % 251) as u8)
        .collect();
    let other_c3_host: Vec<u8> = source_c3_host
        .iter()
        .enumerate()
        .map(|(index, value)| value.saturating_add((index % 11 == 0) as u8))
        .collect();
    let source_c4_host: Vec<u8> = (0..width * height * 4)
        .map(|index| (index % 251) as u8)
        .collect();
    let other_c4_host: Vec<u8> = source_c4_host
        .iter()
        .enumerate()
        .map(|(index, value)| value.saturating_add((index % 13 == 0) as u8))
        .collect();
    let source_c1_memory = DeviceMemory::from_slice(&source_c1_host)?;
    let other_c1_memory = DeviceMemory::from_slice(&other_c1_host)?;
    let source_c3_memory = DeviceMemory::from_slice(&source_c3_host)?;
    let other_c3_memory = DeviceMemory::from_slice(&other_c3_host)?;
    let source_c4_memory = DeviceMemory::from_slice(&source_c4_host)?;
    let other_c4_memory = DeviceMemory::from_slice(&other_c4_host)?;
    let source_c1 = ImageView::<_, C1>::from_memory(&source_c1_memory, Size::new(width, height))?;
    let other_c1 = ImageView::<_, C1>::from_memory(&other_c1_memory, Size::new(width, height))?;
    let source_c3 = ImageView::<_, C3>::from_memory(&source_c3_memory, Size::new(width, height))?;
    let other_c3 = ImageView::<_, C3>::from_memory(&other_c3_memory, Size::new(width, height))?;
    let source_c4 = ImageView::<_, C4>::from_memory(&source_c4_memory, Size::new(width, height))?;
    let other_c4 = ImageView::<_, C4>::from_memory(&other_c4_memory, Size::new(width, height))?;
    let mut workspace = Workspace::create();

    let c1_ssim = ImagePipeline::from_view(&stream_context, &mut workspace, source_c1)
        .structural_similarity(&other_c1)?;
    let c3_ssim = ImagePipeline::from_view(&stream_context, &mut workspace, source_c3)
        .structural_similarity(&other_c3)?;
    let c1_msssim = ImagePipeline::from_view(&stream_context, &mut workspace, source_c1)
        .multiscale_structural_similarity(&other_c1)?;
    let c1_wmsssim = ImagePipeline::from_view(&stream_context, &mut workspace, source_c1)
        .weighted_multiscale_structural_similarity(&other_c1)?;
    let c3_wmsssim = ImagePipeline::from_view(&stream_context, &mut workspace, source_c3)
        .weighted_multiscale_structural_similarity(&other_c3)?;
    let c4_wmsssim = ImagePipeline::from_view(&stream_context, &mut workspace, source_c4)
        .weighted_multiscale_structural_similarity(&other_c4)?;

    stream.synchronize()?;

    assert_eq!(c1_ssim.len(), 1);
    assert_eq!(c3_ssim.len(), 3);
    assert_eq!(c1_msssim.len(), 1);
    assert_eq!(c1_wmsssim.len(), 1);
    assert_eq!(c3_wmsssim.len(), 3);
    assert_eq!(c4_wmsssim.len(), 4);
    for values in [
        c1_ssim.values().copy_to_host_vec()?,
        c3_ssim.values().copy_to_host_vec()?,
        c1_msssim.values().copy_to_host_vec()?,
        c1_wmsssim.values().copy_to_host_vec()?,
        c3_wmsssim.values().copy_to_host_vec()?,
        c4_wmsssim.values().copy_to_host_vec()?,
    ] {
        assert!(values.into_iter().all(f32::is_finite));
    }

    Ok(())
}

#[test]
fn fluent_image_pipeline_runs_radial_profile_statistics() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let width = 8;
    let height = 8;
    let source_host: Vec<f32> = (0..width * height).map(|index| index as f32).collect();
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(width, height))?;
    let center = Point32f { x: 3.5, y: 3.5 };
    let mut workspace = Workspace::create();

    let circular = ImagePipeline::from_view(&stream_context, &mut workspace, source)
        .circular_radial_profile(center, 4)?;
    let elliptical = ImagePipeline::from_view(&stream_context, &mut workspace, source)
        .elliptical_radial_profile(center, 1.25, 0.5, 4)?;

    stream.synchronize()?;

    assert_eq!(circular.len(), 4);
    assert_eq!(elliptical.len(), 4);
    assert!(circular.iter().any(|sample| sample.pixels > 0));
    assert!(elliptical.iter().any(|sample| sample.pixels > 0));
    assert!(
        circular
            .iter()
            .all(|sample| sample.mean_intensity.is_finite()
                && sample.standard_deviation_intensity.is_finite())
    );
    assert!(
        elliptical
            .iter()
            .all(|sample| sample.mean_intensity.is_finite()
                && sample.standard_deviation_intensity.is_finite())
    );

    Ok(())
}

#[test]
fn fluent_image_pipeline_runs_mean_standard_deviation_statistics() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_c1_host = [1u8, 2, 3, 4];
    let source_c3_host = [1u8, 2, 3, 4, 5, 6];
    let source_c1_memory = DeviceMemory::from_slice(&source_c1_host)?;
    let source_c3_memory = DeviceMemory::from_slice(&source_c3_host)?;
    let source_c1 = ImageView::<_, C1>::from_memory(&source_c1_memory, Size::new(2, 2))?;
    let source_c3 = ImageView::<_, C3>::from_memory(&source_c3_memory, Size::new(2, 1))?;
    let mut workspace = Workspace::create();

    let c1 = ImagePipeline::from_view(&stream_context, &mut workspace, source_c1)
        .mean_standard_deviation()?;
    let c3_channel_0 = ImagePipeline::from_view(&stream_context, &mut workspace, source_c3)
        .mean_standard_deviation_channel(0)?;
    let c3_channel_2 = ImagePipeline::from_view(&stream_context, &mut workspace, source_c3)
        .mean_standard_deviation_channel(2)?;

    stream.synchronize()?;

    assert_eq!(c1.mean.values().copy_to_host_vec()?, [2.5]);
    assert!((c1.standard_deviation.values().copy_to_host_vec()?[0] - 1.25f64.sqrt()).abs() < 1e-9);
    assert_eq!(c3_channel_0.mean.values().copy_to_host_vec()?, [2.5]);
    assert_eq!(c3_channel_2.mean.values().copy_to_host_vec()?, [4.5]);
    assert!(
        (c3_channel_0
            .standard_deviation
            .values()
            .copy_to_host_vec()?[0]
            - 1.5)
            .abs()
            < 1e-9
    );
    assert!(
        (c3_channel_2
            .standard_deviation
            .values()
            .copy_to_host_vec()?[0]
            - 1.5)
            .abs()
            < 1e-9
    );

    Ok(())
}

#[test]
fn fluent_image_pipeline_chains_processing_into_scalar_statistic() -> Result<()> {
    let (stream, stream_context) = create_stream_context()?;
    let source_host = [1u8, 2, 3, 4];
    let source_memory = DeviceMemory::from_slice(&source_host)?;
    let source = ImageView::<_, C1>::from_memory(&source_memory, Size::new(2, 2))?;
    let mut workspace = Workspace::create();

    let sum = ImagePipeline::from_view(&stream_context, &mut workspace, source)
        .add_constant(1, 0)?
        .sum()?;

    stream.synchronize()?;

    assert_eq!(sum.values().copy_to_host_vec()?, [14.0]);

    Ok(())
}

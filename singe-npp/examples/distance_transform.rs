mod common;

use singe_cuda::memory::DeviceMemory;
use singe_npp::{
    image::{
        raw::filtering,
        view::{C1, C2, ImageView, ImageViewMut},
    },
    types::Size,
};

use common::{
    Result, count_nonzero, create_stream, download, image_path, normalize_f32_to_u8,
    normalize_i16_c2_magnitude_to_u8, normalize_u16_to_u8, output_dir, read_raw_u8, upload,
    write_png,
};

fn main() -> Result<()> {
    let (stream, stream_context) = create_stream()?;
    let output = output_dir("distance_transform")?;

    let dolphin = run_case(
        &stream_context,
        "Dolphin1_313x317_8u.raw",
        Size::new(313, 317),
    )?;
    let diamond = run_case(
        &stream_context,
        "TestImage3_diamond_64x64_8u.raw",
        Size::new(64, 64),
    )?;
    stream.synchronize()?;

    write_png(
        output.join("DistanceTransformTruncated_Dolphin1_313x317_16u.png"),
        &normalize_u16_to_u8(&dolphin.truncated),
        Size::new(313, 317),
    )?;
    write_png(
        output.join("DistanceTransformVoronoi_Dolphin1_313x317_16s.png"),
        &normalize_i16_c2_magnitude_to_u8(&dolphin.voronoi),
        Size::new(313, 317),
    )?;
    write_png(
        output.join("DistanceTransformTrue_Dolphin1_313x317_32f.png"),
        &normalize_f32_to_u8(&dolphin.transform),
        Size::new(313, 317),
    )?;
    write_png(
        output.join("DistanceTransformTruncated_TestImage3_diamond_64x64_16u.png"),
        &normalize_u16_to_u8(&diamond.truncated),
        Size::new(64, 64),
    )?;
    write_png(
        output.join("DistanceTransformVoronoi_TestImage3_64x64_16s.png"),
        &normalize_i16_c2_magnitude_to_u8(&diamond.voronoi),
        Size::new(64, 64),
    )?;
    write_png(
        output.join("DistanceTransformTrue_TestImage3_diamond_64x64_32f.png"),
        &normalize_f32_to_u8(&diamond.transform),
        Size::new(64, 64),
    )?;

    assert!(count_nonzero(&dolphin.truncated) > 0);
    assert!(diamond.transform.iter().any(|&value| value > 0.0));
    println!(
        "distance_transform: wrote outputs for Dolphin1 and TestImage3 to {}",
        output.display()
    );
    Ok(())
}

struct Outputs {
    truncated: Vec<u16>,
    voronoi: Vec<i16>,
    transform: Vec<f32>,
}

fn run_case(
    stream_context: &singe_npp::context::StreamContext,
    filename: &str,
    size: Size,
) -> Result<Outputs> {
    let input = read_raw_u8(image_path("distance_transform", filename), size)?;
    let source_device = upload(&input)?;
    let source = ImageView::<u8, C1>::from_memory(&source_device, size)?;

    let mut truncated_device =
        DeviceMemory::<u16>::zeroes(size.width as usize * size.height as usize)?;
    let mut voronoi_device =
        DeviceMemory::<i16>::zeroes(size.width as usize * size.height as usize * 2)?;
    let mut transform_device =
        DeviceMemory::<f32>::zeroes(size.width as usize * size.height as usize)?;

    let mut truncated = ImageViewMut::<u16, C1>::from_memory(&mut truncated_device, size)?;
    let mut voronoi = ImageViewMut::<i16, C2>::from_memory(&mut voronoi_device, size)?;
    let mut transform = ImageViewMut::<f32, C1>::from_memory(&mut transform_device, size)?;

    filtering::distance_transform_pba_u8_to_u16_c1(
        stream_context,
        &source,
        0,
        0,
        None,
        None,
        None,
        Some(&mut truncated),
    )?;
    filtering::distance_transform_pba_u8_to_u16_c1(
        stream_context,
        &source,
        0,
        0,
        Some(&mut voronoi),
        None,
        None,
        None,
    )?;
    filtering::distance_transform_pba_u8_to_f32_c1(
        stream_context,
        &source,
        0,
        0,
        None,
        None,
        None,
        Some(&mut transform),
    )?;

    Ok(Outputs {
        truncated: download(&truncated_device)?,
        voronoi: download(&voronoi_device)?,
        transform: download(&transform_device)?,
    })
}

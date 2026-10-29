mod common;

use singe_npp::{
    image::view::{C1, ImageView},
    pipeline::{ImagePipeline, Workspace},
    types::{ImageNormalization, Size},
};

use common::{
    Result, create_stream, image_path, normalize_u32_to_u8, output_dir, read_raw_u8, upload,
    write_png,
};

const SIZE: Size = Size::new(2048, 1024);

fn main() -> Result<()> {
    let (stream, stream_context) = create_stream()?;
    let input = read_raw_u8(
        image_path("find_contour", "CircuitBoard_2048x1024_8u.raw"),
        SIZE,
    )?;
    let source_device = upload(stream.context(), &input)?;
    let source = ImageView::<u8, C1>::from_buffer(&source_device, SIZE)?;
    let mut workspace = Workspace::create();
    let labels = ImagePipeline::from_view(&stream_context, &mut workspace, source)?
        .label_markers_uf(ImageNormalization::Infinity)?;
    let compressed = labels.compress(&stream_context)?;
    let compressed_count = compressed.label_count();
    assert!(compressed_count > 0);
    let contours = compressed.contours(&stream_context)?;
    let contour_pixels = contours.totals().total_image_pixel_contour_count;
    assert!(contour_pixels > 0);
    let labels_host = contours.labels().values().copy_to_host_vec()?;
    let contours_host = contours.image().values().copy_to_host_vec()?;

    let output = output_dir("find_contour")?;
    write_png(
        output.join("CircuitBoard_CompressedMarkerLabelsUF_8Way_2048x1024_32u.png"),
        &normalize_u32_to_u8(&labels_host),
        SIZE,
    )?;
    write_png(
        output.join("CircuitBoard_Contours_8Way_2048x1024_8u.png"),
        &contours_host,
        SIZE,
    )?;

    println!(
        "find_contour: compressed labels {}, contour pixels {}, wrote {}",
        compressed_count,
        contour_pixels,
        output.display()
    );
    Ok(())
}

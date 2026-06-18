mod common;

use singe_npp::{
    image::view::{C3, ImageView},
    pipeline::{ImagePipeline, Resize, Workspace},
    types::{
        BorderType, ComparisonOperation, ImageNormalization, InterpolationMode, Point, Size,
        WatershedSegmentBoundaryType,
    },
};

use common::{Result, create_stream, examples_dir, output_dir, upload, write_png};

const OUTPUT_SIZE: Size = Size {
    width: 768,
    height: 768,
};
const WATERSHED_SIZE: Size = Size {
    width: 384,
    height: 384,
};

fn main() -> Result<()> {
    let input_path = examples_dir().join("images").join("cat-1.jpg");
    let output_path = output_dir("pipeline_cat")?.join("cat_pipeline.png");
    let image = image::open(&input_path)?.to_rgb8();
    let size = Size::new(image.width() as i32, image.height() as i32);

    let (stream, stream_context) = create_stream()?;
    let source_device = upload(image.as_raw())?;
    let source = ImageView::<u8, C3>::from_memory(&source_device, size)?;

    let mut workspace = Workspace::create();
    let crop = size.centered_rectangle();

    let output_image = ImagePipeline::from_view(&stream_context, &mut workspace, source)
        .resize_with(Resize {
            source_roi: crop,
            destination_roi: OUTPUT_SIZE.rectangle(),
            interpolation: InterpolationMode::Lanczos,
        })?
        .rgb_to_gray()?
        .resize(WATERSHED_SIZE, InterpolationMode::Linear)?
        .threshold_less_greater_value(24, 0, 235, 255)?
        .threshold_value(92, 32, ComparisonOperation::Less)?
        .erode_3x3()?
        .dilate_3x3()?
        .segment_watershed(
            None, // marker_labels
            ImageNormalization::Infinity,
            WatershedSegmentBoundaryType::White,
        )?
        .filter_sharpen_border(Point::ZERO, BorderType::Replicate)?
        .finish()?;

    stream.synchronize()?;

    write_png(
        output_path,
        &output_image.copy_to_host_vec()?,
        WATERSHED_SIZE,
    )?;

    Ok(())
}

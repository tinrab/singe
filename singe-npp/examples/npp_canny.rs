mod common;

use singe_cuda::memory::DeviceMemory;
use singe_npp::{
    image::{
        raw::filtering,
        view::{C1, C3, ImageView, ImageViewMut},
    },
    types::{BorderType, DifferentialKernel, ImageNormalization, MaskSize, Point, Size},
};

use common::{
    Result, count_nonzero, create_stream, download, image_path, output_dir, upload, write_png,
};

fn main() -> Result<()> {
    let input_path = image_path("npp_canny", "example_input.png");
    let image = image::open(&input_path)?.to_rgb8();
    let (width, height) = image.dimensions();
    let size = Size::new(width as i32, height as i32);

    let (stream, stream_context) = create_stream()?;
    let source_device = upload(image.as_raw())?;
    let mut destination_device =
        DeviceMemory::<u8>::zeroes(size.width as usize * size.height as usize)?;

    let source = ImageView::<u8, C3>::from_memory(&source_device, size)?;
    let mut destination = ImageViewMut::<u8, C1>::from_memory(&mut destination_device, size)?;

    filtering::filter_canny_border_c3_to_c1(
        &stream_context,
        &source,
        Point { x: 0, y: 0 },
        &mut destination,
        DifferentialKernel::Sobel,
        MaskSize::Mask3x3,
        50,
        100,
        ImageNormalization::L2,
        BorderType::Replicate,
    )?;
    stream.synchronize()?;

    let edges = download(&destination_device)?;
    let edge_count = count_nonzero(&edges);
    assert!(edge_count > 0);

    let output_path = output_dir("npp_canny")?.join("edges_npp.png");
    write_png(&output_path, &edges, size)?;

    println!(
        "npp_canny: {}x{}, {edge_count} edge pixels, wrote {}",
        width,
        height,
        output_path.display()
    );
    Ok(())
}

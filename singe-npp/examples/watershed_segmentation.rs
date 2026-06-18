mod common;

use singe_cuda::memory::DeviceMemory;
use singe_npp::{
    image::{
        raw::filtering,
        view::{C1, ImageViewMut},
    },
    types::{ImageNormalization, Size, WatershedSegmentBoundaryType},
};

use common::{
    Result, count_nonzero, create_stream, download, image_path, normalize_u32_to_u8, output_dir,
    read_raw_u8, upload, write_png,
};

struct InputImage {
    name: &'static str,
    size: Size,
}

fn main() -> Result<()> {
    let images = [
        InputImage {
            name: "Lena_512x512_8u_Gray.raw",
            size: Size::new(512, 512),
        },
        InputImage {
            name: "CT_skull_512x512_8u_Gray.raw",
            size: Size::new(512, 512),
        },
        InputImage {
            name: "Rocks_512x512_8u_Gray.raw",
            size: Size::new(512, 512),
        },
        InputImage {
            name: "coins_500x383_8u_Gray.raw",
            size: Size::new(500, 383),
        },
        InputImage {
            name: "coins_overlay_500x569_8u_Gray.raw",
            size: Size::new(500, 569),
        },
    ];

    let (stream, stream_context) = create_stream()?;
    let output = output_dir("watershed_segmentation")?;

    for image in images {
        let input = read_raw_u8(image_path("watershed_segmentation", image.name), image.size)?;
        let mut segments_device = upload(&input)?;
        let mut labels_device =
            DeviceMemory::<u32>::zeroes(image.size.width as usize * image.size.height as usize)?;
        let mut segments = ImageViewMut::<u8, C1>::from_memory(&mut segments_device, image.size)?;
        let mut labels = ImageViewMut::<u32, C1>::from_memory(&mut labels_device, image.size)?;

        filtering::segment_watershed_u8_c1_in_place(
            &stream_context,
            &mut segments,
            Some(&mut labels),
            ImageNormalization::Infinity,
            WatershedSegmentBoundaryType::None,
        )?;
        stream.synchronize()?;

        let segments_host = download(&segments_device)?;
        let labels_host = download(&labels_device)?;
        assert!(count_nonzero(&labels_host) > 0);

        let stem = image.name.strip_suffix(".raw").unwrap_or(image.name);
        write_png(
            output.join(format!("Segmented_{stem}.png")),
            &segments_host,
            image.size,
        )?;
        write_png(
            output.join(format!("Labels_{stem}.png")),
            &normalize_u32_to_u8(&labels_host),
            image.size,
        )?;
        println!(
            "watershed_segmentation: processed {} ({} labels nonzero)",
            image.name,
            count_nonzero(&labels_host)
        );
    }

    Ok(())
}

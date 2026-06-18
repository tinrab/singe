mod common;

use singe_cuda::memory::DeviceMemory;
use singe_npp::{
    image::{
        raw::filtering,
        view::{C1, ImageView, ImageViewMut},
    },
    types::{
        CompressedMarkerLabelsInfo, ContourPixelDirectionInfo, ContourTotalsInfo,
        ImageNormalization, Size,
    },
};

use common::{
    Result, count_nonzero, create_stream, download, image_path, normalize_u32_to_u8, output_dir,
    read_raw_u8, upload, write_png,
};

const SIZE: Size = Size::new(2048, 1024);
const LENGTH: usize = SIZE.width as usize * SIZE.height as usize;

fn main() -> Result<()> {
    let (stream, stream_context) = create_stream()?;
    let input = read_raw_u8(
        image_path("find_contour", "CircuitBoard_2048x1024_8u.raw"),
        SIZE,
    )?;
    let source_device = upload(&input)?;
    let mut labels_device =
        DeviceMemory::<u32>::zeroes(SIZE.width as usize * SIZE.height as usize)?;

    let source = ImageView::<u8, C1>::from_memory(&source_device, SIZE)?;
    let mut labels = ImageViewMut::<u32, C1>::from_memory(&mut labels_device, SIZE)?;

    filtering::label_markers_uf_u8_to_u32_c1(
        &stream_context,
        &source,
        &mut labels,
        ImageNormalization::Infinity,
    )?;
    let compressed_count = filtering::compress_marker_labels_uf_u32_c1_in_place(
        &stream_context,
        &mut labels,
        SIZE.width * SIZE.height,
    )?;
    assert!(compressed_count > 0);

    let list_len = compressed_count as usize + 4;
    let mut contours_device = DeviceMemory::<u8>::zeroes(LENGTH)?;
    let mut directions_device = DeviceMemory::<ContourPixelDirectionInfo>::zeroes(LENGTH)?;
    let mut info_device = DeviceMemory::<CompressedMarkerLabelsInfo>::zeroes(list_len)?;
    let mut contour_counts_device = DeviceMemory::<u32>::zeroes(list_len)?;
    let mut contour_offsets_device = DeviceMemory::<u32>::zeroes(list_len)?;
    let mut contour_counts_host = vec![0_u32; list_len];
    let mut contour_offsets_host = vec![0_u32; list_len];
    let mut totals = ContourTotalsInfo::default();

    let compressed_labels = ImageView::<u32, C1>::from_memory(&labels_device, SIZE)?;
    let mut contours = ImageViewMut::<u8, C1>::from_memory(&mut contours_device, SIZE)?;
    let mut directions =
        ImageViewMut::<ContourPixelDirectionInfo, C1>::from_memory(&mut directions_device, SIZE)?;

    filtering::compressed_marker_labels_uf_info_u32_c1(
        &stream_context,
        &compressed_labels,
        compressed_count,
        &mut info_device,
        &mut contours,
        &mut directions,
        &mut totals,
        &mut contour_counts_device,
        &mut contour_counts_host,
        &mut contour_offsets_device,
        &mut contour_offsets_host,
    )?;
    stream.synchronize()?;

    let labels_host = download(&labels_device)?;
    let contours_host = download(&contours_device)?;
    assert!(count_nonzero(&contours_host) > 0);

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
        count_nonzero(&contours_host),
        output.display()
    );
    Ok(())
}

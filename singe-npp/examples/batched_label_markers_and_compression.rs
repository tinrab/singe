mod common;

use singe_cuda::memory::DeviceMemory;
use singe_npp::{
    image::{
        raw::filtering,
        view::{C1, ImageView, ImageViewMut},
    },
    types::{ImageNormalization, Size},
};

use common::{
    Result, create_stream, download, image_path, normalize_u32_to_u8, output_dir, read_raw_u8,
    upload, write_png,
};

struct InputImage {
    name: &'static str,
    label_output: &'static str,
    compressed_output: &'static str,
    size: Size,
}

fn main() -> Result<()> {
    let images = [
        InputImage {
            name: "lena_512x512_8u.raw",
            label_output: "Lena_LabelMarkersUFBatch_8Way_512x512_32u.png",
            compressed_output: "Lena_CompressedMarkerLabelsUFBatch_8Way_512x512_32u.png",
            size: Size::new(512, 512),
        },
        InputImage {
            name: "CT_skull_512x512_8u.raw",
            label_output: "CT_skull_LabelMarkersUFBatch_8Way_512x512_32u.png",
            compressed_output: "CT_skull_CompressedMarkerLabelsUFBatch_8Way_512x512_32u.png",
            size: Size::new(512, 512),
        },
        InputImage {
            name: "PCB_METAL_509x335_8u.raw",
            label_output: "PCB_METAL_LabelMarkersUFBatch_8Way_509x335_32u.png",
            compressed_output: "PCB_METAL_CompressedMarkerLabelsUFBatch_8Way_509x335_32u.png",
            size: Size::new(509, 335),
        },
        InputImage {
            name: "PCB2_1024x683_8u.raw",
            label_output: "PCB2_LabelMarkersUFBatch_8Way_1024x683_32u.png",
            compressed_output: "PCB2_CompressedMarkerLabelsUFBatch_8Way_1024x683_32u.png",
            size: Size::new(1024, 683),
        },
        InputImage {
            name: "PCB_1280x720_8u.raw",
            label_output: "PCB_LabelMarkersUFBatch_8Way_1280x720_32u.png",
            compressed_output: "PCB_CompressedMarkerLabelsUFBatch_8Way_1280x720_32u.png",
            size: Size::new(1280, 720),
        },
    ];

    let (stream, stream_context) = create_stream()?;
    let output = output_dir("batched_label_markers_and_compression")?;

    let mut source_devices = Vec::with_capacity(images.len());
    let mut label_devices = Vec::with_capacity(images.len());
    for image in &images {
        let input = read_raw_u8(
            image_path("batched_label_markers_and_compression", image.name),
            image.size,
        )?;
        source_devices.push(upload(&input)?);
        label_devices.push(DeviceMemory::<u32>::zeroes(
            image.size.width as usize * image.size.height as usize,
        )?);
    }

    let sources = images
        .iter()
        .zip(source_devices.iter())
        .map(|(image, memory)| {
            let img = ImageView::<u8, C1>::from_memory(memory, image.size)?;
            Ok(img)
        })
        .collect::<Result<Vec<_>>>()?;

    let mut label_views = images
        .iter()
        .zip(label_devices.iter_mut())
        .map(|(image, memory)| {
            let img = ImageViewMut::<u32, C1>::from_memory(memory, image.size)?;
            Ok(img)
        })
        .collect::<Result<Vec<_>>>()?;

    filtering::label_markers_uf_batch_u8_to_u32_c1_advanced(
        &stream_context,
        &sources,
        &mut label_views,
        ImageNormalization::Infinity,
    )?;
    stream.synchronize()?;
    drop(label_views);

    for (image, labels) in images.iter().zip(label_devices.iter()) {
        write_png(
            output.join(image.label_output),
            &normalize_u32_to_u8(&download(labels)?),
            image.size,
        )?;
    }

    let mut label_views = images
        .iter()
        .zip(label_devices.iter_mut())
        .map(|(image, memory)| {
            let img = ImageViewMut::<u32, C1>::from_memory(memory, image.size)?;
            Ok(img)
        })
        .collect::<Result<Vec<_>>>()?;

    let mut compressed_counts_device = DeviceMemory::<u32>::zeroes(images.len())?;
    filtering::compress_marker_labels_uf_batch_u32_c1_in_place_advanced(
        &stream_context,
        &mut label_views,
        &mut compressed_counts_device,
    )?;
    stream.synchronize()?;

    let compressed_counts = download(&compressed_counts_device)?;
    for ((image, labels), count) in images
        .iter()
        .zip(label_devices.iter())
        .zip(compressed_counts.iter())
    {
        assert!(*count > 0);
        write_png(
            output.join(image.compressed_output),
            &normalize_u32_to_u8(&download(labels)?),
            image.size,
        )?;
        println!(
            "batched_label_markers_and_compression: {} compressed label count {}",
            image.name, count
        );
    }

    Ok(())
}

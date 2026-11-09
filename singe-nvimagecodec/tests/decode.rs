#![cfg(feature = "testing")]

use std::{error::Error, io, mem::MaybeUninit, path::PathBuf, slice};

use singe_cuda::{context::Context as CudaContext, device::Device};
use singe_nvimagecodec::{
    code_stream::CodeStream,
    codec::ExecutionParams,
    decode::{DecodeItem, DecodeParams, Decoder},
    image::{Image, ImageInfo, ImagePlaneInfo},
    instance::{Instance, InstanceCreateInfo},
    types::{ChromaSubsampling, ColorSpec, ProcessingStatus, SampleDataType, SampleFormat},
};

#[test]
fn test_decode_sample_jpeg_to_host_rgb() -> Result<(), Box<dyn Error>> {
    let device_count = match Device::count() {
        Ok(count) => count,
        Err(error) => {
            eprintln!("skipping decode_sample_jpeg_to_host_rgb: no cuda device count: {error}");
            return Ok(());
        }
    };
    if device_count == 0 {
        eprintln!("skipping decode_sample_jpeg_to_host_rgb: no cuda devices");
        return Ok(());
    }

    let input_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("images")
        .join("cat-1.jpg");

    let mut create_info = InstanceCreateInfo::new();
    create_info
        .set_load_builtin_modules(true)
        .set_load_extension_modules(true);
    let instance = Instance::create(&create_info)?;

    let code_stream = CodeStream::from_file(&instance, &input_path, None)?;
    let input_info = code_stream.image_info()?;
    let input_plane = input_info
        .plane(0)
        .ok_or_else(|| invalid_data("image has no planes"))?;

    let width = input_plane.width();
    let height = input_plane.height();
    let sample_type = input_plane.sample_type();
    let bytes_per_sample = bytes_per_sample(sample_type)?;
    let row_stride = checked_mul(width as usize, bytes_per_sample, "row stride")?;
    let plane_size = checked_mul(row_stride, height as usize, "plane size")?;
    let output_size = checked_mul(plane_size, 3, "output size")?;
    let mut output = vec![MaybeUninit::<u8>::uninit(); output_size];

    let mut planes = [
        ImagePlaneInfo::create(width, height, row_stride, 1, sample_type)?,
        ImagePlaneInfo::create(width, height, row_stride, 1, sample_type)?,
        ImagePlaneInfo::create(width, height, row_stride, 1, sample_type)?,
    ];
    for plane in &mut planes {
        plane.set_precision(input_plane.precision());
    }

    {
        let mut output_info =
            ImageInfo::from_host_output(&mut output, SampleFormat::PlanarRgb, &planes)?;
        output_info
            .set_color_spec(ColorSpec::Srgb)
            .set_chroma_subsampling(ChromaSubsampling::None);
        let mut image = Image::create(&instance, output_info)?;

        let context = CudaContext::retain_primary_for_device(Device::new(0))?;
        let execution_params = ExecutionParams::new(&context);
        let mut decoder = Decoder::create(&instance, &execution_params, None)?;

        let mut decode_params = DecodeParams::new();
        decode_params.set_apply_exif_orientation(true);
        let item = DecodeItem::new(&code_stream, &mut image);

        assert_eq!(
            decoder.can_decode_batch(slice::from_ref(&item), &decode_params, false)?,
            [ProcessingStatus::SUCCESS]
        );
        decoder.decode_one(item.stream, item.output, &decode_params)?;
        let mut has_nonzero_samples = false;
        for plane in 0..3 {
            for row in 0..height as usize {
                has_nonzero_samples |= image.host_row(plane, row)?.iter().any(|byte| *byte != 0);
            }
        }
        assert!(has_nonzero_samples);
    }
    Ok(())
}

fn bytes_per_sample(sample_type: SampleDataType) -> Result<usize, io::Error> {
    sample_type
        .byte_size()
        .ok_or_else(|| invalid_data(format!("unsupported sample data type: {sample_type}")))
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, io::Error> {
    left.checked_mul(right)
        .ok_or_else(|| invalid_data(format!("{name} overflows usize")))
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

use std::{error::Error, io, mem::MaybeUninit, path::PathBuf, slice};

use anyhow::Context as _;
use singe_cuda::{context::Context as CudaContext, device::Device};
use singe_nvimagecodec::{
    code_stream::CodeStream,
    codec::ExecutionParams,
    decode::{DecodeItem, DecodeParams, Decoder},
    image::{Image, ImageInfo, ImagePlaneInfo},
    instance::{Instance, InstanceCreateInfo},
    types::{ChromaSubsampling, ColorSpec, ProcessingStatus, SampleDataType, SampleFormat},
};

fn main() -> Result<(), Box<dyn Error>> {
    let input_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("images")
        .join("cat-1.jpg");

    let mut create_info = InstanceCreateInfo::new();
    create_info
        .set_load_builtin_modules(true)
        .set_load_extension_modules(true)
        .set_create_debug_messenger(true);
    let instance = Instance::create(&create_info).context("create nvimagecodec instance")?;

    let code_stream =
        CodeStream::from_file(&instance, &input_path, None).context("open input code stream")?;
    let input_info = code_stream.image_info().context("read input image info")?;
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

    let checksum = {
        let mut output_info =
            ImageInfo::from_host_output(&mut output, SampleFormat::PlanarRgb, &planes)?;
        output_info
            .set_color_spec(ColorSpec::Srgb)
            .set_chroma_subsampling(ChromaSubsampling::None);
        let mut image = Image::create(&instance, output_info).context("create output image")?;

        let context = CudaContext::retain_primary_for_device(Device::new(0))?;
        let execution_params = ExecutionParams::new(&context);
        let mut decoder =
            Decoder::create(&instance, &execution_params, None).context("create decoder")?;

        let mut decode_params = DecodeParams::new();
        decode_params.set_apply_exif_orientation(true);

        let item = DecodeItem::new(&code_stream, &mut image);
        let can_decode = decoder
            .can_decode_batch(slice::from_ref(&item), &decode_params, false)
            .context("query decoder capability")?;
        if can_decode != [ProcessingStatus::SUCCESS] {
            return Err(
                invalid_data(format!("decoder cannot decode image: {can_decode:?}")).into(),
            );
        }

        let status = decoder
            .decode_one(item.stream, item.output, &decode_params)
            .context("submit decode")?;

        if status != ProcessingStatus::SUCCESS {
            return Err(invalid_data(format!("decode failed: {status:?}")).into());
        }

        let mut checksum = 0_u64;
        for plane in 0..3 {
            for row in 0..height as usize {
                for byte in image.host_row(plane, row)? {
                    checksum = checksum.wrapping_add(u64::from(*byte));
                }
            }
        }
        checksum
    };
    println!(
        "decoded {} as {}x{} {:?} planar RGB into {} host bytes (checksum {})",
        input_info.codec_name(),
        width,
        height,
        sample_type,
        output.len(),
        checksum
    );

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

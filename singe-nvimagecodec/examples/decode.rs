use std::{mem::MaybeUninit, path::PathBuf, slice};

use anyhow::{Context as _, anyhow, bail};
use singe_cuda::{context::Context as CudaContext, device::Device};
use singe_nvimagecodec::{
    code_stream::CodeStream,
    codec::ExecutionParams,
    decode::{DecodeItem, DecodeParams, Decoder},
    image::{Image, ImageInfo, ImagePlaneInfo},
    instance::{Instance, InstanceOptions},
    types::{ChromaSubsampling, CodecFallback, ColorSpec, ProcessingStatus, SampleFormat},
};

fn main() -> anyhow::Result<()> {
    let input_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("images")
        .join("cat-1.jpg");

    let instance = Instance::create(&InstanceOptions {
        create_debug_messenger: true,
        ..InstanceOptions::default()
    })
    .context("create nvimagecodec instance")?;

    let code_stream =
        CodeStream::from_file(&instance, &input_path, None).context("open input code stream")?;
    let input_info = code_stream.image_info().context("read input image info")?;
    let input_plane = input_info
        .plane(0)
        .ok_or_else(|| anyhow!("image has no planes"))?;

    let width = input_plane.width();
    let height = input_plane.height();
    let sample_type = input_plane.sample_type()?;
    let bytes_per_sample = sample_type
        .byte_size()
        .ok_or_else(|| anyhow!("unsupported sample data type: {sample_type}"))?;
    let rows = usize::try_from(height)?;
    let row_stride = usize::try_from(width)?
        .checked_mul(bytes_per_sample)
        .context("row stride overflows usize")?;
    let output_size = row_stride
        .checked_mul(rows)
        .and_then(|plane_size| plane_size.checked_mul(3))
        .context("output size overflows usize")?;
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

        let context = CudaContext::create_for_device(Device::new(0))?;
        let execution_params = ExecutionParams::new(&context);
        let decoder =
            Decoder::create(&instance, &execution_params, None).context("create decoder")?;

        let mut decode_params = DecodeParams::new();
        decode_params.set_apply_exif_orientation(true);

        let item = DecodeItem::new(&code_stream, &mut image);
        let can_decode = decoder
            .can_decode(
                slice::from_ref(&item),
                &decode_params,
                CodecFallback::StopAtPreferred,
            )
            .context("query decoder capability")?;
        if can_decode != [ProcessingStatus::SUCCESS] {
            bail!("decoder cannot decode image: {can_decode:?}");
        }

        decoder
            .decode_one(item.stream, item.output, &decode_params)
            .context("decode image")?;

        let mut checksum = 0_u64;
        for plane in 0..3 {
            for row in 0..rows {
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

#![cfg(feature = "testing")]

use std::{mem::MaybeUninit, path::PathBuf, slice};

use anyhow::{Context as _, Result, anyhow};
use singe_nvimagecodec::{
    code_stream::CodeStream,
    codec::ExecutionParams,
    decode::{DecodeItem, DecodeParams, Decoder},
    image::{Image, ImageInfo, ImagePlaneInfo},
    testing::setup_context,
    types::{ChromaSubsampling, CodecFallback, ColorSpec, ProcessingStatus, SampleFormat},
};

#[test]
fn test_decode_sample_jpeg_to_host_rgb() -> Result<()> {
    let test = setup_context()?;
    let instance = test.instance();
    let input_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("images")
        .join("cat-1.jpg");

    let code_stream = CodeStream::from_file(instance, &input_path, None)?;
    let input_info = code_stream.image_info()?;
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

    {
        let mut output_info =
            ImageInfo::from_host_output(&mut output, SampleFormat::PlanarRgb, &planes)?;
        output_info
            .set_color_spec(ColorSpec::Srgb)
            .set_chroma_subsampling(ChromaSubsampling::None);
        let mut image = Image::create(instance, output_info)?;

        let execution_params = ExecutionParams::new(test.cuda_context());
        let decoder = Decoder::create(instance, &execution_params, None)?;

        let mut decode_params = DecodeParams::new();
        decode_params.set_apply_exif_orientation(true);
        let item = DecodeItem::new(&code_stream, &mut image);

        assert_eq!(
            decoder.can_decode(
                slice::from_ref(&item),
                &decode_params,
                CodecFallback::StopAtPreferred
            )?,
            [ProcessingStatus::SUCCESS]
        );
        decoder.decode_one(item.stream, item.output, &decode_params)?;
        let mut has_nonzero_samples = false;
        for plane in 0..3 {
            for row in 0..rows {
                has_nonzero_samples |= image.host_row(plane, row)?.iter().any(|byte| *byte != 0);
            }
        }
        assert!(has_nonzero_samples);
    }
    Ok(())
}

mod common;

use std::{
    env,
    path::{Path, PathBuf},
    process,
    time::Instant,
};

use common::{ExampleResult, invalid_input};
use singe_cuda::{context::Context, memory::DeviceMemory, stream::Stream};
use singe_nvtiff::{
    decode::{DecodeOptions, DecodeRegion, Decoder},
    encode::{EncodeParams, Encoder},
    error::Result,
    image::ImageInfo,
    stream::{TiffImage, TiffStream},
};

use crate::common::{next_path, next_value};

#[derive(Debug)]
struct Options {
    file: PathBuf,
    frame_begin: Option<u32>,
    frame_end: Option<u32>,
    decoded_output_dir: PathBuf,
    encoded_output: Option<PathBuf>,
    verbose: bool,
}

fn main() -> ExampleResult<()> {
    let options = Options::parse()?;

    let context = Context::create()?;
    let stream = context.create_stream()?;

    let parse_start = Instant::now();
    let tiff = TiffStream::open_file(&options.file)?;
    println!(
        "parsed {} in {:?}",
        options.file.display(),
        parse_start.elapsed()
    );
    if options.verbose {
        tiff.print()?;
    }

    let all_images = tiff.images()?.collect::<Result<Vec<_>>>()?;
    let image_count = all_images.len();
    if image_count == 0 {
        return Err(invalid_input("TIFF stream contains no images"));
    }

    let (frame_begin, frame_end) = frame_range(&options, image_count)?;
    let images = all_images
        .iter()
        .copied()
        .skip(frame_begin as usize)
        .take((frame_end - frame_begin + 1) as usize)
        .collect::<Vec<_>>();
    let ifd_offsets = images.iter().map(TiffImage::ifd_offset).collect::<Vec<_>>();
    let image_infos = images
        .iter()
        .map(|image| Ok(image.info()?))
        .collect::<ExampleResult<Vec<_>>>()?;
    let layouts = image_infos
        .iter()
        .map(|info| Ok(info.default_decode_layout()?))
        .collect::<ExampleResult<Vec<_>>>()?;
    let mut decoder = Decoder::create(&stream)?;
    println!(
        "decoding {} image(s), range [{frame_begin}, {frame_end}]",
        ifd_offsets.len()
    );
    let decode_start = Instant::now();
    let mut outputs = Vec::with_capacity(ifd_offsets.len());
    for (ifd_offset, layout) in ifd_offsets.iter().copied().zip(&layouts) {
        decoder.set_options(
            DecodeOptions::new()
                .with_output_format(layout.format)
                .with_region(DecodeRegion::full_image(ifd_offset)),
        )?;
        outputs.push(decoder.decode_to_device(&tiff)?);
    }

    println!("decode completed in {:?}", decode_start.elapsed());

    for index in 0..ifd_offsets.len() {
        let host = outputs[index].copy_to_host_vec()?;
        let path = common::write_decoded_pnm(
            &options.file,
            &options.decoded_output_dir,
            ifd_offsets[index],
            &host,
            layouts[index],
        )?;
        println!("wrote {}", path.display());
    }

    if let Some(encoded_output) = options.encoded_output.as_deref() {
        encode_decoded_images(
            encoded_output,
            &image_infos,
            &outputs,
            &stream,
            ifd_offsets.len(),
        )?;
    }

    Ok(())
}

impl Options {
    fn parse() -> ExampleResult<Self> {
        let mut file = None;
        let mut frame_begin = None;
        let mut frame_end = None;
        let mut decoded_output_dir = None;
        let mut encoded_output = None;
        let mut verbose = false;
        let mut args = env::args().skip(1);

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "-h" | "--help" => {
                    print_usage(&env::args().next().unwrap_or_else(|| "decode_encode".into()));
                    process::exit(0);
                }
                "-f" | "--file" => file = Some(next_path(&mut args, &arg)?),
                "-b" | "--frame-beg" => frame_begin = Some(next_value(&mut args, &arg)?.parse()?),
                "-e" | "--frame-end" => frame_end = Some(next_value(&mut args, &arg)?.parse()?),
                "--decode-out" => decoded_output_dir = Some(next_path(&mut args, &arg)?),
                "--encode-out" => encoded_output = Some(next_path(&mut args, &arg)?),
                "-v" | "--verbose" => verbose = true,
                _ if !arg.starts_with('-') && file.is_none() => file = Some(PathBuf::from(arg)),
                _ => return Err(invalid_input(format!("unknown argument `{arg}`"))),
            }
        }

        Ok(Self {
            file: file.unwrap_or_else(common::sample_tiff_path),
            frame_begin,
            frame_end,
            decoded_output_dir: decoded_output_dir.unwrap_or_else(common::output_dir_path),
            encoded_output,
            verbose,
        })
    }
}

fn encode_decoded_images(
    encoded_output: &Path,
    image_infos: &[ImageInfo],
    outputs: &[DeviceMemory<u8>],
    stream: &Stream,
    image_count: usize,
) -> ExampleResult<()> {
    if !common::compatible_for_single_encode_params(image_infos) {
        return Err(invalid_input(
            "encoding this range requires identical image metadata",
        ));
    }

    let mut encoder = Encoder::create(stream)?;
    let mut params = EncodeParams::create(stream)?;
    let image_info = image_infos[0].to_encode_info()?;
    params.set_image_info(&image_info)?;

    println!(
        "encoding {} decoded image(s) to {}",
        image_count,
        encoded_output.display()
    );
    let encode_start = Instant::now();
    let tiff_file = encoder.encode(&mut params, outputs.iter().take(image_count))?;
    let size = tiff_file.size();
    tiff_file.write_file(encoded_output)?;
    println!(
        "encode completed in {:?}; metadata={} bytes, bitstream={} bytes",
        encode_start.elapsed(),
        size.metadata,
        size.bitstream,
    );

    Ok(())
}

fn frame_range(options: &Options, image_count: usize) -> ExampleResult<(u32, u32)> {
    let image_count = u32::try_from(image_count)?;
    let begin = options.frame_begin.unwrap_or(0).min(image_count - 1);
    let end = options
        .frame_end
        .unwrap_or(image_count - 1)
        .min(image_count - 1);
    if begin > end {
        return Err(invalid_input(
            "frame begin must be less than or equal to frame end",
        ));
    }
    Ok((begin, end))
}

fn print_usage(program: &str) {
    println!(
        "Usage: {program} [--file TIFF_FILE] [--frame-beg N] [--frame-end N] [--decode-out DIR] [--encode-out TIFF_FILE] [--verbose]\n\
         Defaults to {} and decoded output directory {}. Passing --encode-out enables the encode/write part.",
        common::sample_tiff_path().display(),
        common::output_dir_path().display(),
    );
}

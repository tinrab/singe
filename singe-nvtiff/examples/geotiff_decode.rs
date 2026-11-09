mod common;

use std::{env, path::PathBuf, process, time::Instant};

use common::{ExampleResult, invalid_input};
use singe_cuda::{context::Context, stream::Stream};
use singe_nvtiff::{
    decode::{DecodeOptions, DecodeRegion, Decoder},
    error::Result,
    image::IfdOffset,
    stream::{TiffImage, TiffStream},
    types::{GeoKeyValue, Tag, TagValue},
};

use crate::common::{next_path, next_value};

#[derive(Debug)]
struct Options {
    file: PathBuf,
    frame_begin: Option<u32>,
    frame_end: Option<u32>,
    decoded_output_dir: PathBuf,
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

    print_geotiff_metadata(&tiff)?;
    decode_images(&options, &tiff, &stream)?;

    Ok(())
}

impl Options {
    fn parse() -> ExampleResult<Self> {
        let mut file = None;
        let mut frame_begin = None;
        let mut frame_end = None;
        let mut decoded_output_dir = None;
        let mut verbose = false;
        let mut args = env::args().skip(1);

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "-h" | "--help" => {
                    print_usage(
                        &env::args()
                            .next()
                            .unwrap_or_else(|| "geotiff_decode".into()),
                    );
                    process::exit(0);
                }
                "-f" | "--file" => file = Some(next_path(&mut args, &arg)?),
                "-b" | "--frame-beg" => frame_begin = Some(next_value(&mut args, &arg)?.parse()?),
                "-e" | "--frame-end" => frame_end = Some(next_value(&mut args, &arg)?.parse()?),
                "--decode-out" => decoded_output_dir = Some(next_path(&mut args, &arg)?),
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
            verbose,
        })
    }
}

fn print_geotiff_metadata(tiff: &TiffStream<'_>) -> ExampleResult<()> {
    for tag in [
        Tag::ModelPixelScale,
        Tag::ModelTiePoint,
        Tag::ModelTransformation,
    ] {
        match tiff.tag_value(IfdOffset::new(0), tag) {
            Ok(TagValue::Double(values)) => {
                println!("{tag}: {values:?}");
            }
            Ok(value) => println!("{tag}: skipping unsupported tag value {value:?}"),
            Err(error) if error.is_missing_metadata() => {}
            Err(error) => return Err(error.into()),
        }
    }

    match tiff.geo_keys() {
        Ok(keys) if keys.is_empty() => println!("no GeoTIFF keys found"),
        Ok(keys) => {
            println!("GeoTIFF keys:");
            for key in keys {
                match tiff.geo_key_value(key)? {
                    GeoKeyValue::Short(values) => println!("  {key}: {values:?}"),
                    GeoKeyValue::Double(values) => println!("  {key}: {values:?}"),
                    GeoKeyValue::Ascii(value) => println!("  {key}: {value}"),
                    GeoKeyValue::Unknown(bytes) => println!("  {key}: unknown value {bytes:?}"),
                    _ => println!("  {key}: unsupported GeoTIFF value"),
                }
            }
        }
        Err(error) if error.is_missing_metadata() => println!("no GeoTIFF keys found"),
        Err(error) => return Err(error.into()),
    }

    Ok(())
}

fn decode_images(options: &Options, tiff: &TiffStream<'_>, stream: &Stream) -> ExampleResult<()> {
    let images = tiff.images()?.collect::<Result<Vec<_>>>()?;
    let image_count = u32::try_from(images.len())?;
    if image_count == 0 {
        return Err(invalid_input("TIFF stream contains no images"));
    }

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

    let ifd_offsets = images[begin as usize..=end as usize]
        .iter()
        .map(TiffImage::ifd_offset)
        .collect::<Vec<_>>();
    let image_infos = ifd_offsets
        .iter()
        .map(|ifd_offset| Ok(tiff.image_info(*ifd_offset)?))
        .collect::<ExampleResult<Vec<_>>>()?;
    let layouts = image_infos
        .iter()
        .map(|info| Ok(info.default_decode_layout()?))
        .collect::<ExampleResult<Vec<_>>>()?;

    let mut decoder = Decoder::create(stream)?;
    println!(
        "decoding {} image(s), range [{begin}, {end}]",
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
        outputs.push(decoder.decode_to_device(tiff)?);
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

    Ok(())
}

fn print_usage(program: &str) {
    println!(
        "Usage: {program} [--file TIFF_FILE] [--frame-beg N] [--frame-end N] [--decode-out DIR] [--verbose]\n\
         Defaults to {} and decoded output directory {}.",
        common::sample_tiff_path().display(),
        common::output_dir_path().display(),
    );
}

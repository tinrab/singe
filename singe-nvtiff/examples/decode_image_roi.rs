mod common;

use std::{env, path::PathBuf, process, time::Instant};

use common::{ExampleResult, invalid_input};
use singe_cuda::context::Context;
use singe_nvtiff::{
    decode::{DecodeOptions, DecodeRegion, Decoder, Rectangle},
    image::{IfdOffset, ImageInfo},
    stream::TiffStream,
};

use crate::common::{next_path, next_value};

#[derive(Debug)]
struct Options {
    file: PathBuf,
    ifd_offset: IfdOffset,
    roi: Option<(i32, i32, i32, i32)>,
    output_dir: PathBuf,
}

fn main() -> ExampleResult<()> {
    let options = Options::parse()?;
    let tiff = TiffStream::open_file(&options.file)?;
    let image_info = tiff.image_info(options.ifd_offset)?;

    let context = Context::create()?;
    let stream = context.create_stream()?;

    let roi = normalized_roi(
        &image_info,
        options.roi.unwrap_or_else(|| default_roi(&image_info)),
    )?;
    let default_layout = image_info.default_decode_layout()?;
    let layout =
        image_info.decode_layout_for_extent(default_layout.format, roi.width, roi.height)?;
    let decode_options = DecodeOptions::new()
        .with_output_format(layout.format)
        .with_region(DecodeRegion::region(options.ifd_offset, roi));

    println!(
        "decoding ROI {},{},{},{} from image {} in {}",
        roi.x,
        roi.y,
        roi.width,
        roi.height,
        options.ifd_offset,
        options.file.display()
    );
    let mut decoder = Decoder::create(&stream)?;
    decoder.set_options(decode_options)?;
    let start = Instant::now();
    let output = decoder.decode_to_device(&tiff)?;
    println!("decode completed in {:?}", start.elapsed());

    let host = output.copy_to_host_vec()?;
    let path = common::write_decoded_pnm(
        &options.file,
        &options.output_dir,
        options.ifd_offset,
        &host,
        layout,
    )?;
    println!("wrote {}", path.display());

    Ok(())
}

impl Options {
    fn parse() -> ExampleResult<Self> {
        let mut file = None;
        let mut ifd_offset = IfdOffset::new(0);
        let mut roi = None;
        let mut output_dir = None;
        let mut args = env::args().skip(1);

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "-h" | "--help" => {
                    print_usage(
                        &env::args()
                            .next()
                            .unwrap_or_else(|| "decode_image_roi".into()),
                    );
                    process::exit(0);
                }
                "-f" | "--file" => file = Some(next_path(&mut args, &arg)?),
                "--ifd-offset" => {
                    ifd_offset = IfdOffset::new(next_value(&mut args, &arg)?.parse()?);
                }
                "--roi" => roi = Some(common::parse_roi(&next_value(&mut args, &arg)?)?),
                "-o" | "--output-dir" => output_dir = Some(next_path(&mut args, &arg)?),
                _ if !arg.starts_with('-') && file.is_none() => file = Some(PathBuf::from(arg)),
                _ => return Err(invalid_input(format!("unknown argument `{arg}`"))),
            }
        }

        Ok(Self {
            file: file.unwrap_or_else(common::sample_tiff_path),
            ifd_offset,
            roi,
            output_dir: output_dir.unwrap_or_else(common::output_dir_path),
        })
    }
}

fn normalized_roi(image_info: &ImageInfo, roi: (i32, i32, i32, i32)) -> ExampleResult<Rectangle> {
    let (x, y, width, height) = roi;
    if x < 0 || y < 0 {
        return Err(invalid_input("ROI offsets must be non-negative"));
    }

    let image_width = i32::try_from(image_info.width())?;
    let image_height = i32::try_from(image_info.height())?;
    if x >= image_width || y >= image_height {
        return Err(invalid_input("ROI origin is outside the image"));
    }

    let width = if width < 0 { image_width - x } else { width };
    let height = if height < 0 { image_height - y } else { height };
    if width <= 0 || height <= 0 {
        return Err(invalid_input("ROI width and height must be positive"));
    }

    Ok(Rectangle::create(
        x,
        y,
        width.min(image_width - x),
        height.min(image_height - y),
    )?)
}

fn default_roi(image_info: &ImageInfo) -> (i32, i32, i32, i32) {
    let width = i32::try_from(image_info.width()).unwrap_or(i32::MAX);
    let height = i32::try_from(image_info.height()).unwrap_or(i32::MAX);
    let roi_width = (width / 2).max(1);
    let roi_height = (height / 2).max(1);
    (
        (width - roi_width) / 2,
        (height - roi_height) / 2,
        roi_width,
        roi_height,
    )
}

fn print_usage(program: &str) {
    println!(
        "Usage: {program} [--file TIFF_FILE] [--ifd-offset N] [--roi x,y,width,height] [--output-dir DIR]\n\
         Defaults to {}, a centered half-image ROI, and output directory {}.",
        common::sample_tiff_path().display(),
        common::output_dir_path().display()
    );
}

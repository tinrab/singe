#![allow(dead_code)]

use std::{
    fs::{self, File},
    io::{self, Write},
    path::{Path, PathBuf},
};

use anyhow::{Error, Result, anyhow};
use singe_nvtiff::image::{DecodeOutputLayout, IfdOffset, ImageInfo};

pub type ExampleResult<T> = Result<T>;

pub fn sample_tiff_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/images/bali_notiles.tif")
}

pub fn output_dir_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/output")
}

pub fn write_decoded_pnm(
    input_path: &Path,
    output_dir: &Path,
    ifd_offset: IfdOffset,
    data: &[u8],
    layout: DecodeOutputLayout,
) -> ExampleResult<PathBuf> {
    let samples_to_write = samples_to_write(layout)?;
    if layout.bits_per_sample != 8 && layout.bits_per_sample != 16 {
        return Err(invalid_input(format!(
            "PNM output only supports 8-bit and 16-bit samples, got {}",
            layout.bits_per_sample
        )));
    }

    fs::create_dir_all(output_dir)?;
    let stem = input_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("image");
    let extension = if samples_to_write == 1 { "pgm" } else { "ppm" };
    let path = output_dir.join(format!("{stem}_nvtiff_out_{ifd_offset}.{extension}"));
    write_pnm(&path, data, layout)?;
    Ok(path)
}

pub fn compatible_for_single_encode_params(infos: &[ImageInfo]) -> bool {
    let Some(first) = infos.first() else {
        return false;
    };

    infos.iter().all(|info| {
        info.width() == first.width()
            && info.height() == first.height()
            && info.bits_per_pixel() == first.bits_per_pixel()
            && info.samples_per_pixel() == first.samples_per_pixel()
            && info.photometric_interpretation() == first.photometric_interpretation()
            && info.planar_config() == first.planar_config()
            && (0..first.samples_per_pixel() as usize).all(|index| {
                info.bits_per_sample(index) == first.bits_per_sample(index)
                    && info.sample_format(index) == first.sample_format(index)
            })
    })
}

pub fn parse_roi(value: &str) -> ExampleResult<(i32, i32, i32, i32)> {
    let parts = value
        .split(',')
        .map(str::trim)
        .map(str::parse::<i32>)
        .collect::<Result<Vec<_>, _>>()?;
    if parts.len() != 4 {
        return Err(invalid_input("ROI must be `x,y,width,height`"));
    }
    Ok((parts[0], parts[1], parts[2], parts[3]))
}

pub fn invalid_input(message: impl Into<String>) -> Error {
    anyhow!(message.into())
}

fn write_pnm(path: &Path, data: &[u8], layout: DecodeOutputLayout) -> ExampleResult<()> {
    let samples_to_write = samples_to_write(layout)?;
    let bytes_per_sample = usize::from(layout.bits_per_sample / 8);
    let expected = checked_pixels(layout.width, layout.height)?
        .checked_mul(layout.samples_per_pixel as usize)
        .and_then(|size| size.checked_mul(bytes_per_sample))
        .ok_or_else(|| invalid_input("decoded image size overflow"))?;
    if data.len() < expected {
        return Err(invalid_input(format!(
            "decoded buffer is too small for PNM output (expected at least {expected}, got {})",
            data.len()
        )));
    }

    let mut file = File::create(path)?;
    let magic = if samples_to_write == 1 { "P5" } else { "P6" };
    let max_value = (1_u32 << u32::from(layout.bits_per_sample)) - 1;
    let width = layout.width;
    let height = layout.height;
    write!(file, "{magic}\n# nvTIFF\n{width} {height}\n{max_value}\n")?;

    for y in 0..height as usize {
        for x in 0..width as usize {
            let pixel = (y * width as usize + x) * layout.samples_per_pixel as usize;
            for channel in 0..samples_to_write as usize {
                let sample = (pixel + channel) * bytes_per_sample;
                if layout.bits_per_sample == 8 {
                    file.write_all(&data[sample..=sample])?;
                } else {
                    let value = u16::from_ne_bytes([data[sample], data[sample + 1]]);
                    file.write_all(&value.to_be_bytes())?;
                }
            }
        }
    }

    Ok(())
}

fn samples_to_write(layout: DecodeOutputLayout) -> ExampleResult<u16> {
    match layout.samples_per_pixel {
        1 => Ok(1),
        3 | 4 => Ok(3),
        samples => Err(invalid_input(format!(
            "PNM output only supports 1, 3, or 4 channels, got {samples}"
        ))),
    }
}

fn checked_pixels(width: u32, height: u32) -> ExampleResult<usize> {
    usize::try_from(width)?
        .checked_mul(usize::try_from(height)?)
        .ok_or_else(|| invalid_input("image dimensions overflow"))
}

pub fn next_path(args: &mut impl Iterator<Item = String>, name: &str) -> ExampleResult<PathBuf> {
    Ok(PathBuf::from(next_value(args, name)?))
}

pub fn next_value(args: &mut impl Iterator<Item = String>, name: &str) -> ExampleResult<String> {
    args.next()
        .ok_or_else(|| invalid_input(format!("missing value for `{name}`")))
}

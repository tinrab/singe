use std::{
    env,
    error::Error as StdError,
    fs,
    io::{Error, ErrorKind, Write},
    path::{Path, PathBuf},
};

use singe_cuda::context::Context as CudaContext;
use singe_nvjpeg::{image::Size, k2::prelude::*};

type ExampleResult<T> = Result<T, Box<dyn StdError>>;

fn main() -> ExampleResult<()> {
    let input = input_path()?;

    let context = CudaContext::create()?;
    let stream = context.create_stream()?;
    let options = DecodeOptions::new().with_output_format(ImageFormat::Planar);
    let mut decoder = Decoder::builder()
        .backend(Backend::Default)
        .options(options)
        .create(&context)?;
    let image = decoder.decode_file_to_device(&input, &stream)?;

    match image.components() {
        [gray] => decode_gray(gray)?,
        [red, green, blue] => decode_rgb(red, green, blue)?,
        _ => {
            return Err(invalid_input("example supports only 1- or 3-component JPEG 2000").into());
        }
    }

    Ok(())
}

fn decode_gray(component: &DecodedComponent) -> ExampleResult<()> {
    let info = component.spec().info();
    ensure_u8_unsigned(&info)?;
    let size = component_size(&info);

    let output = output_path("jpeg2k_decode.pgm");
    fs::create_dir_all(
        output
            .parent()
            .ok_or_else(|| invalid_input("missing output parent"))?,
    )?;
    let gray = component.data().copy_to_host_vec()?;
    write_pgm(&output, size.width(), size.height(), &gray)?;
    println!("wrote {}", output.display());
    Ok(())
}

fn decode_rgb(
    red: &DecodedComponent,
    green: &DecodedComponent,
    blue: &DecodedComponent,
) -> ExampleResult<()> {
    let red_info = red.spec().info();
    let green_info = green.spec().info();
    let blue_info = blue.spec().info();
    ensure_u8_unsigned(&red_info)?;
    ensure_u8_unsigned(&green_info)?;
    ensure_u8_unsigned(&blue_info)?;
    let size = component_size(&red_info);
    if component_size(&green_info) != size || component_size(&blue_info) != size {
        return Err(invalid_input("example requires equal RGB component dimensions").into());
    }

    let red = red.data().copy_to_host_vec()?;
    let green = green.data().copy_to_host_vec()?;
    let blue = blue.data().copy_to_host_vec()?;
    let rgb = interleave_rgb(&red, &green, &blue)?;

    let output = output_path("jpeg2k_decode.ppm");
    fs::create_dir_all(
        output
            .parent()
            .ok_or_else(|| invalid_input("missing output parent"))?,
    )?;
    write_ppm(&output, size.width(), size.height(), &rgb)?;
    println!("wrote {}", output.display());
    Ok(())
}

fn input_path() -> ExampleResult<PathBuf> {
    env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| {
            invalid_input("usage: cargo run -p singe-nvjpeg --features k2 --example jpeg2k_decode -- <input.jp2>")
        })
        .map_err(Into::into)
}

fn output_path(filename: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("output")
        .join(filename)
}

fn ensure_u8_unsigned(component: &ImageComponentInfo) -> ExampleResult<()> {
    if component.precision() <= 8 && !component.signed() {
        Ok(())
    } else {
        Err(invalid_input("example supports only unsigned 8-bit components").into())
    }
}

fn component_size(component: &ImageComponentInfo) -> Size {
    let size = component.size();
    Size::new(size.width(), size.height())
}

fn interleave_rgb(red: &[u8], green: &[u8], blue: &[u8]) -> ExampleResult<Vec<u8>> {
    if red.len() != green.len() || red.len() != blue.len() {
        return Err(invalid_input("RGB component lengths differ").into());
    }

    let mut rgb = vec![
        0;
        red.len()
            .checked_mul(3)
            .ok_or_else(|| invalid_input("RGB output size overflow"))?
    ];
    for index in 0..red.len() {
        let offset = index * 3;
        rgb[offset] = red[index];
        rgb[offset + 1] = green[index];
        rgb[offset + 2] = blue[index];
    }
    Ok(rgb)
}

fn write_pgm(path: &Path, width: u32, height: u32, gray: &[u8]) -> ExampleResult<()> {
    let mut file = fs::File::create(path)?;
    write!(file, "P5\n{width} {height}\n255\n")?;
    file.write_all(gray)?;
    Ok(())
}

fn write_ppm(path: &Path, width: u32, height: u32, rgb: &[u8]) -> ExampleResult<()> {
    let mut file = fs::File::create(path)?;
    write!(file, "P6\n{width} {height}\n255\n")?;
    file.write_all(rgb)?;
    Ok(())
}

fn invalid_input(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidInput, message.into())
}

use std::{
    error::Error as StdError,
    fs,
    io::{Error, ErrorKind},
    path::PathBuf,
};

use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory};
use singe_nvjpeg::{image::Plane, k2::prelude::*};

type ExampleResult<T> = Result<T, Box<dyn StdError>>;

const WIDTH: u32 = 256;
const HEIGHT: u32 = 160;

fn main() -> ExampleResult<()> {
    let output = output_path();
    fs::create_dir_all(
        output
            .parent()
            .ok_or_else(|| invalid_input("missing output parent"))?,
    )?;

    let (red, green, blue) = rgb_gradient(WIDTH, HEIGHT)?;

    let context = CudaContext::create()?;
    let stream = context.create_stream()?;
    let red = DeviceMemory::<u8>::from_slice(stream.context(), &red)?;
    let green = DeviceMemory::<u8>::from_slice(stream.context(), &green)?;
    let blue = DeviceMemory::<u8>::from_slice(stream.context(), &blue)?;

    let plane_size = Size::new(WIDTH, HEIGHT);
    let pitch = WIDTH as usize;
    let image = EncodeInput::planar([
        Plane::create(red.view(), plane_size, pitch)?,
        Plane::create(green.view(), plane_size, pitch)?,
        Plane::create(blue.view(), plane_size, pitch)?,
    ])?;

    let config = EncodeConfig::builder(plane_size, BitstreamType::Jp2, ColorSpace::Srgb)?
        .rgb_u8()?
        .lossless()
        .build()?;

    let options = EncodeOptions::new().with_input_format(ImageFormat::Planar);
    let mut encoder = Encoder::builder()
        .options(options)
        .encode_config(config)
        .create(&context)?;

    encoder.encode_to_file(&image, &output, &stream)?;
    println!("wrote {}", output.display());
    Ok(())
}

fn output_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("output")
        .join("jpeg2k_encode.jp2")
}

fn rgb_gradient(width: u32, height: u32) -> ExampleResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let len = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| invalid_input("input image size overflow"))?;
    let mut red = vec![0; len];
    let mut green = vec![0; len];
    let mut blue = vec![0; len];

    for y in 0..height {
        for x in 0..width {
            let offset = y as usize * width as usize + x as usize;
            red[offset] = x.to_le_bytes()[0];
            green[offset] = y.to_le_bytes()[0];
            blue[offset] = x.wrapping_add(y).to_le_bytes()[0];
        }
    }

    Ok((red, green, blue))
}

fn invalid_input(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidInput, message.into())
}

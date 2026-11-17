use std::{
    env, fs,
    io::{Error, ErrorKind},
};

use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory, stream::StreamOptions};
use singe_nvjpeg::{
    image::{Plane, Size},
    k2::{
        encode::{EncodeConfig, EncodeOptions, Encoder, EncoderHandle},
        image::EncodeInput,
        types::{BitstreamType, ColorSpace, ImageFormat},
    },
};

const WIDTH: u32 = 256;
const HEIGHT: u32 = 160;

fn main() -> anyhow::Result<()> {
    let output = env::temp_dir().join("jpeg2k_encode.jp2");
    let (red, green, blue) = rgb_gradient(WIDTH, HEIGHT)?;

    let context = CudaContext::create()?;
    let stream = context.create_stream(StreamOptions::default())?;
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
    let handle = EncoderHandle::create(&context)?;
    let mut encoder = Encoder::create(&handle, config, options)?;

    fs::write(&output, encoder.encode_to_vec(&image, &stream)?)?;
    println!("wrote {}", output.display());
    Ok(())
}

fn rgb_gradient(width: u32, height: u32) -> anyhow::Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
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

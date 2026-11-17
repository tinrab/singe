use std::{
    env, fs,
    io::{Error, ErrorKind},
};

use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory, stream::StreamOptions};
use singe_nvjpeg::{
    context::{Context, ContextOptions},
    encode::{EncodeInput, EncodeOptions, Encoder, Quality},
    image::Size,
    types::{ChromaSubsampling, EncoderBackend, InputFormat},
};

const WIDTH: u32 = 256;
const HEIGHT: u32 = 160;

fn main() -> anyhow::Result<()> {
    let output = env::temp_dir().join("encode.jpg");
    let host = rgb_gradient(WIDTH, HEIGHT)?;

    let cuda_context = CudaContext::create()?;
    let stream = cuda_context.create_stream(StreamOptions::default())?;
    let context = Context::create(&cuda_context, &ContextOptions::default())?;
    let device = DeviceMemory::<u8>::from_slice(stream.context(), &host)?;
    let row_bytes = checked_row_bytes(WIDTH)?;
    let input = EncodeInput::from_interleaved(
        device.view(),
        Size::new(WIDTH, HEIGHT),
        row_bytes,
        InputFormat::RgbInterleaved,
    )?;

    let options = EncodeOptions::new()
        .with_quality(Quality::create(90)?)
        .with_sampling(ChromaSubsampling::Css420);
    let mut encoder = Encoder::create(&context, EncoderBackend::Default, options, &stream)?;
    let jpeg = encoder.encode_to_device(&input, &stream)?;
    fs::write(&output, jpeg.copy_to_host_vec()?)?;
    println!("wrote {}", output.display());
    Ok(())
}

fn rgb_gradient(width: u32, height: u32) -> anyhow::Result<Vec<u8>> {
    let row_bytes = checked_row_bytes(width)?;
    let len = row_bytes
        .checked_mul(height as usize)
        .ok_or_else(|| invalid_input("input image size overflow"))?;
    let mut pixels = vec![0; len];

    for y in 0..height {
        for x in 0..width {
            let offset = (y as usize * row_bytes) + (x as usize * 3);
            pixels[offset] = x.to_le_bytes()[0];
            pixels[offset + 1] = y.to_le_bytes()[0];
            pixels[offset + 2] = x.wrapping_add(y).to_le_bytes()[0];
        }
    }

    Ok(pixels)
}

fn checked_row_bytes(width: u32) -> anyhow::Result<usize> {
    (width as usize)
        .checked_mul(3)
        .ok_or_else(|| invalid_input("row byte count overflow").into())
}

fn invalid_input(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidInput, message.into())
}

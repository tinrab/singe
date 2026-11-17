use std::{
    env, fs,
    io::{Error, ErrorKind, Write},
    path::{Path, PathBuf},
};

use singe_cuda::{context::Context as CudaContext, stream::StreamOptions};
use singe_nvjpeg::{
    context::{Context, ContextOptions},
    decode::{DecodeOptions, Decoder},
    types::OutputFormat,
};

fn main() -> anyhow::Result<()> {
    let input = fs::read(input_path()?)?;
    let output = env::temp_dir().join("decode.ppm");

    let cuda_context = CudaContext::create()?;
    let stream = cuda_context.create_stream(StreamOptions::default())?;
    let context = Context::create(&cuda_context, &ContextOptions::default())?;
    let options = DecodeOptions::new().with_output_format(OutputFormat::RgbInterleaved);
    let mut decoder = Decoder::create(&context, options)?;
    let image = decoder.decode_to_device(&input, &stream)?;
    let plane = image.single_plane()?;
    let size = plane.spec().size();
    let host = plane.data().copy_to_host_vec()?;
    write_ppm(&output, size.width(), size.height(), &host)?;
    println!("wrote {}", output.display());
    Ok(())
}

fn input_path() -> anyhow::Result<PathBuf> {
    env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| {
            invalid_input("usage: cargo run -p singe-nvjpeg --example decode -- <input.jpg>")
        })
        .map_err(Into::into)
}

fn write_ppm(path: &Path, width: u32, height: u32, rgb: &[u8]) -> anyhow::Result<()> {
    let mut file = fs::File::create(path)?;
    write!(file, "P6\n{width} {height}\n255\n")?;
    file.write_all(rgb)?;
    Ok(())
}

fn invalid_input(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidInput, message.into())
}

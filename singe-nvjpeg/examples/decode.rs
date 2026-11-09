use std::{
    env,
    error::Error as StdError,
    fs,
    io::{Error, ErrorKind, Write},
    path::{Path, PathBuf},
};

use singe_cuda::context::Context as CudaContext;
use singe_nvjpeg::prelude::*;

type ExampleResult<T> = Result<T, Box<dyn StdError>>;

fn main() -> ExampleResult<()> {
    let input = input_path()?;
    let output = output_path();
    fs::create_dir_all(
        output
            .parent()
            .ok_or_else(|| invalid_input("missing output parent"))?,
    )?;

    let context = CudaContext::create()?;
    let stream = context.create_stream()?;
    let options = DecodeOptions::new().with_output_format(OutputFormat::RgbInterleaved);
    let mut decoder = Decoder::builder()
        .backend(Backend::Default)
        .options(options)
        .create(&context)?;
    let image = decoder.decode_file_to_device(&input, &stream)?;
    let plane = image.single_plane()?;
    let size = plane.spec().size();
    let host = plane.data().copy_to_host_vec()?;
    write_ppm(&output, size.width(), size.height(), &host)?;
    println!("wrote {}", output.display());
    Ok(())
}

fn input_path() -> ExampleResult<PathBuf> {
    env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| {
            invalid_input("usage: cargo run -p singe-nvjpeg --example decode -- <input.jpg>")
        })
        .map_err(Into::into)
}

fn output_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("output")
        .join("decode.ppm")
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

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
    let inputs = input_paths()?;
    let output_dir = output_dir();
    fs::create_dir_all(&output_dir)?;

    let jpeg_data = inputs.iter().map(fs::read).collect::<Result<Vec<_>, _>>()?;
    let jpeg_refs = jpeg_data.iter().map(Vec::as_slice).collect::<Vec<_>>();

    let context = CudaContext::create()?;
    let stream = context.create_stream()?;
    let config = BatchConfig::create(jpeg_refs.len(), 1, OutputFormat::RgbInterleaved)?;
    let mut decoder = BatchedDecoder::builder(config)
        .backend(Backend::Default)
        .create(&context)?;
    let outputs = decoder.decode_to_device(&jpeg_refs, &stream)?;

    for (index, output) in outputs.iter().enumerate() {
        let plane = output.single_plane()?;
        let size = plane.spec().size();
        let host = plane.data().copy_to_host_vec()?;
        let path = output_dir.join(format!("batch_decode_{index}.ppm"));
        write_ppm(&path, size.width(), size.height(), &host)?;
        println!("wrote {}", path.display());
    }

    Ok(())
}

fn input_paths() -> ExampleResult<Vec<PathBuf>> {
    let inputs = env::args_os()
        .skip(1)
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    if inputs.is_empty() {
        return Err(invalid_input(
            "usage: cargo run -p singe-nvjpeg --example batch_decode -- <input0.jpg> [input1.jpg ...]",
        )
        .into());
    }
    Ok(inputs)
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("output")
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

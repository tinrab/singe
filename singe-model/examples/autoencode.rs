// Encodes an image file to Stable Diffusion's latents with a TAESD snapshot and decodes it again: `--image FILE --output FILE`, at `--size`, 512 by default.

mod support;

use anyhow::Result;
use singe::{DataType, Shape, Tensor};
use singe_model::vision::image_to_image::load_autoencoder;

use crate::support::{Arguments, read_image, write_image};

fn main() -> Result<()> {
    let arguments = Arguments::parse()?;
    let size = arguments.value("size").map_or(Ok(512), str::parse::<u32>)?;
    let pixels = read_image(arguments.required("image")?, size, size)?;
    let extent = i64::from(size);
    // TAESD reads images in [-1, 1].
    let images = Tensor::from_vec(
        arguments.device,
        Shape::create([1, 3, extent, extent])?,
        pixels,
    )?
    .mul_scalar(2.0)?
    .sub_scalar(1.0)?;
    let model = load_autoencoder(&arguments.snapshot, arguments.device, DataType::F32)?;
    let latents = model.encode(&images)?;
    eprintln!("latents {:?}", latents.shape().dimensions());
    let decoded = model
        .decode(&latents)?
        .add_scalar(1.0)?
        .div_scalar(2.0)?
        .cast(DataType::F32)?
        .to_vec::<f32>()?;
    write_image(arguments.required("output")?, &decoded, size, size)
}

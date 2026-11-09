// Embeds an image file with a DINOv2 snapshot, resized to the model's trained size and normalized with ImageNet's statistics, and prints its class embedding's norm and first values.

mod support;

use anyhow::Result;
use singe::{DataType, Shape, Tensor};
use singe_model::{
    config::Snapshot,
    vision::image_feature_extraction::{dinov2::Dinov2Config, load_image_encoder},
};

use crate::support::{Arguments, read_image};

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

fn main() -> Result<()> {
    let arguments = Arguments::parse()?;
    let config: Dinov2Config = Snapshot::new(&arguments.snapshot).config()?;
    let size = u32::try_from(config.image_size)?;
    let mut pixels = read_image(arguments.required("image")?, size, size)?;
    let plane = pixels.len() / 3;
    for (index, value) in pixels.iter_mut().enumerate() {
        let channel = index / plane;
        *value = (*value - MEAN[channel]) / STD[channel];
    }

    let extent = i64::from(size);
    let images = Tensor::from_vec(
        arguments.device,
        Shape::create([1, 3, extent, extent])?,
        pixels,
    )?;
    let model = load_image_encoder(&arguments.snapshot, arguments.device, DataType::F32)?;
    let hidden = model.forward(&images)?;
    let embedding = model
        .layer()
        .pooled(&hidden)?
        .cast(DataType::F32)?
        .to_vec::<f32>()?;
    let norm = embedding
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    println!(
        "class embedding of {} values, norm {norm:.3}: {:?}",
        embedding.len(),
        &embedding[..8]
    );
    Ok(())
}

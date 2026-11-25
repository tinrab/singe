//! Writes a loaded model for one task in a form that runs without Singe: [`cuda`] as CUDA C++ programs, and [`onnx`] as an ONNX file.

#[cfg(feature = "cuda_13_4")]
pub mod cuda;
#[cfg(feature = "onnx")]
pub mod onnx;

use crate::{
    error::{Error, Result},
    processor::image::{ImageTransforms, Resize},
};

/// Returns the `[height, width]` an image processor gives every image, which an image encoder's program reads.
fn image_size(processor: &ImageTransforms) -> Result<[i64; 2]> {
    match (processor.crop, processor.resize) {
        (Some(crop), _) => Ok(crop),
        (None, Some(Resize::Exact { height, width })) => Ok([height, width]),
        _ => Err(Error::unsupported(
            "exporting an image encoder whose processor keeps the aspect ratio",
        )),
    }
}

//! Writes a loaded model for one task in a form that runs without Singe: [`cuda`] as CUDA C++ programs, and [`onnx`] as an ONNX file.

#[cfg(feature = "cuda_13_4")]
pub mod cuda;
#[cfg(feature = "onnx")]
pub mod onnx;
#[cfg(feature = "onnx")]
pub mod rewrite;

#[cfg(any(feature = "onnx", feature = "cuda_13_4"))]
use singe::shape::spatial::{Spatial2d, Spatial2dI64};
#[cfg(feature = "onnx")]
use singe::{DataType, Tensor};

#[cfg(any(feature = "onnx", feature = "cuda_13_4"))]
use crate::{
    error::{Error, Result},
    processor::image::{ImageTransforms, Resize},
};

#[cfg(feature = "onnx")]
#[derive(Debug)]
struct Compared {
    line: String,
    matched: bool,
}

#[cfg(feature = "onnx")]
fn compare(label: &str, want: &Tensor, have: &Tensor) -> Result<Compared> {
    let data_type = want.data_type();
    let want = want.cast(DataType::F32)?.to_vec::<f32>()?;
    let have = have.cast(DataType::F32)?.to_vec::<f32>()?;
    let scale = want
        .iter()
        .fold(0.0_f32, |scale, value| scale.max(value.abs()));
    let difference = want
        .iter()
        .zip(&have)
        .fold(0.0_f32, |largest, (want, have)| {
            let difference = (want - have).abs();
            if difference.is_nan() {
                f32::INFINITY
            } else {
                largest.max(difference)
            }
        });
    let tolerance = match data_type {
        DataType::F32 => 1e-2,
        _ => 5e-2,
    } * scale.max(1.0);
    Ok(Compared {
        line: format!("{label}: largest difference {difference:.3e}, values up to {scale:.3e}"),
        matched: want.len() == have.len() && difference <= tolerance,
    })
}

#[cfg(feature = "onnx")]
fn matched_lines(comparisons: Vec<Compared>) -> Result<Vec<String>> {
    let failures = comparisons
        .iter()
        .filter(|compared| !compared.matched)
        .map(|compared| compared.line.as_str())
        .collect::<Vec<_>>();
    if !failures.is_empty() {
        return Err(Error::invalid(format!(
            "outputs that differ from Singe's: {}",
            failures.join("; ")
        )));
    }
    Ok(comparisons
        .into_iter()
        .map(|compared| compared.line)
        .collect())
}

/// Returns the size an image processor gives every image, which an image encoder's program reads.
#[cfg(any(feature = "onnx", feature = "cuda_13_4"))]
fn image_size(processor: &ImageTransforms) -> Result<Spatial2dI64> {
    match (processor.crop, processor.resize) {
        (Some(crop), _) => Ok(crop),
        (None, Some(Resize::Exact { height, width })) => Ok(Spatial2d::new(height, width)),
        _ => Err(Error::unsupported(
            "exporting an image encoder whose processor keeps the aspect ratio",
        )),
    }
}

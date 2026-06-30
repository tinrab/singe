#![allow(clippy::module_inception)]

mod advanced;
mod attention;
mod block_scale;
mod compile;
mod convolution;
mod normalization;
mod operation;
mod pointwise;
mod reduction;
mod resample;
mod rng;
mod stats;
mod tensor_ops;

pub use advanced::*;
pub use attention::*;
pub use block_scale::*;
pub use compile::*;
pub use convolution::*;
pub use normalization::*;
pub use operation::*;
pub use pointwise::*;
pub use reduction::*;
pub use resample::*;
pub use rng::*;
pub use stats::*;
pub use tensor_ops::*;

use crate::tensor::TensorId;

pub(crate) trait FrontendOperationTensors {
    fn append_tensor_ids(&self, tensors: &mut Vec<TensorId>);
}

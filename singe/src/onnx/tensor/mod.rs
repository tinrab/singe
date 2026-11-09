mod data;
mod data_type;
mod external;

use std::path::Path;

use singe_onnx::Tensor;

pub use self::data_type::data_type;
use crate::{
    onnx::{
        error::{Error, Result},
        options::OnnxOptions,
        shape::shape_from_dimensions,
    },
    tensor::Tensor as SingeTensor,
};

pub fn tensor_from_onnx(
    tensor: &Tensor,
    options: &OnnxOptions,
    model_dir: Option<&Path>,
) -> Result<SingeTensor> {
    if tensor.segment.is_some() {
        return Err(Error::invalid_initializer(
            tensor_name(tensor),
            "segmented tensors are not supported",
        ));
    }

    let name = tensor_name(tensor);
    let data_type = data_type(&name, tensor.data_type)?;
    let shape =
        shape_from_dimensions(tensor.dimensions.clone()).map_err(|_| Error::UnsupportedShape {
            name: name.clone(),
            reason: "invalid tensor dimensions".into(),
        })?;
    let bytes = data::tensor_bytes(tensor, data_type, model_dir)?;
    SingeTensor::from_bytes(options.device, data_type, shape, &bytes).map_err(Into::into)
}

pub fn tensor_name(tensor: &Tensor) -> String {
    if tensor.name.is_empty() {
        "<unnamed>".into()
    } else {
        tensor.name.clone()
    }
}

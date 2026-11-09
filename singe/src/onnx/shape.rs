use std::result::Result as StdResult;

use singe_onnx::{DimensionValue, TensorShape, TensorType, TypeValue, ValueInfo};

use crate::{
    backend::spec,
    error::Error as SingeError,
    onnx::{
        error::{Error, Result},
        options::OnnxOptions,
    },
    shape::Shape,
};

pub fn concrete_shape(
    name: &str,
    tensor_shape: Option<&TensorShape>,
    options: &OnnxOptions,
) -> Result<Shape> {
    let Some(tensor_shape) = tensor_shape else {
        return Err(Error::UnsupportedShape {
            name: name.into(),
            reason: "missing tensor shape".into(),
        });
    };
    let override_shape = options.input_shapes.get(name);
    let mut dimensions = Vec::with_capacity(tensor_shape.dimensions.len());
    for (axis, dimension) in tensor_shape.dimensions.iter().enumerate() {
        let value = match &dimension.value {
            Some(DimensionValue::Value(value)) => {
                if override_shape
                    .and_then(|shape| shape.get(axis))
                    .is_some_and(|specified| specified != value)
                {
                    return Err(Error::UnsupportedShape {
                        name: name.into(),
                        reason: format!(
                            "shape override conflicts with fixed dimension at axis {axis}"
                        ),
                    });
                }
                *value
            }
            Some(DimensionValue::Parameter(parameter)) => override_shape
                .and_then(|shape| shape.get(axis).copied())
                .or_else(|| options.symbolic_dimensions.get(parameter).copied())
                .ok_or_else(|| Error::UnsupportedShape {
                    name: name.into(),
                    reason: format!("unresolved symbolic dimension `{parameter}` at axis {axis}"),
                })?,
            None => override_shape
                .and_then(|shape| shape.get(axis).copied())
                .ok_or_else(|| Error::UnsupportedShape {
                    name: name.into(),
                    reason: format!("unknown dimension at axis {axis}"),
                })?,
        };
        dimensions.push(value);
    }

    if let Some(override_shape) = override_shape
        && override_shape.len() != dimensions.len()
    {
        return Err(Error::UnsupportedShape {
            name: name.into(),
            reason: format!(
                "shape override rank {} does not match onnx rank {}",
                override_shape.len(),
                dimensions.len()
            ),
        });
    }
    shape_from_dimensions(dimensions).map_err(|_| Error::UnsupportedShape {
        name: name.into(),
        reason: "invalid concrete shape".into(),
    })
}

pub fn shape_from_dimensions(dimensions: impl Into<Vec<i64>>) -> StdResult<Shape, SingeError> {
    let dimensions = dimensions.into();
    if dimensions.is_empty() {
        Ok(Shape::scalar())
    } else {
        Shape::create(dimensions)
    }
}

pub fn value_tensor_type(value: &ValueInfo) -> Result<&TensorType> {
    let value_type = value
        .value_type
        .as_ref()
        .ok_or_else(|| Error::UnsupportedShape {
            name: value.name.clone(),
            reason: "missing value type".into(),
        })?;
    match &value_type.value {
        TypeValue::Tensor(tensor) => Ok(tensor),
        _ => Err(Error::UnsupportedShape {
            name: value.name.clone(),
            reason: "only tensor values are supported".into(),
        }),
    }
}

pub fn conv_output_shape(
    input: &Shape,
    weight: &Shape,
    stride: (i64, i64),
    padding: (i64, i64, i64, i64),
    dilation: (i64, i64),
    groups: i64,
) -> Result<Shape> {
    spec::conv2d_output_shape(input, weight, stride, padding, dilation, groups).map_err(Into::into)
}

pub fn pool_output_shape(
    input: &Shape,
    kernel: (i64, i64),
    stride: (i64, i64),
    padding: (i64, i64, i64, i64),
    dilation: (i64, i64),
    ceil_mode: bool,
) -> Result<Shape> {
    if input.rank() != 4 {
        return Err(Error::UnsupportedShape {
            name: "pool input".into(),
            reason: format!(
                "pool output shape expects rank-4 input, got rank {} shape {:?}",
                input.rank(),
                input.dimensions()
            ),
        });
    }
    spec::pool2d_output_shape(input, kernel, stride, padding, dilation, ceil_mode)
        .map_err(Into::into)
}

pub fn gemm_output_shape(a: &Shape, b: &Shape, trans_a: bool, trans_b: bool) -> Result<Shape> {
    spec::gemm_output_shape(a, b, trans_a, trans_b).map_err(Into::into)
}

pub fn concat_shape(inputs: &[Shape], axis: i64) -> Result<Shape> {
    spec::concat_output_shape(inputs, axis).map_err(Into::into)
}

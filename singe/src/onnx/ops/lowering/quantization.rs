use singe_onnx::Node;

use crate::{
    data_type::DataType,
    onnx::{
        error::{Error, Result},
        importer::Importer,
        ops::lowering::optional_input,
    },
    tensor::TensorDataType,
};

pub fn quantization_scale(
    importer: &Importer,
    node: &Node,
    index: usize,
    name: &str,
) -> Result<f32> {
    let scale = scalar_initializer::<f32>(importer, node, index, name)?;
    if !scale.is_finite() || scale <= 0.0 {
        let input_name = &node.inputs[index];
        return Err(Error::invalid_initializer(
            input_name,
            format!("{name} must be finite and positive"),
        ));
    }
    Ok(scale)
}

pub fn optional_quantization_zero_point(
    importer: &Importer,
    node: &Node,
    index: usize,
    name: &str,
    data_type: DataType,
    default: i64,
) -> Result<i64> {
    match optional_input(importer, node, index)? {
        Some(zero_point) => {
            if zero_point.data_type != data_type {
                return Err(Error::unsupported_operator(
                    node.domain.as_deref(),
                    &node.operator_type,
                    importer.opset_version(node.domain.as_deref()),
                    format!("{name} dtype must match the corresponding input dtype"),
                ));
            }
            zero_point_initializer(importer, node, index, name, data_type)
        }
        None => Ok(default),
    }
}

pub fn zero_point_initializer(
    importer: &Importer,
    node: &Node,
    index: usize,
    name: &str,
    data_type: DataType,
) -> Result<i64> {
    match data_type {
        DataType::I8 => Ok(i64::from(scalar_initializer::<i8>(
            importer, node, index, name,
        )?)),
        DataType::U8 => Ok(i64::from(scalar_initializer::<u8>(
            importer, node, index, name,
        )?)),
        _ => Err(Error::unsupported_operator(
            node.domain.as_deref(),
            &node.operator_type,
            importer.opset_version(node.domain.as_deref()),
            format!("{name} must be an int8 or uint8 scalar"),
        )),
    }
}

pub fn scalar_initializer<T: TensorDataType>(
    importer: &Importer,
    node: &Node,
    index: usize,
    name: &str,
) -> Result<T> {
    let (input_name, tensor) = importer.required_static_input_tensor_with_reasons(
        node,
        index,
        name,
        format!("{name} input is optional only for unsupported quantization modes"),
        format!("{name} must be a constant scalar initializer or Constant node output"),
    )?;
    if tensor.shape().rank() != 0 {
        return Err(Error::invalid_initializer(
            &input_name,
            "only rank-zero scalar operands are supported",
        ));
    }

    if tensor.data_type() != T::data_type() {
        return Err(Error::invalid_initializer(
            &input_name,
            format!("expected scalar {} tensor", T::data_type()),
        ));
    }

    let values = tensor.to_vec::<T>()?;
    match values.as_slice() {
        [value] => Ok(*value),
        _ => Err(Error::invalid_initializer(
            &input_name,
            format!("{name} must contain exactly one element"),
        )),
    }
}

pub fn optional_static_f32_scalar(
    importer: &Importer,
    node: &Node,
    index: usize,
    name: &str,
) -> Result<Option<f32>> {
    let Some(input_name) = node.inputs.get(index) else {
        return Ok(None);
    };
    if input_name.is_empty() {
        return Ok(None);
    }

    let Some((_, tensor)) = importer.optional_static_input_tensor(node, index)? else {
        return Err(Error::unsupported_operator(
            node.domain.as_deref(),
            &node.operator_type,
            importer.opset_version(node.domain.as_deref()),
            format!("{name} must be a constant scalar initializer or Constant node output"),
        ));
    };
    if tensor.shape().rank() != 0 {
        return Err(Error::invalid_initializer(
            input_name,
            "only rank-zero scalar operands are supported",
        ));
    }

    if tensor.data_type() != DataType::F32 {
        return Err(Error::invalid_initializer(
            input_name,
            format!("{name} must be an f32 scalar tensor"),
        ));
    }

    let values = tensor.to_vec::<f32>()?;
    match values.as_slice() {
        [value] => Ok(Some(*value)),
        _ => Err(Error::invalid_initializer(
            input_name,
            format!("{name} must contain exactly one element"),
        )),
    }
}

pub fn static_f32_scalar(
    importer: &Importer,
    node: &Node,
    index: usize,
    name: &str,
) -> Result<f32> {
    optional_static_f32_scalar(importer, node, index, name)?.ok_or_else(|| {
        Error::unsupported_operator(
            node.domain.as_deref(),
            &node.operator_type,
            importer.opset_version(node.domain.as_deref()),
            format!("{name} input is required"),
        )
    })
}

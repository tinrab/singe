use singe_onnx::{
    AttributeValue, Node, Tensor as OnnxTensor, TensorDataType as OnnxTensorDataType,
};

use crate::{
    data_type::DataType,
    onnx::{
        attributes::Attributes,
        error::{Error, Result},
        importer::{ImportedValue, Importer},
        ops::{
            expect_input_count, expect_outputs, register_node_outputs,
            shape_from_static_dimensions, shape_input_values,
        },
        tensor::tensor_from_onnx,
        utility::node_label,
    },
    shape::Shape,
    tensor::{Tensor, TensorDataType},
};

pub fn lower_constant(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_count(importer, node, 0)?;
    expect_outputs(importer, node, 1)?;
    let attrs = Attributes::create(node)?;
    let values = [
        "value",
        "value_float",
        "value_int",
        "value_floats",
        "value_ints",
        "value_string",
        "value_strings",
        "sparse_value",
    ]
    .into_iter()
    .filter_map(|name| attrs.value(name))
    .collect::<Vec<_>>();
    let [value] = values.as_slice() else {
        return Err(Error::invalid_attribute(
            node_label(node),
            "value",
            "exactly one constant value attribute is required",
        ));
    };
    let tensor = constant_attribute_tensor(importer, node, value)?;
    register_constant_tensor(importer, node, &tensor)
}

pub fn register_constant_tensor(
    importer: &mut Importer,
    node: &Node,
    tensor: &Tensor,
) -> Result<()> {
    let id = importer.graph().add_constant(tensor.clone());
    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(
            id,
            tensor.data_type(),
            tensor.shape().clone(),
        )],
    )
}

pub fn constant_attribute_tensor(
    importer: &Importer,
    node: &Node,
    value: &AttributeValue,
) -> Result<Tensor> {
    match value {
        AttributeValue::Tensor(tensor) => {
            tensor_from_onnx(tensor, importer.options(), importer.model_dir())
        }
        AttributeValue::Float(value) => {
            Tensor::from_vec(importer.options().device, Shape::scalar(), vec![*value])
                .map_err(Into::into)
        }
        AttributeValue::Int(value) => {
            Tensor::from_vec(importer.options().device, Shape::scalar(), vec![*value])
                .map_err(Into::into)
        }
        AttributeValue::Floats(values) => {
            let shape = Shape::create([i64::try_from(values.len()).map_err(|_| {
                Error::UnsupportedShape {
                    name: node_label(node),
                    reason: "constant float list is too large".into(),
                }
            })?])?;
            Tensor::from_vec(importer.options().device, shape, values.clone()).map_err(Into::into)
        }
        AttributeValue::Ints(values) => {
            let shape = Shape::create([i64::try_from(values.len()).map_err(|_| {
                Error::UnsupportedShape {
                    name: node_label(node),
                    reason: "constant int list is too large".into(),
                }
            })?])?;
            Tensor::from_vec(importer.options().device, shape, values.clone()).map_err(Into::into)
        }
        _ => Err(Error::unsupported_operator(
            node.domain.as_deref(),
            "Constant",
            importer.opset_version(node.domain.as_deref()),
            "only tensor, float, int, floats, and ints value attributes are supported",
        )),
    }
}

pub fn lower_constant_of_shape(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_count(importer, node, 1)?;
    expect_outputs(importer, node, 1)?;
    let attrs = Attributes::create(node)?;
    let dimensions = shape_input_values(importer, node, 0, "shape")?;
    let shape = shape_from_static_dimensions(dimensions)?;
    let tensor = match attrs.value("value") {
        Some(AttributeValue::Tensor(value)) => {
            constant_of_shape_tensor(importer, node, &shape, value)?
        }
        Some(_) => {
            return Err(Error::invalid_attribute(
                node_label(node),
                "value",
                "expected scalar tensor",
            ));
        }
        None => Tensor::full(importer.options().device, shape, 0.0_f32)?,
    };
    register_constant_tensor(importer, node, &tensor)
}

pub fn constant_of_shape_tensor(
    importer: &Importer,
    node: &Node,
    shape: &Shape,
    value: &OnnxTensor,
) -> Result<Tensor> {
    let value = tensor_from_onnx(value, importer.options(), importer.model_dir())?;
    if value.shape().element_count() != 1 {
        return Err(Error::invalid_attribute(
            node_label(node),
            "value",
            "expected scalar tensor",
        ));
    }
    match value.data_type() {
        DataType::F32 => fill_from_scalar::<f32>(importer, node, shape, &value),
        DataType::I64 => fill_from_scalar::<i64>(importer, node, shape, &value),
        data_type => Err(Error::UnsupportedType {
            name: node_label(node),
            data_type: OnnxTensorDataType::Undefined,
            reason: format!("constantofshape scalar dtype {data_type} is not supported"),
        }),
    }
}

fn fill_from_scalar<T>(
    importer: &Importer,
    node: &Node,
    shape: &Shape,
    value: &Tensor,
) -> Result<Tensor>
where
    T: TensorDataType,
{
    let [value]: [T; 1] = value
        .to_vec::<T>()?
        .try_into()
        .map_err(|_| Error::UnsupportedShape {
            name: node_label(node),
            reason: format!(
                "constantofshape {} scalar value has invalid element count",
                T::data_type()
            ),
        })?;
    Tensor::full(importer.options().device, shape.clone(), value).map_err(Into::into)
}

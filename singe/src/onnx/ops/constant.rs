use singe_onnx::{
    AttributeValue, Node, Tensor as OnnxTensor, TensorDataType as OnnxTensorDataType,
};

use crate::{
    data_type::DataType,
    device::Device,
    graph::{DimensionExpr, SymbolicAttributes, ir::ValueId},
    onnx::{
        attributes::Attributes,
        error::{Error, Result},
        importer::{ImportedValue, Importer},
        ops::{
            expect_input_count, expect_outputs, register_node_outputs,
            shape_from_static_dimensions, shape_input_expressions,
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
    register_constant_tensor(importer, node, &tensor)?;
    Ok(())
}

pub fn register_constant_tensor(
    importer: &mut Importer,
    node: &Node,
    tensor: &Tensor,
) -> Result<ValueId> {
    let id = importer.graph().add_constant(tensor.clone());
    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(
            id,
            tensor.data_type(),
            tensor.shape().clone(),
        )],
    )?;
    Ok(id)
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
    let expressions = shape_input_expressions(importer, node, 0, "shape")?;
    let shape = shape_from_static_dimensions(importer.evaluate_dimensions(&expressions)?)?;
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
    let id = register_constant_tensor(importer, node, &tensor)?;
    if !expressions.iter().all(DimensionExpr::is_constant) {
        let value = tensor
            .to_device(Device::Cpu)?
            .reshape(vec![-1])?
            .narrow(0, 0, 1)?;
        importer.graph().shape_program.attributes.insert(
            id,
            SymbolicAttributes::Fill {
                dimensions: expressions,
                value: value.contiguous()?,
            },
        );
    }
    Ok(())
}

// ONNX `Range`, whose bounds may follow run-time axes when it counts `I64` values by one.
pub fn lower_range(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_count(importer, node, 3)?;
    expect_outputs(importer, node, 1)?;
    let [start, limit, delta] = [0, 1, 2].map(|index| importer.required_input(node, index));
    let data_type = start?.data_type;
    if limit?.data_type != data_type || delta?.data_type != data_type {
        return Err(Error::UnsupportedShape {
            name: node_label(node),
            reason: "range operands must share one data type".into(),
        });
    }

    if data_type == DataType::I64 {
        let [start, limit, delta] =
            [0, 1, 2].map(|index| shape_input_expressions(importer, node, index, "range operand"));
        let ([start], [limit], [delta]) = (&start?[..], &limit?[..], &delta?[..]) else {
            return Err(Error::UnsupportedShape {
                name: node_label(node),
                reason: "range operands must be scalars".into(),
            });
        };
        let step = importer.evaluate_dimension(delta)?;
        let (first, last) = (
            importer.evaluate_dimension(start)?,
            importer.evaluate_dimension(limit)?,
        );
        if start.is_constant() && limit.is_constant() && delta.is_constant() {
            let tensor = Tensor::arange(importer.options().device, first, last, step)?;
            register_constant_tensor(importer, node, &tensor)?;
            return Ok(());
        }

        if !delta.is_constant() || step != 1 {
            return Err(Error::UnsupportedShape {
                name: node_label(node),
                reason: "a run-time range must count by a constant one".into(),
            });
        }

        let tensor = Tensor::arange(importer.options().device, first, last.max(first), 1)?;
        let id = register_constant_tensor(importer, node, &tensor)?;
        importer.graph().shape_program.attributes.insert(
            id,
            SymbolicAttributes::Range {
                start: start.clone(),
                end: limit.clone(),
            },
        );
        return Ok(());
    }

    let scalar = |index| -> Result<f64> {
        let (name, tensor) = importer.required_static_input_tensor(node, index, "range operand")?;
        let values = tensor
            .to_device(Device::Cpu)?
            .cast(DataType::F64)?
            .to_vec::<f64>()?;
        match values[..] {
            [value] => Ok(value),
            _ => Err(Error::invalid_initializer(
                &name,
                "range operands must be scalars",
            )),
        }
    };
    let (start, limit, delta) = (scalar(0)?, scalar(1)?, scalar(2)?);
    let device = importer.options().device;
    let tensor = match data_type {
        DataType::F64 => Tensor::arange(device, start, limit, delta)?,
        DataType::F32 => Tensor::arange(device, start as f32, limit as f32, delta as f32)?,
        data_type => {
            return Err(Error::UnsupportedShape {
                name: node_label(node),
                reason: format!("range of {data_type} values"),
            });
        }
    };
    register_constant_tensor(importer, node, &tensor)?;
    Ok(())
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

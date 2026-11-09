use singe_onnx::Node;

use crate::{
    data_type::DataType,
    graph::{DimensionExpr, ReshapeExpression, ShapeValue, node::NodeKind},
    onnx::{
        attributes::Attributes,
        error::{Error, Result},
        importer::{ImportedValue, Importer},
        ops::{
            expand_output_shape, expect_input_count, expect_input_range, expect_outputs,
            normalize_shape_bound, register_node_outputs, shape_input_values,
            shape_ops::onnx_dimension_to_usize,
        },
        utility::{node_label, node_name},
    },
    shape::Shape,
    tensor::Tensor,
};

pub fn lower_expand(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_count(importer, node, 2)?;
    expect_outputs(importer, node, 1)?;
    let input = importer.required_input(node, 0)?;
    let dimensions = shape_input_values(importer, node, 1, "shape")?;
    let shape = expand_output_shape(&node_label(node), &input.shape, dimensions)?;
    let device = importer.options().device;
    let id = importer.graph().add_named_operation(
        node_name(node),
        NodeKind::expand(input.node, shape.clone()),
        input.data_type,
        shape.clone(),
        device,
    )?;
    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(id, input.data_type, shape)],
    )
}

pub fn lower_reshape(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_range(importer, node, 2, 2)?;
    expect_outputs(importer, node, 1)?;
    let attrs = Attributes::create(node)?;
    let input = importer.required_input(node, 0)?;
    let target = importer.required_input(node, 1)?;
    let tracked = importer.metadata_shape_value(&node.inputs[1])?.cloned();
    if tracked.is_some() && target.shape.rank() != 1 {
        return Err(Error::UnsupportedShape {
            name: node_label(node),
            reason: "reshape shape operand must have rank one".into(),
        });
    }

    let mut dimensions = if let Some(value) = &tracked {
        importer.evaluate_shape_value(value)?
    } else {
        shape_input_values(importer, node, 1, "shape")?
    };
    let expressions = tracked.map_or_else(
        || {
            dimensions
                .iter()
                .copied()
                .map(DimensionExpr::constant)
                .collect()
        },
        |value| value.elements().to_vec(),
    );
    let allow_zero = attrs.int("allowzero", 0)? != 0;
    if allow_zero && dimensions.contains(&0) && dimensions.contains(&-1) {
        return Err(Error::UnsupportedShape {
            name: node_label(node),
            reason: "reshape allowzero=1 cannot be combined with inferred dimensions".into(),
        });
    }

    if !allow_zero {
        for (axis, dimension) in dimensions.iter_mut().enumerate() {
            if *dimension == 0 {
                *dimension =
                    *input
                        .shape
                        .dimensions()
                        .get(axis)
                        .ok_or_else(|| Error::UnsupportedShape {
                            name: node_label(node),
                            reason: "reshape zero copy axis is out of range".into(),
                        })?;
            }
        }
    }

    let shape = input.shape.reshape(dimensions)?;
    let device = importer.options().device;
    let id = importer.graph().add_named_operation(
        node_name(node),
        NodeKind::reshape(input.node, shape.clone()),
        input.data_type,
        shape.clone(),
        device,
    )?;
    importer.graph().shape_program.reshapes.insert(
        id,
        ReshapeExpression {
            dimensions: expressions,
            allow_zero,
        },
    );
    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(id, input.data_type, shape)],
    )
}

pub fn lower_shape(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_count(importer, node, 1)?;
    expect_outputs(importer, node, 1)?;
    let attrs = Attributes::create(node)?;
    let (_, input_shape) = importer.input_meta(node, 0)?;
    let rank = input_shape.rank();
    let start = attrs.int("start", 0)?;
    let end = attrs.int("end", rank)?;
    let start = normalize_shape_bound(start, rank);
    let end = normalize_shape_bound(end, rank);
    let end = end.max(start);
    let start = onnx_dimension_to_usize(node, "shape start", start)?;
    let end = onnx_dimension_to_usize(node, "shape end", end)?;
    let dimensions = input_shape.dimensions()[start..end].to_vec();
    let shape =
        Shape::create([
            i64::try_from(dimensions.len()).map_err(|_| Error::UnsupportedShape {
                name: node_label(node),
                reason: "shape output rank is too large".into(),
            })?,
        ])?;
    let tensor = Tensor::from_vec(importer.options().device, shape.clone(), dimensions)?;
    let id = importer.graph().add_constant(tensor);
    let input = importer.required_input(node, 0)?;
    let value = ShapeValue::Vector(
        (start..end)
            .map(|axis| importer.metadata_dimension(input.node, axis))
            .collect::<Result<Vec<_>>>()?,
    );
    importer.graph().shape_program.values.insert(id, value);
    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(id, DataType::I64, shape)],
    )
}

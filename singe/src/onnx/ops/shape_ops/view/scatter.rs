use singe_onnx::Node;

use crate::{
    backend::registry::{IndexPutAttributes, Operation, OperationAttributes},
    data_type::DataType,
    graph::{GraphOperationInputs, GraphOperationOutput, GraphOperationRequest},
    onnx::{
        attributes::Attributes,
        error::{Error, Result},
        importer::{ImportedValue, Importer},
        ops::{expect_input_count, expect_outputs, register_node_outputs},
        utility::node_name,
    },
};

/// Lowers `ScatterElements`: on rank-one data it writes `updates` at `indices` like `index_put` along dimension zero, the form the exporter emits for `strided_scatter`.
///
/// Higher-rank data at fixed shapes scatters as `Tensor::scatter` does, or adds as `Tensor::scatter_add` does under reduction `add`; other reductions are unsupported.
pub fn lower_scatter_elements(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_count(importer, node, 3)?;
    expect_outputs(importer, node, 1)?;
    let attributes = Attributes::create(node)?;
    let data = importer.required_input(node, 0)?;
    let indices = importer.required_input(node, 1)?;
    let updates = importer.required_input(node, 2)?;
    if indices.data_type != DataType::I64 {
        return Err(refuse(importer, node, "indices must be i64"));
    }

    let axis = attributes.int("axis", 0)?;
    let adds = match attributes.string("reduction", "none")?.as_str() {
        "none" => false,
        "add" => true,
        _ => {
            return Err(refuse(
                importer,
                node,
                "only reductions none and add are supported",
            ));
        }
    };
    if data.shape.rank() != 1 || adds {
        let operands = [data.node, indices.node, updates.node];
        if !importer.have_fixed_shapes(&operands)? {
            return Err(refuse(
                importer,
                node,
                "scatters of higher rank with run-time axes",
            ));
        }

        let device = importer.options().device;
        let ids = importer
            .graph()
            .append_traced(&operands, device, |inputs| {
                let scattered = if adds {
                    inputs[0].scatter_add(axis, inputs[1], inputs[2])?
                } else {
                    inputs[0].scatter(axis, inputs[1], inputs[2])?
                };
                Ok(vec![scattered])
            })?;
        let &[id] = ids.as_slice() else {
            return Err(refuse(importer, node, "a scatter without one result"));
        };
        return register_node_outputs(
            importer,
            node,
            vec![ImportedValue::new(id, data.data_type, data.shape)],
        );
    }

    if !matches!(axis, 0 | -1) {
        return Err(refuse(
            importer,
            node,
            "rank-one data scatters along axis zero",
        ));
    }

    let device = importer.options().device;
    let id = importer
        .graph()
        .add_operation_request(GraphOperationRequest::new(
            node_name(node),
            GraphOperationInputs::new(
                Operation::IndexPut,
                OperationAttributes::IndexPut(IndexPutAttributes {
                    dim: 0,
                    accumulate: false,
                }),
                vec![data.node, updates.node, indices.node],
            ),
            GraphOperationOutput::new(data.data_type, data.shape.clone(), device),
        ))?;
    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(id, data.data_type, data.shape)],
    )
}

fn refuse(importer: &Importer, node: &Node, details: &str) -> Error {
    Error::unsupported_operator(
        node.domain.as_deref(),
        "ScatterElements",
        importer.opset_version(node.domain.as_deref()),
        details,
    )
}

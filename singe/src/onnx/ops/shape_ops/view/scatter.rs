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

/// Lowers `ScatterElements` on rank-one data, where it writes `updates` at `indices` like `index_put` along dimension zero.
///
/// The exporter emits this form for `strided_scatter`; scatters over higher-rank data and reductions other than `none` are unsupported.
pub fn lower_scatter_elements(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_count(importer, node, 3)?;
    expect_outputs(importer, node, 1)?;
    let attributes = Attributes::create(node)?;
    let data = importer.required_input(node, 0)?;
    let indices = importer.required_input(node, 1)?;
    let updates = importer.required_input(node, 2)?;
    let unsupported = |details: &str| {
        Error::unsupported_operator(
            node.domain.as_deref(),
            "ScatterElements",
            importer.opset_version(node.domain.as_deref()),
            details,
        )
    };
    if data.shape.rank() != 1 || !matches!(attributes.int("axis", 0)?, 0 | -1) {
        return Err(unsupported("only rank-one data is supported"));
    }

    if attributes.string("reduction", "none")? != "none" {
        return Err(unsupported("only reduction none is supported"));
    }

    if indices.data_type != DataType::I64 {
        return Err(unsupported("indices must be i64"));
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

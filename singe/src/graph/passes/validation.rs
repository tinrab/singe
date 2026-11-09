use crate::{
    error::{Error, Result},
    graph::{Graph, GraphSignature},
    layout::Layout,
};

pub fn validate_initializers(graph: &Graph) -> Result<usize> {
    // Defaults obey the logical signature's dimension constraints.
    // A concrete specialization may have another valid shape when the caller overrides one.
    GraphSignature::create(graph)?;
    Ok(graph.initializers.len())
}

pub fn validate_value_metadata(graph: &Graph) -> Result<usize> {
    for (id, node) in graph.node_entries() {
        node.validate_terminal_declared_metadata(&format!(
            "graph pass validate_value_metadata node {}",
            id.index()
        ))?;
        if let Some(input) = node.alias_input() {
            let source = &graph.required_node(input)?;
            let expected =
                if node.view_output_device(source.output_device()) == source.output_device() {
                    node.view_output_layout(source.output_layout())?
                } else {
                    Layout::from_shape(node.output_shape())
                };
            let actual = node.output_layout();
            if &expected != actual {
                return Err(Error::layout_mismatch(
                    format!(
                        "graph pass validate_value_metadata node {} alias layout",
                        id.index()
                    ),
                    expected.dimensions().to_vec(),
                    expected.strides().to_vec(),
                    expected.storage_offset(),
                    actual.dimensions().to_vec(),
                    actual.strides().to_vec(),
                    actual.storage_offset(),
                ));
            }
        }
    }
    Ok(graph.node_count())
}

pub fn validate_node_contracts(graph: &Graph) -> Result<usize> {
    for (id, node) in graph.node_entries() {
        graph
            .validate_node_contracts_with_schema(
                id,
                &node,
                &format!("graph pass validate_node_contracts node {}", id.index()),
            )
            .map_err(|error| error.with_graph_source(node.source_location()))?;
    }
    Ok(graph.node_count())
}

pub fn validate_outputs(graph: &Graph) -> Result<usize> {
    for output in graph.output_projections() {
        let Ok(output_count) = graph.node_output_count(output.node()) else {
            return Err(Error::unsupported_operation_with_details(
                "graph pass validate_outputs",
                format!("missing graph output node {}", output.node().index()),
            ));
        };
        if output.output_index() >= output_count {
            return Err(Error::unsupported_operation_with_details(
                "graph pass validate_outputs",
                format!(
                    "missing graph output value {}:{}",
                    output.node().index(),
                    output.output_index()
                ),
            ));
        }
    }
    Ok(graph.output_projections().len())
}

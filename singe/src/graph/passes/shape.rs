use crate::{
    backend::spec as backend_spec,
    error::Result,
    graph::{Graph, ir::ValueId},
    shape::Shape,
};

pub fn infer_value_metadata(graph: &Graph) -> Result<(Graph, Vec<(ValueId, ValueId)>)> {
    let mut graph = graph.clone();
    let source_count = graph.node_count();
    graph.infer_value_metadata()?;
    if graph.node_count() == source_count {
        return Ok((graph, Vec::new()));
    }

    // Inserted copies have later ids than their readers.
    // Restore dependency order before optimizers that traverse construction nodes in insertion order.
    let remap = graph.renumber_in_execution_order()?;
    let remapped = (0..source_count)
        .map(ValueId::new)
        .filter_map(|old| Some((old, remap.get(old)?)))
        .collect();
    Ok((graph, remapped))
}

pub fn repeat_kv_output_shape_for_input(
    graph: &Graph,
    input: ValueId,
    repeats: i64,
) -> Result<Shape> {
    let input_shape = graph.required_value_shape(input)?;
    Ok(backend_spec::repeat_kv_problem(
        input_shape,
        repeats,
        "graph pass infer_value_metadata repeat_kv",
    )?
    .output_shape)
}

pub fn pad_output_shape_for_input(
    graph: &Graph,
    input: ValueId,
    padding: &[(i64, i64)],
) -> Result<Shape> {
    let input_shape = graph.required_value_shape(input)?;
    Ok(
        backend_spec::pad_problem(input_shape, padding, "graph pass infer_value_metadata pad")?
            .output_shape,
    )
}

pub fn concat_output_shape_for_inputs(
    graph: &Graph,
    inputs: &[ValueId],
    dim: i64,
) -> Result<Shape> {
    let shapes = inputs
        .iter()
        .map(|input| {
            graph
                .required_node(*input)
                .map(|node| node.output_shape().clone())
        })
        .collect::<Result<Vec<_>>>()?;
    let shape_refs = shapes.iter().collect::<Vec<_>>();
    Ok(
        backend_spec::concat_problem(&shape_refs, dim, "graph pass infer_value_metadata concat")?
            .output_shape,
    )
}

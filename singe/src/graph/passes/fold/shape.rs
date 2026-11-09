use crate::{
    backend::registry::Operation,
    error::{Error, Result},
    graph::{Graph, ir::ValueId, node::NodeRef, passes::fold::constant_input},
    tensor::Tensor,
};

pub fn folded_small_shape_gather_constant(
    graph: &Graph,
    node: &NodeRef<'_>,
) -> Result<Option<Tensor>> {
    if node.operation() == Some(Operation::Take) {
        let inputs = node.inputs();
        let Some(input) = constant_input(graph, inputs[0])? else {
            return Ok(None);
        };
        let Some(indices) = constant_input(graph, inputs[1])? else {
            return Ok(None);
        };
        let dim = node
            .attributes()
            .index_select_dimension()
            .ok_or_else(|| Error::missing_node("folded small shape gather constant"))?;
        return Ok(Some(input.take(dim, &indices)?));
    }

    if let Some((input, padding, value)) = node.pad() {
        let Some(input) = constant_input(graph, input)? else {
            return Ok(None);
        };
        return Ok(Some(input.pad(&padding, value)?));
    }

    if let Some((inputs, dim)) = node.concat() {
        let Some(inputs) = constant_input_list(graph, &inputs)? else {
            return Ok(None);
        };
        return Ok(Some(Tensor::concat(&inputs, dim)?));
    }

    if let Some((input, dim, indices)) = node.index_select() {
        let Some(input) = constant_input(graph, input)? else {
            return Ok(None);
        };
        let Some(indices) = constant_input(graph, indices)? else {
            return Ok(None);
        };
        return Ok(Some(input.index_select(dim, &indices)?));
    }

    if let Some((weight, indices)) = node.embedding_inputs() {
        let Some(weight) = constant_input(graph, weight)? else {
            return Ok(None);
        };
        let Some(indices) = constant_input(graph, indices)? else {
            return Ok(None);
        };
        return Ok(Some(weight.embedding(&indices)?));
    }
    Ok(None)
}

pub fn folded_small_view_constant(graph: &Graph, node: &NodeRef<'_>) -> Result<Option<Tensor>> {
    let Some(input) = node.view_input() else {
        return Ok(None);
    };
    let Some(input) = constant_input(graph, input)? else {
        return Ok(None);
    };
    node.folded_view_constant(&input)
}

fn constant_input_list(graph: &Graph, inputs: &[ValueId]) -> Result<Option<Vec<Tensor>>> {
    let mut tensors = Vec::with_capacity(inputs.len());
    for input in inputs {
        let Some(tensor) = constant_input(graph, *input)? else {
            return Ok(None);
        };
        tensors.push(tensor);
    }
    Ok(Some(tensors))
}

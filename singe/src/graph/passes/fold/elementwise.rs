use crate::{
    backend::{registry::Operation, spec::clamp_problem},
    dispatch::call_one,
    error::Result,
    graph::{
        Graph,
        node::NodeRef,
        passes::fold::{constant_input, constant_inputs},
    },
    tensor::Tensor,
};

/// Folds unary, binary, and scalar elementwise nodes by dispatching the node's own operation and attributes on its constant inputs.
pub fn folded_small_elementwise_constant(
    graph: &Graph,
    node: &NodeRef<'_>,
    operation: Option<Operation>,
) -> Result<Option<Tensor>> {
    let Some(operation) = operation.filter(|operation| {
        operation.is_unary_elementwise()
            || operation.is_binary_elementwise()
            || operation.is_scalar_elementwise()
    }) else {
        return Ok(None);
    };
    let mut inputs = Vec::new();
    for &input in node.inputs() {
        let Some(input) = constant_input(graph, input)? else {
            return Ok(None);
        };
        inputs.push(input);
    }

    let inputs = inputs.iter().collect::<Vec<_>>();
    call_one(operation, node.attributes(), &inputs).map(Some)
}

pub fn folded_small_matmul_constant(graph: &Graph, node: &NodeRef<'_>) -> Result<Option<Tensor>> {
    let Some(inputs) = node.matmul_inputs() else {
        return Ok(None);
    };
    let Some([lhs, rhs]) = constant_inputs(graph, &inputs)? else {
        return Ok(None);
    };
    Ok(Some(lhs.matmul(&rhs)?))
}

pub fn folded_small_misc_elementwise_constant(
    graph: &Graph,
    node: &NodeRef<'_>,
) -> Result<Option<Tensor>> {
    if let Some((input, min, max)) = node.clamp_bounds() {
        clamp_problem(node.output_shape(), min, max)?;
        let Some(input) = constant_input(graph, input)? else {
            return Ok(None);
        };
        return Ok(Some(input.clamp(min, max)?));
    }

    if let Some((a, b, t)) = node.lerp_inputs() {
        let Some(a) = constant_input(graph, a)? else {
            return Ok(None);
        };
        let Some(b) = constant_input(graph, b)? else {
            return Ok(None);
        };
        return Ok(Some(a.lerp(&b, t)?));
    }

    if let Some((condition, x, y)) = node.where_inputs() {
        let Some(condition) = constant_input(graph, condition)? else {
            return Ok(None);
        };
        let Some(x) = constant_input(graph, x)? else {
            return Ok(None);
        };
        let Some(y) = constant_input(graph, y)? else {
            return Ok(None);
        };
        return Ok(Some(Tensor::where_select(&condition, &x, &y)?));
    }
    Ok(None)
}

pub fn folded_small_dtype_transform_constant(
    graph: &Graph,
    node: &NodeRef<'_>,
    operation: Option<Operation>,
) -> Result<Option<Tensor>> {
    if let Some((input, data_type)) = node.cast() {
        return constant_input(graph, input)?
            .map(|input| input.cast(data_type))
            .transpose();
    }

    let Some(operation) = operation
        .filter(|operation| matches!(operation, Operation::Quantize | Operation::Dequantize))
    else {
        return Ok(None);
    };
    let [input, scale, zero_point] = node.inputs()[..] else {
        return Ok(None);
    };
    let (Some(input), Some(scale), Some(zero_point)) = (
        constant_input(graph, input)?,
        constant_input(graph, scale)?,
        constant_input(graph, zero_point)?,
    ) else {
        return Ok(None);
    };
    call_one(operation, node.attributes(), &[&input, &scale, &zero_point]).map(Some)
}

pub fn folded_small_reduction_constant(
    graph: &Graph,
    node: &NodeRef<'_>,
    operation: Option<Operation>,
) -> Result<Option<Tensor>> {
    let Some((input, dimensions, keepdim)) = node.reduction_shape_inputs() else {
        return Ok(None);
    };
    let Some(input) = constant_input(graph, input)? else {
        return Ok(None);
    };
    Ok(Some(match operation {
        Some(Operation::Sum) => input.sum(&dimensions, keepdim)?,
        Some(Operation::Mean) => input.mean(&dimensions, keepdim)?,
        Some(Operation::Max) => input.max_reduce(&dimensions, keepdim)?,
        Some(Operation::Min) => input.min_reduce(&dimensions, keepdim)?,
        Some(Operation::Product) => input.product(&dimensions, keepdim)?,
        Some(Operation::LogSumExp) => input.logsumexp(&dimensions, keepdim)?,
        _ => return Ok(None),
    }))
}

pub fn folded_small_index_reduction_constant(
    graph: &Graph,
    node: &NodeRef<'_>,
    operation: Option<Operation>,
) -> Result<Option<Tensor>> {
    let Some((input, dim)) = node.arg_reduction_input() else {
        return Ok(None);
    };
    let Some(input) = constant_input(graph, input)? else {
        return Ok(None);
    };
    Ok(Some(match operation {
        Some(Operation::ArgMax) => input.argmax(dim)?,
        Some(Operation::ArgMin) => input.argmin(dim)?,
        _ => return Ok(None),
    }))
}

pub fn folded_small_topk_constant(
    graph: &Graph,
    node: &NodeRef<'_>,
    operation: Option<Operation>,
) -> Result<Option<Tensor>> {
    let Some((input, k, dim, largest, sorted)) = node.topk() else {
        return Ok(None);
    };
    let Some(input) = constant_input(graph, input)? else {
        return Ok(None);
    };
    let (values, indices) = input.topk(k, dim, largest, sorted)?;
    Ok(Some(match operation {
        Some(Operation::TopKValues) => values,
        Some(Operation::TopKIndices) => indices,
        _ => return Ok(None),
    }))
}

pub fn folded_small_softmax_constant(
    graph: &Graph,
    node: &NodeRef<'_>,
    operation: Option<Operation>,
) -> Result<Option<Tensor>> {
    let Some((input, dim, _)) = node.softmax_fold_inputs() else {
        return Ok(None);
    };
    let Some(input) = constant_input(graph, input)? else {
        return Ok(None);
    };
    Ok(Some(match operation {
        Some(Operation::Softmax) => input.softmax(dim)?,
        Some(Operation::LogSoftmax) => input.log_softmax(dim)?,
        _ => return Ok(None),
    }))
}

use std::collections::HashMap;

use crate::{
    backend::registry::{LayerNormalizationAttributes, Operation, OperationAttributes},
    data_type::DataType,
    error::{Error, Result},
    graph::{
        Graph,
        ir::ValueId,
        node::{Node, NodeKind, NodeRef},
        passes::{fold::constant_input, remap::remapped_node},
    },
    scalar::Scalar,
};

// Rewrites primitive spellings of row operations as the operations, so imported and ported models reach the row kernels:
// - `x · rsqrt(mean(x², -1) + ε)` and `x / sqrt(mean(x², -1) + ε)` become `rms_norm(x)`;
// - the same of `x − mean(x, -1)` becomes `layer_norm(x)`;
// - `exp(a) / sum(exp(a), d)` becomes `softmax(a, d)`, and `a − log(sum(exp(a), d))` becomes `log_softmax(a, d)`, where `a` is `x` or `x − max(x, d)`.
// Weights and biases stay separate multiplications and additions, which fusion runs in the row kernel's epilogue.
// The matched intermediates stay, and the dead-node pruning after the passes removes those nothing else reads.
pub fn recognize_row_operations(graph: &Graph) -> Result<(Graph, Vec<(ValueId, ValueId)>)> {
    let mut id_map = HashMap::with_capacity(graph.node_count());
    let mut nodes = Vec::with_capacity(graph.node_count());
    let mut recognized = false;
    for index in 0..graph.node_count() {
        let id = ValueId::new(index);
        let node = graph.required_node(id)?;
        let new_id = match row_operation(graph, &node) {
            Some(row) => {
                recognized = true;
                let input = id_map
                    .get(&row.input())
                    .copied()
                    .ok_or_else(|| Error::missing_node("recognize row operations"))?;
                push(&mut nodes, node.with_kind(row.kind(input, &node)))
            }
            None => push(&mut nodes, remapped_node(&node, &id_map)?),
        };
        id_map.insert(id, new_id);
    }

    if !recognized {
        return Ok((graph.clone(), Vec::new()));
    }

    let remapped = graph.remapped_node_pairs(&id_map)?;
    Ok((Graph::from_retained_nodes(graph, &id_map, nodes)?, remapped))
}

fn push(nodes: &mut Vec<Node>, node: Node) -> ValueId {
    nodes.push(node);
    ValueId::new(nodes.len() - 1)
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum RowOperation {
    RmsNorm { input: ValueId, eps: f32 },
    LayerNorm { input: ValueId, eps: f32 },
    Softmax { input: ValueId, dim: i64 },
    LogSoftmax { input: ValueId, dim: i64 },
}

impl RowOperation {
    const fn input(self) -> ValueId {
        match self {
            Self::RmsNorm { input, .. }
            | Self::LayerNorm { input, .. }
            | Self::Softmax { input, .. }
            | Self::LogSoftmax { input, .. } => input,
        }
    }

    fn kind(self, input: ValueId, node: &NodeRef<'_>) -> NodeKind {
        let (operation, attributes) = match self {
            Self::RmsNorm { eps, .. } | Self::LayerNorm { eps, .. } => (
                if matches!(self, Self::RmsNorm { .. }) {
                    Operation::RmsNorm
                } else {
                    Operation::LayerNorm
                },
                OperationAttributes::LayerNormalization(LayerNormalizationAttributes {
                    normalized_shape: node
                        .output_shape()
                        .dimensions()
                        .last()
                        .copied()
                        .into_iter()
                        .collect(),
                    has_weight: false,
                    has_bias: false,
                    eps,
                }),
            ),
            Self::Softmax { dim, .. } => (Operation::Softmax, OperationAttributes::Dimension(dim)),
            Self::LogSoftmax { dim, .. } => {
                (Operation::LogSoftmax, OperationAttributes::Dimension(dim))
            }
        };
        NodeKind::operation_invocation(operation, attributes, vec![input])
    }
}

// The row operation `node` spells, when its input has the node's floating-point type and shape.
fn row_operation(graph: &Graph, node: &NodeRef<'_>) -> Option<RowOperation> {
    if !node.output_data_type().is_float() || node.output_shape().rank() == 0 {
        return None;
    }

    let row = normalization(graph, node).or_else(|| softmax(graph, node))?;
    let input = graph.node(row.input())?;
    (input.output_data_type() == node.output_data_type()
        && input.output_shape() == node.output_shape())
    .then_some(row)
}

// `x · rsqrt(v + ε)`, `x · (1 / sqrt(v + ε))`, or `x / sqrt(v + ε)` with `v = mean(x², -1)`, which is layer norm of `y` when `x = y − mean(y, -1)`.
fn normalization(graph: &Graph, node: &NodeRef<'_>) -> Option<RowOperation> {
    let [left, right] = binary(node)?;
    let (input, variance) = match node.operation()? {
        Operation::Mul => [(left, right), (right, left)]
            .into_iter()
            .find_map(|(input, scale)| Some((input, inverse_root(graph, scale)?)))?,
        Operation::Div => (left, root(graph, right)?),
        _ => return None,
    };
    let (variance, eps) = added_scalar(graph, variance)?;
    let squared = last_axis_mean(graph, variance)?;
    if square_of(graph, squared)? != input {
        return None;
    }

    let eps = f32_value(eps)?;
    let centered = graph.node(input)?;
    if centered.operation() == Some(Operation::Sub)
        && let Some([value, mean]) = binary(&centered)
        && last_axis_mean(graph, mean) == Some(value)
    {
        return Some(RowOperation::LayerNorm { input: value, eps });
    }
    Some(RowOperation::RmsNorm { input, eps })
}

// `exp(a) / sum(exp(a), d)` or `a − log(sum(exp(a), d))`, reading through `a = x − max(x, d)`.
fn softmax(graph: &Graph, node: &NodeRef<'_>) -> Option<RowOperation> {
    let [left, right] = binary(node)?;
    match node.operation()? {
        Operation::Div => {
            let (exponentials, dim) = kept_sum(graph, right)?;
            if exponentials != left {
                return None;
            }

            let shifted = unary(graph, left, Operation::Exp)?;
            let input = unshifted(graph, shifted, dim);
            Some(RowOperation::Softmax { input, dim })
        }
        Operation::Sub => {
            let (exponentials, dim) = kept_sum(graph, unary(graph, right, Operation::Log)?)?;
            if unary(graph, exponentials, Operation::Exp)? != left {
                return None;
            }

            let input = unshifted(graph, left, dim);
            Some(RowOperation::LogSoftmax { input, dim })
        }
        _ => None,
    }
}

// `x` when `shifted` is `x − max(x, dim)`, which the row kernels subtract themselves, and `shifted` otherwise.
fn unshifted(graph: &Graph, shifted: ValueId, dim: i64) -> ValueId {
    let subtracted = graph.node(shifted).and_then(|node| {
        if node.operation() != Some(Operation::Sub) {
            return None;
        }

        let [value, largest] = binary(&node)?;
        let (input, dimensions, keepdim) = reduced(graph, largest, Operation::Max)?;
        let rank = node.output_shape().rank();
        (input == value && keepdim && normalized(&dimensions, rank)? == [dim]).then_some(value)
    });
    subtracted.unwrap_or(shifted)
}

// The summed input and its normalized axis, when `id` is a sum over one axis that keeps it.
fn kept_sum(graph: &Graph, id: ValueId) -> Option<(ValueId, i64)> {
    let (input, dimensions, keepdim) = reduced(graph, id, Operation::Sum)?;
    let rank = graph.node(input)?.output_shape().rank();
    let &[dim] = normalized(&dimensions, rank)?.as_slice() else {
        return None;
    };
    keepdim.then_some((input, dim))
}

// The averaged input, when `id` is a mean over the last axis that keeps it.
fn last_axis_mean(graph: &Graph, id: ValueId) -> Option<ValueId> {
    let (input, dimensions, keepdim) = reduced(graph, id, Operation::Mean)?;
    let rank = graph.node(input)?.output_shape().rank();
    (keepdim && normalized(&dimensions, rank)? == [rank - 1]).then_some(input)
}

fn reduced(graph: &Graph, id: ValueId, operation: Operation) -> Option<(ValueId, Vec<i64>, bool)> {
    let node = graph.node(id)?;
    if node.operation() != Some(operation) {
        return None;
    }
    node.reduction_shape_inputs()
}

// Nonnegative axes, sorted; an empty list names every axis.
fn normalized(dimensions: &[i64], rank: i64) -> Option<Vec<i64>> {
    if dimensions.is_empty() {
        return Some((0..rank).collect());
    }

    let mut axes = dimensions
        .iter()
        .map(|&dim| {
            let axis = if dim < 0 { dim + rank } else { dim };
            (0..rank).contains(&axis).then_some(axis)
        })
        .collect::<Option<Vec<_>>>()?;
    axes.sort_unstable();
    Some(axes)
}

// The radicand of `1 / sqrt(v)`, spelled `rsqrt(v)`, `reciprocal(sqrt(v))`, or a division of one by `sqrt(v)`.
fn inverse_root(graph: &Graph, id: ValueId) -> Option<ValueId> {
    if let Some(radicand) = unary(graph, id, Operation::Rsqrt) {
        return Some(radicand);
    }

    if let Some(root) = unary(graph, id, Operation::Reciprocal) {
        return self::root(graph, root);
    }

    let node = graph.node(id)?;
    if let Some((root, numerator)) = node.scalar_operation(Operation::RDivScalar) {
        return (scalar_value(numerator) == Some(1.0))
            .then(|| self::root(graph, root))
            .flatten();
    }

    if node.operation() == Some(Operation::Div) {
        let [numerator, root] = binary(&node)?;
        return (constant_value(graph, numerator)? == 1.0)
            .then(|| self::root(graph, root))
            .flatten();
    }
    None
}

fn root(graph: &Graph, id: ValueId) -> Option<ValueId> {
    unary(graph, id, Operation::Sqrt)
}

// The base of `x · x`, `x²` as a scalar power, or `x²` as a power of a constant.
fn square_of(graph: &Graph, id: ValueId) -> Option<ValueId> {
    let node = graph.node(id)?;
    if let Some((base, exponent)) = node.scalar_operation(Operation::PowScalar) {
        return (scalar_value(exponent) == Some(2.0)).then_some(base);
    }

    let [left, right] = binary(&node)?;
    match node.operation()? {
        Operation::Mul if left == right => Some(left),
        Operation::Pow if constant_value(graph, right)? == 2.0 => Some(left),
        _ => None,
    }
}

// The other operand and the scalar of `x + c`, as a scalar operation or an addition of a one-element constant.
fn added_scalar(graph: &Graph, id: ValueId) -> Option<(ValueId, f64)> {
    let node = graph.node(id)?;
    if let Some((input, scalar)) = node.scalar_operation(Operation::AddScalar) {
        return Some((input, scalar_value(scalar)?));
    }

    if node.operation() != Some(Operation::Add) {
        return None;
    }

    let [left, right] = binary(&node)?;
    constant_value(graph, right)
        .map(|value| (left, value))
        .or_else(|| constant_value(graph, left).map(|value| (right, value)))
}

fn unary(graph: &Graph, id: ValueId, operation: Operation) -> Option<ValueId> {
    let node = graph.node(id)?;
    if node.operation() != Some(operation) {
        return None;
    }

    let &[input] = node.inputs() else {
        return None;
    };
    Some(input)
}

fn binary(node: &NodeRef<'_>) -> Option<[ValueId; 2]> {
    let &[left, right] = node.inputs() else {
        return None;
    };
    Some([left, right])
}

// The value of a one-element constant.
fn constant_value(graph: &Graph, id: ValueId) -> Option<f64> {
    let tensor = constant_input(graph, id).ok()??;
    if tensor.shape().element_count() != 1 {
        return None;
    }
    tensor
        .cast(DataType::F64)
        .ok()?
        .to_vec::<f64>()
        .ok()?
        .first()
        .copied()
}

#[expect(
    clippy::cast_precision_loss,
    reason = "the scalars these patterns compare are small exact values"
)]
const fn scalar_value(scalar: Scalar) -> Option<f64> {
    match scalar {
        Scalar::Float(value) => Some(value),
        Scalar::Signed(value) => Some(value as f64),
        Scalar::Unsigned(value) => Some(value as f64),
        Scalar::Bool(_) => None,
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "normalization epsilons are `f32` attributes"
)]
fn f32_value(value: f64) -> Option<f32> {
    (value.is_finite() && value > 0.0).then_some(value as f32)
}

#[cfg(test)]
mod tests;

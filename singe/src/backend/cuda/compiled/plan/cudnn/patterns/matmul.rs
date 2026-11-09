use std::{
    collections::{HashMap, HashSet},
    hash::RandomState,
};

use singe_cublas::lt::types::Epilogue;

use crate::{
    backend::cuda::{
        compiled::plan::cudnn::patterns::traversal::CudnnSubgraphMatcher,
        graph_value::CudaGraphValue,
    },
    error::Result,
    graph::ir::{Graph, ValueId},
};

/// Finds matrix multiplication followed by bias addition followed by `Relu` with ready operands.
/// Returns `None` when graph outputs, consumer topology, execution order, or operand readiness prevent fusion.
///
/// # Errors
///
/// Returns an error if referenced graph nodes lack valid primary results or producing operations, or a candidate consumer has an invalid operand list.
pub fn cudnn_matmul_add_relu_pattern(
    ir: &Graph,
    matmul: ValueId,
    matmul_inputs: [ValueId; 2],
    positions: &HashMap<ValueId, usize, RandomState>,
    users: &HashMap<ValueId, Vec<ValueId>, RandomState>,
    outputs: &HashSet<ValueId, RandomState>,
    cache: &HashMap<ValueId, CudaGraphValue, RandomState>,
) -> Result<Option<CudnnMatmulAddReluPattern>> {
    let matcher = CudnnSubgraphMatcher::new(ir, positions, users, outputs, cache);
    if matcher.is_output(matmul)? {
        return Ok(None);
    }

    let Some((add, bias)) = matcher.add_with_operand(matmul)? else {
        return Ok(None);
    };
    if matcher.is_output(add)? || !matcher.is_ready(bias) {
        return Ok(None);
    }

    let Some(relu) = matcher.relu_after(add)? else {
        return Ok(None);
    };
    let Some(add_index) = matcher.ordered_after(matmul, add) else {
        return Ok(None);
    };
    let Some(relu_index) = matcher.ordered_after(add, relu) else {
        return Ok(None);
    };
    if !matcher.all_ready(&matmul_inputs) {
        return Ok(None);
    }

    Ok(Some(CudnnMatmulAddReluPattern::new(
        add, relu, bias, add_index, relu_index,
    )))
}

/// Finds matrix multiplication followed by bias addition with ready operands.
/// Returns `None` when graph outputs, consumer topology, execution order, or operand readiness prevent fusion.
///
/// # Errors
///
/// Returns an error if referenced graph nodes lack valid primary results or producing operations, or a candidate consumer has an invalid operand list.
pub fn cudnn_matmul_add_pattern(
    ir: &Graph,
    matmul: ValueId,
    matmul_inputs: [ValueId; 2],
    positions: &HashMap<ValueId, usize, RandomState>,
    users: &HashMap<ValueId, Vec<ValueId>, RandomState>,
    outputs: &HashSet<ValueId, RandomState>,
    cache: &HashMap<ValueId, CudaGraphValue, RandomState>,
) -> Result<Option<CudnnMatmulAddPattern>> {
    let matcher = CudnnSubgraphMatcher::new(ir, positions, users, outputs, cache);
    if matcher.is_output(matmul)? {
        return Ok(None);
    }

    let Some((add, bias)) = matcher.add_with_operand(matmul)? else {
        return Ok(None);
    };
    if !matcher.is_ready(bias) {
        return Ok(None);
    }

    let Some(add_index) = matcher.ordered_after(matmul, add) else {
        return Ok(None);
    };
    if !matcher.all_ready(&matmul_inputs) {
        return Ok(None);
    }

    Ok(Some(CudnnMatmulAddPattern::new(add, bias, add_index)))
}

/// Matched matrix multiplication, bias addition, and `Relu` fusion nodes.
#[derive(Debug, Clone, Copy)]
pub struct CudnnMatmulAddReluPattern {
    add: ValueId,
    relu: ValueId,
    bias: ValueId,
    add_index: usize,
    relu_index: usize,
}

/// Matched matrix multiplication and bias addition fusion nodes.
#[derive(Debug, Clone, Copy)]
pub struct CudnnMatmulAddPattern {
    add: ValueId,
    bias: ValueId,
    add_index: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct CudnnMatmulActivationPattern {
    activation: ValueId,
    activation_index: usize,
    epilogue: Epilogue,
}

impl CudnnMatmulAddReluPattern {
    pub(crate) fn new(
        add: ValueId,
        relu: ValueId,
        bias: ValueId,
        add_index: usize,
        relu_index: usize,
    ) -> Self {
        Self {
            add,
            relu,
            bias,
            add_index,
            relu_index,
        }
    }

    /// Returns the bias addition node.
    #[must_use]
    pub fn add(&self) -> ValueId {
        self.add
    }

    /// Returns the `Relu` activation node.
    #[must_use]
    pub fn relu(&self) -> ValueId {
        self.relu
    }

    /// Returns the node that supplies the bias operand.
    #[must_use]
    pub fn bias(&self) -> ValueId {
        self.bias
    }

    /// Returns the bias addition node's position in the execution order.
    #[must_use]
    pub fn add_index(&self) -> usize {
        self.add_index
    }

    /// Returns the activation node's position in the execution order.
    #[must_use]
    pub fn relu_index(&self) -> usize {
        self.relu_index
    }
}

impl CudnnMatmulAddPattern {
    pub(crate) fn new(add: ValueId, bias: ValueId, add_index: usize) -> Self {
        Self {
            add,
            bias,
            add_index,
        }
    }

    /// Returns the bias addition node.
    #[must_use]
    pub fn add(&self) -> ValueId {
        self.add
    }

    /// Returns the node that supplies the bias operand.
    #[must_use]
    pub fn bias(&self) -> ValueId {
        self.bias
    }

    /// Returns the bias addition node's position in the execution order.
    #[must_use]
    pub fn add_index(&self) -> usize {
        self.add_index
    }
}

impl CudnnMatmulActivationPattern {
    #[must_use]
    pub fn activation(&self) -> ValueId {
        self.activation
    }

    #[must_use]
    pub fn activation_index(&self) -> usize {
        self.activation_index
    }

    #[must_use]
    pub fn epilogue(&self) -> Epilogue {
        self.epilogue
    }
}

/// Finds a single-user matrix multiplication followed by `Relu` or tanh-approximate GELU.
/// Returns `None` when the matrix result is a graph output, the activation is absent, or an operand is not ready.
///
/// # Errors
///
/// Returns an error if referenced graph nodes lack valid primary results or producing operations.
pub fn cudnn_matmul_activation_pattern(
    ir: &Graph,
    matmul: ValueId,
    matmul_inputs: [ValueId; 2],
    positions: &HashMap<ValueId, usize, RandomState>,
    users: &HashMap<ValueId, Vec<ValueId>, RandomState>,
    outputs: &HashSet<ValueId, RandomState>,
    cache: &HashMap<ValueId, CudaGraphValue, RandomState>,
) -> Result<Option<CudnnMatmulActivationPattern>> {
    let matcher = CudnnSubgraphMatcher::new(ir, positions, users, outputs, cache);
    if matcher.is_output(matmul)? {
        return Ok(None);
    }

    let (activation, epilogue) = if let Some(relu) = matcher.relu_after(matmul)? {
        (relu, Epilogue::Relu)
    } else if let Some(gelu) = matcher.gelu_after(matmul)? {
        (gelu, Epilogue::Gelu)
    } else {
        return Ok(None);
    };
    let Some(activation_index) = matcher.ordered_after(matmul, activation) else {
        return Ok(None);
    };
    if !matcher.all_ready(&matmul_inputs) {
        return Ok(None);
    }
    Ok(Some(CudnnMatmulActivationPattern {
        activation,
        activation_index,
        epilogue,
    }))
}

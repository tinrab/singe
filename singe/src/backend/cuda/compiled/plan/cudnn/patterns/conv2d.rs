use std::{
    collections::{HashMap, HashSet},
    hash::RandomState,
};

use crate::{
    backend::cuda::{
        compiled::plan::cudnn::patterns::traversal::CudnnSubgraphMatcher,
        graph_value::CudaGraphValue,
    },
    error::Result,
    graph::ir::{Graph, ValueId},
};

/// A matched convolution with bias followed by a `Relu` activation.
#[derive(Debug, Clone, Copy)]
pub struct CudnnConv2dBiasReluPattern {
    relu: ValueId,
    bias: ValueId,
    relu_index: usize,
}

impl CudnnConv2dBiasReluPattern {
    /// Returns the activation node.
    #[must_use]
    pub fn relu(&self) -> ValueId {
        self.relu
    }

    /// Returns the convolution bias operand node.
    #[must_use]
    pub fn bias(&self) -> ValueId {
        self.bias
    }

    /// Returns the activation node's position in the execution order.
    #[must_use]
    pub fn relu_index(&self) -> usize {
        self.relu_index
    }
}

/// Finds a single-user convolution-to-`Relu` pattern with ready input, filter, and bias values.
/// Returns `None` when the convolution is a graph output, inputs are not ready, or the consumer or execution order cannot be fused.
///
/// # Errors
///
/// Returns an error if a referenced node has no valid primary result or producing operation, or a candidate activation has an invalid operand list.
pub fn cudnn_conv2d_bias_relu_pattern(
    ir: &Graph,
    conv2d: ValueId,
    conv2d_inputs: [ValueId; 3],
    positions: &HashMap<ValueId, usize, RandomState>,
    users: &HashMap<ValueId, Vec<ValueId>, RandomState>,
    outputs: &HashSet<ValueId, RandomState>,
    cache: &HashMap<ValueId, CudaGraphValue, RandomState>,
) -> Result<Option<CudnnConv2dBiasReluPattern>> {
    let matcher = CudnnSubgraphMatcher::new(ir, positions, users, outputs, cache);
    if matcher.is_output(conv2d)? || !matcher.all_ready(&conv2d_inputs) {
        return Ok(None);
    }

    let Some(relu) = matcher.relu_after(conv2d)? else {
        return Ok(None);
    };
    let Some(relu_index) = matcher.ordered_after(conv2d, relu) else {
        return Ok(None);
    };
    Ok(Some(CudnnConv2dBiasReluPattern {
        relu,
        bias: conv2d_inputs[2],
        relu_index,
    }))
}

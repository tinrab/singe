use crate::{
    backend::registry::{
        BatchNormalizationAttributes, ContractionAttributes, LayerNormalizationAttributes,
        Operation, OperationAttributes, ScaledDotProductAttentionAttributes,
    },
    graph::{
        ir::ValueId,
        operation::{GraphOperationInputs, GraphOperationOutput, GraphOperationRequest},
    },
};

impl GraphOperationRequest {
    /// Creates an attention request with ordered query, key, and value operands and optional scale and causal masking.
    /// The request stores operands and attributes without validating them; graph insertion performs logical inference.
    #[must_use]
    pub fn scaled_dot_product_attention(
        name: Option<String>,
        query: ValueId,
        key: ValueId,
        value: ValueId,
        scale: Option<f32>,
        causal: bool,
        output: GraphOperationOutput,
    ) -> Self {
        Self {
            name,
            inputs: GraphOperationInputs::new(
                Operation::ScaledDotProductAttention,
                OperationAttributes::ScaledDotProductAttention(
                    ScaledDotProductAttentionAttributes {
                        scale,
                        causal,
                        window: None,
                        softcap: None,
                        masked: false,
                        sinks: false,
                        key_lengths: false,
                    },
                ),
                vec![query, key, value],
            ),
            output,
        }
    }

    /// Creates a rotary embedding request with input, cosine, and sine operands.
    /// The request stores operands and attributes without validating them; graph insertion performs logical inference.
    #[must_use]
    pub fn rotary_embedding(
        name: Option<String>,
        input: ValueId,
        cos: ValueId,
        sin: ValueId,
        output: GraphOperationOutput,
    ) -> Self {
        Self {
            name,
            inputs: GraphOperationInputs::new(
                Operation::RotaryEmbedding,
                OperationAttributes::None,
                vec![input, cos, sin],
            ),
            output,
        }
    }
}

impl GraphOperationRequest {
    /// Creates a scaled tensor contraction request with three operands and explicit mode labels.
    /// `alpha` scales the contraction of A and B, and `beta` scales C in the output mode order.
    /// The request stores operands and attributes without validating them; graph insertion performs logical inference.
    #[must_use]
    pub fn contraction(
        name: Option<String>,
        a: ValueId,
        a_mode: Vec<i64>,
        b: ValueId,
        b_mode: Vec<i64>,
        c: ValueId,
        c_mode: Vec<i64>,
        d_mode: Vec<i64>,
        alpha: f32,
        beta: f32,
        output: GraphOperationOutput,
    ) -> Self {
        Self {
            name,
            inputs: GraphOperationInputs::new(
                Operation::Contraction,
                OperationAttributes::Contraction(ContractionAttributes {
                    a_mode,
                    b_mode,
                    c_mode,
                    d_mode,
                    alpha,
                    beta,
                }),
                vec![a, b, c],
            ),
            output,
        }
    }
}

impl GraphOperationRequest {
    /// Creates a batch normalization request with optional affine and running-statistic operands.
    /// Present operands follow input, weight, bias, running mean, and running variance order.
    /// The request stores operands and attributes without validating them; graph insertion performs logical inference.
    #[must_use]
    pub fn batch_normalization(
        name: Option<String>,
        input: ValueId,
        weight: Option<ValueId>,
        bias: Option<ValueId>,
        running_mean: Option<ValueId>,
        running_var: Option<ValueId>,
        training: bool,
        momentum: f32,
        eps: f32,
        output: GraphOperationOutput,
    ) -> Self {
        Self {
            name,
            inputs: GraphOperationInputs::new(
                Operation::BatchNorm,
                OperationAttributes::BatchNormalization(BatchNormalizationAttributes {
                    has_weight: weight.is_some(),
                    has_bias: bias.is_some(),
                    has_running_mean: running_mean.is_some(),
                    has_running_var: running_var.is_some(),
                    training,
                    momentum,
                    eps,
                }),
                [Some(input), weight, bias, running_mean, running_var]
                    .into_iter()
                    .flatten()
                    .collect(),
            ),
            output,
        }
    }

    /// Creates a normalization request with input, optional weight, and optional bias operands.
    /// `operation` selects the normalization, and `normalized_shape` supplies the trailing normalization dimensions.
    /// The request stores operands and attributes without validating them; graph insertion performs logical inference.
    #[must_use]
    pub fn layer_normalization(
        name: Option<String>,
        operation: Operation,
        input: ValueId,
        normalized_shape: Vec<i64>,
        weight: Option<ValueId>,
        bias: Option<ValueId>,
        eps: f32,
        output: GraphOperationOutput,
    ) -> Self {
        Self {
            name,
            inputs: GraphOperationInputs::new(
                operation,
                OperationAttributes::LayerNormalization(LayerNormalizationAttributes {
                    normalized_shape,
                    has_weight: weight.is_some(),
                    has_bias: bias.is_some(),
                    eps,
                }),
                [Some(input), weight, bias].into_iter().flatten().collect(),
            ),
            output,
        }
    }
}

use crate::{
    backend::registry::{Operation, OperationAttributeKind, ScaledDotProductAttentionAttributes},
    graph::{
        ir::{ValueId, operation_attributes},
        node::NodeRef,
    },
};

impl NodeRef<'_> {
    /// Returns the query, key, value, and optional mask inputs and the attributes of scaled dot-product attention nodes.
    pub(crate) fn scaled_dot_product_attention_parameters(
        &self,
    ) -> Option<(&[ValueId], ScaledDotProductAttentionAttributes)> {
        if self.operation() != Some(Operation::ScaledDotProductAttention) {
            return None;
        }

        let attributes = operation_attributes(self)
            .scaled_dot_product_attention()
            .copied()?;
        Some((self.inputs(), attributes))
    }
}

impl NodeRef<'_> {
    /// Returns weight and indices inputs for embedding nodes.
    pub(crate) fn embedding_inputs(&self) -> Option<(ValueId, ValueId)> {
        if self.operation() != Some(Operation::Embedding) {
            return None;
        }

        let inputs = self.inputs();
        Some((*inputs.first()?, *inputs.get(1)?))
    }

    /// Returns input, axis, and indices input for index-select nodes.
    pub(crate) fn index_select(&self) -> Option<(ValueId, i64, ValueId)> {
        if self.operation() != Some(Operation::IndexSelect) {
            return None;
        }

        let inputs = self.inputs();
        Some((
            *inputs.first()?,
            operation_attributes(self).index_select_dimension()?,
            *inputs.get(1)?,
        ))
    }

    /// Returns input and trigonometric tables for rotary embedding nodes.
    pub(crate) fn rotary_embedding_inputs(&self) -> Option<(ValueId, ValueId, ValueId)> {
        if self.operation() != Some(Operation::RotaryEmbedding) {
            return None;
        }

        let inputs = self.inputs();
        Some((*inputs.first()?, *inputs.get(1)?, *inputs.get(2)?))
    }
}

impl NodeRef<'_> {
    /// Returns input and optional parameter inputs for batch-normalization nodes.
    pub(crate) fn batch_normalization_inputs(&self) -> Option<BatchNormalizationInputs> {
        if self.operation() != Some(Operation::BatchNorm) {
            return None;
        }

        let inputs = self.inputs();
        let attributes = operation_attributes(self).batch_normalization().copied()?;
        let input = *inputs.first()?;
        let mut next_input = 1;
        let weight = if attributes.has_weight {
            let weight = *inputs.get(next_input)?;
            next_input += 1;
            Some(weight)
        } else {
            None
        };
        let bias = if attributes.has_bias {
            let bias = *inputs.get(next_input)?;
            next_input += 1;
            Some(bias)
        } else {
            None
        };
        let running_mean = if attributes.has_running_mean {
            let running_mean = *inputs.get(next_input)?;
            next_input += 1;
            Some(running_mean)
        } else {
            None
        };
        let running_var = if attributes.has_running_var {
            Some(*inputs.get(next_input)?)
        } else {
            None
        };
        Some((input, weight, bias, running_mean, running_var))
    }

    /// Returns input, optional parameters, and scalar attributes for batch-normalization nodes.
    pub(crate) fn batch_normalization_parameters(&self) -> Option<BatchNormalizationParameters> {
        let (input, weight, bias, running_mean, running_var) = self.batch_normalization_inputs()?;
        let attributes = operation_attributes(self).batch_normalization().copied()?;
        Some((
            input,
            weight,
            bias,
            running_mean,
            running_var,
            attributes.training,
            attributes.momentum,
            attributes.eps,
        ))
    }
}

impl NodeRef<'_> {
    /// Returns input, normalized shape, parameters, and epsilon for normalization nodes.
    pub(crate) fn normalization_parameters(&self) -> Option<NormalizationParameters> {
        if !self
            .operation()?
            .has_attribute_kind(OperationAttributeKind::LayerNormalization)
        {
            return None;
        }

        let inputs = self.inputs();
        let operation_attributes = operation_attributes(self);
        let attributes = operation_attributes.layer_normalization()?;
        let input = *inputs.first()?;
        let mut next_input = 1;
        let weight = if attributes.has_weight {
            let weight = *inputs.get(next_input)?;
            next_input += 1;
            Some(weight)
        } else {
            None
        };
        let bias = if attributes.has_bias {
            Some(*inputs.get(next_input)?)
        } else {
            None
        };
        Some((
            input,
            attributes.normalized_shape.clone(),
            weight,
            bias,
            attributes.eps,
        ))
    }
}

impl NodeRef<'_> {
    /// Returns input and full parameters for 2D pooling nodes.
    pub(crate) fn pool2d_parameters(&self) -> Option<Pool2DParameters> {
        if !self
            .operation()?
            .has_attribute_kind(OperationAttributeKind::Pool2d)
        {
            return None;
        }

        let input = self.inputs().first().copied()?;
        let attributes = operation_attributes(self).pool2d().copied()?;
        let count_include_pad = attributes.count_include_pad;
        Some((
            input,
            attributes.kernel_size,
            attributes.stride,
            attributes.padding,
            attributes.dilation,
            attributes.ceil_mode,
            count_include_pad.is_some(),
            count_include_pad.unwrap_or(false),
        ))
    }

    /// Returns inputs and shape attributes for 2D convolution nodes.
    pub(crate) fn conv2d(&self) -> Option<Conv2DParameters> {
        if self.operation() != Some(Operation::Conv2d) {
            return None;
        }

        let inputs = self.inputs();
        let attributes = operation_attributes(self).conv2d().copied()?;
        Some((
            *inputs.first()?,
            *inputs.get(1)?,
            attributes
                .has_bias
                .then(|| inputs.get(2).copied())
                .flatten(),
            attributes.stride,
            attributes.padding,
            attributes.dilation,
            attributes.groups,
        ))
    }
}

type BatchNormalizationInputs = (
    ValueId,
    Option<ValueId>,
    Option<ValueId>,
    Option<ValueId>,
    Option<ValueId>,
);

type BatchNormalizationParameters = (
    ValueId,
    Option<ValueId>,
    Option<ValueId>,
    Option<ValueId>,
    Option<ValueId>,
    bool,
    f32,
    f32,
);

type NormalizationParameters = (ValueId, Vec<i64>, Option<ValueId>, Option<ValueId>, f32);

type Pool2DParameters = (
    ValueId,
    (i64, i64),
    (i64, i64),
    (i64, i64, i64, i64),
    (i64, i64),
    bool,
    bool,
    bool,
);

type Conv2DParameters = (
    ValueId,
    ValueId,
    Option<ValueId>,
    (i64, i64),
    (i64, i64, i64, i64),
    (i64, i64),
    i64,
);

use crate::{
    backend::registry::Operation,
    error::Result,
    graph::{
        Graph,
        node::NodeRef,
        passes::fold::{OptionalConstantInput, constant_input, optional_constant_input},
    },
    tensor::{AttentionOptions, Tensor},
};

pub fn folded_small_neural_constant(
    graph: &Graph,
    node: &NodeRef<'_>,
    operation: Option<Operation>,
) -> Result<Option<Tensor>> {
    if let Some(tensor) = folded_small_normalization_constant(graph, node, operation)? {
        return Ok(Some(tensor));
    }

    if let Some(tensor) = folded_small_attention_constant(graph, node)? {
        return Ok(Some(tensor));
    }
    folded_small_spatial_constant(graph, node, operation)
}

pub fn folded_small_normalization_constant(
    graph: &Graph,
    node: &NodeRef<'_>,
    operation: Option<Operation>,
) -> Result<Option<Tensor>> {
    if let Some((input, weight, bias, running_mean, running_var, training, momentum, eps)) =
        node.batch_normalization_parameters()
    {
        let Some(input) = constant_input(graph, input)? else {
            return Ok(None);
        };
        let OptionalConstantInput::Constant(weight) = optional_constant_input(graph, weight)?
        else {
            return Ok(None);
        };
        let OptionalConstantInput::Constant(bias) = optional_constant_input(graph, bias)? else {
            return Ok(None);
        };
        let OptionalConstantInput::Constant(running_mean) =
            optional_constant_input(graph, running_mean)?
        else {
            return Ok(None);
        };
        let OptionalConstantInput::Constant(running_var) =
            optional_constant_input(graph, running_var)?
        else {
            return Ok(None);
        };
        return Ok(Some(input.batch_norm(
            weight.as_ref(),
            bias.as_ref(),
            running_mean.as_ref(),
            running_var.as_ref(),
            training,
            momentum,
            eps,
        )?));
    }

    if let Some((input, normalized_shape, weight, bias, eps)) = node.normalization_parameters() {
        let Some(input) = constant_input(graph, input)? else {
            return Ok(None);
        };
        let OptionalConstantInput::Constant(weight) = optional_constant_input(graph, weight)?
        else {
            return Ok(None);
        };
        let OptionalConstantInput::Constant(bias) = optional_constant_input(graph, bias)? else {
            return Ok(None);
        };
        return Ok(Some(match operation {
            Some(Operation::LayerNorm) => {
                input.layer_norm(&normalized_shape, weight.as_ref(), bias.as_ref(), eps)?
            }
            Some(Operation::RmsNorm) => {
                input.rms_norm(&normalized_shape, weight.as_ref(), bias.as_ref(), eps)?
            }
            _ => return Ok(None),
        }));
    }
    Ok(None)
}

pub fn folded_small_spatial_constant(
    graph: &Graph,
    node: &NodeRef<'_>,
    operation: Option<Operation>,
) -> Result<Option<Tensor>> {
    if let Some((
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        is_avg_pool,
        count_include_pad,
    )) = node.pool2d_parameters()
    {
        let Some(input) = constant_input(graph, input)? else {
            return Ok(None);
        };
        return Ok(Some(match operation {
            Some(Operation::MaxPool2d) => {
                input.max_pool2d(kernel_size, stride, padding, dilation, ceil_mode)?
            }
            Some(Operation::AvgPool2d) if is_avg_pool => input.avg_pool2d(
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
                count_include_pad,
            )?,
            _ => return Ok(None),
        }));
    }
    Ok(None)
}

pub fn folded_small_attention_constant(
    graph: &Graph,
    node: &NodeRef<'_>,
) -> Result<Option<Tensor>> {
    if let Some((inputs, attention)) = node.scaled_dot_product_attention_parameters() {
        let mut tensors = Vec::with_capacity(inputs.len());
        for &input in inputs {
            let Some(tensor) = constant_input(graph, input)? else {
                return Ok(None);
            };
            tensors.push(tensor);
        }

        let [query, key, value, rest @ ..] = tensors.as_slice() else {
            return Ok(None);
        };
        let mask = rest.first().filter(|_| attention.masked);
        let sinks = rest
            .get(usize::from(attention.masked))
            .filter(|_| attention.sinks);
        if attention.key_lengths {
            return Ok(None);
        }

        let options = AttentionOptions {
            scale: attention.scale,
            causal: attention.causal,
            window: attention.window,
            softcap: attention.softcap,
        };
        return Ok(Some(query.attend(key, value, mask, sinks, options)?));
    }
    Ok(None)
}

pub fn folded_small_backend_structural_constant(
    graph: &Graph,
    node: &NodeRef<'_>,
) -> Result<Option<Tensor>> {
    if let Some((input, weight, bias, stride, padding, dilation, groups)) = node.conv2d() {
        let Some(input) = constant_input(graph, input)? else {
            return Ok(None);
        };
        let Some(weight) = constant_input(graph, weight)? else {
            return Ok(None);
        };
        let OptionalConstantInput::Constant(bias) = optional_constant_input(graph, bias)? else {
            return Ok(None);
        };
        return Ok(Some(input.convolution2d(
            &weight,
            bias.as_ref(),
            stride,
            padding,
            dilation,
            groups,
        )?));
    }

    if let Some((a, a_mode, b, b_mode, c, c_mode, d_mode, alpha, beta)) = node.contraction() {
        let Some(a) = constant_input(graph, a)? else {
            return Ok(None);
        };
        let Some(b) = constant_input(graph, b)? else {
            return Ok(None);
        };
        let Some(c) = constant_input(graph, c)? else {
            return Ok(None);
        };
        return Ok(Some(a.contraction(
            &a_mode, &b, &b_mode, &c, &c_mode, &d_mode, alpha, beta,
        )?));
    }

    if let Some((input, cos, sin)) = node.rotary_embedding_inputs() {
        let Some(input) = constant_input(graph, input)? else {
            return Ok(None);
        };
        let Some(cos) = constant_input(graph, cos)? else {
            return Ok(None);
        };
        let Some(sin) = constant_input(graph, sin)? else {
            return Ok(None);
        };
        return Ok(Some(input.rotary_embedding(&cos, &sin)?));
    }

    if let Some((input, repeats)) = node.repeat_kv() {
        let Some(input) = constant_input(graph, input)? else {
            return Ok(None);
        };
        return Ok(Some(input.repeat_kv(repeats)?));
    }
    Ok(None)
}

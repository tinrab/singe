use crate::{
    backend::{
        registry::{
            OperationAttributes, OperationInference,
            infer::{
                InferenceContext, elementwise::check_quantization_parameters, require_indices,
            },
        },
        spec,
    },
    data_type::DataType,
    error::{Error, Result},
    shape::Shape,
    value::ValueSpec,
};

/// Multiplies matrices of one data type with broadcast batch dimensions.
pub fn matmul(context: &InferenceContext<'_>) -> Result<OperationInference> {
    context.require_same_types()?;
    let problem = spec::matmul_problem(
        context.shape(),
        context.input(1)?.shape(),
        "operation matmul",
    )?;
    // Boolean products have no meaning, but a product without terms is all false, as PyTorch computes it without checking the type.
    if context.first().data_type() == DataType::Bool && !problem.is_zero_work() {
        return Err(Error::unsupported_data_type("matmul of boolean tensors"));
    }
    Ok(context.same_type(&problem.output_shape))
}

/// Multiplies quantized matrices into an `F32` result.
pub fn quantized_matmul(context: &InferenceContext<'_>) -> Result<OperationInference> {
    for (scale, zero_point) in [(2, 3), (4, 5)] {
        check_quantization_parameters(
            context.input(scale - 2)?.shape(),
            context.input(scale)?,
            context.input(zero_point)?,
            None,
            None,
        )?;
    }

    let problem = spec::matmul_problem(
        context.shape(),
        context.input(1)?.shape(),
        "operation quantized matmul",
    )?;
    Ok(context.finish(&problem.output_shape, DataType::F32))
}

/// Contracts operands of one data type by their mode labels.
pub fn contraction(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let contraction = context.attributes(OperationAttributes::contraction)?;
    context.require_same_types()?;
    let problem = spec::contraction_problem(
        context.shape(),
        &contraction.a_mode,
        context.input(1)?.shape(),
        &contraction.b_mode,
        context.input(2)?.shape(),
        &contraction.c_mode,
        &contraction.d_mode,
    )?;
    Ok(context.same_type(&problem.output_shape))
}

/// Normalizes along one dimension.
pub fn softmax(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let dim = context.value(OperationAttributes::dimension)?;
    // PyTorch normalizes only floating values, but returns an empty tensor of any type unchanged.
    if !matches!(
        context.first().data_type(),
        DataType::F16 | DataType::Bf16 | DataType::F32 | DataType::F64
    ) && context.shape().element_count() != 0
    {
        return Err(Error::unsupported_data_type(
            "softmax of non-floating values",
        ));
    }

    // A scalar is one slice of one element.
    if context.shape().rank() == 0 {
        return if matches!(dim, 0 | -1) {
            Ok(context.same_type(context.shape()))
        } else {
            Err(Error::DimensionOutOfBounds)
        };
    }
    Ok(context.same_type(&spec::softmax_problem(context.shape(), dim)?.output_shape))
}

/// Looks up rows of the weight with `I64` indices.
pub fn embedding(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let indices = context.input(1)?;
    require_indices(indices)?;
    let problem = spec::embedding_problem(context.shape(), indices.shape(), "operation embedding")?;
    Ok(context.same_type(&problem.output_shape))
}

/// Rotates the input with cosine and sine tables.
pub fn rotary_embedding(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let problem = spec::rotary_embedding_problem(
        context.shape(),
        context.input(1)?.shape(),
        context.input(2)?.shape(),
        "operation rotary embedding",
    )?;
    Ok(context.same_type(&problem.output_shape))
}

/// Convolves the input with a weight and an optional bias.
pub fn conv2d(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let conv = context.attributes(OperationAttributes::conv2d)?;
    let problem = spec::conv2d_problem(
        context.shape(),
        context.input(1)?.shape(),
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        "operation conv2d",
    )?;
    spec::validate_conv2d_bias(
        &problem,
        context.optional_shape(2, conv.has_bias)?,
        "operation conv2d",
    )?;
    Ok(context.same_type(&problem.output_shape))
}

// Convolves three spatial axes.
pub fn conv3d(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let conv = context.attributes(OperationAttributes::conv3d)?;
    let output = spec::ConvolutionWindow::conv3d(conv)
        .output_shape(context.shape(), context.input(1)?.shape())?;
    if let Some(bias) = context.optional_shape(2, conv.has_bias)?
        && bias.dimensions() != [output.dimensions()[1]]
    {
        return Err(Error::invalid_argument(
            "operation conv3d needs one bias per output channel",
        ));
    }
    Ok(context.same_type(&output))
}

/// Reduces spatial windows to their maximum.
pub fn max_pool2d(context: &InferenceContext<'_>) -> Result<OperationInference> {
    pool2d(context, false)
}

/// Averages spatial windows.
pub fn avg_pool2d(context: &InferenceContext<'_>) -> Result<OperationInference> {
    pool2d(context, true)
}

fn pool2d(context: &InferenceContext<'_>, average: bool) -> Result<OperationInference> {
    let pool = context.attributes(OperationAttributes::pool2d)?;
    let problem = spec::pool2d_problem(
        context.shape(),
        pool.kernel_size,
        pool.stride,
        pool.padding,
        pool.dilation,
        pool.ceil_mode,
        average,
        "operation pool2d",
    )?;
    Ok(context.same_type(&problem.output_shape))
}

/// Normalizes channels with running statistics and optional affine operands.
pub fn batch_norm(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let norm = context.attributes(OperationAttributes::batch_normalization)?;
    // Training updates running statistics in place, which a call does not write.
    if norm.training && (norm.has_running_mean || norm.has_running_var) {
        return Err(Error::unsupported_operation(
            "batch norm running-state mutation",
        ));
    }

    let weight = context.optional_shape(1, norm.has_weight)?;
    let bias_index = 1 + usize::from(norm.has_weight);
    let bias = context.optional_shape(bias_index, norm.has_bias)?;
    let mean_index = bias_index + usize::from(norm.has_bias);
    let mean = context.optional_shape(mean_index, norm.has_running_mean)?;
    let variance = context.optional_shape(
        mean_index + usize::from(norm.has_running_mean),
        norm.has_running_var,
    )?;
    let problem = spec::batch_norm_problem(
        context.shape(),
        weight,
        bias,
        mean,
        variance,
        "operation batch norm",
    )?;
    Ok(context.same_type(&problem.output_shape))
}

/// Normalizes trailing dimensions with optional affine operands.
pub fn layer_norm(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let norm = context.attributes(OperationAttributes::layer_normalization)?;
    let problem = spec::normalization_problem(
        context.shape(),
        &norm.normalized_shape,
        "operation normalization",
    )?;
    spec::validate_normalization_parameters(
        &problem,
        context.optional_shape(1, norm.has_weight)?,
        context.optional_shape(1 + usize::from(norm.has_weight), norm.has_bias)?,
        "operation normalization",
    )?;
    Ok(context.same_type(&problem.output_shape))
}

/// Attends from queries to keys and values.
pub fn attention(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let attention = context.attributes(OperationAttributes::scaled_dot_product_attention)?;
    let problem = spec::attention_problem(
        context.shape(),
        context.input(1)?.shape(),
        context.input(2)?.shape(),
        "operation attention",
    )?;
    let query = context.input(0)?.data_type();
    let same_type = |operand: &ValueSpec, name: &'static str| {
        if operand.data_type() == query {
            Ok(())
        } else {
            Err(Error::data_type_mismatch(name, query, operand.data_type()))
        }
    };
    if attention.masked {
        let mask = context.input(3)?;
        spec::attention_mask_shape(&problem, mask.shape(), "operation attention")?;
        same_type(mask, "operation attention mask")?;
    }

    // One sink logit per query head.
    if attention.sinks {
        let sinks = context.input(3 + usize::from(attention.masked))?;
        if sinks.shape().dimensions() != [problem.heads] {
            return Err(Error::shape_mismatch(
                "operation attention sinks",
                vec![problem.heads],
                sinks.shape().dimensions().to_vec(),
            ));
        }

        same_type(sinks, "operation attention sinks")?;
    }

    // One count of keys per batch.
    if attention.key_lengths {
        let lengths =
            context.input(3 + usize::from(attention.masked) + usize::from(attention.sinks))?;
        if lengths.shape().dimensions() != [problem.batch] {
            return Err(Error::shape_mismatch(
                "operation attention key lengths",
                vec![problem.batch],
                lengths.shape().dimensions().to_vec(),
            ));
        }

        if lengths.data_type() != DataType::I32 {
            return Err(Error::data_type_mismatch(
                "operation attention key lengths",
                DataType::I32,
                lengths.data_type(),
            ));
        }
    }
    Ok(context.same_type(&problem.output_shape))
}

/// Attends ragged queries over a paged key and value cache, keeping the queries' shape and data type.
pub fn paged_attention(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let (output_shape, lse_shape) = paged_attention_shapes(context)?;
    let mut inference = context.same_type(&output_shape);
    inference.outputs[1] = ValueSpec::new(DataType::F32, lse_shape);
    Ok(inference)
}

// The output alone, as graph capture records paged attention's first result.
pub fn paged_attention_output(context: &InferenceContext<'_>) -> Result<OperationInference> {
    Ok(context.same_type(&paged_attention_shapes(context)?.0))
}

// The log-sum-exp alone, as graph capture records paged attention's second result; its role makes it `F32`.
pub fn paged_attention_lse(context: &InferenceContext<'_>) -> Result<OperationInference> {
    Ok(context.same_type(&paged_attention_shapes(context)?.1))
}

fn paged_attention_shapes(context: &InferenceContext<'_>) -> Result<(Shape, Shape)> {
    let attention = context.attributes(OperationAttributes::scaled_dot_product_attention)?;
    if attention.masked || attention.sinks || attention.key_lengths {
        return Err(Error::unsupported_operation(
            "paged attention with a mask, sinks, or key lengths",
        ));
    }

    let index = |position: usize| -> Result<(&Shape, DataType)> {
        let input = context.input(position)?;
        Ok((input.shape(), input.data_type()))
    };
    let problem = spec::paged_attention_problem(
        context.shape(),
        context.input(1)?.shape(),
        context.input(2)?.shape(),
        [index(3)?, index(4)?, index(5)?, index(6)?],
        "operation paged attention",
    )?;
    let query = context.input(0)?.data_type();
    for position in [1, 2] {
        let cache = context.input(position)?.data_type();
        if cache != query {
            return Err(Error::data_type_mismatch(
                "operation paged attention cache",
                query,
                cache,
            ));
        }
    }

    let lse_shape = Shape::create([problem.tokens, problem.heads])?;
    Ok((problem.output_shape, lse_shape))
}

// Checks a step's query offsets `[S + 1]`, real token counts `[S]`, and positions `[T]` against `sequences` sequences of `tokens` rows, from operand `first` on.
fn check_sequence_spans(
    context: &InferenceContext<'_>,
    first: usize,
    sequences: i64,
    tokens: i64,
    label: &'static str,
) -> Result<()> {
    for (index, length) in [
        (first, sequences + 1),
        (first + 1, sequences),
        (first + 2, tokens),
    ] {
        let spans = context.input(index)?;
        if spans.shape().dimensions() != [length] {
            return Err(Error::shape_mismatch(
                label,
                vec![length],
                spans.shape().dimensions().to_vec(),
            ));
        }
    }
    Ok(())
}

// A causal convolution's inputs `[T, C]`, taps `[C, K]`, and state `[S, C, K − 1]`, all of one type, giving `[T, C]`.
fn causal_conv1d_shapes(context: &InferenceContext<'_>, state: usize) -> Result<(i64, i64, i64)> {
    let (input, state) = (context.input(0)?, context.input(state)?);
    let (&[tokens, channels], &[sequences, state_channels, history]) =
        (input.shape().dimensions(), state.shape().dimensions())
    else {
        return Err(Error::invalid_argument(
            "causal convolution inputs are [tokens, channels] and its state [sequences, channels, taps − 1]",
        ));
    };
    if state_channels != channels || history < 1 {
        return Err(Error::shape_mismatch(
            "causal convolution state",
            vec![sequences, channels, history],
            state.shape().dimensions().to_vec(),
        ));
    }

    if state.data_type() != input.data_type() {
        return Err(Error::data_type_mismatch(
            "causal convolution state",
            input.data_type(),
            state.data_type(),
        ));
    }

    Ok((tokens, sequences, history))
}

// Convolves each sequence's rows after its state, giving the inputs' shape and type.
pub fn causal_conv1d(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let (tokens, sequences, history) = causal_conv1d_shapes(context, 2)?;
    let (input, weight) = (context.input(0)?, context.input(1)?);
    let channels = input.shape().dimensions()[1];
    if weight.shape().dimensions() != [channels, history + 1] {
        return Err(Error::shape_mismatch(
            "causal convolution taps",
            vec![channels, history + 1],
            weight.shape().dimensions().to_vec(),
        ));
    }

    if weight.data_type() != input.data_type() {
        return Err(Error::data_type_mismatch(
            "causal convolution taps",
            input.data_type(),
            weight.data_type(),
        ));
    }

    check_sequence_spans(context, 3, sequences, tokens, "causal convolution spans")?;
    Ok(context.same_type(input.shape()))
}

// A causal convolution's state after the step, of the state's shape and type.
pub fn causal_conv1d_state(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let (tokens, sequences, _) = causal_conv1d_shapes(context, 1)?;
    check_sequence_spans(context, 2, sequences, tokens, "causal convolution spans")?;
    let state = context.input(1)?;
    let mut inference = context.same_type(state.shape());
    inference.outputs[0] = ValueSpec::new(state.data_type(), state.shape().clone());
    Ok(inference)
}

// The gated delta rule's queries and keys `[T, H, Dk]`, values `[T, H, Dv]`, decays and strengths `[T, H]`, and `F32` state `[S, H, Dk, Dv]`, returning `(T, S, H, Dk, Dv)`.
fn gated_delta_rule_shapes(context: &InferenceContext<'_>) -> Result<[i64; 5]> {
    let (query, key, value) = (context.input(0)?, context.input(1)?, context.input(2)?);
    let (&[tokens, heads, key_dim], &[_, _, value_dim]) =
        (query.shape().dimensions(), value.shape().dimensions())
    else {
        return Err(Error::invalid_argument(
            "the gated delta rule's queries and values are [tokens, heads, dim]",
        ));
    };
    let state = context.input(5)?;
    let &[sequences, ..] = state.shape().dimensions() else {
        return Err(Error::invalid_argument(
            "the gated delta rule's state has a rank",
        ));
    };
    let expected = [
        (key, vec![tokens, heads, key_dim]),
        (value, vec![tokens, heads, value_dim]),
        (context.input(3)?, vec![tokens, heads]),
        (context.input(4)?, vec![tokens, heads]),
        (state, vec![sequences, heads, key_dim, value_dim]),
    ];
    for (operand, shape) in expected {
        if operand.shape().dimensions() != shape.as_slice() {
            return Err(Error::shape_mismatch(
                "gated delta rule operand",
                shape,
                operand.shape().dimensions().to_vec(),
            ));
        }
    }

    for operand in [key, value] {
        if operand.data_type() != query.data_type() {
            return Err(Error::data_type_mismatch(
                "gated delta rule operand",
                query.data_type(),
                operand.data_type(),
            ));
        }
    }

    if state.data_type() != DataType::F32 {
        return Err(Error::data_type_mismatch(
            "gated delta rule state",
            DataType::F32,
            state.data_type(),
        ));
    }

    check_sequence_spans(context, 6, sequences, tokens, "gated delta rule spans")?;
    Ok([tokens, sequences, heads, key_dim, value_dim])
}

// The gated delta rule's output, of the values' shape and type.
pub fn gated_delta_rule(context: &InferenceContext<'_>) -> Result<OperationInference> {
    gated_delta_rule_shapes(context)?;
    let value = context.input(2)?;
    let mut inference = context.same_type(value.shape());
    inference.outputs[0] = ValueSpec::new(value.data_type(), value.shape().clone());
    Ok(inference)
}

// The gated delta rule's state after the step, `F32` of the state's shape.
pub fn gated_delta_rule_state(context: &InferenceContext<'_>) -> Result<OperationInference> {
    gated_delta_rule_shapes(context)?;
    let state = context.input(5)?;
    let mut inference = context.same_type(state.shape());
    inference.outputs[0] = ValueSpec::new(DataType::F32, state.shape().clone());
    Ok(inference)
}

// Multiplies rows sorted by group by their groups' weights, keeping the rows' type.
pub fn grouped_matmul(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let (input, weights, offsets) = (context.input(0)?, context.input(1)?, context.input(2)?);
    if input.data_type() != weights.data_type() {
        return Err(Error::data_type_mismatch(
            "operation grouped matmul weights",
            input.data_type(),
            weights.data_type(),
        ));
    }

    let shape = spec::grouped_matmul_shape(
        input.shape(),
        weights.shape(),
        (offsets.shape(), offsets.data_type()),
        "operation grouped matmul",
    )?;
    Ok(context.same_type(&shape))
}

// Latent attention over paged caches, returning the output, of the queries' type and shape, and each row's base-two log-sum-exp in `F32`.
pub fn paged_mla_attention(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let (output_shape, lse_shape) = paged_mla_attention_shapes(context)?;
    let mut inference = context.same_type(&output_shape);
    inference.outputs[1] = ValueSpec::new(DataType::F32, lse_shape);
    Ok(inference)
}

// The output alone, as graph capture records latent attention's first result.
pub fn paged_mla_attention_output(context: &InferenceContext<'_>) -> Result<OperationInference> {
    Ok(context.same_type(&paged_mla_attention_shapes(context)?.0))
}

// The log-sum-exp alone, as graph capture records latent attention's second result; its role makes it `F32`.
pub fn paged_mla_attention_lse(context: &InferenceContext<'_>) -> Result<OperationInference> {
    Ok(context.same_type(&paged_mla_attention_shapes(context)?.1))
}

fn paged_mla_attention_shapes(context: &InferenceContext<'_>) -> Result<(Shape, Shape)> {
    let attention = context.attributes(OperationAttributes::scaled_dot_product_attention)?;
    if attention.masked
        || attention.sinks
        || attention.key_lengths
        || attention.window.is_some()
        || attention.softcap.is_some()
    {
        return Err(Error::unsupported_operation(
            "paged mla attention with a mask, sinks, a window, or a cap",
        ));
    }

    let index = |position: usize| -> Result<(&Shape, DataType)> {
        let input = context.input(position)?;
        Ok((input.shape(), input.data_type()))
    };
    let problem = spec::paged_mla_attention_problem(
        context.shape(),
        context.input(1)?.shape(),
        context.input(2)?.shape(),
        context.input(3)?.shape(),
        [index(4)?, index(5)?, index(6)?, index(7)?],
        "operation paged mla attention",
    )?;
    let query = context.input(0)?.data_type();
    for position in [1, 2, 3] {
        let operand = context.input(position)?.data_type();
        if operand != query {
            return Err(Error::data_type_mismatch(
                "operation paged mla attention operands",
                query,
                operand,
            ));
        }
    }
    Ok((problem.output_shape, problem.lse_shape))
}

/// Multiplies FP8 matrices scaled by `F32` factors into the attribute's output type, `F16`, `BF16`, or `F32`.
pub fn scaled_matmul(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let output = context.value(OperationAttributes::cast_data_type)?;
    if !matches!(output, DataType::F16 | DataType::Bf16 | DataType::F32) {
        return Err(Error::unsupported_data_type("operation scaled_mm output"));
    }

    let operand = |position: usize| -> Result<(&Shape, DataType)> {
        let input = context.input(position)?;
        Ok((input.shape(), input.data_type()))
    };
    let problem = spec::scaled_matmul_problem(
        operand(0)?,
        operand(1)?,
        operand(2)?,
        operand(3)?,
        "operation scaled_mm",
    )?;
    Ok(context.finish(&problem.output_shape, output))
}

/// Transforms a floating real signal along the last axis into `[..., n / 2 + 1, 2]` frequencies.
pub fn rfft(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let mut dimensions = context.shape().dimensions().to_vec();
    let length = dimensions
        .pop()
        .filter(|&length| length > 0)
        .ok_or_else(|| Error::invalid_argument("operation rfft needs a nonempty last axis"))?;
    if !context.input(0)?.data_type().is_float() {
        return Err(Error::unsupported_data_type("operation rfft"));
    }

    dimensions.extend([length / 2 + 1, 2]);
    Ok(context.same_type(&Shape::create(dimensions)?))
}

/// Transforms `[..., k, 2]` floating frequencies into a real signal of the attribute's length.
pub fn irfft(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let length = context.value(OperationAttributes::length)?;
    let dimensions = context.shape().dimensions();
    let (&[.., _, 2], true) = (dimensions, length > 0) else {
        return Err(Error::invalid_argument(
            "operation irfft needs [..., k, 2] frequencies and a positive length",
        ));
    };
    if !context.input(0)?.data_type().is_float() {
        return Err(Error::unsupported_data_type("operation irfft"));
    }

    let mut output = dimensions[..dimensions.len() - 2].to_vec();
    output.push(length);
    Ok(context.same_type(&Shape::create(output)?))
}

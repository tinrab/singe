mod convolution;
mod fourier;

use crate::{
    backend::{
        registry::{
            ContractionAttributes, Operation, OperationAttributes, QuantizationAttributes,
            ScaledDotProductAttentionAttributes, TopKAttributes,
        },
        spec,
    },
    data_type::DataType,
    dispatch::call_one,
    error::{Error, Result},
    graph::capture,
    scalar::Scalar,
    shape::Shape,
    tensor::Tensor,
    utility::to_i64,
};

// Decompositions write an operation as others, which run on every backend and lower in compiled graphs.
// Graph passes trace them on meta tensors, so a value they need is a `capture::literal` and never read back.

pub fn has_decomposition(operation: Operation) -> bool {
    matches!(
        operation,
        Operation::TopK
            | Operation::TopKValues
            | Operation::TopKIndices
            | Operation::Pad
            | Operation::RotaryEmbedding
            | Operation::ScaledDotProductAttention
            | Operation::Contraction
            | Operation::BatchNorm
            | Operation::QuantizedMatMul
            | Operation::ScaledMatMul
            | Operation::Rfft
            | Operation::Conv3d
            | Operation::Irfft
            | Operation::Softmax
            | Operation::LogSoftmax
            | Operation::MatMul
    )
}

// Returns whether the decomposition runs in place of the backend kernels for inputs of `data_type`, which they do not compute.
pub fn replaces_kernels(
    operation: Operation,
    attributes: &OperationAttributes,
    data_type: DataType,
) -> bool {
    match operation {
        // Native contractions, the CPU's and cuTENSOR's, compute `F32` only.
        Operation::Contraction => data_type != DataType::F32,
        // Kernels normalize with running statistics; training normalizes with the batch's.
        Operation::BatchNorm => attributes
            .batch_normalization()
            .is_some_and(|batch| batch.training),
        // Rotary kernels compute `F32` and widen half precision.
        Operation::RotaryEmbedding => data_type == DataType::F64,
        // Inference holds only the empty tensors of these, which kernels have no type for.
        Operation::Softmax | Operation::LogSoftmax => !matches!(
            data_type,
            DataType::F16 | DataType::Bf16 | DataType::F32 | DataType::F64
        ),
        Operation::MatMul => data_type == DataType::Bool,
        _ => false,
    }
}

// Returns whether compiled graphs run the decomposition of a node, since they have no kernel for it.
pub fn decomposes_in_graphs(
    operation: Operation,
    attributes: &OperationAttributes,
    data_type: DataType,
) -> bool {
    matches!(
        operation,
        Operation::TopK
            | Operation::TopKValues
            | Operation::TopKIndices
            | Operation::Pad
            | Operation::RotaryEmbedding
            | Operation::QuantizedMatMul
    ) || replaces_kernels(operation, attributes, data_type)
}

pub fn decompose(
    operation: Operation,
    attributes: &OperationAttributes,
    inputs: &[&Tensor],
) -> Result<Vec<Tensor>> {
    let input = *inputs
        .first()
        .ok_or_else(|| Error::missing_input("decompose"))?;
    let attribute_error = || Error::unsupported_operation("decomposition attributes");
    match operation {
        Operation::TopK | Operation::TopKValues | Operation::TopKIndices => {
            let topk = attributes.topk().ok_or_else(attribute_error)?;
            let (values, indices) = topk_of(input, topk)?;
            Ok(match operation {
                Operation::TopK => vec![values, indices],
                Operation::TopKValues => vec![values],
                _ => vec![indices],
            })
        }
        Operation::Pad => {
            let pad = attributes.pad().ok_or_else(attribute_error)?;
            Ok(vec![pad_of(input, &pad.padding, pad.value)?])
        }
        Operation::RotaryEmbedding => {
            let [_, cos, sin] = inputs else {
                return Err(Error::missing_input("decompose"));
            };
            Ok(vec![rotary_embedding(input, cos, sin)?])
        }
        Operation::ScaledDotProductAttention => {
            let [_, key, value, rest @ ..] = inputs else {
                return Err(Error::missing_input("decompose"));
            };
            let attention = attributes
                .scaled_dot_product_attention()
                .ok_or_else(attribute_error)?;
            let mask = rest.first().copied().filter(|_| attention.masked);
            let sinks = rest
                .get(usize::from(attention.masked))
                .copied()
                .filter(|_| attention.sinks);
            let lengths = rest
                .get(usize::from(attention.masked) + usize::from(attention.sinks))
                .copied()
                .filter(|_| attention.key_lengths);
            // Decomposed, key lengths are the mask that hides each batch's keys past its count.
            let mask = match lengths {
                Some(lengths) => {
                    let hidden =
                        length_mask(lengths, key.shape().dimensions()[2], input.data_type())?;
                    Some(match mask {
                        Some(mask) => mask.add(&hidden)?,
                        None => hidden,
                    })
                }
                None => mask.cloned(),
            };
            Ok(vec![scaled_dot_product_attention(
                input,
                key,
                value,
                mask.as_ref(),
                sinks,
                attention,
            )?])
        }
        Operation::Contraction => {
            let [a, b, c] = inputs else {
                return Err(Error::missing_input("decompose"));
            };
            let contraction = attributes.contraction().ok_or_else(attribute_error)?;
            Ok(vec![contraction_of(a, b, c, contraction)?])
        }
        Operation::BatchNorm => {
            let batch = attributes
                .batch_normalization()
                .ok_or_else(attribute_error)?;
            if batch.has_running_mean || batch.has_running_var {
                return Err(Error::unsupported_operation(
                    "batch norm running-state mutation",
                ));
            }

            let weight = inputs.get(1).filter(|_| batch.has_weight).copied();
            let bias = inputs
                .get(1 + usize::from(batch.has_weight))
                .filter(|_| batch.has_bias)
                .copied();
            Ok(vec![batch_norm_training(
                input,
                weight,
                bias,
                None,
                None,
                batch.momentum,
                batch.eps,
            )?])
        }
        // An empty input normalizes to an empty copy of itself.
        Operation::Softmax | Operation::LogSoftmax => Ok(vec![input.copy()?]),
        // A product without terms is all false.
        Operation::MatMul => {
            let [_, rhs] = inputs else {
                return Err(Error::missing_input("decompose"));
            };
            let problem = spec::matmul_problem(input.shape(), rhs.shape(), "matmul")?;
            let shape = problem.output_shape;
            Ok(vec![capture::literal(input.device(), |device| {
                Scalar::Bool(false).full(device, input.data_type(), &shape)
            })?])
        }
        Operation::Conv3d => {
            let [_, weight, bias @ ..] = inputs else {
                return Err(Error::missing_input("decompose"));
            };
            let conv = attributes.conv3d().ok_or_else(attribute_error)?;
            Ok(vec![input.convolution3d_by_planes(
                weight,
                bias.first().copied(),
                conv,
            )?])
        }
        Operation::Rfft => Ok(vec![input.rfft_by_basis()?]),
        Operation::Irfft => {
            let length = attributes.length().ok_or_else(attribute_error)?;
            Ok(vec![input.irfft_by_basis(length)?])
        }
        Operation::ScaledMatMul => {
            let [lhs, rhs, lhs_scale, rhs_scale] = inputs else {
                return Err(Error::missing_input("decompose"));
            };
            let output = attributes.cast_data_type().ok_or_else(attribute_error)?;
            Ok(vec![scaled_matmul(lhs, rhs, lhs_scale, rhs_scale, output)?])
        }
        Operation::QuantizedMatMul => {
            let [lhs, rhs, lhs_scale, lhs_zero, rhs_scale, rhs_zero] = inputs else {
                return Err(Error::missing_input("decompose"));
            };
            let per_tensor = OperationAttributes::Quantization(QuantizationAttributes {
                axis: None,
                block_size: None,
                data_type: None,
                packed: None,
            });
            let lhs = call_one(
                Operation::Dequantize,
                per_tensor.clone(),
                &[lhs, lhs_scale, lhs_zero],
            )?;
            let rhs = call_one(
                Operation::Dequantize,
                per_tensor,
                &[rhs, rhs_scale, rhs_zero],
            )?;
            Ok(vec![lhs.matmul(&rhs)?])
        }
        _ => Err(Error::unsupported_operation_with_details(
            operation.name(),
            "no decomposition",
        )),
    }
}

// The first `k` positions of a stable sort, and the values there.
fn topk_of(input: &Tensor, topk: &TopKAttributes) -> Result<(Tensor, Tensor)> {
    if input.shape().rank() == 0 {
        let (values, indices) = topk_of(
            &input.unsqueeze(0)?,
            &TopKAttributes {
                k: topk.k,
                dim: 0,
                largest: topk.largest,
                sorted: topk.sorted,
            },
        )?;
        return Ok((values.squeeze(0)?, indices.squeeze(0)?));
    }

    let indices = input
        .argsort(topk.dim, topk.largest)?
        .narrow(topk.dim, 0, topk.k)?
        .contiguous()?;
    let values = input.gather(topk.dim, &indices)?;
    Ok((values, indices))
}

// Crops each axis by its negative widths, then concatenates blocks of `value` on each side.
fn pad_of(input: &Tensor, padding: &[(i64, i64)], value: f32) -> Result<Tensor> {
    let problem = spec::pad_problem(input.shape(), padding, "pad")?;
    let mut output = input.clone();
    for (axis, &(before, after)) in problem.padding.iter().enumerate() {
        let dim = i64::try_from(axis).map_err(|_| Error::DimensionOutOfBounds)?;
        let extent = output.shape().dimensions()[axis];
        let (start, end) = ((-before).max(0), (-after).max(0));
        if start > 0 || end > 0 {
            output = output.narrow(dim, start, extent - start - end)?;
        }

        if before <= 0 && after <= 0 {
            continue;
        }

        let block = |width: i64| -> Result<Tensor> {
            let mut dimensions = output.shape().dimensions().to_vec();
            dimensions[axis] = width;
            let shape = Shape::create(dimensions)?;
            capture::literal(output.device(), |device| {
                Scalar::Float(f64::from(value)).full(device, output.data_type(), &shape)
            })
        };
        let mut parts = Vec::with_capacity(3);
        if before > 0 {
            parts.push(block(before)?);
        }

        parts.push(output.clone());
        if after > 0 {
            parts.push(block(after)?);
        }

        output = Tensor::concat(&parts, dim)?;
    }

    // Padding that only crops, or does nothing, still returns a tensor of its own.
    if !problem
        .padding
        .iter()
        .any(|&(before, after)| before > 0 || after > 0)
    {
        output = output.copy()?;
    }
    Ok(output)
}

// Rotates each adjacent pair of the leading `2 · pairs` features by its position's angle.
fn rotary_embedding(input: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let problem = spec::rotary_embedding_problem(
        input.shape(),
        cos.shape(),
        sin.shape(),
        "rotary_embedding",
    )?;
    let (pairs, head_dim) = (problem.rotary_pairs, problem.head_dim);
    let mut paired = input.shape().dimensions()[..3].to_vec();
    paired.extend([pairs, 2]);
    let rotated = input.narrow(3, 0, 2 * pairs)?.reshape(paired)?;
    let (even, odd) = (rotated.select(4, 0)?, rotated.select(4, 1)?);
    let first = even.mul(cos)?.sub(&odd.mul(sin)?)?;
    let second = even.mul(sin)?.add(&odd.mul(cos)?)?;
    let mut interleaved = input.shape().dimensions()[..3].to_vec();
    interleaved.push(2 * pairs);
    let rotated =
        Tensor::concat(&[first.unsqueeze(4)?, second.unsqueeze(4)?], 4)?.reshape(interleaved)?;
    if 2 * pairs == head_dim {
        return Ok(rotated);
    }
    Tensor::concat(
        &[rotated, input.narrow(3, 2 * pairs, head_dim - 2 * pairs)?],
        3,
    )
}

// `softmax(q kᵀ · scale + mask) v`; half precision computes in `F32` and rounds once, as PyTorch's attention does, since rounding the scores and weights loses what the exponentials amplify.
// The additive mask, `[B, 1, 1, S]` of `data_type`, that hides each batch's keys from its count in `lengths` on.
fn length_mask(lengths: &Tensor, keys: i64, data_type: DataType) -> Result<Tensor> {
    let batch = lengths.shape().dimensions()[0];
    let positions = capture::literal(lengths.device(), |device| {
        Tensor::from_vec(
            device,
            Shape::create([1, keys])?,
            (0..keys)
                .map(|key| i32::try_from(key).unwrap_or(i32::MAX))
                .collect(),
        )
    })?;
    let fill = |value: f64| {
        capture::literal(lengths.device(), |device| {
            Scalar::Float(value).tensor(device, data_type)
        })
    };
    Tensor::where_select(
        &positions.greater_equal(&lengths.unsqueeze(1)?)?,
        &fill(f64::NEG_INFINITY)?,
        &fill(0.0)?,
    )?
    .reshape([batch, 1, 1, keys])
}

fn scaled_dot_product_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    mask: Option<&Tensor>,
    sinks: Option<&Tensor>,
    attention: &ScaledDotProductAttentionAttributes,
) -> Result<Tensor> {
    let data_type = query.data_type();
    if matches!(data_type, DataType::F16 | DataType::Bf16) {
        let wide = |tensor: &Tensor| tensor.cast(DataType::F32);
        let mask = mask.map(wide).transpose()?;
        let sinks = sinks.map(wide).transpose()?;
        return scaled_dot_product_attention(
            &wide(query)?,
            &wide(key)?,
            &wide(value)?,
            mask.as_ref(),
            sinks.as_ref(),
            attention,
        )?
        .cast(data_type);
    }

    let problem = spec::attention_problem(
        query.shape(),
        key.shape(),
        value.shape(),
        "scaled_dot_product_attention",
    )?;
    // Grouped key and value heads repeat for the query heads that share them.
    let groups = query.shape().dimensions()[1] / key.shape().dimensions()[1];
    if groups > 1 {
        return scaled_dot_product_attention(
            query,
            &key.repeat_kv(groups)?,
            &value.repeat_kv(groups)?,
            mask,
            sinks,
            attention,
        );
    }

    let head_dim = query.shape().dimensions()[3];
    let scale = attention
        .scale
        .map_or_else(|| (head_dim as f64).sqrt().recip(), f64::from);
    let mut scores = query
        .matmul(&key.transpose(2, 3)?)?
        .mul_scalar(Scalar::Float(scale))?;
    if let Some(cap) = attention.softcap {
        let cap = f64::from(cap);
        scores = scores
            .div_scalar(cap)?
            .tanh()?
            .mul_scalar(Scalar::Float(cap))?;
    }

    let scores = if attention.causal || attention.window.is_some() {
        let mask = capture::literal(query.device(), |device| {
            Tensor::attention_mask(
                device,
                DataType::F32,
                problem.query_len,
                problem.key_len,
                attention.causal,
                attention.window,
            )
        })?;
        let mask = if data_type == DataType::F32 {
            mask
        } else {
            mask.cast(data_type)?
        };
        scores.add(&mask)?
    } else {
        scores
    };
    let scores = match mask {
        Some(mask) => scores.add(mask)?,
        None => scores,
    };
    // A sink logit per head joins each row's softmax as one more score, whose weight the product drops.
    let Some(sinks) = sinks else {
        return scores.softmax(-1)?.matmul(value);
    };
    let [batch, heads, rows, keys] = *scores.shape().dimensions() else {
        return Err(Error::invalid_argument("attention scores rank"));
    };
    let column = sinks
        .reshape([1, heads, 1, 1])?
        .expand(&Shape::create([batch, heads, rows, 1])?)?;
    Tensor::concat(&[scores, column], 3)?
        .softmax(-1)?
        .narrow(3, 0, keys)?
        .matmul(value)
}

// `alpha · Σ a b + beta · c` as one batched matmul: `a`'s modes become batch, free, and contracted axes, `b`'s batch, contracted, and free ones, and the product's axes permute to `d`'s modes.
fn contraction_of(
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    contraction: &ContractionAttributes,
) -> Result<Tensor> {
    let ContractionAttributes {
        a_mode,
        b_mode,
        c_mode,
        d_mode,
        alpha,
        beta,
    } = contraction;
    let spec = spec::contraction_problem(
        a.shape(),
        a_mode,
        b.shape(),
        b_mode,
        c.shape(),
        c_mode,
        d_mode,
    )?;
    for modes in [a_mode, b_mode, c_mode, d_mode] {
        let mut sorted = modes.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != modes.len() {
            return Err(Error::unsupported_operation(
                "contraction with a repeated mode",
            ));
        }
    }

    let has = |modes: &[i64], mode: i64| modes.contains(&mode);
    let batch = d_mode
        .iter()
        .copied()
        .filter(|&mode| has(a_mode, mode) && has(b_mode, mode))
        .collect::<Vec<_>>();
    let contracted = a_mode
        .iter()
        .copied()
        .filter(|&mode| has(b_mode, mode) && !has(d_mode, mode))
        .collect::<Vec<_>>();
    let a_free = d_mode
        .iter()
        .copied()
        .filter(|&mode| has(a_mode, mode) && !has(b_mode, mode))
        .collect::<Vec<_>>();
    let b_free = d_mode
        .iter()
        .copied()
        .filter(|&mode| has(b_mode, mode) && !has(a_mode, mode))
        .collect::<Vec<_>>();
    // Modes of one operand that neither the other nor the output has are summed away first.
    let operand =
        |tensor: &Tensor, modes: &[i64], other: &[i64], order: [&[i64]; 3]| -> Result<Tensor> {
            let summed = modes
                .iter()
                .enumerate()
                .filter(|&(_, &mode)| !has(other, mode) && !has(d_mode, mode))
                .map(|(axis, _)| i64::try_from(axis).map_err(|_| Error::DimensionOutOfBounds))
                .collect::<Result<Vec<_>>>()?;
            let (tensor, modes) = if summed.is_empty() {
                (tensor.clone(), modes.to_vec())
            } else {
                let kept = modes
                    .iter()
                    .copied()
                    .filter(|&mode| has(other, mode) || has(d_mode, mode))
                    .collect::<Vec<_>>();
                (tensor.sum(&summed, false)?, kept)
            };
            let position = |mode: i64| {
                modes
                    .iter()
                    .position(|&candidate| candidate == mode)
                    .and_then(|axis| i64::try_from(axis).ok())
                    .ok_or_else(|| Error::out_of_range(format!("contraction mode {mode}")))
            };
            let permutation = order
                .iter()
                .flat_map(|group| group.iter())
                .map(|&mode| position(mode))
                .collect::<Result<Vec<_>>>()?;
            let extent = |group: &[i64]| -> Result<i64> {
                group.iter().try_fold(1_i64, |size, &mode| {
                    Ok(size
                        * tensor.shape().dimensions()[usize::try_from(position(mode)?)
                            .map_err(|_| Error::DimensionOutOfBounds)?])
                })
            };
            let sizes = [extent(order[0])?, extent(order[1])?, extent(order[2])?];
            tensor.permute(&permutation)?.reshape(sizes.to_vec())
        };
    let lhs = operand(a, a_mode, b_mode, [&batch, &a_free, &contracted])?;
    let rhs = operand(b, b_mode, a_mode, [&batch, &contracted, &b_free])?;
    let product_modes = [batch.as_slice(), &a_free, &b_free].concat();
    let mode_extent = |mode: i64| -> Result<i64> {
        d_mode
            .iter()
            .position(|&candidate| candidate == mode)
            .map(|axis| spec.output_shape.dimensions()[axis])
            .ok_or_else(|| Error::out_of_range(format!("contraction output mode {mode}")))
    };
    let product_shape = product_modes
        .iter()
        .map(|&mode| mode_extent(mode))
        .collect::<Result<Vec<_>>>()?;
    let to_output = |modes: &[i64]| -> Result<Vec<i64>> {
        d_mode
            .iter()
            .map(|&mode| {
                modes
                    .iter()
                    .position(|&candidate| candidate == mode)
                    .and_then(|axis| i64::try_from(axis).ok())
                    .ok_or_else(|| Error::out_of_range(format!("contraction output mode {mode}")))
            })
            .collect()
    };
    let product = lhs
        .matmul(&rhs)?
        .reshape(product_shape)?
        .permute(&to_output(&product_modes)?)?;
    let scaled = product.mul_scalar(Scalar::Float(f64::from(*alpha)))?;
    let accumulator = c
        .permute(&to_output(c_mode)?)?
        .mul_scalar(Scalar::Float(f64::from(*beta)))?;
    scaled.add(&accumulator)
}

// Normalizes with the batch's mean and biased variance over every axis but the channels, which the running-statistics kernels then apply, and updates running statistics in place with the unbiased variance, as PyTorch's training does; half precision computes in `F32`.
// The operation never writes its inputs, so its decomposition takes none; the tensor method passes them and records their updates as writes.
pub fn batch_norm_training(
    input: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    running_mean: Option<&Tensor>,
    running_var: Option<&Tensor>,
    momentum: f32,
    eps: f32,
) -> Result<Tensor> {
    let data_type = input.data_type();
    let compute = if matches!(data_type, DataType::F16 | DataType::Bf16) {
        DataType::F32
    } else {
        data_type
    };
    let wide = |tensor: &Tensor| {
        if tensor.data_type() == compute {
            Ok(tensor.clone())
        } else {
            tensor.cast(compute)
        }
    };
    let x = wide(input)?;
    let rank = x.shape().rank();
    let axes = (0..rank).filter(|&axis| axis != 1).collect::<Vec<_>>();
    let mean = x.mean(&axes, true)?;
    let variance = x.sub(&mean)?.square()?.mean(&axes, false)?;
    let mean = mean.mean(&axes, false)?;
    let output = x.batch_norm(
        weight.map(wide).transpose()?.as_ref(),
        bias.map(wide).transpose()?.as_ref(),
        Some(&mean),
        Some(&variance),
        false,
        momentum,
        eps,
    )?;
    let momentum = f64::from(momentum);
    let update = |running: &Tensor, batch: &Tensor| -> Result<()> {
        let updated = wide(running)?
            .mul_scalar(Scalar::Float(1.0 - momentum))?
            .add(&batch.mul_scalar(Scalar::Float(momentum))?)?;
        running.assign(&updated.cast(running.data_type())?)
    };
    if let Some(running_mean) = running_mean {
        update(running_mean, &mean)?;
    }

    if let Some(running_var) = running_var {
        let channels = x.shape().dimensions().get(1).copied().unwrap_or(1).max(1);
        let samples = to_i64(x.shape().element_count(), "batch norm sample count")? / channels;
        #[expect(
            clippy::cast_precision_loss,
            reason = "sample counts are far below 2^52"
        )]
        let correction = samples as f64 / (samples - 1) as f64;
        update(
            running_var,
            &variance.mul_scalar(Scalar::Float(correction))?,
        )?;
    }
    if compute == data_type {
        Ok(output)
    } else {
        output.cast(data_type)
    }
}

// `scaled_mm` as a product of its dequantized operands: each FP8 operand widens to `F32` and multiplies by its scales, which broadcast over it, and the `F32` product rounds once to `output`.
fn scaled_matmul(
    lhs: &Tensor,
    rhs: &Tensor,
    lhs_scale: &Tensor,
    rhs_scale: &Tensor,
    output: DataType,
) -> Result<Tensor> {
    let widened = |operand: &Tensor, scale: &Tensor| operand.cast(DataType::F32)?.mul(scale);
    widened(lhs, lhs_scale)?
        .matmul(&widened(rhs, rhs_scale)?)?
        .cast(output)
}

// A `scaled_mm` with one scale per row and column as the unscaled FP8 product, which tensor cores run in compiled graphs, in `F32`, times the outer product of the scales, rounded once.
pub fn rowwise_scaled_matmul(
    attributes: &OperationAttributes,
    inputs: &[&Tensor],
) -> Result<Vec<Tensor>> {
    let [lhs, rhs, lhs_scale, rhs_scale] = inputs else {
        return Err(Error::missing_input("rowwise scaled matmul"));
    };
    let output = attributes
        .cast_data_type()
        .ok_or_else(|| Error::unsupported_operation("decomposition attributes"))?;
    let unit = capture::literal(lhs.device(), |device| Tensor::scalar(device, 1.0_f32))?;
    let product = lhs
        .scaled_matmul(rhs, &unit, &unit, DataType::F32)?
        .mul(lhs_scale)?
        .mul(rhs_scale)?
        .cast(output)?;
    Ok(vec![product])
}

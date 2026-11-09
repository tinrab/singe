//! Backward formulas: the gradient of each input from the gradient of the outputs.
//!
//! Formulas run with gradient recording disabled and compose existing operations, so they run on every backend.

use crate::{
    autograd::{set_grad_enabled, spellings},
    backend::{
        decomposition,
        registry::{Operation, OperationAttributes, ReductionAttributes},
    },
    error::{Error, Result},
    scalar::Scalar,
    shape::Shape,
    tensor::Tensor,
};

/// Returns the gradient of each input of `operation`, or `None` for inputs without one.
#[expect(
    clippy::too_many_lines,
    reason = "the formula table keeps every differentiable operation together"
)]
pub fn backward(
    operation: Operation,
    attributes: &OperationAttributes,
    inputs: &[Tensor],
    outputs: &[Tensor],
    gradients: &[Tensor],
) -> Result<Vec<Option<Tensor>>> {
    use Operation as Op;
    let gradient = gradients
        .first()
        .ok_or_else(|| Error::missing_output("backward"))?;
    let input = |index: usize| {
        inputs
            .get(index)
            .ok_or_else(|| Error::missing_input("backward"))
    };
    let output = || {
        outputs
            .first()
            .ok_or_else(|| Error::missing_output("backward"))
    };
    let unary = |value: Tensor| Ok(vec![Some(value)]);
    match operation {
        Op::Add => Ok(vec![
            Some(sum_to(gradient, input(0)?.shape())?),
            Some(sum_to(gradient, input(1)?.shape())?),
        ]),
        Op::Sub => Ok(vec![
            Some(sum_to(gradient, input(0)?.shape())?),
            Some(sum_to(&gradient.neg()?, input(1)?.shape())?),
        ]),
        Op::Mul => Ok(vec![
            Some(sum_to(&gradient.mul(input(1)?)?, input(0)?.shape())?),
            Some(sum_to(&gradient.mul(input(0)?)?, input(1)?.shape())?),
        ]),
        Op::Div => {
            let (lhs, rhs) = (input(0)?, input(1)?);
            let quotient = gradient.div(rhs)?;
            Ok(vec![
                Some(sum_to(&quotient, lhs.shape())?),
                Some(sum_to(&quotient.mul(output()?)?.neg()?, rhs.shape())?),
            ])
        }
        Op::MatMul => {
            let (lhs, rhs) = (input(0)?, input(1)?);
            if lhs.shape().rank() < 2 || rhs.shape().rank() < 2 {
                return Err(unsupported(operation, "vector operands"));
            }
            Ok(vec![
                Some(sum_to(
                    &gradient.matmul(&rhs.transpose(-1, -2)?)?,
                    lhs.shape(),
                )?),
                Some(sum_to(
                    &lhs.transpose(-1, -2)?.matmul(gradient)?,
                    rhs.shape(),
                )?),
            ])
        }
        Op::Neg | Op::RSubScalar => unary(gradient.neg()?),
        Op::Scale => unary(gradient.mul_scalar(scalar(attributes)?)?),
        Op::AddScalar | Op::SubScalar | Op::Contiguous | Op::Copy => unary(gradient.clone()),
        Op::RDivScalar => unary(gradient.mul(&output()?.div(input(0)?)?)?.neg()?),
        Op::Exp => unary(gradient.mul(output()?)?),
        Op::Log => unary(gradient.div(input(0)?)?),
        Op::Sqrt => unary(gradient.div(&output()?.mul_scalar(2.0)?)?),
        Op::Square => unary(gradient.mul(&input(0)?.mul_scalar(2.0)?)?),
        Op::Reciprocal => unary(gradient.mul(&output()?.square()?)?.neg()?),
        Op::Tanh => unary(gradient.mul(&output()?.square()?.rsub_scalar(1.0)?)?),
        Op::Sigmoid => {
            let output = output()?;
            unary(gradient.mul(&output.mul(&output.rsub_scalar(1.0)?)?)?)
        }
        Op::Relu => {
            let positive = input(0)?.greater(&input(0)?.zeros_like()?)?;
            unary(Tensor::where_select(
                &positive,
                gradient,
                &gradient.zeros_like()?,
            )?)
        }
        Op::Silu => {
            let x = input(0)?;
            let sigmoid = x.sigmoid()?;
            // d/dx x·σ(x) = σ(x)·(1 + x·(1 − σ(x)))
            let slope = sigmoid.mul(&x.mul(&sigmoid.rsub_scalar(1.0)?)?.add_scalar(1.0)?)?;
            unary(gradient.mul(&slope)?)
        }
        Op::Sum | Op::Mean => {
            let reduction = attributes
                .reduction()
                .ok_or_else(|| Error::invalid_argument("reduction attributes"))?;
            let source = input(0)?;
            let expanded = expand_reduced(gradient, source.shape(), reduction)?;
            if operation == Op::Sum {
                return unary(expanded);
            }

            let count = source.element_count() / output()?.element_count().max(1);
            #[expect(
                clippy::cast_precision_loss,
                reason = "reduced element counts are far below 2^52"
            )]
            let scale = 1.0 / count.max(1) as f64;
            unary(expanded.mul_scalar(scale)?)
        }
        // d/dx log(sum(exp(x))) = exp(x − log(sum(exp(x)))).
        Op::LogSumExp => {
            let reduction = attributes
                .reduction()
                .ok_or_else(|| Error::invalid_argument("reduction attributes"))?;
            let source = input(0)?;
            let total = expand_reduced(output()?, source.shape(), reduction)?;
            let expanded = expand_reduced(gradient, source.shape(), reduction)?;
            unary(expanded.mul(&source.sub(&total)?.exp()?)?)
        }
        Op::Softmax => {
            let dim = dimension(attributes)?;
            let output = output()?;
            let weighted = gradient.mul(output)?.sum(&[dim], true)?;
            unary(output.mul(&gradient.sub(&weighted)?)?)
        }
        Op::LogSoftmax => {
            let dim = dimension(attributes)?;
            let total = gradient.sum(&[dim], true)?;
            unary(gradient.sub(&output()?.exp()?.mul(&total)?)?)
        }
        Op::Transpose => {
            let transpose = attributes
                .transpose()
                .ok_or_else(|| Error::invalid_argument("transpose attributes"))?;
            unary(gradient.transpose(transpose.dim0, transpose.dim1)?)
        }
        Op::Permute => {
            let order = attributes
                .permutation_order()
                .ok_or_else(|| Error::invalid_argument("permute attributes"))?;
            let rank = i64::try_from(order.len()).map_err(|_| Error::DimensionOutOfBounds)?;
            let mut inverse = vec![0_i64; order.len()];
            for (position, &axis) in order.iter().enumerate() {
                let axis = if axis < 0 { axis + rank } else { axis };
                let slot = usize::try_from(axis).map_err(|_| Error::DimensionOutOfBounds)?;
                *inverse.get_mut(slot).ok_or(Error::DimensionOutOfBounds)? =
                    i64::try_from(position).map_err(|_| Error::DimensionOutOfBounds)?;
            }
            unary(gradient.permute(&inverse)?)
        }
        Op::Reshape | Op::Flatten | Op::Squeeze | Op::Unsqueeze => {
            unary(gradient.reshape(input(0)?.shape().dimensions())?)
        }
        Op::Expand => unary(sum_to(gradient, input(0)?.shape())?),
        Op::Slice => {
            let slice = attributes
                .slice()
                .ok_or_else(|| Error::invalid_argument("slice attributes"))?;
            let full = input(0)?.zeros_like()?;
            full.strided_view_unrecorded(full.layout().slice(
                slice.dim,
                slice.start,
                slice.length,
                slice.step,
            )?)?
            .assign(&gradient.contiguous()?)?;
            unary(full)
        }
        Op::Concat => {
            let dim = attributes
                .concat_dimension()
                .ok_or_else(|| Error::invalid_argument("concat attributes"))?;
            let rank = gradient.shape().rank();
            let axis = if dim < 0 { dim + rank } else { dim };
            let axis_index = usize::try_from(axis).map_err(|_| Error::DimensionOutOfBounds)?;
            let mut start = 0;
            inputs
                .iter()
                .map(|part| {
                    let length = part.shape().dimensions()[axis_index];
                    let piece = gradient.narrow(axis, start, length)?;
                    start += length;
                    Ok(Some(piece))
                })
                .collect()
        }
        Op::Where => {
            let condition = input(0)?;
            let zeros = gradient.zeros_like()?;
            Ok(vec![
                None,
                Some(sum_to(
                    &Tensor::where_select(condition, gradient, &zeros)?,
                    input(1)?.shape(),
                )?),
                Some(sum_to(
                    &Tensor::where_select(condition, &zeros, gradient)?,
                    input(2)?.shape(),
                )?),
            ])
        }
        Op::Cast => unary(gradient.cast(input(0)?.data_type())?),
        Op::Embedding => {
            let (weight, indices) = (input(0)?, input(1)?);
            let width = weight.shape().dimensions()[1..].iter().product::<i64>();
            let count =
                i64::try_from(indices.element_count()).map_err(|_| Error::DimensionOutOfBounds)?;
            let rows = gradient.reshape([count, width])?;
            let flat = indices.reshape([count])?;
            let accumulated = weight
                .zeros_like()?
                .reshape([weight.shape().dimensions()[0], width])?
                .index_add(0, &flat, &rows)?;
            Ok(vec![
                Some(accumulated.reshape(weight.shape().dimensions())?),
                None,
            ])
        }
        // Reversing the heads' repeat sums each head's copies.
        Op::RepeatKv => {
            let source = input(0)?.shape().dimensions().to_vec();
            let &[batch, heads, keys, width] = source.as_slice() else {
                return Err(unsupported(
                    operation,
                    "repeated keys and values have rank four",
                ));
            };
            let repeats = gradient.shape().dimensions()[1] / heads.max(1);
            unary(
                gradient
                    .contiguous()?
                    .reshape([batch, heads, repeats, keys, width])?
                    .sum(&[2], false)?,
            )
        }
        Op::RmsNorm
        | Op::LayerNorm
        | Op::BatchNorm
        | Op::Conv2d
        | Op::MaxPool2d
        | Op::AvgPool2d => {
            let shape = output()?.shape().dimensions().to_vec();
            through(inputs, gradients, |leaves| {
                let output = match operation {
                    Op::RmsNorm | Op::LayerNorm => {
                        let norm = attributes
                            .layer_normalization()
                            .ok_or_else(|| Error::invalid_argument("normalization attributes"))?;
                        if operation == Op::RmsNorm {
                            spellings::rms_norm(leaves, norm)?
                        } else {
                            spellings::layer_norm(leaves, norm)?
                        }
                    }
                    Op::BatchNorm => spellings::batch_norm(
                        leaves,
                        attributes
                            .batch_normalization()
                            .ok_or_else(|| Error::invalid_argument("batch norm attributes"))?,
                    )?,
                    Op::Conv2d => spellings::conv2d(
                        leaves,
                        attributes
                            .conv2d()
                            .ok_or_else(|| Error::invalid_argument("convolution attributes"))?,
                        &shape,
                    )?,
                    _ => {
                        let pool = attributes
                            .pool2d()
                            .ok_or_else(|| Error::invalid_argument("pooling attributes"))?;
                        if operation == Op::MaxPool2d {
                            spellings::max_pool2d(&leaves[0], pool, &shape)?
                        } else {
                            spellings::avg_pool2d(&leaves[0], pool, &shape)?
                        }
                    }
                };
                Ok(vec![output])
            })
        }
        Op::ScaledDotProductAttention | Op::Conv3d | Op::Pad => {
            through(inputs, gradients, |leaves| {
                decomposition::decompose(operation, attributes, &leaves.iter().collect::<Vec<_>>())
            })
        }
        _ => Err(unsupported(operation, "no backward formula")),
    }
}

// Differentiates an operation through `form`, which computes the same values from operations with formulas, on fresh leaves that collect the gradient of each input.
fn through(
    inputs: &[Tensor],
    gradients: &[Tensor],
    form: impl FnOnce(&[Tensor]) -> Result<Vec<Tensor>>,
) -> Result<Vec<Option<Tensor>>> {
    let leaves = inputs
        .iter()
        .map(|input| {
            let leaf = input.detach();
            if leaf.data_type().is_float() {
                leaf.set_requires_grad(true)?;
            }
            Ok(leaf)
        })
        .collect::<Result<Vec<_>>>()?;
    let recomputed = {
        let _enabled = set_grad_enabled(true);
        form(&leaves)?
    };
    for (output, gradient) in recomputed.iter().zip(gradients) {
        if output.requires_grad() {
            output.backward_with(gradient)?;
        }
    }
    Ok(leaves.iter().map(Tensor::grad).collect())
}

fn unsupported(operation: Operation, details: &'static str) -> Error {
    Error::unsupported_operation_with_details(format!("{} backward", operation.name()), details)
}

fn scalar(attributes: &OperationAttributes) -> Result<Scalar> {
    attributes
        .scalar()
        .ok_or_else(|| Error::invalid_argument("scalar attributes"))
}

fn dimension(attributes: &OperationAttributes) -> Result<i64> {
    attributes
        .dimension()
        .ok_or_else(|| Error::invalid_argument("dimension attributes"))
}

/// Sums `gradient` over the dimensions broadcasting added, so it has the shape of the operand that broadcast.
fn sum_to(gradient: &Tensor, shape: &Shape) -> Result<Tensor> {
    if gradient.shape() == shape {
        return Ok(gradient.clone());
    }

    let dimensions = gradient.shape().dimensions();
    let leading = dimensions.len() - shape.dimensions().len();
    let reduced = (0..dimensions.len())
        .filter(|&axis| {
            axis < leading || (shape.dimensions()[axis - leading] == 1 && dimensions[axis] != 1)
        })
        .map(|axis| i64::try_from(axis).map_err(|_| Error::DimensionOutOfBounds))
        .collect::<Result<Vec<_>>>()?;
    let summed = if reduced.is_empty() {
        gradient.clone()
    } else {
        gradient.sum(&reduced, true)?
    };
    summed.reshape(shape.dimensions())
}

/// Expands the gradient of a reduction back over the reduced dimensions of `shape`.
fn expand_reduced(
    gradient: &Tensor,
    shape: &Shape,
    reduction: &ReductionAttributes,
) -> Result<Tensor> {
    let rank = shape.rank();
    let mut kept = gradient.clone();
    if !reduction.keepdim {
        let mut axes = if reduction.dimensions.is_empty() {
            (0..rank).collect::<Vec<_>>()
        } else {
            reduction
                .dimensions
                .iter()
                .map(|&axis| if axis < 0 { axis + rank } else { axis })
                .collect()
        };
        axes.sort_unstable();
        for axis in axes {
            kept = kept.unsqueeze(axis)?;
        }
    }
    kept.expand(shape)?.contiguous()
}

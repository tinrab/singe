use std::collections::{BTreeMap, BTreeSet};

use crate::{
    backend::registry::{ContractionAttributes, Operation, OperationAttributes, PackedFormat},
    data_type::DataType,
    dispatch::call_one,
    error::{Error, Result},
    graph::capture,
    scalar::Scalar,
    shape::Shape,
    tensor::{Tensor, elementwise::scalar_parameters},
};

impl Tensor {
    // Multiplies FP8 matrices `self [M, K]` and `rhs [K, N]`, each scaled by its `F32` scale, one value or `[M, 1]` and `[1, N]` per row and column, into `output`, `F16`, `BF16`, or `F32`, as `torch._scaled_mm` does.
    // CUDA runs it on FP8 tensor cores when `self` is row-major and `rhs` column-major, as a linear layer's transposed weight is, with extents that are multiples of 16 and one scale per operand, and as a dequantized `F32` product otherwise.
    pub fn scaled_matmul(
        &self,
        rhs: &Self,
        scale: &Self,
        rhs_scale: &Self,
        output: DataType,
    ) -> Result<Self> {
        call_one(
            Operation::ScaledMatMul,
            OperationAttributes::Cast(output),
            &[self, rhs, scale, rhs_scale],
        )
    }
}

impl Tensor {
    /// Matrix-multiplies tensors over their final two dimensions.
    /// CUDA supports broadcasting leading batch dimensions between operands of rank two or higher.
    /// CPU execution currently supports rank-two FP32 operands.
    /// Operands must share their device and datatype.
    ///
    /// This records a graph operation when called inside graph capture.
    ///
    /// # Errors
    ///
    /// Returns an error if shapes, devices, or data types are unsupported by the backend.
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        let lhs_vector = self.shape().rank() == 1;
        let rhs_vector = rhs.shape().rank() == 1;
        if lhs_vector || rhs_vector {
            let lhs = if lhs_vector {
                self.unsqueeze(0)?
            } else {
                self.clone()
            };
            let rhs = if rhs_vector {
                rhs.unsqueeze(-1)?
            } else {
                rhs.clone()
            };
            let mut output = lhs.matmul(&rhs)?;
            if lhs_vector {
                output = output.squeeze(-2)?;
            }

            if rhs_vector {
                output = output.squeeze(-1)?;
            }
            return Ok(output);
        }
        call_one(Operation::MatMul, OperationAttributes::None, &[self, rhs])
    }

    /// Matrix-multiplies two quantized rank-2 tensors and returns F32 output.
    ///
    /// Inputs are dequantized with `(input - zero_point) * scale` before the reference matmul.
    /// This records a single graph operation when called inside graph capture.
    ///
    /// # Errors
    ///
    /// Returns an error if quantization parameters are invalid or the backend cannot dequantize either input or multiply the dequantized tensors.
    pub fn quantized_matmul(
        &self,
        rhs: &Self,
        lhs_scale: f32,
        lhs_zero_point: i64,
        rhs_scale: f32,
        rhs_zero_point: i64,
    ) -> Result<Self> {
        let (lhs_scale, lhs_zero_point) =
            scalar_parameters(self.device(), lhs_scale, lhs_zero_point)?;
        let (rhs_scale, rhs_zero_point) =
            scalar_parameters(rhs.device(), rhs_scale, rhs_zero_point)?;
        call_one(
            Operation::QuantizedMatMul,
            OperationAttributes::None,
            &[
                self,
                rhs,
                &lhs_scale,
                &lhs_zero_point,
                &rhs_scale,
                &rhs_zero_point,
            ],
        )
    }

    /// Computes `alpha * contraction(self, b) + beta * c`.
    ///
    /// The mode arrays label dimensions in the input tensors and output tensor.
    /// This records a graph operation when called inside graph capture.
    ///
    /// # Errors
    ///
    /// Returns an error if shapes, modes, data types, devices, or backend capabilities are incompatible.
    pub fn contraction(
        &self,
        a_mode: &[i64],
        b: &Self,
        b_mode: &[i64],
        c: &Self,
        c_mode: &[i64],
        d_mode: &[i64],
        alpha: f32,
        beta: f32,
    ) -> Result<Self> {
        call_one(
            Operation::Contraction,
            OperationAttributes::Contraction(ContractionAttributes {
                a_mode: a_mode.to_vec(),
                b_mode: b_mode.to_vec(),
                c_mode: c_mode.to_vec(),
                d_mode: d_mode.to_vec(),
                alpha,
                beta,
            }),
            &[self, b, c],
        )
    }
}

impl Tensor {
    // Evaluates an Einstein summation such as `"bij,bjk->bik"` over `operands`: one operand transposes and sums, and several contract pairwise from the left, each contraction keeping the indices the output or a later operand reads.
    // Without `->`, the output holds the indices that appear once, in alphabetical order.
    // Ellipses and an index repeated within one operand, which takes a diagonal, are not supported.
    pub fn einsum(equation: &str, operands: &[&Self]) -> Result<Self> {
        let equation = equation.replace(' ', "");
        let (inputs, explicit) = match equation.split_once("->") {
            Some((inputs, output)) => (inputs, Some(output)),
            None => (equation.as_str(), None),
        };
        let terms = inputs.split(',').collect::<Vec<_>>();
        if terms.len() != operands.len() {
            return Err(Error::invalid_argument(
                "einsum names one subscript per operand",
            ));
        }

        let mut width = 0;
        let mut prepared = Vec::with_capacity(operands.len());
        for (term, operand) in terms.iter().zip(operands) {
            let (indices, ellipsis) = einsum_term(term, operand.shape().dimensions().len())?;
            width = width.max(ellipsis);
            prepared.push(without_repeated_indices((*operand).clone(), indices)?);
        }

        let output = einsum_output(inputs, explicit, width)?;
        // An extent of one broadcasts against any other, and so does the ellipsis's.
        let mut extents = BTreeMap::new();
        for (tensor, indices) in &prepared {
            for (&index, &extent) in indices.iter().zip(tensor.shape().dimensions()) {
                let known = extents.entry(index).or_insert(extent);
                if *known == 1 {
                    *known = extent;
                } else if extent != 1 && extent != *known {
                    return Err(Error::invalid_argument(
                        "einsum indices name dimensions of one extent",
                    ));
                }
            }
        }

        let prepared = prepared
            .into_iter()
            .map(|operand| without_broadcast_indices(operand, &extents))
            .collect::<Result<Vec<_>>>()?;
        if output.iter().any(|index| !extents.contains_key(index)) {
            return Err(Error::invalid_argument(
                "einsum outputs only indices its operands name",
            ));
        }

        let mode = |indices: &[char]| {
            indices
                .iter()
                .map(|&index| index as i64)
                .collect::<Vec<_>>()
        };
        let (first, rest) = prepared
            .split_first()
            .ok_or_else(|| Error::invalid_argument("einsum takes an operand"))?;
        let mut current = first.clone();
        for (position, (operand, next)) in rest.iter().enumerate() {
            let later = rest[position + 1..]
                .iter()
                .flat_map(|(_, indices)| indices.iter().copied())
                .collect::<Vec<_>>();
            let needed = |index: &char| output.contains(index) || later.contains(index);
            // An index only one side names, and nothing after needs, is summed before the contraction.
            current = sum_out(current, |index| next.contains(index) || needed(index))?;
            let (operand, subscript) = sum_out((operand.clone(), next.clone()), |index| {
                current.1.contains(index) || needed(index)
            })?;
            let subscript = &subscript;
            let kept = current
                .1
                .iter()
                .chain(subscript)
                .copied()
                .filter(|index| needed(index))
                .fold(Vec::new(), |mut kept, index| {
                    if !kept.contains(&index) {
                        kept.push(index);
                    }
                    kept
                });
            let shape = kept.iter().map(|index| extents[index]).collect::<Vec<_>>();
            let (device, data_type) = (first.0.device(), first.0.data_type());
            let accumulator = capture::literal(device, |device| {
                Self::zeroed_on(device, data_type, Shape::create(shape.clone())?)
            })?;
            let product = current.0.contraction(
                &mode(&current.1),
                &operand,
                &mode(subscript),
                &accumulator,
                &mode(&kept),
                &mode(&kept),
                1.0,
                0.0,
            )?;
            current = (product, kept);
        }

        // Indices the output does not name are summed, and the rest move into the output's order.
        let (tensor, indices) = current;
        let summed = indices
            .iter()
            .enumerate()
            .filter(|(_, index)| !output.contains(index))
            .map(|(axis, _)| i64::try_from(axis).map_err(|_| Error::DimensionOutOfBounds))
            .collect::<Result<Vec<_>>>()?;
        let remaining = indices
            .iter()
            .filter(|index| output.contains(index))
            .copied()
            .collect::<Vec<_>>();
        let tensor = if summed.is_empty() {
            tensor
        } else {
            tensor.sum(&summed, false)?
        };
        let order = output
            .iter()
            .map(|index| {
                remaining
                    .iter()
                    .position(|other| other == index)
                    .and_then(|axis| i64::try_from(axis).ok())
                    .ok_or(Error::DimensionOutOfBounds)
            })
            .collect::<Result<Vec<_>>>()?;
        if order.iter().copied().eq(0..) {
            Ok(tensor)
        } else {
            tensor.permute(&order)
        }
    }
}

// The first private-use code point, which names the ellipsis's dimensions counted from the right, since subscripts are letters.
const EINSUM_ELLIPSIS: u32 = 0xE000;

fn einsum_ellipsis_index(from_right: usize) -> Result<char> {
    u32::try_from(from_right)
        .ok()
        .and_then(|offset| char::from_u32(EINSUM_ELLIPSIS + offset))
        .ok_or_else(|| Error::out_of_range("einsum ellipsis"))
}

fn einsum_letters(letters: &str) -> Result<Vec<char>> {
    if letters.chars().all(|index| index.is_ascii_alphabetic()) {
        Ok(letters.chars().collect())
    } else {
        Err(Error::invalid_argument(
            "einsum subscripts are letters and one ellipsis",
        ))
    }
}

// The index of each of an operand's `rank` dimensions, and how many the term's ellipsis covers.
fn einsum_term(term: &str, rank: usize) -> Result<(Vec<char>, usize)> {
    let wrong_rank = || Error::invalid_argument("einsum subscripts name each operand dimension");
    let Some((before, after)) = term.split_once("...") else {
        let indices = einsum_letters(term)?;
        return if indices.len() == rank {
            Ok((indices, 0))
        } else {
            Err(wrong_rank())
        };
    };
    let (before, after) = (einsum_letters(before)?, einsum_letters(after)?);
    let width = rank
        .checked_sub(before.len() + after.len())
        .ok_or_else(wrong_rank)?;
    let mut indices = before;
    for from_right in (0..width).rev() {
        indices.push(einsum_ellipsis_index(from_right)?);
    }

    indices.extend(after);
    Ok((indices, width))
}

// The output's indices, where the implicit form puts the ellipsis first and then, sorted, the letters that appear once; an explicit output without an ellipsis sums its dimensions.
fn einsum_output(inputs: &str, explicit: Option<&str>, width: usize) -> Result<Vec<char>> {
    let ellipsis = (0..width)
        .rev()
        .map(einsum_ellipsis_index)
        .collect::<Result<Vec<_>>>()?;
    let Some(explicit) = explicit else {
        let letters = inputs.replace("...", "").replace(',', "");
        let mut once = letters
            .chars()
            .filter(|&index| letters.matches(index).count() == 1)
            .collect::<Vec<_>>();
        once.sort_unstable();
        let mut output = ellipsis;
        output.extend(once);
        return Ok(output);
    };
    let output = match explicit.split_once("...") {
        Some((before, after)) => {
            let mut output = einsum_letters(before)?;
            output.extend(ellipsis);
            output.extend(einsum_letters(after)?);
            output
        }
        None => einsum_letters(explicit)?,
    };
    let mut seen = BTreeSet::new();
    if output.iter().all(|&index| seen.insert(index)) {
        Ok(output)
    } else {
        Err(Error::invalid_argument(
            "einsum outputs each index at most once",
        ))
    }
}

// `tensor` with each index it repeats, as `ii` does, replaced by the diagonal of those dimensions.
fn without_repeated_indices(tensor: Tensor, indices: Vec<char>) -> Result<(Tensor, Vec<char>)> {
    let (mut tensor, mut indices) = (tensor, indices);
    loop {
        let repeated = indices.iter().enumerate().find_map(|(first, index)| {
            let second = indices[first + 1..]
                .iter()
                .position(|other| other == index)?;
            Some((first, first + 1 + second))
        });
        let Some((first, second)) = repeated else {
            return Ok((tensor, indices));
        };
        let dimensions = tensor.shape().dimensions();
        if dimensions[first] != dimensions[second] {
            return Err(Error::invalid_argument(
                "einsum indices name dimensions of one extent",
            ));
        }

        let axis = |axis: usize| i64::try_from(axis).map_err(|_| Error::DimensionOutOfBounds);
        tensor = tensor.diagonal(0, axis(first)?, axis(second)?)?;
        let index = indices.remove(second);
        indices.remove(first);
        indices.push(index);
    }
}

// `tensor` without the dimensions of extent one that broadcast against a longer one, since the other operand's index then runs over its extent alone.
fn without_broadcast_indices(
    (tensor, indices): (Tensor, Vec<char>),
    extents: &BTreeMap<char, i64>,
) -> Result<(Tensor, Vec<char>)> {
    let mut tensor = tensor;
    let mut kept = Vec::with_capacity(indices.len());
    for index in indices {
        let axis = i64::try_from(kept.len()).map_err(|_| Error::DimensionOutOfBounds)?;
        let extent = tensor.shape().dimensions()[kept.len()];
        if extent == 1 && extents.get(&index).is_some_and(|&known| known != 1) {
            tensor = tensor.squeeze(axis)?;
        } else {
            kept.push(index);
        }
    }
    Ok((tensor, kept))
}

// `tensor` with the dimensions its `indices` name summed away where `keep` rejects them, and the indices that remain.
fn sum_out(
    (tensor, indices): (Tensor, Vec<char>),
    keep: impl Fn(&char) -> bool,
) -> Result<(Tensor, Vec<char>)> {
    let summed = indices
        .iter()
        .enumerate()
        .filter(|(_, index)| !keep(index))
        .map(|(axis, _)| i64::try_from(axis).map_err(|_| Error::DimensionOutOfBounds))
        .collect::<Result<Vec<_>>>()?;
    if summed.is_empty() {
        return Ok((tensor, indices));
    }

    let remaining = indices
        .iter()
        .copied()
        .filter(|index| keep(index))
        .collect();
    Ok((tensor.sum(&summed, false)?, remaining))
}

// The block-scaled formats of cuBLASLt's block-scaled matmuls, as OCP's microscaling and NVIDIA's NVFP4 define them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockScaledFormat {
    // `F8E4M3` values, each block of 32 along the contraction scaled by a power of two, whose `U8` exponent is biased by 127, as E8M0 stores it.
    Mxfp8,
    // Pairs of E2M1 values packed in `U8` bytes, the low nibble first, each block of 16 along the contraction scaled by an `F8E4M3` scale.
    Nvfp4,
}

impl Tensor {
    // Multiplies block-scaled `self [M, K]` by block-scaled `rhs [N, K]`, both stored along the contraction as cuBLASLt's TN layout takes them, with block scales `[M, K / block]` and `[N, K / block]`, times `alpha`, into `output`; NVFP4 operands hold `K / 2` bytes a row.
    // Every path decodes both operands to `F32` and multiplies once, the product cuBLASLt's block-scaled modes compute on targets that run them.
    pub fn block_scaled_matmul(
        &self,
        rhs: &Self,
        scales: &Self,
        rhs_scales: &Self,
        format: BlockScaledFormat,
        alpha: f64,
        output: DataType,
    ) -> Result<Self> {
        let device = self.device();
        let decode = |values: &Self, scales: &Self| -> Result<Self> {
            match format {
                BlockScaledFormat::Mxfp8 => {
                    let exponent = scales.cast(DataType::F32)?.add_scalar(-127.0)?;
                    let shape = exponent.shape().clone();
                    let two = capture::literal(device, |device| {
                        Scalar::Float(2.0).full(device, DataType::F32, &shape)
                    })?;
                    values.dequantize_blocks(&two.pow(&exponent)?, (1, 32))
                }
                BlockScaledFormat::Nvfp4 => {
                    let (scale, zero_point) = scalar_parameters(device, 1.0, 0)?;
                    values
                        .dequantize_packed(&scale, &zero_point, None, PackedFormat::Float4E2M1)?
                        .dequantize_blocks(scales, (1, 16))
                }
            }
        };
        decode(self, scales)?
            .matmul(&decode(rhs, rhs_scales)?.transpose(0, 1)?)?
            .mul_scalar(alpha)?
            .cast(output)
    }
}

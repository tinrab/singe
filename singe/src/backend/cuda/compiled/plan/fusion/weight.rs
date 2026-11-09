//! The dequantized weights generated matmuls compute as they load them, and which matmuls run as generated kernels.

use std::{
    collections::{HashMap, HashSet},
    hash::BuildHasher,
};

use crate::{
    backend::{
        cuda::{
            compiled::plan::lowering::{CudaGraphProvider, LoweringPlan},
            provider_selection::{KernelChoice, Operand, matmul_kernel},
        },
        registry::{Operation, OperationAttributes, QuantizationAttributes},
    },
    data_type::DataType,
    error::Result,
    graph::ir::{Graph, ValueId},
};

/// A matmul's right operand computed by a dequantization the kernel performs as it reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DequantizedWeight {
    pub quantized: ValueId,
    pub scale: ValueId,
    pub zero_point: ValueId,
    pub attributes: QuantizationAttributes,
    /// The type a cast converts the dequantized weight to before the product.
    pub rounded: Option<DataType>,
    pub transposed: bool,
    /// The dequantization, cast, and transpose, which the fused matmul replaces.
    members: Vec<ValueId>,
}

impl DequantizedWeight {
    /// Returns the dequantization and the casts and transposes after it, which the fused matmul replaces.
    #[must_use]
    pub fn members(&self) -> &[ValueId] {
        &self.members
    }
}

/// Returns the matmuls of `ir` whose right operand a dequantization feeds alone and that run as generated kernels on `device`, with their weights; the members are nodes lowering skips.
///
/// # Errors
///
/// Returns an error if the graph's metadata is missing.
pub fn dequantized_weights<S: BuildHasher, T: BuildHasher>(
    ir: &Graph,
    execution_order: &[ValueId],
    users: &HashMap<ValueId, Vec<ValueId>, S>,
    graph_outputs: &HashSet<ValueId, T>,
    lowering_plan: &LoweringPlan,
    library_matmuls: Option<&HashSet<ValueId>>,
) -> Result<HashMap<ValueId, DequantizedWeight>> {
    let mut weights = HashMap::new();
    for &id in execution_order {
        let Ok(operation) = ir.required_operation_for_value(id) else {
            continue;
        };
        let (Some(Operation::Dequantize), Some(attributes), [quantized, scale, zero_point]) = (
            operation.opcode(),
            operation.attributes().quantization().copied(),
            operation.inputs(),
        ) else {
            continue;
        };
        if attributes.block_size.is_some() || ir.required_value_shape(*quantized)?.rank() != 2 {
            continue;
        }

        let mut members = vec![id];
        let (mut rounded, mut transposed) = (None, false);
        let mut current = id;
        let matmul = loop {
            let (&[reader], false) = (
                users.get(&current).map_or(&[][..], Vec::as_slice),
                graph_outputs.contains(&current),
            ) else {
                break None;
            };
            let next = ir.required_operation_for_value(reader)?;
            match (next.opcode(), next.attributes()) {
                (Some(Operation::Cast), &OperationAttributes::Cast(data_type))
                    if rounded.is_none() =>
                {
                    rounded = Some(data_type);
                }
                (Some(Operation::Transpose), attributes) if !transposed => {
                    let Some(transpose) = attributes.transpose() else {
                        break None;
                    };
                    let mut dims = [transpose.dim0, transpose.dim1].map(|dim| dim.rem_euclid(2));
                    dims.sort_unstable();
                    if dims != [0, 1] {
                        break None;
                    }

                    transposed = true;
                }
                (Some(Operation::MatMul), _) if next.inputs().get(1) == Some(&current) => {
                    break Some(reader);
                }
                _ => break None,
            }

            members.push(reader);
            current = reader;
        };
        let Some(matmul) = matmul else {
            continue;
        };
        if !generated(ir, matmul, lowering_plan, library_matmuls)? {
            continue;
        }

        weights.insert(
            matmul,
            DequantizedWeight {
                quantized: *quantized,
                scale: *scale,
                zero_point: *zero_point,
                attributes,
                rounded,
                transposed,
                members,
            },
        );
    }
    Ok(weights)
}

/// Returns whether matmul `id` runs as a generated kernel: the lowering policy allows no library for it, or provider tuning, measuring its operands now when it has not, chose the generated kernel over the libraries.
pub fn generated(
    ir: &Graph,
    id: ValueId,
    lowering_plan: &LoweringPlan,
    library_matmuls: Option<&HashSet<ValueId>>,
) -> Result<bool> {
    let operation = ir.required_operation_for_value(id)?;
    let &[lhs, rhs] = operation.inputs() else {
        return Ok(false);
    };
    if !lowering_plan.should_try_provider(id, CudaGraphProvider::Kernel) {
        return Ok(false);
    }

    // A program decides which matmuls its libraries run before it plans fusion.
    if let Some(library_matmuls) = library_matmuls {
        return Ok(!library_matmuls.contains(&id));
    }

    if ![
        CudaGraphProvider::Cublas,
        CudaGraphProvider::CublasLt,
        CudaGraphProvider::CudnnBackend,
    ]
    .iter()
    .any(|&provider| lowering_plan.should_try_provider(id, provider))
    {
        return Ok(true);
    }

    let operand = |value: ValueId| -> Result<Operand<'_>> {
        Ok(Operand {
            data_type: ir.required_value_data_type(value)?,
            layout: ir.required_value_layout(value)?,
        })
    };
    let (lhs, rhs) = (operand(lhs)?, operand(rhs)?);
    if lhs.data_type != rhs.data_type {
        return Ok(false);
    }
    Ok(matmul_kernel(lhs, rhs) == KernelChoice::General)
}

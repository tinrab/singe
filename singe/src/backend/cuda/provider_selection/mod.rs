//! The kernel choice that eager CUDA calls and compiled-graph lowering share.
//!
//! Both paths describe an operation's operands as [`Operand`] values and ask the same function which kernel runs it, so they cannot disagree on what a kernel accepts.
//! The general kernel runs operations unless a library measures faster for their operands.
//! Compiled graphs may still replace a node with a library fusion, which [`KernelChoice`] does not describe.

use serde::{Deserialize, Serialize};

use crate::{
    backend::{cuda::gemm::GemmLayout, registry::Operation},
    data_type::DataType,
    layout::Layout,
    tensor::Tensor,
};

/// The data type and layout of one operand, read from an eager tensor or a graph value.
#[derive(Debug, Clone, Copy)]
pub struct Operand<'a> {
    pub data_type: DataType,
    pub layout: &'a Layout,
}

impl<'a> Operand<'a> {
    /// Describes an eager tensor.
    pub fn of(tensor: &'a Tensor) -> Self {
        Self {
            data_type: tensor.data_type(),
            layout: tensor.layout(),
        }
    }
}

/// The kernel that runs one operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KernelChoice {
    /// A generated kernel over strided, broadcast operands of any floating type.
    General,
    /// A library provider, such as cuDNN, for operations the general kernel does not cover.
    Library,
}

/// Returns whether `operation` runs through [`elementwise_kernel`]; `copy` has its own kernels.
#[must_use]
pub fn is_elementwise(operation: Operation) -> bool {
    (operation.is_unary_elementwise() && operation != Operation::Copy)
        || operation.is_binary_elementwise()
        || operation.is_scalar_elementwise()
        || matches!(
            operation,
            Operation::Clamp
                | Operation::Lerp
                | Operation::Where
                | Operation::Cast
                | Operation::SwiGlu
        )
}

/// Returns the kernel that runs `lhs @ rhs` on `device`: a cuBLAS GEMM for operands of one `F16`, `BF16`, or `F32` type that it reads in place, and the general kernel for every other layout and type.
///
/// Compiled graphs may pick another library for a [`KernelChoice::Library`] choice, such as cuBLASLt, and tuning records may pick another kernel for them.
#[must_use]
pub fn matmul_kernel(lhs: Operand<'_>, rhs: Operand<'_>) -> KernelChoice {
    if library_matmul_accepts(lhs, rhs) {
        KernelChoice::Library
    } else {
        KernelChoice::General
    }
}

/// Returns whether cuBLAS runs `lhs @ rhs` as one strided batched GEMM, so both it and the general kernel are candidates.
#[must_use]
pub fn library_matmul_accepts(lhs: Operand<'_>, rhs: Operand<'_>) -> bool {
    lhs.data_type == rhs.data_type
        && matches!(
            lhs.data_type,
            DataType::F16 | DataType::Bf16 | DataType::F32
        )
        && GemmLayout::create(lhs.layout, rhs.layout).is_some()
}

#[cfg(test)]
mod tests;

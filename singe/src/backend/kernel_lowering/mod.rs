//! Lowers graph operations to `singe-codegen` kernels, so code generation sees what operations compute instead of how a target writes them.

mod domain;

mod element_wise;
mod epilogue;
mod expression;
mod gather;
mod group;
mod matmul;
mod quantization;
mod reduction;
mod rotary;
mod window;

use singe_codegen::ir::ElementType;

pub use self::{
    domain::{ArgumentLabels, LoweredKernel},
    element_wise::{ElementWiseKernel, copy},
    epilogue::{EpilogueNode, EpilogueValue, RowEpilogue},
    gather::{Selection, gather, scatter},
    group::{
        ElementGroup, FusedOperand, FusedStep, FusedStores, is_fusible, is_row_opcode,
        is_row_operation,
    },
    matmul::{Epilogue, Lhs, Weight, contraction, matmul},
    quantization::quantization,
    reduction::{batch_norm, normalization, normalization_with_epilogue, reduction, scan, softmax},
    rotary::rotary_embedding,
    window::{conv2d, pool2d},
};

use crate::{
    data_type::DataType,
    error::{Error, Result},
};

/// The status code a kernel reports for an index outside its dimension.
pub const STATUS_INDEX_OUT_OF_BOUNDS: u32 = 1;

/// The status code a kernel reports for a quantization scale that is not finite and positive.
pub const STATUS_INVALID_QUANTIZATION_SCALE: u32 = 2;

/// The status code a kernel reports for an integer division or remainder by zero.
pub const STATUS_DIVISION_BY_ZERO: u32 = 3;

/// Returns the error a kernel's status word `code` records, or nothing for zero.
///
/// # Errors
///
/// Returns the error `code` names: an index out of bounds, an invalid quantization scale, a division by zero, or an unknown code.
pub fn check_status_code(code: u32) -> Result<()> {
    match code {
        0 => Ok(()),
        STATUS_INDEX_OUT_OF_BOUNDS => Err(Error::IndexOutOfBounds),
        STATUS_INVALID_QUANTIZATION_SCALE => Err(Error::invalid_argument("quantization scale")),
        STATUS_DIVISION_BY_ZERO => Err(Error::invalid_argument("integer division by zero")),
        _ => Err(Error::invalid_argument("kernel status code")),
    }
}

/// Returns the kernel element type of `data_type`.
///
/// # Errors
///
/// Returns an error for the packed 4-bit floating type, which kernels cannot hold yet.
pub fn element_type(data_type: DataType) -> Result<ElementType> {
    Ok(match data_type {
        DataType::Bool => ElementType::Bool,
        DataType::U8 => ElementType::U8,
        DataType::I8 => ElementType::I8,
        DataType::U16 => ElementType::U16,
        DataType::I16 => ElementType::I16,
        DataType::U32 => ElementType::U32,
        DataType::I32 => ElementType::I32,
        DataType::U64 => ElementType::U64,
        DataType::I64 => ElementType::I64,
        DataType::F16 => ElementType::F16,
        DataType::Bf16 => ElementType::Bf16,
        DataType::F32 => ElementType::F32,
        DataType::F64 => ElementType::F64,
        DataType::F8E4M3 => ElementType::F8E4M3,
        DataType::F8E5M2 => ElementType::F8E5M2,
        DataType::F4E2M1 => {
            return Err(Error::unsupported_data_type("generated kernel element"));
        }
    })
}

#[cfg(test)]
mod tests;

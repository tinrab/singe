//! The `sm_80` matrix load and multiply instructions.

use std::{array, sync::Arc};

use crate::{
    ast::{Instruction, Operand, ScalarType, StateSpace},
    emulator::{
        compiler::FunctionCompiler,
        error::{CompileError, Result},
        ir::{self, CollectiveOperation, Operation},
        lowering::{Head, operands, require},
    },
};

pub fn load(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    let span = instruction.span;
    require(
        head.take("sync") && head.take("aligned") && head.take("m8n8"),
        span,
        "matrix load requires sync.aligned.m8n8",
    )?;
    let [destination, address] = operands(instruction)?;
    let layout = if head.take("x4") {
        ir::MatrixLoad::Rows4(destinations(compiler, destination, ScalarType::B32)?)
    } else {
        require(
            head.take("x2") && head.take("trans"),
            span,
            "matrix load supports x4 or x2.trans",
        )?;
        ir::MatrixLoad::Columns2(destinations(compiler, destination, ScalarType::B32)?)
    };
    require(
        head.space() == Some(StateSpace::Shared) && head.ty()? == ScalarType::B16,
        span,
        "matrix load requires shared.b16",
    )?;
    Ok(Operation::Collective(CollectiveOperation::MatrixLoad {
        layout,
        address: compiler.address(address, Some(StateSpace::Shared), false)?,
    }))
}

pub fn multiply(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    let span = instruction.span;
    require(
        head.take("sync")
            && head.take("aligned")
            && head.take("m16n8k16")
            && head.take("row")
            && head.take("col"),
        span,
        "matrix multiply requires sync.aligned.m16n8k16.row.col",
    )?;
    require(
        head.ty()? == ScalarType::F32
            && head.ty()? == ScalarType::F16
            && head.ty()? == ScalarType::F16
            && head.ty()? == ScalarType::F32,
        span,
        "matrix multiply requires f32.f16.f16.f32",
    )?;
    let [d, a, b, c] = operands(instruction)?;
    Ok(Operation::Collective(CollectiveOperation::MatrixMultiply(
        Arc::new(ir::MatrixMultiply {
            destinations: destinations(compiler, d, ScalarType::F32)?,
            a: sources(compiler, a, ScalarType::B32)?,
            b: sources(compiler, b, ScalarType::B32)?,
            c: sources(compiler, c, ScalarType::F32)?,
        }),
    )))
}

fn vector<const N: usize>(operand: &Operand) -> Result<&[Operand; N]> {
    let Operand::Vector { elements, .. } = operand else {
        return Err(CompileError::instruction(
            operand.span(),
            "matrix operand requires a vector",
        ));
    };
    elements.as_slice().try_into().map_err(|_| {
        CompileError::instruction(operand.span(), "matrix vector has the wrong length")
    })
}

fn destinations<const N: usize>(
    compiler: &FunctionCompiler<'_, '_>,
    operand: &Operand,
    ty: ScalarType,
) -> Result<[ir::Destination; N]> {
    let elements = vector::<N>(operand)?;
    let mut registers = [ir::Destination::Discard; N];
    for (register, element) in registers.iter_mut().zip(elements) {
        *register = compiler.destination(element, ty)?;
    }
    Ok(registers)
}

fn sources<const N: usize>(
    compiler: &FunctionCompiler<'_, '_>,
    operand: &Operand,
    ty: ScalarType,
) -> Result<[ir::Source; N]> {
    let elements = vector::<N>(operand)?;
    let mut sources = array::from_fn(|_| ir::Source::Immediate(0));
    for (source, element) in sources.iter_mut().zip(elements) {
        *source = compiler.source(element, ty)?;
    }
    Ok(sources)
}

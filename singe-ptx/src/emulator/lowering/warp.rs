//! Warp-wide shuffles, votes, and reductions.

use crate::{
    ast::{Instruction, ScalarType},
    emulator::{
        compiler::FunctionCompiler,
        error::{CompileError, Result},
        ir::{self, CollectiveOperation, Operation},
        lowering::{Head, operands, paired_destination, require},
    },
};

/// Lowers `shfl.sync`.
pub fn shuffle(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    let span = instruction.span;
    require(
        head.take("sync"),
        span,
        "only synchronized shuffle is supported",
    )?;
    let mode = match head.next()? {
        "up" => ir::Shuffle::Up,
        "down" => ir::Shuffle::Down,
        "bfly" => ir::Shuffle::Butterfly,
        "idx" => ir::Shuffle::Index,
        _ => return Err(CompileError::instruction(span, "invalid shuffle mode")),
    };
    require(head.ty()? == ScalarType::B32, span, "shuffle requires b32")?;
    let [d, a, b, c, mask] = operands(instruction)?;
    let (destination, valid) = paired_destination(compiler, d, ScalarType::B32)?;
    Ok(Operation::Collective(CollectiveOperation::Shuffle {
        mode,
        destination,
        valid,
        value: compiler.source(a, ScalarType::B32)?,
        lane: compiler.source(b, ScalarType::U32)?,
        control: compiler.source(c, ScalarType::U32)?,
        mask: compiler.source(mask, ScalarType::B32)?,
    }))
}

/// Lowers `vote.sync`.
pub fn vote(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    let span = instruction.span;
    require(
        head.take("sync"),
        span,
        "only synchronized vote is supported",
    )?;
    let mode = match head.next()? {
        "all" => ir::Vote::All,
        "any" => ir::Vote::Any,
        "uni" => ir::Vote::Uniform,
        "ballot" => ir::Vote::Ballot,
        _ => return Err(CompileError::instruction(span, "invalid vote mode")),
    };
    let ty = head.ty()?;
    let expected = if mode == ir::Vote::Ballot {
        ScalarType::B32
    } else {
        ScalarType::Pred
    };
    require(
        ty == expected,
        span,
        "vote destination type does not match its mode",
    )?;
    let [d, a, mask] = operands(instruction)?;
    Ok(Operation::Collective(CollectiveOperation::Vote {
        mode,
        destination: compiler.destination(d, ty)?,
        predicate: compiler.predicate(a)?,
        mask: compiler.source(mask, ScalarType::B32)?,
    }))
}

pub fn active_mask(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    require(
        head.ty()? == ScalarType::B32,
        instruction.span,
        "activemask requires b32",
    )?;
    let [d] = operands(instruction)?;
    Ok(Operation::Collective(CollectiveOperation::ActiveMask {
        destination: compiler.destination(d, ScalarType::B32)?,
    }))
}

/// Lowers `redux.sync.add.u32`.
pub fn reduce(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    let span = instruction.span;
    require(head.take("sync"), span, "redux requires sync")?;
    require(head.take("add"), span, "redux requires add")?;
    require(
        head.ty()? == ScalarType::U32,
        span,
        "redux add requires u32",
    )?;
    let [d, value, mask] = operands(instruction)?;
    Ok(Operation::Collective(CollectiveOperation::ReduceAdd {
        destination: compiler.destination(d, ScalarType::U32)?,
        value: compiler.source(value, ScalarType::U32)?,
        mask: compiler.source(mask, ScalarType::B32)?,
    }))
}

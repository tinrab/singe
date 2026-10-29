//! Branches, calls, returns, and CTA barriers.

use crate::{
    ast::{Instruction, Operand, ScalarType},
    emulator::{
        compiler::FunctionCompiler,
        error::{CompileError, Result},
        ir::{self, CollectiveOperation, ControlOperation, Operation},
        lowering::{Head, Opcode, operands, require},
        target::{MAX_BLOCK_THREADS, NAMED_BARRIERS, WARP_SIZE},
    },
};

pub fn branch(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    head.take("uni");
    let [label] = operands(instruction)?;
    Ok(Operation::Control(ControlOperation::Branch {
        target: compiler.branch_target(label)?,
    }))
}

/// Lowers a direct `call` with optional return and argument tuples.
pub fn call(
    compiler: &mut FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    head.take("uni");
    let (returns, function, arguments) = match instruction.operands.as_slice() {
        [
            Operand::Tuple {
                elements: returns, ..
            },
            function,
            Operand::Tuple {
                elements: arguments,
                ..
            },
        ] => (returns.as_slice(), function, arguments.as_slice()),
        [
            Operand::Tuple {
                elements: returns, ..
            },
            function,
        ] => (returns.as_slice(), function, &[][..]),
        [
            function,
            Operand::Tuple {
                elements: arguments,
                ..
            },
        ] => (&[][..], function, arguments.as_slice()),
        [function] => (&[][..], function, &[][..]),
        _ => {
            return Err(CompileError::instruction(
                instruction.span,
                "call requires an optional return tuple, direct function, and optional argument tuple",
            ));
        }
    };
    compiler.call(function, arguments, returns)
}

pub fn ret(instruction: &Instruction, head: &mut Head<'_>) -> Result<Operation> {
    head.take("uni");
    operands::<0>(instruction)?;
    Ok(Operation::Control(ControlOperation::Return))
}

/// Lowers `exit`, `trap`, and `brkpt`.
pub fn terminate(instruction: &Instruction, opcode: Opcode) -> Result<Operation> {
    operands::<0>(instruction)?;
    Ok(match opcode {
        Opcode::Exit => Operation::Control(ControlOperation::Exit),
        Opcode::Trap => Operation::Control(ControlOperation::Trap),
        _ => Operation::Control(ControlOperation::Breakpoint),
    })
}

pub fn barrier(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
    opcode: Opcode,
) -> Result<Operation> {
    if opcode == Opcode::Bar && head.take("warp") {
        require(head.take("sync"), instruction.span, "expected warp sync")?;
        let [mask] = operands(instruction)?;
        return Ok(Operation::Collective(CollectiveOperation::WarpBarrier {
            mask: compiler.source(mask, ScalarType::B32)?,
        }));
    }
    head.take("cta");
    require(
        head.take("sync"),
        instruction.span,
        "only sync CTA barriers are supported",
    )?;
    if opcode != Opcode::Bar {
        head.take("aligned");
    }
    let (barrier, count) = match instruction.operands.as_slice() {
        [barrier] => (barrier, None),
        [barrier, count] => (barrier, Some(compiler.source(count, ScalarType::U32)?)),
        _ => {
            return Err(CompileError::instruction(
                instruction.span,
                "barrier expects an id and optional thread count",
            ));
        }
    };
    let barrier = compiler.source(barrier, ScalarType::U32)?;
    if let ir::Source::Immediate(id) = barrier {
        require(
            id < NAMED_BARRIERS as u128,
            instruction.span,
            "barrier id must be less than 16",
        )?;
    }
    if let Some(ir::Source::Immediate(count)) = count {
        require(
            count <= u128::from(MAX_BLOCK_THREADS) && count.is_multiple_of(WARP_SIZE as u128),
            instruction.span,
            "barrier count must be a warp multiple at most 1024",
        )?;
    }
    Ok(Operation::Collective(CollectiveOperation::CtaBarrier {
        barrier,
        count,
    }))
}

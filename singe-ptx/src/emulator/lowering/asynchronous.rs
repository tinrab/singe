//! Asynchronous copies and memory barriers (`cp.async`, `mbarrier`).

use crate::{
    ast::{Instruction, Operand, ScalarType, StateSpace},
    emulator::{
        compiler::FunctionCompiler,
        error::{CompileError, Result},
        ir::{self, AsyncOperation, Operation},
        lowering::{Head, operands, require},
        target::PtxVersion,
    },
};

pub fn copy(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    let span = instruction.span;
    require(head.take("async"), span, "expected asynchronous copy")?;
    if head.take("commit_group") {
        let [] = operands(instruction)?;
        return Ok(Operation::Async(AsyncOperation::CommitCopies));
    }
    if head.take("wait_all") {
        let [] = operands(instruction)?;
        return Ok(Operation::Async(AsyncOperation::WaitCopies {
            pending: 0,
            commit: true,
        }));
    }
    if head.take("wait_group") {
        let [pending] = operands(instruction)?;
        let pending = compiler.source(pending, ScalarType::U32)?;
        let ir::Source::Immediate(pending) = pending else {
            return Err(CompileError::instruction(
                span,
                "copy wait count must be an immediate",
            ));
        };
        return Ok(Operation::Async(AsyncOperation::WaitCopies {
            pending: u32::try_from(pending)
                .map_err(|_| CompileError::instruction(span, "copy wait count exceeds u32"))?,
            commit: false,
        }));
    }
    if head.take("mbarrier") {
        return arrive(compiler, instruction, head);
    }
    transfer(compiler, instruction, head)
}

/// Lowers `cp.async.mbarrier.arrive`.
fn arrive(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    let span = instruction.span;
    require(
        head.take("arrive"),
        span,
        "expected asynchronous barrier arrival",
    )?;
    let no_increment = head.take("noinc");
    let space = head.space();
    require(
        matches!(space, None | Some(StateSpace::Shared)),
        span,
        "mbarrier requires shared storage",
    )?;
    require(head.ty()? == ScalarType::B64, span, "mbarrier requires b64")?;
    let [address] = operands(instruction)?;
    Ok(Operation::Async(AsyncOperation::TrackCopies {
        address: compiler.address(address, space, true)?,
        space,
        no_increment,
    }))
}

/// Lowers the `cp.async.ca` and `cp.async.cg` copy itself.
fn transfer(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    let span = instruction.span;
    let global_cache_only = if head.take("ca") {
        false
    } else {
        require(
            head.take("cg"),
            span,
            "copy requires ca or cg cache qualifier",
        )?;
        true
    };
    require(
        head.space() == Some(StateSpace::Shared),
        span,
        "copy destination requires shared storage",
    )?;
    require(
        head.space() == Some(StateSpace::Global),
        span,
        "copy source requires global storage",
    )?;
    let (destination, source, size, input) = match instruction.operands.as_slice() {
        [destination, source, size] => (destination, source, size, None),
        [destination, source, size, input] => (destination, source, size, Some(input)),
        _ => {
            return Err(CompileError::instruction(
                span,
                "copy expects three or four operands",
            ));
        }
    };
    let ir::Source::Immediate(size @ (4 | 8 | 16)) = compiler.source(size, ScalarType::U32)? else {
        return Err(CompileError::instruction(
            span,
            "copy size must be an immediate 4, 8, or 16",
        ));
    };
    require(
        !global_cache_only || size == 16,
        span,
        "cg copy size must be 16",
    )?;
    let input = input
        .map(|operand| {
            if let Ok(bytes) = compiler.source(operand, ScalarType::U32) {
                if let ir::Source::Immediate(bytes) = bytes {
                    require(bytes <= size, span, "copy source size exceeds copy size")?;
                }
                Ok(ir::AsyncCopyInput::Bytes(bytes))
            } else {
                compiler.require_version(PtxVersion::new(7, 5), span)?;
                Ok(ir::AsyncCopyInput::Ignore(compiler.predicate(operand)?))
            }
        })
        .transpose()?;
    Ok(Operation::Async(AsyncOperation::Copy(ir::AsyncCopy {
        destination: compiler.address(destination, Some(StateSpace::Shared), true)?,
        source: compiler.address(source, Some(StateSpace::Global), false)?,
        size: size as usize,
        input,
    })))
}
pub fn barrier(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    let span = instruction.span;
    let operation = head.next()?;
    let parity = operation == "test_wait" && head.take("parity");
    if parity {
        compiler.require_version(PtxVersion::new(7, 1), span)?;
    }
    let space = head.space();
    require(
        matches!(space, None | Some(StateSpace::Shared)),
        span,
        "mbarrier requires shared storage",
    )?;
    require(head.ty()? == ScalarType::B64, span, "mbarrier requires b64")?;
    Ok(match operation {
        "init" => {
            let [address, count] = operands(instruction)?;
            let count = compiler.source(count, ScalarType::U32)?;
            if let ir::Source::Immediate(count) = count {
                require(
                    (1..1 << 20).contains(&count),
                    span,
                    "mbarrier count must be in 1..=1048575",
                )?;
            }
            Operation::Async(AsyncOperation::BarrierInit {
                address: compiler.address(address, space, true)?,
                space,
                count,
            })
        }
        "inval" => {
            let [address] = operands(instruction)?;
            Operation::Async(AsyncOperation::BarrierInvalidate {
                address: compiler.address(address, space, true)?,
                space,
            })
        }
        "arrive" => {
            let [destination, address] = operands(instruction)?;
            let destination = compiler.destination(destination, ScalarType::B64)?;
            if destination == ir::Destination::Discard {
                compiler.require_version(PtxVersion::new(7, 1), span)?;
            }
            Operation::Async(AsyncOperation::BarrierArrive {
                destination,
                address: compiler.address(address, space, true)?,
                space,
            })
        }
        "test_wait" => {
            let [destination, address, state] = operands(instruction)?;
            require(
                !matches!(destination, Operand::BitBucket { .. }),
                span,
                "barrier wait requires a predicate destination",
            )?;
            Operation::Async(AsyncOperation::BarrierTest {
                destination: compiler.destination(destination, ScalarType::Pred)?,
                address: compiler.address(address, space, true)?,
                space,
                state: compiler.source(
                    state,
                    if parity {
                        ScalarType::U32
                    } else {
                        ScalarType::B64
                    },
                )?,
                parity,
            })
        }
        _ => {
            return Err(CompileError::unsupported(
                span,
                "unsupported mbarrier operation",
            ));
        }
    })
}

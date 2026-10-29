//! Loads, stores, atomics, and fences.

use std::slice;

use crate::{
    ast::{Instruction, Operand, ScalarType, StateSpace},
    emulator::{
        compiler::FunctionCompiler,
        error::{CompileError, Result},
        ir::{self, MemoryOperation, Operation},
        layout::scalar_bytes,
        lowering::{Head, Opcode, operands, require},
    },
};

/// Memory ordering qualifiers. The interpreter serializes every access, so orderings
/// are validated here and not kept in the executable form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ordering {
    Weak,
    Volatile,
    Relaxed,
    Acquire,
    Release,
    AcquireRelease,
}

pub fn access(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
    write: bool,
) -> Result<Operation> {
    let ordering = access_ordering(head, write)?;
    let space = head.space();
    require(
        !write || space != Some(StateSpace::Const),
        instruction.span,
        "stores cannot target constant memory",
    )?;
    if ordering != Ordering::Weak {
        require(
            matches!(space, None | Some(StateSpace::Global | StateSpace::Shared)),
            instruction.span,
            "memory ordering requires generic, global, or shared addressing",
        )?;
    }
    let count = if head.take("v2") {
        2
    } else if head.take("v4") {
        4
    } else {
        1
    };
    let ty = head.ty()?;
    require(
        ty != ScalarType::Pred,
        instruction.span,
        "predicates cannot be loaded or stored",
    )?;
    require(
        !matches!(ty, ScalarType::Bf16 | ScalarType::F16 | ScalarType::F16x2),
        instruction.span,
        "half data uses bit types for loads and stores",
    )?;
    let bytes = scalar_bytes(ty, instruction.span)?;
    require(
        bytes * count <= 16,
        instruction.span,
        "memory vector exceeds 128 bits",
    )?;
    let [first, second] = operands(instruction)?;
    let (address, value) = if write {
        (first, second)
    } else {
        (second, first)
    };
    let address = compiler.address(address, space, write)?;
    let values = if count == 1 {
        slice::from_ref(value)
    } else {
        let Operand::Vector { elements, .. } = value else {
            return Err(CompileError::instruction(
                value.span(),
                "vector memory access requires a vector operand",
            ));
        };
        require(
            elements.len() == count as usize,
            value.span(),
            "memory vector length does not match its qualifier",
        )?;
        elements.as_slice()
    };
    if write {
        let sources = values
            .iter()
            .map(|value| compiler.wide_source(value, ty))
            .collect::<Result<_>>()?;
        Ok(Operation::Memory(MemoryOperation::Store {
            ty,
            space,
            address,
            sources,
        }))
    } else {
        let destinations = values
            .iter()
            .map(|value| compiler.wide_destination(value, ty))
            .collect::<Result<_>>()?;
        Ok(Operation::Memory(MemoryOperation::Load {
            ty,
            space,
            destinations,
            address,
        }))
    }
}

/// Consumes the ordering and, for relaxed, acquire, and release accesses, the scope of
/// `ld` or `st`.
fn access_ordering(head: &mut Head<'_>, write: bool) -> Result<Ordering> {
    let ordering = if head.take("volatile") {
        Ordering::Volatile
    } else if head.take("relaxed") {
        Ordering::Relaxed
    } else if !write && head.take("acquire") {
        Ordering::Acquire
    } else if write && head.take("release") {
        Ordering::Release
    } else {
        head.take("weak");
        Ordering::Weak
    };
    if matches!(
        ordering,
        Ordering::Relaxed | Ordering::Acquire | Ordering::Release
    ) {
        require(
            head.take_scope(),
            head.span,
            "ordered memory access requires a scope",
        )?;
    }
    Ok(ordering)
}

pub fn atomic(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
    returns_old: bool,
) -> Result<Operation> {
    use ir::Atomic::{Add, And, CompareSwap, Decrement, Exchange, Increment, Max, Min, Or, Xor};

    use ScalarType::{B32, B64, F32, S32, S64, U32, U64};

    let ordering = if head.take("acquire") {
        Ordering::Acquire
    } else if head.take("release") {
        Ordering::Release
    } else if head.take("acq_rel") {
        Ordering::AcquireRelease
    } else {
        head.take("relaxed");
        Ordering::Relaxed
    };
    head.take_scope();
    let space = head.space();
    require(
        matches!(space, None | Some(StateSpace::Global | StateSpace::Shared)),
        instruction.span,
        "atomic operation requires generic, global, or shared addressing",
    )?;
    let name = head.next()?;
    let ty = head.ty()?;
    let valid = match name {
        "add" => matches!(ty, U32 | S32 | U64) || (ty == F32 && space == Some(StateSpace::Global)),
        "min" | "max" => matches!(ty, U32 | S32 | U64 | S64),
        "inc" | "dec" => ty == U32,
        "and" | "or" | "xor" | "exch" | "cas" => matches!(ty, B32 | B64),
        _ => {
            return Err(CompileError::unsupported(
                instruction.span,
                "atomic operation",
            ));
        }
    };
    require(
        valid,
        instruction.span,
        "atomic type does not match its operation",
    )?;
    let update = |name| match name {
        "add" => Add,
        "min" => Min,
        "max" => Max,
        "and" => And,
        "or" => Or,
        "xor" => Xor,
        "inc" => Increment,
        "dec" => Decrement,
        _ => Exchange,
    };
    let (op, destination, address, value) = if !returns_old {
        require(
            !matches!(name, "exch" | "cas"),
            instruction.span,
            "red does not support exchange or compare-swap",
        )?;
        require(
            matches!(ordering, Ordering::Relaxed | Ordering::Release),
            instruction.span,
            "red requires relaxed or release ordering",
        )?;
        let [address, value] = operands(instruction)?;
        (update(name), ir::Destination::Discard, address, value)
    } else if name == "cas" {
        let [d, address, compare, value] = operands(instruction)?;
        let compare = compiler.source(compare, ty)?;
        (
            CompareSwap { compare },
            compiler.destination(d, ty)?,
            address,
            value,
        )
    } else {
        let [d, address, value] = operands(instruction)?;
        (update(name), compiler.destination(d, ty)?, address, value)
    };
    Ok(Operation::Memory(MemoryOperation::Atomic {
        op,
        ty,
        space,
        destination,
        address: compiler.address(address, space, true)?,
        value: compiler.source(value, ty)?,
    }))
}

/// Lowers `membar` and `fence`; the interpreter already serializes every access.
pub fn fence(instruction: &Instruction, head: &mut Head<'_>, opcode: Opcode) -> Result<Operation> {
    operands::<0>(instruction)?;
    if opcode == Opcode::Membar {
        require(
            matches!(head.next()?, "cta" | "gl" | "sys"),
            instruction.span,
            "invalid membar scope",
        )?;
    } else {
        if !head.take("sc") {
            head.take("acq_rel");
        }
        require(
            head.take_scope(),
            instruction.span,
            "fence requires a scope",
        )?;
    }
    Ok(Operation::Memory(MemoryOperation::Fence))
}

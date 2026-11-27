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
        target::Architecture,
    },
};

/// Memory ordering qualifiers.
/// The interpreter serializes every access, so orderings are validated here and not kept in the executable form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ordering {
    Weak,
    Volatile,
    Relaxed,
    Acquire,
    Release,
    AcquireRelease,
}

pub(crate) fn access(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
    write: bool,
) -> Result<Operation> {
    let ordering = access_ordering(head, write)?;
    let space = head.space();
    // `ld.global.nc` reads through the non-coherent cache, which only changes caching of data the kernel does not write.
    if !write && space == Some(StateSpace::Global) && head.take("nc") {
        require(
            ordering == Ordering::Weak,
            instruction.span,
            "non-coherent loads cannot be ordered",
        )?;
    }

    require(
        !write || space != Some(StateSpace::Const),
        instruction.span,
        "stores cannot target constant memory",
    )?;
    if !matches!(ordering, Ordering::Weak | Ordering::Volatile) {
        require(
            matches!(space, None | Some(StateSpace::Global | StateSpace::Shared)),
            instruction.span,
            "memory ordering requires generic, global, or shared addressing",
        )?;
    }

    take_cache_hints(head, write);

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

/// Consumes cache operators and eviction hints, which change only where data is cached.
fn take_cache_hints(head: &mut Head<'_>, write: bool) {
    let operators: &[&str] = if write {
        &["wb", "cg", "cs", "wt"]
    } else {
        &["ca", "cg", "cs", "lu", "cv"]
    };
    operators.iter().any(|operator| head.take(operator));
    while let Some(part) = head.parts.first()
        && (part.starts_with("L1::evict_")
            || part.starts_with("L2::evict_")
            || matches!(
                *part,
                "L1::no_allocate" | "L2::64B" | "L2::128B" | "L2::256B"
            ))
    {
        head.parts = &head.parts[1..];
    }
}

/// Consumes the ordering and, for relaxed, acquire, and release accesses, the scope of `ld` or `st`.
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

fn atomic_ordering(head: &mut Head<'_>) -> Option<Ordering> {
    if head.take("acquire") {
        Some(Ordering::Acquire)
    } else if head.take("release") {
        Some(Ordering::Release)
    } else if head.take("acq_rel") {
        Some(Ordering::AcquireRelease)
    } else if head.take("relaxed") {
        Some(Ordering::Relaxed)
    } else {
        None
    }
}

pub(crate) fn atomic(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
    returns_old: bool,
) -> Result<Operation> {
    use ir::Atomic::{Add, And, CompareSwap, Decrement, Exchange, Increment, Max, Min, Or, Xor};

    use ScalarType::{B32, B64, F32, S32, S64, U32, U64};

    // ptxas also accepts the operation before the ordering, scope, and space, as NCCL's inline assembly writes it.
    let mut name = None;
    let mut ordering = None;
    let mut scoped = false;
    let mut space = None;
    loop {
        if ordering.is_none()
            && let Some(found) = atomic_ordering(head)
        {
            ordering = Some(found);
        } else if !scoped && head.take_scope() {
            scoped = true;
        } else if space.is_none()
            && let Some(found) = head.space()
        {
            space = Some(found);
        } else if name.is_none()
            && let Some(&found) = head.parts.first()
            && matches!(
                found,
                "add" | "min" | "max" | "and" | "or" | "xor" | "inc" | "dec" | "exch" | "cas"
            )
        {
            head.parts = &head.parts[1..];
            name = Some(found);
        } else {
            break;
        }
    }

    let ordering = ordering.unwrap_or(Ordering::Relaxed);
    require(
        matches!(space, None | Some(StateSpace::Global | StateSpace::Shared)),
        instruction.span,
        "atomic operation requires generic, global, or shared addressing",
    )?;
    let name = match name {
        Some(name) => name,
        None => head.next()?,
    };
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

/// Lowers `multimem.ld_reduce`, `multimem.st`, and `multimem.red`, which need `sm_90`.
pub(crate) fn multimem(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    use ScalarType::{B32, B64, F32, F64, S32, S64, U32, U64};

    let span = instruction.span;
    compiler.require_architecture(Architecture::Sm90, span)?;
    let kind = head.next()?;
    let orderings: &[&str] = match kind {
        "ld_reduce" => &["weak", "relaxed", "acquire"],
        "st" => &["weak", "relaxed", "release"],
        "red" => &["relaxed", "release"],
        _ => {
            return Err(CompileError::unsupported(
                span,
                format!("instruction multimem.{kind}"),
            ));
        }
    };
    let ordered = orderings
        .iter()
        .find(|ordering| head.take(ordering))
        .is_some_and(|ordering| *ordering != "weak");
    if ordered {
        require(
            head.take_scope() || head.take("cluster"),
            span,
            "ordered multimem access requires a scope",
        )?;
    }

    let space = if head.take("global") {
        Some(StateSpace::Global)
    } else {
        None
    };
    let access = match kind {
        "ld_reduce" => ir::MulticastAccess::LoadReduce(multimem_reduction(head)?),
        "red" => ir::MulticastAccess::Reduce(multimem_reduction(head)?),
        _ => ir::MulticastAccess::Store,
    };
    let count = if head.take("v2") {
        2
    } else if head.take("v4") {
        4
    } else {
        1
    };
    let ty = head.ty()?;
    let valid = match &access {
        ir::MulticastAccess::Store => matches!(ty, B32 | B64 | U32 | S32 | U64 | S64 | F32 | F64),
        ir::MulticastAccess::LoadReduce(op) | ir::MulticastAccess::Reduce(op) => match op {
            ir::Atomic::Add => matches!(ty, U32 | S32 | U64 | F32),
            ir::Atomic::Min | ir::Atomic::Max => matches!(ty, U32 | S32 | U64 | S64),
            _ => matches!(ty, B32 | B64),
        },
    };
    require(valid, span, "multimem type does not match its operation")?;
    require(
        count == 1 || ty == F32,
        span,
        "multimem vectors hold f32 elements",
    )?;

    let load = matches!(access, ir::MulticastAccess::LoadReduce(_));
    let [first, second] = operands(instruction)?;
    let (address, values) = if load {
        (second, first)
    } else {
        (first, second)
    };
    let values = if count == 1 {
        slice::from_ref(values)
    } else {
        let Operand::Vector { elements, .. } = values else {
            return Err(CompileError::instruction(
                values.span(),
                "vector multimem access requires a vector operand",
            ));
        };
        require(
            elements.len() == count,
            values.span(),
            "multimem vector length does not match its qualifier",
        )?;
        elements.as_slice()
    };
    let address = compiler.address(address, space, !load)?;
    let (destinations, sources) = if load {
        let destinations = values
            .iter()
            .map(|value| compiler.destination(value, ty))
            .collect::<Result<_>>()?;
        (destinations, Vec::new())
    } else {
        let sources = values
            .iter()
            .map(|value| compiler.source(value, ty))
            .collect::<Result<_>>()?;
        (Vec::new(), sources)
    };
    Ok(Operation::Memory(MemoryOperation::Multicast {
        access,
        ty,
        space,
        address,
        destinations,
        sources,
    }))
}

fn multimem_reduction(head: &mut Head<'_>) -> Result<ir::Atomic> {
    match head.next()? {
        "add" => Ok(ir::Atomic::Add),
        "min" => Ok(ir::Atomic::Min),
        "max" => Ok(ir::Atomic::Max),
        "and" => Ok(ir::Atomic::And),
        "or" => Ok(ir::Atomic::Or),
        "xor" => Ok(ir::Atomic::Xor),
        name => Err(CompileError::unsupported(
            head.span,
            format!("multimem reduction {name}"),
        )),
    }
}

/// Lowers `membar` and `fence`; the interpreter already serializes every access.
pub(crate) fn fence(
    instruction: &Instruction,
    head: &mut Head<'_>,
    opcode: Opcode,
) -> Result<Operation> {
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

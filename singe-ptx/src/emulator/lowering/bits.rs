//! Bit-field, bit-count, permute, and funnel instructions.

use crate::{
    ast::{Instruction, Operand, ScalarType},
    emulator::{
        compiler::FunctionCompiler,
        error::{CompileError, Result},
        ir::{self, IntegerOperation, Operation},
        lowering::{Head, Opcode, operands, paired_destination, require},
        target::PtxVersion,
    },
};

/// Consumes the instruction type and checks it against `valid`.
fn ty(head: &mut Head<'_>, valid: &[ScalarType]) -> Result<ScalarType> {
    let ty = head.ty()?;
    require(valid.contains(&ty), head.span, "invalid bit operation type")?;
    Ok(ty)
}

const BITS: &[ScalarType] = &[ScalarType::B32, ScalarType::B64];
const WIDE_BITS: &[ScalarType] = &[ScalarType::B16, ScalarType::B32, ScalarType::B64];
const INTEGERS: &[ScalarType] = &[
    ScalarType::U32,
    ScalarType::S32,
    ScalarType::U64,
    ScalarType::S64,
];

/// Lowers `popc`, `clz`, `brev`, `bfind`, and `cnot`.
pub fn unary(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
    opcode: Opcode,
) -> Result<Operation> {
    let (op, ty) = match opcode {
        Opcode::Popc => (ir::BitUnary::Popc, ty(head, BITS)?),
        Opcode::Clz => (ir::BitUnary::Clz, ty(head, BITS)?),
        Opcode::Brev => (ir::BitUnary::Brev, ty(head, BITS)?),
        Opcode::Cnot => (ir::BitUnary::Cnot, ty(head, WIDE_BITS)?),
        _ => {
            let shift = head.take("shiftamt");
            (ir::BitUnary::Bfind { shift }, ty(head, INTEGERS)?)
        }
    };
    let destination_ty = if matches!(opcode, Opcode::Brev | Opcode::Cnot) {
        ty
    } else {
        ScalarType::U32
    };
    let [d, a] = operands(instruction)?;
    Ok(Operation::Integer(IntegerOperation::BitUnary {
        op,
        ty,
        destination: compiler.destination(d, destination_ty)?,
        source: compiler.source(a, ty)?,
    }))
}

/// Lowers `bfe`.
pub fn extract(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    let ty = ty(head, INTEGERS)?;
    let [d, a, b, c] = operands(instruction)?;
    Ok(Operation::Integer(IntegerOperation::Extract {
        ty,
        destination: compiler.destination(d, ty)?,
        value: compiler.source(a, ty)?,
        position: compiler.source(b, ScalarType::U32)?,
        length: compiler.source(c, ScalarType::U32)?,
    }))
}

/// Lowers `bfi`.
pub fn insert(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    let ty = ty(head, BITS)?;
    let [d, a, b, c, length] = operands(instruction)?;
    Ok(Operation::Integer(IntegerOperation::Insert {
        ty,
        destination: compiler.destination(d, ty)?,
        value: compiler.source(a, ty)?,
        base: compiler.source(b, ty)?,
        position: compiler.source(c, ScalarType::U32)?,
        length: compiler.source(length, ScalarType::U32)?,
    }))
}

/// Lowers `lop3`, including the PTX 8.2 form that also combines a predicate.
pub fn lop3(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    let span = instruction.span;
    let boolean = if head.take("and") {
        Some(ir::BooleanOp::And)
    } else if head.take("or") {
        Some(ir::BooleanOp::Or)
    } else {
        None
    };
    let ty = ty(head, &[ScalarType::B32])?;
    let (destination, a, b, c, lut, predicate) = if let Some(op) = boolean {
        compiler.require_version(PtxVersion::new(8, 2), span)?;
        let [d, a, b, c, lut, combined] = operands(instruction)?;
        require(
            matches!(d, Operand::Pair { .. }),
            span,
            "predicate lop3 requires paired destinations",
        )?;
        let (destination, predicate) = paired_destination(compiler, d, ty)?;
        let predicate = predicate
            .ok_or_else(|| CompileError::instruction(span, "missing predicate destination"))?;
        require(
            predicate != ir::Destination::Discard,
            span,
            "lop3 predicate destination cannot be discarded",
        )?;
        let predicate = ir::Lop3Predicate {
            destination: predicate,
            combine: ir::Combine {
                op,
                predicate: compiler.predicate(combined)?,
            },
        };
        (destination, a, b, c, lut, Some(predicate))
    } else {
        let [d, a, b, c, lut] = operands(instruction)?;
        (compiler.destination(d, ty)?, a, b, c, lut, None)
    };
    let ir::Source::Immediate(lut) = compiler.source(lut, ScalarType::U32)? else {
        return Err(CompileError::instruction(
            span,
            "lop3 LUT must be immediate",
        ));
    };
    let lut = u8::try_from(lut)
        .map_err(|_| CompileError::instruction(span, "lop3 LUT exceeds eight bits"))?;
    Ok(Operation::Integer(IntegerOperation::Lop3 {
        destination,
        a: compiler.source(a, ty)?,
        b: compiler.source(b, ty)?,
        c: compiler.source(c, ty)?,
        lut,
        predicate,
    }))
}

/// Lowers `prmt`.
pub fn permute(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    let ty = ty(head, &[ScalarType::B32])?;
    let mode = if head.is_empty() {
        ir::Permute::Generic
    } else {
        match head.next()? {
            "f4e" => ir::Permute::Forward,
            "b4e" => ir::Permute::Backward,
            "rc8" => ir::Permute::ReplicateByte,
            "ecl" => ir::Permute::ClampLeft,
            "ecr" => ir::Permute::ClampRight,
            "rc16" => ir::Permute::ReplicateHalf,
            _ => {
                return Err(CompileError::instruction(
                    instruction.span,
                    "invalid permute mode",
                ));
            }
        }
    };
    let [d, a, b, c] = operands(instruction)?;
    Ok(Operation::Integer(IntegerOperation::Permute {
        mode,
        destination: compiler.destination(d, ty)?,
        a: compiler.source(a, ty)?,
        b: compiler.source(b, ty)?,
        control: compiler.source(c, ScalarType::U32)?,
    }))
}

/// Lowers `shf`.
pub fn funnel(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    let span = instruction.span;
    let direction = match head.next()? {
        "l" => ir::ShiftDirection::Left,
        "r" => ir::ShiftDirection::Right,
        _ => return Err(CompileError::instruction(span, "invalid funnel direction")),
    };
    let range = head.range("invalid funnel mode")?;
    require(
        head.ty()? == ScalarType::B32,
        span,
        "funnel shift requires b32",
    )?;
    let [d, a, b, c] = operands(instruction)?;
    Ok(Operation::Integer(IntegerOperation::Funnel {
        direction,
        range,
        destination: compiler.destination(d, ScalarType::B32)?,
        a: compiler.source(a, ScalarType::B32)?,
        b: compiler.source(b, ScalarType::B32)?,
        amount: compiler.source(c, ScalarType::U32)?,
    }))
}

/// Lowers `fns`.
pub fn find_set(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    ty(head, &[ScalarType::B32])?;
    let [d, a, b, c] = operands(instruction)?;
    Ok(Operation::Integer(IntegerOperation::FindSet {
        destination: compiler.destination(d, ScalarType::B32)?,
        mask: compiler.source(a, ScalarType::B32)?,
        base: compiler.source(b, ScalarType::U32)?,
        offset: compiler.source(c, ScalarType::S32)?,
    }))
}

/// Lowers `bmsk`.
pub fn mask(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    compiler.require_version(PtxVersion::new(7, 6), instruction.span)?;
    let range = head.range("expected clamp or wrap")?;
    let ty = ty(head, &[ScalarType::B32])?;
    let [d, a, b] = operands(instruction)?;
    Ok(Operation::Integer(IntegerOperation::MaskBits {
        range,
        destination: compiler.destination(d, ty)?,
        position: compiler.source(a, ScalarType::U32)?,
        length: compiler.source(b, ScalarType::U32)?,
    }))
}

/// Lowers `szext`.
pub fn extend(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    compiler.require_version(PtxVersion::new(7, 6), instruction.span)?;
    let range = head.range("expected clamp or wrap")?;
    let ty = ty(head, &[ScalarType::U32, ScalarType::S32])?;
    let [d, a, b] = operands(instruction)?;
    Ok(Operation::Integer(IntegerOperation::ExtendBits {
        range,
        ty,
        destination: compiler.destination(d, ty)?,
        value: compiler.source(a, ty)?,
        length: compiler.source(b, ScalarType::U32)?,
    }))
}

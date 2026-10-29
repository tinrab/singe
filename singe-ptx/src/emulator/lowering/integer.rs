//! Integer arithmetic, logic, and shift instructions.

use crate::{
    ast::{Instruction, MulMode, ScalarType},
    emulator::{
        compiler::FunctionCompiler,
        error::{CompileError, Result},
        ir::{self, IntegerOperation, Operation},
        lowering::{Head, Opcode, integer, operands, require},
        target::PtxVersion,
    },
};

/// Lowers `add`, `sub`, `div`, `rem`, `min`, and `max` on an integer type.
pub fn binary(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
    opcode: Opcode,
) -> Result<Operation> {
    let span = instruction.span;
    let saturate = head.take("sat");
    let ty = head.ty()?;
    require(
        integer(ty),
        span,
        "expected a 16-bit, 32-bit, or 64-bit integer type",
    )?;
    require(
        !saturate || (matches!(opcode, Opcode::Add | Opcode::Sub) && ty == ScalarType::S32),
        span,
        "saturation requires add.s32 or sub.s32",
    )?;
    let op = match opcode {
        Opcode::Add => ir::IntegerBinary::Add,
        Opcode::Sub => ir::IntegerBinary::Sub,
        Opcode::Div => ir::IntegerBinary::Div,
        Opcode::Rem => ir::IntegerBinary::Rem,
        Opcode::Min => ir::IntegerBinary::Min,
        _ => ir::IntegerBinary::Max,
    };
    let [d, a, b] = operands(instruction)?;
    Ok(Operation::Integer(IntegerOperation::Binary {
        op,
        ty,
        saturate,
        destination: compiler.destination(d, ty)?,
        a: compiler.source(a, ty)?,
        b: compiler.source(b, ty)?,
    }))
}

/// Lowers `mul` and `mad` with a `lo`, `hi`, or `wide` mode.
pub fn multiply(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
    opcode: Opcode,
) -> Result<Operation> {
    let span = instruction.span;
    let mode = match head.next()? {
        "lo" => MulMode::Lo,
        "hi" => MulMode::Hi,
        "wide" => MulMode::Wide,
        _ => {
            return Err(CompileError::instruction(
                span,
                "expected lo, hi, or wide multiplication",
            ));
        }
    };
    let saturate = head.take("sat");
    let ty = head.ty()?;
    require(
        !saturate || (opcode == Opcode::Mad && mode == MulMode::Hi && ty == ScalarType::S32),
        span,
        "saturation requires mad.hi.sat.s32",
    )?;
    require(
        integer(ty),
        span,
        "integer multiplication requires an integer type",
    )?;
    let destination_ty = if mode == MulMode::Wide {
        match ty {
            ScalarType::U16 => ScalarType::U32,
            ScalarType::S16 => ScalarType::S32,
            ScalarType::U32 => ScalarType::U64,
            ScalarType::S32 => ScalarType::S64,
            _ => {
                return Err(CompileError::instruction(
                    span,
                    "wide multiplication requires 16-bit or 32-bit inputs",
                ));
            }
        }
    } else {
        ty
    };
    let (d, a, b, addend) = if opcode == Opcode::Mad {
        let [d, a, b, c] = operands(instruction)?;
        (d, a, b, Some(compiler.source(c, destination_ty)?))
    } else {
        let [d, a, b] = operands(instruction)?;
        (d, a, b, None)
    };
    Ok(Operation::Integer(IntegerOperation::Multiply {
        saturate,
        ty,
        mode,
        destination: compiler.destination(d, destination_ty)?,
        a: compiler.source(a, ty)?,
        b: compiler.source(b, ty)?,
        c: addend,
    }))
}

/// Lowers `mul24` and `mad24`.
pub fn multiply24(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
    opcode: Opcode,
) -> Result<Operation> {
    let span = instruction.span;
    let half = head.half("expected lo or hi")?;
    let saturate = head.take("sat");
    let ty = head.ty()?;
    require(
        matches!(ty, ScalarType::U32 | ScalarType::S32),
        span,
        "24-bit multiply requires s32 or u32",
    )?;
    require(
        !saturate || (opcode == Opcode::Mad24 && half == ir::Half::High && ty == ScalarType::S32),
        span,
        "saturation requires mad24.hi.sat.s32",
    )?;
    let (d, a, b, c) = if opcode == Opcode::Mad24 {
        let [d, a, b, c] = operands(instruction)?;
        (d, a, b, Some(compiler.source(c, ty)?))
    } else {
        let [d, a, b] = operands(instruction)?;
        (d, a, b, None)
    };
    Ok(Operation::Integer(IntegerOperation::Multiply24 {
        ty,
        half,
        saturate,
        destination: compiler.destination(d, ty)?,
        a: compiler.source(a, ty)?,
        b: compiler.source(b, ty)?,
        c,
    }))
}

/// Lowers `sad`.
pub fn sum_difference(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    let ty = head.ty()?;
    require(
        integer(ty),
        instruction.span,
        "sad requires scalar integers",
    )?;
    let [d, a, b, c] = operands(instruction)?;
    Ok(Operation::Integer(IntegerOperation::SumDifference {
        ty,
        destination: compiler.destination(d, ty)?,
        a: compiler.source(a, ty)?,
        b: compiler.source(b, ty)?,
        c: compiler.source(c, ty)?,
    }))
}

/// Lowers `clmad`.
pub fn carryless(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    let span = instruction.span;
    compiler.require_version(PtxVersion::new(9, 3), span)?;
    let half = head.half("expected lo or hi")?;
    require(head.ty()? == ScalarType::U64, span, "clmad requires u64")?;
    let [d, a, b, c] = operands(instruction)?;
    Ok(Operation::Integer(IntegerOperation::Carryless {
        half,
        destination: compiler.destination(d, ScalarType::U64)?,
        a: compiler.source(a, ScalarType::U64)?,
        b: compiler.source(b, ScalarType::U64)?,
        c: compiler.source(c, ScalarType::U64)?,
    }))
}

/// Lowers `addc`, `subc`, `madc`, and the `.cc` forms of `add`, `sub`, and `mad`.
pub fn carry(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
    opcode: Opcode,
) -> Result<Operation> {
    let multiply = matches!(opcode, Opcode::Mad | Opcode::MadCarry);
    let half = if multiply {
        Some(head.half("carry multiply requires lo or hi")?)
    } else {
        None
    };
    let output = head.take("cc");
    let input = matches!(
        opcode,
        Opcode::AddCarry | Opcode::SubCarry | Opcode::MadCarry
    );
    let ty = head.ty()?;
    require(
        matches!(
            ty,
            ScalarType::S32 | ScalarType::U32 | ScalarType::S64 | ScalarType::U64
        ) && (input || output),
        instruction.span,
        "carry arithmetic requires 32/64-bit integers and a carry input or output",
    )?;
    let (d, arithmetic) = if let Some(half) = half {
        let [d, a, b, c] = operands(instruction)?;
        (
            d,
            ir::CarryArithmetic::MultiplyAdd {
                a: compiler.source(a, ty)?,
                b: compiler.source(b, ty)?,
                c: compiler.source(c, ty)?,
                half,
            },
        )
    } else {
        let [d, a, b] = operands(instruction)?;
        let a = compiler.source(a, ty)?;
        let b = compiler.source(b, ty)?;
        let arithmetic = if matches!(opcode, Opcode::Sub | Opcode::SubCarry) {
            ir::CarryArithmetic::Subtract { a, b }
        } else {
            ir::CarryArithmetic::Add { a, b }
        };
        (d, arithmetic)
    };
    Ok(Operation::Integer(IntegerOperation::Carry {
        ty,
        arithmetic,
        input,
        output,
        destination: compiler.destination(d, ty)?,
    }))
}

/// Lowers `dp4a` and `dp2a`.
pub fn dot(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
    opcode: Opcode,
) -> Result<Operation> {
    let span = instruction.span;
    let mode = if opcode == Opcode::Dp4a {
        ir::DotMode::FourBytes
    } else {
        match head.half("dp2a requires lo or hi")? {
            ir::Half::Low => ir::DotMode::TwoLow,
            ir::Half::High => ir::DotMode::TwoHigh,
        }
    };
    let atype = head.ty()?;
    let btype = head.ty()?;
    require(
        matches!(atype, ScalarType::S32 | ScalarType::U32)
            && matches!(btype, ScalarType::S32 | ScalarType::U32),
        span,
        "dot product requires s32 or u32 input formats",
    )?;
    let result = if atype.is_signed() || btype.is_signed() {
        ScalarType::S32
    } else {
        ScalarType::U32
    };
    let [d, a, b, c] = operands(instruction)?;
    Ok(Operation::Integer(IntegerOperation::DotProduct {
        mode,
        a_ty: atype,
        b_ty: btype,
        destination: compiler.destination(d, result)?,
        a: compiler.source(a, atype)?,
        b: compiler.source(b, btype)?,
        c: compiler.source(c, result)?,
    }))
}

/// Lowers `neg` and `abs` on a signed integer type.
pub fn unary(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
    opcode: Opcode,
) -> Result<Operation> {
    let ty = head.ty()?;
    require(
        matches!(ty, ScalarType::S16 | ScalarType::S32 | ScalarType::S64),
        instruction.span,
        "integer neg/abs requires a signed type",
    )?;
    let [d, a] = operands(instruction)?;
    Ok(Operation::Integer(IntegerOperation::Unary {
        op: if opcode == Opcode::Neg {
            ir::Unary::Neg
        } else {
            ir::Unary::Abs
        },
        ty,
        destination: compiler.destination(d, ty)?,
        source: compiler.source(a, ty)?,
    }))
}

/// Lowers `and`, `or`, `xor`, and `not`.
pub fn bitwise(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
    opcode: Opcode,
) -> Result<Operation> {
    let ty = head.ty()?;
    require(
        matches!(
            ty,
            ScalarType::B16 | ScalarType::B32 | ScalarType::B64 | ScalarType::Pred
        ),
        instruction.span,
        "bitwise operation requires a bit or predicate type",
    )?;
    if opcode == Opcode::Not {
        let [d, a] = operands(instruction)?;
        return Ok(Operation::Integer(IntegerOperation::Not {
            ty,
            destination: compiler.destination(d, ty)?,
            source: compiler.source(a, ty)?,
        }));
    }
    let op = match opcode {
        Opcode::And => ir::BitwiseBinary::And,
        Opcode::Or => ir::BitwiseBinary::Or,
        _ => ir::BitwiseBinary::Xor,
    };
    let [d, a, b] = operands(instruction)?;
    Ok(Operation::Integer(IntegerOperation::Bitwise {
        op,
        ty,
        destination: compiler.destination(d, ty)?,
        a: compiler.source(a, ty)?,
        b: compiler.source(b, ty)?,
    }))
}

/// Lowers `shl` and `shr`.
pub fn shift(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
    opcode: Opcode,
) -> Result<Operation> {
    let direction = if opcode == Opcode::Shr {
        ir::ShiftDirection::Right
    } else {
        ir::ShiftDirection::Left
    };
    let ty = head.ty()?;
    require(
        matches!(ty, ScalarType::B16 | ScalarType::B32 | ScalarType::B64)
            || (direction == ir::ShiftDirection::Right && integer(ty)),
        instruction.span,
        "unsupported shift type",
    )?;
    let [d, a, b] = operands(instruction)?;
    Ok(Operation::Integer(IntegerOperation::Shift {
        direction,
        ty,
        destination: compiler.destination(d, ty)?,
        value: compiler.source(a, ty)?,
        amount: compiler.source(b, ScalarType::U32)?,
    }))
}

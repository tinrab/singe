//! Floating-point arithmetic, math, and classification instructions.

use crate::{
    ast::{FloatRoundingMode, Instruction, ScalarType},
    emulator::{
        compiler::FunctionCompiler,
        error::{CompileError, Result},
        ir::{self, FloatOperation, Operation},
        lowering::{Head, Opcode, float, operands, require},
    },
};

/// Lowers `add`, `sub`, `mul`, `div`, `min`, and `max` on a floating format.
pub fn binary(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
    opcode: Opcode,
) -> Result<Operation> {
    let (format, control) = control(instruction, head, opcode)?;
    let op = match opcode {
        Opcode::Add => ir::FloatBinary::Add,
        Opcode::Sub => ir::FloatBinary::Sub,
        Opcode::Mul => ir::FloatBinary::Mul,
        Opcode::Div => ir::FloatBinary::Div,
        Opcode::Min => ir::FloatBinary::Min,
        Opcode::Max => ir::FloatBinary::Max,
        _ => {
            return Err(CompileError::unsupported(
                instruction.span,
                "floating remainder",
            ));
        }
    };
    let [d, a, b] = operands(instruction)?;
    Ok(Operation::Float(FloatOperation::Binary {
        op,
        format,
        control,
        destination: compiler.float_destination(d, format)?,
        a: compiler.float_source(a, format)?,
        b: compiler.float_source(b, format)?,
    }))
}

/// Lowers `neg` and `abs` on a floating format.
pub fn unary(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
    opcode: Opcode,
) -> Result<Operation> {
    let span = instruction.span;
    let ftz = head.take("ftz");
    let format = ir::FloatFormat::parse(head.next()?)
        .ok_or_else(|| CompileError::instruction(span, "invalid floating format"))?;
    require(
        !ftz || (!format.bf16() && format != ir::FloatFormat::F64),
        span,
        "ftz is invalid for this format",
    )?;
    let [d, a] = operands(instruction)?;
    Ok(Operation::Float(FloatOperation::Unary {
        op: if opcode == Opcode::Neg {
            ir::Unary::Neg
        } else {
            ir::Unary::Abs
        },
        format,
        ftz,
        destination: compiler.float_destination(d, format)?,
        source: compiler.float_source(a, format)?,
    }))
}

pub fn fma(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    let (format, control) = control(instruction, head, Opcode::Fma)?;
    let [d, a, b, c] = operands(instruction)?;
    Ok(Operation::Float(FloatOperation::Fma {
        format,
        control,
        destination: compiler.float_destination(d, format)?,
        a: compiler.float_source(a, format)?,
        b: compiler.float_source(b, format)?,
        c: compiler.float_source(c, format)?,
    }))
}

pub fn exponential(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    let span = instruction.span;
    require(head.take("approx"), span, "ex2 requires approx")?;
    let ftz = head.take("ftz");
    require(head.ty()? == ScalarType::F32, span, "ex2 requires f32")?;
    let [d, a] = operands(instruction)?;
    Ok(Operation::Float(FloatOperation::Exponential {
        ftz,
        destination: compiler.destination(d, ScalarType::F32)?,
        source: compiler.source(a, ScalarType::F32)?,
    }))
}

/// Lowers `rcp`, `sqrt`, and `rsqrt`.
pub fn math(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
    opcode: Opcode,
) -> Result<Operation> {
    let span = instruction.span;
    let approximate = head.take("approx");
    let rounding = if approximate {
        FloatRoundingMode::Rn
    } else {
        head.rounding().ok_or_else(|| {
            CompileError::instruction(span, "rounding or approx qualifier required")
        })?
    };
    let ftz = head.take("ftz");
    let ty = head.ty()?;
    require(float(ty), span, "math operation requires f32 or f64")?;
    require(
        opcode != Opcode::Rsqrt || approximate,
        span,
        "rsqrt requires approx",
    )?;
    require(
        !(ty == ScalarType::F64 && ftz) || (approximate && opcode != Opcode::Sqrt),
        span,
        "ftz is invalid for this math form",
    )?;
    require(
        !(ty == ScalarType::F64 && approximate)
            || opcode == Opcode::Rsqrt
            || (opcode == Opcode::Rcp && ftz),
        span,
        "invalid approximate f64 math form",
    )?;
    let [d, a] = operands(instruction)?;
    Ok(Operation::Float(FloatOperation::Math {
        op: match opcode {
            Opcode::Rcp => ir::FloatMath::Reciprocal,
            Opcode::Sqrt => ir::FloatMath::SquareRoot,
            _ => ir::FloatMath::ReciprocalSquareRoot,
        },
        ty,
        control: ir::MathControl {
            rounding,
            ftz,
            upper_word: ty == ScalarType::F64 && approximate && ftz,
        },
        destination: compiler.destination(d, ty)?,
        source: compiler.source(a, ty)?,
    }))
}

/// Lowers `testp`.
pub fn test(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    let span = instruction.span;
    let property = match head.next()? {
        "finite" => ir::FloatProperty::Finite,
        "infinite" => ir::FloatProperty::Infinite,
        "number" => ir::FloatProperty::Number,
        "notanumber" => ir::FloatProperty::Nan,
        "normal" => ir::FloatProperty::Normal,
        "subnormal" => ir::FloatProperty::Subnormal,
        _ => return Err(CompileError::instruction(span, "invalid floating property")),
    };
    let ty = head.ty()?;
    require(float(ty), span, "testp requires f32 or f64")?;
    let [d, a] = operands(instruction)?;
    Ok(Operation::Float(FloatOperation::Test {
        property,
        ty,
        destination: compiler.destination(d, ScalarType::Pred)?,
        source: compiler.source(a, ty)?,
    }))
}

pub fn copy_sign(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    let ty = head.ty()?;
    require(float(ty), instruction.span, "copysign requires f32 or f64")?;
    let [d, a, b] = operands(instruction)?;
    Ok(Operation::Float(FloatOperation::CopySign {
        ty,
        destination: compiler.destination(d, ty)?,
        sign: compiler.source(a, ty)?,
        magnitude: compiler.source(b, ty)?,
    }))
}

/// Parses the rounding, flush, saturation, and format qualifiers shared by arithmetic forms.
fn control(
    instruction: &Instruction,
    head: &mut Head<'_>,
    opcode: Opcode,
) -> Result<(ir::FloatFormat, ir::FloatControl)> {
    let span = instruction.span;
    let round = head.rounding();
    let ftz = head.take("ftz");
    let saturate = head.take("sat");
    let relu = head.take("relu");
    let nan = head.take("NaN");
    let format = ir::FloatFormat::parse(head.next()?)
        .ok_or_else(|| CompileError::instruction(span, "invalid floating format"))?;
    let minmax = matches!(opcode, Opcode::Min | Opcode::Max);
    let half = format.half();
    let bf = format.bf16();
    let double = format == ir::FloatFormat::F64;
    require(
        !ftz || (!bf && !double),
        span,
        "ftz is invalid for this format",
    )?;
    require(
        !saturate || (!bf && !double && !minmax && opcode != Opcode::Div),
        span,
        "div requires rounding and no saturation; invalid floating saturation",
    )?;
    require(
        !relu || (opcode == Opcode::Fma && half),
        span,
        "relu requires half fma",
    )?;
    require(
        !nan || (minmax && !double),
        span,
        "NaN modifier requires half or f32 min/max",
    )?;
    require(
        !minmax || (round.is_none() && !saturate && !relu),
        span,
        "min/max does not take rounding or saturation",
    )?;
    require(
        !bf || matches!(opcode, Opcode::Fma | Opcode::Min | Opcode::Max),
        span,
        "bf16 add/sub/mul requires sm_90",
    )?;
    require(
        !half || (opcode != Opcode::Div && round.is_none_or(|r| r == FloatRoundingMode::Rn)),
        span,
        "half arithmetic supports rn rounding only",
    )?;
    require(
        !matches!(opcode, Opcode::Div | Opcode::Fma) || round.is_some(),
        span,
        "div/fma requires explicit rounding",
    )?;
    require(!(saturate && relu), span, "sat and relu are exclusive")?;
    Ok((
        format,
        ir::FloatControl {
            rounding: round.unwrap_or(FloatRoundingMode::Rn),
            ftz,
            clamp: ir::Clamp::new(saturate, relu),
            nan,
        },
    ))
}

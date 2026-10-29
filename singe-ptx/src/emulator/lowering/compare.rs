//! Comparison and selection instructions.

use crate::{
    ast::{CompareOp, Instruction, Operand, ScalarType},
    emulator::{
        compiler::FunctionCompiler,
        error::{CompileError, Result},
        ir::{self, DataOperation, Operation},
        layout::scalar_bytes,
        lowering::{Head, float, integer, operands, paired_destination, require, scalar},
    },
};

pub fn compare(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    let (comparison, boolean, ftz) = prefix(head)?;
    let ty = head.ty()?;
    require(
        supports(comparison, ty)
            && (!ftz || matches!(ty, ScalarType::F32 | ScalarType::F16 | ScalarType::F16x2)),
        instruction.span,
        "comparison modifiers do not apply to this type",
    )?;
    let (d, a, b, combine) = operands_with_combine(compiler, instruction, boolean)?;
    if ty == ScalarType::F16 {
        require(
            !matches!(d, Operand::Pair { .. }),
            instruction.span,
            "scalar f16 setp has one destination",
        )?;
    }
    if ty == ScalarType::F16x2 {
        require(
            matches!(d, Operand::Pair { .. }),
            instruction.span,
            "packed setp requires paired predicates",
        )?;
    }
    let (destination, inverse) = paired_destination(compiler, d, ScalarType::Pred)?;
    Ok(Operation::Data(DataOperation::Compare {
        comparison,
        ty,
        ftz,
        destination,
        inverse,
        a: source(compiler, a, ty)?,
        b: source(compiler, b, ty)?,
        combine,
    }))
}

pub fn set(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    let (comparison, boolean, ftz) = prefix(head)?;
    let to = head.ty()?;
    let from = head.ty()?;
    let valid = match from {
        ScalarType::F16 => matches!(
            to,
            ScalarType::F16 | ScalarType::S16 | ScalarType::U16 | ScalarType::S32 | ScalarType::U32
        ),
        ScalarType::F16x2 => matches!(to, ScalarType::F16x2 | ScalarType::S32 | ScalarType::U32),
        _ => {
            scalar(from)
                && matches!(
                    to,
                    ScalarType::S32 | ScalarType::U32 | ScalarType::F32 | ScalarType::F16
                )
        }
    };
    require(
        valid && supports(comparison, from),
        instruction.span,
        "invalid set source or destination format",
    )?;
    require(
        to != ScalarType::F16
            || !matches!(
                comparison,
                CompareOp::Lo | CompareOp::Ls | CompareOp::Hi | CompareOp::Hs
            ),
        instruction.span,
        "half set does not support unsigned comparison aliases",
    )?;
    require(
        !ftz || matches!(from, ScalarType::F16 | ScalarType::F16x2 | ScalarType::F32),
        instruction.span,
        "ftz does not apply to this set form",
    )?;
    let (d, a, b, combine) = operands_with_combine(compiler, instruction, boolean)?;
    let destination = match to {
        ScalarType::F16 => compiler.float_destination(d, ir::FloatFormat::F16)?,
        ScalarType::F16x2 => compiler.float_destination(d, ir::FloatFormat::F16x2)?,
        _ => compiler.destination(d, to)?,
    };
    Ok(Operation::Data(DataOperation::CompareValue {
        comparison,
        from,
        to,
        ftz,
        destination,
        a: source(compiler, a, from)?,
        b: source(compiler, b, from)?,
        combine,
    }))
}

/// Lowers `selp`.
pub fn select(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    let ty = head.ty()?;
    require(scalar(ty), instruction.span, "unsupported selection type")?;
    let [d, a, b, c] = operands(instruction)?;
    Ok(Operation::Data(DataOperation::Select {
        ty,
        destination: compiler.destination(d, ty)?,
        a: compiler.source(a, ty)?,
        b: compiler.source(b, ty)?,
        predicate: compiler.predicate(c)?,
    }))
}

pub fn select_sign(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &Instruction,
    head: &mut Head<'_>,
) -> Result<Operation> {
    let ftz = head.take("ftz");
    let dtype = head.ty()?;
    let selector_ty = head.ty()?;
    require(
        scalar(dtype)
            && matches!(selector_ty, ScalarType::S32 | ScalarType::F32)
            && (!ftz || selector_ty == ScalarType::F32),
        instruction.span,
        "invalid slct types or modifiers",
    )?;
    let ty = match scalar_bytes(dtype, instruction.span)? {
        2 => ScalarType::B16,
        4 => ScalarType::B32,
        _ => ScalarType::B64,
    };
    let [d, a, b, c] = operands(instruction)?;
    Ok(Operation::Data(DataOperation::SelectSign {
        ty,
        selector_ty,
        ftz,
        destination: compiler.destination(d, ty)?,
        a: compiler.source(a, ty)?,
        b: compiler.source(b, ty)?,
        selector: compiler.source(c, selector_ty)?,
    }))
}

fn prefix(head: &mut Head<'_>) -> Result<(CompareOp, Option<ir::BooleanOp>, bool)> {
    use CompareOp::{
        Eq, Equ, Ge, Geu, Gt, Gtu, Hi, Hs, Le, Leu, Lo, Ls, Lt, Ltu, Nan, Ne, Neu, Num,
    };
    let op = match head.next()? {
        "eq" => Eq,
        "ne" => Ne,
        "lt" => Lt,
        "le" => Le,
        "gt" => Gt,
        "ge" => Ge,
        "lo" => Lo,
        "ls" => Ls,
        "hi" => Hi,
        "hs" => Hs,
        "equ" => Equ,
        "neu" => Neu,
        "ltu" => Ltu,
        "leu" => Leu,
        "gtu" => Gtu,
        "geu" => Geu,
        "num" => Num,
        "nan" => Nan,
        _ => {
            return Err(CompileError::instruction(
                head.span,
                "invalid comparison operator",
            ));
        }
    };
    let boolean = if head.take("and") {
        Some(ir::BooleanOp::And)
    } else if head.take("or") {
        Some(ir::BooleanOp::Or)
    } else if head.take("xor") {
        Some(ir::BooleanOp::Xor)
    } else {
        None
    };
    Ok((op, boolean, head.take("ftz")))
}

fn supports(op: CompareOp, ty: ScalarType) -> bool {
    use CompareOp::{
        Eq, Equ, Ge, Geu, Gt, Gtu, Hi, Hs, Le, Leu, Lo, Ls, Lt, Ltu, Nan, Ne, Neu, Num,
    };
    let floating = float(ty) || matches!(ty, ScalarType::F16 | ScalarType::F16x2);
    match op {
        Eq | Ne => scalar(ty) || floating,
        Lt | Le | Gt | Ge => integer(ty) || floating,
        Lo | Ls | Hi | Hs => matches!(ty, ScalarType::U16 | ScalarType::U32 | ScalarType::U64),
        Equ | Neu | Ltu | Leu | Gtu | Geu | Num | Nan => floating,
    }
}

type ComparisonOperands<'a> = (&'a Operand, &'a Operand, &'a Operand, Option<ir::Combine>);

fn operands_with_combine<'a>(
    compiler: &FunctionCompiler<'_, '_>,
    instruction: &'a Instruction,
    boolean: Option<ir::BooleanOp>,
) -> Result<ComparisonOperands<'a>> {
    if let Some(op) = boolean {
        let [d, a, b, c] = operands(instruction)?;
        Ok((
            d,
            a,
            b,
            Some(ir::Combine {
                op,
                predicate: compiler.predicate(c)?,
            }),
        ))
    } else {
        let [d, a, b] = operands(instruction)?;
        Ok((d, a, b, None))
    }
}

fn source(
    compiler: &FunctionCompiler<'_, '_>,
    operand: &Operand,
    ty: ScalarType,
) -> Result<ir::Source> {
    match ty {
        ScalarType::F16 => compiler.float_source(operand, ir::FloatFormat::F16),
        ScalarType::F16x2 => compiler.float_source(operand, ir::FloatFormat::F16x2),
        _ => compiler.source(operand, ty),
    }
}

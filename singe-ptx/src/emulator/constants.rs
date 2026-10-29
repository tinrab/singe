use std::collections::BTreeMap;

use rustc_apfloat::{
    Float, FloatConvert, Round,
    ieee::{Double, Single},
};

use crate::{
    ast::{
        FloatLiteral, Initializer, InitializerBinaryOp as Binary, InitializerCastType,
        InitializerUnaryOp as Unary, ScalarType, Span,
    },
    emulator::{
        error::{CompileError, CompileErrorKind, Result},
        ir::GlobalId,
        target::PtxVersion,
    },
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Relocation {
    pub target: GlobalId,
    pub offset: i64,
    pub generic: bool,
    /// Mask selecting one byte of the relocated address, which is then shifted down to
    /// the low bits.
    pub mask: Option<u64>,
}

#[derive(Debug, Clone)]
pub enum Constant {
    Signed(i64),
    Unsigned(u64),
    Float(Double),
    Single(u32),
    Address(Relocation),
}

fn error(span: Span, reason: impl Into<String>) -> CompileError {
    CompileError::new(span, CompileErrorKind::InvalidInitializer(reason.into()))
}

pub fn float_constant(literal: &FloatLiteral, span: Span) -> Result<Constant> {
    Ok(match literal {
        FloatLiteral::Decimal(text) => Constant::Float(
            Double::from_str_r(text, Round::NearestTiesToEven)
                .map_err(|_| error(span, "invalid decimal floating-point constant"))?
                .value,
        ),
        FloatLiteral::Binary64(bits) => Constant::Float(Double::from_bits(u128::from(*bits))),
        FloatLiteral::Binary32(bits) => Constant::Single(*bits),
    })
}

impl Constant {
    pub(crate) fn bits(self, ty: ScalarType, span: Span) -> Result<u128> {
        use ScalarType::{B32, B64, Bf16, F16, F16x2, F32, F64, Tf32};
        match self {
            Self::Signed(value) if !matches!(ty, F16 | F16x2 | Bf16 | F32 | F64 | Tf32) => {
                Ok(value as u128)
            }
            Self::Unsigned(value) if !matches!(ty, F16 | F16x2 | Bf16 | F32 | F64 | Tf32) => {
                Ok(u128::from(value))
            }
            Self::Single(bits) if matches!(ty, F32 | B32) => Ok(u128::from(bits)),
            Self::Float(value) if matches!(ty, F32 | B32) => {
                let single: Single = value.convert(&mut false).value;
                Ok(single.to_bits())
            }
            Self::Float(value) if matches!(ty, F64 | B64) => Ok(value.to_bits()),
            _ => Err(error(
                span,
                "constant cannot be represented by the declared type",
            )),
        }
    }

    fn integer(&self, span: Span) -> Result<u64> {
        match self {
            Self::Signed(value) => Ok(*value as u64),
            Self::Unsigned(value) => Ok(*value),
            _ => Err(error(span, "integer expression required")),
        }
    }

    fn truth(&self, span: Span) -> Result<bool> {
        Ok(self.integer(span)? != 0)
    }
}

pub fn evaluate(value: &Initializer, symbols: &BTreeMap<String, GlobalId>) -> Result<Constant> {
    match value {
        Initializer::Integer { value, .. } => Ok(Constant::Signed(*value)),
        Initializer::UnsignedInteger { value, .. } => Ok(Constant::Unsigned(*value)),
        Initializer::Float { span, value } => float_constant(value, *span),
        Initializer::Symbol { span, name } => {
            let target = *symbols.get(name).ok_or_else(|| {
                CompileError::new(*span, CompileErrorKind::UndefinedSymbol(name.clone()))
            })?;
            Ok(Constant::Address(Relocation {
                target,
                offset: 0,
                generic: false,
                mask: None,
            }))
        }
        Initializer::Generic { span, expr } => match evaluate(expr, symbols)? {
            Constant::Address(mut address) => {
                address.generic = true;
                Ok(Constant::Address(address))
            }
            _ => Err(error(*span, "generic requires a symbol address")),
        },
        Initializer::Masked { span, mask, value } => {
            let shift = mask.trailing_zeros();
            if shift % 8 != 0 || shift > 56 || *mask != (0xffu64 << shift) {
                return Err(error(
                    *span,
                    "initializer mask must select one byte at a byte boundary",
                ));
            }
            match evaluate(value, symbols)? {
                Constant::Address(mut address) => {
                    if address.mask.is_some() {
                        return Err(error(*span, "nested address masks are unsupported"));
                    }
                    address.mask = Some(*mask);
                    Ok(Constant::Address(address))
                }
                value => Ok(Constant::Unsigned((value.integer(*span)? & mask) >> shift)),
            }
        }
        Initializer::Cast { span, ty, expr } => {
            let value = evaluate(expr, symbols)?.integer(*span)?;
            Ok(match ty {
                InitializerCastType::S64 => Constant::Signed((value).cast_signed()),
                InitializerCastType::U64 => Constant::Unsigned(value),
            })
        }
        Initializer::Unary { span, op, expr } => unary(*op, evaluate(expr, symbols)?, *span),
        Initializer::Conditional {
            span,
            condition,
            then_expr,
            else_expr,
        } => {
            let condition = evaluate(condition, symbols)?.truth(*span)?;
            let a = evaluate(then_expr, symbols)?;
            let b = evaluate(else_expr, symbols)?;
            match (&a, &b) {
                (Constant::Float(_), Constant::Float(_)) => Ok(if condition { a } else { b }),
                (
                    Constant::Signed(_) | Constant::Unsigned(_),
                    Constant::Signed(_) | Constant::Unsigned(_),
                ) => {
                    let unsigned =
                        matches!(a, Constant::Unsigned(_)) || matches!(b, Constant::Unsigned(_));
                    let value = if condition { a } else { b };
                    if unsigned {
                        Ok(Constant::Unsigned(value.integer(*span)?))
                    } else {
                        Ok(value)
                    }
                }
                _ => Err(error(
                    *span,
                    "conditional branches must both be integers or both be binary64 constants",
                )),
            }
        }
        Initializer::Binary {
            span,
            op,
            left,
            right,
        } => {
            let lhs = evaluate(left, symbols)?;
            binary(lhs, *op, evaluate(right, symbols)?, *span)
        }
        Initializer::List { span, .. } => Err(error(*span, "scalar expression required")),
    }
}

/// Checks that every operator in an initializer exists in PTX `version`.
///
/// Returns whether the expression refers to a symbol.
pub fn validate_version(value: &Initializer, version: PtxVersion) -> Result<bool> {
    Ok(match value {
        Initializer::Symbol { .. } => true,
        Initializer::Masked { span, value, .. } => {
            let symbolic = validate_version(value, version)?;
            let minimum = PtxVersion::new(7, if symbolic { 1 } else { 3 });
            if version < minimum {
                return Err(CompileError::unsupported(
                    *span,
                    format!("initializer mask requires PTX {minimum}"),
                ));
            }
            symbolic
        }
        Initializer::Generic { expr, .. }
        | Initializer::Cast { expr, .. }
        | Initializer::Unary { expr, .. } => validate_version(expr, version)?,
        Initializer::Binary { left, right, .. } => {
            let left = validate_version(left, version)?;
            let right = validate_version(right, version)?;
            left || right
        }
        Initializer::Conditional {
            condition,
            then_expr,
            else_expr,
            ..
        } => {
            let condition = validate_version(condition, version)?;
            let then_expr = validate_version(then_expr, version)?;
            let else_expr = validate_version(else_expr, version)?;
            condition || then_expr || else_expr
        }
        Initializer::List { values, .. } => {
            let mut symbolic = false;
            for value in values {
                symbolic |= validate_version(value, version)?;
            }
            symbolic
        }
        Initializer::Integer { .. }
        | Initializer::UnsignedInteger { .. }
        | Initializer::Float { .. } => false,
    })
}

fn unary(op: Unary, value: Constant, span: Span) -> Result<Constant> {
    match (op, value) {
        (
            Unary::Plus,
            value @ (Constant::Signed(_) | Constant::Unsigned(_) | Constant::Float(_)),
        ) => Ok(value),
        (Unary::Minus, Constant::Float(value)) => Ok(Constant::Float(-value)),
        (Unary::Minus, Constant::Signed(value)) => Ok(Constant::Signed(value.wrapping_neg())),
        (Unary::Minus, Constant::Unsigned(value)) => Ok(Constant::Unsigned(value.wrapping_neg())),
        (Unary::BitwiseNot, Constant::Signed(value)) => Ok(Constant::Unsigned(!(value as u64))),
        (Unary::BitwiseNot, Constant::Unsigned(value)) => Ok(Constant::Unsigned(!value)),
        (Unary::LogicalNot, value) => Ok(Constant::Signed(i64::from(!value.truth(span)?))),
        _ => Err(error(span, "invalid unary constant expression")),
    }
}

fn binary(lhs: Constant, op: Binary, rhs: Constant, span: Span) -> Result<Constant> {
    match (lhs, rhs) {
        (Constant::Address(address), rhs) => address_binary(address, op, rhs, span),
        (lhs, Constant::Address(address)) if op == Binary::Add => {
            address_binary(address, op, lhs, span)
        }
        (Constant::Float(a), Constant::Float(b)) => float_binary(a, op, b, span),
        (Constant::Float(_), _) | (_, Constant::Float(_)) => Err(error(
            span,
            "floating-point expressions cannot mix integer operands",
        )),
        (lhs, rhs) => integer_binary(&lhs, op, &rhs, span),
    }
}

/// Evaluates symbol-address arithmetic: a displacement, or the difference of two addresses
/// of the same symbol.
fn address_binary(
    mut address: Relocation,
    op: Binary,
    rhs: Constant,
    span: Span,
) -> Result<Constant> {
    let offset = match rhs {
        Constant::Address(other) => {
            if op == Binary::Sub
                && address.target == other.target
                && address.generic == other.generic
                && address.mask == other.mask
            {
                return address
                    .offset
                    .checked_sub(other.offset)
                    .map(Constant::Signed)
                    .ok_or_else(|| error(span, "address difference overflow"));
            }
            return Err(error(
                span,
                "unsupported expression between symbol addresses",
            ));
        }
        Constant::Signed(value) => value,
        Constant::Unsigned(value) => {
            i64::try_from(value).map_err(|_| error(span, "address displacement overflow"))?
        }
        _ => return Err(error(span, "address displacement must be an integer")),
    };
    address.offset = match op {
        Binary::Add => address.offset.checked_add(offset),
        Binary::Sub => address.offset.checked_sub(offset),
        _ => None,
    }
    .ok_or_else(|| error(span, "invalid address arithmetic"))?;
    if address.mask.is_some() {
        return Err(error(
            span,
            "arithmetic after an address mask is unsupported",
        ));
    }
    Ok(Constant::Address(address))
}

fn float_binary(a: Double, op: Binary, b: Double, span: Span) -> Result<Constant> {
    match op {
        Binary::Add => Ok(Constant::Float((a + b).value)),
        Binary::Sub => Ok(Constant::Float((a - b).value)),
        Binary::Mul => Ok(Constant::Float((a * b).value)),
        Binary::Div if b.is_zero() => Err(error(span, "division by zero")),
        Binary::Div => Ok(Constant::Float((a / b).value)),
        Binary::Eq => Ok(Constant::Signed(i64::from(a == b))),
        Binary::Ne => Ok(Constant::Signed(i64::from(a != b))),
        Binary::Lt => Ok(Constant::Signed(i64::from(a < b))),
        Binary::Le => Ok(Constant::Signed(i64::from(a <= b))),
        Binary::Gt => Ok(Constant::Signed(i64::from(a > b))),
        Binary::Ge => Ok(Constant::Signed(i64::from(a >= b))),
        _ => Err(error(span, "unsupported floating-point constant operator")),
    }
}

/// Evaluates 64-bit integer arithmetic; an unsigned operand makes the operation unsigned.
fn integer_binary(lhs: &Constant, op: Binary, rhs: &Constant, span: Span) -> Result<Constant> {
    let left_unsigned = matches!(lhs, Constant::Unsigned(_));
    let unsigned = left_unsigned || matches!(rhs, Constant::Unsigned(_));
    let (a, b) = (lhs.integer(span)?, rhs.integer(span)?);
    let signed_cmp = (a).cast_signed().cmp(&(b).cast_signed());
    let comparison = if unsigned { a.cmp(&b) } else { signed_cmp };
    let result = match op {
        Binary::Add => a.wrapping_add(b),
        Binary::Sub => a.wrapping_sub(b),
        Binary::Mul => a.wrapping_mul(b),
        Binary::Div | Binary::Rem => {
            if b == 0 {
                return Err(error(span, "division by zero"));
            }
            if op == Binary::Rem {
                a % b
            } else if unsigned {
                a / b
            } else {
                (a).cast_signed().wrapping_div((b).cast_signed()) as u64
            }
        }
        Binary::Shl | Binary::Shr => {
            let shift = (b & 63) as u32;
            if op == Binary::Shl {
                a << shift
            } else if left_unsigned {
                a >> shift
            } else {
                ((a).cast_signed() >> shift) as u64
            }
        }
        Binary::BitAnd => a & b,
        Binary::BitOr => a | b,
        Binary::BitXor => a ^ b,
        Binary::Lt => u64::from(comparison.is_lt()),
        Binary::Le => u64::from(comparison.is_le()),
        Binary::Gt => u64::from(comparison.is_gt()),
        Binary::Ge => u64::from(comparison.is_ge()),
        Binary::Eq => u64::from(a == b),
        Binary::Ne => u64::from(a != b),
        Binary::LogicalAnd => u64::from(a != 0 && b != 0),
        Binary::LogicalOr => u64::from(a != 0 || b != 0),
    };
    let unsigned_result = match op {
        Binary::Lt
        | Binary::Le
        | Binary::Gt
        | Binary::Ge
        | Binary::Eq
        | Binary::Ne
        | Binary::LogicalAnd
        | Binary::LogicalOr
        | Binary::Rem => false,
        Binary::Shl | Binary::Shr => left_unsigned,
        _ => unsigned,
    };
    Ok(if unsigned_result {
        Constant::Unsigned(result)
    } else {
        Constant::Signed((result).cast_signed())
    })
}

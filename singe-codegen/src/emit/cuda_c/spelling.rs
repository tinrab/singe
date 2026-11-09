use crate::ir::{BinaryOp, Combiner, ElementType, Literal, UnaryOp};

/// Returns the C type values of `element` are computed in.
#[must_use]
pub fn compute_type(element: ElementType) -> &'static str {
    match element {
        ElementType::Bool => "bool",
        ElementType::U8 => "unsigned char",
        ElementType::I8 => "signed char",
        ElementType::U16 => "unsigned short",
        ElementType::I16 => "short",
        ElementType::U32 => "unsigned int",
        ElementType::I32 => "int",
        ElementType::U64 => "unsigned long long",
        ElementType::I64 => "long long",
        ElementType::F16
        | ElementType::Bf16
        | ElementType::F32
        | ElementType::F8E4M3
        | ElementType::F8E5M2 => "float",
        ElementType::F64 => "double",
    }
}

/// Returns the C type elements of `element` are stored as.
#[must_use]
pub fn storage_type(element: ElementType) -> &'static str {
    match element {
        ElementType::Bool | ElementType::F8E4M3 | ElementType::F8E5M2 => "unsigned char",
        ElementType::F16 | ElementType::Bf16 => "unsigned short",
        other => compute_type(other),
    }
}

#[must_use]
pub fn load_expression(element: ElementType, address: &str) -> String {
    match element {
        ElementType::Bool => format!("({address} != 0)"),
        ElementType::F16 => format!("load_half({address})"),
        ElementType::Bf16 => format!("load_bfloat({address})"),
        ElementType::F8E4M3 => format!("singe_load_fp8<4, 3, true>({address})"),
        ElementType::F8E5M2 => format!("singe_load_fp8<5, 2, false>({address})"),
        _ => address.to_owned(),
    }
}

#[must_use]
pub fn store_expression(element: ElementType, value: &str) -> String {
    match element {
        ElementType::Bool => format!("static_cast<unsigned char>({value})"),
        ElementType::F16 => format!("store_half({value})"),
        ElementType::Bf16 => format!("store_bfloat({value})"),
        ElementType::F8E4M3 => format!("singe_store_fp8<4, 3, true>({value})"),
        ElementType::F8E5M2 => format!("singe_store_fp8<5, 2, false>({value})"),
        _ => value.to_owned(),
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "an `F32` literal is its value rounded to `F32`"
)]
#[must_use]
pub fn literal_expression(element: ElementType, literal: Literal) -> String {
    match (element, literal) {
        (_, Literal::Bool(value)) => value.to_string(),
        (ElementType::F64, Literal::Float(value)) => {
            format!("__longlong_as_double(0x{:016x}ULL)", value.to_bits())
        }
        (_, Literal::Float(value)) => {
            format!("__uint_as_float(0x{:08x}U)", (value as f32).to_bits())
        }
        (element, Literal::Int(value)) => {
            format!("static_cast<{}>({value}LL)", compute_type(element))
        }
    }
}

#[must_use]
pub fn unary_expression(op: UnaryOp, element: ElementType, x: &str) -> String {
    if element.is_integer() {
        // Negation wraps as the binary operations do, so the smallest signed value is its own negation and absolute value, as in `PyTorch`.
        let negated = format!("(0ULL - {})", unsigned(x));
        return match op {
            UnaryOp::Neg => negated,
            UnaryOp::BitNot => format!("(~{x})"),
            _ => format!("({x} < 0 ? {negated} : {})", unsigned(x)),
        };
    }

    let suffix = if element == ElementType::F32 { "f" } else { "" };
    let function = |name: &str| format!("{name}{suffix}({x})");
    match op {
        UnaryOp::Neg => format!("(-{x})"),
        UnaryOp::Abs => function("fabs"),
        UnaryOp::Exp => function("exp"),
        UnaryOp::Log => function("log"),
        UnaryOp::Sqrt => function("sqrt"),
        UnaryOp::Rsqrt => function("rsqrt"),
        UnaryOp::Reciprocal => format!("(1.0{suffix} / {x})"),
        UnaryOp::Sin => function("sin"),
        UnaryOp::Cos => function("cos"),
        UnaryOp::Tan => function("tan"),
        UnaryOp::Tanh => function("tanh"),
        UnaryOp::Asin => function("asin"),
        UnaryOp::Acos => function("acos"),
        UnaryOp::Atan => function("atan"),
        UnaryOp::Sinh => function("sinh"),
        UnaryOp::Cosh => function("cosh"),
        UnaryOp::Asinh => function("asinh"),
        UnaryOp::Acosh => function("acosh"),
        UnaryOp::Atanh => function("atanh"),
        UnaryOp::Erf => function("erf"),
        UnaryOp::Log1p => function("log1p"),
        UnaryOp::Expm1 => function("expm1"),
        UnaryOp::Floor => function("floor"),
        UnaryOp::Ceil => function("ceil"),
        UnaryOp::Trunc => function("trunc"),
        UnaryOp::Round => function("rint"),
        UnaryOp::Not => format!("(!{x})"),
        UnaryOp::BitNot => unreachable!("verification allows complements of integers only"),
        UnaryOp::IsNan => format!("isnan({x})"),
    }
}

#[must_use]
pub fn binary_expression(op: BinaryOp, element: ElementType, x: &str, y: &str) -> String {
    let suffix = if element == ElementType::F32 { "f" } else { "" };
    let infix = |operator: &str| format!("({x} {operator} {y})");
    // Integer sums, differences, and products wrap: computed as 64-bit unsigned values, whose low bits the narrowing store keeps, they never overflow a signed type, which C++ leaves undefined.
    let wrapping = |operator: &str| format!("({} {operator} {})", unsigned(x), unsigned(y));
    match op {
        BinaryOp::Add if element.is_integer() => wrapping("+"),
        BinaryOp::Pow if element.is_integer() => format!("singe_integer_power({x}, {y})"),
        BinaryOp::Sub if element.is_integer() => wrapping("-"),
        BinaryOp::Mul if element.is_integer() => wrapping("*"),
        BinaryOp::Add => infix("+"),
        BinaryOp::Sub => infix("-"),
        BinaryOp::Mul => infix("*"),
        BinaryOp::Div => infix("/"),
        BinaryOp::Rem if element.is_integer() => infix("%"),
        BinaryOp::Rem => format!("fmod{suffix}({x}, {y})"),
        BinaryOp::Pow => format!("pow{suffix}({x}, {y})"),
        BinaryOp::Max => format!("singe_max({x}, {y})"),
        BinaryOp::Min => format!("singe_min({x}, {y})"),
        BinaryOp::Atan2 => format!("atan2{suffix}({x}, {y})"),
        BinaryOp::Equal => infix("=="),
        BinaryOp::NotEqual => infix("!="),
        BinaryOp::Less => infix("<"),
        BinaryOp::LessEqual => infix("<="),
        BinaryOp::Greater => infix(">"),
        BinaryOp::GreaterEqual => infix(">="),
        BinaryOp::And => infix("&&"),
        BinaryOp::Or => infix("||"),
        BinaryOp::BitAnd => infix("&"),
        BinaryOp::BitOr => infix("|"),
        BinaryOp::BitXor => infix("^"),
        BinaryOp::ShiftLeft => format!("singe_shift_left({x}, {y})"),
        BinaryOp::ShiftRight => format!("singe_shift_right({x}, {y})"),
    }
}

/// Returns `x` converted to a 64-bit unsigned integer, whose arithmetic wraps.
#[must_use]
pub fn unsigned(x: &str) -> String {
    format!("static_cast<unsigned long long>({x})")
}

/// Returns the template tag `singe_combine` selects `combiner` by.
#[must_use]
pub const fn combiner_tag(combiner: Combiner) -> u8 {
    match combiner {
        Combiner::Sum => 0,
        Combiner::Product => 1,
        Combiner::Max => 2,
        Combiner::Min => 3,
        Combiner::LogSumExp => 4,
    }
}

/// Returns the value a reduction with `combiner` starts from.
#[must_use]
pub fn identity(combiner: Combiner, element: ElementType) -> String {
    let ty = compute_type(element);
    let infinity = match element {
        ElementType::F64 => "__longlong_as_double(0x7ff0000000000000ULL)",
        _ => "__uint_as_float(0x7f800000U)",
    };
    match combiner {
        Combiner::Sum => format!("static_cast<{ty}>(0)"),
        Combiner::Product => format!("static_cast<{ty}>(1)"),
        Combiner::Max | Combiner::LogSumExp if element.is_float() => format!("(-{infinity})"),
        Combiner::Min if element.is_float() => infinity.to_owned(),
        Combiner::Max | Combiner::LogSumExp => integer_limit(element, false),
        Combiner::Min => integer_limit(element, true),
    }
}

/// Returns the largest (`high`) or smallest value of an integer type as a C expression.
#[must_use]
pub fn integer_limit(element: ElementType, high: bool) -> String {
    let ty = compute_type(element);
    let (low, high_value): (i128, i128) = match element {
        ElementType::U8 => (0, 0xff),
        ElementType::I8 => (-0x80, 0x7f),
        ElementType::U16 => (0, 0xffff),
        ElementType::I16 => (-0x8000, 0x7fff),
        ElementType::U32 => (0, 0xffff_ffff),
        ElementType::I32 => (-0x8000_0000, 0x7fff_ffff),
        ElementType::U64 => (0, 0xffff_ffff_ffff_ffff),
        _ => (-0x8000_0000_0000_0000, 0x7fff_ffff_ffff_ffff),
    };
    let value = if high { high_value } else { low };
    if value == -0x8000_0000_0000_0000 {
        // The literal 9223372036854775808 does not fit `long long`, so the minimum is written as a difference.
        return format!("static_cast<{ty}>(-9223372036854775807LL - 1)");
    }
    format!("static_cast<{ty}>({value}ULL)")
}

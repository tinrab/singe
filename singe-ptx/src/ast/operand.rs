use crate::ast::Span;

/// One instruction operand, as written in the source.
#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    /// Named register, variable, label, or function.
    Symbol {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Symbol spelling preserved from the PTX source, including a leading `%`.
        name: String,
    },
    /// Literal constant operand.
    Immediate {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Literal constant represented by this operand.
        value: Immediate,
    },
    /// Bracketed memory address operand, `[base+offset]`.
    Address {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Register or symbol whose value or address is displaced.
        base: String,
        /// Signed byte displacement from the base, zero when omitted.
        offset: i64,
    },
    /// Brace-delimited vector of operands.
    Vector {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Vector components in source order.
        elements: Vec<Self>,
    },
    /// Parenthesized operand tuple.
    Tuple {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Tuple elements in source order.
        elements: Vec<Self>,
    },
    /// Two operands separated by `|`, as in the `setp` destination `%p|%q`.
    Pair {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Operand before the `|` separator.
        first: Box<Self>,
        /// Operand after the `|` separator.
        second: Box<Self>,
    },
    /// Operand with a leading `!`, as in a negated predicate source `!%p`.
    Negated {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
        /// Operand after the `!`.
        operand: Box<Self>,
    },
    /// Discarded destination register, `_`.
    BitBucket {
        /// Byte range of this syntax node in the original PTX source.
        span: Span,
    },
}

/// Integer or floating-point literal used as an instruction operand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Immediate {
    /// Signed integer literal, before target-width conversion.
    Integer(i64),
    /// Unsigned integer literal, before target-width conversion.
    UnsignedInteger(u64),
    /// Floating-point literal.
    Float(FloatLiteral),
}

/// Floating-point literal as written. PTX evaluates floating-point constants in binary64.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FloatLiteral {
    /// Decimal floating-point text before conversion to a target precision.
    Decimal(String),
    /// IEEE 754 binary32 bit pattern from a `0f` literal.
    Binary32(u32),
    /// IEEE 754 binary64 bit pattern from a `0d` literal.
    Binary64(u64),
}

impl Operand {
    /// Returns the byte range occupied by this operand in the original PTX source.
    #[must_use]
    pub fn span(&self) -> Span {
        match self {
            Self::Symbol { span, .. }
            | Self::Immediate { span, .. }
            | Self::Address { span, .. }
            | Self::Vector { span, .. }
            | Self::Tuple { span, .. }
            | Self::Pair { span, .. }
            | Self::Negated { span, .. }
            | Self::BitBucket { span } => *span,
        }
    }
}

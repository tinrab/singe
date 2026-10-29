use crate::ast::{Operand, Span};

/// One PTX instruction, such as `@%p add.rn.f32 %f1, %f2, %f3;`.
#[derive(Debug, Clone, PartialEq)]
pub struct Instruction {
    /// Byte range of this syntax node in the original PTX source.
    pub span: Span,
    /// Predicate guard such as `@%p`, if present.
    pub guard: Option<PredicateGuard>,
    /// Opcode with its qualifiers as written, such as `add.rn.f32`.
    pub head: String,
    /// Instruction operands in source order.
    pub operands: Vec<Operand>,
}

/// Guard that runs an instruction only in lanes where a predicate is true, or false
/// for `@!%p`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredicateGuard {
    /// Byte range of this syntax node in the original PTX source.
    pub span: Span,
    /// Whether the guard is negated with `!`.
    pub negated: bool,
    /// Predicate register name, including its leading `%`.
    pub register: String,
}

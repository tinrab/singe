use crate::{
    ast::{Operand, Span},
    generated::instruction_set::InstructionOpcode,
};

#[derive(Debug, Clone, PartialEq)]
pub struct Instruction {
    pub span: Span,
    pub guard: Option<PredicateGuard>,
    pub opcode: String,
    pub opcode_kind: Option<InstructionOpcode>,
    pub modifiers: Vec<String>,
    pub operands: Vec<Operand>,
    pub setp_details: Option<SetPredicateDetails>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PredicateGuard {
    pub span: Span,
    pub negated: bool,
    pub register: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SetPredicateDetails {
    pub span: Span,
    pub secondary_destination: Option<Operand>,
    pub predicate_input_negated: bool,
}

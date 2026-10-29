use nom::{
    Parser, bytes::complete::take_while1, character::complete::char, combinator::opt,
    multi::separated_list0,
};

use crate::{
    ast::{Instruction, PredicateGuard, Span},
    parser::{
        common::{
            Input, PResult, SyntaxError, SyntaxErrorKind, comma_sep, identifier, lexeme, semicolon,
            text, with_span, ws,
        },
        operand::operand,
    },
};

fn guard_predicate(input: Input<'_>) -> PResult<'_, PredicateGuard> {
    let (input, (span, (_, negated, name))) =
        with_span((char('@'), opt(char('!')), identifier)).parse(input)?;
    Ok((
        input,
        PredicateGuard {
            span,
            negated: negated.is_some(),
            register: name.to_owned(),
        },
    ))
}

pub fn instruction(input: Input<'_>, depth: usize) -> PResult<'_, Instruction> {
    let start = input.location_offset();
    let (input, guard) = opt(lexeme(guard_predicate)).parse(input)?;
    let (input, ()) = ws(input)?;
    let (input, head) =
        take_while1(|c: char| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | ':'))
            .parse(input)?;
    if !head.starts_with(|c: char| c.is_ascii_alphabetic())
        || head.split('.').any(str::is_empty)
        || head
            .split("::")
            .any(|part| part.is_empty() || part.contains(':'))
    {
        return Err(SyntaxError::failure(input, SyntaxErrorKind::Unexpected));
    }
    let (input, operands) =
        separated_list0(comma_sep, lexeme(|input| operand(input, depth))).parse(input)?;
    let (input, ()) = semicolon(input)?;
    Ok((
        input,
        Instruction {
            span: Span::new(start, input.location_offset()),
            guard,
            head: text(head).to_owned(),
            operands,
        },
    ))
}

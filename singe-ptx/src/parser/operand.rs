use nom::{
    Parser,
    branch::alt,
    bytes::complete::{tag, take_while, take_while1},
    character::complete::char,
    combinator::{map, not, opt, recognize},
    multi::{separated_list0, separated_list1},
    sequence::{delimited, pair, preceded},
};

use crate::{
    ast::{Immediate, Operand, Span},
    parser::{
        common::{
            Input, PResult, comma_sep, displacement, float_literal, identifier, integer_literal,
            lexeme, text, with_span, ws,
        },
        depth,
    },
};

fn register_name(input: Input<'_>) -> PResult<'_, &str> {
    map(
        recognize(preceded(
            char('%'),
            pair(
                take_while1(|c: char| c.is_ascii_alphanumeric() || c == '_' || c == '$'),
                take_while(|c: char| c == '.' || c.is_ascii_alphanumeric() || c == '_' || c == '$'),
            ),
        )),
        text,
    )
    .parse(input)
}

fn register(input: Input<'_>) -> PResult<'_, Operand> {
    let (input, (span, name)) = with_span(register_name).parse(input)?;
    Ok((
        input,
        Operand::Symbol {
            span,
            name: name.to_owned(),
        },
    ))
}

fn immediate(input: Input<'_>) -> PResult<'_, Operand> {
    let (input, (span, value)) = with_span(alt((
        map(float_literal, Immediate::Float),
        map(integer_literal, |(value, unsigned)| {
            if unsigned {
                Immediate::UnsignedInteger(value)
            } else {
                Immediate::Integer(value.cast_signed())
            }
        }),
    )))
    .parse(input)?;
    Ok((input, Operand::Immediate { span, value }))
}

fn address_operand(input: Input<'_>) -> PResult<'_, Operand> {
    let (input, (span, (_, (), base, offset, _))) = with_span((
        char('['),
        ws,
        alt((register_name, identifier)),
        opt(lexeme(displacement)),
        lexeme(char(']')),
    ))
    .parse(input)?;
    Ok((
        input,
        Operand::Address {
            span,
            base: base.to_owned(),
            offset: offset.unwrap_or(0),
        },
    ))
}

fn vector_operand(input: Input<'_>, depth: usize) -> PResult<'_, Operand> {
    let (input, (span, elements)) = with_span(delimited(
        lexeme(char('{')),
        separated_list1(comma_sep, lexeme(|input| operand(input, depth + 1))),
        lexeme(char('}')),
    ))
    .parse(input)?;
    Ok((input, Operand::Vector { span, elements }))
}

fn tuple_operand(input: Input<'_>, depth: usize) -> PResult<'_, Operand> {
    let (input, (span, elements)) = with_span(delimited(
        lexeme(char('(')),
        separated_list0(
            comma_sep,
            lexeme(preceded(not(char(')')), |input| operand(input, depth + 1))),
        ),
        lexeme(char(')')),
    ))
    .parse(input)?;
    Ok((input, Operand::Tuple { span, elements }))
}

fn symbol_operand(input: Input<'_>) -> PResult<'_, Operand> {
    let (input, (span, name)) = with_span(alt((identifier, map(tag("_"), text)))).parse(input)?;
    Ok((
        input,
        if name == "_" {
            Operand::BitBucket { span }
        } else {
            Operand::Symbol {
                span,
                name: name.to_owned(),
            }
        },
    ))
}

fn atom(input: Input<'_>, depth: usize) -> PResult<'_, Operand> {
    let start = input.location_offset();
    let (input, negated) = opt(char('!')).parse(input)?;
    let child_depth = depth + usize::from(negated.is_some());
    depth::check(child_depth, input)?;
    let (input, operand) = alt((
        register,
        |input| vector_operand(input, child_depth),
        |input| tuple_operand(input, child_depth),
        address_operand,
        immediate,
        symbol_operand,
    ))
    .parse(input)?;
    Ok((
        input,
        if negated.is_some() {
            Operand::Negated {
                span: Span::new(start, input.location_offset()),
                operand: Box::new(operand),
            }
        } else {
            operand
        },
    ))
}

pub fn operand(input: Input<'_>, depth: usize) -> PResult<'_, Operand> {
    depth::check(depth, input)?;
    let start = input.location_offset();
    let (input, first) = atom(input, depth)?;
    let (input, second) = opt(preceded(
        lexeme(char('|')),
        lexeme(|input| atom(input, depth)),
    ))
    .parse(input)?;
    if let Some(second) = &second {
        depth::check(
            depth + 1 + operand_depth(&first).max(operand_depth(second)),
            input,
        )?;
    }
    Ok((
        input,
        match second {
            Some(second) => Operand::Pair {
                span: Span::new(start, input.location_offset()),
                first: Box::new(first),
                second: Box::new(second),
            },
            None => first,
        },
    ))
}

fn operand_depth(value: &Operand) -> usize {
    match value {
        Operand::Vector { elements, .. } | Operand::Tuple { elements, .. } => elements
            .iter()
            .map(operand_depth)
            .max()
            .map_or(0, |depth| depth + 1),
        Operand::Pair { first, second, .. } => 1 + operand_depth(first).max(operand_depth(second)),
        Operand::Negated { operand, .. } => 1 + operand_depth(operand),
        Operand::Symbol { .. }
        | Operand::Immediate { .. }
        | Operand::Address { .. }
        | Operand::BitBucket { .. } => 0,
    }
}

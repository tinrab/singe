use nom::{
    Parser,
    branch::alt,
    character::complete::char,
    combinator::{cut, map, opt, peek},
    multi::many0,
    sequence::{preceded, terminated},
};

use crate::{
    ast::{Function, Span, Statement},
    parser::{
        common,
        common::{Input, PResult, identifier, keyword, lexeme, linking_directive, semicolon},
        depth, directive,
        directive::{attributes, directive_statement, function_directive, param_list},
        instruction::instruction,
    },
};

fn label(input: Input<'_>) -> PResult<'_, Statement> {
    let start = input.location_offset();
    let (input, name) = lexeme(identifier).parse(input)?;
    let (input, _) = lexeme(char(':')).parse(input)?;
    Ok((
        input,
        Statement::Label {
            span: Span::new(start, input.location_offset()),
            name: name.to_string(),
        },
    ))
}

fn block(input: Input<'_>, depth: usize) -> PResult<'_, Statement> {
    let start = input.location_offset();
    peek(char('{')).parse(input)?;
    let (input, statements) = body(input, depth + 1)?;
    Ok((
        input,
        Statement::Block {
            span: Span::new(start, input.location_offset()),
            statements,
        },
    ))
}

fn statement(input: Input<'_>, depth: usize) -> PResult<'_, Statement> {
    alt((
        |input| block(input, depth),
        label,
        map(
            |input| directive::variable_decl(input, depth),
            Statement::Variable,
        ),
        map(directive_statement, Statement::Directive),
        map(|input| instruction(input, depth), Statement::Instruction),
    ))
    .parse(input)
}

fn body(input: Input<'_>, depth: usize) -> PResult<'_, Vec<Statement>> {
    depth::check(depth, input)?;
    preceded(
        lexeme(char('{')),
        cut(terminated(
            many0(lexeme(|input| statement(input, depth))),
            lexeme(char('}')),
        )),
    )
    .parse(input)
}

fn has_return_param_list(input: Input<'_>) -> PResult<'_, ()> {
    let (input, _) = lexeme(char('(')).parse(input)?;
    let (input, _) = lexeme(alt((keyword(".param"), keyword(".reg")))).parse(input)?;
    Ok((input, ()))
}

pub fn function(input: Input<'_>) -> PResult<'_, Function> {
    let (input, ()) = common::ws(input)?;
    let start = input.location_offset();
    let (input, linkage) = opt(lexeme(linking_directive)).parse(input)?;
    let (input, kind) = lexeme(alt((keyword(".entry"), keyword(".func")))).parse(input)?;
    let entry = *kind.fragment() == ".entry";
    let (input, attributes) = opt(lexeme(attributes)).parse(input)?;

    let (input, return_params) = if !entry && peek(has_return_param_list).parse(input).is_ok() {
        map(param_list, Some).parse(input)?
    } else {
        (input, None)
    };

    let (input, name) = lexeme(identifier).parse(input)?;
    let (input, params) = opt(param_list).parse(input)?;
    let (input, directives) = many0(lexeme(function_directive)).parse(input)?;
    let (input, fn_body) = opt(|input| body(input, 0)).parse(input)?;

    let input = if fn_body.is_none() {
        let (input, ()) = semicolon(input)?;
        input
    } else {
        input
    };

    Ok((
        input,
        Function {
            span: Span::new(start, input.location_offset()),
            linkage,
            entry,
            name: name.to_string(),
            attributes: attributes.unwrap_or_default(),
            directives,
            params: params.unwrap_or_default(),
            return_params: return_params.unwrap_or_default(),
            body: fn_body,
        },
    ))
}

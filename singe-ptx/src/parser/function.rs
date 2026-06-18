use nom::{
    Parser,
    branch::alt,
    bytes::complete::tag,
    character::complete::char,
    combinator::{map, opt, peek},
    multi::many0,
    sequence::delimited,
};

use crate::{
    ast::{Function, Statement},
    parser::{
        common::{PResult, identifier, lexeme, linking_directive, semicolon},
        directive::{attributes, directive_statement, function_directive, param_list},
        instruction::instruction,
        offset,
    },
};

fn label<'a>(source: &'a str, input: &'a str) -> PResult<'a, Statement> {
    let start = offset(source, input);
    let (input, name) = lexeme(identifier).parse(input)?;
    let (input, _) = lexeme(char(':')).parse(input)?;
    Ok((
        input,
        Statement::Label {
            span: crate::ast::Span::new(start, offset(source, input)),
            name: name.to_string(),
        },
    ))
}

fn block<'a>(source: &'a str, input: &'a str) -> PResult<'a, Statement> {
    let start = offset(source, input);
    let (input, statements) = body(source, input)?;
    Ok((
        input,
        Statement::Block {
            span: crate::ast::Span::new(start, offset(source, input)),
            statements,
        },
    ))
}

fn statement<'a>(source: &'a str, input: &'a str) -> PResult<'a, Statement> {
    alt((
        |input| block(source, input),
        |input| label(source, input),
        map(
            |input| super::directive::variable_decl(source, input),
            Statement::Variable,
        ),
        map(
            |input| directive_statement(source, input),
            Statement::Directive,
        ),
        map(|input| instruction(source, input), Statement::Instruction),
    ))
    .parse(input)
}

fn body<'a>(source: &'a str, input: &'a str) -> PResult<'a, Vec<Statement>> {
    delimited(
        lexeme(char('{')),
        many0(lexeme(|input| statement(source, input))),
        lexeme(char('}')),
    )
    .parse(input)
}

fn has_return_param_list(input: &str) -> PResult<'_, ()> {
    let (input, _) = lexeme(char('(')).parse(input)?;
    let (input, _) = lexeme(alt((tag(".param"), tag(".reg")))).parse(input)?;
    Ok((input, ()))
}

pub fn function<'a>(source: &'a str, input: &'a str) -> PResult<'a, Function> {
    let (input, _) = super::common::ws(input)?;
    let start = offset(source, input);
    let (input, linkage) = opt(lexeme(linking_directive)).parse(input)?;
    let (input, kind) = lexeme(alt((tag(".entry"), tag(".func")))).parse(input)?;
    let entry = kind == ".entry";
    let (input, attributes) = opt(lexeme(|input| attributes(source, input))).parse(input)?;

    let (input, return_params) = if !entry && peek(has_return_param_list).parse(input).is_ok() {
        map(|input| param_list(source, input), Some).parse(input)?
    } else {
        (input, None)
    };

    let (input, name) = lexeme(identifier).parse(input)?;
    let (input, params) = opt(|input| param_list(source, input)).parse(input)?;
    let (input, directives) =
        many0(lexeme(|input| function_directive(source, input))).parse(input)?;
    let (input, fn_body) = opt(|input| body(source, input)).parse(input)?;

    let input = if fn_body.is_none() {
        let (input, _) = semicolon(input)?;
        input
    } else {
        input
    };

    Ok((
        input,
        Function {
            span: crate::ast::Span::new(start, offset(source, input)),
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

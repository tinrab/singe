use nom::{
    Parser,
    bytes::complete::tag,
    character::complete::char,
    combinator::opt,
    multi::{many0, separated_list1},
};

use crate::{
    ast::{AddressSize, Module, TargetDirective, VersionDirective},
    parser::{
        common::{PResult, comma_sep, decimal_integer, identifier, lexeme, with_span, ws},
        directive::top_level_item,
    },
};

fn non_negative_component(
    value: i64,
    input: &str,
) -> Result<u32, nom::Err<nom::error::Error<&str>>> {
    u32::try_from(value)
        .map_err(|_| nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Digit)))
}

fn version<'a>(source: &'a str, input: &'a str) -> PResult<'a, VersionDirective> {
    let (input, _) = ws(input)?;
    let (input, (span, (_, _, major, _, minor))) = with_span(
        source,
        (
            tag(".version"),
            ws,
            decimal_integer,
            char('.'),
            decimal_integer,
        ),
    )
    .parse(input)?;
    Ok((
        input,
        VersionDirective {
            span,
            major: non_negative_component(major, input)?,
            minor: non_negative_component(minor, input)?,
        },
    ))
}

fn target_specifier(input: &str) -> PResult<'_, String> {
    let (input, _) = ws(input)?;
    let (input, name) = identifier(input)?;
    Ok((input, name.to_string()))
}

fn target<'a>(source: &'a str, input: &'a str) -> PResult<'a, TargetDirective> {
    let (input, _) = ws(input)?;
    let (input, (span, (_, specifiers))) = with_span(
        source,
        (tag(".target"), separated_list1(comma_sep, target_specifier)),
    )
    .parse(input)?;
    Ok((input, TargetDirective { span, specifiers }))
}

fn address_size(input: &str) -> PResult<'_, AddressSize> {
    let (input, _) = lexeme(tag(".address_size")).parse(input)?;
    let (input, _) = ws(input)?;
    let (input, size) = decimal_integer(input)?;
    match size {
        32 => Ok((input, AddressSize::Bits32)),
        64 => Ok((input, AddressSize::Bits64)),
        _ => Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Tag,
        ))),
    }
}

pub fn module(input: &str) -> PResult<'_, Module> {
    let source = input;
    let (input, ver) = version(source, input)?;
    let (input, tgt) = target(source, input)?;
    let (input, addr) = opt(address_size).parse(input)?;
    let (input, items) = many0(lexeme(|input| top_level_item(source, input))).parse(input)?;
    let (input, _) = ws(input)?;
    Ok((
        input,
        Module {
            span: crate::ast::Span::new(0, source.len() - input.len()),
            version: ver,
            target: tgt,
            address_size: addr.unwrap_or(AddressSize::Bits32),
            items,
        },
    ))
}

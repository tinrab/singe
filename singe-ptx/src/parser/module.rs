use nom::{
    Parser,
    character::complete::char,
    combinator::opt,
    multi::{many0, separated_list1},
};

use crate::{
    ast::{AddressSize, Module, Span, TargetDirective, VersionDirective},
    parser::{
        common::{
            Input, PResult, SyntaxError, comma_sep, decimal, identifier, keyword, lexeme,
            with_span, ws,
        },
        directive::top_level_item,
    },
};

fn version(input: Input<'_>) -> PResult<'_, VersionDirective> {
    let (input, ()) = ws(input)?;
    let (input, (span, (_, (), major, _, minor))) =
        with_span((keyword(".version"), ws, decimal, char('.'), decimal)).parse(input)?;
    Ok((input, VersionDirective { span, major, minor }))
}

fn target_specifier(input: Input<'_>) -> PResult<'_, String> {
    let (input, ()) = ws(input)?;
    let (input, name) = identifier(input)?;
    Ok((input, name.to_string()))
}

fn target(input: Input<'_>) -> PResult<'_, TargetDirective> {
    let (input, ()) = ws(input)?;
    let (input, (span, (_, specifiers))) = with_span((
        keyword(".target"),
        separated_list1(comma_sep, target_specifier),
    ))
    .parse(input)?;
    Ok((input, TargetDirective { span, specifiers }))
}

fn address_size(input: Input<'_>) -> PResult<'_, AddressSize> {
    let (input, _) = lexeme(keyword(".address_size")).parse(input)?;
    let (input, ()) = ws(input)?;
    let (input, size) = decimal::<u32>(input)?;
    match size {
        32 => Ok((input, AddressSize::Bits32)),
        64 => Ok((input, AddressSize::Bits64)),
        _ => Err(SyntaxError::unexpected(input)),
    }
}

pub fn module(input: Input<'_>) -> PResult<'_, Module> {
    let (input, ver) = version(input)?;
    let (input, tgt) = target(input)?;
    let (input, addr) = opt(address_size).parse(input)?;
    let (input, items) = many0(lexeme(top_level_item)).parse(input)?;
    let (input, ()) = ws(input)?;
    Ok((
        input,
        Module {
            span: Span::new(0, input.location_offset()),
            version: ver,
            target: tgt,
            address_size: addr.unwrap_or(AddressSize::Bits32),
            items,
        },
    ))
}

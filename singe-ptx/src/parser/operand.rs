use nom::{
    Parser,
    branch::alt,
    bytes::complete::{tag, take_while, take_while1},
    character::complete::char,
    combinator::{opt, recognize},
    multi::separated_list1,
    sequence::{delimited, pair, preceded},
};

use crate::{
    ast::{Address, Immediate, Operand},
    parser::common::{
        PResult, binary_integer, comma_sep, decimal_integer, float_literal, hex_integer,
        identifier, lexeme, octal_integer, with_span, ws,
    },
};

fn register<'a>(source: &'a str, input: &'a str) -> PResult<'a, Operand> {
    let (input, (span, name)) = with_span(
        source,
        recognize(preceded(
            char('%'),
            pair(
                take_while1(|c: char| c.is_ascii_alphanumeric() || c == '_'),
                take_while(|c: char| c == '.' || c.is_ascii_alphanumeric() || c == '_'),
            ),
        )),
    )
    .parse(input)?;
    Ok((
        input,
        Operand::Register {
            span,
            name: name.to_string(),
        },
    ))
}

fn register_name(input: &str) -> PResult<'_, &str> {
    recognize(preceded(
        char('%'),
        pair(
            take_while1(|c: char| c.is_ascii_alphanumeric() || c == '_'),
            take_while(|c: char| c == '.' || c.is_ascii_alphanumeric() || c == '_'),
        ),
    ))
    .parse(input)
}

fn hex_float_single<'a>(source: &'a str, input: &'a str) -> PResult<'a, Immediate> {
    let (input, (span, (_, digits))) = with_span(
        source,
        pair(alt((tag("0f"), tag("0F"))), take_while_m_n_hex::<8>),
    )
    .parse(input)?;
    let bits = u32::from_str_radix(digits, 16).map_err(|_| {
        nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::HexDigit,
        ))
    })?;
    Ok((
        input,
        Immediate::HexFloat32 {
            span,
            value: f32::from_bits(bits),
        },
    ))
}

fn hex_float_double<'a>(source: &'a str, input: &'a str) -> PResult<'a, Immediate> {
    let (input, (span, (_, digits))) = with_span(
        source,
        pair(alt((tag("0d"), tag("0D"))), take_while_m_n_hex::<16>),
    )
    .parse(input)?;
    let bits = u64::from_str_radix(digits, 16).map_err(|_| {
        nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::HexDigit,
        ))
    })?;
    Ok((
        input,
        Immediate::HexFloat64 {
            span,
            value: f64::from_bits(bits),
        },
    ))
}

fn take_while_m_n_hex<const N: usize>(input: &str) -> PResult<'_, &str> {
    let (input, digits) = recognize(take_while1(|c: char| c.is_ascii_hexdigit())).parse(input)?;
    if digits.len() == N {
        Ok((input, digits))
    } else {
        Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::HexDigit,
        )))
    }
}

fn float_immediate<'a>(source: &'a str, input: &'a str) -> PResult<'a, Immediate> {
    let (input, (span, value)) = with_span(source, float_literal).parse(input)?;
    Ok((input, Immediate::Float { span, value }))
}

fn hex_immediate<'a>(source: &'a str, input: &'a str) -> PResult<'a, Immediate> {
    let (input, (span, value)) = with_span(source, hex_integer).parse(input)?;
    Ok((input, Immediate::UnsignedInteger { span, value }))
}

fn octal_immediate<'a>(source: &'a str, input: &'a str) -> PResult<'a, Immediate> {
    let (input, (span, value)) = with_span(source, octal_integer).parse(input)?;
    Ok((input, Immediate::UnsignedInteger { span, value }))
}

fn binary_immediate<'a>(source: &'a str, input: &'a str) -> PResult<'a, Immediate> {
    let (input, (span, value)) = with_span(source, binary_integer).parse(input)?;
    Ok((input, Immediate::UnsignedInteger { span, value }))
}

fn decimal_immediate<'a>(source: &'a str, input: &'a str) -> PResult<'a, Immediate> {
    let (input, (span, value)) = with_span(source, decimal_integer).parse(input)?;
    Ok((input, Immediate::Integer { span, value }))
}

fn immediate<'a>(source: &'a str, input: &'a str) -> PResult<'a, Operand> {
    let (input, (span, immediate)) = with_span(
        source,
        alt((
            |input| hex_float_single(source, input),
            |input| hex_float_double(source, input),
            |input| float_immediate(source, input),
            |input| hex_immediate(source, input),
            |input| octal_immediate(source, input),
            |input| binary_immediate(source, input),
            |input| decimal_immediate(source, input),
        )),
    )
    .parse(input)?;
    Ok((input, Operand::Immediate { span, immediate }))
}

fn address_operand<'a>(source: &'a str, input: &'a str) -> PResult<'a, Operand> {
    let (input, (span, (_, _, base, _, offset, _, _))) = with_span(
        source,
        (
            char('['),
            ws,
            alt((register_name, identifier)),
            ws,
            opt(pair(
                preceded(ws, alt((char('+'), char('-')))),
                preceded(ws, decimal_integer),
            )),
            ws,
            char(']'),
        ),
    )
    .parse(input)?;

    let offset = offset.map(|(sign, value)| {
        if sign == '-' {
            -value.abs()
        } else {
            value.abs()
        }
    });

    let addr = if base.starts_with('%') {
        Address::RegisterOffset {
            span,
            register: base.to_string(),
            offset: offset.unwrap_or(0),
        }
    } else {
        match offset {
            Some(off) => Address::SymbolOffset {
                span,
                symbol: base.to_string(),
                offset: off,
            },
            None => Address::Symbol {
                span,
                name: base.to_string(),
            },
        }
    };
    Ok((
        input,
        Operand::Address {
            span,
            address: addr,
        },
    ))
}

fn vector_operand<'a>(source: &'a str, input: &'a str) -> PResult<'a, Operand> {
    let (input, (span, elements)) = with_span(
        source,
        delimited(
            lexeme(char('{')),
            separated_list1(comma_sep, lexeme(|input| operand(source, input))),
            lexeme(char('}')),
        ),
    )
    .parse(input)?;
    Ok((input, Operand::Vector { span, elements }))
}

fn bit_bucket<'a>(source: &'a str, input: &'a str) -> PResult<'a, Operand> {
    let (input, (span, _)) = with_span(source, char('_')).parse(input)?;
    Ok((input, Operand::BitBucket { span }))
}

fn label_operand<'a>(source: &'a str, input: &'a str) -> PResult<'a, Operand> {
    let (input, (span, name)) = with_span(source, identifier).parse(input)?;
    Ok((
        input,
        Operand::Label {
            span,
            name: name.to_string(),
        },
    ))
}

pub fn operand<'a>(source: &'a str, input: &'a str) -> PResult<'a, Operand> {
    alt((
        |input| register(source, input),
        |input| vector_operand(source, input),
        |input| address_operand(source, input),
        |input| immediate(source, input),
        |input| bit_bucket(source, input),
        |input| label_operand(source, input),
    ))
    .parse(input)
}

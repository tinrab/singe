use std::str::FromStr;

use nom::{
    Err as NomErr, IResult, Input as NomInput, Parser,
    branch::alt,
    bytes::complete::{tag, take_while, take_while1},
    character::complete::{char, multispace1, one_of, satisfy},
    combinator::{map, not, opt, recognize, value},
    error::{ErrorKind, ParseError as NomParseError},
    multi::many0,
    sequence::{delimited, pair, preceded, terminated},
};

use nom_locate::LocatedSpan;

use crate::ast::{FloatLiteral, LinkingDirective, ScalarType, Span, StateSpace, VectorSize};

/// Parser input: the unparsed suffix of the source, which knows its own byte offset.
pub type Input<'a> = LocatedSpan<&'a str>;

/// Returns the source text an input or recognized token covers.
pub fn text(input: Input<'_>) -> &str {
    input.fragment()
}

/// Skips `count` bytes of `input`.
fn advance(input: Input<'_>, count: usize) -> Input<'_> {
    input.take_from(count)
}

/// Consumes `prefix` from the start of `input`, ignoring ASCII case.
fn strip_prefix_ignore_case<'a>(input: Input<'a>, prefix: &str) -> Option<Input<'a>> {
    input
        .fragment()
        .get(..prefix.len())
        .filter(|head| head.eq_ignore_ascii_case(prefix))
        .map(|_| advance(input, prefix.len()))
}

/// Why a parser stopped, with the input remaining at that point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyntaxErrorKind {
    /// The input does not match the grammar here.
    Unexpected,
    /// Nesting or constructed expression depth exceeds the parser limit.
    DepthLimitExceeded,
}

/// Parser error carrying the remaining input, from which
/// [`parse_module`](crate::parser::parse_module) computes the error span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyntaxError<'a> {
    pub input: Input<'a>,
    pub kind: SyntaxErrorKind,
}

impl<'a> SyntaxError<'a> {
    /// Returns a recoverable mismatch, which lets `alt` try other branches.
    pub fn unexpected(input: Input<'a>) -> NomErr<Self> {
        NomErr::Error(Self {
            input,
            kind: SyntaxErrorKind::Unexpected,
        })
    }

    /// Returns a failure that stops the whole parse at `input`.
    pub fn failure(input: Input<'a>, kind: SyntaxErrorKind) -> NomErr<Self> {
        NomErr::Failure(Self { input, kind })
    }
}

impl<'a> NomParseError<Input<'a>> for SyntaxError<'a> {
    fn from_error_kind(input: Input<'a>, _kind: ErrorKind) -> Self {
        Self {
            input,
            kind: SyntaxErrorKind::Unexpected,
        }
    }

    fn append(_input: Input<'a>, _kind: ErrorKind, other: Self) -> Self {
        other
    }
}

pub type PResult<'a, T> = IResult<Input<'a>, T, SyntaxError<'a>>;

pub fn keyword<'a>(
    token: &'static str,
) -> impl Parser<Input<'a>, Output = Input<'a>, Error = SyntaxError<'a>> {
    terminated(
        tag(token),
        not(satisfy(|character: char| {
            character.is_ascii_alphanumeric() || matches!(character, '_' | '$')
        })),
    )
}

fn line_comment(input: Input<'_>) -> PResult<'_, ()> {
    value((), pair(tag("//"), take_while(|c| c != '\n'))).parse(input)
}

fn block_comment(input: Input<'_>) -> PResult<'_, ()> {
    let (input, _) = tag("/*").parse(input)?;
    // An unterminated comment consumes the rest of the input.
    input.fragment().find("*/").map_or_else(
        || {
            Err(SyntaxError::failure(
                advance(input, input.len()),
                SyntaxErrorKind::Unexpected,
            ))
        },
        |end| Ok((advance(input, end + 2), ())),
    )
}

fn preprocessor_line(input: Input<'_>) -> PResult<'_, ()> {
    value((), pair(char('#'), take_while(|c| c != '\n'))).parse(input)
}

pub fn ws(input: Input<'_>) -> PResult<'_, ()> {
    value(
        (),
        many0(alt((
            value((), multispace1),
            line_comment,
            block_comment,
            preprocessor_line,
        ))),
    )
    .parse(input)
}

pub fn lexeme<'a, O>(
    mut inner: impl Parser<Input<'a>, Output = O, Error = SyntaxError<'a>>,
) -> impl Parser<Input<'a>, Output = O, Error = SyntaxError<'a>> {
    move |input: Input<'a>| {
        let (input, ()) = ws(input)?;
        inner.parse(input)
    }
}

pub fn with_span<'a, O>(
    mut inner: impl Parser<Input<'a>, Output = O, Error = SyntaxError<'a>>,
) -> impl Parser<Input<'a>, Output = (Span, O), Error = SyntaxError<'a>> {
    move |input: Input<'a>| {
        let start = input.location_offset();
        let (input, output) = inner.parse(input)?;
        let end = input.location_offset();
        Ok((input, (Span::new(start, end), output)))
    }
}

pub fn identifier(input: Input<'_>) -> PResult<'_, &str> {
    map(
        alt((
            recognize(pair(
                take_while1(|c: char| c.is_ascii_alphabetic()),
                take_while(|c: char| c.is_ascii_alphanumeric() || c == '_' || c == '$'),
            )),
            recognize(pair(
                one_of("_$%"),
                take_while1(|c: char| c.is_ascii_alphanumeric() || c == '_' || c == '$'),
            )),
        )),
        text,
    )
    .parse(input)
}

/// Parses unsigned integer digits in hexadecimal (`0x`), binary (`0b`), octal (a leading
/// `0`), or decimal, with an optional `U` suffix. Returns the value and whether the
/// suffix was present.
fn integer_digits(input: Input<'_>) -> PResult<'_, (u64, bool)> {
    let original = input;
    let (input, (radix, digits)) = if let Some(tail) = strip_prefix_ignore_case(input, "0x") {
        let (rest, digits) = take_while1(|c: char| c.is_ascii_hexdigit()).parse(tail)?;
        (rest, (16, digits))
    } else if let Some(tail) = strip_prefix_ignore_case(input, "0b") {
        let (rest, digits) = take_while1(|c: char| c == '0' || c == '1').parse(tail)?;
        (rest, (2, digits))
    } else {
        let (rest, digits) = take_while1(|c: char| c.is_ascii_digit()).parse(input)?;
        (rest, (if digits.starts_with('0') { 8 } else { 10 }, digits))
    };
    let (input, suffix) = opt(char('U')).parse(input)?;
    let value =
        u64::from_str_radix(&digits, radix).map_err(|_| SyntaxError::unexpected(original))?;
    Ok((input, (value, suffix.is_some())))
}

/// Parses an integer literal with an optional sign. Returns its 64-bit pattern and
/// whether PTX treats it as unsigned: it has a `U` suffix or exceeds `i64::MAX`.
pub fn integer_literal(input: Input<'_>) -> PResult<'_, (u64, bool)> {
    let (input, sign) = opt(one_of("+-")).parse(input)?;
    let (input, (value, suffix)) = integer_digits(input)?;
    let unsigned = suffix || value > i64::MAX.cast_unsigned();
    let value = if sign == Some('-') {
        value.wrapping_neg()
    } else {
        value
    };
    Ok((input, (value, unsigned)))
}

/// Parses an unsigned integer literal in any PTX radix.
pub fn unsigned_literal(input: Input<'_>) -> PResult<'_, u64> {
    map(integer_digits, |(value, _)| value).parse(input)
}

/// Parses an address displacement written as `+ literal` or `- literal`.
pub fn displacement(input: Input<'_>) -> PResult<'_, i64> {
    let start = input;
    let (input, (sign, magnitude)) =
        (lexeme(one_of("+-")), lexeme(unsigned_literal)).parse(input)?;
    let magnitude = i128::from(magnitude);
    i64::try_from(if sign == '-' { -magnitude } else { magnitude })
        .map(|value| (input, value))
        .map_err(|_| SyntaxError::failure(start, SyntaxErrorKind::Unexpected))
}

/// Parses a double-quoted string. A backslash makes the next character literal.
pub fn string_literal(input: Input<'_>) -> PResult<'_, String> {
    let (rest, _) = char('"').parse(input)?;
    let mut value = String::new();
    let mut chars = rest.fragment().char_indices();
    while let Some((index, character)) = chars.next() {
        match character {
            '"' => return Ok((advance(rest, index + 1), value)),
            '\\' => match chars.next() {
                Some((_, escaped)) => value.push(escaped),
                None => break,
            },
            character => value.push(character),
        }
    }
    Err(SyntaxError::failure(
        advance(rest, rest.len()),
        SyntaxErrorKind::Unexpected,
    ))
}

pub fn float_literal(input: Input<'_>) -> PResult<'_, FloatLiteral> {
    if let Some(tail) = strip_prefix_ignore_case(input, "0f") {
        return hex_float(tail, 8).map(|(remaining, bits)| {
            (
                remaining,
                FloatLiteral::Binary32(
                    u32::try_from(bits).expect("eight hexadecimal digits fit u32"),
                ),
            )
        });
    }
    if let Some(tail) = strip_prefix_ignore_case(input, "0d") {
        return hex_float(tail, 16)
            .map(|(remaining, bits)| (remaining, FloatLiteral::Binary64(bits)));
    }
    let fraction = alt((
        recognize((
            take_while1(|c: char| c.is_ascii_digit()),
            char('.'),
            opt(take_while1(|c: char| c.is_ascii_digit())),
        )),
        recognize((char('.'), take_while1(|c: char| c.is_ascii_digit()))),
    ));
    let mut parser = recognize((
        opt(one_of("+-")),
        alt((
            recognize((fraction, opt(decimal_exponent))),
            recognize((take_while1(|c: char| c.is_ascii_digit()), decimal_exponent)),
        )),
    ));

    let (input, literal) = parser.parse(input)?;
    Ok((input, FloatLiteral::Decimal(text(literal).to_owned())))
}

fn decimal_exponent(input: Input<'_>) -> PResult<'_, Input<'_>> {
    recognize((
        one_of("eE"),
        opt(one_of("+-")),
        take_while1(|c: char| c.is_ascii_digit()),
    ))
    .parse(input)
}

fn hex_float(input: Input<'_>, width: usize) -> PResult<'_, u64> {
    let (remaining, digits) = take_while1(|c: char| c.is_ascii_hexdigit()).parse(input)?;
    if digits.len() != width {
        return Err(SyntaxError::unexpected(input));
    }
    let bits = u64::from_str_radix(&digits, 16).map_err(|_| SyntaxError::unexpected(input))?;
    Ok((remaining, bits))
}

pub fn comma_sep(input: Input<'_>) -> PResult<'_, ()> {
    value((), delimited(ws, char(','), ws)).parse(input)
}

pub fn semicolon(input: Input<'_>) -> PResult<'_, ()> {
    value((), lexeme(char(';'))).parse(input)
}

pub fn scalar_type(input: Input<'_>) -> PResult<'_, ScalarType> {
    let (rest, name) = preceded(
        char('.'),
        take_while1(|c: char| c.is_ascii_alphanumeric() || matches!(c, '_' | '$')),
    )
    .parse(input)?;
    ScalarType::from_suffix(&name)
        .map(|ty| (rest, ty))
        .ok_or_else(|| SyntaxError::unexpected(input))
}

pub fn state_space(input: Input<'_>) -> PResult<'_, StateSpace> {
    alt((
        value(StateSpace::SReg, keyword(".sreg")),
        value(StateSpace::Shared, keyword(".shared")),
        value(StateSpace::Reg, keyword(".reg")),
        value(StateSpace::Const, keyword(".const")),
        value(StateSpace::Global, keyword(".global")),
        value(StateSpace::Local, keyword(".local")),
        value(StateSpace::Param, keyword(".param")),
        value(StateSpace::Tex, keyword(".tex")),
    ))
    .parse(input)
}

pub fn linking_directive(input: Input<'_>) -> PResult<'_, LinkingDirective> {
    alt((
        value(LinkingDirective::Extern, keyword(".extern")),
        value(LinkingDirective::Visible, keyword(".visible")),
        value(LinkingDirective::Weak, keyword(".weak")),
        value(LinkingDirective::Common, keyword(".common")),
    ))
    .parse(input)
}

pub fn vector_size(input: Input<'_>) -> PResult<'_, VectorSize> {
    alt((
        value(VectorSize::V2, keyword(".v2")),
        value(VectorSize::V4, keyword(".v4")),
    ))
    .parse(input)
}

/// Parses decimal digits without a radix prefix, as in versions and directive counts.
pub fn decimal<T: FromStr>(input: Input<'_>) -> PResult<'_, T> {
    let (rest, digits) = take_while1(|c: char| c.is_ascii_digit()).parse(input)?;
    let value = digits.parse().map_err(|_| SyntaxError::unexpected(input))?;
    Ok((rest, value))
}

pub fn alignment(input: Input<'_>) -> PResult<'_, u32> {
    preceded(keyword(".align"), lexeme(decimal)).parse(input)
}

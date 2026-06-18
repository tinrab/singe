use nom::{
    IResult, Parser,
    branch::alt,
    bytes::complete::{tag, take_while, take_while1},
    character::complete::{char, multispace1},
    combinator::{opt, recognize, value},
    multi::many0,
    sequence::{delimited, pair, preceded},
};

use crate::ast::{
    CompareOp, FloatRoundingMode, IntRoundingMode, LinkingDirective, MulMode, ScalarType, Span,
    StateSpace, VectorSize,
};

pub type PResult<'a, T> = IResult<&'a str, T>;

fn line_comment(input: &str) -> PResult<'_, ()> {
    value((), pair(tag("//"), take_while(|c| c != '\n'))).parse(input)
}

fn block_comment(input: &str) -> PResult<'_, ()> {
    let (input, _) = tag("/*").parse(input)?;
    let mut chars = input.char_indices();
    while let Some((i, c)) = chars.next() {
        if c == '*' && matches!(chars.clone().next(), Some((_, '/'))) {
            return Ok((&input[i + 2..], ()));
        }
    }
    Err(nom::Err::Error(nom::error::Error::new(
        input,
        nom::error::ErrorKind::TakeUntil,
    )))
}

fn preprocessor_line(input: &str) -> PResult<'_, ()> {
    value((), pair(char('#'), take_while(|c| c != '\n'))).parse(input)
}

pub fn ws(input: &str) -> PResult<'_, ()> {
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
    mut inner: impl Parser<&'a str, Output = O, Error = nom::error::Error<&'a str>>,
) -> impl Parser<&'a str, Output = O, Error = nom::error::Error<&'a str>> {
    move |input: &'a str| {
        let (input, _) = ws(input)?;
        inner.parse(input)
    }
}

pub fn with_span<'a, O>(
    source: &'a str,
    mut inner: impl Parser<&'a str, Output = O, Error = nom::error::Error<&'a str>>,
) -> impl Parser<&'a str, Output = (Span, O), Error = nom::error::Error<&'a str>> {
    move |input: &'a str| {
        let start = source.len() - input.len();
        let (input, output) = inner.parse(input)?;
        let end = source.len() - input.len();
        Ok((input, (Span::new(start, end), output)))
    }
}

pub fn identifier(input: &str) -> PResult<'_, &str> {
    alt((
        recognize(pair(
            take_while1(|c: char| c.is_ascii_alphabetic()),
            take_while(|c: char| c.is_ascii_alphanumeric() || c == '_' || c == '$'),
        )),
        recognize(pair(
            take_while1(|c: char| c == '_' || c == '$' || c == '%'),
            take_while1(|c: char| c.is_ascii_alphanumeric() || c == '_' || c == '$'),
        )),
    ))
    .parse(input)
}

pub fn decimal_integer(input: &str) -> PResult<'_, i64> {
    let (input, sign) = opt(char('-')).parse(input)?;
    let (input, digits) = take_while1(|c: char| c.is_ascii_digit()).parse(input)?;
    let (input, _) = opt(char('U')).parse(input)?;
    let val: i64 = digits.parse().map_err(|_| {
        nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Digit))
    })?;
    Ok((input, if sign.is_some() { -val } else { val }))
}

pub fn hex_integer(input: &str) -> PResult<'_, u64> {
    let (input, _) = alt((tag("0x"), tag("0X"))).parse(input)?;
    let (input, digits) = take_while1(|c: char| c.is_ascii_hexdigit()).parse(input)?;
    let (input, _) = opt(char('U')).parse(input)?;
    let val = u64::from_str_radix(digits, 16).map_err(|_| {
        nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::HexDigit,
        ))
    })?;
    Ok((input, val))
}

pub fn octal_integer(input: &str) -> PResult<'_, u64> {
    let (input, _) = alt((tag("0o"), tag("0O"))).parse(input)?;
    let (input, digits) = take_while1(|c: char| ('0'..='7').contains(&c)).parse(input)?;
    let (input, _) = opt(char('U')).parse(input)?;
    let val = u64::from_str_radix(digits, 8).map_err(|_| {
        nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Digit))
    })?;
    Ok((input, val))
}

pub fn binary_integer(input: &str) -> PResult<'_, u64> {
    let (input, _) = alt((tag("0b"), tag("0B"))).parse(input)?;
    let (input, digits) = take_while1(|c: char| c == '0' || c == '1').parse(input)?;
    let (input, _) = opt(char('U')).parse(input)?;
    let val = u64::from_str_radix(digits, 2).map_err(|_| {
        nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Digit))
    })?;
    Ok((input, val))
}

pub fn string_literal(input: &str) -> PResult<'_, String> {
    let (input, _) = char('"').parse(input)?;
    let (input, content) = take_while(|c| c != '"').parse(input)?;
    let (input, _) = char('"').parse(input)?;
    Ok((input, content.to_string()))
}

pub fn float_literal(input: &str) -> PResult<'_, f64> {
    let mut parser = recognize((
        opt(alt((char('+'), char('-')))),
        alt((
            recognize((
                take_while1(|c: char| c.is_ascii_digit()),
                char('.'),
                opt(take_while1(|c: char| c.is_ascii_digit())),
            )),
            recognize((char('.'), take_while1(|c: char| c.is_ascii_digit()))),
            recognize((
                take_while1(|c: char| c.is_ascii_digit()),
                alt((char('e'), char('E'))),
                opt(alt((char('+'), char('-')))),
                take_while1(|c: char| c.is_ascii_digit()),
            )),
        )),
        opt(recognize((
            alt((char('e'), char('E'))),
            opt(alt((char('+'), char('-')))),
            take_while1(|c: char| c.is_ascii_digit()),
        ))),
    ));

    let (input, literal) = parser.parse(input)?;
    let value = literal.parse().map_err(|_| {
        nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Float))
    })?;
    Ok((input, value))
}

pub fn comma_sep(input: &str) -> PResult<'_, ()> {
    value((), delimited(ws, char(','), ws)).parse(input)
}

pub fn semicolon(input: &str) -> PResult<'_, ()> {
    value((), lexeme(char(';'))).parse(input)
}

pub fn scalar_type(input: &str) -> PResult<'_, ScalarType> {
    alt((
        alt((
            value(ScalarType::B128, tag(".b128")),
            value(ScalarType::B64, tag(".b64")),
            value(ScalarType::B32, tag(".b32")),
            value(ScalarType::B16, tag(".b16")),
            value(ScalarType::B8, tag(".b8")),
            value(ScalarType::S64, tag(".s64")),
            value(ScalarType::S32, tag(".s32")),
            value(ScalarType::S16, tag(".s16")),
            value(ScalarType::S8, tag(".s8")),
            value(ScalarType::U64, tag(".u64")),
        )),
        alt((
            value(ScalarType::U32, tag(".u32")),
            value(ScalarType::U16, tag(".u16")),
            value(ScalarType::U8, tag(".u8")),
            value(ScalarType::F16x2, tag(".f16x2")),
            value(ScalarType::F16, tag(".f16")),
            value(ScalarType::F64, tag(".f64")),
            value(ScalarType::F32, tag(".f32")),
            value(ScalarType::Bf16, tag(".bf16")),
            value(ScalarType::Tf32, tag(".tf32")),
            value(ScalarType::Pred, tag(".pred")),
            value(ScalarType::TexRef, tag(".texref")),
            value(ScalarType::SamplerRef, tag(".samplerref")),
            value(ScalarType::SurfRef, tag(".surfref")),
        )),
    ))
    .parse(input)
}

pub fn state_space(input: &str) -> PResult<'_, StateSpace> {
    alt((
        value(StateSpace::SReg, tag(".sreg")),
        value(StateSpace::Shared, tag(".shared")),
        value(StateSpace::Reg, tag(".reg")),
        value(StateSpace::Const, tag(".const")),
        value(StateSpace::Global, tag(".global")),
        value(StateSpace::Local, tag(".local")),
        value(StateSpace::Param, tag(".param")),
        value(StateSpace::Tex, tag(".tex")),
    ))
    .parse(input)
}

pub fn linking_directive(input: &str) -> PResult<'_, LinkingDirective> {
    alt((
        value(LinkingDirective::Extern, tag(".extern")),
        value(LinkingDirective::Visible, tag(".visible")),
        value(LinkingDirective::Weak, tag(".weak")),
        value(LinkingDirective::Common, tag(".common")),
    ))
    .parse(input)
}

pub fn vector_size(input: &str) -> PResult<'_, VectorSize> {
    alt((
        value(VectorSize::V2, tag(".v2")),
        value(VectorSize::V4, tag(".v4")),
    ))
    .parse(input)
}

#[allow(dead_code)]
pub fn float_rounding_mode(input: &str) -> PResult<'_, FloatRoundingMode> {
    alt((
        value(FloatRoundingMode::Rna, tag(".rna")),
        value(FloatRoundingMode::Rn, tag(".rn")),
        value(FloatRoundingMode::Rz, tag(".rz")),
        value(FloatRoundingMode::Rm, tag(".rm")),
        value(FloatRoundingMode::Rp, tag(".rp")),
    ))
    .parse(input)
}

#[allow(dead_code)]
pub fn int_rounding_mode(input: &str) -> PResult<'_, IntRoundingMode> {
    alt((
        value(IntRoundingMode::Rni, tag(".rni")),
        value(IntRoundingMode::Rzi, tag(".rzi")),
        value(IntRoundingMode::Rmi, tag(".rmi")),
        value(IntRoundingMode::Rpi, tag(".rpi")),
    ))
    .parse(input)
}

#[allow(dead_code)]
pub fn compare_op(input: &str) -> PResult<'_, CompareOp> {
    alt((
        alt((
            value(CompareOp::Equ, tag(".equ")),
            value(CompareOp::Neu, tag(".neu")),
            value(CompareOp::Ltu, tag(".ltu")),
            value(CompareOp::Leu, tag(".leu")),
            value(CompareOp::Gtu, tag(".gtu")),
            value(CompareOp::Geu, tag(".geu")),
        )),
        alt((
            value(CompareOp::Eq, tag(".eq")),
            value(CompareOp::Ne, tag(".ne")),
            value(CompareOp::Lt, tag(".lt")),
            value(CompareOp::Le, tag(".le")),
            value(CompareOp::Gt, tag(".gt")),
            value(CompareOp::Ge, tag(".ge")),
            value(CompareOp::Lo, tag(".lo")),
            value(CompareOp::Ls, tag(".ls")),
            value(CompareOp::Hi, tag(".hi")),
            value(CompareOp::Hs, tag(".hs")),
        )),
        alt((
            value(CompareOp::Num, tag(".num")),
            value(CompareOp::Nan, tag(".nan")),
        )),
    ))
    .parse(input)
}

#[allow(dead_code)]
pub fn mul_mode(input: &str) -> PResult<'_, MulMode> {
    alt((
        value(MulMode::Hi, tag(".hi")),
        value(MulMode::Lo, tag(".lo")),
        value(MulMode::Wide, tag(".wide")),
    ))
    .parse(input)
}

pub fn parse_u32(input: &str) -> PResult<'_, u32> {
    let (input, digits) = take_while1(|c: char| c.is_ascii_digit()).parse(input)?;
    let val: u32 = digits.parse().map_err(|_| {
        nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Digit))
    })?;
    Ok((input, val))
}

pub fn alignment(input: &str) -> PResult<'_, u32> {
    preceded(tag(".align"), lexeme(parse_u32)).parse(input)
}

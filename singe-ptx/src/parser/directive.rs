use nom::{
    Parser,
    branch::alt,
    bytes::complete::{tag, take_while1},
    character::complete::char,
    combinator::{map, opt, value},
    multi::{many0, separated_list0, separated_list1},
    sequence::{delimited, pair, preceded, terminated},
};

use crate::{
    ast::{
        Attribute, CallPrototype, DirectiveStatement, FunctionDirective, Initializer,
        InitializerBinaryOp, InitializerCastType, InitializerUnaryOp, LabelPattern, LocDirective,
        LocFunctionName, LocInlineSite, Parameter, SectionDataWidth, SectionDirective, SectionLine,
        SectionValue, TopLevelItem, VariableDecl,
    },
    parser::{
        common::{
            PResult, alignment, binary_integer, comma_sep, decimal_integer, float_literal,
            hex_integer, identifier, lexeme, linking_directive, octal_integer, parse_u32,
            scalar_type, semicolon, state_space, string_literal, vector_size, with_span, ws,
        },
        function::function,
    },
};

fn dotted_symbol(input: &str) -> PResult<'_, &str> {
    alt((
        identifier,
        nom::combinator::recognize(pair(
            char('.'),
            take_while1(|c: char| c.is_ascii_alphanumeric() || c == '_' || c == '.'),
        )),
    ))
    .parse(input)
}

fn parse_u64(input: &str) -> PResult<'_, u64> {
    let (input, digits) = take_while1(|c: char| c.is_ascii_digit()).parse(input)?;
    let val = digits.parse().map_err(|_| {
        nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Digit))
    })?;
    Ok((input, val))
}

fn hex_or_decimal_u64(input: &str) -> PResult<'_, u64> {
    alt((
        hex_integer,
        octal_integer,
        binary_integer,
        map(decimal_integer, |value| value as u64),
    ))
    .parse(input)
}

fn attribute<'a>(source: &'a str, input: &'a str) -> PResult<'a, Attribute> {
    alt((
        map(with_span(source, tag(".managed")), |(span, _)| {
            Attribute::Managed { span }
        }),
        map(
            with_span(
                source,
                preceded(
                    tag(".unified"),
                    delimited(
                        lexeme(char('(')),
                        pair(
                            lexeme(hex_or_decimal_u64),
                            preceded(comma_sep, lexeme(hex_or_decimal_u64)),
                        ),
                        lexeme(char(')')),
                    ),
                ),
            ),
            |(span, (uuid1, uuid2))| Attribute::Unified { span, uuid1, uuid2 },
        ),
    ))
    .parse(input)
}

pub fn attributes<'a>(source: &'a str, input: &'a str) -> PResult<'a, Vec<Attribute>> {
    preceded(
        tag(".attribute"),
        delimited(
            lexeme(char('(')),
            separated_list1(comma_sep, lexeme(|input| attribute(source, input))),
            lexeme(char(')')),
        ),
    )
    .parse(input)
}

fn float_initializer<'a>(source: &'a str, input: &'a str) -> PResult<'a, Initializer> {
    let (input, (span, value)) = with_span(source, float_literal).parse(input)?;
    Ok((input, Initializer::Float { span, value }))
}

fn integer_initializer<'a>(source: &'a str, input: &'a str) -> PResult<'a, Initializer> {
    alt((
        map(with_span(source, hex_integer), |(span, value)| {
            Initializer::UnsignedInteger { span, value }
        }),
        map(with_span(source, octal_integer), |(span, value)| {
            Initializer::UnsignedInteger { span, value }
        }),
        map(with_span(source, binary_integer), |(span, value)| {
            Initializer::UnsignedInteger { span, value }
        }),
        map(with_span(source, decimal_integer), |(span, value)| {
            Initializer::Integer { span, value }
        }),
    ))
    .parse(input)
}

fn symbol_initializer<'a>(source: &'a str, input: &'a str) -> PResult<'a, Initializer> {
    let (input, (span, symbol)) = with_span(source, identifier).parse(input)?;
    Ok((
        input,
        Initializer::Symbol {
            span,
            name: symbol.to_string(),
        },
    ))
}

fn generic_initializer<'a>(source: &'a str, input: &'a str) -> PResult<'a, Initializer> {
    let (input, (span, (_, expr))) = with_span(
        source,
        pair(
            tag("generic"),
            delimited(
                lexeme(char('(')),
                lexeme(|input| expression(source, input)),
                lexeme(char(')')),
            ),
        ),
    )
    .parse(input)?;
    Ok((
        input,
        Initializer::Generic {
            span,
            expr: Box::new(expr),
        },
    ))
}

fn masked_initializer<'a>(source: &'a str, input: &'a str) -> PResult<'a, Initializer> {
    let (input, (span, (mask, value))) = with_span(
        source,
        pair(
            alt((hex_integer, octal_integer, binary_integer)),
            delimited(
                lexeme(char('(')),
                lexeme(|input| expression(source, input)),
                lexeme(char(')')),
            ),
        ),
    )
    .parse(input)?;
    Ok((
        input,
        Initializer::Masked {
            span,
            mask,
            value: Box::new(value),
        },
    ))
}

fn initializer_list<'a>(source: &'a str, input: &'a str) -> PResult<'a, Initializer> {
    let (input, (span, values)) = with_span(
        source,
        delimited(
            lexeme(char('{')),
            separated_list0(comma_sep, lexeme(|input| initializer(source, input))),
            lexeme(char('}')),
        ),
    )
    .parse(input)?;
    Ok((input, Initializer::List { span, values }))
}

fn cast_initializer<'a>(source: &'a str, input: &'a str) -> PResult<'a, Initializer> {
    let (input, (span, (ty, expr))) = with_span(
        source,
        pair(
            delimited(
                lexeme(char('(')),
                alt((
                    map(tag(".s64"), |_| InitializerCastType::S64),
                    map(tag(".u64"), |_| InitializerCastType::U64),
                )),
                lexeme(char(')')),
            ),
            lexeme(|input| unary_expression(source, input)),
        ),
    )
    .parse(input)?;
    Ok((
        input,
        Initializer::Cast {
            span,
            ty,
            expr: Box::new(expr),
        },
    ))
}

fn primary_expression<'a>(source: &'a str, input: &'a str) -> PResult<'a, Initializer> {
    alt((
        delimited(
            lexeme(char('(')),
            lexeme(|input| expression(source, input)),
            lexeme(char(')')),
        ),
        |input| generic_initializer(source, input),
        |input| masked_initializer(source, input),
        |input| float_initializer(source, input),
        |input| integer_initializer(source, input),
        |input| symbol_initializer(source, input),
    ))
    .parse(input)
}

fn unary_expression<'a>(source: &'a str, input: &'a str) -> PResult<'a, Initializer> {
    alt((
        |input| cast_initializer(source, input),
        map(
            with_span(
                source,
                preceded(lexeme(char('+')), |input| unary_expression(source, input)),
            ),
            |(span, expr)| Initializer::Unary {
                span,
                op: InitializerUnaryOp::Plus,
                expr: Box::new(expr),
            },
        ),
        map(
            with_span(
                source,
                preceded(lexeme(char('-')), |input| unary_expression(source, input)),
            ),
            |(span, expr)| Initializer::Unary {
                span,
                op: InitializerUnaryOp::Minus,
                expr: Box::new(expr),
            },
        ),
        map(
            with_span(
                source,
                preceded(lexeme(char('!')), |input| unary_expression(source, input)),
            ),
            |(span, expr)| Initializer::Unary {
                span,
                op: InitializerUnaryOp::LogicalNot,
                expr: Box::new(expr),
            },
        ),
        map(
            with_span(
                source,
                preceded(lexeme(char('~')), |input| unary_expression(source, input)),
            ),
            |(span, expr)| Initializer::Unary {
                span,
                op: InitializerUnaryOp::BitwiseNot,
                expr: Box::new(expr),
            },
        ),
        |input| primary_expression(source, input),
    ))
    .parse(input)
}

fn binary_expression<'a, F>(
    source: &'a str,
    mut input: &'a str,
    next: F,
    operators: &[(&str, InitializerBinaryOp)],
) -> PResult<'a, Initializer>
where
    F: Copy + Fn(&'a str, &'a str) -> PResult<'a, Initializer>,
{
    let (next_input, _) = ws(input)?;
    let (next_input, mut left) = next(source, next_input)?;
    input = next_input;

    loop {
        let mut matched = None;
        for (token, op) in operators {
            if let Ok((after_op, _)) = lexeme(tag(*token)).parse(input) {
                matched = Some((after_op, *op));
                break;
            }
        }

        let Some((after_op, op)) = matched else {
            break;
        };

        let (after_rhs, _) = ws(after_op)?;
        let (after_rhs, right) = next(source, after_rhs)?;
        let span = crate::ast::Span::new(left_span(&left), right_span(&right));
        left = Initializer::Binary {
            span,
            op,
            left: Box::new(left),
            right: Box::new(right),
        };
        input = after_rhs;
    }

    Ok((input, left))
}

fn multiplicative_expression<'a>(source: &'a str, input: &'a str) -> PResult<'a, Initializer> {
    binary_expression(
        source,
        input,
        unary_expression,
        &[
            ("*", InitializerBinaryOp::Mul),
            ("/", InitializerBinaryOp::Div),
            ("%", InitializerBinaryOp::Rem),
        ],
    )
}

fn additive_expression<'a>(source: &'a str, input: &'a str) -> PResult<'a, Initializer> {
    binary_expression(
        source,
        input,
        multiplicative_expression,
        &[
            ("+", InitializerBinaryOp::Add),
            ("-", InitializerBinaryOp::Sub),
        ],
    )
}

fn shift_expression<'a>(source: &'a str, input: &'a str) -> PResult<'a, Initializer> {
    binary_expression(
        source,
        input,
        additive_expression,
        &[
            ("<<", InitializerBinaryOp::Shl),
            (">>", InitializerBinaryOp::Shr),
        ],
    )
}

fn relational_expression<'a>(source: &'a str, input: &'a str) -> PResult<'a, Initializer> {
    binary_expression(
        source,
        input,
        shift_expression,
        &[
            ("<=", InitializerBinaryOp::Le),
            (">=", InitializerBinaryOp::Ge),
            ("<", InitializerBinaryOp::Lt),
            (">", InitializerBinaryOp::Gt),
        ],
    )
}

fn equality_expression<'a>(source: &'a str, input: &'a str) -> PResult<'a, Initializer> {
    binary_expression(
        source,
        input,
        relational_expression,
        &[
            ("==", InitializerBinaryOp::Eq),
            ("!=", InitializerBinaryOp::Ne),
        ],
    )
}

fn bitand_expression<'a>(source: &'a str, input: &'a str) -> PResult<'a, Initializer> {
    binary_expression(
        source,
        input,
        equality_expression,
        &[("&", InitializerBinaryOp::BitAnd)],
    )
}

fn bitxor_expression<'a>(source: &'a str, input: &'a str) -> PResult<'a, Initializer> {
    binary_expression(
        source,
        input,
        bitand_expression,
        &[("^", InitializerBinaryOp::BitXor)],
    )
}

fn bitor_expression<'a>(source: &'a str, input: &'a str) -> PResult<'a, Initializer> {
    binary_expression(
        source,
        input,
        bitxor_expression,
        &[("|", InitializerBinaryOp::BitOr)],
    )
}

fn logical_and_expression<'a>(source: &'a str, input: &'a str) -> PResult<'a, Initializer> {
    binary_expression(
        source,
        input,
        bitor_expression,
        &[("&&", InitializerBinaryOp::LogicalAnd)],
    )
}

fn logical_or_expression<'a>(source: &'a str, input: &'a str) -> PResult<'a, Initializer> {
    binary_expression(
        source,
        input,
        logical_and_expression,
        &[("||", InitializerBinaryOp::LogicalOr)],
    )
}

fn conditional_expression<'a>(source: &'a str, input: &'a str) -> PResult<'a, Initializer> {
    let start = source.len() - input.len();
    let (input, condition) = logical_or_expression(source, input)?;
    let (input, question) = opt(lexeme(char('?'))).parse(input)?;
    if question.is_none() {
        return Ok((input, condition));
    }

    let (input, then_expr) = expression(source, input)?;
    let (input, _) = lexeme(char(':')).parse(input)?;
    let (input, else_expr) = conditional_expression(source, input)?;
    Ok((
        input,
        Initializer::Conditional {
            span: crate::ast::Span::new(start, source.len() - input.len()),
            condition: Box::new(condition),
            then_expr: Box::new(then_expr),
            else_expr: Box::new(else_expr),
        },
    ))
}

fn expression<'a>(source: &'a str, input: &'a str) -> PResult<'a, Initializer> {
    conditional_expression(source, input)
}

fn initializer<'a>(source: &'a str, input: &'a str) -> PResult<'a, Initializer> {
    alt((
        |input| initializer_list(source, input),
        |input| expression(source, input),
    ))
    .parse(input)
}

fn array_bounds(input: &str) -> PResult<'_, Vec<Option<u32>>> {
    many0(delimited(
        lexeme(char('[')),
        opt(lexeme(parse_u32)),
        lexeme(char(']')),
    ))
    .parse(input)
}

fn left_span(initializer: &Initializer) -> usize {
    match initializer {
        Initializer::Integer { span, .. }
        | Initializer::UnsignedInteger { span, .. }
        | Initializer::Float { span, .. }
        | Initializer::Symbol { span, .. }
        | Initializer::Cast { span, .. }
        | Initializer::Generic { span, .. }
        | Initializer::Masked { span, .. }
        | Initializer::List { span, .. }
        | Initializer::Unary { span, .. }
        | Initializer::Binary { span, .. }
        | Initializer::Conditional { span, .. } => span.start,
    }
}

fn right_span(initializer: &Initializer) -> usize {
    match initializer {
        Initializer::Integer { span, .. }
        | Initializer::UnsignedInteger { span, .. }
        | Initializer::Float { span, .. }
        | Initializer::Symbol { span, .. }
        | Initializer::Cast { span, .. }
        | Initializer::Generic { span, .. }
        | Initializer::Masked { span, .. }
        | Initializer::List { span, .. }
        | Initializer::Unary { span, .. }
        | Initializer::Binary { span, .. }
        | Initializer::Conditional { span, .. } => span.end,
    }
}

pub fn parameter<'a>(source: &'a str, input: &'a str) -> PResult<'a, Parameter> {
    let (input, (span, (space, align, ty, ptr, name, bounds))) = with_span(
        source,
        (
            opt(lexeme(state_space)),
            opt(lexeme(alignment)),
            lexeme(scalar_type),
            opt(lexeme(tag(".ptr"))),
            lexeme(alt((identifier, value("_", char('_'))))),
            array_bounds,
        ),
    )
    .parse(input)?;

    Ok((
        input,
        Parameter {
            span,
            name: name.to_string(),
            ty,
            state_space: space,
            alignment: align,
            array_bounds: bounds,
            ptr: ptr.is_some(),
        },
    ))
}

pub fn param_list<'a>(source: &'a str, input: &'a str) -> PResult<'a, Vec<Parameter>> {
    delimited(
        lexeme(char('(')),
        separated_list0(comma_sep, lexeme(|input| parameter(source, input))),
        lexeme(char(')')),
    )
    .parse(input)
}

pub fn variable_decl<'a>(source: &'a str, input: &'a str) -> PResult<'a, VariableDecl> {
    let (input, (span, (linkage, space, attributes, align, vec, ty, name, count, bounds, init, _))) =
        with_span(
            source,
            (
                opt(lexeme(linking_directive)),
                opt(lexeme(state_space)),
                opt(lexeme(|input| attributes(source, input))),
                opt(lexeme(alignment)),
                opt(lexeme(vector_size)),
                lexeme(scalar_type),
                lexeme(identifier),
                opt(delimited(
                    lexeme(char('<')),
                    lexeme(parse_u32),
                    lexeme(char('>')),
                )),
                array_bounds,
                opt(preceded(
                    lexeme(char('=')),
                    lexeme(|input| initializer(source, input)),
                )),
                semicolon,
            ),
        )
        .parse(input)?;

    Ok((
        input,
        VariableDecl {
            span,
            name: name.to_string(),
            ty,
            state_space: space,
            linkage,
            attributes: attributes.unwrap_or_default(),
            alignment: align,
            vector: vec,
            count,
            array_bounds: bounds,
            initializer: init,
        },
    ))
}

pub fn function_directive<'a>(source: &'a str, input: &'a str) -> PResult<'a, FunctionDirective> {
    alt((
        map(with_span(source, tag(".noreturn")), |(span, _)| {
            FunctionDirective::NoReturn { span }
        }),
        map(
            with_span(source, preceded(tag(".maxnreg"), lexeme(parse_u32))),
            |(span, value)| FunctionDirective::MaxNReg { span, value },
        ),
        map(
            with_span(
                source,
                preceded(
                    tag(".maxntid"),
                    separated_list1(comma_sep, lexeme(parse_u32)),
                ),
            ),
            |(span, values)| FunctionDirective::MaxNTid { span, values },
        ),
        map(
            with_span(
                source,
                preceded(
                    tag(".reqntid"),
                    separated_list1(comma_sep, lexeme(parse_u32)),
                ),
            ),
            |(span, values)| FunctionDirective::ReqNTid { span, values },
        ),
        map(
            with_span(source, preceded(tag(".minnctapersm"), lexeme(parse_u32))),
            |(span, value)| FunctionDirective::MinNCtaPerSm { span, value },
        ),
        map(
            with_span(source, preceded(tag(".abi_preserve"), lexeme(parse_u32))),
            |(span, value)| FunctionDirective::AbiPreserve { span, value },
        ),
        map(
            with_span(
                source,
                preceded(tag(".abi_preserve_control"), lexeme(parse_u32)),
            ),
            |(span, value)| FunctionDirective::AbiPreserveControl { span, value },
        ),
        map(
            with_span(
                source,
                preceded(
                    tag(".pragma"),
                    terminated(lexeme(string_literal), opt(semicolon)),
                ),
            ),
            |(span, value)| FunctionDirective::Pragma { span, value },
        ),
    ))
    .parse(input)
}

fn pragma<'a>(source: &'a str, input: &'a str) -> PResult<'a, TopLevelItem> {
    let (input, (span, (_, value, _))) = with_span(
        source,
        (lexeme(tag(".pragma")), lexeme(string_literal), semicolon),
    )
    .parse(input)?;
    Ok((input, TopLevelItem::Pragma { span, value }))
}

fn file_directive<'a>(source: &'a str, input: &'a str) -> PResult<'a, TopLevelItem> {
    let (input, (span, (_, _, index, name, metadata, _))) = with_span(
        source,
        (
            lexeme(tag(".file")),
            ws,
            parse_u32,
            lexeme(string_literal),
            opt(preceded(
                comma_sep,
                pair(lexeme(parse_u64), preceded(comma_sep, lexeme(parse_u64))),
            )),
            opt(semicolon),
        ),
    )
    .parse(input)?;

    Ok((
        input,
        TopLevelItem::File {
            span,
            index,
            name,
            timestamp: Some(metadata.map(|(timestamp, _)| timestamp).unwrap_or(0)),
            file_size: Some(metadata.map(|(_, file_size)| file_size).unwrap_or(0)),
        },
    ))
}

fn alias_directive<'a>(source: &'a str, input: &'a str) -> PResult<'a, TopLevelItem> {
    let (input, (span, (_, alias, _, target, _))) = with_span(
        source,
        (
            lexeme(tag(".alias")),
            lexeme(identifier),
            comma_sep,
            lexeme(identifier),
            semicolon,
        ),
    )
    .parse(input)?;

    Ok((
        input,
        TopLevelItem::Alias {
            span,
            alias: alias.to_string(),
            target: target.to_string(),
        },
    ))
}

fn section_width(input: &str) -> PResult<'_, SectionDataWidth> {
    alt((
        value(SectionDataWidth::B8, tag(".b8")),
        value(SectionDataWidth::B16, tag(".b16")),
        value(SectionDataWidth::B32, tag(".b32")),
        value(SectionDataWidth::B64, tag(".b64")),
    ))
    .parse(input)
}

fn section_value<'a>(source: &'a str, input: &'a str) -> PResult<'a, SectionValue> {
    let label_value = map(
        with_span(source, identifier),
        |(span, label): (crate::ast::Span, &str)| SectionValue::Label {
            span,
            name: label.to_string(),
        },
    );
    let label_offset = map(
        with_span(
            source,
            pair(
                identifier,
                preceded(lexeme(char('+')), lexeme(decimal_integer)),
            ),
        ),
        |(span, (label, offset))| SectionValue::LabelOffset {
            span,
            label: label.to_string(),
            offset,
        },
    );
    let label_difference = map(
        with_span(
            source,
            pair(identifier, preceded(lexeme(char('-')), lexeme(identifier))),
        ),
        |(span, (left, right))| SectionValue::LabelDifference {
            span,
            left: left.to_string(),
            right: right.to_string(),
        },
    );

    alt((
        map(with_span(source, hex_integer), |(span, value)| {
            SectionValue::UnsignedInteger { span, value }
        }),
        map(with_span(source, octal_integer), |(span, value)| {
            SectionValue::UnsignedInteger { span, value }
        }),
        map(with_span(source, binary_integer), |(span, value)| {
            SectionValue::UnsignedInteger { span, value }
        }),
        map(with_span(source, decimal_integer), |(span, value)| {
            SectionValue::Integer { span, value }
        }),
        label_difference,
        label_offset,
        label_value,
        map(
            with_span(source, dotted_symbol),
            |(span, label): (crate::ast::Span, &str)| SectionValue::Label {
                span,
                name: label.to_string(),
            },
        ),
    ))
    .parse(input)
}

fn section_line<'a>(source: &'a str, input: &'a str) -> PResult<'a, SectionLine> {
    alt((
        map(
            with_span(source, terminated(lexeme(identifier), lexeme(char(':')))),
            |(span, label)| SectionLine::Label {
                span,
                name: label.to_string(),
            },
        ),
        map(
            with_span(
                source,
                pair(
                    lexeme(section_width),
                    terminated(
                        separated_list1(comma_sep, lexeme(|input| section_value(source, input))),
                        opt(semicolon),
                    ),
                ),
            ),
            |(span, (width, values))| SectionLine::Data {
                span,
                width,
                values,
            },
        ),
    ))
    .parse(input)
}

fn section_directive<'a>(source: &'a str, input: &'a str) -> PResult<'a, TopLevelItem> {
    let (input, (span, (_, name, lines))) = with_span(
        source,
        (
            lexeme(tag(".section")),
            lexeme(dotted_symbol),
            delimited(
                lexeme(char('{')),
                many0(lexeme(|input| section_line(source, input))),
                lexeme(char('}')),
            ),
        ),
    )
    .parse(input)?;

    Ok((
        input,
        TopLevelItem::Section(SectionDirective {
            span,
            name: name.to_string(),
            lines,
        }),
    ))
}

fn label_pattern<'a>(source: &'a str, input: &'a str) -> PResult<'a, LabelPattern> {
    let (input, (span, (name, count))) = with_span(
        source,
        pair(
            identifier,
            opt(delimited(
                lexeme(char('<')),
                lexeme(parse_u32),
                lexeme(char('>')),
            )),
        ),
    )
    .parse(input)?;

    let pattern = match count {
        Some(count) => LabelPattern::Range {
            span,
            prefix: name.to_string(),
            count,
        },
        None => LabelPattern::Name {
            span,
            name: name.to_string(),
        },
    };
    Ok((input, pattern))
}

fn loc_function_name<'a>(source: &'a str, input: &'a str) -> PResult<'a, LocFunctionName> {
    map(
        with_span(
            source,
            pair(
                dotted_symbol,
                opt(preceded(lexeme(char('+')), lexeme(decimal_integer))),
            ),
        ),
        |(span, (label, offset))| match offset {
            Some(offset) => LocFunctionName::LabelOffset {
                span,
                label: label.to_string(),
                offset,
            },
            None => LocFunctionName::Label {
                span,
                label: label.to_string(),
            },
        },
    )
    .parse(input)
}

fn loc_directive<'a>(source: &'a str, input: &'a str) -> PResult<'a, LocDirective> {
    let (input, (span, (_, file, line, column, function_name, inlined_at))) = with_span(
        source,
        (
            tag(".loc"),
            lexeme(parse_u32),
            lexeme(parse_u32),
            lexeme(parse_u32),
            opt(preceded(
                lexeme(char(',')),
                preceded(
                    lexeme(tag("function_name")),
                    lexeme(|input| loc_function_name(source, input)),
                ),
            )),
            opt(preceded(
                lexeme(char(',')),
                preceded(
                    lexeme(tag("inlined_at")),
                    map(
                        with_span(
                            source,
                            (lexeme(parse_u32), lexeme(parse_u32), lexeme(parse_u32)),
                        ),
                        |(span, (file, line, column))| LocInlineSite {
                            span,
                            file,
                            line,
                            column,
                        },
                    ),
                ),
            )),
        ),
    )
    .parse(input)?;

    Ok((
        input,
        LocDirective {
            span,
            file,
            line,
            column,
            function_name,
            inlined_at,
        },
    ))
}

pub fn directive_statement<'a>(source: &'a str, input: &'a str) -> PResult<'a, DirectiveStatement> {
    alt((
        map(
            with_span(
                source,
                preceded(
                    tag(".pragma"),
                    terminated(lexeme(string_literal), semicolon),
                ),
            ),
            |(span, value)| DirectiveStatement::Pragma { span, value },
        ),
        map(
            |input| loc_directive(source, input),
            DirectiveStatement::Loc,
        ),
        map(
            with_span(
                source,
                preceded(
                    tag(".branchtargets"),
                    terminated(
                        separated_list1(comma_sep, lexeme(|input| label_pattern(source, input))),
                        semicolon,
                    ),
                ),
            ),
            |(span, labels)| DirectiveStatement::BranchTargets { span, labels },
        ),
        map(
            with_span(
                source,
                preceded(
                    tag(".calltargets"),
                    terminated(
                        separated_list1(comma_sep, lexeme(map(identifier, str::to_string))),
                        semicolon,
                    ),
                ),
            ),
            |(span, targets)| DirectiveStatement::CallTargets { span, targets },
        ),
        map(
            with_span(
                source,
                preceded(
                    tag(".callprototype"),
                    terminated(
                        (
                            opt(|input| param_list(source, input)),
                            lexeme(alt((identifier, value("_", char('_'))))),
                            opt(|input| param_list(source, input)),
                            many0(lexeme(|input| function_directive(source, input))),
                        ),
                        semicolon,
                    ),
                ),
            ),
            |(span, (return_params, placeholder, params, directives))| {
                DirectiveStatement::CallPrototype {
                    span,
                    prototype: CallPrototype {
                        span,
                        return_params: return_params.unwrap_or_default(),
                        placeholder: placeholder.to_string(),
                        params: params.unwrap_or_default(),
                        directives,
                    },
                }
            },
        ),
    ))
    .parse(input)
}

pub fn top_level_item<'a>(source: &'a str, input: &'a str) -> PResult<'a, TopLevelItem> {
    alt((
        map(|input| function(source, input), TopLevelItem::Function),
        |input| pragma(source, input),
        |input| file_directive(source, input),
        |input| alias_directive(source, input),
        |input| section_directive(source, input),
        map(|input| variable_decl(source, input), TopLevelItem::Variable),
    ))
    .parse(input)
}

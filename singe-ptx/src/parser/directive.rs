use nom::{
    Parser,
    branch::alt,
    bytes::complete::{tag, tag_no_case, take_while1},
    character::complete::char,
    combinator::{map, not, opt, peek, recognize, value},
    multi::{many0, separated_list0, separated_list1},
    sequence::{delimited, pair, preceded, terminated},
};

use crate::{
    ast::{
        Attribute, CallPrototype, DirectiveStatement, FunctionDirective, Initializer,
        InitializerBinaryOp, InitializerCastType, InitializerUnaryOp, LabelPattern, LocDirective,
        LocFunctionName, LocInlineSite, Parameter, PointerAnnotation, SectionDataWidth,
        SectionDirective, SectionLine, SectionValue, Span, TopLevelItem, VariableDecl,
    },
    parser::{
        common::{
            Input, PResult, alignment, comma_sep, decimal, displacement, float_literal, identifier,
            integer_literal, keyword, lexeme, linking_directive, scalar_type, semicolon,
            state_space, string_literal, text, unsigned_literal, vector_size, with_span, ws,
        },
        depth,
        function::function,
    },
};

fn dotted_symbol(input: Input<'_>) -> PResult<'_, &str> {
    alt((
        identifier,
        map(
            recognize(pair(
                char('.'),
                take_while1(|c: char| c.is_ascii_alphanumeric() || c == '_' || c == '.'),
            )),
            text,
        ),
    ))
    .parse(input)
}

fn attribute(input: Input<'_>) -> PResult<'_, Attribute> {
    alt((
        map(with_span(keyword(".managed")), |(span, _)| {
            Attribute::Managed { span }
        }),
        map(
            with_span(preceded(
                keyword(".unified"),
                delimited(
                    lexeme(char('(')),
                    pair(
                        lexeme(unsigned_literal),
                        preceded(comma_sep, lexeme(unsigned_literal)),
                    ),
                    lexeme(char(')')),
                ),
            )),
            |(span, (uuid1, uuid2))| Attribute::Unified { span, uuid1, uuid2 },
        ),
    ))
    .parse(input)
}

pub fn attributes(input: Input<'_>) -> PResult<'_, Vec<Attribute>> {
    preceded(
        keyword(".attribute"),
        delimited(
            lexeme(char('(')),
            separated_list1(comma_sep, lexeme(attribute)),
            lexeme(char(')')),
        ),
    )
    .parse(input)
}

fn float_initializer(input: Input<'_>) -> PResult<'_, Initializer> {
    let (input, (span, value)) = with_span(float_literal).parse(input)?;
    Ok((input, Initializer::Float { span, value }))
}

fn integer_initializer(input: Input<'_>) -> PResult<'_, Initializer> {
    let (input, (span, (value, unsigned))) = with_span(integer_literal).parse(input)?;
    let initializer = if unsigned {
        Initializer::UnsignedInteger { span, value }
    } else {
        Initializer::Integer {
            span,
            value: (value).cast_signed(),
        }
    };
    Ok((input, initializer))
}

fn symbol_initializer(input: Input<'_>) -> PResult<'_, Initializer> {
    let (input, (span, symbol)) = with_span(identifier).parse(input)?;
    Ok((
        input,
        Initializer::Symbol {
            span,
            name: symbol.to_string(),
        },
    ))
}

fn initializer_depth(value: &Initializer) -> usize {
    match value {
        Initializer::Cast { expr, .. }
        | Initializer::Generic { expr, .. }
        | Initializer::Unary { expr, .. } => 1 + initializer_depth(expr),
        Initializer::Masked { value, .. } => 1 + initializer_depth(value),
        Initializer::List { values, .. } => values
            .iter()
            .map(initializer_depth)
            .max()
            .map_or(0, |depth| depth + 1),
        Initializer::Binary { left, right, .. } => {
            1 + initializer_depth(left).max(initializer_depth(right))
        }
        Initializer::Conditional {
            condition,
            then_expr,
            else_expr,
            ..
        } => {
            1 + initializer_depth(condition)
                .max(initializer_depth(then_expr))
                .max(initializer_depth(else_expr))
        }
        Initializer::Integer { .. }
        | Initializer::UnsignedInteger { .. }
        | Initializer::Float { .. }
        | Initializer::Symbol { .. } => 0,
    }
}

fn generic_initializer(input: Input<'_>, depth: usize) -> PResult<'_, Initializer> {
    let (input, (span, (_, expr))) = with_span(pair(
        keyword("generic"),
        delimited(
            lexeme(char('(')),
            lexeme(|input| expression(input, depth + 1)),
            lexeme(char(')')),
        ),
    ))
    .parse(input)?;
    Ok((
        input,
        Initializer::Generic {
            span,
            expr: Box::new(expr),
        },
    ))
}

fn masked_initializer(input: Input<'_>, depth: usize) -> PResult<'_, Initializer> {
    let (input, (span, (mask, value))) = with_span(pair(
        preceded(peek(tag_no_case("0x")), unsigned_literal),
        delimited(
            lexeme(char('(')),
            lexeme(|input| expression(input, depth + 1)),
            lexeme(char(')')),
        ),
    ))
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

fn initializer_list(input: Input<'_>, depth: usize) -> PResult<'_, Initializer> {
    let (input, (span, values)) = with_span(delimited(
        lexeme(char('{')),
        separated_list0(
            comma_sep,
            lexeme(preceded(not(char('}')), |input| {
                initializer(input, depth + 1)
            })),
        ),
        lexeme(char('}')),
    ))
    .parse(input)?;
    Ok((input, Initializer::List { span, values }))
}

fn cast_initializer(input: Input<'_>, depth: usize) -> PResult<'_, Initializer> {
    let (input, (span, (ty, expr))) = with_span(pair(
        delimited(
            lexeme(char('(')),
            alt((
                map(keyword(".s64"), |_| InitializerCastType::S64),
                map(keyword(".u64"), |_| InitializerCastType::U64),
            )),
            lexeme(char(')')),
        ),
        lexeme(|input| unary_expression(input, depth + 1)),
    ))
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

fn primary_expression(input: Input<'_>, depth: usize) -> PResult<'_, Initializer> {
    alt((
        delimited(
            lexeme(char('(')),
            lexeme(|input| expression(input, depth + 1)),
            lexeme(char(')')),
        ),
        |input| generic_initializer(input, depth),
        |input| masked_initializer(input, depth),
        float_initializer,
        integer_initializer,
        symbol_initializer,
    ))
    .parse(input)
}

fn unary_expression(input: Input<'_>, depth: usize) -> PResult<'_, Initializer> {
    depth::check(depth, input)?;
    alt((
        |input| cast_initializer(input, depth),
        map(
            with_span(preceded(lexeme(char('+')), |input| {
                unary_expression(input, depth + 1)
            })),
            |(span, expr)| Initializer::Unary {
                span,
                op: InitializerUnaryOp::Plus,
                expr: Box::new(expr),
            },
        ),
        map(
            with_span(preceded(lexeme(char('-')), |input| {
                unary_expression(input, depth + 1)
            })),
            |(span, expr)| Initializer::Unary {
                span,
                op: InitializerUnaryOp::Minus,
                expr: Box::new(expr),
            },
        ),
        map(
            with_span(preceded(lexeme(char('!')), |input| {
                unary_expression(input, depth + 1)
            })),
            |(span, expr)| Initializer::Unary {
                span,
                op: InitializerUnaryOp::LogicalNot,
                expr: Box::new(expr),
            },
        ),
        map(
            with_span(preceded(lexeme(char('~')), |input| {
                unary_expression(input, depth + 1)
            })),
            |(span, expr)| Initializer::Unary {
                span,
                op: InitializerUnaryOp::BitwiseNot,
                expr: Box::new(expr),
            },
        ),
        |input| primary_expression(input, depth),
    ))
    .parse(input)
}

fn binary_expression<'a, F>(
    mut input: Input<'a>,
    depth: usize,
    next: F,
    operators: &[(&str, InitializerBinaryOp)],
) -> PResult<'a, Initializer>
where
    F: Copy + Fn(Input<'a>, usize) -> PResult<'a, Initializer>,
{
    let (next_input, ()) = ws(input)?;
    let (next_input, mut left) = next(next_input, depth)?;
    input = next_input;

    loop {
        let mut matched = None;
        for (token, op) in operators {
            if let Ok((after_op, _)) = lexeme(tag(*token)).parse(input) {
                // Leave && and || for the lower-precedence logical expressions.
                if matches!(*token, "&" | "|") && after_op.fragment().starts_with(*token) {
                    continue;
                }
                matched = Some((after_op, *op));
                break;
            }
        }

        let Some((after_op, op)) = matched else {
            break;
        };

        let (after_rhs, ()) = ws(after_op)?;
        let (after_rhs, right) = next(after_rhs, depth)?;
        let span = Span::new(left.span().start, right.span().end);
        // Reject before wrapping so every partial tree stays safe to clone and drop.
        depth::check(
            depth + 1 + initializer_depth(&left).max(initializer_depth(&right)),
            after_op,
        )?;
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

fn multiplicative_expression(input: Input<'_>, depth: usize) -> PResult<'_, Initializer> {
    binary_expression(
        input,
        depth,
        unary_expression,
        &[
            ("*", InitializerBinaryOp::Mul),
            ("/", InitializerBinaryOp::Div),
            ("%", InitializerBinaryOp::Rem),
        ],
    )
}

fn additive_expression(input: Input<'_>, depth: usize) -> PResult<'_, Initializer> {
    binary_expression(
        input,
        depth,
        multiplicative_expression,
        &[
            ("+", InitializerBinaryOp::Add),
            ("-", InitializerBinaryOp::Sub),
        ],
    )
}

fn shift_expression(input: Input<'_>, depth: usize) -> PResult<'_, Initializer> {
    binary_expression(
        input,
        depth,
        additive_expression,
        &[
            ("<<", InitializerBinaryOp::Shl),
            (">>", InitializerBinaryOp::Shr),
        ],
    )
}

fn relational_expression(input: Input<'_>, depth: usize) -> PResult<'_, Initializer> {
    binary_expression(
        input,
        depth,
        shift_expression,
        &[
            ("<=", InitializerBinaryOp::Le),
            (">=", InitializerBinaryOp::Ge),
            ("<", InitializerBinaryOp::Lt),
            (">", InitializerBinaryOp::Gt),
        ],
    )
}

fn equality_expression(input: Input<'_>, depth: usize) -> PResult<'_, Initializer> {
    binary_expression(
        input,
        depth,
        relational_expression,
        &[
            ("==", InitializerBinaryOp::Eq),
            ("!=", InitializerBinaryOp::Ne),
        ],
    )
}

fn bitand_expression(input: Input<'_>, depth: usize) -> PResult<'_, Initializer> {
    binary_expression(
        input,
        depth,
        equality_expression,
        &[("&", InitializerBinaryOp::BitAnd)],
    )
}

fn bitxor_expression(input: Input<'_>, depth: usize) -> PResult<'_, Initializer> {
    binary_expression(
        input,
        depth,
        bitand_expression,
        &[("^", InitializerBinaryOp::BitXor)],
    )
}

fn bitor_expression(input: Input<'_>, depth: usize) -> PResult<'_, Initializer> {
    binary_expression(
        input,
        depth,
        bitxor_expression,
        &[("|", InitializerBinaryOp::BitOr)],
    )
}

fn logical_and_expression(input: Input<'_>, depth: usize) -> PResult<'_, Initializer> {
    binary_expression(
        input,
        depth,
        bitor_expression,
        &[("&&", InitializerBinaryOp::LogicalAnd)],
    )
}

fn logical_or_expression(input: Input<'_>, depth: usize) -> PResult<'_, Initializer> {
    binary_expression(
        input,
        depth,
        logical_and_expression,
        &[("||", InitializerBinaryOp::LogicalOr)],
    )
}

fn conditional_expression(input: Input<'_>, depth: usize) -> PResult<'_, Initializer> {
    depth::check(depth, input)?;
    let start = input.location_offset();
    let (input, condition) = logical_or_expression(input, depth)?;
    let (input, question) = opt(lexeme(char('?'))).parse(input)?;
    if question.is_none() {
        return Ok((input, condition));
    }

    let (input, then_expr) = expression(input, depth + 1)?;
    let (input, _) = lexeme(char(':')).parse(input)?;
    let (input, else_expr) = conditional_expression(input, depth + 1)?;
    depth::check(
        depth
            + 1
            + initializer_depth(&condition)
                .max(initializer_depth(&then_expr))
                .max(initializer_depth(&else_expr)),
        input,
    )?;
    Ok((
        input,
        Initializer::Conditional {
            span: Span::new(start, input.location_offset()),
            condition: Box::new(condition),
            then_expr: Box::new(then_expr),
            else_expr: Box::new(else_expr),
        },
    ))
}

fn expression(input: Input<'_>, depth: usize) -> PResult<'_, Initializer> {
    depth::check(depth, input)?;
    conditional_expression(input, depth)
}

fn initializer(input: Input<'_>, depth: usize) -> PResult<'_, Initializer> {
    depth::check(depth, input)?;
    alt((
        |input| initializer_list(input, depth),
        |input| expression(input, depth),
    ))
    .parse(input)
}

fn array_bounds(input: Input<'_>) -> PResult<'_, Vec<Option<u32>>> {
    many0(delimited(
        lexeme(char('[')),
        opt(lexeme(decimal)),
        lexeme(char(']')),
    ))
    .parse(input)
}

pub fn parameter(input: Input<'_>) -> PResult<'_, Parameter> {
    let (input, (span, (space, align, ty, ptr, name, bounds))) = with_span((
        opt(lexeme(state_space)),
        opt(lexeme(alignment)),
        lexeme(scalar_type),
        opt(preceded(
            lexeme(keyword(".ptr")),
            (opt(lexeme(state_space)), opt(lexeme(alignment))),
        )),
        lexeme(alt((identifier, value("_", char('_'))))),
        array_bounds,
    ))
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
            pointer: ptr.map(|(state_space, alignment)| PointerAnnotation {
                state_space,
                alignment,
            }),
        },
    ))
}

pub fn param_list(input: Input<'_>) -> PResult<'_, Vec<Parameter>> {
    delimited(
        lexeme(char('(')),
        separated_list0(comma_sep, lexeme(parameter)),
        lexeme(char(')')),
    )
    .parse(input)
}

pub fn variable_decl(input: Input<'_>, depth: usize) -> PResult<'_, VariableDecl> {
    let (
        input,
        (span, (linkage, space, attributes, align, vec, ty, name, count, bounds, init, ())),
    ) = with_span((
        opt(lexeme(linking_directive)),
        opt(lexeme(state_space)),
        opt(lexeme(attributes)),
        opt(lexeme(alignment)),
        opt(lexeme(vector_size)),
        lexeme(scalar_type),
        lexeme(identifier),
        opt(delimited(
            lexeme(char('<')),
            lexeme(decimal),
            lexeme(char('>')),
        )),
        array_bounds,
        opt(preceded(
            lexeme(char('=')),
            lexeme(|input| initializer(input, depth)),
        )),
        semicolon,
    ))
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

pub fn function_directive(input: Input<'_>) -> PResult<'_, FunctionDirective> {
    alt((
        map(with_span(keyword(".noreturn")), |(span, _)| {
            FunctionDirective::NoReturn { span }
        }),
        map(
            with_span(preceded(keyword(".maxnreg"), lexeme(decimal))),
            |(span, value)| FunctionDirective::MaxNReg { span, value },
        ),
        map(
            with_span(preceded(
                keyword(".maxntid"),
                separated_list1(comma_sep, lexeme(decimal)),
            )),
            |(span, values)| FunctionDirective::MaxNTid { span, values },
        ),
        map(
            with_span(preceded(
                keyword(".reqntid"),
                separated_list1(comma_sep, lexeme(decimal)),
            )),
            |(span, values)| FunctionDirective::ReqNTid { span, values },
        ),
        map(
            with_span(preceded(keyword(".minnctapersm"), lexeme(decimal))),
            |(span, value)| FunctionDirective::MinNCtaPerSm { span, value },
        ),
        map(
            with_span(preceded(keyword(".abi_preserve"), lexeme(decimal))),
            |(span, value)| FunctionDirective::AbiPreserve { span, value },
        ),
        map(
            with_span(preceded(keyword(".abi_preserve_control"), lexeme(decimal))),
            |(span, value)| FunctionDirective::AbiPreserveControl { span, value },
        ),
        map(
            with_span(preceded(
                keyword(".pragma"),
                terminated(lexeme(string_literal), opt(semicolon)),
            )),
            |(span, value)| FunctionDirective::Pragma { span, value },
        ),
    ))
    .parse(input)
}

fn pragma(input: Input<'_>) -> PResult<'_, TopLevelItem> {
    let (input, (span, (_, value, ()))) = with_span((
        lexeme(keyword(".pragma")),
        lexeme(string_literal),
        semicolon,
    ))
    .parse(input)?;
    Ok((input, TopLevelItem::Pragma { span, value }))
}

fn file_directive(input: Input<'_>) -> PResult<'_, TopLevelItem> {
    let (input, (span, (_, (), index, name, metadata, _))) = with_span((
        lexeme(keyword(".file")),
        ws,
        decimal,
        lexeme(string_literal),
        opt(preceded(
            comma_sep,
            pair(lexeme(decimal), preceded(comma_sep, lexeme(decimal))),
        )),
        opt(semicolon),
    ))
    .parse(input)?;

    Ok((
        input,
        TopLevelItem::File {
            span,
            index,
            name,
            timestamp: Some(metadata.map_or(0, |(timestamp, _)| timestamp)),
            file_size: Some(metadata.map_or(0, |(_, file_size)| file_size)),
        },
    ))
}

fn alias_directive(input: Input<'_>) -> PResult<'_, TopLevelItem> {
    let (input, (span, (_, alias, (), target, ()))) = with_span((
        lexeme(keyword(".alias")),
        lexeme(identifier),
        comma_sep,
        lexeme(identifier),
        semicolon,
    ))
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

fn section_width(input: Input<'_>) -> PResult<'_, SectionDataWidth> {
    alt((
        value(SectionDataWidth::B8, keyword(".b8")),
        value(SectionDataWidth::B16, keyword(".b16")),
        value(SectionDataWidth::B32, keyword(".b32")),
        value(SectionDataWidth::B64, keyword(".b64")),
    ))
    .parse(input)
}

fn section_value(input: Input<'_>) -> PResult<'_, SectionValue> {
    let label_value = map(with_span(identifier), |(span, label): (Span, &str)| {
        SectionValue::Label {
            span,
            name: label.to_string(),
        }
    });
    let label_offset = map(
        with_span(pair(identifier, lexeme(displacement))),
        |(span, (label, offset))| SectionValue::LabelOffset {
            span,
            label: label.to_string(),
            offset,
        },
    );
    let label_difference = map(
        with_span(pair(
            identifier,
            preceded(lexeme(char('-')), lexeme(identifier)),
        )),
        |(span, (left, right))| SectionValue::LabelDifference {
            span,
            left: left.to_string(),
            right: right.to_string(),
        },
    );

    alt((
        map(with_span(integer_literal), |(span, (value, unsigned))| {
            if unsigned {
                SectionValue::UnsignedInteger { span, value }
            } else {
                SectionValue::Integer {
                    span,
                    value: value.cast_signed(),
                }
            }
        }),
        label_difference,
        label_offset,
        label_value,
        map(with_span(dotted_symbol), |(span, label): (Span, &str)| {
            SectionValue::Label {
                span,
                name: label.to_string(),
            }
        }),
    ))
    .parse(input)
}

fn section_line(input: Input<'_>) -> PResult<'_, SectionLine> {
    alt((
        map(
            with_span(terminated(lexeme(identifier), lexeme(char(':')))),
            |(span, label)| SectionLine::Label {
                span,
                name: label.to_string(),
            },
        ),
        map(
            with_span(pair(
                lexeme(section_width),
                terminated(
                    separated_list1(comma_sep, lexeme(section_value)),
                    opt(semicolon),
                ),
            )),
            |(span, (width, values))| SectionLine::Data {
                span,
                width,
                values,
            },
        ),
    ))
    .parse(input)
}

fn section_directive(input: Input<'_>) -> PResult<'_, TopLevelItem> {
    let (input, (span, (_, name, lines))) = with_span((
        lexeme(keyword(".section")),
        lexeme(dotted_symbol),
        delimited(
            lexeme(char('{')),
            many0(lexeme(section_line)),
            lexeme(char('}')),
        ),
    ))
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

fn label_pattern(input: Input<'_>) -> PResult<'_, LabelPattern> {
    let (input, (span, (name, count))) = with_span(pair(
        identifier,
        opt(delimited(
            lexeme(char('<')),
            lexeme(decimal),
            lexeme(char('>')),
        )),
    ))
    .parse(input)?;

    let pattern = count.map_or_else(
        || LabelPattern::Name {
            span,
            name: name.to_string(),
        },
        |count| LabelPattern::Range {
            span,
            prefix: name.to_string(),
            count,
        },
    );
    Ok((input, pattern))
}

fn loc_function_name(input: Input<'_>) -> PResult<'_, LocFunctionName> {
    map(
        with_span(pair(dotted_symbol, opt(lexeme(displacement)))),
        |(span, (label, offset))| {
            offset.map_or_else(
                || LocFunctionName::Label {
                    span,
                    label: label.to_string(),
                },
                |offset| LocFunctionName::LabelOffset {
                    span,
                    label: label.to_string(),
                    offset,
                },
            )
        },
    )
    .parse(input)
}

fn loc_directive(input: Input<'_>) -> PResult<'_, LocDirective> {
    let (input, (span, (_, file, line, column, function_name, inlined_at))) = with_span((
        keyword(".loc"),
        lexeme(decimal),
        lexeme(decimal),
        lexeme(decimal),
        opt(preceded(
            lexeme(char(',')),
            preceded(lexeme(keyword("function_name")), lexeme(loc_function_name)),
        )),
        opt(preceded(
            lexeme(char(',')),
            preceded(
                lexeme(keyword("inlined_at")),
                map(
                    with_span((lexeme(decimal), lexeme(decimal), lexeme(decimal))),
                    |(span, (file, line, column))| LocInlineSite {
                        span,
                        file,
                        line,
                        column,
                    },
                ),
            ),
        )),
    ))
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

pub fn directive_statement(input: Input<'_>) -> PResult<'_, DirectiveStatement> {
    alt((
        map(
            with_span(preceded(
                keyword(".pragma"),
                terminated(lexeme(string_literal), semicolon),
            )),
            |(span, value)| DirectiveStatement::Pragma { span, value },
        ),
        map(loc_directive, DirectiveStatement::Loc),
        map(
            with_span(preceded(
                keyword(".branchtargets"),
                terminated(separated_list1(comma_sep, lexeme(label_pattern)), semicolon),
            )),
            |(span, labels)| DirectiveStatement::BranchTargets { span, labels },
        ),
        map(
            with_span(preceded(
                keyword(".calltargets"),
                terminated(
                    separated_list1(comma_sep, lexeme(map(identifier, str::to_string))),
                    semicolon,
                ),
            )),
            |(span, targets)| DirectiveStatement::CallTargets { span, targets },
        ),
        map(
            with_span(preceded(
                keyword(".callprototype"),
                terminated(
                    (
                        opt(param_list),
                        lexeme(alt((identifier, value("_", char('_'))))),
                        opt(param_list),
                        many0(lexeme(function_directive)),
                    ),
                    semicolon,
                ),
            )),
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

pub fn top_level_item(input: Input<'_>) -> PResult<'_, TopLevelItem> {
    alt((
        map(function, TopLevelItem::Function),
        pragma,
        file_directive,
        alias_directive,
        section_directive,
        map(|input| variable_decl(input, 0), TopLevelItem::Variable),
    ))
    .parse(input)
}

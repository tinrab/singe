use nom::{
    Parser, bytes::complete::take_while1, character::complete::char, combinator::opt,
    multi::separated_list0, sequence::preceded,
};

use crate::{
    ast::{Instruction, Operand, PredicateGuard, SetPredicateDetails, Span},
    instruction_opcode,
    parser::{
        common::{PResult, comma_sep, lexeme, semicolon, with_span, ws},
        offset,
        operand::operand,
    },
    syntax_for_opcode,
};

fn guard_predicate<'a>(source: &'a str, input: &'a str) -> PResult<'a, PredicateGuard> {
    let (input, (span, (_, negated, _, name))) = with_span(
        source,
        (
            char('@'),
            opt(char('!')),
            char('%'),
            take_while1(|c: char| c.is_ascii_alphanumeric() || c == '_'),
        ),
    )
    .parse(input)?;
    Ok((
        input,
        PredicateGuard {
            span,
            negated: negated.is_some(),
            register: format!("%{name}"),
        },
    ))
}

fn instruction_head(input: &str) -> PResult<'_, (&str, Vec<String>)> {
    let (input, head) =
        take_while1(|c: char| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | ':'))
            .parse(input)?;

    let mut best_opcode = None;
    for candidate in crate::generated::instruction_set::PTX_91_INSTRUCTION_OPCODES {
        if head == *candidate {
            best_opcode = Some(*candidate);
            break;
        }
        if head.starts_with(candidate)
            && head[candidate.len()..].starts_with('.')
            && best_opcode.is_none_or(|current: &str| candidate.len() > current.len())
        {
            best_opcode = Some(*candidate);
        }
    }

    let opcode = match best_opcode {
        Some(opcode) => opcode,
        None => {
            return Err(nom::Err::Error(nom::error::Error::new(
                input,
                nom::error::ErrorKind::Tag,
            )));
        }
    };

    let suffix = &head[opcode.len()..];
    let modifiers: Vec<String> = suffix
        .split('.')
        .filter_map(|segment: &str| {
            if segment.is_empty() {
                None
            } else {
                Some(segment.to_string())
            }
        })
        .collect();
    Ok((input, (opcode, modifiers)))
}

fn matches_known_syntax(opcode: &str, modifiers: &[String], operand_count: usize) -> bool {
    let variants = syntax_for_opcode(opcode);
    variants.is_empty()
        || variants.iter().any(|variant| {
            operand_count >= variant.min_operand_arity
                && operand_count <= variant.max_operand_arity
                && modifiers_match_variant(modifiers, variant.syntax, variant.required_modifiers)
        })
}

fn modifiers_match_variant(
    modifiers: &[String],
    syntax: &str,
    required_modifiers: &[&str],
) -> bool {
    let mut allowed = Vec::new();
    for required in required_modifiers {
        let modifier = required.trim_start_matches('.');
        if !modifier.is_empty() && modifier != "type" {
            allowed.push(modifier);
        }
    }

    let head = syntax.split_once(' ').map_or(syntax, |(head, _)| head);
    for atom in modifier_atoms(head) {
        match atom {
            atom if is_type_placeholder(atom) => allowed.extend(TYPE_MODIFIERS),
            atom if is_rounding_placeholder(atom) => allowed.extend(ROUNDING_MODIFIERS),
            atom if is_compare_placeholder(atom) => allowed.extend(COMPARE_MODIFIERS),
            atom if is_boolean_placeholder(atom) => allowed.extend(BOOLEAN_MODIFIERS),
            atom if is_state_space_placeholder(atom) => allowed.extend(STATE_SPACE_MODIFIERS),
            atom if is_mode_placeholder(atom) => allowed.extend(MODE_MODIFIERS),
            atom if is_size_placeholder(atom) => allowed.extend(SIZE_MODIFIERS),
            atom if is_plain_placeholder(atom) => {}
            atom => allowed.push(atom.trim_start_matches('.')),
        }
    }

    if !required_modifiers.iter().all(|required| {
        let required = required.trim_start_matches('.');
        required.is_empty()
            || required == "type"
            || modifiers.iter().any(|modifier| modifier == required)
    }) {
        return false;
    }

    modifiers
        .iter()
        .all(|modifier| allowed.iter().any(|allowed| modifier == allowed))
}

const TYPE_MODIFIERS: &[&str] = &[
    "pred", "b8", "b16", "b32", "b64", "b128", "b1024", "u8", "u16", "u32", "u64", "s8", "s16",
    "s32", "s64", "f16", "f16x2", "f32", "f32x2", "f64", "bf16", "bf16x2", "tf32", "ue8m0x2",
    "fp16x2", "s2f6x2",
];

const ROUNDING_MODIFIERS: &[&str] = &[
    "rn", "rna", "rz", "rm", "rp", "rni", "rzi", "rmi", "rpi", "rs",
];

const COMPARE_MODIFIERS: &[&str] = &[
    "eq", "ne", "lt", "le", "gt", "ge", "lo", "ls", "hi", "hs", "equ", "neu", "ltu", "leu", "gtu",
    "geu", "num", "nan",
];

const BOOLEAN_MODIFIERS: &[&str] = &["and", "or", "xor"];
const STATE_SPACE_MODIFIERS: &[&str] = &["const", "global", "local", "param", "shared"];
const MODE_MODIFIERS: &[&str] = &["clamp", "wrap", "hi", "lo", "wide"];
const SIZE_MODIFIERS: &[&str] = &["u32", "u64"];

fn modifier_atoms(head: &str) -> impl Iterator<Item = &str> {
    head.match_indices('.').filter_map(|(start, _)| {
        let tail = &head[start..];
        let end = tail
            .char_indices()
            .skip(1)
            .take_while(|(_, character)| {
                character.is_ascii_alphanumeric() || matches!(character, '_' | ':')
            })
            .last()
            .map_or(1, |(index, character)| index + character.len_utf8());
        (end > 1).then_some(&tail[..end])
    })
}

fn is_type_placeholder(atom: &str) -> bool {
    atom.trim_start_matches('.')
        .to_ascii_lowercase()
        .contains("type")
}

fn is_rounding_placeholder(atom: &str) -> bool {
    atom.trim_start_matches('.').eq_ignore_ascii_case("rnd")
}

fn is_compare_placeholder(atom: &str) -> bool {
    atom.trim_start_matches('.')
        .to_ascii_lowercase()
        .contains("cmpop")
}

fn is_boolean_placeholder(atom: &str) -> bool {
    atom.trim_start_matches('.')
        .to_ascii_lowercase()
        .contains("boolop")
}

fn is_plain_placeholder(atom: &str) -> bool {
    matches!(
        atom.trim_start_matches('.').to_ascii_lowercase().as_str(),
        "" | "op"
            | "sem"
            | "scope"
            | "cop"
            | "cmp"
            | "dim"
            | "shape"
            | "vec"
            | "permute"
            | "kind"
            | "level::cache_hint"
            | "completion_mechanism"
    )
}

fn is_state_space_placeholder(atom: &str) -> bool {
    matches!(
        atom.trim_start_matches('.').to_ascii_lowercase().as_str(),
        "space" | "ss"
    )
}

fn is_mode_placeholder(atom: &str) -> bool {
    atom.trim_start_matches('.').eq_ignore_ascii_case("mode")
}

fn is_size_placeholder(atom: &str) -> bool {
    atom.trim_start_matches('.').eq_ignore_ascii_case("size")
}

fn setp_destination<'a>(
    source: &'a str,
    input: &'a str,
) -> PResult<'a, (Operand, Option<Operand>)> {
    let (input, first) = operand(source, input)?;
    let (input, second) =
        opt(preceded(lexeme(char('|')), |input| operand(source, input))).parse(input)?;
    Ok((input, (first, second)))
}

fn setp_predicate_input<'a>(source: &'a str, input: &'a str) -> PResult<'a, (bool, Operand)> {
    let (input, negated) = opt(lexeme(char('!'))).parse(input)?;
    let (input, predicate) = operand(source, input)?;
    Ok((input, (negated.is_some(), predicate)))
}

pub fn instruction<'a>(source: &'a str, input: &'a str) -> PResult<'a, Instruction> {
    let start = offset(source, input);
    let (input, guard) = opt(lexeme(|input| guard_predicate(source, input))).parse(input)?;

    let (input, (opcode, modifiers)) = lexeme(instruction_head).parse(input)?;

    let (input, _) = ws(input)?;

    let (input, operands, setp_details) = if opcode == "setp" {
        let (input, (destination, secondary_destination)) =
            lexeme(|input| setp_destination(source, input)).parse(input)?;
        let (input, _) = opt(comma_sep).parse(input)?;
        let (input, lhs) = lexeme(|input| operand(source, input)).parse(input)?;
        let (input, _) = comma_sep(input)?;
        let (input, rhs) = lexeme(|input| operand(source, input)).parse(input)?;
        let (input, tail) = opt(preceded(
            comma_sep,
            lexeme(|input| setp_predicate_input(source, input)),
        ))
        .parse(input)?;

        let mut operands = vec![destination, lhs, rhs];
        let mut predicate_input_negated = false;
        if let Some((negated, predicate_input)) = tail {
            predicate_input_negated = negated;
            operands.push(predicate_input);
        }

        (
            input,
            operands,
            Some(SetPredicateDetails {
                span: Span::new(start, offset(source, input)),
                secondary_destination,
                predicate_input_negated,
            }),
        )
    } else {
        let (input, operands) =
            separated_list0(comma_sep, lexeme(|input| operand(source, input))).parse(input)?;
        (input, operands, None)
    };

    if !matches_known_syntax(opcode, &modifiers, operands.len()) {
        return Err(nom::Err::Failure(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Verify,
        )));
    }

    let (input, _) = semicolon(input)?;

    Ok((
        input,
        Instruction {
            span: Span::new(start, offset(source, input)),
            guard,
            opcode: opcode.to_string(),
            opcode_kind: instruction_opcode(opcode),
            modifiers,
            operands,
            setp_details,
        },
    ))
}

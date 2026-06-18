use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

use anyhow::{Context, Result};
use bomboni_core::string::{Case, str_to_case};
use proc_macro2::{Ident, Literal, Span};
use quote::quote;
use serde::Deserialize;

use crate::utility::workspace_root;

const PTX_INSTRUCTIONS_JSON: &str = include_str!("../docs/ptx-9.1-instructions.json");

pub fn generate_ptx_instruction_parser() -> Result<()> {
    let root = workspace_root()?;
    let spec = parse_spec()?;
    let model = build_model(&spec.instructions);
    let output_path = root.join("singe-ptx/src/generated/instruction_set.rs");
    let contents = render_instruction_set(&model)?;
    write_if_changed(&output_path, &contents)?;
    println!("updated {}", rel(&root, &output_path));
    Ok(())
}

fn parse_spec() -> Result<InstructionSpecFile> {
    serde_json::from_str(PTX_INSTRUCTIONS_JSON).context("failed to parse PTX instruction JSON")
}

fn build_model(instructions: &[InstructionSpec]) -> InstructionSetModel {
    let mut by_name: BTreeMap<String, Vec<InstructionVariant>> = BTreeMap::new();
    let mut summaries: BTreeMap<String, String> = BTreeMap::new();

    for instruction in instructions {
        let filtered_names: Vec<String> = instruction
            .names
            .iter()
            .filter(|name| is_instruction_name(name))
            .cloned()
            .collect();
        if filtered_names.is_empty() {
            continue;
        }

        let summary = instruction.summary.trim();
        if !summary.is_empty() {
            for name in &filtered_names {
                summaries
                    .entry(name.clone())
                    .or_insert_with(|| summary.to_string());
            }
        }

        let grouped = classify_syntax(&filtered_names, &instruction.syntax);
        for (name, variants) in grouped {
            by_name.entry(name).or_default().extend(variants);
        }
    }

    let instructions = by_name
        .into_iter()
        .map(|(name, mut variants)| {
            variants.sort_by(|left, right| left.syntax.cmp(&right.syntax));
            variants.dedup_by(|left, right| left.syntax == right.syntax);
            InstructionEntry {
                enum_name: opcode_enum_name(&name),
                summary: summaries.remove(&name).unwrap_or_default(),
                name,
                variants,
            }
        })
        .collect();

    InstructionSetModel { instructions }
}

fn classify_syntax(
    names: &[String],
    syntax_lines: &[String],
) -> BTreeMap<String, Vec<InstructionVariant>> {
    let known_names: Vec<String> = names.iter().map(|name| normalize_opcode(name)).collect();
    let known_name_set: BTreeSet<&str> = known_names.iter().map(String::as_str).collect();
    let placeholder_names = collect_placeholder_names(syntax_lines);

    let mut grouped = BTreeMap::<String, Vec<InstructionVariant>>::new();
    for line in syntax_lines {
        if line.trim_start().starts_with('.') {
            continue;
        }
        let syntax = normalize_syntax_line(line);
        if syntax.is_empty() {
            continue;
        }
        if let Some((name, variant)) = parse_variant(&syntax, &known_name_set, &placeholder_names) {
            grouped.entry(name).or_default().push(variant);
        }
    }

    for name in known_names {
        grouped.entry(name).or_default();
    }

    grouped
}

fn is_instruction_name(name: &str) -> bool {
    let Some(first) = name.chars().next() else {
        return false;
    };
    first.is_ascii_alphabetic() && !name.contains('{') && !name.contains('}')
}

fn parse_variant(
    syntax: &str,
    known_names: &BTreeSet<&str>,
    placeholder_names: &BTreeSet<String>,
) -> Option<(String, InstructionVariant)> {
    let trimmed = syntax.trim_end_matches(';').trim();
    if trimmed.is_empty() {
        return None;
    }

    let split_at = trimmed.find(char::is_whitespace).unwrap_or(trimmed.len());
    let head = &trimmed[..split_at];
    let operands = trimmed[split_at..].trim();
    let min_operand_arity = if operands.is_empty() {
        0
    } else {
        operands
            .split(',')
            .filter(|segment| !segment.contains('{'))
            .count()
    };
    let max_operand_arity = if operands.is_empty() {
        0
    } else {
        operands.split(',').count()
    };

    let opcode = longest_matching_opcode(head, known_names)?;
    let suffix = &head[opcode.len()..];
    let required_modifiers = extract_required_modifiers(suffix, placeholder_names);
    Some((
        opcode.to_string(),
        InstructionVariant {
            syntax: syntax.to_string(),
            min_operand_arity,
            max_operand_arity,
            required_modifiers,
        },
    ))
}

fn collect_placeholder_names(syntax_lines: &[String]) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for line in syntax_lines {
        let trimmed = line.trim();
        if !trimmed.starts_with('.') {
            continue;
        }
        let Some((left, _)) = trimmed.split_once('=') else {
            continue;
        };
        let name = left.trim();
        if !name.is_empty() {
            names.insert(name.trim_end_matches(';').to_string());
        }
    }
    names
}

fn longest_matching_opcode<'a>(head: &'a str, known_names: &BTreeSet<&str>) -> Option<&'a str> {
    let mut best = None;
    for name in known_names {
        if head == *name {
            return Some(&head[..name.len()]);
        }
        if head.starts_with(name)
            && matches!(head[name.len()..].chars().next(), Some('.' | '{'))
            && best.is_none_or(|current: &str| name.len() > current.len())
        {
            best = Some(&head[..name.len()]);
        }
    }
    best
}

fn extract_required_modifiers(suffix: &str, placeholder_names: &BTreeSet<String>) -> Vec<String> {
    let mut modifiers = Vec::new();
    let bytes = suffix.as_bytes();
    let mut index = 0;
    let mut brace_depth: usize = 0;

    while index < bytes.len() {
        match bytes[index] {
            b'{' => {
                brace_depth += 1;
                index += 1;
                continue;
            }
            b'}' => {
                brace_depth = brace_depth.saturating_sub(1);
                index += 1;
                continue;
            }
            _ => {}
        }

        if bytes[index] != b'.' || brace_depth > 0 {
            index += 1;
            continue;
        }

        let start = index;
        index += 1;
        while index < bytes.len() {
            let ch = bytes[index] as char;
            if ch.is_ascii_alphanumeric() || matches!(ch, '_' | ':') {
                index += 1;
            } else {
                break;
            }
        }

        if start < index {
            let modifier = suffix[start..index].to_string();
            if !placeholder_names.contains(&modifier) {
                modifiers.push(modifier);
            }
        }
    }

    modifiers
}

fn normalize_opcode(name: &str) -> String {
    name.trim().to_string()
}

fn normalize_syntax_line(line: &str) -> String {
    line.split("//").next().unwrap_or(line).trim().to_string()
}

fn opcode_enum_name(name: &str) -> String {
    if let Some(override_name) = opcode_enum_name_override(name) {
        return override_name.to_string();
    }

    let mut out = String::new();

    for part in name.split(['.', ':', '_']) {
        let cleaned: String = part
            .chars()
            .filter(|ch| ch.is_ascii_alphanumeric())
            .collect();
        if cleaned.is_empty() {
            continue;
        }

        let mut chars = cleaned.chars();
        let Some(first) = chars.next() else {
            continue;
        };

        if first.is_ascii_digit() {
            out.push('_');
            out.push(first);
        } else {
            out.push(first.to_ascii_uppercase());
        }

        for ch in chars {
            out.push(ch.to_ascii_lowercase());
        }
    }

    if out.is_empty() {
        out.push_str("Opcode");
    }

    str_to_case(out, Case::Pascal)
}

fn opcode_enum_name_override(name: &str) -> Option<&'static str> {
    match name {
        "activemask" => Some("ActiveMask"),
        "applypriority" => Some("ApplyPriority"),
        "clusterlaunchcontrol.query_cancel" => Some("ClusterLaunchControlQueryCancel"),
        "clusterlaunchcontrol.try_cancel" => Some("ClusterLaunchControlTryCancel"),
        "createpolicy" => Some("CreatePolicy"),
        "getctarank" => Some("GetCtaRank"),
        "griddepcontrol" => Some("GridDepControl"),
        "isspacep" => Some("IsSpaceP"),
        "istypep" => Some("IsTypeP"),
        "ldmatrix" => Some("LdMatrix"),
        "mma.sp.ordered_metadata" => Some("MmaSpOrderedMetadata"),
        "movmatrix" => Some("MovMatrix"),
        "multimem.ld_reduce" => Some("MultimemLdReduce"),
        "pmevent" => Some("PmEvent"),
        "setmaxnreg" => Some("SetMaxNReg"),
        "stackrestore" => Some("StackRestore"),
        "stacksave" => Some("StackSave"),
        "stmatrix" => Some("StMatrix"),
        "tensormap.cp_fenceproxy" => Some("TensorMapCpFenceProxy"),
        "vabsdiff" => Some("VAbsDiff"),
        "vabsdiff2" => Some("VAbsDiff2"),
        "vabsdiff4" => Some("VAbsDiff4"),
        "vadd" => Some("VAdd"),
        "vadd2" => Some("VAdd2"),
        "vadd4" => Some("VAdd4"),
        "vavrg2" => Some("VAvrg2"),
        "vavrg4" => Some("VAvrg4"),
        "vmad" => Some("VMad"),
        "vmax" => Some("VMax"),
        "vmax2" => Some("VMax2"),
        "vmax4" => Some("VMax4"),
        "vmin" => Some("VMin"),
        "vmin2" => Some("VMin2"),
        "vmin4" => Some("VMin4"),
        "vset" => Some("VSet"),
        "vset2" => Some("VSet2"),
        "vset4" => Some("VSet4"),
        "vshl" => Some("VShl"),
        "vshr" => Some("VShr"),
        "vsub" => Some("VSub"),
        "vsub2" => Some("VSub2"),
        "vsub4" => Some("VSub4"),
        _ => None,
    }
}

fn render_instruction_set(model: &InstructionSetModel) -> Result<String> {
    let opcode_variants = model.instructions.iter().map(|instruction| {
        let opcode_ident = make_ident(&instruction.enum_name);
        let summary = Literal::string(&instruction.summary);
        quote! {
            #[doc = #summary]
            #opcode_ident
        }
    });

    let opcode_literals: Vec<Literal> = model
        .instructions
        .iter()
        .map(|instruction| Literal::string(&instruction.name))
        .collect();

    let all_syntax_entries = model.instructions.iter().flat_map(|instruction| {
        let opcode_ident = make_ident(&instruction.enum_name);
        let name_literal = Literal::string(&instruction.name);
        instruction.variants.iter().map(move |variant| {
            let syntax_literal = Literal::string(&variant.syntax);
            let min_operand_arity = variant.min_operand_arity;
            let max_operand_arity = variant.max_operand_arity;
            let required_modifiers: Vec<Literal> = variant
                .required_modifiers
                .iter()
                .map(|modifier| Literal::string(modifier))
                .collect();

            quote! {
                InstructionSyntax {
                    opcode: InstructionOpcode::#opcode_ident,
                    name: #name_literal,
                    syntax: #syntax_literal,
                    min_operand_arity: #min_operand_arity,
                    max_operand_arity: #max_operand_arity,
                    required_modifiers: &[#(#required_modifiers),*],
                }
            }
        })
    });

    let opcode_match_arms = model.instructions.iter().map(|instruction| {
        let opcode_ident = make_ident(&instruction.enum_name);
        let name_literal = Literal::string(&instruction.name);
        quote! {
            #name_literal => Some(InstructionOpcode::#opcode_ident)
        }
    });

    let syntax_match_arms = model.instructions.iter().map(|instruction| {
        let name_literal = Literal::string(&instruction.name);
        let opcode_ident = make_ident(&instruction.enum_name);
        let variants = instruction.variants.iter().map(|variant| {
            let syntax_literal = Literal::string(&variant.syntax);
            let min_operand_arity = variant.min_operand_arity;
            let max_operand_arity = variant.max_operand_arity;
            let required_modifiers: Vec<Literal> = variant
                .required_modifiers
                .iter()
                .map(|modifier| Literal::string(modifier))
                .collect();

            quote! {
                InstructionSyntax {
                    opcode: InstructionOpcode::#opcode_ident,
                    name: #name_literal,
                    syntax: #syntax_literal,
                    min_operand_arity: #min_operand_arity,
                    max_operand_arity: #max_operand_arity,
                    required_modifiers: &[#(#required_modifiers),*],
                }
            }
        });

        quote! {
            #name_literal => &[#(#variants),*]
        }
    });

    let tokens = quote! {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum InstructionOpcode {
            #(#opcode_variants),*
        }

        #[derive(Debug, Clone, PartialEq, Eq)]
        pub struct InstructionSyntax {
            pub opcode: InstructionOpcode,
            pub name: &'static str,
            pub syntax: &'static str,
            pub min_operand_arity: usize,
            pub max_operand_arity: usize,
            pub required_modifiers: &'static [&'static str],
        }

        pub const PTX_91_INSTRUCTION_OPCODES: &[&str] = &[
            #(#opcode_literals),*
        ];

        pub const PTX_91_INSTRUCTION_SYNTAX: &[InstructionSyntax] = &[
            #(#all_syntax_entries),*
        ];

        pub fn instruction_opcode(name: &str) -> Option<InstructionOpcode> {
            match name {
                #(#opcode_match_arms,)*
                _ => None,
            }
        }

        pub fn syntax_for_opcode(name: &str) -> &'static [InstructionSyntax] {
            match name {
                #(#syntax_match_arms,)*
                _ => &[],
            }
        }
    };

    let file = syn::parse2(tokens).context("failed to build generated Rust AST")?;
    let mut output = String::new();
    output.push_str("// auto-generated by `cargo xtask gen-ptx-instructions`\n");
    output.push_str("// do not edit manually\n\n");
    output.push_str(&format_doc_comment_spacing(&prettyplease::unparse(&file)));
    Ok(output)
}

fn format_doc_comment_spacing(input: &str) -> String {
    input
        .lines()
        .map(|line| {
            let Some((prefix, rest)) = line.split_once("///") else {
                return line.to_string();
            };
            if rest.is_empty()
                || rest.starts_with(' ')
                || rest.starts_with('/')
                || rest.starts_with('!')
            {
                line.to_string()
            } else {
                format!("{prefix}/// {rest}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}

fn make_ident(name: &str) -> Ident {
    Ident::new(name, Span::call_site())
}

fn write_if_changed(path: &Path, contents: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let current = fs::read_to_string(path).ok();
    if current.as_deref() == Some(contents) {
        return Ok(());
    }

    fs::write(path, contents).with_context(|| format!("failed to write {}", path.display()))
}

fn rel(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}

#[derive(Debug, Deserialize)]
struct InstructionSpecFile {
    instructions: Vec<InstructionSpec>,
}

#[derive(Debug, Deserialize)]
struct InstructionSpec {
    names: Vec<String>,
    summary: String,
    syntax: Vec<String>,
}

#[derive(Debug)]
struct InstructionSetModel {
    instructions: Vec<InstructionEntry>,
}

#[derive(Debug)]
struct InstructionEntry {
    enum_name: String,
    summary: String,
    name: String,
    variants: Vec<InstructionVariant>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct InstructionVariant {
    syntax: String,
    min_operand_arity: usize,
    max_operand_arity: usize,
    required_modifiers: Vec<String>,
}

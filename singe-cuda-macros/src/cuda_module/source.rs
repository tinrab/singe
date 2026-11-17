// Kernel declarations read from the CUDA source with tree-sitter, and the Rust scalar each C parameter type maps to.

use std::{collections::HashMap, mem};

use proc_macro2::Span;
use syn::{Error, Result};
use tree_sitter::{Node, Parser};

pub(crate) struct SourceKernel {
    pub(crate) name: String,
    pub(crate) params: Vec<SourceParam>,
}

pub(crate) struct SourceParam {
    pub(crate) name: String,
    pub(crate) declaration: String,
    pub(crate) kind: SourceParamKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SourceParamKind {
    Scalar(Option<RustScalar>),
    Pointer {
        levels: Vec<bool>,
        element: Option<RustScalar>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RustScalar {
    Bool,
    F16,
    Bf16,
    F32,
    F64,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    USize,
    ISize,
}

// Parses the `__global__` functions of `source`, reporting a parser failure at `span`.
pub(crate) fn parse_source_kernels(source: &str, span: Span) -> Result<Vec<SourceKernel>> {
    let mut parser = Parser::new();
    let language = tree_sitter_c::LANGUAGE;
    parser
        .set_language(&language.into())
        .map_err(|error| Error::new(span, format!("failed to parse cuda source: {error}")))?;
    let tree = parser.parse(source, None).ok_or_else(|| {
        Error::new(
            span,
            "failed to parse cuda source: the parser returned no tree",
        )
    })?;

    let aliases = collect_type_aliases(source, tree.root_node());
    let mut kernels = Vec::new();
    collect_source_kernels(source, tree.root_node(), &aliases, &mut kernels);
    Ok(kernels)
}

fn collect_source_kernels(
    source: &str,
    node: Node<'_>,
    aliases: &HashMap<String, RustScalar>,
    kernels: &mut Vec<SourceKernel>,
) {
    if node.kind() == "function_definition"
        && let Some(kernel) = parse_source_kernel(source, node, aliases)
    {
        kernels.push(kernel);
    }

    for index in 0..node.child_count() {
        if let Some(child) = node.child(index) {
            collect_source_kernels(source, child, aliases, kernels);
        }
    }
}

fn parse_source_kernel(
    source: &str,
    node: Node<'_>,
    aliases: &HashMap<String, RustScalar>,
) -> Option<SourceKernel> {
    let declarator = function_declarator(node)?;
    let prefix = &source[node.start_byte()..declarator.start_byte()];
    if !contains_identifier(prefix, "__global__") {
        return None;
    }

    let name_node = declarator
        .child_by_field_name("declarator")
        .and_then(declarator_name)?;
    let name = node_text(source, name_node).trim().to_string();
    let params = declarator
        .child_by_field_name("parameters")
        .or_else(|| find_descendant_kind(declarator, "parameter_list"))
        .map_or_default(|parameters| parse_source_parameters(source, parameters, aliases));

    Some(SourceKernel { name, params })
}

fn function_declarator(node: Node<'_>) -> Option<Node<'_>> {
    node.child_by_field_name("declarator")
        .and_then(|declarator| find_descendant_kind(declarator, "function_declarator"))
}

fn parse_source_parameters(
    source: &str,
    parameters: Node<'_>,
    aliases: &HashMap<String, RustScalar>,
) -> Vec<SourceParam> {
    let mut params = Vec::new();
    for index in 0..parameters.child_count() {
        let Some(child) = parameters.child(index) else {
            continue;
        };
        if child.kind() != "parameter_declaration" {
            continue;
        }
        if let Some(param) = parse_source_parameter(source, child, params.len(), aliases) {
            params.push(param);
        }
    }
    params
}

fn parse_source_parameter(
    source: &str,
    node: Node<'_>,
    index: usize,
    aliases: &HashMap<String, RustScalar>,
) -> Option<SourceParam> {
    let declaration = node_text(source, node).trim();
    if declaration.is_empty() || declaration == "void" {
        return None;
    }

    let declarator = node.child_by_field_name("declarator");
    let name_node = declarator.and_then(declarator_name);
    let name = name_node
        .map(|node| node_text(source, node).trim().to_string())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| format!("arg{index}"));
    let array = declarator.is_some_and(|node| declarator_has_kind(node, "array_declarator"));
    let levels = pointer_levels_from_declaration(declaration, array);
    let scalar = source_scalar_type(declaration, &name, aliases);
    let kind = if levels.is_empty() {
        SourceParamKind::Scalar(scalar)
    } else {
        SourceParamKind::Pointer {
            levels,
            element: scalar,
        }
    };

    Some(SourceParam {
        name,
        declaration: declaration.to_string(),
        kind,
    })
}

fn pointer_levels_from_declaration(declaration: &str, array: bool) -> Vec<bool> {
    let mut levels = Vec::new();
    let mut previous = 0usize;
    for (star, _) in declaration.match_indices('*') {
        levels.push(!contains_identifier(&declaration[previous..star], "const"));
        previous = star + 1;
    }

    if array {
        let segment = if levels.is_empty() {
            declaration
        } else {
            &declaration[previous..]
        };
        levels.push(!contains_identifier(segment, "const"));
    }
    levels
}

fn source_scalar_type(
    declaration: &str,
    name: &str,
    aliases: &HashMap<String, RustScalar>,
) -> Option<RustScalar> {
    let tokens = declaration_type_tokens(declaration, name);
    resolve_scalar_tokens(&tokens, aliases)
}

fn resolve_scalar_tokens(
    tokens: &[String],
    aliases: &HashMap<String, RustScalar>,
) -> Option<RustScalar> {
    if tokens.iter().any(|token| token == "void") {
        return None;
    }

    if let Some(scalar) = tokens.iter().find_map(|token| aliases.get(token).copied()) {
        return Some(scalar);
    }

    if tokens
        .iter()
        .any(|token| token == "bool" || token == "_Bool")
    {
        return Some(RustScalar::Bool);
    }

    if tokens.iter().any(|token| {
        matches!(
            token.as_str(),
            "__half" | "half" | "__nv_half" | "cuda_fp16"
        )
    }) {
        return Some(RustScalar::F16);
    }

    if tokens.iter().any(|token| {
        matches!(
            token.as_str(),
            "__nv_bfloat16" | "nv_bfloat16" | "bfloat16" | "bf16"
        )
    }) {
        return Some(RustScalar::Bf16);
    }

    if tokens.iter().any(|token| token == "double") {
        // `long double` has no Rust equivalent.
        return (!tokens.iter().any(|token| token == "long")).then_some(RustScalar::F64);
    }

    if tokens.iter().any(|token| token == "float") {
        return Some(RustScalar::F32);
    }

    if tokens.iter().any(|token| token == "size_t") {
        return Some(RustScalar::USize);
    }

    if tokens
        .iter()
        .any(|token| matches!(token.as_str(), "ssize_t" | "ptrdiff_t"))
    {
        return Some(RustScalar::ISize);
    }

    if tokens
        .iter()
        .any(|token| matches!(token.as_str(), "uint64_t" | "uintptr_t"))
    {
        return Some(RustScalar::U64);
    }

    if tokens
        .iter()
        .any(|token| matches!(token.as_str(), "int64_t" | "intptr_t"))
    {
        return Some(RustScalar::I64);
    }

    if tokens.iter().any(|token| token == "uint32_t") {
        return Some(RustScalar::U32);
    }

    if tokens.iter().any(|token| token == "int32_t") {
        return Some(RustScalar::I32);
    }

    if tokens.iter().any(|token| token == "uint16_t") {
        return Some(RustScalar::U16);
    }

    if tokens.iter().any(|token| token == "int16_t") {
        return Some(RustScalar::I16);
    }

    if tokens.iter().any(|token| token == "uint8_t") {
        return Some(RustScalar::U8);
    }

    if tokens.iter().any(|token| token == "int8_t") {
        return Some(RustScalar::I8);
    }

    resolve_integer_tokens(tokens)
}

fn resolve_integer_tokens(tokens: &[String]) -> Option<RustScalar> {
    let unsigned = tokens.iter().any(|token| token == "unsigned");
    let signed = tokens.iter().any(|token| token == "signed");
    let long_count = tokens.iter().filter(|token| *token == "long").count();
    // Device code follows the host ABI: `long` has 64 bits on LP64 targets, and 32 on Windows and 32-bit targets.
    if long_count >= 2 || (long_count == 1 && cfg!(all(target_pointer_width = "64", not(windows))))
    {
        return Some(if unsigned {
            RustScalar::U64
        } else {
            RustScalar::I64
        });
    }

    if long_count == 1 {
        return Some(if unsigned {
            RustScalar::U32
        } else {
            RustScalar::I32
        });
    }

    if tokens.iter().any(|token| token == "short") {
        return Some(if unsigned {
            RustScalar::U16
        } else {
            RustScalar::I16
        });
    }

    if tokens.iter().any(|token| token == "char") {
        return match (unsigned, signed) {
            (true, _) => Some(RustScalar::U8),
            (false, true) => Some(RustScalar::I8),
            _ => None,
        };
    }

    if tokens.iter().any(|token| token == "int") {
        return Some(if unsigned {
            RustScalar::U32
        } else {
            RustScalar::I32
        });
    }
    None
}

fn collect_type_aliases(source: &str, root: Node<'_>) -> HashMap<String, RustScalar> {
    let mut aliases = HashMap::new();
    collect_typedef_aliases(source, root, &mut aliases);
    collect_using_aliases(source, &mut aliases);
    aliases
}

fn collect_typedef_aliases(
    source: &str,
    node: Node<'_>,
    aliases: &mut HashMap<String, RustScalar>,
) {
    if node.kind() == "type_definition" {
        collect_typedef_alias(source, node, aliases);
    }

    for index in 0..node.child_count() {
        if let Some(child) = node.child(index) {
            collect_typedef_aliases(source, child, aliases);
        }
    }
}

fn collect_typedef_alias(source: &str, node: Node<'_>, aliases: &mut HashMap<String, RustScalar>) {
    let Some(type_node) = node.child_by_field_name("type") else {
        return;
    };
    let base = node_text(source, type_node).trim();
    let scalar = source_scalar_type(base, "", aliases);
    for index in 0..node.child_count() {
        let Some(child) = node.child(index) else {
            continue;
        };
        if child.kind() == "type_identifier"
            && let Some(scalar) = scalar
        {
            aliases.insert(node_text(source, child).trim().to_owned(), scalar);
        }
    }
}

fn collect_using_aliases(source: &str, aliases: &mut HashMap<String, RustScalar>) {
    for statement in source.split(';') {
        let statement = statement.trim();
        let Some(rest) = statement.strip_prefix("using ") else {
            continue;
        };
        let Some((alias, ty)) = rest.split_once('=') else {
            continue;
        };
        let alias = alias.trim();
        if alias.is_empty() || !alias.chars().all(is_c_identifier_char) {
            continue;
        }

        let ty = ty.trim();
        if ty.contains('*') || ty.contains('&') {
            continue;
        }

        let Some(scalar) = source_scalar_type(ty, "", aliases) else {
            continue;
        };
        aliases.insert(alias.to_owned(), scalar);
    }
}

fn declaration_type_tokens(declaration: &str, name: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in declaration.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            current.push(ch);
            continue;
        }

        push_type_token(&mut tokens, &mut current, name);
    }

    push_type_token(&mut tokens, &mut current, name);
    tokens
}

fn push_type_token(tokens: &mut Vec<String>, current: &mut String, name: &str) {
    if current.is_empty() {
        return;
    }
    if current != name && !is_ignored_type_token(current) {
        tokens.push(mem::take(current));
    } else {
        current.clear();
    }
}

fn is_ignored_type_token(token: &str) -> bool {
    matches!(
        token,
        "const"
            | "volatile"
            | "restrict"
            | "__restrict"
            | "__restrict__"
            | "static"
            | "register"
            | "extern"
            | "__global__"
            | "__device__"
            | "__host__"
            | "__forceinline__"
            | "inline"
    )
}

fn find_descendant_kind<'a>(node: Node<'a>, kind: &str) -> Option<Node<'a>> {
    if node.kind() == kind {
        return Some(node);
    }

    for index in 0..node.child_count() {
        if let Some(child) = node.child(index)
            && let Some(found) = find_descendant_kind(child, kind)
        {
            return Some(found);
        }
    }
    None
}

fn declarator_name(node: Node<'_>) -> Option<Node<'_>> {
    if matches!(node.kind(), "identifier" | "field_identifier") {
        return Some(node);
    }

    if let Some(declarator) = node.child_by_field_name("declarator")
        && let Some(name) = declarator_name(declarator)
    {
        return Some(name);
    }

    let mut last = None;
    for index in 0..node.child_count() {
        if let Some(child) = node.child(index)
            && child.is_named()
            && let Some(name) = declarator_name(child)
        {
            last = Some(name);
        }
    }
    last
}

fn declarator_has_kind(node: Node<'_>, kind: &str) -> bool {
    if node.kind() == kind {
        return true;
    }

    for index in 0..node.child_count() {
        if let Some(child) = node.child(index)
            && declarator_has_kind(child, kind)
        {
            return true;
        }
    }
    false
}

fn contains_identifier(source: &str, needle: &str) -> bool {
    let mut start = 0usize;
    while let Some(relative) = source[start..].find(needle) {
        let index = start + relative;
        let end = index + needle.len();
        let before = source[..index]
            .chars()
            .next_back()
            .is_none_or(|ch| !is_c_identifier_char(ch));
        let after = source[end..]
            .chars()
            .next()
            .is_none_or(|ch| !is_c_identifier_char(ch));
        if before && after {
            return true;
        }

        start = end;
    }
    false
}

pub(crate) fn is_c_identifier_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}

fn node_text<'a>(source: &'a str, node: Node<'_>) -> &'a str {
    &source[node.start_byte()..node.end_byte()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_kernel_parameters() {
        let source = r#"
            __device__ void helper(float* value) {}

            extern "C" __global__ void scale_add(
                const float* input,
                float* output,
                float alpha,
                int len,
                size_t count,
                unsigned long long seed,
                const __half* halves,
                unsigned int flags,
                const float* const* batches
            ) {
                int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
                if (i < len) {
                    output[i] = input[i] * alpha + 1.0f;
                }
            }
        "#;
        let kernels = parse_source_kernels(source, Span::call_site()).unwrap();
        assert_eq!(kernels.len(), 1);

        let kernel = &kernels[0];
        assert_eq!(kernel.name, "scale_add");
        assert_eq!(kernel.params.len(), 9);
        assert_pointer(&kernel.params[0], &[false], Some(RustScalar::F32));
        assert_pointer(&kernel.params[1], &[true], Some(RustScalar::F32));
        assert_scalar(&kernel.params[2], Some(RustScalar::F32));
        assert_scalar(&kernel.params[3], Some(RustScalar::I32));
        assert_scalar(&kernel.params[4], Some(RustScalar::USize));
        assert_scalar(&kernel.params[5], Some(RustScalar::U64));
        assert_pointer(&kernel.params[6], &[false], Some(RustScalar::F16));
        assert_scalar(&kernel.params[7], Some(RustScalar::U32));
        assert_pointer(&kernel.params[8], &[false, false], Some(RustScalar::F32));
    }

    #[test]
    fn test_handle_array_parameters() {
        let source = r#"
            extern "C" __global__ void arrays(
                const float input[],
                unsigned int output[static 4],
                void* scratch
            ) {}
        "#;
        let kernels = parse_source_kernels(source, Span::call_site()).unwrap();
        let params = &kernels[0].params;
        assert_pointer(&params[0], &[false], Some(RustScalar::F32));
        assert_pointer(&params[1], &[true], Some(RustScalar::U32));
        assert_pointer(&params[2], &[true], None);
    }

    #[test]
    fn test_resolve_scalar_alias_parameters() {
        let source = r#"
            using index_t = int;
            using count_t = index_t;
            typedef unsigned int flags_t;
            typedef unsigned long long seed_t;

            extern "C" __global__ void aliases(
                index_t index,
                count_t count,
                flags_t flags,
                seed_t seed
            ) {}
        "#;
        let kernels = parse_source_kernels(source, Span::call_site()).unwrap();
        let params = &kernels[0].params;
        assert_scalar(&params[0], Some(RustScalar::I32));
        assert_scalar(&params[1], Some(RustScalar::I32));
        assert_scalar(&params[2], Some(RustScalar::U32));
        assert_scalar(&params[3], Some(RustScalar::U64));
    }

    fn assert_scalar(param: &SourceParam, expected: Option<RustScalar>) {
        assert_eq!(param.kind, SourceParamKind::Scalar(expected));
    }

    fn assert_pointer(param: &SourceParam, levels: &[bool], element: Option<RustScalar>) {
        assert_eq!(
            param.kind,
            SourceParamKind::Pointer {
                levels: levels.to_vec(),
                element,
            }
        );
    }
}

// Paged attention for what the tensor-core kernels refuse: other floating types, other head dimensions, strided or misaligned operands, and targets before `sm_80`.
// One warp per row of queries reads any layout, finds its sequence, and goes over its keys once, its lanes splitting each dot product and each value, with a running largest score that keeps the exponentials in range.
// A decoding step has few rows and many keys, as Whisper's cross-attention over 1500 encoder outputs does, so the lanes share a row rather than each taking one.
// Key counts are run-time values, so this is one kernel rather than a decomposition, which would have fixed shapes.

use std::f64::consts::LOG2_E;

use singe_codegen::emit::cuda_c::spelling::store_expression;

use crate::{
    backend::{
        cuda::kernel_plan::source::{
            KernelInput, KernelPlan,
            paged_attention::{PagedSequence, report_index_error},
            writer::KernelWriter,
        },
        kernel_lowering::element_type,
        registry::{Operation, ScaledDotProductAttentionAttributes},
    },
    data_type::DataType,
    error::{Error, Result},
    layout::Layout,
    utility::emit,
    value::ValueSpec,
};

// The lanes that share a row.
const WARP: usize = 32;

// One part of a score: the dot product of a query, by `query` in the plan's inputs, with a cached key, by `key`.
#[derive(Debug, Clone, Copy)]
pub struct ScoreTerm {
    pub query: usize,
    pub key: usize,
    pub width: usize,
}

// What differs between grouped-query and latent attention.
#[derive(Debug, Clone)]
pub struct GeneralPaged<'a> {
    pub label: &'static str,
    pub operation: Operation,
    pub inputs: &'a [KernelInput],
    pub attention: ScaledDotProductAttentionAttributes,
    pub terms: &'a [ScoreTerm],
    // The input the attended values come from, and their width.
    pub value: usize,
    pub value_width: usize,
    // Query heads per key head, for a cache with a head axis.
    pub groups: Option<usize>,
    // The inputs after the caches: query offsets, page offsets, pages, and last page lengths.
    pub indices: usize,
    pub tokens: usize,
    pub heads: usize,
    pub page_size: usize,
    pub cache_pages: usize,
    pub sequences: usize,
    pub output: ValueSpec,
    pub lse: ValueSpec,
    pub default_scale: f32,
}

// Addresses one element of `layout`, from `indices` in axis order, relative to its first element, where an input's pointer starts.
fn element_at(layout: &Layout, indices: &[&str]) -> String {
    let mut address = "0LL".to_owned();
    for (index, stride) in indices.iter().zip(layout.strides()) {
        emit!(address, " + static_cast<long long>({index}) * {stride}LL");
    }
    address
}

// Plans the kernel of `plan`.
pub fn general_paged(plan: &GeneralPaged<'_>) -> Result<KernelPlan> {
    let GeneralPaged {
        inputs,
        attention,
        terms,
        value,
        value_width,
        indices,
        tokens,
        heads,
        page_size,
        cache_pages,
        sequences,
        ..
    } = *plan;
    let data_type = inputs[0].data_type;
    if !matches!(
        data_type,
        DataType::F16 | DataType::Bf16 | DataType::F32 | DataType::F64
    ) || inputs[..indices]
        .iter()
        .any(|input| input.data_type != data_type)
    {
        return Err(Error::unsupported_data_type("cuda paged attention"));
    }

    let wide = data_type == DataType::F64;
    let (accumulator, exp, log, tanh, suffix) = if wide {
        ("double", "exp", "log", "tanh", "")
    } else {
        ("float", "expf", "logf", "tanhf", "f")
    };
    let mut writer = KernelWriter::new();
    let names = inputs
        .iter()
        .enumerate()
        .map(|(index, input)| writer.input(input.data_type, &format!("operand_{index}")))
        .collect::<Result<Vec<_>>>()?;
    let output = writer.output(data_type, "output")?;
    let lse = writer.output(DataType::F32, "lse")?;
    writer.status();
    let rows = tokens * heads;
    let launch = writer.elements(rows * WARP);
    // Each lane holds the value dimensions `lane`, `lane + 32`, and so on.
    let per_lane = value_width.div_ceil(WARP);
    let load = |value: String| match data_type {
        DataType::F16 => format!("load_half({value})"),
        DataType::Bf16 => format!("load_bfloat({value})"),
        _ => format!("static_cast<{accumulator}>({value})"),
    };
    let read = |input: usize, indices: &[&str]| {
        load(format!(
            "{}[{}]",
            names[input],
            element_at(&inputs[input].layout, indices)
        ))
    };
    let index_value = |input: usize, position: &str| {
        format!(
            "static_cast<long long>({}[{}])",
            names[input],
            element_at(&inputs[input].layout, &[position])
        )
    };
    let key_indices = |dimension: &'static str| -> Vec<&'static str> {
        if plan.groups.is_some() {
            vec!["page", "slot", "key_head", dimension]
        } else {
            vec!["page", "slot", dimension]
        }
    };
    let mut score = String::new();
    for term in terms {
        emit!(
            score,
            "for (unsigned long long d = lane; d < {width}ULL; d += 32ULL) score += {query} * {key};",
            width = term.width,
            query = read(term.query, &["token", "head", "d"]),
            key = read(term.key, &key_indices("d")),
        );
    }

    let scale = attention.scale.unwrap_or(plan.default_scale);
    let scaled = attention.softcap.map_or_else(
        || format!("score * {scale:?}{suffix}"),
        |cap| {
            format!(
                "{cap:?}{suffix} * {tanh}(score * {inverse:?}{suffix})",
                inverse = scale / cap
            )
        },
    );
    let causal = attention.causal;
    let first_key = attention.window.map_or_else(
        || "0LL".to_owned(),
        |window| format!("position + 1LL > {window}LL ? position + 1LL - {window}LL : 0LL"),
    );
    let sequence_values = PagedSequence {
        query_offsets: &names[indices],
        page_offsets: &names[indices + 1],
        last_page_lengths: &names[indices + 3],
        tokens,
        page_table: inputs[indices + 2].layout.shape().element_count(),
        page_size,
        rows_per_query: heads,
    }
    .checked_values();
    let read_page = format!(
        "const long long table_entry = page_start + key / {page_size}LL;
const unsigned long long page = static_cast<unsigned long long>({page_value});
if (page >= {cache_pages}ULL) {{ {report} return; }}
const unsigned long long slot = static_cast<unsigned long long>(key) % {page_size}ULL;",
        page_value = index_value(indices + 2, "table_entry"),
        report = report_index_error(),
    );
    let key_head = plan.groups.map_or_else(String::new, |groups| {
        format!("const unsigned long long key_head = head / {groups}ULL;")
    });
    let value_read = read(value, &key_indices("d"));
    let element = element_type(data_type)?;
    let stored = store_expression(element, "result");
    writer.lines(&format!(
        "
const unsigned long long row = index / 32ULL;
const unsigned long long lane = index % 32ULL;
const unsigned long long token = row / {heads}ULL, head = row % {heads}ULL;
{key_head}
{accumulator} values[{per_lane}];
for (unsigned int part = 0; part < {per_lane}U; ++part) values[part] = 0;
{accumulator} largest = -__uint_as_float(0x7f800000U);
{accumulator} total = 0;
for (unsigned int sequence = 0; sequence < {sequences}U; ++sequence) {{
{sequence_values}
if (static_cast<long long>(token) < query_start || static_cast<long long>(token) >= query_end) continue;
// The queries are the sequence's last tokens, so this one sits at this key position.
const long long position = keys - (query_end - query_start) + (static_cast<long long>(token) - query_start);
const long long end = {causal} ? (position + 1LL < keys ? position + 1LL : keys) : keys;
const long long start = {first_key};
for (long long key = start; key < end; ++key) {{
    {read_page}
    {accumulator} score = 0;
    {score}
    // Every lane ends with the whole dot product, so all of them weigh the key alike.
    for (int offset = 16; offset > 0; offset /= 2) score += __shfl_xor_sync(0xffffffffU, score, offset);
    const {accumulator} scaled = {scaled};
    const {accumulator} next = scaled > largest ? scaled : largest;
    const {accumulator} correction = {exp}(largest - next);
    const {accumulator} weight = {exp}(scaled - next);
    total = total * correction + weight;
    for (unsigned int part = 0; part < {per_lane}U; ++part) {{
        const unsigned long long d = lane + 32ULL * part;
        if (d < {value_width}ULL) values[part] = values[part] * correction + weight * {value_read};
    }}
    largest = next;
}}
break;
}}
// A row that sees no key keeps a zero output and a log-sum-exp of negative infinity.
const {accumulator} inverse = total > 0 ? 1 / total : 0;
for (unsigned int part = 0; part < {per_lane}U; ++part) {{
const unsigned long long d = lane + 32ULL * part;
if (d >= {value_width}ULL) break;
const {accumulator} result = values[part] * inverse;
{output}[row * {value_width}ULL + d] = {stored};
}}
if (lane == 0) {lse}[row] = total > 0 ? static_cast<float>((largest + {log}(total)) * {LOG2_E:?}{suffix}) : -__uint_as_float(0x7f800000U);"
    ));
    Ok(writer.finish(
        plan.label,
        launch,
        vec![plan.output.clone(), plan.lse.clone()],
        inputs,
        &[plan.operation],
    ))
}

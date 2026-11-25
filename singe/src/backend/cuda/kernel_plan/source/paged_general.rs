// Paged attention for what the tensor-core kernels refuse: other floating types, other head dimensions, strided or misaligned operands, and targets before `sm_80`.
// One warp per row of queries reads any layout, finds its sequence, and goes over its keys once, with a running largest score that keeps the exponentials in range.
// A decoding step has few rows and many keys, as Whisper's cross-attention over 1500 encoder outputs does, so a block of warps shares a row: each warp takes every `WARPS`-th slice of keys, groups of its lanes each take a key at a time, splitting its dot product and value, and the groups' and then the warps' running sums merge at the end, as FlashInfer's split-KV decoding splits a row's keys.
// Key counts are run-time values, so this is one kernel rather than a decomposition, which would have fixed shapes.

use std::f64::consts::LOG2_E;

use singe_codegen::emit::cuda_c::spelling::store_expression;

use crate::{
    backend::{
        cuda::kernel_plan::source::{
            KernelInput, KernelPlan,
            paging::{PagedSequence, report_index_error},
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

// The lanes of a warp.
const WARP: usize = 32;

// The warps that share a row, each taking every `WARPS`-th slice of its keys.
const WARPS: usize = 8;

const THREADS: u32 = 256;

const _: () = assert!(THREADS as usize == WARPS * WARP);

const LANE_ELEMENTS: usize = 16;

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
    // The input of a bias by distance, `[heads, distances]`, whose column of a query's position less a key's adds to their score: the first for a key after its query, the last for a distance past it.
    pub bias: Option<usize>,
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
    let launch = writer.rows(rows, THREADS);
    // The lanes that share a key, as few as keep each lane's share of a value to about 16 elements, and the keys a warp takes at once.
    let widest = terms
        .iter()
        .map(|term| term.width)
        .chain([value_width])
        .max()
        .unwrap_or(1);
    let lanes = widest
        .div_ceil(LANE_ELEMENTS)
        .next_power_of_two()
        .clamp(1, WARP);
    let groups = WARP / lanes;
    let stride = WARPS * groups;
    // Each lane holds the value dimensions `sub`, `sub + lanes`, and so on.
    let per_lane = value_width.div_ceil(lanes);
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
    // Each lane's share of the query, read once per row rather than once per key.
    let mut query = String::new();
    let mut score = String::new();
    for (index, term) in terms.iter().enumerate() {
        let parts = term.width.div_ceil(lanes);
        emit!(
            query,
            "{accumulator} query_{index}[{parts}];
for (unsigned int part = 0; part < {parts}U; ++part) {{ const unsigned long long d = sub + {lanes}ULL * part; query_{index}[part] = d < {width}ULL ? {read} : 0; }}
",
            width = term.width,
            read = read(term.query, &["token", "head", "d"]),
        );
        emit!(
            score,
            "if (live) for (unsigned int part = 0; part < {parts}U; ++part) {{ const unsigned long long d = sub + {lanes}ULL * part; if (d < {width}ULL) score += query_{index}[part] * {key}; }}",
            width = term.width,
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
    let bias = plan
        .bias
        .map(|bias| -> Result<String> {
            let &[_, distances] = inputs[bias].layout.dimensions() else {
                return Err(Error::invalid_argument("cuda paged attention bias"));
            };
            Ok(format!(
                "const long long distance = position - key;
const long long column = distance < 0LL ? 0LL : (distance < {distances}LL ? distance : {distances}LL - 1LL);
scaled += {read};",
                read = read(bias, &["head", "column"]),
            ))
        })
        .transpose()?
        .unwrap_or_default();
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
        "const long long table_entry = page_start + (live ? key : start) / {page_size}LL;
const unsigned long long page = static_cast<unsigned long long>({page_value});
// A page past the cache reports and takes no part, since returning would leave the other groups' shuffles short.
if (live && page >= {cache_pages}ULL) {{ {report} live = false; }}
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
const unsigned int warp = lane / 32U;
const unsigned long long sub = (lane % 32U) % {lanes}ULL, group = (lane % 32U) / {lanes}ULL;
const unsigned long long token = row / {heads}ULL, head = row % {heads}ULL;
{key_head}
{accumulator} values[{per_lane}];
for (unsigned int part = 0; part < {per_lane}U; ++part) values[part] = 0;
const {accumulator} lowest = -__uint_as_float(0x7f800000U);
{accumulator} largest = lowest;
{accumulator} total = 0;
for (unsigned int sequence = 0; sequence < {sequences}U; ++sequence) {{
{sequence_values}
if (static_cast<long long>(token) < query_start || static_cast<long long>(token) >= query_end) continue;
// The queries are the sequence's last tokens, so this one sits at this key position.
const long long position = keys - (query_end - query_start) + (static_cast<long long>(token) - query_start);
const long long end = {causal} ? (position + 1LL < keys ? position + 1LL : keys) : keys;
const long long start = {first_key};
{query}
for (long long base = start + static_cast<long long>(warp) * {groups}LL; base < end; base += {stride}LL) {{
    const long long key = base + static_cast<long long>(group);
    bool live = key < end;
    {read_page}
    {accumulator} score = 0;
    {score}
    // The lanes of a group end with its key's whole dot product.
    for (int offset = {lanes} / 2; offset > 0; offset /= 2) score += __shfl_xor_sync(0xffffffffU, score, offset);
    if (live) {{
        {accumulator} scaled = {scaled};
        {bias}
        const {accumulator} next = scaled > largest ? scaled : largest;
        const {accumulator} correction = {exp}(largest - next);
        const {accumulator} weight = {exp}(scaled - next);
        total = total * correction + weight;
        for (unsigned int part = 0; part < {per_lane}U; ++part) {{
            const unsigned long long d = sub + {lanes}ULL * part;
            if (d < {value_width}ULL) values[part] = values[part] * correction + weight * {value_read};
        }}
        largest = next;
    }}
}}
break;
}}
// The groups merge their running sums, each rescaled to the larger of two largest scores; a group that saw no key adds nothing.
for (int offset = {lanes}; offset < 32; offset *= 2) {{
    const {accumulator} other_largest = __shfl_xor_sync(0xffffffffU, largest, offset);
    const {accumulator} other_total = __shfl_xor_sync(0xffffffffU, total, offset);
    const {accumulator} next = other_largest > largest ? other_largest : largest;
    const {accumulator} mine = largest == lowest ? 0 : {exp}(largest - next);
    const {accumulator} theirs = other_largest == lowest ? 0 : {exp}(other_largest - next);
    total = total * mine + other_total * theirs;
    for (unsigned int part = 0; part < {per_lane}U; ++part) {{
        const {accumulator} other = __shfl_xor_sync(0xffffffffU, values[part], offset);
        values[part] = values[part] * mine + other * theirs;
    }}
    largest = next;
}}
// The warps merge through shared memory: the lanes of each warp's first group hold its sums, which the first warp rescales and adds.
__shared__ {accumulator} warp_largest[{WARPS}];
__shared__ {accumulator} warp_total[{WARPS}];
__shared__ {accumulator} warp_values[{WARPS}][{value_width}];
if (group == 0) {{
for (unsigned int part = 0; part < {per_lane}U; ++part) {{
const unsigned long long d = sub + {lanes}ULL * part;
if (d < {value_width}ULL) warp_values[warp][d] = values[part];
}}
if (sub == 0) {{ warp_largest[warp] = largest; warp_total[warp] = total; }}
}}
__syncthreads();
if (warp == 0) {{
{accumulator} best = lowest;
for (unsigned int other = 0; other < {WARPS}U; ++other) best = warp_largest[other] > best ? warp_largest[other] : best;
{accumulator} scales[{WARPS}];
{accumulator} sum = 0;
for (unsigned int other = 0; other < {WARPS}U; ++other) {{
scales[other] = warp_largest[other] == lowest ? 0 : {exp}(warp_largest[other] - best);
sum += warp_total[other] * scales[other];
}}
// A row that sees no key keeps a zero output and a log-sum-exp of negative infinity.
const {accumulator} inverse = sum > 0 ? 1 / sum : 0;
for (unsigned long long d = lane; d < {value_width}ULL; d += 32ULL) {{
{accumulator} weighted = 0;
for (unsigned int other = 0; other < {WARPS}U; ++other) weighted += warp_values[other][d] * scales[other];
const {accumulator} result = weighted * inverse;
{output}[row * {value_width}ULL + d] = {stored};
}}
if (lane == 0) {lse}[row] = sum > 0 ? static_cast<float>((best + {log}(sum)) * {LOG2_E:?}{suffix}) : lowest;
}}"
    ));
    Ok(writer.finish(
        plan.label,
        launch,
        vec![plan.output.clone(), plan.lse.clone()],
        inputs,
        &[plan.operation],
    ))
}

// The causal convolution and the gated delta rule over a step's sequences, and the states they leave.
// A sequence's rows are `offsets[s]` onwards, its first `real[s]` real, and it starts from a zero state when its first real row is at position zero.

use std::fmt::Write as _;

use singe_codegen::emit::cuda_c::spelling::{load_expression, store_expression};

use crate::{
    backend::{
        cuda::kernel_plan::source::{KernelInput, KernelPlan, writer::KernelWriter},
        kernel_lowering::element_type,
        registry::Operation,
    },
    data_type::DataType,
    error::{Error, Result},
    utility::to_usize,
    value::ValueSpec,
};

// The threads of one block of the delta rule: one warp, so a head's columns spread over several blocks.
const WARP: usize = 32;

// How many lanes share one value feature's column of the delta rule's state.
const DELTA_RULE_PARTS: usize = 4;

// The warps of one block of the delta rule's prompt form, which share each chunk's work on its keys.
const CHUNK_WARPS: usize = 8;

// The most rows a chunk of the delta rule's prompt form holds.
const DELTA_RULE_CHUNK: usize = 16;

// The shared bytes a chunk's keys, queries, and reads may take.
const CHUNK_SHARED_BYTES: usize = 40 << 10;

// The type a kernel computes `data_type` values in.
const fn accumulator(data_type: DataType) -> &'static str {
    if matches!(data_type, DataType::F64) {
        "double"
    } else {
        "float"
    }
}

fn extents<const N: usize>(input: &KernelInput, label: &'static str) -> Result<[usize; N]> {
    let dimensions = input.layout.shape().dimensions();
    let dimensions: &[i64; N] = dimensions
        .try_into()
        .map_err(|_| Error::invalid_argument(label))?;
    let mut extents = [0; N];
    for (extent, &dimension) in extents.iter_mut().zip(dimensions) {
        *extent = to_usize(dimension, label)?;
    }
    Ok(extents)
}

impl KernelPlan {
    // Plans one of the causal convolution, its state, the gated delta rule, and its state, on contiguous inputs.
    pub(crate) fn state_recurrence(operation: Operation, inputs: &[KernelInput]) -> Result<Self> {
        if inputs.iter().any(|input| !input.layout.is_contiguous()) {
            return Err(Error::unsupported_operation(
                "cuda state recurrence of strided inputs",
            ));
        }

        match (operation, inputs) {
            (Operation::CausalConv1d, [input, weight, state, offsets, real, positions]) => {
                Self::causal_conv1d(
                    inputs,
                    input,
                    Some(weight),
                    state,
                    [offsets, real, positions],
                )
            }
            (Operation::CausalConv1dState, [input, state, offsets, real, positions]) => {
                Self::causal_conv1d(inputs, input, None, state, [offsets, real, positions])
            }
            (
                Operation::GatedDeltaRule
                | Operation::GatedDeltaRuleOutput
                | Operation::GatedDeltaRuleState,
                [_, _, _, _, _, _, _, _, _],
            ) => Self::gated_delta_rule(inputs),
            _ => Err(Error::missing_input("cuda state recurrence operands")),
        }
    }

    // One thread per output element, which reads its window from the sequence's rows and, before them, its state.
    fn causal_conv1d(
        inputs: &[KernelInput],
        input: &KernelInput,
        weight: Option<&KernelInput>,
        state: &KernelInput,
        spans: [&KernelInput; 3],
    ) -> Result<Self> {
        let [tokens, channels] = extents::<2>(input, "cuda causal convolution input")?;
        let [sequences, _, history] = extents::<3>(state, "cuda causal convolution state")?;
        let data_type = input.data_type;
        let element = element_type(data_type)?;
        let ty = accumulator(data_type);
        let mut writer = KernelWriter::new();
        let rows = writer.input(data_type, "rows")?;
        let taps = weight
            .map(|weight| writer.input(weight.data_type, "taps"))
            .transpose()?;
        let held = writer.input(state.data_type, "held")?;
        let [offsets, real, positions] = [
            ("offsets", spans[0]),
            ("real", spans[1]),
            ("positions", spans[2]),
        ]
        .map(|(name, span)| writer.input(span.data_type, name));
        let (offsets, real, positions) = (offsets?, real?, positions?);
        let output = writer.output(data_type, "output")?;
        let zero = store_expression(element, &format!("static_cast<{ty}>(0)"));
        let (launch, label, spec) = if let (Some(taps), Some(weight)) = (taps, weight) {
            let launch = writer.elements(tokens * channels);
            let tap = load_expression(
                element_type(weight.data_type)?,
                &format!("{taps}[channel * {width}ULL + j]", width = history + 1),
            );
            let row = load_expression(
                element,
                &format!("{rows}[(start + u) * {channels}ULL + channel]"),
            );
            let kept = load_expression(
                element,
                &format!(
                    "{held}[(sequence * {channels}ULL + channel) * {history}ULL + {history} + u]"
                ),
            );
            let sum = store_expression(element, "sum");
            writer.lines(&format!(
                "
const long long token = static_cast<long long>(index / {channels}ULL);
const unsigned long long channel = index % {channels}ULL;
long long sequence = -1;
for (unsigned long long s = 0; s < {sequences}ULL; ++s) {{
    const long long first = static_cast<long long>({offsets}[s]);
    if (token >= first && token < first + static_cast<long long>({real}[s])) {{
        sequence = static_cast<long long>(s);
        break;
    }}
}}
if (sequence < 0) {{
    {output}[index] = {zero};
    return;
}}
const long long start = static_cast<long long>({offsets}[sequence]);
const bool fresh = static_cast<long long>({positions}[start]) == 0;
{ty} sum = 0;
for (long long j = 0; j <= {history}LL; ++j) {{
    const long long u = token - start + j - {history}LL;
    const {ty} value = u >= 0 ? static_cast<{ty}>({row}) : (fresh ? static_cast<{ty}>(0) : static_cast<{ty}>({kept}));
    sum += static_cast<{ty}>({tap}) * value;
}}
{output}[index] = {sum};"
            ));
            (
                launch,
                "causal_conv1d",
                ValueSpec::new(data_type, input.layout.shape().clone()),
            )
        } else {
            // A sequence's state after `n` real rows is the last `history` of its state followed by those rows.
            let launch = writer.elements(sequences * channels * history);
            writer.lines(&format!(
                "
const unsigned long long sequence = index / ({channels}ULL * {history}ULL);
const unsigned long long channel = (index / {history}ULL) % {channels}ULL;
const long long place = static_cast<long long>(index % {history}ULL);
const long long real_rows = static_cast<long long>({real}[sequence]);
if (real_rows == 0) {{
    {output}[index] = {held}[index];
    return;
}}
const long long start = static_cast<long long>({offsets}[sequence]);
const long long w = real_rows + place;
if (w >= {history}LL) {{
    {output}[index] = {rows}[(start + w - {history}LL) * {channels}ULL + channel];
}} else if (static_cast<long long>({positions}[start]) == 0) {{
    {output}[index] = {zero};
}} else {{
    {output}[index] = {held}[(sequence * {channels}ULL + channel) * {history}ULL + w];
}}"
            ));
            (
                launch,
                "causal_conv1d_state",
                ValueSpec::new(state.data_type, state.layout.shape().clone()),
            )
        };
        let operation = if weight.is_some() {
            Operation::CausalConv1d
        } else {
            Operation::CausalConv1dState
        };
        Ok(writer.finish(label, launch, vec![spec], inputs, &[operation]))
    }

    // One block per sequence, head, and slice of value features, writing the output and the state the step leaves.
    // Each value feature's column of the state, `Dk` values, is split across `parts` lanes, which sum what they read with shuffles.
    // A step of single rows runs one warp per block, token by token; a step that reads prompts runs each sequence's rows in chunks, as `transformers`' `torch_chunk_gated_delta_rule` does, with four warps sharing each chunk's work on its keys.
    fn gated_delta_rule(inputs: &[KernelInput]) -> Result<Self> {
        let [query, key, value, decay, strength, state, ..] = inputs else {
            return Err(Error::missing_input("cuda gated delta rule operands"));
        };
        let [tokens, heads, key_dim] = extents::<3>(query, "cuda gated delta rule queries")?;
        let [sequences, _, _, value_dim] = extents::<4>(state, "cuda gated delta rule state")?;
        if sequences == 0 || heads == 0 {
            return Err(Error::unsupported_operation(
                "cuda gated delta rule of no sequences or heads",
            ));
        }

        let data_type = value.data_type;
        let element = element_type(data_type)?;
        let ty = accumulator(data_type);
        let mut writer = KernelWriter::new();
        let names = [
            "query",
            "key",
            "value",
            "decay",
            "strength",
            "held",
            "offsets",
            "real",
            "positions",
        ];
        let mut declared = Vec::with_capacity(names.len());
        for (input, name) in inputs.iter().zip(names) {
            declared.push(writer.input(input.data_type, name)?);
        }

        let [
            queries,
            keys,
            values,
            decays,
            strengths,
            held,
            offsets,
            real,
            positions,
        ] = <[String; 9]>::try_from(declared)
            .map_err(|_| Error::missing_input("cuda gated delta rule operands"))?;
        let output = writer.output(data_type, "output")?;
        let left = writer.output(DataType::F32, "left")?;
        let parts = if key_dim.is_multiple_of(DELTA_RULE_PARTS) {
            DELTA_RULE_PARTS
        } else {
            1
        };
        let prompt = tokens > sequences;
        let threads = if prompt { CHUNK_WARPS * WARP } else { WARP };
        let columns = threads / parts;
        let launch = writer.grid(
            [sequences, heads, value_dim.div_ceil(columns)],
            u32::try_from(threads)
                .map_err(|_| Error::out_of_range("cuda gated delta rule threads"))?,
        )?;
        let owned = key_dim / parts;
        let load = |name: &str, input: &KernelInput, index: &str| -> Result<String> {
            Ok(format!(
                "static_cast<{ty}>({})",
                load_expression(element_type(input.data_type)?, &format!("{name}[{index}]"))
            ))
        };
        let stored = store_expression(element, "out");
        let zero = store_expression(element, &format!("static_cast<{ty}>(0)"));
        let prologue = format!(
            "
const unsigned long long sequence = blockIdx.x, head = blockIdx.y;
const unsigned int part = lane % {parts}U;
const unsigned long long column = static_cast<unsigned long long>(blockIdx.z) * {columns}ULL + lane / {parts}U;
const bool active = column < {value_dim}ULL;
const long long start = static_cast<long long>({offsets}[sequence]);
const long long count = static_cast<long long>({real}[sequence]);
const bool fresh = count > 0 && static_cast<long long>({positions}[start]) == 0;
const unsigned long long base = (sequence * {heads}ULL + head) * {key_dim}ULL * {value_dim}ULL + column;
{ty} column_state[{owned}];
#pragma unroll
for (unsigned int t = 0; t < {owned}U; ++t) {{
    const unsigned long long d = part + t * {parts}ULL;
    column_state[t] = active && !fresh ? static_cast<{ty}>({held}[base + d * {value_dim}ULL]) : static_cast<{ty}>(0);
}}"
        );
        // Rows that are not real are zero: those after a sequence's real rows up to the next sequence's or the step's end, and any before the first sequence.
        let epilogue = format!(
            "
if (active) {{
    #pragma unroll
    for (unsigned int t = 0; t < {owned}U; ++t) {left}[base + (part + t * {parts}ULL) * {value_dim}ULL] = static_cast<float>(column_state[t]);
}}
if (active && part == 0) {{
    const long long end = sequence + 1 == {sequences}ULL ? {tokens}LL : static_cast<long long>({offsets}[sequence + 1]);
    for (long long row = start + count; row < end; ++row) {output}[(static_cast<unsigned long long>(row) * {heads}ULL + head) * {value_dim}ULL + column] = {zero};
    if (sequence == 0) {{
        for (long long row = 0; row < start; ++row) {output}[(static_cast<unsigned long long>(row) * {heads}ULL + head) * {value_dim}ULL + column] = {zero};
    }}
}}"
        );
        let load_key = load(&keys, key, &format!("r * {key_dim}ULL + d"))?;
        let load_query = load(&queries, query, &format!("r * {key_dim}ULL + d"))?;
        let load_value = load(&values, value, &format!("r * {value_dim}ULL + column"))?;
        let load_decay = load(&decays, decay, "r")?;
        let load_strength = load(&strengths, strength, "r")?;
        // Lanes of one column sum their parts with shuffles within groups of `parts` lanes, which every lane then holds.
        let mut gather = String::new();
        let mut width = 1;
        while width < parts {
            let _ = write!(
                gather,
                "\n    read += __shfl_xor_sync(0xffffffffU, read, {width});"
            );
            width *= 2;
        }

        let gather_out = gather.replace("read", "out");
        let body = if prompt {
            let size = if ty == "double" { 8 } else { 4 };
            let chunk = (CHUNK_SHARED_BYTES / (3 * key_dim * size)).clamp(1, DELTA_RULE_CHUNK);
            let gather_sum = gather.replace("read", "sum");
            // Per chunk of `n` rows, padded to `C` with rows that change nothing: `g` sums each row's decay from the chunk's start, `A` is the strictly lower `−β_i (k_i · k_j) e^(g_i − g_j)`, `T` solves `(I − A) T = I` by forward substitution, and `W = T diag(β e^g) K` is what each row reads of the chunk's starting state.
            // Each column then takes `u = T diag(β) v` and `v' = u − W S`, writes `e^(g_i) q_i S + Σ_(j ≤ i) (q_i · k_j) e^(g_i − g_j) v'_j`, and leaves `S e^(g_C) + Σ_i k_i e^(g_C − g_i) v'_i`; every row of a chunk reads its starting state, so none waits for another.
            format!(
                "
__shared__ {ty} chunk_key[{chunk}][{key_dim}];
__shared__ {ty} chunk_query[{chunk}][{key_dim}];
__shared__ {ty} chunk_read[{chunk}][{key_dim}];
__shared__ {ty} solved[{chunk}][{chunk}];
__shared__ {ty} scores[{chunk}][{chunk}];
__shared__ {ty} decayed[{chunk}];
__shared__ {ty} chunk_strength[{chunk}];
__shared__ {ty} risen[{chunk}];
__shared__ {ty} weight[{chunk}];
__shared__ {ty} remaining[{chunk}];{prologue}
for (long long first = start; first < start + count; first += {chunk}LL) {{
    const long long n = start + count - first < {chunk}LL ? start + count - first : {chunk}LL;
    for (unsigned int e = lane; e < {chunk}U * {key_dim}U; e += blockDim.x) {{
        const unsigned int i = e / {key_dim}U, d = e % {key_dim}U;
        const unsigned long long r = static_cast<unsigned long long>(first + i) * {heads}ULL + head;
        chunk_key[i][d] = i < n ? {load_key} : static_cast<{ty}>(0);
        chunk_query[i][d] = i < n ? {load_query} : static_cast<{ty}>(0);
    }}
    if (lane == 0) {{
        {ty} sum = 0;
        for (unsigned int i = 0; i < {chunk}U; ++i) {{
            const unsigned long long r = static_cast<unsigned long long>(first + i) * {heads}ULL + head;
            sum += i < n ? {load_decay} : static_cast<{ty}>(0);
            decayed[i] = sum;
            chunk_strength[i] = i < n ? {load_strength} : static_cast<{ty}>(0);
            risen[i] = exp(sum);
            weight[i] = chunk_strength[i] * risen[i];
        }}
        for (unsigned int i = 0; i < {chunk}U; ++i) remaining[i] = exp(sum - decayed[i]);
    }}
    __syncthreads();
    for (unsigned int e = lane; e < {chunk}U * {chunk}U; e += blockDim.x) {{
        const unsigned int i = e / {chunk}U, j = e % {chunk}U;
        {ty} kk = 0, qk = 0;
        if (j <= i) {{
            for (unsigned int d = 0; d < {key_dim}U; ++d) {{
                kk += chunk_key[i][d] * chunk_key[j][d];
                qk += chunk_query[i][d] * chunk_key[j][d];
            }}
        }}
        const {ty} fall = j <= i ? exp(decayed[i] - decayed[j]) : static_cast<{ty}>(0);
        solved[i][j] = j < i ? -chunk_strength[i] * kk * fall : static_cast<{ty}>(0);
        scores[i][j] = qk * fall;
    }}
    __syncthreads();
    for (unsigned int i = 1; i < {chunk}U; ++i) {{
        {ty} entry = 0;
        if (lane < i) {{
            entry = solved[i][lane];
            for (unsigned int m = lane + 1; m < i; ++m) entry += solved[i][m] * solved[m][lane];
        }}
        __syncthreads();
        if (lane < i) solved[i][lane] = entry;
        __syncthreads();
    }}
    for (unsigned int e = lane; e < {chunk}U * {key_dim}U; e += blockDim.x) {{
        const unsigned int i = e / {key_dim}U, d = e % {key_dim}U;
        {ty} sum = weight[i] * chunk_key[i][d];
        for (unsigned int j = 0; j < i; ++j) sum += solved[i][j] * weight[j] * chunk_key[j][d];
        chunk_read[i][d] = sum;
    }}
    __syncthreads();
    {ty} weighted[{chunk}], fresh_values[{chunk}];
    #pragma unroll
    for (unsigned int i = 0; i < {chunk}U; ++i) {{
        const unsigned long long r = static_cast<unsigned long long>(first + i) * {heads}ULL + head;
        weighted[i] = i < n && active ? chunk_strength[i] * {load_value} : static_cast<{ty}>(0);
    }}
    #pragma unroll
    for (unsigned int i = 0; i < {chunk}U; ++i) {{
        {ty} u = weighted[i];
        #pragma unroll
        for (unsigned int j = 0; j < i; ++j) u += solved[i][j] * weighted[j];
        {ty} sum = 0;
        #pragma unroll
        for (unsigned int t = 0; t < {owned}U; ++t) sum += chunk_read[i][part + t * {parts}U] * column_state[t];{gather_sum}
        fresh_values[i] = u - sum;
    }}
    #pragma unroll
    for (unsigned int i = 0; i < {chunk}U; ++i) {{
        {ty} sum = 0;
        #pragma unroll
        for (unsigned int t = 0; t < {owned}U; ++t) sum += chunk_query[i][part + t * {parts}U] * column_state[t];{gather_sum}
        {ty} out = risen[i] * sum;
        #pragma unroll
        for (unsigned int j = 0; j <= i; ++j) out += scores[i][j] * fresh_values[j];
        const unsigned long long r = static_cast<unsigned long long>(first + i) * {heads}ULL + head;
        if (i < n && active && part == 0) {output}[r * {value_dim}ULL + column] = {stored};
    }}
    #pragma unroll
    for (unsigned int i = 0; i < {chunk}U; ++i) fresh_values[i] *= remaining[i];
    const {ty} last = risen[{chunk} - 1];
    #pragma unroll
    for (unsigned int t = 0; t < {owned}U; ++t) {{
        {ty} entry = column_state[t] * last;
        #pragma unroll
        for (unsigned int i = 0; i < {chunk}U; ++i) entry += chunk_key[i][part + t * {parts}U] * fresh_values[i];
        column_state[t] = entry;
    }}
    __syncthreads();
}}{epilogue}"
            )
        } else {
            format!(
                "
__shared__ {ty} shared_key[{key_dim}];
__shared__ {ty} shared_query[{key_dim}];{prologue}
for (long long row = start; row < start + count; ++row) {{
    const unsigned long long r = static_cast<unsigned long long>(row) * {heads}ULL + head;
    for (unsigned int d = lane; d < {key_dim}U; d += blockDim.x) {{
        shared_key[d] = {load_key};
        shared_query[d] = {load_query};
    }}
    __syncwarp();
    const {ty} factor = exp({load_decay});
    {ty} read = 0;
    #pragma unroll
    for (unsigned int t = 0; t < {owned}U; ++t) {{
        column_state[t] *= factor;
        read += column_state[t] * shared_key[part + t * {parts}U];
    }}{gather}
    const {ty} delta = ((active ? {load_value} : static_cast<{ty}>(0)) - read) * {load_strength};
    {ty} out = 0;
    #pragma unroll
    for (unsigned int t = 0; t < {owned}U; ++t) {{
        column_state[t] += shared_key[part + t * {parts}U] * delta;
        out += column_state[t] * shared_query[part + t * {parts}U];
    }}{gather_out}
    if (active && part == 0) {output}[r * {value_dim}ULL + column] = {stored};
    __syncwarp();
}}{epilogue}"
            )
        };
        writer.lines(&body);
        Ok(writer.finish(
            "gated_delta_rule",
            launch,
            vec![
                ValueSpec::new(data_type, value.layout.shape().clone()),
                ValueSpec::new(DataType::F32, state.layout.shape().clone()),
            ],
            inputs,
            &[Operation::GatedDeltaRule],
        ))
    }
}

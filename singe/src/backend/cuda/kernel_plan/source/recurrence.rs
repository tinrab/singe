// The causal convolution and the gated delta rule over a step's sequences, and the states they leave.
// A sequence's rows are `offsets[s]` onwards, its first `real[s]` real, and it starts from a zero state when its first real row is at position zero.

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

// The most value features one block of the delta rule holds, one per thread.
const DELTA_RULE_THREADS: usize = 128;

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
                Operation::GatedDeltaRule | Operation::GatedDeltaRuleState,
                [_, _, _, _, _, _, _, _, _],
            ) => Self::gated_delta_rule(operation, inputs),
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

    // One block per sequence, head, and slice of value features, each thread holding one feature's column of the state, `Dk` values, through the sequence's rows.
    fn gated_delta_rule(operation: Operation, inputs: &[KernelInput]) -> Result<Self> {
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
        let want_state = operation == Operation::GatedDeltaRuleState;
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
        let output_type = if want_state { DataType::F32 } else { data_type };
        let output = writer.output(output_type, "output")?;
        let threads = value_dim.clamp(1, DELTA_RULE_THREADS).next_multiple_of(32);
        let launch = writer.grid(
            [sequences, heads, value_dim.div_ceil(threads)],
            u32::try_from(threads)
                .map_err(|_| Error::out_of_range("cuda gated delta rule threads"))?,
        )?;
        let load = |name: &str, input: &KernelInput, index: &str| -> Result<String> {
            Ok(format!(
                "static_cast<{ty}>({})",
                load_expression(element_type(input.data_type)?, &format!("{name}[{index}]"))
            ))
        };
        let load_key = load(&keys, key, &format!("r * {key_dim}ULL + d"))?;
        let load_query = load(&queries, query, &format!("r * {key_dim}ULL + d"))?;
        let load_value = load(&values, value, &format!("r * {value_dim}ULL + column"))?;
        let load_decay = load(&decays, decay, "r")?;
        let load_strength = load(&strengths, strength, "r")?;
        let stored = store_expression(element, "out");
        let zero = store_expression(element, &format!("static_cast<{ty}>(0)"));
        let finish = if want_state {
            format!(
                "
if (active) {{
    #pragma unroll
    for (unsigned int d = 0; d < {key_dim}U; ++d) {output}[base + d * {value_dim}ULL] = static_cast<float>(column_state[d]);
}}"
            )
        } else {
            // Rows that are not real are zero: those after a sequence's real rows up to the next sequence's or the step's end, and any before the first sequence.
            format!(
                "
if (active) {{
    const long long end = sequence + 1 == {sequences}ULL ? {tokens}LL : static_cast<long long>({offsets}[sequence + 1]);
    for (long long row = start + count; row < end; ++row) {output}[(static_cast<unsigned long long>(row) * {heads}ULL + head) * {value_dim}ULL + column] = {zero};
    if (sequence == 0) {{
        for (long long row = 0; row < start; ++row) {output}[(static_cast<unsigned long long>(row) * {heads}ULL + head) * {value_dim}ULL + column] = {zero};
    }}
}}"
            )
        };
        let write_output = if want_state {
            String::new()
        } else {
            format!("if (active) {output}[r * {value_dim}ULL + column] = {stored};")
        };
        writer.lines(&format!(
            "
__shared__ {ty} shared_key[{key_dim}];
__shared__ {ty} shared_query[{key_dim}];
const unsigned long long sequence = blockIdx.x, head = blockIdx.y;
const unsigned long long column = static_cast<unsigned long long>(blockIdx.z) * blockDim.x + lane;
const bool active = column < {value_dim}ULL;
const long long start = static_cast<long long>({offsets}[sequence]);
const long long count = static_cast<long long>({real}[sequence]);
const bool fresh = count > 0 && static_cast<long long>({positions}[start]) == 0;
const unsigned long long base = (sequence * {heads}ULL + head) * {key_dim}ULL * {value_dim}ULL + column;
{ty} column_state[{key_dim}];
#pragma unroll
for (unsigned int d = 0; d < {key_dim}U; ++d) {{
    column_state[d] = active && !fresh ? static_cast<{ty}>({held}[base + d * {value_dim}ULL]) : static_cast<{ty}>(0);
}}
for (long long row = start; row < start + count; ++row) {{
    const unsigned long long r = static_cast<unsigned long long>(row) * {heads}ULL + head;
    for (unsigned int d = lane; d < {key_dim}U; d += blockDim.x) {{
        shared_key[d] = {load_key};
        shared_query[d] = {load_query};
    }}
    __syncthreads();
    const {ty} factor = exp({load_decay});
    {ty} read = 0;
    #pragma unroll
    for (unsigned int d = 0; d < {key_dim}U; ++d) {{
        column_state[d] *= factor;
        read += column_state[d] * shared_key[d];
    }}
    const {ty} delta = ((active ? {load_value} : static_cast<{ty}>(0)) - read) * {load_strength};
    {ty} out = 0;
    #pragma unroll
    for (unsigned int d = 0; d < {key_dim}U; ++d) {{
        column_state[d] += shared_key[d] * delta;
        out += column_state[d] * shared_query[d];
    }}
    {write_output}
    __syncthreads();
}}{finish}"
        ));
        let shape = if want_state {
            state.layout.shape().clone()
        } else {
            value.layout.shape().clone()
        };
        let label = if want_state {
            "gated_delta_rule_state"
        } else {
            "gated_delta_rule"
        };
        Ok(writer.finish(
            label,
            launch,
            vec![ValueSpec::new(output_type, shape)],
            inputs,
            &[operation],
        ))
    }
}

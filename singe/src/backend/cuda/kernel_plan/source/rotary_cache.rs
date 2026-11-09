// Rotates queries and keys and writes keys and values to a paged cache as one kernel.
// It follows FlashInfer's `cake_fused_qk_rmsnorm_rope_append_paged_kv_cache` and vLLM's `csrc/fused_qknorm_rope_cache_kernel.cu`, both Apache-2.0: blocks run over one flat list of a step's heads, each head taken by a few threads.
// They read one projected `qkv` tensor, where our graphs have three, which only changes the addressing.
// Each thread here owns whole rotary pairs rather than runs of a head, so a rotation stays in registers and needs none of vLLM's trips through shared memory.

use singe_codegen::emit::cuda_c::spelling::{compute_type, load_expression, store_expression};

use crate::{
    backend::{
        cuda::kernel_plan::source::{
            KernelInput, KernelPlan, KernelPlanner,
            paged_attention::report_index_error,
            parameters::{KernelParameters, RotaryCacheParameters, parameter_key},
            writer::KernelWriter,
        },
        kernel_lowering::element_type,
        registry::Operation,
    },
    data_type::DataType,
    error::{Error, Result},
    utility::to_usize,
    value::ValueSpec,
};

// The RMS normalization a rotation applies first, where a graph has one.
#[derive(Debug, Clone)]
pub struct RotationNorm {
    pub eps: f32,
    pub weight: Option<KernelInput>,
}

// What the kernel reads and writes: `[T, H, D]` queries, `[T, G, D]` keys and values, `[T, 1, D / 2]` angles, `[T]` pages and slots, and `[P, S, G, D]` caches.
// Latent attention, as vLLM's `fused_rope_unified_mla_kv_cache_update` runs it, has one positional key per token and no heads in its caches: `[T, 1, D]` keys, `[T, L]` values, and `[P, S, D]` and `[P, S, L]` caches.
#[derive(Debug, Clone)]
pub struct RotaryCacheInputs {
    // Absent when attention's kernel rotates its own query.
    pub query: Option<KernelInput>,
    pub key: KernelInput,
    pub value: KernelInput,
    pub cos: KernelInput,
    pub sin: KernelInput,
    pub pages: KernelInput,
    pub slots: KernelInput,
    pub key_cache: KernelInput,
    pub value_cache: KernelInput,
    pub query_norm: Option<RotationNorm>,
    pub key_norm: Option<RotationNorm>,
}

impl RotaryCacheInputs {
    // What the kernel's launch reads, in the order its parameters take them.
    fn operands(&self) -> Vec<KernelInput> {
        let mut operands = self.query.iter().cloned().collect::<Vec<_>>();
        operands.extend([
            self.key.clone(),
            self.value.clone(),
            self.cos.clone(),
            self.sin.clone(),
            self.pages.clone(),
            self.slots.clone(),
        ]);
        for norm in [&self.query_norm, &self.key_norm].into_iter().flatten() {
            operands.extend(norm.weight.clone());
        }
        operands
    }

    fn parameter_key(&self) -> String {
        let eps = |norm: &Option<RotationNorm>| norm.as_ref().map(|norm| norm.eps.to_bits());
        let mut operands = self.operands();
        operands.extend([self.key_cache.clone(), self.value_cache.clone()]);
        parameter_key(
            "rotary_cache_write",
            &operands,
            &format!("{:?}{:?}", eps(&self.query_norm), eps(&self.key_norm)),
        )
    }
}

// The expression of a buffer's index along a head of `[T, H, D]`, at `token` and `head`, relative to its first element, where an input's pointer starts.
fn head_start(input: &KernelInput, head: &str) -> String {
    let strides = input.layout.strides();
    format!("token * {}LL + {head} * {}LL", strides[0], strides[1])
}

struct Rotation<'a> {
    source: &'a str,
    input: &'a KernelInput,
    weight: Option<&'a str>,
    norm: Option<&'a RotationNorm>,
    // The statements that store `first` and `second` at `pair`.
    store: String,
}

// The statements that rotate this thread's pairs of one head, which `head` names.
fn rotation_code(
    rotation: &Rotation<'_>,
    (cos, sin): (&str, &str),
    (cos_input, sin_input): (&KernelInput, &KernelInput),
    data_type: DataType,
    half: usize,
    threads_per_head: u32,
) -> Result<String> {
    let element = element_type(data_type)?;
    let ty = compute_type(element);
    let step = rotation.input.layout.strides()[2];
    let source = rotation.source;
    let load = |offset: &str| {
        let address =
            format!("{source}[from + (static_cast<long long>(pair) + {offset}LL) * {step}LL]");
        format!("static_cast<{ty}>({})", load_expression(element, &address))
    };
    let (load_first, load_second) = (load("0"), load(&half.to_string()));
    let table = |name: &str, input: &KernelInput| -> Result<String> {
        let strides = input.layout.strides();
        let address = format!(
            "{name}[token * {}LL + static_cast<long long>(pair) * {}LL]",
            strides[0], strides[2],
        );
        Ok(format!(
            "static_cast<{ty}>({})",
            load_expression(element_type(input.data_type)?, &address)
        ))
    };
    let (load_cos, load_sin) = (table(cos, cos_input)?, table(sin, sin_input)?);
    let (scale, weighted) = rotation.norm.map_or_else(
        || (String::new(), ""),
        |norm| {
            // Each thread sums the squares of its pairs, and shuffles within the head's threads combine them.
            let bits = if threads_per_head == 32 {
                u32::MAX
            } else {
                (1_u32 << threads_per_head) - 1
            };
            (
                format!(
                    "
{ty} squares = 0;
for (unsigned int pair = first_pair; pair < last_pair; ++pair) {{
    const {ty} first = {load_first};
    const {ty} second = {load_second};
    squares += first * first + second * second;
}}
const unsigned int mask = {bits}U << ((lane % 32U) / {threads_per_head}U * {threads_per_head}U);
for (unsigned int offset = 1U; offset < {threads_per_head}U; offset <<= 1U) {{
    squares += __shfl_xor_sync(mask, squares, offset, 32);
}}
const {ty} scale = rsqrt(squares / static_cast<{ty}>({dim}) + static_cast<{ty}>({eps:?}));",
                    dim = half * 2,
                    eps = f64::from(norm.eps),
                ),
                " * scale",
            )
        },
    );
    let weight = |offset: &str| -> Result<String> {
        let (Some(name), Some(input)) = (
            rotation.weight,
            rotation.norm.and_then(|norm| norm.weight.as_ref()),
        ) else {
            return Ok(String::new());
        };
        let address = format!(
            "{name}[(static_cast<long long>(pair) + {offset}LL) * {}LL]",
            input.layout.strides()[0],
        );
        Ok(format!(
            " * static_cast<{ty}>({})",
            load_expression(element_type(input.data_type)?, &address)
        ))
    };
    let (weight_first, weight_second) = (weight("0")?, weight(&half.to_string())?);
    Ok(format!(
        "{scale}
for (unsigned int pair = first_pair; pair < last_pair; ++pair) {{
    const {ty} a = {load_first}{weighted}{weight_first};
    const {ty} b = {load_second}{weighted}{weight_second};
    const {ty} cosine = {load_cos};
    const {ty} sine = {load_sin};
    const {ty} first = a * cosine - b * sine;
    const {ty} second = b * cosine + a * sine;
{store}
}}",
        store = rotation.store,
    ))
}

impl KernelPlanner {
    // The parameters this kernel may run with, the default first.
    pub(crate) fn rotary_cache_candidates(&self, head_dim: usize) -> Vec<RotaryCacheParameters> {
        let default = RotaryCacheParameters::for_head(head_dim);
        let mut candidates = vec![default];
        for threads_per_head in [8, 16, 32] {
            for threads_per_block in [64, 128, 256] {
                let candidate = RotaryCacheParameters {
                    threads_per_head,
                    threads_per_block,
                    min_blocks: None,
                };
                if candidate != default {
                    candidates.push(candidate);
                }
            }
        }

        candidates.extend(
            self.launch_bound_candidates(default.threads_per_block as usize)
                .into_iter()
                .flatten()
                .map(|bound| RotaryCacheParameters {
                    min_blocks: Some(bound),
                    ..default
                }),
        );
        candidates
    }

    // The key this kernel's parameters are recorded under and what tuning may choose among.
    pub(crate) fn rotary_cache_parameters(
        &self,
        inputs: &RotaryCacheInputs,
    ) -> Result<(String, Vec<KernelParameters>)> {
        let head_dim = to_usize(
            inputs.key.layout.shape().dimensions()[2],
            "cuda rotary cache head",
        )?;
        Ok((
            inputs.parameter_key(),
            self.rotary_cache_candidates(head_dim)
                .into_iter()
                .map(KernelParameters::RotaryCache)
                .collect(),
        ))
    }

    // Plans the kernel, whose outputs are the rotated queries and the two caches with this step written.
    // The caller fills the caches first, since only the selected slots are written.
    pub(crate) fn rotary_cache_write(&self, inputs: &RotaryCacheInputs) -> Result<KernelPlan> {
        let extent = |value: i64| to_usize(value, "cuda rotary cache extent");
        let key_shape = inputs.key.layout.shape().dimensions();
        let cache_shape = inputs.key_cache.layout.shape().dimensions();
        let (tokens, head_dim) = (extent(key_shape[0])?, extent(key_shape[2])?);
        let query_heads = inputs
            .query
            .as_ref()
            .map_or(Ok(0), |query| extent(query.layout.shape().dimensions()[1]))?;
        let cache_heads = extent(key_shape[1])?;
        let value_shape = inputs.value.layout.shape().dimensions();
        let value_dim = extent(value_shape[value_shape.len() - 1])?;
        let (pages, slots) = (extent(cache_shape[0])?, extent(cache_shape[1])?);
        let half = head_dim / 2;
        let default = RotaryCacheParameters::for_head(head_dim);
        let key = inputs.parameter_key();
        let parameters = match self.tuned_parameters(&key) {
            Some(KernelParameters::RotaryCache(parameters))
                if self.rotary_cache_candidates(head_dim).contains(&parameters) =>
            {
                parameters
            }
            Some(_) => return Err(Self::stale_parameters(&key)),
            None => default,
        };
        let data_type = inputs.key.data_type;
        let element = element_type(data_type)?;

        let mut writer = KernelWriter::new();
        writer.min_blocks(parameters.min_blocks);
        let query = inputs
            .query
            .as_ref()
            .map(|_| writer.input(data_type, "query"))
            .transpose()?;
        let key = writer.input(data_type, "key")?;
        let value = writer.input(data_type, "value")?;
        let cos = writer.input(inputs.cos.data_type, "cos")?;
        let sin = writer.input(inputs.sin.data_type, "sin")?;
        let page_index = writer.input(inputs.pages.data_type, "page_index")?;
        let slot_index = writer.input(inputs.slots.data_type, "slot_index")?;
        let mut weight_of = |norm: &Option<RotationNorm>, name: &str| -> Result<Option<String>> {
            norm.as_ref()
                .and_then(|norm| norm.weight.as_ref())
                .map(|weight| writer.input(weight.data_type, name))
                .transpose()
        };
        let query_weight = weight_of(&inputs.query_norm, "query_weight")?;
        let key_weight = weight_of(&inputs.key_norm, "key_weight")?;
        let query_out = inputs
            .query
            .as_ref()
            .map(|_| writer.output(data_type, "query_out"))
            .transpose()?;
        let key_cache = writer.output(data_type, "key_cache")?;
        let value_cache = writer.output(data_type, "value_cache")?;
        writer.status();

        let threads_per_head = parameters.threads_per_head;
        let heads_per_block = parameters.threads_per_block / threads_per_head;
        let total_heads = query_heads + 2 * cache_heads;
        let items = tokens
            .checked_mul(total_heads)
            .ok_or_else(|| Error::out_of_range("cuda rotary cache heads"))?;
        let launch = writer.blocks(
            items.div_ceil(heads_per_block as usize),
            parameters.threads_per_block,
        );
        let pairs_per_thread = half.div_ceil(threads_per_head as usize);
        let values_per_thread = value_dim.div_ceil(threads_per_head as usize);
        let store = |name: &str, base: &str| {
            format!(
                "    {name}[{base} + pair] = {};
    {name}[{base} + {half}LL + pair] = {};",
                store_expression(element, "first"),
                store_expression(element, "second"),
            )
        };
        let query_block = match (&inputs.query, &query, &query_out) {
            (Some(input), Some(source), Some(output)) => {
                let code = rotation_code(
                    &Rotation {
                        source,
                        input,
                        weight: query_weight.as_deref(),
                        norm: inputs.query_norm.as_ref(),
                        store: store(output, "to"),
                    },
                    (&cos, &sin),
                    (&inputs.cos, &inputs.sin),
                    data_type,
                    half,
                    threads_per_head,
                )?;
                format!(
                    "if (local < {query_heads}LL) {{
    const long long head = local;
    const long long from = {query_start};
    const long long to = (token * {query_heads}LL + head) * {head_dim}LL;
    {code}
    return;
}}",
                    query_start = head_start(input, "head"),
                )
            }
            _ => String::new(),
        };
        let key_code = rotation_code(
            &Rotation {
                source: &key,
                input: &inputs.key,
                weight: key_weight.as_deref(),
                norm: inputs.key_norm.as_ref(),
                store: store(&key_cache, "to"),
            },
            (&cos, &sin),
            (&inputs.cos, &inputs.sin),
            data_type,
            half,
            threads_per_head,
        )?;
        let index_at = |name: &str, input: &KernelInput| {
            format!("{name}[token * {}LL]", input.layout.strides()[0])
        };
        let value_strides = inputs.value.layout.strides();
        let value_step = value_strides[value_strides.len() - 1];
        let value_start = if value_shape.len() == 3 {
            head_start(&inputs.value, "head")
        } else {
            format!("token * {}LL", value_strides[0])
        };
        writer.lines(&format!(
            "
const unsigned int head_in_block = lane / {threads_per_head}U;
const unsigned int part = lane % {threads_per_head}U;
const unsigned long long item = static_cast<unsigned long long>(blockIdx.x) * {heads_per_block}ULL + head_in_block;
if (item >= {items}ULL) return;
const long long token = static_cast<long long>(item / {total_heads}ULL);
const long long local = static_cast<long long>(item % {total_heads}ULL);
const unsigned int first_pair = min(part * {pairs_per_thread}U, {half}U);
const unsigned int last_pair = min(first_pair + {pairs_per_thread}U, {half}U);
{query_block}
long long page = {page};
long long slot = {slot};
if (page < 0) page += {pages}LL;
if (slot < 0) slot += {slots}LL;
if (page < 0 || page >= {pages}LL || slot < 0 || slot >= {slots}LL) {{
    {report}
    return;
}}
const long long cell = (page * {slots}LL + slot) * {cache_heads}LL;
if (local < {keys_end}LL) {{
    const long long head = local - {query_heads}LL;
    const long long from = {key_start};
    const long long to = (cell + head) * {head_dim}LL;
    {key_code}
    return;
}}
const long long head = local - {keys_end}LL;
const long long from = {value_start};
const long long to = (cell + head) * {value_dim}LL;
const unsigned int first_value = min(part * {values_per_thread}U, {value_dim}U);
const unsigned int last_value = min(first_value + {values_per_thread}U, {value_dim}U);
for (unsigned int dim = first_value; dim < last_value; ++dim) {{
    {value_cache}[to + dim] = {value}[from + static_cast<long long>(dim) * {value_step}LL];
}}",
            key_start = head_start(&inputs.key, "head"),
            page = index_at(&page_index, &inputs.pages),
            slot = index_at(&slot_index, &inputs.slots),
            report = report_index_error(),
            keys_end = query_heads + cache_heads,
        ));
        let mut outputs = inputs
            .query
            .iter()
            .map(|query| ValueSpec::new(data_type, query.layout.shape().clone()))
            .collect::<Vec<_>>();
        outputs.extend([
            ValueSpec::new(data_type, inputs.key_cache.layout.shape().clone()),
            ValueSpec::new(data_type, inputs.value_cache.layout.shape().clone()),
        ]);
        Ok(writer.finish(
            "rotary_cache_write",
            launch,
            outputs,
            &inputs.operands(),
            &[Operation::RotaryEmbedding, Operation::IndexPut],
        ))
    }
}

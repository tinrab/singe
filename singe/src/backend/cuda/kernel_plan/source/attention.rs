// Fused scaled dot-product attention on tensor cores, after FlashAttention-2 and LeetCUDA's `flash_attn_mma` kernels: each block takes 64 query rows of one head, 16 per warp, and walks the keys 64 at a time, keeping the softmax online so the scores never leave registers.

use std::f32::consts::LOG2_E;

use singe_codegen::emit::cuda_c::spelling::{load_expression, store_expression};

use crate::{
    backend::{
        cuda::{
            codegen::CompileRequirements,
            kernel_plan::source::{
                KernelInput, KernelPlan, KernelPlanner,
                parameters::{AttentionParameters, KernelParameters, has_tcgen05, parameter_key},
                tcgen05_attention::{EngineAttention, engine_attention},
                tensor_core::{
                    COMBINE_ROW_TOTALS, fold_key_tile, key_tile_loop, load_query_fragments,
                    operand_kind, row_accumulators, scaled_score_expression, shared_tiles,
                    tensor_core_helpers, tile_launch_bytes,
                },
                writer::KernelWriter,
            },
        },
        kernel_lowering::element_type,
        registry::{Operation, ScaledDotProductAttentionAttributes},
        spec,
    },
    data_type::DataType,
    error::{Error, Result},
    shape::Shape,
    utility::{to_u32, to_u64, to_usize},
    value::ValueSpec,
};

fn dimensions(input: &KernelInput) -> &[i64] {
    input.layout.shape().dimensions()
}

// Plans attention of operands `fused_attention` took on the tensor engine.
fn engine_plain_attention(
    [query, key, value]: [&KernelInput; 3],
    (batch, heads, queries, dim): (usize, usize, usize, usize),
    (groups, keys): (usize, usize),
    attention: ScaledDotProductAttentionAttributes,
) -> Result<KernelPlan> {
    let data_type = query.data_type;
    let packed = groups * queries;
    let tiles = packed.div_ceil(128);
    let causal = if attention.causal { "true" } else { "false" };
    let windowed = attention.window.map_or_else(
        || "false".to_owned(),
        |window| format!("static_cast<long long>(key) + {window}LL <= position"),
    );
    #[expect(
        clippy::cast_precision_loss,
        reason = "head dimensions are at most 128"
    )]
    let scale = attention.scale.unwrap_or_else(|| 1.0 / (dim as f32).sqrt());
    let score =
        scaled_score_expression(scale, attention.softcap).replace("scores[j][r]", "scores[column]");
    let stored = store_expression(element_type(data_type)?, "value");
    let mut writer = KernelWriter::new();
    let qs = writer.input(data_type, "query_rows")?;
    let ks = writer.input(data_type, "key_rows")?;
    let vs = writer.input(data_type, "value_rows")?;
    let os = writer.output(data_type, "output")?;
    let launch = engine_attention(
        &mut writer,
        &EngineAttention {
            data_type,
            dim,
            blocks: batch * (heads / groups) * tiles,
            setup: &format!(
                "const unsigned long long kv_head = blockIdx.x / {tiles}ULL;
const unsigned int first_row = (blockIdx.x % {tiles}U) * 128U;
const unsigned int packed = {packed}U;
const unsigned int keys = {keys}U;"
            ),
            position: &format!("static_cast<long long>(row % {queries}U)"),
            query_source: &format!(
                "const bool valid = first_row + r < packed;
    const unsigned short* source = valid ? {qs} + (kv_head * {packed}ULL + first_row + r) * {dim}ULL + column : {qs};"
            ),
            key_source: &format!(
                "const bool valid = first_key + r < keys;
    const unsigned short* source = valid ? {ks} + (kv_head * {keys}ULL + first_key + r) * {dim}ULL + column : {ks};"
            ),
            value_source: &format!(
                "const bool valid = first_key + r < keys;
    const unsigned short* source = valid ? {vs} + (kv_head * {keys}ULL + first_key + r) * {dim}ULL + column : {vs};"
            ),
            hidden: &format!(
                "key >= keys || !row_valid || ({causal} && static_cast<long long>(key) > position) || ({windowed})"
            ),
            score: &score,
            store: &format!("{os}[(kv_head * {packed}ULL + row) * {dim}ULL + at] = {stored};"),
            finish: "",
        },
    )?;
    Ok(writer.finish(
        "tcgen05_attention",
        launch,
        vec![ValueSpec::new(data_type, query.layout.shape().clone())],
        &[query.clone(), key.clone(), value.clone()],
        &[Operation::ScaledDotProductAttention],
    ))
}

// Shared rows are padded by one 16-byte chunk, so the rows an `ldmatrix` reads fall in different banks.
pub const PAD: usize = 8;

impl KernelPlanner {
    // The parameters attention of `inputs` may run with, the default first: `key_tile` holds the queries too on their way to registers, and two tiles fit static shared memory.
    pub(crate) fn attention_candidates(&self, dim: usize) -> Vec<AttentionParameters> {
        let default = AttentionParameters::default();
        let mut candidates = vec![default];
        for warps in [4, 2, 8] {
            for key_tile in [64, 32, 96, 128] {
                if key_tile < warps * 16 {
                    continue;
                }

                for split_below_blocks in [64, 0, 256] {
                    candidates.push(AttentionParameters {
                        warps,
                        key_tile,
                        stages: 1,
                        split_below_blocks,
                        min_blocks: None,
                    });
                }

                candidates.push(AttentionParameters {
                    warps,
                    key_tile,
                    stages: 2,
                    ..default
                });
            }
        }

        candidates.extend(
            self.launch_bound_candidates(default.warps * 32)
                .into_iter()
                .flatten()
                .map(|bound| AttentionParameters {
                    min_blocks: Some(bound),
                    ..default
                }),
        );
        candidates.retain(|candidate| candidate.fits(dim + PAD));
        let mut unique = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            if !unique.contains(&candidate) {
                unique.push(candidate);
            }
        }
        unique
    }

    // The parameters a record gives attention with this key, or the default.
    pub(crate) fn attention_parameters(
        &self,
        key: &str,
        dim: usize,
    ) -> Result<AttentionParameters> {
        match self.tuned_parameters(key) {
            Some(KernelParameters::Attention(parameters))
                if self.attention_candidates(dim).contains(&parameters) =>
            {
                Ok(parameters)
            }
            Some(_) => Err(Self::stale_parameters(key)),
            None => Ok(AttentionParameters::default()),
        }
    }

    // Plans `softmax(q kᵀ · scale) v` as one kernel, or returns `None` for operands it does not take: contiguous `F16` or `BF16` `[B, H, L, D]` queries and `[B, G, S, D]` keys and values with `G` dividing `H`, read in place for grouped-query heads, a head dimension that is a multiple of 16 up to 128, and a target with `sm_80`'s `mma.sync`.
    // A causal mask hides key `j` from query `i` when `j > i`, as `Tensor::causal_attention_mask` does.
    pub(crate) fn fused_attention(
        &self,
        query: &KernelInput,
        key: &KernelInput,
        value: &KernelInput,
        mask: Option<&KernelInput>,
        sinks: Option<&KernelInput>,
        key_lengths: Option<&KernelInput>,
        attention: ScaledDotProductAttentionAttributes,
    ) -> Result<Option<KernelPlan>> {
        let ScaledDotProductAttentionAttributes {
            scale,
            causal,
            window,
            softcap,
            masked,
            sinks: sunk,
            key_lengths: counted,
        } = attention;
        if masked != mask.is_some() || sunk != sinks.is_some() || counted != key_lengths.is_some() {
            return Err(Error::missing_input("fused attention"));
        }

        let data_type = query.data_type;
        let kind = operand_kind(data_type);
        let (q, k) = (dimensions(query), dimensions(key));
        if !self.takes_tensor_cores(&[query, key, value])
            || [mask, sinks]
                .into_iter()
                .flatten()
                .any(|input| input.data_type != data_type)
            || key_lengths.is_some_and(|lengths| lengths.data_type != DataType::I32)
            // A split's combining kernel reads the sinks, which would leave the key lengths out of place among this kernel's inputs.
            || (key_lengths.is_some() && sinks.is_some())
            || !spec::fuses_attention(
                data_type,
                query.layout.shape(),
                key.layout.shape(),
                value.layout.shape(),
            )
        {
            return Ok(None);
        }

        let extent = |value: i64| to_usize(value, "cuda attention extent");
        let (batch, heads, queries, dim) =
            (extent(q[0])?, extent(q[1])?, extent(q[2])?, extent(q[3])?);
        let (groups, keys) = (heads / extent(k[1])?, extent(k[2])?);
        if has_tcgen05(self.context.target)
            && mask.is_none()
            && sinks.is_none()
            && key_lengths.is_none()
        {
            return engine_plain_attention(
                [query, key, value],
                (batch, heads, queries, dim),
                (groups, keys),
                attention,
            )
            .map(Some);
        }

        let keyed = [query, key, value]
            .into_iter()
            .chain(mask)
            .chain(sinks)
            .chain(key_lengths)
            .cloned()
            .collect::<Vec<_>>();
        let parameters = self.attention_parameters(
            &parameter_key("attention", &keyed, &format!("{attention:?}")),
            dim,
        )?;
        let AttentionParameters {
            warps,
            key_tile,
            stages,
            split_below_blocks,
            min_blocks,
        } = parameters;
        let query_rows = warps * 16;
        #[expect(
            clippy::cast_precision_loss,
            reason = "head dimensions are at most 128"
        )]
        let scale = scale.unwrap_or_else(|| 1.0 / (dim as f32).sqrt());
        let scaled = scaled_score_expression(scale, softcap);
        let window = window
            .map(|window| to_u32(window, "cuda attention window"))
            .transpose()?;
        let windowed = window.map_or_else(
            || "false".to_owned(),
            |window| format!("key + {window}U <= positions[r / 2]"),
        );
        // A window lets the block skip key tiles wholly before its first query's window.
        let first_key = window.map_or_else(
            || "0U".to_owned(),
            |window| format!("(first_position + 1U > {window}U ? (first_position + 1U - {window}U) / {key_tile}U * {key_tile}U : 0U)"),
        );
        let row = dim + PAD;
        let (steps, fragments) = (dim / 16, dim / 8);
        // A block takes 64 rows of the queries of all the heads that share one key and value head, which are consecutive in memory, so it reads that cache once for them all.
        let packed = groups * queries;
        let blocks = batch * (heads / groups) * packed.div_ceil(query_rows);
        // Too few blocks to fill the GPU, as in decoding at small batches, split the keys into chunks attended separately and combined after.
        let (splits, chunk) = if blocks >= split_below_blocks || keys <= 4 * key_tile {
            (1, keys)
        } else {
            let wanted = (2 * split_below_blocks)
                .div_ceil(blocks)
                .min(keys.div_ceil(4 * key_tile));
            let chunk = keys.div_ceil(wanted).div_ceil(key_tile) * key_tile;
            (keys.div_ceil(chunk), chunk)
        };
        let rows_total = batch * heads * queries;
        let causal = if causal { "true" } else { "false" };
        let threads = u32::try_from(warps * 32)
            .map_err(|_| Error::invalid_argument("cuda attention threads"))?;

        let mut writer = KernelWriter::new();
        writer.require(CompileRequirements::TENSOR_CORES);
        writer.min_blocks(min_blocks);
        let qs = writer.input(data_type, "queries")?;
        let ks = writer.input(data_type, "keys")?;
        let vs = writer.input(data_type, "values")?;
        // A mask adds to each score in base two; its strides over `[B, H, L, S]` broadcast it, and each row finds its own once.
        let (mask_rows, masked_score) = match mask {
            Some(mask) => {
                let name = writer.input(data_type, "mask")?;
                let scores = Shape::create([q[0], q[1], q[2], k[2]])?;
                let strides = mask
                    .layout
                    .expand(&scores)?
                    .strides()
                    .iter()
                    .map(|&stride| to_u64(stride, "cuda attention mask stride"))
                    .collect::<Result<Vec<_>>>()?;
                let load = load_expression(
                    element_type(data_type)?,
                    &format!("{name}[mask_rows[r / 2] + key * {}ULL]", strides[3]),
                );
                (
                    format!(
                        "
unsigned long long mask_rows[2];
#pragma unroll
for (int h = 0; h < 2; ++h) {{
    const unsigned int row = rows[h] < {packed}U ? rows[h] : 0U;
    const unsigned long long head = kv_head % {kv_heads}ULL * {groups}ULL + row / {queries}U;
    mask_rows[h] = kv_head / {kv_heads}ULL * {}ULL + head * {}ULL + (row % {queries}U) * {}ULL;
}}",
                        strides[0],
                        strides[1],
                        strides[2],
                        kv_heads = heads / groups,
                    ),
                    format!(" + {load} * {LOG2_E:?}f"),
                )
            }
            None => (String::new(), String::new()),
        };
        // A sink logit per query head joins each row's softmax sum, in base two against the row's largest score, once its keys are done.
        let sink_stride = sinks
            .map(|sinks| {
                to_u64(
                    sinks.layout.strides().first().copied().unwrap_or(0),
                    "cuda attention sinks stride",
                )
            })
            .transpose()?;
        let sink_read = sink_stride.map(|stride| SinkRead {
            data_type,
            stride,
            queries,
            heads,
        });
        let sink_rows = if splits == 1 {
            sink_read
                .map(|read| {
                    read.weight(
                        &mut writer,
                        &format!("kv_head * {packed}ULL + rows[h]"),
                        "largest[h]",
                    )
                })
                .transpose()?
                .unwrap_or_default()
        } else {
            String::new()
        };
        let sink_rows = if sink_rows.is_empty() {
            sink_rows
        } else {
            format!(
                "
#pragma unroll
for (int h = 0; h < 2; ++h)
    if (rows[h] < {packed}U) total[h] += {sink_rows};"
            )
        };
        // A batch's count of keys ends its keys, and the block's keys are of one batch.
        let key_count = key_lengths
            .map(|lengths| {
                let stride = to_u64(
                    lengths.layout.strides().first().copied().unwrap_or(1),
                    "cuda attention key lengths stride",
                )?;
                let name = writer.input(DataType::I32, "key_lengths")?;
                Ok::<_, Error>(format!(
                    "min(static_cast<unsigned int>(max({name}[kv_head / {kv_heads}ULL * {stride}ULL], 0)), {keys}U)",
                    kv_heads = heads / groups,
                ))
            })
            .transpose()?
            .unwrap_or_else(|| format!("{keys}U"));
        let os = writer.output(data_type, "output")?;
        let scratch = if splits > 1 {
            Some((
                writer.scratch(DataType::F32, splits * rows_total * dim, "partials")?,
                writer.scratch(DataType::F32, splits * rows_total * 2, "statistics")?,
            ))
        } else {
            None
        };
        let mut launch = writer.blocks(blocks * splits, threads);
        launch.shared_bytes = tile_launch_bytes(parameters, row)?;
        writer.declare(&tensor_core_helpers(kind));
        writer.declare(&format!(
            "
// Copies `rows` rows of `{dim}` values from `source`, `{dim}` apart, into a padded shared tile, zeros past `limit`.
__device__ __forceinline__ void singe_load_tile(unsigned short (*tile)[{row}], const unsigned short* source, unsigned int first, unsigned int limit) {{
    for (unsigned int chunk = threadIdx.x; chunk < {tile_chunks}U; chunk += blockDim.x) {{
        const unsigned int r = chunk / {row_chunks}U, column = (chunk % {row_chunks}U) * 8U;
        const bool valid = first + r < limit;
        singe_copy_async(singe_shared_address(&tile[r][column]), valid ? source + static_cast<unsigned long long>(first + r) * {dim}ULL + column : source, valid);
    }}
}}",
            tile_chunks = key_tile * dim / 8,
            row_chunks = dim / 8,
        ));
        // Unsplit, the block normalizes and stores its rows; split, it leaves each row's unnormalized output, maximum, and sum for the combining kernel.
        let finish = match &scratch {
            None => format!(
                "unsigned short* output_head = {os} + kv_head * {packed}ULL * {dim}ULL;
#pragma unroll
for (int j = 0; j < {fragments}; ++j)
#pragma unroll
    for (int h = 0; h < 2; ++h) {{
        if (rows[h] >= {packed}U) continue;
        const float inverse = total[h] > 0.0f ? 1.0f / total[h] : 0.0f;
        const unsigned int column = j * 8 + (warp_lane % 4) * 2;
        *reinterpret_cast<unsigned int*>(output_head + static_cast<unsigned long long>(rows[h]) * {dim}ULL + column) = singe_pack(accumulated[j][h * 2] * inverse, accumulated[j][h * 2 + 1] * inverse);
    }}"
            ),
            Some((partials, statistics)) => format!(
                "const unsigned long long first_row = {rows_total}ULL * split + kv_head * {packed}ULL;
#pragma unroll
for (int j = 0; j < {fragments}; ++j)
#pragma unroll
    for (int h = 0; h < 2; ++h) {{
        if (rows[h] >= {packed}U) continue;
        float* partial = {partials} + (first_row + rows[h]) * {dim}ULL + j * 8 + (warp_lane % 4) * 2;
        partial[0] = accumulated[j][h * 2];
        partial[1] = accumulated[j][h * 2 + 1];
    }}
if (warp_lane % 4 == 0)
#pragma unroll
    for (int h = 0; h < 2; ++h) {{
        if (rows[h] >= {packed}U) continue;
        {statistics}[(first_row + rows[h]) * 2ULL] = largest[h];
        {statistics}[(first_row + rows[h]) * 2ULL + 1ULL] = total[h];
    }}"
            ),
        };
        writer.lines(&format!(
            "
{tiles_declared}
const unsigned int warp = lane / 32;
const unsigned int warp_lane = lane % 32;
const unsigned int tiles = {query_tiles}U;
// `kv_head` counts key and value heads across the batch; the block's rows are rows of its group's queries, `{queries}` per query head.
// Blocks of one split of the keys are consecutive; `chunk` keys from the split's first.
const unsigned int split = blockIdx.x / {blocks}U;
const unsigned int block = blockIdx.x % {blocks}U;
const unsigned long long kv_head = block / tiles;
const unsigned int first_query = (block % tiles) * {query_rows}U;
const unsigned short* query_head = {qs} + kv_head * {packed}ULL * {dim}ULL;
const unsigned short* key_head = {ks} + kv_head * {keys}ULL * {dim}ULL;
const unsigned short* value_head = {vs} + kv_head * {keys}ULL * {dim}ULL;
// The queries pass through the key tile into each warp's registers, 16 rows a warp.
singe_load_tile(key_tile, query_head, first_query, {packed}U);
asm volatile(\"cp.async.commit_group;\\n\" ::);
asm volatile(\"cp.async.wait_group 0;\\n\" ::);
__syncthreads();
{query_fragments}
// Each lane holds two rows of the warp's 16, `group` and `group + 8`.
const unsigned int group = warp_lane / 4;
const unsigned int row_low = first_query + warp * 16 + group;
const unsigned int rows[2] = {{row_low, row_low + 8}};
// A row's position among its head's queries, which the causal mask compares keys with.
const unsigned int positions[2] = {{row_low % {queries}U, (row_low + 8) % {queries}U}};
{mask_rows}
{accumulators}
// A causal block's last key is its last query's.
const unsigned int last_row = min(first_query + {query_rows}U, {packed}U) - 1U;
const unsigned int last_position = first_query / {queries}U == last_row / {queries}U ? last_row % {queries}U : {queries}U - 1U;
const unsigned int key_count = {key_count};
const unsigned int key_end = min({causal} ? min(key_count, last_position + 1U) : key_count, (split + 1U) * {chunk}U);
const unsigned int first_position = first_query / {queries}U == last_row / {queries}U ? first_query % {queries}U : 0U;
{key_loop}
{COMBINE_ROW_TOTALS}
{sink_rows}
{finish}",
            query_tiles = packed.div_ceil(query_rows),
            query_fragments = load_query_fragments(steps),
            accumulators = row_accumulators(fragments),
            tiles_declared = shared_tiles(parameters, row),
            key_loop = key_tile_loop(
                &format!("max(split * {chunk}U, {first_key})"),
                (key_tile, stages),
                &|first| format!(
                    "    singe_load_tile(key_tile, key_head, {first}, {keys}U);
    asm volatile(\"cp.async.commit_group;\\n\" ::);
    singe_load_tile(value_tile, value_head, {first}, {keys}U);
    asm volatile(\"cp.async.commit_group;\\n\" ::);"
                ),
                &fold_key_tile(
                &format!("key >= key_count || rows[r / 2] >= {packed}U || ({causal} && key > positions[r / 2]) || ({windowed})"),
                &format!("{scaled}{masked_score}"),
                steps,
                fragments,
                (key_tile, stages),
            ),
            ),
        ));
        let output = ValueSpec::new(data_type, query.layout.shape().clone());
        let inputs = [query, key, value]
            .into_iter()
            .chain(mask)
            .chain(sinks)
            .chain(key_lengths)
            .cloned()
            .collect::<Vec<_>>();
        let attend = writer.finish(
            "fused_attention",
            launch,
            vec![output.clone()],
            &inputs,
            &[Operation::ScaledDotProductAttention],
        );
        if splits == 1 {
            return Ok(Some(attend));
        }
        Ok(Some(combine_splits(
            attend,
            &CombinedSplits {
                data_type,
                splits,
                rows_total,
                dim,
                masked: mask.is_some() || key_lengths.is_some(),
                sink: sink_read,
                output,
            },
            &inputs,
        )?))
    }
}

// How a block reads the sink logit of a row's query head.
#[derive(Debug, Clone, Copy)]
struct SinkRead {
    data_type: DataType,
    stride: u64,
    queries: usize,
    heads: usize,
}

impl SinkRead {
    // Declares the sinks input of `writer`, and returns the logit of `row`'s query head, in base two against `largest`.
    fn weight(self, writer: &mut KernelWriter, row: &str, largest: &str) -> Result<String> {
        let Self {
            data_type,
            stride,
            queries,
            heads,
        } = self;
        let name = writer.input(data_type, "sinks")?;
        let logit = load_expression(
            element_type(data_type)?,
            &format!("{name}[(({row}) / {queries}ULL) % {heads}ULL * {stride}ULL]"),
        );
        Ok(format!("exp2f({logit} * {LOG2_E:?}f - {largest})"))
    }
}

// What the kernel that combines split keys reads.
#[derive(Debug, Clone)]
struct CombinedSplits {
    data_type: DataType,
    splits: usize,
    rows_total: usize,
    dim: usize,
    masked: bool,
    sink: Option<SinkRead>,
    output: ValueSpec,
}

// Runs `attend`, which leaves each split's unnormalized output, maximum, and sum, before a kernel where each output element weighs its splits' outputs by their maxima, as the online softmax weighs tiles.
fn combine_splits(
    attend: KernelPlan,
    combined: &CombinedSplits,
    inputs: &[KernelInput],
) -> Result<KernelPlan> {
    let &CombinedSplits {
        data_type,
        splits,
        rows_total,
        dim,
        masked,
        sink,
        ..
    } = combined;
    let mut combine = KernelWriter::new();
    // The combining kernel reads only the sinks, but declares the inputs before them so each takes its place.
    let sunk_total = if let Some(sink) = sink {
        for (name, present) in [
            ("queries", true),
            ("keys", true),
            ("values", true),
            ("mask", masked),
        ] {
            if present {
                combine.input(data_type, name)?;
            }
        }
        format!(
            "\nif (largest != -__uint_as_float(0x7f800000U)) total += {};",
            sink.weight(&mut combine, "row", "largest")?
        )
    } else {
        String::new()
    };
    let os = combine.output(data_type, "output")?;
    let partials = combine.scratch(DataType::F32, splits * rows_total * dim, "partials")?;
    let statistics = combine.scratch(DataType::F32, splits * rows_total * 2, "statistics")?;
    let launch = combine.elements(rows_total * dim);
    let stored = store_expression(
        element_type(data_type)?,
        "total > 0.0f ? value / total : 0.0f",
    );
    combine.lines(&format!(
        "
const unsigned long long row = index / {dim}ULL;
float largest = -__uint_as_float(0x7f800000U);
for (unsigned int split = 0; split < {splits}U; ++split) largest = fmaxf(largest, {statistics}[(split * {rows_total}ULL + row) * 2ULL]);
float total = 0.0f, value = 0.0f;
if (largest != -__uint_as_float(0x7f800000U)) {{
    for (unsigned int split = 0; split < {splits}U; ++split) {{
        const float weight = exp2f({statistics}[(split * {rows_total}ULL + row) * 2ULL] - largest);
        total += weight * {statistics}[(split * {rows_total}ULL + row) * 2ULL + 1ULL];
        value += weight * {partials}[split * {rows_total}ULL * {dim}ULL + index];
    }}
}}{sunk_total}
{os}[index] = {stored};"
    ));
    let mut plan = combine.finish(
        "fused_attention_combine",
        launch,
        vec![combined.output.clone()],
        inputs,
        &[Operation::ScaledDotProductAttention],
    );
    plan.before.push(attend.kernel);
    Ok(plan)
}

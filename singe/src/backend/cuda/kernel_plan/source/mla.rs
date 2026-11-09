// Fused latent attention over paged caches, as DeepSeek's absorbed MLA runs it: every head of a token attends over one shared latent cache, whose rows are both the keys, with a positional part beside them, and the values.
// Each block takes one group of 16 rows of one sequence's queries, a token's heads consecutive, when decoding, and two groups sharing each key tile when prefilling, on tensor cores.
// A group's four warps split each tile of 32 keys for the scores, eight keys a warp over the full key width, and split the latent width for the output, so a tile of keys is read once, as keys and as values.
// The rows' running maxima and sums combine across a group's warps through shared memory.

use std::f32::consts::LOG2_E;

use crate::{
    backend::{
        cuda::{
            codegen::CompileRequirements,
            kernel_plan::source::{
                KernelInput, KernelPlan, KernelPlanner,
                paged_attention::{PagedSequence, paged_indices, report_index_error},
                paged_general::{GeneralPaged, ScoreTerm, general_paged},
                parameters::{KernelParameters, MlaParameters, parameter_key},
                tensor_core::{operand_kind, row_accumulators, tensor_core_helpers},
                writer::KernelWriter,
            },
        },
        registry::{Operation, ScaledDotProductAttentionAttributes},
        spec,
    },
    data_type::DataType,
    error::{Error, Result},
    utility::{to_u32, to_usize},
    value::ValueSpec,
};

const ROWS: usize = 16;
const PAD: usize = 8;
const SPLIT_KEYS: usize = 64;
const MAX_SPLITS: usize = 32;

impl KernelPlanner {
    // The parameters latent attention with `width` query values may run with, the default first.
    pub(crate) fn mla_candidates(&self, latent: usize, width: usize) -> Vec<MlaParameters> {
        let default = MlaParameters::default();
        let mut candidates = vec![default];
        // Each warp computes eight keys of a tile, so a tile is as long as its warps make it.
        for warps in [4, 2, 8] {
            for target_blocks in [128, 64, 256] {
                candidates.push(MlaParameters {
                    warps,
                    key_tile: warps * 8,
                    target_blocks,
                    min_blocks: None,
                });
            }
        }

        candidates.extend(
            self.launch_bound_candidates(default.warps * 32)
                .into_iter()
                .flatten()
                .map(|bound| MlaParameters {
                    min_blocks: Some(bound),
                    ..default
                }),
        );
        let row = width + PAD;
        candidates.retain(|candidate| {
            latent.is_multiple_of(candidate.warps * 8)
                && 2 * (2 * ROWS + candidate.key_tile * 2) * row <= 96 * 1024
        });
        let mut unique = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            if !unique.contains(&candidate) {
                unique.push(candidate);
            }
        }
        unique
    }

    // Plans latent attention of `inputs`, the latent and positional queries, the latent and positional caches, and the query offsets, page offsets, pages, and last page lengths, or returns `None` for operands it does not take: contiguous `F16` or `BF16` operands, a latent width that is a multiple of 32 up to 512, a positional width that is a multiple of 8 up to 128 and makes the key width a multiple of 16, contiguous index tensors, and a target with `sm_80`'s `mma.sync`.
    pub(crate) fn paged_mla_attention(
        &self,
        inputs: &[KernelInput],
        attention: ScaledDotProductAttentionAttributes,
    ) -> Result<KernelPlan> {
        let [
            query,
            query_positions,
            latent_cache,
            position_cache,
            query_offsets,
            page_offsets,
            pages,
            last_page_lengths,
        ] = inputs
        else {
            return Err(Error::missing_input("paged mla attention"));
        };
        let problem = spec::paged_mla_attention_problem(
            query.layout.shape(),
            query_positions.layout.shape(),
            latent_cache.layout.shape(),
            position_cache.layout.shape(),
            paged_indices([query_offsets, page_offsets, pages, last_page_lengths]),
            "cuda paged mla attention",
        )?;
        let data_type = query.data_type;
        let extent = |value: i64| to_usize(value, "cuda paged mla attention extent");
        let (tokens, heads, latent, positional, page_size, sequences) = (
            extent(problem.tokens)?,
            extent(problem.heads)?,
            extent(problem.latent_dim)?,
            extent(problem.position_dim)?,
            extent(problem.page_size)?,
            extent(problem.sequences)?,
        );
        let width = latent + positional;
        let key = parameter_key("paged_mla_attention", inputs, &format!("{attention:?}"));
        let MlaParameters {
            warps,
            key_tile,
            target_blocks,
            min_blocks,
        } = match self.tuned_parameters(&key) {
            Some(KernelParameters::Mla(parameters))
                if self.mla_candidates(latent, width).contains(&parameters) =>
            {
                parameters
            }
            Some(_) => return Err(Self::stale_parameters(&key)),
            None => MlaParameters::default(),
        };
        if attention.masked || attention.sinks {
            return Err(Error::unsupported_operation(
                "cuda paged mla attention with a mask or sinks",
            ));
        }

        if !self.takes_tensor_cores(&[query, query_positions, latent_cache, position_cache])
            || attention.window.is_some()
            || attention.softcap.is_some()
            || [query_offsets, page_offsets, pages, last_page_lengths]
                .iter()
                .any(|input| !input.layout.is_contiguous())
            || !latent.is_multiple_of(warps * 8)
            || latent > 512
            || !positional.is_multiple_of(8)
            || positional > 128
            || !width.is_multiple_of(16)
        {
            return general_paged(&GeneralPaged {
                label: "paged_mla_attention_general",
                operation: Operation::PagedMlaAttention,
                inputs,
                attention,
                terms: &[
                    ScoreTerm {
                        query: 0,
                        key: 2,
                        width: latent,
                    },
                    ScoreTerm {
                        query: 1,
                        key: 3,
                        width: positional,
                    },
                ],
                value: 2,
                value_width: latent,
                groups: None,
                indices: 4,
                tokens,
                heads,
                page_size,
                cache_pages: extent(latent_cache.layout.shape().dimensions()[0])?,
                sequences,
                output: ValueSpec::new(data_type, problem.output_shape.clone()),
                lse: ValueSpec::new(DataType::F32, problem.lse_shape),
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "head dimensions round to the score's precision"
                )]
                default_scale: 1.0 / (width as f32).sqrt(),
            });
        }

        let kind = operand_kind(data_type);
        // Prefill blocks take two groups of 16 rows that share each key tile, so a tile is read once for both; decoding blocks take one group, whose next key tile loads while it computes on this one.
        let groups = if tokens > sequences { 2 } else { 1 };
        let stages = 3 - groups;
        let block_rows = ROWS * groups;
        // Every sequence's rows fit in this many tiles, since no sequence has more queries than all of them.
        let tiles = (heads * tokens).div_ceil(block_rows).max(1);
        // Few long sequences split their keys across blocks, about `target_blocks` in all, each split at least `SPLIT_KEYS` keys long on average, and a second kernel merges the splits by their log-sum-exps.
        let average_keys = pages.layout.shape().element_count() * page_size / sequences.max(1);
        let splits = (target_blocks / (sequences * tiles).max(1))
            .min(average_keys / SPLIT_KEYS)
            .clamp(1, MAX_SPLITS);
        let blocks = (sequences * tiles * splits).max(1);
        let rows_total = tokens * heads;
        #[expect(
            clippy::cast_precision_loss,
            reason = "head dimensions are at most 640"
        )]
        let scale = attention
            .scale
            .unwrap_or_else(|| 1.0 / (width as f32).sqrt());
        let causal = if attention.causal { "true" } else { "false" };
        let row = width + PAD;
        let probability_row = key_tile + PAD;
        // Each warp's share of the latent width, in fragments of eight columns.
        let columns = latent / warps;
        let fragments = columns / 8;
        let (steps, row_chunks) = (width / 16, width / 8);
        let shared_bytes = 2
            * ((block_rows + stages * key_tile) * row + block_rows * probability_row)
            + 4 * 2 * groups * warps * ROWS;
        let threads = u32::try_from(groups * warps * 32)
            .map_err(|_| Error::invalid_argument("cuda mla attention threads"))?;

        let mut writer = KernelWriter::new();
        writer.require(CompileRequirements::TENSOR_CORES);
        writer.min_blocks(min_blocks);
        let qn = writer.input(data_type, "latent_queries")?;
        let qp = writer.input(data_type, "positional_queries")?;
        let kc = writer.input(data_type, "latent_cache")?;
        let kp = writer.input(data_type, "positional_cache")?;
        let qo = writer.input(query_offsets.data_type, "query_offsets")?;
        let po = writer.input(page_offsets.data_type, "page_offsets")?;
        let pg = writer.input(pages.data_type, "pages")?;
        let lp = writer.input(last_page_lengths.data_type, "last_page_lengths")?;
        let (os, ls) = if splits == 1 {
            (
                writer.output(data_type, "output")?,
                writer.output(DataType::F32, "lse")?,
            )
        } else {
            (
                writer.scratch(
                    DataType::F32,
                    splits * rows_total * latent,
                    "partial_outputs",
                )?,
                writer.scratch(DataType::F32, splits * rows_total, "partial_lse")?,
            )
        };
        // A split writes its rows' normalized partial outputs and log-sum-exps at its own offset.
        let (stored, row_offset) = if splits == 1 {
            (
                "*reinterpret_cast<unsigned int*>(OUT) = singe_pack(LOW, HIGH)",
                String::new(),
            )
        } else {
            (
                "*reinterpret_cast<float2*>(OUT) = make_float2(LOW, HIGH)",
                format!(" + static_cast<unsigned long long>(split) * {rows_total}ULL"),
            )
        };
        let stored = stored
            .replace("OUT", &format!("{os} + output_row * {latent}ULL + column"))
            .replace("LOW", "accumulated[j][h * 2] * inverse")
            .replace("HIGH", "accumulated[j][h * 2 + 1] * inverse");
        let mut launch = writer.blocks(blocks, threads);
        launch.shared_bytes = to_u32(shared_bytes, "cuda mla attention shared memory")?;
        writer.status();
        let report = report_index_error();
        let page_total = to_usize(
            latent_cache.layout.shape().dimensions()[0],
            "cuda mla attention pages",
        )?;
        let sequence_values = PagedSequence {
            query_offsets: &qo,
            page_offsets: &po,
            last_page_lengths: &lp,
            tokens,
            page_table: to_usize(
                pages.layout.shape().dimensions()[0],
                "cuda mla attention page table",
            )?,
            page_size,
            rows_per_query: heads,
        }
        .source();
        writer.declare(&tensor_core_helpers(kind));
        writer.declare(&format!(
            "
// Copies 16-byte chunk `chunk` of a row of `{width}` values, the first `{latent}` from `latent`'s row `slot` and the rest from `positional`'s, into `destination`, or zeros when `valid` is false.
__device__ __forceinline__ void singe_copy_mla_chunk(unsigned int destination, const unsigned short* latent, const unsigned short* positional, unsigned long long slot, unsigned int column, bool valid) {{
    const unsigned short* source = column < {latent}U ? latent + slot * {latent}ULL + column : positional + slot * {positional}ULL + (column - {latent}U);
    singe_copy_async(destination, valid ? source : latent, valid);
}}"
        ));
        // A decoding block loads its next key tile while it computes on this one; a prefill block's two groups load each tile once, before computing on it.
        let key_loop = if stages == 2 {
            format!(
                "unsigned int stage = 0U;
if (split * split_keys < split_end) load_tile(0U, split * split_keys);
for (unsigned int first_key = split * split_keys; first_key < split_end; first_key += {key_tile}U, stage ^= 1U) {{
    if (first_key + {key_tile}U < split_end) {{
        load_tile(stage ^ 1U, first_key + {key_tile}U);
        asm volatile(\"cp.async.wait_group 1;\\n\" ::);
    }} else {{
        asm volatile(\"cp.async.wait_group 0;\\n\" ::);
    }}
    __syncthreads();"
            )
        } else {
            format!(
                "const unsigned int stage = 0U;
for (unsigned int first_key = split * split_keys; first_key < split_end; first_key += {key_tile}U) {{
    load_tile(0U, first_key);
    asm volatile(\"cp.async.wait_group 0;\\n\" ::);
    __syncthreads();"
            )
        };
        writer.lines(&format!(
            "
extern __shared__ __align__(16) unsigned char singe_shared[];
unsigned short (*query_tile)[{row}] = reinterpret_cast<unsigned short (*)[{row}]>(singe_shared);
// One or two stages of key tiles.
unsigned short (*key_tiles)[{key_tile}][{row}] = reinterpret_cast<unsigned short (*)[{key_tile}][{row}]>(singe_shared + {query_bytes});
unsigned short (*probability_tile)[{probability_row}] = reinterpret_cast<unsigned short (*)[{probability_row}]>(singe_shared + {key_offset});
float (*warp_largest)[{ROWS}] = reinterpret_cast<float (*)[{ROWS}]>(singe_shared + {statistics_offset});
float (*warp_total)[{ROWS}] = reinterpret_cast<float (*)[{ROWS}]>(singe_shared + {statistics_offset} + {statistics_bytes});
// Each group of four warps takes 16 rows of the block, and each warp of a group a share of the keys and of the latent width.
const unsigned int row_group = lane / 32 / {warps}U;
const unsigned int warp = lane / 32 % {warps}U;
const unsigned int statistics_row = row_group * {warps}U;
const unsigned int warp_lane = lane % 32;
const unsigned int split = blockIdx.x % {splits}U;
const unsigned int sequence = blockIdx.x / {splits}U / {tiles}U;
const unsigned int first_row = (blockIdx.x / {splits}U % {tiles}U) * {block_rows}U;
if (sequence >= {sequences}U) return;
// The sequence's queries, `{heads}` rows each, and its keys, all of its pages full but the last.
{sequence_values}
// The queries are the sequence's last tokens, so query `q` sits at key position `keys - queries + q`.
const long long delta = static_cast<long long>(keys) - static_cast<long long>(queries);
const unsigned long long row_base = static_cast<unsigned long long>(query_start) * {heads}ULL + first_row;
for (unsigned int chunk = threadIdx.x; chunk < {query_chunks}U; chunk += blockDim.x) {{
    const unsigned int r = chunk / {row_chunks}U, column = (chunk % {row_chunks}U) * 8U;
    singe_copy_mla_chunk(singe_shared_address(&query_tile[r][column]), {qn}, {qp}, row_base + r, column, first_row + r < packed);
}}
asm volatile(\"cp.async.commit_group;\\n\" ::);
// Each lane holds two rows of the 16, `group` and `group + 8`, and two adjacent keys of its warp's eight.
const unsigned int group = warp_lane / 4;
const unsigned int group_row = row_group * {ROWS}U + group;
const unsigned int rows[2] = {{first_row + group_row, first_row + group_row + 8}};
const long long positions[2] = {{delta + rows[0] / {heads}U, delta + rows[1] / {heads}U}};
{accumulators}
// A causal block's last key is its last query's.
const long long last_position = delta + (min(first_row + {block_rows}U, packed) - 1U) / {heads}U;
const unsigned int key_end = {causal} ? (last_position < 0LL ? 0U : min(keys, static_cast<unsigned int>(last_position + 1LL))) : keys;
// The split's keys, a whole number of tiles.
const unsigned int split_keys = ((keys + {splits}U - 1U) / {splits}U + {key_tile}U - 1U) / {key_tile}U * {key_tile}U;
const unsigned int split_end = min(key_end, (split + 1U) * split_keys);
const auto load_tile = [&](unsigned int stage, unsigned int first_key) {{
    for (unsigned int chunk = threadIdx.x; chunk < {key_chunks}U; chunk += blockDim.x) {{
        const unsigned int r = chunk / {row_chunks}U, column = (chunk % {row_chunks}U) * 8U;
        const unsigned int key = first_key + r;
        const bool wanted = key < keys;
        const unsigned long long page = wanted ? static_cast<unsigned long long>({pg}[page_start + key / {page_size}U]) : 0ULL;
        const bool valid = wanted && page < {page_total}ULL;
        if (wanted && !valid) {{ {report} }}
        singe_copy_mla_chunk(singe_shared_address(&key_tiles[stage][r][column]), {kc}, {kp}, page * {page_size}ULL + key % {page_size}U, column, valid);
    }}
    asm volatile(\"cp.async.commit_group;\\n\" ::);
}};
{key_loop}
    unsigned short (*key_tile)[{row}] = key_tiles[stage];
    // The warp's eight keys against all 16 rows, over the full key width, in four independent sums so consecutive products need not wait for each other.
    float partial[4][4] = {{}};
#pragma unroll
    for (int step = 0; step < {steps}; ++step) {{
        unsigned int a[4];
        unsigned int b[2];
        const unsigned int query_address = singe_shared_address(&query_tile[row_group * {ROWS}U + warp_lane % 16][step * 16 + (warp_lane / 16) * 8]);
        asm volatile(\"ldmatrix.sync.aligned.x4.m8n8.shared.b16 {{%0, %1, %2, %3}}, [%4];\\n\" : \"=r\"(a[0]), \"=r\"(a[1]), \"=r\"(a[2]), \"=r\"(a[3]) : \"r\"(query_address));
        const unsigned int key_address = singe_shared_address(&key_tile[warp * 8 + warp_lane % 8][step * 16 + ((warp_lane / 8) % 2) * 8]);
        asm volatile(\"ldmatrix.sync.aligned.x2.m8n8.shared.b16 {{%0, %1}}, [%2];\\n\" : \"=r\"(b[0]), \"=r\"(b[1]) : \"r\"(key_address));
        singe_mma(partial[step % 4], a, b);
    }}
    float scores[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) scores[r] = (partial[0][r] + partial[1][r]) + (partial[2][r] + partial[3][r]);
    // Scale into base two, hide keys past the end and, for a causal mask, after the query, and share each row's largest score of the warp's keys.
    float row_largest[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {{
#pragma unroll
        for (int k = 0; k < 2; ++k) {{
            const unsigned int key = first_key + warp * 8 + (warp_lane % 4) * 2 + k;
            const bool hidden = key >= keys || rows[h] >= packed || ({causal} && static_cast<long long>(key) > positions[h]);
            scores[h * 2 + k] = hidden ? -__uint_as_float(0x7f800000U) : scores[h * 2 + k] * {scaled:?}f;
        }}
        row_largest[h] = fmaxf(scores[h * 2], scores[h * 2 + 1]);
        row_largest[h] = fmaxf(row_largest[h], __shfl_xor_sync(0xffffffffu, row_largest[h], 1));
        row_largest[h] = fmaxf(row_largest[h], __shfl_xor_sync(0xffffffffu, row_largest[h], 2));
        if (warp_lane % 4 == 0) warp_largest[statistics_row + warp][group + h * 8] = row_largest[h];
    }}
    __syncthreads();
    float rescale[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {{
        float tile_largest = largest[h];
#pragma unroll
        for (int w = 0; w < {warps}; ++w) tile_largest = fmaxf(tile_largest, warp_largest[statistics_row + w][group + h * 8]);
        // A row with every key so far hidden stays at zero weight.
        const float base = tile_largest == -__uint_as_float(0x7f800000U) ? 0.0f : tile_largest;
        rescale[h] = exp2f(largest[h] - base);
        largest[h] = tile_largest;
        scores[h * 2] = exp2f(scores[h * 2] - base);
        scores[h * 2 + 1] = exp2f(scores[h * 2 + 1] - base);
        float row_total = scores[h * 2] + scores[h * 2 + 1];
        row_total += __shfl_xor_sync(0xffffffffu, row_total, 1);
        row_total += __shfl_xor_sync(0xffffffffu, row_total, 2);
        if (warp_lane % 4 == 0) warp_total[statistics_row + warp][group + h * 8] = row_total;
        *reinterpret_cast<unsigned int*>(&probability_tile[group_row + h * 8][warp * 8 + (warp_lane % 4) * 2]) = singe_pack(scores[h * 2], scores[h * 2 + 1]);
    }}
    __syncthreads();
#pragma unroll
    for (int h = 0; h < 2; ++h) {{
        float tile_total = 0.0f;
#pragma unroll
        for (int w = 0; w < {warps}; ++w) tile_total += warp_total[statistics_row + w][group + h * 8];
        total[h] = total[h] * rescale[h] + tile_total;
    }}
#pragma unroll
    for (int j = 0; j < {fragments}; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) accumulated[j][r] *= rescale[r / 2];
    // The tile's probabilities against the latent part of its keys, the warp's columns of it.
#pragma unroll
    for (int step = 0; step < {key_steps}; ++step) {{
        unsigned int a[4];
        const unsigned int probability_address = singe_shared_address(&probability_tile[row_group * {ROWS}U + warp_lane % 16][step * 16 + (warp_lane / 16) * 8]);
        asm volatile(\"ldmatrix.sync.aligned.x4.m8n8.shared.b16 {{%0, %1, %2, %3}}, [%4];\\n\" : \"=r\"(a[0]), \"=r\"(a[1]), \"=r\"(a[2]), \"=r\"(a[3]) : \"r\"(probability_address));
#pragma unroll
        for (int j = 0; j < {fragments}; ++j) {{
            unsigned int b[2];
            const unsigned int value_address = singe_shared_address(&key_tile[step * 16 + warp_lane % 16][warp * {columns} + j * 8]);
            asm volatile(\"ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {{%0, %1}}, [%2];\\n\" : \"=r\"(b[0]), \"=r\"(b[1]) : \"r\"(value_address));
            singe_mma(accumulated[j], a, b);
        }}
    }}
    __syncthreads();
}}
asm volatile(\"cp.async.wait_group 0;\\n\" ::);
// A row that sees no key keeps a zero output and a log-sum-exp of negative infinity.
#pragma unroll
for (int h = 0; h < 2; ++h) {{
    if (rows[h] >= packed) continue;
    const float inverse = total[h] > 0.0f ? 1.0f / total[h] : 0.0f;
    const unsigned long long output_row = row_base + group_row + h * 8{row_offset};
#pragma unroll
    for (int j = 0; j < {fragments}; ++j) {{
        const unsigned int column = warp * {columns} + j * 8 + (warp_lane % 4) * 2;
        {stored};
    }}
    if (warp == 0 && warp_lane % 4 == 0) {ls}[output_row] = total[h] > 0.0f ? largest[h] + log2f(total[h]) : -__uint_as_float(0x7f800000U);
}}",
            accumulators = row_accumulators(fragments),
            query_bytes = 2 * block_rows * row,
            key_offset = 2 * (block_rows + stages * key_tile) * row,
            statistics_offset = 2 * ((block_rows + stages * key_tile) * row + block_rows * probability_row),
            statistics_bytes = 4 * groups * warps * ROWS,
            query_chunks = block_rows * row_chunks,
            key_chunks = key_tile * row_chunks,
            key_steps = key_tile / 16,
            scaled = scale * LOG2_E,
        ));
        let outputs = vec![
            ValueSpec::new(data_type, query.layout.shape().clone()),
            ValueSpec::new(DataType::F32, problem.lse_shape),
        ];
        let attention = writer.finish(
            "paged_mla_attention",
            launch,
            outputs.clone(),
            inputs,
            &[Operation::PagedMlaAttention],
        );
        if splits == 1 {
            return Ok(attention);
        }

        merge_splits(
            attention,
            &MergedSplits {
                data_type,
                kind,
                splits,
                rows_total,
                latent,
                threads,
                outputs,
            },
            inputs,
        )
    }
}

// What the kernel that merges split keys reads.
#[derive(Debug, Clone)]
struct MergedSplits {
    data_type: DataType,
    kind: &'static str,
    splits: usize,
    rows_total: usize,
    latent: usize,
    threads: u32,
    outputs: Vec<ValueSpec>,
}

// Runs `attention`, which leaves each split's normalized output and log-sum-exp, before one block a row, which weighs each split's output by its share of the row's total, `2^(lse - largest)` over their sum.
fn merge_splits(
    attention: KernelPlan,
    merged: &MergedSplits,
    inputs: &[KernelInput],
) -> Result<KernelPlan> {
    let &MergedSplits {
        data_type,
        kind,
        splits,
        rows_total,
        latent,
        threads,
        ..
    } = merged;
    let mut merge = KernelWriter::new();
    merge.require(CompileRequirements::TENSOR_CORES);
    let partial_outputs = merge.scratch(
        DataType::F32,
        splits * rows_total * latent,
        "partial_outputs",
    )?;
    let partial_lse = merge.scratch(DataType::F32, splits * rows_total, "partial_lse")?;
    let output = merge.output(data_type, "output")?;
    let lse = merge.output(DataType::F32, "lse")?;
    let launch = merge.rows(rows_total, threads);
    merge.declare(&tensor_core_helpers(kind));
    merge.lines(&format!(
        "
float largest = -__uint_as_float(0x7f800000U);
for (unsigned int split = 0; split < {splits}U; ++split) largest = fmaxf(largest, {partial_lse}[split * {rows_total}ULL + row]);
float total = 0.0f;
if (largest != -__uint_as_float(0x7f800000U))
for (unsigned int split = 0; split < {splits}U; ++split) total += exp2f({partial_lse}[split * {rows_total}ULL + row] - largest);
const float inverse = total > 0.0f ? 1.0f / total : 0.0f;
for (unsigned int column = lane * 2U; column < {latent}U; column += blockDim.x * 2U) {{
float low = 0.0f, high = 0.0f;
if (total > 0.0f)
    for (unsigned int split = 0; split < {splits}U; ++split) {{
        const float weight = exp2f({partial_lse}[split * {rows_total}ULL + row] - largest) * inverse;
        const float2 part = *reinterpret_cast<const float2*>({partial_outputs} + (split * {rows_total}ULL + row) * {latent}ULL + column);
        low += weight * part.x;
        high += weight * part.y;
    }}
*reinterpret_cast<unsigned int*>({output} + row * {latent}ULL + column) = singe_pack(low, high);
}}
if (lane == 0) {lse}[row] = total > 0.0f ? largest + log2f(total) : largest;"
    ));
    let mut plan = merge.finish(
        "paged_mla_attention_merge",
        launch,
        merged.outputs.clone(),
        inputs,
        &[Operation::PagedMlaAttention],
    );
    plan.before = vec![attention.kernel];
    Ok(plan)
}

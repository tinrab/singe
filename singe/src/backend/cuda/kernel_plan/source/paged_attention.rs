// Fused attention over a paged key and value cache, as FlashInfer batches decode and prefill, on the dense kernel's tensor-core main loop.
// Each block takes 64 rows of one sequence's queries for one key and value head: the query heads that share it, a token's heads consecutive, so the block reads each key page once for them all.
// The grid covers every sequence's longest possible run of rows, and blocks past their sequence's rows return at once.

use singe_codegen::emit::cuda_c::spelling::{compute_type, load_expression, store_expression};

use crate::{
    backend::{
        cuda::{
            codegen::CompileRequirements,
            kernel_plan::source::{
                KernelInput, KernelPlan, KernelPlanner,
                attention::PAD,
                paged_general::{GeneralPaged, ScoreTerm, general_paged},
                paging::{PagedSequence, paged_indices, report_index_error},
                parameters::{AttentionParameters, has_tcgen05, parameter_key},
                rotary_cache::RotationNorm,
                tcgen05_attention::{EngineAttention, engine_attention},
                tensor_core::{
                    COMBINE_ROW_TOTALS, fold_key_tile, key_tile_loop, load_query_fragments,
                    operand_kind, row_accumulators, scaled_score_expression, shared_tiles,
                    tensor_core_helpers, tile_launch_bytes,
                },
                writer::{KernelWriter, value_type},
            },
        },
        kernel_lowering::element_type,
        registry::{Operation, ScaledDotProductAttentionAttributes},
        spec,
    },
    data_type::DataType,
    error::{Error, Result},
    shape::Shape,
    utility::to_usize,
    value::ValueSpec,
};

#[derive(Clone, Copy)]
struct PagedShape {
    tokens: usize,
    heads: usize,
    kv_heads: usize,
    dim: usize,
    page_size: usize,
    sequences: usize,
}

// Plans paged attention of operands the tensor-core kernel takes on the tensor engine.
fn engine_paged_attention(
    inputs: &[KernelInput],
    attention: ScaledDotProductAttentionAttributes,
    shape: PagedShape,
) -> Result<KernelPlan> {
    let PagedShape {
        tokens,
        heads,
        kv_heads,
        dim,
        page_size,
        sequences,
    } = shape;
    let Some(query) = inputs.first() else {
        return Err(Error::missing_input("cuda paged attention"));
    };
    let data_type = query.data_type;
    let groups = heads / kv_heads;
    let tiles = (groups * tokens).div_ceil(128).max(1);
    let blocks = (sequences * kv_heads * tiles).max(1);
    let page_total = cache_pages(inputs)?;
    let causal = if attention.causal { "true" } else { "false" };
    let windowed = attention.window.map_or_else(
        || "false".to_owned(),
        |window| format!("static_cast<long long>(key) + {window}LL <= position"),
    );
    let scale = attention.scale.unwrap_or_else(|| default_scale(dim));
    let score =
        scaled_score_expression(scale, attention.softcap).replace("scores[j][r]", "scores[column]");
    let stored = store_expression(element_type(data_type)?, "value");
    let report = report_index_error();
    let mut writer = KernelWriter::new();
    let PagedInputs {
        qs,
        ks,
        vs,
        qo,
        po,
        pg,
        lp,
    } = paged_inputs(&mut writer, inputs)?;
    let os = writer.output(data_type, "output")?;
    let ls = writer.output(DataType::F32, "lse")?;
    writer.status();
    let sequence_values = paged_sequence([&qo, &po, &lp], inputs, tokens, page_size, groups)?;
    let cache_source = |cache: &str| {
        format!(
            "const unsigned int key = first_key + r;
    const bool wanted = key < keys;
    const unsigned long long page = wanted ? static_cast<unsigned long long>({pg}[page_start + key / {page_size}U]) : 0ULL;
    const bool valid = wanted && page < {page_total}ULL;
    if (wanted && !valid) {{ {report} }}
    const unsigned long long offset = ((page * {page_size}ULL + key % {page_size}U) * {kv_heads}ULL + kv_head) * {dim}ULL + column;
    const unsigned short* source = valid ? {cache} + offset : {cache};"
        )
    };
    let row_index = |row: &str| {
        format!(
            "(static_cast<unsigned long long>(query_start) + {row} / {groups}U) * {heads}ULL + kv_head * {groups}U + {row} % {groups}U"
        )
    };
    let launch = engine_attention(
        &mut writer,
        &EngineAttention {
            data_type,
            dim,
            blocks,
            setup: &format!(
                "const unsigned int sequence = blockIdx.x / ({kv_heads}U * {tiles}U);
const unsigned int kv_head = (blockIdx.x / {tiles}U) % {kv_heads}U;
const unsigned int first_row = (blockIdx.x % {tiles}U) * 128U;
if (sequence >= {sequences}U) return;
{sequence_values}
const long long delta = static_cast<long long>(keys) - static_cast<long long>(queries);"
            ),
            position: &format!("delta + row / {groups}U"),
            query_source: &format!(
                "const unsigned int query_row = first_row + r;
    const bool valid = query_row < packed;
    const unsigned short* source = valid ? {qs} + ({query_row}) * {dim}ULL + column : {qs};",
                query_row = row_index("query_row"),
            ),
            key_source: &cache_source(&ks),
            value_source: &cache_source(&vs),
            hidden: &format!(
                "key >= keys || !row_valid || ({causal} && static_cast<long long>(key) > position) || ({windowed})"
            ),
            score: &score,
            store: &format!(
                "{os}[({row}) * {dim}ULL + at] = {stored};",
                row = row_index("row")
            ),
            finish: &format!(
                "if (row_valid) {ls}[{row}] = total > 0.0f ? largest + log2f(total) : -infinity;",
                row = row_index("row"),
            ),
        },
    )?;
    Ok(writer.finish(
        "tcgen05_paged_attention",
        launch,
        vec![
            ValueSpec::new(data_type, query.layout.shape().clone()),
            ValueSpec::new(
                DataType::F32,
                Shape::create([
                    query.layout.shape().dimensions()[0],
                    query.layout.shape().dimensions()[1],
                ])?,
            ),
        ],
        inputs,
        &[Operation::PagedAttention],
    ))
}

impl KernelPlanner {
    // Plans paged attention of `inputs`, the queries, the key and value caches, and the query offsets, page offsets, pages, and last page lengths, or returns `None` for operands it does not take: contiguous `F16` or `BF16` queries `[T, H, D]` and caches `[P, S, G, D]`, a head dimension that is a multiple of 16 up to 128, contiguous index tensors, and a target with `sm_80`'s `mma.sync`.
    pub(crate) fn paged_attention(
        &self,
        inputs: &[KernelInput],
        attention: ScaledDotProductAttentionAttributes,
    ) -> Result<KernelPlan> {
        self.paged_attention_kernel(inputs, attention, None)
    }

    // Whether the fused kernel of attention over a query it rotates itself can run on these operands: `mma.sync`, whose query stage it replaces, on contiguous operands of a head it can tile.
    pub(crate) fn takes_rotary_attention(
        &self,
        inputs: &[KernelInput],
        attention: ScaledDotProductAttentionAttributes,
    ) -> Result<bool> {
        let [query, key_cache, value_cache, indices @ ..] = inputs else {
            return Ok(false);
        };
        let dim = to_usize(
            query.layout.shape().dimensions()[2],
            "cuda paged attention head",
        )?;
        Ok(!attention.masked
            && !attention.sinks
            && !has_tcgen05(self.context.target)
            && self.takes_tensor_cores(&[query, key_cache, value_cache])
            && indices.iter().all(|input| input.layout.is_contiguous())
            && dim.is_multiple_of(16)
            && dim <= 128)
    }

    // Plans attention over a query this kernel normalizes and rotates itself, as `takes_rotary_attention` allows.
    // `inputs` hold the query before its rotation, and then what `paged_attention` reads and what `rotation` names.
    pub(crate) fn rotary_paged_attention(
        &self,
        inputs: &[KernelInput],
        attention: ScaledDotProductAttentionAttributes,
        rotation: &QueryRotation,
    ) -> Result<KernelPlan> {
        self.paged_attention_kernel(inputs, attention, Some(rotation))
    }

    fn paged_attention_kernel(
        &self,
        inputs: &[KernelInput],
        attention: ScaledDotProductAttentionAttributes,
        rotation: Option<&QueryRotation>,
    ) -> Result<KernelPlan> {
        let [
            query,
            key_cache,
            value_cache,
            query_offsets,
            page_offsets,
            pages,
            last_page_lengths,
            ..,
        ] = inputs
        else {
            return Err(Error::missing_input("paged attention"));
        };
        let problem = spec::paged_attention_problem(
            query.layout.shape(),
            key_cache.layout.shape(),
            value_cache.layout.shape(),
            paged_indices([query_offsets, page_offsets, pages, last_page_lengths]),
            "cuda paged attention",
        )?;
        let data_type = query.data_type;
        let extent = |value: i64| to_usize(value, "cuda paged attention extent");
        let (tokens, heads, kv_heads, dim, page_size, sequences) = (
            extent(problem.tokens)?,
            extent(problem.heads)?,
            extent(problem.kv_heads)?,
            extent(problem.head_dim)?,
            extent(problem.page_size)?,
            extent(problem.sequences)?,
        );
        if attention.masked || attention.sinks {
            return Err(Error::unsupported_operation(
                "cuda paged attention with a mask or sinks",
            ));
        }

        if !self.takes_tensor_cores(&[query, key_cache, value_cache])
            || [query_offsets, page_offsets, pages, last_page_lengths]
                .iter()
                .any(|input| !input.layout.is_contiguous())
            || !dim.is_multiple_of(16)
            || dim > 128
        {
            return general_paged(&GeneralPaged {
                label: "paged_attention_general",
                operation: Operation::PagedAttention,
                inputs,
                attention,
                terms: &[ScoreTerm {
                    query: 0,
                    key: 1,
                    width: dim,
                }],
                value: 2,
                value_width: dim,
                groups: Some(heads / kv_heads),
                indices: 3,
                tokens,
                heads,
                page_size,
                cache_pages: extent(key_cache.layout.shape().dimensions()[0])?,
                sequences,
                output: ValueSpec::new(data_type, problem.output_shape.clone()),
                lse: ValueSpec::new(
                    DataType::F32,
                    Shape::create([problem.tokens, problem.heads])?,
                ),
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "head dimensions round to the score's precision"
                )]
                default_scale: 1.0 / (dim as f32).sqrt(),
            });
        }

        let kind = operand_kind(data_type);
        let groups = heads / kv_heads;
        if has_tcgen05(self.context.target) {
            return engine_paged_attention(
                inputs,
                attention,
                PagedShape {
                    tokens,
                    heads,
                    kv_heads,
                    dim,
                    page_size,
                    sequences,
                },
            );
        }

        let parameters = self.attention_parameters(
            &parameter_key("paged_attention", inputs, &format!("{attention:?}")),
            dim,
        )?;
        let AttentionParameters {
            warps,
            key_tile,
            stages,
            min_blocks,
            ..
        } = parameters;
        let query_rows = warps * 16;
        // Every sequence's rows fit in this many tiles, since no sequence has more queries than all of them.
        let tiles = (groups * tokens).div_ceil(query_rows).max(1);
        let blocks = (sequences * kv_heads * tiles).max(1);
        let scale = attention.scale.unwrap_or_else(|| default_scale(dim));
        let scaled = scaled_score_expression(scale, attention.softcap);
        let window = attention.window;
        let windowed = window.map_or_else(
            || "false".to_owned(),
            |window| format!("static_cast<long long>(key) + {window}LL <= positions[r / 2]"),
        );
        // A window lets the block skip key tiles wholly before its first query's window.
        let first_key = window.map_or_else(
            || "0U".to_owned(),
            |window| format!("(first_position + 1LL > {window}LL ? static_cast<unsigned int>(first_position + 1LL - {window}LL) / {key_tile}U * {key_tile}U : 0U)"),
        );
        let causal = if attention.causal { "true" } else { "false" };
        let first_position = if window.is_some() {
            format!("const long long first_position = delta + first_row / {groups}U;\n")
        } else {
            String::new()
        };
        let row = dim + PAD;
        let (steps, fragments) = (dim / 16, dim / 8);
        let threads = u32::try_from(warps * 32)
            .map_err(|_| Error::invalid_argument("cuda attention threads"))?;

        let mut writer = KernelWriter::new();
        writer.require(CompileRequirements::TENSOR_CORES);
        writer.min_blocks(min_blocks);
        let PagedInputs {
            qs,
            ks,
            vs,
            qo,
            po,
            pg,
            lp,
        } = paged_inputs(&mut writer, inputs)?;
        let os = writer.output(data_type, "output")?;
        // A fused kernel's module is what nothing reads the log-sum-exp of, so it goes to scratch.
        let ls = if rotation.is_some() {
            writer.scratch(DataType::F32, tokens * heads, "lse")?
        } else {
            writer.output(DataType::F32, "lse")?
        };
        let rotation_operands = rotation
            .map(|rotation| {
                Ok::<_, Error>((
                    writer.input(rotation.cos.data_type, "cos")?,
                    writer.input(rotation.sin.data_type, "sin")?,
                    rotation
                        .norm
                        .as_ref()
                        .and_then(|norm| norm.weight.as_ref())
                        .map(|weight| writer.input(weight.data_type, "query_weight"))
                        .transpose()?,
                ))
            })
            .transpose()?;
        writer.status();
        let mut launch = writer.blocks(blocks, threads);
        launch.shared_bytes = tile_launch_bytes(parameters, row)?;
        let report = report_index_error();
        let page_total = cache_pages(inputs)?;
        let sequence_values = paged_sequence([&qo, &po, &lp], inputs, tokens, page_size, groups)?;
        let page_type = value_type(pages.data_type)?;
        writer.declare(&tensor_core_helpers(kind));
        writer.declare(&format!(
            "
// Copies 64 keys from `first` of a sequence whose pages start at `page_start` into a padded shared tile, each key from its page's slot, zeros past `keys`.
__device__ __forceinline__ void singe_load_page_tile(unsigned short (*tile)[{row}], const unsigned short* cache, const {page_type}* pages, long long page_start, unsigned int first, unsigned int keys, unsigned int kv_head, unsigned int* status) {{
    for (unsigned int chunk = threadIdx.x; chunk < {tile_chunks}U; chunk += blockDim.x) {{
        const unsigned int r = chunk / {row_chunks}U, column = (chunk % {row_chunks}U) * 8U;
        const unsigned int key = first + r;
        const bool wanted = key < keys;
        const unsigned long long page = wanted ? static_cast<unsigned long long>(pages[page_start + key / {page_size}U]) : 0ULL;
        const bool valid = wanted && page < {page_total}ULL;
        if (wanted && !valid) {{ {report} }}
        const unsigned long long offset = ((page * {page_size}ULL + key % {page_size}U) * {kv_heads}ULL + kv_head) * {dim}ULL + column;
        singe_copy_async(singe_shared_address(&tile[r][column]), valid ? cache + offset : cache, valid);
    }}
}}",
            tile_chunks = key_tile * dim / 8,
            row_chunks = dim / 8,
        ));
        let query_stage = match (rotation, rotation_operands.as_ref()) {
            (Some(rotation), Some((cos, sin, weight))) => rotated_query_stage(
                rotation,
                (&qs, cos, sin, weight.as_deref()),
                data_type,
                [dim, groups, heads],
            )?,
            _ => format!(
                "// The queries pass through the key tile into each warp's registers, 16 rows a warp.
for (unsigned int chunk = threadIdx.x; chunk < {tile_chunks}U; chunk += blockDim.x) {{
    const unsigned int r = chunk / {row_chunks}U, column = (chunk % {row_chunks}U) * 8U;
    const unsigned int query_row = first_row + r;
    const bool valid = query_row < packed;
    const unsigned long long offset = ((static_cast<unsigned long long>(query_start) + query_row / {groups}U) * {heads}ULL + kv_head * {groups}U + query_row % {groups}U) * {dim}ULL + column;
    singe_copy_async(singe_shared_address(&key_tile[r][column]), valid ? {qs} + offset : {qs}, valid);
}}
asm volatile(\"cp.async.commit_group;\\n\" ::);
asm volatile(\"cp.async.wait_group 0;\\n\" ::);
__syncthreads();",
                tile_chunks = key_tile * dim / 8,
                row_chunks = dim / 8,
            ),
        };
        writer.lines(&format!(
            "
{tiles_declared}
const unsigned int warp = lane / 32;
const unsigned int warp_lane = lane % 32;
const unsigned int sequence = blockIdx.x / ({kv_heads}U * {tiles}U);
const unsigned int kv_head = (blockIdx.x / {tiles}U) % {kv_heads}U;
const unsigned int first_row = (blockIdx.x % {tiles}U) * {query_rows}U;
if (sequence >= {sequences}U) return;
// The sequence's queries, `{groups}` rows each, and its keys, all of its pages full but the last.
{sequence_values}
// The queries are the sequence's last tokens, so query `q` sits at key position `keys - queries + q`.
const long long delta = static_cast<long long>(keys) - static_cast<long long>(queries);
{query_stage}
{query_fragments}
// Each lane holds two rows of the warp's 16, `group` and `group + 8`.
const unsigned int group = warp_lane / 4;
const unsigned int row_low = first_row + warp * 16 + group;
const unsigned int rows[2] = {{row_low, row_low + 8}};
const long long positions[2] = {{delta + row_low / {groups}U, delta + (row_low + 8) / {groups}U}};
{accumulators}
// A causal block's last key is its last query's.
{first_position}const long long last_position = delta + (min(first_row + {query_rows}U, packed) - 1U) / {groups}U;
const unsigned int key_end = {causal} ? (last_position < 0LL ? 0U : min(keys, static_cast<unsigned int>(last_position + 1LL))) : keys;
{key_loop}
{COMBINE_ROW_TOTALS}
// A row that sees no key stays zero.
#pragma unroll
for (int j = 0; j < {fragments}; ++j)
#pragma unroll
    for (int h = 0; h < 2; ++h) {{
        if (rows[h] >= packed) continue;
        const float inverse = total[h] > 0.0f ? 1.0f / total[h] : 0.0f;
        const unsigned int column = j * 8 + (warp_lane % 4) * 2;
        const unsigned long long offset = ((static_cast<unsigned long long>(query_start) + rows[h] / {groups}U) * {heads}ULL + kv_head * {groups}U + rows[h] % {groups}U) * {dim}ULL + column;
        *reinterpret_cast<unsigned int*>({os} + offset) = singe_pack(accumulated[j][h * 2] * inverse, accumulated[j][h * 2 + 1] * inverse);
    }}
// A row that sees no key has a log-sum-exp of negative infinity.
#pragma unroll
for (int h = 0; h < 2; ++h) {{
    if (rows[h] >= packed || warp_lane % 4 != 0) continue;
    const unsigned long long row = (static_cast<unsigned long long>(query_start) + rows[h] / {groups}U) * {heads}ULL + kv_head * {groups}U + rows[h] % {groups}U;
    {ls}[row] = total[h] > 0.0f ? largest[h] + log2f(total[h]) : -__uint_as_float(0x7f800000U);
}}",
            query_fragments = load_query_fragments(steps),
            accumulators = row_accumulators(fragments),
            tiles_declared = shared_tiles(parameters, row),
            key_loop = key_tile_loop(
                &first_key,
                (key_tile, stages),
                &|first| format!(
                    "    singe_load_page_tile(key_tile, {ks}, {pg}, page_start, {first}, keys, kv_head, status);
    asm volatile(\"cp.async.commit_group;\\n\" ::);
    singe_load_page_tile(value_tile, {vs}, {pg}, page_start, {first}, keys, kv_head, status);
    asm volatile(\"cp.async.commit_group;\\n\" ::);"
                ),
                &fold_key_tile(
                &format!("key >= keys || rows[r / 2] >= packed || ({causal} && static_cast<long long>(key) > positions[r / 2]) || ({windowed})"),
                &scaled,
                steps,
                fragments,
                (key_tile, stages),
            ),
            ),
        ));
        let output = ValueSpec::new(data_type, query.layout.shape().clone());
        if rotation.is_some() {
            return Ok(writer.finish(
                "rotary_paged_attention",
                launch,
                vec![output],
                inputs,
                &[Operation::PagedAttention],
            ));
        }

        Ok(writer.finish(
            "paged_attention",
            launch,
            vec![
                output,
                ValueSpec::new(
                    DataType::F32,
                    Shape::create([problem.tokens, problem.heads])?,
                ),
            ],
            inputs,
            &[Operation::PagedAttention],
        ))
    }
}

// What a kernel normalizes and rotates its query by, before it attends with it.
#[derive(Debug, Clone)]
pub struct QueryRotation {
    pub cos: KernelInput,
    pub sin: KernelInput,
    pub norm: Option<RotationNorm>,
}

// The statements that put a block's rotated queries in the key tile, in place of copying them there.
// Each row belongs to two lanes of its warp, which take its pairs in turn, so a row's norm is one shuffle away.
fn rotated_query_stage(
    rotation: &QueryRotation,
    (query, cos, sin, weight): (&str, &str, &str, Option<&str>),
    data_type: DataType,
    [dim, groups, heads]: [usize; 3],
) -> Result<String> {
    let element = element_type(data_type)?;
    let ty = compute_type(element);
    let half = dim / 2;
    let load = |offset: usize| {
        format!(
            "static_cast<{ty}>({})",
            load_expression(element, &format!("{query}[from + pair + {offset}LL]"))
        )
    };
    let (load_first, load_second) = (load(0), load(half));
    let table = |name: &str, input: &KernelInput| -> Result<String> {
        let strides = input.layout.strides();
        Ok(format!(
            "static_cast<{ty}>({})",
            load_expression(
                element_type(input.data_type)?,
                &format!("{name}[token * {}LL + pair * {}LL]", strides[0], strides[2])
            )
        ))
    };
    let (load_cos, load_sin) = (table(cos, &rotation.cos)?, table(sin, &rotation.sin)?);
    let (scale, scaled) = rotation.norm.as_ref().map_or_else(
        || (String::new(), ""),
        |norm| {
            (
                format!(
                    "    {ty} squares = 0;
    if (valid) {{
        for (long long pair = warp_lane % 2U; pair < {half}LL; pair += 2LL) {{
            const {ty} first = {load_first};
            const {ty} second = {load_second};
            squares += first * first + second * second;
        }}
    }}
    squares += __shfl_xor_sync(0xffffffffU, squares, 1, 32);
    const {ty} scale = rsqrt(squares / static_cast<{ty}>({dim}) + static_cast<{ty}>({eps:?}));
",
                    eps = f64::from(norm.eps),
                ),
                " * scale",
            )
        },
    );
    let weight_of = |offset: usize| -> Result<String> {
        let (Some(name), Some(input)) = (
            weight,
            rotation.norm.as_ref().and_then(|norm| norm.weight.as_ref()),
        ) else {
            return Ok(String::new());
        };
        Ok(format!(
            " * static_cast<{ty}>({})",
            load_expression(
                element_type(input.data_type)?,
                &format!(
                    "{name}[(pair + {offset}LL) * {}LL]",
                    input.layout.strides()[0]
                )
            )
        ))
    };
    let (weight_first, weight_second) = (weight_of(0)?, weight_of(half)?);
    Ok(format!(
        "{{
    const unsigned int r = warp * 16U + warp_lane / 2U;
    const unsigned int query_row = first_row + r;
    const bool valid = query_row < packed;
    const long long token = static_cast<long long>(query_start) + query_row / {groups}U;
    const long long from = (token * {heads}LL + kv_head * {groups}U + query_row % {groups}U) * {dim}LL;
{scale}    for (long long pair = warp_lane % 2U; pair < {half}LL; pair += 2LL) {{
        if (!valid) {{
            key_tile[r][pair] = 0;
            key_tile[r][pair + {half}LL] = 0;
            continue;
        }}

        const {ty} a = {load_first}{scaled}{weight_first};
        const {ty} b = {load_second}{scaled}{weight_second};
        const {ty} cosine = {load_cos};
        const {ty} sine = {load_sin};
        const {ty} first = a * cosine - b * sine;
        const {ty} second = b * cosine + a * sine;
        key_tile[r][pair] = {store_first};
        key_tile[r][pair + {half}LL] = {store_second};
    }}
}}
__syncthreads();",
        store_first = store_expression(element, "first"),
        store_second = store_expression(element, "second"),
    ))
}

struct PagedInputs {
    qs: String,
    ks: String,
    vs: String,
    qo: String,
    po: String,
    pg: String,
    lp: String,
}

fn paged_inputs(writer: &mut KernelWriter, inputs: &[KernelInput]) -> Result<PagedInputs> {
    // Both kernels read the queries, the caches, and the paging values first, in this order.
    let [
        query,
        _,
        _,
        query_offsets,
        page_offsets,
        pages,
        last_page_lengths,
        ..,
    ] = inputs
    else {
        return Err(Error::missing_input("cuda paged attention"));
    };
    Ok(PagedInputs {
        qs: writer.input(query.data_type, "query_rows")?,
        ks: writer.input(query.data_type, "key_cache")?,
        vs: writer.input(query.data_type, "value_cache")?,
        qo: writer.input(query_offsets.data_type, "query_offsets")?,
        po: writer.input(page_offsets.data_type, "page_offsets")?,
        pg: writer.input(pages.data_type, "pages")?,
        lp: writer.input(last_page_lengths.data_type, "last_page_lengths")?,
    })
}

fn paged_sequence(
    [query_offsets, page_offsets, last_page_lengths]: [&str; 3],
    inputs: &[KernelInput],
    tokens: usize,
    page_size: usize,
    rows_per_query: usize,
) -> Result<String> {
    let pages = inputs
        .get(5)
        .ok_or_else(|| Error::missing_input("cuda paged attention pages"))?;
    Ok(PagedSequence {
        query_offsets,
        page_offsets,
        last_page_lengths,
        tokens,
        page_table: to_usize(
            pages.layout.shape().dimensions()[0],
            "cuda paged attention page table",
        )?,
        page_size,
        rows_per_query,
    }
    .source())
}

fn cache_pages(inputs: &[KernelInput]) -> Result<usize> {
    let key_cache = inputs
        .get(1)
        .ok_or_else(|| Error::missing_input("cuda paged attention key cache"))?;
    to_usize(
        key_cache.layout.shape().dimensions()[0],
        "cuda paged attention pages",
    )
}

#[expect(
    clippy::cast_precision_loss,
    reason = "head dimensions are at most 128"
)]
fn default_scale(dim: usize) -> f32 {
    1.0 / (dim as f32).sqrt()
}

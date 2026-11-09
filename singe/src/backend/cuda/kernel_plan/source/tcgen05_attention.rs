// Scaled dot-product attention on Blackwell's tensor engine, for targets that assemble `tcgen05`, after FlashInfer's `attention/blackwell/` kernels, Apache-2.0, with `tcgen05.rs`'s operands.
// Those feed shared memory with TMA from warps of their own; here every thread loads with `cp.async`, as in `tcgen05.rs`, and every thread owns one of a block's 128 rows, which keeps the softmax in its registers.
// No GPU here runs this kernel, so no test covers it; only `ptxas` has checked it.

use singe_codegen::emit::cuda_c::spelling::store_expression;

use crate::{
    backend::{
        cuda::{
            codegen::{CompileRequirements, KernelLaunch},
            kernel_plan::source::{
                tcgen05::{
                    HELPERS, MN_MAJOR_DESCRIPTOR, SHARED_DESCRIPTOR, instruction_descriptor,
                    operand_format,
                },
                tensor_core::tensor_core_helpers,
                writer::KernelWriter,
            },
        },
        kernel_lowering::element_type,
    },
    data_type::DataType,
    error::{Error, Result},
    utility::to_u32,
};

// The 128 rows, and the 128 keys, a block holds at once.
const TILE: usize = 128;

// The columns of tensor memory the scores take, before the output's.
const SCORE_COLUMNS: usize = 128;

// A swizzled atom holds 8 rows of 64 values, and a tile 16 such atoms.
const ATOM_VALUES: usize = TILE * 64;

// What differs between plain and paged attention, as statements over the names the shared body defines.
pub struct EngineAttention<'a> {
    pub data_type: DataType,
    pub dim: usize,
    pub blocks: usize,
    // Defines `first_row`, `packed`, and `keys`, and returns from a block with no rows.
    pub setup: &'a str,
    // The position of `row` among its sequence's keys, as a `long long`.
    pub position: &'a str,
    // Define `valid` and `source` for chunk `column` of tile row `r`.
    pub query_source: &'a str,
    pub key_source: &'a str,
    pub value_source: &'a str,
    // Whether `key` is hidden from this thread's row, given `row_valid` and `position`.
    pub hidden: &'a str,
    // The kept score of `scores[column]`.
    pub score: &'a str,
    // Stores `value` at column `at` of `row`.
    pub store: &'a str,
    // What a thread does last, given `row`, `row_valid`, `largest`, and `total`.
    pub finish: &'a str,
}

// The statements that copy a tile of `TILE` rows of `dim` values into shared memory, in `SWIZZLE_128B` atoms.
fn load_tile(tile: &str, source: &str, dim: usize) -> String {
    let row_chunks = dim / 8;
    let chunks = TILE * row_chunks;
    format!(
        "for (unsigned int chunk = lane; chunk < {chunks}U; chunk += {TILE}U) {{
    const unsigned int r = chunk / {row_chunks}U, part = chunk % {row_chunks}U;
    const unsigned long long column = part * 8ULL;
    const unsigned int placed = (part / 8U) * {ATOM_VALUES}U + r * 64U + ((part % 8U) ^ (r & 7U)) * 8U;
    {source}
    singe_copy_async(singe_shared_address(&{tile}[placed]), source, valid);
}}
asm volatile(\"cp.async.commit_group;\\n\" ::);"
    )
}

// Writes the body of the kernel and returns its launch; the caller has declared its buffers and finishes the plan.
pub fn engine_attention(
    writer: &mut KernelWriter,
    attention: &EngineAttention<'_>,
) -> Result<KernelLaunch> {
    let EngineAttention {
        data_type,
        dim,
        blocks,
        setup,
        position,
        query_source,
        key_source,
        value_source,
        hidden,
        score,
        store,
        finish,
    } = *attention;
    let format = operand_format(data_type)
        .ok_or_else(|| Error::unsupported_data_type("cuda tensor engine attention"))?;
    let element = element_type(data_type)?;
    let weight = store_expression(element, "weight");
    // The operands' tiles and the scores, which a product reads as 128 keys in two atoms of tiles.
    let atoms = dim.div_ceil(64);
    let operand_values = atoms * ATOM_VALUES;
    let shared_bytes = (3 * operand_values + 2 * ATOM_VALUES) * 2;
    let mut launch = writer.blocks(
        blocks,
        u32::try_from(TILE).map_err(|_| Error::out_of_range("cuda tile"))?,
    );
    launch.shared_bytes = to_u32(shared_bytes, "cuda tensor engine attention shared memory")?;
    writer.require(CompileRequirements::TENSOR_ENGINE);
    writer.declare(&tensor_core_helpers(if data_type == DataType::F16 {
        "f16"
    } else {
        "bf16"
    }));
    writer.declare(HELPERS);
    let score_instruction = instruction_descriptor(format, 128, false);
    let value_instruction = instruction_descriptor(format, to_u32(dim, "cuda head")?, true);
    let body = format!(
        "
extern __shared__ __align__(1024) unsigned char singe_shared[];
__shared__ __align__(8) unsigned long long finished;
__shared__ unsigned int tensor_memory;
unsigned short* query_tile = reinterpret_cast<unsigned short*>(singe_shared);
unsigned short* key_tile = query_tile + {operand_values}ULL;
unsigned short* value_tile = key_tile + {operand_values}ULL;
unsigned short* weight_tile = value_tile + {operand_values}ULL;
{setup}
const unsigned int warp = singe_warp_uniform(lane / 32U);
const unsigned int warp_lane = lane % 32U;
const unsigned int quarter = warp % 4U;
const unsigned int row = first_row + quarter * 32U + warp_lane;
const bool row_valid = row < packed;
const long long position = {position};
const unsigned int barrier = singe_shared_address(&finished);
if (lane == 0U) {{
    asm volatile(\"mbarrier.init.shared::cta.b64 [%0], 1;\" :: \"r\"(barrier));
}}
if (warp == 0U) {{
    asm volatile(\"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;\" :: \"r\"(singe_shared_address(&tensor_memory)) : \"memory\");
    asm volatile(\"tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\");
}}
singe_tcgen05_before_sync();
__syncthreads();
singe_tcgen05_after_sync();
const unsigned int score_memory = tensor_memory;
const unsigned int output_memory = score_memory + {SCORE_COLUMNS}U;
const unsigned int my_rows = static_cast<unsigned int>(quarter * 32U) << 16U;
{load_query}
asm volatile(\"cp.async.wait_group 0;\\n\" ::);
singe_tcgen05_before_sync();
__syncthreads();
singe_tcgen05_after_sync();
const float infinity = __uint_as_float(0x7f800000U);
float largest = -infinity;
float total = 0.0f;
unsigned int phase = 0U;
bool started = false;
for (unsigned int first_key = 0U; first_key < keys; first_key += {TILE}U) {{
    {load_key}
    {load_value}
    asm volatile(\"cp.async.wait_group 0;\\n\" ::);
    singe_tcgen05_before_sync();
    __syncthreads();
    singe_tcgen05_after_sync();
    // Scores: 128 rows of queries against 128 keys, in the depth of a head.
    if (warp == 0U) {{
#pragma unroll
        for (unsigned int depth = 0U; depth < {depths}U; ++depth) {{
            const unsigned int offset = (depth / 4U) * {atom_bytes}U + (depth % 4U) * 32U;
            singe_tcgen05_mma(score_memory, {SHARED_DESCRIPTOR}ULL | ((singe_shared_address(query_tile) + offset) >> 4U), {SHARED_DESCRIPTOR}ULL | ((singe_shared_address(key_tile) + offset) >> 4U), {score_instruction}U, static_cast<int>(depth));
        }}
        singe_tcgen05_commit(barrier);
        __syncwarp();
    }}
    singe_barrier_wait(barrier, phase);
    phase ^= 1U;
    singe_tcgen05_after_sync();
    // This thread's row, folded into its running maximum and sum.
    float scores[{TILE}];
    for (unsigned int first = 0U; first < {TILE}U; first += 16U) {{
        float values[16];
        singe_tmem_load(values, score_memory + my_rows + first);
#pragma unroll
        for (unsigned int column = 0U; column < 16U; ++column) scores[first + column] = values[column];
    }}
    float tile_largest = largest;
    for (unsigned int column = 0U; column < {TILE}U; ++column) {{
        const unsigned int key = first_key + column;
        const bool is_hidden = {hidden};
        scores[column] = is_hidden ? -infinity : {score};
        tile_largest = fmaxf(tile_largest, scores[column]);
    }}
    const float base = tile_largest == -infinity ? 0.0f : tile_largest;
    const float rescale = exp2f(largest - base);
    largest = tile_largest;
    total *= rescale;
    // The weights go to shared memory, where the next product reads them, and what is already accumulated rescales to the new maximum.
    for (unsigned int column = 0U; column < {TILE}U; ++column) {{
        const float weight = exp2f(scores[column] - base);
        total += weight;
        const unsigned int local = column % 64U;
        weight_tile[(column / 64U) * {ATOM_VALUES}U + (row - first_row) * 64U + (((local / 8U) ^ ((row - first_row) & 7U)) * 8U) + local % 8U] = {weight};
    }}
    if (started) {{
        for (unsigned int first = 0U; first < {dim}U; first += 16U) {{
            float values[16];
            singe_tmem_load(values, output_memory + my_rows + first);
#pragma unroll
            for (unsigned int column = 0U; column < 16U; ++column) values[column] *= rescale;
            singe_tmem_store(values, output_memory + my_rows + first);
        }}
    }}
    asm volatile(\"fence.proxy.async.shared::cta;\" ::: \"memory\");
    singe_tcgen05_before_sync();
    __syncthreads();
    singe_tcgen05_after_sync();
    // The output: the weights against the values, whose rows are keys, so the right operand is MN-major.
    if (warp == 0U) {{
#pragma unroll
        for (unsigned int depth = 0U; depth < 8U; ++depth) {{
            const unsigned int weight_offset = (depth / 4U) * {ATOM_VALUES}U * 2U + (depth % 4U) * 32U;
            const unsigned int value_offset = depth * 2048U;
            singe_tcgen05_mma(output_memory, {SHARED_DESCRIPTOR}ULL | ((singe_shared_address(weight_tile) + weight_offset) >> 4U), {MN_MAJOR_DESCRIPTOR}ULL | ((singe_shared_address(value_tile) + value_offset) >> 4U), {value_instruction}U, static_cast<int>(started) | static_cast<int>(depth));
        }}
        singe_tcgen05_commit(barrier);
        __syncwarp();
    }}
    singe_barrier_wait(barrier, phase);
    phase ^= 1U;
    singe_tcgen05_after_sync();
    __syncthreads();
    started = true;
}}
for (unsigned int first = 0U; first < {dim}U; first += 16U) {{
    float values[16];
    singe_tmem_load(values, output_memory + my_rows + first);
#pragma unroll
    for (unsigned int column = 0U; column < 16U; ++column) {{
        const unsigned int at = first + column;
        const float value = started && total > 0.0f ? values[column] / total : 0.0f;
        if (row_valid) {{
            {store}
        }}
    }}
}}
{finish}
singe_tcgen05_before_sync();
__syncthreads();
if (warp == 0U) {{
    asm volatile(\"tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;\" :: \"r\"(score_memory));
}}",
        load_query = load_tile("query_tile", query_source, dim),
        load_key = load_tile("key_tile", key_source, dim),
        load_value = load_tile("value_tile", value_source, dim),
        depths = dim / 16,
        atom_bytes = ATOM_VALUES * 2,
    );
    writer.lines(&body);
    Ok(launch)
}

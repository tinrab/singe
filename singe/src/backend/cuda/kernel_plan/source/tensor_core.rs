// What the tensor-core attention kernels share: when they apply, the device helpers, and the C++ of their common phases.
// A block holds a warp's query rows in registers as `mma.sync` fragments, and folds each tile of keys into a running softmax, so the scores never leave registers.

use std::f32::consts::LOG2_E;

use crate::{
    backend::cuda::kernel_plan::source::{
        KernelInput, KernelPlanner,
        parameters::{AttentionParameters, MAX_STATIC_SHARED},
    },
    data_type::DataType,
    error::Result,
    utility::to_u32,
};

impl KernelPlanner {
    // Whether a tensor-core kernel can run on `operands`: contiguous, aligned half-precision floats of one type, on a target with `sm_80`'s `mma.sync`.
    pub fn takes_tensor_cores(&self, operands: &[&KernelInput]) -> bool {
        let Some(first) = operands.first() else {
            return false;
        };
        matches!(first.data_type, DataType::F16 | DataType::Bf16)
            && operands.iter().all(|input| {
                input.data_type == first.data_type
                    && input.layout.is_contiguous()
                    && input.layout.storage_offset().is_multiple_of(8)
            })
            && self
                .context
                .target
                .is_none_or(|target| target.compute_capability() >= (8, 0))
    }
}

// The `mma.sync` operand kind of `data_type`, one of the types [`KernelPlanner::takes_tensor_cores`] takes.
pub fn operand_kind(data_type: DataType) -> &'static str {
    if data_type == DataType::F16 {
        "f16"
    } else {
        "bf16"
    }
}

// The scaled score of `scores[j][r]`: scores are scaled into base two, so the softmax uses `exp2`, and a cap bounds them in natural units.
pub fn scaled_score_expression(scale: f32, softcap: Option<f32>) -> String {
    softcap.map_or_else(
        || format!("scores[j][r] * {:?}f", scale * LOG2_E),
        |cap| {
            format!(
                "{cap:?}f * tanhf(scores[j][r] * {inverse:?}f) * {LOG2_E:?}f",
                inverse = scale / cap
            )
        },
    )
}

// The device helpers: shared-memory addresses, asynchronous 16-byte copies, `mma.sync` m16n8k16 products of `kind` operands into `F32`, and packing two values into an operand fragment.
pub fn tensor_core_helpers(kind: &str) -> String {
    format!(
        "
__device__ __forceinline__ unsigned int singe_shared_address(const void* pointer) {{
    return static_cast<unsigned int>(__cvta_generic_to_shared(pointer));
}}
// Copies 16 bytes from global to shared memory, or zeros when `valid` is false.
__device__ __forceinline__ void singe_copy_async(unsigned int destination, const void* source, bool valid) {{
    asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16, %2;\\n\" :: \"r\"(destination), \"l\"(source), \"r\"(valid ? 16 : 0));
}}
__device__ __forceinline__ void singe_mma(float* accumulator, const unsigned int* a, const unsigned int* b) {{
    asm volatile(
        \"mma.sync.aligned.m16n8k16.row.col.f32.{kind}.{kind}.f32 {{%0, %1, %2, %3}}, {{%4, %5, %6, %7}}, {{%8, %9}}, {{%0, %1, %2, %3}};\\n\"
        : \"+f\"(accumulator[0]), \"+f\"(accumulator[1]), \"+f\"(accumulator[2]), \"+f\"(accumulator[3])
        : \"r\"(a[0]), \"r\"(a[1]), \"r\"(a[2]), \"r\"(a[3]), \"r\"(b[0]), \"r\"(b[1]));
}}
// Two values rounded to the operand type, the first in the low half, as fragments hold them.
__device__ __forceinline__ unsigned int singe_pack(float low, float high) {{
    unsigned int packed;
    asm(\"cvt.rn.{kind}x2.f32 %0, %1, %2;\" : \"=r\"(packed) : \"f\"(high), \"f\"(low));
    return packed;
}}"
    )
}

// Reads the warp's 16 query rows out of `key_tile`, where they were loaded, into each lane's registers.
pub fn load_query_fragments(steps: usize) -> String {
    format!(
        "unsigned int query_fragments[{steps}][4];
#pragma unroll
for (int step = 0; step < {steps}; ++step) {{
    const unsigned int address = singe_shared_address(&key_tile[warp * 16 + warp_lane % 16][step * 16 + (warp_lane / 16) * 8]);
    asm volatile(\"ldmatrix.sync.aligned.x4.m8n8.shared.b16 {{%0, %1, %2, %3}}, [%4];\\n\" : \"=r\"(query_fragments[step][0]), \"=r\"(query_fragments[step][1]), \"=r\"(query_fragments[step][2]), \"=r\"(query_fragments[step][3]) : \"r\"(address));
}}
__syncthreads();"
    )
}

// Declares each row's running output, largest score, and sum.
pub fn row_accumulators(fragments: usize) -> String {
    format!(
        "float accumulated[{fragments}][4];
#pragma unroll
for (int j = 0; j < {fragments}; ++j)
#pragma unroll
    for (int r = 0; r < 4; ++r) accumulated[j][r] = 0.0f;
float largest[2] = {{-__uint_as_float(0x7f800000U), -__uint_as_float(0x7f800000U)}};
float total[2] = {{0.0f, 0.0f}};"
    )
}

// Folds the 64 keys of `key_tile` and values of `value_tile` into each row's accumulators, inside a loop whose `first_key` is the tile's.
//
// `hidden` says whether key `key` is hidden from row `r / 2`, and `score` is a kept score of `scores[j][r]`.
pub fn fold_key_tile(
    hidden: &str,
    score: &str,
    steps: usize,
    fragments: usize,
    (key_tile_keys, stages): (usize, usize),
) -> String {
    let value_wait = 2 * (stages - 1);
    let key_fragments = key_tile_keys / 8;
    let key_steps = key_tile_keys / 16;
    format!(
        "    float scores[{key_fragments}][4];
#pragma unroll
    for (int j = 0; j < {key_fragments}; ++j) {{
#pragma unroll
        for (int r = 0; r < 4; ++r) scores[j][r] = 0.0f;
#pragma unroll
        for (int step = 0; step < {steps}; ++step) {{
            unsigned int b[2];
            const unsigned int address = singe_shared_address(&key_tile[j * 8 + warp_lane % 8][step * 16 + ((warp_lane / 8) % 2) * 8]);
            asm volatile(\"ldmatrix.sync.aligned.x2.m8n8.shared.b16 {{%0, %1}}, [%2];\\n\" : \"=r\"(b[0]), \"=r\"(b[1]) : \"r\"(address));
            singe_mma(scores[j], query_fragments[step], b);
        }}
    }}
    // Scale, mask keys past the end and, for a causal mask, after the query, and fold the tile into each row's running maximum and sum.
    float tile_largest[2] = {{largest[0], largest[1]}};
#pragma unroll
    for (int j = 0; j < {key_fragments}; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) {{
            const unsigned int key = first_key + j * 8 + (warp_lane % 4) * 2 + (r % 2);
            const bool hidden = {hidden};
            scores[j][r] = hidden ? -__uint_as_float(0x7f800000U) : {score};
            tile_largest[r / 2] = fmaxf(tile_largest[r / 2], scores[j][r]);
        }}
#pragma unroll
    for (int h = 0; h < 2; ++h) {{
        tile_largest[h] = fmaxf(tile_largest[h], __shfl_xor_sync(0xffffffffu, tile_largest[h], 1));
        tile_largest[h] = fmaxf(tile_largest[h], __shfl_xor_sync(0xffffffffu, tile_largest[h], 2));
    }}
    float rescale[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {{
        // A row with every key so far hidden stays at zero weight.
        const float base = tile_largest[h] == -__uint_as_float(0x7f800000U) ? 0.0f : tile_largest[h];
        rescale[h] = exp2f(largest[h] - base);
        largest[h] = tile_largest[h];
        total[h] *= rescale[h];
        tile_largest[h] = base;
    }}
#pragma unroll
    for (int j = 0; j < {key_fragments}; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) {{
            scores[j][r] = exp2f(scores[j][r] - tile_largest[r / 2]);
            total[r / 2] += scores[j][r];
        }}
#pragma unroll
    for (int j = 0; j < {fragments}; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) accumulated[j][r] *= rescale[r / 2];
    asm volatile(\"cp.async.wait_group {value_wait};\\n\" ::);
    __syncthreads();
    // The scores of two adjacent key fragments are the operand fragment of 16 keys.
#pragma unroll
    for (int step = 0; step < {key_steps}; ++step) {{
        const unsigned int a[4] = {{
            singe_pack(scores[2 * step][0], scores[2 * step][1]),
            singe_pack(scores[2 * step][2], scores[2 * step][3]),
            singe_pack(scores[2 * step + 1][0], scores[2 * step + 1][1]),
            singe_pack(scores[2 * step + 1][2], scores[2 * step + 1][3]),
        }};
#pragma unroll
        for (int j = 0; j < {fragments}; ++j) {{
            unsigned int b[2];
            const unsigned int address = singe_shared_address(&value_tile[step * 16 + warp_lane % 16][j * 8]);
            asm volatile(\"ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {{%0, %1}}, [%2];\\n\" : \"=r\"(b[0]), \"=r\"(b[1]) : \"r\"(address));
            singe_mma(accumulated[j], a, b);
        }}
    }}
    __syncthreads();"
    )
}

// The shared tiles of keys and values, which the kernel reads as `key_tile` and `value_tile`.
pub fn shared_tiles(parameters: AttentionParameters, row: usize) -> String {
    let AttentionParameters {
        key_tile, stages, ..
    } = parameters;
    // Past what a kernel may declare statically, its tiles live in what the launch asks for.
    if parameters.shared_bytes(row) > MAX_STATIC_SHARED {
        return format!(
            "extern __shared__ __align__(16) unsigned char dynamic_shared[];
auto key_tiles = reinterpret_cast<unsigned short (*)[{key_tile}][{row}]>(dynamic_shared);
auto value_tiles = key_tiles + {stages};
unsigned short (*key_tile)[{row}] = key_tiles[0];
unsigned short (*value_tile)[{row}] = value_tiles[0];"
        );
    }

    if stages == 1 {
        return format!(
            "__shared__ __align__(16) unsigned short key_tile[{key_tile}][{row}];
__shared__ __align__(16) unsigned short value_tile[{key_tile}][{row}];"
        );
    }
    format!(
        "__shared__ __align__(16) unsigned short key_tiles[{stages}][{key_tile}][{row}];
__shared__ __align__(16) unsigned short value_tiles[{stages}][{key_tile}][{row}];
unsigned short (*key_tile)[{row}] = key_tiles[0];
unsigned short (*value_tile)[{row}] = value_tiles[0];"
    )
}

// What a launch must ask for the kernel's tiles, and zero when they are declared statically.
//
// # Errors
//
// Returns an error if their bytes do not fit in a launch's count.
pub fn tile_launch_bytes(parameters: AttentionParameters, row: usize) -> Result<u32> {
    let bytes = parameters.shared_bytes(row);
    if bytes > MAX_STATIC_SHARED {
        to_u32(bytes, "cuda attention shared memory")
    } else {
        Ok(0)
    }
}

// The loop over key tiles from `first`, whose `load` copies the tiles at the first key it is given and commits them, and whose `fold` reads them.
// A single stage loads each tile just before folding it; more prefetch, keeping `stages` tiles on their way or landed, so each wait is for the oldest, after FlashInfer's `decode.cuh`.
// Every iteration commits, even past the last key where its copies are predicated off, since each wait counts commits.
pub fn key_tile_loop(
    first: &str,
    (key_tile, stages): (usize, usize),
    load: &dyn Fn(&str) -> String,
    fold: &str,
) -> String {
    if stages == 1 {
        return format!(
            "for (unsigned int first_key = {first}; first_key < key_end; first_key += {key_tile}U) {{
{load}
    asm volatile(\"cp.async.wait_group 1;\\n\" ::);
    __syncthreads();
{fold}
}}",
            load = load("first_key"),
        );
    }

    let key_wait = 2 * stages - 1;
    format!(
        "const unsigned int start_key = {first};
for (unsigned int prefill = 0U; prefill < {stages}U; ++prefill) {{
    key_tile = key_tiles[prefill];
    value_tile = value_tiles[prefill];
{prefill}
}}
unsigned int stage = 0U;
for (unsigned int first_key = start_key; first_key < key_end; first_key += {key_tile}U) {{
    key_tile = key_tiles[stage];
    value_tile = value_tiles[stage];
    asm volatile(\"cp.async.wait_group {key_wait};\\n\" ::);
    __syncthreads();
{fold}
{next}
    stage = (stage + 1U) % {stages}U;
}}
asm volatile(\"cp.async.wait_group 0;\\n\" ::);",
        prefill = load(&format!("start_key + prefill * {key_tile}U")),
        next = load(&format!("first_key + {stages}U * {key_tile}U")),
    )
}

// Sums each row's total across the four lanes that hold it.
pub const COMBINE_ROW_TOTALS: &str = "// Each row's sum covers the four lanes that hold it.
#pragma unroll
for (int h = 0; h < 2; ++h) {
    total[h] += __shfl_xor_sync(0xffffffffu, total[h], 1);
    total[h] += __shfl_xor_sync(0xffffffffu, total[h], 2);
}";

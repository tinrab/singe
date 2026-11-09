// A half-precision matmul on Blackwell's asynchronous tensor engine, for targets that assemble `tcgen05`.
// It follows FlashInfer's `dense_bf16_gemm_sm100_splitk.py` and vLLM's `gemm_rs_ar.py` and `csrc/qwen4_exp/hc_up_gate_mix.cu`, all Apache-2.0, with the descriptor fields of CUTLASS's `mma_sm100_desc.hpp`.
// Those load with TMA, whose tensor maps our launches do not pass, so every thread here loads with `cp.async` instead.
// No GPU here runs this kernel, so no test covers it; only `ptxas` has checked it.

use singe_codegen::emit::cuda_c::spelling::store_expression;

use crate::{
    backend::{
        cuda::{
            codegen::CompileRequirements,
            kernel_plan::source::{
                KernelInput, KernelPlan, mma::Problem, tensor_core::tensor_core_helpers,
                writer::KernelWriter,
            },
        },
        kernel_lowering::element_type,
        registry::Operation,
    },
    data_type::DataType,
    error::Result,
    value::ValueSpec,
};

// A block is four warps, one per quarter of the 128 accumulator rows in tensor memory.
const THREADS: u32 = 128;

const BLOCK_M: usize = 128;

const BLOCK_N: usize = 128;

// A shared row holds 64 values, which four 16-deep products read in turn.
const BLOCK_K: usize = 64;

const PRODUCT_K: usize = 16;

// The bytes a `SWIZZLE_128B` atom of eight rows spans.
const ATOM_BYTES: u64 = 8 * 64 * 2;

// The shared-memory descriptor's fixed fields: the stride of its atoms, version one, and `SWIZZLE_128B`.
pub const SHARED_DESCRIPTOR: u64 = ((ATOM_BYTES >> 4) << 32) | (1 << 46) | (2 << 61);

// The same for an MN-major operand, whose rows are its depth, with 16 atoms of 8 rows before its next 64 columns.
pub const MN_MAJOR_DESCRIPTOR: u64 = SHARED_DESCRIPTOR | (((16 * ATOM_BYTES) >> 4) << 16);

// The operand format of half-precision `data_type`.
pub const fn operand_format(data_type: DataType) -> Option<u32> {
    match data_type {
        DataType::F16 => Some(0),
        DataType::Bf16 => Some(1),
        _ => None,
    }
}

// The instruction descriptor of a product of 128 rows by `columns` into `F32` accumulators, whose left operand is K-major and whose right is K-major or MN-major.
pub const fn instruction_descriptor(
    operand_format: u32,
    columns: u32,
    right_mn_major: bool,
) -> u32 {
    let accumulator_is_f32 = 1 << 4;
    let right_major = if right_mn_major { 1 << 16 } else { 0 };
    let rows = (128 >> 4) << 24;
    accumulator_is_f32
        | (operand_format << 7)
        | (operand_format << 10)
        | right_major
        | ((columns >> 3) << 17)
        | rows
}

pub const HELPERS: &str = "
__device__ __forceinline__ unsigned int singe_warp_uniform(unsigned int value) {
    unsigned int result;
    asm volatile(\"shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;\" : \"=r\"(result) : \"r\"(value));
    return result;
}
__device__ __forceinline__ void singe_tcgen05_before_sync() {
    asm volatile(\"tcgen05.fence::before_thread_sync;\" ::: \"memory\");
}
__device__ __forceinline__ void singe_tcgen05_after_sync() {
    asm volatile(\"tcgen05.fence::after_thread_sync;\" ::: \"memory\");
}
// Issues one product from the warp's elected lane; `accumulate` adds to the accumulators instead of replacing them.
__device__ __forceinline__ void singe_tcgen05_mma(unsigned int accumulators, unsigned long long a, unsigned long long b, unsigned int instruction, int accumulate) {
    asm volatile(
        \"{\\n\\t\"
        \".reg .pred leader, add;\\n\\t\"
        \"elect.sync _|leader, 0xFFFFFFFF;\\n\\t\"
        \"setp.ne.b32 add, %4, 0;\\n\\t\"
        \"@leader tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, add;\\n\\t\"
        \"}\\n\"
        :: \"r\"(accumulators), \"l\"(a), \"l\"(b), \"r\"(instruction), \"r\"(accumulate));
}
// Arrives at the barrier once every product issued before it has finished.
__device__ __forceinline__ void singe_tcgen05_commit(unsigned int barrier) {
    asm volatile(
        \"{\\n\\t\"
        \".reg .pred leader;\\n\\t\"
        \"elect.sync _|leader, 0xFFFFFFFF;\\n\\t\"
        \"@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\\n\\t\"
        \"}\\n\"
        :: \"r\"(barrier) : \"memory\");
}
__device__ __forceinline__ void singe_barrier_wait(unsigned int barrier, unsigned int phase) {
    asm volatile(
        \"{\\n\\t\"
        \".reg .pred ready;\\n\\t\"
        \"WAIT_AGAIN:\\n\\t\"
        \"mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 ready, [%0], %1;\\n\\t\"
        \"@ready bra.uni WAIT_DONE;\\n\\t\"
        \"bra.uni WAIT_AGAIN;\\n\\t\"
        \"WAIT_DONE:\\n\\t\"
        \"}\\n\"
        :: \"r\"(barrier), \"r\"(phase) : \"memory\");
}
// Reads 16 accumulator columns of this lane's row from tensor memory.
__device__ __forceinline__ void singe_tmem_load(float* values, unsigned int address) {
    asm volatile(
        \"tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];\"
        : \"=f\"(values[0]), \"=f\"(values[1]), \"=f\"(values[2]), \"=f\"(values[3]), \"=f\"(values[4]), \"=f\"(values[5]), \"=f\"(values[6]), \"=f\"(values[7]),
          \"=f\"(values[8]), \"=f\"(values[9]), \"=f\"(values[10]), \"=f\"(values[11]), \"=f\"(values[12]), \"=f\"(values[13]), \"=f\"(values[14]), \"=f\"(values[15])
        : \"r\"(address));
    asm volatile(\"tcgen05.wait::ld.sync.aligned;\");
}
// Writes 16 accumulator columns of this lane's row back to tensor memory.
__device__ __forceinline__ void singe_tmem_store(const float* values, unsigned int address) {
    asm volatile(
        \"tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};\"
        :: \"r\"(address), \"f\"(values[0]), \"f\"(values[1]), \"f\"(values[2]), \"f\"(values[3]), \"f\"(values[4]), \"f\"(values[5]), \"f\"(values[6]), \"f\"(values[7]),
           \"f\"(values[8]), \"f\"(values[9]), \"f\"(values[10]), \"f\"(values[11]), \"f\"(values[12]), \"f\"(values[13]), \"f\"(values[14]), \"f\"(values[15]));
    asm volatile(\"tcgen05.wait::st.sync.aligned;\");
}";

impl KernelPlan {
    // Plans `lhs × rhs` on the tensor engine, or returns `None` for operands it does not take: half-precision matrices whose right operand is the transpose of `[n, k]` rows, as a linear layer's weight is, which keeps both operands K-major.
    pub(crate) fn tcgen05_matmul(lhs: &KernelInput, rhs: &KernelInput) -> Result<Option<Self>> {
        let data_type = lhs.data_type;
        let Some(format) = operand_format(data_type) else {
            return Ok(None);
        };
        if rhs.data_type != data_type {
            return Ok(None);
        }

        let Some(problem) = Problem::create(lhs, rhs)? else {
            return Ok(None);
        };
        if !problem.rhs_transposed {
            return Ok(None);
        }

        let Problem { m, n, k, batch, .. } = problem;
        let (tiles_m, tiles_n) = (m.div_ceil(BLOCK_M), n.div_ceil(BLOCK_N));
        let lhs_stride = if problem.lhs_batched { m * k } else { 0 };
        let rhs_stride = if problem.rhs_batched { n * k } else { 0 };
        let blocks = batch * tiles_m * tiles_n;
        let stored = store_expression(element_type(data_type)?, "values[column]");

        let mut writer = KernelWriter::new();
        writer.require(CompileRequirements::TENSOR_ENGINE);
        let a = writer.input(data_type, "lhs")?;
        let b = writer.input(data_type, "rhs")?;
        let c = writer.output(data_type, "output")?;
        let launch = writer.blocks(blocks, THREADS);
        writer.declare(&tensor_core_helpers(if data_type == DataType::F16 {
            "f16"
        } else {
            "bf16"
        }));
        writer.declare(HELPERS);
        writer.lines(&format!(
            "
__shared__ __align__(1024) unsigned short lhs_tile[{BLOCK_M} * {BLOCK_K}];
__shared__ __align__(1024) unsigned short rhs_tile[{BLOCK_N} * {BLOCK_K}];
__shared__ __align__(8) unsigned long long finished;
__shared__ unsigned int tensor_memory;
const unsigned int warp = singe_warp_uniform(lane / 32U);
const unsigned int warp_lane = lane % 32U;
const unsigned long long batch = blockIdx.x / {tiles}ULL;
const unsigned long long tile_m = (blockIdx.x % {tiles}ULL) / {tiles_n}ULL;
const unsigned long long tile_n = (blockIdx.x % {tiles}ULL) % {tiles_n}ULL;
const unsigned short* lhs_matrix = {a} + batch * {lhs_stride}ULL;
const unsigned short* rhs_matrix = {b} + batch * {rhs_stride}ULL;
const unsigned int barrier = singe_shared_address(&finished);
if (lane == 0U) {{
    asm volatile(\"mbarrier.init.shared::cta.b64 [%0], 1;\" :: \"r\"(barrier));
}}
if (warp == 0U) {{
    asm volatile(\"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], {BLOCK_N};\" :: \"r\"(singe_shared_address(&tensor_memory)) : \"memory\");
    asm volatile(\"tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\");
}}
singe_tcgen05_before_sync();
__syncthreads();
singe_tcgen05_after_sync();
const unsigned int accumulators = tensor_memory;
unsigned int phase = 0U;
for (unsigned int slice = 0U; slice < {slices}U; ++slice) {{
    // `SWIZZLE_128B` lays out 128-byte rows with 16-byte chunk `c` of row `r` at chunk `c ^ (r & 7)`, which spreads the rows a product reads across banks.
    for (unsigned int chunk = lane; chunk < {chunks}U; chunk += {THREADS}U) {{
        const unsigned int row = chunk / 8U, part = chunk % 8U;
        const unsigned int placed = (part ^ (row & 7U)) * 8U;
        const unsigned long long column = slice * {BLOCK_K}ULL + part * 8ULL;
        const unsigned long long lhs_row = tile_m * {BLOCK_M}ULL + row;
        const unsigned long long rhs_row = tile_n * {BLOCK_N}ULL + row;
        const bool lhs_valid = lhs_row < {m}ULL && column < {k}ULL;
        const bool rhs_valid = rhs_row < {n}ULL && column < {k}ULL;
        singe_copy_async(singe_shared_address(&lhs_tile[row * {BLOCK_K}U + placed]), lhs_valid ? lhs_matrix + lhs_row * {k}ULL + column : lhs_matrix, lhs_valid);
        singe_copy_async(singe_shared_address(&rhs_tile[row * {BLOCK_K}U + placed]), rhs_valid ? rhs_matrix + rhs_row * {k}ULL + column : rhs_matrix, rhs_valid);
    }}
    asm volatile(\"cp.async.commit_group;\\n\" ::);
    asm volatile(\"cp.async.wait_group 0;\\n\" ::);
    singe_tcgen05_before_sync();
    __syncthreads();
    singe_tcgen05_after_sync();
    if (warp == 0U) {{
        const unsigned long long lhs_descriptor = {descriptor}ULL | (singe_shared_address(lhs_tile) >> 4U);
        const unsigned long long rhs_descriptor = {descriptor}ULL | (singe_shared_address(rhs_tile) >> 4U);
#pragma unroll
        for (unsigned int depth = 0U; depth < {products}U; ++depth) {{
            // Each 16-deep product starts 32 bytes further into the shared rows.
            singe_tcgen05_mma(accumulators, lhs_descriptor + depth * 2ULL, rhs_descriptor + depth * 2ULL, {instruction}U, static_cast<int>(slice | depth));
        }}
        singe_tcgen05_commit(barrier);
        __syncwarp();
    }}
    singe_barrier_wait(barrier, phase);
    phase ^= 1U;
    singe_tcgen05_after_sync();
    __syncthreads();
}}
// A warp reads only its own quarter of the accumulator rows.
const unsigned int quarter = warp % 4U;
const unsigned long long row = tile_m * {BLOCK_M}ULL + quarter * 32ULL + warp_lane;
for (unsigned int first = 0U; first < {BLOCK_N}U; first += 16U) {{
    float values[16];
    singe_tmem_load(values, accumulators + (quarter * 32U << 16U) + first);
#pragma unroll
    for (unsigned int column = 0U; column < 16U; ++column) {{
        const unsigned long long at = tile_n * {BLOCK_N}ULL + first + column;
        if (row < {m}ULL && at < {n}ULL) {{
            {c}[batch * {output_stride}ULL + row * {n}ULL + at] = {stored};
        }}
    }}
}}
singe_tcgen05_before_sync();
__syncthreads();
if (warp == 0U) {{
    asm volatile(\"tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, {BLOCK_N};\" :: \"r\"(accumulators));
}}",
            tiles = tiles_m * tiles_n,
            slices = k.div_ceil(BLOCK_K),
            chunks = BLOCK_M * BLOCK_K / 8,
            descriptor = SHARED_DESCRIPTOR,
            products = BLOCK_K / PRODUCT_K,
            instruction = instruction_descriptor(format, 128, false),
            output_stride = m * n,
        ));
        Ok(Some(writer.finish(
            "tcgen05_matmul",
            launch,
            vec![ValueSpec::new(data_type, problem.output)],
            &[lhs.clone(), rhs.clone()],
            &[Operation::MatMul],
        )))
    }
}

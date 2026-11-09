// A tensor-core matmul of half-precision operands, after LeetCUDA's staged `mma.sync` HGEMM, with `F32` accumulators, edge tiles, and batches.

use singe_codegen::emit::cuda_c::spelling::store_expression;

use serde::{Deserialize, Serialize};

use crate::{
    backend::{
        cuda::{
            codegen::CompileRequirements,
            kernel_plan::source::{
                KernelInput, KernelPlan, KernelPlanner,
                parameters::MmaParameters,
                tensor_core::{operand_kind, tensor_core_helpers},
                writer::KernelWriter,
            },
        },
        kernel_lowering::element_type,
        registry::Operation,
    },
    data_type::DataType,
    error::{Error, Result},
    shape::Shape,
    utility::to_usize,
    value::ValueSpec,
};

// How a block divides its tile of the product: `warps_m · warps_n` warps, each computing `fragments_m · fragments_n` m16n8 fragments, over `block_k`-deep slices of the operands loaded `stages` ahead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MmaTiling {
    pub warps_m: usize,
    pub warps_n: usize,
    pub fragments_m: usize,
    pub fragments_n: usize,
    pub block_k: usize,
    pub stages: usize,
}

impl MmaTiling {
    // The tiling for `batch` products of `rows` by `columns`, measured on an RTX 50-series GPU against cuBLAS on FlashInfer-Bench's GEMM shapes.
    // Products with at least 64 tall tiles take 128 by 128 tiles over eight warps of 32 by 64, four 16-deep stages ahead, and the largest 256 by 128 tiles; the rest, as in decoding, take 64-row tiles, so more blocks share the weight.
    #[must_use]
    pub const fn for_product(batch: usize, rows: usize, columns: usize) -> Self {
        if batch * rows.div_ceil(128) * columns.div_ceil(128) < 64 {
            Self {
                warps_m: 2,
                warps_n: 4,
                fragments_m: 2,
                fragments_n: 4,
                block_k: 32,
                stages: 3,
            }
        } else if batch * rows.div_ceil(256) * columns.div_ceil(128) < 128 {
            Self {
                warps_m: 4,
                warps_n: 2,
                fragments_m: 2,
                fragments_n: 8,
                block_k: 16,
                stages: 4,
            }
        } else {
            // Products with at least 128 tiles of 256 by 128 take them, three 32-deep stages in about 90 KiB of dynamic shared memory, which halves how often each operand is loaded.
            Self {
                warps_m: 4,
                warps_n: 2,
                fragments_m: 4,
                fragments_n: 8,
                block_k: 32,
                stages: 3,
            }
        }
    }

    const fn block_m(self) -> usize {
        self.warps_m * self.fragments_m * 16
    }

    const fn block_n(self) -> usize {
        self.warps_n * self.fragments_n * 8
    }

    const fn threads(self) -> usize {
        self.warps_m * self.warps_n * 32
    }

    // Whether this tiling's chunks divide among its threads and its stages fit a block's shared memory.
    const fn fits(self, rhs_transposed: bool) -> bool {
        let (block_m, block_n, threads) = (self.block_m(), self.block_n(), self.threads());
        let lhs_row = self.block_k + PAD;
        let (rhs_rows, rhs_row) = if rhs_transposed {
            (block_n, self.block_k + PAD)
        } else {
            (self.block_k, block_n + PAD)
        };
        self.block_k.is_multiple_of(16)
            && (block_m * self.block_k / 8).is_multiple_of(threads)
            && (self.block_k * block_n / 8).is_multiple_of(threads)
            && self.stages >= 2
            && self.stages * (block_m * lhs_row + rhs_rows * rhs_row) * 2 <= MAX_SHARED
    }
}

// The most dynamic shared memory a block's stages take.
const MAX_SHARED: usize = 96 * 1024;

// Products of fewer tiles than this split their depth, so they still fill the GPU.
pub const SPLIT_BELOW_BLOCKS: usize = 64;

// Shared rows are padded by one 16-byte chunk, so the eight rows an `ldmatrix` reads fall in different banks; LeetCUDA's swizzled layouts reach the same without the padding.
const PAD: usize = 8;

// The operands' shapes: `batch` products of `[m, k]` by `[k, n]`, each operand either one per batch or one shared by all.
pub struct Problem {
    pub batch: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub lhs_batched: bool,
    pub rhs_batched: bool,
    // Whether the right operand is the transpose of contiguous `[n, k]` rows, as a linear layer's weight is.
    pub rhs_transposed: bool,
    pub output: Shape,
}

impl Problem {
    // The problem `lhs × rhs` poses, when both are matrices or batches of them whose 16-byte rows the kernel can copy whole.
    pub fn create(lhs: &KernelInput, rhs: &KernelInput) -> Result<Option<Self>> {
        let (left, right) = (
            lhs.layout.shape().dimensions(),
            rhs.layout.shape().dimensions(),
        );
        if left.len() < 2
            || right.len() < 2
            || !lhs.layout.is_contiguous()
            || !lhs.layout.storage_offset().is_multiple_of(8)
            || !rhs.layout.storage_offset().is_multiple_of(8)
        {
            return Ok(None);
        }

        // The right operand is contiguous `[k, n]`, or the transpose of contiguous `[n, k]`.
        let rank = i64::try_from(right.len()).map_err(|_| Error::DimensionOutOfBounds)?;
        let rhs_transposed = if rhs.layout.is_contiguous() {
            false
        } else if rhs.layout.transpose(rank - 2, rank - 1)?.is_contiguous() {
            true
        } else {
            return Ok(None);
        };
        let extent = |value: i64| to_usize(value, "cuda mma extent");
        let (m, k) = (extent(left[left.len() - 2])?, extent(left[left.len() - 1])?);
        let (rhs_k, n) = (
            extent(right[right.len() - 2])?,
            extent(right[right.len() - 1])?,
        );
        let (left_batch, right_batch) = (&left[..left.len() - 2], &right[..right.len() - 2]);
        let count = |dimensions: &[i64]| dimensions.iter().product::<i64>();
        // A batched operand gives the product's batch dimensions, and the other has fewer dimensions, all of extent one.
        let (batch_dimensions, lhs_batched, rhs_batched) =
            if count(right_batch) == 1 && right_batch.len() <= left_batch.len() {
                (left_batch, count(left_batch) > 1, false)
            } else if count(left_batch) == 1 && left_batch.len() <= right_batch.len() {
                (right_batch, false, true)
            } else if left_batch == right_batch {
                (left_batch, true, true)
            } else {
                return Ok(None);
            };
        // Up to four rows run as matrix-vector products, which read the weight once without tensor cores.
        if rhs_k != k
            || k % 8 != 0
            || n % 8 != 0
            || m * extent(count(left_batch))? <= 4
            || n == 0
            || k == 0
        {
            return Ok(None);
        }

        let mut output = batch_dimensions.to_vec();
        output.extend([left[left.len() - 2], right[right.len() - 1]]);
        Ok(Some(Self {
            batch: extent(count(batch_dimensions))?,
            m,
            n,
            k,
            lhs_batched,
            rhs_batched,
            rhs_transposed,
            output: Shape::create(output)?,
        }))
    }
}

// Blocks run in index order, so the operand with fewer tiles varies fastest: the blocks resident at once share tiles of the other, larger one, which then streams through L2 once, as a linear layer's weight does under few rows.
fn tile_order(tiles_m: usize, tiles_n: usize) -> String {
    if tiles_m <= tiles_n {
        format!(
            "const unsigned int tile_m = block % {tiles_m}U;
const unsigned int tile_n = (block / {tiles_m}U) % {tiles_n}U;"
        )
    } else {
        format!(
            "const unsigned int tile_n = block % {tiles_n}U;
const unsigned int tile_m = (block / {tiles_n}U) % {tiles_m}U;"
        )
    }
}

// How many splits a product of `base` blocks over `total_slices` slices takes, and how many slices each holds.
fn depth_splits(base: usize, total_slices: usize, split_below_blocks: usize) -> (usize, usize) {
    if base < split_below_blocks && total_slices >= 8 {
        let wanted = (2 * split_below_blocks)
            .div_ceil(base)
            .min(total_slices / 4);
        let chunk = total_slices.div_ceil(wanted);
        (total_slices.div_ceil(chunk), chunk)
    } else {
        (1, total_slices)
    }
}

impl KernelPlanner {
    // Plans `lhs × rhs` on tensor cores, or returns `None` when the operands are not `F16` or `BF16` matrices it reads whole, or the target has no `mma.sync` of 16-deep half-precision fragments, which arrived with `sm_80`.
    pub(crate) fn mma_matmul(
        &self,
        lhs: &KernelInput,
        rhs: &KernelInput,
    ) -> Result<Option<KernelPlan>> {
        self.mma_matmul_with(lhs, rhs, None)
    }

    // The tilings tuning tries for `lhs × rhs`, the default first, each with the depth split and launch bound to try; those whose shared memory or chunks do not fit are not listed.
    pub(crate) fn mma_candidates(
        &self,
        lhs: &KernelInput,
        rhs: &KernelInput,
    ) -> Result<Vec<MmaParameters>> {
        // Tuning lists only what `mma_matmul_tuned` plans, since a record it refuses is an error.
        if !self.mma_takes(lhs, rhs) {
            return Ok(Vec::new());
        }

        let Some(problem) = Problem::create(lhs, rhs)? else {
            return Ok(Vec::new());
        };
        let default = MmaTiling::for_product(problem.batch, problem.m, problem.n);
        let tiling = |warps_m, warps_n, fragments_m, fragments_n, block_k, stages| MmaTiling {
            warps_m,
            warps_n,
            fragments_m,
            fragments_n,
            block_k,
            stages,
        };
        let tilings = [
            default,
            tiling(2, 4, 2, 4, 32, 3),
            tiling(4, 2, 2, 8, 16, 4),
            tiling(4, 2, 4, 8, 32, 3),
            tiling(2, 2, 2, 4, 32, 3),
            tiling(2, 2, 2, 4, 16, 4),
            tiling(1, 4, 4, 4, 32, 3),
            tiling(2, 4, 2, 4, 16, 4),
        ];
        let mut candidates = Vec::new();
        for tiling in tilings {
            if !tiling.fits(problem.rhs_transposed) {
                continue;
            }
            for split_below_blocks in [SPLIT_BELOW_BLOCKS, 0, 256] {
                let candidate = MmaParameters {
                    tiling,
                    split_below_blocks,
                    min_blocks: None,
                };
                if !candidates.contains(&candidate) {
                    candidates.push(candidate);
                }
            }
        }

        candidates.extend(
            self.launch_bound_candidates(default.threads())
                .into_iter()
                .flatten()
                .map(|bound| MmaParameters {
                    tiling: default,
                    split_below_blocks: SPLIT_BELOW_BLOCKS,
                    min_blocks: Some(bound),
                }),
        );
        Ok(candidates)
    }

    // Plans `lhs × rhs` on tensor cores under `tiling`, or the tiling `MmaTiling::for_product` picks.
    pub(crate) fn mma_matmul_with(
        &self,
        lhs: &KernelInput,
        rhs: &KernelInput,
        tiling: Option<MmaTiling>,
    ) -> Result<Option<KernelPlan>> {
        self.mma_matmul_tuned(
            lhs,
            rhs,
            tiling.map(|tiling| MmaParameters {
                tiling,
                split_below_blocks: SPLIT_BELOW_BLOCKS,
                min_blocks: None,
            }),
        )
    }

    // Whether tensor cores take operands of these types on this target: matching `F16` or `BF16`, from `sm_80`.
    fn mma_takes(&self, lhs: &KernelInput, rhs: &KernelInput) -> bool {
        matches!(lhs.data_type, DataType::F16 | DataType::Bf16)
            && rhs.data_type == lhs.data_type
            && self
                .context
                .target
                .is_none_or(|target| target.compute_capability() >= (8, 0))
    }

    // Plans `lhs × rhs` on tensor cores under `parameters`, or what `MmaTiling::for_product` picks.
    pub(crate) fn mma_matmul_tuned(
        &self,
        lhs: &KernelInput,
        rhs: &KernelInput,
        parameters: Option<MmaParameters>,
    ) -> Result<Option<KernelPlan>> {
        if !self.mma_takes(lhs, rhs) {
            return Ok(None);
        }

        let data_type = lhs.data_type;
        let kind = operand_kind(data_type);
        let Some(problem) = Problem::create(lhs, rhs)? else {
            return Ok(None);
        };
        let MmaParameters {
            tiling,
            split_below_blocks,
            min_blocks,
        } = parameters.unwrap_or_else(|| MmaParameters {
            tiling: MmaTiling::for_product(problem.batch, problem.m, problem.n),
            split_below_blocks: SPLIT_BELOW_BLOCKS,
            min_blocks: None,
        });
        let MmaTiling {
            warps_m,
            fragments_m,
            fragments_n,
            block_k,
            stages,
            ..
        } = tiling;
        let (block_m, block_n, threads) = (tiling.block_m(), tiling.block_n(), tiling.threads());
        let (lhs_row, rhs_row) = (
            block_k + PAD,
            if problem.rhs_transposed {
                block_k
            } else {
                block_n
            } + PAD,
        );
        let (lhs_chunks, rhs_chunks) = (block_m * block_k / 8, block_k * block_n / 8);
        // The stages live in dynamic shared memory, of which a block takes at most 96 KiB on every architecture from `sm_80` on.
        let rhs_rows = if problem.rhs_transposed {
            block_n
        } else {
            block_k
        };
        let shared = stages * (block_m * lhs_row + rhs_rows * rhs_row) * 2;
        if block_k % 16 != 0
            || lhs_chunks % threads != 0
            || rhs_chunks % threads != 0
            || stages < 2
            || shared > MAX_SHARED
        {
            return Err(Error::invalid_argument("cuda mma tiling"));
        }

        let (tiles_m, tiles_n) = (problem.m.div_ceil(block_m), problem.n.div_ceil(block_n));
        let Problem { m, n, k, .. } = problem;
        let lhs_stride = if problem.lhs_batched { m * k } else { 0 };
        let rhs_stride = if problem.rhs_batched { k * n } else { 0 };
        let store = store_expression(element_type(data_type)?, "value");

        let mut writer = KernelWriter::new();
        writer.require(CompileRequirements::TENSOR_CORES);
        writer.min_blocks(min_blocks);
        let a = writer.input(data_type, "lhs")?;
        let b = writer.input(data_type, "rhs")?;
        let c = writer.output(data_type, "output")?;
        // Products whose tiles cannot fill the GPU split their depth among blocks, which leave `F32` partial products for a second kernel to sum.
        let base = problem.batch * tiles_m * tiles_n;
        let total_slices = k.div_ceil(block_k);
        let (splits, chunk) = depth_splits(base, total_slices, split_below_blocks);
        let elements = problem.batch * m * n;
        let partials = if splits > 1 {
            Some(writer.scratch(DataType::F32, splits * elements, "partials")?)
        } else {
            None
        };
        let threads =
            u32::try_from(threads).map_err(|_| Error::invalid_argument("cuda mma tiling"))?;
        let mut launch = writer.blocks(base * splits, threads);
        launch.shared_bytes =
            u32::try_from(shared).map_err(|_| Error::invalid_argument("cuda mma tiling"))?;
        writer.declare(&tensor_core_helpers(kind));
        let tile_order = tile_order(tiles_m, tiles_n);
        // A `[k, n]` right operand is copied as `k` rows of `n` and read transposed into fragments; a transposed one, `[n, k]` rows, is copied as they are and read directly, as fragments of a column-major operand hold them.
        let (rhs_load, rhs_fragment) = if problem.rhs_transposed {
            (
                format!(
                    "        const unsigned int row = chunk / {chunks}U, column = (chunk % {chunks}U) * 8U;
        const unsigned long long global_row = tile_n * {block_n}ULL + row, global_column = slice * {block_k}ULL + column;
        const bool valid = global_row < {n}ULL && global_column < {k}ULL;
        singe_copy_async(singe_shared_address(&rhs_tile[stage][row][column]), valid ? rhs_matrix + global_row * {k}ULL + global_column : rhs_matrix, valid);",
                    chunks = block_k / 8,
                ),
                "            const unsigned int address = singe_shared_address(&rhs_tile[stage][warp_column + j * 8 + warp_lane % 8][depth + ((warp_lane / 8) % 2) * 8]);
            asm volatile(\"ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\\n\" : \"=r\"(b[j][0]), \"=r\"(b[j][1]) : \"r\"(address));".to_owned(),
            )
        } else {
            (
                format!(
                    "        const unsigned int row = chunk / {chunks}U, column = (chunk % {chunks}U) * 8U;
        const unsigned long long global_row = slice * {block_k}ULL + row, global_column = tile_n * {block_n}ULL + column;
        const bool valid = global_row < {k}ULL && global_column < {n}ULL;
        singe_copy_async(singe_shared_address(&rhs_tile[stage][row][column]), valid ? rhs_matrix + global_row * {n}ULL + global_column : rhs_matrix, valid);",
                    chunks = block_n / 8,
                ),
                "            const unsigned int address = singe_shared_address(&rhs_tile[stage][depth + warp_lane % 16][warp_column + j * 8]);
            asm volatile(\"ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\\n\" : \"=r\"(b[j][0]), \"=r\"(b[j][1]) : \"r\"(address));".to_owned(),
            )
        };
        // Unsplit, each lane rounds its two adjacent columns once to the output type; split, it leaves them in `F32` for the sum.
        let finish = partials.as_ref().map_or_else(
            || {
                format!(
                    "                float value = accumulators[i][j][half * 2];
                const unsigned int low = {store};
                value = accumulators[i][j][half * 2 + 1];
                const unsigned int high = {store};
                *reinterpret_cast<unsigned int*>(product + row * {n}ULL + column) = low | (high << 16);"
                )
            },
            |partials| {
                format!(
                    "                float* partial = {partials} + split * {elements}ULL + batch * {output_stride}ULL + row * {n}ULL + column;
                partial[0] = accumulators[i][j][half * 2];
                partial[1] = accumulators[i][j][half * 2 + 1];",
                    output_stride = m * n,
                )
            },
        );
        writer.lines(&format!(
            "
extern __shared__ __align__(16) unsigned char dynamic_shared[];
auto lhs_tile = reinterpret_cast<unsigned short (*)[{block_m}][{lhs_row}]>(dynamic_shared);
auto rhs_tile = reinterpret_cast<unsigned short (*)[{rhs_rows}][{rhs_row}]>(dynamic_shared + {lhs_bytes});
// Blocks of one split of the depth are consecutive.
const unsigned int split = blockIdx.x / {base}U;
const unsigned int block = blockIdx.x % {base}U;
{tile_order}
const unsigned long long batch = block / ({tiles_n}U * {tiles_m}U);
const unsigned short* lhs_matrix = {a} + batch * {lhs_stride}ULL;
const unsigned short* rhs_matrix = {b} + batch * {rhs_stride}ULL;
unsigned short* product = {c} + batch * {output_stride}ULL;
const unsigned int warp = lane / 32;
const unsigned int warp_lane = lane % 32;
const unsigned int warp_row = (warp % {warps_m}U) * {warp_m}U;
const unsigned int warp_column = (warp / {warps_m}U) * {warp_n}U;
float accumulators[{fragments_m}][{fragments_n}][4];
#pragma unroll
for (int i = 0; i < {fragments_m}; ++i)
#pragma unroll
    for (int j = 0; j < {fragments_n}; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) accumulators[i][j][r] = 0.0f;
// Each thread copies its 16-byte chunks of slice `slice` into stage `stage`, zeros past the edges.
auto load = [&](unsigned int slice, unsigned int stage) {{
#pragma unroll
    for (unsigned int chunk = lane; chunk < {lhs_chunks}U; chunk += {threads}U) {{
        const unsigned int row = chunk / {lhs_row_chunks}U, column = (chunk % {lhs_row_chunks}U) * 8U;
        const unsigned long long global_row = tile_m * {block_m}ULL + row, global_column = slice * {block_k}ULL + column;
        const bool valid = global_row < {m}ULL && global_column < {k}ULL;
        singe_copy_async(singe_shared_address(&lhs_tile[stage][row][column]), valid ? lhs_matrix + global_row * {k}ULL + global_column : lhs_matrix, valid);
    }}
#pragma unroll
    for (unsigned int chunk = lane; chunk < {rhs_chunks}U; chunk += {threads}U) {{
{rhs_load}
    }}
}};
const unsigned int first_slice = split * {chunk}U;
const unsigned int slices = min({total_slices}U, first_slice + {chunk}U) - first_slice;
#pragma unroll
for (unsigned int slice = 0; slice < {stages}U - 1U; ++slice) {{
    if (slice < slices) load(first_slice + slice, slice);
    asm volatile(\"cp.async.commit_group;\\n\" ::);
}}
for (unsigned int slice = 0; slice < slices; ++slice) {{
    asm volatile(\"cp.async.wait_group %0;\\n\" :: \"n\"({stages} - 2));
    __syncthreads();
    // The stage refilled here was read in the previous slice, which every warp has finished.
    const unsigned int ahead = slice + {stages}U - 1U;
    if (ahead < slices) load(first_slice + ahead, ahead % {stages}U);
    asm volatile(\"cp.async.commit_group;\\n\" ::);
    const unsigned int stage = slice % {stages}U;
#pragma unroll
    for (unsigned int depth = 0; depth < {block_k}U; depth += 16U) {{
        unsigned int a[{fragments_m}][4];
        unsigned int b[{fragments_n}][2];
#pragma unroll
        for (int i = 0; i < {fragments_m}; ++i) {{
            const unsigned int address = singe_shared_address(&lhs_tile[stage][warp_row + i * 16 + warp_lane % 16][depth + (warp_lane / 16) * 8]);
            asm volatile(\"ldmatrix.sync.aligned.x4.m8n8.shared.b16 {{%0, %1, %2, %3}}, [%4];\\n\" : \"=r\"(a[i][0]), \"=r\"(a[i][1]), \"=r\"(a[i][2]), \"=r\"(a[i][3]) : \"r\"(address));
        }}
#pragma unroll
        for (int j = 0; j < {fragments_n}; ++j) {{
{rhs_fragment}
        }}
#pragma unroll
        for (int i = 0; i < {fragments_m}; ++i)
#pragma unroll
            for (int j = 0; j < {fragments_n}; ++j) singe_mma(accumulators[i][j], a[i], b[j]);
    }}
}}
asm volatile(\"cp.async.wait_group 0;\\n\" ::);
// Each lane holds two adjacent columns of two rows of every fragment, rounded once to the output type.
#pragma unroll
for (int i = 0; i < {fragments_m}; ++i)
#pragma unroll
    for (int j = 0; j < {fragments_n}; ++j)
#pragma unroll
        for (int half = 0; half < 2; ++half) {{
            const unsigned long long row = tile_m * {block_m}ULL + warp_row + i * 16 + warp_lane / 4 + half * 8;
            const unsigned long long column = tile_n * {block_n}ULL + warp_column + j * 8 + (warp_lane % 4) * 2;
            if (row < {m}ULL && column < {n}ULL) {{
{finish}
            }}
        }}",
            output_stride = m * n,
            lhs_bytes = stages * block_m * lhs_row * 2,
            warp_m = fragments_m * 16,
            warp_n = fragments_n * 8,
            lhs_row_chunks = block_k / 8,


        ));
        let output = ValueSpec::new(data_type, problem.output);
        let inputs = [lhs.clone(), rhs.clone()];
        let multiply = writer.finish(
            "mma_matmul",
            launch,
            vec![output.clone()],
            &inputs,
            &[Operation::MatMul],
        );
        if splits == 1 {
            return Ok(Some(multiply));
        }

        let mut combine = KernelWriter::new();
        let c = combine.output(data_type, "output")?;
        let partials = combine.scratch(DataType::F32, splits * elements, "partials")?;
        let launch = combine.elements(elements);
        let stored = store_expression(element_type(data_type)?, "sum");
        combine.lines(&format!(
            "
float sum = 0.0f;
for (unsigned int split = 0; split < {splits}U; ++split) sum += {partials}[split * {elements}ULL + index];
{c}[index] = {stored};"
        ));
        let mut plan = combine.finish(
            "mma_matmul_sum",
            launch,
            vec![output],
            &inputs,
            &[Operation::MatMul],
        );
        plan.before.push(multiply.kernel);
        Ok(Some(plan))
    }
}

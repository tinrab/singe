// A matmul of a few rows against a linear layer's weight as dot products, `[m, k] × [n, k]ᵀ`, after vLLM's `_ll_bf16_dotprod.py` and `_skinny_gemm.py`, FlashInfer's `dense_bf16_gemm_direct.py`, and `csrc/fp32_router_gemm.cu`: the weight is read once, a warp per output column walking its row a vector per lane, and every row reuses each value it loaded.

use singe_codegen::emit::cuda_c::spelling::{load_expression, store_expression};

use crate::{
    backend::{
        cuda::kernel_plan::source::{
            KernelInput, KernelPlan, KernelPlanner,
            parameters::{DotProductParameters, has_wide_loads},
            writer::KernelWriter,
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

// The most rows this kernel takes unless a record chooses it for more; it matched cuBLAS up to four rows on FlashInfer-Bench's weights and fell behind at eight, where tensor cores reuse each load across more of them.
const DEFAULT_MAX_ROWS: usize = 4;

// The most rows a lane's registers hold running sums for.
const MAX_ROWS: usize = 16;

// Rows times depth in shared memory, at most 48 KiB of `F32`.
const MAX_STAGED: usize = 12 * 1024;

// What this kernel reads, when it can run on these operands.
struct Problem {
    data_type: DataType,
    rows: usize,
    depth: usize,
    columns: usize,
}

impl Problem {
    fn create(lhs: &KernelInput, rhs: &KernelInput) -> Result<Option<Self>> {
        let data_type = lhs.data_type;
        let (left, right) = (
            lhs.layout.shape().dimensions(),
            rhs.layout.shape().dimensions(),
        );
        let (&[.., rows, depth], &[weight_depth, columns]) = (left, right) else {
            return Ok(None);
        };
        let leading = left[..left.len() - 2].iter().product::<i64>();
        let extent = |value: i64| to_usize(value, "cuda dot product extent");
        let (rows, depth, columns) = (extent(leading * rows)?, extent(depth)?, extent(columns)?);
        if !matches!(data_type, DataType::F16 | DataType::Bf16 | DataType::F32)
            || rhs.data_type != data_type
            || !lhs.layout.is_contiguous()
            || rhs.layout.is_contiguous()
            || !rhs.layout.transpose(0, 1)?.is_contiguous()
            || !lhs.layout.storage_offset().is_multiple_of(8)
            || !rhs.layout.storage_offset().is_multiple_of(8)
            || weight_depth != left[left.len() - 1]
            || rows == 0
            || rows > MAX_ROWS
            || columns == 0
            || depth == 0
            || !depth.is_multiple_of(8)
            || rows * depth > MAX_STAGED
        {
            return Ok(None);
        }
        Ok(Some(Self {
            data_type,
            rows,
            depth,
            columns,
        }))
    }
}

impl KernelPlanner {
    // The parameters this kernel may run `lhs × rhs` with, the default first, or none when it cannot run them.
    pub(crate) fn dot_product_candidates(
        &self,
        lhs: &KernelInput,
        rhs: &KernelInput,
    ) -> Result<Vec<DotProductParameters>> {
        let Some(problem) = Problem::create(lhs, rhs)? else {
            return Ok(Vec::new());
        };
        let wide_loads = has_wide_loads(self.context.target)
            && (problem.depth * problem.data_type.size_bytes()).is_multiple_of(32);
        let mut candidates = Vec::new();
        for warps in [8, 4, 16, 2] {
            for wide in [false, true] {
                if wide && !wide_loads {
                    continue;
                }

                candidates.push(DotProductParameters {
                    warps,
                    wide_loads: wide,
                    min_blocks: None,
                });
            }
        }

        candidates.extend(
            self.launch_bound_candidates(DotProductParameters::default().warps * 32)
                .into_iter()
                .flatten()
                .map(|bound| DotProductParameters {
                    min_blocks: Some(bound),
                    ..DotProductParameters::default()
                }),
        );
        Ok(candidates)
    }

    // Plans `lhs × rhs` as untuned matmuls run it, up to the rows dot products take by default.
    pub(crate) fn default_dot_product(
        &self,
        lhs: &KernelInput,
        rhs: &KernelInput,
    ) -> Result<Option<KernelPlan>> {
        let Some(problem) = Problem::create(lhs, rhs)? else {
            return Ok(None);
        };
        if problem.rows > DEFAULT_MAX_ROWS {
            return Ok(None);
        }
        self.dot_product(lhs, rhs, DotProductParameters::default())
    }

    // Plans `lhs × rhs` as dot products, or returns `None` for operands it does not take: up to `MAX_ROWS` contiguous floating rows of depth `k`, a multiple of eight, against a weight of contiguous `[n, k]` rows.
    pub(crate) fn dot_product(
        &self,
        lhs: &KernelInput,
        rhs: &KernelInput,
        DotProductParameters {
            warps,
            wide_loads,
            min_blocks,
        }: DotProductParameters,
    ) -> Result<Option<KernelPlan>> {
        let Some(Problem {
            data_type,
            rows,
            depth,
            columns,
        }) = Problem::create(lhs, rhs)?
        else {
            return Ok(None);
        };
        let wide = wide_loads
            && has_wide_loads(self.context.target)
            && (depth * data_type.size_bytes()).is_multiple_of(32);
        let element = element_type(data_type)?;
        let ty = if data_type == DataType::F32 {
            "float"
        } else {
            "unsigned short"
        };
        // Lanes load 16 or 32 bytes of the weight row at a time.
        let bytes = if wide { 32 } else { 16 };
        let vector = bytes / data_type.size_bytes();
        let load = |source: &str| load_expression(element, source);
        let (load_row, load_weight) = (load("staged[r * DEPTH + d + e]"), load("raw[e]"));
        let stored = store_expression(element, "sum");
        let threads = u32::try_from(warps * 32)
            .map_err(|_| Error::invalid_argument("cuda dot product threads"))?;
        // A 256-bit load needs its 32-byte alignment, which a weight's own alignment gives or does not, so a lane checks once per column.
        let read = if wide {
            format!(
                "    if (aligned) {{
        unsigned int* words = reinterpret_cast<unsigned int*>(raw);
        const void* address = weight_row + d;
        asm volatile(\"ld.global.v8.u32 {{%0, %1, %2, %3, %4, %5, %6, %7}}, [%8];\\n\" : \"=r\"(words[0]), \"=r\"(words[1]), \"=r\"(words[2]), \"=r\"(words[3]), \"=r\"(words[4]), \"=r\"(words[5]), \"=r\"(words[6]), \"=r\"(words[7]) : \"l\"(address));
    }} else {{
#pragma unroll
        for (int half = 0; half < 2; ++half)
            reinterpret_cast<uint4*>(raw)[half] = *reinterpret_cast<const uint4*>(weight_row + d + half * {half_vector});
    }}",
                half_vector = vector / 2,
            )
        } else {
            "    *reinterpret_cast<uint4*>(raw) = *reinterpret_cast<const uint4*>(weight_row + d);"
                .to_owned()
        };
        let aligned = if wide {
            "const bool aligned = (reinterpret_cast<unsigned long long>(weight_row) & 31ULL) == 0ULL;"
        } else {
            ""
        };

        let mut writer = KernelWriter::new();
        writer.min_blocks(min_blocks);
        let a = writer.input(data_type, "rows")?;
        let w = writer.input(data_type, "weight")?;
        let out = writer.output(data_type, "output")?;
        let launch = writer.blocks(columns.div_ceil(warps), threads);
        writer.lines(&format!(
            "
constexpr unsigned int DEPTH = {depth}U;
__shared__ __align__(16) {ty} staged[{staged}];
for (unsigned int i = lane; i < {staged}U; i += blockDim.x) staged[i] = {a}[i];
__syncthreads();
const unsigned int warp_lane = lane % 32;
const unsigned long long column = static_cast<unsigned long long>(blockIdx.x) * {warps}ULL + lane / 32;
if (column >= {columns}ULL) return;
float sums[{rows}];
#pragma unroll
for (int r = 0; r < {rows}; ++r) sums[r] = 0.0f;
const {ty}* weight_row = {w} + column * DEPTH;
{aligned}
for (unsigned int d = warp_lane * {vector}U; d < DEPTH; d += 32U * {vector}U) {{
    alignas({bytes}) {ty} raw[{vector}];
{read}
#pragma unroll
    for (int e = 0; e < {vector}; ++e) {{
        const float weight = {load_weight};
#pragma unroll
        for (int r = 0; r < {rows}; ++r) sums[r] += {load_row} * weight;
    }}
}}
#pragma unroll
for (int r = 0; r < {rows}; ++r) {{
    float sum = sums[r];
    for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    if (warp_lane == 0) {out}[r * {columns}ULL + column] = {stored};
}}",
            staged = rows * depth,
        ));
        let left = lhs.layout.shape().dimensions();
        let right = rhs.layout.shape().dimensions();
        let mut output = left[..left.len() - 1].to_vec();
        output.push(right[1]);
        let output = ValueSpec::new(data_type, Shape::create(output)?);
        Ok(Some(writer.finish(
            "dot_product",
            launch,
            vec![output],
            &[lhs.clone(), rhs.clone()],
            &[Operation::MatMul],
        )))
    }
}

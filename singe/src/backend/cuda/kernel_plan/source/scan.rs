use std::slice;

use singe_codegen::{
    emit::cuda_c::spelling::{combiner_tag, identity, load_expression, store_expression},
    ir::Combiner,
};

use crate::{
    backend::{
        cuda::kernel_plan::source::{
            KernelInput, KernelPlan,
            parameters::SplitParameters,
            writer::{KernelWriter, value_type},
        },
        kernel_lowering::element_type,
        registry::{Operation, TopKAttributes},
        spec,
    },
    data_type::DataType,
    error::{Error, Result},
    shape::{Shape, normalize_dim_usize},
    utility::to_usize,
    value::ValueSpec,
};

/// Slices of a contiguous tensor along one dimension: `outer * inner` slices of `length` elements, `inner` apart.
pub struct ScanSlices {
    outer: usize,
    length: usize,
    inner: usize,
}

impl ScanSlices {
    /// Splits `shape` into slices along `dim`; a scalar is one slice of one element.
    ///
    /// # Errors
    ///
    /// Returns an error if `dim` is out of range or an extent does not fit `usize`.
    pub fn create(shape: &Shape, dim: i64) -> Result<Self> {
        let dimensions = shape.dimensions();
        if dimensions.is_empty() {
            return if matches!(dim, 0 | -1) {
                Ok(Self {
                    outer: 1,
                    length: 1,
                    inner: 1,
                })
            } else {
                Err(Error::DimensionOutOfBounds)
            };
        }

        let axis = normalize_dim_usize(dim, shape.rank())?;
        let size = |dimensions: &[i64]| {
            dimensions.iter().try_fold(1_usize, |product, &dimension| {
                usize::try_from(dimension)
                    .ok()
                    .and_then(|dimension| product.checked_mul(dimension))
                    .ok_or_else(|| Error::out_of_range("cuda scan extent"))
            })
        };
        Ok(Self {
            outer: size(&dimensions[..axis])?,
            length: size(&dimensions[axis..=axis])?,
            inner: size(&dimensions[axis + 1..])?,
        })
    }

    /// Returns the number of slices.
    pub const fn count(&self) -> usize {
        self.outer * self.inner
    }

    /// Returns the slice length rounded up to a power of two, which a bitonic sort needs.
    pub const fn padded_length(&self) -> usize {
        self.length.next_power_of_two()
    }

    /// Generates the offset of the first element of slice `row`.
    fn base(&self) -> String {
        format!(
            "const unsigned long long base = (row / {inner}ULL) * {length}ULL * {inner}ULL + row % {inner}ULL;\n",
            inner = self.inner,
            length = self.length,
        )
    }
}

/// Threads per block for a slice of `length` elements: a power of two from 32 to 1024.
fn threads(length: usize) -> u32 {
    u32::try_from(length.div_ceil(2).next_power_of_two().clamp(32, 1024))
        .expect("the thread count is at most 1024")
}

impl KernelPlan {
    /// Plans a stable bitonic `argsort` of each slice of a contiguous input, with one block per slice.
    ///
    /// The kernel sorts positions in a scratch buffer of `padded_length` positions per slice.
    /// NaN sorts as the largest value, and positions beyond the slice sort last.
    pub(crate) fn argsort(input: &KernelInput, dim: i64, descending: bool) -> Result<Self> {
        Self::argsort_with(input, dim, descending, descending)
    }

    // Plans `argsort` with NaN first or last, independently of the direction: an argmin, which PyTorch lets NaN win, sorts ascending with NaN first.
    pub(crate) fn argsort_with(
        input: &KernelInput,
        dim: i64,
        descending: bool,
        nan_first: bool,
    ) -> Result<Self> {
        if matches!(
            input.data_type,
            DataType::F16 | DataType::Bf16 | DataType::F8E4M3 | DataType::F8E5M2
        ) {
            return Err(Error::unsupported_data_type(
                "cuda argsort of a type inference converts to f32",
            ));
        }

        if !input.layout.is_contiguous() {
            return Err(Error::invalid_argument(
                "cuda argsort input must be contiguous",
            ));
        }

        let ty = value_type(input.data_type)?;
        let shape = input.layout.shape();
        let slices = ScanSlices::create(shape, dim)?;
        let (length, inner, padded) = (slices.length, slices.inner, slices.padded_length());
        let ordered = if descending { "a > b" } else { "a < b" };
        let nan_wins = if nan_first { "an" } else { "bn" };
        let mut writer = KernelWriter::new();
        let values = writer.input(input.data_type, "values")?;
        let keys = writer.scratch(DataType::I64, slices.count() * padded, "keys")?;
        let output = writer.output(DataType::I64, "output")?;
        let launch = writer.rows(slices.count(), threads(padded));
        writer.declare(&format!(
            "
__device__ bool precedes(const {ty}* values, unsigned long long base, long long x, long long y) {{
    if (x >= {length}LL) return false;
    if (y >= {length}LL) return true;
    const {ty} a = values[base + static_cast<unsigned long long>(x) * {inner}ULL];
    const {ty} b = values[base + static_cast<unsigned long long>(y) * {inner}ULL];
    const bool an = a != a;
    const bool bn = b != b;
    if (an != bn) return {nan_wins};
    if (!an && a != b) return {ordered};
    return x < y;
}}"
        ));
        writer.lines(&slices.base());
        writer.lines(&format!(
            "
long long* sorted = {keys} + row * {padded}ULL;
for (unsigned long long i = lane; i < {padded}ULL; i += blockDim.x) sorted[i] = static_cast<long long>(i);
__syncthreads();
for (unsigned long long k = 2; k <= {padded}ULL; k <<= 1) {{
    for (unsigned long long j = k >> 1; j > 0; j >>= 1) {{
        for (unsigned long long i = lane; i < {padded}ULL; i += blockDim.x) {{
            const unsigned long long l = i ^ j;
            if (l > i) {{
                const long long a = sorted[i];
                const long long b = sorted[l];
                if (precedes({values}, base, b, a) == ((i & k) == 0)) {{ sorted[i] = b; sorted[l] = a; }}
            }}
        }}
        __syncthreads();
    }}
}}
for (unsigned long long i = lane; i < {length}ULL; i += blockDim.x) {output}[base + i * {inner}ULL] = sorted[i];"
        ));
        Ok(writer.finish(
            "argsort",
            launch,
            vec![ValueSpec::new(DataType::I64, shape.clone())],
            slice::from_ref(input),
            &[Operation::Argsort],
        ))
    }
}

const SELECT_THREADS: u32 = 128;

impl KernelPlan {
    // Plans `topk` of a contiguous input as a selection, or returns `None` for half-precision or strided inputs and more than `SELECTED` values: one block a slice picks its `k` values in `k` passes, each the best element after the previous pick in the order `argsort` sorts by, so ties and NaN come out as the sort puts them, without sorting the row.
    pub(crate) fn topk_select(input: &KernelInput, topk: &TopKAttributes) -> Result<Option<Self>> {
        let shape = input.layout.shape();
        if !spec::selects_topk(input.data_type, &input.layout, topk) {
            return Ok(None);
        }

        let slices = ScanSlices::create(shape, topk.dim)?;
        let k = to_usize(topk.k, "cuda topk k")?;
        if k > slices.length {
            return Err(Error::invalid_argument("cuda topk k exceeds the dimension"));
        }

        let axis = normalize_dim_usize(topk.dim, shape.rank())?;
        let mut dimensions = shape.dimensions().to_vec();
        dimensions[axis] = topk.k;
        let selected = Shape::create(dimensions)?;
        let ty = value_type(input.data_type)?;
        let (length, inner) = (slices.length, slices.inner);
        let ordered = if topk.largest { "a > b" } else { "a < b" };
        let nan_wins = if topk.largest { "an" } else { "bn" };
        let warps = SELECT_THREADS / 32;
        let mut writer = KernelWriter::new();
        let values = writer.input(input.data_type, "values")?;
        let picked_values = writer.output(input.data_type, "picked_values")?;
        let picked = writer.output(DataType::I64, "picked")?;
        let launch = writer.rows(slices.count(), SELECT_THREADS);
        writer.declare(&format!(
            "
__device__ bool precedes(const {ty}* values, unsigned long long base, long long x, long long y) {{
    const {ty} a = values[base + static_cast<unsigned long long>(x) * {inner}ULL];
    const {ty} b = values[base + static_cast<unsigned long long>(y) * {inner}ULL];
    const bool an = a != a;
    const bool bn = b != b;
    if (an != bn) return {nan_wins};
    if (!an && a != b) return {ordered};
    return x < y;
}}"
        ));
        writer.lines(&slices.base());
        writer.lines(&format!(
            "
const unsigned long long picked_base = (row / {inner}ULL) * {k}ULL * {inner}ULL + row % {inner}ULL;
__shared__ long long warp_best[{warps}];
__shared__ long long chosen;
long long previous = -1;
for (unsigned long long pick = 0; pick < {k}ULL; ++pick) {{
    long long best = -1;
    for (long long i = lane; i < {length}LL; i += blockDim.x) {{
        if (previous >= 0 && !precedes({values}, base, previous, i)) continue;
        if (best < 0 || precedes({values}, base, i, best)) best = i;
    }}
    for (unsigned int offset = 16; offset > 0; offset >>= 1) {{
        const long long other = __shfl_down_sync(0xffffffffu, best, offset);
        if (other >= 0 && (best < 0 || precedes({values}, base, other, best))) best = other;
    }}
    if (lane % 32 == 0) warp_best[lane / 32] = best;
    __syncthreads();
    if (lane == 0) {{
        long long winner = -1;
        for (unsigned int warp = 0; warp < {warps}U; ++warp) {{
            const long long other = warp_best[warp];
            if (other >= 0 && (winner < 0 || precedes({values}, base, other, winner))) winner = other;
        }}
        chosen = winner;
        {picked}[picked_base + pick * {inner}ULL] = winner;
        {picked_values}[picked_base + pick * {inner}ULL] = {values}[base + static_cast<unsigned long long>(winner) * {inner}ULL];
    }}
    __syncthreads();
    previous = chosen;
}}"
        ));
        Ok(Some(writer.finish(
            "topk_select",
            launch,
            vec![
                ValueSpec::new(input.data_type, selected.clone()),
                ValueSpec::new(DataType::I64, selected),
            ],
            slice::from_ref(input),
            &[Operation::TopK],
        )))
    }
}

// Scans of fewer rows than this split each row across blocks, so a few long rows still fill the GPU.
const SPLIT_BELOW_ROWS: usize = 64;

// The fewest points a split row has; shorter rows scan fast enough in one block.
const SPLIT_FROM_WIDTH: usize = 1 << 15;

pub const DEFAULT_SPLIT_THREADS: u32 = 256;

// The input values each thread of a split scan holds in registers.
const SPLIT_HELD: usize = 8;

impl KernelPlan {
    // Plans `cumsum` or `cumprod` of a contiguous floating input along its last axis as a decoupled look-back, or returns `None` when there are enough rows for the row kernel or they are short: each block scans one small chunk of a row, reading what the chunks before it total as they publish it.
    pub(crate) fn split_scan(
        input: &KernelInput,
        dim: i64,
        operation: Operation,
        split: SplitParameters,
    ) -> Result<Option<Self>> {
        let threads = split.threads;
        if !split.enabled {
            return Ok(None);
        }

        let data_type = input.data_type;
        let shape = input.layout.shape();
        let dimensions = shape.dimensions();
        let rank = dimensions.len();
        if !data_type.is_float()
            || !input.layout.is_contiguous()
            || rank == 0
            || normalize_dim_usize(dim, shape.rank())? != rank - 1
        {
            return Ok(None);
        }

        let width = to_usize(dimensions[rank - 1], "cuda split scan width")?;
        let rows = shape.element_count() / width.max(1);
        if rows == 0 || rows >= SPLIT_BELOW_ROWS || width < SPLIT_FROM_WIDTH {
            return Ok(None);
        }

        let combiner = match operation {
            Operation::Cumsum => Combiner::Sum,
            Operation::Cumprod => Combiner::Product,
            _ => return Ok(None),
        };
        let chunk = threads as usize * SPLIT_HELD;
        let chunks = width.div_ceil(chunk);
        let element = element_type(data_type)?;
        let (accumulator, ty) = if data_type == DataType::F64 {
            (DataType::F64, "double")
        } else {
            (DataType::F32, "float")
        };
        let tag = combiner_tag(combiner);
        let start = identity(combiner, element_type(accumulator)?);
        let output = ValueSpec::new(data_type, shape.clone());
        let inputs = slice::from_ref(input);
        let warps = threads / 32;
        let tiles = rows * chunks;

        // Compiled graphs keep their scratch buffers across replays, so every run clears the flags that publish each chunk's total.
        let mut clear = KernelWriter::new();
        clear.scratch(accumulator, tiles, "chunk_totals")?;
        let ready = clear.scratch(DataType::U32, tiles, "chunk_ready")?;
        let launch = clear.elements(tiles);
        clear.lines(&format!("{ready}[index] = 0U;"));
        let clear = clear.finish(
            "split_scan_clear",
            launch,
            vec![output.clone()],
            inputs,
            &[operation],
        );

        // A block holds its chunk in registers, so it reads the input once: it totals the chunk, waits for the total of everything before it, publishes its own, and writes the chunk's results from there.
        // The wait relies on blocks starting in grid order, so the blocks a block waits for have started, and no block that waits can hold back one that has not.
        let mut scan = KernelWriter::new();
        let values = scan.input(data_type, "values")?;
        let result = scan.output(data_type, "output")?;
        let totals = scan.scratch(accumulator, tiles, "chunk_totals")?;
        let ready = scan.scratch(DataType::U32, tiles, "chunk_ready")?;
        let launch = scan.blocks(tiles, threads);
        let load = load_expression(element, &format!("{values}[i]"));
        let stored = store_expression(element, "running");
        scan.lines(&format!(
            "
__shared__ {ty} partials[32];
__shared__ {ty} warp_totals[{warps}];
__shared__ {ty} before;
const unsigned long long row = blockIdx.x / {chunks}U, part = blockIdx.x % {chunks}U;
const unsigned long long begin = row * {width}ULL + part * {chunk}ULL;
const unsigned long long end = row * {width}ULL + min((part + 1ULL) * {chunk}ULL, {width}ULL);
const unsigned long long run = (end - begin + blockDim.x - 1) / blockDim.x;
const unsigned long long run_begin = min(begin + lane * run, end);
const unsigned int held_count = static_cast<unsigned int>(min(run_begin + run, end) - run_begin);
{ty} held[{SPLIT_HELD}];
{ty} local = {start};
for (unsigned int k = 0; k < held_count; ++k) {{
    const unsigned long long i = run_begin + k;
    held[k] = static_cast<{ty}>({load});
    local = singe_combine<{ty}, {tag}>(local, held[k]);
}}
const {ty} prefix = singe_block_exclusive_scan<{ty}, {tag}>(local, partials, {start});
const {ty} total = singe_block_reduce<{ty}, {tag}>(local, warp_totals);
if (lane == 0) {{
    {ty} previous = {start};
    if (part > 0) {{
        while (atomicAdd(&{ready}[blockIdx.x - 1], 0U) == 0U) {{}}
        __threadfence();
        previous = {totals}[blockIdx.x - 1];
    }}
    before = previous;
    {totals}[blockIdx.x] = singe_combine<{ty}, {tag}>(previous, total);
    __threadfence();
    atomicExch(&{ready}[blockIdx.x], 1U);
}}
__syncthreads();
{ty} running = singe_combine<{ty}, {tag}>(before, prefix);
for (unsigned int k = 0; k < held_count; ++k) {{
    running = singe_combine<{ty}, {tag}>(running, held[k]);
    {result}[run_begin + k] = {stored};
}}"
        ));
        let mut plan = scan.finish("split_scan", launch, vec![output], inputs, &[operation]);
        plan.before = vec![clear.kernel];
        Ok(Some(plan))
    }
}

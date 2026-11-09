use std::slice;

use singe_codegen::{
    emit::cuda_c::spelling::{
        combiner_tag, compute_type, identity, load_expression, store_expression,
    },
    ir::{Combiner, ElementType},
};

use crate::{
    backend::{
        cuda::kernel_plan::source::{
            KernelInput, KernelPlan, parameters::SplitParameters, writer::KernelWriter,
        },
        kernel_lowering::element_type,
        registry::Operation,
    },
    data_type::DataType,
    error::Result,
    utility::to_u64,
    value::ValueSpec,
};

// The fewest elements a whole-tensor reduction splits across blocks; below it one block of a row kernel is as fast.
const SPLIT_ELEMENTS: usize = 1 << 15;

// The threads of each block of a split reduction, and the most blocks it launches, which bounds the partials the last block combines.
pub const DEFAULT_SPLIT_THREADS: u32 = 256;
const SPLIT_BLOCKS: usize = 1024;

// The elements each thread folds before a block has enough work, so small inputs launch fewer blocks.
const ELEMENTS_PER_THREAD: usize = 8;

impl KernelPlan {
    // Plans a reduction of a whole contiguous floating-point input to one value split across blocks, or returns `None` for inputs one block reduces as fast.
    // Each block folds a grid-strided share of the elements and writes its partial to scratch; the last block to arrive, counted in a scratch word it resets for the next launch, combines the partials in block order, so the result is the same on every launch.
    pub(crate) fn split_reduction(
        operation: Operation,
        input: &KernelInput,
        output: &ValueSpec,
        split: SplitParameters,
    ) -> Result<Option<Self>> {
        let threads = split.threads;
        if !split.enabled {
            return Ok(None);
        }

        let count = input.layout.shape().element_count();
        if count < SPLIT_ELEMENTS
            || output.shape().element_count() != 1
            || output.data_type() != input.data_type
            || !input.data_type.is_float()
            || !input.layout.is_contiguous()
        {
            return Ok(None);
        }

        let combiner = match operation {
            Operation::Sum | Operation::Mean => Combiner::Sum,
            Operation::Product => Combiner::Product,
            Operation::Max => Combiner::Max,
            Operation::Min => Combiner::Min,
            Operation::LogSumExp => Combiner::LogSumExp,
            _ => return Ok(None),
        };
        let element = element_type(input.data_type)?;
        let accumulator = if element == ElementType::F64 {
            DataType::F64
        } else {
            DataType::F32
        };
        let ty = compute_type(element);
        let tag = combiner_tag(combiner);
        let start = identity(combiner, element);
        let per_block = threads as usize * ELEMENTS_PER_THREAD;
        let blocks = count.div_ceil(per_block).min(SPLIT_BLOCKS);
        let warps = threads / 32;

        let mut writer = KernelWriter::new();
        let values = writer.input(input.data_type, "values")?;
        let partials = writer.scratch(accumulator, blocks, "partials")?;
        let arrivals = writer.scratch(DataType::U32, 1, "arrivals")?;
        let result = writer.output(input.data_type, "output")?;
        let count = to_u64(count, "cuda split reduction count")?;
        let count = writer.size("count", count, "element count");
        let loaded = load_expression(element, &format!("{values}[i]"));
        let total = if operation == Operation::Mean {
            format!("total / static_cast<{ty}>({count})")
        } else {
            "total".to_owned()
        };
        let stored = store_expression(element, &total);
        let launch = writer.blocks(blocks, threads);
        writer.lines(&format!(
            "
__shared__ {ty} warps[{warps}];
__shared__ bool last;
{ty} value = {start};
for (unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + lane; i < {count}; i += static_cast<unsigned long long>(gridDim.x) * blockDim.x) {{
    value = singe_combine<{ty}, {tag}>(value, {loaded});
}}
value = singe_block_reduce<{ty}, {tag}>(value, warps);
if (lane == 0) {{
    {partials}[blockIdx.x] = value;
    __threadfence();
    last = atomicAdd({arrivals}, 1U) == gridDim.x - 1;
}}
__syncthreads();
if (!last) return;
__threadfence();
{ty} total = {start};
for (unsigned int block = lane; block < gridDim.x; block += blockDim.x) {{
    total = singe_combine<{ty}, {tag}>(total, static_cast<volatile {ty}*>({partials})[block]);
}}
total = singe_block_reduce<{ty}, {tag}>(total, warps);
if (lane == 0) {{
    {result}[0] = {stored};
    *{arrivals} = 0U;
}}"
        ));
        Ok(Some(writer.finish(
            &format!("split_{}", operation.name()),
            launch,
            vec![output.clone()],
            slice::from_ref(input),
            &[operation],
        )))
    }
}

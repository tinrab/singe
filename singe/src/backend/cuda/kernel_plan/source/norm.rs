// Layer and RMS normalization of too few long rows for one block per row to fill the GPU, as three kernels: blocks reduce chunks of the rows, one block per row combines its chunks into its statistics, and an element-wise kernel normalizes.

use std::iter;

use singe_codegen::emit::cuda_c::spelling::{load_expression, store_expression};

use crate::{
    backend::{
        cuda::kernel_plan::source::{
            KernelInput, KernelPlan, parameters::SplitParameters, writer::KernelWriter,
        },
        kernel_lowering::element_type,
        registry::{LayerNormalizationAttributes, Operation},
    },
    data_type::DataType,
    error::{Error, Result},
    value::ValueSpec,
};

// Normalizations of fewer rows than this split them, so each row's reduction spreads over several blocks.
const SPLIT_BELOW_ROWS: usize = 64;

// The fewest points a split row has; shorter rows reduce fast enough in one block.
const SPLIT_FROM_WIDTH: usize = 1 << 16;
pub const DEFAULT_THREADS: u32 = 256;

impl KernelPlan {
    // Plans `layer_norm` or `rms_norm` of contiguous floating `input`, with contiguous `parameters` shaped as the normalized dimensions, split across blocks, or returns `None` when there are enough rows for the row kernel or they are short.
    pub(crate) fn split_normalization(
        operation: Operation,
        attributes: &LayerNormalizationAttributes,
        input: &KernelInput,
        parameters: &[KernelInput],
        split: SplitParameters,
    ) -> Result<Option<Self>> {
        let threads = split.threads;
        if !split.enabled {
            return Ok(None);
        }

        let data_type = input.data_type;
        let dimensions = input.layout.shape().dimensions();
        let normalized = attributes.normalized_shape.len();
        if !data_type.is_float()
            || !matches!(operation, Operation::LayerNorm | Operation::RmsNorm)
            || !input.layout.is_contiguous()
            || normalized == 0
            || normalized > dimensions.len()
            || dimensions[dimensions.len() - normalized..] != attributes.normalized_shape[..]
            || parameters.iter().any(|parameter| {
                !parameter.layout.is_contiguous()
                    || parameter.layout.shape().dimensions() != attributes.normalized_shape
            })
        {
            return Ok(None);
        }

        let extent = |values: &[i64]| -> Result<usize> {
            values.iter().try_fold(1_usize, |product, &value| {
                usize::try_from(value)
                    .ok()
                    .and_then(|value| product.checked_mul(value))
                    .ok_or_else(|| Error::out_of_range("cuda normalization extent"))
            })
        };
        let width = extent(&dimensions[dimensions.len() - normalized..])?;
        let rows = extent(&dimensions[..dimensions.len() - normalized])?;
        if rows == 0 || rows >= SPLIT_BELOW_ROWS || width < SPLIT_FROM_WIDTH {
            return Ok(None);
        }

        let chunks = (2 * SPLIT_BELOW_ROWS * 4).div_ceil(rows);
        let chunk = width.div_ceil(chunks);
        let chunks = width.div_ceil(chunk);
        let element = element_type(data_type)?;
        let (accumulator, ty) = if data_type == DataType::F64 {
            (DataType::F64, "double")
        } else {
            (DataType::F32, "float")
        };
        let centered = operation == Operation::LayerNorm;
        let warps = threads / 32;
        let inputs = iter::once(input.clone())
            .chain(parameters.iter().cloned())
            .collect::<Vec<_>>();
        let output = ValueSpec::new(data_type, input.layout.shape().clone());

        // Each chunk's point count is fixed, so its statistics are its mean and the sum of its squared deviations, or for RMS norm the sum of its squares.
        let mut reduce = KernelWriter::new();
        let values = reduce.input(data_type, "values")?;
        let chunk_statistics =
            reduce.scratch(accumulator, rows * chunks * 2, "chunk_statistics")?;
        reduce.scratch(accumulator, rows * 2, "row_statistics")?;
        let launch = reduce.blocks(rows * chunks, threads);
        let load = load_expression(element, &format!("{values}[i]"));
        // One pass sums values shifted by the chunk's first, which keeps the sum of squared deviations, `Σ(x − s)² − (Σ(x − s))² / n`, from cancelling when the values sit far from zero.
        let shift = if centered {
            format!(
                "static_cast<{ty}>({})",
                load_expression(element, &format!("{values}[begin]"))
            )
        } else {
            format!("static_cast<{ty}>(0)")
        };
        let statistics = if centered {
            format!(
                "
    const {ty} points = static_cast<{ty}>(end - begin);
    {chunk_statistics}[blockIdx.x * 2ULL] = shift + sum / points;
    {chunk_statistics}[blockIdx.x * 2ULL + 1ULL] = squares - sum * sum / points;"
            )
        } else {
            format!(
                "
    {chunk_statistics}[blockIdx.x * 2ULL] = 0;
    {chunk_statistics}[blockIdx.x * 2ULL + 1ULL] = squares;"
            )
        };
        reduce.lines(&format!(
            "
__shared__ {ty} warps[{warps}];
const unsigned long long row = blockIdx.x / {chunks}U, part = blockIdx.x % {chunks}U;
const unsigned long long begin = row * {width}ULL + part * {chunk}ULL;
const unsigned long long end = row * {width}ULL + min((part + 1ULL) * {chunk}ULL, {width}ULL);
const {ty} shift = {shift};
{ty} sum = 0, squares = 0;
for (unsigned long long i = begin + lane; i < end; i += blockDim.x) {{
    const {ty} value = static_cast<{ty}>({load}) - shift;
    sum += value;
    squares += value * value;
}}
sum = singe_block_reduce<{ty}, 0>(sum, warps);
squares = singe_block_reduce<{ty}, 0>(squares, warps);
if (lane == 0) {{{statistics}
}}"
        ));
        let reduce = reduce.finish(
            "split_normalization_chunks",
            launch,
            vec![output.clone()],
            &inputs,
            &[operation],
        );

        // Chan's update combines the chunks' means and squared deviations exactly; every chunk but the last has `chunk` points.
        let mut combine = KernelWriter::new();
        let chunk_statistics =
            combine.scratch(accumulator, rows * chunks * 2, "chunk_statistics")?;
        let row_statistics = combine.scratch(accumulator, rows * 2, "row_statistics")?;
        let launch = combine.blocks(rows, 32);
        let eps = f64::from(attributes.eps);
        let combined = if centered {
            format!(
                "
    {ty} count = 0, mean = 0, squares = 0;
    for (unsigned int part = 0; part < {chunks}U; ++part) {{
        const {ty} points = static_cast<{ty}>(min({chunk}ULL, {width}ULL - part * {chunk}ULL));
        const {ty} part_mean = {chunk_statistics}[(row * {chunks}ULL + part) * 2ULL];
        const {ty} delta = part_mean - mean;
        const {ty} total = count + points;
        mean += delta * points / total;
        squares += {chunk_statistics}[(row * {chunks}ULL + part) * 2ULL + 1ULL] + delta * delta * count * points / total;
        count = total;
    }}
    {row_statistics}[row * 2ULL] = mean;
    {row_statistics}[row * 2ULL + 1ULL] = rsqrt(squares / static_cast<{ty}>({width}ULL) + static_cast<{ty}>({eps:?}));"
            )
        } else {
            format!(
                "
    {ty} squares = 0;
    for (unsigned int part = 0; part < {chunks}U; ++part) squares += {chunk_statistics}[(row * {chunks}ULL + part) * 2ULL + 1ULL];
    {row_statistics}[row * 2ULL] = 0;
    {row_statistics}[row * 2ULL + 1ULL] = rsqrt(squares / static_cast<{ty}>({width}ULL) + static_cast<{ty}>({eps:?}));"
            )
        };
        combine.lines(&format!(
            "
if (lane == 0) {{
    const unsigned long long row = blockIdx.x;{combined}
}}"
        ));
        let combine = combine.finish(
            "split_normalization_rows",
            launch,
            vec![output.clone()],
            &inputs,
            &[operation],
        );

        // The weight and bias, when present, are the inputs after the values, indexed by a point's place in its row.
        let mut normalize = KernelWriter::new();
        let values = normalize.input(data_type, "values")?;
        let mut affine = Vec::new();
        for (index, parameter) in parameters.iter().enumerate() {
            let name = if index == 0 && attributes.has_weight {
                "weight"
            } else {
                "bias"
            };
            affine.push((
                normalize.input(parameter.data_type, name)?,
                parameter.data_type,
            ));
        }

        let result = normalize.output(data_type, "output")?;
        normalize.scratch(accumulator, rows * chunks * 2, "chunk_statistics")?;
        let row_statistics = normalize.scratch(accumulator, rows * 2, "row_statistics")?;
        let launch = normalize.elements(rows * width);
        let mut value = format!(
            "(static_cast<{ty}>({}) - {row_statistics}[row * 2ULL]) * {row_statistics}[row * 2ULL + 1ULL]",
            load_expression(element, &format!("{values}[index]"))
        );
        let mut parameter = affine.iter();
        if attributes.has_weight
            && let Some((name, parameter_type)) = parameter.next()
        {
            value = format!(
                "({value}) * static_cast<{ty}>({})",
                load_expression(
                    element_type(*parameter_type)?,
                    &format!("{name}[index % {width}ULL]")
                )
            );
        }

        if attributes.has_bias
            && let Some((name, parameter_type)) = parameter.next()
        {
            value = format!(
                "({value}) + static_cast<{ty}>({})",
                load_expression(
                    element_type(*parameter_type)?,
                    &format!("{name}[index % {width}ULL]")
                )
            );
        }

        let stored = store_expression(element, &value);
        normalize.lines(&format!(
            "
const unsigned long long row = index / {width}ULL;
{result}[index] = {stored};"
        ));
        let mut plan = normalize.finish(
            "split_normalization",
            launch,
            vec![output],
            &inputs,
            &[operation],
        );
        plan.before = vec![reduce.kernel, combine.kernel];
        Ok(Some(plan))
    }
}

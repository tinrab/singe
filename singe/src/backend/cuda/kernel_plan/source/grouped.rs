// Grouped matmuls, as mixture-of-experts layers run them: rows sorted by group, whose group `g` is rows `offsets[g]..offsets[g + 1]`, each times its group's weight.
// Half-precision rows run on tensor cores: a block of a three-dimensional grid of row tiles, column tiles, and groups takes a 64 by 64 tile of one group's rows and the columns, after a first kernel zeroes the output, so rows outside every group stay zero.
// Other floating types, and targets before `sm_80`, take one thread per output element, which finds its row's group by bisecting the offsets.

use singe_codegen::emit::cuda_c::spelling::store_expression;

use crate::{
    backend::{
        cuda::{
            codegen::CompileRequirements,
            kernel_plan::source::{
                KernelInput, KernelPlan, KernelPlanner,
                parameters::{GroupedParameters, KernelParameters, parameter_key},
                tensor_core::tensor_core_helpers,
                writer::KernelWriter,
            },
        },
        kernel_lowering::element_type,
        registry::Operation,
        spec,
    },
    data_type::DataType,
    error::{Error, Result},
    utility::to_usize,
    value::ValueSpec,
};

const TILE: usize = 64;
const PAD: usize = 8;
const THREADS: u32 = 128;

impl KernelPlanner {
    // The parameters grouped matmul may run with, the default first.
    pub(crate) fn grouped_candidates(&self) -> Vec<GroupedParameters> {
        let default = GroupedParameters::default();
        let mut candidates = vec![default];
        candidates.extend([16, 64].map(|depth| GroupedParameters {
            depth,
            min_blocks: None,
        }));
        candidates.extend(
            self.launch_bound_candidates(THREADS as usize)
                .into_iter()
                .flatten()
                .map(|bound| GroupedParameters {
                    min_blocks: Some(bound),
                    ..default
                }),
        );
        candidates
    }

    pub(crate) fn grouped_matmul(
        &self,
        input: &KernelInput,
        weights: &KernelInput,
        offsets: &KernelInput,
    ) -> Result<KernelPlan> {
        let shape = spec::grouped_matmul_shape(
            input.layout.shape(),
            weights.layout.shape(),
            (offsets.layout.shape(), offsets.data_type),
            "cuda grouped matmul",
        )?;
        let data_type = input.data_type;
        if !data_type.is_float() || data_type == DataType::F4E2M1 || weights.data_type != data_type
        {
            return Err(Error::unsupported_data_type("cuda grouped matmul"));
        }

        let extent = |value: i64| to_usize(value, "cuda grouped matmul extent");
        let dimensions = weights.layout.shape().dimensions();
        let (rows, inner, groups, columns) = (
            extent(input.layout.shape().dimensions()[0])?,
            extent(dimensions[1])?,
            extent(dimensions[0])?,
            extent(dimensions[2])?,
        );
        let (input_strides, weight_strides) = (input.layout.strides(), weights.layout.strides());
        let geometry = Geometry {
            rows,
            inner,
            groups,
            columns,
            input_strides: [input_strides[0], input_strides[1]],
            weight_strides: [weight_strides[0], weight_strides[1], weight_strides[2]],
        };
        let output = ValueSpec::new(data_type, shape);
        let inputs = [input.clone(), weights.clone(), offsets.clone()];
        let tensor_cores = matches!(data_type, DataType::F16 | DataType::Bf16)
            && self
                .context
                .target
                .is_none_or(|target| target.compute_capability() >= (8, 0));
        if tensor_cores {
            let key = parameter_key("grouped_matmul", &inputs, "");
            let parameters = match self.tuned_parameters(&key) {
                Some(KernelParameters::Grouped(parameters))
                    if self.grouped_candidates().contains(&parameters) =>
                {
                    parameters
                }
                Some(_) => return Err(Self::stale_parameters(&key)),
                None => GroupedParameters::default(),
            };
            grouped_on_tensor_cores(
                &geometry,
                data_type,
                offsets.data_type,
                output,
                &inputs,
                parameters,
            )
        } else {
            grouped_on_cuda_cores(&geometry, data_type, offsets.data_type, output, &inputs)
        }
    }
}

struct Geometry {
    rows: usize,
    inner: usize,
    groups: usize,
    columns: usize,
    input_strides: [i64; 2],
    weight_strides: [i64; 3],
}

impl Geometry {
    // The element offsets of the rows' value at `row`, `k`, and the weights' at `group`, `k`, `column`, as C++ expressions.
    fn input_at(&self, row: &str, k: &str) -> String {
        let [row_stride, k_stride] = self.input_strides;
        format!(
            "static_cast<long long>({row}) * {row_stride}LL + static_cast<long long>({k}) * {k_stride}LL"
        )
    }

    fn weight_at(&self, group: &str, k: &str, column: &str) -> String {
        let [group_stride, k_stride, column_stride] = self.weight_strides;
        format!(
            "static_cast<long long>({group}) * {group_stride}LL + static_cast<long long>({k}) * {k_stride}LL + static_cast<long long>({column}) * {column_stride}LL",
        )
    }
}

fn grouped_on_cuda_cores(
    geometry: &Geometry,
    data_type: DataType,
    offsets_type: DataType,
    output: ValueSpec,
    inputs: &[KernelInput],
) -> Result<KernelPlan> {
    let Geometry {
        rows,
        inner,
        groups,
        columns,
        ..
    } = *geometry;
    let element = element_type(data_type)?;
    let (accumulator, load) = if data_type == DataType::F64 {
        ("double", "static_cast<double>")
    } else {
        ("float", "static_cast<float>")
    };
    let mut writer = KernelWriter::new();
    let values = writer.input(data_type, "rows")?;
    let weights = writer.input(data_type, "weights")?;
    let offsets = writer.input(offsets_type, "offsets")?;
    let result = writer.output(data_type, "output")?;
    let launch = writer.elements(rows * columns);
    let convert = |value: &str| match data_type {
        DataType::F16 => format!("load_half({value})"),
        DataType::Bf16 => format!("load_bfloat({value})"),
        _ => format!("{load}({value})"),
    };
    let lhs = convert(&format!("{values}[{}]", geometry.input_at("row", "k")));
    let rhs = convert(&format!(
        "{weights}[{}]",
        geometry.weight_at("group", "k", "column")
    ));
    let stored = store_expression(element, "sum");
    writer.lines(&format!(
        "
const unsigned long long row = index / {columns}ULL, column = index % {columns}ULL;
// `low` counts the groups that start at or before this row, so the row is in the last of them unless that one ends before it.
unsigned long long low = 0ULL, high = {groups}ULL;
while (low < high) {{
    const unsigned long long middle = (low + high) / 2ULL;
    if (static_cast<long long>({offsets}[middle]) <= static_cast<long long>(row)) low = middle + 1ULL; else high = middle;
}}
const unsigned long long group = low - 1ULL;
{accumulator} sum = 0;
if (low > 0ULL && static_cast<long long>(row) < static_cast<long long>({offsets}[low]))
    for (unsigned long long k = 0; k < {inner}ULL; ++k) sum += {lhs} * {rhs};
{result}[index] = {stored};"
    ));
    Ok(writer.finish(
        "grouped_matmul",
        launch,
        vec![output],
        inputs,
        &[Operation::GroupedMatMul],
    ))
}

fn grouped_on_tensor_cores(
    geometry: &Geometry,
    data_type: DataType,
    offsets_type: DataType,
    output: ValueSpec,
    inputs: &[KernelInput],
    GroupedParameters { depth, min_blocks }: GroupedParameters,
) -> Result<KernelPlan> {
    let Geometry {
        rows,
        inner,
        groups,
        columns,
        ..
    } = *geometry;
    let kind = if data_type == DataType::F16 {
        "f16"
    } else {
        "bf16"
    };
    let element = element_type(data_type)?;

    // The output starts zeroed, for rows outside every group.
    let mut zero = KernelWriter::new();
    let zeroed = zero.output(data_type, "output")?;
    let launch = zero.elements(rows * columns);
    zero.lines(&format!("{zeroed}[index] = 0;"));
    let zero = zero.finish(
        "grouped_matmul_zero",
        launch,
        vec![output.clone()],
        inputs,
        &[Operation::GroupedMatMul],
    );

    let row_tiles = rows.div_ceil(TILE).max(1);
    let column_tiles = columns.div_ceil(TILE).max(1);
    let mut writer = KernelWriter::new();
    writer.require(CompileRequirements::TENSOR_CORES);
    writer.min_blocks(min_blocks);
    let values = writer.input(data_type, "rows")?;
    let weights = writer.input(data_type, "weights")?;
    let offsets = writer.input(offsets_type, "offsets")?;
    let result = writer.output(data_type, "output")?;
    let launch = writer.grid([row_tiles, column_tiles, groups], THREADS)?;
    writer.declare(&tensor_core_helpers(kind));
    let stored = store_expression(element, "accumulated[i][j][r]");
    let lhs = geometry.input_at("row", "k");
    let rhs = geometry.weight_at("group", "k", "column");
    let (lhs_row, rhs_row) = (depth + PAD, TILE + PAD);
    writer.lines(&format!(
        "
__shared__ __align__(16) unsigned short lhs_tile[{TILE}][{lhs_row}];
__shared__ __align__(16) unsigned short rhs_tile[{depth}][{rhs_row}];
const unsigned int warp = lane / 32U, warp_lane = lane % 32U;
const unsigned int warp_row = warp / 2U, warp_column = warp % 2U;
const unsigned int row_tile = blockIdx.x, column_tile = blockIdx.y, group = blockIdx.z;
const long long end = static_cast<long long>({offsets}[group + 1U]);
const long long first_row = static_cast<long long>({offsets}[group]) + static_cast<long long>(row_tile) * {TILE}LL;
if (first_row >= end) return;
float accumulated[2][4][4] = {{}};
for (unsigned int first = 0; first < {inner}U; first += {depth}U) {{
    for (unsigned int point = threadIdx.x; point < {TILE}U * {depth}U; point += blockDim.x) {{
        const unsigned int r = point / {depth}U, c = point % {depth}U;
        const long long row = first_row + r;
        const unsigned int k = first + c;
        lhs_tile[r][c] = row < end && k < {inner}U ? {values}[{lhs}] : static_cast<unsigned short>(0);
    }}
    for (unsigned int point = threadIdx.x; point < {depth}U * {TILE}U; point += blockDim.x) {{
        const unsigned int r = point / {TILE}U, c = point % {TILE}U;
        const unsigned int k = first + r, column = column_tile * {TILE}U + c;
        rhs_tile[r][c] = k < {inner}U && column < {columns}U ? {weights}[{rhs}] : static_cast<unsigned short>(0);
    }}
    __syncthreads();
#pragma unroll
    for (unsigned int step = 0; step < {depth}U / 16U; ++step) {{
        unsigned int a[2][4];
#pragma unroll
        for (unsigned int i = 0; i < 2U; ++i) {{
            const unsigned int address = singe_shared_address(&lhs_tile[warp_row * 32U + i * 16U + warp_lane % 16U][step * 16U + (warp_lane / 16U) * 8U]);
            asm volatile(\"ldmatrix.sync.aligned.x4.m8n8.shared.b16 {{%0, %1, %2, %3}}, [%4];\\n\" : \"=r\"(a[i][0]), \"=r\"(a[i][1]), \"=r\"(a[i][2]), \"=r\"(a[i][3]) : \"r\"(address));
        }}
#pragma unroll
        for (unsigned int j = 0; j < 4U; ++j) {{
            unsigned int b[2];
            const unsigned int address = singe_shared_address(&rhs_tile[step * 16U + warp_lane % 16U][warp_column * 32U + j * 8U]);
            asm volatile(\"ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {{%0, %1}}, [%2];\\n\" : \"=r\"(b[0]), \"=r\"(b[1]) : \"r\"(address));
#pragma unroll
            for (unsigned int i = 0; i < 2U; ++i) singe_mma(accumulated[i][j], a[i], b);
        }}
    }}
    __syncthreads();
}}
// Each lane holds rows `warp_lane / 4` and `warp_lane / 4 + 8` of each fragment, at two adjacent columns.
#pragma unroll
for (unsigned int i = 0; i < 2U; ++i)
#pragma unroll
for (unsigned int j = 0; j < 4U; ++j)
#pragma unroll
for (unsigned int r = 0; r < 4U; ++r) {{
    const long long row = first_row + warp_row * 32U + i * 16U + warp_lane / 4U + (r / 2U) * 8U;
    const unsigned int column = column_tile * {TILE}U + warp_column * 32U + j * 8U + (warp_lane % 4U) * 2U + r % 2U;
    if (row < end && column < {columns}U) {result}[static_cast<unsigned long long>(row) * {columns}ULL + column] = {stored};
}}"
    ));
    let mut plan = writer.finish(
        "grouped_matmul",
        launch,
        vec![output],
        inputs,
        &[Operation::GroupedMatMul],
    );
    plan.before = vec![zero.kernel];
    Ok(plan)
}

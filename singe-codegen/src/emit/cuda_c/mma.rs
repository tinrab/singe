// Contractions on tensor cores: an `mma.sync` m16n8k16 main loop for a fused contraction, whose factors are computed into shared memory as the half-precision values they hold and whose epilogue runs on each accumulated sum.
// A block of four warps covers an `m` by `n` tile of the rows and columns at one point of the other parallel axes, two warps along each, and walks the contraction `k` points at a time.

use crate::{
    emit::{
        EmitError,
        cuda_c::{Writer, spelling::store_expression},
    },
    ir::{AxisId, ElementType, Expr, NodeId},
    schedule::{MMA_THREADS, contraction::FusedContraction},
};

const PAD: u32 = 8;

impl Writer<'_> {
    pub fn mma_body(&mut self, m: u32, n: u32, k: u32) -> Result<(), EmitError> {
        let contraction = FusedContraction::of(self.kernel).ok_or(EmitError::Unsupported {
            reason: "an mma.sync schedule of a kernel that is not a contraction of half-precision factors",
        })?;
        let FusedContraction {
            sum,
            lhs,
            rhs,
            element,
            reduced,
            rows,
            columns,
            ref batch,
        } = contraction;
        let [store] = self.kernel.stores.as_slice() else {
            unreachable!("a fused contraction stores once");
        };
        let ix = self.index;
        let kind = if element == ElementType::F16 {
            "f16"
        } else {
            "bf16"
        };
        let (warp_rows, warp_columns) = (m / 2, n / 2);
        let (row_fragments, column_fragments) = (warp_rows / 16, warp_columns / 8);
        let (row_extent, column_extent, depth) = (
            self.axis_extent(rows),
            self.axis_extent(columns),
            self.axis_extent(reduced),
        );
        self.source.declare(&format!(
            "
__device__ __forceinline__ unsigned int singe_shared_address(const void* pointer) {{
    return static_cast<unsigned int>(__cvta_generic_to_shared(pointer));
}}
// Copies 16 bytes from global to shared memory without waiting for them.
__device__ __forceinline__ void singe_copy_async(unsigned int destination, const void* source) {{
    asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(destination), \"l\"(source));
}}
__device__ __forceinline__ void singe_mma(float* accumulator, const unsigned int* a, const unsigned int* b) {{
    asm volatile(
        \"mma.sync.aligned.m16n8k16.row.col.f32.{kind}.{kind}.f32 {{%0, %1, %2, %3}}, {{%4, %5, %6, %7}}, {{%8, %9}}, {{%0, %1, %2, %3}};\\n\"
        : \"+f\"(accumulator[0]), \"+f\"(accumulator[1]), \"+f\"(accumulator[2]), \"+f\"(accumulator[3])
        : \"r\"(a[0]), \"r\"(a[1]), \"r\"(a[2]), \"r\"(a[3]), \"r\"(b[0]), \"r\"(b[1]));
}}"
        ));
        // A weight that is one plain load contiguous along the contraction, as linear layers store theirs, keeps that order in shared memory, a row a column, and `ldmatrix` reads it untransposed.
        let rhs_by_column = match self.cone(rhs, None).as_slice() {
            &[node] => match &self.kernel.nodes[node.0].expr {
                Expr::Load(access) => {
                    access.indirect.is_empty()
                        && self.kernel.operands[access.operand.0].element == element
                        && self.walks_contiguously(access, reduced)
                        && !self.walks_contiguously(access, columns)
                }
                _ => false,
            },
            _ => false,
        };
        let lhs_row = k + PAD;
        let (rhs_rows, rhs_row) = if rhs_by_column {
            (n, k + PAD)
        } else {
            (k, n + PAD)
        };
        self.source.lines(&format!(
            "
__shared__ __align__(16) unsigned short lhs_stages[2][{m}][{lhs_row}];
__shared__ __align__(16) unsigned short rhs_stages[2][{rhs_rows}][{rhs_row}];
const unsigned int warp = threadIdx.x / 32U, lane = threadIdx.x % 32U;
const unsigned int warp_row = warp / 2U, warp_column = warp % 2U;
const {ix} column_tiles = ({column_extent} + {n}U - 1U) / {n}U;
const {ix} row_tiles = ({row_extent} + {m}U - 1U) / {m}U;
const {ix} column_tile = static_cast<{ix}>(blockIdx.x) % column_tiles;
const {ix} row_tile = static_cast<{ix}>(blockIdx.x) / column_tiles % row_tiles;
const {ix} batch_point = static_cast<{ix}>(blockIdx.x) / column_tiles / row_tiles;"
        ));
        if !batch.is_empty() {
            self.decompose("batch_point", batch);
        }

        // Two stages of tiles: while tensor cores multiply one, the next one fills, its plain contiguous loads copied asynchronously.
        self.source.lines(&format!(
            "
float accumulated[{row_fragments}][{column_fragments}][4] = {{}};
const auto fill = [&](const {ix} first, const unsigned int stage) {{
unsigned short (*lhs_tile)[{lhs_row}] = lhs_stages[stage];
unsigned short (*rhs_tile)[{rhs_row}] = rhs_stages[stage];"
        ));
        let rhs_layout = if rhs_by_column {
            (
                rhs,
                "rhs_tile",
                columns,
                reduced,
                n,
                k,
                "column_tile",
                "first",
            )
        } else {
            (
                rhs,
                "rhs_tile",
                reduced,
                columns,
                k,
                n,
                "first",
                "column_tile",
            )
        };
        for (factor, tile, outer_axis, inner_axis, outer, inner, outer_base, inner_base) in [
            (lhs, "lhs_tile", rows, reduced, m, k, "row_tile", "first"),
            rhs_layout,
        ] {
            let cone = self.cone(factor, None);
            let outer_fastest = self.loads_contiguously(&cone, outer_axis)
                && !self.loads_contiguously(&cone, inner_axis);
            let base = |name: &str, size: u32| {
                if name == "first" {
                    "first".to_owned()
                } else {
                    format!("{name} * {size}U")
                }
            };
            let (outer_extent, inner_extent) =
                (self.axis_extent(outer_axis), self.axis_extent(inner_axis));
            let (outer_base_value, inner_base_value) =
                (base(outer_base, outer), base(inner_base, inner));
            let (outer_axis, inner_axis) = (outer_axis.0, inner_axis.0);
            // A factor that is one load of the half-precision operand, contiguous along the tile's rows, copies runs of eight values, 16 bytes, as they are.
            let copied = match cone.as_slice() {
                &[node] => match &self.kernel.nodes[node.0].expr {
                    Expr::Load(access)
                        if access.indirect.is_empty()
                            && !outer_fastest
                            && self.kernel.operands[access.operand.0].element == element
                            && self.walks_contiguously(access, AxisId(inner_axis)) =>
                    {
                        Some(access.clone())
                    }
                    _ => None,
                },
                _ => None,
            };
            if let Some(access) = copied {
                let operand = access.operand.0;
                let offset = self.offset(&access);
                let item_offset = self.offset_at(
                    &access,
                    &[(AxisId(inner_axis), &format!("(a{inner_axis} + item)"))],
                );
                self.source.lines(&format!(
                    "
for (unsigned int run = threadIdx.x; run < {runs}U; run += {MMA_THREADS}U) {{
    const unsigned int outer = run / {run_width}U, inner = (run % {run_width}U) * 8U;
    const {ix} a{outer_axis} = {outer_base_value} + outer;
    const {ix} a{inner_axis} = {inner_base_value} + inner;
    const unsigned short* source = o{operand} + ({offset});
    if (a{outer_axis} < {outer_extent} && a{inner_axis} + 7U < {inner_extent} && (reinterpret_cast<unsigned long long>(source) & 15ULL) == 0ULL) {{
        singe_copy_async(singe_shared_address(&{tile}[outer][inner]), source);
    }} else {{
        for (unsigned int item = 0; item < 8U; ++item)
            {tile}[outer][inner + item] = a{outer_axis} < {outer_extent} && a{inner_axis} + item < {inner_extent} ? o{operand}[{item_offset}] : static_cast<unsigned short>(0);
    }}
}}",
                    runs = outer * inner / 8,
                    run_width = inner / 8,
                ));
                continue;
            }

            // Other factors are computed, consecutive threads on consecutive points along the axis their loads walk contiguously, so they coalesce.
            let (outer_index, inner_index) = if outer_fastest {
                (format!("point % {outer}U"), format!("point / {outer}U"))
            } else {
                (format!("point / {inner}U"), format!("point % {inner}U"))
            };
            self.source.lines(&format!(
                "
#pragma unroll 4
for (unsigned int point = threadIdx.x; point < {count}U; point += {MMA_THREADS}U) {{
    const unsigned int outer = {outer_index}, inner = {inner_index};
    const {ix} a{outer_axis} = {outer_base_value} + outer;
    const {ix} a{inner_axis} = {inner_base_value} + inner;
    unsigned short bits = 0;
    if (a{outer_axis} < {outer_extent} && a{inner_axis} < {inner_extent}) {{",
                count = outer * inner,
            ));
            for node in cone {
                let expression = self.expression(node);
                self.define(node, &expression);
            }

            self.source.lines(&format!(
                "
        bits = {};
    }}
    {tile}[outer][inner] = bits;
}}",
                store_expression(element, &format!("v{}", factor.0))
            ));
        }

        self.source.lines(&format!(
            "
asm volatile(\"cp.async.commit_group;\\n\" ::);
}};
fill(0, 0U);
unsigned int stage = 0U;
for ({ix} first = 0; first < {depth}; first += {k}U, stage ^= 1U) {{
if (first + {k}U < {depth}) {{
    fill(first + {k}U, stage ^ 1U);
    asm volatile(\"cp.async.wait_group 1;\\n\" ::);
}} else {{
    asm volatile(\"cp.async.wait_group 0;\\n\" ::);
}}
__syncthreads();
unsigned short (*lhs_tile)[{lhs_row}] = lhs_stages[stage];
unsigned short (*rhs_tile)[{rhs_row}] = rhs_stages[stage];"
        ));
        let rhs_fragment = if rhs_by_column {
            format!(
                "        const unsigned int address = singe_shared_address(&rhs_tile[warp_column * {warp_columns}U + j * 8U + lane % 8U][step * 16U + ((lane / 8U) % 2U) * 8U]);
        asm volatile(\"ldmatrix.sync.aligned.x2.m8n8.shared.b16 {{%0, %1}}, [%2];\\n\" : \"=r\"(b[0]), \"=r\"(b[1]) : \"r\"(address));"
            )
        } else {
            format!(
                "        const unsigned int address = singe_shared_address(&rhs_tile[step * 16U + lane % 16U][warp_column * {warp_columns}U + j * 8U]);
        asm volatile(\"ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {{%0, %1}}, [%2];\\n\" : \"=r\"(b[0]), \"=r\"(b[1]) : \"r\"(address));"
            )
        };
        self.source.lines(&format!(
            "
#pragma unroll
for (unsigned int step = 0; step < {steps}U; ++step) {{
    unsigned int a[{row_fragments}][4];
#pragma unroll
    for (unsigned int i = 0; i < {row_fragments}U; ++i) {{
        const unsigned int address = singe_shared_address(&lhs_tile[warp_row * {warp_rows}U + i * 16U + lane % 16U][step * 16U + (lane / 16U) * 8U]);
        asm volatile(\"ldmatrix.sync.aligned.x4.m8n8.shared.b16 {{%0, %1, %2, %3}}, [%4];\\n\" : \"=r\"(a[i][0]), \"=r\"(a[i][1]), \"=r\"(a[i][2]), \"=r\"(a[i][3]) : \"r\"(address));
    }}
#pragma unroll
    for (unsigned int j = 0; j < {column_fragments}U; ++j) {{
        unsigned int b[2];
{rhs_fragment}
#pragma unroll
        for (unsigned int i = 0; i < {row_fragments}U; ++i) singe_mma(accumulated[i][j], a[i], b);
    }}
}}
__syncthreads();
}}
// Each lane holds rows `lane / 4` and `lane / 4 + 8` of each fragment, at two adjacent columns.
#pragma unroll
for (unsigned int i = 0; i < {row_fragments}U; ++i)
#pragma unroll
for (unsigned int j = 0; j < {column_fragments}U; ++j)
#pragma unroll
for (unsigned int r = 0; r < 4U; ++r) {{
    const {ix} a{rows} = row_tile * {m}U + warp_row * {warp_rows}U + i * 16U + lane / 4U + (r / 2U) * 8U;
    const {ix} a{columns} = column_tile * {n}U + warp_column * {warp_columns}U + j * 8U + (lane % 4U) * 2U + r % 2U;
    if (a{rows} < {row_extent} && a{columns} < {column_extent}) {{
        const float v{sum} = accumulated[i][j][r];",
            steps = k / 16,
            rhs_fragment = rhs_fragment,
            rows = rows.0,
            columns = columns.0,
            sum = sum.0,
        ));
        for node in self.cone(store.value, Some(sum)) {
            let expression = self.expression(node);
            self.define(node, &expression);
        }

        let write = self.write(store);
        self.source.lines(&format!(
            "
        {write}
    }}
}}"
        ));
        Ok(())
    }

    // The nodes `root` depends on and itself, in node order, without `stop` and what only it depends on.
    fn cone(&self, root: NodeId, stop: Option<NodeId>) -> Vec<NodeId> {
        let mut needed = vec![false; self.kernel.nodes.len()];
        let mut pending = vec![root];
        while let Some(node) = pending.pop() {
            if needed[node.0] || Some(node) == stop {
                continue;
            }

            needed[node.0] = true;
            pending.extend(self.kernel.nodes[node.0].expr.inputs());
        }
        needed
            .iter()
            .enumerate()
            .filter(|(_, needed)| **needed)
            .map(|(node, _)| NodeId(node))
            .collect()
    }

    // Whether a load among `nodes` steps through memory one element at a time along `axis`.
    fn loads_contiguously(&self, nodes: &[NodeId], axis: AxisId) -> bool {
        nodes
            .iter()
            .any(|node| match &self.kernel.nodes[node.0].expr {
                Expr::Load(access) => self.walks_contiguously(access, axis),
                _ => false,
            })
    }
}

impl Writer<'_> {
    fn axis_extent(&self, axis: AxisId) -> String {
        self.size(self.kernel.axes[axis.0].extent)
    }
}

use crate::{
    emit::{EMITTER_VERSION, EmitError, EntryLinkage, IndexWidth, KernelSource, cuda_c},
    ir::{Access, AxisKind, BinaryOp, Combiner, ElementType, Kernel, OperandRole, Size, UnaryOp},
    schedule::{
        Launch, Schedule, ScheduleError, ThreadSchedule, element_schedule, reduction_vector,
        space::candidates,
    },
};

/// Builds `out[row] = x[row, :] / sum(exp(x[row, :]))` for BF16 rows of an argument width.
fn normalized_exponentials() -> Kernel {
    let mut kernel = Kernel::new("singe_test");
    let x = kernel.add_operand(ElementType::Bf16, OperandRole::Input);
    let out = kernel.add_operand(ElementType::Bf16, OperandRole::Output);
    let rows = kernel.add_size(Size::Argument);
    let width = kernel.add_size(Size::Argument);
    let one = kernel.add_size(Size::Static(1));
    let row = kernel.add_axis(rows, AxisKind::Parallel);
    let column = kernel.add_axis(width, AxisKind::Reduction);
    let matrix = |operand| Access {
        operand,
        base: None,
        indirect: Vec::new(),
        strides: vec![(row, width), (column, one)],
    };
    let value = kernel.load(matrix(x));
    let exponential = kernel.unary(UnaryOp::Exp, value);
    let sum = kernel.reduce(Combiner::Sum, vec![column], exponential);
    let ratio = kernel.binary(BinaryOp::Div, exponential, sum);
    kernel.store(matrix(out), ratio);
    kernel
}

/// Emits `kernel` with 64-bit indices, as the expected sources below are written.
fn emit(kernel: &Kernel, schedule: Schedule) -> Result<KernelSource, EmitError> {
    cuda_c::emit(
        kernel,
        schedule.thread().expect("a thread schedule"),
        IndexWidth::U64,
    )
}

#[test]
fn test_element_schedules_walk_reductions_inside_each_thread() {
    let source = emit(
        &normalized_exponentials(),
        Schedule::Elements {
            threads: 128,
            vector: 1,
        },
    )
    .unwrap();
    assert_eq!(
        source.render_definition(EntryLinkage::C),
        "extern \"C\" __global__ void singe_test(const unsigned short* __restrict__ o0, unsigned short* __restrict__ o1, const unsigned long long s0, const unsigned long long s1) {
const unsigned long long linear = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
if (linear >= s0) return;
const unsigned long long a0 = linear;
float r2 = static_cast<float>(0);
for (unsigned long long a1 = 0; a1 < s1; ++a1)
{
const float v0 = load_bfloat(o0[a0 * s1 + a1]);
const float v1 = __expf(v0);
r2 = singe_combine<float, 0>(r2, v1);
}
const float v2 = r2;
for (unsigned long long a1 = 0; a1 < s1; ++a1)
{
const float v0 = load_bfloat(o0[a0 * s1 + a1]);
const float v1 = __expf(v0);
const float v3 = __fdividef(v1, v2);
o1[a0 * s1 + a1] = store_bfloat(v3);
}
}
"
    );
}

#[test]
fn test_row_schedules_split_reductions_across_a_block() {
    let source = emit(
        &normalized_exponentials(),
        Schedule::Rows {
            threads: 128,
            vector: 1,
            tiles: [1, 1],
        },
    )
    .unwrap();
    let body = source.render_definition(EntryLinkage::C);
    assert!(body.contains("const unsigned long long linear = blockIdx.x;"));
    assert!(
        body.contains(
            "for (unsigned long long flat = threadIdx.x; flat < s1; flat += blockDim.x) {"
        )
    );
    assert!(body.contains("const float v2 = singe_block_reduce<float, 0>(r2, partials2);"));
}

#[test]
fn test_warp_schedules_give_each_row_a_warp_of_a_shared_block() {
    let kernel = normalized_exponentials();
    let schedule = Schedule::Warps {
        threads: 128,
        vector: 1,
        lanes: 32,
        tiles: [1, 1],
    };
    let source = emit(&kernel, schedule).unwrap();
    let body = source.render_definition(EntryLinkage::C);
    assert!(body.contains("const unsigned long long row = static_cast<unsigned long long>(blockIdx.x) * (blockDim.x / 32U) + threadIdx.x / 32U;"));
    assert!(body.contains("for (unsigned long long flat = warp_lane; flat < s1; flat += 32U) {"));
    assert!(body.contains("const float v2 = singe_warp_reduce<float, 0, 32>(r2);"));
    assert!(!body.contains("__syncthreads"));
    assert!(body.contains("if (row >= s0) return;"));
    // Four warps a block cover nine rows in three blocks.
    assert_eq!(
        schedule.launch(&kernel, &[9, 40]).unwrap(),
        Launch {
            blocks: 3,
            threads: 128
        }
    );
}

#[test]
fn test_emission_rejects_invalid_kernels_and_schedules() {
    let mut kernel = normalized_exponentials();
    kernel.stores.clear();
    assert!(matches!(
        emit(
            &kernel,
            Schedule::Elements {
                threads: 128,
                vector: 1
            }
        ),
        Err(EmitError::Verify(_))
    ));
    assert!(matches!(
        emit(
            &normalized_exponentials(),
            Schedule::Rows {
                threads: 100,
                vector: 1,
                tiles: [1, 1]
            }
        ),
        Err(EmitError::Schedule(_))
    ));
}

/// Builds `out[i, j] = x[i, j] + y[j]` for BF16 `x`, F32 `y` and `out`, five columns, and an argument row count, with `x` transposed when asked.
fn broadcast_addition(transposed: bool) -> Kernel {
    let mut kernel = Kernel::new("singe_test");
    let x = kernel.add_operand(ElementType::Bf16, OperandRole::Input);
    let y = kernel.add_operand(ElementType::F32, OperandRole::Input);
    let out = kernel.add_operand(ElementType::F32, OperandRole::Output);
    let rows = kernel.add_size(Size::Argument);
    let columns = kernel.add_size(Size::Static(5));
    let one = kernel.add_size(Size::Static(1));
    let i = kernel.add_axis(rows, AxisKind::Parallel);
    let j = kernel.add_axis(columns, AxisKind::Parallel);
    let lhs = kernel.load(Access {
        operand: x,
        base: None,
        indirect: Vec::new(),
        strides: if transposed {
            vec![(i, one), (j, rows)]
        } else {
            vec![(i, columns), (j, one)]
        },
    });
    let rhs = kernel.load(Access {
        operand: y,
        base: None,
        indirect: Vec::new(),
        strides: vec![(j, one)],
    });
    let sum = kernel.binary(BinaryOp::Add, lhs, rhs);
    kernel.store(
        Access {
            operand: out,
            base: None,
            indirect: Vec::new(),
            strides: vec![(i, columns), (j, one)],
        },
        sum,
    );
    kernel
}

#[test]
fn test_vector_schedules_stage_contiguous_accesses_through_wide_ones() {
    let kernel = broadcast_addition(false);
    let schedule = element_schedule(&kernel, 128);
    assert_eq!(
        schedule,
        Schedule::Elements {
            threads: 128,
            vector: 4
        }
    );
    // Three rows of five columns take two vectors each.
    assert_eq!(
        schedule.launch(&kernel, &[3]).unwrap(),
        Launch {
            blocks: 1,
            threads: 128
        }
    );
    assert_eq!(
        Schedule::Elements {
            threads: 2,
            vector: 4
        }
        .launch(&kernel, &[3])
        .unwrap()
        .blocks,
        3
    );
    let source = emit(&kernel, schedule).unwrap();
    let body = source.render_definition(EntryLinkage::C);
    assert!(body.contains("if (linear >= s0 * ((5ULL + 3ULL) / 4ULL)) return;"));
    assert!(body.contains("const unsigned long long a1_base = a1_group * 4ULL;"));
    // Four BF16 inputs make one 8-byte access, and four F32 values one 16-byte access.
    assert!(body.contains("*reinterpret_cast<uint2*>(raw0) = *reinterpret_cast<const uint2*>((o0 + a0 * 5ULL + a1_base));"));
    assert!(body.contains(
        "*reinterpret_cast<uint4*>(raw1) = *reinterpret_cast<const uint4*>((o1 + a1_base));"
    ));
    assert!(body.contains("*reinterpret_cast<uint4*>((o2 + a0 * 5ULL + a1_base)) = *reinterpret_cast<const uint4*>(staged0);"));
    assert!(body.contains("const float v0 = load_bfloat(raw0[lane]);"));
}

#[test]
fn test_element_schedules_take_one_lane_unless_every_access_is_contiguous() {
    assert_eq!(
        element_schedule(&broadcast_addition(true), 64),
        Schedule::Elements {
            threads: 64,
            vector: 1
        }
    );
    assert_eq!(
        element_schedule(&normalized_exponentials(), 64),
        Schedule::Elements {
            threads: 64,
            vector: 1
        }
    );
    // A transposed operand is read lane by lane under a forced vector.
    let source = emit(
        &broadcast_addition(true),
        Schedule::Elements {
            threads: 64,
            vector: 2,
        },
    )
    .unwrap();
    assert!(source.render_definition(EntryLinkage::C).contains(
        "const float v0 = (a1 < 5ULL ? load_bfloat(o0[a0 + a1 * s0]) : static_cast<float>(0));"
    ));
    assert!(matches!(
        emit(
            &normalized_exponentials(),
            Schedule::Elements {
                threads: 64,
                vector: 2
            }
        ),
        Err(EmitError::Unsupported { .. })
    ));
    assert!(matches!(
        emit(
            &broadcast_addition(false),
            Schedule::Elements {
                threads: 64,
                vector: 3
            }
        ),
        Err(EmitError::Schedule(_))
    ));
}

#[test]
fn test_indexed_accesses_check_their_indices_and_scatters_add_atomically() {
    // out[index[i], j] += x[i, j] for rows of four.
    let mut kernel = Kernel::new("singe_test");
    let x = kernel.add_operand(ElementType::F32, OperandRole::Input);
    let index = kernel.add_operand(ElementType::I64, OperandRole::Input);
    let out = kernel.add_operand(ElementType::F32, OperandRole::Output);
    let rows = kernel.add_size(Size::Argument);
    let width = kernel.add_size(Size::Static(4));
    let one = kernel.add_size(Size::Static(1));
    let targets = kernel.add_size(Size::Static(9));
    let i = kernel.add_axis(rows, AxisKind::Parallel);
    let j = kernel.add_axis(width, AxisKind::Parallel);
    let value = kernel.load(Access {
        operand: x,
        base: None,
        indirect: Vec::new(),
        strides: vec![(i, width), (j, one)],
    });
    let position = kernel.load(Access {
        operand: index,
        base: None,
        indirect: Vec::new(),
        strides: vec![(i, one)],
    });
    let position = kernel.check_index(position, targets, true, 1);
    kernel.store_sum(
        Access {
            operand: out,
            base: None,
            indirect: vec![(position, width)],
            strides: vec![(j, one)],
        },
        value,
    );
    assert!(kernel.reports_status());
    assert!(
        kernel
            .to_string()
            .contains("store o2[a1*s2 + %2*s1] sum= %0")
    );
    // Atomic stores never share a vector.
    let schedule = element_schedule(&kernel, 128);
    assert_eq!(
        schedule,
        Schedule::Elements {
            threads: 128,
            vector: 1
        }
    );
    let source = emit(&kernel, schedule).unwrap();
    let body = source.render_definition(EntryLinkage::C);
    assert!(
        body.contains("float* __restrict__ o2, unsigned int* status, const unsigned long long s0")
    );
    assert!(body.contains("const unsigned long long v2 = singe_check_index(status, static_cast<long long>(v1), 9ULL, true, 1U);"));
    assert!(body.contains("atomicAdd(&o2[a1 + static_cast<unsigned long long>(v2) * 4ULL], v0);"));
}

#[test]
fn test_row_schedules_cover_a_vector_of_the_innermost_reduction_axis_per_iteration() {
    let kernel = normalized_exponentials();
    assert_eq!(reduction_vector(&kernel, 16), 8);
    assert_eq!(reduction_vector(&kernel, 4), 4);
    let source = emit(
        &kernel,
        Schedule::Warps {
            threads: 128,
            vector: 8,
            lanes: 32,
            tiles: [1, 1],
        },
    )
    .unwrap();
    let body = source.render_definition(EntryLinkage::C);
    assert!(body.contains(
        "for (unsigned long long flat = warp_lane; flat < ((s1 + 7ULL) / 8ULL); flat += 32U) {"
    ));
    // Eight BF16 elements make one 16-byte load, and one 16-byte store.
    assert!(body.contains("*reinterpret_cast<uint4*>(raw0) = *reinterpret_cast<const uint4*>((o0 + a0 * s1 + a1_base));"));
    assert!(body.contains("*reinterpret_cast<uint4*>((o1 + a0 * s1 + a1_base)) = *reinterpret_cast<const uint4*>(staged);"));
    assert!(body.contains("r2 = singe_combine<float, 0>(r2, v1);"));
    assert!(body.contains("if (a1 < s1) {"));
}

#[test]
fn test_rows_of_static_width_keep_the_loads_every_loop_reads_in_registers() {
    // The softmax of `normalized_exponentials`, with rows of 256 baked into the code.
    let mut kernel = normalized_exponentials();
    kernel.sizes[1] = Size::Static(256);
    let source = emit(
        &kernel,
        Schedule::Warps {
            threads: 128,
            vector: 1,
            lanes: 32,
            tiles: [1, 1],
        },
    )
    .unwrap();
    let cached = source.render_definition(EntryLinkage::C);
    assert!(cached.contains("float cached0[8];"));
    assert!(cached.contains("for (unsigned int point = 0; point < 8U; ++point) {"));
    // The first loop loads each point and keeps it; the second reads the register.
    assert_eq!(cached.matches("cached0[point] = v0;").count(), 1);
    assert_eq!(
        cached.matches("const float v0 = cached0[point];").count(),
        1
    );
    // Rows longer than 64 points per thread walk memory instead.
    kernel.sizes[1] = Size::Static(4096);
    let source = emit(
        &kernel,
        Schedule::Warps {
            threads: 128,
            vector: 1,
            lanes: 32,
            tiles: [1, 1],
        },
    )
    .unwrap();
    assert!(
        !source
            .render_definition(EntryLinkage::C)
            .contains("cached0[")
    );
}

#[test]
fn test_vector_rows_of_static_width_load_each_vector_once_into_registers() {
    let mut kernel = normalized_exponentials();
    kernel.sizes[1] = Size::Static(256);
    let source = emit(
        &kernel,
        Schedule::Warps {
            threads: 128,
            vector: 8,
            lanes: 32,
            tiles: [1, 1],
        },
    )
    .unwrap();
    let cached = source.render_definition(EntryLinkage::C);
    assert!(cached.contains("float cached0[8];"));
    // One 16-byte load fills the registers, and the second loop reads them.
    assert_eq!(cached.matches("*reinterpret_cast<uint4*>(raw0)").count(), 1);
    assert_eq!(
        cached.matches("cached0[point * 8U + lane] = v0;").count(),
        1
    );
    assert_eq!(
        cached
            .matches("const float v0 = cached0[point * 8U + lane];")
            .count(),
        1
    );
}

#[test]
fn test_warp_schedules_with_fewer_lanes_put_several_rows_in_a_warp() {
    let kernel = normalized_exponentials();
    let schedule = Schedule::Warps {
        threads: 128,
        vector: 1,
        lanes: 8,
        tiles: [1, 1],
    };
    let source = emit(&kernel, schedule).unwrap();
    let body = source.render_definition(EntryLinkage::C);
    assert!(body.contains("const unsigned int warp_lane = threadIdx.x % 8U;"));
    assert!(body.contains("for (unsigned long long flat = warp_lane; flat < s1; flat += 8U) {"));
    assert!(body.contains("const float v2 = singe_warp_reduce<float, 0, 8>(r2);"));
    // Sixteen rows a block cover forty rows in three blocks.
    assert_eq!(schedule.launch(&kernel, &[40, 7]).unwrap().blocks, 3);
    assert!(matches!(
        Schedule::Warps {
            threads: 128,
            vector: 1,
            lanes: 3,
            tiles: [1, 1]
        }
        .validate(),
        Err(ScheduleError::Lanes { lanes: 3 })
    ));
}

#[test]
fn test_32_bit_indices_narrow_argument_sizes_and_write_32_bit_literals() {
    let kernel = normalized_exponentials();
    let source = cuda_c::emit(
        &kernel,
        ThreadSchedule::Rows {
            threads: 128,
            vector: 1,
            tiles: [1, 1],
        },
        IndexWidth::U32,
    )
    .unwrap();
    let body = source.render_definition(EntryLinkage::C);
    assert!(body.contains("const unsigned long long wide_s0, const unsigned long long wide_s1) {"));
    assert!(body.contains("const unsigned int s0 = static_cast<unsigned int>(wide_s0);"));
    assert!(body.contains("const unsigned int linear = blockIdx.x;"));
    assert!(!body.contains("ULL"), "{body}");
    assert!(
        !body[body.find(") {").unwrap()..].contains("unsigned long long"),
        "{body}"
    );

    // Spans of 2^31 elements or more keep 64-bit indices.
    assert_eq!(IndexWidth::of(&kernel, &[4, 1024]), IndexWidth::U32);
    assert_eq!(IndexWidth::of(&kernel, &[1 << 21, 1024]), IndexWidth::U64);
    assert_eq!(IndexWidth::of(&kernel, &[1, 1 << 31]), IndexWidth::U64);
}

// The emitter's version and the FNV-1a hash of the text it writes for `emitted_corpus`; both change together.
const RECORDED_EMISSION: (u32, u64) = (7, 520_183_809_769_659_205);

// Every kernel above under every candidate schedule and both index widths, so a change to any emitted text changes the corpus.
fn emitted_corpus() -> String {
    let kernels = [
        normalized_exponentials(),
        broadcast_addition(false),
        broadcast_addition(true),
    ];
    let mut corpus = String::new();
    for kernel in &kernels {
        let sizes = vec![64; kernel.argument_sizes().count()];
        for schedule in candidates(kernel, &sizes, false)
            .into_iter()
            .filter_map(Schedule::thread)
        {
            for index in [IndexWidth::U32, IndexWidth::U64] {
                if let Ok(source) = cuda_c::emit(kernel, schedule, index) {
                    corpus.push_str(&source.render_definition(EntryLinkage::C));
                }
            }
        }
    }
    corpus
}

#[test]
fn test_emitted_text_changes_only_with_the_emitter_version() {
    let hash = emitted_corpus()
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
        });
    assert_eq!(
        (EMITTER_VERSION, hash),
        RECORDED_EMISSION,
        "the emitted text changed: bump `EMITTER_VERSION` and record the new hash"
    );
}

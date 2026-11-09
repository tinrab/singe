use std::{collections::HashMap, iter};

use singe_codegen::{emit::EntryLinkage, schedule::Schedule};
use singe_cuda::architecture::GpuArchitecture;

use super::SplitParameters;
use crate::{
    backend::{
        cuda::{
            self,
            codegen::{
                BufferDescription, KernelArgument, KernelCode, KernelLanguage, PlanningContext,
            },
            compiled::executable::GraphCompileProviderDecision,
            kernel_plan::{self, KernelInput, KernelPlan, KernelPlanner},
        },
        kernel_lowering,
        registry::{
            LayerNormalizationAttributes, Operation, OperationAttributes, ReductionAttributes,
            ScaledDotProductAttentionAttributes,
        },
    },
    data_type::DataType,
    device::Device,
    dispatch::call_one,
    error::Result,
    graph::{Graph, GraphCudaOptions},
    layout::Layout,
    shape::Shape,
    tensor::{AttentionOptions, BlockScaledFormat, Paging, Tensor},
    testing::{maybe_lock_device, skip_if_stub},
};

fn input(data_type: DataType, dimensions: &[i64]) -> KernelInput {
    KernelInput::new(data_type, Layout::contiguous(dimensions).unwrap())
}

fn add(lhs: KernelInput, rhs: KernelInput) -> KernelPlan {
    KernelPlanner::default()
        .create(Operation::Add, &OperationAttributes::None, &[lhs, rhs])
        .unwrap()
}

fn softmax(dimensions: &[i64]) -> KernelPlan {
    KernelPlanner::default()
        .create(
            Operation::Softmax,
            &OperationAttributes::Dimension(-1),
            &[input(DataType::Bf16, dimensions)],
        )
        .unwrap()
}

fn rms_norm(rows: i64, width: i64) -> KernelPlan {
    let attributes = OperationAttributes::LayerNormalization(LayerNormalizationAttributes {
        normalized_shape: vec![width],
        has_weight: true,
        has_bias: false,
        eps: 1e-6,
    });
    KernelPlanner::default()
        .create(
            Operation::RmsNorm,
            &attributes,
            &[
                input(DataType::Bf16, &[rows, width]),
                input(DataType::Bf16, &[width]),
            ],
        )
        .unwrap()
}

#[test]
fn test_contiguous_kernels_share_one_source_across_shapes() {
    let small = add(input(DataType::F32, &[4, 8]), input(DataType::F32, &[4, 8]));
    let large = add(
        input(DataType::F32, &[16, 8]),
        input(DataType::F32, &[16, 8]),
    );
    assert_eq!(small.kernel.code, large.kernel.code);
    assert_eq!((extents(&small), extents(&large)), (vec![32], vec![128]));

    assert_eq!(
        softmax(&[2, 513]).kernel.code,
        softmax(&[3, 513]).kernel.code
    );
    // Rows are part of the code; the row count is passed at launch.
    assert_eq!(extents(&softmax(&[3, 513])), vec![3]);
    assert_eq!(rms_norm(3, 1024).kernel.code, rms_norm(5, 1024).kernel.code);
    assert_eq!(
        rms_norm(128, 1024).kernel.code,
        rms_norm(256, 1024).kernel.code
    );
}

#[test]
fn test_strided_operands_specialize_the_source_to_their_layout() {
    let transposed = |rows: i64| {
        KernelInput::new(
            DataType::F32,
            Layout::contiguous([rows, 4])
                .unwrap()
                .transpose(0, 1)
                .unwrap(),
        )
    };
    let narrow = add(transposed(8), input(DataType::F32, &[4, 8]));
    let wide = add(transposed(16), input(DataType::F32, &[4, 16]));
    assert_ne!(narrow.kernel.code, wide.kernel.code);
}

/// Returns the number of input buffers the kernel reads.
fn inputs(plan: &KernelPlan) -> usize {
    plan.kernel
        .abi
        .arguments
        .iter()
        .filter(|argument| matches!(argument, KernelArgument::Input(_)))
        .count()
}

/// Returns the values of the kernel's extent arguments.
fn extents(plan: &KernelPlan) -> Vec<u64> {
    plan.kernel
        .abi
        .arguments
        .iter()
        .filter_map(|argument| match argument {
            KernelArgument::Size(extent) => Some(*extent),
            _ => None,
        })
        .collect()
}

/// Returns the kernel's declaration line.
fn signature(plan: &KernelPlan) -> String {
    plan.kernel
        .code
        .source()
        .expect("the kernel is CUDA C++")
        .render_definition(EntryLinkage::C)
        .lines()
        .find(|line| line.contains("__global__"))
        .unwrap()
        .to_owned()
}

#[test]
fn test_kernel_signatures_list_the_inputs_they_read_and_a_status_word_only_when_checking() {
    let relu = KernelPlanner::default()
        .create(
            Operation::Relu,
            &OperationAttributes::None,
            &[input(DataType::F32, &[4])],
        )
        .unwrap();
    assert_eq!(inputs(&relu), 1);
    assert!(!signature(&relu).contains("p1"));
    assert!(!signature(&relu).contains("status"));

    let gather = KernelPlanner::default()
        .create(
            Operation::IndexSelect,
            &OperationAttributes::IndexSelect(0),
            &[input(DataType::F32, &[4, 3]), input(DataType::I64, &[2])],
        )
        .unwrap();
    assert!(gather.kernel.abi.checks());
    assert_eq!(inputs(&gather), 2);
    assert!(signature(&gather).contains("unsigned int* status"));
}

#[test]
fn test_plans_describe_their_buffers_operations_and_launch_values() {
    let norm = rms_norm(6, 32).kernel.description;
    assert_eq!(norm.operations, ["rms_norm"]);
    let buffers = |buffers: &[BufferDescription]| {
        buffers
            .iter()
            .map(|buffer| {
                (
                    buffer.data_type,
                    buffer.layout.dimensions().to_vec(),
                    buffer.layout.strides().to_vec(),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        buffers(&norm.inputs),
        [
            (DataType::Bf16, vec![6, 32], vec![32, 1]),
            (DataType::Bf16, vec![32], vec![1]),
        ]
    );
    assert_eq!(
        buffers(&norm.outputs),
        [(DataType::Bf16, vec![6, 32], vec![32, 1])]
    );
    assert_eq!(norm.outputs[0].producer.as_deref(), Some("rms_norm"));
    assert_eq!(norm.scalars, ["normalized width", "epsilon"]);
    assert_eq!(norm.sizes, ["outermost parallel extent"]);

    let broadcast = add(
        input(DataType::F32, &[4, 8]).with_name("x"),
        input(DataType::F32, &[8]),
    )
    .kernel
    .description;
    assert_eq!(broadcast.operations, ["add"]);
    assert_eq!(broadcast.inputs[0].name.as_deref(), Some("x"));
    assert_eq!(broadcast.inputs[1].name, None);
    assert_eq!(
        buffers(&broadcast.inputs),
        [
            (DataType::F32, vec![4, 8], vec![8, 1]),
            (DataType::F32, vec![8], vec![1]),
        ]
    );
    assert_eq!(broadcast.sizes, ["outermost extent"]);
}

#[test]
fn test_planning_for_another_architecture_records_it_and_gates_tile_ir_on_it() {
    let planner = |language, target| {
        KernelPlanner::new(PlanningContext {
            language,
            target: Some(target),
            ..PlanningContext::default()
        })
    };
    let add = |planner: &KernelPlanner| {
        planner
            .create(
                Operation::Add,
                &OperationAttributes::None,
                &[input(DataType::F32, &[4, 8]), input(DataType::F32, &[4, 8])],
            )
            .unwrap()
            .kernel
    };
    let hopper = add(&planner(KernelLanguage::CudaC, GpuArchitecture::Sm90));
    assert_eq!(hopper.target, Some(GpuArchitecture::Sm90));
    assert!(matches!(hopper.code, KernelCode::Source(_)));
    // `tileiras` compiles for `sm_80` and later, so Tile IR planning keeps CUDA C++ for `sm_75`.
    let tiled = add(&planner(KernelLanguage::TileIr, GpuArchitecture::Sm120));
    assert!(matches!(tiled.code, KernelCode::TileIr(_)));
    let turing = add(&planner(KernelLanguage::TileIr, GpuArchitecture::Sm75));
    assert!(matches!(turing.code, KernelCode::Source(_)));
    assert_eq!(turing.target, Some(GpuArchitecture::Sm75));
}

fn reduce(operation: Operation, data_type: DataType, dimensions: &[i64]) -> KernelPlan {
    KernelPlanner::default()
        .create(
            operation,
            &OperationAttributes::Reduction(ReductionAttributes {
                dimensions: Vec::new(),
                keepdim: false,
            }),
            &[input(data_type, dimensions)],
        )
        .unwrap()
}

#[test]
fn test_whole_tensor_reductions_split_across_blocks_with_declared_scratch() {
    let split = reduce(Operation::Sum, DataType::Bf16, &[512, 1024]);
    assert!(
        split.kernel.name.contains("split_sum"),
        "{}",
        split.kernel.name
    );
    assert_eq!(split.kernel.launch.blocks, 256);
    // `F32` partials, one per block, and the arrival count the last block resets.
    assert_eq!(split.kernel.abi.scratch, [256 * 4, 4]);
    assert!(
        split
            .kernel
            .abi
            .arguments
            .contains(&KernelArgument::Scratch(1))
    );

    let rows = reduce(Operation::Sum, DataType::Bf16, &[64, 64]);
    assert!(!rows.kernel.name.contains("split"), "{}", rows.kernel.name);
    assert!(rows.kernel.abi.scratch.is_empty());
}

#[test]
fn test_split_reductions_match_the_cpu_on_every_launch() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let count: i64 = 300_001;
    let data = (0..count)
        .map(|index| ((index * 7919 % 1013) as f64 / 1013.0) - 0.5)
        .collect::<Vec<_>>();
    let cpu = Tensor::from_vec(Device::Cpu, Shape::create([count]).unwrap(), data).unwrap();
    let Some(device) = skip_if_stub(cpu.to_device(Device::Cuda(0))) else {
        return;
    };
    for data_type in [DataType::F64, DataType::Bf16] {
        let (cpu, device) = (
            cpu.cast(data_type).unwrap(),
            device.cast(data_type).unwrap(),
        );
        for operation in [
            Operation::Sum,
            Operation::Mean,
            Operation::Max,
            Operation::Min,
        ] {
            let reduced = |tensor: &Tensor| {
                call_one(
                    operation,
                    OperationAttributes::Reduction(ReductionAttributes {
                        dimensions: Vec::new(),
                        keepdim: false,
                    }),
                    &[tensor],
                )
                .unwrap()
                .cast(DataType::F64)
                .unwrap()
                .to_vec::<f64>()
                .unwrap()
            };
            let expected = reduced(&cpu);
            // A second launch finds the arrival count the first reset.
            for _ in 0..2 {
                let actual = reduced(&device);
                let tolerance = if data_type == DataType::Bf16 {
                    0.05
                } else {
                    1e-9
                };
                assert!(
                    (actual[0] - expected[0]).abs() <= tolerance * expected[0].abs().max(1.0),
                    "{operation:?} {data_type:?}: {actual:?} against {expected:?}"
                );
            }
        }
    }
}

#[test]
fn test_tensor_core_matmuls_match_the_cpu_on_edges_and_batches() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let values = |dimensions: &[i64], seed: i64| {
        let shape = Shape::create(dimensions).unwrap();
        let count = i64::try_from(shape.element_count()).unwrap();
        let data = (0..count)
            .map(|index| ((index * 37 + seed * 11) % 29) as f32 / 29.0 - 0.5)
            .collect::<Vec<_>>();
        Tensor::from_vec(Device::Cpu, shape, data).unwrap()
    };
    // The right operand is `[k, n]`, or, transposed, `[n, k]` rows used as their transpose, as a linear layer's weight is.
    // The first two have too few tiles to fill the GPU, so their depth splits across blocks and sums after.
    // The third has 128 tiles of 256 by 128, which the largest tiling takes.
    let cases: [(&[i64], &[i64], bool); 12] = [
        (&[32, 1024], &[64, 1024], true),
        (&[20, 512], &[512, 40], false),
        (&[512, 64], &[64, 8192], false),
        (&[16, 8], &[8, 8], false),
        (&[130, 72], &[72, 136], false),
        (&[256, 256], &[256, 256], false),
        (&[3, 40, 64], &[64, 24], false),
        (&[40, 64], &[2, 64, 24], false),
        (&[2, 1, 33, 48], &[2, 1, 48, 16], false),
        (&[16, 64], &[136, 64], true),
        (&[130, 72], &[200, 72], true),
        (&[2, 33, 48], &[2, 16, 48], true),
    ];
    for data_type in [DataType::F16, DataType::Bf16] {
        for (lhs, rhs, transposed) in cases {
            let (lhs, rhs) = (values(lhs, 1), values(rhs, 2));
            let oriented = |tensor: Tensor| {
                if transposed {
                    tensor.transpose(-2, -1).unwrap()
                } else {
                    tensor
                }
            };
            let expected = lhs
                .cast(data_type)
                .unwrap()
                .cast(DataType::F32)
                .unwrap()
                .matmul(&oriented(
                    rhs.cast(data_type).unwrap().cast(DataType::F32).unwrap(),
                ))
                .unwrap();
            let Some(device_lhs) = skip_if_stub(lhs.to_device(Device::Cuda(0))) else {
                return;
            };
            let device_lhs = device_lhs.cast(data_type).unwrap();
            let device_rhs = oriented(
                rhs.to_device(Device::Cuda(0))
                    .unwrap()
                    .cast(data_type)
                    .unwrap(),
            );
            let plan = KernelPlanner::default()
                .create(
                    Operation::MatMul,
                    &OperationAttributes::None,
                    &[
                        KernelInput::from_source(&device_lhs),
                        KernelInput::from_source(&device_rhs),
                    ],
                )
                .unwrap();
            assert!(
                plan.kernel.name.contains("mma_matmul"),
                "{}",
                plan.kernel.name
            );
            let actual = kernel_plan::run(&plan, &[&device_lhs, &device_rhs]).unwrap();
            assert_eq!(actual.shape(), expected.shape());
            let (actual, expected) = (
                actual.cast(DataType::F32).unwrap().to_vec::<f32>().unwrap(),
                expected.to_vec::<f32>().unwrap(),
            );
            let tolerance = if data_type == DataType::Bf16 {
                0.06
            } else {
                0.01
            };
            for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
                    "{data_type:?} {:?}: element {index} is {actual}, not {expected}",
                    plan.output().shape()
                );
            }
        }
    }
}

#[test]
fn test_fused_attention_matches_the_cpu_with_masks_edges_and_grouped_heads() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let values = |dimensions: &[i64], seed: i64| {
        let shape = Shape::create(dimensions).unwrap();
        let count = i64::try_from(shape.element_count()).unwrap();
        let data = (0..count)
            .map(|index| ((index * 37 + seed * 11) % 41) as f32 / 20.0 - 1.0)
            .collect::<Vec<_>>();
        Tensor::from_vec(Device::Cpu, shape, data).unwrap()
    };
    // Queries `[B, H, L, D]`, keys and values `[B, G, S, D]`, and whether the mask is causal.
    // The last two have too few blocks to fill the GPU, so their keys split across blocks and combine after.
    let cases: [([i64; 4], [i64; 4], bool); 8] = [
        ([1, 8, 1, 64], [1, 2, 1000, 64], false),
        ([1, 1, 100, 64], [1, 1, 300, 64], true),
        ([1, 2, 70, 64], [1, 2, 70, 64], true),
        ([2, 6, 23, 64], [2, 2, 23, 64], true),
        ([1, 2, 70, 64], [1, 2, 70, 64], false),
        ([2, 4, 33, 128], [2, 2, 100, 128], false),
        ([1, 8, 1, 64], [1, 2, 200, 64], false),
        ([1, 1, 130, 32], [1, 1, 130, 32], true),
    ];
    // Each case runs with and without an additive mask, per head on even cases and broadcast over heads on odd ones, hiding every seventh key, and with and without sink logits.
    for data_type in [DataType::F16, DataType::Bf16] {
        for (case, (queries, keys, causal)) in cases.into_iter().enumerate() {
            for (masked, sunk) in [(false, false), (true, false), (false, true), (true, true)] {
                let (query, key, value) = (values(&queries, 1), values(&keys, 2), values(&keys, 3));
                let mask_heads = if case % 2 == 0 { queries[1] } else { 1 };
                let mask = values(&[queries[0], mask_heads, queries[2], keys[2]], 4);
                let hidden = Tensor::from_vec(
                    Device::Cpu,
                    Shape::create([keys[2]]).unwrap(),
                    (0..keys[2]).map(|key| key % 7 == 3).collect(),
                )
                .unwrap();
                let infinity = Tensor::scalar(Device::Cpu, f32::NEG_INFINITY).unwrap();
                let mask = Tensor::where_select(&hidden, &infinity, &mask).unwrap();
                let sinks = values(&[queries[1]], 5).mul_scalar(2.0).unwrap();
                let groups = queries[1] / keys[1];
                let rounded =
                    |tensor: &Tensor| tensor.cast(data_type).unwrap().cast(DataType::F32).unwrap();
                let options = AttentionOptions {
                    causal,
                    ..AttentionOptions::default()
                };
                let expected = rounded(&query)
                    .attend(
                        &rounded(&key).repeat_kv(groups).unwrap(),
                        &rounded(&value).repeat_kv(groups).unwrap(),
                        masked.then(|| rounded(&mask)).as_ref(),
                        sunk.then(|| rounded(&sinks)).as_ref(),
                        options,
                    )
                    .unwrap()
                    .to_vec::<f32>()
                    .unwrap();
                let device = |tensor: &Tensor| {
                    skip_if_stub(tensor.to_device(Device::Cuda(0)))
                        .map(|tensor| tensor.cast(data_type).unwrap())
                };
                let (Some(query), Some(key), Some(value), Some(mask), Some(sinks)) = (
                    device(&query),
                    device(&key),
                    device(&value),
                    device(&mask),
                    device(&sinks),
                ) else {
                    return;
                };
                let mask = masked.then_some(&mask);
                let sinks = sunk.then_some(&sinks);
                let plan = KernelPlanner::default()
                    .fused_attention(
                        &KernelInput::from_source(&query),
                        &KernelInput::from_source(&key),
                        &KernelInput::from_source(&value),
                        mask.map(KernelInput::from_source).as_ref(),
                        sinks.map(KernelInput::from_source).as_ref(),
                        None,
                        ScaledDotProductAttentionAttributes {
                            scale: None,
                            causal,
                            window: None,
                            softcap: None,
                            masked,
                            sinks: sunk,
                            key_lengths: false,
                        },
                    )
                    .unwrap()
                    .unwrap();
                let inputs = [&query, &key, &value]
                    .into_iter()
                    .chain(mask)
                    .chain(sinks)
                    .collect::<Vec<_>>();
                let actual = kernel_plan::run(&plan, &inputs)
                    .unwrap()
                    .cast(DataType::F32)
                    .unwrap()
                    .to_vec::<f32>()
                    .unwrap();
                let tolerance = if data_type == DataType::Bf16 {
                    0.03
                } else {
                    0.005
                };
                for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "{data_type:?} {queries:?} by {keys:?}, causal {causal}, masked {masked}, sinks {sunk}: element {index} is {actual}, not {expected}"
                    );
                }
            }
        }
    }
}

#[test]
fn test_matrix_vector_products_match_the_cpu() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let values = |dimensions: &[i64], seed: i64| {
        let shape = Shape::create(dimensions).unwrap();
        let count = i64::try_from(shape.element_count()).unwrap();
        let data = (0..count)
            .map(|index| ((index * 37 + seed * 11) % 29) as f32 / 29.0 - 0.5)
            .collect::<Vec<_>>();
        Tensor::from_vec(Device::Cpu, shape, data).unwrap()
    };
    // Rows, and `[n, k]` weight rows used transposed, as a linear layer's are.
    let cases: [(&[i64], &[i64]); 4] = [
        (&[1, 2048], &[1000, 2048]),
        (&[3, 72], &[77, 72]),
        (&[4, 1024], &[300, 1024]),
        (&[2, 1, 64], &[33, 64]),
    ];
    for data_type in [DataType::F32, DataType::F16, DataType::Bf16] {
        for (rows, weight) in cases {
            let (rows, weight) = (values(rows, 1), values(weight, 2));
            let rounded =
                |tensor: &Tensor| tensor.cast(data_type).unwrap().cast(DataType::F32).unwrap();
            let expected = rounded(&rows)
                .matmul(&rounded(&weight).transpose(0, 1).unwrap())
                .unwrap()
                .to_vec::<f32>()
                .unwrap();
            let Some(device_rows) = skip_if_stub(rows.to_device(Device::Cuda(0))) else {
                return;
            };
            let device_rows = device_rows.cast(data_type).unwrap();
            let device_weight = weight
                .to_device(Device::Cuda(0))
                .unwrap()
                .cast(data_type)
                .unwrap()
                .transpose(0, 1)
                .unwrap();
            let plan = KernelPlanner::default()
                .create(
                    Operation::MatMul,
                    &OperationAttributes::None,
                    &[
                        KernelInput::from_source(&device_rows),
                        KernelInput::from_source(&device_weight),
                    ],
                )
                .unwrap();
            assert!(
                plan.kernel.name.contains("dot_product"),
                "{}",
                plan.kernel.name
            );
            let actual = kernel_plan::run(&plan, &[&device_rows, &device_weight])
                .unwrap()
                .cast(DataType::F32)
                .unwrap()
                .to_vec::<f32>()
                .unwrap();
            let tolerance = match data_type {
                DataType::Bf16 => 0.06,
                DataType::F16 => 0.01,
                _ => 1e-4,
            };
            for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
                    "{data_type:?}: element {index} is {actual}, not {expected}"
                );
            }
        }
    }
}

#[test]
fn test_rotary_embedding_runs_as_one_generated_kernel() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let values = |dimensions: &[i64], seed: i64| {
        let shape = Shape::create(dimensions).unwrap();
        let count = i64::try_from(shape.element_count()).unwrap();
        let data = (0..count)
            .map(|index| ((index * 37 + seed * 11) % 29) as f32 / 29.0 - 0.5)
            .collect::<Vec<_>>();
        Tensor::from_vec(Device::Cpu, shape, data).unwrap()
    };
    let (input, cos, sin) = (
        values(&[2, 3, 5, 8], 1),
        values(&[5, 4], 2),
        values(&[5, 4], 3),
    );
    let expected = input
        .rotary_embedding(&cos, &sin)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    let Some(device_input) = skip_if_stub(input.to_device(Device::Cuda(0))) else {
        return;
    };
    let (device_cos, device_sin) = (
        cos.to_device(Device::Cuda(0)).unwrap(),
        sin.to_device(Device::Cuda(0)).unwrap(),
    );
    let plan = KernelPlanner::default()
        .create(
            Operation::RotaryEmbedding,
            &OperationAttributes::None,
            &[
                KernelInput::from_source(&device_input),
                KernelInput::from_source(&device_cos),
                KernelInput::from_source(&device_sin),
            ],
        )
        .unwrap();
    assert!(plan.kernel.name.contains("rotary"), "{}", plan.kernel.name);
    let actual = kernel_plan::run(&plan, &[&device_input, &device_cos, &device_sin])
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-6,
            "element {index} is {actual}, not {expected}"
        );
    }
}

// A normalization's input shape, normalized shape, operation, whether it has a weight and a bias, and its data type.
type Normalization = (
    &'static [i64],
    &'static [i64],
    Operation,
    bool,
    bool,
    DataType,
);

#[test]
fn test_normalizations_of_few_long_rows_split_across_blocks() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let values = |dimensions: &[i64], seed: i64| {
        let shape = Shape::create(dimensions).unwrap();
        let count = i64::try_from(shape.element_count()).unwrap();
        let data = (0..count)
            .map(|index| ((index * 37 + seed * 11) % 101) as f32 / 25.0 + 3.0)
            .collect::<Vec<_>>();
        Tensor::from_vec(Device::Cpu, shape, data).unwrap()
    };
    // Rows, the normalized shape, the operation, and whether it has a weight and a bias.
    let cases: [Normalization; 3] = [
        (
            &[3, 70000],
            &[70000],
            Operation::LayerNorm,
            true,
            true,
            DataType::F32,
        ),
        (
            &[2, 4, 32, 1024],
            &[4, 32, 1024],
            Operation::RmsNorm,
            true,
            false,
            DataType::Bf16,
        ),
        (
            &[5, 66000],
            &[66000],
            Operation::LayerNorm,
            false,
            false,
            DataType::F64,
        ),
    ];
    for (shape, normalized, operation, has_weight, has_bias, data_type) in cases {
        let input = values(shape, 1).cast(data_type).unwrap();
        let mut parameters = Vec::new();
        if has_weight {
            parameters.push(values(normalized, 2).cast(data_type).unwrap());
        }
        if has_bias {
            parameters.push(values(normalized, 3).cast(data_type).unwrap());
        }
        let attributes = OperationAttributes::LayerNormalization(LayerNormalizationAttributes {
            normalized_shape: normalized.to_vec(),
            has_weight,
            has_bias,
            eps: 1e-5,
        });
        // The reference normalizes the same rounded values in `F64`, row by row.
        let as_f64 = |tensor: &Tensor| tensor.cast(DataType::F64).unwrap().to_vec::<f64>().unwrap();
        let (points, weight, bias) = (
            as_f64(&input),
            has_weight.then(|| as_f64(&parameters[0])),
            has_bias.then(|| as_f64(parameters.last().unwrap())),
        );
        let width = usize::try_from(normalized.iter().product::<i64>()).unwrap();
        let mut expected = Vec::with_capacity(points.len());
        for row in points.chunks(width) {
            let mean = if operation == Operation::LayerNorm {
                row.iter().sum::<f64>() / width as f64
            } else {
                0.0
            };
            let variance =
                row.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / width as f64;
            let scale = (variance + 1e-5).sqrt().recip();
            for (position, value) in row.iter().enumerate() {
                let mut normalized = (value - mean) * scale;
                if let Some(weight) = &weight {
                    normalized *= weight[position];
                }
                if let Some(bias) = &bias {
                    normalized += bias[position];
                }
                expected.push(normalized);
            }
        }
        let Some(device_input) = skip_if_stub(input.to_device(Device::Cuda(0))) else {
            return;
        };
        let device_parameters = parameters
            .iter()
            .map(|parameter| parameter.to_device(Device::Cuda(0)).unwrap())
            .collect::<Vec<_>>();
        let device_inputs = iter::once(&device_input)
            .chain(&device_parameters)
            .collect::<Vec<_>>();
        let plan = KernelPlanner::default()
            .create(
                operation,
                &attributes,
                &device_inputs
                    .iter()
                    .map(|input| KernelInput::from_source(*input))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        assert_eq!(plan.before.len(), 2, "{}", plan.kernel.name);
        let actual = kernel_plan::run(&plan, &device_inputs)
            .unwrap()
            .cast(DataType::F64)
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        let tolerance = if data_type == DataType::Bf16 {
            0.05
        } else {
            1e-4
        };
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
                "{operation:?} {data_type:?}: element {index} is {actual}, not {expected}"
            );
        }
    }
}

#[test]
fn test_paged_attention_matches_the_cpu_over_ragged_sequences() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let values = |dimensions: &[i64], seed: i64| {
        let shape = Shape::create(dimensions).unwrap();
        let count = i64::try_from(shape.element_count()).unwrap();
        let data = (0..count)
            .map(|index| ((index * 37 + seed * 11) % 41) as f32 / 20.0 - 1.0)
            .collect::<Vec<_>>();
        Tensor::from_vec(Device::Cpu, shape, data).unwrap()
    };
    let indices = |values: Vec<i32>| {
        let shape = Shape::create([i64::try_from(values.len()).unwrap()]).unwrap();
        Tensor::from_vec(Device::Cpu, shape, values).unwrap()
    };
    // Each sequence's queries and keys: one decoding query over a long cache, a prefill of its whole cache, a chunk of a prefill, and a sequence with no keys.
    let sequences: [(i32, i32); 4] = [(1, 150), (70, 70), (20, 90), (3, 0)];
    let options = [
        AttentionOptions::default(),
        AttentionOptions {
            causal: true,
            ..AttentionOptions::default()
        },
        AttentionOptions {
            causal: true,
            window: Some(16),
            softcap: Some(30.0),
            ..AttentionOptions::default()
        },
    ];
    for (heads, kv_heads, dim, page_size) in [(8, 2, 64, 1), (8, 1, 128, 16), (4, 4, 64, 7)] {
        // Pages are handed out in a shuffled order, so the kernel must follow the page table.
        let mut query_offsets = vec![0];
        let mut page_offsets = vec![0];
        let mut last_page_lengths = Vec::new();
        let mut pages_needed = 0;
        for &(queries, keys) in &sequences {
            query_offsets.push(query_offsets.last().unwrap() + queries);
            let pages = (keys + page_size - 1) / page_size;
            pages_needed += pages;
            page_offsets.push(pages_needed);
            last_page_lengths.push(if keys == 0 {
                0
            } else {
                keys - (pages - 1) * page_size
            });
        }
        let page_count = pages_needed + 3;
        let pages = (0..pages_needed)
            .map(|page| (page * 7 + 2) % page_count)
            .collect::<Vec<_>>();
        let tokens = i64::from(*query_offsets.last().unwrap());
        let query = values(&[tokens, heads, dim], 1);
        let cache = [i64::from(page_count), i64::from(page_size), kv_heads, dim];
        let (key_cache, value_cache) = (values(&cache, 2), values(&cache, 3));
        let index_tensors = [
            indices(query_offsets),
            indices(page_offsets),
            indices(pages),
            indices(last_page_lengths),
        ];
        for option in options {
            for data_type in [DataType::F16, DataType::Bf16] {
                let rounded =
                    |tensor: &Tensor| tensor.cast(data_type).unwrap().cast(DataType::F32).unwrap();
                let expected = rounded(&query)
                    .paged_attention(
                        &rounded(&key_cache),
                        &rounded(&value_cache),
                        Paging {
                            query_offsets: &index_tensors[0],
                            page_offsets: &index_tensors[1],
                            pages: &index_tensors[2],
                            last_page_lengths: &index_tensors[3],
                        },
                        option,
                    )
                    .unwrap()
                    .0
                    .to_vec::<f32>()
                    .unwrap();
                let device = |tensor: &Tensor, cast: bool| {
                    skip_if_stub(tensor.to_device(Device::Cuda(0))).map(|tensor| {
                        if cast {
                            tensor.cast(data_type).unwrap()
                        } else {
                            tensor
                        }
                    })
                };
                let (Some(query), Some(key_cache), Some(value_cache)) = (
                    device(&query, true),
                    device(&key_cache, true),
                    device(&value_cache, true),
                ) else {
                    return;
                };
                let on_device = index_tensors
                    .iter()
                    .map(|tensor| device(tensor, false).unwrap())
                    .collect::<Vec<_>>();
                let actual = query
                    .paged_attention(
                        &key_cache,
                        &value_cache,
                        Paging {
                            query_offsets: &on_device[0],
                            page_offsets: &on_device[1],
                            pages: &on_device[2],
                            last_page_lengths: &on_device[3],
                        },
                        option,
                    )
                    .unwrap()
                    .0
                    .cast(DataType::F32)
                    .unwrap()
                    .to_vec::<f32>()
                    .unwrap();
                let tolerance = if data_type == DataType::Bf16 {
                    0.03
                } else {
                    0.005
                };
                for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "{data_type:?} heads {heads}/{kv_heads}, dim {dim}, page {page_size}, {option:?}: element {index} is {actual}, not {expected}"
                    );
                }
            }
        }
    }
}

#[test]
fn test_scaled_matmuls_on_fp8_tensor_cores_match_their_decomposition() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let values = |dimensions: &[i64], seed: i64| {
        let shape = Shape::create(dimensions).unwrap();
        let count = i64::try_from(shape.element_count()).unwrap();
        let data = (0..count)
            .map(|index| ((index * 37 + seed * 11) % 41) as f32 / 10.0 - 2.0)
            .collect::<Vec<_>>();
        Tensor::from_vec(Device::Cpu, shape, data).unwrap()
    };
    let scale = |value: f32| {
        Tensor::from_vec(Device::Cpu, Shape::create([1]).unwrap(), vec![value]).unwrap()
    };
    // A linear layer's input against its `[N, K]` weight, transposed into the column-major operand cuBLASLt takes, and a shape it does not.
    for (rows, depth, columns) in [(32_i64, 64_i64, 48_i64), (5, 24, 7)] {
        for (lhs_type, rhs_type) in [
            (DataType::F8E4M3, DataType::F8E4M3),
            (DataType::F8E4M3, DataType::F8E5M2),
        ] {
            for (output, rowwise) in [
                (DataType::Bf16, false),
                (DataType::F32, false),
                (DataType::Bf16, true),
            ] {
                let lhs = values(&[rows, depth], 1).cast(lhs_type).unwrap();
                let rhs = values(&[columns, depth], 2)
                    .cast(rhs_type)
                    .unwrap()
                    .transpose(0, 1)
                    .unwrap();
                // Row-wise scales give each row of the left operand and each column of the right its own.
                let (lhs_scale, rhs_scale) = if rowwise {
                    let vector = |count: i64, shape: [i64; 2], base: f32| {
                        Tensor::from_vec(
                            Device::Cpu,
                            Shape::create(shape).unwrap(),
                            (0..count).map(|index| base + index as f32 / 16.0).collect(),
                        )
                        .unwrap()
                    };
                    (
                        vector(rows, [rows, 1], 0.5),
                        vector(columns, [1, columns], 1.0),
                    )
                } else {
                    (scale(0.5), scale(1.5))
                };
                let expected = lhs
                    .scaled_matmul(&rhs, &lhs_scale, &rhs_scale, output)
                    .unwrap()
                    .cast(DataType::F32)
                    .unwrap()
                    .to_vec::<f32>()
                    .unwrap();
                let device = |tensor: &Tensor| skip_if_stub(tensor.to_device(Device::Cuda(0)));
                let (Some(lhs), Some(rhs), Some(lhs_scale), Some(rhs_scale)) = (
                    device(&lhs),
                    device(&values(&[columns, depth], 2).cast(rhs_type).unwrap()),
                    device(&lhs_scale),
                    device(&rhs_scale),
                ) else {
                    return;
                };
                let rhs = rhs.transpose(0, 1).unwrap();
                let eager = lhs
                    .scaled_matmul(&rhs, &lhs_scale, &rhs_scale, output)
                    .unwrap();
                let (graph, _) = Graph::capture(
                    [
                        ("lhs", lhs.clone()),
                        ("weight", rhs.transpose(0, 1).unwrap().contiguous().unwrap()),
                        ("lhs_scale", lhs_scale.clone()),
                        ("rhs_scale", rhs_scale.clone()),
                    ],
                    |inputs| {
                        Ok(vec![inputs[0].scaled_matmul(
                            &inputs[1].transpose(0, 1)?,
                            &inputs[2],
                            &inputs[3],
                            output,
                        )?])
                    },
                )
                .unwrap();
                let mut compiled = cuda::compile_graph(&graph, &GraphCudaOptions::new()).unwrap();
                if rows == 32 {
                    // Row-wise scales run the unscaled product on tensor cores too, and one kernel scales and rounds it.
                    assert_eq!(compiled.stats().cublas_child_graph_nodes(), 1);
                    assert_eq!(
                        compiled.stats().kernel_child_graph_nodes(),
                        usize::from(rowwise)
                    );
                }
                let compiled_output = compiled
                    .run(&HashMap::from([
                        ("lhs".to_owned(), lhs.clone()),
                        (
                            "weight".to_owned(),
                            rhs.transpose(0, 1).unwrap().contiguous().unwrap(),
                        ),
                        ("lhs_scale".to_owned(), lhs_scale.clone()),
                        ("rhs_scale".to_owned(), rhs_scale.clone()),
                    ]))
                    .unwrap()
                    .remove(0);
                // FP8 tensor cores add the products in their own order, and the output rounds once.
                let tolerance = if output == DataType::Bf16 { 0.02 } else { 1e-4 };
                for actual in [eager, compiled_output] {
                    let actual = actual.cast(DataType::F32).unwrap().to_vec::<f32>().unwrap();
                    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                        assert!(
                            (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
                            "{rows}x{depth}x{columns} {lhs_type:?} by {rhs_type:?} into {output:?}, row-wise {rowwise}: element {index} is {actual}, not {expected}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_split_scans_of_few_long_rows_match_the_cpu() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    // Few rows of many points, one of a width no chunk divides, scan across several blocks each.
    for dimensions in [[2_i64, 70000], [5, 40000]] {
        let shape = Shape::create(dimensions).unwrap();
        let count = i64::try_from(shape.element_count()).unwrap();
        let values = (0..count)
            .map(|index| (((index * 37) % 41) as f32 - 20.0) / 40000.0 + 1.0)
            .collect::<Vec<_>>();
        let input = Tensor::from_vec(Device::Cpu, &shape, values).unwrap();
        let Some(device_input) = skip_if_stub(input.to_device(Device::Cuda(0))) else {
            return;
        };
        for operation in [Operation::Cumsum, Operation::Cumprod] {
            let plan = KernelPlan::split_scan(
                &KernelInput::from_source(&device_input),
                1,
                operation,
                SplitParameters::with_threads(256),
            )
            .unwrap()
            .expect("few long rows split");
            let actual = kernel_plan::run(&plan, &[&device_input])
                .unwrap()
                .to_vec::<f32>()
                .unwrap();
            let expected = call_one(operation, OperationAttributes::Dimension(1), &[&input])
                .unwrap()
                .to_vec::<f32>()
                .unwrap();
            for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    actual == expected
                        || (actual - expected).abs() <= 1e-3 * expected.abs().max(1.0),
                    "{operation:?} of {dimensions:?}: element {index} is {actual}, not {expected}"
                );
            }
        }
    }
}

#[test]
fn test_compiled_graphs_transform_long_signals_with_cufft() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let shape = Shape::create([3, 1024]).unwrap();
    let values = (0..3 * 1024)
        .map(|index| ((index * 37) % 41) as f32 / 20.0 - 1.0)
        .collect::<Vec<_>>();
    let input = Tensor::from_vec(Device::Cpu, &shape, values).unwrap();
    let expected_spectrum = input.rfft().unwrap().to_vec::<f32>().unwrap();
    let expected_signal = input
        .rfft()
        .unwrap()
        .irfft(1024)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    let Some(device_input) = skip_if_stub(input.to_device(Device::Cuda(0))) else {
        return;
    };
    let (graph, _) = Graph::capture([("x", device_input.clone())], |inputs| {
        let spectrum = inputs[0].rfft()?;
        Ok(vec![spectrum.clone(), spectrum.irfft(1024)?])
    })
    .unwrap();
    let mut compiled = cuda::compile_graph(&graph, &GraphCudaOptions::new()).unwrap();
    for name in ["rfft", "irfft"] {
        let report = compiled
            .report()
            .operations()
            .iter()
            .find(|operation| operation.operation() == name)
            .unwrap_or_else(|| panic!("no {name} report"));
        assert_eq!(
            report
                .realized_decision()
                .and_then(GraphCompileProviderDecision::provider),
            Some(cuda::CudaGraphProvider::Cufft),
            "{name}"
        );
    }
    let outputs = compiled
        .run(&HashMap::from([("x".to_owned(), device_input)]))
        .unwrap();
    for (actual, expected) in outputs.iter().zip([&expected_spectrum, &expected_signal]) {
        let actual = actual.to_vec::<f32>().unwrap();
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-3 * expected.abs().max(1.0),
                "element {index} is {actual}, not {expected}"
            );
        }
    }
}

#[test]
fn test_paged_mla_attention_matches_the_cpu_over_ragged_sequences() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let values = |dimensions: &[i64], seed: i64| {
        let shape = Shape::create(dimensions).unwrap();
        let count = i64::try_from(shape.element_count()).unwrap();
        let data = (0..count)
            .map(|index| ((index * 37 + seed * 11) % 41) as f32 / 40.0 - 0.5)
            .collect::<Vec<_>>();
        Tensor::from_vec(Device::Cpu, shape, data).unwrap()
    };
    let indices = |values: Vec<i32>| {
        let shape = Shape::create([i64::try_from(values.len()).unwrap()]).unwrap();
        Tensor::from_vec(Device::Cpu, shape, values).unwrap()
    };
    // One decoding query over a long cache, a prefill of its whole cache, a chunk of a prefill, and a sequence with no keys; then few long sequences, whose keys split across blocks; then decoding alone, whose blocks take one group of rows.
    let batches: [&[(i32, i32)]; 3] = [
        &[(1, 150), (9, 9), (5, 70), (2, 0)],
        &[(1, 2000), (2, 1500)],
        &[(1, 300), (1, 40), (1, 0)],
    ];
    for (sequences, (heads, latent, positional, page_size)) in batches
        .into_iter()
        .flat_map(|sequences| [(16, 512, 64, 1), (8, 128, 32, 5)].map(|shape| (sequences, shape)))
    {
        let mut query_offsets = vec![0];
        let mut page_offsets = vec![0];
        let mut last_page_lengths = Vec::new();
        let mut pages_needed = 0;
        for &(queries, keys) in sequences {
            query_offsets.push(query_offsets.last().unwrap() + queries);
            let pages = (keys + page_size - 1) / page_size;
            pages_needed += pages;
            page_offsets.push(pages_needed);
            last_page_lengths.push(if keys == 0 {
                0
            } else {
                keys - (pages - 1) * page_size
            });
        }
        let page_count = pages_needed + 3;
        let pages = (0..pages_needed)
            .map(|page| (page * 7 + 2) % page_count)
            .collect::<Vec<_>>();
        let tokens = i64::from(*query_offsets.last().unwrap());
        let operands = [
            values(&[tokens, heads, latent], 1),
            values(&[tokens, heads, positional], 2),
            values(&[i64::from(page_count), i64::from(page_size), latent], 3),
            values(
                &[i64::from(page_count), i64::from(page_size), positional],
                4,
            ),
            indices(query_offsets),
            indices(page_offsets),
            indices(pages),
            indices(last_page_lengths),
        ];
        for causal in [false, true] {
            let options = AttentionOptions {
                causal,
                scale: Some(0.07),
                ..AttentionOptions::default()
            };
            for data_type in [DataType::F16, DataType::Bf16] {
                let prepared = |device: Device| {
                    operands
                        .iter()
                        .enumerate()
                        .map(|(index, operand)| {
                            let operand = if index < 4 {
                                operand.cast(data_type).unwrap()
                            } else {
                                operand.clone()
                            };
                            operand.to_device(device)
                        })
                        .collect::<Result<Vec<_>>>()
                };
                let attend = |operands: &[Tensor]| {
                    operands[0].paged_mla_attention(
                        &operands[1],
                        &operands[2],
                        &operands[3],
                        Paging {
                            query_offsets: &operands[4],
                            page_offsets: &operands[5],
                            pages: &operands[6],
                            last_page_lengths: &operands[7],
                        },
                        options,
                    )
                };
                let (expected, expected_lse) = attend(&prepared(Device::Cpu).unwrap()).unwrap();
                let Some(on_device) = skip_if_stub(prepared(Device::Cuda(0))) else {
                    return;
                };
                let (actual, actual_lse) = attend(&on_device).unwrap();
                let read = |tensor: &Tensor| {
                    tensor
                        .cast(DataType::F32)
                        .unwrap()
                        .to_device(Device::Cpu)
                        .unwrap()
                        .to_vec::<f32>()
                        .unwrap()
                };
                let tolerance = if data_type == DataType::Bf16 {
                    0.02
                } else {
                    0.003
                };
                for (index, (actual, expected)) in
                    read(&actual).iter().zip(&read(&expected)).enumerate()
                {
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "{data_type:?} causal {causal} latent {latent}: output {index} is {actual}, not {expected}"
                    );
                }
                for (index, (actual, expected)) in read(&actual_lse)
                    .iter()
                    .zip(&read(&expected_lse))
                    .enumerate()
                {
                    assert!(
                        actual == expected
                            || (actual - expected).abs() <= 1e-3 * expected.abs().max(1.0),
                        "{data_type:?} causal {causal} latent {latent}: lse {index} is {actual}, not {expected}"
                    );
                }
            }
        }
    }
}

#[test]
fn test_generated_half_contractions_on_tensor_cores_match_cuda_cores() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    // Edge tiles in every direction, a batch, and a weight read with its contraction axis contiguous, as linear layers store it.
    for (batch, rows, inner, columns, transposed) in [
        (1_i64, 128_i64, 256_i64, 128_i64, false),
        (2, 50, 100, 70, true),
        (1, 300, 64, 33, false),
    ] {
        for data_type in [DataType::F16, DataType::Bf16] {
            let sample = |dimensions: &[i64], seed: i64| {
                let shape = Shape::create(dimensions).unwrap();
                let count = i64::try_from(shape.element_count()).unwrap();
                Tensor::from_vec(
                    Device::Cpu,
                    shape,
                    (0..count)
                        .map(|index| ((index * 37 + seed * 11) % 41) as f32 / 20.0 - 1.0)
                        .collect::<Vec<_>>(),
                )
                .unwrap()
                .cast(data_type)
                .unwrap()
            };
            let lhs = sample(&[batch, rows, inner], 1);
            let rhs = if transposed {
                sample(&[batch, columns, inner], 2).transpose(1, 2).unwrap()
            } else {
                sample(&[batch, inner, columns], 2)
            };
            let lower = || kernel_lowering::matmul(data_type, lhs.layout(), rhs.layout()).unwrap();
            let tensor_cores = lower();
            assert!(
                matches!(tensor_cores.schedule, Schedule::Mma { .. }),
                "{:?}",
                tensor_cores.schedule
            );
            let mut cuda_cores = lower();
            cuda_cores.schedule = Schedule::Warps {
                threads: 128,
                vector: 1,
                lanes: 32,
                tiles: [1, 1],
            };
            let Some(device) = skip_if_stub(lhs.to_device(Device::Cuda(0))) else {
                return;
            };
            let operands = [device, rhs.to_device(Device::Cuda(0)).unwrap()];
            let metadata = operands
                .iter()
                .map(KernelInput::from_source)
                .collect::<Vec<_>>();
            let run = |lowered| {
                let plan = KernelPlanner::default()
                    .lowered(lowered, &metadata, &[Operation::MatMul])
                    .unwrap();
                kernel_plan::run(&plan, &[&operands[0], &operands[1]])
                    .unwrap()
                    .cast(DataType::F32)
                    .unwrap()
                    .to_vec::<f32>()
                    .unwrap()
            };
            let (actual, expected) = (run(tensor_cores), run(cuda_cores));
            assert_eq!(actual.len(), expected.len());
            let tolerance = if data_type == DataType::Bf16 {
                0.06
            } else {
                0.01
            };
            for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
                    "{data_type:?} {rows}x{inner}x{columns}: element {index} is {actual}, not {expected}"
                );
            }
        }
    }
}

#[test]
fn test_grouped_matmuls_match_the_cpu_with_empty_groups_and_rows_outside_them() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let sample = |dimensions: &[i64], seed: i64, data_type: DataType| {
        let shape = Shape::create(dimensions).unwrap();
        let count = i64::try_from(shape.element_count()).unwrap();
        Tensor::from_vec(
            Device::Cpu,
            shape,
            (0..count)
                .map(|index| ((index * 37 + seed * 11) % 41) as f32 / 20.0 - 1.0)
                .collect::<Vec<_>>(),
        )
        .unwrap()
        .cast(data_type)
        .unwrap()
    };
    // Rows 0 and 1 and the last row belong to no group, and the second group is empty.
    let offsets = Tensor::from_vec(
        Device::Cpu,
        Shape::create([4]).unwrap(),
        vec![2_i32, 70, 70, 140],
    )
    .unwrap();
    for data_type in [DataType::F16, DataType::Bf16, DataType::F32] {
        let input = sample(&[141, 96], 1, data_type);
        // Linear layers store each expert's weight output-major, so the product reads it transposed.
        let weights = sample(&[3, 72, 96], 2, data_type).transpose(1, 2).unwrap();
        let expected = input
            .grouped_matmul(&weights, &offsets)
            .unwrap()
            .cast(DataType::F32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let Some(device_input) = skip_if_stub(input.to_device(Device::Cuda(0))) else {
            return;
        };
        let actual = device_input
            .grouped_matmul(
                &weights.to_device(Device::Cuda(0)).unwrap(),
                &offsets.to_device(Device::Cuda(0)).unwrap(),
            )
            .unwrap()
            .cast(DataType::F32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let tolerance = match data_type {
            DataType::Bf16 => 0.08,
            DataType::F16 => 0.02,
            _ => 1e-4,
        };
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
                "{data_type:?}: element {index} is {actual}, not {expected}"
            );
        }
        assert!(expected[..2 * 72].iter().all(|value| *value == 0.0));
    }
}

#[test]
fn test_moe_matches_each_token_through_its_experts() {
    let sample = |dimensions: &[i64], seed: i64| {
        let shape = Shape::create(dimensions).unwrap();
        let count = i64::try_from(shape.element_count()).unwrap();
        Tensor::from_vec(
            Device::Cpu,
            shape,
            (0..count)
                .map(|index| ((index * 37 + seed * 11) % 41) as f32 / 40.0 - 0.5)
                .collect::<Vec<_>>(),
        )
        .unwrap()
    };
    let (tokens, hidden, experts, intermediate, chosen) = (5_i64, 16_i64, 4_i64, 8_i64, 2_i64);
    let x = sample(&[tokens, hidden], 1);
    let gate_up = sample(&[experts, 2 * intermediate, hidden], 2);
    let down = sample(&[experts, hidden, intermediate], 3);
    // Expert 2 takes no token.
    let ids = vec![0_i64, 3, 1, 0, 3, 1, 0, 1, 3, 3];
    let weights = vec![0.7_f32, 0.3, 0.6, 0.4, 0.5, 0.5, 0.9, 0.1, 0.2, 0.8];
    let expert_ids = Tensor::from_vec(
        Device::Cpu,
        Shape::create([tokens, chosen]).unwrap(),
        ids.clone(),
    )
    .unwrap();
    let expert_weights = Tensor::from_vec(
        Device::Cpu,
        Shape::create([tokens, chosen]).unwrap(),
        weights.clone(),
    )
    .unwrap();
    let mut expected = Vec::new();
    for token in 0..tokens {
        let row = x.narrow(0, token, 1).unwrap();
        let mut sum = Tensor::from_vec(
            Device::Cpu,
            Shape::create([1, hidden]).unwrap(),
            vec![0.0_f32; 16],
        )
        .unwrap();
        for slot in 0..chosen {
            let index = usize::try_from(token * chosen + slot).unwrap();
            let expert = ids[index];
            let projected = row
                .matmul(&gate_up.select(0, expert).unwrap().transpose(0, 1).unwrap())
                .unwrap();
            let activated = projected
                .narrow(1, 0, intermediate)
                .unwrap()
                .silu()
                .unwrap()
                .mul(&projected.narrow(1, intermediate, intermediate).unwrap())
                .unwrap();
            let output = activated
                .matmul(&down.select(0, expert).unwrap().transpose(0, 1).unwrap())
                .unwrap();
            sum = sum
                .add(&output.mul_scalar(f64::from(weights[index])).unwrap())
                .unwrap();
        }
        expected.extend(sum.to_vec::<f32>().unwrap());
    }
    let cpu = x
        .moe(&expert_ids, &expert_weights, &gate_up, &down)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    for (actual, expected) in cpu.iter().zip(&expected) {
        assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
    }
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let Some(device_x) = skip_if_stub(x.to_device(Device::Cuda(0))) else {
        return;
    };
    let on_device = |tensor: &Tensor| tensor.to_device(Device::Cuda(0)).unwrap();
    for data_type in [DataType::F32, DataType::Bf16] {
        let cast = |tensor: &Tensor| on_device(tensor).cast(data_type).unwrap();
        let cuda = device_x
            .cast(data_type)
            .unwrap()
            .moe(
                &on_device(&expert_ids),
                &on_device(&expert_weights),
                &cast(&gate_up),
                &cast(&down),
            )
            .unwrap()
            .cast(DataType::F32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let tolerance = if data_type == DataType::Bf16 {
            0.03
        } else {
            1e-4
        };
        for (actual, expected) in cuda.iter().zip(&expected) {
            assert!(
                (actual - expected).abs() <= tolerance,
                "{data_type:?}: {actual} != {expected}"
            );
        }
    }
}

#[test]
fn test_kernels_compile_for_their_requirements_or_say_why_not() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let Some(input) = skip_if_stub(Tensor::from_vec(
        Device::Cuda(0),
        Shape::create([4]).unwrap(),
        vec![1.0_f32, -2.0, 3.0, -4.0],
    )) else {
        return;
    };
    let plan = || {
        KernelPlanner::default()
            .create(
                Operation::Neg,
                &OperationAttributes::None,
                &[KernelInput::from_source(&input)],
            )
            .unwrap()
    };
    // A capability no device has names the kernel and what it needs.
    let mut impossible = plan();
    impossible.kernel.requirements.minimum_capability = Some((99, 0));
    let error = kernel_plan::run(&impossible, &[&input])
        .unwrap_err()
        .to_string();
    assert!(error.contains("needs compute capability 99.0"), "{error}");
    // Architecture-specific code compiles for the `a` variant of the device's target, where there is one.
    let mut specific = plan();
    specific.kernel.requirements.architecture_specific = true;
    match kernel_plan::run(&specific, &[&input]) {
        Ok(output) => assert_eq!(output.to_vec::<f32>().unwrap(), vec![-1.0, 2.0, -3.0, 4.0]),
        Err(error) => assert!(
            error
                .to_string()
                .contains("no architecture-specific target"),
            "{error}"
        ),
    }
}

#[test]
fn test_block_scaled_matmuls_decode_mxfp8_and_nvfp4() {
    let tensor = |dimensions: &[i64], values: Vec<f32>| {
        Tensor::from_vec(Device::Cpu, Shape::create(dimensions).unwrap(), values).unwrap()
    };
    let bytes = |dimensions: &[i64], values: Vec<u8>| {
        Tensor::from_vec(Device::Cpu, Shape::create(dimensions).unwrap(), values).unwrap()
    };
    // The F32 product of decoded rows `[M, K]` and `[N, K]`, row by column.
    let product = |lhs: &[f32], rhs: &[f32], inner: usize| {
        let (rows, columns) = (lhs.len() / inner, rhs.len() / inner);
        (0..rows * columns)
            .map(|index| {
                let (row, column) = (index / columns, index % columns);
                (0..inner)
                    .map(|k| lhs[row * inner + k] * rhs[column * inner + k])
                    .sum::<f32>()
            })
            .collect::<Vec<_>>()
    };
    // MXFP8: two blocks of 32 a row, scaled by 2^-1 and 2^1 for the rows, 2^0 and 2^2 for the columns.
    let small = |count: usize, seed: usize| {
        (0..count)
            .map(|index| ((index * 7 + seed) % 9) as f32 / 2.0 - 2.0)
            .collect::<Vec<_>>()
    };
    let (lhs_values, rhs_values) = (small(2 * 64, 1), small(3 * 64, 4));
    let lhs = tensor(&[2, 64], lhs_values.clone())
        .cast(DataType::F8E4M3)
        .unwrap();
    let rhs = tensor(&[3, 64], rhs_values.clone())
        .cast(DataType::F8E4M3)
        .unwrap();
    let lhs_exponents = bytes(&[2, 2], vec![126, 128, 126, 128]);
    let rhs_exponents = bytes(&[3, 2], vec![127, 129, 127, 129, 127, 129]);
    let scaled = |values: &[f32], factors: [f32; 2]| {
        values
            .iter()
            .enumerate()
            .map(|(index, value)| value * factors[index % 64 / 32])
            .collect::<Vec<_>>()
    };
    let mxfp8 = product(
        &scaled(&lhs_values, [0.5, 2.0]),
        &scaled(&rhs_values, [1.0, 4.0]),
        64,
    );
    // NVFP4: nibbles 2 (1.0), 5 (3.0), 9 (-0.5), and 15 (-6.0), two blocks of 16 a row, scaled by 0.5 and 2.0.
    let nibbles = [2_u8, 5, 9, 15];
    let decoded = [1.0_f32, 3.0, -0.5, -6.0];
    let packed = |rows: usize, seed: usize| {
        let codes = (0..rows * 32)
            .map(|index| (index * 3 + seed) % 4)
            .collect::<Vec<_>>();
        let bytes = codes
            .chunks(2)
            .map(|pair| nibbles[pair[0]] | (nibbles[pair[1]] << 4))
            .collect::<Vec<_>>();
        let values = codes
            .iter()
            .enumerate()
            .map(|(index, &code)| decoded[code] * [0.5, 2.0][index % 32 / 16])
            .collect::<Vec<_>>();
        (bytes, values)
    };
    let ((lhs_bytes, lhs_decoded), (rhs_bytes, rhs_decoded)) = (packed(2, 0), packed(3, 1));
    let nvfp4 = product(&lhs_decoded, &rhs_decoded, 32);
    let fp4_scales = |rows: usize| {
        tensor(
            &[i64::try_from(rows).unwrap(), 2],
            [0.5_f32, 2.0].repeat(rows),
        )
        .cast(DataType::F8E4M3)
        .unwrap()
    };
    let cases = [
        (
            BlockScaledFormat::Mxfp8,
            [lhs, rhs, lhs_exponents, rhs_exponents],
            mxfp8,
        ),
        (
            BlockScaledFormat::Nvfp4,
            [
                bytes(&[2, 16], lhs_bytes),
                bytes(&[3, 16], rhs_bytes),
                fp4_scales(2),
                fp4_scales(3),
            ],
            nvfp4,
        ),
    ];
    let lock = maybe_lock_device();
    for (format, operands, expected) in &cases {
        let multiply = |operands: &[Tensor]| {
            operands[0]
                .block_scaled_matmul(
                    &operands[1],
                    &operands[2],
                    &operands[3],
                    *format,
                    0.5,
                    DataType::F32,
                )
                .unwrap()
                .to_device(Device::Cpu)
                .unwrap()
                .to_vec::<f32>()
                .unwrap()
        };
        let check = |actual: Vec<f32>| {
            for (actual, expected) in actual.iter().zip(expected) {
                assert!(
                    (actual - expected * 0.5).abs() <= 1e-4 * expected.abs().max(1.0),
                    "{format:?}: {actual} != {}",
                    expected * 0.5
                );
            }
        };
        check(multiply(operands));
        if lock.is_some()
            && let Some(on_device) = skip_if_stub(
                operands
                    .iter()
                    .map(|operand| operand.to_device(Device::Cuda(0)))
                    .collect::<Result<Vec<_>>>(),
            )
        {
            check(multiply(&on_device));
        }
    }
}

#[test]
fn test_compiled_graphs_run_both_results_of_paged_mla_attention_as_eager_calls_do() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let values = |dimensions: &[i64], seed: i64| {
        let shape = Shape::create(dimensions).unwrap();
        let count = i64::try_from(shape.element_count()).unwrap();
        Tensor::from_vec(
            Device::Cpu,
            shape,
            (0..count)
                .map(|index| ((index * 37 + seed * 11) % 41) as f32 / 40.0 - 0.5)
                .collect::<Vec<_>>(),
        )
        .unwrap()
    };
    let indices = |values: Vec<i32>| {
        let shape = Shape::create([i64::try_from(values.len()).unwrap()]).unwrap();
        Tensor::from_vec(Device::Cpu, shape, values).unwrap()
    };
    // A decoding query over 300 keys, whose keys split across blocks, and a prefill chunk of 3 queries over 40.
    let operands = [
        values(&[4, 16, 128], 1).cast(DataType::Bf16).unwrap(),
        values(&[4, 16, 32], 2).cast(DataType::Bf16).unwrap(),
        values(&[340, 1, 128], 3).cast(DataType::Bf16).unwrap(),
        values(&[340, 1, 32], 4).cast(DataType::Bf16).unwrap(),
        indices(vec![0, 1, 4]),
        indices(vec![0, 300, 340]),
        indices((0..340).rev().collect()),
        indices(vec![1, 1]),
    ];
    let Some(on_device) = skip_if_stub(
        operands
            .iter()
            .map(|operand| operand.to_device(device))
            .collect::<Result<Vec<_>>>(),
    ) else {
        return;
    };
    let options = AttentionOptions {
        causal: true,
        scale: Some(0.08),
        ..AttentionOptions::default()
    };
    let names = [
        "latent_queries",
        "positional_queries",
        "latent_cache",
        "positional_cache",
        "query_offsets",
        "page_offsets",
        "pages",
        "last_page_lengths",
    ];
    let (graph, eager) = Graph::capture(
        names.iter().copied().zip(on_device.iter().cloned()),
        |inputs| {
            let (output, lse) = inputs[0].paged_mla_attention(
                &inputs[1],
                &inputs[2],
                &inputs[3],
                Paging {
                    query_offsets: &inputs[4],
                    page_offsets: &inputs[5],
                    pages: &inputs[6],
                    last_page_lengths: &inputs[7],
                },
                options,
            )?;
            Ok(vec![output, lse])
        },
    )
    .unwrap();
    let mut compiled = cuda::compile_graph(&graph, &GraphCudaOptions::new()).unwrap();
    assert_eq!(
        compiled.stats().kernel_child_graph_nodes(),
        1,
        "both results come from one kernel"
    );
    let outputs = compiled
        .run(
            &names
                .iter()
                .map(|name| (*name).to_owned())
                .zip(on_device.iter().cloned())
                .collect::<HashMap<_, _>>(),
        )
        .unwrap();
    for (actual, expected) in outputs.iter().zip(&eager) {
        let read = |tensor: &Tensor| tensor.cast(DataType::F32).unwrap().to_vec::<f32>().unwrap();
        assert_eq!(read(actual), read(expected));
    }
}

#[test]
fn test_selection_topk_matches_the_sorting_cpu_with_ties_and_nan() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    // Router scores over 256 experts, with many ties and one NaN a row.
    let mut values = (0..3 * 256)
        .map(|index| ((index * 37) % 23) as f32 / 4.0)
        .collect::<Vec<_>>();
    for row in 0..3 {
        values[row * 256 + 17 * (row + 1)] = f32::NAN;
    }
    let input = Tensor::from_vec(Device::Cpu, Shape::create([3, 256]).unwrap(), values).unwrap();
    let Some(device_input) = skip_if_stub(input.to_device(Device::Cuda(0))) else {
        return;
    };
    for (k, dim, largest) in [(8, 1, true), (5, 1, false), (2, 0, true), (32, -1, true)] {
        let (expected_values, expected_indices) = input.topk(k, dim, largest, true).unwrap();
        let (values, indices) = device_input.topk(k, dim, largest, true).unwrap();
        assert_eq!(
            indices.to_vec::<i64>().unwrap(),
            expected_indices.to_vec::<i64>().unwrap(),
            "k {k} dim {dim} largest {largest}"
        );
        let bits = |tensor: &Tensor| {
            tensor
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&values), bits(&expected_values));
    }
    // A compiled graph that reads both results selects once.
    let (graph, _) = Graph::capture([("input", device_input.clone())], |inputs| {
        let (values, indices) = inputs[0].topk(8, 1, true, true)?;
        Ok(vec![values, indices])
    })
    .unwrap();
    let mut compiled = cuda::compile_graph(&graph, &GraphCudaOptions::new()).unwrap();
    assert_eq!(compiled.stats().kernel_child_graph_nodes(), 1);
    let outputs = compiled
        .run(&HashMap::from([("input".to_owned(), device_input)]))
        .unwrap();
    let (expected_values, expected_indices) = input.topk(8, 1, true, true).unwrap();
    assert_eq!(
        outputs[1].to_vec::<i64>().unwrap(),
        expected_indices.to_vec::<i64>().unwrap()
    );
    let bits = |tensor: &Tensor| {
        tensor
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(&outputs[0]), bits(&expected_values));
}

// FlashInfer-Bench 020 end to end: DeepSeek's router picks the experts, block-scaled FP8 storage holds the weights, and the layer runs them, as the CPU runs each piece.
#[test]
fn test_deepseek_moe_composes_routing_block_storage_and_experts() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let sample = |dimensions: &[i64], seed: i64| {
        let shape = Shape::create(dimensions).unwrap();
        let count = i64::try_from(shape.element_count()).unwrap();
        Tensor::from_vec(
            Device::Cpu,
            shape,
            (0..count)
                .map(|index| ((index * 37 + seed * 11) % 41) as f32 / 40.0 - 0.5)
                .collect::<Vec<_>>(),
        )
        .unwrap()
    };
    let (tokens, hidden, experts, intermediate) = (6_i64, 16_i64, 8_i64, 8_i64);
    let operands = [
        sample(&[tokens, hidden], 1),
        sample(&[tokens, experts], 2),
        sample(&[experts], 3),
        sample(&[experts, 2 * intermediate, hidden], 4),
        sample(&[experts, 2, 2], 5).add_scalar(1.0).unwrap(),
        sample(&[experts, hidden, intermediate], 6),
        sample(&[experts, 2, 1], 7).add_scalar(1.0).unwrap(),
    ];
    let run = |operands: &[Tensor]| {
        let [x, logits, bias, gate_up, gate_up_scales, down, down_scales] = operands else {
            unreachable!();
        };
        let (ids, weights) = logits.grouped_topk_routing(bias, 4, 2, 3, 2.5).unwrap();
        let gate_up = gate_up
            .cast(DataType::F8E4M3)
            .unwrap()
            .dequantize_blocks(gate_up_scales, (8, 8))
            .unwrap();
        let down = down
            .cast(DataType::F8E4M3)
            .unwrap()
            .dequantize_blocks(down_scales, (8, 8))
            .unwrap();
        let output = x.moe(&ids, &weights, &gate_up, &down).unwrap();
        (
            ids.to_vec::<i64>().unwrap(),
            output.to_vec::<f32>().unwrap(),
        )
    };
    let (expected_ids, expected) = run(&operands);
    let Some(on_device) = skip_if_stub(
        operands
            .iter()
            .map(|operand| operand.to_device(Device::Cuda(0)))
            .collect::<Result<Vec<_>>>(),
    ) else {
        return;
    };
    let (ids, output) = run(&on_device);
    assert_eq!(ids, expected_ids);
    for (actual, expected) in output.iter().zip(&expected) {
        assert!((actual - expected).abs() <= 1e-3, "{actual} != {expected}");
    }
}

// Compiled graphs keep a split scan's scratch across replays, so each replay must wait for its own chunk totals.
#[test]
fn test_split_scans_replay_in_compiled_graphs() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let shape = Shape::create([2, 70000]).unwrap();
    let values = |scale: f32| {
        (0..140_000_u32)
            .map(|index| (index % 41) as f32 * scale / 40000.0)
            .collect::<Vec<_>>()
    };
    let input = Tensor::from_vec(Device::Cpu, &shape, values(1.0)).unwrap();
    let Some(device_input) = skip_if_stub(input.to_device(Device::Cuda(0))) else {
        return;
    };
    let (graph, _) = Graph::capture([("x", device_input)], |inputs| {
        Ok(vec![inputs[0].cumsum(1)?])
    })
    .unwrap();
    let mut compiled = cuda::compile_graph(&graph, &GraphCudaOptions::new()).unwrap();
    for scale in [1.0_f32, 2.0, 0.5] {
        let cpu = Tensor::from_vec(Device::Cpu, &shape, values(scale)).unwrap();
        let actual = compiled
            .run(&HashMap::from([(
                "x".to_owned(),
                cpu.to_device(Device::Cuda(0)).unwrap(),
            )]))
            .unwrap()
            .remove(0)
            .to_vec::<f32>()
            .unwrap();
        let expected = cpu.cumsum(1).unwrap().to_vec::<f32>().unwrap();
        for (actual, expected) in actual.iter().zip(&expected) {
            assert!(
                (actual - expected).abs() <= 1e-3 * expected.abs().max(1.0),
                "{actual} != {expected}"
            );
        }
    }
}

// The HF spelling of RMS norm is recognized, and takes two kernels, since a row kernel keeps one data type and its casts run apart.
#[test]
fn test_hf_rms_norm_spelling_takes_two_kernels() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let input = Tensor::from_vec(
        Device::Cpu,
        Shape::create([4, 256]).unwrap(),
        (0..1024)
            .map(|index| index as f32 / 512.0 - 1.0)
            .collect::<Vec<_>>(),
    )
    .unwrap()
    .cast(DataType::F16)
    .unwrap();
    let Some(input) = skip_if_stub(input.to_device(Device::Cuda(0))) else {
        return;
    };
    let (graph, _) = Graph::capture([("x", input)], |inputs| {
        let wide = inputs[0].cast(DataType::F32)?;
        let normed = wide.mul(
            &wide
                .mul(&wide)?
                .mean(&[-1], true)?
                .add_scalar(1e-6)?
                .rsqrt()?,
        )?;
        Ok(vec![normed.cast(DataType::F16)?])
    })
    .unwrap();
    let compiled = cuda::compile_graph(&graph, &GraphCudaOptions::new()).unwrap();
    assert_eq!(compiled.stats().kernel_child_graph_nodes(), 2);
}

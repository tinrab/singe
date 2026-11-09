use std::{collections::HashMap, time::Duration};

use crate::{
    backend::{
        cuda::{
            self,
            codegen::PlanningContext,
            compiled::{
                executable::CudaCompiledGraph,
                plan::{
                    fusion::{FusionPlan, GroupKind},
                    group::module_parameters,
                    lowering::{LoweringPolicy, ModuleKernel},
                },
            },
            export::CudaExport,
            kernel_plan::{KernelParameters, KernelPlanner},
            runtime::runtime_for_device,
            tuning::{
                GraphTuneOptions, Measurement, TunedChoice, TunedRecord, TunedWay, TuningRecords,
                group_key, tune_graph,
            },
        },
        registry::{LayerNormalizationAttributes, Operation, OperationAttributes},
    },
    data_type::DataType,
    device::Device,
    dispatch::call_one,
    error::Result,
    graph::{Graph, GraphCudaOptions},
    index::TensorIndex,
    shape::Shape,
    tensor::Tensor,
    testing::{CompiledProgram, maybe_lock_device, program_target, skip_if_stub},
};

const DEVICE: Device = Device::Cuda(0);
const MODULE: &str = "module: rotary cache write";

fn tensor(dimensions: &[i64], data_type: DataType, seed: u32) -> Tensor {
    let count = usize::try_from(dimensions.iter().product::<i64>()).unwrap();
    let mut state = seed;
    let values = (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            f32::from(u16::try_from(state >> 16).unwrap()) / 32_768.0 - 1.0
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(DEVICE, Shape::create(dimensions.to_vec()).unwrap(), values)
        .unwrap()
        .cast(data_type)
        .unwrap()
}

fn integers(values: Vec<i64>) -> Tensor {
    Tensor::from_vec(
        DEVICE,
        Shape::create([i64::try_from(values.len()).unwrap()]).unwrap(),
        values,
    )
    .unwrap()
}

fn rotate(input: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = input.shape().dimensions()[2] / 2;
    let first = input.narrow(-1, 0, half)?;
    let second = input.narrow(-1, half, half)?;
    Tensor::concat(
        &[
            first.mul(cos)?.sub(&second.mul(sin)?)?,
            second.mul(cos)?.add(&first.mul(sin)?)?,
        ],
        -1,
    )
}

#[derive(Debug, Clone, Copy)]
struct StepShape {
    tokens: i64,
    query_heads: i64,
    cache_heads: i64,
    head_dim: i64,
    pages: i64,
    slots: i64,
}

// A decode step's tensors, whose tokens write distinct slots, some through negative indices.
fn step_inputs(shape: StepShape, data_type: DataType) -> Vec<(&'static str, Tensor)> {
    let StepShape {
        tokens,
        query_heads,
        cache_heads,
        head_dim,
        pages,
        slots,
    } = shape;
    let half = head_dim / 2;
    let slot = |token: i64| {
        let index = token * 3 + 1;
        (index / slots % pages, index % slots)
    };
    let page_values = (0..tokens)
        .map(|token| {
            let page = slot(token).0;
            if token % 2 == 1 { page - pages } else { page }
        })
        .collect();
    let slot_values = (0..tokens)
        .map(|token| {
            let at = slot(token).1;
            if token % 3 == 2 { at - slots } else { at }
        })
        .collect();
    vec![
        (
            "query",
            tensor(&[tokens, query_heads, head_dim], data_type, 1),
        ),
        (
            "key",
            tensor(&[tokens, cache_heads, head_dim], data_type, 2),
        ),
        (
            "value",
            tensor(&[tokens, cache_heads, head_dim], data_type, 3),
        ),
        ("cos", tensor(&[tokens, 1, half], data_type, 4)),
        ("sin", tensor(&[tokens, 1, half], data_type, 5)),
        ("pages", integers(page_values)),
        ("slots", integers(slot_values)),
        (
            "keys",
            tensor(&[pages, slots, cache_heads, head_dim], data_type, 6),
        ),
        (
            "values",
            tensor(&[pages, slots, cache_heads, head_dim], data_type, 7),
        ),
    ]
}

fn slots_of(inputs: &[Tensor]) -> [TensorIndex; 2] {
    [
        TensorIndex::Indices(inputs[5].clone()),
        TensorIndex::Indices(inputs[6].clone()),
    ]
}

fn step(inputs: &[Tensor]) -> Result<Vec<Tensor>> {
    let query = rotate(&inputs[0], &inputs[3], &inputs[4])?;
    let key = rotate(&inputs[1], &inputs[3], &inputs[4])?;
    Ok(vec![
        query,
        inputs[7].index_put(&slots_of(inputs), &key, false)?,
        inputs[8].index_put(&slots_of(inputs), &inputs[2], false)?,
    ])
}

fn compile(
    graph: &Graph,
    policy: LoweringPolicy,
    records: Option<TuningRecords>,
) -> CudaCompiledGraph {
    let mut options = GraphCudaOptions::new().with_lowering_policy(policy);
    if let Some(records) = records {
        options = options.with_tuning_records(records);
    }
    cuda::compile_graph(graph, &options).unwrap()
}

fn named(inputs: &[(&'static str, Tensor)]) -> HashMap<String, Tensor> {
    inputs
        .iter()
        .map(|(name, tensor)| ((*name).to_owned(), tensor.clone()))
        .collect()
}

fn values(tensors: &[Tensor]) -> Vec<Vec<f32>> {
    tensors
        .iter()
        .map(|tensor| tensor.cast(DataType::F32).unwrap().to_vec::<f32>().unwrap())
        .collect()
}

fn assert_all_close(actual: &[Vec<f32>], expected: &[Vec<f32>], tolerance: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}");
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.len(), expected.len(), "{what}");
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "{what}[{index}]: {actual} != {expected}"
            );
        }
    }
}

fn tolerance(data_type: DataType) -> f32 {
    match data_type {
        DataType::F16 => 2e-2,
        DataType::Bf16 => 8e-2,
        _ => 1e-5,
    }
}

fn plan_of(compiled: &CudaCompiledGraph) -> FusionPlan {
    compiled.report().fusion_plan().clone()
}

fn ready() -> Option<impl Sized> {
    let lock = maybe_lock_device()?;
    skip_if_stub(runtime_for_device(DEVICE))?;
    Some(lock)
}

#[test]
fn test_modules_run_correctly_across_types_heads_and_paging() {
    let Some(_lock) = ready() else {
        return;
    };
    for data_type in [DataType::F16, DataType::Bf16, DataType::F32] {
        for shape in [
            StepShape {
                tokens: 1,
                query_heads: 2,
                cache_heads: 1,
                head_dim: 8,
                pages: 2,
                slots: 4,
            },
            StepShape {
                tokens: 5,
                query_heads: 8,
                cache_heads: 2,
                head_dim: 16,
                pages: 3,
                slots: 8,
            },
            StepShape {
                tokens: 7,
                query_heads: 4,
                cache_heads: 4,
                head_dim: 128,
                pages: 4,
                slots: 8,
            },
        ] {
            let inputs = step_inputs(shape, data_type);
            let (graph, eager) = Graph::capture(inputs.clone(), step).unwrap();
            let mut compiled = compile(&graph, LoweringPolicy::generated_kernels(), None);
            let plan = plan_of(&compiled);
            assert!(plan.to_string().contains(MODULE), "{shape:?}\n{plan}");
            let outputs = compiled.run(&named(&inputs)).unwrap();
            assert_all_close(
                &values(&outputs),
                &values(&eager),
                tolerance(data_type),
                &format!("{data_type:?} {shape:?}"),
            );
        }
    }
}

#[test]
fn test_modules_write_a_cache_in_place() {
    let Some(_lock) = ready() else {
        return;
    };
    let shape = StepShape {
        tokens: 4,
        query_heads: 4,
        cache_heads: 2,
        head_dim: 16,
        pages: 3,
        slots: 8,
    };
    let inputs = step_inputs(shape, DataType::F32);
    let function = |inputs: &[Tensor]| {
        let query = rotate(&inputs[0], &inputs[3], &inputs[4])?;
        let key = rotate(&inputs[1], &inputs[3], &inputs[4])?;
        inputs[7].index_put_(&slots_of(inputs), &key)?;
        inputs[8].index_put_(&slots_of(inputs), &inputs[2])?;
        Ok(vec![query.add(&query)?])
    };
    let copies = step_inputs(shape, DataType::F32);
    let (graph, eager) = Graph::capture(copies.clone(), function).unwrap();
    let options = GraphCudaOptions::new()
        .with_lowering_policy(LoweringPolicy::generated_kernels())
        .with_inputs_in_place(named(&inputs));
    let mut compiled = cuda::compile_graph(&graph, &options).unwrap();
    let plan = plan_of(&compiled);
    assert!(plan.to_string().contains(MODULE), "{plan}");
    let outputs = compiled.run(&named(&inputs)).unwrap();
    assert_all_close(&values(&outputs), &values(&eager), 1e-5, "output");
    let caches = |inputs: &[(&str, Tensor)]| values(&[inputs[7].1.clone(), inputs[8].1.clone()]);
    assert_all_close(&caches(&inputs), &caches(&copies), 1e-5, "caches");
}

// A rotation's operands may be views that start inside their storage, as a projection split into queries, keys, and values is.
#[test]
fn test_modules_read_operands_that_start_inside_their_storage() {
    let Some(_lock) = ready() else {
        return;
    };
    let shape = StepShape {
        tokens: 5,
        query_heads: 4,
        cache_heads: 2,
        head_dim: 16,
        pages: 3,
        slots: 8,
    };
    let mut inputs = step_inputs(shape, DataType::F32);
    // Each rotated or written operand gains heads before those the step reads, and the angles gain rows before theirs.
    inputs[0].1 = tensor(&[5, 6, 16], DataType::F32, 11);
    inputs[1].1 = tensor(&[5, 3, 16], DataType::F32, 12);
    inputs[2].1 = tensor(&[5, 3, 16], DataType::F32, 13);
    inputs[3].1 = tensor(&[7, 1, 8], DataType::F32, 14);
    inputs[4].1 = tensor(&[7, 1, 8], DataType::F32, 15);
    let function = |inputs: &[Tensor]| {
        let cos = inputs[3].narrow(0, 2, 5)?;
        let sin = inputs[4].narrow(0, 2, 5)?;
        let query = rotate(&inputs[0].narrow(1, 2, 4)?, &cos, &sin)?;
        let key = rotate(&inputs[1].narrow(1, 1, 2)?, &cos, &sin)?;
        Ok(vec![
            query,
            inputs[7].index_put(&slots_of(inputs), &key, false)?,
            inputs[8].index_put(&slots_of(inputs), &inputs[2].narrow(1, 1, 2)?, false)?,
        ])
    };
    let (graph, eager) = Graph::capture(inputs.clone(), function).unwrap();
    let mut compiled = compile(&graph, LoweringPolicy::generated_kernels(), None);
    let plan = plan_of(&compiled);
    assert!(plan.to_string().contains(MODULE), "{plan}");
    let outputs = compiled.run(&named(&inputs)).unwrap();
    assert_all_close(&values(&outputs), &values(&eager), 1e-5, "views");
}

#[test]
fn test_matchers_decline_what_the_kernel_cannot_take_and_keep_the_generic_route() {
    type Case = fn(&[Tensor]) -> Result<Vec<Tensor>>;

    let Some(_lock) = ready() else {
        return;
    };
    let shape = StepShape {
        tokens: 3,
        query_heads: 2,
        cache_heads: 2,
        head_dim: 8,
        pages: 2,
        slots: 4,
    };
    let inputs = step_inputs(shape, DataType::F32);
    let cases: [(&str, Case); 4] = [
        ("no query", |inputs| {
            let key = rotate(&inputs[1], &inputs[3], &inputs[4])?;
            Ok(vec![
                inputs[7].index_put(&slots_of(inputs), &key, false)?,
                inputs[8].index_put(&slots_of(inputs), &inputs[2], false)?,
            ])
        }),
        ("rotated keys read elsewhere", |inputs| {
            let query = rotate(&inputs[0], &inputs[3], &inputs[4])?;
            let key = rotate(&inputs[1], &inputs[3], &inputs[4])?;
            Ok(vec![
                query,
                key.clone(),
                inputs[7].index_put(&slots_of(inputs), &key, false)?,
                inputs[8].index_put(&slots_of(inputs), &inputs[2], false)?,
            ])
        }),
        ("rotation of a part of a head", |inputs| {
            let query = rotate(&inputs[0], &inputs[3], &inputs[4])?;
            let half = inputs[3].shape().dimensions()[2];
            let first = inputs[1].narrow(-1, 0, half)?;
            let second = inputs[1].narrow(-1, half, half)?;
            let key = Tensor::concat(
                &[
                    first.mul(&inputs[3])?.sub(&second.mul(&inputs[4])?)?,
                    second.mul(&inputs[3])?.sub(&first.mul(&inputs[4])?)?,
                ],
                -1,
            )?;
            Ok(vec![
                query,
                inputs[7].index_put(&slots_of(inputs), &key, false)?,
                inputs[8].index_put(&slots_of(inputs), &inputs[2], false)?,
            ])
        }),
        ("no value write", |inputs| {
            let query = rotate(&inputs[0], &inputs[3], &inputs[4])?;
            let key = rotate(&inputs[1], &inputs[3], &inputs[4])?;
            Ok(vec![
                query,
                inputs[7].index_put(&slots_of(inputs), &key, false)?,
            ])
        }),
    ];
    for (what, function) in cases {
        let (graph, eager) = Graph::capture(inputs.clone(), function).unwrap();
        let mut compiled = compile(&graph, LoweringPolicy::generated_kernels(), None);
        let plan = plan_of(&compiled);
        assert!(!plan.to_string().contains(MODULE), "{what}\n{plan}");
        // A case that leaves an input unread takes it out of the graph.
        let mut operands = named(&inputs);
        let read = compiled
            .ir()
            .input_values()
            .map(|(_, name)| name.to_owned())
            .collect::<Vec<_>>();
        operands.retain(|name, _| read.contains(name));
        let outputs = compiled.run(&operands).unwrap();
        assert_all_close(&values(&outputs), &values(&eager), 1e-5, what);
    }
}

#[test]
fn test_a_policy_and_a_record_keep_a_module_off() {
    let Some(_lock) = ready() else {
        return;
    };
    let shape = StepShape {
        tokens: 3,
        query_heads: 2,
        cache_heads: 2,
        head_dim: 8,
        pages: 2,
        slots: 4,
    };
    let inputs = step_inputs(shape, DataType::F32);
    let (graph, eager) = Graph::capture(inputs.clone(), step).unwrap();
    let on = compile(&graph, LoweringPolicy::generated_kernels(), None);
    let plan = plan_of(&on);
    let group = plan
        .groups()
        .iter()
        .find(|group| matches!(group.kind, GroupKind::Module(_)))
        .unwrap();
    let key = group_key(on.ir(), &group.members).unwrap();
    let mut records = TuningRecords::new(None);
    records.insert(
        key,
        TunedRecord {
            choice: TunedChoice::Way(TunedWay::Separate),
            default_ns: None,
            best_ns: 0,
            bytes: 0,
        },
    );
    for (what, mut compiled) in [
        (
            "policy",
            compile(
                &graph,
                LoweringPolicy::generated_kernels().with_modules([]),
                None,
            ),
        ),
        (
            "record",
            compile(&graph, LoweringPolicy::generated_kernels(), Some(records)),
        ),
    ] {
        let plan = plan_of(&compiled);
        assert!(!plan.to_string().contains(MODULE), "{what}\n{plan}");
        let outputs = compiled.run(&named(&inputs)).unwrap();
        assert_all_close(&values(&outputs), &values(&eager), 1e-5, what);
    }
    let named_only = compile(
        &graph,
        LoweringPolicy::generated_kernels().with_modules([ModuleKernel::RotaryCacheWrite]),
        None,
    );
    assert!(plan_of(&named_only).to_string().contains(MODULE));
}

#[test]
fn test_modules_take_query_and_key_norms() {
    let Some(_lock) = ready() else {
        return;
    };
    let shape = StepShape {
        tokens: 5,
        query_heads: 3,
        cache_heads: 2,
        head_dim: 32,
        pages: 3,
        slots: 8,
    };
    for data_type in [DataType::F16, DataType::Bf16, DataType::F32] {
        let mut inputs = step_inputs(shape, data_type);
        inputs.push(("query_weight", tensor(&[32], data_type, 8)));
        let attributes = |has_weight| {
            OperationAttributes::LayerNormalization(LayerNormalizationAttributes {
                normalized_shape: vec![32],
                has_weight,
                has_bias: false,
                eps: 1e-5,
            })
        };
        let function = |inputs: &[Tensor]| {
            let query = call_one(
                Operation::RmsNorm,
                attributes(true),
                &[&inputs[0], &inputs[9]],
            )?;
            let key = call_one(Operation::RmsNorm, attributes(false), &[&inputs[1]])?;
            let query = rotate(&query, &inputs[3], &inputs[4])?;
            let key = rotate(&key, &inputs[3], &inputs[4])?;
            Ok(vec![
                query,
                inputs[7].index_put(&slots_of(inputs), &key, false)?,
                inputs[8].index_put(&slots_of(inputs), &inputs[2], false)?,
            ])
        };
        let (graph, eager) = Graph::capture(inputs.clone(), function).unwrap();
        let mut compiled = compile(&graph, LoweringPolicy::generated_kernels(), None);
        let plan = plan_of(&compiled);
        assert!(plan.to_string().contains(MODULE), "{plan}");
        let outputs = compiled.run(&named(&inputs)).unwrap();
        assert_all_close(
            &values(&outputs),
            &values(&eager),
            tolerance(data_type),
            &format!("{data_type:?}"),
        );
    }
}

#[test]
fn test_every_module_parameter_computes_what_the_default_does() {
    let Some(_lock) = ready() else {
        return;
    };
    let shape = StepShape {
        tokens: 5,
        query_heads: 4,
        cache_heads: 2,
        head_dim: 64,
        pages: 3,
        slots: 8,
    };
    let inputs = step_inputs(shape, DataType::Bf16);
    let (graph, eager) = Graph::capture(inputs.clone(), step).unwrap();
    let default = compile(&graph, LoweringPolicy::generated_kernels(), None);
    let plan = plan_of(&default);
    let group = plan
        .groups()
        .iter()
        .find(|group| matches!(group.kind, GroupKind::Module(_)))
        .unwrap();
    let GroupKind::Module(kind) = &group.kind else {
        unreachable!();
    };
    let target = runtime_for_device(DEVICE).unwrap().architecture();
    let planner = KernelPlanner::new(PlanningContext {
        target: Some(target),
        ..PlanningContext::default()
    });
    let (key, candidates) = module_parameters(&planner, default.ir(), kind).unwrap();
    assert!(candidates.len() > 3);
    for parameters in candidates {
        let KernelParameters::RotaryCache(_) = parameters else {
            panic!("{parameters:?}");
        };
        let mut records = TuningRecords::new(None);
        records.insert(
            key.clone(),
            TunedRecord {
                choice: TunedChoice::Parameters(parameters),
                default_ns: None,
                best_ns: 0,
                bytes: 0,
            },
        );
        let mut compiled = compile(&graph, LoweringPolicy::generated_kernels(), Some(records));
        let outputs = compiled.run(&named(&inputs)).unwrap();
        assert_all_close(
            &values(&outputs),
            &values(&eager),
            tolerance(DataType::Bf16),
            &format!("{parameters:?}"),
        );
    }
}

#[test]
fn test_a_tuned_module_compiles_and_exports_from_its_records() {
    let Some(_lock) = ready() else {
        return;
    };
    let Some(architecture) = program_target(DEVICE) else {
        return;
    };
    let shape = StepShape {
        tokens: 5,
        query_heads: 4,
        cache_heads: 2,
        head_dim: 16,
        pages: 3,
        slots: 8,
    };
    let inputs = step_inputs(shape, DataType::F32);
    let (graph, eager) = Graph::capture(inputs.clone(), step).unwrap();
    let records = tune_graph(
        &graph,
        &named(&inputs),
        DEVICE,
        &GraphTuneOptions {
            policy: LoweringPolicy::generated_kernels(),
            choice_budget: Duration::from_millis(500),
            runs: 2,
            schedules: false,
            plan_budget: Duration::ZERO,
            plan_speed: false,
            ..GraphTuneOptions::default()
        },
    )
    .unwrap();
    let mut compiled = compile(
        &graph,
        LoweringPolicy::generated_kernels(),
        Some(records.clone()),
    );
    let outputs = compiled.run(&named(&inputs)).unwrap();
    assert_all_close(&values(&outputs), &values(&eager), 1e-5, "compiled");

    let export = CudaExport::new(architecture).without_libraries();
    // A record may choose to run the module's nodes separately, so only the untuned program must hold its kernel.
    for (export, tuned) in [
        (export.clone(), false),
        // Records hold only for the way they were measured, so the program reads these replay records as its own, which only a test does.
        (
            export.with_tuning_records(records.measured_as(Measurement::Program)),
            true,
        ),
    ] {
        let program = export.export(&graph).unwrap();
        assert!(
            tuned
                || program
                    .steps
                    .iter()
                    .any(|step| format!("{step:?}").contains("rotary_cache_write")),
            "the program has no module kernel",
        );
        let binary = CompiledProgram::create(&program, "rotary_cache").unwrap();
        let operands = program
            .inputs
            .iter()
            .map(|input| {
                inputs
                    .iter()
                    .find(|(name, _)| *name == input.name)
                    .unwrap()
                    .1
                    .clone()
            })
            .collect::<Vec<_>>();
        let outputs = binary.instance().unwrap().run(&operands, 2).unwrap();
        assert_all_close(&values(&outputs), &values(&eager), 1e-5, "program");
    }
}

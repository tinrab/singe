use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    process, thread,
    time::Duration,
};

use singe_cudnn::backend::{engine::EngineIndex, execution::EngineChoice};

use super::{
    measure::{
        Expected, Measurer, SmallGraph, WholeGraphRunner, agrees, evaluate, export_program,
        listed_library_choices, remove_abandoned_candidates, subgraph, whole_graph,
    },
    plan::{
        Alternative, Candidate, ChoicePoint, MeasureError, Measured, PlanLimits, PlanRunner,
        check_plan, plan_records,
    },
    report::{CandidateOutcome, DropReason, FailureStage, SecondReference},
    tune::forced_records,
};
use crate::{
    backend::{
        cuda::{
            CudaEvent, CudaStream, EventTiming,
            compiled::plan::lowering::{CudaGraphProvider, LoweringPolicy},
            export::{CudaExport, ProgramStep, compiled::CompiledProgram},
            graph_binding::{InPlaceWrites, WritableInputs, graph_users},
            runtime::runtime_for_device,
            tuning::{
                GraphTuneOptions, Measurement, TunedChoice, TunedRecord, TunedWay,
                TuningEnvironment, TuningRecords, group_key, library_key, merged_projections_key,
                tune_graph, way_key,
            },
        },
        registry::Operation,
    },
    data_type::DataType,
    device::Device,
    error::Error,
    graph::{
        Graph, GraphCompileOptions, GraphCudaOptions, GraphPrepareOptions,
        capture::constant,
        ir::{Graph as IrGraph, ValueId},
        merged_projection_bytes,
    },
    index::TensorIndex,
    shape::Shape,
    tensor::{AttentionOptions, Tensor},
    testing::{maybe_lock_device, skip_if_stub},
};

const DEVICE: Device = Device::Cuda(0);

fn measure(
    small: &SmallGraph,
    records: &TuningRecords,
    device: Device,
    options: &GraphTuneOptions,
    refuse_over: Option<u64>,
    abandon_above: Option<Duration>,
) -> Result<Measured, MeasureError> {
    Measurer::new(device, options).measure(small, records, refuse_over, abandon_above)
}

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

fn record(choice: TunedChoice) -> TunedRecord {
    TunedRecord {
        choice,
        default_ns: None,
        best_ns: 0,
        bytes: 0,
    }
}

// A decoder step's shapes in small: a matmul of a few rows and attention, which each have several ways to run.
fn small_step() -> Option<(Graph, Vec<(&'static str, Tensor)>)> {
    let lhs = skip_if_stub(Ok(tensor(&[2, 64], DataType::Bf16, 1)))?;
    let inputs = vec![
        ("lhs", lhs),
        ("weight", tensor(&[32, 64], DataType::Bf16, 2)),
        ("query", tensor(&[1, 2, 3, 16], DataType::F16, 3)),
        ("key", tensor(&[1, 2, 4, 16], DataType::F16, 4)),
        ("value", tensor(&[1, 2, 4, 16], DataType::F16, 5)),
    ];
    let (graph, _) = Graph::capture(inputs.clone(), |inputs| {
        Ok(vec![
            inputs[0].matmul(&inputs[1].transpose(0, 1)?)?,
            inputs[2].attend(
                &inputs[3],
                &inputs[4],
                None,
                None,
                AttentionOptions {
                    causal: true,
                    ..AttentionOptions::default()
                },
            )?,
        ])
    })
    .unwrap();
    Some((graph, inputs))
}

fn named(inputs: &[(&str, Tensor)]) -> HashMap<String, Tensor> {
    inputs
        .iter()
        .map(|(name, tensor)| ((*name).to_owned(), tensor.clone()))
        .collect()
}

fn options(policy: LoweringPolicy) -> GraphTuneOptions {
    GraphTuneOptions {
        policy,
        choice_budget: Duration::from_secs(3),
        runs: 3,
        schedules: false,
        plan_budget: Duration::ZERO,
        plan_speed: false,
        ..GraphTuneOptions::default()
    }
}

fn run_with(
    graph: &Graph,
    inputs: &HashMap<String, Tensor>,
    records: TuningRecords,
) -> (Vec<Vec<f32>>, String) {
    let mut prepared = graph
        .prepare(
            GraphPrepareOptions::new(DEVICE)
                .with_cuda_options(GraphCudaOptions::new().with_tuning_records(records)),
        )
        .unwrap();
    let outputs = prepared.run(inputs).unwrap();
    let values = outputs
        .iter()
        .map(|output| output.cast(DataType::F32).unwrap().to_vec::<f32>().unwrap())
        .collect();
    (values, prepared.report().unwrap())
}

#[test]
fn test_a_record_chooses_how_a_node_runs_and_one_that_cannot_hold_is_an_error() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let Some((graph, inputs)) = small_step() else {
        return;
    };
    let inputs = named(&inputs);
    let compiled = graph
        .compile_with_options(
            GraphCompileOptions::new()
                .with_cuda_providers(LoweringPolicy::default().graph_providers()),
        )
        .unwrap();
    let ir = compiled.ir();
    let attention = ir
        .operations()
        .iter()
        .find(|operation| operation.opcode() == Some(Operation::ScaledDotProductAttention))
        .and_then(|operation| operation.result_values().next())
        .unwrap();
    let key = way_key(ir, attention).unwrap();
    let with_way = |way| {
        let mut records = TuningRecords::new(None);
        records.insert(key.clone(), record(TunedChoice::Way(way)));
        records
    };
    let (_, report) = run_with(&graph, &inputs, with_way(TunedWay::Generated));
    assert!(
        report.contains("scaled_dot_product_attention: provider Some(Kernel)"),
        "{report}"
    );
    let (_, report) = run_with(
        &graph,
        &inputs,
        with_way(TunedWay::Library(CudaGraphProvider::CudnnBackend)),
    );
    assert!(
        report.contains("scaled_dot_product_attention: provider Some(CudnnBackend)"),
        "{report}"
    );
    // cuBLAS cannot run attention, so preparing with this record fails, naming it.
    let error = graph
        .prepare(
            GraphPrepareOptions::new(DEVICE).with_cuda_options(
                GraphCudaOptions::new()
                    .with_tuning_records(with_way(TunedWay::Library(CudaGraphProvider::Cublas))),
            ),
        )
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("tuning records chose"), "{error}");
    // Records made under another environment are an error too, saying what differs.
    let elsewhere = TuningEnvironment::current(DEVICE, Measurement::Program).unwrap();
    let mut records = TuningRecords::new(Some(elsewhere));
    records.insert(key.clone(), record(TunedChoice::Way(TunedWay::Generated)));
    let error = graph
        .prepare(
            GraphPrepareOptions::new(DEVICE)
                .with_cuda_options(GraphCudaOptions::new().with_tuning_records(records)),
        )
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("measured as Program"), "{error}");
}

#[test]
fn test_a_tuned_graph_reads_back_from_its_file_and_exports_only_its_winners() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let Some((graph, inputs)) = small_step() else {
        return;
    };
    let named_inputs = named(&inputs);
    let records = tune_graph(
        &graph,
        &named_inputs,
        DEVICE,
        &options(LoweringPolicy::default()),
    )
    .unwrap();
    assert!(!records.is_empty());
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target")
        .join(format!("tuning-records-{}.json", process::id()));
    records.write(&path).unwrap();
    let read = TuningRecords::read(&path).unwrap();
    fs::remove_file(&path).unwrap();
    assert_eq!(read, records);

    let (expected, _) = run_with(&graph, &named_inputs, TuningRecords::new(None));
    assert_eq!(read.report(), records.report());
    let (actual, report) = run_with(&graph, &named_inputs, read.clone());
    assert!(report.contains("tuning records:"), "{report}");
    for (actual, expected) in actual.iter().zip(&expected) {
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 0.1,
                "{actual} is not {expected}"
            );
        }
    }

    // A generated winner leaves no library call in the program.
    let mut environment = read.environment().unwrap().clone();
    environment.measurement = Measurement::Program;
    let mut generated = TuningRecords::new(Some(environment.clone()));
    let compiled = graph
        .compile_with_options(
            GraphCompileOptions::new()
                .with_cuda_providers(LoweringPolicy::default().graph_providers()),
        )
        .unwrap();
    for operation in compiled.ir().operations() {
        if operation.opcode().is_some_and(|opcode| {
            matches!(
                opcode,
                Operation::MatMul | Operation::ScaledDotProductAttention
            )
        }) {
            let id = operation.result_values().next().unwrap();
            generated.insert(
                way_key(compiled.ir(), id).unwrap(),
                record(TunedChoice::Way(TunedWay::Generated)),
            );
        }
    }
    let program = CudaExport::new(environment.architecture.parse().unwrap())
        .with_tuning_records(generated)
        .export(&graph)
        .unwrap();
    assert!(
        program
            .steps
            .iter()
            .all(|step| !matches!(step, ProgramStep::Library { .. })),
        "{:?}",
        program.providers
    );
    assert!(program.record_coverage.used >= 2);
}

#[test]
fn test_a_record_can_keep_a_fusion_group_apart() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let Some(input) = skip_if_stub(Ok(tensor(&[8, 64], DataType::F32, 7))) else {
        return;
    };
    let inputs = [("x", input)];
    let (graph, _) = Graph::capture(inputs.clone(), |inputs| {
        Ok(vec![inputs[0].add(&inputs[0])?.exp()?])
    })
    .unwrap();
    let named_inputs = named(&inputs);
    let (expected, report) = run_with(&graph, &named_inputs, TuningRecords::new(None));
    assert!(report.contains("group"), "{report}");
    let compiled = graph
        .compile_with_options(
            GraphCompileOptions::new()
                .with_cuda_providers(LoweringPolicy::default().graph_providers()),
        )
        .unwrap();
    let members = compiled
        .ir()
        .operations()
        .iter()
        .filter_map(|operation| operation.result_values().next())
        .collect::<Vec<_>>();
    let mut records = TuningRecords::new(None);
    records.insert(
        group_key(compiled.ir(), &members).unwrap(),
        record(TunedChoice::Way(TunedWay::Separate)),
    );
    let (actual, report) = run_with(&graph, &named_inputs, records);
    assert!(report.contains("reads used"), "{report}");
    assert!(!report.contains("Pointwise"), "{report}");
    for (actual, expected) in actual[0].iter().zip(&expected[0]) {
        assert!((actual - expected).abs() <= 1e-4 * expected.abs().max(1.0));
    }
}

// Records hold only for the way they were measured, since a graph replay says nothing about a program's host cost, and the reverse, and using them elsewhere is an error that says what differs.
#[test]
fn test_records_are_read_only_where_they_were_measured() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let Some((graph, inputs)) = small_step() else {
        return;
    };
    let target = runtime_for_device(DEVICE).unwrap().architecture();
    let records_of = |environment| {
        let mut records = TuningRecords::new(Some(environment));
        records.insert(
            "unused".to_owned(),
            record(TunedChoice::Way(TunedWay::Generated)),
        );
        records
    };
    let records_for =
        |measurement| records_of(TuningEnvironment::current(DEVICE, measurement).unwrap());
    let export = |measurement| {
        CudaExport::new(target)
            .without_libraries()
            .with_tuning_records(records_for(measurement))
            .export(&graph)
            .map(|program| program.record_coverage)
    };
    let error = export(Measurement::GraphReplay).unwrap_err().to_string();
    assert!(error.contains("measured as GraphReplay"), "{error}");
    export(Measurement::Program).unwrap();
    let prepared = |records| {
        graph
            .prepare(
                GraphPrepareOptions::new(DEVICE)
                    .with_cuda_options(GraphCudaOptions::new().with_tuning_records(records)),
            )
            .map(drop)
    };
    let error = prepared(records_for(Measurement::Program))
        .unwrap_err()
        .to_string();
    assert!(error.contains("measured as Program"), "{error}");
    run_with(
        &graph,
        &named(&inputs),
        records_for(Measurement::GraphReplay),
    );

    // An engine or algorithm a record names means something only to one version of its library.
    for (change, named) in [
        (
            (|environment: &mut TuningEnvironment| environment.cublas = "other".to_owned())
                as fn(&mut TuningEnvironment),
            "with cuBLAS",
        ),
        (
            |environment| environment.cudnn = "other".to_owned(),
            "with cuDNN",
        ),
    ] {
        let mut environment = TuningEnvironment::current(DEVICE, Measurement::GraphReplay).unwrap();
        change(&mut environment);
        let error = prepared(records_of(environment)).unwrap_err().to_string();
        assert!(error.contains(named), "{error}");
    }
}

// A cache write stays in place in its small graph, so no cache copy is timed, and each candidate is judged on what it wrote.
#[test]
fn test_a_small_graph_keeps_a_cache_write_in_place_and_compares_it() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let Some(cache) = skip_if_stub(Ok(tensor(&[3, 4, 2, 16], DataType::F32, 30))) else {
        return;
    };
    let indices =
        |values: Vec<i64>| Tensor::from_vec(DEVICE, Shape::create([2]).unwrap(), values).unwrap();
    let inputs = |cache: Tensor| {
        vec![
            ("cache", cache),
            ("value", tensor(&[2, 2, 16], DataType::F32, 31)),
            ("pages", indices(vec![0, 2])),
            ("slots", indices(vec![1, 3])),
            ("other", tensor(&[2, 16], DataType::F32, 32)),
        ]
    };
    // Capturing runs the write, so what the small graph starts from is another cache.
    let captured = tensor(&[3, 4, 2, 16], DataType::F32, 30);
    let (graph, _) = Graph::capture(inputs(captured), |inputs| {
        inputs[0].index_put_(
            &[
                TensorIndex::Indices(inputs[2].clone()),
                TensorIndex::Indices(inputs[3].clone()),
            ],
            &inputs[1],
        )?;
        Ok(vec![inputs[4].relu()?])
    })
    .unwrap();
    let compiled = graph
        .placed_on(DEVICE)
        .unwrap()
        .compile_with_options(GraphCompileOptions::new())
        .unwrap();
    let ir = compiled.shared_ir();
    let order = compiled.value_order().to_vec();
    let write = order
        .iter()
        .copied()
        .find(|&id| {
            ir.required_operation_for_value(id).unwrap().opcode() == Some(Operation::IndexPut)
        })
        .unwrap();
    let relu = order
        .iter()
        .copied()
        .find(|&id| ir.required_operation_for_value(id).unwrap().opcode() == Some(Operation::Relu))
        .unwrap();
    let evaluated = evaluate(&ir, &order, &named(&inputs(cache)), DEVICE).unwrap();
    let mut small = subgraph(
        &ir,
        &evaluated,
        &[write, relu],
        &[write, relu],
        &every_write(&ir, &order),
        DEVICE,
    )
    .unwrap();
    assert_eq!(small.written.len(), 1);
    let options = GraphTuneOptions::default();
    let environment = TuningEnvironment::current(DEVICE, Measurement::GraphReplay).unwrap();
    let candidate = Candidate {
        choices: Vec::new(),
    };
    let records = forced_records(&candidate, &environment);
    assert!(
        measure(&small, &records, DEVICE, &options, None, None)
            .unwrap()
            .agrees
    );
    // The output, then the cache, which holds the write and starts again from where it began.
    assert_eq!(small.expected.len(), 2);
    let (name, before) = small.written[0].clone();
    let written = small.inputs[&name]
        .cast(DataType::F64)
        .unwrap()
        .to_vec::<f64>()
        .unwrap();
    assert_eq!(written.len(), 3 * 4 * 2 * 16);
    let original = before.cast(DataType::F64).unwrap().to_vec::<f64>().unwrap();
    assert_ne!(written, original);
    small.inputs[&name]
        .assign(&before.add_scalar(1.0).unwrap())
        .unwrap();
    assert!(
        measure(&small, &records, DEVICE, &options, None, None)
            .unwrap()
            .agrees
    );
    // A candidate that left the cache otherwise would not agree.
    small.expected[1] = small.expected[1].shifted(1.0).unwrap();
    assert!(
        !measure(&small, &records, DEVICE, &options, None, None)
            .unwrap()
            .agrees
    );
}

// More elements than one comparison reads at a time, so a difference in the last has to be reached.
#[test]
fn test_agreement_reads_every_chunk_of_a_tensor() {
    const COUNT: usize = 600_000;
    let shape = || Shape::create([i64::try_from(COUNT).unwrap()]).unwrap();
    let values = (0..COUNT)
        .map(|index| (index % 7) as f32)
        .collect::<Vec<_>>();
    let expected =
        Expected::create(&Tensor::from_vec(Device::Cpu, shape(), values.clone()).unwrap()).unwrap();
    let judge = |values: Vec<f32>| {
        let tensor = Tensor::from_vec(Device::Cpu, shape(), values).unwrap();
        agrees(DataType::F32, 1.0, &tensor, &expected).unwrap()
    };
    assert!(judge(values.clone()));
    // The tolerance is relative to the largest magnitude, six here.
    let mut close = values.clone();
    close[COUNT - 1] += 5e-4;
    assert!(judge(close));
    let mut wrong = values.clone();
    wrong[COUNT - 1] += 1e-2;
    assert!(!judge(wrong));
    let mut wrong = values.clone();
    wrong[0] = f32::NAN;
    assert!(!judge(wrong));

    // Matching NaNs and infinities agree, and an infinity does not count towards the magnitude.
    let mut special = values;
    special[COUNT - 1] = f32::NAN;
    special[COUNT - 2] = f32::INFINITY;
    let expected =
        Expected::create(&Tensor::from_vec(Device::Cpu, shape(), special.clone()).unwrap())
            .unwrap();
    let tensor = Tensor::from_vec(Device::Cpu, shape(), special).unwrap();
    assert!(agrees(DataType::F32, 1.0, &tensor, &expected).unwrap());

    // A tensor of another size does not agree.
    let shorter =
        Tensor::from_vec(Device::Cpu, Shape::create([3]).unwrap(), vec![0.0_f32; 3]).unwrap();
    assert!(!agrees(DataType::F32, 1.0, &shorter, &expected).unwrap());
}

fn every_write(ir: &IrGraph, order: &[ValueId]) -> InPlaceWrites {
    InPlaceWrites::analyze(
        ir,
        order,
        &graph_users(ir, order).unwrap(),
        &ir.output_value_set(),
        WritableInputs::Every,
    )
    .unwrap()
}

// A small graph of one relu, as a stand-in for any choice point.
fn relu_small() -> Option<SmallGraph> {
    let input = skip_if_stub(Ok(tensor(&[2, 8], DataType::F32, 40)))?;
    let inputs = [("x", input)];
    let (graph, _) = Graph::capture(inputs.clone(), |inputs| Ok(vec![inputs[0].relu()?])).unwrap();
    let compiled = graph
        .compile_with_options(GraphCompileOptions::new())
        .unwrap();
    let ir = compiled.shared_ir();
    let order = compiled.value_order().to_vec();
    let relu = order
        .iter()
        .copied()
        .find(|&id| ir.required_operation_for_value(id).unwrap().opcode() == Some(Operation::Relu))
        .unwrap();
    let evaluated = evaluate(&ir, &order, &named(&inputs), DEVICE).unwrap();
    Some(
        subgraph(
            &ir,
            &evaluated,
            &[relu],
            &[relu],
            &every_write(&ir, &order),
            DEVICE,
        )
        .unwrap(),
    )
}

fn stage_of(run: Result<Measured, MeasureError>) -> FailureStage {
    match run {
        Err(MeasureError::Failed(failure)) => {
            assert!(!failure.message.is_empty());
            failure.stage
        }
        other => panic!("expected a failure, found {other:?}"),
    }
}

#[test]
fn test_a_candidate_that_fails_says_at_which_stage() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let Some(mut small) = relu_small() else {
        return;
    };
    let options = GraphTuneOptions::default();
    let replay = TuningEnvironment::current(DEVICE, Measurement::GraphReplay).unwrap();
    let program = TuningEnvironment::current(DEVICE, Measurement::Program).unwrap();
    let candidate = Candidate {
        choices: Vec::new(),
    };

    let mut wrong_target = program.clone();
    wrong_target.architecture = "not an architecture".to_owned();
    let run = measure(
        &small,
        &forced_records(&candidate, &wrong_target),
        DEVICE,
        &options,
        None,
        None,
    );
    assert_eq!(stage_of(run), FailureStage::Export);

    // A directory cannot be made beneath a file.
    let blocker = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target")
        .join(format!("tuning-blocker-{}", process::id()));
    fs::write(&blocker, b"").unwrap();
    let unbuildable = GraphTuneOptions {
        build_directory: blocker.join("programs"),
        ..GraphTuneOptions::default()
    };
    let run = measure(
        &small,
        &forced_records(&candidate, &program),
        DEVICE,
        &unbuildable,
        None,
        None,
    );
    fs::remove_file(&blocker).unwrap();
    assert_eq!(stage_of(run), FailureStage::Build);

    let run = measure(
        &small,
        &forced_records(&candidate, &replay),
        Device::Cuda(99),
        &options,
        None,
        None,
    );
    assert_eq!(stage_of(run), FailureStage::Prepare);

    small.inputs.clear();
    let run = measure(
        &small,
        &forced_records(&candidate, &replay),
        DEVICE,
        &options,
        None,
        None,
    );
    assert_eq!(stage_of(run), FailureStage::Run);
}

#[test]
fn test_a_candidate_over_the_limit_is_refused_before_it_is_timed() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let Some(input) = skip_if_stub(Ok(tensor(&[8, 64], DataType::F32, 41))) else {
        return;
    };
    let inputs = [("x", input)];
    let (graph, _) = Graph::capture(inputs.clone(), |inputs| {
        Ok(vec![inputs[0].add(&inputs[0])?.exp()?])
    })
    .unwrap();
    let compiled = graph
        .compile_with_options(GraphCompileOptions::new())
        .unwrap();
    let ir = compiled.shared_ir();
    let order = compiled.value_order().to_vec();
    let evaluated = evaluate(&ir, &order, &named(&inputs), DEVICE).unwrap();
    let whole = whole_graph(&graph, &ir, &order, &evaluated, &named(&inputs)).unwrap();
    let environment = TuningEnvironment::current(DEVICE, Measurement::GraphReplay).unwrap();
    let separate = Candidate {
        choices: vec![(
            group_key(
                &ir,
                &order
                    .iter()
                    .copied()
                    .filter(|&id| {
                        ir.required_operation_for_value(id).is_ok_and(|operation| {
                            matches!(operation.opcode(), Some(Operation::Add | Operation::Exp))
                        })
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            TunedChoice::Way(TunedWay::Separate),
        )],
    };
    let options = options(LoweringPolicy::default());
    let forced = forced_records(&separate, &environment);
    let default = forced_records(
        &Candidate {
            choices: Vec::new(),
        },
        &environment,
    );
    let default_bytes = measure(&whole, &default, DEVICE, &options, None, None)
        .unwrap()
        .bytes;
    let separate_bytes = measure(&whole, &forced, DEVICE, &options, None, None)
        .unwrap()
        .bytes;
    assert!(
        separate_bytes > default_bytes,
        "{separate_bytes} {default_bytes}"
    );
    assert_eq!(
        measure(
            &whole,
            &forced,
            DEVICE,
            &options,
            Some(separate_bytes - 1),
            None
        )
        .unwrap_err(),
        MeasureError::OverMemory(separate_bytes)
    );

    // On a real plan, a limit between the two keeps what fits.
    let points = vec![ChoicePoint {
        key: "group".to_owned(),
        default_ns: Some(1),
        default_bytes: Some(default_bytes),
        alternatives: vec![Alternative {
            candidate: separate,
            median_ns: 1,
            bytes: separate_bytes,
        }],
        chosen: Some(0),
    }];
    let mut runner = WholeGraphRunner::new(&whole, DEVICE, &options);
    let limits = |memory| PlanLimits {
        memory: Some(memory),
        speed: false,
    };
    let (plan, report) = check_plan(
        &points,
        vec![Some(0)],
        Some(&environment),
        limits(separate_bytes - 1),
        &mut runner,
    );
    assert_eq!(plan, vec![None]);
    assert!(report.plan_bytes.unwrap() < separate_bytes);
    assert_eq!(report.dropped[0].reason, DropReason::OverMemory);
    assert!(!report.default_over_limit);
    let (plan, report) = check_plan(
        &points,
        vec![Some(0)],
        Some(&environment),
        limits(separate_bytes),
        &mut runner,
    );
    assert_eq!(plan, vec![Some(0)]);
    assert_eq!(report.plan_bytes, Some(separate_bytes));
    if default_bytes > 0 {
        let (plan, report) = check_plan(
            &points,
            vec![Some(0)],
            Some(&environment),
            limits(default_bytes - 1),
            &mut runner,
        );
        assert_eq!(plan, vec![None]);
        assert!(report.default_over_limit);
    }
}

#[test]
fn test_a_tuned_plan_holds_as_a_whole_and_explains_itself() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let Some((graph, inputs)) = small_step() else {
        return;
    };
    let named_inputs = named(&inputs);
    for measurement in [Measurement::GraphReplay, Measurement::Program] {
        let records = tune_graph(
            &graph,
            &named_inputs,
            DEVICE,
            &GraphTuneOptions {
                measurement,
                build_directory: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../target/programs/tuning"),
                choice_budget: Duration::from_secs(4),
                plan_budget: Duration::from_secs(3),
                ..options(LoweringPolicy::default())
            },
        )
        .unwrap();
        // Runs wrote the caller's tensors only if the graph writes any, and left them as they were.
        for (name, tensor) in &inputs {
            assert_eq!(
                tensor.to_bytes().unwrap(),
                named_inputs[*name].to_bytes().unwrap()
            );
        }
        let report = records.report().unwrap();
        let summary = report.to_string();
        assert!(!report.choices.is_empty(), "{summary}");
        assert_eq!(report.problems().count(), 0, "{summary}");
        let check = &report.checks[0];
        assert!(check.default_ns.is_some(), "{summary}");
        assert!(check.plan_ns.is_some(), "{summary}");
        assert!(check.default_failure.is_none(), "{summary}");
        assert!(!check.default_disagreed, "{summary}");
        assert!(check.dropped.is_empty(), "{summary}");
        assert!(report.searches[0].plans_run >= 1, "{summary}");
        assert!(summary.contains("plan search"), "{summary}");
    }
}

// What the evaluated graph says wins over every plan: a whole plan judged wrong keeps nothing.
#[test]
fn test_a_plan_the_whole_graph_disagrees_with_loses_its_records() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let Some((graph, inputs)) = small_step() else {
        return;
    };
    let named_inputs = named(&inputs);
    let compiled = graph
        .compile_with_options(
            GraphCompileOptions::new()
                .with_cuda_providers(LoweringPolicy::default().graph_providers()),
        )
        .unwrap();
    let ir = compiled.shared_ir();
    let order = compiled.value_order().to_vec();
    let evaluated = evaluate(&ir, &order, &named_inputs, DEVICE).unwrap();
    let mut whole = whole_graph(&graph, &ir, &order, &evaluated, &named_inputs).unwrap();
    let environment = TuningEnvironment::current(DEVICE, Measurement::GraphReplay).unwrap();
    let options = options(LoweringPolicy::default());
    let attention = order
        .iter()
        .copied()
        .find(|&id| {
            ir.required_operation_for_value(id).is_ok_and(|operation| {
                operation.opcode() == Some(Operation::ScaledDotProductAttention)
            })
        })
        .unwrap();
    let points = vec![ChoicePoint {
        key: "attention".to_owned(),
        default_ns: Some(1),
        default_bytes: Some(0),
        alternatives: vec![Alternative {
            candidate: Candidate {
                choices: vec![(
                    way_key(&ir, attention).unwrap(),
                    TunedChoice::Way(TunedWay::Generated),
                )],
            },
            median_ns: 1,
            bytes: 0,
        }],
        chosen: Some(0),
    }];
    let limits = PlanLimits::default();
    {
        let mut runner = WholeGraphRunner::new(&whole, DEVICE, &options);
        let (plan, report) = check_plan(
            &points,
            vec![Some(0)],
            Some(&environment),
            limits,
            &mut runner,
        );
        assert_eq!(plan, vec![Some(0)], "{report:?}");
        // The runner reports what the plan reads, which changes when the plan does.
        let records = plan_records(&points, &plan, Some(&environment));
        let print = runner.fingerprint(&records).unwrap();
        assert_ne!(
            print,
            runner
                .fingerprint(&TuningRecords::new(Some(environment.clone())))
                .unwrap()
        );
        assert_eq!(print, runner.fingerprint(&records).unwrap());
    }
    whole.expected[0] = whole.expected[0].shifted(1000.0).unwrap();
    let mut runner = WholeGraphRunner::new(&whole, DEVICE, &options);
    let (plan, report) = check_plan(
        &points,
        vec![Some(0)],
        Some(&environment),
        limits,
        &mut runner,
    );
    assert_eq!(plan, vec![None]);
    assert!(report.default_disagreed);
    assert_eq!(report.dropped[0].reason, DropReason::Disagreed);
}

// Candidates are judged against the evaluated graph, so wrong expectations reach every candidate, the first included.
#[test]
fn test_a_wrong_value_is_reported_as_disagreeing() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let Some(mut small) = relu_small() else {
        return;
    };
    small.expected[0] = small.expected[0].shifted(1000.0).unwrap();
    let environment = TuningEnvironment::current(DEVICE, Measurement::GraphReplay).unwrap();
    let candidates = [Candidate {
        choices: Vec::new(),
    }];
    let options = options(LoweringPolicy::default());
    let (point, report) = super::plan::decide(
        "relu".to_owned(),
        &candidates,
        Duration::from_secs(1),
        |candidate| {
            measure(
                &small,
                &forced_records(candidate, &environment),
                DEVICE,
                &options,
                None,
                None,
            )
        },
    );
    assert_eq!(point.chosen, None);
    assert_eq!(report.candidates[0].outcome, CandidateOutcome::Disagreed);
}

fn library_options() -> GraphTuneOptions {
    GraphTuneOptions {
        build_directory: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../target/programs/tuning"),
        choice_budget: Duration::from_secs(6),
        ..options(LoweringPolicy::default())
    }
}

// What `provider` lists for `operation` reaches the plan and a program, and what it refuses falls back and is counted.
fn check_library_choices(
    graph: &Graph,
    inputs: &HashMap<String, Tensor>,
    operation: Operation,
    provider: CudaGraphProvider,
) {
    let options = library_options();
    let compiled = graph
        .compile_with_options(
            GraphCompileOptions::new()
                .with_cuda_providers(LoweringPolicy::default().graph_providers()),
        )
        .unwrap();
    let ir = compiled.shared_ir();
    let order = compiled.value_order().to_vec();
    let evaluated = evaluate(&ir, &order, inputs, DEVICE).unwrap();
    let whole = whole_graph(graph, &ir, &order, &evaluated, inputs).unwrap();
    let id = order
        .iter()
        .copied()
        .find(|&id| {
            ir.required_operation_for_value(id)
                .is_ok_and(|node| node.opcode() == Some(operation))
        })
        .unwrap();
    let forced = |measurement, choice: Option<TunedChoice>| {
        let environment = TuningEnvironment::current(DEVICE, measurement).unwrap();
        let mut records = TuningRecords::new(Some(environment));
        records.insert(
            way_key(&ir, id).unwrap(),
            record(TunedChoice::Way(TunedWay::Library(provider))),
        );
        if let Some(choice) = choice {
            records.insert(library_key(&ir, id).unwrap(), record(choice));
        }
        records
    };
    let listed = listed_library_choices(
        &whole,
        &forced(Measurement::GraphReplay, None),
        DEVICE,
        &options,
        &way_key(&ir, id).unwrap(),
        provider,
    );
    eprintln!(
        "{provider:?} lists {} for {operation:?}: {listed:?}",
        listed.len()
    );
    assert!(listed.len() >= 2, "{provider:?} listed {listed:?}");

    let mut runner = WholeGraphRunner::new(&whole, DEVICE, &options);
    let mut plans = Vec::new();
    let mut programs = Vec::new();
    for choice in &listed[..2] {
        let replay = forced(Measurement::GraphReplay, Some(choice.clone()));
        plans.push(runner.fingerprint(&replay).unwrap());
        runner.run(&replay, None).unwrap();
        run_with(graph, inputs, replay);

        let program = forced(Measurement::Program, Some(choice.clone()));
        programs.push(runner.fingerprint(&program).unwrap());
        runner.run(&program, None).unwrap();
    }
    assert_ne!(plans[0], plans[1], "{provider:?}: {listed:?}");
    assert_ne!(programs[0], programs[1], "{provider:?}: {listed:?}");

    let refused = match &listed[0] {
        TunedChoice::CublasLt(config) => {
            let mut config = *config;
            config.id = 1_000_000;
            TunedChoice::CublasLt(config)
        }
        _ => TunedChoice::Cudnn(EngineChoice::new(EngineIndex::new(1_000_000), Vec::new())),
    };
    // A configuration the library refuses fails, in a compiled graph and in a program alike, rather than leaving the library to choose.
    let replay = forced(Measurement::GraphReplay, Some(refused.clone()));
    assert!(runner.run(&replay, None).is_err(), "{provider:?}");
    let error = graph
        .prepare(
            GraphPrepareOptions::new(DEVICE)
                .with_cuda_options(GraphCudaOptions::new().with_tuning_records(replay)),
        )
        .err()
        .unwrap();
    assert!(
        matches!(error, Error::ConfigurationRefused { .. })
            || error.to_string().contains("refused the configuration"),
        "{provider:?} {error}"
    );
    let refused = forced(Measurement::Program, Some(refused));
    assert!(runner.run(&refused, None).is_err(), "{provider:?}");

    // Configurations of one library at one choice point share a build, reading a table where what ships has literals.
    let export = |records: &TuningRecords, library_table| {
        let program = export_program(&whole, records, &options, library_table).unwrap();
        let files = program.render("model").unwrap();
        (program, files)
    };
    let table_of = |choice: &TunedChoice| forced(Measurement::Program, Some(choice.clone()));
    let (_, first) = export(&table_of(&listed[0]), true);
    let (_, second) = export(&table_of(&listed[1]), true);
    assert_eq!(first.source, second.source, "{provider:?}: {listed:?}");
    assert_ne!(first.library_table, second.library_table, "{provider:?}");
    assert!(first.forced_configurations >= 1, "{provider:?}");

    // Through a table, a configuration runs as itself and computes what it does through literals, and one that is refused fails the program, saying so.
    let (literal, literal_files) = export(&table_of(&listed[0]), false);
    assert!(literal_files.library_table.is_empty());
    let literal = CompiledProgram::create_in(
        &literal,
        options
            .build_directory
            .join(format!("literal-{}-{provider:?}", process::id())),
    )
    .unwrap();
    literal.instance().unwrap();
    let (refused_program, _) = export(&refused, false);
    let refused_program = CompiledProgram::create_in(
        &refused_program,
        options
            .build_directory
            .join(format!("refused-{}-{provider:?}", process::id())),
    )
    .unwrap();
    let error = refused_program.instance().err().unwrap();
    assert!(error.is_refusal(), "{provider:?} {error}");
    assert!(
        error.to_string().contains("refused") || error.to_string().contains("AlgoCheck"),
        "{provider:?} {error}"
    );
    let mut measurer = Measurer::for_choice_points(DEVICE, &options);
    measurer.build_ahead(
        &whole,
        &[table_of(&listed[0]), table_of(&listed[1]), refused.clone()],
    );
    for choice in &listed[..2] {
        let timed = measurer
            .measure(&whole, &table_of(choice), None, None)
            .unwrap();
        assert!(timed.agrees, "{provider:?} {choice:?}");
    }
    assert_eq!(
        measurer.measure(&whole, &refused, None, None),
        Err(MeasureError::Refused),
        "{provider:?}"
    );
    let compared = measurer
        .compare(&whole, &table_of(&listed[0]), &table_of(&listed[1]))
        .unwrap();
    assert!(compared.together, "{provider:?}");
    let compared = measurer
        .compare_in_blocks(&whole, &table_of(&listed[0]), &table_of(&listed[1]))
        .unwrap();
    assert!(!compared.together, "{provider:?}");
}

#[test]
fn test_candidates_of_a_process_that_stopped_are_removed() {
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target/programs")
        .join(format!("abandoned-{}", process::id()));
    let ours = directory.join(format!("candidate-{}-0", process::id()));
    // Above what Linux gives a process, so nothing runs with this id.
    let theirs = directory.join("candidate-4294967294-0");
    let other = directory.join("literal-4294967294");
    for kept in [&ours, &theirs, &other] {
        fs::create_dir_all(kept).unwrap();
    }

    remove_abandoned_candidates(&directory);
    assert!(ours.exists());
    assert!(other.exists());
    assert_eq!(theirs.exists(), !Path::new("/proc/self").exists());
    fs::remove_dir_all(&directory).unwrap();
}

// What a graph can choose does not depend on how long anything took.
#[test]
fn test_one_graph_lists_the_same_choices_twice() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let Some((graph, inputs)) = small_step() else {
        return;
    };
    let named_inputs = named(&inputs);
    let options = GraphTuneOptions {
        choice_budget: Duration::from_secs(3600),
        self_comparisons: true,
        ..library_options()
    };
    let listed = || {
        let records = tune_graph(&graph, &named_inputs, DEVICE, &options).unwrap();
        let report = records.report().unwrap().clone();
        assert!(!report.self_comparisons.is_empty(), "{report}");
        report
            .choices
            .iter()
            .map(|choice| {
                let candidates = choice
                    .candidates
                    .iter()
                    .map(|candidate| candidate.choice.clone())
                    .collect::<Vec<_>>();
                (choice.key.clone(), candidates)
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(listed(), listed());
}

#[test]
fn test_library_configurations_are_choices_and_a_refused_one_falls_back() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let Some((graph, inputs)) = small_step() else {
        return;
    };
    let named_inputs = named(&inputs);
    check_library_choices(
        &graph,
        &named_inputs,
        Operation::MatMul,
        CudaGraphProvider::CublasLt,
    );
    check_library_choices(
        &graph,
        &named_inputs,
        Operation::ScaledDotProductAttention,
        CudaGraphProvider::CudnnBackend,
    );
    // cuDNN's matrix products keep to half precision here, since the default policy refuses what they do to `F32`.
    let half = vec![
        ("lhs", tensor(&[2, 64], DataType::F16, 1)),
        ("rhs", tensor(&[64, 32], DataType::F16, 2)),
    ];
    let (matmul, _) = Graph::capture(half.clone(), |inputs| {
        Ok(vec![inputs[0].matmul(&inputs[1])?])
    })
    .unwrap();
    check_library_choices(
        &matmul,
        &named(&half),
        Operation::MatMul,
        CudaGraphProvider::CudnnBackend,
    );

    // Tuning lists them for what the policy enables, and only for that.
    let options = library_options();
    let records = tune_graph(&graph, &named_inputs, DEVICE, &options).unwrap();
    let report = records.report().unwrap();
    assert_eq!(report.problems().count(), 0, "{report}");
    let described = report
        .choices
        .iter()
        .flat_map(|choice| &choice.candidates)
        .map(|candidate| candidate.choice.as_str())
        .collect::<Vec<_>>();
    for listed in ["CublasLt(", "Cudnn("] {
        assert!(
            described.iter().any(|choice| choice.contains(listed)),
            "{report}"
        );
    }
    let records = tune_graph(
        &graph,
        &named_inputs,
        DEVICE,
        &GraphTuneOptions {
            policy: LoweringPolicy::generated_kernels(),
            ..options
        },
    )
    .unwrap();
    let report = records.report().unwrap();
    for choice in &report.choices {
        for candidate in &choice.candidates {
            assert!(
                !candidate.choice.contains("CublasLt(") && !candidate.choice.contains("Cudnn("),
                "{report}"
            );
        }
    }
}

// Sibling matmuls running as one is a choice for a whole graph, which the plan check holds to the same values, in graphs and in programs.
#[test]
fn test_merged_projections_are_a_choice_for_a_whole_graph() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let Some(left) = skip_if_stub(Ok(tensor(&[2, 8, 64], DataType::Bf16, 1))) else {
        return;
    };
    let weights = [
        tensor(&[32, 64], DataType::Bf16, 2),
        tensor(&[16, 64], DataType::Bf16, 3),
    ];
    let inputs = vec![("left", left)];
    let (graph, _) = Graph::capture(inputs.clone(), |inputs| {
        Ok(vec![
            inputs[0].matmul(&constant("first", &weights[0])?.transpose(0, 1)?)?,
            inputs[0].matmul(&constant("second", &weights[1])?.transpose(0, 1)?)?,
        ])
    })
    .unwrap();
    let named_inputs = named(&inputs);
    for measurement in [Measurement::GraphReplay, Measurement::Program] {
        let environment = TuningEnvironment::current(DEVICE, measurement).unwrap();
        let mut records = TuningRecords::new(Some(environment));
        records.insert(
            merged_projections_key(),
            record(TunedChoice::MergedProjections),
        );
        let options = GraphTuneOptions {
            measurement,
            build_directory: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../target/programs/tuning"),
            ..options(LoweringPolicy::default())
        };
        let compiled = graph
            .compile_with_options(
                GraphCompileOptions::new()
                    .with_cuda_providers(LoweringPolicy::default().graph_providers()),
            )
            .unwrap();
        let ir = compiled.shared_ir();
        let order = compiled.value_order().to_vec();
        let evaluated = evaluate(&ir, &order, &named_inputs, DEVICE).unwrap();
        let whole = whole_graph(&graph, &ir, &order, &evaluated, &named_inputs).unwrap();
        let merged_bytes = merged_projection_bytes(&graph).unwrap();
        let mut runner =
            WholeGraphRunner::new(&whole, DEVICE, &options).with_merged_bytes(merged_bytes);
        let measured = runner.run(&records, None).unwrap();
        assert!(measured.agrees, "{measurement:?}");
        assert!(measured.bytes >= merged_bytes, "{measurement:?}");

        // Joined weights count wherever a whole plan is judged, so a limit under them loses the merge.
        let points = [ChoicePoint {
            key: merged_projections_key(),
            default_ns: None,
            default_bytes: None,
            alternatives: vec![Alternative {
                candidate: Candidate {
                    choices: vec![(merged_projections_key(), TunedChoice::MergedProjections)],
                },
                median_ns: 1,
                bytes: measured.bytes,
            }],
            chosen: Some(0),
        }];
        let (plan, report) = check_plan(
            &points,
            vec![Some(0)],
            records.environment(),
            PlanLimits {
                memory: Some(measured.bytes - 1),
                speed: false,
            },
            &mut runner,
        );
        assert_eq!(plan, vec![None], "{measurement:?}");
        assert_eq!(report.dropped[0].reason, DropReason::OverMemory);
        assert!(!report.default_over_limit, "{measurement:?}");

        let tuned = tune_graph(&graph, &named_inputs, DEVICE, &options).unwrap();
        let report = tuned.report().unwrap();
        assert_eq!(report.problems().count(), 0, "{report}");
        // The merged matmul of 48 outputs is tuned for itself, under its own key, before the plans compare.
        assert!(
            report
                .choices
                .iter()
                .any(|choice| choice.key.contains("way|") && choice.key.contains("48")),
            "{report}"
        );
        assert!(
            report
                .choices
                .iter()
                .any(|choice| choice.key == merged_projections_key()
                    && choice.candidates.iter().all(|candidate| matches!(
                        candidate.outcome,
                        CandidateOutcome::Timed { .. }
                    ))),
            "{report}"
        );
    }
}

#[test]
fn test_a_held_stream_starts_nothing_until_it_is_released() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let Some(stream) = skip_if_stub(CudaStream::create(DEVICE)) else {
        return;
    };
    let event = CudaEvent::create(DEVICE, EventTiming::Recorded).unwrap();
    let hold = stream.hold().unwrap();
    event.record(&stream).unwrap();
    thread::sleep(Duration::from_millis(50));
    assert!(!event.query().unwrap());
    hold.release();
    event.synchronize().unwrap();
    assert!(!hold.overran());
}

// One set of inputs can hide a kernel's error, so every candidate is also judged on a second set, of other signs and magnitudes, and a value wrong only there is a disagreement.
#[test]
fn test_a_candidate_is_judged_on_a_second_set_of_inputs() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let Some(mut small) = relu_small() else {
        return;
    };
    let second = small.second.as_ref().unwrap();
    assert_eq!(second.reference, SecondReference::Host);
    let name = second.values[0].0.clone();
    let first = small.inputs[&name].to_vec::<f32>().unwrap();
    let other = second.values[0].1.to_vec::<f32>().unwrap();
    assert!(
        first
            .iter()
            .zip(&other)
            .all(|(first, other)| first.signum() != other.signum() || *first == 0.0)
    );
    let environment = TuningEnvironment::current(DEVICE, Measurement::GraphReplay).unwrap();
    let records = forced_records(
        &Candidate {
            choices: Vec::new(),
        },
        &environment,
    );
    let options = options(LoweringPolicy::default());
    let measured = measure(&small, &records, DEVICE, &options, None, None).unwrap();
    assert!(measured.agrees);
    // The first inputs are back in place, so what is timed reads them.
    assert_eq!(small.inputs[&name].to_vec::<f32>().unwrap(), first);
    // A value wrong only on the second inputs, which the first alone would never show.
    let second = small.second.as_mut().unwrap();
    second.expected[0] = second.expected[0].shifted(1000.0).unwrap();
    let measured = measure(&small, &records, DEVICE, &options, None, None).unwrap();
    assert!(!measured.agrees);
}

// The second inputs' values come from the host where the inputs fit, which shares no library with a candidate on the device, and from the device otherwise.
#[test]
fn test_second_inputs_are_evaluated_on_the_host_where_they_fit() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let small_matmul = |rows: i64| {
        let inputs = [
            ("x", tensor(&[rows, 64], DataType::Bf16, 50)),
            ("w", tensor(&[32, 64], DataType::Bf16, 51)),
        ];
        let (graph, _) = Graph::capture(inputs.clone(), |inputs| {
            Ok(vec![inputs[0].matmul(&inputs[1].transpose(0, 1)?)?])
        })
        .unwrap();
        let compiled = graph
            .compile_with_options(GraphCompileOptions::new())
            .unwrap();
        let ir = compiled.shared_ir();
        let order = compiled.value_order().to_vec();
        let matmul = order
            .iter()
            .copied()
            .find(|&id| {
                ir.required_operation_for_value(id).unwrap().opcode() == Some(Operation::MatMul)
            })
            .unwrap();
        let evaluated = evaluate(&ir, &order, &named(&inputs), DEVICE).unwrap();
        subgraph(
            &ir,
            &evaluated,
            &[matmul],
            &[matmul],
            &every_write(&ir, &order),
            DEVICE,
        )
        .unwrap()
    };
    let small = small_matmul(4);
    assert_eq!(
        small.second.as_ref().unwrap().reference,
        SecondReference::Host
    );
    // cuBLAS computes the device's eager matmul, and the host's agrees with what a candidate computes.
    let environment = TuningEnvironment::current(DEVICE, Measurement::GraphReplay).unwrap();
    let measured = measure(
        &small,
        &forced_records(
            &Candidate {
                choices: Vec::new(),
            },
            &environment,
        ),
        DEVICE,
        &options(LoweringPolicy::default()),
        None,
        None,
    )
    .unwrap();
    assert!(measured.agrees);
    // Over the host's share, the device evaluates them.
    let large = small_matmul(1 << 20);
    assert_eq!(
        large.second.as_ref().unwrap().reference,
        SecondReference::Device
    );
}

// A record keys a matmul by the elements its operands address, so a view whose axis of extent one has another stride reads the same record.
#[test]
fn test_a_record_holds_whatever_stride_an_axis_of_extent_one_has() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let key_of = |transposed: bool| {
        let lhs = if transposed {
            tensor(&[64, 1], DataType::F32, 60)
        } else {
            tensor(&[1, 64], DataType::F32, 60)
        };
        let inputs = [("lhs", lhs), ("rhs", tensor(&[64, 8], DataType::F32, 61))];
        let (graph, _) = Graph::capture(inputs, |inputs| {
            let lhs = if transposed {
                inputs[0].transpose(0, 1)?
            } else {
                inputs[0].clone()
            };
            Ok(vec![lhs.matmul(&inputs[1])?])
        })
        .unwrap();
        let compiled = graph
            .compile_with_options(GraphCompileOptions::new())
            .unwrap();
        let ir = compiled.ir();
        let matmul = ir
            .operations()
            .iter()
            .find(|operation| operation.opcode() == Some(Operation::MatMul))
            .and_then(|operation| operation.result_values().next())
            .unwrap();
        let lhs = ir.required_operation_for_value(matmul).unwrap().inputs()[0];
        (
            way_key(ir, matmul).unwrap(),
            ir.required_value_layout(lhs).unwrap().strides().to_vec(),
        )
    };
    let (plain, plain_strides) = key_of(false);
    let (viewed, viewed_strides) = key_of(true);
    assert_ne!(plain_strides, viewed_strides);
    assert_eq!(plain, viewed);
}

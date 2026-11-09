use std::{collections::HashMap, fs, path::PathBuf, process, slice, time::Duration};

use crate::{
    backend::{
        cuda::{
            self, CudaGraphProvider,
            codegen::{KernelDescription, emit_kernel},
            compiled::plan::lowering::LoweringPolicy,
            kernel_tuning::{KernelBench, Outcome, TuneOptions, tune},
            tuning::{GraphTuneOptions, TuningRecords, tune_graph},
        },
        kernel_lowering::{ElementGroup, LoweredKernel, normalization, softmax},
        registry::{LayerNormalizationAttributes, Operation},
    },
    data_type::DataType,
    device::Device,
    graph::{Graph, GraphCudaOptions, GraphOperationOutput, GraphOperationRequest},
    layout::Layout,
    shape::Shape,
    tensor::Tensor,
    testing::{maybe_lock_device, skip_if_stub},
};

/// Lowers RMS norm of BF16 `rows` × `width` with a weight.
fn rms_norm(rows: i64, width: i64) -> LoweredKernel {
    let input = Layout::contiguous([rows, width]).unwrap();
    let weight = Layout::contiguous([width]).unwrap();
    normalization(
        Operation::RmsNorm,
        &LayerNormalizationAttributes {
            normalized_shape: vec![width],
            has_weight: true,
            has_bias: false,
            eps: 1e-6,
        },
        &ElementGroup::input(DataType::Bf16, &input),
        &[(DataType::Bf16, &weight)],
    )
    .unwrap()
}

#[test]
fn test_tuning_times_matching_candidates_and_keeps_the_default_unless_beaten() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let lowered = rms_norm(256, 1024);
    let Some(mut bench) = skip_if_stub(KernelBench::create(
        device,
        &lowered.kernel,
        &lowered.bindings,
    )) else {
        return;
    };
    // Two warm passes over the same kernel agree, so the search, which times warm, can tell candidates apart.
    // A cold sample subtracts a flush of twice the L2 cache, whose own variation exceeds a 2 µs kernel.
    let generated = emit_kernel(
        &lowered.kernel,
        lowered.schedule,
        &lowered.bindings,
        &KernelDescription::default(),
    )
    .unwrap();
    let first = bench.time(slice::from_ref(&generated), 20, false).unwrap()[0];
    let second = bench.time(slice::from_ref(&generated), 20, false).unwrap()[0];
    let ratio = first.median.as_secs_f64() / second.median.as_secs_f64();
    assert!(
        (0.75..1.33).contains(&ratio),
        "{first:?} against {second:?}"
    );
    assert!(
        !bench.outputs(&generated).unwrap()[0]
            .iter()
            .all(|&byte| byte == 0)
    );

    let options = TuneOptions {
        tile: true,
        rounds: 10,
        budget: Duration::from_secs(120),
    };
    let result = tune(
        device,
        &lowered.kernel,
        lowered.schedule,
        &lowered.bindings,
        options,
    )
    .unwrap();
    assert_eq!(result.default, lowered.schedule);
    assert!(matches!(result.candidates[0].1, Outcome::Timed(_)));
    assert!(
        result
            .candidates
            .iter()
            .all(|(_, outcome)| !matches!(outcome, Outcome::Mismatched)),
        "{result:?}"
    );
    let default = result.timing(result.default).unwrap();
    let best = result.timing(result.best).unwrap();
    assert!(best.median <= default.median);
}

#[test]
fn test_a_tuned_graph_compiles_from_its_file_and_computes_the_same_values() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let shape = Shape::create([64, 512]).unwrap();
    let values = (0..64 * 512)
        .map(|index| f32::from(u16::try_from(index % 29).unwrap()) / 29.0)
        .collect::<Vec<_>>();
    let Some(input) = skip_if_stub(Tensor::from_vec(device, &shape, values.clone())) else {
        return;
    };
    // `exp(x + x)` fuses into one generated kernel.
    let mut graph = Graph::new();
    let input_id = graph.add_input("input", DataType::F32, shape.clone(), device);
    let output = GraphOperationOutput::new(DataType::F32, shape, device);
    let doubled = graph
        .add_operation_request(GraphOperationRequest::binary(
            None,
            Operation::Add,
            [input_id, input_id],
            output.clone(),
        ))
        .unwrap();
    let exponent = graph
        .add_operation_request(GraphOperationRequest::unary(
            None,
            Operation::Exp,
            doubled,
            output,
        ))
        .unwrap();
    graph.set_outputs(vec![exponent]).unwrap();
    let policy = LoweringPolicy::new(vec![
        CudaGraphProvider::Alias,
        CudaGraphProvider::Kernel,
        CudaGraphProvider::Fusion,
    ]);
    let inputs = HashMap::from([("input".to_owned(), input.clone())]);
    let tuned = tune_graph(
        &graph,
        &inputs,
        device,
        &GraphTuneOptions {
            policy: policy.clone(),
            choice_budget: Duration::from_secs(20),
            runs: 5,
            schedules: true,
            plan_budget: Duration::ZERO,
            plan_speed: false,
            ..GraphTuneOptions::default()
        },
    );
    let Some(tuned) = skip_if_stub(tuned) else {
        return;
    };
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target")
        .join(format!("graph-tuning-{}.json", process::id()));
    tuned.write(&path).unwrap();
    let records = TuningRecords::read(&path).unwrap();
    fs::remove_file(&path).unwrap();
    assert_eq!(records, tuned);
    let options = GraphCudaOptions::new()
        .with_lowering_policy(policy)
        .with_tuning_records(records);
    let mut compiled = cuda::compile_graph(&graph, &options).unwrap();

    // Whatever schedule the search chose, the compiled graph computes the same values.
    let outputs = compiled
        .run(&HashMap::from([("input".to_owned(), input)]))
        .unwrap();
    for (value, input) in outputs[0].to_vec::<f32>().unwrap().iter().zip(&values) {
        let expected = (input + input).exp();
        assert!(
            (value - expected).abs() <= 1e-5 * expected,
            "{value} {expected}"
        );
    }
}

#[test]
fn test_every_listed_schedule_of_the_milestone_rows_matches_the_default() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let scores = Layout::contiguous([16, 256, 256]).unwrap();
    let kernels = [
        rms_norm(1, 1024),
        rms_norm(16, 1024),
        rms_norm(4096, 128),
        softmax(
            Operation::Softmax,
            &ElementGroup::input(DataType::Bf16, &scores),
            -1,
            None,
        )
        .unwrap(),
    ];
    let options = TuneOptions {
        tile: false,
        rounds: 1,
        budget: Duration::from_secs(120),
    };
    for lowered in kernels {
        let Some(result) = skip_if_stub(tune(
            device,
            &lowered.kernel,
            lowered.schedule,
            &lowered.bindings,
            options,
        )) else {
            return;
        };
        assert!(result.candidates.len() > 10, "{result:?}");
        assert!(
            result
                .candidates
                .iter()
                .all(|(_, outcome)| matches!(outcome, Outcome::Timed(_))),
            "{} {result:?}",
            lowered.kernel.name
        );
    }
}

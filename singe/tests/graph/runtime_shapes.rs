use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use singe::{
    DataType, Device, Result, Shape, Tensor,
    graph::{CaptureBindings, DimensionExpr, Graph, GraphInputSpec, GraphPrepareOptions},
    nn::{Layer, Model, ModelProgram, ModelState, StateBuilder},
    tensor::InterpolateMode,
    value::ValueSpec,
};

#[cfg(feature = "testing")]
use singe::testing::TensorActivity;

#[derive(Debug, Clone, Copy)]
enum Arithmetic {
    Multiply(i64),
    DivideByPrevious,
    TruncatedNegativeHalf,
}

#[derive(Debug)]
struct ShapeModule {
    arithmetic: Arithmetic,
    constructions: Arc<AtomicUsize>,
}

impl Layer for ShapeModule {
    fn forward(&self, input: &Tensor, _state: &ModelState) -> Result<Tensor> {
        self.constructions.fetch_add(1, Ordering::Relaxed);
        let batch = input.dimension(0)?;
        let merged = match self.arithmetic {
            Arithmetic::Multiply(factor) => batch * factor,
            Arithmetic::DivideByPrevious => batch.clone() / (batch - 1),
            Arithmetic::TruncatedNegativeHalf => {
                DimensionExpr::constant(0) - ((DimensionExpr::constant(0) - batch) / 2)
            }
        };
        input.reshape_with(&[merged, 0.into()])
    }
}

// Narrows, resizes, counts, and expands by its input's run-time height and width.
#[derive(Debug)]
struct SizedModule;

impl Layer for SizedModule {
    fn forward(&self, input: &Tensor, _state: &ModelState) -> Result<Tensor> {
        let device = input.device();
        let (height, width) = (input.dimension(2)?, input.dimension(3)?);
        let rows = height.clone() * 2;
        let resized = input
            .narrow_with(3, 1_i64, width.clone() - 1)?
            .interpolate_with((rows.clone(), width.clone()), InterpolateMode::Bilinear)?;
        let columns = Tensor::range(device, 0_i64, width.clone())?
            .cast(DataType::F32)?
            .unsqueeze(0)?
            .expand_with(&[rows, width.clone()])?;
        let width = Tensor::from_dimensions(device, &[height, width])?
            .narrow(0, 1, 1)?
            .cast(DataType::F32)?;
        resized.add(&columns)?.add(&width)
    }
}

fn program(batch: i64, arithmetic: Arithmetic, constructions: Arc<AtomicUsize>) -> ModelProgram {
    let model = Model::new(
        ShapeModule {
            arithmetic,
            constructions,
        },
        StateBuilder::new().finish().unwrap(),
    );
    let spec = GraphInputSpec::new(ValueSpec::new(
        DataType::F32,
        Shape::create([batch, 0]).unwrap(),
    ))
    .with_dimension(0, "batch", Some(1), None)
    .unwrap();
    model.build("x", spec).unwrap()
}

fn input(batch: i64) -> HashMap<String, Tensor> {
    HashMap::from([(
        "x".into(),
        Tensor::from_vec(
            Device::Cpu,
            Shape::create([batch, 0]).unwrap(),
            Vec::<f32>::new(),
        )
        .unwrap(),
    )])
}

fn exercise_shape_specialization(device: Device) {
    let constructions = Arc::new(AtomicUsize::new(0));
    let program = program(2, Arithmetic::Multiply(2), Arc::clone(&constructions));
    assert_eq!(constructions.load(Ordering::Relaxed), 1);
    let mut executor = program
        .prepare(GraphPrepareOptions::new(device).with_cache_capacity(2))
        .unwrap();
    for batch in [2, 3, 4, 2] {
        let inputs = input(batch)
            .into_iter()
            .map(|(name, tensor)| (name, tensor.to_device(device).unwrap()))
            .collect();
        let outputs = executor.run(&inputs).unwrap();
        assert_eq!(outputs[0].shape().dimensions(), [batch * 2, 0]);
    }
    assert_eq!(constructions.load(Ordering::Relaxed), 1);

    // Slices, resizes, ranges, expansions, and dimension vectors sized by run-time axes match the eager forward at each shape.
    let state = StateBuilder::new().finish().unwrap();
    let spec = GraphInputSpec::new(ValueSpec::new(
        DataType::F32,
        Shape::create([1, 2, 3, 4]).unwrap(),
    ))
    .with_dimension(2, "height", Some(2), None)
    .unwrap()
    .with_dimension(3, "width", Some(2), None)
    .unwrap();
    let program = Model::new(SizedModule, state.clone())
        .build("x", spec)
        .unwrap();
    let mut executor = program.prepare(GraphPrepareOptions::new(device)).unwrap();
    for (height, width) in [(3, 4), (5, 7), (3, 4)] {
        let count = 2 * height * width;
        let values = (0..count).map(|value| value as f32 / 7.0).collect();
        let x = Tensor::from_vec(
            Device::Cpu,
            Shape::create([1, 2, height, width]).unwrap(),
            values,
        )
        .and_then(|x| x.to_device(device))
        .unwrap();
        let expected = SizedModule.forward(&x, &state).unwrap();
        let output = executor
            .run(&HashMap::from([("x".into(), x)]))
            .unwrap()
            .remove(0);
        assert_eq!(output.shape().dimensions(), [1, 2, 2 * height, width]);
        let difference = output
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .zip(expected.to_vec::<f32>().unwrap())
            .map(|(output, expected)| (output - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(difference < 1e-5, "{height}x{width}: {difference}");
    }
}

#[test]
fn test_cpu_shape_specialization_never_reruns_layer_rust_code() {
    exercise_shape_specialization(Device::Cpu);
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_shape_specialization_never_reruns_layer_rust_code() {
    exercise_shape_specialization(Device::Cuda(0));
}

#[test]
fn test_overflow_and_division_by_zero_fail_before_dispatch_or_cache_publication() {
    for (arithmetic, batch) in [
        (Arithmetic::Multiply(2), i64::MAX),
        (Arithmetic::DivideByPrevious, 1),
    ] {
        let program = program(2, arithmetic, Arc::new(AtomicUsize::new(0)));
        let mut executor = program
            .prepare(GraphPrepareOptions::new(Device::Cpu))
            .unwrap();
        let inputs = input(batch);
        let cache = executor.cache_stats();
        #[cfg(feature = "testing")]
        let before = TensorActivity::current();
        assert!(executor.run(&inputs).is_err());
        #[cfg(feature = "testing")]
        assert_eq!(TensorActivity::current(), before);
        assert_eq!(executor.cache_stats(), cache);
        assert_eq!(
            executor.run(&input(2)).unwrap()[0].shape().dimensions()[1],
            0
        );
    }
}

#[test]
fn test_invalid_symbol_declarations_and_foreign_dimension_expressions_are_rejected() {
    let spec = ValueSpec::new(DataType::F32, Shape::create([2, 0]).unwrap());
    assert!(
        GraphInputSpec::new(spec.clone())
            .with_dimension(0, "batch", Some(3), None)
            .is_err()
    );
    assert!(
        GraphInputSpec::new(spec.clone())
            .with_dimension(2, "axis", None, None)
            .is_err()
    );
    assert!(
        GraphInputSpec::new(spec.clone())
            .with_dimension(0, "", None, None)
            .is_err()
    );
    let mut foreign = None;
    Graph::build([("x", spec.clone())], CaptureBindings::new(), |inputs| {
        foreign = Some(inputs[0].dimension(0)?);
        Ok(vec![inputs[0].clone()])
    })
    .unwrap();
    let foreign = foreign.unwrap();
    assert!(
        Graph::build([("y", spec)], CaptureBindings::new(), |inputs| {
            Ok(vec![inputs[0].reshape_with(&[foreign, 0.into()])?])
        })
        .is_err()
    );
}

#[test]
fn test_integer_dimension_division_truncates_toward_zero() {
    let program = program(
        2,
        Arithmetic::TruncatedNegativeHalf,
        Arc::new(AtomicUsize::new(0)),
    );
    let mut executor = program
        .prepare(GraphPrepareOptions::new(Device::Cpu))
        .unwrap();
    assert_eq!(
        executor.run(&input(3)).unwrap()[0].shape().dimensions(),
        [1, 0]
    );
}

#[test]
fn test_metadata_only_shape_dependencies_and_shared_symbols_survive_pruning() {
    let spec = GraphInputSpec::new(ValueSpec::new(
        DataType::F32,
        Shape::create([2, 0]).unwrap(),
    ))
    .with_dimension(0, "batch", Some(1), Some(8))
    .unwrap();
    let graph = Graph::build(
        [("x", spec.clone()), ("y", spec)],
        CaptureBindings::new(),
        |inputs| {
            let dimension = inputs[1].dimension(0)?;
            Ok(vec![inputs[0].reshape_with(&[dimension, 0.into()])?])
        },
    )
    .unwrap();
    let program = graph.program().unwrap();
    let state = program.bind_state(&HashMap::new()).unwrap();
    let mut executor = program
        .prepare(&state, GraphPrepareOptions::new(Device::Cpu))
        .unwrap();
    let x = input(3).remove("x").unwrap();
    let y = input(4).remove("x").unwrap();
    assert!(
        executor
            .run(&HashMap::from([("x".into(), x.clone()), ("y".into(), y)]))
            .is_err()
    );
    let output = executor
        .run(&HashMap::from([("x".into(), x.clone()), ("y".into(), x)]))
        .unwrap();
    assert_eq!(output[0].shape().dimensions(), [3, 0]);
}

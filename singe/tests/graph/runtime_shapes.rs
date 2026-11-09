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
            Arithmetic::Multiply(factor) => batch * factor.into(),
            Arithmetic::DivideByPrevious => batch.clone() / (batch - 1.into()),
            Arithmetic::TruncatedNegativeHalf => {
                DimensionExpr::constant(0) - ((DimensionExpr::constant(0) - batch) / 2.into())
            }
        };
        input.reshape_with(&[merged, 0.into()])
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

#[test]
fn test_expression_identity_excludes_representatives_and_includes_arithmetic() {
    let constructions = Arc::new(AtomicUsize::new(0));
    let first = program(1, Arithmetic::Multiply(2), Arc::clone(&constructions));
    let second = program(3, Arithmetic::Multiply(2), Arc::clone(&constructions));
    let other = program(1, Arithmetic::Multiply(3), constructions);
    assert_eq!(
        first.program().fingerprint().unwrap(),
        second.program().fingerprint().unwrap()
    );
    assert_ne!(
        first.program().fingerprint().unwrap(),
        other.program().fingerprint().unwrap()
    );
}

#[test]
fn test_shape_specialization_never_reruns_layer_rust_code() {
    let constructions = Arc::new(AtomicUsize::new(0));
    let program = program(2, Arithmetic::Multiply(2), Arc::clone(&constructions));
    assert_eq!(constructions.load(Ordering::Relaxed), 1);
    let mut executor = program
        .prepare(GraphPrepareOptions::new(Device::Cpu).with_cache_capacity(2))
        .unwrap();
    for batch in [2, 3, 4, 2] {
        let outputs = executor.run(&input(batch)).unwrap();
        assert_eq!(outputs[0].shape().dimensions(), [batch * 2, 0]);
    }
    assert_eq!(constructions.load(Ordering::Relaxed), 1);
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

use std::{collections::HashMap, thread};

use singe::{
    Device, Shape, Tensor,
    graph::{Graph, GraphPrepareOptions, ProgramInputBinding, StateBindingKind, StateFreezePolicy},
};

fn vector(values: &[f32]) -> Tensor {
    Tensor::from_vec(
        Device::Cpu,
        Shape::create(vec![i64::try_from(values.len()).unwrap()]).unwrap(),
        values.to_vec(),
    )
    .unwrap()
}

fn source() -> Graph {
    let (captured, _) = Graph::capture(
        [
            ("x", vector(&[1., 2.])),
            ("weight", vector(&[2., 3.])),
            ("bias", vector(&[1., 1.])),
        ],
        |inputs| Ok(vec![inputs[0].mul(&inputs[1])?.add(&inputs[2])?]),
    )
    .unwrap();
    let mut graph = captured;
    graph.add_parameter("weight", vector(&[2., 3.])).unwrap();
    graph.add_buffer("bias", vector(&[1., 1.])).unwrap();
    graph
}

fn bindings() -> HashMap<String, Tensor> {
    HashMap::from([
        ("weight".into(), vector(&[2., 3.])),
        ("bias".into(), vector(&[1., 1.])),
    ])
}

fn input() -> HashMap<String, Tensor> {
    HashMap::from([("x".into(), vector(&[2., 4.]))])
}

fn two_parameters() -> Graph {
    let (capture, _) = Graph::capture(
        [
            ("x", vector(&[1., 2.])),
            ("a", vector(&[2., 3.])),
            ("b", vector(&[2., 3.])),
        ],
        |inputs| Ok(vec![inputs[0].mul(&inputs[1])?.add(&inputs[2])?]),
    )
    .unwrap();
    let mut graph = capture;
    graph.add_parameter("a", vector(&[2., 3.])).unwrap();
    graph.add_parameter("b", vector(&[2., 3.])).unwrap();
    graph
}

fn exercise_tied_state(device: Device) {
    let untied = two_parameters().program().unwrap();
    let program = untied.tie_state("a", "b").unwrap();
    let slots = program.state_schema().collect::<Vec<_>>();
    assert_eq!(slots[0].id(), slots[1].id());
    assert_ne!(
        program.fingerprint().unwrap(),
        untied.fingerprint().unwrap()
    );
    assert_eq!(
        program.fingerprint().unwrap(),
        untied.tie_state("b", "a").unwrap().fingerprint().unwrap()
    );
    let state = program
        .bind_state(&HashMap::from([("b".into(), vector(&[2., 3.]))]))
        .unwrap();
    assert_eq!(
        state.tensor("a").unwrap().id(),
        state.tensor("b").unwrap().id()
    );
    let moved = state.to_device(device).unwrap();
    assert_eq!(
        moved.tensor("a").unwrap().id(),
        moved.tensor("b").unwrap().id()
    );
    let mut executor = program
        .prepare(&moved, GraphPrepareOptions::new(device))
        .unwrap();
    let mut old = executor.create_instance().unwrap();
    let inputs = HashMap::from([("x".into(), vector(&[3., 4.]).to_device(device).unwrap())]);
    let retained = executor.run(&inputs).unwrap();
    let key = executor.specialization_key().clone();
    let replacement = state
        .replace(&HashMap::from([("a".into(), vector(&[4., 5.]))]))
        .unwrap();
    assert_eq!(
        replacement.tensor("a").unwrap().id(),
        replacement.tensor("b").unwrap().id()
    );
    executor.rebind_state(&replacement).unwrap();
    assert_ne!(executor.specialization_key(), &key);
    assert_eq!(executor.specialization_key().program(), key.program());
    assert_eq!(
        executor.run(&inputs).unwrap()[0].to_vec::<f32>().unwrap(),
        [16., 25.]
    );
    assert_eq!(
        old.run(&inputs).unwrap()[0].to_vec::<f32>().unwrap(),
        [8., 15.]
    );
    assert_eq!(retained[0].to_vec::<f32>().unwrap(), [8., 15.]);
}

#[test]
fn test_declared_state_ties_survive_binding_placement_replacement_and_old_executors() {
    exercise_tied_state(Device::Cpu);
}

#[test]
fn test_tied_binding_conflicts_are_rejected_before_publishing_any_replacement() {
    let program = two_parameters()
        .program()
        .unwrap()
        .tie_state("a", "b")
        .unwrap();
    assert!(
        program
            .bind_state(&HashMap::from([
                ("a".into(), vector(&[2., 3.])),
                ("b".into(), vector(&[4., 5.]))
            ]))
            .is_err()
    );
    let state = program
        .bind_state(&HashMap::from([
            ("a".into(), vector(&[2., 3.])),
            ("b".into(), vector(&[2., 3.])),
        ]))
        .unwrap();
    assert_eq!(
        state.tensor("a").unwrap().id(),
        state.tensor("b").unwrap().id()
    );
    let revision = state.revision();
    assert!(
        state
            .replace(&HashMap::from([
                ("a".into(), vector(&[4., 5.])),
                ("b".into(), vector(&[4., 6.]))
            ]))
            .is_err()
    );
    assert!(
        state
            .replace(&HashMap::from([("a".into(), vector(&[4.]))]))
            .is_err()
    );
    assert_eq!(state.revision(), revision);
    assert_eq!(
        state.tensor("a").unwrap().to_vec::<f32>().unwrap(),
        [2., 3.]
    );
    assert_eq!(
        state.tensor("b").unwrap().to_vec::<f32>().unwrap(),
        [2., 3.]
    );
    let original = two_parameters().program().unwrap();
    let original_state = original
        .bind_state(&HashMap::from([
            ("a".into(), vector(&[2., 3.])),
            ("b".into(), vector(&[2., 3.])),
        ]))
        .unwrap();
    let tied = original.tie_state("a", "b").unwrap();
    assert!(
        tied.prepare(&original_state, GraphPrepareOptions::new(Device::Cpu))
            .is_err()
    );
    assert!(
        source()
            .program()
            .unwrap()
            .tie_state("weight", "bias")
            .is_err()
    );
}

#[cfg(feature = "cuda")]
#[test]
fn test_native_cuda_preserves_declared_ties_and_old_state_through_rebinding() {
    exercise_tied_state(Device::Cuda(0));
}

#[test]
fn test_state_replacement_is_atomic_and_old_instances_keep_their_snapshot() {
    let program = source().program().unwrap();
    let state = program.bind_state(&bindings()).unwrap();
    let fingerprint = program.fingerprint().unwrap();
    let mut executor = program
        .prepare(&state, GraphPrepareOptions::new(Device::Cpu))
        .unwrap();
    let mut old = executor.create_instance().unwrap();
    let retained = executor.run(&input()).unwrap();
    let old_key = executor.specialization_key().clone();
    let invalid = state.replace(&HashMap::from([
        ("weight".into(), vector(&[5., 6.])),
        ("bias".into(), vector(&[7.])),
    ]));
    assert!(invalid.is_err());
    assert_eq!(
        state.tensor("weight").unwrap().to_vec::<f32>().unwrap(),
        [2., 3.]
    );
    let replacement = state
        .replace(&HashMap::from([
            ("weight".into(), vector(&[5., 6.])),
            ("bias".into(), vector(&[7., 8.])),
        ]))
        .unwrap();
    assert_ne!(state.revision(), replacement.revision());
    executor.rebind_state(&replacement).unwrap();
    assert_ne!(&old_key, executor.specialization_key());
    assert_eq!(executor.specialization_key().program(), fingerprint);
    assert_eq!(
        executor.run(&input()).unwrap()[0].to_vec::<f32>().unwrap(),
        [17., 32.]
    );
    assert_eq!(
        old.run(&input()).unwrap()[0].to_vec::<f32>().unwrap(),
        [5., 13.]
    );
    assert_eq!(retained[0].to_vec::<f32>().unwrap(), [5., 13.]);
    let foreign = source().program().unwrap().bind_state(&bindings()).unwrap();
    assert!(executor.rebind_state(&foreign).is_err());
    assert_eq!(
        executor.run(&input()).unwrap()[0].to_vec::<f32>().unwrap(),
        [17., 32.]
    );
}

#[test]
fn test_parameter_freezing_is_explicit_and_preserves_the_public_signature() {
    let program = source().program().unwrap();
    let state = program.bind_state(&bindings()).unwrap();
    let dynamic = program
        .prepare(&state, GraphPrepareOptions::new(Device::Cpu))
        .unwrap();
    let mut frozen = program
        .prepare(
            &state,
            GraphPrepareOptions::new(Device::Cpu).with_freeze_policy(StateFreezePolicy::Parameters),
        )
        .unwrap();
    assert_eq!(
        dynamic.specialization_key().program(),
        frozen.specialization_key().program()
    );
    assert_ne!(dynamic.specialization_key(), frozen.specialization_key());
    assert!(
        dynamic
            .analysis()
            .ir()
            .values()
            .iter()
            .any(|value| value.binding().input_name() == Some("weight"))
    );
    assert!(
        !frozen
            .analysis()
            .ir()
            .values()
            .iter()
            .any(|value| value.binding().input_name() == Some("weight"))
    );
    assert_eq!(frozen.analysis().signature().inputs().len(), 3);
    let mut override_input = input();
    override_input.insert("weight".into(), vector(&[9., 9.]));
    assert!(frozen.run(&override_input).is_err());
    assert_eq!(
        frozen.run(&input()).unwrap()[0].to_vec::<f32>().unwrap(),
        [5., 13.]
    );
    let new = state
        .replace(&HashMap::from([("weight".into(), vector(&[5., 6.]))]))
        .unwrap();
    frozen.rebind_state(&new).unwrap();
    assert_eq!(
        frozen.run(&input()).unwrap()[0].to_vec::<f32>().unwrap(),
        [11., 25.]
    );
}

#[test]
fn test_program_identity_excludes_state_values_and_allocations_but_includes_roles() {
    let first = source().program().unwrap();
    let mut changed = source();
    changed
        .add_parameter("weight", vector(&[99., 100.]))
        .unwrap();
    let second = changed.program().unwrap();
    assert_eq!(first.fingerprint().unwrap(), second.fingerprint().unwrap());
    let weight = first
        .state_schema()
        .find(|binding| binding.name() == "weight")
        .unwrap();
    assert_eq!(weight.kind(), StateBindingKind::Parameter);
    assert_eq!(
        first.signature().inputs()[1].binding(),
        ProgramInputBinding::State(StateBindingKind::Parameter)
    );
    changed.add_buffer("weight", vector(&[99., 100.])).unwrap();
    assert_ne!(
        first.fingerprint().unwrap(),
        changed.program().unwrap().fingerprint().unwrap()
    );
}

#[test]
fn test_concurrent_publications_have_distinct_revision_keys() {
    let program = source().program().unwrap();
    let state = program.bind_state(&bindings()).unwrap();
    let left = state.clone();
    let right = state.clone();
    let left = thread::spawn(move || {
        left.replace(&HashMap::from([("weight".into(), vector(&[4., 5.]))]))
            .unwrap()
    });
    let right = thread::spawn(move || {
        right
            .replace(&HashMap::from([("weight".into(), vector(&[6., 7.]))]))
            .unwrap()
    });
    let left = left.join().unwrap();
    let right = right.join().unwrap();
    assert_eq!(left.revision().identity(), right.revision().identity());
    assert_ne!(left.revision(), right.revision());
    assert_eq!(state.revision().revision(), 0);
}

#[test]
fn test_defaults_remain_overridable_when_all_non_default_state_is_frozen() {
    let mut graph = source();
    graph.add_initializer("bias", vector(&[1., 1.])).unwrap();
    let program = graph.program().unwrap();
    let state = program.bind_state(&bindings()).unwrap();
    let mut executor = program
        .prepare(
            &state,
            GraphPrepareOptions::new(Device::Cpu).with_freeze_policy(StateFreezePolicy::AllState),
        )
        .unwrap();
    let mut inputs = input();
    inputs.insert("bias".into(), vector(&[7., 8.]));
    assert_eq!(
        executor.run(&inputs).unwrap()[0].to_vec::<f32>().unwrap(),
        [11., 20.]
    );
}

#[cfg(feature = "cuda")]
#[test]
fn test_cpu_and_cuda_share_logical_identity_and_rebind_distinct_specializations() {
    let program = source().program().unwrap();
    let state = program.bind_state(&bindings()).unwrap();
    let cpu = program
        .prepare(&state, GraphPrepareOptions::new(Device::Cpu))
        .unwrap();
    let mut cuda = program
        .prepare(&state, GraphPrepareOptions::new(Device::Cuda(0)))
        .unwrap();
    let mut previous = cuda.create_instance().unwrap();
    assert_eq!(
        cpu.specialization_key().program(),
        cuda.specialization_key().program()
    );
    assert_ne!(cpu.specialization_key(), cuda.specialization_key());
    assert!(
        cuda.specialization_key()
            .native_context_identity()
            .is_some()
    );
    assert_eq!(cuda.specialization_key(), previous.specialization_key());
    let inputs = HashMap::from([(
        "x".into(),
        vector(&[2., 4.]).to_device(Device::Cuda(0)).unwrap(),
    )]);
    let retained = cuda.run(&inputs).unwrap();
    let old_key = cuda.specialization_key().clone();
    let replacement = state
        .replace(&HashMap::from([("weight".into(), vector(&[5., 6.]))]))
        .unwrap();
    cuda.rebind_state(&replacement).unwrap();
    assert_ne!(&old_key, cuda.specialization_key());
    assert_eq!(
        cuda.run(&inputs).unwrap()[0].to_vec::<f32>().unwrap(),
        [11., 25.]
    );
    assert_eq!(
        previous.run(&inputs).unwrap()[0].to_vec::<f32>().unwrap(),
        [5., 13.]
    );
    assert_eq!(retained[0].to_vec::<f32>().unwrap(), [5., 13.]);
    let mut frozen = program
        .prepare(
            &state,
            GraphPrepareOptions::new(Device::Cuda(0))
                .with_freeze_policy(StateFreezePolicy::Parameters),
        )
        .unwrap();
    assert_eq!(
        frozen.run(&inputs).unwrap()[0].to_vec::<f32>().unwrap(),
        [5., 13.]
    );
    frozen.rebind_state(&replacement).unwrap();
    assert_eq!(
        frozen.run(&inputs).unwrap()[0].to_vec::<f32>().unwrap(),
        [11., 25.]
    );
}

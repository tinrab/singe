use std::collections::HashMap;

use crate::{
    backend::registry::Operation,
    data_type::DataType,
    device::Device,
    graph::{Graph, GraphCompileOptions, GraphPrepareOptions, GraphProgram, capture::constant},
    shape::Shape,
    tensor::Tensor,
};

fn tensor(dimensions: &[i64], seed: u32) -> Tensor {
    let count = usize::try_from(dimensions.iter().product::<i64>()).unwrap();
    let mut state = seed;
    let values = (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            f32::from(u16::try_from(state >> 16).unwrap()) / 32_768.0 - 1.0
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(
        Device::Cpu,
        Shape::create(dimensions.to_vec()).unwrap(),
        values,
    )
    .unwrap()
}

fn matmuls(graph: &Graph, merged: bool) -> usize {
    let compiled = graph
        .compile_with_options(GraphCompileOptions::new().with_merged_projections(merged))
        .unwrap();
    compiled
        .ir()
        .operations()
        .iter()
        .filter(|operation| operation.opcode() == Some(Operation::MatMul))
        .count()
}

fn run(graph: &Graph, inputs: &HashMap<String, Tensor>, merged: bool) -> Vec<Vec<f32>> {
    let mut prepared = graph
        .prepare(
            GraphPrepareOptions::new(Device::Cpu)
                .with_compile_options(GraphCompileOptions::new().with_merged_projections(merged)),
        )
        .unwrap();
    prepared
        .run(inputs)
        .unwrap()
        .iter()
        .map(|output| output.cast(DataType::F32).unwrap().to_vec::<f32>().unwrap())
        .collect()
}

fn assert_close(actual: &[Vec<f32>], expected: &[Vec<f32>]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-4, "{actual} != {expected}");
        }
    }
}

// Linear layers over one input read their weights as `input.matmul(weight.transpose(0, 1))`.
fn projections(input: &Tensor, weights: &[&Tensor]) -> Vec<Tensor> {
    weights
        .iter()
        .map(|weight| input.matmul(&weight.transpose(0, 1).unwrap()).unwrap())
        .collect()
}

#[test]
fn test_sibling_projections_merge_and_compute_what_they_did() {
    let inputs = vec![("x", tensor(&[2, 3, 8], 1))];
    let (first, second, third) = (tensor(&[4, 8], 2), tensor(&[6, 8], 3), tensor(&[5, 8], 4));
    let (graph, _) = Graph::capture(inputs.clone(), |inputs| {
        let weights = [
            constant("first", &first)?,
            constant("second", &second)?,
            constant("third", &third)?,
        ];
        let [a, b, c] = projections(&inputs[0], &[&weights[0], &weights[1], &weights[2]])
            .try_into()
            .unwrap();
        // One sibling's result feeds another step, and each result stays an output.
        Ok(vec![a.clone(), b, c, a.relu()?])
    })
    .unwrap();
    assert_eq!(matmuls(&graph, false), 3);
    assert_eq!(matmuls(&graph, true), 1);

    let named = HashMap::from([("x".to_owned(), inputs[0].1.clone())]);
    assert_close(&run(&graph, &named, true), &run(&graph, &named, false));
}

// What differs in type or depth, or reads another left side, is not a sibling.
#[test]
fn test_only_matmuls_of_one_left_side_type_and_depth_merge() {
    let inputs = vec![
        ("x", tensor(&[3, 8], 1)),
        ("y", tensor(&[3, 8], 2)),
        ("deep", tensor(&[3, 6], 3)),
    ];
    let (same, half, other, deep) = (
        tensor(&[4, 8], 4),
        tensor(&[4, 8], 5).cast(DataType::F16).unwrap(),
        tensor(&[4, 8], 6),
        tensor(&[4, 6], 7),
    );
    let (graph, _) = Graph::capture(inputs, |inputs| {
        let weight = |name: &str, tensor: &Tensor| constant(name, tensor)?.transpose(0, 1);
        Ok(vec![
            inputs[0].matmul(&weight("same", &same)?)?,
            inputs[0]
                .cast(DataType::F16)?
                .matmul(&weight("half", &half)?)?,
            inputs[1].matmul(&weight("other", &other)?)?,
            inputs[2].matmul(&weight("deep_weight", &deep)?)?,
        ])
    })
    .unwrap();
    assert_eq!(matmuls(&graph, true), 4);
}

// Preparing with other weights rebuilds what they were joined into, so changed weights are never stale.
#[test]
fn test_merged_projections_follow_weights_that_change() {
    let left = tensor(&[3, 8], 1);
    let (first_weight, second_weight) = (tensor(&[4, 8], 2), tensor(&[6, 8], 3));
    let inputs = vec![
        ("x", left.clone()),
        ("first", first_weight.clone()),
        ("second", second_weight.clone()),
    ];
    let (mut source, _) = Graph::capture(inputs, |inputs| {
        Ok(projections(&inputs[0], &[&inputs[1], &inputs[2]]))
    })
    .unwrap();
    source
        .add_static_initializer("first", first_weight.clone())
        .unwrap();
    source
        .add_static_initializer("second", second_weight.clone())
        .unwrap();
    assert_eq!(matmuls(&source, true), 1);

    let program = GraphProgram::create(&source).unwrap();
    let options = GraphPrepareOptions::new(Device::Cpu)
        .with_compile_options(GraphCompileOptions::new().with_merged_projections(true));
    let bind = |first: Tensor, second: Tensor| {
        program
            .bind_state(&HashMap::from([
                ("first".to_owned(), first),
                ("second".to_owned(), second),
            ]))
            .unwrap()
    };
    let (new_first, new_second) = (tensor(&[4, 8], 12), tensor(&[6, 8], 13));
    let mut prepared = program
        .prepare(&bind(first_weight, second_weight), options)
        .unwrap();
    prepared
        .rebind_state(&bind(new_first.clone(), new_second.clone()))
        .unwrap();
    let outputs = prepared
        .run(&HashMap::from([("x".to_owned(), left.clone())]))
        .unwrap();
    let host = |tensors: &[Tensor]| {
        tensors
            .iter()
            .map(|output| output.to_vec::<f32>().unwrap())
            .collect::<Vec<_>>()
    };
    assert_close(
        &host(&outputs),
        &host(&projections(&left, &[&new_first, &new_second])),
    );

    // A run cannot replace static state, merged or not.
    assert!(
        prepared
            .run(&HashMap::from([
                ("x".to_owned(), left),
                ("first".to_owned(), tensor(&[4, 8], 22)),
            ]))
            .is_err()
    );
}

// A weight a run may replace stays apart, since a joined copy would go on using the old one.
#[test]
fn test_weights_a_run_may_replace_do_not_merge() {
    let left = tensor(&[3, 8], 1);
    let (first, second) = (tensor(&[4, 8], 2), tensor(&[6, 8], 3));
    let inputs = vec![
        ("x", left.clone()),
        ("first", first.clone()),
        ("second", second.clone()),
    ];
    let (mut source, _) = Graph::capture(inputs, |inputs| {
        Ok(projections(&inputs[0], &[&inputs[1], &inputs[2]]))
    })
    .unwrap();
    source.add_initializer("first", first).unwrap();
    source.add_initializer("second", second.clone()).unwrap();
    assert_eq!(matmuls(&source, true), 2);

    let replaced = tensor(&[4, 8], 22);
    let mut prepared = source
        .prepare(
            GraphPrepareOptions::new(Device::Cpu)
                .with_compile_options(GraphCompileOptions::new().with_merged_projections(true)),
        )
        .unwrap();
    let outputs = prepared
        .run(&HashMap::from([
            ("x".to_owned(), left.clone()),
            ("first".to_owned(), replaced.clone()),
        ]))
        .unwrap();
    assert_close(
        &[outputs[0].to_vec::<f32>().unwrap()],
        &[projections(&left, &[&replaced, &second])[0]
            .to_vec::<f32>()
            .unwrap()],
    );
}

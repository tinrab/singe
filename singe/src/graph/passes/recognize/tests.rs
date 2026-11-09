use std::collections::HashMap;

use singe_core::assert_close;

use crate::{
    backend::registry::Operation,
    device::Device,
    error::Result,
    graph::{Graph, GraphPrepareOptions, ir::OperationNode, test_support::patterned_tensor},
    tensor::Tensor,
};

fn values(dimensions: &[i64], seed: usize) -> Tensor {
    patterned_tensor(Device::Cpu, dimensions, seed)
}

// Captures `function` of one `[3, 8]` input, then checks that the compiled graph runs `expected` and none of `absent`, and gives the eager result.
fn assert_recognized(
    function: impl Fn(&Tensor) -> Result<Tensor>,
    expected: Operation,
    absent: &[Operation],
) {
    let input = values(&[3, 8], 1);
    let (graph, _) = Graph::capture([("x", input.clone())], |inputs| {
        Ok(vec![function(&inputs[0])?])
    })
    .unwrap();
    let compiled = graph.compile().unwrap();
    let operations = compiled
        .ir()
        .operations()
        .iter()
        .filter_map(OperationNode::opcode)
        .collect::<Vec<_>>();
    assert!(operations.contains(&expected), "{operations:?}");
    for operation in absent {
        assert!(!operations.contains(operation), "{operations:?}");
    }

    let expected = function(&input).unwrap().to_vec::<f32>().unwrap();
    let mut prepared = graph
        .prepare(GraphPrepareOptions::new(Device::Cpu))
        .unwrap();
    let inputs = HashMap::from([("x".to_owned(), input)]);
    let actual = prepared.run(&inputs).unwrap()[0].to_vec::<f32>().unwrap();
    assert_close!(&actual, &expected, 1e-5);
}

#[test]
fn test_rms_norm_spellings_become_rms_norm() {
    let variance = |x: &Tensor| x.mul(x)?.mean(&[-1], true)?.add_scalar(1e-6);
    assert_recognized(
        |x| x.mul(&variance(x)?.rsqrt()?),
        Operation::RmsNorm,
        &[Operation::Mean, Operation::Rsqrt],
    );
    assert_recognized(
        |x| x.div(&variance(x)?.sqrt()?),
        Operation::RmsNorm,
        &[Operation::Mean, Operation::Sqrt],
    );
    assert_recognized(
        |x| {
            let squared = x.pow_scalar(2.0)?.mean(&[1], true)?.add_scalar(1e-5)?;
            squared.sqrt()?.reciprocal()?.mul(x)
        },
        Operation::RmsNorm,
        &[Operation::Mean, Operation::Reciprocal],
    );
}

#[test]
fn test_centered_rms_norm_becomes_layer_norm() {
    assert_recognized(
        |x| {
            let centered = x.sub(&x.mean(&[-1], true)?)?;
            let variance = centered.mul(&centered)?.mean(&[-1], true)?;
            centered.mul(&variance.add_scalar(1e-5)?.rsqrt()?)
        },
        Operation::LayerNorm,
        &[Operation::Mean, Operation::RmsNorm],
    );
}

#[test]
fn test_softmax_spellings_become_softmax_and_log_softmax() {
    let shifted = |x: &Tensor| x.sub(&x.max_reduce(&[-1], true)?);
    assert_recognized(
        |x| {
            let exponentials = shifted(x)?.exp()?;
            exponentials.div(&exponentials.sum(&[-1], true)?)
        },
        Operation::Softmax,
        &[Operation::Exp, Operation::Max],
    );
    assert_recognized(
        |x| x.exp()?.div(&x.exp()?.sum(&[1], true)?),
        Operation::Softmax,
        &[Operation::Sum],
    );
    assert_recognized(
        |x| {
            let shifted = shifted(x)?;
            shifted.sub(&shifted.exp()?.sum(&[-1], true)?.log()?)
        },
        Operation::LogSoftmax,
        &[Operation::Exp, Operation::Log],
    );
}

#[test]
fn test_near_misses_stay_primitive() {
    // A mean over the first axis is not a row normalization, and a sum that drops its axis is not a softmax denominator.
    let input = values(&[3, 8], 2);
    let (graph, _) = Graph::capture([("x", input)], |inputs| {
        let x = &inputs[0];
        let columns = x.mul(&x.mul(x)?.mean(&[0], true)?.add_scalar(1e-6)?.rsqrt()?)?;
        let dropped = x.exp()?.sum(&[-1], false)?.unsqueeze(-1)?;
        Ok(vec![columns, x.exp()?.div(&dropped)?])
    })
    .unwrap();
    let compiled = graph.compile().unwrap();
    let operations = compiled
        .ir()
        .operations()
        .iter()
        .filter_map(OperationNode::opcode)
        .collect::<Vec<_>>();
    assert!(!operations.contains(&Operation::RmsNorm), "{operations:?}");
    assert!(!operations.contains(&Operation::Softmax), "{operations:?}");
}

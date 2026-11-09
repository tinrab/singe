use std::collections::HashMap;

use singe_core::assert_close;

#[cfg(feature = "cuda")]
use crate::shape::Shape;
use crate::{
    backend::registry::Operation,
    device::Device,
    error::Result,
    graph::{Graph, GraphPrepareOptions, ir::OperationNode, test_support::patterned_tensor},
    tensor::Tensor,
};
#[cfg(feature = "cuda")]
use crate::{
    data_type::DataType,
    testing::{maybe_lock_device, skip_if_stub},
};

/// Returns a tensor of `dimensions` on `device` with small varied values.
/// Attention of `q` over `k` and `v` whose KV heads repeat for the query heads: `softmax(q kᵀ) v`.
fn attention(inputs: &[Tensor], groups: i64) -> Result<Vec<Tensor>> {
    let keys = inputs[1].repeat_kv(groups)?;
    let values = inputs[2].repeat_kv(groups)?;
    let scores = inputs[0].matmul(&keys.transpose(2, 3)?)?;
    Ok(vec![scores.softmax(-1)?.matmul(&values)?])
}

fn captured(device: Device, query: &[i64], cache: &[i64]) -> (Graph, HashMap<String, Tensor>) {
    let inputs = [
        ("q", patterned_tensor(device, query, 1)),
        ("k", patterned_tensor(device, cache, 2)),
        ("v", patterned_tensor(device, cache, 3)),
    ];
    let groups = query[1] / cache[1];
    let named = inputs
        .iter()
        .map(|(name, tensor)| ((*name).to_owned(), tensor.clone()))
        .collect();
    let (graph, _) = Graph::capture(inputs, |inputs| attention(inputs, groups)).unwrap();
    (graph, named)
}

fn operations(graph: &Graph) -> Vec<Operation> {
    graph
        .compile()
        .unwrap()
        .ir()
        .operations()
        .iter()
        .filter_map(OperationNode::opcode)
        .collect()
}

#[test]
fn test_grouped_query_attention_folds_repeated_heads_into_matmul_rows() {
    let (graph, inputs) = captured(Device::Cpu, &[2, 4, 3, 8], &[2, 2, 5, 8]);
    let compiled = operations(&graph);
    assert!(!compiled.contains(&Operation::RepeatKv), "{compiled:?}");
    assert_eq!(
        compiled
            .iter()
            .filter(|&&operation| operation == Operation::MatMul)
            .count(),
        2
    );

    let groups = 2;
    let ordered = [&inputs["q"], &inputs["k"], &inputs["v"]].map(Clone::clone);
    let expected = attention(&ordered, groups).unwrap()[0]
        .to_vec::<f32>()
        .unwrap();
    let mut prepared = graph
        .prepare(GraphPrepareOptions::new(Device::Cpu))
        .unwrap();
    let actual = prepared.run(&inputs).unwrap()[0].to_vec::<f32>().unwrap();
    assert_close!(&actual, &expected, 1e-5);
}

#[test]
fn test_grouped_query_attention_keeps_repeat_kv_that_other_operations_read() {
    let device = Device::Cpu;
    let inputs = [
        ("q", patterned_tensor(device, &[1, 4, 2, 8], 1)),
        ("k", patterned_tensor(device, &[1, 2, 6, 8], 2)),
    ];
    let (graph, _) = Graph::capture(inputs, |inputs| {
        let keys = inputs[1].repeat_kv(2)?;
        let scores = inputs[0].matmul(&keys.transpose(2, 3)?)?;
        Ok(vec![scores, keys])
    })
    .unwrap();
    let compiled = operations(&graph);
    assert_eq!(
        compiled
            .iter()
            .filter(|&&operation| operation == Operation::RepeatKv)
            .count(),
        1,
        "{compiled:?}"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn test_grouped_query_attention_folds_on_cuda_at_decode_shapes() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let Some(probe) = skip_if_stub(Tensor::zeroed_on(
        device,
        DataType::F32,
        Shape::create([1]).unwrap(),
    )) else {
        return;
    };
    drop(probe);
    let (graph, inputs) = captured(device, &[4, 16, 1, 128], &[4, 8, 513, 128]);
    let ordered = [&inputs["q"], &inputs["k"], &inputs["v"]].map(Clone::clone);
    let expected = attention(&ordered, 2).unwrap()[0].to_vec::<f32>().unwrap();
    let mut prepared = graph.prepare(GraphPrepareOptions::new(device)).unwrap();
    let actual = prepared.run(&inputs).unwrap()[0].to_vec::<f32>().unwrap();
    assert_close!(&actual, &expected, 1e-3);
}

/// Returns the attention of `values`-filled `q`, `k`, and `v` on `device`, from a compiled graph and from an eager call, as `F32` values.
fn scaled_attention(device: Device, causal: bool) -> (Vec<Operation>, Vec<f32>, Vec<f32>) {
    let inputs = [
        ("q", patterned_tensor(device, &[2, 3, 5, 8], 1)),
        ("k", patterned_tensor(device, &[2, 3, 7, 8], 2)),
        ("v", patterned_tensor(device, &[2, 3, 7, 8], 3)),
    ];
    let named = inputs
        .iter()
        .map(|(name, tensor)| ((*name).to_owned(), tensor.clone()))
        .collect::<HashMap<_, _>>();
    let eager = inputs[0]
        .1
        .scaled_dot_product_attention(&inputs[1].1, &inputs[2].1, None, causal)
        .unwrap();
    let (graph, _) = Graph::capture(inputs, |inputs| {
        Ok(vec![inputs[0].scaled_dot_product_attention(
            &inputs[1], &inputs[2], None, causal,
        )?])
    })
    .unwrap();
    let mut prepared = graph.prepare(GraphPrepareOptions::new(device)).unwrap();
    let compiled = prepared.run(&named).unwrap()[0].to_vec::<f32>().unwrap();
    (operations(&graph), compiled, eager.to_vec::<f32>().unwrap())
}

#[test]
fn test_attention_decomposes_into_matmuls_and_a_softmax() {
    for causal in [false, true] {
        let (operations, compiled, native) = scaled_attention(Device::Cpu, causal);
        assert!(!operations.contains(&Operation::ScaledDotProductAttention));
        assert!(operations.contains(&Operation::Softmax));
        assert_close!(&compiled, &native, 1e-5);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_attention_runs_on_cuda_eagerly_and_compiled() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let Some(probe) = skip_if_stub(Tensor::zeroed_on(
        Device::Cuda(0),
        DataType::F32,
        Shape::create([1]).unwrap(),
    )) else {
        return;
    };
    drop(probe);
    for causal in [false, true] {
        let (_, _, reference) = scaled_attention(Device::Cpu, causal);
        let (_, compiled, eager) = scaled_attention(Device::Cuda(0), causal);
        assert_close!(&eager, &reference, 1e-4);
        assert_close!(&compiled, &reference, 1e-4);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_half_precision_attention_stays_whole_and_runs_fused_in_compiled_graphs() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let Some(probe) = skip_if_stub(Tensor::zeroed_on(
        device,
        DataType::F32,
        Shape::create([1]).unwrap(),
    )) else {
        return;
    };
    drop(probe);
    let half = |dimensions: &[i64], seed| {
        patterned_tensor(device, dimensions, seed)
            .cast(DataType::F16)
            .unwrap()
    };
    let inputs = [
        ("q", half(&[1, 2, 40, 64], 1)),
        ("k", half(&[1, 2, 40, 64], 2)),
        ("v", half(&[1, 2, 40, 64], 3)),
    ];
    let named = inputs
        .iter()
        .map(|(name, tensor)| ((*name).to_owned(), tensor.clone()))
        .collect::<HashMap<_, _>>();
    let wide = |tensor: &Tensor| tensor.cast(DataType::F32).unwrap();
    let expected = wide(&inputs[0].1)
        .scaled_dot_product_attention(&wide(&inputs[1].1), &wide(&inputs[2].1), None, true)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    let (graph, _) = Graph::capture(inputs, |inputs| {
        Ok(vec![inputs[0].scaled_dot_product_attention(
            &inputs[1], &inputs[2], None, true,
        )?])
    })
    .unwrap();
    assert_eq!(operations(&graph), [Operation::ScaledDotProductAttention]);
    let mut prepared = graph.prepare(GraphPrepareOptions::new(device)).unwrap();
    let actual = wide(&prepared.run(&named).unwrap()[0])
        .to_vec::<f32>()
        .unwrap();
    assert_close!(&actual, &expected, 5e-3);
}

#[test]
fn test_grouped_key_heads_attend_as_their_repeats_do() {
    // Grouped-query attention with fewer key heads equals attention over the repeated heads, eagerly and in compiled `F32` graphs, which decompose it.
    let device = Device::Cpu;
    let inputs = [
        ("q", patterned_tensor(device, &[1, 4, 6, 8], 1)),
        ("k", patterned_tensor(device, &[1, 2, 9, 8], 2)),
        ("v", patterned_tensor(device, &[1, 2, 9, 8], 3)),
    ];
    let named = inputs
        .iter()
        .map(|(name, tensor)| ((*name).to_owned(), tensor.clone()))
        .collect::<HashMap<_, _>>();
    let expected = inputs[0]
        .1
        .scaled_dot_product_attention(
            &inputs[1].1.repeat_kv(2).unwrap(),
            &inputs[2].1.repeat_kv(2).unwrap(),
            None,
            true,
        )
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    let eager = inputs[0]
        .1
        .scaled_dot_product_attention(&inputs[1].1, &inputs[2].1, None, true)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    assert_close!(&eager, &expected, 1e-5);
    let (graph, _) = Graph::capture(inputs, |inputs| {
        Ok(vec![inputs[0].scaled_dot_product_attention(
            &inputs[1], &inputs[2], None, true,
        )?])
    })
    .unwrap();
    let mut prepared = graph.prepare(GraphPrepareOptions::new(device)).unwrap();
    let compiled = prepared.run(&named).unwrap()[0].to_vec::<f32>().unwrap();
    assert_close!(&compiled, &expected, 1e-5);
}

#[cfg(feature = "cuda")]
#[test]
fn test_compiled_graphs_fold_repeated_heads_into_fused_attention() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let Some(probe) = skip_if_stub(Tensor::zeroed_on(
        device,
        DataType::F32,
        Shape::create([1]).unwrap(),
    )) else {
        return;
    };
    drop(probe);
    let half = |dimensions: &[i64], seed| {
        patterned_tensor(device, dimensions, seed)
            .cast(DataType::F16)
            .unwrap()
    };
    let inputs = [
        ("q", half(&[1, 8, 3, 64], 1)),
        ("k", half(&[1, 2, 50, 64], 2)),
        ("v", half(&[1, 2, 50, 64], 3)),
    ];
    let named = inputs
        .iter()
        .map(|(name, tensor)| ((*name).to_owned(), tensor.clone()))
        .collect::<HashMap<_, _>>();
    let attend = |inputs: &[Tensor]| -> Result<Vec<Tensor>> {
        Ok(vec![inputs[0].scaled_dot_product_attention(
            &inputs[1].repeat_kv(4)?,
            &inputs[2].repeat_kv(4)?,
            None,
            false,
        )?])
    };
    let wide = |tensor: &Tensor| tensor.cast(DataType::F32).unwrap();
    let expected = attend(
        &inputs
            .iter()
            .map(|(_, tensor)| wide(tensor))
            .collect::<Vec<_>>(),
    )
    .unwrap()[0]
        .to_vec::<f32>()
        .unwrap();
    let (graph, _) = Graph::capture(inputs, attend).unwrap();
    assert_eq!(operations(&graph), [Operation::ScaledDotProductAttention]);
    let mut prepared = graph.prepare(GraphPrepareOptions::new(device)).unwrap();
    let actual = wide(&prepared.run(&named).unwrap()[0])
        .to_vec::<f32>()
        .unwrap();
    assert_close!(&actual, &expected, 5e-3);
}

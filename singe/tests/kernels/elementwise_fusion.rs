#![cfg(all(feature = "testing", feature = "cuda"))]

use std::collections::HashMap;

use singe::{
    Device, Shape, Tensor,
    backend::cuda::{self, CudaGraphProvider, KernelLanguage, LoweringPolicy},
    error::Result,
    graph::{Graph, GraphCudaOptions, GraphPrepareOptions},
};
use singe_cuda::testing::device_lock;

fn values(count: usize, scale: f32) -> Vec<f32> {
    (0..count)
        .map(|value| (value as f32).mul_add(scale, -1.5))
        .collect()
}

fn chain(x: &Tensor, y: &Tensor) -> Result<Tensor> {
    x.mul_scalar(2.0)?
        .add(y)?
        .relu()?
        .sub_scalar(0.5)?
        .sigmoid()
}

#[test]
fn test_element_wise_chains_compile_to_one_kernel_and_match_eager_results() {
    let _lock = device_lock(0).unwrap();
    let device = Device::Cuda(0);
    let x = Tensor::from_vec(device, Shape::create([4, 33]).unwrap(), values(132, 0.03)).unwrap();
    // `y` broadcasts along the rows inside the fused kernel.
    let y = Tensor::from_vec(device, Shape::create([33]).unwrap(), values(33, -0.07)).unwrap();
    let expected = chain(&x, &y).unwrap();
    let (captured, _) = Graph::capture([("x", x.clone()), ("y", y.clone())], |inputs| {
        Ok(vec![chain(&inputs[0], &inputs[1])?])
    })
    .unwrap();

    let mut compiled = cuda::compile_graph(&captured, &GraphCudaOptions::default()).unwrap();
    assert_eq!(compiled.stats().kernel_child_graph_nodes(), 1);
    let inputs = HashMap::from([("x".to_owned(), x), ("y".to_owned(), y)]);
    let actual = compiled.run(&inputs).unwrap();
    let close = actual[0]
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .zip(expected.to_vec::<f32>().unwrap())
        .all(|(actual, expected)| (actual - expected).abs() <= 1e-6 * expected.abs().max(1.0));
    assert!(close);

    let mut prepared = captured.prepare(GraphPrepareOptions::new(device)).unwrap();
    assert_eq!(
        prepared.run(&inputs).unwrap()[0].to_vec::<f32>().unwrap(),
        actual[0].to_vec::<f32>().unwrap()
    );
}

#[test]
fn test_fusion_groups_store_every_value_read_outside_the_group() {
    let _lock = device_lock(0).unwrap();
    let device = Device::Cuda(0);
    let x = Tensor::from_vec(device, Shape::create([64]).unwrap(), values(64, 0.05)).unwrap();
    let (captured, _) = Graph::capture([("x", x.clone())], |inputs| {
        let shared = inputs[0].exp()?;
        // `shared` has two readers, which join it in one kernel that stores both outputs.
        Ok(vec![shared.relu()?.neg()?, shared.sqrt()?.abs()?])
    })
    .unwrap();
    let mut compiled = cuda::compile_graph(&captured, &GraphCudaOptions::default()).unwrap();
    assert_eq!(compiled.stats().kernel_child_graph_nodes(), 1);
    let outputs = compiled
        .run(&HashMap::from([("x".to_owned(), x.clone())]))
        .unwrap();
    let shared = x.exp().unwrap();
    for (output, expected) in outputs.iter().zip([
        shared.relu().unwrap().neg().unwrap(),
        shared.sqrt().unwrap().abs().unwrap(),
    ]) {
        assert_eq!(
            output.to_vec::<f32>().unwrap(),
            expected.to_vec::<f32>().unwrap()
        );
    }
}

#[test]
fn test_policies_without_the_fusion_provider_lower_one_kernel_per_operation() {
    let _lock = device_lock(0).unwrap();
    let device = Device::Cuda(0);
    let x = Tensor::from_vec(device, Shape::create([4, 33]).unwrap(), values(132, 0.03)).unwrap();
    let y = Tensor::from_vec(device, Shape::create([33]).unwrap(), values(33, -0.07)).unwrap();
    let (captured, _) = Graph::capture([("x", x.clone()), ("y", y.clone())], |inputs| {
        Ok(vec![chain(&inputs[0], &inputs[1])?])
    })
    .unwrap();
    let policy = LoweringPolicy::new(vec![CudaGraphProvider::Alias, CudaGraphProvider::Kernel]);
    let mut compiled = cuda::compile_graph(
        &captured,
        &GraphCudaOptions::new().with_lowering_policy(policy),
    )
    .unwrap();
    assert_eq!(compiled.stats().kernel_child_graph_nodes(), 5);
    let actual = compiled
        .run(&HashMap::from([
            ("x".to_owned(), x.clone()),
            ("y".to_owned(), y.clone()),
        ]))
        .unwrap();
    assert_eq!(
        actual[0].to_vec::<f32>().unwrap(),
        chain(&x, &y).unwrap().to_vec::<f32>().unwrap()
    );
}

/// Two chains that meet in one operation: the reader fuses with the chain lowered last, whose sibling is already available.
#[test]
fn test_chains_that_meet_fuse_once_their_other_operands_are_lowered() {
    let _lock = device_lock(0).unwrap();
    let device = Device::Cuda(0);
    let x = Tensor::from_vec(device, Shape::create([3, 8]).unwrap(), values(24, 0.11)).unwrap();
    let y = Tensor::from_vec(device, Shape::create([3, 8]).unwrap(), values(24, -0.05)).unwrap();
    let meet = |x: &Tensor, y: &Tensor| x.mul(x)?.sub(&y.mul(y)?.add_scalar(1.0)?);
    let expected = meet(&x, &y).unwrap().to_vec::<f32>().unwrap();
    let (captured, _) = Graph::capture([("x", x.clone()), ("y", y.clone())], |inputs| {
        Ok(vec![meet(&inputs[0], &inputs[1])?])
    })
    .unwrap();
    let mut compiled = cuda::compile_graph(&captured, &GraphCudaOptions::default()).unwrap();
    let inputs = HashMap::from([("x".to_owned(), x), ("y".to_owned(), y)]);
    let actual = compiled.run(&inputs).unwrap()[0].to_vec::<f32>().unwrap();
    for (actual, expected) in actual.iter().zip(&expected) {
        assert!((actual - expected).abs() <= 1e-6 * expected.abs().max(1.0));
    }
}

#[test]
fn test_branching_element_wise_groups_compile_to_one_kernel() {
    let _lock = device_lock(0).unwrap();
    let device = Device::Cuda(0);
    let tensor = |seed| {
        Tensor::from_vec(device, Shape::create([4, 33]).unwrap(), values(132, seed)).unwrap()
    };
    let (a, b, c, d) = (tensor(0.01), tensor(-0.02), tensor(0.03), tensor(0.05));
    // The rotary embedding's `first × cos - second × sin`: two products of different inputs meet in one difference.
    let function = |inputs: &[Tensor]| -> Result<Tensor> {
        inputs[0]
            .mul(&inputs[1])?
            .sub(&inputs[2].mul(&inputs[3])?)?
            .mul_scalar(0.5)
    };
    let named = [("a", a), ("b", b), ("c", c), ("d", d)];
    let expected = function(
        &named
            .iter()
            .map(|(_, tensor)| tensor.clone())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let (captured, _) =
        Graph::capture(named.clone(), |inputs| Ok(vec![function(inputs)?])).unwrap();
    let mut compiled = cuda::compile_graph(&captured, &GraphCudaOptions::default()).unwrap();
    assert_eq!(compiled.stats().kernel_child_graph_nodes(), 1);
    let inputs = named
        .into_iter()
        .map(|(name, tensor)| (name.to_owned(), tensor))
        .collect::<HashMap<_, _>>();
    let actual = compiled.run(&inputs).unwrap();
    assert_eq!(
        actual[0].to_vec::<f32>().unwrap(),
        expected.to_vec::<f32>().unwrap()
    );
}

#[test]
fn test_groups_that_read_two_views_of_one_value_depend_on_its_producer_once() {
    let _lock = device_lock(0).unwrap();
    let device = Device::Cuda(0);
    let x = Tensor::from_vec(device, Shape::create([4, 16]).unwrap(), values(64, 0.02)).unwrap();
    // Both halves of `exp(x)` are views of one graph node, as rotary embedding reads both halves of a head.
    let function = |x: &Tensor| -> Result<Tensor> {
        let y = x.exp()?;
        y.narrow(1, 0, 8)?.mul(&y.narrow(1, 8, 8)?)?.mul_scalar(2.0)
    };
    let expected = function(&x).unwrap();
    let (captured, _) =
        Graph::capture([("x", x.clone())], |inputs| Ok(vec![function(&inputs[0])?])).unwrap();
    let mut compiled = cuda::compile_graph(&captured, &GraphCudaOptions::default()).unwrap();
    let actual = compiled.run(&HashMap::from([("x".to_owned(), x)])).unwrap();
    assert_eq!(
        actual[0].to_vec::<f32>().unwrap(),
        expected.to_vec::<f32>().unwrap()
    );
}

#[test]
fn test_concatenations_of_element_wise_groups_compile_to_one_kernel() {
    let _lock = device_lock(0).unwrap();
    let device = Device::Cuda(0);
    let x = Tensor::from_vec(
        device,
        Shape::create([2, 3, 5, 8]).unwrap(),
        values(240, 0.013),
    )
    .unwrap();
    let cos = Tensor::from_vec(device, Shape::create([5, 4]).unwrap(), values(20, 0.11)).unwrap();
    let sin = Tensor::from_vec(device, Shape::create([5, 4]).unwrap(), values(20, -0.09)).unwrap();
    // Rotary embedding: both rotated halves of each head land in one concatenation.
    let rotary = |inputs: &[Tensor]| -> Result<Tensor> {
        let (x, cos, sin) = (&inputs[0], &inputs[1], &inputs[2]);
        let first = x.narrow(3, 0, 4)?;
        let second = x.narrow(3, 4, 4)?;
        Tensor::concat(
            &[
                first.mul(cos)?.sub(&second.mul(sin)?)?,
                second.mul(cos)?.add(&first.mul(sin)?)?,
            ],
            3,
        )
    };
    // A graph input joins as a slice the kernel copies.
    let mixed = |inputs: &[Tensor]| -> Result<Tensor> {
        Tensor::concat(&[inputs[0].exp()?.mul_scalar(0.5)?, inputs[0].clone()], 1)
    };
    let named = [("x", x), ("cos", cos), ("sin", sin)];
    let tensors = named
        .iter()
        .map(|(_, tensor)| tensor.clone())
        .collect::<Vec<_>>();
    let expected = [rotary(&tensors).unwrap(), mixed(&tensors).unwrap()];
    let (captured, _) = Graph::capture(named.clone(), |inputs| {
        Ok(vec![rotary(inputs)?, mixed(inputs)?])
    })
    .unwrap();
    let mut compiled = cuda::compile_graph(&captured, &GraphCudaOptions::default()).unwrap();
    assert_eq!(compiled.stats().kernel_child_graph_nodes(), 2);
    let inputs = named
        .into_iter()
        .map(|(name, tensor)| (name.to_owned(), tensor))
        .collect::<HashMap<_, _>>();
    let actual = compiled.run(&inputs).unwrap();
    for (actual, expected) in actual.iter().zip(&expected) {
        assert_eq!(actual.shape(), expected.shape());
        assert_eq!(
            actual.to_vec::<f32>().unwrap(),
            expected.to_vec::<f32>().unwrap()
        );
    }
}

#[test]
fn test_row_operations_compute_the_element_wise_groups_they_read() {
    let _lock = device_lock(0).unwrap();
    let device = Device::Cuda(0);
    let scores = Tensor::from_vec(
        device,
        Shape::create([2, 3, 40]).unwrap(),
        values(240, 0.011),
    )
    .unwrap();
    let mask =
        Tensor::from_vec(device, Shape::create([3, 40]).unwrap(), values(120, -0.02)).unwrap();
    // Attention's scaled, masked scores feed the softmax, and a scale alone feeds a sum.
    let function = |inputs: &[Tensor]| -> Result<Vec<Tensor>> {
        Ok(vec![
            inputs[0].mul_scalar(0.25)?.add(&inputs[1])?.softmax(-1)?,
            inputs[0].mul_scalar(2.0)?.sum(&[1], false)?,
        ])
    };
    let named = [("scores", scores), ("mask", mask)];
    let tensors = named
        .iter()
        .map(|(_, tensor)| tensor.clone())
        .collect::<Vec<_>>();
    let expected = function(&tensors).unwrap();
    let (captured, _) = Graph::capture(named.clone(), function).unwrap();
    let mut compiled = cuda::compile_graph(&captured, &GraphCudaOptions::default()).unwrap();
    assert_eq!(compiled.stats().kernel_child_graph_nodes(), 2);
    let inputs = named
        .into_iter()
        .map(|(name, tensor)| (name.to_owned(), tensor))
        .collect::<HashMap<_, _>>();
    let actual = compiled.run(&inputs).unwrap();
    for (actual, expected) in actual.iter().zip(&expected) {
        assert_eq!(actual.shape(), expected.shape());
        let close = actual
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .zip(expected.to_vec::<f32>().unwrap())
            .all(|(actual, expected)| (actual - expected).abs() <= 1e-5 * expected.abs().max(1.0));
        assert!(close);
    }
}

#[test]
fn test_row_kernels_also_store_groups_that_other_operations_read() {
    let _lock = device_lock(0).unwrap();
    let device = Device::Cuda(0);
    let tensor = |seed| {
        Tensor::from_vec(device, Shape::create([6, 40]).unwrap(), values(240, seed)).unwrap()
    };
    // A residual sum feeds the row operation and is read again after it, as a decoder's hidden state is.
    let function = |inputs: &[Tensor]| -> Result<Vec<Tensor>> {
        let hidden = inputs[0].add(&inputs[1])?;
        Ok(vec![hidden.softmax(-1)?.mul(&hidden)?])
    };
    let named = [("x", tensor(0.01)), ("y", tensor(-0.02))];
    let tensors = named
        .iter()
        .map(|(_, tensor)| tensor.clone())
        .collect::<Vec<_>>();
    let expected = function(&tensors).unwrap();
    let (captured, _) = Graph::capture(named.clone(), function).unwrap();
    let mut compiled = cuda::compile_graph(&captured, &GraphCudaOptions::default()).unwrap();
    assert_eq!(compiled.stats().kernel_child_graph_nodes(), 2);
    let inputs = named
        .into_iter()
        .map(|(name, tensor)| (name.to_owned(), tensor))
        .collect::<HashMap<_, _>>();
    let actual = compiled.run(&inputs).unwrap();
    let close = actual[0]
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .zip(expected[0].to_vec::<f32>().unwrap())
        .all(|(actual, expected)| (actual - expected).abs() <= 1e-5 * expected.abs().max(1.0));
    assert!(close);
}

#[test]
fn test_fused_groups_compile_to_tile_kernels_that_match_eager_results() {
    let _lock = device_lock(0).unwrap();
    let device = Device::Cuda(0);
    let tensor = |dimensions: &[i64], seed| {
        let count = usize::try_from(dimensions.iter().product::<i64>()).unwrap();
        Tensor::from_vec(
            device,
            Shape::create(dimensions).unwrap(),
            values(count, seed),
        )
        .unwrap()
    };
    // Rotary embedding's concatenation, a masked softmax, and a residual sum the softmax reads and the output reuses.
    let function = |inputs: &[Tensor]| -> Result<Vec<Tensor>> {
        let (x, cos, sin, mask) = (&inputs[0], &inputs[1], &inputs[2], &inputs[3]);
        let first = x.narrow(2, 0, 4)?;
        let second = x.narrow(2, 4, 4)?;
        let rotated = Tensor::concat(
            &[
                first.mul(cos)?.sub(&second.mul(sin)?)?,
                second.mul(cos)?.add(&first.mul(sin)?)?,
            ],
            2,
        )?;
        let hidden = x.add(&rotated)?;
        let attention = hidden.mul_scalar(0.25)?.add(mask)?.softmax(-1)?;
        Ok(vec![rotated, attention.mul(&hidden)?])
    };
    let named = [
        ("x", tensor(&[3, 5, 8], 0.013)),
        ("cos", tensor(&[5, 4], 0.11)),
        ("sin", tensor(&[5, 4], -0.09)),
        ("mask", tensor(&[5, 8], -0.02)),
    ];
    let tensors = named
        .iter()
        .map(|(_, tensor)| tensor.clone())
        .collect::<Vec<_>>();
    let expected = function(&tensors).unwrap();
    let (captured, _) = Graph::capture(named.clone(), function).unwrap();
    let compiled = cuda::compile_graph(
        &captured,
        &GraphCudaOptions::default().with_kernel_language(KernelLanguage::TileIr),
    );
    let mut compiled = compiled.unwrap();
    let inputs = named
        .into_iter()
        .map(|(name, tensor)| (name.to_owned(), tensor))
        .collect::<HashMap<_, _>>();
    let actual = compiled.run(&inputs).unwrap();
    for (actual, expected) in actual.iter().zip(&expected) {
        assert_eq!(actual.shape(), expected.shape());
        let close = actual
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .zip(expected.to_vec::<f32>().unwrap())
            .all(|(actual, expected)| (actual - expected).abs() <= 1e-5 * expected.abs().max(1.0));
        assert!(close);
    }
}

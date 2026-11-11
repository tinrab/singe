#![cfg(all(feature = "testing", feature = "cuda"))]

use std::collections::HashMap;

use singe::{
    Device, Shape, Tensor,
    error::Error,
    graph::{Graph, GraphPrepareOptions},
};
use singe_cuda::testing::device_lock;

fn tensor(device: Device, dimensions: &[i64], values: Vec<f32>) -> Tensor {
    Tensor::from_vec(device, Shape::create(dimensions.to_vec()).unwrap(), values).unwrap()
}

#[test]
fn test_compiled_cuda_graphs_report_out_of_range_indices_after_replay() {
    let _lock = device_lock(0).unwrap();
    let device = Device::Cuda(0);
    let input = tensor(device, &[3, 2], vec![1., 2., 3., 4., 5., 6.]);
    let rows = Tensor::from_vec(device, Shape::create([2]).unwrap(), vec![0_i64, 2]).unwrap();
    let source = tensor(device, &[2, 2], vec![10., 20., 30., 40.]);
    let (captured, _) = Graph::capture(
        [
            ("x", input.clone()),
            ("rows", rows.clone()),
            ("source", source.clone()),
        ],
        |inputs| Ok(vec![inputs[0].index_add(0, &inputs[1], &inputs[2])?]),
    )
    .unwrap();
    let mut prepared = captured.prepare(GraphPrepareOptions::new(device)).unwrap();
    let mut inputs = HashMap::from([
        ("x".to_owned(), input),
        ("rows".to_owned(), rows),
        ("source".to_owned(), source),
    ]);
    assert_eq!(
        prepared.run(&inputs).unwrap()[0].to_vec::<f32>().unwrap(),
        [11., 22., 3., 4., 35., 46.]
    );

    let bad = Tensor::from_vec(device, Shape::create([2]).unwrap(), vec![0_i64, 7]).unwrap();
    inputs.insert("rows".to_owned(), bad);
    assert!(matches!(
        prepared.run(&inputs),
        Err(Error::IndexOutOfBounds)
    ));

    // The status word is cleared, so a valid replay succeeds again.
    let good = Tensor::from_vec(device, Shape::create([2]).unwrap(), vec![1_i64, 1]).unwrap();
    inputs.insert("rows".to_owned(), good);
    assert_eq!(
        prepared.run(&inputs).unwrap()[0].to_vec::<f32>().unwrap(),
        [1., 2., 43., 64., 5., 6.]
    );
}

#![cfg(all(feature = "testing", feature = "cuda"))]

use std::collections::HashMap;

use singe::{
    Device, Shape, Tensor,
    error::{Error, Result},
    graph::{Graph, GraphPrepareOptions},
};
use singe_cuda::testing::device_lock;

fn tensor(device: Device, dimensions: &[i64], values: Vec<f32>) -> Tensor {
    Tensor::from_vec(device, Shape::create(dimensions.to_vec()).unwrap(), values).unwrap()
}

#[test]
fn test_compiled_cuda_graphs_sort_scan_and_index_with_computed_indices() {
    let _lock = device_lock(0).unwrap();
    let device = Device::Cuda(0);
    let values = (0..24)
        .map(|value| ((value * 7) % 11) as f32)
        .collect::<Vec<_>>();
    let input = tensor(device, &[4, 6], values);
    let rows = Tensor::from_vec(device, Shape::create([2]).unwrap(), vec![3_i64, 1]).unwrap();
    let step = |x: &Tensor, rows: &Tensor| -> Result<Vec<Tensor>> {
        let (sorted, order) = x.sort(-1, true)?;
        // The sort order is computed inside the graph and indexes the input again.
        let reordered = x.gather(1, &order)?;
        let sums = x.cumsum(1)?;
        let added = x.index_add(0, rows, &sorted.narrow(0, 0, 2)?)?;
        Ok(vec![sorted, order, reordered, sums, added])
    };
    let expected = step(&input, &rows).unwrap();
    let (captured, _) = Graph::capture([("x", input.clone()), ("rows", rows.clone())], |inputs| {
        step(&inputs[0], &inputs[1])
    })
    .unwrap();
    let mut prepared = captured.prepare(GraphPrepareOptions::new(device)).unwrap();
    let inputs = HashMap::from([("x".to_owned(), input), ("rows".to_owned(), rows)]);
    for _ in 0..2 {
        let outputs = prepared.run(&inputs).unwrap();
        assert_eq!(outputs.len(), expected.len());
        for (output, expected) in outputs.iter().zip(&expected) {
            assert_eq!(output.data_type(), expected.data_type());
            assert_eq!(output.to_bytes().unwrap(), expected.to_bytes().unwrap());
        }
    }
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

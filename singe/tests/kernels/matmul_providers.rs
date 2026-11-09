#![cfg(all(feature = "testing", feature = "cuda"))]

use std::collections::HashMap;

use singe::{
    DataType, Device, Shape, Tensor,
    graph::{Graph, GraphPrepareOptions},
};
use singe_cuda::testing::device_lock;

fn matrix(dimensions: &[i64], seed: usize) -> Tensor {
    let count = usize::try_from(dimensions.iter().product::<i64>()).unwrap();
    let values = (0..count)
        .map(|index| ((index * 7 + seed * 3) % 11) as f32 / 8.0 - 0.6)
        .collect::<Vec<_>>();
    Tensor::from_vec(
        Device::Cuda(0),
        Shape::create(dimensions.to_vec()).unwrap(),
        values,
    )
    .unwrap()
}

fn as_f32(tensor: &Tensor) -> Vec<f32> {
    tensor.cast(DataType::F32).unwrap().to_vec::<f32>().unwrap()
}

#[test]
fn test_cuda_matmul_agrees_between_eager_and_compiled_graphs_for_every_kernel() {
    let _lock = device_lock(0).unwrap();
    let cases: [(&[i64], &[i64], DataType, f32); 5] = [
        (&[33, 17], &[17, 9], DataType::F32, 1e-5),
        (&[33, 17], &[17, 9], DataType::F16, 2e-3),
        (&[33, 17], &[17, 9], DataType::Bf16, 1e-2),
        (&[8, 5], &[5, 6], DataType::F64, 1e-12),
        (&[2, 8, 5], &[5, 6], DataType::F32, 1e-5),
    ];
    for (lhs_shape, rhs_shape, data_type, tolerance) in cases {
        let lhs = matrix(lhs_shape, 1).cast(data_type).unwrap();
        let rhs = matrix(rhs_shape, 2).cast(data_type).unwrap();
        let reference = as_f32(
            &lhs.cast(DataType::F64)
                .unwrap()
                .matmul(&rhs.cast(DataType::F64).unwrap())
                .unwrap(),
        );
        let eager = lhs.matmul(&rhs).unwrap();
        assert_eq!(eager.data_type(), data_type);
        let (captured, _) = Graph::capture([("a", lhs.clone()), ("b", rhs.clone())], |inputs| {
            Ok(vec![inputs[0].matmul(&inputs[1])?])
        })
        .unwrap();
        let mut prepared = captured
            .prepare(GraphPrepareOptions::new(Device::Cuda(0)))
            .unwrap();
        let compiled = prepared
            .run(&HashMap::from([
                ("a".to_owned(), lhs),
                ("b".to_owned(), rhs),
            ]))
            .unwrap()
            .remove(0);
        for result in [as_f32(&eager), as_f32(&compiled)] {
            for (actual, expected) in result.iter().zip(&reference) {
                assert!(
                    (actual - expected).abs() <= tolerance * expected.abs().max(1.0) * 4.0,
                    "{data_type:?} {lhs_shape:?}: {actual} differs from {expected}"
                );
            }
        }
    }
}

use std::{collections::HashMap, sync::Arc};

use singe::{
    DataType, Device, Shape, Tensor,
    graph::{Graph, GraphPrepareOptions},
    layout::Layout,
    storage::cpu::CpuStorage,
};

fn matrix() -> Tensor {
    Tensor::from_vec(
        Device::Cpu,
        Shape::create([2, 3]).unwrap(),
        vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0],
    )
    .unwrap()
}

#[test]
fn test_view_rejects_incompatible_transposed_layout() {
    let input = matrix().transpose(0, 1).unwrap();
    assert!(input.view(Shape::create([6]).unwrap()).is_err());
}

#[test]
fn test_strided_layout_cannot_escape_its_allocation() {
    let input =
        Tensor::from_vec(Device::Cpu, Shape::create([2]).unwrap(), vec![1.0_f32, 2.0]).unwrap();
    assert!(
        Tensor::create_from_storage(
            Arc::clone(input.storage()),
            DataType::F32,
            Layout::with_strides([2], [100]).unwrap()
        )
        .is_err()
    );
    assert!(
        Tensor::create_from_storage(
            Arc::clone(input.storage()),
            DataType::F32,
            Layout::contiguous([2]).unwrap().with_offset(1)
        )
        .is_err()
    );
}

#[test]
fn test_compatible_strided_chunks_and_broadcasts_reshape_without_copying() {
    let storage = Arc::clone(matrix().storage());
    let input = Tensor::create_from_storage(
        storage,
        DataType::F32,
        Layout::with_strides([3], [2]).unwrap(),
    )
    .unwrap();
    let reshaped = input.view(Shape::create([1, 3, 1]).unwrap()).unwrap();
    assert!(Arc::ptr_eq(input.storage(), reshaped.storage()));
    assert_eq!(reshaped.to_vec::<f32>().unwrap(), [0.0, 2.0, 4.0]);

    let storage =
        CpuStorage::create_from_bytes(Device::Cpu, DataType::F32, &2.0_f32.to_ne_bytes()).unwrap();
    let broadcast = Tensor::create(
        storage,
        DataType::F32,
        Layout::with_strides([2, 3], [0, 0]).unwrap(),
    )
    .unwrap();
    let flat = broadcast.reshape(vec![6]).unwrap();
    assert!(Arc::ptr_eq(broadcast.storage(), flat.storage()));
    assert_eq!(flat.to_vec::<f32>().unwrap(), [2.0; 6]);
}

#[test]
fn test_empty_and_unit_dimensions_do_not_require_copying() {
    let empty = Tensor::from_vec(
        Device::Cpu,
        Shape::create([0, 3]).unwrap(),
        Vec::<f32>::new(),
    )
    .unwrap();
    let flat = empty.reshape(vec![3, 0, 1]).unwrap();
    assert!(Arc::ptr_eq(empty.storage(), flat.storage()));
    assert_eq!(flat.to_vec::<f32>().unwrap(), [] as [f32; 0]);
    let storage = Arc::clone(matrix().storage());
    let input = Tensor::create_from_storage(
        storage,
        DataType::F32,
        Layout::with_strides([1, 2], [i64::MAX, 1]).unwrap(),
    )
    .unwrap();
    assert_eq!(
        input.reshape(vec![2]).unwrap().to_vec::<f32>().unwrap(),
        [0.0, 1.0]
    );
}

#[cfg(all(feature = "cuda", feature = "testing"))]
#[test]
fn test_cuda_strict_replay_preserves_scalar_rounding_and_exceptional_values() {
    use singe::testing::maybe_lock_device;

    let Some(_device_lock) = maybe_lock_device() else {
        return;
    };
    let input = Tensor::from_vec(
        Device::Cuda(0),
        Shape::create([5]).unwrap(),
        vec![1.0_f32, -0.0, f32::MAX, f32::INFINITY, f32::NAN],
    )
    .unwrap();
    let (captured, captured_outputs) = Graph::capture([("x", input.clone())], |inputs| {
        Ok(vec![
            inputs[0].add_scalar(1e20)?.sub_scalar(1e20)?,
            inputs[0].mul_scalar(2.0)?.mul_scalar(0.5)?,
            inputs[0].mul_scalar(0.0)?,
            inputs[0].add_scalar(0.0)?,
        ])
    })
    .unwrap();
    let mut executor = captured
        .prepare(GraphPrepareOptions::new(Device::Cuda(0)))
        .unwrap();
    let outputs = executor
        .run(&HashMap::from([("x".to_string(), input)]))
        .unwrap();
    for (expected, actual) in captured_outputs.iter().zip(outputs) {
        for (expected, actual) in expected
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .zip(actual.to_vec::<f32>().unwrap())
        {
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(expected.to_bits(), actual.to_bits());
            }
        }
    }
}

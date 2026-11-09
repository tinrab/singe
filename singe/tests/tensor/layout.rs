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
fn test_reshape_preserves_transposed_logical_order() {
    let input = matrix().transpose(0, 1).unwrap();
    let output = input.reshape(vec![6]).unwrap();
    assert_eq!(
        output.to_vec::<f32>().unwrap(),
        [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
    );
    assert!(!Arc::ptr_eq(input.storage(), output.storage()));
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
fn test_reshape_preserves_narrowed_transposed_values_and_offset() {
    let input = matrix().narrow(0, 1, 1).unwrap().transpose(0, 1).unwrap();
    let output = input.reshape(vec![3]).unwrap();
    assert_eq!(output.to_vec::<f32>().unwrap(), [3.0, 4.0, 5.0]);
    assert!(Arc::ptr_eq(input.storage(), output.storage()));
    assert_eq!(Layout::from_tensor(&output).storage_offset(), 3);
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

#[test]
fn test_capture_records_materialization_needed_by_reshape() {
    let (captured, _) = Graph::capture([("x", matrix())], |inputs| {
        Ok(vec![inputs[0].transpose(0, 1)?.reshape(vec![6])?])
    })
    .unwrap();
    let outputs = captured
        .run(&HashMap::from([(
            "x".to_string(),
            matrix().add_scalar(10.0).unwrap(),
        )]))
        .unwrap();
    assert_eq!(
        outputs[0].to_vec::<f32>().unwrap(),
        [10.0, 13.0, 11.0, 14.0, 12.0, 15.0]
    );
}

#[test]
fn test_conversions_produce_valid_contiguous_storage_from_strided_views() {
    let input = Tensor::create_from_storage(
        Arc::clone(matrix().storage()),
        DataType::F32,
        Layout::with_strides([3], [2]).unwrap(),
    )
    .unwrap();
    let cast = input.cast(DataType::F64).unwrap();
    assert_eq!(cast.to_vec::<f64>().unwrap(), [0.0, 2.0, 4.0]);
    let quantized = input.quantize(1.0, 0, DataType::I8).unwrap();
    assert_eq!(quantized.to_vec::<i8>().unwrap(), [0, 2, 4]);
    assert!(cast.layout().is_contiguous());
    assert!(quantized.layout().is_contiguous());
}

#[cfg(all(feature = "cuda", feature = "testing"))]
#[test]
fn test_cuda_reshape_and_replay_preserve_transposed_values() {
    use singe::testing::maybe_lock_device;

    let Some(_device_lock) = maybe_lock_device() else {
        return;
    };
    let input = matrix().to_device(Device::Cuda(0)).unwrap();
    let (captured, captured_outputs) = Graph::capture([("x", input.clone())], |inputs| {
        Ok(vec![inputs[0].transpose(0, 1)?.reshape(vec![6])?])
    })
    .unwrap();
    assert_eq!(
        captured_outputs[0].to_vec::<f32>().unwrap(),
        [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
    );
    let mut executor = captured
        .prepare(GraphPrepareOptions::new(Device::Cuda(0)))
        .unwrap();
    let outputs = executor
        .run(&HashMap::from([("x".to_string(), input)]))
        .unwrap();
    assert_eq!(
        outputs[0].to_vec::<f32>().unwrap(),
        [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
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

#[test]
fn test_cpu_concat_preserves_integral_bits_and_strided_logical_order() {
    let large = (1_i64 << 54) + 1;
    let input = Tensor::from_vec(
        Device::Cpu,
        Shape::create([2, 3]).unwrap(),
        vec![large, 2, 3, 4, 5, 6],
    )
    .unwrap()
    .transpose(0, 1)
    .unwrap();
    let output = Tensor::concat(&[input.clone(), input.clone()], 1).unwrap();
    assert_eq!(
        output.to_vec::<i64>().unwrap(),
        [large, 4, large, 4, 2, 5, 2, 5, 3, 6, 3, 6]
    );
    let (captured, _) = Graph::capture([("x", input.clone())], |inputs| {
        Ok(vec![Tensor::concat(
            &[inputs[0].clone(), inputs[0].clone()],
            1,
        )?])
    })
    .unwrap();
    let mut executor = captured
        .prepare(GraphPrepareOptions::new(Device::Cpu))
        .unwrap();
    assert_eq!(
        executor.run(&HashMap::from([("x".into(), input)])).unwrap()[0]
            .to_bytes()
            .unwrap(),
        output.to_bytes().unwrap()
    );
    let boolean =
        Tensor::from_vec(Device::Cpu, Shape::create([2]).unwrap(), vec![true, false]).unwrap();
    assert_eq!(
        Tensor::concat(&[boolean.clone(), boolean], 0)
            .unwrap()
            .to_vec::<bool>()
            .unwrap(),
        [true, false, true, false]
    );
}

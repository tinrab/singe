use std::collections::HashMap;

use singe::{
    Device, Shape, Tensor,
    data_type::{bf16, f16},
    error::Error,
    graph::Graph,
    idx,
};
#[cfg(all(feature = "testing", feature = "cuda"))]
use singe::{backend::cuda, graph::GraphCudaOptions};
#[cfg(all(feature = "testing", feature = "cuda"))]
use singe_cuda::testing::device_lock;

fn assert_invalid_reductions(device: Device) {
    let empty =
        Tensor::from_vec(device, Shape::create([2, 0]).unwrap(), Vec::<f32>::new()).unwrap();
    assert!(matches!(
        empty.max_reduce(&[1], false),
        Err(Error::InvalidArgument { .. })
    ));
    assert!(matches!(
        empty.min_reduce(&[1], false),
        Err(Error::InvalidArgument { .. })
    ));
    let input = Tensor::full(device, Shape::create([2, 3]).unwrap(), 1.0_f32).unwrap();
    assert!(matches!(
        input.sum(&[0, -2], false),
        Err(Error::InvalidArgument { .. })
    ));
    assert!(matches!(
        input.mean(&[2], false),
        Err(Error::DimensionOutOfBounds)
    ));
}

#[test]
fn test_cpu_reduction_validation_rejects_empty_extrema_duplicate_axes_and_invalid_axes() {
    assert_invalid_reductions(Device::Cpu);
}

#[test]
fn test_scalar_views_reject_axes_outside_the_virtual_scalar_axis() {
    let scalar = Tensor::scalar(Device::Cpu, 3.0_f32).unwrap();
    for invalid in [-2, 1] {
        for result in [
            scalar.transpose(invalid, 0),
            scalar.transpose(0, invalid),
            scalar.flatten(invalid, 0),
            scalar.flatten(0, invalid),
        ] {
            assert!(matches!(result, Err(Error::DimensionOutOfBounds)));
        }
    }
}

#[cfg(all(feature = "testing", feature = "cuda"))]
#[test]
fn test_cuda_reduction_validation_and_mixed_device_rejection_follow_the_tensor_contract() {
    let _lock = device_lock(0).unwrap();
    let device = Device::Cuda(0);
    assert_invalid_reductions(device);
    let cpu = Tensor::scalar(Device::Cpu, 1.0_f32).unwrap();
    let cuda = Tensor::scalar(device, 1.0_f32).unwrap();
    for result in [cpu.add(&cuda), cuda.equal(&cpu), cpu.div(&cuda)] {
        assert!(matches!(result, Err(Error::DeviceMismatch { .. })));
    }
    let cpu_mask = Tensor::scalar(Device::Cpu, true).unwrap();
    let cuda_mask = Tensor::scalar(device, true).unwrap();
    assert!(matches!(
        cpu_mask.logical_and(&cuda_mask),
        Err(Error::DeviceMismatch { .. })
    ));
    assert!(matches!(
        Tensor::where_select(&cpu_mask, &cuda, &cuda),
        Err(Error::DeviceMismatch { .. })
    ));
    assert!(matches!(
        Tensor::where_select(&cuda_mask, &cpu, &cuda),
        Err(Error::DeviceMismatch { .. })
    ));
}

#[cfg(all(feature = "testing", feature = "cuda"))]
#[test]
fn test_cuda_general_kernel_replay_retains_modules_after_cache_clear_and_outputs_after_reuse() {
    let _lock = device_lock(0).unwrap();
    let device = Device::Cuda(0);
    let input = Tensor::from_vec(
        device,
        Shape::create([2, 3]).unwrap(),
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();
    let (captured, _) = Graph::capture([("x", input.clone())], |inputs| {
        let transposed = inputs[0].transpose(0, 1)?;
        Ok(vec![transposed.add_scalar(0.25)?.sum(&[1], false)?])
    })
    .unwrap();
    let mut compiled = cuda::compile_graph(&captured, &GraphCudaOptions::default()).unwrap();
    cuda::clear_cache(device).unwrap();
    let first = compiled
        .run(&HashMap::from([("x".into(), input.clone())]))
        .unwrap();
    let changed = input.add_scalar(10.0).unwrap();
    cuda::clear_cache(device).unwrap();
    let second = compiled
        .run(&HashMap::from([("x".into(), changed)]))
        .unwrap();
    assert_eq!(first[0].to_vec::<f64>().unwrap(), [5.5, 7.5, 9.5]);
    assert_eq!(second[0].to_vec::<f64>().unwrap(), [25.5, 27.5, 29.5]);
}

#[test]
fn test_copies_are_independent_and_broadcast_views_share_storage() {
    assert_copy_and_view_identity(Device::Cpu);
}

#[cfg(all(feature = "testing", feature = "cuda"))]
#[test]
fn test_cuda_copies_and_broadcast_views_preserve_allocation_identity_contracts() {
    let _lock = device_lock(0).unwrap();
    assert_copy_and_view_identity(Device::Cuda(0));
}

fn assert_copy_and_view_identity(device: Device) {
    use std::sync::Arc;

    let input =
        Tensor::from_vec(device, Shape::create([3]).unwrap(), vec![1.0_f32, 2.0, 3.0]).unwrap();
    let cloned = input.clone();
    assert!(Arc::ptr_eq(input.storage(), cloned.storage()));
    let copied = input.copy().unwrap();
    assert!(!Arc::ptr_eq(input.storage(), copied.storage()));
    assert_eq!(copied.to_vec::<f32>().unwrap(), [1.0, 2.0, 3.0]);
    let expanded = input.expand(Shape::create([2, 3]).unwrap()).unwrap();
    assert!(Arc::ptr_eq(input.storage(), expanded.storage()));
    assert_eq!(expanded.layout().strides(), &[0, 1]);
    assert_eq!(
        expanded.to_vec::<f32>().unwrap(),
        [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    );
    let offset = input.narrow(0, 1, 2).unwrap();
    assert!(offset.layout().is_contiguous());
    assert_eq!(offset.layout().storage_offset(), 1);
    let (captured, _) = Graph::capture([("x", input.clone())], |inputs| {
        Ok(vec![inputs[0].copy()?, inputs[0].copy()?])
    })
    .unwrap();
    let outputs = captured
        .compile()
        .unwrap()
        .run(&HashMap::from([("x".into(), input.clone())]))
        .unwrap();
    assert!(!Arc::ptr_eq(input.storage(), outputs[0].storage()));
    assert!(!Arc::ptr_eq(outputs[0].storage(), outputs[1].storage()));
    #[cfg(all(feature = "testing", feature = "cuda"))]
    if device.is_cuda() {
        let outputs = cuda::compile_graph(&captured, &GraphCudaOptions::default())
            .unwrap()
            .run(&HashMap::from([("x".into(), input.clone())]))
            .unwrap();
        assert_ne!(
            input.storage().storage_identity(),
            outputs[0].storage().storage_identity()
        );
        assert_ne!(
            outputs[0].storage().storage_identity(),
            outputs[1].storage().storage_identity()
        );
        assert_eq!(outputs[0].to_vec::<f32>().unwrap(), [1.0, 2.0, 3.0]);
    }
}

#[test]
fn test_selected_scalar_and_offset_views_replay_changed_inputs() {
    let original = Tensor::from_vec(
        Device::Cpu,
        Shape::create([2, 2]).unwrap(),
        vec![1.0_f32, 2.0, 3.0, 4.0],
    )
    .unwrap();
    let (captured, _) = Graph::capture([("x", original.clone())], |inputs| {
        Ok(vec![inputs[0].index(&idx![1, 0])?])
    })
    .unwrap();
    let changed = original.add_scalar(10.0).unwrap();
    let result = captured
        .compile()
        .unwrap()
        .run(&HashMap::from([("x".into(), changed)]))
        .unwrap();
    assert_eq!(result[0].shape(), &Shape::scalar());
    assert_eq!(result[0].to_vec::<f32>().unwrap(), [13.0]);
}

#[test]
fn test_cpu_storage_import_is_aligned_and_validates_boolean_bytes() {
    use singe::{DataType, storage::cpu::CpuStorage};

    let bytes = [0_u8, 0, 0, 0, 0, 0, 0, 240, 63];
    let storage = CpuStorage::create_from_bytes(Device::Cpu, DataType::F64, &bytes[1..]).unwrap();
    assert_eq!(*storage.as_typed_slice::<f64>().unwrap(), [1.0]);
    assert_eq!(storage.as_ptr().unwrap().addr() % align_of::<f64>(), 0);
    assert!(CpuStorage::create_from_bytes(Device::Cpu, DataType::Bool, &[2]).is_err());
    let mut booleans = CpuStorage::create(DataType::Bool, 2).unwrap();
    booleans.as_typed_slice_mut::<bool>().unwrap()[1] = true;
    assert_eq!(*booleans.as_slice().unwrap(), [0, 1]);
}

#[test]
fn test_scalar_vector_and_empty_shapes_remain_distinct() {
    let scalar = Tensor::scalar(Device::Cpu, 3.0_f32).unwrap();
    assert_eq!(scalar.shape().dimensions(), &[] as &[i64]);
    assert_eq!(scalar.shape().rank(), 0);
    assert_eq!(scalar.to_vec::<f32>().unwrap(), [3.0]);
    let vector = scalar.unsqueeze(0).unwrap();
    assert_eq!(scalar.squeeze(-1).unwrap().shape(), scalar.shape());
    assert_eq!(
        vector.squeeze(0).unwrap().squeeze(0).unwrap().shape(),
        scalar.shape()
    );
    assert_eq!(vector.shape().dimensions(), &[1]);
    assert_eq!(
        vector.squeeze(0).unwrap().shape().dimensions(),
        &[] as &[i64]
    );
    assert_eq!(
        vector.index(&idx![0]).unwrap().shape().dimensions(),
        &[] as &[i64]
    );
    assert_eq!(scalar.reshape(vec![1]).unwrap().shape().dimensions(), &[1]);
    let empty =
        Tensor::from_vec(Device::Cpu, Shape::create([0]).unwrap(), Vec::<f32>::new()).unwrap();
    assert_eq!(empty.shape().element_count(), 0);
    assert!(empty.reshape(vec![]).is_err());
    let fingerprint = |input: Tensor| {
        Graph::capture([("x", input)], |inputs| Ok(vec![inputs[0].clone()]))
            .unwrap()
            .0
            .fingerprint()
            .unwrap()
    };
    let scalar_fingerprint = fingerprint(scalar);
    let vector_fingerprint = fingerprint(vector);
    let empty_fingerprint = fingerprint(empty);
    assert_ne!(scalar_fingerprint, vector_fingerprint);
    assert_ne!(scalar_fingerprint, empty_fingerprint);
    assert_ne!(vector_fingerprint, empty_fingerprint);
}

#[test]
fn test_byte_export_preserves_strided_fp64_bits_and_bool_broadcasts() {
    let values = vec![
        1.000_000_000_000_000_2_f64,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    let input = Tensor::from_vec(Device::Cpu, Shape::create([2, 2]).unwrap(), values)
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    let expected = [
        1.000_000_000_000_000_2_f64,
        f64::INFINITY,
        -0.0,
        f64::NEG_INFINITY,
    ];
    assert_eq!(
        input.to_bytes().unwrap(),
        expected
            .into_iter()
            .flat_map(f64::to_ne_bytes)
            .collect::<Vec<_>>()
    );
    let restored = Tensor::from_bytes(
        Device::Cpu,
        input.data_type(),
        input.shape(),
        &input.to_bytes().unwrap(),
    )
    .unwrap();
    assert_eq!(
        restored
            .to_vec::<f64>()
            .unwrap()
            .into_iter()
            .map(f64::to_bits)
            .collect::<Vec<_>>(),
        expected.into_iter().map(f64::to_bits).collect::<Vec<_>>()
    );
    let mask = Tensor::scalar(Device::Cpu, true)
        .unwrap()
        .expand(Shape::create([2, 3]).unwrap())
        .unwrap();
    assert_eq!(mask.to_bytes().unwrap(), [1_u8; 6]);
    assert_eq!(mask.to_vec::<bool>().unwrap(), [true; 6]);
}

#[test]
fn test_typed_fill_and_like_constructors_preserve_bool_half_and_integer_values() {
    let shape = Shape::create([2, 3]).unwrap();
    assert_eq!(
        Tensor::ones::<bool>(Device::Cpu, &shape)
            .unwrap()
            .to_vec::<bool>()
            .unwrap(),
        [true; 6]
    );
    assert_eq!(
        Tensor::ones::<f16>(Device::Cpu, &shape)
            .unwrap()
            .to_vec::<f16>()
            .unwrap(),
        [f16::from_f32(1.0); 6]
    );
    assert_eq!(
        Tensor::ones::<bf16>(Device::Cpu, &shape)
            .unwrap()
            .to_vec::<bf16>()
            .unwrap(),
        [bf16::from_f32(1.0); 6]
    );
    let large = (1_i64 << 60) + 1;
    let input = Tensor::full(Device::Cpu, shape, large)
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    assert_eq!(
        input.full_like(large).unwrap().to_vec::<i64>().unwrap(),
        [large; 6]
    );
    assert_eq!(input.ones_like().unwrap().to_vec::<i64>().unwrap(), [1; 6]);
    assert_eq!(input.zeros_like().unwrap().to_vec::<i64>().unwrap(), [0; 6]);
    assert!(input.full_like(1.0_f32).is_err());
}

#[test]
fn test_reduction_and_capture_preserve_rank_zero_outputs() {
    let input = Tensor::from_vec(
        Device::Cpu,
        Shape::create([3]).unwrap(),
        vec![1.0_f32, 2.0, 3.0],
    )
    .unwrap();
    let (captured, _) = Graph::capture([("x", input.clone())], |inputs| {
        Ok(vec![inputs[0].sum(&[0], false)?, inputs[0].argmax(0)?])
    })
    .unwrap();
    let outputs = captured.run(&HashMap::from([("x".into(), input)])).unwrap();
    assert_eq!(outputs[0].shape().dimensions(), &[] as &[i64]);
    assert_eq!(outputs[0].to_vec::<f32>().unwrap(), [6.0]);
    assert_eq!(outputs[1].shape().dimensions(), &[] as &[i64]);
    assert_eq!(outputs[1].to_vec::<i64>().unwrap(), [2]);
}

#[cfg(all(feature = "testing", feature = "cuda"))]
#[test]
fn test_cuda_nan_extrema_preserve_first_indices_and_nan_results() {
    let _lock = device_lock(0).expect("the CUDA contract tests require device 0");
    for (values, expected) in [
        (vec![f32::NAN, 2.0, 2.0, -1.0], 0),
        (vec![2.0, f32::NAN, f32::NAN, -1.0], 1),
        (vec![f32::INFINITY, f32::NAN, f32::NEG_INFINITY], 1),
        (vec![f32::NAN, f32::NAN], 0),
    ] {
        let input = Tensor::from_vec(
            Device::Cuda(0),
            Shape::create([i64::try_from(values.len()).unwrap()]).unwrap(),
            values,
        )
        .unwrap();
        let (captured, _) = Graph::capture([("x", input.clone())], |inputs| {
            Ok(vec![
                inputs[0].argmax(0)?,
                inputs[0].argmin(0)?,
                inputs[0].max_reduce(&[0], false)?,
                inputs[0].min_reduce(&[0], false)?,
            ])
        })
        .unwrap();
        let mut compiled = cuda::compile_graph(&captured, &GraphCudaOptions::default()).unwrap();
        let replayed = compiled
            .run(&HashMap::from([("x".into(), input.clone())]))
            .unwrap();
        assert_eq!(replayed[0].to_vec::<i64>().unwrap(), [expected]);
        assert_eq!(replayed[1].to_vec::<i64>().unwrap(), [expected]);
        let outputs = captured.run(&HashMap::from([("x".into(), input)])).unwrap();
        assert_eq!(outputs[0].to_vec::<i64>().unwrap(), [expected]);
        assert_eq!(outputs[1].to_vec::<i64>().unwrap(), [expected]);
        assert!(outputs[2].to_vec::<f32>().unwrap()[0].is_nan());
        assert!(outputs[3].to_vec::<f32>().unwrap()[0].is_nan());
    }
}

#[cfg(all(feature = "testing", feature = "cuda"))]
#[test]
fn test_cuda_comparisons_replay_bool_outputs_and_expand_replays_as_a_view() {
    let _lock = device_lock(0).expect("the CUDA contract tests require device 0");
    let input = Tensor::from_vec(
        Device::Cuda(0),
        Shape::create([3]).unwrap(),
        vec![1.0_f32, 2.0, 3.0],
    )
    .unwrap();
    let (captured, _) = Graph::capture([("x", input.clone())], |inputs| {
        Ok(vec![inputs[0].less(&inputs[0].add_scalar(1.0)?)?])
    })
    .unwrap();
    let outputs = cuda::compile_graph(&captured, &GraphCudaOptions::default())
        .unwrap()
        .run(&HashMap::from([("x".into(), input.clone())]))
        .unwrap();
    assert_eq!(outputs[0].to_vec::<bool>().unwrap(), [true, true, true]);
    let (expanded, _) = Graph::capture([("x", input.clone())], |inputs| {
        Ok(vec![inputs[0].expand(&Shape::create([2, 3])?)?])
    })
    .unwrap();
    let outputs = cuda::compile_graph(&expanded, &GraphCudaOptions::default())
        .unwrap()
        .run(&HashMap::from([("x".into(), input)]))
        .unwrap();
    assert_eq!(
        outputs[0].to_vec::<f32>().unwrap(),
        [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    );
}

#[cfg(all(feature = "testing", feature = "cuda"))]
#[test]
fn test_cuda_host_readback_and_moves_preserve_strided_fp64_and_bool_values() {
    let _lock = device_lock(0).expect("the CUDA contract tests require device 0");
    let input = Tensor::from_vec(
        Device::Cuda(0),
        Shape::create([2, 3]).unwrap(),
        vec![1.000_000_000_000_000_2_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap()
    .transpose(0, 1)
    .unwrap()
    .narrow(0, 1, 2)
    .unwrap();
    assert_eq!(input.to_vec::<f64>().unwrap(), [2.0, 5.0, 3.0, 6.0]);
    let moved = input.to_device(Device::Cpu).unwrap();
    assert_eq!(moved.data_type(), input.data_type());
    assert_eq!(moved.to_bytes().unwrap(), input.to_bytes().unwrap());
    let scalar = Tensor::scalar(Device::Cuda(0), 1.000_000_000_000_000_2_f64).unwrap();
    assert_eq!(
        scalar
            .to_device(Device::Cpu)
            .unwrap()
            .to_vec::<f64>()
            .unwrap()[0]
            .to_bits(),
        1.000_000_000_000_000_2_f64.to_bits()
    );
    let mask = Tensor::scalar(Device::Cuda(0), true)
        .unwrap()
        .expand(Shape::create([2, 3]).unwrap())
        .unwrap();
    assert_eq!(mask.to_vec::<bool>().unwrap(), [true; 6]);
    assert_eq!(
        mask.to_device(Device::Cpu)
            .unwrap()
            .to_vec::<bool>()
            .unwrap(),
        [true; 6]
    );
}

#[cfg(all(feature = "testing", feature = "cuda"))]
#[test]
fn test_cuda_empty_reductions_replay_with_the_original_input_signature() {
    let _lock = device_lock(0).expect("the CUDA contract tests require device 0");
    let input = Tensor::from_vec(
        Device::Cuda(0),
        Shape::create([2, 0]).unwrap(),
        Vec::<f32>::new(),
    )
    .unwrap();
    let (captured, _) = Graph::capture([("x", input.clone())], |inputs| {
        Ok(vec![
            inputs[0].sum(&[1], false)?,
            inputs[0].product(&[1], false)?,
            inputs[0].mean(&[1], false)?,
        ])
    })
    .unwrap();
    let mut compiled = cuda::compile_graph(&captured, &GraphCudaOptions::default()).unwrap();
    let bindings = HashMap::from([("x".into(), input.clone())]);
    for _ in 0..2 {
        let outputs = compiled.run(&bindings).unwrap();
        assert_eq!(outputs[0].to_vec::<f32>().unwrap(), [0.0; 2]);
        assert_eq!(outputs[1].to_vec::<f32>().unwrap(), [1.0; 2]);
        assert!(
            outputs[2]
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|value| value.is_nan())
        );
    }
    assert!(input.max_reduce(&[1], false).is_err());
    assert!(input.min_reduce(&[1], false).is_err());
    let wrong = Tensor::from_vec(
        Device::Cuda(0),
        Shape::create([2, 1]).unwrap(),
        vec![0.0_f32; 2],
    )
    .unwrap();
    assert!(compiled.run(&HashMap::from([("x".into(), wrong)])).is_err());
}

fn assert_rank_zero_dimension_operations(device: Device) {
    let scalar = Tensor::scalar(device, 2.5_f32).unwrap();
    for dim in [0, -1] {
        let softmax = scalar.softmax(dim).unwrap();
        assert_eq!(softmax.shape().dimensions(), &[] as &[i64]);
        assert_eq!(softmax.to_vec::<f32>().unwrap(), [1.0]);
        assert_eq!(
            scalar.log_softmax(dim).unwrap().to_vec::<f32>().unwrap(),
            [0.0]
        );
        for index in [scalar.argmax(dim).unwrap(), scalar.argmin(dim).unwrap()] {
            assert_eq!(index.shape().dimensions(), &[] as &[i64]);
            assert_eq!(index.to_vec::<i64>().unwrap(), [0]);
        }
    }
    assert!(scalar.softmax(1).is_err());
    assert!(scalar.argmax(2).is_err());

    let (captured, _) = Graph::capture([("x", scalar.clone())], |inputs| {
        Ok(vec![inputs[0].softmax(0)?, inputs[0].argmax(-1)?])
    })
    .unwrap();
    let outputs = captured
        .run(&HashMap::from([("x".into(), scalar)]))
        .unwrap();
    assert_eq!(outputs[0].to_vec::<f32>().unwrap(), [1.0]);
    assert_eq!(outputs[1].to_vec::<i64>().unwrap(), [0]);
}

#[test]
fn test_rank_zero_inputs_reach_softmax_and_arg_reduction_kernels() {
    assert_rank_zero_dimension_operations(Device::Cpu);
}

#[cfg(all(feature = "testing", feature = "cuda"))]
#[test]
fn test_cuda_rank_zero_inputs_reach_softmax_and_arg_reduction_kernels() {
    let _lock = device_lock(0).expect("the CUDA contract tests require device 0");
    assert_rank_zero_dimension_operations(Device::Cuda(0));
}

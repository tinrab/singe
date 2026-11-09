use singe::{
    Device, Shape, Tensor,
    graph::{Graph, GraphPrepareOptions},
};

#[test]
fn test_owned_host_vectors_become_cpu_storage_without_copying() {
    let values = vec![1_f32, 2., 3., 4.];
    let pointer = values.as_ptr().addr();
    let tensor = Tensor::from_vec(Device::Cpu, Shape::create([2, 2]).unwrap(), values).unwrap();
    assert_eq!(tensor.storage().data_pointer_identity(), Some(pointer));
    assert_eq!(tensor.to_vec::<f32>().unwrap(), [1., 2., 3., 4.]);
}

#[test]
fn test_spare_vector_capacity_is_retained_but_never_part_of_the_readable_span() {
    let mut values = Vec::with_capacity(1024);
    values.extend([1_f32, 2., 3., 4.]);
    let pointer = values.as_ptr().addr();
    let capacity = values.capacity() * size_of::<f32>();
    let tensor = Tensor::from_vec(Device::Cpu, Shape::create([4]).unwrap(), values).unwrap();
    assert_eq!(tensor.storage().data_pointer_identity(), Some(pointer));
    assert_eq!(tensor.storage().byte_size(), 16);
    assert_eq!(tensor.storage().allocation_byte_size(), capacity);
    assert_eq!(tensor.to_vec::<f32>().unwrap(), [1., 2., 3., 4.]);
    let mut graph = Graph::new();
    let constant = graph.add_constant(tensor);
    graph.set_outputs(vec![constant]).unwrap();
    let executor = graph
        .prepare(GraphPrepareOptions::new(Device::Cpu))
        .unwrap();
    assert!(executor.memory_stats().unwrap().bound_allocation_bytes() >= capacity);
}

#[test]
fn test_typed_activation_iteration_preserves_offsets_transposes_and_broadcasts() {
    let tensor = Tensor::from_vec(
        Device::Cpu,
        Shape::create([3, 2]).unwrap(),
        vec![-9_f32, 8., -1., 2., -3., 4.],
    )
    .unwrap();
    let offset = tensor.narrow(0, 1, 2).unwrap();
    assert_eq!(
        offset.relu().unwrap().to_vec::<f32>().unwrap(),
        [0., 2., 0., 4.]
    );
    assert_eq!(
        offset
            .transpose(0, 1)
            .unwrap()
            .relu()
            .unwrap()
            .to_vec::<f32>()
            .unwrap(),
        [0., 0., 2., 4.]
    );
    let broadcast = offset
        .narrow(0, 0, 1)
        .unwrap()
        .expand(Shape::create([3, 2]).unwrap())
        .unwrap();
    assert_eq!(
        broadcast.relu().unwrap().to_vec::<f32>().unwrap(),
        [0., 2., 0., 2., 0., 2.]
    );
}

#[cfg(feature = "testing")]
#[test]
fn test_contiguous_cpu_activations_do_not_stage_bytes() {
    use singe::storage::cpu::CpuStorage;

    let input = Tensor::from_vec(
        Device::Cpu,
        Shape::create([4]).unwrap(),
        vec![-1_f32, 2., -3., 4.],
    )
    .unwrap();
    CpuStorage::reset_copy_stats();
    let output = input.relu().unwrap().sigmoid().unwrap();
    let copies = CpuStorage::copy_stats();
    assert_eq!(copies.imports, 0);
    assert_eq!(copies.exports, 0);
    assert_eq!(copies.clones, 0);
    assert_eq!(output.shape(), input.shape());
}

#[test]
fn test_interpreted_liveness_preserves_early_aliases_duplicate_outputs_and_prior_runs() {
    check_interpreted_liveness(Device::Cpu);
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_eager_liveness_preserves_completed_outputs_across_later_runs() {
    check_interpreted_liveness(Device::Cuda(0));
}

fn check_interpreted_liveness(device: Device) {
    use std::collections::HashMap;

    use singe::graph::{Graph, GraphExecutionMode, GraphPrepareOptions};

    let input = Tensor::from_vec(
        Device::Cpu,
        Shape::create([4]).unwrap(),
        vec![-1_f32, 2., -3., 4.],
    )
    .unwrap();
    let (captured, _) = Graph::capture([("x", input.clone())], |inputs| {
        let first = inputs[0].relu()?;
        let alias = first.narrow(0, 1, 2)?;
        let mut value = first;
        for _ in 0..32 {
            value = value.sigmoid()?.tanh()?;
        }
        Ok(vec![alias, value.clone(), value])
    })
    .unwrap();
    let mut executor = captured
        .prepare(GraphPrepareOptions::new(device).with_execution_mode(GraphExecutionMode::Eager))
        .unwrap();
    let first = executor
        .run(&HashMap::from([(
            "x".into(),
            input.to_device(device).unwrap(),
        )]))
        .unwrap();
    let stats = executor.last_run_stats().unwrap();
    assert!(stats.peak_live_values() <= 5, "{stats:?}");
    assert!(stats.released_values() >= 64, "{stats:?}");
    let second = executor
        .run(&HashMap::from([(
            "x".into(),
            Tensor::full(device, Shape::create([4]).unwrap(), 9_f32).unwrap(),
        )]))
        .unwrap();
    assert_eq!(first[0].to_vec::<f32>().unwrap(), [2., 0.]);
    assert_eq!(second[0].to_vec::<f32>().unwrap(), [9., 9.]);
    assert_eq!(first[1].to_bytes().unwrap(), first[2].to_bytes().unwrap());
}

use std::{collections::HashMap, slice};

use crate::{
    backend::{
        cuda::{self, compiled::plan::memory::GraphMemoryPlanner, runtime::runtime_for_device},
        registry::Operation,
    },
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    graph::{
        Graph, GraphCudaOptions, GraphOperationOutput, GraphOperationRequest, GraphPrepareOptions,
    },
    shape::Shape,
    tensor::Tensor,
    testing::{maybe_lock_device, skip_if_stub},
};

#[test]
fn test_cuda_compiled_graph_uses_memset_node_for_zero_inner_matmul() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let lhs_shape = Shape::create([2, 0]).unwrap();
    let rhs_shape = Shape::create([0, 3]).unwrap();
    let output_shape = Shape::create([2, 3]).unwrap();
    let Some(lhs) = skip_if_stub(Tensor::zeroed_on(
        Device::Cuda(0),
        DataType::F32,
        &lhs_shape,
    )) else {
        return;
    };
    let Some(rhs) = skip_if_stub(Tensor::zeroed_on(
        Device::Cuda(0),
        DataType::F32,
        &rhs_shape,
    )) else {
        return;
    };

    let mut graph = Graph::new();
    let lhs_id = graph.add_input("lhs", DataType::F32, lhs_shape, Device::Cuda(0));
    let rhs_id = graph.add_input("rhs", DataType::F32, rhs_shape, Device::Cuda(0));
    let output = graph
        .add_operation_request(GraphOperationRequest::binary(
            None,
            Operation::MatMul,
            [lhs_id, rhs_id],
            GraphOperationOutput::new(DataType::F32, output_shape.clone(), Device::Cuda(0)),
        ))
        .unwrap();
    graph.set_outputs(vec![output]).unwrap();

    let Some(_) = skip_if_stub(runtime_for_device(Device::Cuda(0))) else {
        return;
    };
    let mut compiled = cuda::compile_graph(&graph, &GraphCudaOptions::default()).unwrap();
    assert_eq!(compiled.stats().memset_nodes(), 1);
    assert_eq!(compiled.stats().native_provider_nodes(), 1);

    let inputs = HashMap::from([("lhs".to_string(), lhs), ("rhs".to_string(), rhs)]);
    let outputs = compiled.run(&inputs).unwrap();

    assert_eq!(outputs[0].shape().dimensions(), output_shape.dimensions());
    assert_eq!(outputs[0].to_vec::<f32>().unwrap(), vec![0.0; 6]);
}

#[test]
fn test_cuda_compiled_graph_reuses_released_buffers_for_other_data_types() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let values = (0..64).map(|value| value as f32 / 64.0).collect::<Vec<_>>();
    let Some(x) = skip_if_stub(Tensor::from_vec(
        Device::Cuda(0),
        Shape::create([64]).unwrap(),
        values,
    )) else {
        return;
    };
    // A scan never fuses into the sum, so its buffer outlives it.
    let function = |x: &Tensor| x.cumsum(0)?.sum(&[0], false)?.cast(DataType::F16);
    let expected = function(&x).unwrap();
    let (captured, _) =
        Graph::capture([("x", x.clone())], |inputs| Ok(vec![function(&inputs[0])?])).unwrap();

    // The scan's F32 buffer is released after the sum and holds the F16 output.
    let mut compiled = cuda::compile_graph(&captured, &GraphCudaOptions::default()).unwrap();
    assert_eq!(compiled.stats().reused_buffers(), 1);
    let outputs = compiled.run(&HashMap::from([("x".to_owned(), x)])).unwrap();
    assert_eq!(outputs[0].data_type(), DataType::F16);
    assert_eq!(
        outputs[0]
            .cast(DataType::F32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap(),
        expected
            .cast(DataType::F32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap()
    );
}

#[test]
fn test_cuda_graph_memory_reuse_waits_for_every_reader_of_the_released_buffer() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let Some(context) = skip_if_stub(cuda::execution_context(Device::Cuda(0))) else {
        return;
    };
    let shape = Shape::create([4]).unwrap();
    let output = || GraphOperationOutput::new(DataType::F32, shape.clone(), Device::Cuda(0));
    let mut graph = Graph::new();
    let x = graph.add_input("x", DataType::F32, shape.clone(), Device::Cuda(0));
    let unary = |graph: &mut Graph, operation, input| {
        graph
            .add_operation_request(GraphOperationRequest::unary(
                None,
                operation,
                input,
                output(),
            ))
            .unwrap()
    };
    let shared = unary(&mut graph, Operation::Exp, x);
    let first = unary(&mut graph, Operation::Neg, shared);
    let second = unary(&mut graph, Operation::Sin, shared);
    let sum = graph
        .add_operation_request(GraphOperationRequest::binary(
            None,
            Operation::Add,
            [first, second],
            output(),
        ))
        .unwrap();
    let result = unary(&mut graph, Operation::Exp, sum);
    graph.set_outputs(vec![result]).unwrap();
    let order = [shared, first, second, sum, result];

    let mut planner = GraphMemoryPlanner::create(&graph, &order).unwrap();
    let allocator = context.allocator();
    let mut native = context.runtime().context().create_graph().unwrap();
    let mut values = HashMap::new();
    let shared_value = planner.allocate(allocator, &graph, shared).unwrap();
    let shared_node = native.add_empty_node(&[]).unwrap();
    values.insert(shared, shared_value.into_value());
    let first_node = native
        .add_empty_node(slice::from_ref(&shared_node))
        .unwrap();
    planner.release_dependencies(first, 1, &values, &first_node);
    let second_node = native.add_empty_node(&[shared_node]).unwrap();
    planner.release_dependencies(second, 2, &values, &second_node);

    // `sum` reuses the buffer of `shared`, which both readers may still be reading.
    let reused = planner.allocate(allocator, &graph, sum).unwrap();
    assert_eq!(reused.dependencies(), vec![first_node, second_node]);
}

#[test]
fn test_cuda_compiled_graph_reads_inputs_bound_in_place_without_copying() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let shape = Shape::create([4]).unwrap();
    let Some(x) = skip_if_stub(Tensor::from_vec(
        device,
        &shape,
        vec![1.0_f32, 2.0, 3.0, 4.0],
    )) else {
        return;
    };
    let y = Tensor::from_vec(device, &shape, vec![10.0_f32, 20.0, 30.0, 40.0]).unwrap();
    let (graph, _) = Graph::capture([("x", x.clone()), ("y", y.clone())], |inputs| {
        Ok(vec![inputs[0].add(&inputs[1])?])
    })
    .unwrap();
    let inputs = HashMap::from([("x".to_owned(), x.clone()), ("y".to_owned(), y)]);
    let options = GraphCudaOptions::new().with_inputs_in_place([("x".to_owned(), x.clone())]);
    let mut compiled = cuda::compile_graph(&graph, &options).unwrap();
    assert_eq!(
        compiled.run(&inputs).unwrap()[0].to_vec::<f32>().unwrap(),
        [11.0, 22.0, 33.0, 44.0]
    );
    // The graph reads `x` where it is, so a write into it shows in the next replay.
    x.assign(&Tensor::from_vec(device, &shape, vec![5.0_f32, 6.0, 7.0, 8.0]).unwrap())
        .unwrap();
    assert_eq!(
        compiled.run(&inputs).unwrap()[0].to_vec::<f32>().unwrap(),
        [15.0, 26.0, 37.0, 48.0]
    );
    // Another tensor for `x` would be silently ignored, so replay rejects it.
    let mut replaced = inputs.clone();
    replaced.insert(
        "x".to_owned(),
        Tensor::from_vec(device, &shape, vec![0.0_f32; 4]).unwrap(),
    );
    assert!(compiled.run(&replaced).is_err());
}

#[test]
fn test_prepared_executors_read_inputs_bound_in_place() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let shape = Shape::create([4]).unwrap();
    let Some(x) = skip_if_stub(Tensor::from_vec(
        device,
        &shape,
        vec![1.0_f32, 2.0, 3.0, 4.0],
    )) else {
        return;
    };
    let y = Tensor::from_vec(device, &shape, vec![10.0_f32, 20.0, 30.0, 40.0]).unwrap();
    let (graph, _) = Graph::capture([("x", x.clone()), ("y", y.clone())], |inputs| {
        Ok(vec![inputs[0].mul(&inputs[1])?])
    })
    .unwrap();
    let cuda = GraphCudaOptions::new().with_inputs_in_place([("x".to_owned(), x.clone())]);
    let mut executor = graph
        .prepare(GraphPrepareOptions::new(device).with_cuda_options(cuda))
        .unwrap();
    let inputs = HashMap::from([("x".to_owned(), x.clone()), ("y".to_owned(), y)]);
    assert_eq!(
        executor.run(&inputs).unwrap()[0].to_vec::<f32>().unwrap(),
        [10.0, 40.0, 90.0, 160.0]
    );
    x.assign(&Tensor::from_vec(device, &shape, vec![2.0_f32; 4]).unwrap())
        .unwrap();
    assert_eq!(
        executor.run(&inputs).unwrap()[0].to_vec::<f32>().unwrap(),
        [20.0, 40.0, 60.0, 80.0]
    );
    let mut replaced = inputs.clone();
    replaced.insert(
        "x".to_owned(),
        Tensor::from_vec(device, &shape, vec![0.0_f32; 4]).unwrap(),
    );
    assert!(executor.run(&replaced).is_err());
}

#[test]
fn test_executor_instances_read_their_own_inputs_in_place() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let shape = Shape::create([4]).unwrap();
    let Some(x) = skip_if_stub(Tensor::from_vec(
        device,
        &shape,
        vec![1.0_f32, 2.0, 3.0, 4.0],
    )) else {
        return;
    };
    let worker_x = Tensor::from_vec(device, &shape, vec![3.0_f32; 4]).unwrap();
    let y = Tensor::from_vec(device, &shape, vec![10.0_f32, 20.0, 30.0, 40.0]).unwrap();
    let (graph, _) = Graph::capture([("x", x.clone()), ("y", y.clone())], |inputs| {
        Ok(vec![inputs[0].mul(&inputs[1])?])
    })
    .unwrap();
    let cuda = GraphCudaOptions::new().with_inputs_in_place([("x".to_owned(), x.clone())]);
    let mut executor = graph
        .prepare(GraphPrepareOptions::new(device).with_cuda_options(cuda))
        .unwrap();
    let mut worker = executor
        .execution_plan()
        .create_instance_with_inputs_in_place([("x".to_owned(), worker_x.clone())])
        .unwrap();
    let inputs = HashMap::from([("x".to_owned(), x), ("y".to_owned(), y.clone())]);
    let worker_inputs = HashMap::from([("x".to_owned(), worker_x.clone()), ("y".to_owned(), y)]);
    worker_x
        .assign(&Tensor::from_vec(device, &shape, vec![-1.0_f32; 4]).unwrap())
        .unwrap();
    assert_eq!(
        worker.run(&worker_inputs).unwrap()[0]
            .to_vec::<f32>()
            .unwrap(),
        [-10.0, -20.0, -30.0, -40.0]
    );
    assert_eq!(
        executor.run(&inputs).unwrap()[0].to_vec::<f32>().unwrap(),
        [10.0, 40.0, 90.0, 160.0]
    );
    // Each instance is bound to its own tensor, so neither takes the other's.
    assert!(worker.run(&inputs).is_err());
    assert!(executor.run(&worker_inputs).is_err());
}

#[test]
fn test_prepared_executors_read_inputs_in_place_across_specializations() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let shape = Shape::create([4]).unwrap();
    let Some(x) = skip_if_stub(Tensor::from_vec(
        device,
        &shape,
        vec![1.0_f32, 2.0, 3.0, 4.0],
    )) else {
        return;
    };
    let y = Tensor::from_vec(device, &shape, vec![10.0_f32; 4]).unwrap();
    let (mut graph, _) = Graph::capture([("x", x.clone()), ("y", y.clone())], |inputs| {
        Ok(vec![inputs[0].add(&inputs[1])?])
    })
    .unwrap();
    for name in ["x", "y"] {
        graph
            .add_input_dimension_guard(name, 0, "n", Some(1), Some(8))
            .unwrap();
    }
    let cuda = GraphCudaOptions::new().with_inputs_in_place([("x".to_owned(), x.clone())]);
    let mut executor = graph
        .prepare(GraphPrepareOptions::new(device).with_cuda_options(cuda))
        .unwrap();
    let inputs = HashMap::from([("x".to_owned(), x.clone()), ("y".to_owned(), y)]);
    assert_eq!(
        executor.run(&inputs).unwrap()[0].to_vec::<f32>().unwrap(),
        [11.0, 12.0, 13.0, 14.0]
    );
    // A specialization the bound tensor does not fit copies whatever tensor the step passes.
    let longer = Shape::create([6]).unwrap();
    let other = HashMap::from([
        (
            "x".to_owned(),
            Tensor::from_vec(device, &longer, vec![1.0_f32; 6]).unwrap(),
        ),
        (
            "y".to_owned(),
            Tensor::from_vec(device, &longer, vec![2.0_f32; 6]).unwrap(),
        ),
    ]);
    assert_eq!(
        executor.run(&other).unwrap()[0].to_vec::<f32>().unwrap(),
        [3.0; 6]
    );
    // Back on the first specialization, the graph reads the bound tensor where it is again.
    x.assign(&Tensor::from_vec(device, &shape, vec![5.0_f32; 4]).unwrap())
        .unwrap();
    assert_eq!(
        executor.run(&inputs).unwrap()[0].to_vec::<f32>().unwrap(),
        [15.0; 4]
    );
    assert_eq!(executor.cache_stats().misses(), 2);
    let mut replaced = inputs.clone();
    replaced.insert(
        "x".to_owned(),
        Tensor::from_vec(device, &shape, vec![0.0_f32; 4]).unwrap(),
    );
    assert!(executor.run(&replaced).is_err());
}

// A written input read in place is copied back by the graph's own last node, and every replay leaves it updated.
#[test]
fn test_compiled_graphs_write_inputs_they_read_in_place() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let rows = Shape::create([2, 3]).unwrap();
    let row = |values: Vec<f32>| {
        Tensor::from_vec(device, Shape::create([1, 3]).unwrap(), values).unwrap()
    };
    let Some(cache) = skip_if_stub(Tensor::from_vec(device, &rows, vec![0.0_f32; 6])) else {
        return;
    };
    let (graph, _) = Graph::capture(
        [("cache", cache), ("row", row(vec![1., 2., 3.]))],
        |inputs| {
            inputs[0].narrow(0, 1, 1)?.assign(&inputs[1])?;
            Ok(vec![inputs[0].add(&inputs[0])?])
        },
    )
    .unwrap();
    let replayed = Tensor::from_vec(device, &rows, vec![5.0_f32; 6]).unwrap();
    let options =
        GraphCudaOptions::new().with_inputs_in_place([("cache".to_owned(), replayed.clone())]);
    let compiled = cuda::compile_graph(&graph, &options).unwrap();
    assert!(compiled.writes_input_in_graph("cache"));
    assert!(compiled.stats().memcpy_nodes() >= 1);
    let mut executor = graph
        .prepare(GraphPrepareOptions::new(device).with_cuda_options(options))
        .unwrap();
    for (values, doubled) in [
        (vec![7., 8., 9.], [10., 10., 10., 14., 16., 18.]),
        (vec![1., 1., 1.], [10., 10., 10., 2., 2., 2.]),
    ] {
        let outputs = executor
            .run(&HashMap::from([
                ("cache".to_owned(), replayed.clone()),
                ("row".to_owned(), row(values.clone())),
            ]))
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].to_vec::<f32>().unwrap(), doubled);
        let cache = replayed.to_vec::<f32>().unwrap();
        assert_eq!(cache[3..], values[..]);
    }
}

// A scatter whose result replaces an input bound in place writes that input where it is, and no replay copies it.
#[test]
fn test_scatters_write_inputs_bound_in_place_without_copies() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let tensor = |dimensions: &[i64], values: Vec<f32>| {
        Tensor::from_vec(device, Shape::create(dimensions).unwrap(), values)
    };
    let Some(cache) = skip_if_stub(tensor(&[3, 2], vec![0.0; 6])) else {
        return;
    };
    let position =
        |value: i64| Tensor::from_vec(device, Shape::create([1]).unwrap(), vec![value]).unwrap();
    let (graph, _) = Graph::capture(
        [
            ("cache", cache.clone()),
            ("row", tensor(&[1, 2], vec![1., 2.]).unwrap()),
            ("position", position(0)),
        ],
        |inputs| {
            inputs[0].assign(&inputs[0].index_copy(0, &inputs[2], &inputs[1])?)?;
            Ok(vec![inputs[0].sum(&[0], false)?])
        },
    )
    .unwrap();
    let options =
        GraphCudaOptions::new().with_inputs_in_place([("cache".to_owned(), cache.clone())]);
    let compiled = cuda::compile_graph(&graph, &options).unwrap();
    assert!(compiled.writes_input_in_graph("cache"));
    assert_eq!(compiled.stats().memcpy_nodes(), 0);
    let mut executor = graph
        .prepare(GraphPrepareOptions::new(device).with_cuda_options(options))
        .unwrap();
    for (slot, row, total, contents) in [
        (0, vec![1., 2.], [1., 2.], [1., 2., 0., 0., 0., 0.]),
        (2, vec![3., 4.], [4., 6.], [1., 2., 0., 0., 3., 4.]),
        (0, vec![5., 5.], [8., 9.], [5., 5., 0., 0., 3., 4.]),
    ] {
        let outputs = executor
            .run(&HashMap::from([
                ("cache".to_owned(), cache.clone()),
                ("row".to_owned(), tensor(&[1, 2], row).unwrap()),
                ("position".to_owned(), position(slot)),
            ]))
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].to_vec::<f32>().unwrap(), total);
        assert_eq!(cache.to_vec::<f32>().unwrap(), contents);
    }
}

// A scatter copies its input when anything else needs the old values, or the caller keeps the new ones.
#[test]
fn test_scatters_copy_inputs_something_else_needs() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let tensor = |dimensions: &[i64], values: Vec<f32>| {
        Tensor::from_vec(device, Shape::create(dimensions).unwrap(), values)
    };
    let Some(cache) = skip_if_stub(tensor(&[2], vec![1., 2.])) else {
        return;
    };
    let index = Tensor::from_vec(device, Shape::create([1]).unwrap(), vec![0_i64]).unwrap();
    let value = tensor(&[1], vec![9.]).unwrap();
    let named = [
        ("cache", cache.clone()),
        ("value", value.clone()),
        ("index", index.clone()),
    ];
    let options =
        GraphCudaOptions::new().with_inputs_in_place([("cache".to_owned(), cache.clone())]);
    let inputs = HashMap::from([
        ("cache".to_owned(), cache.clone()),
        ("value".to_owned(), value),
        ("index".to_owned(), index),
    ]);

    // The old values are read after the scatter ran.
    let (reads_old, _) = Graph::capture(named.clone(), |inputs| {
        let old = inputs[0].mul_scalar(2.0)?;
        inputs[0].assign(&inputs[0].index_copy(0, &inputs[2], &inputs[1])?)?;
        Ok(vec![inputs[0].add(&old)?])
    })
    .unwrap();
    cache.assign(&tensor(&[2], vec![1., 2.]).unwrap()).unwrap();
    let mut executor = reads_old
        .prepare(GraphPrepareOptions::new(device).with_cuda_options(options.clone()))
        .unwrap();
    cache.assign(&tensor(&[2], vec![1., 2.]).unwrap()).unwrap();
    assert_eq!(
        executor.run(&inputs).unwrap()[0].to_vec::<f32>().unwrap(),
        [11., 6.]
    );
    assert_eq!(cache.to_vec::<f32>().unwrap(), [9., 2.]);

    // The result the caller keeps must survive the next run's write.
    let (keeps, _) = Graph::capture(named, |inputs| {
        inputs[0].assign(&inputs[0].index_copy(0, &inputs[2], &inputs[1])?)?;
        Ok(vec![inputs[0].clone()])
    })
    .unwrap();
    let mut executor = keeps
        .prepare(GraphPrepareOptions::new(device).with_cuda_options(options))
        .unwrap();
    cache.assign(&tensor(&[2], vec![1., 2.]).unwrap()).unwrap();
    let kept = executor.run(&inputs).unwrap().remove(0);
    assert_eq!(kept.to_vec::<f32>().unwrap(), [9., 2.]);
    cache.assign(&tensor(&[2], vec![0., 0.]).unwrap()).unwrap();
    assert_eq!(kept.to_vec::<f32>().unwrap(), [9., 2.]);
}

// A scatter writes its base in place only when nothing else reads the base.
#[test]
fn test_scatters_write_bases_nothing_else_reads_in_place() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let shape = Shape::create([4]).unwrap();
    let Some(x) = skip_if_stub(Tensor::from_vec(
        device,
        &shape,
        vec![1.0_f32, 2.0, 3.0, 4.0],
    )) else {
        return;
    };
    let index = Tensor::from_vec(device, Shape::create([1]).unwrap(), vec![1_i64]).unwrap();
    let source = Tensor::from_vec(device, Shape::create([1]).unwrap(), vec![9.0_f32]).unwrap();
    let named = [
        ("x", x.clone()),
        ("index", index.clone()),
        ("source", source.clone()),
    ];
    let inputs = HashMap::from([
        ("x".to_owned(), x),
        ("index".to_owned(), index),
        ("source".to_owned(), source),
    ]);
    let (alone, _) = Graph::capture(named.clone(), |inputs| {
        let base = inputs[0].add_scalar(1.0)?;
        let scattered = base.index_copy(0, &inputs[1], &inputs[2])?;
        Ok(vec![scattered.mul_scalar(2.0)?])
    })
    .unwrap();
    let (shared, _) = Graph::capture(named, |inputs| {
        let base = inputs[0].add_scalar(1.0)?;
        let scattered = base.index_copy(0, &inputs[1], &inputs[2])?;
        Ok(vec![scattered.mul_scalar(2.0)?, base.mul_scalar(3.0)?])
    })
    .unwrap();
    let mut alone = cuda::compile_graph(&alone, &GraphCudaOptions::new()).unwrap();
    let mut shared = cuda::compile_graph(&shared, &GraphCudaOptions::new()).unwrap();
    for _ in 0..2 {
        assert_eq!(
            alone.run(&inputs).unwrap()[0].to_vec::<f32>().unwrap(),
            [4.0, 18.0, 8.0, 10.0]
        );
        let outputs = shared.run(&inputs).unwrap();
        assert_eq!(outputs[0].to_vec::<f32>().unwrap(), [4.0, 18.0, 8.0, 10.0]);
        assert_eq!(outputs[1].to_vec::<f32>().unwrap(), [6.0, 9.0, 12.0, 15.0]);
    }
}

// Specializations of one program share their replay buffers, and results stay right as they alternate.
#[test]
fn test_prepared_specializations_share_one_arena() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let tensors = |count: i64| {
        let shape = Shape::create([count]).unwrap();
        let x = (0..count).map(|index| index as f32).collect::<Vec<_>>();
        HashMap::from([
            ("x".to_owned(), Tensor::from_vec(device, &shape, x).unwrap()),
            (
                "y".to_owned(),
                Tensor::from_vec(
                    device,
                    &shape,
                    vec![2.0_f32; usize::try_from(count).unwrap()],
                )
                .unwrap(),
            ),
        ])
    };
    let long = tensors(8);
    let Some(x) = skip_if_stub(Ok::<_, Error>(long["x"].clone())) else {
        return;
    };
    let (mut graph, _) = Graph::capture([("x", x), ("y", long["y"].clone())], |inputs| {
        Ok(vec![
            inputs[0]
                .add(&inputs[1])?
                .softmax(-1)?
                .mul(&inputs[1])?
                .softmax(-1)?,
        ])
    })
    .unwrap();
    for name in ["x", "y"] {
        graph
            .add_input_dimension_guard(name, 0, "n", Some(1), Some(8))
            .unwrap();
    }
    let mut executor = graph.prepare(GraphPrepareOptions::new(device)).unwrap();
    let expected = |inputs: &HashMap<String, Tensor>| {
        let (x, y) = (&inputs["x"], &inputs["y"]);
        x.add(y)
            .unwrap()
            .softmax(-1)
            .unwrap()
            .mul(y)
            .unwrap()
            .softmax(-1)
            .unwrap()
            .to_vec::<f32>()
            .unwrap()
    };
    let short = tensors(4);
    assert_eq!(
        executor.run(&long).unwrap()[0].to_vec::<f32>().unwrap(),
        expected(&long)
    );
    let first = executor
        .memory_stats()
        .unwrap()
        .active_replay_allocation_bytes();
    assert!(first > 0);
    assert_eq!(
        executor.run(&short).unwrap()[0].to_vec::<f32>().unwrap(),
        expected(&short)
    );
    assert_eq!(executor.cache_stats().misses(), 2);
    // The shorter specialization fits entirely in the longer one's buffers.
    let stats = executor.memory_stats().unwrap();
    assert_eq!(
        stats.active_replay_allocation_bytes() + stats.inactive_replay_allocation_bytes(),
        first
    );
    for inputs in [&long, &short, &long] {
        assert_eq!(
            executor.run(inputs).unwrap()[0].to_vec::<f32>().unwrap(),
            expected(inputs)
        );
    }
}

// A copy only strided-reading kernels read is removed, and a copy a view or the caller needs stays.
#[test]
fn test_contiguous_copies_are_elided_where_readers_take_strides() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let Some(x) = skip_if_stub(Tensor::from_vec(
        device,
        Shape::create([2, 3]).unwrap(),
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
    )) else {
        return;
    };
    let copies = |graph: &Graph| {
        let compiled = graph.compile().unwrap();
        compiled
            .ir()
            .operations()
            .iter()
            .filter(|operation| operation.opcode() == Some(Operation::Contiguous))
            .count()
    };
    let (elided, _) = Graph::capture([("x", x.clone())], |inputs| {
        let copy = inputs[0].transpose(0, 1)?.contiguous()?;
        Ok(vec![copy.relu()?, copy.sum(&[1], false)?])
    })
    .unwrap();
    assert_eq!(copies(&elided), 0);
    let (kept, _) = Graph::capture([("x", x.clone())], |inputs| {
        let copy = inputs[0].transpose(0, 1)?.contiguous()?;
        Ok(vec![copy.reshape([6])?, copy.relu()?])
    })
    .unwrap();
    assert_eq!(copies(&kept), 1);
    let (returned, _) = Graph::capture([("x", x.clone())], |inputs| {
        Ok(vec![inputs[0].transpose(0, 1)?.contiguous()?])
    })
    .unwrap();
    assert_eq!(copies(&returned), 1);
    let mut compiled = cuda::compile_graph(&elided, &GraphCudaOptions::new()).unwrap();
    let outputs = compiled.run(&HashMap::from([("x".to_owned(), x)])).unwrap();
    assert_eq!(
        outputs[0].to_vec::<f32>().unwrap(),
        [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
    assert_eq!(outputs[1].to_vec::<f32>().unwrap(), [5.0, 7.0, 9.0]);
}

// A convolution only another convolution reads writes channels-last, and the chain computes what eager calls do.
#[test]
fn test_convolution_chains_run_channels_last() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let values = |dimensions: &[i64], seed: i64| {
        let shape = Shape::create(dimensions).unwrap();
        let count = i64::try_from(shape.element_count()).unwrap();
        Tensor::from_vec(
            device,
            shape,
            (0..count)
                .map(|index| ((index * 37 + seed * 11) % 41) as f32 / 40.0 - 0.5)
                .collect::<Vec<_>>(),
        )
    };
    let Some(x) = skip_if_stub(values(&[2, 3, 16, 16], 1)) else {
        return;
    };
    let first = values(&[8, 3, 3, 3], 2).unwrap();
    let second = values(&[4, 8, 3, 3], 3).unwrap();
    let chain = |inputs: &[Tensor]| -> Result<Vec<Tensor>> {
        let hidden = inputs[0].convolution2d(&inputs[1], None, (1, 1), (1, 1, 1, 1), (1, 1), 1)?;
        Ok(vec![hidden.convolution2d(
            &inputs[2],
            None,
            (1, 1),
            (0, 0, 0, 0),
            (1, 1),
            1,
        )?])
    };
    let named = [
        ("x", x.clone()),
        ("first", first.clone()),
        ("second", second.clone()),
    ];
    let (graph, eager) = Graph::capture(named, chain).unwrap();
    let mut compiled = cuda::compile_graph(&graph, &GraphCudaOptions::new()).unwrap();
    let strided = compiled
        .ir()
        .values()
        .iter()
        .filter(|value| !value.layout().is_contiguous())
        .count();
    assert_eq!(strided, 1);
    let outputs = compiled
        .run(&HashMap::from([
            ("x".to_owned(), x),
            ("first".to_owned(), first),
            ("second".to_owned(), second),
        ]))
        .unwrap();
    let (actual, expected) = (
        outputs[0].to_vec::<f32>().unwrap(),
        eager[0].to_vec::<f32>().unwrap(),
    );
    for (actual, expected) in actual.iter().zip(&expected) {
        assert!((actual - expected).abs() <= 1e-4, "{actual} != {expected}");
    }
}

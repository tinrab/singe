use crate::{
    backend::cuda::cutensor::tests::compiled::{report::assert_cutensor_contraction_report, *},
    graph::GraphCudaOptions,
};

#[test]
fn test_cuda_compiled_graph_replays_cutensor_contraction() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let a_shape = Shape::create([2, 3]).unwrap();
    let b_shape = Shape::create([3, 2]).unwrap();
    let c_shape = Shape::create([2, 2]).unwrap();
    let Some(a) = maybe_cuda_f32_tensor(&a_shape, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]) else {
        return;
    };
    let b = f32_tensor(&b_shape, vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let c = f32_tensor(&c_shape, vec![1.0_f32, 2.0, 3.0, 4.0]);
    let output = cuda_f32_output(&c_shape);
    let (graph, _) = contraction_graph(a_shape, b_shape, c_shape, &output, 0.5, 2.0);

    let Some(runtime) = skip_if_stub(runtime_for_device(cuda_device())) else {
        return;
    };
    runtime.cutensor_contraction_plans().lock().unwrap().clear();

    let Some(mut compiled) =
        skip_if_stub(cuda::compile_graph(&graph, &GraphCudaOptions::default()))
    else {
        return;
    };
    assert_cutensor_contraction_report(&compiled);
    let first_plan = {
        let cache = runtime.cutensor_contraction_plans().lock().unwrap();
        assert_eq!(cache.len(), 1);
        Arc::clone(cache.values().next().unwrap().plan())
    };
    let Some(_second_compiled) =
        skip_if_stub(cuda::compile_graph(&graph, &GraphCudaOptions::default()))
    else {
        return;
    };
    let second_plan = {
        let cache = runtime.cutensor_contraction_plans().lock().unwrap();
        assert_eq!(cache.len(), 1);
        Arc::clone(cache.values().next().unwrap().plan())
    };
    assert!(Arc::ptr_eq(&first_plan, &second_plan));

    let outputs = compiled.run(&contraction_inputs(a, b, c)).unwrap();

    assert_close!(
        &outputs[0].to_vec::<f32>().unwrap(),
        &[31.0, 36.0, 75.5, 85.0],
    );

    cuda::clear_cache(cuda_device()).unwrap();
    assert!(
        runtime
            .cutensor_contraction_plans()
            .lock()
            .unwrap()
            .is_empty()
    );
}

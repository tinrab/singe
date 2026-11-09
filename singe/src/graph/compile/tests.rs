use std::{collections::HashMap, ptr};

#[cfg(feature = "cuda")]
use crate::backend::cuda::CudaStream;
use crate::{device::Device, graph::Graph, shape::Shape, tensor::Tensor};

fn matrix(values: &[f32]) -> Tensor {
    Tensor::from_vec(Device::Cpu, Shape::create([2, 2]).unwrap(), values.to_vec()).unwrap()
}

#[test]
fn test_compilation_lowers_once_and_interpretation_reuses_the_same_analysis() {
    let (captured, _) = Graph::capture([("x", matrix(&[1., 2., 3., 4.]))], |inputs| {
        Ok(vec![
            inputs[0]
                .add_scalar(2.)?
                .transpose(0, 1)?
                .reshape(vec![4])?,
        ])
    })
    .unwrap();
    let before = Graph::lowering_count_for_test();
    let compiled = captured.compile().unwrap();
    assert_eq!(Graph::lowering_count_for_test() - before, 1);
    let shared = compiled.clone();
    assert!(ptr::eq(compiled.ir(), shared.ir()));
    let first = compiled
        .run(&HashMap::from([("x".into(), matrix(&[1., 2., 3., 4.]))]))
        .unwrap();
    let second = shared
        .run(&HashMap::from([("x".into(), matrix(&[5., 6., 7., 8.]))]))
        .unwrap();
    assert_eq!(first[0].to_vec::<f32>().unwrap(), [3., 5., 4., 6.]);
    assert_eq!(second[0].to_vec::<f32>().unwrap(), [7., 9., 8., 10.]);
    assert_eq!(Graph::lowering_count_for_test() - before, 1);
}

#[cfg(feature = "cuda")]
#[test]
fn test_native_compilation_consumes_prepared_ir_without_lowering_the_construction_graph() {
    use crate::{
        backend::cuda::{
            self,
            compiled::plan::{cache_key::DynamicShapePolicy, lowering::LoweringPolicy},
        },
        graph::GraphCudaOptions,
    };

    let (captured, _) = Graph::capture([("x", matrix(&[1., 2., 3., 4.]))], |inputs| {
        Ok(vec![inputs[0].add_scalar(2.)?])
    })
    .unwrap();
    let graph = captured.placed_on(Device::Cuda(0)).unwrap();
    let compiled = graph.compile().unwrap();
    let shared = compiled.clone();
    let before = Graph::lowering_count_for_test();
    let mut native = cuda::compile(
        &compiled,
        &GraphCudaOptions::new()
            .with_dynamic_shape_policy(DynamicShapePolicy::exact_shape())
            .with_lowering_policy(LoweringPolicy::default()),
    )
    .unwrap();
    assert!(ptr::eq(native.ir(), shared.ir()));
    let stream = CudaStream::create(Device::Cuda(0)).unwrap();
    let inputs = HashMap::from([(
        "x".into(),
        matrix(&[1., 2., 3., 4.])
            .to_device(Device::Cuda(0))
            .unwrap(),
    )]);
    assert_eq!(
        native.run_on(&inputs, &stream).unwrap()[0]
            .to_vec::<f32>()
            .unwrap(),
        [3., 4., 5., 6.]
    );
    assert_eq!(Graph::lowering_count_for_test(), before);
}

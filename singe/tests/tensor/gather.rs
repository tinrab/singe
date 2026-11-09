use std::collections::HashMap;

#[cfg(feature = "cuda")]
use singe::graph::GraphExecutor;
use singe::{
    DataType, Device, Error, Shape, Tensor,
    graph::{Graph, GraphExecutionMode, GraphPrepareOptions},
};

fn assert_index_error(mut error: &Error) {
    while let Error::GraphSource { source, .. } = error {
        error = source;
    }
    assert!(matches!(error, Error::IndexOutOfBounds), "{error}");
}

fn exercise(device: Device) {
    let table = Tensor::from_vec(
        device,
        Shape::create([3, 4]).unwrap(),
        (0..12).map(|value| value as f32).collect(),
    )
    .unwrap()
    .transpose(0, 1)
    .unwrap();
    let indices = Tensor::from_vec(
        device,
        Shape::create([2, 2]).unwrap(),
        vec![-1_i64, 0, 1, -2],
    )
    .unwrap()
    .transpose(0, 1)
    .unwrap();
    let expected = [
        8_f32, 4., 0., 4., 9., 5., 1., 5., 10., 6., 2., 6., 11., 7., 3., 7.,
    ];
    let output = table.take(-1, &indices).unwrap();
    assert_eq!(output.shape().dimensions(), [4, 2, 2]);
    assert_eq!(output.to_vec::<f32>().unwrap(), expected);
    let (captured, _) = Graph::capture([("x", table.clone()), ("i", indices.clone())], |xs| {
        Ok(vec![xs[0].take(-1, &xs[1])?])
    })
    .unwrap();
    for mode in [GraphExecutionMode::Eager, GraphExecutionMode::Compiled] {
        let mut executor = captured
            .prepare(GraphPrepareOptions::new(device).with_execution_mode(mode))
            .unwrap();
        let outputs = executor
            .run(&HashMap::from([
                ("x".into(), table.clone()),
                ("i".into(), indices.clone()),
            ]))
            .unwrap();
        assert_eq!(outputs[0].to_vec::<f32>().unwrap(), expected);
        let bad = Tensor::from_vec(device, Shape::create([2, 2]).unwrap(), vec![0_i64, 1, 3, 0])
            .unwrap()
            .transpose(0, 1)
            .unwrap();
        assert_index_error(
            &executor
                .run(&HashMap::from([
                    ("x".into(), table.clone()),
                    ("i".into(), bad),
                ]))
                .unwrap_err(),
        );
        assert_eq!(outputs[0].to_vec::<f32>().unwrap(), expected);
    }
    let scalar = Tensor::from_vec(device, Shape::scalar(), vec![-1_i32]).unwrap();
    let output = table.take(0, &scalar).unwrap();
    assert_eq!(output.shape().dimensions(), [3]);
    assert_eq!(output.to_vec::<f32>().unwrap(), [3., 7., 11.]);
    let empty =
        Tensor::from_vec(device, Shape::create([0, 2]).unwrap(), Vec::<i64>::new()).unwrap();
    assert_eq!(
        table.take(0, &empty).unwrap().shape().dimensions(),
        [0, 2, 3]
    );
    for value in [i64::MIN, -4, 3, i64::MAX] {
        let indices = Tensor::from_vec(device, Shape::scalar(), vec![value]).unwrap();
        assert_index_error(&table.take(1, &indices).unwrap_err());
    }
    let negative = Tensor::from_vec(device, Shape::create([1]).unwrap(), vec![-1_i64]).unwrap();
    assert_index_error(&table.index_select(1, &negative).unwrap_err());
    assert_index_error(&table.embedding(&negative).unwrap_err());
}

fn exercise_storage(device: Device) {
    let indices =
        Tensor::from_vec(device, Shape::create([3]).unwrap(), vec![1_i64, -1, 0]).unwrap();
    for (dtype, source, expected) in [
        (
            DataType::F16,
            [0_u16, 0x8000, 1, 0x7e55, 0x3c00, 0xfc00],
            [1, 0x3c00, 0, 0x7e55, 0xfc00, 0x8000],
        ),
        (
            DataType::Bf16,
            [0_u16, 0x8000, 1, 0x7fc5, 0x3f80, 0xff80],
            [1, 0x3f80, 0, 0x7fc5, 0xff80, 0x8000],
        ),
    ] {
        let bytes = source
            .into_iter()
            .flat_map(u16::to_ne_bytes)
            .collect::<Vec<_>>();
        let expected = expected
            .into_iter()
            .flat_map(u16::to_ne_bytes)
            .collect::<Vec<_>>();
        let table = Tensor::from_bytes(device, dtype, Shape::create([3, 2]).unwrap(), &bytes)
            .unwrap()
            .transpose(0, 1)
            .unwrap();
        let (captured, _) = Graph::capture([("x", table.clone()), ("i", indices.clone())], |xs| {
            Ok(vec![xs[0].take(1, &xs[1])?])
        })
        .unwrap();
        assert_eq!(
            table.take(1, &indices).unwrap().to_bytes().unwrap(),
            expected
        );
        for mode in [GraphExecutionMode::Eager, GraphExecutionMode::Compiled] {
            let output = captured
                .prepare(GraphPrepareOptions::new(device).with_execution_mode(mode))
                .unwrap()
                .run(&HashMap::from([
                    ("x".into(), table.clone()),
                    ("i".into(), indices.clone()),
                ]))
                .unwrap();
            assert_eq!(output[0].to_bytes().unwrap(), expected);
        }
    }
    let source = Tensor::from_vec(
        device,
        Shape::create([3, 2]).unwrap(),
        (0..6).map(|value| (1_i64 << 60) + value * 3 + 1).collect(),
    )
    .unwrap()
    .narrow(0, 1, 2)
    .unwrap()
    .transpose(0, 1)
    .unwrap();
    let index = Tensor::from_vec(device, Shape::scalar(), vec![-1_i64]).unwrap();
    assert_eq!(
        source.take(1, &index).unwrap().to_vec::<i64>().unwrap(),
        [(1_i64 << 60) + 13, (1_i64 << 60) + 16]
    );
}

#[test]
fn test_gather_handles_axes_strides_scalars_and_runtime_bounds_on_cpu() {
    exercise(Device::Cpu);
    exercise_storage(Device::Cpu);
}

#[cfg(feature = "cuda")]
#[test]
fn test_gather_handles_axes_strides_scalars_and_runtime_bounds_on_cuda() {
    exercise(Device::Cuda(0));
    exercise_storage(Device::Cuda(0));
}

#[cfg(feature = "cuda")]
#[test]
fn test_native_gather_reports_bad_device_computed_indices_after_replay() {
    let table = Tensor::from_vec(
        Device::Cpu,
        Shape::create([3]).unwrap(),
        vec![1_f32, 2., 3.],
    )
    .unwrap();
    let indices =
        Tensor::from_vec(Device::Cpu, Shape::create([2]).unwrap(), vec![0_i64, 2]).unwrap();
    let (captured, _) = Graph::capture([("x", table), ("i", indices)], |xs| {
        Ok(vec![xs[0].take(0, &xs[1].copy()?)?])
    })
    .unwrap();
    let mut prepared = captured
        .prepare(GraphPrepareOptions::new(Device::Cuda(0)))
        .unwrap();
    let run = |prepared: &mut GraphExecutor, indices: Vec<i64>| {
        let inputs = HashMap::from([
            (
                "x".to_owned(),
                Tensor::from_vec(
                    Device::Cuda(0),
                    Shape::create([3]).unwrap(),
                    vec![1_f32, 2., 3.],
                )
                .unwrap(),
            ),
            (
                "i".to_owned(),
                Tensor::from_vec(Device::Cuda(0), Shape::create([2]).unwrap(), indices).unwrap(),
            ),
        ]);
        prepared.run(&inputs)
    };
    assert_eq!(
        run(&mut prepared, vec![0, 2]).unwrap()[0]
            .to_vec::<f32>()
            .unwrap(),
        [1., 3.]
    );
    let error = run(&mut prepared, vec![0, 5]).unwrap_err();
    assert!(matches!(error, Error::IndexOutOfBounds), "{error}");
}

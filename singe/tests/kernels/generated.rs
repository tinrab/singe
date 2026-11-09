#![cfg(all(feature = "testing", feature = "cuda"))]

use std::collections::HashMap;

use singe::{
    DataType, Device, Shape, Tensor,
    backend::{
        cuda::{CudaGraphProvider, LoweringPolicy},
        registry::{Operation, OperationAttributes},
    },
    dispatch::call_one,
    graph::{Graph, GraphCudaOptions, GraphPrepareOptions},
};
use singe_cuda::testing::device_lock;

fn bf16(device: Device, dimensions: &[i64]) -> Tensor {
    let count = dimensions.iter().product::<i64>();
    let values = (0..count)
        .map(|value| (value as f32).mul_add(0.25, -3.0))
        .collect::<Vec<_>>();
    Tensor::from_vec(device, Shape::create(dimensions).unwrap(), values)
        .unwrap()
        .cast(DataType::Bf16)
        .unwrap()
}

fn host_values(tensor: &Tensor) -> Vec<f32> {
    tensor
        .to_device(Device::Cpu)
        .unwrap()
        .cast(DataType::F32)
        .unwrap()
        .to_vec::<f32>()
        .unwrap()
}

#[test]
fn test_copies_of_misaligned_views_fall_back_to_element_copies() {
    let _lock = device_lock(0).unwrap();
    for start in [0, 1, 3] {
        let input = bf16(Device::Cuda(0), &[4, 16]).narrow(1, start, 8).unwrap();
        let expected = bf16(Device::Cpu, &[4, 16]).narrow(1, start, 8).unwrap();

        let copied = input.contiguous().unwrap();

        assert_eq!(
            host_values(&copied),
            host_values(&expected.contiguous().unwrap())
        );
    }
}

#[test]
fn test_compiled_graphs_copy_strided_views_without_cudnn() {
    let _lock = device_lock(0).unwrap();
    let device = Device::Cuda(0);
    let x = bf16(device, &[2, 3, 5, 8]);
    let function = |x: &Tensor| x.transpose(1, 2)?.contiguous()?.reshape([2, 5, 24]);
    let expected = function(&bf16(Device::Cpu, &[2, 3, 5, 8])).unwrap();
    let (captured, _) =
        Graph::capture([("x", x.clone())], |inputs| Ok(vec![function(&inputs[0])?])).unwrap();
    let options = GraphPrepareOptions::new(device).with_cuda_options(
        GraphCudaOptions::new().with_lowering_policy(LoweringPolicy::new(vec![
            CudaGraphProvider::Alias,
            CudaGraphProvider::Kernel,
        ])),
    );

    let mut prepared = captured.prepare(options).unwrap();
    let actual = prepared
        .run(&HashMap::from([("x".to_owned(), x)]))
        .unwrap()
        .remove(0);

    assert_eq!(host_values(&actual), host_values(&expected));
}

#[test]
fn test_index_select_and_embedding_of_strided_tables_match_the_cpu() {
    let _lock = device_lock(0).unwrap();
    let rows = |device| {
        Tensor::from_vec(
            device,
            Shape::create([3, 2]).unwrap(),
            vec![5_i64, 0, 2, 2, 1, 4],
        )
        .unwrap()
    };
    let columns =
        |device| Tensor::from_vec(device, Shape::create([3]).unwrap(), vec![3_i64, 0, 1]).unwrap();
    let lookups = |table: &Tensor, rows: &Tensor, columns: &Tensor| -> singe::Result<Vec<Tensor>> {
        Ok(vec![
            table.embedding(rows)?,
            call_one(
                Operation::IndexSelect,
                OperationAttributes::IndexSelect(1),
                &[table, columns],
            )?,
        ])
    };
    for data_type in [DataType::Bf16, DataType::F32] {
        // The transposed table reads its rows with strides.
        let table = |device| {
            bf16(device, &[4, 6])
                .cast(data_type)
                .unwrap()
                .transpose(0, 1)
                .unwrap()
        };
        let expected = lookups(
            &table(Device::Cpu),
            &rows(Device::Cpu),
            &columns(Device::Cpu),
        )
        .unwrap()
        .iter()
        .map(|tensor| tensor.contiguous().unwrap().to_bytes().unwrap())
        .collect::<Vec<_>>();
        let cuda = Device::Cuda(0);
        let (table, rows, columns) = (table(cuda), rows(cuda), columns(cuda));
        let eager = lookups(&table, &rows, &columns).unwrap();
        let (captured, _) = Graph::capture(
            [
                ("table", table.clone()),
                ("rows", rows.clone()),
                ("columns", columns.clone()),
            ],
            |inputs| lookups(&inputs[0], &inputs[1], &inputs[2]),
        )
        .unwrap();
        let compiled = captured
            .prepare(GraphPrepareOptions::new(cuda))
            .unwrap()
            .run(&HashMap::from([
                ("table".to_owned(), table),
                ("rows".to_owned(), rows),
                ("columns".to_owned(), columns),
            ]))
            .unwrap();
        for outputs in [eager, compiled] {
            let actual = outputs
                .iter()
                .map(|tensor| tensor.contiguous().unwrap().to_bytes().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "{data_type:?}");
        }
    }
}

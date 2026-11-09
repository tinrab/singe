use std::collections::HashMap;

use super::*;
use crate::{
    backend::registry::{BackendExecution, operation_support, registered_operations},
    data_type::DataType,
    graph::Graph,
    scalar::Scalar,
    shape::Shape,
    value::ValueSpec,
};

fn vector(values: &[f32]) -> Tensor {
    Tensor::from_vec(
        Device::Cpu,
        Shape::create([i64::try_from(values.len()).unwrap()]).unwrap(),
        values.to_vec(),
    )
    .unwrap()
}

#[test]
fn test_every_declared_backend_has_a_kernel() {
    for &operation in registered_operations() {
        let support = operation_support(operation);
        if support.supports(BackendExecution::CpuEager) {
            assert!(
                has_backend_kernel(operation, DispatchKey::Cpu),
                "{} declares CPU support without a CPU kernel",
                operation.name()
            );
        }
        #[cfg(feature = "cuda")]
        if support.supports(BackendExecution::CudaEager) {
            assert!(
                has_backend_kernel(operation, DispatchKey::Cuda),
                "{} declares CUDA support without a CUDA kernel",
                operation.name()
            );
        }
    }
}

#[test]
fn test_composite_kernel_runs_when_backend_has_none() {
    let input = vector(&[1., 2., -1.]);
    let gate = vector(&[0.5, -2., 3.]);
    let composite = input.swiglu(&gate).unwrap();
    let expected = input.mul(&gate.silu().unwrap()).unwrap();
    assert_eq!(
        composite.to_vec::<f32>().unwrap(),
        expected.to_vec::<f32>().unwrap()
    );
}

#[test]
fn test_exclude_guard_restores_previous_keys() {
    assert!(!excluded().contains(DispatchKey::Capture));
    {
        let _outer = ExcludeGuard::new(DispatchKey::Capture.and_higher());
        assert!(excluded().contains(DispatchKey::Autocast));
        assert!(excluded().contains(DispatchKey::Capture));
    }
    assert!(!excluded().contains(DispatchKey::Capture));
}

#[test]
fn test_autocast_rejects_full_precision_and_nests() {
    assert!(autocast(DataType::F32).is_err());
    {
        let _half = autocast(DataType::F16).unwrap();
        {
            let _bfloat = autocast(DataType::Bf16).unwrap();
            assert_eq!(autocast_data_type(), Some(DataType::Bf16));
        }
        assert_eq!(autocast_data_type(), Some(DataType::F16));
    }
    assert_eq!(autocast_data_type(), None);
}

#[test]
fn test_autocast_leaves_cpu_matmul_in_full_precision() {
    let lhs =
        Tensor::from_vec(Device::Cpu, Shape::create([1, 2]).unwrap(), vec![1_f32, 2.]).unwrap();
    let rhs =
        Tensor::from_vec(Device::Cpu, Shape::create([2, 1]).unwrap(), vec![3_f32, 4.]).unwrap();
    let _scope = autocast(DataType::F16).unwrap();
    let output = lhs.matmul(&rhs).unwrap();
    assert_eq!(output.data_type(), DataType::F32);
    assert_eq!(output.to_vec::<f32>().unwrap(), [11.]);
}

fn scaled_add() -> CustomOperation {
    CustomOperation::register(
        "test.scaled_add",
        2,
        1,
        &[DataType::F32],
        |inputs: &[ValueSpec], attributes: &[Scalar]| {
            if attributes.len() != 1 {
                return Err(Error::invalid_argument("scaled_add attributes"));
            }
            Ok(vec![inputs[0].clone()])
        },
    )
    .unwrap()
    .register_kernel(
        DispatchKey::Cpu,
        Kernel::single(|call| {
            let OperationAttributes::Custom(values) = call.attributes() else {
                return Err(Error::invalid_argument("scaled_add attributes"));
            };
            call.input(0)?.add(&call.input(1)?.mul_scalar(values[0])?)
        }),
    )
    .unwrap()
}

#[test]
fn test_custom_operation_runs_records_and_reports_support() {
    let operation = Operation::Custom(scaled_add());
    let attributes = OperationAttributes::Custom(vec![Scalar::from(2_f32)]);
    let lhs = vector(&[1., 2.]);
    let rhs = vector(&[10., 20.]);

    let eager = call_one(operation, attributes.clone(), &[&lhs, &rhs]).unwrap();
    assert_eq!(eager.to_vec::<f32>().unwrap(), [21., 42.]);

    let (captured, _) = Graph::capture([("lhs", lhs), ("rhs", rhs)], |inputs| {
        Ok(vec![call_one(
            operation,
            attributes.clone(),
            &[&inputs[0], &inputs[1]],
        )?])
    })
    .unwrap();
    let replayed = captured
        .run(&HashMap::from([
            ("lhs".to_owned(), vector(&[3., 4.])),
            ("rhs".to_owned(), vector(&[1., 1.])),
        ]))
        .unwrap();
    assert_eq!(replayed[0].to_vec::<f32>().unwrap(), [5., 6.]);

    let support = operation_support(operation);
    assert!(support.supports(BackendExecution::CpuEager));
    assert!(!support.supports(BackendExecution::CudaEager));
}

#[test]
fn test_missing_kernel_reports_operation_and_backend() {
    let operation = Operation::Custom(
        CustomOperation::register("test.unimplemented", 1, 1, &[DataType::F32], |inputs, _| {
            Ok(vec![inputs[0].clone()])
        })
        .unwrap(),
    );
    let error = call_one(
        operation,
        OperationAttributes::Custom(Vec::new()),
        &[&vector(&[1.])],
    )
    .unwrap_err();
    assert!(
        matches!(&error, Error::Unsupported { operation, .. } if operation == "test.unimplemented"),
        "unexpected error {error:?}"
    );
}

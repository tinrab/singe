use std::{
    collections::HashMap,
    panic::{self, AssertUnwindSafe},
    thread,
};

use singe::{
    DataType, Device, Error, Shape, Tensor,
    error::CaptureError,
    graph::{
        CaptureBindings, CaptureOptions, Graph, GraphPrepareOptions, StateBindingKind, capture,
    },
};
#[cfg(all(feature = "testing", feature = "cuda"))]
use singe_cuda::testing::device_lock;

fn vector(values: &[f32]) -> Tensor {
    Tensor::from_vec(
        Device::Cpu,
        Shape::create([i64::try_from(values.len()).unwrap()]).unwrap(),
        values.to_vec(),
    )
    .unwrap()
}

#[test]
fn test_unrecorded_operands_and_outputs_are_rejected_instead_of_becoming_constants() {
    let closed = vector(&[3., 4.]);
    let operand = Graph::capture([("x", vector(&[1., 2.]))], |inputs| {
        Ok(vec![inputs[0].add(&closed)?])
    });
    assert!(matches!(
        operand,
        Err(Error::Capture(CaptureError::UnrecordedTensor))
    ));
    let output = Graph::capture([("x", closed.clone())], |_| Ok(vec![closed.clone()]));
    assert!(matches!(
        output,
        Err(Error::Capture(CaptureError::UnrecordedTensor))
    ));
}

#[test]
fn test_explicit_parameters_buffers_and_literals_share_the_prepared_execution_path() {
    let weight = vector(&[2., 3.]);
    let bias = vector(&[1., 2.]);
    let literal = vector(&[10., 20.]);
    let bindings = CaptureBindings::new()
        .with_parameter("weight", weight.clone())
        .with_buffer("bias", bias.clone())
        .with_constant("literal", literal.clone());
    let (captured, _) =
        Graph::capture_with_bindings([("x", vector(&[1., 2.]))], bindings, |inputs| {
            let first = inputs[0].mul(&weight)?;
            Ok(vec![first.add(&weight)?.add(&bias)?.add(&literal)?])
        })
        .unwrap();
    let program = captured.program().unwrap();
    let roles = program
        .state_schema()
        .map(|binding| (binding.name().to_owned(), binding.kind()))
        .collect::<HashMap<_, _>>();
    assert_eq!(
        roles,
        HashMap::from([
            ("weight".into(), StateBindingKind::Parameter),
            ("bias".into(), StateBindingKind::Buffer)
        ])
    );
    assert!(
        !program
            .capture_info()
            .unwrap()
            .allows_value_specialization()
    );
    let mut executor = captured
        .prepare(GraphPrepareOptions::new(Device::Cpu))
        .unwrap();
    let outputs = executor
        .run(&HashMap::from([("x".into(), vector(&[3., 4.]))]))
        .unwrap();
    assert_eq!(outputs[0].to_vec::<f32>().unwrap(), [19., 37.]);
    assert!(
        executor
            .report()
            .unwrap()
            .contains("value specialization: false")
    );
}

#[test]
fn test_literals_created_inside_capture_require_explicit_registration() {
    assert!(matches!(
        capture::constant("outside", &vector(&[1.])),
        Err(Error::Capture(CaptureError::NotActive))
    ));
    let (captured, _) = Graph::capture([("x", vector(&[1., 2.]))], |inputs| {
        let literal = capture::constant("offset", &vector(&[3., 4.]))?;
        Ok(vec![inputs[0].add(&literal)?])
    })
    .unwrap();
    assert_eq!(
        captured
            .run(&HashMap::from([("x".into(), vector(&[5., 6.]))]))
            .unwrap()[0]
            .to_vec::<f32>()
            .unwrap(),
        [8., 10.]
    );
}

fn check_readback_is_rejected(device: Device) {
    let original = vector(&[1., 2.]).to_device(device).unwrap();
    let (_, captured_outputs) = Graph::capture([("x", original.clone())], |inputs| {
        let output = inputs[0].add_scalar(2.)?;
        for tensor in [&original, &inputs[0], &output] {
            assert!(matches!(
                tensor.to_vec::<f32>(),
                Err(Error::Capture(CaptureError::Readback))
            ));
            assert!(matches!(
                tensor.to_bytes(),
                Err(Error::Capture(CaptureError::Readback))
            ));
            assert!(matches!(
                tensor.get_element::<f32>(&[0]),
                Err(Error::Capture(CaptureError::Readback))
            ));
            let mut bytes = vec![0; tensor.storage().byte_size()];
            assert!(matches!(
                tensor.storage().copy_to_host(&mut bytes),
                Err(Error::Capture(CaptureError::Readback))
            ));
            if device.is_cpu() {
                let storage = tensor.storage().as_cpu().unwrap();
                assert!(matches!(
                    storage.as_slice(),
                    Err(Error::Capture(CaptureError::Readback))
                ));
                assert!(matches!(
                    storage.as_typed_slice::<f32>(),
                    Err(Error::Capture(CaptureError::Readback))
                ));
            }
        }
        assert_eq!(inputs[0].shape().dimensions(), [2]);
        Ok(vec![output])
    })
    .unwrap();
    assert_eq!(captured_outputs[0].to_vec::<f32>().unwrap(), [3., 4.]);
    assert_eq!(original.to_vec::<f32>().unwrap(), [1., 2.]);
}

#[test]
fn test_tracked_readback_is_rejected_through_tensor_storage_and_allocation_aliases() {
    check_readback_is_rejected(Device::Cpu);
}

#[test]
fn test_empty_tensor_readback_obeys_the_same_capture_policy() {
    let (_, captured_outputs) = Graph::capture([("empty", vector(&[]))], |inputs| {
        assert!(matches!(
            inputs[0].to_vec::<f32>(),
            Err(Error::Capture(CaptureError::Readback))
        ));
        assert!(matches!(
            inputs[0].to_bytes(),
            Err(Error::Capture(CaptureError::Readback))
        ));
        Ok(vec![inputs[0].clone()])
    })
    .unwrap();
    assert_eq!(captured_outputs[0].to_vec::<f32>().unwrap(), [] as [f32; 0]);
}

#[test]
fn test_explicit_value_specialization_retains_the_observed_host_coefficient() {
    let (captured, _) = Graph::capture_with_options(
        [("x", vector(&[2., 3.]))],
        CaptureBindings::new(),
        CaptureOptions::new().with_value_specialization(true),
        |inputs| {
            let coefficient = inputs[0].to_vec::<f32>()?[0];
            Ok(vec![inputs[0].mul_scalar(coefficient)?])
        },
    )
    .unwrap();
    assert!(
        captured
            .capture_info()
            .unwrap()
            .allows_value_specialization()
    );
    let mut executor = captured
        .prepare(GraphPrepareOptions::new(Device::Cpu))
        .unwrap();
    assert!(
        executor
            .report()
            .unwrap()
            .contains("value specialization: true")
    );
    let outputs = executor
        .run(&HashMap::from([("x".into(), vector(&[4., 6.]))]))
        .unwrap();
    assert_eq!(outputs[0].to_vec::<f32>().unwrap(), [8., 12.]);
}

#[test]
fn test_nested_capture_and_caught_errors_leave_the_outer_trace_usable() {
    let (_, captured_outputs) = Graph::capture([("x", vector(&[1., 2.]))], |inputs| {
        let nested = Graph::capture(
            [("nested", inputs[0].clone())],
            |nested| Ok(nested.to_vec()),
        );
        assert!(matches!(
            nested,
            Err(Error::Capture(CaptureError::AlreadyActive))
        ));
        assert!(matches!(
            inputs[0].to_vec::<f32>(),
            Err(Error::Capture(CaptureError::Readback))
        ));
        Ok(vec![inputs[0].add_scalar(3.)?])
    })
    .unwrap();
    assert_eq!(captured_outputs[0].to_vec::<f32>().unwrap(), [4., 5.]);
}

#[test]
fn test_error_and_unwind_cleanup_restore_capture_and_readback() {
    let original = vector(&[1., 2.]);
    let failed = Graph::capture([("x", original.clone())], |inputs| {
        inputs[0].to_vec::<f32>()?;
        Ok(inputs.to_vec())
    });
    assert!(matches!(
        failed,
        Err(Error::Capture(CaptureError::Readback))
    ));
    let unwound = panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = Graph::capture(
            [("x", original.clone())],
            |_| -> Result<Vec<Tensor>, Error> { panic!("expected capture unwind") },
        );
    }));
    assert!(unwound.is_err());
    assert_eq!(original.to_vec::<f32>().unwrap(), [1., 2.]);
    let (_, captured_outputs) = Graph::capture([("x", original)], |inputs| {
        Ok(vec![inputs[0].add_scalar(1.)?])
    })
    .unwrap();
    assert_eq!(captured_outputs[0].to_vec::<f32>().unwrap(), [2., 3.]);
}

#[test]
fn test_spawned_thread_work_is_outside_the_trace_and_cannot_supply_outputs() {
    let result = Graph::capture([("x", vector(&[1., 2.]))], |inputs| {
        let input = inputs[0].clone();
        let output = thread::spawn(move || {
            assert_eq!(input.to_vec::<f32>().unwrap(), [1., 2.]);
            input.add_scalar(2.)
        })
        .join()
        .unwrap()?;
        Ok(vec![output])
    });
    assert!(matches!(
        result,
        Err(Error::Capture(CaptureError::UnrecordedTensor))
    ));
}

#[test]
fn test_dynamic_dimension_declarations_specialize_the_recorded_path_without_retracing() {
    let (mut captured, _) = Graph::capture([("x", vector(&[1., 2.]))], |inputs| {
        let offset = if inputs[0].shape().dimensions()[0] == 2 {
            10.
        } else {
            20.
        };
        Ok(vec![inputs[0].add_scalar(offset)?])
    })
    .unwrap();
    let changed = HashMap::from([("x".into(), vector(&[1., 2., 3.]))]);
    let mut fixed = captured
        .prepare(GraphPrepareOptions::new(Device::Cpu))
        .unwrap();
    assert!(fixed.run(&changed).is_err());
    // Declaring this dimension dynamic accepts the previously recorded offset.
    captured
        .add_input_dimension_guard("x", 0, "batch", Some(1), Some(4))
        .unwrap();
    let mut dynamic = captured
        .prepare(GraphPrepareOptions::new(Device::Cpu))
        .unwrap();
    assert_eq!(
        dynamic.run(&changed).unwrap()[0].to_vec::<f32>().unwrap(),
        [11., 12., 13.]
    );
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_tracked_readback_requires_explicit_value_specialization() {
    check_readback_is_rejected(Device::Cuda(0));
}

#[cfg(feature = "cuda")]
#[test]
fn test_explicit_capture_state_prepares_native_cuda_and_rebinds_without_recapture() {
    let weight = vector(&[2., 3.]);
    let (captured, _) = Graph::capture_with_bindings(
        [("x", vector(&[1., 2.]))],
        CaptureBindings::new().with_parameter("weight", weight.clone()),
        |inputs| Ok(vec![inputs[0].mul(&weight)?]),
    )
    .unwrap();
    let mut executor = captured
        .prepare(GraphPrepareOptions::new(Device::Cuda(0)))
        .unwrap();
    let input = HashMap::from([(
        "x".into(),
        vector(&[3., 4.]).to_device(Device::Cuda(0)).unwrap(),
    )]);
    let retained = executor.run(&input).unwrap();
    let state = executor
        .state()
        .replace(&HashMap::from([("weight".into(), vector(&[4., 5.]))]))
        .unwrap();
    executor.rebind_state(&state).unwrap();
    assert_eq!(
        executor.run(&input).unwrap()[0].to_vec::<f32>().unwrap(),
        [12., 20.]
    );
    assert_eq!(retained[0].to_vec::<f32>().unwrap(), [6., 12.]);
}

fn assert_value_dependent_outputs_narrow_on_each_run(device: Device) {
    let input = |values: &[f32]| vector(values).to_device(device).unwrap();
    let threshold = input(&[1.]);
    let (graph, eager) = Graph::capture(
        [("x", input(&[0., 2., 0., 3.])), ("t", threshold.clone())],
        |inputs| {
            let x = &inputs[0];
            Ok(vec![
                x.nonzero()?,
                x.masked_select(&x.greater(&inputs[1])?)?,
            ])
        },
    )
    .unwrap();
    assert_eq!(eager[0].shape().dimensions(), [2, 1]);
    let mut prepared = graph.prepare(GraphPrepareOptions::new(device)).unwrap();
    // Another input selects another number of elements, which each run narrows its results to.
    let outputs = prepared
        .run(&HashMap::from([
            ("x".to_owned(), input(&[1., 0., 5., 7.])),
            ("t".to_owned(), threshold),
        ]))
        .unwrap();
    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].to_vec::<i64>().unwrap(), [0, 2, 3]);
    assert_eq!(outputs[1].to_vec::<f32>().unwrap(), [5., 7.]);
    // A graph returns such a result but computes nothing from it.
    let error = Graph::capture([("x", input(&[0., 2.]))], |inputs| {
        Ok(vec![inputs[0].nonzero()?.cast(DataType::F32)?])
    })
    .unwrap_err();
    assert!(matches!(error, Error::Unsupported { .. }), "{error:?}");
}

#[test]
fn test_cpu_graphs_return_value_dependent_results_narrowed_on_each_run() {
    assert_value_dependent_outputs_narrow_on_each_run(Device::Cpu);
}

#[cfg(all(feature = "testing", feature = "cuda"))]
#[test]
fn test_cuda_graphs_return_value_dependent_results_narrowed_on_each_run() {
    let _lock = device_lock(0).unwrap();
    assert_value_dependent_outputs_narrow_on_each_run(Device::Cuda(0));
}

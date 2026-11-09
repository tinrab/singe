use std::{panic, panic::AssertUnwindSafe, sync::Mutex};

use super::*;
use crate::error::UnsupportedTarget;

#[test]
fn test_graph_capture_preserves_duplicate_input_tensor_ids() {
    let shape = Shape::create(vec![2]).unwrap();
    let input = Tensor::from_vec(Device::Cpu, &shape, vec![1.0_f32, 2.0]).unwrap();

    let (captured, _) = Graph::capture(
        [("a", input.clone()), ("b", input)],
        |inputs| -> Result<Vec<Tensor>> {
            let output = inputs[0].add(&inputs[1])?;
            Ok(vec![output])
        },
    )
    .unwrap();

    let mut inputs = HashMap::new();
    inputs.insert(
        "a".to_string(),
        Tensor::from_vec(Device::Cpu, &shape, vec![2.0_f32, 3.0]).unwrap(),
    );
    inputs.insert(
        "b".to_string(),
        Tensor::from_vec(Device::Cpu, &shape, vec![10.0_f32, 20.0]).unwrap(),
    );

    let outputs = captured.run(&inputs).unwrap();
    assert_close!(&outputs[0].to_vec::<f32>().unwrap(), &[12.0, 23.0], 1e-6);
}

#[test]
fn test_graph_capture_rejects_duplicate_input_names() {
    let shape = Shape::create(vec![2]).unwrap();
    let a = Tensor::from_vec(Device::Cpu, &shape, vec![1.0_f32, 2.0]).unwrap();
    let b = Tensor::from_vec(Device::Cpu, &shape, vec![3.0_f32, 4.0]).unwrap();

    let error = Graph::capture([("x", a), ("x", b)], |inputs| -> Result<Vec<Tensor>> {
        Ok(vec![inputs[0].clone()])
    })
    .unwrap_err();

    assert!(matches!(error, Error::DuplicateInput));
}

#[test]
fn test_graph_capture_clears_active_session_after_panic() {
    static PANIC_HOOK_LOCK: Mutex<()> = Mutex::new(());

    let _hook_lock = PANIC_HOOK_LOCK.lock().unwrap();
    let previous_hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));

    let shape = Shape::create(vec![2]).unwrap();
    let input = Tensor::from_vec(Device::Cpu, &shape, vec![1.0_f32, 2.0]).unwrap();

    let panic_result = panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = Graph::capture([("x", input.clone())], |_inputs| -> Result<Vec<Tensor>> {
            panic!("intentional capture panic")
        });
    }));
    panic::set_hook(previous_hook);
    assert!(panic_result.is_err());

    let (captured, _) = Graph::capture([("x", input)], |inputs| -> Result<Vec<Tensor>> {
        let output = inputs[0].mul_scalar(2.0)?;
        Ok(vec![output])
    })
    .unwrap();

    let ir = captured.to_ir().unwrap();
    assert_eq!(ir.values().len(), 2);
}

#[test]
fn test_graph_capture_clears_active_session_after_error() {
    let shape = Shape::create(vec![2]).unwrap();
    let input = Tensor::from_vec(Device::Cpu, &shape, vec![1.0_f32, 2.0]).unwrap();

    let error = Graph::capture([("x", input.clone())], |inputs| -> Result<Vec<Tensor>> {
        let _output = inputs[0].mul_scalar(2.0)?;
        Err(Error::unsupported_operation("test capture error"))
    })
    .unwrap_err();
    assert!(matches!(
        error,
        Error::Unsupported {
            target: UnsupportedTarget::Operation,
            ..
        }
    ));

    let (captured, _) = Graph::capture([("x", input)], |inputs| -> Result<Vec<Tensor>> {
        let output = inputs[0].mul_scalar(2.0)?;
        Ok(vec![output])
    })
    .unwrap();

    let ir = captured.to_ir().unwrap();
    assert_eq!(ir.values().len(), 2);
}

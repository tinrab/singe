use singe::{Device, Shape, Tensor};

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

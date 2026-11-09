use crate::{
    device::Device,
    nn::{Layer, Linear, StateBuilder},
    optim::Sgd,
    shape::Shape,
    tensor::Tensor,
};

fn tensor(dimensions: &[i64], values: Vec<f32>) -> Tensor {
    Tensor::from_vec(
        Device::Cpu,
        Shape::create(dimensions.to_vec()).unwrap(),
        values,
    )
    .unwrap()
}

#[test]
fn test_parameters_skip_buffers_and_list_tied_tensors_once() {
    let mut builder = StateBuilder::new();
    let weight = builder
        .parameter("weight", tensor(&[1, 2], vec![0.0, 0.0]))
        .unwrap();
    builder.buffer("count", tensor(&[1], vec![0.0])).unwrap();
    builder.tie("weight_alias", &weight).unwrap();
    let state = builder.finish().unwrap();
    let names = state
        .named_parameters()
        .into_iter()
        .map(|(name, _)| name.to_owned())
        .collect::<Vec<_>>();
    assert_eq!(names, ["weight"]);
    assert_eq!(state.parameters().len(), 1);
}

#[test]
fn test_optimizers_train_module_parameters_in_place() {
    let mut builder = StateBuilder::new();
    let weight = builder
        .parameter("weight", tensor(&[1, 2], vec![0.0, 0.0]))
        .unwrap();
    let bias = builder.parameter("bias", tensor(&[1], vec![0.0])).unwrap();
    let state = builder.finish().unwrap();
    let linear = Linear::new(weight, Some(bias));
    for parameter in state.parameters() {
        parameter.set_requires_grad(true).unwrap();
    }
    // y = 2 x0 - x1 + 0.5
    let x = tensor(&[4, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0]);
    let y = tensor(&[4, 1], vec![2.5, -0.5, 1.5, 5.5]);
    let mut optimizer = Sgd::new(state.parameters(), 0.1);
    let loss = || {
        linear
            .forward(&x, &state)
            .unwrap()
            .sub(&y)
            .unwrap()
            .square()
            .unwrap()
            .mean(&[], false)
            .unwrap()
    };
    let first = loss().to_vec::<f32>().unwrap()[0];
    for _ in 0..200 {
        optimizer.zero_grad();
        loss().backward().unwrap();
        optimizer.step().unwrap();
    }
    let last = loss().to_vec::<f32>().unwrap()[0];
    assert!(last < first * 1e-3, "{first} -> {last}");
    let weight = state
        .named_tensor("weight")
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    assert!(
        (weight[0] - 2.0).abs() < 0.05 && (weight[1] + 1.0).abs() < 0.05,
        "{weight:?}"
    );
}

use crate::{
    autograd::checkpoint,
    data_type::DataType,
    device::Device,
    error::Result,
    optim::{AdamW, Sgd},
    random::Generator,
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

fn parameter(dimensions: &[i64], generator: &Generator) -> Tensor {
    let parameter = Tensor::randn(
        Device::Cpu,
        Shape::create(dimensions.to_vec()).unwrap(),
        DataType::F32,
        Some(generator),
    )
    .unwrap()
    .mul_scalar(0.5)
    .unwrap();
    parameter.set_requires_grad(true).unwrap();
    parameter
}

/// Mean squared error of a two-layer MLP, with its hidden layer recomputed by `checkpoint` when `checkpointed` is true.
fn loss(parameters: &[Tensor], x: &Tensor, y: &Tensor, checkpointed: bool) -> Tensor {
    let hidden = |inputs: &[Tensor]| -> Result<Vec<Tensor>> {
        Ok(vec![inputs[0].matmul(&inputs[1])?.add(&inputs[2])?.tanh()?])
    };
    let inputs = [x.clone(), parameters[0].clone(), parameters[1].clone()];
    let hidden = if checkpointed {
        checkpoint(&inputs, hidden).unwrap().remove(0)
    } else {
        hidden(&inputs).unwrap().remove(0)
    };
    hidden
        .matmul(&parameters[2])
        .unwrap()
        .sub(y)
        .unwrap()
        .square()
        .unwrap()
        .mean(&[], false)
        .unwrap()
}

fn train(optimizer: &mut dyn FnMut() -> f32, steps: usize) -> (f32, f32) {
    let first = optimizer();
    let mut last = first;
    for _ in 1..steps {
        last = optimizer();
    }
    (first, last)
}

fn regression() -> (Tensor, Tensor) {
    let x = tensor(
        &[8, 2],
        (0..16).map(|value| (value as f32 * 0.37).sin()).collect(),
    );
    let y = tensor(
        &[8, 1],
        (0..8)
            .map(|row| {
                let a = (row as f32 * 2.0 * 0.37).sin();
                let b = ((row as f32).mul_add(2.0, 1.0) * 0.37).sin();
                a.mul_add(0.8, -0.5 * b)
            })
            .collect(),
    );
    (x, y)
}

#[test]
fn test_sgd_with_momentum_and_adamw_reduce_a_regression_loss() {
    let (x, y) = regression();
    let generator = Generator::new(4);
    for adam in [false, true] {
        for checkpointed in [false, true] {
            let parameters = vec![
                parameter(&[2, 8], &generator),
                parameter(&[8], &generator),
                parameter(&[8, 1], &generator),
            ];
            let mut sgd = Sgd::new(parameters.clone(), 0.1).with_momentum(0.9);
            let mut adamw = AdamW::new(parameters.clone(), 0.05);
            let mut step = || -> f32 {
                let value = loss(&parameters, &x, &y, checkpointed);
                if adam {
                    adamw.zero_grad();
                } else {
                    sgd.zero_grad();
                }
                value.backward().unwrap();
                if adam {
                    adamw.step().unwrap();
                } else {
                    sgd.step().unwrap();
                }
                value.to_vec::<f32>().unwrap()[0]
            };
            let (first, last) = train(&mut step, 60);
            assert!(
                last < first * 0.2,
                "adam {adam} checkpointed {checkpointed}: {first} -> {last}"
            );
        }
    }
}

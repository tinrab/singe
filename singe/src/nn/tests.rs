use crate::{device::Device, nn::StateBuilder, shape::Shape, tensor::Tensor};

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

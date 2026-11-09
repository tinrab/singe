use singe::{DataType, Device, Shape, Tensor, random::Generator, set_num_threads};

fn random(generator: &Generator, dimensions: &[i64]) -> Tensor {
    Tensor::rand(
        Device::Cpu,
        Shape::create(dimensions.to_vec()).unwrap(),
        DataType::F32,
        Some(generator),
    )
    .unwrap()
}

fn results() -> Vec<Vec<f32>> {
    let generator = Generator::new(5);
    let values = random(&generator, &[512, 512]);
    let transposed = values.transpose(0, 1).unwrap();
    let image = random(&generator, &[2, 8, 64, 64]);
    let filters = random(&generator, &[16, 4, 3, 3]);
    let channel = random(&generator, &[8]);
    let bias = random(&generator, &[16]);
    let row = random(&generator, &[512]);
    let query = random(&generator, &[2, 4, 64, 32]);
    let key = random(&generator, &[2, 4, 64, 32]);
    let rotary = random(&generator, &[64, 16]);
    [
        values.add(&transposed).unwrap(),
        values.exp().unwrap(),
        values.sum(&[1], false).unwrap(),
        transposed.sum(&[0, 1], false).unwrap(),
        values.softmax(1).unwrap(),
        values.matmul(&transposed).unwrap(),
        image
            .convolution2d(&filters, Some(&bias), (1, 1), (1, 1, 1, 1), (1, 1), 2)
            .unwrap(),
        image
            .max_pool2d((3, 3), (2, 2), (1, 1, 1, 1), (1, 1), false)
            .unwrap(),
        image
            .avg_pool2d((3, 3), (2, 2), (1, 1, 1, 1), (1, 1), false, true)
            .unwrap(),
        values
            .layer_norm(&[512], Some(&row), Some(&row), 1e-5)
            .unwrap(),
        values.rms_norm(&[512], Some(&row), None, 1e-5).unwrap(),
        image
            .batch_norm(
                Some(&channel),
                Some(&channel),
                Some(&channel),
                Some(&channel),
                false,
                0.1,
                1e-5,
            )
            .unwrap(),
        query
            .scaled_dot_product_attention(&key, &key, None, true)
            .unwrap(),
        query.rotary_embedding(&rotary, &rotary).unwrap(),
    ]
    .iter()
    .map(|tensor| tensor.to_vec::<f32>().unwrap())
    .collect()
}

#[test]
fn test_cpu_results_do_not_depend_on_the_thread_count() {
    set_num_threads(1);
    let serial = results();
    set_num_threads(8);
    let parallel = results();
    set_num_threads(0);
    assert_eq!(serial, parallel);
}

use std::collections::HashMap;

#[cfg(feature = "cuda")]
use singe::graph::GraphPrepareOptions;
use singe::{DataType, Device, Shape, Tensor, graph::Graph, random::Generator};

#[cfg(all(feature = "testing", feature = "cuda"))]
use singe_cuda::testing::device_lock;

fn uniform(device: Device, count: i64, generator: &Generator) -> Vec<f32> {
    Tensor::rand(
        device,
        Shape::create([count]).unwrap(),
        DataType::F32,
        Some(generator),
    )
    .unwrap()
    .to_vec::<f32>()
    .unwrap()
}

fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len() as f32
}

#[test]
fn test_generators_are_reproducible_and_advance() {
    let first = Generator::new(7);
    let second = Generator::new(7);
    let a = uniform(Device::Cpu, 9, &first);
    assert_eq!(a, uniform(Device::Cpu, 9, &second));
    assert_ne!(a, uniform(Device::Cpu, 9, &first));
    assert_ne!(a, uniform(Device::Cpu, 9, &Generator::new(8)));
    first.manual_seed(7).unwrap();
    assert_eq!(a, uniform(Device::Cpu, 9, &first));
}

#[test]
fn test_cpu_distributions_have_expected_moments() {
    let generator = Generator::new(1);
    let values = Tensor::uniform(
        Device::Cpu,
        Shape::create([20_000]).unwrap(),
        -2.0,
        2.0,
        DataType::F32,
        Some(&generator),
    )
    .unwrap()
    .to_vec::<f32>()
    .unwrap();
    assert!(values.iter().all(|value| (-2.0..2.0).contains(value)));
    assert!(mean(&values).abs() < 0.05);

    let normal = Tensor::randn(
        Device::Cpu,
        Shape::create([20_000]).unwrap(),
        DataType::F32,
        Some(&generator),
    )
    .unwrap()
    .to_vec::<f32>()
    .unwrap();
    let center = mean(&normal);
    let variance = normal
        .iter()
        .map(|value| (value - center).powi(2))
        .sum::<f32>()
        / normal.len() as f32;
    assert!(center.abs() < 0.05);
    assert!((variance - 1.0).abs() < 0.05);

    let probabilities = Tensor::from_vec(
        Device::Cpu,
        Shape::create([10_000]).unwrap(),
        vec![0.25_f32; 10_000],
    )
    .unwrap();
    let hits = probabilities
        .bernoulli(Some(&generator))
        .unwrap()
        .to_vec::<bool>()
        .unwrap()
        .into_iter()
        .filter(|hit| *hit)
        .count();
    assert!((2_250..2_750).contains(&hits));

    let weights = Tensor::from_vec(
        Device::Cpu,
        Shape::create([2, 3]).unwrap(),
        vec![0_f32, 1., 0., 1., 0., 3.],
    )
    .unwrap();
    let samples = weights
        .multinomial(4_000, Some(&generator))
        .unwrap()
        .to_vec::<i64>()
        .unwrap();
    assert!(samples[..4_000].iter().all(|index| *index == 1));
    let thirds = samples[4_000..].iter().filter(|index| **index == 2).count();
    assert!((2_850..3_150).contains(&thirds));
}

#[test]
fn test_captured_random_operations_continue_the_generator_on_each_run() {
    let generator = Generator::new(11);
    let zeros = Tensor::from_vec(Device::Cpu, Shape::create([5]).unwrap(), vec![0_f32; 5]).unwrap();
    let (captured, _) = Graph::capture([("x", zeros.clone())], |inputs| {
        let noise = Tensor::rand(
            Device::Cpu,
            Shape::create([5]).unwrap(),
            DataType::F32,
            Some(&generator),
        )?;
        Ok(vec![inputs[0].add(&noise)?])
    })
    .unwrap();
    let inputs = HashMap::from([("x".to_owned(), zeros)]);
    let first = captured.run(&inputs).unwrap()[0].to_vec::<f32>().unwrap();
    let second = captured.run(&inputs).unwrap()[0].to_vec::<f32>().unwrap();

    let eager = Generator::new(11);
    let _during_capture = uniform(Device::Cpu, 5, &eager);
    assert_eq!(first, uniform(Device::Cpu, 5, &eager));
    assert_eq!(second, uniform(Device::Cpu, 5, &eager));
}

#[cfg(all(feature = "testing", feature = "cuda"))]
#[test]
fn test_cuda_matches_cpu_for_the_same_generator_state() {
    let _lock = device_lock(0).unwrap();
    let cuda = Device::Cuda(0);
    let cpu_generator = Generator::new(3);
    let cuda_generator = Generator::new(3);
    assert_eq!(
        uniform(Device::Cpu, 37, &cpu_generator),
        uniform(cuda, 37, &cuda_generator)
    );

    let shape = Shape::create([37]).unwrap();
    let cpu = Tensor::randn(
        Device::Cpu,
        shape.clone(),
        DataType::F32,
        Some(&cpu_generator),
    )
    .unwrap()
    .to_vec::<f32>()
    .unwrap();
    let gpu = Tensor::randn(cuda, shape, DataType::F32, Some(&cuda_generator))
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    for (cpu, gpu) in cpu.iter().zip(&gpu) {
        assert!((cpu - gpu).abs() < 1e-5, "{cpu} != {gpu}");
    }

    let weights = vec![0.5_f32, 0.25, 0.25, 0.1, 0.1, 0.8];
    let cpu_weights =
        Tensor::from_vec(Device::Cpu, Shape::create([2, 3]).unwrap(), weights.clone()).unwrap();
    let cuda_weights = Tensor::from_vec(cuda, Shape::create([2, 3]).unwrap(), weights).unwrap();
    assert_eq!(
        cpu_weights
            .multinomial(16, Some(&cpu_generator))
            .unwrap()
            .to_vec::<i64>()
            .unwrap(),
        cuda_weights
            .multinomial(16, Some(&cuda_generator))
            .unwrap()
            .to_vec::<i64>()
            .unwrap()
    );
    assert_eq!(
        cpu_weights
            .bernoulli(Some(&cpu_generator))
            .unwrap()
            .to_vec::<bool>()
            .unwrap(),
        cuda_weights
            .bernoulli(Some(&cuda_generator))
            .unwrap()
            .to_vec::<bool>()
            .unwrap()
    );
}

#[cfg(all(feature = "testing", feature = "cuda"))]
#[test]
fn test_compiled_cuda_graphs_replay_random_operations_natively() {
    let _lock = device_lock(0).unwrap();
    let device = Device::Cuda(0);
    let generator = Generator::new(19);
    let weights = Tensor::from_vec(
        device,
        Shape::create([2, 4]).unwrap(),
        vec![1_f32, 2., 3., 4., 4., 3., 2., 1.],
    )
    .unwrap();
    let (captured, _) = Graph::capture([("weights", weights.clone())], |inputs| {
        let noise = Tensor::rand(
            device,
            Shape::create([2, 4]).unwrap(),
            DataType::F32,
            Some(&generator),
        )?;
        let samples = inputs[0].multinomial(3, Some(&generator))?;
        Ok(vec![inputs[0].add(&noise)?, samples])
    })
    .unwrap();
    let mut prepared = captured.prepare(GraphPrepareOptions::new(device)).unwrap();
    let inputs = HashMap::from([("weights".to_owned(), weights.clone())]);
    let first = prepared.run(&inputs).unwrap();
    let second = prepared.run(&inputs).unwrap();

    let eager = Generator::new(19);
    // Capture ran each operation once, advancing the generator.
    Tensor::rand(
        device,
        Shape::create([2, 4]).unwrap(),
        DataType::F32,
        Some(&eager),
    )
    .unwrap();
    weights.multinomial(3, Some(&eager)).unwrap();
    for outputs in [first, second] {
        let noise = Tensor::rand(
            device,
            Shape::create([2, 4]).unwrap(),
            DataType::F32,
            Some(&eager),
        )
        .unwrap();
        let samples = weights.multinomial(3, Some(&eager)).unwrap();
        assert_eq!(
            outputs[0].to_vec::<f32>().unwrap(),
            weights.add(&noise).unwrap().to_vec::<f32>().unwrap()
        );
        assert_eq!(
            outputs[1].to_vec::<i64>().unwrap(),
            samples.to_vec::<i64>().unwrap()
        );
    }
}

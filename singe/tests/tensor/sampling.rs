use std::collections::HashMap;

use singe::{Device, Shape, Tensor, graph::Graph, random::Generator, tensor::SamplingOptions};

#[cfg(all(feature = "testing", feature = "cuda"))]
use singe_cuda::testing::device_lock;

fn logits(device: Device, rows: i64, values: &[f32]) -> Tensor {
    let columns = i64::try_from(values.len()).unwrap();
    let repeated = (0..rows)
        .flat_map(|_| values.iter().copied())
        .collect::<Vec<_>>();
    Tensor::from_vec(device, Shape::create([rows, columns]).unwrap(), repeated).unwrap()
}

/// Logits whose probabilities are proportional to `1, 2, ..., count`.
fn linear_probabilities(count: usize) -> Vec<f32> {
    (1..=count).map(|weight| (weight as f32).ln()).collect()
}

fn frequencies(indices: &[i64], categories: usize) -> Vec<f64> {
    let mut counts = vec![0_usize; categories];
    for &index in indices {
        counts[usize::try_from(index).unwrap()] += 1;
    }
    counts
        .into_iter()
        .map(|count| count as f64 / indices.len() as f64)
        .collect()
}

#[test]
fn test_near_zero_temperature_selects_the_most_likely_tokens() {
    let logits = logits(Device::Cpu, 2, &[0.5, 3., -1., 2.]);
    let options = SamplingOptions::new().with_temperature(0.);
    let greedy = logits.sample(&options, None).unwrap();
    assert_eq!(greedy.shape().dimensions(), [2, 1]);
    assert_eq!(greedy.to_vec::<i64>().unwrap(), [1, 1]);
    let two = logits
        .sample(&options.with_samples(2), None)
        .unwrap()
        .to_vec::<i64>()
        .unwrap();
    assert_eq!(two, [1, 3, 1, 3]);
}

#[test]
fn test_filtered_tokens_are_never_drawn_and_frequencies_match() {
    let generator = Generator::new(11);
    let values = linear_probabilities(8);
    let vector = Tensor::from_vec(Device::Cpu, Shape::create([8]).unwrap(), values).unwrap();
    let draws = 20_000;

    let top_k = vector
        .sample(
            &SamplingOptions::new()
                .with_top_k(Some(3))
                .with_samples(draws),
            Some(&generator),
        )
        .unwrap()
        .to_vec::<i64>()
        .unwrap();
    assert!(top_k.iter().all(|&index| index >= 5));
    let observed = frequencies(&top_k, 8);
    for (index, weight) in [(5, 6.), (6, 7.), (7, 8.)] {
        assert!(
            (observed[index] - weight / 21.).abs() < 0.02,
            "{observed:?}"
        );
    }

    // Probabilities are 1/36, ..., 8/36; the four most likely sum to 26/36, the first past 0.6.
    let top_p = vector
        .sample(
            &SamplingOptions::new().with_top_p(0.6).with_samples(draws),
            Some(&generator),
        )
        .unwrap()
        .to_vec::<i64>()
        .unwrap();
    assert!(top_p.iter().all(|&index| index >= 4));
    let observed = frequencies(&top_p, 8);
    for (index, weight) in [(4, 5.), (5, 6.), (6, 7.), (7, 8.)] {
        assert!(
            (observed[index] - weight / 26.).abs() < 0.02,
            "{observed:?}"
        );
    }

    let unique = vector
        .sample(
            &SamplingOptions::new()
                .with_samples(8)
                .with_replacement(false),
            Some(&generator),
        )
        .unwrap();
    let mut distinct = unique.to_vec::<i64>().unwrap();
    distinct.sort_unstable();
    assert_eq!(distinct, (0..8).collect::<Vec<_>>());
    assert!(
        vector
            .sample(
                &SamplingOptions::new()
                    .with_samples(9)
                    .with_replacement(false),
                None,
            )
            .is_err()
    );
}

#[test]
fn test_sampling_is_reproducible_and_recorded_by_capture() {
    let options = SamplingOptions::new()
        .with_temperature(0.7)
        .with_top_k(Some(6))
        .with_top_p(0.9)
        .with_samples(4);
    let input = logits(Device::Cpu, 3, &linear_probabilities(10));
    let first = input.sample(&options, Some(&Generator::new(5))).unwrap();
    let second = input.sample(&options, Some(&Generator::new(5))).unwrap();
    assert_eq!(
        first.to_vec::<i64>().unwrap(),
        second.to_vec::<i64>().unwrap()
    );

    let generator = Generator::new(5);
    let (captured, _) = Graph::capture([("logits", input.clone())], |inputs| {
        Ok(vec![inputs[0].sample(&options, Some(&generator))?])
    })
    .unwrap();
    let inputs = HashMap::from([("logits".to_owned(), input)]);
    let replayed = captured.run(&inputs).unwrap();
    assert_eq!(replayed[0].shape().dimensions(), [3, 4]);
    assert!(
        replayed[0]
            .to_vec::<i64>()
            .unwrap()
            .iter()
            .all(|&index| index >= 4)
    );
}

#[cfg(all(feature = "testing", feature = "cuda"))]
#[test]
fn test_cuda_sampling_draws_the_same_tokens_as_cpu() {
    let _lock = device_lock(0).unwrap();
    let values = linear_probabilities(32);
    let cpu = logits(Device::Cpu, 2, &values);
    let cuda = cpu.to_device(Device::Cuda(0)).unwrap();
    for options in [
        SamplingOptions::new().with_samples(16),
        SamplingOptions::new()
            .with_temperature(0.8)
            .with_top_k(Some(10))
            .with_top_p(0.8)
            .with_samples(16),
        SamplingOptions::new()
            .with_samples(5)
            .with_replacement(false),
        SamplingOptions::new().with_temperature(0.),
    ] {
        let expected = cpu.sample(&options, Some(&Generator::new(3))).unwrap();
        let actual = cuda.sample(&options, Some(&Generator::new(3))).unwrap();
        assert_eq!(actual.device(), Device::Cuda(0));
        assert_eq!(
            actual.to_vec::<i64>().unwrap(),
            expected.to_vec::<i64>().unwrap(),
            "{options:?}"
        );
    }
}

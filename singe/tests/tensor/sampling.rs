use singe::{
    Device, Shape, Tensor,
    random::Generator,
    tensor::{Draws, SamplingOptions},
};

#[cfg(all(feature = "testing", feature = "cuda"))]
use singe_cuda::{device::Device as CudaDevice, testing::device_lock};

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
    let greedy = SamplingOptions::new().with_temperature(0.);
    for options in [vec![Some(greedy)], vec![Some(greedy), None], vec![None]] {
        let chosen = logits.sample(&options, Draws::Generator(None)).unwrap();
        assert_eq!(chosen.shape().dimensions(), [2]);
        assert_eq!(chosen.to_vec::<i64>().unwrap(), [1, 1], "{options:?}");
    }
}

#[test]
fn test_filtered_tokens_are_never_drawn_and_frequencies_match() {
    let generator = Generator::new(11);
    let rows = 20_000;
    let logits = logits(Device::Cpu, rows, &linear_probabilities(8));
    let top_k = SamplingOptions::new().with_top_k(Some(3));
    // Probabilities are 1/36, ..., 8/36; the four most likely sum to 26/36, the first past 0.6.
    let top_p = SamplingOptions::new().with_top_p(0.6);
    let check = |drawn: &[i64], least: i64, weights: &[(usize, f64)]| {
        assert!(drawn.iter().all(|&index| index >= least));
        let observed = frequencies(drawn, 8);
        let total = weights.iter().map(|(_, weight)| weight).sum::<f64>();
        for &(index, weight) in weights {
            assert!(
                (observed[index] - weight / total).abs() < 0.02,
                "{observed:?}"
            );
        }
    };
    let top_k_weights = [(5, 6.), (6, 7.), (7, 8.)];
    let top_p_weights = [(4, 5.), (5, 6.), (6, 7.), (7, 8.)];
    for (options, least, weights) in [
        (top_k, 5, &top_k_weights[..]),
        (top_p, 4, &top_p_weights[..]),
    ] {
        let drawn = logits
            .sample(&[Some(options)], Draws::Generator(Some(&generator)))
            .unwrap()
            .to_vec::<i64>()
            .unwrap();
        check(&drawn, least, weights);
    }

    // Rows alternate top-k and top-p in one call, each from a stream of its own.
    let options = (0..rows)
        .map(|row| Some(if row % 2 == 0 { top_k } else { top_p }))
        .collect::<Vec<_>>();
    let states = streams(Device::Cpu, rows);
    let drawn = logits
        .sample(&options, Draws::Streams(&states))
        .unwrap()
        .to_vec::<i64>()
        .unwrap();
    let alternate = |parity| {
        drawn
            .iter()
            .skip(parity)
            .step_by(2)
            .copied()
            .collect::<Vec<_>>()
    };
    check(&alternate(0), 5, &top_k_weights);
    check(&alternate(1), 4, &top_p_weights);

    // A row's draw depends only on its stream, not on the rows beside it.
    for row in [0, 1, 4_321] {
        let alone = logits
            .narrow(0, row, 1)
            .unwrap()
            .sample(
                &[options[usize::try_from(row).unwrap()]],
                Draws::Streams(&states.narrow(0, row, 1).unwrap().contiguous().unwrap()),
            )
            .unwrap()
            .to_vec::<i64>()
            .unwrap();
        assert_eq!(alone, [drawn[usize::try_from(row).unwrap()]]);
    }
}

fn streams(device: Device, rows: i64) -> Tensor {
    let states = (0..rows)
        .flat_map(|row| [row * 7 + 1, row % 5])
        .collect::<Vec<_>>();
    Tensor::from_vec(device, Shape::create([rows, 2]).unwrap(), states).unwrap()
}

#[cfg(all(feature = "testing", feature = "cuda"))]
#[test]
fn test_cuda_sampling_draws_the_same_tokens_as_cpu() {
    let _lock = device_lock(CudaDevice::new(0)).unwrap();
    let rows = 16;
    let cpu = logits(Device::Cpu, rows, &linear_probabilities(32));
    let cuda = cpu.to_device(Device::Cuda(0)).unwrap();
    let filtered = SamplingOptions::new()
        .with_temperature(0.8)
        .with_top_k(Some(10))
        .with_top_p(0.8);
    let mixed = (0..rows)
        .map(|row| match row % 4 {
            0 => None,
            1 => Some(SamplingOptions::new()),
            2 => Some(filtered),
            _ => Some(SamplingOptions::new().with_min_p(0.2)),
        })
        .collect::<Vec<_>>();
    for options in [
        vec![Some(SamplingOptions::new())],
        vec![Some(filtered)],
        mixed,
    ] {
        let expected = cpu
            .sample(&options, Draws::Generator(Some(&Generator::new(3))))
            .unwrap();
        let actual = cuda
            .sample(&options, Draws::Generator(Some(&Generator::new(3))))
            .unwrap();
        assert_eq!(actual.device(), Device::Cuda(0));
        assert_eq!(
            actual.to_vec::<i64>().unwrap(),
            expected.to_vec::<i64>().unwrap(),
            "{options:?}"
        );

        let expected = cpu
            .sample(&options, Draws::Streams(&streams(Device::Cpu, rows)))
            .unwrap();
        let actual = cuda
            .sample(&options, Draws::Streams(&streams(Device::Cuda(0), rows)))
            .unwrap();
        assert_eq!(
            actual.to_vec::<i64>().unwrap(),
            expected.to_vec::<i64>().unwrap(),
            "{options:?}"
        );
    }
}

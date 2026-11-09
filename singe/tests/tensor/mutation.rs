use std::collections::HashMap;

#[cfg(feature = "cuda")]
use singe::graph::GraphPrepareOptions;
use singe::{Device, Shape, Tensor, graph::Graph};

#[cfg(all(feature = "testing", feature = "cuda"))]
use singe_cuda::testing::device_lock;

fn matrix(device: Device) -> Tensor {
    Tensor::from_vec(
        device,
        Shape::create([2, 3]).unwrap(),
        vec![1_f32, 2., 3., 4., 5., 6.],
    )
    .unwrap()
}

fn assert_writes(device: Device) {
    let base = matrix(device);
    let column = base.narrow(1, 1, 1).unwrap();
    column.fill_(0_f32).unwrap();
    assert_eq!(base.to_vec::<f32>().unwrap(), [1., 0., 3., 4., 0., 6.]);
    assert_eq!(base.version(), 1);
    assert_eq!(column.version(), 1);

    let transposed = base.transpose(0, 1).unwrap();
    transposed
        .assign(
            &Tensor::from_vec(
                device,
                Shape::create([3, 2]).unwrap(),
                vec![10_f32, 40., 20., 50., 30., 60.],
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(
        base.to_vec::<f32>().unwrap(),
        [10., 20., 30., 40., 50., 60.]
    );

    base.add_(&base).unwrap();
    assert_eq!(
        base.to_vec::<f32>().unwrap(),
        [20., 40., 60., 80., 100., 120.]
    );

    let shifted = matrix(device);
    shifted
        .narrow(1, 1, 2)
        .unwrap()
        .assign(&shifted.narrow(1, 0, 2).unwrap())
        .unwrap();
    assert_eq!(shifted.to_vec::<f32>().unwrap(), [1., 1., 2., 4., 4., 5.]);

    let mask =
        Tensor::from_vec(device, Shape::create([3]).unwrap(), vec![true, false, true]).unwrap();
    let masked = matrix(device);
    masked.masked_fill_(&mask, -1_f32).unwrap();
    assert_eq!(
        masked.to_vec::<f32>().unwrap(),
        [-1., 2., -1., -1., 5., -1.]
    );

    let expanded = Tensor::from_vec(device, Shape::create([1, 3]).unwrap(), vec![1_f32; 3])
        .unwrap()
        .expand(Shape::create([2, 3]).unwrap())
        .unwrap();
    assert!(expanded.fill_(2_f32).is_err());
    assert_eq!(expanded.version(), 0);
}

#[test]
fn test_cpu_writes_are_visible_through_views_and_bump_versions() {
    assert_writes(Device::Cpu);
}

#[cfg(all(feature = "testing", feature = "cuda"))]
#[test]
fn test_cuda_writes_are_visible_through_views_and_bump_versions() {
    let _lock = device_lock(0).unwrap();
    assert_writes(Device::Cuda(0));
}

#[test]
fn test_in_place_results_must_fit_the_destination_type() {
    let integers =
        Tensor::from_vec(Device::Cpu, Shape::create([2]).unwrap(), vec![1_i64, 2]).unwrap();
    let floats =
        Tensor::from_vec(Device::Cpu, Shape::create([2]).unwrap(), vec![0.5_f32, 1.5]).unwrap();
    assert!(integers.add_(&floats).is_err());
    floats.add_(&integers).unwrap();
    assert_eq!(floats.to_vec::<f32>().unwrap(), [1.5, 3.5]);
}

fn assert_capture_replays_writes(device: Device) {
    let rows = |values: Vec<f32>| {
        Tensor::from_vec(device, Shape::create([2, 3]).unwrap(), values).unwrap()
    };
    let row = |values: Vec<f32>| {
        Tensor::from_vec(device, Shape::create([1, 3]).unwrap(), values).unwrap()
    };
    let cache = rows(vec![0.; 6]);
    let (captured, _) = Graph::capture(
        [("cache", cache.clone()), ("row", row(vec![1., 2., 3.]))],
        |inputs| {
            let slot = inputs[0].narrow(0, 1, 1)?;
            slot.assign(&inputs[1])?;
            Ok(vec![inputs[0].add(&inputs[0])?, slot])
        },
    )
    .unwrap();
    assert_eq!(cache.to_vec::<f32>().unwrap(), [0., 0., 0., 1., 2., 3.]);
    assert_eq!(cache.version(), 1);

    let replayed = rows(vec![5.; 6]);
    let outputs = captured
        .run(&HashMap::from([
            ("cache".to_owned(), replayed.clone()),
            ("row".to_owned(), row(vec![7., 8., 9.])),
        ]))
        .unwrap();
    assert_eq!(outputs.len(), 2);
    assert_eq!(
        outputs[0].to_vec::<f32>().unwrap(),
        [10., 10., 10., 14., 16., 18.]
    );
    assert_eq!(outputs[1].to_vec::<f32>().unwrap(), [7., 8., 9.]);
    assert_eq!(replayed.to_vec::<f32>().unwrap(), [5., 5., 5., 7., 8., 9.]);
    assert_eq!(replayed.version(), 1);

    let compiled = captured.compile().unwrap();
    let again = rows(vec![1.; 6]);
    compiled
        .run(&HashMap::from([
            ("cache".to_owned(), again.clone()),
            ("row".to_owned(), row(vec![2., 2., 2.])),
        ]))
        .unwrap();
    assert_eq!(again.to_vec::<f32>().unwrap(), [1., 1., 1., 2., 2., 2.]);
}

#[test]
fn test_cpu_capture_records_writes_and_replay_writes_back() {
    assert_capture_replays_writes(Device::Cpu);
}

#[cfg(all(feature = "testing", feature = "cuda"))]
#[test]
fn test_cuda_capture_records_writes_and_replay_writes_back() {
    let _lock = device_lock(0).unwrap();
    assert_capture_replays_writes(Device::Cuda(0));
}

/// Writes `x` into row `position` of `cache`, then attends from `x` over the cache.
fn decoder_step(cache: &Tensor, x: &Tensor, position: i64) -> singe::Result<Tensor> {
    cache.narrow(0, position, 1)?.assign(x)?;
    let scores = x.matmul(&cache.transpose(0, 1)?)?.softmax(1)?;
    scores.matmul(cache)
}

fn assert_decoder_cache_replays(device: Device) {
    let tensor = |shape: [i64; 2], values: Vec<f32>| {
        Tensor::from_vec(device, Shape::create(shape).unwrap(), values).unwrap()
    };
    let initial = vec![0.5_f32, -1., 0.25, 2., 0., 0., 0., 0.];
    let x = tensor([1, 2], vec![1., -0.5]);

    let eager_cache = tensor([4, 2], initial.clone());
    let eager = decoder_step(&eager_cache, &x, 2).unwrap();

    let capture_cache = tensor([4, 2], initial.clone());
    let (captured, _) = Graph::capture([("cache", capture_cache), ("x", x.clone())], |inputs| {
        Ok(vec![decoder_step(&inputs[0], &inputs[1], 2)?])
    })
    .unwrap();
    let replay_cache = tensor([4, 2], initial);
    let replayed = captured
        .run(&HashMap::from([
            ("cache".to_owned(), replay_cache.clone()),
            ("x".to_owned(), x),
        ]))
        .unwrap();

    assert_eq!(replayed.len(), 1);
    assert_eq!(
        replayed[0].to_vec::<f32>().unwrap(),
        eager.to_vec::<f32>().unwrap()
    );
    assert_eq!(
        replay_cache.to_vec::<f32>().unwrap(),
        eager_cache.to_vec::<f32>().unwrap()
    );
}

#[test]
fn test_cpu_decoder_cache_updates_match_eager_after_replay() {
    assert_decoder_cache_replays(Device::Cpu);
}

#[cfg(all(feature = "testing", feature = "cuda"))]
#[test]
fn test_cuda_decoder_cache_updates_match_eager_after_replay() {
    let _lock = device_lock(0).unwrap();
    assert_decoder_cache_replays(Device::Cuda(0));
}

#[cfg(all(feature = "testing", feature = "cuda"))]
#[test]
fn test_compiled_cuda_graphs_write_inputs_back_natively() {
    let _lock = device_lock(0).unwrap();
    let device = Device::Cuda(0);
    let cache = Tensor::from_vec(device, Shape::create([4, 3]).unwrap(), vec![0_f32; 12]).unwrap();
    let x = Tensor::from_vec(device, Shape::create([1, 3]).unwrap(), vec![1_f32, 2., 3.]).unwrap();
    let (captured, _) = Graph::capture([("cache", cache), ("x", x)], |inputs| {
        Ok(vec![decoder_step(&inputs[0], &inputs[1], 2)?])
    })
    .unwrap();
    let mut prepared = captured.prepare(GraphPrepareOptions::new(device)).unwrap();

    for step in 0..3 {
        let replayed =
            Tensor::from_vec(device, Shape::create([4, 3]).unwrap(), vec![0.5_f32; 12]).unwrap();
        let input = Tensor::from_vec(
            device,
            Shape::create([1, 3]).unwrap(),
            vec![step as f32, 1., 2.],
        )
        .unwrap();
        let reference = replayed.copy().unwrap();
        let expected = decoder_step(&reference, &input, 2).unwrap();
        let outputs = prepared
            .run(&HashMap::from([
                ("cache".to_owned(), replayed.clone()),
                ("x".to_owned(), input),
            ]))
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(
            outputs[0].to_vec::<f32>().unwrap(),
            expected.to_vec::<f32>().unwrap()
        );
        assert_eq!(
            replayed.to_vec::<f32>().unwrap(),
            reference.to_vec::<f32>().unwrap()
        );
    }
}

#[test]
fn test_training_batch_norm_updates_running_statistics_in_place() {
    let input = Tensor::from_vec(
        Device::Cpu,
        Shape::create([2, 1, 2]).unwrap(),
        vec![1_f32, 2., 3., 4.],
    )
    .unwrap();
    let shape = Shape::create([1]).unwrap();
    let mean = Tensor::from_vec(Device::Cpu, shape.clone(), vec![0_f32]).unwrap();
    let variance = Tensor::from_vec(Device::Cpu, shape, vec![1_f32]).unwrap();
    let output = input
        .batch_norm(None, None, Some(&mean), Some(&variance), true, 0.1, 0.0)
        .unwrap();
    // The batch's mean is 2.5, its biased variance 1.25 normalizes, and its unbiased variance 5/3 updates the running one.
    let deviation = 1.25_f32.sqrt();
    let expected = [-1.5_f32, -0.5, 0.5, 1.5].map(|value| value / deviation);
    for (actual, expected) in output.to_vec::<f32>().unwrap().iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
    }
    assert!((mean.to_vec::<f32>().unwrap()[0] - 0.25).abs() < 1e-6);
    let updated = 0.9_f32 + 0.1 * 5.0 / 3.0;
    assert!((variance.to_vec::<f32>().unwrap()[0] - updated).abs() < 1e-6);
}

use super::*;

#[test]
fn test_cuda_eager_causal_depthwise_conv1d_matches_the_cpu() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    // Mamba's short convolution: each channel convolves its own sequence with four taps, padded on the left so no point reads a later one.
    let (batch, channels, length, taps) = (2_i64, 6_i64, 37_i64, 4_i64);
    let values = |count: i64, seed: usize| {
        (0..usize::try_from(count).unwrap())
            .map(|index| ((index * 37 + seed * 11) % 23) as f32 / 11.5 - 1.0)
            .collect::<Vec<_>>()
    };
    let shape = |dimensions: &[i64]| Shape::create(dimensions.to_vec()).unwrap();
    let input = values(batch * channels * length, 1);
    let weight = values(channels * taps, 2);
    let bias = values(channels, 3);
    let convolve = |device: Device| {
        let input = Tensor::from_vec(device, shape(&[batch, channels, length]), input.clone())?;
        let weight = Tensor::from_vec(device, shape(&[channels, 1, taps]), weight.clone())?;
        let bias = Tensor::from_vec(device, shape(&[channels]), bias.clone())?;
        input.convolution1d(&weight, Some(&bias), 1, (taps - 1, 0), 1, channels)
    };
    let expected = convolve(Device::Cpu).unwrap();
    let Some(actual) = skip_if_stub(convolve(Device::Cuda(0))) else {
        return;
    };
    assert_eq!(actual.shape(), expected.shape());
    assert_close!(
        &actual.to_vec::<f32>().unwrap(),
        &expected.to_vec::<f32>().unwrap(),
        1e-5
    );
}

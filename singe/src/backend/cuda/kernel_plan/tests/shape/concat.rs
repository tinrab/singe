use super::*;

#[test]
fn test_cuda_eager_concat_f32_two_inputs_matches_expected() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let lhs_shape = Shape::create([3_i64, 17]).unwrap();
    let rhs_shape = Shape::create([3_i64, 26]).unwrap();
    let lhs_values = (0..51usize)
        .map(|value| (value as f32).mul_add(0.25, -3.0))
        .collect::<Vec<_>>();
    let rhs_values = (0..78usize)
        .map(|value| (value as f32).mul_add(-0.125, 5.0))
        .collect::<Vec<_>>();
    let expected = expected_concat_f32(&lhs_values, &rhs_values, &[3, 17], &[3, 26], 1);
    let Some(lhs) = skip_if_stub(Tensor::from_vec(Device::Cuda(0), &lhs_shape, lhs_values)) else {
        return;
    };
    let rhs = Tensor::from_vec(Device::Cuda(0), &rhs_shape, rhs_values).unwrap();

    let output = Tensor::concat(&[lhs, rhs], 1).unwrap();

    assert_eq!(output.shape().dimensions(), &[3_i64, 43]);
    assert_close!(&output.to_vec::<f32>().unwrap(), &expected, 1.0e-5);
}

#[test]
fn test_cuda_eager_concat_f32_two_strided_inputs_matches_expected() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let lhs_shape = Shape::create([2_i64, 3, 11]).unwrap();
    let rhs_shape = Shape::create([4_i64, 3, 11]).unwrap();
    let lhs_values = (0..66usize)
        .map(|value| (value as f32).mul_add(0.125, -4.0))
        .collect::<Vec<_>>();
    let rhs_values = (0..132usize)
        .map(|value| (value as f32).mul_add(-0.0625, 7.0))
        .collect::<Vec<_>>();
    let mut lhs_permuted_values = Vec::with_capacity(lhs_values.len());
    for i in 0..11usize {
        for j in 0..2usize {
            for k in 0..3usize {
                lhs_permuted_values.push(lhs_values[j * 3 * 11 + k * 11 + i]);
            }
        }
    }
    let mut rhs_permuted_values = Vec::with_capacity(rhs_values.len());
    for i in 0..11usize {
        for j in 0..4usize {
            for k in 0..3usize {
                rhs_permuted_values.push(rhs_values[j * 3 * 11 + k * 11 + i]);
            }
        }
    }
    let expected = expected_concat_f32(
        &lhs_permuted_values,
        &rhs_permuted_values,
        &[11, 2, 3],
        &[11, 4, 3],
        1,
    );
    let Some(lhs) = skip_if_stub(Tensor::from_vec(Device::Cuda(0), &lhs_shape, lhs_values)) else {
        return;
    };
    let rhs = Tensor::from_vec(Device::Cuda(0), &rhs_shape, rhs_values).unwrap();
    let lhs = lhs.permute(&[2, 0, 1]).unwrap();
    let rhs = rhs.permute(&[2, 0, 1]).unwrap();
    assert!(!lhs.is_contiguous());
    assert!(!rhs.is_contiguous());

    let output = Tensor::concat(&[lhs, rhs], 1).unwrap();

    assert_eq!(output.shape().dimensions(), &[11_i64, 6, 3]);
    assert_close!(&output.to_vec::<f32>().unwrap(), &expected, 1.0e-5);
}

// Concatenations of several inputs, of rank five, and along inner axes run the generated copy kernel.
#[test]
fn test_cuda_eager_concat_matches_the_cpu_across_strides_and_types() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let cases = [
        (vec![2_i64, 3], 3, 0),
        (vec![2_i64, 2, 2, 2, 2], 2, 0),
        (vec![2_i64, 2, 2, 2, 2], 2, 3),
    ];
    for (dimensions, count, dim) in cases {
        let shape = Shape::create(dimensions).unwrap();
        let inputs = (0..count)
            .map(|index| {
                let values = (0..shape.element_count())
                    .map(|value| (value + index * 100) as f32)
                    .collect::<Vec<_>>();
                Tensor::from_vec(Device::Cpu, &shape, values).unwrap()
            })
            .collect::<Vec<_>>();
        let Some(cuda) = skip_if_stub(
            inputs
                .iter()
                .map(|input| input.to_device(Device::Cuda(0)))
                .collect::<Result<Vec<_>, _>>(),
        ) else {
            return;
        };
        let expected = Tensor::concat(&inputs, dim).unwrap();
        let actual = Tensor::concat(&cuda, dim).unwrap();
        assert_eq!(actual.shape(), expected.shape());
        assert_eq!(
            actual
                .to_device(Device::Cpu)
                .unwrap()
                .to_vec::<f32>()
                .unwrap(),
            expected.to_vec::<f32>().unwrap()
        );
    }
}

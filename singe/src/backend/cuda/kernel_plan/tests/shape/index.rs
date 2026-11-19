use super::*;
use crate::idx;

fn cpu_indices(values: &[i64]) -> Tensor {
    Tensor::from_vec(
        Device::Cpu,
        Shape::create([i64::try_from(values.len()).unwrap()]).unwrap(),
        values.to_vec(),
    )
    .unwrap()
}

#[test]
fn test_cuda_index_and_index_put_match_cpu() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let shape = Shape::create([3, 4, 2]).unwrap();
    let values = (0..24).map(|value| value as f32).collect::<Vec<_>>();
    let cpu = Tensor::from_vec(Device::Cpu, &shape, values).unwrap();
    let Some(cuda) = skip_if_stub(cpu.to_device(Device::Cuda(0))) else {
        return;
    };
    let rows = cpu_indices(&[2, -3, 1]);
    let columns = Tensor::from_vec(
        Device::Cpu,
        Shape::create([1, 3]).unwrap(),
        vec![0_i64, 3, -1],
    )
    .unwrap();
    let expected = cpu.index(&idx![&rows, &columns]).unwrap();
    let actual = cuda.index(&idx![&rows, &columns]).unwrap();
    assert_eq!(actual.shape(), expected.shape());
    assert_eq!(
        actual.to_vec::<f32>().unwrap(),
        expected.to_vec::<f32>().unwrap()
    );

    let updates = Tensor::from_vec(
        Device::Cpu,
        Shape::create([1, 3, 2]).unwrap(),
        (0..6).map(|value| value as f32 * 10.).collect::<Vec<_>>(),
    )
    .unwrap();
    for accumulate in [false, true] {
        let expected = cpu
            .index_put(&idx![&rows, &columns], &updates, accumulate)
            .unwrap();
        let actual = cuda
            .index_put(&idx![&rows, &columns], &updates, accumulate)
            .unwrap();
        assert_eq!(
            actual.to_vec::<f32>().unwrap(),
            expected.to_vec::<f32>().unwrap()
        );
    }

    let repeated = cuda
        .index_add(
            0,
            &cpu_indices(&[1, 1]).to_device(Device::Cuda(0)).unwrap(),
            &cuda.index(&idx![0..2]).unwrap(),
        )
        .unwrap();
    let expected = cpu
        .index_add(0, &cpu_indices(&[1, 1]), &cpu.index(&idx![0..2]).unwrap())
        .unwrap();
    assert_eq!(
        repeated.to_vec::<f32>().unwrap(),
        expected.to_vec::<f32>().unwrap()
    );
}

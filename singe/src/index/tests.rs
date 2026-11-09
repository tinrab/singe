use std::collections::HashMap;

use crate::{
    data_type::DataType, device::Device, error::Error, graph::Graph, index::TensorIndex,
    layout::Layout, shape::Shape, storage::cpu::CpuStorage, tensor::Tensor,
};

fn range_tensor(shape: &[i64]) -> Tensor {
    let count = shape.iter().product::<i64>();
    let values = (0..count).map(|value| value as f32).collect::<Vec<_>>();
    Tensor::from_vec(Device::Cpu, Shape::create(shape.to_vec()).unwrap(), values).unwrap()
}

fn values(tensor: &Tensor) -> Vec<f32> {
    tensor.to_vec::<f32>().unwrap()
}

#[test]
fn test_index_integers_and_slices_select_views() {
    let t = range_tensor(&[2, 3, 4]);
    let row = t.index(&idx![1]).unwrap();
    assert_eq!(row.shape().dimensions(), [3, 4]);
    assert_eq!(values(&row)[0], 12.);
    let window = t.index(&idx![1, 1..3]).unwrap();
    assert_eq!(window.shape().dimensions(), [2, 4]);
    let last = t.index(&idx![-1, -1, -1]).unwrap();
    assert_eq!(last.shape().rank(), 0);
    assert_eq!(values(&last), [23.]);
}

#[test]
fn test_index_steps_ellipsis_and_new_axes() {
    let t = range_tensor(&[2, 6]);
    let stepped = t.index(&idx![.., 1..;2]).unwrap();
    assert_eq!(values(&stepped), [1., 3., 5., 7., 9., 11.]);
    let tail = t.index(&idx![..., 4..]).unwrap();
    assert_eq!(values(&tail), [4., 5., 10., 11.]);
    let expanded = t.index(&idx![None, 0, ..., None]).unwrap();
    assert_eq!(expanded.shape().dimensions(), [1, 6, 1]);
    let (start, end) = (4_i64, 2_i64);
    let reversed_bounds = t.index(&idx![.., start..end]).unwrap();
    assert_eq!(reversed_bounds.shape().dimensions(), [2, 0]);
}

#[test]
fn test_index_tensor_indices_and_masks() {
    let t = range_tensor(&[3, 4]);
    let rows = Tensor::from_vec(Device::Cpu, Shape::create([2]).unwrap(), vec![2_i64, 0]).unwrap();
    assert_eq!(
        values(&t.index(&idx![&rows]).unwrap()),
        [8., 9., 10., 11., 0., 1., 2., 3.]
    );
    let columns =
        Tensor::from_vec(Device::Cpu, Shape::create([2]).unwrap(), vec![1_i64, -1]).unwrap();
    assert_eq!(values(&t.index(&idx![&rows, &columns]).unwrap()), [9., 3.]);
    let separated = range_tensor(&[2, 3, 4]);
    let first = Tensor::from_vec(Device::Cpu, Shape::create([2]).unwrap(), vec![0_i64, 1]).unwrap();
    let last = Tensor::from_vec(Device::Cpu, Shape::create([2]).unwrap(), vec![3_i64, 0]).unwrap();
    let picked = separated.index(&idx![&first, .., &last]).unwrap();
    assert_eq!(picked.shape().dimensions(), [2, 3]);
    assert_eq!(values(&picked), [3., 7., 11., 12., 16., 20.]);
    let mask = t
        .greater(&Tensor::scalar(Device::Cpu, 8_f32).unwrap())
        .unwrap();
    assert_eq!(values(&t.index(&idx![&mask]).unwrap()), [9., 10., 11.]);
}

#[test]
fn test_index_put_writes_through_basic_and_tensor_indices() {
    let t = range_tensor(&[3, 4]);
    t.index_put_(
        &idx![.., 1..;2],
        &Tensor::scalar(Device::Cpu, -1_f32).unwrap(),
    )
    .unwrap();
    assert_eq!(values(&t)[..4], [0., -1., 2., -1.]);
    let rows = Tensor::from_vec(Device::Cpu, Shape::create([1]).unwrap(), vec![2_i64]).unwrap();
    let columns = Tensor::from_vec(Device::Cpu, Shape::create([1]).unwrap(), vec![0_i64]).unwrap();
    t.index_put_(
        &idx![&rows, &columns],
        &Tensor::scalar(Device::Cpu, 100_f32).unwrap(),
    )
    .unwrap();
    assert_eq!(values(&t)[8], 100.);
    let mask = t
        .less(&Tensor::scalar(Device::Cpu, 0_f32).unwrap())
        .unwrap();
    t.index_put_(&idx![&mask], &Tensor::scalar(Device::Cpu, 0_f32).unwrap())
        .unwrap();
    assert!(values(&t).iter().all(|value| *value >= 0.));
}

#[test]
fn test_index_rejects_invalid_entries() {
    let t = range_tensor(&[2, 3]);
    assert!(matches!(
        t.index(&[TensorIndex::Single(i64::MIN)]).unwrap_err(),
        Error::IndexOutOfBounds
    ));
    assert!(matches!(
        t.index(&idx![0, 0, 0]).unwrap_err(),
        Error::DimensionOutOfBounds
    ));
    assert!(t.index(&idx![..., 0, ...]).is_err());
    assert!(t.index(&idx![..;0]).is_err());
    let clamped = t.index(&[TensorIndex::from(i64::MIN..i64::MAX)]).unwrap();
    assert_eq!(values(&clamped), [0., 1., 2., 3., 4., 5.]);
}

#[test]
fn test_get_element() {
    let t = range_tensor(&[2, 3]);
    assert_eq!(t.get_element::<f32>(&[0, 0]).unwrap(), 0.0);
    assert_eq!(t.get_element::<f32>(&[1, 2]).unwrap(), 5.0);
}

#[test]
fn test_to_vec_non_contiguous() {
    let t = range_tensor(&[2, 3]);
    let tr = t.transpose(0, 1).unwrap();
    assert_eq!(values(&tr), [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
}

#[test]
fn test_slice_input_rejects_overflowing_storage_span() {
    let shape = Layout::with_strides(vec![2], vec![i64::MAX]).unwrap();
    let error = Tensor::create(
        CpuStorage::create(DataType::F32, 2).unwrap(),
        DataType::F32,
        shape,
    )
    .unwrap_err();
    assert!(matches!(error, Error::OutOfRange { .. }));
}

fn indices(values: &[i64]) -> Tensor {
    Tensor::from_vec(
        Device::Cpu,
        Shape::create([i64::try_from(values.len()).unwrap()]).unwrap(),
        values.to_vec(),
    )
    .unwrap()
}

#[test]
fn test_index_put_replaces_or_accumulates_repeated_positions() {
    let base = Tensor::zeroed_on(Device::Cpu, DataType::F32, Shape::create([4]).unwrap()).unwrap();
    let positions = indices(&[0, 2, 0]);
    let updates = Tensor::from_vec(
        Device::Cpu,
        Shape::create([3]).unwrap(),
        vec![1_f32, 2., 3.],
    )
    .unwrap();
    let summed = base.index_put(&idx![&positions], &updates, true).unwrap();
    assert_eq!(values(&summed), [4., 0., 2., 0.]);
    let replaced = base.index_put(&idx![&positions], &updates, false).unwrap();
    assert_eq!(values(&replaced), [3., 0., 2., 0.]);
    assert_eq!(values(&base), [0.; 4]);

    let rows = range_tensor(&[3, 2]);
    let added = rows
        .index_add(0, &indices(&[1, 1]), &range_tensor(&[2, 2]))
        .unwrap();
    assert_eq!(values(&added), [0., 1., 4., 7., 4., 5.]);
}

#[test]
fn test_index_with_several_tensors_is_captured_and_replays_new_indices() {
    let t = range_tensor(&[3, 4]);
    let (captured, _) = Graph::capture(
        [
            ("x", t.clone()),
            ("rows", indices(&[2, 0])),
            ("columns", indices(&[1, -1])),
        ],
        |inputs| Ok(vec![inputs[0].index(&idx![&inputs[1], &inputs[2]])?]),
    )
    .unwrap();
    let run = |rows: &[i64], columns: &[i64]| {
        captured.run(&HashMap::from([
            ("x".to_string(), t.clone()),
            ("rows".to_string(), indices(rows)),
            ("columns".to_string(), indices(columns)),
        ]))
    };
    assert_eq!(values(&run(&[2, 0], &[1, -1]).unwrap()[0]), [9., 3.]);
    assert_eq!(values(&run(&[1, 1], &[0, 3]).unwrap()[0]), [4., 7.]);
    assert!(matches!(
        run(&[3, 0], &[0, 0]),
        Err(Error::IndexOutOfBounds)
    ));
}

#[test]
fn test_index_put_with_several_tensors_writes_through_the_view() {
    let t = range_tensor(&[2, 3, 4]);
    let first = indices(&[0, 1]);
    let last = indices(&[3, 0]);
    let updates =
        Tensor::from_vec(Device::Cpu, Shape::create([2, 3]).unwrap(), vec![-1_f32; 6]).unwrap();
    t.index_put_(&idx![&first, .., &last], &updates).unwrap();
    let written = t.index(&idx![&first, .., &last]).unwrap();
    assert_eq!(written.shape().dimensions(), [2, 3]);
    assert_eq!(values(&written), [-1.; 6]);
    assert_eq!(values(&t.index(&idx![0, 0, 0]).unwrap()), [0.]);
}

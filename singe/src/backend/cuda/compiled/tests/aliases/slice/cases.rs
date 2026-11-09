use super::*;

#[test]
fn test_cuda_compiled_graph_accepts_slice_alias_for_non_f32_dtype() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    assert_i64_slice_alias(
        &[6],
        &[3],
        vec![10, 20, 30, 40, 50, 60],
        SliceAliasCase {
            dim: 0,
            start: 0,
            length: 3,
            expected_strides: &[1],
            expected_storage_offset: 0,
        },
    );
}

#[test]
fn test_cuda_compiled_graph_accepts_slice_alias_with_offset_for_non_f32_dtype() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    assert_i64_slice_alias(
        &[2, 3],
        &[2, 2],
        vec![1, 2, 3, 4, 5, 6],
        SliceAliasCase {
            dim: 1,
            start: 1,
            length: 2,
            expected_strides: &[3, 1],
            expected_storage_offset: 1,
        },
    );
}

#[test]
fn test_cuda_compiled_graph_accepts_slice_alias_with_negative_dim_for_non_f32_dtype() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    assert_i64_slice_alias(
        &[3, 2],
        &[3, 1],
        vec![10, 20, 30, 40, 50, 60],
        SliceAliasCase {
            dim: -1,
            start: 0,
            length: 1,
            expected_strides: &[2, 1],
            expected_storage_offset: 0,
        },
    );
}

#[test]
fn test_cuda_compiled_graph_accepts_slice_alias_at_end_with_zero_length() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let outputs = run_i64_slice_alias(
        &[3, 2],
        &[3, 0],
        vec![1, 2, 3, 4, 5, 6],
        SliceAliasCase {
            dim: 1,
            start: 2,
            length: 0,
            expected_strides: &[2, 1],
            expected_storage_offset: 2,
        },
    );
    let Some(outputs) = outputs else {
        return;
    };

    assert_eq!(outputs[0].shape().dimensions(), &[3, 0]);
    assert_eq!(outputs[0].element_count(), 0);
}

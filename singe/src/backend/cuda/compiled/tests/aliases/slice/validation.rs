use super::*;

#[test]
fn test_cuda_compiled_graph_rejects_slice_alias_with_negative_start() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let shape = contiguous_shape(&[6]);
    let output_shape = contiguous_shape(&[3]);
    if maybe_i64_tensor(&shape, vec![1_i64, 2, 3, 4, 5, 6]).is_none() {
        return;
    }
    let Some(output) = maybe_i64_output(&output_shape) else {
        return;
    };

    let mut graph = Graph::new();
    let input = add_i64_input(&mut graph, shape);
    assert!(matches!(
        add_i64_slice_operation(
            &mut graph,
            input,
            &output,
            SliceAliasCase {
                dim: 0,
                start: -1,
                length: 3,
                expected_strides: &[],
                expected_storage_offset: 0,
            },
        ),
        Err(Error::InvalidArgument { name }) if name == "graph slice start"
    ));
}

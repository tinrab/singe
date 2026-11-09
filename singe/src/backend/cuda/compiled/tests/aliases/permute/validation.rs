use super::*;

#[test]
fn test_cuda_graph_builder_rejects_permute_alias_with_duplicate_dimensions() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let Some(graph) = build_invalid_permute_graph(&Shape::create([2, 2]).unwrap(), vec![0, 0])
    else {
        return;
    };

    let Some(_) = skip_if_stub(runtime_for_device(Device::Cuda(0))) else {
        return;
    };
    assert!(matches!(
        graph,
        Err(Error::InvalidArgument { name }) if name == "permute order"
    ));
}

#[test]
fn test_cuda_graph_builder_rejects_permute_alias_with_invalid_dimension_count() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let Some(graph) = build_invalid_permute_graph(&Shape::create([2, 3]).unwrap(), vec![0, 1, 2])
    else {
        return;
    };

    let Some(_) = skip_if_stub(runtime_for_device(Device::Cuda(0))) else {
        return;
    };
    assert!(matches!(
        graph,
        Err(Error::LengthMismatch { name, .. }) if name == "permute order"
    ));
}

#[test]
fn test_cuda_graph_builder_rejects_permute_alias_with_out_of_bounds_dimension() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let Some(graph) = build_invalid_permute_graph(&Shape::create([2, 3]).unwrap(), vec![0, 5])
    else {
        return;
    };

    let Some(_) = skip_if_stub(runtime_for_device(Device::Cuda(0))) else {
        return;
    };
    assert!(matches!(graph, Err(Error::DimensionOutOfBounds)));
}

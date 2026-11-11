use super::*;

#[test]
fn test_cuda_backend_clear_cache_preserves_active_allocations() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let device = Device::Cuda(0);
    let Some(_) = skip_if_stub(runtime_for_device(device)) else {
        return;
    };

    clear_allocator_cache(device);
    let before = cuda::allocator_stats(device).unwrap();
    assert_eq!(before.cached_blocks(), 0);
    assert_eq!(before.cached_bytes(), 0);

    let live_byte_size: usize = 123_511;
    let cached_byte_size: usize = 123_513;
    let live_shape = byte_shape(live_byte_size);
    let cached_shape = byte_shape(cached_byte_size);

    let live_storage = Tensor::zeroed_on(device, DataType::U8, &live_shape).unwrap();
    let cached_storage = Tensor::zeroed_on(device, DataType::U8, &cached_shape).unwrap();
    drop(cached_storage);

    let before_clear = cuda::allocator_stats(device).unwrap();
    assert_eq!(
        before_clear.active_bytes(),
        before.active_bytes() + live_byte_size
    );
    assert_eq!(before_clear.cached_bytes(), cached_byte_size);
    assert!(before_clear.cached_blocks() > 0);

    clear_allocator_cache(device);
    let after_clear = cuda::allocator_stats(device).unwrap();
    assert_eq!(
        after_clear.active_bytes(),
        before.active_bytes() + live_byte_size
    );
    assert_eq!(
        after_clear.allocated_bytes(),
        before_clear.allocated_bytes()
    );
    assert_eq!(
        after_clear.allocation_count(),
        before_clear.allocation_count()
    );
    assert_eq!(
        after_clear.cache_hit_count(),
        before_clear.cache_hit_count()
    );
    assert_eq!(after_clear.cached_blocks(), 0);
    assert_eq!(after_clear.cached_bytes(), 0);

    drop(live_storage);
    clear_allocator_cache(device);
}

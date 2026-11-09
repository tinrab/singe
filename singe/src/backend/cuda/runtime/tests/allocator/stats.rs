use super::*;

#[test]
fn test_cuda_backend_allocator_stats_track_allocations() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let device = Device::Cuda(0);
    clear_allocator_cache(device);
    let byte_size: usize = 123_507;
    let shape = byte_shape(byte_size);
    let before = cuda::allocator_stats(device).unwrap();
    let expected_active = before.active_bytes() + byte_size;
    let expected_cached = before.cached_bytes();

    let storage = Tensor::zeroed_on(device, DataType::U8, &shape).unwrap();
    let after_allocate = cuda::allocator_stats(device).unwrap();
    assert_eq!(
        after_allocate.allocated_bytes(),
        before.allocated_bytes() + byte_size
    );
    assert_eq!(
        after_allocate.allocation_count(),
        before.allocation_count() + 1
    );
    assert_eq!(after_allocate.cache_hit_count(), before.cache_hit_count());
    assert_eq!(after_allocate.active_bytes(), expected_active);
    assert_eq!(after_allocate.cached_bytes(), expected_cached);
    assert_eq!(
        after_allocate.peak_bytes(),
        expected_active.max(before.peak_bytes())
    );
    drop(storage);

    let after_drop = cuda::allocator_stats(device).unwrap();
    assert_eq!(after_drop.active_bytes(), before.active_bytes());
    assert_eq!(after_drop.cached_bytes(), before.cached_bytes() + byte_size);
    assert_eq!(after_drop.allocation_count(), before.allocation_count() + 1);
    assert_eq!(after_drop.cache_hit_count(), before.cache_hit_count());

    let storage = Tensor::zeroed_on(device, DataType::U8, &shape).unwrap();
    let after_reuse = cuda::allocator_stats(device).unwrap();
    assert_eq!(
        after_reuse.allocated_bytes(),
        before.allocated_bytes() + byte_size
    );
    assert_eq!(
        after_reuse.allocation_count(),
        before.allocation_count() + 1
    );
    assert_eq!(after_reuse.cache_hit_count(), before.cache_hit_count() + 1);
    assert_eq!(after_reuse.active_bytes(), expected_active);
    assert_eq!(after_reuse.cached_bytes(), before.cached_bytes());
    drop(storage);

    let after_released = cuda::allocator_stats(device).unwrap();
    assert_eq!(
        after_released.allocated_bytes(),
        before.allocated_bytes() + byte_size
    );
    assert_eq!(after_released.active_bytes(), before.active_bytes());
    assert_eq!(
        after_released.cached_bytes(),
        before.cached_bytes() + byte_size
    );
    assert_eq!(
        after_released.allocation_count(),
        before.allocation_count() + 1
    );
    assert_eq!(
        after_released.cache_hit_count(),
        before.cache_hit_count() + 1
    );
}

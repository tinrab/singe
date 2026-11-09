use super::*;

#[test]
fn test_caching_allocator_reuses_released_blocks() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let Some(allocator) = caching_allocator() else {
        return;
    };
    allocator.clear_cache().unwrap();

    let byte_size = 123_457;
    let before = allocator.stats().unwrap();
    let storage = allocator.zeroed(DataType::U8, byte_size).unwrap();
    let after_allocate = allocator.stats().unwrap();
    assert_eq!(
        after_allocate.allocation_count(),
        before.allocation_count() + 1
    );
    assert_eq!(after_allocate.cache_hit_count(), before.cache_hit_count());
    assert_eq!(
        after_allocate.active_bytes(),
        before.active_bytes() + byte_size
    );
    assert!(after_allocate.peak_bytes() >= after_allocate.active_bytes());
    drop(storage);

    let after_drop = allocator.stats().unwrap();
    assert_eq!(after_drop.active_bytes(), before.active_bytes());
    assert_eq!(after_drop.cached_bytes(), before.cached_bytes() + byte_size);
    assert_eq!(after_drop.allocation_count(), before.allocation_count() + 1);
    assert_eq!(after_drop.cache_hit_count(), before.cache_hit_count());

    let before_reuse = allocator.cached_block_count_for(byte_size).unwrap();
    assert!(before_reuse > 0);

    let storage = allocator.zeroed(DataType::U8, byte_size).unwrap();
    let after_reuse = allocator.stats().unwrap();
    assert_eq!(
        after_reuse.cache_hit_count(),
        after_drop.cache_hit_count() + 1
    );
    assert_eq!(
        after_reuse.allocation_count(),
        after_drop.allocation_count()
    );
    assert_eq!(
        after_reuse.active_bytes(),
        before.active_bytes() + byte_size
    );
    assert_eq!(after_reuse.cached_bytes(), before.cached_bytes());
    assert_eq!(
        allocator.cached_block_count_for(byte_size).unwrap(),
        before_reuse - 1
    );
    drop(storage);

    let after_release = allocator.stats().unwrap();
    assert_eq!(after_release.active_bytes(), before.active_bytes());
    assert_eq!(
        after_release.cached_bytes(),
        before.cached_bytes() + byte_size
    );
    assert_eq!(
        allocator.cached_block_count_for(byte_size).unwrap(),
        before_reuse
    );
    let stats = allocator.stats().unwrap();
    assert!(stats.cached_blocks() >= before_reuse);
    assert!(stats.cached_bytes() >= byte_size * before_reuse);
}

use super::*;

#[test]
fn test_caching_allocator_clear_cache_releases_blocks() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let Some(allocator) = caching_allocator() else {
        return;
    };

    let byte_size = 123_499;
    let storage = allocator.zeroed(DataType::U8, byte_size).unwrap();
    drop(storage);

    let before_clear = allocator.cached_block_count_for(byte_size).unwrap();
    assert!(before_clear > 0);

    allocator.clear_cache().unwrap();
    assert_eq!(allocator.cached_block_count_for(byte_size).unwrap(), 0);
    assert_eq!(allocator.stats().unwrap().cached_blocks(), 0);
}

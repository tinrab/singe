use super::*;
use crate::{backend::cuda::CudaStream, storage::cuda::CudaMemoryRecycler};

#[test]
fn test_caching_allocator_reuses_stream_ordered_blocks() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let Some(stream) = skip_if_stub(CudaStream::create(Device::Cuda(0))) else {
        return;
    };
    let Some(allocator) = caching_allocator_with_stream(stream.raw_arc()) else {
        return;
    };

    let byte_size = 123_491;
    let storage = allocator.zeroed(DataType::U8, byte_size).unwrap();
    drop(storage);

    let before_reuse = allocator.cached_block_count_for(byte_size).unwrap();
    assert!(before_reuse > 0);

    let storage = allocator.zeroed(DataType::U8, byte_size).unwrap();
    stream.synchronize().unwrap();
    drop(storage);
    assert_eq!(
        allocator.cached_block_count_for(byte_size).unwrap(),
        before_reuse
    );
}

#[test]
fn test_caching_allocator_reuses_blocks_with_cross_stream_wait_dependency() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let Some(context) = skip_if_stub(cuda_context_for_device(0)) else {
        return;
    };
    let Some(stream1) = default_stream(&context) else {
        return;
    };
    let Some(stream2) = default_stream(&context) else {
        return;
    };

    let stream1 = Arc::new(stream1);
    let stream2 = Arc::new(stream2);

    let Some(allocator_stream1) = caching_allocator_with_stream(Arc::clone(&stream1)) else {
        return;
    };
    let Some(allocator_stream2) = caching_allocator_with_stream(Arc::clone(&stream2)) else {
        return;
    };

    allocator_stream1.clear_cache().unwrap();
    let byte_size = 1_048_576;
    let shape = Some(vec![i64::try_from(byte_size).unwrap()]);
    let mut memory = allocator_stream1
        .take_or_create(byte_size, DataType::U8, shape.clone())
        .unwrap();
    let reused_ptr = memory.as_ptr();
    for _ in 0..1024 {
        // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
        unsafe {
            memory
                .set_value_async_unchecked(0x11, stream1.as_ref())
                .unwrap();
        }
    }

    let producer_done = record_disabled_timing_event(&context, &stream1);
    let producer_query = record_disabled_timing_event(&context, &stream1);
    allocator_stream1.cache.recycle(memory, Some(producer_done));

    let mut reused_memory = allocator_stream2
        .take_or_create(byte_size, DataType::U8, shape)
        .unwrap();
    assert_eq!(reused_memory.as_ptr(), reused_ptr);

    // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        reused_memory
            .set_value_async_unchecked(0x22, stream2.as_ref())
            .unwrap();
    }
    let consumer_done = record_disabled_timing_event(&context, &stream2);

    let mut violates_ordering = false;
    for _ in 0..5_000 {
        let producer_finished = producer_query.query().unwrap();
        let consumer_finished = consumer_done.query().unwrap();
        if consumer_finished && !producer_finished {
            violates_ordering = true;
            break;
        }
        if producer_finished && consumer_finished {
            break;
        }
    }

    producer_query.synchronize().unwrap();
    consumer_done.synchronize().unwrap();
    // SAFETY: the completed full-range memset initialized every byte.
    let reused_memory = unsafe { reused_memory.assume_init() };
    let sample = reused_memory.copy_to_host_vec().unwrap();
    let first_byte = sample[0];

    allocator_stream2
        .cache
        .recycle(reused_memory.into_uninit(), None);
    assert!(!violates_ordering);
    assert_eq!(first_byte, 0x22);
    assert_eq!(
        allocator_stream1.cached_block_count_for(byte_size).unwrap(),
        1
    );
}

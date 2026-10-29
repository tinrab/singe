//! Allocate, fill and free stream-ordered buffers from an explicit memory pool.

use std::error::Error;

use singe_cuda::{
    context::Context,
    memory::{
        AsyncDeviceMemory, MemoryAllocationHandleType, MemoryAllocationType, MemoryLocation,
        MemoryPool, MemoryPoolAttribute, MemoryPoolAttributeValue, MemoryPoolProps,
        MemoryPoolUsage, transfer,
    },
};

fn main() -> Result<(), Box<dyn Error>> {
    let context = Context::create()?;
    let stream = context.create_stream()?;
    let mut pool = MemoryPool::create(
        &context,
        MemoryPoolProps {
            alloc_type: MemoryAllocationType::Pinned,
            handle_type: MemoryAllocationHandleType::None,
            location: MemoryLocation::Device(context.device().id()),
            max_size: 0,
            usage: MemoryPoolUsage::empty(),
            gpu_direct_rdma_capable: false,
        },
    )?;
    // Keep up to 8 MiB reserved across synchronization so later allocations can
    // reuse backing memory. This is a retention threshold, not a size limit.
    pool.set_attribute(
        MemoryPoolAttribute::ReleaseThreshold,
        MemoryPoolAttributeValue::Bytes(8 * 1024 * 1024),
    )?;

    for value in [7_u8, 42] {
        let mut buffer = AsyncDeviceMemory::<u8>::zeroed_in_pool(4096, &pool, &stream)?;
        buffer.set_value_async(value)?;
        let output = transfer::copy_to_host_vec(&buffer)?;
        assert_eq!(output, vec![value; 4096]);
        // Enqueue the free on the allocation's stream before the next allocation.
        buffer.free_async()?;
    }
    stream.synchronize()?;
    println!(
        "reserved after freeing buffers: {:?}",
        pool.attribute(MemoryPoolAttribute::ReservedMemoryCurrent)?
    );
    // Return unused backing memory after all allocations have been freed.
    pool.trim_to(0)?;
    println!("verified both buffers and trimmed the pool");
    Ok(())
}

use std::{
    alloc::{Allocator, Layout},
    sync::Arc,
};

use super::*;
use crate::{
    device::DeviceAttribute,
    error::Status,
    stream::{Stream, StreamOptions},
    testing,
    virtual_memory::AllocationHandleTypes,
};

#[test]
fn test_it_works() -> Result<()> {
    let test_device = testing::bootstrap()?;
    let context = test_device.context();
    // SAFETY: the test buffers outlive the synchronous copies in this block.
    unsafe {
        let host_in = [1, 2, 3];

        let device_ptr = DeviceMemory::alloc(context, 3)?;
        let attributes = PointerAttributes::query_non_null(NonNull::new(device_ptr).unwrap())?;
        assert_eq!(attributes.memory_type, MemoryType::Device);

        DeviceMemory::copy(
            context,
            device_ptr,
            host_in.as_ptr(),
            3,
            MemoryCopyKind::HostToDevice,
        )?;
        let mut host_out = [0, 0, 0];
        DeviceMemory::copy(
            context,
            host_out.as_mut_ptr(),
            device_ptr,
            3,
            MemoryCopyKind::DeviceToHost,
        )?;
        assert_eq!(host_out, host_in);

        DeviceMemory::free(context, device_ptr)?;
    }
    Ok(())
}

#[test]
fn test_scoped_async_copy_round_trip() -> Result<()> {
    let test_device = testing::bootstrap()?;
    let context = Arc::clone(test_device.context());
    let stream = context.create_stream(StreamOptions::default())?;

    let host_in = [4_i32, 5, 6];
    let mut device = DeviceMemory::<i32>::zeroed(&context, host_in.len())?;
    let mut host_out = [0_i32; 3];

    stream.sync_scope(|scope| device.copy_from_host_scoped(&host_in, scope))?;
    stream.sync_scope(|scope| device.copy_to_host_scoped(&mut host_out, scope))?;

    assert_eq!(host_out, host_in);

    Ok(())
}

#[test]
fn test_stream_ordered_device_memory_frees_on_stream() -> Result<()> {
    let test_device = testing::bootstrap()?;
    let context = Arc::clone(test_device.context());
    let stream = context.create_stream(StreamOptions::default())?;

    let mut memory = DeviceMemory::<u32>::zeroed_stream_ordered(8, &stream)?;
    assert_eq!(memory.len(), 8);
    assert_eq!(memory.byte_len(), size_of::<[u32; 8]>());
    stream.sync_scope(|scope| memory.set_value_scoped(0, scope))?;
    memory.release()?;
    stream.synchronize()?;

    Ok(())
}

#[test]
fn test_stream_ordered_device_memory_allocates_from_pool() -> Result<()> {
    let test_device = testing::bootstrap()?;
    let context = Arc::clone(test_device.context());
    let stream = context.create_stream(StreamOptions::default())?;
    let props = MemoryPoolProps {
        alloc_type: MemoryAllocationType::Pinned,
        handle_types: AllocationHandleTypes::empty(),
        location: MemoryLocation::Device(context.device().id()),
        max_size: 0,
        usage: MemoryPoolUsage::empty(),
        gpu_direct_rdma_capable: false,
    };
    let pool = MemoryPool::create(&context, props)?;

    let memory = DeviceMemory::<u32>::zeroed_in_pool(8, &pool, &stream)?;
    assert_eq!(memory.len(), 8);
    assert_eq!(memory.stream().map(Stream::as_raw), Some(stream.as_raw()));
    memory.release()?;
    stream.synchronize()?;

    Ok(())
}

#[test]
fn test_pinned_vec_grows_and_reuses_blocks() -> Result<()> {
    let test_device = testing::bootstrap()?;
    let context = Arc::clone(test_device.context());
    let pinned = context.pinned_host();

    let mut values = PinnedVec::new_in(pinned.clone());
    for value in 0_u32..10_000 {
        values.push(value);
    }

    assert!(values.iter().copied().eq(0..10_000));
    assert_eq!(pinned.stats().active_bytes, 65_536);
    drop(values);
    assert_eq!(pinned.stats().active_bytes, 0);

    let chunks = pinned.stats().chunks;
    let source = vec![7_u64; 4096];
    let values = pinned.from_slice(&source)?;
    assert_eq!(values.len(), 4096);
    assert_eq!(pinned.stats().chunks, chunks);

    Ok(())
}

#[test]
fn test_pinned_vec_round_trips_through_device() -> Result<()> {
    let test_device = testing::bootstrap()?;
    let context = Arc::clone(test_device.context());
    let stream = context.create_stream(StreamOptions::default())?;
    let pinned = context.pinned_host();

    let host_in = pinned.from_slice(&[10_i32, 11, 12])?;
    let mut device = DeviceMemory::<i32>::zeroed(&context, host_in.len())?;
    device.copy_from_pinned(&host_in)?;
    let mut host_out = pinned.zeroed::<i32>(host_in.len())?;
    stream.sync_scope(|scope| device.copy_to_pinned_scoped(&mut host_out, scope))?;

    assert_eq!(host_out, host_in);

    Ok(())
}

#[test]
fn test_pinned_layouts() -> Result<()> {
    let test_device = testing::bootstrap()?;
    let context = Arc::clone(test_device.context());
    let pinned = context.pinned_host();

    let empty = pinned.with_capacity::<u64>(0)?;
    assert_eq!(empty.capacity(), 0);
    assert_eq!(pinned.stats().chunks, 0);

    let page = Layout::from_size_align(16, 4096).unwrap();
    let block = pinned.allocate(page).unwrap();
    assert!(block.cast::<u8>().addr().get().is_multiple_of(4096));
    // SAFETY: the block came from this allocator with this layout.
    unsafe { pinned.deallocate(block.cast(), page) };

    let too_aligned = Layout::from_size_align(16, 8192).unwrap();
    assert!(pinned.allocate(too_aligned).is_err());
    assert!(matches!(
        pinned.with_capacity::<Page>(1),
        Err(Error::UnsupportedPinnedAlignment { align: 8192 })
    ));

    let mut values = pinned.from_slice(&[1_u32, 2, 3])?;
    let block = values.as_ptr();
    values.extend([4, 5, 6, 7, 8]);
    assert_eq!(values.as_ptr(), block);
    values.truncate(2);
    values.shrink_to_fit();
    assert_eq!(values.as_ptr(), block);
    assert_eq!(values, [1, 2]);
    values.extend(0..1024);
    assert_ne!(values.as_ptr(), block);
    assert_eq!(values[..3], [1, 2, 0]);

    Ok(())
}

#[test]
fn test_pinned_vec_mapped_device_view() -> Result<()> {
    let test_device = testing::bootstrap()?;
    let context = Arc::clone(test_device.context());
    let device = DeviceMemory::from_slice(&context, &[3_i32, 4, 5])?;
    let mut host = context.pinned_host().zeroed::<i32>(3)?;

    transfer::copy_device(&mut host.mapped_device_view_mut()?, &device)?;
    assert_eq!(host, [3, 4, 5]);
    assert_eq!(host.mapped_device_view()?.len(), 3);

    Ok(())
}

#[test]
fn test_pinned_release_cached() -> Result<()> {
    let test_device = testing::bootstrap()?;
    let context = Arc::clone(test_device.context());
    let pinned = context.pinned_host();

    let small = pinned.with_capacity::<u8>(100)?;
    let large = pinned.with_capacity::<u8>(5 << 20)?;
    assert_eq!(pinned.stats().reserved_bytes, 8 << 20);

    drop(large);
    let other = pinned.with_capacity::<u8>(3 << 20)?;
    assert_eq!(pinned.stats().reserved_bytes, 6 << 20);

    drop(other);
    pinned.release_cached()?;
    assert_eq!(pinned.stats().chunks, 1);

    drop(small);
    pinned.release_cached()?;
    assert_eq!(pinned.stats(), PinnedStats::default());

    Ok(())
}

#[expect(dead_code, reason = "only the alignment matters")]
#[repr(align(8192))]
struct Page(u8);

#[test]
fn test_pitched_device_memory_copy_round_trip() -> Result<()> {
    let test_device = testing::bootstrap()?;
    let context = Arc::clone(test_device.context());

    let host_in = vec![1_u32, 2, 3, 4, 5, 6];
    let mut pitched = PitchedDeviceMemory::from_slice(&context, 3, 2, &host_in)?;
    assert_eq!(pitched.width(), 3);
    assert_eq!(pitched.height(), 2);
    assert!(pitched.pitch() >= 3 * size_of::<u32>());
    pitched.copy_from_host(&host_in)?;

    let mut copied = PitchedDeviceMemory::zeroed(&context, 3, 2)?;
    copied.copy_from_pitched(&pitched)?;

    let mut host_out = vec![0_u32; host_in.len()];
    copied.copy_to_host(&mut host_out)?;
    assert_eq!(host_out, host_in);

    Ok(())
}

#[test]
fn test_pitched_device_memory_scoped_async_copy_round_trip() -> Result<()> {
    let test_device = testing::bootstrap()?;
    let context = Arc::clone(test_device.context());
    let stream = context.create_stream(StreamOptions::default())?;

    let host_in = vec![10_u32, 11, 12, 13, 14, 15];
    let mut pitched = PitchedDeviceMemory::zeroed(&context, 2, 3)?;
    let mut host_out = vec![0_u32; host_in.len()];

    stream.sync_scope(|scope| pitched.copy_from_host_scoped(&host_in, scope))?;
    stream.sync_scope(|scope| pitched.copy_to_host_scoped(&mut host_out, scope))?;

    assert_eq!(host_out, host_in);

    Ok(())
}

#[test]
fn test_pitched_device_memory_allows_empty_shapes() -> Result<()> {
    let test_device = testing::bootstrap()?;
    let context = Arc::clone(test_device.context());

    let mut pitched = PitchedDeviceMemory::<u32>::zeroed(&context, 0, 8)?;
    assert!(pitched.is_empty());
    assert_eq!(pitched.row_byte_width()?, 0);
    assert_eq!(pitched.byte_len()?, 0);
    pitched.copy_from_host(&[])?;
    pitched.copy_to_host(&mut [])?;

    Ok(())
}

#[test]
fn test_pitched_device_volume_copy_round_trip() -> Result<()> {
    let test_device = testing::bootstrap()?;
    let context = Arc::clone(test_device.context());

    let host_in: Vec<u32> = (0..24).collect();
    let mut volume = PitchedDeviceVolume::from_slice(&context, 4, 3, 2, &host_in)?;
    assert_eq!(volume.width(), 4);
    assert_eq!(volume.height(), 3);
    assert_eq!(volume.depth(), 2);
    assert!(volume.pitch() >= 4 * size_of::<u32>());
    volume.copy_from_host(&host_in)?;

    let mut copied = PitchedDeviceVolume::zeroed(&context, 4, 3, 2)?;
    copied.copy_from_volume(&volume)?;

    let mut host_out = vec![0_u32; host_in.len()];
    copied.copy_to_host(&mut host_out)?;
    assert_eq!(host_out, host_in);

    Ok(())
}

#[test]
fn test_pitched_device_volume_scoped_async_copy_round_trip() -> Result<()> {
    let test_device = testing::bootstrap()?;
    let context = Arc::clone(test_device.context());
    let stream = context.create_stream(StreamOptions::default())?;

    let host_in: Vec<u32> = (100..112).collect();
    let mut volume = PitchedDeviceVolume::zeroed(&context, 3, 2, 2)?;
    let mut host_out = vec![0_u32; host_in.len()];

    stream.sync_scope(|scope| volume.copy_from_host_scoped(&host_in, scope))?;
    stream.sync_scope(|scope| volume.copy_to_host_scoped(&mut host_out, scope))?;

    assert_eq!(host_out, host_in);

    Ok(())
}

#[test]
fn test_pitched_device_volume_allows_empty_shapes() -> Result<()> {
    let test_device = testing::bootstrap()?;
    let context = Arc::clone(test_device.context());

    let mut volume = PitchedDeviceVolume::<u32>::zeroed(&context, 4, 0, 2)?;
    assert!(volume.is_empty());
    assert_eq!(volume.row_byte_width()?, 4 * size_of::<u32>());
    assert_eq!(volume.byte_len()?, 0);
    volume.copy_from_host(&[])?;
    volume.copy_to_host(&mut [])?;

    Ok(())
}

#[test]
fn test_registered_host_empty_slices_are_noops() -> Result<()> {
    let test_device = testing::bootstrap()?;
    let context = Arc::clone(test_device.context());

    let mut values: Vec<u32> = Vec::new();
    let registered =
        RegisteredHostMemory::create(&context, &mut values, HostRegisterFlags::DEFAULT)?;
    assert!(registered.is_empty());
    assert_eq!(registered.byte_len()?, 0);

    let readonly = RegisteredHostSlice::create(&context, &values, HostRegisterFlags::READ_ONLY)?;
    assert!(readonly.is_empty());
    assert_eq!(readonly.byte_len()?, 0);
    assert!(readonly.flags().contains(HostRegisterFlags::READ_ONLY));

    Ok(())
}

#[test]
fn test_registered_host_memory_tracks_borrow() -> Result<()> {
    let test_device = testing::bootstrap()?;
    let context = Arc::clone(test_device.context());

    let mut values = vec![1_u32, 2, 3, 4];
    {
        let mut registered =
            RegisteredHostMemory::create(&context, &mut values, HostRegisterFlags::DEFAULT)?;
        assert_eq!(registered.as_slice(), &[1, 2, 3, 4]);
        registered.as_slice_mut()[1] = 7;
        assert_eq!(registered.byte_len()?, 4 * size_of::<u32>());
    }
    assert_eq!(values, vec![1, 7, 3, 4]);

    let read_only_supported = context
        .device()
        .attribute(DeviceAttribute::ReadOnlyHostRegisterSupported)?
        != 0;
    match RegisteredHostSlice::create(&context, &values, HostRegisterFlags::READ_ONLY) {
        Ok(read_only) => {
            assert!(read_only_supported);
            assert_eq!(read_only.as_slice(), values.as_slice());
            assert!(read_only.flags().contains(HostRegisterFlags::READ_ONLY));
        }
        Err(Error::Cuda {
            code: Status::NotSupported,
            ..
        }) => assert!(!read_only_supported),
        Err(error) => return Err(error),
    }

    Ok(())
}

#[test]
fn test_managed_memory_advise_uses_typed_wrapper() -> Result<()> {
    let test_device = testing::bootstrap()?;
    let context = Arc::clone(test_device.context());
    let properties = context.device().properties()?;
    let managed = ManagedMemory::<u32>::zeroed(&context, 4, MemoryAttachFlags::GLOBAL)?;
    let host = MemoryLocation::Host;

    managed.set_read_mostly()?;
    managed.unset_read_mostly()?;
    managed.set_preferred_location(host)?;
    managed.unset_preferred_location()?;

    if properties.concurrent_managed_access {
        let device = MemoryLocation::Device(context.device().id());
        managed.set_accessed_by(device)?;
        managed.unset_accessed_by(device)?;
    }

    Ok(())
}

#[test]
fn test_managed_memory_tracks_metadata_and_views() -> Result<()> {
    let test_device = testing::bootstrap()?;
    let context = Arc::clone(test_device.context());
    let mut managed = ManagedMemory::<u32>::zeroed(&context, 4, MemoryAttachFlags::HOST)?;
    // SAFETY: the test synchronizes the device before touching managed memory from the host.
    unsafe {
        managed.as_host_slice_mut().copy_from_slice(&[1, 2, 3, 4]);
    }
    assert_eq!(managed.len(), 4);
    assert_eq!(managed.byte_len(), 4 * size_of::<u32>());
    assert_eq!(managed.attach_flags(), MemoryAttachFlags::HOST);
    assert_eq!(managed.view().len(), 4);
    assert_eq!(managed.view_mut().len(), 4);
    // SAFETY: the test synchronizes the device before touching managed memory from the host.
    unsafe {
        assert_eq!(managed.as_host_slice(), &[1, 2, 3, 4]);
        managed.as_host_slice_mut()[2] = 9;
        assert_eq!(managed.as_host_slice()[2], 9);
    }
    let address = managed.as_ptr();
    let raw = managed.into_raw().unwrap();
    assert_eq!(raw.ptr.cast_const(), address);
    assert_eq!(raw.len, 4);
    assert_eq!(raw.attach_flags, MemoryAttachFlags::HOST);
    // SAFETY: reconstruct the unique owner using its original allocation and context.
    let _managed = unsafe { ManagedMemory::from_raw(raw) }?;
    Ok(())
}

use std::alloc::Allocator;

use super::*;
use crate::{error::Status, testing};

#[test]
fn test_it_works() -> Result<()> {
    // SAFETY: the test buffers outlive the synchronous copies in this block.
    unsafe {
        let host_in = [1, 2, 3];

        let device_ptr = DeviceMemory::alloc(3)?;
        let attributes = PointerAttributes::query_non_null(NonNull::new(device_ptr).unwrap())?;
        assert_eq!(attributes.memory_type, MemoryType::Device);

        DeviceMemory::copy(
            device_ptr,
            host_in.as_ptr(),
            3,
            MemoryCopyKind::HostToDevice,
        )?;
        let mut host_out = [0, 0, 0];
        DeviceMemory::copy(
            host_out.as_mut_ptr(),
            device_ptr,
            3,
            MemoryCopyKind::DeviceToHost,
        )?;
        assert_eq!(host_out, host_in);

        DeviceMemory::free(device_ptr)?;
    }
    Ok(())
}

#[test]
fn test_scoped_async_copy_round_trip() -> Result<()> {
    let (_lock, context) = testing::bootstrap()?;
    let stream = context.create_stream()?;

    let host_in = [4_i32, 5, 6];
    // SAFETY: the buffer is fully written by the operation before it is read.
    let mut device = unsafe { DeviceMemory::create_unchecked(&context, host_in.len()) }?;
    let mut host_out = [0_i32; 3];

    stream.sync_scope(|scope| device.copy_from_host_async(&host_in, scope))?;
    stream.sync_scope(|scope| device.copy_to_host_async(&mut host_out, scope))?;

    assert_eq!(host_out, host_in);

    Ok(())
}

#[test]
fn test_async_device_memory_frees_on_stream() -> Result<()> {
    let (_lock, context) = testing::bootstrap()?;
    let stream = context.create_stream()?;

    let mut memory = AsyncDeviceMemory::<u32>::zeroed(8, &stream)?;
    assert_eq!(memory.len(), 8);
    assert_eq!(memory.byte_len(), size_of::<[u32; 8]>());
    memory.set_value_async(0)?;
    memory.free_async()?;
    stream.synchronize()?;

    Ok(())
}

#[test]
fn test_async_device_memory_allocates_from_pool() -> Result<()> {
    let (_lock, context) = testing::bootstrap()?;
    let stream = context.create_stream()?;
    let props = MemoryPoolProps {
        alloc_type: MemoryAllocationType::Pinned,
        handle_type: MemoryAllocationHandleType::None,
        location: MemoryLocation::Device(context.device().id()),
        max_size: 0,
        usage: MemoryPoolUsage::empty(),
        gpu_direct_rdma_capable: false,
    };
    let pool = match MemoryPool::create(&context, props) {
        Ok(pool) => pool,
        Err(Error::Cuda {
            code: Status::NotSupported,
            ..
        }) => return Ok(()),
        Err(error) => return Err(error),
    };

    let memory = AsyncDeviceMemory::<u32>::zeroed_in_pool(8, &pool, &stream)?;
    assert_eq!(memory.len(), 8);
    assert_eq!(memory.stream().as_raw(), stream.as_raw());
    memory.free_async()?;
    stream.synchronize()?;

    Ok(())
}

#[test]
fn test_pinned_vec_grows_and_reuses_blocks() -> Result<()> {
    let (_lock, context) = testing::bootstrap()?;
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
    let values = pinned.from_slice(&[7_u64; 4096])?;
    assert_eq!(values.len(), 4096);
    assert_eq!(pinned.stats().chunks, chunks);

    Ok(())
}

#[test]
fn test_pinned_vec_round_trips_through_device() -> Result<()> {
    let (_lock, context) = testing::bootstrap()?;
    let stream = context.create_stream()?;
    let pinned = context.pinned_host();

    let host_in = pinned.from_slice(&[10_i32, 11, 12])?;
    // SAFETY: the buffer is fully written by the operation before it is read.
    let mut device = unsafe { DeviceMemory::create_unchecked(&context, host_in.len()) }?;
    device.copy_from_pinned(&host_in)?;
    let mut host_out = pinned.zeroed::<i32>(host_in.len())?;
    stream.sync_scope(|scope| device.copy_to_pinned_async(&mut host_out, scope))?;

    assert_eq!(host_out, host_in);

    Ok(())
}

#[test]
fn test_pinned_layouts() -> Result<()> {
    let (_lock, context) = testing::bootstrap()?;
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
    let (_lock, context) = testing::bootstrap()?;
    let device = DeviceMemory::from_slice(&context, &[3_i32, 4, 5])?;
    let mut host = context.pinned_host().zeroed::<i32>(3)?;

    transfer::copy_device(&mut host.mapped_device_view_mut()?, &device)?;
    assert_eq!(host, [3, 4, 5]);
    assert_eq!(host.mapped_device_view()?.len(), 3);

    Ok(())
}

#[test]
fn test_pinned_release_cached() -> Result<()> {
    let (_lock, context) = testing::bootstrap()?;
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
    let (_lock, context) = testing::bootstrap()?;

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
    let (_lock, context) = testing::bootstrap()?;
    let stream = context.create_stream()?;

    let host_in = vec![10_u32, 11, 12, 13, 14, 15];
    let mut pitched = PitchedDeviceMemory::zeroed(&context, 2, 3)?;
    let mut host_out = vec![0_u32; host_in.len()];

    stream.sync_scope(|scope| pitched.copy_from_host_async(&host_in, scope))?;
    stream.sync_scope(|scope| pitched.copy_to_host_async(&mut host_out, scope))?;

    assert_eq!(host_out, host_in);

    Ok(())
}

#[test]
fn test_pitched_device_memory_allows_empty_shapes() -> Result<()> {
    let (_lock, context) = testing::bootstrap()?;

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
    let (_lock, context) = testing::bootstrap()?;

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
    let (_lock, context) = testing::bootstrap()?;
    let stream = context.create_stream()?;

    let host_in: Vec<u32> = (100..112).collect();
    let mut volume = PitchedDeviceVolume::zeroed(&context, 3, 2, 2)?;
    let mut host_out = vec![0_u32; host_in.len()];

    stream.sync_scope(|scope| volume.copy_from_host_async(&host_in, scope))?;
    stream.sync_scope(|scope| volume.copy_to_host_async(&mut host_out, scope))?;

    assert_eq!(host_out, host_in);

    Ok(())
}

#[test]
fn test_pitched_device_volume_allows_empty_shapes() -> Result<()> {
    let (_lock, context) = testing::bootstrap()?;

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
    let (_lock, context) = testing::bootstrap()?;

    let mut values: Vec<u32> = Vec::new();
    let registered = RegisteredHostMemory::create(&context, &mut values)?;
    assert!(registered.is_empty());
    assert_eq!(registered.byte_len()?, 0);

    let readonly = RegisteredHostSlice::create_read_only(&context, &values)?;
    assert!(readonly.is_empty());
    assert_eq!(readonly.byte_len()?, 0);
    assert!(readonly.flags().contains(HostRegisterFlags::READ_ONLY));

    let registered_vec = RegisteredHostVec::<u32>::from_slice(&context, &[])?;
    assert!(registered_vec.is_empty());
    assert_eq!(registered_vec.byte_len()?, 0);
    assert!(matches!(
        RegisteredHostVec::from_slice_with_flags(&context, &[1_u32], HostRegisterFlags::READ_ONLY),
        Err(Error::InvalidValue)
    ));

    Ok(())
}

#[test]
fn test_registered_host_memory_tracks_borrow() -> Result<()> {
    let (_lock, context) = testing::bootstrap()?;
    if !context.device().properties()?.host_register_supported {
        return Ok(());
    }

    let mut values = vec![1_u32, 2, 3, 4];
    {
        let mut registered = match RegisteredHostMemory::create(&context, &mut values) {
            Ok(registered) => registered,
            Err(Error::Cuda {
                code: Status::NotSupported,
                ..
            }) => return Ok(()),
            Err(error) => return Err(error),
        };
        assert_eq!(registered.as_slice(), &[1, 2, 3, 4]);
        registered.as_slice_mut()[1] = 7;
        assert_eq!(registered.byte_len()?, 4 * size_of::<u32>());
    }
    assert_eq!(values, vec![1, 7, 3, 4]);

    let read_only = match RegisteredHostSlice::create_read_only(&context, &values) {
        Ok(read_only) => read_only,
        Err(Error::Cuda {
            code: Status::NotSupported,
            ..
        }) => return Ok(()),
        Err(error) => return Err(error),
    };
    assert_eq!(read_only.as_slice(), values.as_slice());
    assert!(read_only.flags().contains(HostRegisterFlags::READ_ONLY));

    let registered_vec = match RegisteredHostVec::from_slice(&context, &values) {
        Ok(registered_vec) => registered_vec,
        Err(Error::Cuda {
            code: Status::NotSupported,
            ..
        }) => return Ok(()),
        Err(error) => return Err(error),
    };
    assert_eq!(registered_vec.as_slice(), values.as_slice());

    Ok(())
}

#[test]
fn test_managed_memory_advise_uses_typed_wrapper() -> Result<()> {
    let (_lock, context) = testing::bootstrap()?;
    let properties = context.device().properties()?;
    if !properties.managed_memory {
        return Ok(());
    }

    let managed = ManagedMemory::<u32>::zeroed(&context, 4, MemoryAttachFlags::GLOBAL)?;
    let host = MemoryLocation::Host;

    for result in [
        managed.set_read_mostly(),
        managed.unset_read_mostly(),
        managed.set_preferred_location(host),
        managed.unset_preferred_location(),
    ] {
        match result {
            Ok(()) => {}
            Err(Error::Cuda {
                code: Status::NotSupported,
                ..
            }) => return Ok(()),
            Err(error) => return Err(error),
        }
    }

    if properties.concurrent_managed_access {
        let device = MemoryLocation::Device(context.device().id());
        managed.set_accessed_by(device)?;
        managed.unset_accessed_by(device)?;
    }

    Ok(())
}

#[test]
fn test_managed_memory_tracks_metadata_and_views() -> Result<()> {
    let (_lock, context) = testing::bootstrap()?;
    if !context.device().properties()?.managed_memory {
        return Ok(());
    }
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
    let (ptr, length, flags, allocation_context) = managed.into_raw_parts().unwrap();
    assert_eq!(ptr.cast_const(), address);
    assert_eq!(length, 4);
    assert_eq!(flags, MemoryAttachFlags::HOST);
    // SAFETY: reconstruct the unique owner using its original allocation and context.
    let _managed =
        unsafe { ManagedMemory::from_raw_parts(&allocation_context, ptr, length, flags) }?;
    Ok(())
}

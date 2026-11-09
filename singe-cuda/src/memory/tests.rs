use super::*;
use crate::{error::Status, testing, view::ZeroableDeviceRepr};

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
fn test_pinned_host_memory_from_slice_and_zeroes() -> Result<()> {
    let (_lock, _ctx) = testing::bootstrap()?;

    let host = PinnedHostMemory::from_slice(&_ctx, &[1_i32, 2, 3])?;
    assert_eq!(host.as_slice(), &[1, 2, 3]);
    assert_eq!(host.len(), 3);
    assert_eq!(host.byte_len()?, 3 * size_of::<i32>());

    let zeroes = PinnedHostMemory::<u32>::zeroed(&_ctx, 4, HostAllocationFlags::DEFAULT)?;
    assert_eq!(zeroes.as_slice(), &[0, 0, 0, 0]);

    Ok(())
}

#[test]
fn test_pinned_host_try_clone_and_raw_round_trip() -> Result<()> {
    let (_lock, _ctx) = testing::bootstrap()?;

    let host = PinnedHostMemory::from_slice_with_flags(
        &_ctx,
        &[1_u32, 2, 3],
        HostAllocationFlags::PORTABLE,
    )?;
    let cloned = host.try_clone()?;
    assert_eq!(cloned.as_slice(), host.as_slice());
    assert_eq!(cloned.flags(), host.flags());

    let (ptr, len, flags, allocation_context) = host.into_raw_parts().unwrap();
    // SAFETY: the pointer and length come from the allocation that the test just made.
    let rebuilt =
        unsafe { PinnedHostMemory::from_raw_parts(&allocation_context, ptr, len, flags) }?;
    assert_eq!(rebuilt.as_slice(), &[1, 2, 3]);
    assert_eq!(rebuilt.flags(), HostAllocationFlags::PORTABLE);

    Ok(())
}

#[test]
fn test_mapped_pinned_host_memory_reports_device_alias() -> Result<()> {
    let (_lock, context) = testing::bootstrap()?;
    if !context.device().properties()?.can_map_host_memory {
        return Ok(());
    }

    let mut host = PinnedHostMemory::from_slice_with_flags(
        &context,
        &[5_u32, 6, 7, 8],
        HostAllocationFlags::MAPPED,
    )?;
    let flags = match host.host_flags() {
        Ok(flags) => flags,
        Err(Error::Cuda {
            code: Status::NotSupported | Status::InvalidValue,
            ..
        }) => return Ok(()),
        Err(error) => return Err(error),
    };
    assert!(flags.contains(HostAllocationFlags::MAPPED));

    let mapped = match host.mapped_device_ptr() {
        Ok(mapped) => mapped,
        Err(Error::Cuda {
            code: Status::NotSupported | Status::InvalidValue,
            ..
        }) => return Ok(()),
        Err(error) => return Err(error),
    };
    assert!(!mapped.is_null());
    assert_eq!(host.mapped_device_view()?.len(), host.len());
    assert_eq!(host.mapped_device_view_mut()?.len(), host.len());

    Ok(())
}

#[test]
fn test_pinned_host_copy_round_trip() -> Result<()> {
    let (_lock, _ctx) = testing::bootstrap()?;

    let host_in = PinnedHostMemory::from_slice(&_ctx, &[7_i32, 8, 9])?;
    // SAFETY: the buffer is fully written by the operation before it is read.
    let mut device = unsafe { DeviceMemory::create_unchecked(&_ctx, host_in.len()) }?;
    device.copy_from_pinned(&host_in)?;

    let mut host_out =
        PinnedHostMemory::<i32>::zeroed(&_ctx, host_in.len(), HostAllocationFlags::DEFAULT)?;
    device.copy_to_pinned(&mut host_out)?;

    assert_eq!(host_out.as_slice(), host_in.as_slice());

    Ok(())
}

#[test]
fn test_pinned_host_scoped_async_copy_round_trip() -> Result<()> {
    let (_lock, context) = testing::bootstrap()?;
    let stream = context.create_stream()?;

    let host_in = PinnedHostMemory::from_slice(&context, &[10_i32, 11, 12])?;
    // SAFETY: the buffer is fully written by the operation before it is read.
    let mut device = unsafe { DeviceMemory::create_unchecked(&context, host_in.len()) }?;
    let mut host_out =
        PinnedHostMemory::<i32>::zeroed(&context, host_in.len(), HostAllocationFlags::DEFAULT)?;

    stream.sync_scope(|scope| device.copy_from_pinned_async(&host_in, scope))?;
    stream.sync_scope(|scope| device.copy_to_pinned_async(&mut host_out, scope))?;

    assert_eq!(host_out.as_slice(), host_in.as_slice());

    Ok(())
}

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
fn test_pinned_host_rejects_non_empty_zst() -> Result<()> {
    let (_lock, context) = testing::bootstrap()?;
    #[derive(Clone, Copy)]
    struct Zst;

    unsafe impl DeviceRepr for Zst {}

    unsafe impl ZeroableDeviceRepr for Zst {}

    assert!(matches!(
        PinnedHostMemory::<Zst>::from_slice(&context, &[Zst]),
        Err(Error::InvalidMemoryAllocationRequest)
    ));
    assert!(matches!(
        PinnedHostMemory::<Zst>::zeroed(&context, 1, HostAllocationFlags::DEFAULT),
        Err(Error::InvalidMemoryAllocationRequest)
    ));
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

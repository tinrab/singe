use singe_cufile_sys as sys;

use crate::{
    config::{set_gpu_bounce_buffer_slab_array, set_posix_pool_slab_array},
    error::Result,
    try_ffi,
    types::{DriverControlFlags, DriverStatus, DriverStatusFlags, FeatureFlags, P2PFlags},
    utility::to_i32,
};

#[derive(Debug)]
pub struct Driver {
    _private: (),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DriverProperties {
    pub major_version: u32,
    pub minor_version: u32,
    pub poll_threshold_size: usize,
    pub max_direct_io_size: usize,
    pub status_flags: DriverStatusFlags,
    pub control_flags: DriverControlFlags,
    pub feature_flags: FeatureFlags,
    pub max_device_cache_size: u32,
    pub per_buffer_cache_size: u32,
    pub max_device_pinned_memory_size: u32,
    pub max_batch_io_size: u32,
    pub max_batch_io_timeout_milliseconds: u32,
}

impl Driver {
    pub fn create() -> Result<Self> {
        unsafe {
            try_ffi!(sys::cuFileDriverOpen())?;
        }
        Ok(Self { _private: () })
    }

    pub fn use_count() -> isize {
        unsafe { sys::cuFileUseCount() as isize }
    }

    pub fn properties(&self) -> Result<DriverProperties> {
        let mut properties = sys::CUfileDrvProps_t::default();
        unsafe {
            try_ffi!(sys::cuFileDriverGetProperties(&raw mut properties))?;
        }

        Ok(DriverProperties {
            major_version: properties.nvfs.major_version,
            minor_version: properties.nvfs.minor_version,
            poll_threshold_size: properties.nvfs.poll_thresh_size as usize,
            max_direct_io_size: properties.nvfs.max_direct_io_size as usize,
            status_flags: DriverStatusFlags::from_bits_retain(properties.nvfs.dstatusflags),
            control_flags: DriverControlFlags::from_bits_retain(properties.nvfs.dcontrolflags),
            feature_flags: FeatureFlags::from_bits_retain(properties.fflags),
            max_device_cache_size: properties.max_device_cache_size,
            per_buffer_cache_size: properties.per_buffer_cache_size,
            max_device_pinned_memory_size: properties.max_device_pinned_mem_size,
            max_batch_io_size: properties.max_batch_io_size,
            max_batch_io_timeout_milliseconds: properties.max_batch_io_timeout_msecs,
        })
    }

    pub fn set_poll_mode(&self, enabled: bool, threshold_size: usize) -> Result<()> {
        unsafe {
            try_ffi!(sys::cuFileDriverSetPollMode(
                enabled,
                threshold_size as sys::size_t,
            ))?;
        }
        Ok(())
    }

    pub fn set_max_direct_io_size(&self, size: usize) -> Result<()> {
        unsafe {
            try_ffi!(sys::cuFileDriverSetMaxDirectIOSize(size as sys::size_t))?;
        }
        Ok(())
    }

    pub fn set_max_cache_size(&self, size: usize) -> Result<()> {
        unsafe {
            try_ffi!(sys::cuFileDriverSetMaxCacheSize(size as sys::size_t))?;
        }
        Ok(())
    }

    pub fn set_max_pinned_memory_size(&self, size: usize) -> Result<()> {
        unsafe {
            try_ffi!(sys::cuFileDriverSetMaxPinnedMemSize(size as sys::size_t))?;
        }
        Ok(())
    }

    pub fn bar_size_in_kb(&self, gpu_index: i32) -> Result<usize> {
        let mut size = 0;
        unsafe {
            try_ffi!(sys::cuFileGetBARSizeInKB(gpu_index, &raw mut size))?;
        }
        Ok(size as usize)
    }

    pub fn p2p_flags(&self, status_flag: DriverStatus) -> Result<P2PFlags> {
        let mut flags = sys::CUfileP2PFlags_t::CUFILE_P2PDMA;
        unsafe {
            try_ffi!(sys::cuFileDriverGetP2PFlags(
                status_flag.into(),
                &raw mut flags,
            ))?;
        }
        Ok(P2PFlags::from_bits_retain(flags.0))
    }

    pub fn set_p2p_flags(&self, status_flag: DriverStatus, flags: P2PFlags) -> Result<()> {
        let raw_flags = sys::CUfileP2PFlags_t(flags.bits());
        unsafe {
            try_ffi!(sys::cuFileDriverSetP2PFlags(status_flag.into(), raw_flags))?;
        }
        Ok(())
    }

    pub fn set_posix_pool_slab_array(&self, slabs: &[Slab]) -> Result<()> {
        set_posix_pool_slab_array(slabs)
    }

    pub fn set_gpu_bounce_buffer_slab_array(&self, slabs: &[Slab]) -> Result<()> {
        set_gpu_bounce_buffer_slab_array(slabs)
    }
}

impl Drop for Driver {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::cuFileDriverClose();
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Slab {
    pub size: usize,
    pub count: usize,
}

impl Slab {
    pub const fn new(size: usize, count: usize) -> Self {
        Self { size, count }
    }
}

pub(crate) fn split_slabs(slabs: &[Slab]) -> Result<(Vec<sys::size_t>, Vec<sys::size_t>, i32)> {
    let len = to_i32(slabs.len(), "slabs")?;
    let sizes = slabs.iter().map(|slab| slab.size as sys::size_t).collect();
    let counts = slabs.iter().map(|slab| slab.count as sys::size_t).collect();
    Ok((sizes, counts, len))
}

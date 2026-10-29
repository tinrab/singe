use std::sync::{Arc, Mutex};

use singe_cufile_sys as sys;

use crate::{
    config::{set_gpu_bounce_buffer_slab_array, set_posix_pool_slab_array},
    error::{Error, Result},
    try_ffi,
    types::{DriverControlFlags, DriverStatus, DriverStatusFlags, FeatureFlags, P2PFlags},
    utility::{to_i32, to_usize},
};

// cuFileDriverOpen reuses one process-wide session; cuFileDriverClose closes it for every caller.
// Serialize acquisition and final close so independent Driver::create calls share its native lifetime.
static DRIVER_SESSIONS: Mutex<usize> = Mutex::new(0);

/// A shared lease on cuFile's process-wide session, retained by its registered resources.
#[derive(Debug)]
pub struct Driver {
    _private: (),
}

/// Driver version, supported I/O paths, current controls, and resource limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DriverProperties {
    /// Major component of the cuFile filesystem driver version.
    pub major_version: u32,
    /// Minor component of the cuFile filesystem driver version.
    pub minor_version: u32,
    /// Maximum I/O size for polling mode, in KB; must be 4 KiB aligned.
    pub poll_threshold_size: usize,
    /// Maximum I/O size sent to the filesystem driver, in KB; must be 4 KiB aligned.
    pub max_direct_io_size: usize,
    /// Supported filesystems and block-device types.
    pub status_flags: DriverStatusFlags,
    /// Enabled driver I/O controls.
    pub control_flags: DriverControlFlags,
    /// Supported library features.
    pub feature_flags: FeatureFlags,
    /// Maximum internal GPU buffer space per device, in KB.
    pub max_device_cache_size: u32,
    /// Size of each GPU bounce buffer in the internal pools, in KB.
    pub per_buffer_cache_size: u32,
    /// Maximum registered buffer space pinned and mapped into the GPU BAR, in KB.
    pub max_device_pinned_memory_size: u32,
    /// Maximum number of I/O requests in a batch.
    pub max_batch_io_size: u32,
    /// Maximum batch I/O timeout, in milliseconds.
    pub max_batch_io_timeout_milliseconds: u32,
}

impl Driver {
    /// Acquires a shared lease on the process-wide cuFile driver session.
    ///
    /// The first lease opens the driver; the last lease closes it.
    ///
    /// # Errors
    ///
    /// Returns an error if the session count overflows or cuFile cannot open the driver.
    ///
    /// # Panics
    ///
    /// Panics if the driver lifecycle mutex is poisoned.
    pub fn create() -> Result<Arc<Self>> {
        let mut sessions = DRIVER_SESSIONS
            .lock()
            .expect("cufile driver lifecycle mutex poisoned");
        let next = sessions
            .checked_add(1)
            .ok_or_else(|| Error::out_of_range("driver session count"))?;
        if *sessions == 0 {
            // SAFETY: `cuFileDriverOpen` takes no arguments.
            unsafe {
                try_ffi!(sys::cuFileDriverOpen())?;
            }
        }
        *sessions = next;
        drop(sessions);
        Ok(Arc::new(Self { _private: () }))
    }

    /// Returns the native cuFile driver use count.
    ///
    /// # Panics
    ///
    /// Panics if the native use count cannot be represented as `isize`.
    #[must_use]
    pub fn use_count() -> isize {
        // SAFETY: `cuFileUseCount` takes no arguments.
        unsafe {
            isize::try_from(sys::cuFileUseCount())
                .expect("driver use count fits the host signed size")
        }
    }

    /// Returns the current driver properties.
    ///
    /// # Errors
    ///
    /// Returns an error if cuFile cannot report its properties or a reported size cannot be represented as `usize`.
    pub fn properties(&self) -> Result<DriverProperties> {
        let mut properties = sys::CUfileDrvProps_t::default();
        // SAFETY: `cuFileDriverGetProperties` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuFileDriverGetProperties(&raw mut properties))?;
        }

        Ok(DriverProperties {
            major_version: properties.nvfs.major_version,
            minor_version: properties.nvfs.minor_version,
            poll_threshold_size: to_usize(properties.nvfs.poll_thresh_size, "poll threshold size")?,
            max_direct_io_size: to_usize(
                properties.nvfs.max_direct_io_size,
                "maximum direct IO size",
            )?,
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

    /// Sets whether reads and writes poll for completion, up to `threshold_size` KB.
    ///
    /// The native setting applies only before the driver opens; it has no effect on an already open session.
    ///
    /// # Errors
    ///
    /// Returns an error if cuFile rejects the value.
    pub fn set_poll_mode(&self, enabled: bool, threshold_size: usize) -> Result<()> {
        // SAFETY: `cuFileDriverSetPollMode` receives only plain values.
        unsafe {
            try_ffi!(sys::cuFileDriverSetPollMode(
                enabled,
                threshold_size as sys::size_t,
            ))?;
        }
        Ok(())
    }

    /// Sets the maximum I/O size sent to the filesystem driver, in KB.
    ///
    /// The native setting applies only before the driver opens; it has no effect on an already open session.
    ///
    /// # Errors
    ///
    /// Returns an error if cuFile rejects the value.
    pub fn set_max_direct_io_size(&self, size: usize) -> Result<()> {
        // SAFETY: `cuFileDriverSetMaxDirectIOSize` receives only plain values.
        unsafe {
            try_ffi!(sys::cuFileDriverSetMaxDirectIOSize(size as sys::size_t))?;
        }
        Ok(())
    }

    /// Sets the maximum internal GPU buffer space per device, in KB.
    ///
    /// The native setting applies only before the driver opens; it has no effect on an already open session.
    ///
    /// # Errors
    ///
    /// Returns an error if cuFile rejects the value, including an unsupported resource limit or a CUDA failure when configuring GPU memory.
    pub fn set_max_cache_size(&self, size: usize) -> Result<()> {
        // SAFETY: `cuFileDriverSetMaxCacheSize` receives only plain values.
        unsafe {
            try_ffi!(sys::cuFileDriverSetMaxCacheSize(size as sys::size_t))?;
        }
        Ok(())
    }

    /// Sets the maximum buffer space pinned for registration, in KB.
    ///
    /// The native setting applies only before the driver opens; it has no effect on an already open session.
    ///
    /// # Errors
    ///
    /// Returns an error if cuFile rejects the value, including an unsupported resource limit or a CUDA failure when configuring GPU memory.
    pub fn set_max_pinned_memory_size(&self, size: usize) -> Result<()> {
        // SAFETY: `cuFileDriverSetMaxPinnedMemSize` receives only plain values.
        unsafe {
            try_ffi!(sys::cuFileDriverSetMaxPinnedMemSize(size as sys::size_t))?;
        }
        Ok(())
    }

    /// Returns the BAR size of `gpu_index`, in KiB.
    ///
    /// # Errors
    ///
    /// Returns an error if cuFile is not initialized, the GPU index is invalid, or the reported size cannot be represented as `usize`.
    pub fn bar_size_in_kb(&self, gpu_index: i32) -> Result<usize> {
        let mut size = 0;
        // SAFETY: `cuFileGetBARSizeInKB` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuFileGetBARSizeInKB(gpu_index, &raw mut size))?;
        }
        to_usize(size, "pinned memory size")
    }

    /// Returns peer-to-peer I/O capabilities for a filesystem or block-device type.
    ///
    /// # Errors
    ///
    /// Returns an error if cuFile cannot query the flags for `status_flag`.
    pub fn p2p_flags(&self, status_flag: DriverStatus) -> Result<P2PFlags> {
        let mut flags = sys::CUfileP2PFlags_t::CUFILE_P2PDMA;
        // SAFETY: `cuFileDriverGetP2PFlags` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuFileDriverGetP2PFlags(
                status_flag.into(),
                &raw mut flags,
            ))?;
        }
        Ok(P2PFlags::from_bits_retain(flags.0))
    }

    /// Sets the permitted peer-to-peer I/O flags for a filesystem or block-device type.
    ///
    /// The native API permits PCI peer-to-peer DMA on x86-64 and chip-to-chip transfers on `AArch64`.
    /// The driver manages NVFS and DMA-buffer flags internally.
    ///
    /// # Errors
    ///
    /// Returns an error if the driver is not initialized, the target type is invalid, or a flag is unsupported on the current platform.
    pub fn set_p2p_flags(&self, status_flag: DriverStatus, flags: P2PFlags) -> Result<()> {
        let raw_flags = sys::CUfileP2PFlags_t(flags.bits());
        // SAFETY: `cuFileDriverSetP2PFlags` receives only plain values.
        unsafe {
            try_ffi!(sys::cuFileDriverSetP2PFlags(status_flag.into(), raw_flags))?;
        }
        Ok(())
    }

    /// Sets the slab sizes and counts for the POSIX memory pool.
    ///
    /// Slab sizes are in KB.
    /// The override is applied at driver load time.
    ///
    /// # Errors
    ///
    /// Returns an error if the slab count exceeds the `i32` range, the driver is already open, or cuFile rejects the size/count pairs.
    pub fn set_posix_pool_slab_array(&self, slabs: &[Slab]) -> Result<()> {
        set_posix_pool_slab_array(slabs)
    }

    /// Sets the slab sizes and counts for the GPU bounce-buffer pool.
    ///
    /// Slab sizes are in KB.
    /// The override is applied at driver load time.
    ///
    /// # Errors
    ///
    /// Returns an error if the slab count exceeds the `i32` range, the driver is already open, or cuFile rejects the size/count pairs.
    pub fn set_gpu_bounce_buffer_slab_array(&self, slabs: &[Slab]) -> Result<()> {
        set_gpu_bounce_buffer_slab_array(slabs)
    }
}

impl Drop for Driver {
    fn drop(&mut self) {
        let mut sessions = DRIVER_SESSIONS
            .lock()
            .expect("cufile driver lifecycle mutex poisoned");
        *sessions -= 1;
        if *sessions == 0 {
            // SAFETY: `cuFileDriverClose` takes no arguments.
            unsafe {
                if let Err(error) = try_ffi!(sys::cuFileDriverClose()) {
                    #[cfg(debug_assertions)]
                    eprintln!("failed to close cufile driver: {error}");
                }
            }
        }
    }
}

/// A buffer-pool size class and its number of preallocated buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Slab {
    /// Size of each buffer in this slab class, in KB.
    pub size: usize,
    /// Number of buffers in this slab class.
    pub count: usize,
}

impl Slab {
    /// Creates a slab size class with `size` in KB and `count` buffers.
    #[must_use]
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

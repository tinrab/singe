use std::sync::{Arc, Mutex, PoisonError};

use singe_core::report_cleanup_failure;
use singe_cufile_sys as sys;

use crate::{
    error::{Error, Result},
    try_ffi,
    types::{DriverControlFlags, DriverStatus, DriverStatusFlags, FeatureFlags, P2PFlags},
    utility::{to_i32, to_u64, to_usize},
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
    pub poll_threshold_size_kb: usize,
    /// Maximum I/O size sent to the filesystem driver, in KB; must be 4 KiB aligned.
    pub max_direct_io_size_kb: usize,
    /// Supported filesystems and block-device types.
    pub status_flags: DriverStatusFlags,
    /// Enabled driver I/O controls.
    pub control_flags: DriverControlFlags,
    /// Supported library features.
    pub feature_flags: FeatureFlags,
    /// Maximum internal GPU buffer space per device, in KB.
    pub max_device_cache_size_kb: u32,
    /// Size of each GPU bounce buffer in the internal pools, in KB.
    pub per_buffer_cache_size_kb: u32,
    /// Maximum registered buffer space pinned and mapped into the GPU BAR, in KB.
    pub max_device_pinned_memory_size_kb: u32,
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
    pub fn create() -> Result<Arc<Self>> {
        let mut sessions = DRIVER_SESSIONS.lock().map_err(|_| Error::LockPoisoned {
            name: "cufile driver sessions".into(),
        })?;
        let next = sessions
            .checked_add(1)
            .ok_or_else(|| Error::out_of_range("driver session count"))?;
        if *sessions == 0 {
            // SAFETY: the call takes no arguments, and the `DRIVER_SESSIONS` lock held here serializes it with the final close.
            unsafe {
                try_ffi!(sys::cuFileDriverOpen())?;
            }
        }

        *sessions = next;
        drop(sessions);
        Ok(Arc::new(Self { _private: () }))
    }

    /// Returns the native cuFile driver use count.
    #[must_use]
    pub fn use_count() -> i64 {
        // SAFETY: the call takes no arguments and only reads the driver's use count.
        // cuFile supports only 64-bit Linux, where its `long` result is `i64`.
        unsafe { sys::cuFileUseCount() }
    }

    /// Returns the current driver properties.
    ///
    /// # Errors
    ///
    /// Returns an error if cuFile cannot report its properties or a reported size cannot be represented as `usize`.
    pub fn properties(&self) -> Result<DriverProperties> {
        let mut properties = sys::CUfileDrvProps_t::default();
        // SAFETY: `properties` is a local `CUfileDrvProps_t` that cuFile fills.
        unsafe {
            try_ffi!(sys::cuFileDriverGetProperties(&raw mut properties))?;
        }

        Ok(DriverProperties {
            major_version: properties.nvfs.major_version,
            minor_version: properties.nvfs.minor_version,
            poll_threshold_size_kb: to_usize(
                properties.nvfs.poll_thresh_size,
                "poll threshold size",
            )?,
            max_direct_io_size_kb: to_usize(
                properties.nvfs.max_direct_io_size,
                "maximum direct IO size",
            )?,
            status_flags: DriverStatusFlags::from_bits_retain(properties.nvfs.dstatusflags),
            control_flags: DriverControlFlags::from_bits_retain(properties.nvfs.dcontrolflags),
            feature_flags: FeatureFlags::from_bits_retain(properties.fflags),
            max_device_cache_size_kb: properties.max_device_cache_size,
            per_buffer_cache_size_kb: properties.per_buffer_cache_size,
            max_device_pinned_memory_size_kb: properties.max_device_pinned_mem_size,
            max_batch_io_size: properties.max_batch_io_size,
            max_batch_io_timeout_milliseconds: properties.max_batch_io_timeout_msecs,
        })
    }

    /// Makes reads and writes of at most `threshold_size_kb` KB poll for completion, or turns polling off for `None`.
    ///
    /// The native setting applies only before the driver opens; it has no effect on an already open session.
    ///
    /// # Errors
    ///
    /// Returns an error if the threshold exceeds the native range or cuFile rejects the value.
    pub fn set_poll_mode(&self, threshold_size_kb: Option<usize>) -> Result<()> {
        let threshold = to_u64(threshold_size_kb.unwrap_or(0), "poll threshold size")?;
        // SAFETY: the flag and threshold are passed by value, and the call touches no Rust-owned memory.
        unsafe {
            try_ffi!(sys::cuFileDriverSetPollMode(
                threshold_size_kb.is_some(),
                threshold,
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
    /// Returns an error if the size exceeds the native range or cuFile rejects the value.
    pub fn set_max_direct_io_size_kb(&self, size_kb: usize) -> Result<()> {
        let size = to_u64(size_kb, "maximum direct io size")?;
        // SAFETY: the size is passed by value, and the call touches no Rust-owned memory.
        unsafe {
            try_ffi!(sys::cuFileDriverSetMaxDirectIOSize(size))?;
        }
        Ok(())
    }

    /// Sets the maximum internal GPU buffer space per device, in KB.
    ///
    /// The native setting applies only before the driver opens; it has no effect on an already open session.
    ///
    /// # Errors
    ///
    /// Returns an error if the size exceeds the native range or cuFile rejects the value, including an unsupported resource limit or a CUDA failure when configuring GPU memory.
    pub fn set_max_cache_size_kb(&self, size_kb: usize) -> Result<()> {
        let size = to_u64(size_kb, "maximum cache size")?;
        // SAFETY: the size is passed by value, and the call touches no Rust-owned memory.
        unsafe {
            try_ffi!(sys::cuFileDriverSetMaxCacheSize(size))?;
        }
        Ok(())
    }

    /// Sets the maximum buffer space pinned for registration, in KB.
    ///
    /// The native setting applies only before the driver opens; it has no effect on an already open session.
    ///
    /// # Errors
    ///
    /// Returns an error if the size exceeds the native range or cuFile rejects the value, including an unsupported resource limit or a CUDA failure when configuring GPU memory.
    pub fn set_max_pinned_memory_size_kb(&self, size_kb: usize) -> Result<()> {
        let size = to_u64(size_kb, "maximum pinned memory size")?;
        // SAFETY: the size is passed by value, and the call touches no Rust-owned memory.
        unsafe {
            try_ffi!(sys::cuFileDriverSetMaxPinnedMemSize(size))?;
        }
        Ok(())
    }

    /// Returns the BAR size of `gpu_index`, in KiB.
    ///
    /// # Errors
    ///
    /// Returns an error if cuFile is not initialized, the GPU index is invalid, or the reported size cannot be represented as `usize`.
    pub fn bar_size_kb(&self, gpu_index: i32) -> Result<usize> {
        let mut size = 0;
        // SAFETY: `size` is a local `size_t` output, and cuFile validates `gpu_index`.
        unsafe {
            try_ffi!(sys::cuFileGetBARSizeInKB(gpu_index, &raw mut size))?;
        }
        to_usize(size, "bar size")
    }

    /// Returns peer-to-peer I/O capabilities for a filesystem or block-device type.
    ///
    /// # Errors
    ///
    /// Returns an error if cuFile cannot query the flags for `status_flag`.
    pub fn p2p_flags(&self, status_flag: DriverStatus) -> Result<P2PFlags> {
        let mut flags = sys::CUfileP2PFlags_t::CUFILE_P2PDMA;
        // SAFETY: `flags` is a local `CUfileP2PFlags_t` output.
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
        // SAFETY: the type and flags are passed by value, and the call touches no Rust-owned memory.
        unsafe {
            try_ffi!(sys::cuFileDriverSetP2PFlags(status_flag.into(), raw_flags))?;
        }
        Ok(())
    }
}

impl Drop for Driver {
    fn drop(&mut self) {
        // The last session must close the driver whatever a panic left here.
        let mut sessions = DRIVER_SESSIONS
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        *sessions -= 1;
        if *sessions == 0 {
            // SAFETY: this is the last lease, so no registered handle, buffer, or batch retains the session, and the `DRIVER_SESSIONS` lock serializes the close with a reopen.
            unsafe {
                if let Err(error) = try_ffi!(sys::cuFileDriverClose()) {
                    report_cleanup_failure("close", "cufile driver", &error);
                }
            }
        }
    }
}

/// A buffer-pool size class and its number of preallocated buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Slab {
    /// Size of each buffer in this slab class, in KB.
    pub size_kb: usize,
    /// Number of buffers in this slab class.
    pub count: usize,
}

impl Slab {
    /// Creates a slab size class of `count` buffers of `size_kb` KB each.
    #[must_use]
    pub const fn new(size_kb: usize, count: usize) -> Self {
        Self { size_kb, count }
    }
}

pub(crate) fn split_slabs(slabs: &[Slab]) -> Result<(Vec<sys::size_t>, Vec<sys::size_t>, i32)> {
    let len = to_i32(slabs.len(), "slabs")?;
    let sizes = slabs
        .iter()
        .map(|slab| to_u64(slab.size_kb, "slab size"))
        .collect::<Result<_>>()?;
    let counts = slabs
        .iter()
        .map(|slab| to_u64(slab.count, "slab count"))
        .collect::<Result<_>>()?;
    Ok((sizes, counts, len))
}

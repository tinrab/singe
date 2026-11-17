//! Device enumeration, properties, attributes, and peer capabilities.

#![expect(
    deprecated,
    reason = "wraps deprecated upstream API values that remain part of the supported surface"
)]

mod attribute;
mod clique;
mod properties;

use std::{
    ffi::{CString, c_char},
    mem,
};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_display, string_from_c_chars};
use singe_cuda_sys::runtime;

use crate::{
    context::ContextFlags,
    error::{Error, Result},
    try_ffi,
    types::FunctionCache,
    utility::to_usize,
};

pub use self::{
    attribute::DeviceAttribute,
    clique::{CliqueInfo, CliqueType},
    properties::{DeviceProperties, Uuid},
};

singe_core::native_enum! {
    /// CUDA Limits.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum Limit: runtime::cudaLimit {
        /// Per-block memory size.
        PerBlockMemorySize = cudaLimitPerBlockMemorySize,
        /// GPU thread stack size.
        StackSize = cudaLimitStackSize,
        /// GPU printf FIFO size.
        PrintfFifoSize = cudaLimitPrintfFifoSize,
        /// GPU malloc heap size.
        MallocHeapSize = cudaLimitMallocHeapSize,
        /// GPU device runtime synchronize depth.
        DevRuntimeSyncDepth = cudaLimitDevRuntimeSyncDepth,
        /// GPU device runtime pending launch count.
        DevRuntimePendingLaunchCount = cudaLimitDevRuntimePendingLaunchCount,
        /// A value between 0 and 128 that indicates the maximum fetch granularity of L2, in bytes.
        /// The value is a hint.
        MaxL2FetchGranularity = cudaLimitMaxL2FetchGranularity,
        /// A size in bytes for L2 persisting lines cache size.
        PersistingL2CacheSize = cudaLimitPersistingL2CacheSize,
    }
}

impl_enum_display!(Limit, {
    Self::PerBlockMemorySize => "cudaLimitPerBlockMemorySize",
    Self::StackSize => "cudaLimitStackSize",
    Self::PrintfFifoSize => "cudaLimitPrintfFifoSize",
    Self::MallocHeapSize => "cudaLimitMallocHeapSize",
    Self::DevRuntimeSyncDepth => "cudaLimitDevRuntimeSyncDepth",
    Self::DevRuntimePendingLaunchCount => "cudaLimitDevRuntimePendingLaunchCount",
    Self::MaxL2FetchGranularity => "cudaLimitMaxL2FetchGranularity",
    Self::PersistingL2CacheSize => "cudaLimitPersistingL2CacheSize",
});

singe_core::native_enum! {
    /// CUDA device compute modes.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum ComputeMode: runtime::cudaComputeMode {
        /// Default compute mode; multiple threads can use [`Device::set_current`] with this device.
        Default = cudaComputeModeDefault,
        /// Compute-exclusive-thread mode; only one thread in one process can use [`Device::set_current`] with this device.
        Exclusive = cudaComputeModeExclusive,
        /// Compute-prohibited mode; no threads can use [`Device::set_current`] with this device.
        Prohibited = cudaComputeModeProhibited,
        /// Compute-exclusive-process mode; many threads in one process can use [`Device::set_current`] with this device.
        ExclusiveProcess = cudaComputeModeExclusiveProcess,
    }
}

singe_core::native_enum! {
    /// Attributes queryable between two devices using [`Device::p2p_attribute`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum PeerToPeerAttribute: runtime::cudaDeviceP2PAttr {
        /// Relative performance rank of the link between two devices.
        PerformanceRank = CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK,
        /// Whether peer access is supported between the devices.
        AccessSupported = CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED,
        /// Whether all CUDA-valid native atomic operations are supported over the peer link.
        NativeAtomicSupported = CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED,
        /// Deprecated selector for CUDA array peer-access support.
        #[deprecated]
        CudaArrayAccessSupported = CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED,
    }
}

impl_enum_display!(PeerToPeerAttribute, {
    Self::PerformanceRank => "cudaDevP2PAttrPerformanceRank",
    Self::AccessSupported => "cudaDevP2PAttrAccessSupported",
    Self::NativeAtomicSupported => "cudaDevP2PAttrNativeAtomicSupported",
    Self::CudaArrayAccessSupported => "cudaDevP2PAttrCudaArrayAccessSupported",
});

/// Inclusive stream priority limits supported by the current device.
///
/// Lower numeric values represent higher priorities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamPriorityRange {
    /// Least supported priority, with the highest numeric value.
    pub least: i32,
    /// Greatest supported priority, with the lowest numeric value.
    pub greatest: i32,
}

/// A CUDA device ordinal.
///
/// The value does not own a CUDA context or validate device availability until an operation queries CUDA.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Device(DeviceId);

/// Signed CUDA device ordinal used by runtime and driver APIs.
pub type DeviceId = i32;

impl Device {
    /// Wraps a device ordinal without querying CUDA, validating it, or making the device current.
    #[must_use]
    pub const fn new(id: DeviceId) -> Self {
        Self(id)
    }

    /// Returns the number of devices with compute capability greater or equal to 2.0 that are available for execution.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query the device count, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics such as [`crate::error::Status::NotInitialized`], [`crate::error::Status::CallRequiresNewerDriver`], or [`crate::error::Status::NoDevice`].
    pub fn count() -> Result<i32> {
        let mut count: i32 = 0;
        // SAFETY: `count` is a local that `cudaGetDeviceCount` writes through an exclusive pointer.
        unsafe {
            try_ffi!(runtime::cudaGetDeviceCount(&raw mut count))?;
        }
        Ok(count)
    }

    /// Returns the current device for the calling host thread.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query the current device, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn current() -> Result<Self> {
        let mut device_id: i32 = 0;
        // SAFETY: `device_id` is a local that `cudaGetDevice` writes through an exclusive pointer.
        unsafe {
            try_ffi!(runtime::cudaGetDevice(&raw mut device_id))?;
        }
        Ok(Self(device_id))
    }

    /// Blocks until the device has completed all preceding requested tasks.
    /// [`Device::synchronize`] returns an error if one of the preceding tasks has failed.
    /// If [`ContextFlags::SCHEDULE_BLOCKING_SYNC`] was set for this device, the host thread blocks until the device has finished its work.
    ///
    /// * Use of [`Device::synchronize`] in device code was deprecated in CUDA 11.6 and removed for `compute_90`+ compilation.
    ///   For compute capability &lt; 9.0, compile-time opt-in with `-D CUDA_FORCE_CDP1_IF_SUPPORTED` is required to continue using [`Device::synchronize`] in device code for now.
    ///   This is different from host-side [`Device::synchronize`], which is still supported.
    ///
    /// # Errors
    ///
    /// Returns an error if synchronization fails, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn synchronize() -> Result<()> {
        // SAFETY: `cudaDeviceSynchronize` takes no arguments and only waits for work on the current device.
        unsafe {
            try_ffi!(runtime::cudaDeviceSynchronize())?;
        }
        Ok(())
    }

    /// Explicitly destroys and cleans up all resources associated with the current device in the current process.
    /// Accessing these resources or passing them to subsequent API calls after reset results in undefined behavior.
    /// These resources include streams, events, arrays, mipmapped arrays, pitched allocations, texture and surface objects, external memory and semaphore objects, and graphics resources owned by the current device state.
    /// These resources also include memory allocations by [`DeviceMemory::alloc`](crate::memory::DeviceMemory::alloc), [`PinnedHost`](crate::memory::PinnedHost), [`DeviceMemory::alloc_managed`](crate::memory::DeviceMemory::alloc_managed), and [`PitchedDeviceMemory`](crate::memory::PitchedDeviceMemory).
    /// Any subsequent call to this device reinitializes it.
    ///
    /// This call resets the device immediately.
    /// Ensure that no other host threads in the process are accessing the device when this is called.
    ///
    /// * [`Device::reset`] does not destroy memory allocated by [`DeviceMemory::alloc_async`](crate::memory::DeviceMemory::alloc_async) or [`sys::cudaMallocFromPoolAsync`](singe_cuda_sys::runtime::cudaMallocFromPoolAsync).
    ///   These memory allocations must be destroyed explicitly.
    /// * If a non-primary CUDA context is current to the thread, [`Device::reset`] will destroy only the internal CUDA runtime state for that context.
    ///
    /// # Safety
    ///
    /// All affected resources must have been released and all work completed.
    /// Exclude other host threads and any later use of pre-reset owners or handles.
    ///
    /// # Errors
    ///
    /// Returns an error if device reset fails, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub unsafe fn reset() -> Result<()> {
        // SAFETY: the caller guarantees every resource of the device is released, its work is complete, and no other thread uses it.
        unsafe {
            try_ffi!(runtime::cudaDeviceReset())?;
        }
        Ok(())
    }

    /// Returns the current size of limit.
    /// The following [`Limit`] values are supported.
    ///
    /// * [`Limit::StackSize`] is the stack size in bytes of each GPU thread.
    /// * [`Limit::PrintfFifoSize`] is the size in bytes of the shared FIFO used by the `printf()` device system call.
    /// * [`Limit::MallocHeapSize`] is the size in bytes of the heap used by the `malloc()` and `free()` device system calls.
    /// * [`Limit::DevRuntimeSyncDepth`] is the maximum grid depth at which a thread can issue the device runtime call [`Device::synchronize`] to wait on child grid launches to complete.
    ///   This feature is removed for devices of compute capability &gt;= 9.0, so such devices return [`crate::error::Status::UnsupportedLimit`].
    /// * [`Limit::DevRuntimePendingLaunchCount`] is the maximum number of outstanding device runtime launches.
    /// * [`Limit::MaxL2FetchGranularity`] is the L2 cache fetch granularity.
    /// * [`Limit::PersistingL2CacheSize`] is the persisting L2 cache size in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if `limit` is unsupported, CUDA cannot query the limit, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn limit(limit: Limit) -> Result<usize> {
        let mut value = 0;
        // SAFETY: `value` is a local that `cudaDeviceGetLimit` writes through an exclusive pointer.
        unsafe {
            try_ffi!(runtime::cudaDeviceGetLimit(&raw mut value, limit.into(),))?;
        }
        to_usize(value, "device limit")
    }

    /// Setting limit to value is a request by the application to update the current limit maintained by the device.
    /// The driver may modify the requested value to meet hardware requirements, such as clamping to minimum or maximum values or rounding up to the nearest element size.
    /// Use [`Device::limit`] to query the effective value.
    ///
    /// Setting each [`Limit`] has its own specific restrictions, so each is discussed here.
    ///
    /// * [`Limit::StackSize`] controls the stack size in bytes of each GPU thread.
    ///
    /// * [`Limit::PrintfFifoSize`] controls the size in bytes of the shared FIFO used by the `printf()` device system call.
    ///   Setting [`Limit::PrintfFifoSize`] must not be performed after launching any kernel that uses the `printf()` device system call; otherwise [`crate::error::Status::InvalidValue`] is returned.
    ///
    /// * [`Limit::MallocHeapSize`] controls the size in bytes of the heap used by the `malloc()` and `free()` device system calls.
    ///   Setting [`Limit::MallocHeapSize`] must not be performed after launching any kernel that uses the `malloc()` or `free()` device system calls; otherwise [`crate::error::Status::InvalidValue`] is returned.
    ///
    /// * [`Limit::DevRuntimeSyncDepth`] controls the maximum nesting depth of a grid at which a thread can safely call [`Device::synchronize`].
    ///   Setting this limit must be performed before any launch of a kernel that uses the device runtime and calls [`Device::synchronize`] above the default sync depth, two levels of grids.
    ///   Calls to [`Device::synchronize`] fail if this limit is violated.
    ///   This limit can be set smaller than the default or up to the maximum launch depth of 24.
    ///   Additional sync-depth levels require the runtime to reserve large amounts of device memory that can no longer be used for application allocations.
    ///   If these device-memory reservations fail, [`Device::set_limit`] returns an error, and the limit can be reset to a lower value.
    ///   This limit is only applicable to devices of compute capability &lt; 9.0.
    ///   Setting this limit on devices with other compute capabilities returns [`crate::error::Status::UnsupportedLimit`].
    ///
    /// * [`Limit::DevRuntimePendingLaunchCount`] controls the maximum number of outstanding device runtime launches that can be made from the current device.
    ///   A grid is outstanding from launch until it is known to have completed.
    ///   Device runtime launches that violate this limit fail.
    ///   If a module using the device runtime needs more pending launches than the default 2048 launches, this limit can be increased.
    ///   Sustaining additional pending launches requires the runtime to reserve larger amounts of device memory up front, which can no longer be used for allocations.
    ///   If these reservations fail, [`Device::set_limit`] returns an error, and the limit can be reset to a lower value.
    ///   This limit is only applicable to devices of compute capability 3.5 and higher.
    ///   Setting this limit on devices with compute capability less than 3.5 returns [`crate::error::Status::UnsupportedLimit`].
    ///
    /// * [`Limit::MaxL2FetchGranularity`] controls the L2 cache fetch granularity.
    ///   Values can range from 0B to 128B.
    ///   Performance hint that can be ignored or clamped depending on the platform.
    ///
    /// * [`Limit::PersistingL2CacheSize`] controls size in bytes available for persisting L2 cache.
    ///   Performance hint that can be ignored or clamped depending on the platform.
    ///
    /// # Errors
    ///
    /// Returns an error if `limit` is unsupported, `value` is invalid for that limit, CUDA cannot set the limit, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn set_limit(limit: Limit, value: usize) -> Result<()> {
        let value = value
            .try_into()
            .map_err(|_| Error::out_of_range("device limit value"))?;
        // SAFETY: `cudaDeviceSetLimit` receives only values.
        unsafe {
            try_ffi!(runtime::cudaDeviceSetLimit(limit.into(), value))?;
        }
        Ok(())
    }

    /// Records flags as the flags for the current device.
    /// If the current device has been set and that device has already been initialized, the previous flags are overwritten.
    /// If the current device has not been initialized, it is initialized with the provided flags.
    /// If no device has been made current to the calling thread, a default device is selected and initialized with the provided flags.
    ///
    /// The three least significant bits of `flags` control how the CPU thread interacts with the OS scheduler while waiting for device results.
    ///
    /// * [`ContextFlags::SCHEDULE_AUTO`]: The default value if `flags` is zero.
    ///   Uses a heuristic based on the number of active CUDA contexts in the process (`C`) and the number of logical processors in the system (`P`).
    ///   If `C > P`, CUDA yields to other OS threads when waiting for the device; otherwise, CUDA actively spins while waiting for results.
    ///   Additionally, on Tegra devices, [`ContextFlags::SCHEDULE_AUTO`] uses a heuristic based on the power profile of the platform and may choose [`ContextFlags::SCHEDULE_BLOCKING_SYNC`] for low-powered devices.
    /// * [`ContextFlags::SCHEDULE_SPIN`]: Instruct CUDA to actively spin when waiting for results from the device.
    ///   This can decrease latency when waiting for the device, but may lower the performance of CPU threads if they are performing work in parallel with the CUDA thread.
    /// * [`ContextFlags::SCHEDULE_YIELD`]: Instruct CUDA to yield its thread when waiting for results from the device.
    ///   This can increase latency when waiting for the device, but can increase the performance of CPU threads performing work in parallel with the device.
    /// * [`ContextFlags::SCHEDULE_BLOCKING_SYNC`]: Instruct CUDA to block the CPU thread on a synchronization primitive when waiting for the device to finish work.
    ///
    /// This matches the deprecated CUDA runtime blocking-sync behavior now represented by [`ContextFlags::SCHEDULE_BLOCKING_SYNC`].
    /// * [`ContextFlags::MAP_HOST`]: This flag enables allocating pinned host memory that is accessible to the device.
    ///   It is implicit for the runtime but may be absent if a context is created using the driver API.
    ///   If this flag is not set, [`sys::cudaHostGetDevicePointer`](singe_cuda_sys::runtime::cudaHostGetDevicePointer) always returns a failure code.
    /// * [`ContextFlags::LOCAL_MEMORY_RESIZE_TO_MAX`]: Instruct CUDA to not reduce local memory after resizing local memory for a kernel.
    ///   This can prevent thrashing by local memory allocations when launching many kernels with high local memory usage at the cost of potentially increased memory usage.
    ///
    /// Deprecated: this behavior is now the default and cannot be disabled.
    /// * [`ContextFlags::SYNC_MEMORY_OPERATIONS`]: Ensures that synchronous memory operations initiated on this context always synchronize.
    ///   See further documentation in the section titled "API Synchronization behavior" to learn more about cases when synchronous memory operations can exhibit asynchronous behavior.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot set the device flags, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn set_flags(flags: ContextFlags) -> Result<()> {
        // SAFETY: `cudaSetDeviceFlags` receives only a flag value, which CUDA validates.
        unsafe { try_ffi!(runtime::cudaSetDeviceFlags(flags.bits())) }
    }

    /// Returns the flags for the current device.
    /// If there is a current device for the calling thread, the flags for the device are returned.
    /// If there is no current device, the flags for the first device are returned, which may be the default flags.
    /// Compare to the behavior of [`Device::set_flags`].
    ///
    /// Typically, the returned flags match the behavior seen if the calling thread uses a device after this call, assuming this thread or another thread does not change the flags or current device in between.
    /// If the device is not initialized, another thread can change the flags for the current device before it is initialized.
    /// Additionally, when using exclusive mode, if this thread has not requested a specific device, it may use a device other than the first device, contrary to the assumption made by this query.
    ///
    /// If a context has been created via the driver API and is current to the calling thread, the flags for that context are always returned.
    ///
    /// Returned flags may specifically include [`ContextFlags::MAP_HOST`] even though it is not accepted by [`Device::set_flags`] because it is implicit in runtime API flags.
    /// The reason for this is that the current context may have been created via the driver API in which case the flag is not implicit and may be unset.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query the device flags, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn flags() -> Result<ContextFlags> {
        let mut flags_raw: u32 = 0;
        // SAFETY: `flags_raw` is a local that `cudaGetDeviceFlags` writes through an exclusive pointer.
        unsafe {
            try_ffi!(runtime::cudaGetDeviceFlags(&raw mut flags_raw))?;
        }
        Ok(ContextFlags::from_bits_retain(flags_raw))
    }

    /// Returns the device which has properties that best match the given prop.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot choose a matching device, the selected device cannot be represented by this wrapper, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn choose(prop: &DeviceProperties) -> Result<Self> {
        // CUDA matches against a template, so only the compute capability and managed-memory support of `prop` take part.
        // SAFETY: this plain C struct is valid when all bytes are zero.
        let mut ffi_prop: runtime::cudaDeviceProp = unsafe { mem::zeroed() };
        ffi_prop.major = prop.major;
        ffi_prop.minor = prop.minor;
        ffi_prop.managedMemory = i32::from(prop.managed_memory);

        let mut device: i32 = -1;
        // SAFETY: `device` is a writable local, and `ffi_prop` is a local template live for the call.
        unsafe {
            try_ffi!(runtime::cudaChooseDevice(
                (&raw mut device).cast(),
                &raw const ffi_prop
            ))?;
        }
        if device == -1 {
            Err(Error::DeviceNotFound)
        } else {
            Ok(Self(device))
        }
    }

    /// Returns a device ordinal given a PCI bus ID string.
    ///
    /// # Errors
    ///
    /// Returns an error if `pci_bus_id` contains an interior NUL byte, CUDA cannot resolve the bus ID, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn by_pci_bus_id(pci_bus_id: &str) -> Result<Self> {
        let c_pci_bus_id = CString::new(pci_bus_id)?;
        let mut device: i32 = -1;
        // SAFETY: `device` is a writable local, and `c_pci_bus_id` is a NUL-terminated string live for the call.
        unsafe {
            try_ffi!(runtime::cudaDeviceGetByPCIBusId(
                (&raw mut device).cast(),
                c_pci_bus_id.as_ptr(),
            ))?;
        }
        if device == -1 {
            Err(Error::DeviceNotFound)
        } else {
            Ok(Self(device))
        }
    }

    /// Sets device as the current device for the calling host thread.
    /// Valid device IDs are 0 to [`Device::count`] minus one.
    ///
    /// Device memory subsequently allocated from this host thread is physically resident on `device`.
    /// Host memory allocated or registered from this host thread has its lifetime associated with `device`.
    /// Streams and events created from this host thread are associated with `device`.
    /// Kernels launched from this host thread execute on `device`.
    ///
    /// This may be called from any host thread, for any device, at any time.
    /// This performs no synchronization with the previous or new device, and usually only takes significant time when it initializes the runtime's context state.
    /// This binds the primary context of the specified device to the calling thread; subsequent memory allocations, stream and event creations, and kernel launches are associated with that primary context.
    /// This also immediately initializes the runtime state on the primary context, and the context is current on the device immediately.
    /// It returns an error if the device is in [`ComputeMode::ExclusiveProcess`] and is occupied by another process, or if it is in [`ComputeMode::Prohibited`].
    ///
    /// It is not required to call [`sys::cudaInitDevice`](singe_cuda_sys::runtime::cudaInitDevice) before using this method.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot set the current device, the device is unavailable due to compute mode restrictions, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn set_current(self) -> Result<()> {
        // SAFETY: `cudaSetDevice` receives only the ordinal, which CUDA validates.
        unsafe {
            try_ffi!(runtime::cudaSetDevice(self.0))?;
        }
        Ok(())
    }

    /// Returns this device's properties.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query the device properties, the returned properties cannot be converted into the safe wrapper type, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn properties(self) -> Result<DeviceProperties> {
        let mut prop = runtime::cudaDeviceProp::default();
        // SAFETY: `cudaGetDeviceProperties` writes only into the local `prop`.
        unsafe {
            try_ffi!(runtime::cudaGetDeviceProperties(&raw mut prop, self.0))?;
        }
        prop.try_into()
    }

    /// Returns the PCI bus ID string identifying the device.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query the PCI bus ID, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn pci_bus_id(self) -> Result<String> {
        const LEN: usize = 16; // Sufficient for typical PCI IDs like 0000:01:00.0
        let mut pci_bus_id_buf: [c_char; LEN] = [0; LEN];
        // SAFETY: `pci_bus_id_buf` is a local buffer of `LEN` = 16 bytes, the length passed to CUDA.
        unsafe {
            try_ffi!(runtime::cudaDeviceGetPCIBusId(
                pci_bus_id_buf.as_mut_ptr().cast(),
                16,
                self.0,
            ))?;
            Ok(string_from_c_chars(&pci_bus_id_buf))
        }
    }

    /// Returns true if this device can directly access memory from `other`.
    /// If direct access from this device to `other` is possible, access may be enabled by calling [`Context::enable_peer_access`](crate::context::Context::enable_peer_access).
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query peer accessibility, either device is invalid, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn can_access_peer(self, other: Self) -> Result<bool> {
        let mut can_access_peer: i32 = 0;
        // SAFETY: `can_access_peer` is a local that `cudaDeviceCanAccessPeer` writes through an exclusive pointer.
        unsafe {
            try_ffi!(runtime::cudaDeviceCanAccessPeer(
                (&raw mut can_access_peer).cast(),
                self.0,
                other.0,
            ))?;
        }
        Ok(can_access_peer != 0)
    }

    /// Returns the value of the requested attribute of the link between devices.
    /// Supported attributes are represented by [`PeerToPeerAttribute`]:
    ///
    /// * [`PeerToPeerAttribute::PerformanceRank`]: relative performance of the link between the two devices.
    ///   Lower values are better.
    /// * [`PeerToPeerAttribute::AccessSupported`]: whether direct peer access is supported.
    /// * [`PeerToPeerAttribute::NativeAtomicSupported`]: whether native atomic operations over the link are supported.
    /// * [`PeerToPeerAttribute::CudaArrayAccessSupported`]: whether CUDA arrays are accessible over the link.
    ///
    /// Returns [`crate::error::Status::InvalidDevice`] if either device is invalid or if they represent the same device.
    ///
    /// Returns [`crate::error::Status::InvalidValue`] if `attrib` is not valid.
    ///
    /// # Errors
    ///
    /// Returns an error if either device is invalid, `attr` is not accepted by CUDA, CUDA cannot query the attribute, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn p2p_attribute(self, attr: PeerToPeerAttribute, other: Self) -> Result<i32> {
        let mut value: i32 = 0;
        // SAFETY: `value` is a local that `cudaDeviceGetP2PAttribute` writes through an exclusive pointer.
        unsafe {
            try_ffi!(runtime::cudaDeviceGetP2PAttribute(
                (&raw mut value).cast(),
                attr.into(),
                self.0,
                other.0,
            ))?;
        }
        Ok(value)
    }

    /// On devices where the L1 cache and shared memory use the same hardware resources, this returns the preferred cache configuration for the current device.
    /// This setting is only a preference.
    /// The runtime uses the requested configuration if possible, but it may choose a different configuration if required to execute functions.
    ///
    /// This returns [`FunctionCache::PreferNone`] on devices where the size of the L1 cache and shared memory are fixed.
    ///
    /// The supported cache configurations are:
    ///
    /// * [`FunctionCache::PreferNone`]: no preference for shared memory or L1 (default)
    /// * [`FunctionCache::PreferShared`]: prefer larger shared memory and smaller L1 cache
    /// * [`FunctionCache::PreferL1`]: prefer larger L1 cache and smaller shared memory
    /// * [`FunctionCache::PreferEqual`]: prefer equal size L1 cache and shared memory
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query the cache configuration, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn cache_config() -> Result<FunctionCache> {
        let mut config = runtime::cudaFuncCache::CU_FUNC_CACHE_PREFER_NONE;
        // SAFETY: `config` is a local that `cudaDeviceGetCacheConfig` writes through an exclusive pointer.
        unsafe {
            try_ffi!(runtime::cudaDeviceGetCacheConfig(&raw mut config))?;
        }
        Ok(config.try_into()?)
    }

    /// On devices where the L1 cache and shared memory use the same hardware resources, this sets through cacheConfig the preferred cache configuration for the current device.
    /// This setting is only a preference.
    /// The runtime uses the requested configuration if possible, but it is free to choose a different configuration if required to execute the kernel.
    /// Any per-kernel cache preference set through the CUDA API takes precedence over this device-wide setting.
    /// Setting the device-wide cache configuration to [`FunctionCache::PreferNone`] causes subsequent kernel launches to prefer not changing the cache configuration unless required to launch the kernel.
    ///
    /// This setting does nothing on devices where the size of the L1 cache and shared memory are fixed.
    ///
    /// Launching a kernel with a different preference than the most recent preference setting may insert a device-side synchronization point.
    ///
    /// The supported cache configurations are:
    ///
    /// * [`FunctionCache::PreferNone`]: no preference for shared memory or L1 (default)
    /// * [`FunctionCache::PreferShared`]: prefer larger shared memory and smaller L1 cache
    /// * [`FunctionCache::PreferL1`]: prefer larger L1 cache and smaller shared memory
    /// * [`FunctionCache::PreferEqual`]: prefer equal size L1 cache and shared memory
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot set the cache configuration, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn set_cache_config(config: FunctionCache) -> Result<()> {
        // SAFETY: `cudaDeviceSetCacheConfig` receives only an enum value.
        unsafe {
            try_ffi!(runtime::cudaDeviceSetCacheConfig(config.into()))?;
        }
        Ok(())
    }

    /// Returns the least and greatest stream priority numerical values.
    /// Stream priorities follow a convention where lower numbers represent greater priorities.
    /// The range of meaningful stream priorities is given by \[greatest, least\].
    /// If a stream is created with a priority value outside this range, the priority is automatically clamped.
    /// See [`StreamOptions::priority`](crate::stream::StreamOptions::priority) for creating a priority stream.
    ///
    /// Returns 0 for both values if the current context's device does not support stream priorities.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query the stream-priority range, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn stream_priority_range() -> Result<StreamPriorityRange> {
        let mut least = 0;
        let mut greatest = 0;
        // SAFETY: `least` and `greatest` are locals that CUDA writes through exclusive pointers.
        unsafe {
            try_ffi!(runtime::cudaDeviceGetStreamPriorityRange(
                &raw mut least,
                &raw mut greatest,
            ))?;
        }
        Ok(StreamPriorityRange { least, greatest })
    }

    /// Returns this device's ordinal.
    #[must_use]
    pub const fn id(self) -> DeviceId {
        self.0
    }
}

impl_enum_display!(ComputeMode, {
    ComputeMode::Default => "cudaComputeModeDefault",
    ComputeMode::Exclusive => "cudaComputeModeExclusive",
    ComputeMode::Prohibited => "cudaComputeModeProhibited",
    ComputeMode::ExclusiveProcess => "cudaComputeModeExclusiveProcess",
});

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;
    use crate::testing;

    #[test]
    fn test_it_works() -> Result<()> {
        let _lock = testing::device_lock(Device::new(0))?;
        assert!(Device::count()? > 0);
        assert!(!Device::new(0).properties()?.name.is_empty());
        Ok(())
    }
}

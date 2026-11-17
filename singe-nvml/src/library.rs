use std::{ffi::CString, mem, ptr, sync::Arc};

use singe_core::{CudaVersion, report_cleanup_failure, string_from_c_chars};
use singe_nvml_sys as sys;

use crate::{
    device::Device,
    error::{Error, Result},
    try_ffi,
    types::{
        ConfComputeKeyRotationThreshold, ConfComputeSystemCaps, ConfComputeSystemSettings,
        ConfComputeSystemState, EventTypes, ExcludedDeviceInfo, GpmMetric, GpmMetricId, HwbcEntry,
        InitFlags, PgpuMetadata, Pid, SystemEventNotification, SystemEventTypes, VgpuCompatibility,
        VgpuDriverCapability, VgpuMetadata, VgpuVersion, VgpuVersionRange,
    },
    unit::Unit,
    utility::{struct_version, to_u32, to_usize},
};

/// Owns an NVML GPU event set and releases it on drop.
///
/// It keeps NVML initialized.
#[derive(Debug)]
pub struct EventSet {
    raw: sys::nvmlEventSet_t,
    library: Library,
}

/// Owns an NVML system event set and releases it on drop.
///
/// It keeps NVML initialized.
#[derive(Debug)]
pub struct SystemEventSet {
    raw: sys::nvmlSystemEventSet_t,
    library: Library,
}

/// Handle to an NVML initialization.
///
/// Clones share one initialization, and every device, unit, instance, event set, and sample queried through it keeps that initialization alive.
/// NVML shuts down when the last of them is dropped.
#[derive(Debug, Clone)]
pub struct Library {
    _initialization: Arc<Initialization>,
}

// Calls `nvmlShutdown` once, after the last `Library` and every object that retains one is gone.
#[derive(Debug)]
struct Initialization;

/// Owns a native GPU Performance Monitoring sample.
///
/// Dropping the sample releases its native allocation.
/// It keeps NVML initialized.
#[derive(Debug)]
pub struct GpmSample {
    raw: sys::nvmlGpmSample_t,
    library: Library,
}

// SAFETY: NVML documents its API as thread-safe, the event set is owned exclusively and freed once on drop, and the retained `Library` is `Send`.
unsafe impl Send for EventSet {}

// SAFETY: NVML documents its API as thread-safe, the only `&self` operation registers devices, and waits, which reuse the set's context storage, take `&mut EventSet`.
unsafe impl Sync for EventSet {}

// SAFETY: NVML documents its API as thread-safe, the system event set is owned exclusively and freed once on drop, and the retained `Library` is `Send`.
unsafe impl Send for SystemEventSet {}

// SAFETY: NVML documents its API as thread-safe, so concurrent registrations and waits through `&SystemEventSet` are serialized by NVML.
unsafe impl Sync for SystemEventSet {}

// SAFETY: NVML documents its API as thread-safe, the sample is owned exclusively and freed once on drop, and the retained `Library` is `Send`.
unsafe impl Send for GpmSample {}

// SAFETY: NVML writes a sample only through `&mut GpmSample`, so shared references only read it, and NVML documents its API as thread-safe.
unsafe impl Sync for GpmSample {}

/// A transferred NVML event-set handle with the initialization it was created under.
#[derive(Debug)]
pub struct RawEventSet {
    /// The owned native event-set handle.
    pub handle: sys::nvmlEventSet_t,
    /// The NVML initialization kept alive until `handle` is freed.
    pub library: Library,
}

/// A transferred NVML system event-set handle with the initialization it was created under.
#[derive(Debug)]
pub struct RawSystemEventSet {
    /// The owned native system event-set handle.
    pub handle: sys::nvmlSystemEventSet_t,
    /// The NVML initialization kept alive until `handle` is freed.
    pub library: Library,
}

/// A transferred NVML GPM sample handle with the initialization it was allocated under.
#[derive(Debug)]
pub struct RawGpmSample {
    /// The owned native GPM sample handle.
    pub handle: sys::nvmlGpmSample_t,
    /// The NVML initialization kept alive until `handle` is freed.
    pub library: Library,
}

// SAFETY: the record owns its event-set handle exclusively, NVML documents its API as thread-safe, and the retained `Library` is `Send`.
unsafe impl Send for RawEventSet {}

// SAFETY: the record owns its system event-set handle exclusively, NVML documents its API as thread-safe, and the retained `Library` is `Send`.
unsafe impl Send for RawSystemEventSet {}

// SAFETY: the record owns its sample handle exclusively, NVML documents its API as thread-safe, and the retained `Library` is `Send`.
unsafe impl Send for RawGpmSample {}

impl Library {
    /// Loads and initializes NVML with `flags`, without initializing any GPU.
    ///
    /// GPUs are initialized lazily when a device handle is requested, so NVML can reach a GPU while other GPUs are in a bad state.
    /// Empty `flags` give NVML's default initialization.
    /// NVML counts initializations and shuts down when the last [`Library`] clone and every object retaining it are dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if the NVIDIA driver is not running, if NVML does not have permission to communicate with the driver, or if NVML reports an unexpected failure.
    pub fn create(flags: InitFlags) -> Result<Self> {
        sys::load()?;
        // SAFETY: `nvmlInitWithFlags` takes only the flag bits, and NVML allows initialization from any thread.
        unsafe {
            try_ffi!(sys::nvmlInitWithFlags(flags.bits()))?;
        }
        Ok(Self::initialized())
    }

    fn initialized() -> Self {
        Self {
            _initialization: Arc::new(Initialization),
        }
    }

    /// Returns the version of the system's graphics driver.
    ///
    /// The version is an alphanumeric string of at most 79 characters.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal version buffer is too small, if NVML rejects the output argument, or if NVML has not been initialized.
    pub fn driver_version(&self) -> Result<String> {
        let mut buffer = [0; sys::NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE as usize];
        // SAFETY: `buffer` holds the `buffer.len()` writable bytes passed as its length.
        unsafe {
            try_ffi!(sys::nvmlSystemGetDriverVersion(
                buffer.as_mut_ptr(),
                to_u32(buffer.len(), "NVML buffer length")?,
            ))?;
        }
        Ok(string_from_c_chars(&buffer))
    }

    /// Returns the version of the installed CUDA driver library.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA driver library or version function cannot be found, if NVML rejects the output argument, or if the reported version is negative.
    pub fn cuda_driver_version(&self) -> Result<CudaVersion> {
        let mut raw = 0;
        // SAFETY: `raw` is a writable local integer.
        unsafe {
            try_ffi!(sys::nvmlSystemGetCudaDriverVersion_v2(&raw mut raw))?;
        }
        CudaVersion::from_raw(raw).ok_or_else(|| Error::out_of_range("cuda driver version"))
    }

    /// Returns the name of the process with the provided process ID.
    ///
    /// The name is encoded in ANSI and truncated to 4096 bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the process ID, if the process does not exist, if the current process lacks permission, or if NVML reports an unexpected failure.
    pub fn process_name(&self, pid: Pid) -> Result<String> {
        let mut buffer = vec![0; 4096];
        // SAFETY: `buffer` holds the `buffer.len()` writable bytes passed as its length.
        unsafe {
            try_ffi!(sys::nvmlSystemGetProcessName(
                pid.0,
                buffer.as_mut_ptr(),
                to_u32(buffer.len(), "NVML buffer length")?,
            ))?;
        }
        Ok(string_from_c_chars(&buffer))
    }

    /// Returns the branch of the installed NVIDIA driver.
    ///
    /// The branch is an alphanumeric string of at most 79 characters.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal driver-branch buffer is too small, if NVML rejects the query arguments, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn driver_branch(&self) -> Result<String> {
        let mut info = sys::nvmlSystemDriverBranchInfo_t {
            version: struct_version::<sys::nvmlSystemDriverBranchInfo_t>(1),
            ..Default::default()
        };
        // SAFETY: `info` is a versioned local struct whose `branch` array holds the `NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE` bytes passed as its length.
        unsafe {
            try_ffi!(sys::nvmlSystemGetDriverBranch(
                &raw mut info,
                sys::NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE,
            ))?;
        }
        Ok(string_from_c_chars(&info.branch))
    }

    /// Returns whether the vGPU driver supports `capability`.
    ///
    /// Requires a Maxwell or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the requested capability, if the current driver state does not support vGPU capability queries, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn vgpu_driver_capability(&self, capability: VgpuDriverCapability) -> Result<bool> {
        let mut result = 0;
        // SAFETY: `capability` converts to a native capability value, and `result` is a writable local integer.
        unsafe {
            try_ffi!(sys::nvmlGetVgpuDriverCapabilities(
                capability.into(),
                &raw mut result,
            ))?;
        }
        Ok(result != 0)
    }

    /// Returns the preset range of supported vGPU versions and the range currently configured by an administrator.
    ///
    /// Both ranges are the same unless [`Library::set_vgpu_version`] overrode the preset range.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML does not support this query, rejects the output buffers, or fails while fetching the version ranges.
    pub fn vgpu_version(&self) -> Result<VgpuVersionRange> {
        let mut supported = sys::nvmlVgpuVersion_t::default();
        let mut current = sys::nvmlVgpuVersion_t::default();
        // SAFETY: `supported` and `current` are writable local structs.
        unsafe {
            try_ffi!(sys::nvmlGetVgpuVersion(
                &raw mut supported,
                &raw mut current
            ))?;
        }
        Ok(VgpuVersionRange {
            supported: supported.into(),
            current: current.into(),
        })
    }

    /// Overrides the preset range of supported vGPU versions with `version`.
    ///
    /// The range must be a subset of the preset range from [`Library::vgpu_version`], and no VM may be running on the host.
    /// It is advertised to guest VMs for version negotiation: a guest driver whose supported range does not overlap it fails to load, and otherwise loads with the highest overlapping version.
    /// The preset range returns after a host reboot or driver reload.
    ///
    /// # Errors
    ///
    /// Returns an error if `version` is invalid or outside the preset supported range, if a VM is running on the host, or if the installed vGPU Compressor does not support overriding the version range.
    pub fn set_vgpu_version(&self, version: VgpuVersion) -> Result<()> {
        let mut version: sys::nvmlVgpuVersion_t = version.into();
        // SAFETY: `version` is a local struct that NVML reads during the call.
        unsafe {
            try_ffi!(sys::nvmlSetVgpuVersion(&raw mut version))?;
        }
        Ok(())
    }

    /// Returns whether a vGPU instance, described by [`VgpuInstance::metadata`](crate::vgpu_instance::VgpuInstance::metadata), can boot on a physical GPU, described by [`Device::vgpu_metadata`].
    ///
    /// When compatibility is limited, the result names the limiting factor.
    /// The check ignores dynamic capacity, which can still prevent a vGPU or its VM from booting.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects either metadata blob or reports an unexpected compatibility-query failure.
    pub fn vgpu_compatibility(
        &self,
        vgpu_metadata: &VgpuMetadata,
        pgpu_metadata: &PgpuMetadata,
    ) -> Result<VgpuCompatibility> {
        let mut vgpu_raw = vgpu_metadata.encode_raw()?;
        let mut pgpu_raw = pgpu_metadata.encode_raw()?;
        let mut compatibility = sys::nvmlVgpuPgpuCompatibility_t::default();
        // SAFETY: `encode_raw` sizes both blobs to at least their native metadata structs, and `compatibility` is a writable local struct.
        unsafe {
            try_ffi!(sys::nvmlGetVgpuCompatibility(
                vgpu_raw.as_mut_ptr().cast(),
                pgpu_raw.as_mut_ptr().cast(),
                &raw mut compatibility,
            ))?;
        }
        Ok(compatibility.into())
    }

    /// Returns the set of GPUs that have CPU affinity with the given CPU number.
    /// Supported on Linux only.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the CPU number, if topology discovery is not supported on this platform, or if NVML fails while collecting the GPU set.
    pub fn topology_gpu_set(&self, cpu_number: u32) -> Result<Vec<Device>> {
        let mut count = 0;
        // SAFETY: a null array with a zero count asks NVML only for the count, written to the local `count`.
        let status = unsafe {
            sys::nvmlSystemGetTopologyGpuSet(cpu_number, &raw mut count, ptr::null_mut())
        };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && count == 0 {
            return Ok(Vec::new());
        }

        if status != sys::nvmlReturn_t::NVML_SUCCESS
            && status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
        {
            return Err(status.into());
        }

        let mut devices = vec![ptr::null_mut(); to_usize(count, "GPU count")?];
        // SAFETY: `devices` holds the `count` writable handle slots passed as its length.
        unsafe {
            try_ffi!(sys::nvmlSystemGetTopologyGpuSet(
                cpu_number,
                &raw mut count,
                devices.as_mut_ptr(),
            ))?;
        }

        devices.truncate(to_usize(count, "GPU count")?);
        Ok(devices
            .into_iter()
            // SAFETY: NVML returned this device handle under `self`, which each `Device` retains to keep NVML initialized.
            .map(|handle| unsafe { Device::from_raw(handle, self) })
            .collect())
    }

    /// Returns the number of GPUs excluded from NVML enumeration.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the count output.
    pub fn excluded_device_count(&self) -> Result<u32> {
        let mut count = 0;
        // SAFETY: `count` is a writable local integer.
        unsafe {
            try_ffi!(sys::nvmlGetExcludedDeviceCount(&raw mut count))?;
        }
        Ok(count)
    }

    /// Returns information about the excluded GPU at `index`.
    ///
    /// Valid indices are below [`Library::excluded_device_count`].
    ///
    /// # Errors
    ///
    /// Returns an error if `index` is out of range or NVML rejects the device information output.
    pub fn excluded_device(&self, index: u32) -> Result<ExcludedDeviceInfo> {
        let mut info = sys::nvmlExcludedDeviceInfo_t::default();
        // SAFETY: `info` is a writable local struct, and NVML validates `index`.
        unsafe {
            try_ffi!(sys::nvmlGetExcludedDeviceInfoByIndex(index, &raw mut info))?;
        }
        Ok(info.into())
    }

    /// Returns information for every device excluded from normal NVML enumeration.
    ///
    /// # Errors
    ///
    /// Returns an error if the excluded-device count or any per-device query fails; no partial list is returned.
    pub fn excluded_devices(&self) -> Result<Vec<ExcludedDeviceInfo>> {
        (0..self.excluded_device_count()?)
            .map(|index| self.excluded_device(index))
            .collect()
    }

    /// Returns the number of GPUs in the system.
    ///
    /// The count includes GPUs for which [`Library::device`] returns [`Status::NoPermission`](crate::error::Status::NoPermission).
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the count output, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn device_count(&self) -> Result<u32> {
        let mut count = 0;
        // SAFETY: `count` is a writable local integer.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetCount_v2(&raw mut count))?;
        }
        Ok(count)
    }

    /// Returns the number of units in the system.
    ///
    /// Units exist only on S-class systems.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the count output, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn unit_count(&self) -> Result<u32> {
        let mut count = 0;
        // SAFETY: `count` is a writable local integer.
        unsafe {
            try_ffi!(sys::nvmlUnitGetCount(&raw mut count))?;
        }
        Ok(count)
    }

    /// Returns the unit at `index`.
    ///
    /// Valid indices are below [`Library::unit_count`].
    /// The enumeration order can change between reboots.
    ///
    /// # Errors
    ///
    /// Returns an error if `index` is out of range, if NVML rejects the handle output, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn unit(&self, index: u32) -> Result<Unit> {
        let mut handle = ptr::null_mut();
        // SAFETY: `handle` is a writable local handle slot, and NVML validates `index`.
        unsafe {
            try_ffi!(sys::nvmlUnitGetHandleByIndex(index, &raw mut handle))?;
        }
        // SAFETY: NVML returned this unit handle under `self`, which the `Unit` retains to keep NVML initialized.
        Ok(unsafe { Unit::from_raw(handle, self) })
    }

    /// Returns all multi-GPU units reported by NVML.
    ///
    /// # Errors
    ///
    /// Returns an error if the unit count or any per-unit query fails; no partial list is returned.
    pub fn units(&self) -> Result<Vec<Unit>> {
        (0..self.unit_count()?)
            .map(|index| self.unit(index))
            .collect()
    }

    /// Returns the GPU at `index` and initializes it.
    ///
    /// Valid indices are below [`Library::device_count`].
    /// The enumeration order can change between reboots and need not match other libraries, such as CUDA; [`Library::device_by_uuid`] and [`Library::device_by_pci_bus_id`] find a GPU stably.
    ///
    /// # Errors
    ///
    /// Returns an error if `index` is out of range, if the process cannot access the target GPU, if the GPU cannot be initialized because of power, interrupt, or bus-access problems, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn device(&self, index: u32) -> Result<Device> {
        let mut handle = ptr::null_mut();
        // SAFETY: `handle` is a writable local handle slot, and NVML validates `index`.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetHandleByIndex_v2(index, &raw mut handle))?;
        }
        // SAFETY: NVML returned this device handle under `self`, which the `Device` retains to keep NVML initialized.
        Ok(unsafe { Device::from_raw(handle, self) })
    }

    /// Returns the GPU with the UUID `uuid` and initializes it.
    ///
    /// NVML may initialize other GPUs while searching.
    ///
    /// # Errors
    ///
    /// Returns an error if `uuid` contains an interior NUL byte or does not identify a device, if NVML cannot initialize one of the GPUs searched, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn device_by_uuid(&self, uuid: &str) -> Result<Device> {
        let uuid = CString::new(uuid)?;
        let mut handle = ptr::null_mut();
        // SAFETY: `uuid` is a NUL-terminated string alive for the call, and `handle` is a writable local handle slot.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetHandleByUUID(
                uuid.as_ptr(),
                &raw mut handle
            ))?;
        }
        // SAFETY: NVML returned this device handle under `self`, which the `Device` retains to keep NVML initialized.
        Ok(unsafe { Device::from_raw(handle, self) })
    }

    /// Returns the GPU with the PCI bus ID `pci_bus_id`, as reported by [`Device::pci_info`], and initializes it.
    ///
    /// # Errors
    ///
    /// Returns an error if `pci_bus_id` contains an interior NUL byte or does not identify a device, if the process cannot access the target GPU, if NVML cannot initialize it because of power, interrupt, or bus-access problems, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn device_by_pci_bus_id(&self, pci_bus_id: &str) -> Result<Device> {
        let pci_bus_id = CString::new(pci_bus_id)?;
        let mut handle = ptr::null_mut();
        // SAFETY: `pci_bus_id` is a NUL-terminated string alive for the call, and `handle` is a writable local handle slot.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetHandleByPciBusId_v2(
                pci_bus_id.as_ptr(),
                &raw mut handle,
            ))?;
        }
        // SAFETY: NVML returned this device handle under `self`, which the `Device` retains to keep NVML initialized.
        Ok(unsafe { Device::from_raw(handle, self) })
    }

    /// Returns the first enumerated device whose serial number matches `serial`.
    ///
    /// # Errors
    ///
    /// Returns an error if enumeration or a serial-number query fails, or [`Status::NotFound`](crate::error::Status::NotFound) if no GPU has the serial number `serial`.
    pub fn device_by_serial(&self, serial: &str) -> Result<Device> {
        for device in self.devices()? {
            if device.serial()? == serial {
                return Ok(device);
            }
        }

        Err(sys::nvmlReturn_t::NVML_ERROR_NOT_FOUND.into())
    }

    /// Returns all devices reported by NVML.
    ///
    /// # Errors
    ///
    /// Returns an error if the device count or any per-device query fails; no partial list is returned.
    pub fn devices(&self) -> Result<Vec<Device>> {
        (0..self.device_count()?)
            .map(|index| self.device(index))
            .collect()
    }

    /// Creates an empty event set, which is freed on drop.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the event-set output, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn event_set(&self) -> Result<EventSet> {
        let mut set = ptr::null_mut();
        // SAFETY: `set` is a writable local handle slot.
        unsafe {
            try_ffi!(sys::nvmlEventSetCreate(&raw mut set))?;
        }
        // SAFETY: NVML created this event set under `self` and nothing else owns it.
        unsafe {
            EventSet::from_raw(RawEventSet {
                handle: set,
                library: self.clone(),
            })
        }
    }

    /// Creates an empty system event set, which is freed on drop.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the request layout, if NVML rejects the request, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn system_event_set(&self) -> Result<SystemEventSet> {
        let mut request = sys::nvmlSystemEventSetCreateRequest_t {
            version: struct_version::<sys::nvmlSystemEventSetCreateRequest_t>(1),
            ..Default::default()
        };
        // SAFETY: `request` is a versioned local struct whose `set` field NVML writes.
        unsafe {
            try_ffi!(sys::nvmlSystemEventSetCreate(&raw mut request))?;
        }
        // SAFETY: NVML created this system event set under `self` and nothing else owns it.
        unsafe {
            SystemEventSet::from_raw(RawSystemEventSet {
                handle: request.set,
                library: self.clone(),
            })
        }
    }

    /// Returns the global `NVLink` bandwidth mode.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the output, if the system does not support global `NVLink` bandwidth mode, or if the current process lacks the required privileges.
    pub fn nvlink_bw_mode(&self) -> Result<u32> {
        let mut mode = 0;
        // SAFETY: `mode` is a writable local integer.
        unsafe {
            try_ffi!(sys::nvmlSystemGetNvlinkBwMode(&raw mut mode))?;
        }
        Ok(mode)
    }

    /// Returns Confidential Computing system capabilities.
    ///
    /// Requires an Ampere or newer GPU on Linux or Windows TCC.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the output, if the system does not support this Confidential Computing query, or if NVML has not been initialized.
    pub fn conf_compute_capabilities(&self) -> Result<ConfComputeSystemCaps> {
        let mut caps = sys::nvmlConfComputeSystemCaps_t::default();
        // SAFETY: `caps` is a writable local struct.
        unsafe {
            try_ffi!(sys::nvmlSystemGetConfComputeCapabilities(&raw mut caps))?;
        }
        Ok(caps.into())
    }

    /// Returns Confidential Computing system state.
    ///
    /// Requires an Ampere or newer GPU on Linux or Windows TCC.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the output, if the system does not support this Confidential Computing query, or if NVML has not been initialized.
    pub fn conf_compute_state(&self) -> Result<ConfComputeSystemState> {
        let mut state = sys::nvmlConfComputeSystemState_t::default();
        // SAFETY: `state` is a writable local struct.
        unsafe {
            try_ffi!(sys::nvmlSystemGetConfComputeState(&raw mut state))?;
        }
        Ok(state.into())
    }

    /// Returns whether the GPUs accept Confidential Computing work.
    ///
    /// Requires an Ampere or newer GPU on Linux or Windows TCC.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the output, if the system does not support this Confidential Computing query, or if NVML has not been initialized.
    pub fn conf_compute_gpus_ready_state(&self) -> Result<bool> {
        let mut accepting_work = 0;
        // SAFETY: `accepting_work` is a writable local integer.
        unsafe {
            try_ffi!(sys::nvmlSystemGetConfComputeGpusReadyState(
                &raw mut accepting_work
            ))?;
        }
        Ok(accepting_work == sys::NVML_CC_ACCEPTING_CLIENT_REQUESTS_TRUE)
    }

    /// Returns the Confidential Computing key rotation thresholds.
    ///
    /// Requires a Hopper or newer GPU on Linux or Windows TCC.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the request, if the system does not support this Confidential Computing query, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn conf_compute_key_rotation_threshold(&self) -> Result<ConfComputeKeyRotationThreshold> {
        let mut info = sys::nvmlConfComputeGetKeyRotationThresholdInfo_t {
            version: struct_version::<sys::nvmlConfComputeGetKeyRotationThresholdInfo_t>(1),
            ..Default::default()
        };
        // SAFETY: `info` is a writable local struct with its version set.
        unsafe {
            try_ffi!(sys::nvmlSystemGetConfComputeKeyRotationThresholdInfo(
                &raw mut info,
            ))?;
        }
        Ok(info.into())
    }

    /// Returns Confidential Computing system settings.
    ///
    /// Requires a Hopper or newer GPU on Linux or Windows TCC.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the request layout, if NVML rejects the request, if a target GPU is inaccessible, if the system does not support Confidential Computing settings, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn conf_compute_settings(&self) -> Result<ConfComputeSystemSettings> {
        let mut settings = sys::nvmlSystemConfComputeSettings_t {
            version: struct_version::<sys::nvmlSystemConfComputeSettings_t>(1),
            ..Default::default()
        };
        // SAFETY: `settings` is a writable local struct with its version set.
        unsafe {
            try_ffi!(sys::nvmlSystemGetConfComputeSettings(&raw mut settings))?;
        }
        Ok(settings.into())
    }

    /// Returns the IDs and firmware versions of the Host Interface Cards (HICs) connected to an S-class system.
    ///
    /// # Errors
    ///
    /// Returns an error if the number of HIC entries changes while the wrapper is fetching them, if NVML rejects the query, or if NVML has not been initialized.
    pub fn hic_versions(&self) -> Result<Vec<HwbcEntry>> {
        let mut count = 0;
        // SAFETY: a null array with a zero count asks NVML only for the count, written to the local `count`.
        let status = unsafe { sys::nvmlSystemGetHicVersion(&raw mut count, ptr::null_mut()) };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && count == 0 {
            return Ok(Vec::new());
        }

        if status != sys::nvmlReturn_t::NVML_SUCCESS
            && status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
        {
            return Err(status.into());
        }

        let mut entries = vec![sys::nvmlHwbcEntry_t::default(); to_usize(count, "HIC count")?];
        // SAFETY: `entries` holds the `count` writable entries passed as its length.
        unsafe {
            try_ffi!(sys::nvmlSystemGetHicVersion(
                &raw mut count,
                entries.as_mut_ptr(),
            ))?;
        }

        entries.truncate(to_usize(count, "HIC count")?);
        Ok(entries.into_iter().map(Into::into).collect())
    }

    /// Allocates a GPU Performance Monitoring sample buffer.
    ///
    /// [`Library::gpm_metrics`] compares two samples; requires a Hopper or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the allocation request or if system memory is insufficient.
    pub fn gpm_sample(&self) -> Result<GpmSample> {
        let mut sample = ptr::null_mut();
        // SAFETY: `sample` is a writable local handle slot.
        unsafe {
            try_ffi!(sys::nvmlGpmSampleAlloc(&raw mut sample))?;
        }
        // SAFETY: NVML allocated this sample under `self` and nothing else owns it.
        unsafe {
            GpmSample::from_raw(RawGpmSample {
                handle: sample,
                library: self.clone(),
            })
        }
    }

    /// Computes the GPU Performance Monitoring metrics `metric_ids` between two samples.
    ///
    /// Fill `sample1` and `sample2` with [`Device::gpm_sample`] more than 100 ms apart; requires a Hopper or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if the metric count exceeds the native array capacity or cannot fit in `u32`, or NVML rejects the samples or metric request.
    /// Per-metric NVML failures are retained in each returned metric record.
    pub fn gpm_metrics(
        &self,
        sample1: &GpmSample,
        sample2: &GpmSample,
        metric_ids: &[GpmMetricId],
    ) -> Result<Vec<GpmMetric>> {
        let mut metrics_get = sys::nvmlGpmMetricsGet_t {
            version: sys::NVML_GPM_METRICS_GET_VERSION,
            numMetrics: to_u32(metric_ids.len(), "metric_ids")?,
            sample1: sample1.raw,
            sample2: sample2.raw,
            ..Default::default()
        };
        if metric_ids.len() > metrics_get.metrics.len() {
            return Err(Error::out_of_range("metric_ids"));
        }

        for (slot, metric_id) in metrics_get.metrics.iter_mut().zip(metric_ids) {
            slot.metricId = metric_id.0;
        }

        // SAFETY: `metrics_get` is a versioned local struct whose metric count was checked against its array, and both samples are live allocations.
        unsafe {
            try_ffi!(sys::nvmlGpmMetricsGet(&raw mut metrics_get))?;
        }

        Ok(metrics_get.metrics[..metric_ids.len()]
            .iter()
            // SAFETY: NVML filled these metrics, and their name pointers refer to strings NVML owns.
            .map(|&metric| unsafe { GpmMetric::from_raw(metric) })
            .collect())
    }
}

impl Drop for Initialization {
    fn drop(&mut self) {
        // SAFETY: each `Initialization` pairs with one successful NVML initialization and drops once, after every owner retaining it.
        unsafe {
            if let Err(error) = try_ffi!(sys::nvmlShutdown()) {
                report_cleanup_failure("shut down", "nvml", &error);
            }
        }
    }
}

impl GpmSample {
    /// Wraps an existing NVML GPM sample handle.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be a valid NVML GPM sample handle allocated while `raw.library`'s initialization was alive.
    /// The returned value takes ownership of the handle and frees it on drop.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullHandle`] if `raw.handle` is null.
    pub unsafe fn from_raw(raw: RawGpmSample) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            raw: raw.handle,
            library: raw.library,
        })
    }

    /// Returns the native sample handle without transferring ownership.
    #[must_use]
    pub const fn as_raw(&self) -> sys::nvmlGpmSample_t {
        self.raw
    }

    /// Returns the NVML initialization that this sample keeps alive.
    #[must_use]
    pub const fn library(&self) -> &Library {
        &self.library
    }

    /// Consumes this sample and returns the owned raw NVML GPM sample handle with its retained [`Library`].
    ///
    /// The caller becomes responsible for freeing the handle before dropping the library.
    #[must_use]
    pub fn into_raw(mut self) -> RawGpmSample {
        RawGpmSample {
            handle: mem::replace(&mut self.raw, ptr::null_mut()),
            library: self.library.clone(),
        }
    }
}

impl Drop for GpmSample {
    fn drop(&mut self) {
        if self.raw.is_null() {
            return;
        }

        // SAFETY: `self` owns the non-null sample and frees it once, while `self.library` keeps NVML initialized.
        unsafe {
            if let Err(error) = try_ffi!(sys::nvmlGpmSampleFree(self.raw)) {
                report_cleanup_failure("free", "nvml gpm sample", &error);
            }
        }
    }
}

impl EventSet {
    /// Wraps an existing NVML event-set handle.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be a valid NVML event-set handle created while `raw.library`'s initialization was alive.
    /// The returned value takes ownership of the handle and frees it on drop.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullHandle`] if `raw.handle` is null.
    pub unsafe fn from_raw(raw: RawEventSet) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            raw: raw.handle,
            library: raw.library,
        })
    }

    /// Returns the raw NVML event-set handle.
    #[must_use]
    pub const fn as_raw(&self) -> sys::nvmlEventSet_t {
        self.raw
    }

    /// Returns the NVML initialization that this event set keeps alive.
    #[must_use]
    pub const fn library(&self) -> &Library {
        &self.library
    }

    /// Consumes this event set and returns the owned raw NVML event-set handle with its retained [`Library`].
    ///
    /// The caller becomes responsible for freeing the handle before dropping the library.
    #[must_use]
    pub fn into_raw(mut self) -> RawEventSet {
        RawEventSet {
            handle: mem::replace(&mut self.raw, ptr::null_mut()),
            library: self.library.clone(),
        }
    }

    /// Starts recording `event_types` from `device` into this event set, on Linux.
    ///
    /// Earlier events are not recorded; [`EventSet::wait`] returns recorded events.
    /// ECC events need an ECC-enabled GPU, and power-capping events need power management support.
    ///
    /// After [`Status::UnknownError`](crate::error::Status::UnknownError) the event set is in an undefined state and must be dropped.
    /// After [`Status::NotSupported`](crate::error::Status::NotSupported) it stays usable, with none of `event_types` registered.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the event set or event mask, if the platform or requested event types are not supported, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn register_device(&self, device: &Device, event_types: EventTypes) -> Result<()> {
        // SAFETY: `device` and `self` hold live NVML handles whose retained libraries keep NVML initialized.
        unsafe {
            try_ffi!(sys::nvmlDeviceRegisterEvents(
                device.as_raw(),
                event_types.bits(),
                self.raw,
            ))
        }
    }
}

impl Drop for EventSet {
    fn drop(&mut self) {
        if self.raw.is_null() {
            return;
        }

        // SAFETY: `self` owns the non-null event set and frees it once, while `self.library` keeps NVML initialized.
        unsafe {
            if let Err(error) = try_ffi!(sys::nvmlEventSetFree(self.raw)) {
                report_cleanup_failure("free", "nvml event set", &error);
            }
        }
    }
}

impl SystemEventSet {
    /// Wraps an existing NVML system event-set handle.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be a valid NVML system event-set handle created while `raw.library`'s initialization was alive.
    /// The returned value takes ownership of the handle and frees it on drop.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullHandle`] if `raw.handle` is null.
    pub unsafe fn from_raw(raw: RawSystemEventSet) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            raw: raw.handle,
            library: raw.library,
        })
    }

    /// Returns the raw NVML system event-set handle.
    #[must_use]
    pub const fn as_raw(&self) -> sys::nvmlSystemEventSet_t {
        self.raw
    }

    /// Returns the NVML initialization that this event set keeps alive.
    #[must_use]
    pub const fn library(&self) -> &Library {
        &self.library
    }

    /// Consumes this event set and returns the owned raw NVML system event-set handle with its retained [`Library`].
    ///
    /// The caller becomes responsible for freeing the handle before dropping the library.
    #[must_use]
    pub fn into_raw(mut self) -> RawSystemEventSet {
        RawSystemEventSet {
            handle: mem::replace(&mut self.raw, ptr::null_mut()),
            library: self.library.clone(),
        }
    }

    /// Starts recording `event_types` into this system event set, on Linux.
    ///
    /// Earlier events are not recorded; [`SystemEventSet::wait`] returns recorded events.
    ///
    /// After [`Status::UnknownError`](crate::error::Status::UnknownError) the event set is in an undefined state and must be dropped.
    /// After [`Status::NotSupported`](crate::error::Status::NotSupported) it stays usable, with none of `event_types` registered.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the request layout, if NVML rejects the event registration request, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn register_events(&self, event_types: SystemEventTypes) -> Result<()> {
        let mut request = sys::nvmlSystemRegisterEventRequest_t {
            version: struct_version::<sys::nvmlSystemRegisterEventRequest_t>(1),
            eventTypes: event_types.bits(),
            set: self.raw,
        };
        // SAFETY: `request` is a versioned local struct naming this live system event set.
        unsafe { try_ffi!(sys::nvmlSystemRegisterEvents(&raw mut request)) }
    }

    /// Waits up to `timeout_ms` for system events and returns at most `max_events` of them.
    ///
    /// Returns immediately when events are ready, and can return early on an interrupt.
    /// When `max_events` events are returned, more may be pending.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the request layout, if NVML rejects the wait request, if no event arrives before `timeout_ms`, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn wait(&self, timeout_ms: u32, max_events: u32) -> Result<Vec<SystemEventNotification>> {
        let mut data =
            vec![sys::nvmlSystemEventData_v1_t::default(); to_usize(max_events, "max_events")?];
        let mut request = sys::nvmlSystemEventSetWaitRequest_t {
            version: struct_version::<sys::nvmlSystemEventSetWaitRequest_t>(1),
            timeoutms: timeout_ms,
            set: self.raw,
            data: data.as_mut_ptr(),
            dataSize: to_u32(data.len(), "NVML buffer length")?,
            ..Default::default()
        };
        // SAFETY: `request` is a versioned local struct naming this live set, and `data` holds the `dataSize` writable entries it points to.
        unsafe {
            try_ffi!(sys::nvmlSystemEventSetWait(&raw mut request))?;
        }

        data.truncate(to_usize(request.numEvent, "event count")?);
        Ok(data.into_iter().map(Into::into).collect())
    }
}

impl Drop for SystemEventSet {
    fn drop(&mut self) {
        if self.raw.is_null() {
            return;
        }

        let mut request = sys::nvmlSystemEventSetFreeRequest_t {
            version: struct_version::<sys::nvmlSystemEventSetFreeRequest_t>(1),
            set: self.raw,
        };
        // SAFETY: `self` owns the non-null system event set and frees it once, while `self.library` keeps NVML initialized.
        unsafe {
            if let Err(error) = try_ffi!(sys::nvmlSystemEventSetFree(&raw mut request)) {
                report_cleanup_failure("free", "nvml system event set", &error);
            }
        }
    }
}

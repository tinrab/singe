#[allow(unused_imports)]
use crate::error::Status;

use std::{
    ffi::CString,
    mem::{ManuallyDrop, MaybeUninit},
    ptr,
};

use singe_core::string_from_c_chars;
use singe_nvml_sys as sys;

use crate::{
    device::Device,
    error::Result,
    try_ffi,
    types::{
        ConfComputeKeyRotationThreshold, ConfComputeSystemCaps, ConfComputeSystemSettings,
        ConfComputeSystemState, CudaDriverVersion, EventData, EventTypes, ExcludedDeviceInfo,
        GpmMetric, GpmMetricId, HwbcEntry, InitFlags, PgpuMetadata, Pid, SystemEventData,
        SystemEventTypes, VgpuCompatibility, VgpuDriverCapability, VgpuMetadata, VgpuVersion,
        VgpuVersionRange,
    },
    unit::Unit,
    utility::struct_version,
};

#[derive(Debug)]
pub struct EventSet(sys::nvmlEventSet_t);

#[derive(Debug)]
pub struct SystemEventSet(sys::nvmlSystemEventSet_t);

#[derive(Debug)]
pub struct Library;

#[derive(Debug)]
pub struct GpmSample(sys::nvmlGpmSample_t);

impl Library {
    /// Initializes NVML, but does not initialize any GPUs yet.
    ///
    /// * Newer NVML initialization adds flags that let callers adjust initialization behavior.
    /// * In NVML 5.319, [`Library::create`] replaced the older initialization path that initialized all GPU devices in the system.
    ///
    /// This allows NVML to communicate with a GPU when other GPUs in the system are unstable or in a bad state.
    /// With this initialization mode, GPUs are discovered and initialized lazily when you request a device handle from this crate.
    ///
    /// In contrast, the older initialization path in NVML 4.304 would fail if any detected GPU was in a bad or unstable state.
    ///
    /// For all products.
    ///
    /// Call this once before invoking any other methods in the library.
    /// A reference count of the number of initializations is maintained.
    /// Shutdown only occurs when the reference count reaches zero.
    ///
    /// # Errors
    ///
    /// Returns an error if the NVIDIA driver is not running, if NVML does not
    /// have permission to communicate with the driver, or if NVML reports an
    /// unexpected failure.
    pub fn create() -> Result<Self> {
        unsafe {
            try_ffi!(sys::nvmlInit_v2())?;
        }
        Ok(Self)
    }

    /// Initializes NVML with the provided initialization flags.
    /// This follows the same reference-counting behavior as [`Library::create`].
    ///
    /// For all products.
    ///
    /// # Errors
    ///
    /// Returns an error if the NVIDIA driver is not running, if NVML does not
    /// have permission to communicate with the driver, or if NVML reports an
    /// unexpected failure.
    pub fn create_with_flags(flags: InitFlags) -> Result<Self> {
        unsafe {
            try_ffi!(sys::nvmlInitWithFlags(flags.bits()))?;
        }
        Ok(Self)
    }

    /// Returns the version of the system's graphics driver.
    ///
    /// For all products.
    ///
    /// The version identifier is an alphanumeric string.
    /// It does not exceed 80 bytes including the terminating NUL byte.
    /// This wrapper allocates the required NVML buffer internally.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal version buffer is too small, if NVML
    /// rejects the output argument, or if NVML has not been initialized.
    pub fn driver_version(&self) -> Result<String> {
        let mut buffer = [0i8; sys::NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE as usize];
        unsafe {
            try_ffi!(sys::nvmlSystemGetDriverVersion(
                buffer.as_mut_ptr(),
                buffer.len() as u32,
            ))?;
        }
        Ok(string_from_c_chars(&buffer))
    }

    /// Returns the version of the NVML library.
    ///
    /// For all products.
    ///
    /// The version identifier is an alphanumeric string.
    /// It does not exceed 80 bytes including the terminating NUL byte.
    /// This wrapper allocates the required NVML buffer internally.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal version buffer is too small or if NVML
    /// rejects the output argument.
    pub fn version(&self) -> Result<String> {
        crate::version()
    }

    /// Returns the version of the CUDA driver from the shared library.
    ///
    /// For all products.
    ///
    /// CUDA driver version obtained by calling the CUDA driver.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA driver library or version function cannot
    /// be found, or if NVML rejects the output argument.
    pub fn cuda_driver_version(&self) -> Result<CudaDriverVersion> {
        let mut raw = 0;
        unsafe {
            try_ffi!(sys::nvmlSystemGetCudaDriverVersion_v2(&raw mut raw))?;
        }
        Ok(CudaDriverVersion::from_raw(raw))
    }

    /// Returns the version of the CUDA driver.
    ///
    /// For all products.
    ///
    /// The CUDA driver version is retrieved from the currently installed version of CUDA.
    /// If the CUDA library is not found, this returns a known supported version number.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the query arguments.
    pub fn cuda_driver_version_fallback(&self) -> Result<CudaDriverVersion> {
        let mut raw = 0;
        unsafe {
            try_ffi!(sys::nvmlSystemGetCudaDriverVersion(&raw mut raw))?;
        }
        Ok(CudaDriverVersion::from_raw(raw))
    }

    pub fn process_name(&self, pid: Pid) -> Result<String> {
        self.process_name_with_capacity(pid, 1024)
    }

    /// Returns the name of the process with the provided process ID.
    ///
    /// For all products.
    ///
    /// The returned process name is truncated to `capacity` and encoded in ANSI.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the process ID or output buffer, if the
    /// process does not exist, if the current process lacks permission, if NVML
    /// has not been initialized, or if NVML reports an unexpected failure.
    pub fn process_name_with_capacity(&self, pid: Pid, capacity: usize) -> Result<String> {
        let mut buffer = vec![0i8; capacity];
        unsafe {
            try_ffi!(sys::nvmlSystemGetProcessName(
                pid.0,
                buffer.as_mut_ptr(),
                buffer.len() as u32,
            ))?;
        }
        Ok(string_from_c_chars(&buffer))
    }

    /// Returns the driver branch of the NVIDIA driver installed on the system.
    ///
    /// For all products.
    ///
    /// The branch identifier is an alphanumeric string.
    /// It does not exceed 80 bytes including the terminating NUL byte.
    /// This wrapper allocates the required NVML buffer internally.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal driver-branch buffer is too small, if
    /// NVML rejects the query arguments, if NVML has not been initialized, or if
    /// NVML reports an unexpected failure.
    pub fn driver_branch(&self) -> Result<String> {
        let mut info = sys::nvmlSystemDriverBranchInfo_t {
            version: struct_version::<sys::nvmlSystemDriverBranchInfo_t>(1),
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlSystemGetDriverBranch(
                &raw mut info,
                sys::NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE,
            ))?;
        }
        Ok(string_from_c_chars(&info.branch))
    }

    /// Returns the requested vGPU driver capability.
    ///
    /// See [`VgpuDriverCapability`] for the supported capabilities.
    /// Returns a boolean indicating whether the capability is supported.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the requested capability, if the current
    /// driver state does not support vGPU capability queries, if NVML has not
    /// been initialized, or if NVML reports an unexpected failure.
    pub fn vgpu_driver_capability(&self, capability: VgpuDriverCapability) -> Result<bool> {
        let mut result = 0;
        unsafe {
            try_ffi!(sys::nvmlGetVgpuDriverCapabilities(
                capability.into(),
                &raw mut result,
            ))?;
        }
        Ok(result != 0)
    }

    /// Query the ranges of supported vGPU versions.
    ///
    /// Returns the preset linear range of supported vGPU versions for the NVIDIA vGPU Manager and the administrator-configured range.
    /// If the preset range has not been overridden by [`Library::set_vgpu_version`], both ranges are the same.
    ///
    /// This wrapper returns both the preset supported range and the administrator-configured current range.
    /// By default, the current range matches the preset range.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML does not support this query, rejects the output
    /// buffers, or fails while fetching the version ranges.
    pub fn vgpu_version(&self) -> Result<VgpuVersionRange> {
        let mut supported = sys::nvmlVgpuVersion_t::default();
        let mut current = sys::nvmlVgpuVersion_t::default();
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

    /// Override the preset range of vGPU versions supported by the NVIDIA vGPU Manager with a range set by an administrator.
    ///
    /// Configures the NVIDIA vGPU Manager with an administrator-provided range of supported vGPU versions.
    /// This range must be a subset of the preset range that the NVIDIA vGPU Manager supports.
    /// The custom range set by an administrator takes precedence over the preset range and is advertised to the guest VM for negotiating the vGPU version.
    /// See [`Library::vgpu_version`] for details of how to query the preset range of versions supported.
    ///
    /// This overrides the preset vGPU version range with the administrator-provided range.
    ///
    /// After host system reboot or driver reload, the range of supported versions reverts to the range that is preset for the NVIDIA vGPU Manager.
    ///
    /// 1. The range set by the administrator must be a subset of the preset range that the NVIDIA vGPU Manager supports.
    ///    Otherwise, an error is returned.
    /// 2. If the range of supported guest driver versions does not overlap the range set by the administrator,
    ///    the guest driver fails to load.
    /// 3. If the range of supported guest driver versions overlaps the range set by the administrator,
    ///    the guest driver loads with a negotiated vGPU version equal to the
    ///    maximum value in the overlapping range.
    /// 4. No VMs must be running on the host when setting the version range.
    ///    If a VM is running on the host, the call fails.
    ///
    /// # Errors
    ///
    /// Returns an error if `version` is invalid or outside the preset supported
    /// range, if a VM is running on the host, or if the installed vGPU Manager
    /// does not support overriding the version range.
    pub fn set_vgpu_version(&self, version: VgpuVersion) -> Result<()> {
        let mut version: sys::nvmlVgpuVersion_t = version.into();
        unsafe {
            try_ffi!(sys::nvmlSetVgpuVersion(&raw mut version))?;
        }
        Ok(())
    }

    /// Takes a vGPU instance metadata structure read from [`VgpuInstance::metadata`](crate::vgpu_instance::VgpuInstance::metadata), and a vGPU metadata structure for a physical GPU read from [`Device::vgpu_metadata`], and returns compatibility information of the vGPU instance and the physical GPU.
    ///
    /// This wrapper returns compatibility information describing whether the vGPU or VM may be booted on the physical GPU.
    /// If the vGPU / VM compatibility with the physical GPU is limited, a limit code indicates the factor limiting compatibility.
    /// See the returned compatibility structure for the reported limit code.
    ///
    /// vGPU compatibility does not take into account dynamic capacity conditions that may limit a system's ability to boot a given vGPU or associated VM.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects either metadata blob or reports an
    /// unexpected compatibility-query failure.
    pub fn vgpu_compatibility(
        &self,
        vgpu_metadata: &VgpuMetadata,
        pgpu_metadata: &PgpuMetadata,
    ) -> Result<VgpuCompatibility> {
        let mut vgpu_raw = vgpu_metadata.encode_raw();
        let mut pgpu_raw = pgpu_metadata.encode_raw();
        let mut compatibility = sys::nvmlVgpuPgpuCompatibility_t::default();
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
    /// Returns an error if NVML rejects the CPU number, if topology discovery is
    /// not supported on this platform, or if NVML fails while collecting the GPU
    /// set.
    pub fn topology_gpu_set(&self, cpu_number: u32) -> Result<Vec<Device>> {
        let mut count = 0;
        let status = unsafe {
            sys::nvmlSystemGetTopologyGpuSet(cpu_number, &raw mut count, ptr::null_mut())
        };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && count == 0 {
            return Ok(Vec::new());
        }
        if status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE {
            return Err(status.into());
        }

        let mut devices = vec![ptr::null_mut(); count as usize];
        unsafe {
            try_ffi!(sys::nvmlSystemGetTopologyGpuSet(
                cpu_number,
                &raw mut count,
                devices.as_mut_ptr(),
            ))?;
        }
        devices.truncate(count as usize);
        Ok(devices
            .into_iter()
            .map(|handle| unsafe { Device::from_raw(handle) })
            .collect())
    }

    /// Returns the number of excluded GPU devices in the system.
    ///
    /// For all products.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the count output.
    pub fn excluded_device_count(&self) -> Result<u32> {
        let mut count = 0;
        unsafe {
            try_ffi!(sys::nvmlGetExcludedDeviceCount(&raw mut count))?;
        }
        Ok(count)
    }

    /// Acquire the device information for an excluded GPU device, based on its index.
    ///
    /// For all products.
    ///
    /// Valid indices are derived from the count returned by [`Library::excluded_device_count`].
    /// For example, if the count is 2 the valid indices are 0 and 1, corresponding to GPU 0 and GPU 1.
    ///
    /// # Errors
    ///
    /// Returns an error if `index` is out of range or NVML rejects the device
    /// information output.
    pub fn excluded_device(&self, index: u32) -> Result<ExcludedDeviceInfo> {
        let mut info = sys::nvmlExcludedDeviceInfo_t::default();
        unsafe {
            try_ffi!(sys::nvmlGetExcludedDeviceInfoByIndex(index, &raw mut info))?;
        }
        Ok(info.into())
    }

    pub fn excluded_devices(&self) -> Result<Vec<ExcludedDeviceInfo>> {
        (0..self.excluded_device_count()?)
            .map(|index| self.excluded_device(index))
            .collect()
    }

    /// Returns the number of compute devices in the system.
    /// A compute device is a single GPU.
    ///
    /// For all products.
    ///
    /// [`Library::device_count`] returns the count of all devices in the system even if [`Library::device`] returns [`Status::NoPermission`] for some devices.
    /// Update your code to handle this error, or use the NVML 4.304 or older header file.
    /// For backward binary compatibility reasons, the `_v1` symbol is still present in the shared library.
    /// The old `_v1` NVML entry point does not count devices that NVML has no permission to talk to.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the count output, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn device_count(&self) -> Result<u32> {
        let mut count = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetCount_v2(&raw mut count))?;
        }
        Ok(count)
    }

    /// Returns the number of units in the system.
    ///
    /// For S-class products.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the count output, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn unit_count(&self) -> Result<u32> {
        let mut count = 0;
        unsafe {
            try_ffi!(sys::nvmlUnitGetCount(&raw mut count))?;
        }
        Ok(count)
    }

    /// Acquire the handle for a particular unit, based on its index.
    ///
    /// For S-class products.
    ///
    /// Valid indices are derived from the count returned by [`Library::unit_count`].
    /// For example, if the count is 2 the valid indices are 0 and 1, corresponding to UNIT 0 and UNIT 1.
    ///
    /// The order in which NVML enumerates units has no guarantees of consistency between reboots.
    ///
    /// # Errors
    ///
    /// Returns an error if `index` is out of range, if NVML rejects the handle
    /// output, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn unit(&self, index: u32) -> Result<Unit> {
        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::nvmlUnitGetHandleByIndex(index, &raw mut handle))?;
        }
        Ok(unsafe { Unit::from_raw(handle) })
    }

    pub fn units(&self) -> Result<Vec<Unit>> {
        (0..self.unit_count()?)
            .map(|index| self.unit(index))
            .collect()
    }

    /// Acquire the handle for a particular device, based on its index.
    ///
    /// For all products.
    ///
    /// Valid indices are derived from the accessible device count returned by [`Library::device_count`].
    /// For example, if the count is 2 the valid indices are 0 and 1, corresponding to GPU 0 and GPU 1.
    ///
    /// The order in which NVML enumerates devices has no guarantees of consistency between reboots.
    /// Prefer PCI bus IDs or UUIDs for stable device lookup.
    /// See [`Library::device_by_uuid`] and [`Library::device_by_pci_bus_id`].
    ///
    /// The NVML index may not correlate with other libraries, such as the CUDA device index.
    ///
    /// Starting from NVML 5, this call causes NVML to initialize the target GPU. NVML may initialize additional GPUs if:
    ///
    /// * The target GPU is an SLI slave.
    ///
    /// [`Library::device_count`] returns the count of all devices in the system even if [`Library::device`] returns [`Status::NoPermission`] for some devices.
    /// Update your code to handle this error, or use the NVML 4.304 or older header file.
    /// For backward binary compatibility reasons, the `_v1` symbol is still present in the shared library.
    /// The old `_v1` NVML entry point does not count devices that NVML has no permission to talk to.
    ///
    /// This means that [`Library::device`] and _v1 can return different devices for the same index.
    /// Code that uses the default `_v2` mappings at the top of the file is unaffected.
    ///
    /// # Errors
    ///
    /// Returns an error if `index` is out of range, if the process cannot access
    /// the target GPU, if the GPU cannot be initialized because of power,
    /// interrupt, or bus-access problems, if NVML has not been initialized, or
    /// if NVML reports an unexpected failure.
    pub fn device(&self, index: u32) -> Result<Device> {
        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::nvmlDeviceGetHandleByIndex_v2(index, &raw mut handle))?;
        }
        Ok(unsafe { Device::from_raw(handle) })
    }

    /// Acquire the handle for a device from its globally unique immutable UUID.
    ///
    /// For all products.
    ///
    /// Starting from NVML 5, this call causes NVML to initialize the target GPU. NVML may initialize additional GPUs as it searches for the target GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if `uuid` contains an interior NUL byte or does not
    /// identify a device, if NVML cannot initialize one of the GPUs searched,
    /// if NVML has not been initialized, or if NVML reports an unexpected
    /// failure.
    pub fn device_by_uuid(&self, uuid: &str) -> Result<Device> {
        // TODO: add a typed nvmlUUID_t-based API for nvmlDeviceGetHandleByUUIDV when we design a clean ASCII/binary UUID abstraction.
        let uuid = CString::new(uuid)?;
        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::nvmlDeviceGetHandleByUUID(
                uuid.as_ptr(),
                &raw mut handle
            ))?;
        }
        Ok(unsafe { Device::from_raw(handle) })
    }

    /// Acquire the handle for a particular device, based on its PCI bus id.
    ///
    /// For all products.
    ///
    /// This value corresponds to the PCI bus ID returned by [`Device::pci_info`].
    ///
    /// Starting from NVML 5, this call causes NVML to initialize the target GPU. NVML may initialize additional GPUs if:
    ///
    /// * The target GPU is an SLI slave.
    ///
    /// Older NVML releases returned [`Status::NotFound`] instead of [`Status::NoPermission`].
    ///
    /// # Errors
    ///
    /// Returns an error if `pci_bus_id` contains an interior NUL byte or does
    /// not identify a device, if the process cannot access the target GPU, if
    /// NVML cannot initialize it because of power, interrupt, or bus-access
    /// problems, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn device_by_pci_bus_id(&self, pci_bus_id: &str) -> Result<Device> {
        let pci_bus_id = CString::new(pci_bus_id)?;
        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::nvmlDeviceGetHandleByPciBusId_v2(
                pci_bus_id.as_ptr(),
                &raw mut handle,
            ))?;
        }
        Ok(unsafe { Device::from_raw(handle) })
    }

    pub fn device_by_serial(&self, serial: &str) -> Result<Device> {
        for device in self.devices()? {
            if device.serial()? == serial {
                return Ok(device);
            }
        }

        Err(sys::nvmlReturn_t::NVML_ERROR_NOT_FOUND.into())
    }

    pub fn devices(&self) -> Result<Vec<Device>> {
        (0..self.device_count()?)
            .map(|index| self.device(index))
            .collect()
    }

    /// Creates an empty set of events.
    /// The returned event set is freed automatically when dropped.
    ///
    /// For Fermi or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the event-set output, if NVML has not
    /// been initialized, or if NVML reports an unexpected failure.
    pub fn event_set(&self) -> Result<EventSet> {
        let mut set = ptr::null_mut();
        unsafe {
            try_ffi!(sys::nvmlEventSetCreate(&raw mut set))?;
        }
        unsafe { EventSet::from_raw(set) }
    }

    /// Creates an empty set of system events.
    /// The returned system event set is freed automatically when dropped.
    ///
    /// For Fermi or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the
    /// request layout, if NVML rejects the request, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn system_event_set(&self) -> Result<SystemEventSet> {
        let mut request = sys::nvmlSystemEventSetCreateRequest_t {
            version: struct_version::<sys::nvmlSystemEventSetCreateRequest_t>(1),
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlSystemEventSetCreate(&raw mut request))?;
        }
        unsafe { SystemEventSet::from_raw(request.set) }
    }

    /// Returns the global NVLink bandwidth mode.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the output, if the system does not
    /// support global NVLink bandwidth mode, or if the current process lacks
    /// the required privileges.
    pub fn nvlink_bw_mode(&self) -> Result<u32> {
        let mut mode = 0;
        unsafe {
            try_ffi!(sys::nvmlSystemGetNvlinkBwMode(&raw mut mode))?;
        }
        Ok(mode)
    }

    /// Returns Confidential Computing system capabilities.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux, Windows TCC.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the output, if the system does not
    /// support this Confidential Computing query, or if NVML has not been
    /// initialized.
    pub fn conf_compute_capabilities(&self) -> Result<ConfComputeSystemCaps> {
        let mut caps = sys::nvmlConfComputeSystemCaps_t::default();
        unsafe {
            try_ffi!(sys::nvmlSystemGetConfComputeCapabilities(&raw mut caps))?;
        }
        Ok(caps.into())
    }

    /// Returns Confidential Computing system state.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux, Windows TCC.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the output, if the system does not
    /// support this Confidential Computing query, or if NVML has not been
    /// initialized.
    pub fn conf_compute_state(&self) -> Result<ConfComputeSystemState> {
        let mut state = sys::nvmlConfComputeSystemState_t::default();
        unsafe {
            try_ffi!(sys::nvmlSystemGetConfComputeState(&raw mut state))?;
        }
        Ok(state.into())
    }

    /// Returns Confidential Computing GPU ready state.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux, Windows TCC.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the output, if the system does not
    /// support this Confidential Computing query, or if NVML has not been
    /// initialized.
    pub fn conf_compute_gpus_ready_state(&self) -> Result<bool> {
        let mut accepting_work = 0;
        unsafe {
            try_ffi!(sys::nvmlSystemGetConfComputeGpusReadyState(
                &raw mut accepting_work
            ))?;
        }
        Ok(accepting_work == sys::NVML_CC_ACCEPTING_CLIENT_REQUESTS_TRUE)
    }

    /// Returns Confidential Computing key rotation threshold detail.
    ///
    /// For Hopper or newer fully supported devices.
    /// Supported on Linux, Windows TCC.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the request, if the system does not
    /// support this Confidential Computing query, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn conf_compute_key_rotation_threshold(&self) -> Result<ConfComputeKeyRotationThreshold> {
        let mut info = sys::nvmlConfComputeGetKeyRotationThresholdInfo_t {
            version: struct_version::<sys::nvmlConfComputeGetKeyRotationThresholdInfo_t>(1),
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlSystemGetConfComputeKeyRotationThresholdInfo(
                &raw mut info,
            ))?;
        }
        Ok(info.into())
    }

    /// Returns Confidential Computing system settings.
    ///
    /// For Hopper or newer fully supported devices.
    /// Supported on Linux, Windows TCC.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the
    /// request layout, if NVML rejects the request, if a target GPU is
    /// inaccessible, if the system does not support Confidential Computing
    /// settings, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn conf_compute_settings(&self) -> Result<ConfComputeSystemSettings> {
        let mut settings = sys::nvmlSystemConfComputeSettings_t {
            version: struct_version::<sys::nvmlSystemConfComputeSettings_t>(1),
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlSystemGetConfComputeSettings(&raw mut settings))?;
        }
        Ok(settings.into())
    }

    /// Returns the IDs and firmware versions for any Host Interface Cards (HICs) in the system.
    ///
    /// For S-class products.
    ///
    /// This wrapper queries the required entry count internally.
    /// The HIC must be connected to an S-class system to be reported.
    ///
    /// # Errors
    ///
    /// Returns an error if the number of HIC entries changes while the wrapper
    /// is fetching them, if NVML rejects the query, or if NVML has not been
    /// initialized.
    pub fn hic_versions(&self) -> Result<Vec<HwbcEntry>> {
        let mut count = 0;
        let status = unsafe { sys::nvmlSystemGetHicVersion(&raw mut count, ptr::null_mut()) };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && count == 0 {
            return Ok(Vec::new());
        }
        if status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE {
            return Err(status.into());
        }

        let mut entries = vec![sys::nvmlHwbcEntry_t::default(); count as usize];
        unsafe {
            try_ffi!(sys::nvmlSystemGetHicVersion(
                &raw mut count,
                entries.as_mut_ptr(),
            ))?;
        }
        entries.truncate(count as usize);
        Ok(entries.into_iter().map(Into::into).collect())
    }

    /// Allocates a sample buffer to be used with NVML GPM.
    /// At least two of these buffers are required to use the NVML GPM feature.
    ///
    /// For Hopper or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the allocation request or if system
    /// memory is insufficient.
    pub fn gpm_sample(&self) -> Result<GpmSample> {
        let mut sample = ptr::null_mut();
        unsafe {
            try_ffi!(sys::nvmlGpmSampleAlloc(&raw mut sample))?;
        }
        unsafe { GpmSample::from_raw(sample) }
    }

    /// Calculate GPM metrics from two samples.
    ///
    /// For Hopper or newer fully supported devices.
    ///
    /// To retrieve metrics, allocate two sample buffers with [`Library::gpm_sample`] and store them in `metrics.sample1` and `metrics.sample2`.
    /// Next, fill each requested metric ID in `metrics.metrics[i].metric_id` and set `metrics.num_metrics` to the total number of metrics to retrieve.
    /// Then call [`Device::gpm_sample`] twice to obtain two samples of counters.
    ///
    /// The interval between these two [`Device::gpm_sample`] calls must be greater than 100 ms due to the internal sample refresh rate.
    /// Finally, call [`Library::gpm_metrics`] to retrieve the metrics into `metrics.metrics`.
    ///
    pub fn gpm_metrics(
        &self,
        sample1: &GpmSample,
        sample2: &GpmSample,
        metric_ids: &[GpmMetricId],
    ) -> Result<Vec<GpmMetric>> {
        let mut metrics_get = sys::nvmlGpmMetricsGet_t {
            version: sys::NVML_GPM_METRICS_GET_VERSION,
            numMetrics: metric_ids.len() as u32,
            sample1: sample1.0,
            sample2: sample2.0,
            ..Default::default()
        };

        for (slot, metric_id) in metrics_get.metrics.iter_mut().zip(metric_ids) {
            slot.metricId = metric_id.0;
        }

        unsafe {
            try_ffi!(sys::nvmlGpmMetricsGet(&raw mut metrics_get))?;
        }

        Ok(metrics_get.metrics[..metric_ids.len()]
            .iter()
            .copied()
            .map(Into::into)
            .collect())
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::nvmlShutdown();
        }
    }
}

impl GpmSample {
    /// Wraps an existing NVML GPM sample handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid NVML GPM sample handle. The returned value
    /// takes ownership of the handle and frees it on drop.
    pub unsafe fn from_raw(handle: sys::nvmlGpmSample_t) -> Result<Self> {
        if handle.is_null() {
            return Err(crate::error::Error::NullHandle);
        }

        Ok(Self(handle))
    }

    pub const fn as_raw(&self) -> sys::nvmlGpmSample_t {
        self.0
    }

    /// Consumes this sample and returns the owned raw NVML GPM sample handle.
    ///
    /// The caller becomes responsible for freeing the handle.
    pub fn into_raw(self) -> sys::nvmlGpmSample_t {
        let this = ManuallyDrop::new(self);
        this.0
    }
}

impl Drop for GpmSample {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::nvmlGpmSampleFree(self.0);
        }
    }
}

impl EventSet {
    /// Wraps an existing NVML event-set handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid NVML event-set handle. The returned value takes
    /// ownership of the handle and frees it on drop.
    pub unsafe fn from_raw(handle: sys::nvmlEventSet_t) -> Result<Self> {
        if handle.is_null() {
            return Err(crate::error::Error::NullHandle);
        }

        Ok(Self(handle))
    }

    /// Returns the raw NVML event-set handle.
    pub const fn as_raw(&self) -> sys::nvmlEventSet_t {
        self.0
    }

    /// Consumes this event set and returns the owned raw NVML event-set handle.
    ///
    /// The caller becomes responsible for freeing the handle.
    pub fn into_raw(self) -> sys::nvmlEventSet_t {
        let this = ManuallyDrop::new(self);
        this.0
    }

    /// Starts recording the requested events for the specified device.
    ///
    /// For Fermi or newer fully supported devices.
    /// ECC events are available only on ECC-enabled devices; power-capping events are available only on devices with power management support.
    ///
    /// For Linux only.
    ///
    /// All events that occurred before this call are not recorded.
    /// Use [`EventSet::wait`] to observe recorded events.
    ///
    /// If NVML reports [`Status::UnknownError`], the event set is in an undefined state and must be freed.
    /// If NVML reports [`Status::NotSupported`], the event set can still be
    /// used, but none of the requested event types are registered.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// event set or event mask, if the platform or requested event types are not
    /// supported, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn register_device(&self, device: Device, event_types: EventTypes) -> Result<()> {
        unsafe {
            try_ffi!(sys::nvmlDeviceRegisterEvents(
                device.as_raw(),
                event_types.bits(),
                self.0,
            ))
        }
    }

    /// Waits for and returns the next event.
    ///
    /// For Fermi or newer fully supported devices.
    ///
    /// If events are ready when this is called, it returns immediately.
    /// If no events are ready, it sleeps until an event arrives or `timeout_ms` expires.
    /// In some conditions, such as an interrupt, this can return before the timeout expires.
    ///
    /// On Windows, after an Xid error, this method returns the most recent Xid error type seen by the system.
    /// If multiple Xid errors are generated before this wait call is made, the last seen Xid error type is returned for all Xid error events.
    ///
    /// On Linux, every Xid error event returns the associated event data and
    /// other information if applicable.
    ///
    /// In MIG mode, if a device handle is provided, NVML reports events for all
    /// available instances only when the caller has appropriate privileges.
    /// Without those privileges, only events affecting all instances, namely
    /// the whole device, are reported.
    ///
    /// This does not currently support per-instance event reporting using MIG device handles.
    ///
    /// # Errors
    ///
    /// Returns an error if a registered GPU is inaccessible, if NVML rejects the
    /// event-data output, if no event arrives before `timeout_ms` or the wait is
    /// interrupted, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn wait(&self, timeout_ms: u32) -> Result<EventData> {
        unsafe {
            let mut data = MaybeUninit::<sys::nvmlEventData_t>::uninit();
            try_ffi!(sys::nvmlEventSetWait_v2(
                self.0,
                data.as_mut_ptr(),
                timeout_ms
            ))?;
            Ok(data.assume_init().into())
        }
    }
}

impl Drop for EventSet {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::nvmlEventSetFree(self.0);
        }
    }
}

impl SystemEventSet {
    /// Wraps an existing NVML system event-set handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid NVML system event-set handle. The returned
    /// value takes ownership of the handle and frees it on drop.
    pub unsafe fn from_raw(handle: sys::nvmlSystemEventSet_t) -> Result<Self> {
        if handle.is_null() {
            return Err(crate::error::Error::NullHandle);
        }

        Ok(Self(handle))
    }

    /// Returns the raw NVML system event-set handle.
    pub const fn as_raw(&self) -> sys::nvmlSystemEventSet_t {
        self.0
    }

    /// Consumes this event set and returns the owned raw NVML system event-set handle.
    ///
    /// The caller becomes responsible for freeing the handle.
    pub fn into_raw(self) -> sys::nvmlSystemEventSet_t {
        let this = ManuallyDrop::new(self);
        this.0
    }

    /// Starts recording the requested events for the system event set.
    ///
    /// For Linux only.
    ///
    /// Starts recording events on the specified device.
    /// All events that occurred before this call are not recorded.
    /// Use [`SystemEventSet::wait`] to check whether an event occurred.
    ///
    /// If NVML reports [`Status::UnknownError`], the event set is in an undefined state and must be freed.
    /// If NVML reports [`Status::NotSupported`], the event set can still be
    /// used, but none of the requested event types are registered.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the
    /// request layout, if NVML rejects the event registration request, if NVML
    /// has not been initialized, or if NVML reports an unexpected failure.
    pub fn register_events(&self, event_types: SystemEventTypes) -> Result<()> {
        let mut request = sys::nvmlSystemRegisterEventRequest_t {
            version: struct_version::<sys::nvmlSystemRegisterEventRequest_t>(1),
            eventTypes: event_types.bits(),
            set: self.0,
        };
        unsafe { try_ffi!(sys::nvmlSystemRegisterEvents(&raw mut request)) }
    }

    /// Waits for system events and returns ready events.
    ///
    /// For Fermi or newer fully supported devices.
    ///
    /// If events are ready when this is called, it returns immediately.
    /// If no events are ready, it sleeps until an event arrives or `timeout_ms` expires.
    /// In some conditions, such as an interrupt, this can return before the timeout expires.
    ///
    /// If the returned event count equals the internal event-buffer capacity, there may be outstanding events.
    /// Call [`SystemEventSet::wait`] again to query all events.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the
    /// request layout, if NVML rejects the wait request, if no event arrives
    /// before `timeout_ms`, if NVML has not been initialized, or if NVML reports
    /// an unexpected failure.
    pub fn wait(&self, timeout_ms: u32, max_events: u32) -> Result<Vec<SystemEventData>> {
        let mut data = vec![sys::nvmlSystemEventData_v1_t::default(); max_events as usize];
        let mut request = sys::nvmlSystemEventSetWaitRequest_t {
            version: struct_version::<sys::nvmlSystemEventSetWaitRequest_t>(1),
            timeoutms: timeout_ms,
            set: self.0,
            data: data.as_mut_ptr(),
            dataSize: data.len() as u32,
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlSystemEventSetWait(&raw mut request))?;
        }
        data.truncate(request.numEvent as usize);
        Ok(data.into_iter().map(Into::into).collect())
    }
}

impl Drop for SystemEventSet {
    fn drop(&mut self) {
        let mut request = sys::nvmlSystemEventSetFreeRequest_t {
            version: struct_version::<sys::nvmlSystemEventSetFreeRequest_t>(1),
            set: self.0,
        };
        unsafe {
            let _ = sys::nvmlSystemEventSetFree(&raw mut request);
        }
    }
}

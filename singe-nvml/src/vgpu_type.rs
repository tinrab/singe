use singe_core::string_from_c_chars;
use singe_nvml_sys as sys;

use crate::{
    device::Device,
    error::Result,
    try_ffi,
    types::{
        VgpuTypeBar1Info, VgpuTypeCapability, VgpuTypeDeviceId, VgpuTypeId, VgpuTypeResolution,
    },
    utility::struct_version,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct VgpuType(VgpuTypeId);

impl VgpuType {
    pub const fn from_id(id: VgpuTypeId) -> Self {
        Self(id)
    }

    pub const fn id(self) -> VgpuTypeId {
        self.0
    }

    /// Returns the class of a vGPU type.
    /// It does not exceed 64 bytes including the terminating NUL byte.
    /// This wrapper allocates the required NVML buffer internally.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal class-name buffer is too small, if
    /// NVML rejects the vGPU type ID or query arguments, or if NVML reports an
    /// unexpected failure.
    pub fn class(self) -> Result<String> {
        let mut buffer = [0i8; sys::NVML_VGPU_NAME_BUFFER_SIZE as usize];
        let mut size = buffer.len() as u32;
        unsafe {
            try_ffi!(sys::nvmlVgpuTypeGetClass(
                self.0.0,
                buffer.as_mut_ptr(),
                &raw mut size,
            ))?;
        }
        Ok(string_from_c_chars(&buffer))
    }

    /// Returns the vGPU type name.
    ///
    /// The name is an alphanumeric vGPU identifier such as `GRID M60-2Q`.
    /// It does not exceed 64 bytes including the terminating NUL byte.
    /// This wrapper allocates the required NVML buffer internally.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal name buffer is too small, if NVML
    /// rejects the vGPU type ID or query arguments, or if NVML reports an
    /// unexpected failure.
    pub fn name(self) -> Result<String> {
        let mut buffer = [0i8; sys::NVML_VGPU_NAME_BUFFER_SIZE as usize];
        let mut size = buffer.len() as u32;
        unsafe {
            try_ffi!(sys::nvmlVgpuTypeGetName(
                self.0.0,
                buffer.as_mut_ptr(),
                &raw mut size,
            ))?;
        }
        Ok(string_from_c_chars(&buffer))
    }

    /// Returns the GPU instance profile ID for this vGPU type ID.
    /// Returns a valid GPU instance profile ID for MIG-capable vGPU types, or
    /// `INVALID_GPU_INSTANCE_PROFILE_ID` otherwise.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the vGPU type ID or query arguments,
    /// if the device is not in vGPU host virtualization mode, or if NVML
    /// reports an unexpected failure.
    pub fn gpu_instance_profile_id(self) -> Result<u32> {
        let mut profile_id = 0;
        unsafe {
            try_ffi!(sys::nvmlVgpuTypeGetGpuInstanceProfileId(
                self.0.0,
                &raw mut profile_id,
            ))?;
        }
        Ok(profile_id)
    }

    /// Returns the device ID of a vGPU type.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the vGPU type ID or query arguments,
    /// if NVML has not been initialized, or if NVML reports an unexpected
    /// failure.
    pub fn device_id(self) -> Result<VgpuTypeDeviceId> {
        let mut device_id = 0;
        let mut subsystem_id = 0;
        unsafe {
            try_ffi!(sys::nvmlVgpuTypeGetDeviceID(
                self.0.0,
                &raw mut device_id,
                &raw mut subsystem_id,
            ))?;
        }
        Ok(VgpuTypeDeviceId {
            device_id,
            subsystem_id,
        })
    }

    /// Returns the vGPU framebuffer size in bytes.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the vGPU type ID or query arguments,
    /// if NVML has not been initialized, or if NVML reports an unexpected
    /// failure.
    pub fn framebuffer_size(self) -> Result<u64> {
        let mut size = 0;
        unsafe {
            try_ffi!(sys::nvmlVgpuTypeGetFramebufferSize(self.0.0, &raw mut size))?;
        }
        Ok(size)
    }

    /// Returns count of vGPU's supported display heads.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the vGPU type ID or query arguments,
    /// if NVML has not been initialized, or if NVML reports an unexpected
    /// failure.
    pub fn num_display_heads(self) -> Result<u32> {
        let mut count = 0;
        unsafe {
            try_ffi!(sys::nvmlVgpuTypeGetNumDisplayHeads(
                self.0.0,
                &raw mut count
            ))?;
        }
        Ok(count)
    }

    /// Returns vGPU display head's maximum supported resolution.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the vGPU type ID or query arguments,
    /// if NVML has not been initialized, or if NVML reports an unexpected
    /// failure.
    pub fn resolution(self, display_index: u32) -> Result<VgpuTypeResolution> {
        let mut x = 0;
        let mut y = 0;
        unsafe {
            try_ffi!(sys::nvmlVgpuTypeGetResolution(
                self.0.0,
                display_index,
                &raw mut x,
                &raw mut y,
            ))?;
        }
        Ok(VgpuTypeResolution { x, y })
    }

    /// Returns license requirements for a vGPU type.
    ///
    /// The license type and version required to run the specified vGPU type is
    /// returned as an alphanumeric string in the form `<license name>,<version>`,
    /// for example `GRID-Virtual-PC,2.0`.
    /// If a vGPU is runnable with more than one license type, the licenses are
    /// delimited by semicolons, for example
    /// `GRID-Virtual-PC,2.0;GRID-Virtual-WS,2.0;GRID-Virtual-WS-Ext,2.0`.
    ///
    /// The returned string does not exceed 128 bytes including the terminating
    /// NUL byte.
    /// This wrapper allocates the required NVML buffer internally.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal license buffer is too small, if NVML
    /// rejects the vGPU type ID or query arguments, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn license(self) -> Result<String> {
        let mut buffer = [0i8; sys::NVML_GRID_LICENSE_BUFFER_SIZE as usize];
        unsafe {
            try_ffi!(sys::nvmlVgpuTypeGetLicense(
                self.0.0,
                buffer.as_mut_ptr(),
                buffer.len() as u32,
            ))?;
        }
        Ok(string_from_c_chars(&buffer))
    }

    /// Returns the static frame-rate limit value of the vGPU type.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the vGPU type ID or query arguments,
    /// if the frame-rate limiter is disabled for this vGPU type, if NVML has
    /// not been initialized, or if NVML reports an unexpected failure.
    pub fn frame_rate_limit(self) -> Result<u32> {
        let mut limit = 0;
        unsafe {
            try_ffi!(sys::nvmlVgpuTypeGetFrameRateLimit(self.0.0, &raw mut limit))?;
        }
        Ok(limit)
    }

    /// Returns the maximum number of vGPU instances creatable on a device for this vGPU type.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the vGPU type ID or query arguments,
    /// if NVML has not been initialized, or if NVML reports an unexpected
    /// failure.
    pub fn max_instances(self, device: Device) -> Result<u32> {
        let mut count = 0;
        unsafe {
            try_ffi!(sys::nvmlVgpuTypeGetMaxInstances(
                device.as_raw(),
                self.0.0,
                &raw mut count,
            ))?;
        }
        Ok(count)
    }

    /// Returns the maximum number of vGPU instances supported per VM for this vGPU type.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the vGPU type ID or query arguments,
    /// if NVML has not been initialized, or if NVML reports an unexpected
    /// failure.
    pub fn max_instances_per_vm(self) -> Result<u32> {
        let mut count = 0;
        unsafe {
            try_ffi!(sys::nvmlVgpuTypeGetMaxInstancesPerVm(
                self.0.0,
                &raw mut count,
            ))?;
        }
        Ok(count)
    }

    /// Returns the maximum number of vGPU instances per GPU instance for this vGPU type.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the versioned request, if the vGPU
    /// type ID or query arguments are invalid, if the host, GPU, or vGPU type
    /// does not support the query, if NVML has not been initialized, or if
    /// NVML reports an unexpected failure.
    pub fn max_instances_per_gpu_instance(self) -> Result<u32> {
        let mut max_instance = sys::nvmlVgpuTypeMaxInstance_t {
            version: struct_version::<sys::nvmlVgpuTypeMaxInstance_t>(1),
            vgpuTypeId: self.0.0,
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlVgpuTypeGetMaxInstancesPerGpuInstance(
                &raw mut max_instance
            ))?;
        }
        Ok(max_instance.maxInstancePerGI)
    }

    /// Returns the BAR1 info for this vGPU type.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the vGPU type ID or query arguments,
    /// if NVML has not been initialized, or if NVML reports an unexpected
    /// failure.
    pub fn bar1_info(self) -> Result<VgpuTypeBar1Info> {
        let mut info = sys::nvmlVgpuTypeBar1Info_t {
            version: 1 | ((size_of::<sys::nvmlVgpuTypeBar1Info_t>() as u32) << 16),
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlVgpuTypeGetBAR1Info(self.0.0, &raw mut info))?;
        }
        Ok(info.into())
    }

    /// Returns the requested capability for the given vGPU type.
    /// See [`VgpuTypeCapability`] for the supported capabilities.
    /// Returns a boolean indicating whether the capability is supported.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the vGPU type ID, capability, or query
    /// arguments, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn capability(self, capability: VgpuTypeCapability) -> Result<bool> {
        let mut result = 0;
        unsafe {
            try_ffi!(sys::nvmlVgpuTypeGetCapabilities(
                self.0.0,
                capability.into(),
                &raw mut result,
            ))?;
        }
        Ok(result != 0)
    }

    /// Returns the static framebuffer reservation of the vGPU type in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the vGPU type ID or query arguments,
    /// if NVML has not been initialized, or if NVML reports an unexpected
    /// failure.
    pub fn fb_reservation(self) -> Result<u64> {
        let mut reservation = 0;
        unsafe {
            try_ffi!(sys::nvmlVgpuTypeGetFbReservation(
                self.0.0,
                &raw mut reservation,
            ))?;
        }
        Ok(reservation)
    }

    /// Returns the static GSP heap size of the vGPU type in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the vGPU type ID or query arguments,
    /// if NVML has not been initialized, or if NVML reports an unexpected
    /// failure.
    pub fn gsp_heap_size(self) -> Result<u64> {
        let mut size = 0;
        unsafe {
            try_ffi!(sys::nvmlVgpuTypeGetGspHeapSize(self.0.0, &raw mut size))?;
        }
        Ok(size)
    }
}

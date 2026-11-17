use std::hash::{Hash, Hasher};

use singe_core::string_from_c_chars;
use singe_nvml_sys as sys;

use crate::{
    device::Device,
    error::Result,
    library::Library,
    try_ffi,
    types::{
        VgpuTypeBar1Info, VgpuTypeCapability, VgpuTypeDeviceId, VgpuTypeId, VgpuTypeResolution,
    },
    utility::{struct_version, to_u32},
};

/// Identifies a vGPU type and provides its resource limits and capabilities.
///
/// It keeps NVML initialized.
/// Equality and hashing compare the type identifier.
#[derive(Debug, Clone)]
pub struct VgpuType {
    id: VgpuTypeId,
    library: Library,
}

impl PartialEq for VgpuType {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for VgpuType {}

impl Hash for VgpuType {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl VgpuType {
    /// Wraps an identifier and keeps `library`'s initialization alive, without checking that the type exists.
    #[must_use]
    pub fn from_id(id: VgpuTypeId, library: &Library) -> Self {
        Self {
            id,
            library: library.clone(),
        }
    }

    /// Returns the typed identifier used for native NVML calls.
    #[must_use]
    pub const fn id(&self) -> VgpuTypeId {
        self.id
    }

    /// Returns the NVML initialization that this type keeps alive.
    #[must_use]
    pub const fn library(&self) -> &Library {
        &self.library
    }

    /// Returns the class of this vGPU type, at most 63 characters.
    ///
    /// Requires a Kepler or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal class-name buffer is too small, if NVML rejects the vGPU type ID or query arguments, or if NVML reports an unexpected failure.
    pub fn class(&self) -> Result<String> {
        let mut buffer = [0; sys::NVML_VGPU_NAME_BUFFER_SIZE as usize];
        let mut size = to_u32(buffer.len(), "NVML buffer length")?;
        // SAFETY: `buffer` holds the `size` writable bytes passed as its capacity, and NVML validates the type ID.
        unsafe {
            try_ffi!(sys::nvmlVgpuTypeGetClass(
                self.id.0,
                buffer.as_mut_ptr(),
                &raw mut size,
            ))?;
        }
        Ok(string_from_c_chars(&buffer))
    }

    /// Returns the vGPU type name.
    ///
    /// The name is an alphanumeric identifier of at most 63 characters, such as `GRID M60-2Q`.
    /// Requires a Kepler or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal name buffer is too small, if NVML rejects the vGPU type ID or query arguments, or if NVML reports an unexpected failure.
    pub fn name(&self) -> Result<String> {
        let mut buffer = [0; sys::NVML_VGPU_NAME_BUFFER_SIZE as usize];
        let mut size = to_u32(buffer.len(), "NVML buffer length")?;
        // SAFETY: `buffer` holds the `size` writable bytes passed as its capacity, and NVML validates the type ID.
        unsafe {
            try_ffi!(sys::nvmlVgpuTypeGetName(
                self.id.0,
                buffer.as_mut_ptr(),
                &raw mut size,
            ))?;
        }
        Ok(string_from_c_chars(&buffer))
    }

    /// Returns the GPU instance profile ID of this vGPU type.
    ///
    /// The ID is valid only for MIG-capable vGPU types; NVML reports an invalid profile ID otherwise.
    /// Requires a Kepler or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the vGPU type ID or query arguments, if the device is not in vGPU host virtualization mode, or if NVML reports an unexpected failure.
    pub fn gpu_instance_profile_id(&self) -> Result<u32> {
        let mut profile_id = 0;
        // SAFETY: NVML validates the type ID, and `profile_id` is a writable local integer.
        unsafe {
            try_ffi!(sys::nvmlVgpuTypeGetGpuInstanceProfileId(
                self.id.0,
                &raw mut profile_id,
            ))?;
        }
        Ok(profile_id)
    }

    /// Returns the PCI device and subsystem IDs of this vGPU type.
    ///
    /// Requires a Kepler or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the vGPU type ID or query arguments, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn device_id(&self) -> Result<VgpuTypeDeviceId> {
        let mut device_id = 0;
        let mut subsystem_id = 0;
        // SAFETY: NVML validates the type ID, and `device_id` and `subsystem_id` are writable local integers.
        unsafe {
            try_ffi!(sys::nvmlVgpuTypeGetDeviceID(
                self.id.0,
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
    /// Requires a Kepler or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the vGPU type ID or query arguments, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn framebuffer_size(&self) -> Result<u64> {
        let mut size = 0;
        // SAFETY: NVML validates the type ID, and `size` is a writable local integer.
        unsafe {
            try_ffi!(sys::nvmlVgpuTypeGetFramebufferSize(
                self.id.0,
                &raw mut size
            ))?;
        }
        Ok(size)
    }

    /// Returns the number of display heads this vGPU type supports.
    ///
    /// Requires a Kepler or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the vGPU type ID or query arguments, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn num_display_heads(&self) -> Result<u32> {
        let mut count = 0;
        // SAFETY: NVML validates the type ID, and `count` is a writable local integer.
        unsafe {
            try_ffi!(sys::nvmlVgpuTypeGetNumDisplayHeads(
                self.id.0,
                &raw mut count
            ))?;
        }
        Ok(count)
    }

    /// Returns the maximum resolution of display head `display_index`.
    ///
    /// Requires a Kepler or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the vGPU type ID or query arguments, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn resolution(&self, display_index: u32) -> Result<VgpuTypeResolution> {
        let mut x = 0;
        let mut y = 0;
        // SAFETY: NVML validates the type ID, and `x` and `y` are writable local integers.
        unsafe {
            try_ffi!(sys::nvmlVgpuTypeGetResolution(
                self.id.0,
                display_index,
                &raw mut x,
                &raw mut y,
            ))?;
        }
        Ok(VgpuTypeResolution { x, y })
    }

    /// Returns the licenses that can run this vGPU type.
    ///
    /// Each license is `<license name>,<version>`, such as `GRID-Virtual-PC,2.0`, and several are separated by semicolons.
    /// The string has at most 127 characters.
    /// Requires a Kepler or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal license buffer is too small, if NVML rejects the vGPU type ID or query arguments, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn license(&self) -> Result<String> {
        let mut buffer = [0; sys::NVML_GRID_LICENSE_BUFFER_SIZE as usize];
        // SAFETY: NVML validates the type ID, and `buffer` holds the `buffer.len()` writable bytes passed as its length.
        unsafe {
            try_ffi!(sys::nvmlVgpuTypeGetLicense(
                self.id.0,
                buffer.as_mut_ptr(),
                to_u32(buffer.len(), "NVML buffer length")?,
            ))?;
        }
        Ok(string_from_c_chars(&buffer))
    }

    /// Returns the static frame-rate limit of this vGPU type.
    ///
    /// Requires a Kepler or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the vGPU type ID or query arguments, if the frame-rate limiter is disabled for this vGPU type, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn frame_rate_limit(&self) -> Result<u32> {
        let mut limit = 0;
        // SAFETY: NVML validates the type ID, and `limit` is a writable local integer.
        unsafe {
            try_ffi!(sys::nvmlVgpuTypeGetFrameRateLimit(
                self.id.0,
                &raw mut limit
            ))?;
        }
        Ok(limit)
    }

    /// Returns how many vGPU instances of this type `device` can host.
    ///
    /// Requires a Kepler or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the vGPU type ID or query arguments, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn max_instances(&self, device: &Device) -> Result<u32> {
        let mut count = 0;
        // SAFETY: `device` holds a live NVML device handle, NVML validates the type ID, and `count` is a writable local integer.
        unsafe {
            try_ffi!(sys::nvmlVgpuTypeGetMaxInstances(
                device.as_raw(),
                self.id.0,
                &raw mut count,
            ))?;
        }
        Ok(count)
    }

    /// Returns the maximum number of vGPU instances supported per VM for this vGPU type.
    ///
    /// Requires a Kepler or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the vGPU type ID or query arguments, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn max_instances_per_vm(&self) -> Result<u32> {
        let mut count = 0;
        // SAFETY: NVML validates the type ID, and `count` is a writable local integer.
        unsafe {
            try_ffi!(sys::nvmlVgpuTypeGetMaxInstancesPerVm(
                self.id.0,
                &raw mut count,
            ))?;
        }
        Ok(count)
    }

    /// Returns the maximum number of vGPU instances per GPU instance for this vGPU type.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the versioned request, if the vGPU type ID or query arguments are invalid, if the host, GPU, or vGPU type does not support the query, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn max_instances_per_gpu_instance(&self) -> Result<u32> {
        let mut max_instance = sys::nvmlVgpuTypeMaxInstance_t {
            version: struct_version::<sys::nvmlVgpuTypeMaxInstance_t>(1),
            vgpuTypeId: self.id.0,
            ..Default::default()
        };
        // SAFETY: `max_instance` is a writable local struct with its version and type ID set.
        unsafe {
            try_ffi!(sys::nvmlVgpuTypeGetMaxInstancesPerGpuInstance(
                &raw mut max_instance
            ))?;
        }
        Ok(max_instance.maxInstancePerGI)
    }

    /// Returns the BAR1 info for this vGPU type.
    ///
    /// Requires a Maxwell or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the vGPU type ID or query arguments, if NVML has not been initialized, or if NVML reports an unexpected failure.
    ///
    /// # Panics
    ///
    /// Panics if the native BAR1-info record size cannot fit in its `u32` ABI version field.
    /// The fixed native record layout fits this field on supported targets.
    pub fn bar1_info(&self) -> Result<VgpuTypeBar1Info> {
        let mut info = sys::nvmlVgpuTypeBar1Info_t {
            version: 1
                | ((u32::try_from(size_of::<sys::nvmlVgpuTypeBar1Info_t>())
                    .expect("the BAR1 info fits its ABI size field"))
                    << 16),
            ..Default::default()
        };
        // SAFETY: NVML validates the type ID, and `info` is a writable local struct with its version set.
        unsafe {
            try_ffi!(sys::nvmlVgpuTypeGetBAR1Info(self.id.0, &raw mut info))?;
        }
        Ok(info.into())
    }

    /// Returns whether this vGPU type supports `capability`.
    ///
    /// Requires a Maxwell or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the vGPU type ID, capability, or query arguments, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn capability(&self, capability: VgpuTypeCapability) -> Result<bool> {
        let mut result = 0;
        // SAFETY: NVML validates the type ID, and `result` is a writable local integer.
        unsafe {
            try_ffi!(sys::nvmlVgpuTypeGetCapabilities(
                self.id.0,
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
    /// Returns an error if NVML rejects the vGPU type ID or query arguments, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn fb_reservation(&self) -> Result<u64> {
        let mut reservation = 0;
        // SAFETY: NVML validates the type ID, and `reservation` is a writable local integer.
        unsafe {
            try_ffi!(sys::nvmlVgpuTypeGetFbReservation(
                self.id.0,
                &raw mut reservation,
            ))?;
        }
        Ok(reservation)
    }

    /// Returns the static GSP heap size of the vGPU type in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the vGPU type ID or query arguments, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn gsp_heap_size(&self) -> Result<u64> {
        let mut size = 0;
        // SAFETY: NVML validates the type ID, and `size` is a writable local integer.
        unsafe {
            try_ffi!(sys::nvmlVgpuTypeGetGspHeapSize(self.id.0, &raw mut size))?;
        }
        Ok(size)
    }
}

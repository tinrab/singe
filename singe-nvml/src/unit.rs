use std::{
    hash::{Hash, Hasher},
    ptr,
};

use singe_nvml_sys as sys;

use crate::{
    device::Device,
    error::Result,
    library::Library,
    try_ffi,
    types::{
        LedState, PsuInfo, UnitFanInfo, UnitFanSpeeds, UnitInfo, UnitTemperature,
        UnitTemperatureType,
    },
    utility::to_usize,
};

/// NVML multi-GPU unit handle.
///
/// It keeps NVML initialized, and clones refer to the same native unit.
/// Equality and hashing compare the native handle.
#[derive(Debug, Clone)]
pub struct Unit {
    raw: sys::nvmlUnit_t,
    library: Library,
}

// SAFETY: NVML documents its API as thread-safe, the handle is an opaque identifier the wrapper never dereferences, and the retained `Library` is `Send`.
unsafe impl Send for Unit {}

// SAFETY: NVML documents its API as thread-safe, and every `&self` method only passes the opaque handle to NVML.
unsafe impl Sync for Unit {}

impl PartialEq for Unit {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}

impl Eq for Unit {}

impl Hash for Unit {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.raw.hash(state);
    }
}

impl Unit {
    /// Wraps an NVML unit handle and keeps `library`'s initialization alive.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `nvmlUnit_t` obtained while `library`'s initialization was alive.
    /// NVML owns the underlying unit handle; this wrapper does not destroy it.
    pub unsafe fn from_raw(handle: sys::nvmlUnit_t, library: &Library) -> Self {
        Self {
            raw: handle,
            library: library.clone(),
        }
    }

    /// Returns the borrowed native handle.
    #[must_use]
    pub const fn as_raw(&self) -> sys::nvmlUnit_t {
        self.raw
    }

    /// Returns whether the stored native handle is null.
    #[must_use]
    pub const fn is_null(&self) -> bool {
        self.raw.is_null()
    }

    /// Returns the NVML initialization that this unit keeps alive.
    #[must_use]
    pub const fn library(&self) -> &Library {
        &self.library
    }

    /// Returns the static information of this unit.
    ///
    /// # Errors
    ///
    /// Returns an error if the unit handle or output argument is invalid, or if NVML has not been initialized.
    pub fn info(&self) -> Result<UnitInfo> {
        let mut info = sys::nvmlUnitInfo_t::default();
        // SAFETY: `self.raw` is a unit handle NVML returned under `self.library`, and `info` is a writable local struct.
        unsafe {
            try_ffi!(sys::nvmlUnitGetUnitInfo(self.raw, &raw mut info))?;
        }
        Ok(info.into())
    }

    /// Returns the LED state of this unit.
    ///
    /// # Errors
    ///
    /// Returns an error if the unit handle or output argument is invalid, if this is not an S-class product, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn led_state(&self) -> Result<LedState> {
        let mut state = sys::nvmlLedState_t::default();
        // SAFETY: `self.raw` is a unit handle NVML returned under `self.library`, and `state` is a writable local struct.
        unsafe {
            try_ffi!(sys::nvmlUnitGetLedState(self.raw, &raw mut state))?;
        }
        Ok(state.into())
    }

    /// Returns the power supply readings of this unit.
    ///
    /// # Errors
    ///
    /// Returns an error if the unit handle or output argument is invalid, if this is not an S-class product, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn psu_info(&self) -> Result<PsuInfo> {
        let mut info = sys::nvmlPSUInfo_t::default();
        // SAFETY: `self.raw` is a unit handle NVML returned under `self.library`, and `info` is a writable local struct.
        unsafe {
            try_ffi!(sys::nvmlUnitGetPsuInfo(self.raw, &raw mut info))?;
        }
        Ok(info.into())
    }

    /// Returns the `kind` temperature of this unit, in degrees Celsius.
    ///
    /// Products report a subset of the intake, exhaust, and board readings.
    ///
    /// # Errors
    ///
    /// Returns an error if the unit handle, temperature type, or output argument is invalid, if this is not an S-class product, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn temperature(&self, kind: UnitTemperatureType) -> Result<UnitTemperature> {
        let mut temperature = 0;
        // SAFETY: `self.raw` is a unit handle NVML returned under `self.library`, and `temperature` is a writable local integer.
        unsafe {
            try_ffi!(sys::nvmlUnitGetTemperature(
                self.raw,
                kind.into(),
                &raw mut temperature,
            ))?;
        }
        Ok(UnitTemperature { kind, temperature })
    }

    /// Returns the fan speeds and states of this unit.
    ///
    /// # Errors
    ///
    /// Returns an error if the unit handle or output argument is invalid, if this is not an S-class product, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn fan_speeds(&self) -> Result<UnitFanSpeeds> {
        let mut fan_speeds = sys::nvmlUnitFanSpeeds_t::default();
        // SAFETY: `self.raw` is a unit handle NVML returned under `self.library`, and `fan_speeds` is a writable local struct.
        unsafe {
            try_ffi!(sys::nvmlUnitGetFanSpeedInfo(self.raw, &raw mut fan_speeds))?;
        }

        let fans = fan_speeds
            .fans
            .iter()
            .take(to_usize(fan_speeds.count, "fan count")?)
            .map(|fan| UnitFanInfo {
                speed: fan.speed,
                state: fan.state.into(),
            })
            .collect();
        Ok(UnitFanSpeeds { fans })
    }

    /// Returns the GPUs attached to this unit.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal device buffer is too small, if the unit handle or query arguments are invalid, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn devices(&self) -> Result<Vec<Device>> {
        let mut count = 0;
        // SAFETY: `self.raw` is a live unit handle, and a null array asks NVML only for the count, written to the local `count`.
        let status = unsafe { sys::nvmlUnitGetDevices(self.raw, &raw mut count, ptr::null_mut()) };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && count == 0 {
            return Ok(Vec::new());
        }

        if status != sys::nvmlReturn_t::NVML_SUCCESS
            && status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
        {
            return Err(status.into());
        }

        let mut devices = vec![ptr::null_mut(); to_usize(count, "GPU count")?];
        // SAFETY: `self.raw` is a live unit handle, and `devices` holds the `count` writable handle slots passed as its length.
        unsafe {
            try_ffi!(sys::nvmlUnitGetDevices(
                self.raw,
                &raw mut count,
                devices.as_mut_ptr()
            ))?;
        }

        devices.truncate(to_usize(count, "GPU count")?);
        Ok(devices
            .into_iter()
            // SAFETY: NVML returned this device handle under `self.library`, which each `Device` retains to keep NVML initialized.
            .map(|handle| unsafe { Device::from_raw(handle, &self.library) })
            .collect())
    }
}

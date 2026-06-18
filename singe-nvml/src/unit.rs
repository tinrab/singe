use std::{mem::MaybeUninit, ptr};

use singe_nvml_sys as sys;

use crate::{
    device::Device,
    error::Result,
    try_ffi,
    types::{
        FanState, LedState, PsuInfo, UnitFanInfo, UnitFanSpeeds, UnitInfo, UnitTemperature,
        UnitTemperatureType,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Unit(sys::nvmlUnit_t);

impl Unit {
    /// Wraps a borrowed NVML unit handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `nvmlUnit_t` for the duration of any calls made
    /// through the returned wrapper. NVML owns the underlying unit handle; this
    /// wrapper does not destroy it.
    pub const unsafe fn from_raw(handle: sys::nvmlUnit_t) -> Self {
        Self(handle)
    }

    pub const fn as_raw(self) -> sys::nvmlUnit_t {
        self.0
    }

    pub const fn is_null(self) -> bool {
        self.0.is_null()
    }

    /// Returns the static information associated with a unit.
    ///
    /// For S-class products.
    ///
    /// Returns the static unit information reported by NVML.
    ///
    /// # Errors
    ///
    /// Returns an error if the unit handle or output argument is invalid, or if
    /// NVML has not been initialized.
    pub fn info(self) -> Result<UnitInfo> {
        unsafe {
            let mut info = MaybeUninit::<sys::nvmlUnitInfo_t>::uninit();
            try_ffi!(sys::nvmlUnitGetUnitInfo(self.0, info.as_mut_ptr()))?;
            Ok(info.assume_init().into())
        }
    }

    /// Returns the LED state associated with this unit.
    ///
    /// For S-class products.
    ///
    /// Returns the LED state reported by NVML.
    ///
    /// # Errors
    ///
    /// Returns an error if the unit handle or output argument is invalid, if this
    /// is not an S-class product, if NVML has not been initialized, or if NVML
    /// reports an unexpected failure.
    pub fn led_state(self) -> Result<LedState> {
        unsafe {
            let mut state = MaybeUninit::<sys::nvmlLedState_t>::uninit();
            try_ffi!(sys::nvmlUnitGetLedState(self.0, state.as_mut_ptr()))?;
            Ok(state.assume_init().into())
        }
    }

    /// Returns the PSU stats for the unit.
    ///
    /// For S-class products.
    ///
    /// Returns the PSU information reported by NVML.
    ///
    /// # Errors
    ///
    /// Returns an error if the unit handle or output argument is invalid, if this
    /// is not an S-class product, if NVML has not been initialized, or if NVML
    /// reports an unexpected failure.
    pub fn psu_info(self) -> Result<PsuInfo> {
        unsafe {
            let mut info = MaybeUninit::<sys::nvmlPSUInfo_t>::uninit();
            try_ffi!(sys::nvmlUnitGetPsuInfo(self.0, info.as_mut_ptr()))?;
            Ok(info.assume_init().into())
        }
    }

    /// Returns the temperature readings for the unit, in degrees C.
    ///
    /// For S-class products.
    ///
    /// Depending on the product, readings may be available for intake (type=0), exhaust (type=1) and board (type=2).
    ///
    /// # Errors
    ///
    /// Returns an error if the unit handle, temperature type, or output argument is
    /// invalid, if this is not an S-class product, if NVML has not been initialized,
    /// or if NVML reports an unexpected failure.
    pub fn temperature(self, kind: UnitTemperatureType) -> Result<UnitTemperature> {
        let mut temperature = 0;
        unsafe {
            try_ffi!(sys::nvmlUnitGetTemperature(
                self.0,
                kind.into(),
                &raw mut temperature,
            ))?;
        }
        Ok(UnitTemperature { kind, temperature })
    }

    /// Returns the fan speed readings for the unit.
    ///
    /// For S-class products.
    ///
    /// Returns the fan speed information reported by NVML.
    ///
    /// # Errors
    ///
    /// Returns an error if the unit handle or output argument is invalid, if this
    /// is not an S-class product, if NVML has not been initialized, or if NVML
    /// reports an unexpected failure.
    pub fn fan_speeds(self) -> Result<UnitFanSpeeds> {
        let mut fan_speeds = sys::nvmlUnitFanSpeeds_t::default();
        unsafe {
            try_ffi!(sys::nvmlUnitGetFanSpeedInfo(self.0, &raw mut fan_speeds))?;
        }
        let fans = fan_speeds.fans[..fan_speeds.count as usize]
            .iter()
            .copied()
            .map(|fan| UnitFanInfo {
                speed: fan.speed,
                state: FanState::from(fan.state),
            })
            .collect();
        Ok(UnitFanSpeeds { fans })
    }

    /// Returns the set of GPU devices that are attached to the specified unit.
    ///
    /// For S-class products.
    ///
    /// This wrapper queries the device count internally and returns attached devices as a [`Vec`].
    ///
    /// # Errors
    ///
    /// Returns an error if the internal device buffer is too small, if the unit
    /// handle or query arguments are invalid, if NVML has not been initialized, or
    /// if NVML reports an unexpected failure.
    pub fn devices(self) -> Result<Vec<Device>> {
        let mut count = 0;
        let status = unsafe { sys::nvmlUnitGetDevices(self.0, &raw mut count, ptr::null_mut()) };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && count == 0 {
            return Ok(Vec::new());
        }
        if status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE {
            return Err(status.into());
        }

        let mut devices = vec![ptr::null_mut(); count as usize];
        unsafe {
            try_ffi!(sys::nvmlUnitGetDevices(
                self.0,
                &raw mut count,
                devices.as_mut_ptr()
            ))?;
        }
        devices.truncate(count as usize);
        Ok(devices
            .into_iter()
            .map(|handle| unsafe { Device::from_raw(handle) })
            .collect())
    }
}

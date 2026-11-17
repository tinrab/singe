use singe_nvml_sys as sys;

use crate::{
    device::Device,
    error::{Error, Result},
    try_ffi,
    types::{
        CoolerInfo, FanPolicy, FanSpeedInfo, MarginTemperature, MinMaxFanSpeed, TemperatureInfo,
        TemperatureSensor, TemperatureThreshold, ThermalSettings, try_from_nvml_enum,
    },
    utility::struct_version,
};

impl Device {
    /// Returns the selected sensor's temperature as a nonnegative integer.
    ///
    /// # Errors
    ///
    /// Propagates sensor-query errors from [`Self::temperature`].
    /// Returns [`Error::OutOfRange`] if the reported temperature is negative.
    pub fn temperature_reading(&self, sensor: TemperatureSensor) -> Result<u32> {
        let value = self.temperature(sensor)?.temperature;
        u32::try_from(value).map_err(|_| Error::out_of_range("temperature"))
    }

    /// Returns the temperature threshold for the GPU with the specified threshold type in degrees C.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// See [`TemperatureThreshold`] for details on available temperature thresholds.
    ///
    /// This is no longer the preferred interface for retrieving the following temperature thresholds on Ada and later architectures: [`TemperatureThreshold::Shutdown`], [`TemperatureThreshold::Slowdown`], [`TemperatureThreshold::MemoryMax`] and [`TemperatureThreshold::GpuMax`].
    ///
    /// Support for reading these temperature thresholds for Ada and later architectures may be removed in future releases.
    /// Use [`Device::field_values`] with the `TEMPERATURE_*` constants of [`FieldId`](crate::types::FieldId) to retrieve temperature thresholds on these architectures.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle, threshold type, or output, if the device does not support the requested temperature threshold, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn temperature_threshold(&self, threshold: TemperatureThreshold) -> Result<u32> {
        let mut temperature = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `temperature` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetTemperatureThreshold(
                self.raw,
                threshold.into(),
                &raw mut temperature,
            ))?;
        }
        Ok(temperature)
    }

    /// Sets the temperature threshold for the GPU with the specified threshold type in degrees C.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// See [`TemperatureThreshold`] for details on available temperature thresholds.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle, threshold type, or temperature value, if the device does not support setting this threshold, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn set_temperature_threshold(
        &self,
        threshold: TemperatureThreshold,
        temperature: i32,
    ) -> Result<i32> {
        let mut temperature = temperature;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `temperature` is a local NVML reads and may update.
        unsafe {
            try_ffi!(sys::nvmlDeviceSetTemperatureThreshold(
                self.raw,
                threshold.into(),
                &raw mut temperature,
            ))?;
        }
        Ok(temperature)
    }

    /// Returns the thermal margin temperature (distance to nearest slowdown threshold).
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the request layout, if the device is inaccessible, if NVML rejects the handle or output, if the platform does not support this query, or if NVML reports an unexpected failure.
    pub fn margin_temperature(&self) -> Result<MarginTemperature> {
        let mut margin = sys::nvmlMarginTemperature_t {
            version: struct_version::<sys::nvmlMarginTemperature_t>(1),
            ..Default::default()
        };
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `margin` is a writable local whose version matches its layout.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetMarginTemperature(
                self.raw,
                &raw mut margin
            ))?;
        }
        Ok(margin.into())
    }

    /// Returns the thermal sensor readings and limits for `sensor_index`.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle, sensor index, or output, if the device does not support thermal settings, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn thermal_settings(&self, sensor_index: u32) -> Result<ThermalSettings> {
        let mut settings = sys::nvmlGpuThermalSettings_t::default();
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `settings` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetThermalSettings(
                self.raw,
                sensor_index,
                &raw mut settings,
            ))?;
        }
        Ok(settings.into())
    }

    /// Returns the number of fans on the device.
    ///
    /// For all discrete products with dedicated fans.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or output, if the device does not have a fan, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn num_fans(&self) -> Result<u32> {
        let mut fans = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `fans` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetNumFans(self.raw, &raw mut fans))?;
        }
        Ok(fans)
    }

    /// Returns the intended operating speed of the device's specified fan.
    ///
    /// The reported speed is the intended fan speed.
    /// If the fan is physically blocked and unable to spin, the output does not match the actual fan speed.
    ///
    /// For all discrete products with dedicated fans.
    ///
    /// The fan speed is expressed as a percentage of the product's maximum noise tolerance fan speed.
    /// This value may exceed 100% in certain cases.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle, fan index, or output, if the device does not have a supported fan interface, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn fan_speed(&self, fan: u32) -> Result<u32> {
        let mut speed = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `speed` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetFanSpeed_v2(self.raw, fan, &raw mut speed))?;
        }
        Ok(speed)
    }

    /// Returns the intended operating speed in rotations per minute (RPM) of the device's specified fan.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// For all discrete products with dedicated fans.
    ///
    /// The reported speed is the intended fan speed.
    /// If the fan is physically blocked and unable to spin, the output does not match the actual fan speed.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the request layout, if NVML rejects the handle, fan index, or output, if the device does not support fan RPM reporting, or if NVML has not been initialized.
    pub fn fan_speed_rpm(&self, fan: u32) -> Result<FanSpeedInfo> {
        let mut info = sys::nvmlFanSpeedInfo_t {
            version: struct_version::<sys::nvmlFanSpeedInfo_t>(1),
            fan,
            ..Default::default()
        };
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `info` is a writable local whose version matches its layout.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetFanSpeedRPM(self.raw, &raw mut info))?;
        }
        Ok(FanSpeedInfo {
            fan: info.fan,
            speed: info.speed,
        })
    }

    /// Returns the current control policy of fan `fan`.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// For all cuda-capable discrete products with fans.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle, fan index, or output, if the device does not support fan policies, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn fan_control_policy(&self, fan: u32) -> Result<FanPolicy> {
        let mut policy = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `policy` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetFanControlPolicy_v2(
                self.raw,
                fan,
                &raw mut policy
            ))?;
        }
        try_from_nvml_enum("fan policy", policy)
    }

    /// Restores the default fan control policy of fan `fan`.
    ///
    /// For all cuda-capable discrete products with fans.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or fan index, if the device does not support fan-speed control, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn set_default_fan_speed(&self, fan: u32) -> Result<()> {
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and the fan index is passed by value.
        unsafe {
            try_ffi!(sys::nvmlDeviceSetDefaultFanSpeed_v2(self.raw, fan))?;
        }
        Ok(())
    }

    /// Sets the control policy of fan `fan`.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// Requires privileged access.
    ///
    /// For all cuda-capable discrete products with fans.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle, fan index, or policy, if the device does not support fan policies, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn set_fan_control_policy(&self, fan: u32, policy: FanPolicy) -> Result<()> {
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and the fan index and policy are passed by value.
        unsafe {
            try_ffi!(sys::nvmlDeviceSetFanControlPolicy(
                self.raw,
                fan,
                policy.into()
            ))?;
        }
        Ok(())
    }

    /// Sets the speed of fan `fan`, as a percentage.
    ///
    /// Warning: this changes the fan control policy to manual.
    /// You must monitor the temperature and adjust the fan speed accordingly.
    /// Setting the fan speed too low can damage the GPU.
    /// Use [`Device::set_default_fan_speed`] to restore default control policy.
    ///
    /// For all cuda-capable discrete products with fans that are Maxwell or Newer.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML has not been initialized, if NVML rejects the handle, fan index, or speed, if the device does not support manual fan speed control, or if NVML reports an unexpected failure.
    pub fn set_fan_speed(&self, fan: u32, speed: u32) -> Result<()> {
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and the fan index and speed are passed by value.
        unsafe {
            try_ffi!(sys::nvmlDeviceSetFanSpeed_v2(self.raw, fan, speed))?;
        }
        Ok(())
    }

    /// Returns the intended target speed of the device's specified fan.
    ///
    /// Normally, the driver dynamically adjusts the fan based on the needs of the GPU.
    /// When the caller sets fan speed with [`Device::set_fan_speed`], the driver attempts to make the fan achieve that setting.
    /// The actual current speed of the fan is reported in [`Device::fan_speed`].
    ///
    /// For all discrete products with dedicated fans.
    ///
    /// The fan speed is expressed as a percentage of the product's maximum noise tolerance fan speed.
    /// This value may exceed 100% in certain cases.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle, fan index, or output, if the device does not have a supported fan interface, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn target_fan_speed(&self, fan: u32) -> Result<u32> {
        let mut speed = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `speed` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetTargetFanSpeed(
                self.raw,
                fan,
                &raw mut speed
            ))?;
        }
        Ok(speed)
    }

    /// Returns the min and max fan speed that can be set for the GPU fan.
    ///
    /// For all cuda-capable discrete products with fans.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or output, if the device does not support fan-speed range reporting, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn min_max_fan_speed(&self) -> Result<MinMaxFanSpeed> {
        let mut min = 0;
        let mut max = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `min` and `max` are writable locals.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetMinMaxFanSpeed(
                self.raw,
                &raw mut min,
                &raw mut max,
            ))?;
        }
        Ok(MinMaxFanSpeed { min, max })
    }

    /// Returns the control signal type and targets of cooler `index`.
    ///
    /// See [`CoolerControl`](crate::types::CoolerControl) for details on available signal types.
    /// Targets may be the GPU, memory, power supply, or all of these.
    /// See [`CoolerTarget`](crate::types::CoolerTarget) for details on available targets.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// For all discrete products with dedicated fans.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the request layout, if NVML rejects the handle or cooler index, if the device does not support cooler reporting, or if NVML has not been initialized.
    pub fn cooler_info(&self, index: u32) -> Result<CoolerInfo> {
        let mut info = sys::nvmlCoolerInfo_t {
            version: struct_version::<sys::nvmlCoolerInfo_t>(1),
            index,
            ..Default::default()
        };
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `info` is a writable local whose version matches its layout.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetCoolerInfo(self.raw, &raw mut info))?;
        }
        Ok(CoolerInfo {
            index: info.index,
            signal_type: info.signalType.into(),
            target: info.target.into(),
        })
    }

    /// Returns the current temperature readings (in degrees C) for the given device.
    ///
    /// For all products.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle, sensor type, or output, if the device does not have the requested sensor, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn temperature(&self, sensor: TemperatureSensor) -> Result<TemperatureInfo> {
        let mut temperature = sys::nvmlTemperature_t {
            version: struct_version::<sys::nvmlTemperature_t>(1),
            sensorType: sensor.into(),
            ..Default::default()
        };
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `temperature` is a writable local whose version matches its layout.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetTemperatureV(
                self.raw,
                &raw mut temperature
            ))?;
        }
        Ok(TemperatureInfo {
            sensor: temperature.sensorType.into(),
            temperature: temperature.temperature,
        })
    }
}

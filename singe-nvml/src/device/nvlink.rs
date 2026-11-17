use std::ptr;

use singe_nvml_sys as sys;

use crate::{
    device::Device,
    error::{Error, Result},
    try_ffi,
    types::{
        EnableState, GpuFabricInfo, NvLinkBwMode, NvLinkCapability, NvLinkErrorCounter, NvLinkInfo,
        NvLinkRemoteDeviceType, NvLinkSupportedBwModes, NvLinkVersion, PciInfo,
    },
    utility::{struct_version, to_u32, to_usize},
};

/// Requested reduced `NVLink` bandwidth mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvLinkBandwidthMode {
    /// Set to the best available bandwidth mode.
    Best,
    /// Requested bandwidth mode from the device's supported modes.
    Mode(u8),
}

/// Per-link `NVLink` telemetry sample type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvLinkTelemetryType {
    /// Raw TX flit counter for a single link.
    Transmit,
    /// Raw RX flit counter for a single link.
    Receive,
}

/// One link and metric requested in a telemetry batch.
#[derive(Debug, Clone, Copy)]
pub struct NvLinkTelemetryRequest {
    /// Link identifier whose telemetry is requested.
    pub link_id: u32,
    /// Transmit or receive telemetry to query.
    pub sample_type: NvLinkTelemetryType,
}

/// Historical samples or a per-slot NVML error.
#[derive(Debug)]
pub struct NvLinkTelemetrySamples {
    /// Request associated with these samples.
    pub request: NvLinkTelemetryRequest,
    /// Raw telemetry samples, or the error reported for this request.
    pub samples: Result<Vec<u64>>,
}

impl Device {
    /// Returns GPU fabric information, including all fabric clique assignments.
    ///
    /// For Hopper or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device does not support GPU fabric reporting.
    pub fn gpu_fabric_info(&self) -> Result<GpuFabricInfo> {
        let mut info = sys::nvmlGpuFabricInfo_v4_t::default();
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `info` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetGpuFabricInfo_v4(self.raw, &raw mut info))?;
        }
        GpuFabricInfo::from_raw(&info)
    }

    /// Returns the state of the device's `NVLink` for the specified link.
    ///
    /// For Pascal or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle, link index, or output, if the device does not support `NVLink` state reporting, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn nvlink_state(&self, link: u32) -> Result<EnableState> {
        let mut state = sys::nvmlEnableState_t::NVML_FEATURE_DISABLED;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `state` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetNvLinkState(
                self.raw,
                link,
                &raw mut state
            ))?;
        }
        Ok(state.into())
    }

    /// Returns the version of the device's `NVLink` for the specified link.
    ///
    /// For Pascal or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle, link index, or output, if the device does not support `NVLink` version reporting, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn nvlink_version(&self, link: u32) -> Result<NvLinkVersion> {
        let mut version = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `version` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetNvLinkVersion(
                self.raw,
                link,
                &raw mut version
            ))?;
        }
        Ok(NvLinkVersion::from(sys::nvmlNvlinkVersion_t(version)))
    }

    /// Returns the requested capability from the device's `NVLink` for the specified link.
    ///
    /// See [`NvLinkCapability`] for the supported capabilities.
    ///
    /// For Pascal or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle, link index, capability, or output, if the device does not support `NVLink` capability reporting, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn nvlink_capability(&self, link: u32, capability: NvLinkCapability) -> Result<bool> {
        let mut result = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `result` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetNvLinkCapability(
                self.raw,
                link,
                capability.into(),
                &raw mut result,
            ))?;
        }
        Ok(result != 0)
    }

    /// Returns the PCI information for the remote node on an `NVLink` link.
    ///
    /// `pci_subsystem_id` is not filled by this query and is indeterminate.
    ///
    /// For Pascal or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle, link index, or output, if the device does not support remote PCI reporting for `NVLink`, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn nvlink_remote_pci_info(&self, link: u32) -> Result<PciInfo> {
        let mut pci = sys::nvmlPciInfo_t::default();
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `pci` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetNvLinkRemotePciInfo_v2(
                self.raw,
                link,
                &raw mut pci,
            ))?;
        }
        Ok(pci.into())
    }

    /// Returns the `NVLink` device type of the remote device connected over the given link.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle, link index, or output, if `NVLink` is unsupported, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn nvlink_remote_device_type(&self, link: u32) -> Result<NvLinkRemoteDeviceType> {
        let mut device_type = sys::nvmlIntNvLinkDeviceType_t::NVML_NVLINK_DEVICE_TYPE_UNKNOWN;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `device_type` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetNvLinkRemoteDeviceType(
                self.raw,
                link,
                &raw mut device_type,
            ))?;
        }
        Ok(device_type.into())
    }

    /// Returns the specified error counter value.
    ///
    /// See [`NvLinkErrorCounter`] for available counters.
    ///
    /// For Pascal or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle, link index, counter, or output, if the device does not support `NVLink` error counters, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn nvlink_error_counter(&self, link: u32, counter: NvLinkErrorCounter) -> Result<u64> {
        let mut value = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `value` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetNvLinkErrorCounter(
                self.raw,
                link,
                counter.into(),
                &raw mut value,
            ))?;
        }
        Ok(value)
    }

    /// Returns the supported `NVLink` reduced bandwidth modes of the device.
    ///
    /// For Blackwell or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the request layout, if NVML rejects the handle or output, or if the device does not support `NVLink` bandwidth modes.
    pub fn nvlink_supported_bw_modes(&self) -> Result<NvLinkSupportedBwModes> {
        let mut modes = sys::nvmlNvlinkSupportedBwModes_t {
            version: struct_version::<sys::nvmlNvlinkSupportedBwModes_t>(1),
            ..Default::default()
        };
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `modes` is a writable local whose version matches its layout.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetNvlinkSupportedBwModes(
                self.raw,
                &raw mut modes
            ))?;
        }
        Ok(modes.into())
    }

    /// Returns the `NVLink` reduced bandwidth mode for the device.
    ///
    /// For Blackwell or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the request layout, if NVML rejects the handle or output, or if the device does not support `NVLink` bandwidth mode reporting.
    pub fn nvlink_bw_mode(&self) -> Result<NvLinkBwMode> {
        let mut mode = sys::nvmlNvlinkGetBwMode_t {
            version: struct_version::<sys::nvmlNvlinkGetBwMode_t>(1),
            ..Default::default()
        };
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `mode` is a writable local whose version matches its layout.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetNvlinkBwMode(self.raw, &raw mut mode))?;
        }
        Ok(mode.into())
    }

    /// Returns the `NVLink` information of this device.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the request layout, if the device is inaccessible, if NVML rejects the handle or output, if the device does not support `NVLink` info reporting, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn nvlink_info(&self) -> Result<NvLinkInfo> {
        let mut info = sys::nvmlNvLinkInfo_t {
            version: struct_version::<sys::nvmlNvLinkInfo_t>(2),
            ..Default::default()
        };
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `info` is a writable local whose version matches its layout.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetNvLinkInfo(self.raw, &raw mut info))?;
        }
        Ok(info.into())
    }

    /// Sets the `NVLink` reduced bandwidth mode of the device.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested bandwidth mode is unsupported, the device is inaccessible, or NVML denies permission to configure `NVLink` bandwidth.
    /// Also propagates NVML configuration failures.
    pub fn set_nvlink_bandwidth_mode(&self, mode: NvLinkBandwidthMode) -> Result<()> {
        let mut params = sys::nvmlNvlinkSetBwMode_t {
            version: struct_version::<sys::nvmlNvlinkSetBwMode_t>(1),
            bSetBest: u32::from(mode == NvLinkBandwidthMode::Best),
            bwMode: match mode {
                NvLinkBandwidthMode::Best => 0,
                NvLinkBandwidthMode::Mode(value) => value,
            },
        };
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `params` is a local whose version matches its layout.
        unsafe {
            try_ffi!(sys::nvmlDeviceSetNvlinkBwMode(
                self.as_raw(),
                &raw mut params
            ))
        }
    }

    /// Requests an `NVLink` Reduced Bandwidth Mode change without waiting for it, and returns the polling timeout in milliseconds.
    ///
    /// For Rubin or newer devices.
    /// Poll fabric information for the completed state to check whether the setting was applied.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML lacks this entry point or rejects the requested mode, device, or caller permissions.
    /// This reports request-submission errors; the returned timeout describes subsequent polling.
    pub fn request_nvlink_bandwidth_mode(&self, mode: NvLinkBandwidthMode) -> Result<u32> {
        let mut params = sys::nvmlNvlinkSetBwModeAsync_v1_t {
            bSetBest: u32::from(mode == NvLinkBandwidthMode::Best),
            bwMode: match mode {
                NvLinkBandwidthMode::Best => 0,
                NvLinkBandwidthMode::Mode(value) => value.into(),
            },
            ..Default::default()
        };
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `params` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceSetNvlinkBwModeAsync_v1(
                self.as_raw(),
                &raw mut params
            ))?;
        }
        Ok(params.asyncPollTimeoutMs)
    }

    /// Returns a batch of historical `NVLink` per-link telemetry samples.
    ///
    /// Samples are taken at approximately 100 ms intervals.
    /// Intended for periodic polling every roughly two seconds.
    /// For Rubin or newer devices.
    ///
    /// # Errors
    ///
    /// Returns an error if request or sample counts cannot fit in `u32`, the native entry point is unavailable, or a batch query fails.
    /// Per-link errors, including insufficient sample capacity, are retained in the corresponding returned record.
    pub fn nvlink_telemetry_samples(
        &self,
        requests: &[NvLinkTelemetryRequest],
    ) -> Result<Vec<NvLinkTelemetrySamples>> {
        let count = to_u32(requests.len(), "request count")?;
        let mut slots: Vec<_> = requests.iter().map(|request| sys::nvmlNvlinkTelemetrySample_v1_t {
            linkId: request.link_id,
            sampleType: match request.sample_type { NvLinkTelemetryType::Transmit => sys::nvmlNvlinkTelemetrySampleType_t::NVML_NVLINK_TELEMETRY_SAMPLE_TYPE_THROUGHPUT_RAW_TX.0, NvLinkTelemetryType::Receive => sys::nvmlNvlinkTelemetrySampleType_t::NVML_NVLINK_TELEMETRY_SAMPLE_TYPE_THROUGHPUT_RAW_RX.0 },
            ..Default::default()
        }).collect();
        let mut batch = sys::nvmlNvlinkTelemetrySamples_v1_t {
            telemetryCount: count,
            telemetrySamples: slots.as_mut_ptr(),
        };
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `batch` points to the `count` entries of `slots`, whose null sample arrays ask NVML only for the sample counts.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetNvLinkTelemetrySamples_v1(
                self.as_raw(),
                &raw mut batch
            ))?;
        }

        let mut buffers = slots
            .iter()
            .map(|slot| {
                Ok(if slot.nvmlReturn == sys::nvmlReturn_t::NVML_SUCCESS {
                    vec![0; to_usize(slot.sampleCount, "telemetry sample count")?]
                } else {
                    Vec::new()
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let initial_status: Vec<_> = slots.iter().map(|slot| slot.nvmlReturn).collect();
        for (slot, buffer) in slots.iter_mut().zip(&mut buffers) {
            slot.samples = if buffer.is_empty() {
                ptr::null_mut()
            } else {
                buffer.as_mut_ptr()
            };
            slot.sampleCount = to_u32(buffer.len(), "telemetry sample count")?;
        }

        batch.telemetrySamples = slots.as_mut_ptr();
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, `batch` points to the `count` entries of `slots`, and each slot's sample array is null or holds its `sampleCount` writable entries in `buffers`.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetNvLinkTelemetrySamples_v1(
                self.as_raw(),
                &raw mut batch
            ))?;
        }
        Ok(requests
            .iter()
            .copied()
            .zip(slots)
            .zip(buffers)
            .zip(initial_status)
            .map(|(((request, slot), mut samples), initial)| {
                let result = if initial != sys::nvmlReturn_t::NVML_SUCCESS {
                    Err(initial.into())
                } else if slot.nvmlReturn != sys::nvmlReturn_t::NVML_SUCCESS {
                    Err(slot.nvmlReturn.into())
                } else {
                    match usize::try_from(slot.sampleCount) {
                        Ok(count) if count <= samples.len() => {
                            samples.truncate(count);
                            Ok(samples)
                        }
                        _ => Err(Error::out_of_range("telemetry sample count")),
                    }
                };
                NvLinkTelemetrySamples {
                    request,
                    samples: result,
                }
            })
            .collect())
    }
}

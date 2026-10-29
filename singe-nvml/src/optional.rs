//! Exact-signature access to header-declared APIs absent from some library builds.
#![expect(non_snake_case, reason = "names mirror the upstream C API")]

use std::sync::OnceLock;

use libloading::Library;
use singe_nvml_sys as sys;

fn library() -> Option<&'static Library> {
    static LIBRARY: OnceLock<Option<Library>> = OnceLock::new();
    // Keep the library loaded for the lifetime of every cached function pointer.
    LIBRARY
        // SAFETY: loading NVML runs only its library initializers, which have no preconditions.
        .get_or_init(|| unsafe {
            Library::new(if cfg!(target_os = "windows") {
                "nvml.dll"
            } else {
                "libnvidia-ml.so.1"
            })
            .ok()
        })
        .as_ref()
}

macro_rules! optional_api {
    ($name:ident($($argument:ident: $ty:ty),* $(,)?) -> $result:ty) => {
        pub unsafe fn $name($($argument: $ty),*) -> $result {
            type Function = unsafe extern "C" fn($($ty),*) -> $result;
            static FUNCTION: OnceLock<Option<Function>> = OnceLock::new();
            let Some(function) = FUNCTION.get_or_init(|| {
                let library = library()?;
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { library.get::<Function>(concat!(stringify!($name), "\0").as_bytes()).ok().map(|symbol| *symbol) }
            }) else { return sys::nvmlReturn_t::NVML_ERROR_NOT_SUPPORTED; };
            // SAFETY: the caller upholds this function's safety contract.
            unsafe { function($($argument),*) }
        }
    };
}

optional_api!(
    nvmlDeviceGetAdaptiveTgpModeInfo_v1(
        device: sys::nvmlDevice_t,
        info: *mut sys::nvmlAdaptiveTgpModeInfo_v1_t,
    ) -> sys::nvmlReturn_t
);

optional_api!(
    nvmlDeviceGetBankRemapperStatus_v1(
        device: sys::nvmlDevice_t,
        pBankRemapperStatus: *mut sys::nvmlEccBankRemapperStatus_v1_t,
    ) -> sys::nvmlReturn_t
);

optional_api!(
    nvmlDeviceGetGpuFabricInfo_v4(
        device: sys::nvmlDevice_t,
        gpuFabricInfo: *mut sys::nvmlGpuFabricInfo_v4_t,
    ) -> sys::nvmlReturn_t
);

optional_api!(
    nvmlDeviceGetMemoryLimits_v1(
        device: sys::nvmlDevice_t,
        limits: *mut sys::nvmlGetMemoryLimits_v1_t,
    ) -> sys::nvmlReturn_t
);

optional_api!(
    nvmlDeviceGetNvLinkTelemetrySamples_v1(
        device: sys::nvmlDevice_t,
        samples: *mut sys::nvmlNvlinkTelemetrySamples_v1_t,
    ) -> sys::nvmlReturn_t
);

optional_api!(
    nvmlDevicePerfMetricsGetSamples_v1(
        device: sys::nvmlDevice_t,
        samples: *mut sys::nvmlPerfMetricsSamples_v1_t,
    ) -> sys::nvmlReturn_t
);

optional_api!(
    nvmlDeviceSetAdaptiveTgpMode_v1(
        device: sys::nvmlDevice_t,
        mode: sys::nvmlEnableState_t,
    ) -> sys::nvmlReturn_t
);

optional_api!(
    nvmlDeviceSetMemoryLimits_v1(
        device: sys::nvmlDevice_t,
        limits: *mut sys::nvmlSetMemoryLimits_v1_t,
    ) -> sys::nvmlReturn_t
);

optional_api!(
    nvmlDeviceSetNvlinkBwModeAsync_v1(
        device: sys::nvmlDevice_t,
        setBwModeAsync: *mut sys::nvmlNvlinkSetBwModeAsync_v1_t,
    ) -> sys::nvmlReturn_t
);

optional_api!(
    nvmlEventSetGetContextCount_v1(
        set: sys::nvmlEventSet_t,
        params: *mut sys::nvmlEventSetGetContextCount_v1_t,
    ) -> sys::nvmlReturn_t
);

optional_api!(
    nvmlEventSetGetContextData_v1(
        set: sys::nvmlEventSet_t,
        params: *mut sys::nvmlEventSetGetContextData_v1_t,
    ) -> sys::nvmlReturn_t
);

optional_api!(
    nvmlEventSetGetContextInfo_v1(
        set: sys::nvmlEventSet_t,
        params: *mut sys::nvmlEventSetGetContextInfo_v1_t,
    ) -> sys::nvmlReturn_t
);

optional_api!(
    nvmlEventSetGetGpuOperationalEventContextLegacyXid_v1(
        set: sys::nvmlEventSet_t,
        params: *mut sys::nvmlEventSetGetGpuOperationalEventContextLegacyXid_v1_t,
    ) -> sys::nvmlReturn_t
);

optional_api!(
    nvmlEventSetRegisterGpuOperationalEvents_v1(
        eventSet: sys::nvmlEventSet_t,
        config: *const sys::nvmlGpuOperationalEventConfig_v1_t,
    ) -> sys::nvmlReturn_t
);

optional_api!(
    nvmlEventSetWait_v3(
        set: sys::nvmlEventSet_t,
        params: *mut sys::nvmlEventSetWait_v3_t,
    ) -> sys::nvmlReturn_t
);

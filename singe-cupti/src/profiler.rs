use std::{
    ffi,
    ffi::{CStr, CString},
    mem::{offset_of, size_of_val},
    ptr,
    ptr::NonNull,
    slice,
    sync::{Mutex, MutexGuard},
};

use singe_cupti_sys as sys;

use crate::{
    context::Context,
    error::{Error, Result, Status},
    try_ffi,
    utility::to_usize,
};

macro_rules! params {
    ($ty:ident, $last:ident $(, $field:ident: $value:expr)* $(,)?) => {{
        let mut params = sys::$ty { $($field: $value,)* ..Default::default() };
        params.structSize = (offset_of!(sys::$ty, $last) + size_of_val(&params.$last)) as _;
        params
    }};
}

static RUNTIME: Mutex<bool> = Mutex::new(false);
static CLOCK_LOCK: Mutex<()> = Mutex::new(());

fn initialize() -> Result<()> {
    let mut initialized = RUNTIME.lock().map_err(|_| Error::LockPoisoned {
        name: "profiler runtime".into(),
    })?;
    if !*initialized {
        let mut params = params!(CUpti_Profiler_Initialize_Params, pPriv);
        // SAFETY: `cuptiProfilerInitialize` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuptiProfilerInitialize(&raw mut params))?;
        }
        *initialized = true;
    }
    drop(initialized);
    Ok(())
}

pub(crate) fn finalize_process() -> Result<()> {
    let mut initialized = RUNTIME.lock().map_err(|_| Error::LockPoisoned {
        name: "profiler runtime".into(),
    })?;
    // SAFETY: `cuptiFinalize` takes no arguments.
    unsafe {
        try_ffi!(sys::cuptiFinalize())?;
    }
    *initialized = false;
    drop(initialized);
    Ok(())
}

// Initialization is process-wide; deinitializing here could invalidate profiling
// objects owned by other CUPTI users in the same process.

/// Determines the frequency to which GPU clocks are pinned for a profiling session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClockControlMode {
    /// Lock GPU clocks to the base (rated TDP) frequency.
    Base,
    /// Lock GPU clocks to the boost frequency. No automatic fallback to base.
    Boost,
}

/// Device-global clock-control state reported by the driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClockControlStatus {
    /// The driver reports no explicit clock lock for this device.
    Unlocked,
    /// Clocks are locked to a fixed level, possibly by an external tool.
    Locked,
    /// Clocks are locked to a floor/minimum level.
    LockedToFloor,
    /// Boost behavior is enabled by clock control.
    BoostEnabled,
    /// Boost behavior is disabled by clock control.
    BoostDisabled,
}

/// A process-wide CUPTI clock lock released when this value is dropped.
///
/// The lock affects all contexts on the GPU. A pre-existing external lock remains
/// untouched. Clocks are not automatically unlocked if the process exits.
#[derive(Debug)]
pub struct ClockLock<'a> {
    _context: &'a Context,
    guard: Option<MutexGuard<'static, ()>>,
}

impl Context {
    /// Queries the device-global GPU clock-control status without modifying it.
    ///
    /// # Errors
    ///
    /// Returns an error if CUPTI cannot query device clock control or reports an unsupported status sentinel.
    pub fn clock_control_status(&self) -> Result<ClockControlStatus> {
        let mut status = sys::CUpti_ClockControl_Status::CUPTI_CLOCK_CONTROL_STATUS_UNLOCKED;
        // SAFETY: `cuptiClockControlGetStatus` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cuptiClockControlGetStatus(
                self.as_raw(),
                &raw mut status
            ))?;
        }
        match status {
            sys::CUpti_ClockControl_Status::CUPTI_CLOCK_CONTROL_STATUS_UNLOCKED => {
                Ok(ClockControlStatus::Unlocked)
            }
            sys::CUpti_ClockControl_Status::CUPTI_CLOCK_CONTROL_STATUS_LOCKED => {
                Ok(ClockControlStatus::Locked)
            }
            sys::CUpti_ClockControl_Status::CUPTI_CLOCK_CONTROL_STATUS_LOCKED_TO_FLOOR => {
                Ok(ClockControlStatus::LockedToFloor)
            }
            sys::CUpti_ClockControl_Status::CUPTI_CLOCK_CONTROL_STATUS_BOOST_ENABLED => {
                Ok(ClockControlStatus::BoostEnabled)
            }
            sys::CUpti_ClockControl_Status::CUPTI_CLOCK_CONTROL_STATUS_BOOST_DISABLED => {
                Ok(ClockControlStatus::BoostDisabled)
            }
            sys::CUpti_ClockControl_Status::CUPTI_CLOCK_CONTROL_STATUS_FORCE_INT => {
                Err(Status::InvalidParameter.into())
            }
        }
    }

    /// Locks GPU clocks to a fixed level for deterministic profiling metrics.
    ///
    /// Call before enabling a profiling session, and release after disabling it.
    /// Only one lock can be active at a time per process.
    ///
    /// # Errors
    ///
    /// Returns an error if another process-local clock lock is active or the lock mutex is poisoned, or if CUPTI rejects the context, clock mode, permissions, or device state.
    pub fn lock_clocks(&self, mode: ClockControlMode) -> Result<ClockLock<'_>> {
        let guard = CLOCK_LOCK
            .try_lock()
            .map_err(|_| Error::from(Status::InvalidOperation))?;
        let mode = match mode {
            ClockControlMode::Base => sys::CUpti_ClockControl_Mode::CUPTI_CLOCK_CONTROL_MODE_BASE,
            ClockControlMode::Boost => sys::CUpti_ClockControl_Mode::CUPTI_CLOCK_CONTROL_MODE_BOOST,
        };
        // SAFETY: `cuptiClockControlLock` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cuptiClockControlLock(self.as_raw(), mode))?;
        }
        Ok(ClockLock {
            _context: self,
            guard: Some(guard),
        })
    }
}

impl ClockLock<'_> {
    /// Releases a clock lock created by CUPTI in this process.
    ///
    /// # Errors
    ///
    /// Returns an error if CUPTI cannot release the process-wide clock lock.
    pub fn release(mut self) -> Result<()> {
        // SAFETY: `cuptiClockControlUnlock` takes no arguments.
        unsafe {
            try_ffi!(sys::cuptiClockControlUnlock())?;
        }
        self.guard.take();
        Ok(())
    }
}

impl Drop for ClockLock<'_> {
    fn drop(&mut self) {
        if self.guard.is_some() {
            // SAFETY: `cuptiClockControlUnlock` takes no arguments.
            unsafe {
                if let Err(error) = try_ffi!(sys::cuptiClockControlUnlock()) {
                    #[cfg(debug_assertions)]
                    eprintln!("failed to unlock cupti clock control: {error}");
                }
            }
        }
    }
}

/// Cache control resets supported GPU cache state before each profiling pass.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum CacheControl {
    /// Disable cache control. Passes may observe cache state from earlier passes.
    #[default]
    None,
    /// Reset all cache state supported by this CUPTI version before each pass.
    All,
    /// Reset supported L2 cache state before each pass.
    L2,
}

/// Type of profiler metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricType {
    /// A counter metric.
    Counter,
    /// A ratio metric.
    Ratio,
    /// A throughput metric.
    Throughput,
}

impl MetricType {
    fn as_raw(self) -> sys::CUpti_MetricType {
        match self {
            Self::Counter => sys::CUpti_MetricType::CUPTI_METRIC_TYPE_COUNTER,
            Self::Ratio => sys::CUpti_MetricType::CUPTI_METRIC_TYPE_RATIO,
            Self::Throughput => sys::CUpti_MetricType::CUPTI_METRIC_TYPE_THROUGHPUT,
        }
    }
}

/// Description, hardware unit and dimensional unit of a metric.
#[derive(Debug, Clone)]
pub struct MetricProperties {
    /// A short description about the metric.
    pub description: CString,
    /// Associated hardware unit for the metric.
    pub hardware_unit: CString,
    /// The dimension of the metric values.
    pub dimensional_unit: CString,
    /// The metric type (counter, ratio or throughput).
    pub metric_type: MetricType,
    /// Whether the metric is collected across the entire device.
    pub device_scope: bool,
}

/// Host object for metric discovery, configuration, and GPU-value evaluation.
#[derive(Debug)]
pub struct ProfilerHost {
    handle: NonNull<sys::CUpti_Profiler_Host_Object>,
    availability: Vec<u8>,
    chip_name: CString,
}

impl ProfilerHost {
    /// Creates and initializes a profiler host object for the context's device.
    ///
    /// The nvperf host and target libraries shipped alongside CUPTI must be
    /// discoverable by the dynamic linker.
    ///
    /// # Errors
    ///
    /// Returns an error if the profiler-runtime mutex is poisoned, native initialization, device or chip discovery, counter-availability querying, or host-object creation fails, a returned size does not fit `usize`, or a required native string or handle is null.
    pub fn create(context: &Context) -> Result<Self> {
        initialize()?;
        let mut chip = params!(CUpti_Device_GetChipName_Params, pChipName,
            deviceIndex: context.device_id()?.get().into());
        // SAFETY: `cuptiDeviceGetChipName` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuptiDeviceGetChipName(&raw mut chip))?;
        }
        let chip_name = copy_string(chip.pChipName)?;
        let mut availability = params!(CUpti_Profiler_GetCounterAvailability_Params, bAllowDeviceLevelCounters,
            ctx: context.as_raw());
        // SAFETY: `cuptiProfilerGetCounterAvailability` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuptiProfilerGetCounterAvailability(
                &raw mut availability
            ))?;
        }
        let mut image = vec![
            0;
            to_usize(
                availability.counterAvailabilityImageSize,
                "counter availability"
            )?
        ];
        availability.pCounterAvailabilityImage = image.as_mut_ptr();
        // SAFETY: `cuptiProfilerGetCounterAvailability` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuptiProfilerGetCounterAvailability(
                &raw mut availability
            ))?;
        }
        Self::from_device_info(chip_name, image)
    }

    /// Creates a profiler host object for a chip supported at the time of release.
    ///
    /// A counter availability image is required for future chip support; use
    /// [`Self::create`] to obtain that image from an actual device context.
    ///
    /// # Errors
    ///
    /// Returns an error if the profiler-runtime mutex is poisoned, CUPTI initialization fails, the chip is unsupported, native host-object creation fails, or the returned handle is null.
    pub fn create_for_chip(chip_name: &CStr) -> Result<Self> {
        initialize()?;
        Self::from_device_info(chip_name.to_owned(), Vec::new())
    }

    fn from_device_info(chip_name: CString, image: Vec<u8>) -> Result<Self> {
        let mut init = params!(CUpti_Profiler_Host_Initialize_Params, pSinglePassMetricSetName,
            profilerType: sys::CUpti_ProfilerType::CUPTI_PROFILER_TYPE_RANGE_PROFILER,
            pChipName: chip_name.as_ptr(), pCounterAvailabilityImage: if image.is_empty() { ptr::null() } else { image.as_ptr() });
        // SAFETY: `cuptiProfilerHostInitialize` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuptiProfilerHostInitialize(&raw mut init))?;
        }
        Ok(Self {
            handle: NonNull::new(init.pHostObject).ok_or(Error::NullHandle)?,
            availability: image,
            chip_name,
        })
    }

    /// Get the list of supported chips.
    ///
    /// # Errors
    ///
    /// Returns an error if profiler initialization or chip enumeration fails, the runtime mutex is poisoned, a native string count does not fit `usize`, or a required string table or string pointer is null.
    pub fn supported_chips() -> Result<Vec<CString>> {
        initialize()?;
        let mut params = params!(CUpti_Profiler_Host_GetSupportedChips_Params, ppChipNames);
        // SAFETY: `cuptiProfilerHostGetSupportedChips` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuptiProfilerHostGetSupportedChips(&raw mut params))?;
        }
        copy_strings(params.ppChipNames, params.numChips)
    }

    /// Get the maximum number of hardware metrics (metric names which do not
    /// include `_sass_`) that can be scheduled in a single pass for a chip.
    /// Practical constraints may prevent reaching this theoretical upper limit.
    ///
    /// # Errors
    ///
    /// Returns an error if CUPTI cannot query the chip and counter-availability configuration, or the native metric count does not fit `usize`.
    pub fn max_hardware_metrics_per_pass(&mut self) -> Result<usize> {
        let mut params = params!(CUpti_Profiler_Host_GetMaxNumHardwareMetricsPerPass_Params, maxMetricsPerPass,
            profilerType: sys::CUpti_ProfilerType::CUPTI_PROFILER_TYPE_RANGE_PROFILER,
            pChipName: self.chip_name.as_ptr(), pCounterAvailabilityImage: if self.availability.is_empty() { ptr::null_mut() } else { self.availability.as_mut_ptr() });
        // SAFETY: `cuptiProfilerHostGetMaxNumHardwareMetricsPerPass` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuptiProfilerHostGetMaxNumHardwareMetricsPerPass(
                &raw mut params
            ))?;
        }
        to_usize(params.maxMetricsPerPass, "hardware metric count")
    }

    /// Get the single pass metric sets defined in the metric config file.
    /// Profiling data for the metrics in a set can be collected in a single pass.
    ///
    /// # Errors
    ///
    /// Returns an error if native set enumeration fails, a returned count does not fit `usize` or exceeds the allocated table, or a required native string pointer is null.
    pub fn single_pass_sets(&self) -> Result<Vec<CString>> {
        let mut params = params!(CUpti_Profiler_Host_GetSinglePassSets_Params, ppSinglePassSets,
            pChipName: self.chip_name.as_ptr());
        // SAFETY: `cuptiProfilerHostGetSinglePassSets` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuptiProfilerHostGetSinglePassSets(&raw mut params))?;
        }
        let mut names =
            vec![ptr::null(); to_usize(params.numOfSinglePassSets, "single pass set count")?];
        if names.is_empty() {
            return Ok(Vec::new());
        }
        params.ppSinglePassSets = names.as_mut_ptr();
        // SAFETY: `cuptiProfilerHostGetSinglePassSets` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuptiProfilerHostGetSinglePassSets(&raw mut params))?;
        }
        if params.numOfSinglePassSets > names.len() as u64 {
            return Err(Status::InvalidParameter.into());
        }
        copy_strings(names.as_ptr(), params.numOfSinglePassSets)
    }

    /// Get all the metrics defined in the single pass metric set.
    /// Profiling data for the metrics in a set can be collected in a single pass.
    ///
    /// # Errors
    ///
    /// Returns an error if CUPTI rejects the set or cannot enumerate its metrics, a count, buffer size, or string offset does not fit `usize`, native results exceed the allocated buffers, or an offset or nul-terminated metric name is invalid.
    pub fn metrics_in_single_pass_set(&self, name: &CStr) -> Result<Vec<CString>> {
        let mut params = params!(CUpti_Profiler_Host_GetMetricsInSinglePassSet_Params, pMetricsBuffer,
            pChipName: self.chip_name.as_ptr(), pSinglePassSetName: name.as_ptr());
        // SAFETY: `cuptiProfilerHostGetMetricsInSinglePassSet` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuptiProfilerHostGetMetricsInSinglePassSet(
                &raw mut params
            ))?;
        }
        let mut indices = vec![0; to_usize(params.numOfMetricsInSinglePassSet, "metric count")?];
        let mut buffer = vec![0; to_usize(params.metricsBufferSize, "metrics buffer size")?];
        if indices.is_empty() {
            return Ok(Vec::new());
        }
        params.pMetricsIndicesBuffer = indices.as_mut_ptr();
        params.pMetricsBuffer = buffer.as_mut_ptr();
        // SAFETY: `cuptiProfilerHostGetMetricsInSinglePassSet` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuptiProfilerHostGetMetricsInSinglePassSet(
                &raw mut params
            ))?;
        }
        if params.numOfMetricsInSinglePassSet > indices.len() as u64
            || params.metricsBufferSize > buffer.len() as u64
        {
            return Err(Status::InvalidParameter.into());
        }
        indices.truncate(to_usize(
            params.numOfMetricsInSinglePassSet,
            "metric count",
        )?);
        indices
            .into_iter()
            .map(|offset| {
                let bytes = buffer
                    .get(to_usize(offset, "metric offset")?..)
                    .ok_or_else(|| Error::from(Status::InvalidParameter))?;
                CStr::from_bytes_until_nul(bytes)
                    .map(CStr::to_owned)
                    .map_err(|_| Status::InvalidParameter.into())
            })
            .collect()
    }

    /// Returns the chip name used to initialize the host object.
    #[must_use]
    pub fn chip_name(&self) -> &CStr {
        &self.chip_name
    }

    /// Gets all base metrics of a metric type supported by the chip.
    ///
    /// # Errors
    ///
    /// Returns an error if CUPTI rejects the metric type or cannot enumerate base metrics, a returned count does not fit `usize`, or a required string table or string pointer is null.
    pub fn base_metrics(&self, metric_type: MetricType) -> Result<Vec<CString>> {
        let mut params = params!(CUpti_Profiler_Host_GetBaseMetrics_Params, numMetrics,
            pHostObject: self.handle.as_ptr(), metricType: metric_type.as_raw());
        // SAFETY: `cuptiProfilerHostGetBaseMetrics` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuptiProfilerHostGetBaseMetrics(&raw mut params))?;
        }
        copy_strings(params.ppMetricNames, params.numMetrics)
    }

    /// Gets submetrics for a metric.
    ///
    /// # Errors
    ///
    /// Returns an error if CUPTI rejects the metric name or type or cannot enumerate submetrics, a returned count does not fit `usize`, or a required string table or string pointer is null.
    pub fn submetrics(&self, metric: &CStr, metric_type: MetricType) -> Result<Vec<CString>> {
        let mut params = params!(CUpti_Profiler_Host_GetSubMetrics_Params, ppSubMetrics,
            pHostObject: self.handle.as_ptr(), pMetricName: metric.as_ptr(), metricType: metric_type.as_raw());
        // SAFETY: `cuptiProfilerHostGetSubMetrics` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuptiProfilerHostGetSubMetrics(&raw mut params))?;
        }
        copy_strings(params.ppSubMetrics, params.numOfSubmetrics)
    }

    /// Gets the properties of a metric.
    ///
    /// # Errors
    ///
    /// Returns an error if CUPTI cannot query the metric, reports an unsupported metric-type sentinel, or returns a null property-string pointer.
    pub fn metric_properties(&self, metric: &CStr) -> Result<MetricProperties> {
        let mut params = params!(CUpti_Profiler_Host_GetMetricProperties_Params, metricCollectionScope,
            pHostObject: self.handle.as_ptr(), pMetricName: metric.as_ptr());
        // SAFETY: `cuptiProfilerHostGetMetricProperties` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuptiProfilerHostGetMetricProperties(&raw mut params))?;
        }
        let metric_type = match params.metricType {
            sys::CUpti_MetricType::CUPTI_METRIC_TYPE_COUNTER => MetricType::Counter,
            sys::CUpti_MetricType::CUPTI_METRIC_TYPE_RATIO => MetricType::Ratio,
            sys::CUpti_MetricType::CUPTI_METRIC_TYPE_THROUGHPUT => MetricType::Throughput,
            sys::CUpti_MetricType::CUPTI_METRIC_TYPE__COUNT => {
                return Err(Status::InvalidParameter.into());
            }
        };
        Ok(MetricProperties {
            description: copy_string(params.pDescription)?,
            hardware_unit: copy_string(params.pHwUnit)?,
            dimensional_unit: copy_string(params.pDimUnit)?,
            metric_type,
            device_scope: params.metricCollectionScope
                == sys::CUpti_MetricCollectionScope::CUPTI_METRIC_COLLECTION_SCOPE_DEVICE,
        })
    }

    /// Adds metrics to the configuration for range profiler metric collection.
    ///
    /// # Errors
    ///
    /// Returns an error if CUPTI rejects a metric name, the host configuration, or the requested metric combination.
    pub fn add_metrics(&mut self, metrics: &[&CStr]) -> Result<()> {
        let mut names: Vec<_> = metrics.iter().map(|name| name.as_ptr()).collect();
        let mut params = params!(CUpti_Profiler_Host_ConfigAddMetrics_Params, numMetrics,
            pHostObject: self.handle.as_ptr(), ppMetricNames: names.as_mut_ptr(), numMetrics: names.len() as _);
        // SAFETY: `cuptiProfilerHostConfigAddMetrics` receives valid output locations.
        unsafe { try_ffi!(sys::cuptiProfilerHostConfigAddMetrics(&raw mut params)) }
    }

    /// Gets the configuration image with scheduling information for metric collection.
    ///
    /// # Errors
    ///
    /// Returns an error if CUPTI cannot size or produce the configuration image, or its reported byte count does not fit `usize`.
    pub fn configuration(&self) -> Result<ProfilerConfiguration> {
        let mut size = params!(CUpti_Profiler_Host_GetConfigImageSize_Params, configImageSize,
            pHostObject: self.handle.as_ptr());
        // SAFETY: `cuptiProfilerHostGetConfigImageSize` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuptiProfilerHostGetConfigImageSize(&raw mut size))?;
        }
        let mut image = vec![0; to_usize(size.configImageSize, "configuration size")?];
        let mut params = params!(CUpti_Profiler_Host_GetConfigImage_Params, pConfigImage,
            pHostObject: self.handle.as_ptr(), configImageSize: image.len() as _, pConfigImage: image.as_mut_ptr());
        // SAFETY: `cuptiProfilerHostGetConfigImage` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuptiProfilerHostGetConfigImage(&raw mut params))?;
        }
        Ok(ProfilerConfiguration { image })
    }

    /// Evaluates collected counter data to GPU values for the requested metrics.
    ///
    /// # Errors
    ///
    /// Returns an error if CUPTI rejects the metric names, range index, host configuration, or counter image, or cannot evaluate the collected data.
    pub fn evaluate(
        &self,
        data: &CounterImage,
        range: usize,
        metrics: &[&CStr],
    ) -> Result<Vec<f64>> {
        let mut names: Vec<_> = metrics.iter().map(|name| name.as_ptr()).collect();
        let mut values = vec![0.0; names.len()];
        let mut params = params!(CUpti_Profiler_Host_EvaluateToGpuValues_Params, pMetricValues,
            pHostObject: self.handle.as_ptr(), pCounterDataImage: data.image.as_ptr(), counterDataImageSize: data.image.len() as _,
            rangeIndex: range as _, ppMetricNames: names.as_mut_ptr(), numMetrics: names.len() as _, pMetricValues: values.as_mut_ptr());
        // SAFETY: `cuptiProfilerHostEvaluateToGpuValues` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuptiProfilerHostEvaluateToGpuValues(&raw mut params))?;
        }
        Ok(values)
    }

    /// Sets device partition information for metric evaluation on partitioned devices.
    ///
    /// # Errors
    ///
    /// Returns an error if CUPTI rejects the device-partition image or cannot apply it to the host object.
    pub fn set_device_partition(&mut self, partition: &DevicePartitionInfo) -> Result<()> {
        let mut image = partition.image.clone();
        let mut params = params!(CUpti_Profiler_Host_SetDevicePartitionInfo_Params, devicePartitionInfoSize,
            pHostObject: self.handle.as_ptr(), pDevicePartitionInfo: image.as_mut_ptr(), devicePartitionInfoSize: image.len() as _);
        // SAFETY: `cuptiProfilerHostSetDevicePartitionInfo` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuptiProfilerHostSetDevicePartitionInfo(
                &raw mut params
            ))
        }
    }
}

impl Drop for ProfilerHost {
    fn drop(&mut self) {
        let mut params = params!(CUpti_Profiler_Host_Deinitialize_Params, pHostObject, pHostObject: self.handle.as_ptr());
        // SAFETY: `cuptiProfilerHostDeinitialize` receives valid output locations.
        unsafe {
            if let Err(error) = try_ffi!(sys::cuptiProfilerHostDeinitialize(&raw mut params)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to deinitialize cupti profiler host: {error}");
            }
        }
    }
}

/// Configuration image containing scheduling information for metric collection.
#[derive(Debug)]
pub struct ProfilerConfiguration {
    image: Vec<u8>,
}

impl ProfilerConfiguration {
    /// Gets the number of passes required to collect the configured metrics.
    ///
    /// # Errors
    ///
    /// Returns an error if CUPTI cannot decode the configuration image or the reported pass count does not fit `usize`.
    pub fn passes(&self) -> Result<usize> {
        let mut image = self.image.clone();
        let mut params = params!(CUpti_Profiler_Host_GetNumOfPasses_Params, numOfPasses,
            configImageSize: image.len() as _, pConfigImage: image.as_mut_ptr());
        // SAFETY: `cuptiProfilerHostGetNumOfPasses` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuptiProfilerHostGetNumOfPasses(&raw mut params))?;
        }
        to_usize(params.numOfPasses, "number of passes")
    }
}

/// Counter data image allocated and initialized for a range profiling session.
#[derive(Debug)]
pub struct CounterImage {
    image: Vec<u8>,
}

impl CounterImage {
    /// Gets the number of ranges stored in the counter data image.
    ///
    /// # Errors
    ///
    /// Returns an error if CUPTI cannot inspect the counter image or the reported range count does not fit `usize`.
    pub fn range_count(&self) -> Result<usize> {
        let mut params = params!(CUpti_RangeProfiler_GetCounterDataInfo_Params, numTotalRanges,
            pCounterDataImage: self.image.as_ptr(), counterDataImageSize: self.image.len() as _);
        // SAFETY: `cuptiRangeProfilerGetCounterDataInfo` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuptiRangeProfilerGetCounterDataInfo(&raw mut params))?;
        }
        to_usize(params.numTotalRanges, "range count")
    }

    /// Gets the range name for a range in the counter data image.
    ///
    /// # Errors
    ///
    /// Returns an error if CUPTI rejects the image, range index, or delimiter, cannot retrieve the range name, or returns a null name pointer.
    pub fn range_name(&self, range: usize, delimiter: &CStr) -> Result<CString> {
        let mut params = params!(CUpti_RangeProfiler_CounterData_GetRangeInfo_Params, rangeName,
            pCounterDataImage: self.image.as_ptr(), counterDataImageSize: self.image.len() as _, rangeIndex: range as _, rangeDelimiter: delimiter.as_ptr());
        // SAFETY: `cuptiRangeProfilerCounterDataGetRangeInfo` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuptiRangeProfilerCounterDataGetRangeInfo(
                &raw mut params
            ))?;
        }
        copy_string(params.rangeName)
    }
}

/// Device partition information used for host metric configuration and evaluation.
#[derive(Debug)]
pub struct DevicePartitionInfo {
    image: Vec<u8>,
}

/// Controls whether ranges are collected for each kernel or explicitly pushed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeMode {
    /// Each kernel forms a profiling range.
    Auto,
    /// Application-defined ranges are pushed and popped explicitly.
    User,
}

/// Replay mode used to collect metrics requiring multiple passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayMode {
    /// The application is relaunched for each pass.
    Application,
    /// CUPTI replays kernels to collect all passes.
    Kernel,
    /// The application resubmits the profiled work for each pass.
    User,
}

/// Configuration for range collection and nesting.
#[derive(Debug, Clone, Copy)]
pub struct RangeOptions {
    /// Profiling range mode.
    pub range: RangeMode,
    /// Replay mode.
    pub replay: ReplayMode,
    /// Maximum number of ranges that can be profiled in a pass.
    pub max_ranges: usize,
    /// Maximum number of range tree nodes; must be at least `max_ranges`.
    pub max_range_tree_nodes: u32,
    /// Number of nesting levels to be profiled. For auto range mode, this should be 1.
    pub nesting_levels: u16,
    /// Minimum nesting level to be profiled.
    pub min_nesting_level: u16,
    /// Pass index for the replay session.
    pub pass_index: usize,
    /// Target nesting level for the replay session.
    pub target_nesting_level: u16,
}

impl Default for RangeOptions {
    fn default() -> Self {
        Self {
            range: RangeMode::Auto,
            replay: ReplayMode::Kernel,
            max_ranges: 1024,
            max_range_tree_nodes: 1024,
            nesting_levels: 1,
            min_nesting_level: 1,
            pass_index: 0,
            target_nesting_level: 1,
        }
    }
}

/// Progress returned after stopping a replay pass.
#[derive(Debug, Clone, Copy)]
pub struct ReplayProgress {
    /// Pass index for the replay session.
    pub pass_index: usize,
    /// Target nesting level for the replay session.
    pub target_nesting_level: usize,
    /// Whether all passes are submitted to the GPU for collection.
    pub all_passes_submitted: bool,
}

/// Range profiler object associated with a CUDA context.
#[derive(Debug)]
pub struct RangeProfiler {
    handle: NonNull<sys::CUpti_RangeProfiler_Object>,
    _context: Context,
    configuration: Option<ProfilerConfiguration>,
    data: Option<CounterImage>,
    running: bool,
    enabled: bool,
}

impl RangeProfiler {
    /// Creates a range profiler object and enables range profiling on the context.
    ///
    /// # Errors
    ///
    /// Returns an error if profiler initialization fails or its runtime mutex is poisoned, CUPTI cannot enable profiling for the context and cache-control mode, or the returned handle is null.
    pub fn create(context: &Context, cache: CacheControl) -> Result<Self> {
        initialize()?;
        let cache = match cache {
            CacheControl::None => sys::CUpti_CacheControl_Mode::CUPTI_CACHE_CONTROL_MODE_NONE,
            CacheControl::All => sys::CUpti_CacheControl_Mode::CUPTI_CACHE_CONTROL_MODE_ALL,
            CacheControl::L2 => sys::CUpti_CacheControl_Mode::CUPTI_CACHE_CONTROL_MODE_L2,
        };
        let mut params = params!(CUpti_RangeProfiler_Enable_Params, cacheControlMode, ctx: context.as_raw(), cacheControlMode: cache);
        // SAFETY: `cuptiRangeProfilerEnable` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuptiRangeProfilerEnable(&raw mut params))?;
        }
        Ok(Self {
            handle: NonNull::new(params.pRangeProfilerObject).ok_or(Error::NullHandle)?,
            _context: context.clone(),
            configuration: None,
            data: None,
            running: false,
            enabled: true,
        })
    }

    /// Sets range/replay configuration and initializes counter data for the metrics.
    ///
    /// # Errors
    ///
    /// Returns an error if collection is running, CUPTI rejects the metric names, image, range limits, nesting, or replay options, counter-image sizing or initialization fails, or its byte count does not fit `usize`.
    pub fn configure(
        &mut self,
        configuration: ProfilerConfiguration,
        metrics: &[&CStr],
        options: RangeOptions,
    ) -> Result<()> {
        if self.running {
            return Err(Status::InvalidOperation.into());
        }
        let mut names: Vec<_> = metrics.iter().map(|name| name.as_ptr()).collect();
        let mut size = params!(CUpti_RangeProfiler_GetCounterDataSize_Params, counterDataSize,
            pRangeProfilerObject: self.handle.as_ptr(), pMetricNames: names.as_mut_ptr(), numMetrics: names.len() as _, maxNumOfRanges: options.max_ranges as _, maxNumRangeTreeNodes: options.max_range_tree_nodes);
        // SAFETY: `cuptiRangeProfilerGetCounterDataSize` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuptiRangeProfilerGetCounterDataSize(&raw mut size))?;
        }
        let mut data = CounterImage {
            image: vec![0; to_usize(size.counterDataSize, "counter data size")?],
        };
        let mut init = params!(CUpti_RangeProfiler_CounterDataImage_Initialize_Params, pCounterData,
            pRangeProfilerObject: self.handle.as_ptr(), counterDataSize: data.image.len() as _, pCounterData: data.image.as_mut_ptr());
        // SAFETY: `cuptiRangeProfilerCounterDataImageInitialize` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuptiRangeProfilerCounterDataImageInitialize(
                &raw mut init
            ))?;
        }
        let range = match options.range {
            RangeMode::Auto => sys::CUpti_ProfilerRange::CUPTI_AutoRange,
            RangeMode::User => sys::CUpti_ProfilerRange::CUPTI_UserRange,
        };
        let replay = match options.replay {
            ReplayMode::Application => sys::CUpti_ProfilerReplayMode::CUPTI_ApplicationReplay,
            ReplayMode::Kernel => sys::CUpti_ProfilerReplayMode::CUPTI_KernelReplay,
            ReplayMode::User => sys::CUpti_ProfilerReplayMode::CUPTI_UserReplay,
        };
        let mut params = params!(CUpti_RangeProfiler_SetConfig_Params, targetNestingLevel,
            pRangeProfilerObject: self.handle.as_ptr(), configSize: configuration.image.len() as _, pConfig: configuration.image.as_ptr(), counterDataImageSize: data.image.len() as _, pCounterDataImage: data.image.as_mut_ptr(), range: range, replayMode: replay, maxRangesPerPass: options.max_ranges as _, numNestingLevels: options.nesting_levels, minNestingLevel: options.min_nesting_level, passIndex: options.pass_index as _, targetNestingLevel: options.target_nesting_level);
        // SAFETY: `cuptiRangeProfilerSetConfig` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuptiRangeProfilerSetConfig(&raw mut params))?;
        }
        self.configuration = Some(configuration);
        self.data = Some(data);
        Ok(())
    }

    /// Starts the range profiler.
    ///
    /// # Errors
    ///
    /// Returns an error if collection is already running or no configuration has been installed, or if CUPTI cannot start the profiling pass.
    pub fn start(&mut self) -> Result<()> {
        if self.running || self.configuration.is_none() {
            return Err(Status::InvalidOperation.into());
        }
        let mut params = params!(CUpti_RangeProfiler_Start_Params, pRangeProfilerObject, pRangeProfilerObject: self.handle.as_ptr());
        // SAFETY: `cuptiRangeProfilerStart` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuptiRangeProfilerStart(&raw mut params))?;
        }
        self.running = true;
        Ok(())
    }

    /// Stops the range profiler and reports replay progress.
    ///
    /// # Errors
    ///
    /// Returns an error if collection is not running, CUPTI cannot stop the pass, or the returned replay progress cannot be represented as `usize`.
    pub fn stop(&mut self) -> Result<ReplayProgress> {
        if !self.running {
            return Err(Status::InvalidOperation.into());
        }
        let mut params = params!(CUpti_RangeProfiler_Stop_Params, isAllPassSubmitted, pRangeProfilerObject: self.handle.as_ptr());
        // SAFETY: `cuptiRangeProfilerStop` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuptiRangeProfilerStop(&raw mut params))?;
        }
        self.running = false;
        Ok(ReplayProgress {
            pass_index: to_usize(params.passIndex, "pass index")?,
            target_nesting_level: to_usize(params.targetNestingLevel, "target nesting level")?,
            all_passes_submitted: params.isAllPassSubmitted != 0,
        })
    }

    /// Pushes a named range onto the range stack.
    ///
    /// # Errors
    ///
    /// Returns an error if CUPTI rejects the range name, range mode, nesting depth, or current profiling state.
    pub fn push_range(&mut self, name: &CStr) -> Result<()> {
        let mut params = params!(CUpti_RangeProfiler_PushRange_Params, pRangeName, pRangeProfilerObject: self.handle.as_ptr(), pRangeName: name.as_ptr());
        // SAFETY: `cuptiRangeProfilerPushRange` receives valid output locations.
        unsafe { try_ffi!(sys::cuptiRangeProfilerPushRange(&raw mut params)) }
    }

    /// Pops the current range off the range stack.
    ///
    /// # Errors
    ///
    /// Returns an error if CUPTI cannot pop a range because the range stack, mode, or profiling state is invalid.
    pub fn pop_range(&mut self) -> Result<()> {
        let mut params = params!(CUpti_RangeProfiler_PopRange_Params, pRangeProfilerObject, pRangeProfilerObject: self.handle.as_ptr());
        // SAFETY: `cuptiRangeProfilerPopRange` receives valid output locations.
        unsafe { try_ffi!(sys::cuptiRangeProfilerPopRange(&raw mut params)) }
    }

    /// Decodes collected profiling data, returning the number of dropped ranges.
    ///
    /// # Errors
    ///
    /// Returns an error if collection is running, CUPTI cannot decode the collected data, or the dropped-range count does not fit `usize`.
    pub fn decode(&mut self) -> Result<usize> {
        if self.running {
            return Err(Status::InvalidOperation.into());
        }
        let mut params = params!(CUpti_RangeProfiler_DecodeData_Params, numOfRangeDropped, pRangeProfilerObject: self.handle.as_ptr());
        // SAFETY: `cuptiRangeProfilerDecodeData` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuptiRangeProfilerDecodeData(&raw mut params))?;
        }
        to_usize(params.numOfRangeDropped, "dropped ranges")
    }

    /// Borrows the counter data image after collection has stopped.
    ///
    /// # Errors
    ///
    /// Returns an error if collection is running or no counter image has been initialized by configuration.
    pub fn counter_data(&self) -> Result<&CounterImage> {
        if self.running {
            return Err(Status::InvalidOperation.into());
        }
        self.data
            .as_ref()
            .ok_or_else(|| Status::InvalidOperation.into())
    }

    /// Gets device partition information for the profiling context.
    ///
    /// # Errors
    ///
    /// Returns an error if CUPTI cannot size or retrieve device-partition information, or its byte count does not fit `usize`.
    pub fn device_partition_info(&self) -> Result<DevicePartitionInfo> {
        let mut params = params!(CUpti_RangeProfiler_GetDevicePartitionInfo_Params, pDevicePartitionInfo, pRangeProfilerObject: self.handle.as_ptr());
        // SAFETY: `cuptiRangeProfilerGetDevicePartitionInfo` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuptiRangeProfilerGetDevicePartitionInfo(
                &raw mut params
            ))?;
        }
        let mut image = vec![0; to_usize(params.devicePartitionInfoSize, "device partition info")?];
        params.pDevicePartitionInfo = image.as_mut_ptr();
        // SAFETY: `cuptiRangeProfilerGetDevicePartitionInfo` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuptiRangeProfilerGetDevicePartitionInfo(
                &raw mut params
            ))?;
        }
        Ok(DevicePartitionInfo { image })
    }

    /// Disables range profiling and returns the owned counter data image.
    ///
    /// # Errors
    ///
    /// Returns an error if stopping, decoding, or disabling native profiling fails, a returned progress count does not fit `usize`, or no configured counter image is available.
    pub fn finish(mut self) -> Result<CounterImage> {
        if self.running {
            self.stop()?;
        }
        self.decode()?;
        let mut params = params!(CUpti_RangeProfiler_Disable_Params, pRangeProfilerObject, pRangeProfilerObject: self.handle.as_ptr());
        // SAFETY: `cuptiRangeProfilerDisable` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuptiRangeProfilerDisable(&raw mut params))?;
        }
        self.enabled = false;
        let data = self
            .data
            .take()
            .ok_or_else(|| Error::from(Status::InvalidOperation))?;
        Ok(data)
    }
}

impl Drop for RangeProfiler {
    fn drop(&mut self) {
        if !self.enabled {
            return;
        }
        if self.running
            && let Err(error) = self.stop()
        {
            #[cfg(debug_assertions)]
            eprintln!("failed to stop cupti range profiler: {error}");
        }
        let mut params = params!(CUpti_RangeProfiler_Disable_Params, pRangeProfilerObject, pRangeProfilerObject: self.handle.as_ptr());
        // SAFETY: `cuptiRangeProfilerDisable` receives valid output locations.
        unsafe {
            if let Err(error) = try_ffi!(sys::cuptiRangeProfilerDisable(&raw mut params)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to disable cupti range profiler: {error}");
            }
        }
    }
}

fn copy_string(value: *const ffi::c_char) -> Result<CString> {
    if value.is_null() {
        return Err(Error::NullHandle);
    }
    // SAFETY: the pointer was checked for null, and the library keeps the NUL-terminated string alive while this value is borrowed.
    Ok(unsafe { CStr::from_ptr(value) }.to_owned())
}

fn copy_strings(values: *const *const ffi::c_char, count: u64) -> Result<Vec<CString>> {
    let count = to_usize(count, "string count")?;
    if count == 0 {
        return Ok(Vec::new());
    }
    if values.is_null() {
        return Err(Error::NullHandle);
    }
    // SAFETY: the library returned `count` string pointers at the non-null `values`, valid while they are copied.
    unsafe { slice::from_raw_parts(values, count) }
        .iter()
        .map(|&value| copy_string(value))
        .collect()
}

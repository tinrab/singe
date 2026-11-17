use std::{
    fmt::{self, Display, Formatter},
    mem::size_of,
    ptr,
};

use super::buffer::note_user_defined_records;
use crate::{
    callback::Subscriber,
    context::Context,
    error::{Error, Result},
    sys, try_ffi,
    types::{
        ActivityAttribute, ActivityAttributeSetting, ActivityAttributeValue, ActivityFlushFlag,
        ActivityKind, ActivityOptions, ActivityThreadIdType, ActivityUnifiedMemoryCounterConfig,
        CallbackId, ExternalCorrelationId, ExternalCorrelationKind, MultipleSubscriberState,
        StreamId,
    },
    utility::{to_u32, to_u64, to_usize},
};

/// Scope of an activity-buffer flush: the process, a context, or a stream.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum ActivityFlushTarget<'a> {
    /// Flush buffers across the process.
    Global,
    /// Flush buffers for the supplied CUDA context.
    Context(&'a Context),
    /// Flush buffers for one stream in the supplied CUDA context.
    Stream {
        /// CUDA context containing the stream to flush.
        context: &'a Context,
        /// CUPTI identifier of the stream to flush.
        stream_id: StreamId,
    },
}

impl Display for ActivityFlushTarget<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Global => write!(f, "global"),
            Self::Context(_) => write!(f, "context"),
            Self::Stream { .. } => write!(f, "stream"),
        }
    }
}

impl ActivityFlushTarget<'_> {
    fn context_and_stream_id(self) -> Result<(sys::CUcontext, u32)> {
        match self {
            Self::Global => Ok((ptr::null_mut(), 0)),
            Self::Context(context) => Ok((context.as_raw(), 0)),
            Self::Stream { context, stream_id } => {
                Ok((context.as_raw(), to_u32(stream_id.as_raw(), "stream_id")?))
            }
        }
    }
}

impl Context {
    /// Enables collection of an activity kind for this context.
    ///
    /// Context-specific activity settings override global activity settings.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
    /// - Returns [`crate::error::Status::InvalidKind`] if the activity kind is invalid or unsupported.
    /// - Returns [`crate::error::Status::NotCompatible`] if the activity kind cannot be enabled in the current configuration.
    pub fn enable_activity(&self, kind: ActivityKind) -> Result<()> {
        // SAFETY: `self.inner` retains the CUDA context for the call, and the activity kind is a valid enum value.
        unsafe {
            try_ffi!(sys::cuptiActivityEnableContext(self.as_raw(), kind.into()))?;
        }
        Ok(())
    }

    /// Disables collection of an activity kind for this context.
    ///
    /// Context-specific activity settings override global activity settings.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
    /// - Returns [`crate::error::Status::InvalidKind`] if the activity kind is invalid or unsupported.
    pub fn disable_activity(&self, kind: ActivityKind) -> Result<()> {
        // SAFETY: `self.inner` retains the CUDA context for the call, and the activity kind is a valid enum value.
        unsafe {
            try_ffi!(sys::cuptiActivityDisableContext(self.as_raw(), kind.into()))?;
        }
        Ok(())
    }

    /// Returns and resets the number of activity records dropped for a stream in this context.
    ///
    /// Dropped records indicate insufficient activity buffer space, including device-side buffers for CUDA Dynamic Parallelism records.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
    /// - Returns [`crate::error::Status::InvalidParameter`] if CUPTI rejects the request.
    pub fn num_dropped_records(&self, stream_id: StreamId) -> Result<u64> {
        let mut dropped = 0;
        let stream_id = to_u32(stream_id.as_raw(), "stream_id")?;
        // SAFETY: `self.inner` retains the CUDA context for the call, and `dropped` is a local output.
        unsafe {
            try_ffi!(sys::cuptiActivityGetNumDroppedRecords(
                self.as_raw(),
                stream_id,
                &raw mut dropped,
            ))?;
        }
        Ok(dropped)
    }
}

/// Globally enables collection of an activity kind.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidKind`] if the activity kind is invalid or unsupported.
/// - Returns [`crate::error::Status::NotCompatible`] if the activity kind cannot be enabled in the current configuration.
pub fn enable_global(kind: ActivityKind) -> Result<()> {
    // SAFETY: the call takes only the activity kind, a valid enum value.
    unsafe {
        try_ffi!(sys::cuptiActivityEnable(kind.into()))?;
    }
    Ok(())
}

/// Globally disables collection of an activity kind.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidKind`] if the activity kind is invalid or unsupported.
pub fn disable_global(kind: ActivityKind) -> Result<()> {
    // SAFETY: the call takes only the activity kind, a valid enum value.
    unsafe {
        try_ffi!(sys::cuptiActivityDisable(kind.into()))?;
    }
    Ok(())
}

/// Globally enables collection of an activity kind and asks CUPTI to emit existing resource records when supported.
///
/// Register activity buffer callbacks before calling this; otherwise CUPTI can enable the kind without delivering records.
/// Resource dumping is mainly useful for late attach to existing devices, contexts, streams, `NVLink`, and `PCIe` records.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidKind`] if the activity kind is invalid or unsupported.
/// - Returns [`crate::error::Status::NotCompatible`] if the activity kind cannot be enabled in the current configuration.
/// - Returns [`crate::error::Status::UnknownError`] if CUPTI reports an internal error.
pub fn enable_global_and_dump(kind: ActivityKind) -> Result<()> {
    // SAFETY: the call takes only the activity kind, a valid enum value.
    unsafe {
        try_ffi!(sys::cuptiActivityEnableAndDump(kind.into()))?;
    }
    Ok(())
}

/// Enables collection of an activity kind for a subscriber.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidKind`] if the activity kind is invalid or unsupported.
/// - Returns [`crate::error::Status::NotCompatible`] if the activity kind cannot be enabled in the current configuration.
pub fn enable(
    subscriber: &Subscriber,
    kind: ActivityKind,
    options: &ActivityOptions,
) -> Result<()> {
    let subscriber = subscriber.as_raw()?;
    options.validate_kind(kind)?;
    note_field_selection(options);
    // SAFETY: the borrowed subscriber keeps `subscriber` live, and `with_raw` keeps the config's field and callback ID arrays alive for the call.
    options.with_raw(|config| unsafe {
        try_ffi!(sys::cuptiActivityEnable_v2(subscriber, kind.into(), config,))?;
        Ok(())
    })
}

// A field selection shapes records only with user-defined records enabled, so it marks later buffers as possibly user-defined.
fn note_field_selection(options: &ActivityOptions) {
    if !options.fields.is_empty() {
        note_user_defined_records();
    }
}

/// Disables collection of an activity kind for a subscriber.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidKind`] if the activity kind is invalid or unsupported.
pub fn disable(
    subscriber: &Subscriber,
    kind: ActivityKind,
    options: &ActivityOptions,
) -> Result<()> {
    let subscriber = subscriber.as_raw()?;
    options.validate_kind(kind)?;
    // SAFETY: the borrowed subscriber keeps `subscriber` live, and `with_raw` keeps the config's field and callback ID arrays alive for the call.
    options.with_raw(|config| unsafe {
        try_ffi!(sys::cuptiActivityDisable_v2(
            subscriber,
            kind.into(),
            config,
        ))?;
        Ok(())
    })
}

/// Enables collection of an activity kind for a subscriber and asks CUPTI to emit existing resource records when supported.
///
/// Register activity buffer callbacks before calling this; otherwise CUPTI can enable the kind without delivering records.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidKind`] if the activity kind is invalid or unsupported.
/// - Returns [`crate::error::Status::NotCompatible`] if the activity kind cannot be enabled in the current configuration.
/// - Returns [`crate::error::Status::UnknownError`] if CUPTI reports an internal error.
pub fn enable_and_dump(
    subscriber: &Subscriber,
    kind: ActivityKind,
    options: &ActivityOptions,
) -> Result<()> {
    let subscriber = subscriber.as_raw()?;
    options.validate_kind(kind)?;
    note_field_selection(options);
    // SAFETY: the borrowed subscriber keeps `subscriber` live, and `with_raw` keeps the config's field and callback ID arrays alive for the call.
    options.with_raw(|config| unsafe {
        try_ffi!(sys::cuptiActivityEnableAndDump_v2(
            subscriber,
            kind.into(),
            config,
        ))?;
        Ok(())
    })
}

/// Returns the activity kinds enabled for `subscriber`.
///
/// # Errors
///
/// Returns an error if the subscriber handle is absent, native enumeration rejects the subscriber or fails, or a reported count cannot be represented as `usize`.
pub fn enabled_kinds(subscriber: &Subscriber) -> Result<Vec<ActivityKind>> {
    let mut buffer_size = 0u32;
    let mut enabled_count = 0u32;
    let subscriber = subscriber.as_raw()?;
    // SAFETY: the borrowed subscriber keeps `subscriber` live, the null list asks only for the count, and both counts are local outputs.
    unsafe {
        try_ffi!(sys::cuptiActivityGetEnabledKinds(
            subscriber,
            ptr::null_mut(),
            &raw mut buffer_size,
            &raw mut enabled_count,
        ))?;
    }

    if enabled_count == 0 {
        return Ok(Vec::new());
    }

    let len = to_usize(enabled_count, "enabled_activity_kinds_count")?;
    let mut kinds = vec![sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_INVALID; len];
    // CUPTI takes the buffer size in bytes, as NVIDIA's multiple-subscriber sample passes it.
    let mut buffer_size = to_u32(
        len.checked_mul(size_of::<sys::CUpti_ActivityKind>())
            .ok_or_else(|| Error::out_of_range("enabled_activity_kinds_size"))?,
        "enabled_activity_kinds_size",
    )?;
    let mut enabled_count = 0u32;
    // SAFETY: the borrowed subscriber keeps `subscriber` live, and `kinds` holds `buffer_size` writable bytes of kinds, matching the size passed.
    unsafe {
        try_ffi!(sys::cuptiActivityGetEnabledKinds(
            subscriber,
            kinds.as_mut_ptr(),
            &raw mut buffer_size,
            &raw mut enabled_count,
        ))?;
    }

    // CUPTI reports the true count, which exceeds the filled entries when more kinds were enabled since the first call.
    kinds.truncate(to_usize(enabled_count, "enabled_activity_kinds_count")?.min(len));
    kinds
        .into_iter()
        .map(|kind| Ok(ActivityKind::try_from(kind)?))
        .collect()
}

/// Returns the byte size of CUPTI's activity record struct for an activity kind and CUPTI API version.
///
/// Pass `0` for `version` to query the current runtime version.
/// Nonzero versions must use CUPTI's `xxyyzz` version format and be at least 13.0.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidKind`] if the activity kind is invalid or unsupported.
/// - Returns [`crate::error::Status::InvalidParameter`] if CUPTI rejects the request.
/// - Returns [`crate::error::Status::NotSupported`] if the requested nonzero CUPTI version is older than 13.0.
pub fn struct_size(kind: ActivityKind, version: u32) -> Result<usize> {
    let mut size = 0;
    // SAFETY: the kind and version are passed by value, and `size` is a local output.
    unsafe {
        try_ffi!(sys::cuptiActivityGetStructSize(
            kind.into(),
            version,
            &raw mut size
        ))?;
    }
    to_usize(size, "activity_struct_size")
}

/// Configures Unified Memory counters before enabling [`crate::types::ActivityKind::UnifiedMemoryCounter`].
///
/// Call this after CUDA driver initialization and before collecting Unified Memory counter activity.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidParameter`] if one of the Unified Memory counter configurations is invalid.
/// - Returns [`crate::error::Status::UnifiedMemoryProfilingNotSupported`] if the platform does not support Unified Memory counters.
/// - Returns [`crate::error::Status::UnifiedMemoryProfilingNotSupportedOnDevice`] if the device does not support Unified Memory counters.
/// - Returns [`crate::error::Status::UnifiedMemoryProfilingNotSupportedOnNonPeerToPeerDevices`] if the multi-GPU topology does not support Unified Memory counters.
pub fn configure_unified_memory_counter(
    config: &mut [ActivityUnifiedMemoryCounterConfig],
) -> Result<()> {
    let count = to_u32(config.len(), "unified_memory_counter_config_count")?;
    let mut config = config
        .iter()
        .copied()
        .map(ActivityUnifiedMemoryCounterConfig::to_raw)
        .collect::<Vec<_>>();
    // SAFETY: `config` holds `count` raw configurations, with `count` converted from its length above.
    unsafe {
        try_ffi!(sys::cuptiActivityConfigureUnifiedMemoryCounter(
            config.as_mut_ptr(),
            count,
        ))?;
    }
    Ok(())
}

/// Controls whether synchronization activity records are emitted for all CUDA event and stream queries.
///
/// When disabled, CUPTI may skip records for queries whose CUDA event or stream operation has not completed.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
pub fn set_all_sync_records_enabled(enabled: bool) -> Result<()> {
    // SAFETY: the call takes only a 0 or 1 flag.
    unsafe {
        try_ffi!(sys::cuptiActivityEnableAllSyncRecords(u8::from(enabled)))?;
    }
    Ok(())
}

/// Controls whether memory allocation activity records include the source shared library path.
///
/// This requires [`crate::types::ActivityKind::Memory2`] and adds runtime overhead.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
pub fn set_allocation_source_enabled(enabled: bool) -> Result<()> {
    // SAFETY: the call takes only a 0 or 1 flag.
    unsafe {
        try_ffi!(sys::cuptiActivityEnableAllocationSource(u8::from(enabled)))?;
    }
    Ok(())
}

/// Controls whether CUDA event activity records include device timestamps.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
pub fn set_cuda_event_device_timestamps_enabled(enabled: bool) -> Result<()> {
    // SAFETY: the call takes only a 0 or 1 flag.
    unsafe {
        try_ffi!(sys::cuptiActivityEnableCudaEventDeviceTimestamps(u8::from(
            enabled
        )))?;
    }
    Ok(())
}

/// Controls whether graph activity collection includes device graph trace records.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
pub fn set_device_graph_enabled(enabled: bool) -> Result<()> {
    // SAFETY: the call takes only a 0 or 1 flag.
    unsafe {
        try_ffi!(sys::cuptiActivityEnableDeviceGraph(u8::from(enabled)))?;
    }
    Ok(())
}

/// Enables or disables tracing for a specific CUDA driver callback ID.
///
/// The broad [`crate::types::ActivityKind::Driver`] setting can override this finer-grained setting.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
pub fn set_driver_api_enabled(callback_id: CallbackId, enabled: bool) -> Result<()> {
    // SAFETY: the call takes a callback ID and a 0 or 1 flag by value, and CUPTI checks the ID.
    unsafe {
        try_ffi!(sys::cuptiActivityEnableDriverApi(
            callback_id.as_raw(),
            u8::from(enabled)
        ))?;
    }
    Ok(())
}

/// Enables or disables tracing for a specific CUDA runtime callback ID.
///
/// The broad [`crate::types::ActivityKind::Runtime`] setting can override this finer-grained setting.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
pub fn set_runtime_api_enabled(callback_id: CallbackId, enabled: bool) -> Result<()> {
    // SAFETY: the call takes a callback ID and a 0 or 1 flag by value, and CUPTI checks the ID.
    unsafe {
        try_ffi!(sys::cuptiActivityEnableRuntimeApi(
            callback_id.as_raw(),
            u8::from(enabled)
        ))?;
    }
    Ok(())
}

/// Enables or disables hardware-event-system tracing for the profiling session.
///
/// Call this before creating CUDA contexts when possible.
/// Once enabled, hardware tracing persists for the session.
///
/// # Errors
///
/// - Returns [`crate::error::Status::CmpDeviceNotSupported`], [`crate::error::Status::ConfidentialComputingNotSupported`], [`crate::error::Status::MigDeviceNotSupported`], [`crate::error::Status::SliDeviceNotSupported`], [`crate::error::Status::VirtualizedDeviceNotSupported`], [`crate::error::Status::WslDeviceNotSupported`], or [`crate::error::Status::HesTraceNotSupportedOnMultiProcessService`] when hardware tracing is unavailable for the current environment.
/// - Returns [`crate::error::Status::NotSupported`] if hardware tracing cannot be enabled on this platform.
pub fn set_hardware_trace_enabled(enabled: bool) -> Result<()> {
    // SAFETY: the call takes only a 0 or 1 flag.
    unsafe {
        try_ffi!(sys::cuptiActivityEnableHWTrace(u8::from(enabled)))?;
    }
    Ok(())
}

/// Controls whether kernel activity records include latency timestamps.
///
/// This is not supported when hardware tracing is enabled.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
pub fn set_latency_timestamps_enabled(enabled: bool) -> Result<()> {
    // SAFETY: the call takes only a 0 or 1 flag.
    unsafe {
        try_ffi!(sys::cuptiActivityEnableLatencyTimestamps(u8::from(enabled)))?;
    }
    Ok(())
}

/// Controls whether kernel activity records include launch attributes.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
pub fn set_launch_attributes_enabled(enabled: bool) -> Result<()> {
    // SAFETY: the call takes only a 0 or 1 flag.
    unsafe {
        try_ffi!(sys::cuptiActivityEnableLaunchAttributes(u8::from(enabled)))?;
    }
    Ok(())
}

/// Flushes activity buffers for a target and delivers completed records to registered callbacks.
///
/// Use [`crate::types::ActivityFlushFlag::Forced`] before ending a profiling session to request delivery of buffers that may contain incomplete records.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidOperation`] if activity buffer callbacks have not been registered.
/// - Returns [`crate::error::Status::UnknownError`] if CUPTI reports an internal error.
pub fn flush(target: ActivityFlushTarget<'_>, flag: ActivityFlushFlag) -> Result<()> {
    let (context, stream_id) = target.context_and_stream_id()?;
    // SAFETY: `context` is null for a global flush or comes from a `Context` that `target` borrows, which retains it for the call.
    unsafe {
        try_ffi!(sys::cuptiActivityFlush(context, stream_id, flag.into()))?;
    }
    Ok(())
}

/// Flushes all activity buffers and delivers completed records to registered callbacks.
///
/// Use [`crate::types::ActivityFlushFlag::Forced`] before ending a profiling session to request delivery of buffers that may contain incomplete records.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidOperation`] if activity buffer callbacks have not been registered.
/// - Returns [`crate::error::Status::UnknownError`] if CUPTI reports an internal error.
pub fn flush_all(flag: ActivityFlushFlag) -> Result<()> {
    // SAFETY: the call takes only the flush flag, a valid enum value.
    unsafe {
        try_ffi!(sys::cuptiActivityFlushAll(flag.into()))?;
    }
    Ok(())
}

/// Sets CUPTI's periodic activity flush interval.
///
/// Passing `0` disables periodic flushing and restores CUPTI's default worker-thread heuristics.
/// Periodic flushes deliver only full buffers whose records are complete.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
pub fn flush_period(time: u32) -> Result<()> {
    // SAFETY: the call takes only the period in milliseconds.
    unsafe {
        try_ffi!(sys::cuptiActivityFlushPeriod(time))?;
    }
    Ok(())
}

fn read_attribute(attr: ActivityAttribute, value_size: *mut u64, value: *mut ()) -> Result<()> {
    // SAFETY: the only caller, `read_attribute_value`, passes a local size and a local value of `*value_size` bytes.
    unsafe {
        try_ffi!(sys::cuptiActivityGetAttribute(
            attr.into(),
            value_size,
            value.cast(),
        ))?;
    }
    Ok(())
}

fn read_attribute_v2(
    subscriber: &Subscriber,
    attr: ActivityAttribute,
    value_size: *mut u64,
    value: *mut (),
) -> Result<()> {
    let subscriber = subscriber.as_raw()?;
    // SAFETY: the borrowed subscriber keeps `subscriber` live, and `read_attribute_value` passes a local size and a local value of `*value_size` bytes.
    unsafe {
        try_ffi!(sys::cuptiActivityGetAttribute_v2(
            subscriber,
            attr.into(),
            value_size,
            value.cast(),
        ))?;
    }
    Ok(())
}

/// Reads a process-wide activity attribute into its typed value.
///
/// # Errors
///
/// Returns an error if the typed accessor does not support the attribute, the native result has an unexpected byte size or enum value, a checked size conversion fails, or CUPTI rejects or cannot read the attribute.
pub fn global_attribute_value(attr: ActivityAttribute) -> Result<ActivityAttributeValue> {
    attribute_value_with(read_attribute, attr)
}

/// Reads an activity attribute for `subscriber` into its typed value.
///
/// # Errors
///
/// Returns an error if the typed accessor does not support the attribute, the native result has an unexpected byte size or enum value, a checked size conversion fails, or CUPTI rejects or cannot read the attribute.
/// Also returns an error if the subscriber handle is absent.
pub fn attribute_value(
    subscriber: &Subscriber,
    attr: ActivityAttribute,
) -> Result<ActivityAttributeValue> {
    attribute_value_with(
        |attr, value_size, value| read_attribute_v2(subscriber, attr, value_size, value),
        attr,
    )
}

fn attribute_value_with(
    mut read: impl FnMut(ActivityAttribute, *mut u64, *mut ()) -> Result<()>,
    attr: ActivityAttribute,
) -> Result<ActivityAttributeValue> {
    #[expect(
        deprecated,
        reason = "wraps deprecated upstream API values that remain part of the supported surface"
    )]
    match attr {
        ActivityAttribute::DeviceBufferSize
        | ActivityAttribute::DeviceBufferSizeCudaDynamicParallelism
        | ActivityAttribute::ProfilingSemaphorePoolSize
        | ActivityAttribute::DeviceBufferSizeDeviceGraphs => Ok(ActivityAttributeValue::Bytes(
            read_attribute_value::<usize>(&mut read, attr)?,
        )),
        ActivityAttribute::DeviceBufferPoolLimit
        | ActivityAttribute::ProfilingSemaphorePoolLimit
        | ActivityAttribute::DeviceBufferPreAllocateValue
        | ActivityAttribute::ProfilingSemaphorePreAllocateValue => Ok(
            ActivityAttributeValue::Count(read_attribute_value::<usize>(&mut read, attr)?),
        ),
        ActivityAttribute::ZeroedOutActivityBuffer
        | ActivityAttribute::MemAllocationTypeHostPinned
        | ActivityAttribute::PerThreadActivityBuffer
        | ActivityAttribute::UserDefinedRecords
        | ActivityAttribute::EnableHes
        | ActivityAttribute::EnableAllocationSourceTracking
        | ActivityAttribute::EnableKernelLatencyTimestamps
        | ActivityAttribute::EnableAllSyncRecords
        | ActivityAttribute::EnableCudaEventDeviceTimestamps
        | ActivityAttribute::EnableKernelLaunchAttributes
        | ActivityAttribute::EnableDeviceGraphTrace
        | ActivityAttribute::EnableMultiSubscriberGraphLevelTrace
        | ActivityAttribute::CigMode => Ok(ActivityAttributeValue::Enabled(
            read_attribute_value::<u8>(&mut read, attr)? != 0,
        )),
        ActivityAttribute::MultipleSubscriberState => {
            Ok(ActivityAttributeValue::MultipleSubscriberState(
                MultipleSubscriberState::from(read_attribute_value::<u8>(&mut read, attr)?),
            ))
        }
        ActivityAttribute::ThreadIdType => {
            let raw = read_attribute_value::<u32>(&mut read, attr)?;
            let value =
                ActivityThreadIdType::try_from(raw).map_err(|_| Error::UnknownEnumValue {
                    name: "activity thread id type".into(),
                    value: raw.into(),
                })?;
            Ok(ActivityAttributeValue::ThreadIdType(value))
        }
        ActivityAttribute::TimestampCallback | ActivityAttribute::DeviceBufferForceInt => {
            Err(Error::InvalidAttribute {
                name: format!("{attr:?}").into(),
            })
        }
    }
}

fn read_attribute_value<T: Copy + Default>(
    read: &mut impl FnMut(ActivityAttribute, *mut u64, *mut ()) -> Result<()>,
    attr: ActivityAttribute,
) -> Result<T> {
    // Only plain integers are read here, so CUPTI cannot write an invalid value.
    let mut value = T::default();
    let mut value_size = to_u64(size_of::<T>(), "activity_attribute_size")?;
    read(attr, &raw mut value_size, (&raw mut value).cast())?;
    let value_size = to_usize(value_size, "activity_attribute_size")?;
    if value_size != size_of::<T>() {
        return Err(Error::SizeMismatch {
            name: "activity_attribute_size".into(),
            expected: size_of::<T>(),
            actual: value_size,
        });
    }
    Ok(value)
}

/// Returns whether a CUPTI tracing session is currently active in the process.
///
/// This query does not initialize CUPTI on its own.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidParameter`] if CUPTI rejects the request.
/// - Returns [`crate::error::Status::UnknownError`] if CUPTI reports an internal error.
pub fn is_tracing_session_running() -> Result<bool> {
    let mut running = 0u8;
    // SAFETY: `running` is a local output for the call.
    unsafe {
        try_ffi!(sys::cuptiIsTracingSessionRunning(&raw mut running))?;
    }
    Ok(running != 0)
}

/// Pops the current external correlation ID for an external API kind on the calling thread.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidParameter`] if the external correlation kind is invalid.
/// - Returns [`crate::error::Status::QueueEmpty`] if no external correlation ID is active for the kind.
pub fn pop_external_correlation_id(kind: ExternalCorrelationKind) -> Result<ExternalCorrelationId> {
    let mut last_id = 0;
    // SAFETY: the kind is a valid enum value, and `last_id` is a local output.
    unsafe {
        try_ffi!(sys::cuptiActivityPopExternalCorrelationId(
            kind.into(),
            &raw mut last_id,
        ))?;
    }
    Ok(ExternalCorrelationId::from(last_id))
}

/// Pushes an external correlation ID for the calling thread.
///
/// When external-correlation activity collection is enabled, CUPTI emits correlation records for activity generated inside the region.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidParameter`] if the external correlation kind is invalid.
pub fn push_external_correlation_id(
    kind: ExternalCorrelationKind,
    id: ExternalCorrelationId,
) -> Result<()> {
    // SAFETY: the call takes the kind, a valid enum value, and the ID by value.
    unsafe {
        try_ffi!(sys::cuptiActivityPushExternalCorrelationId(
            kind.into(),
            id.as_raw()
        ))?;
    }
    Ok(())
}

fn set_attribute(attr: ActivityAttribute, value_size: *mut u64, value: *mut ()) -> Result<()> {
    // SAFETY: the only caller, `set_attribute_value_with`, passes a local size and a local value of `*value_size` bytes.
    unsafe {
        try_ffi!(sys::cuptiActivitySetAttribute(
            attr.into(),
            value_size,
            value.cast(),
        ))?;
    }
    Ok(())
}

fn set_attribute_v2(
    subscriber: &Subscriber,
    attr: ActivityAttribute,
    value_size: *mut u64,
    value: *mut (),
) -> Result<()> {
    let subscriber = subscriber.as_raw()?;
    // SAFETY: the borrowed subscriber keeps `subscriber` live, and `set_attribute_value_with` passes a local size and a local value of `*value_size` bytes.
    unsafe {
        try_ffi!(sys::cuptiActivitySetAttribute_v2(
            subscriber,
            attr.into(),
            value_size,
            value.cast(),
        ))?;
    }
    Ok(())
}

/// Sets a process-wide activity attribute using a typed setting.
///
/// Enabling [`ActivityAttribute::UserDefinedRecords`] makes every activity record completed afterwards in this process decode to [`crate::activity::ActivityRecordPayload::Unsupported`], since its layout is chosen at run time.
///
/// # Errors
///
/// Returns an error if the setting variant does not match a supported attribute, a checked byte-size conversion fails, or CUPTI rejects the attribute or its value.
pub fn set_global_attribute_value(
    attr: ActivityAttribute,
    value: ActivityAttributeSetting,
) -> Result<()> {
    set_attribute_value_with(set_attribute, attr, value)
}

/// Sets an activity attribute for `subscriber` using a typed setting.
///
/// Enabling [`ActivityAttribute::UserDefinedRecords`] makes every activity record completed afterwards in this process decode to [`crate::activity::ActivityRecordPayload::Unsupported`], since its layout is chosen at run time.
///
/// # Errors
///
/// Returns an error if the setting variant does not match a supported attribute, a checked byte-size conversion fails, or CUPTI rejects the attribute or its value.
/// Also returns an error if the subscriber handle is absent.
pub fn set_attribute_value(
    subscriber: &Subscriber,
    attr: ActivityAttribute,
    value: ActivityAttributeSetting,
) -> Result<()> {
    set_attribute_value_with(
        |attr, value_size, value| set_attribute_v2(subscriber, attr, value_size, value),
        attr,
        value,
    )
}

fn set_attribute_value_with(
    mut write: impl FnMut(ActivityAttribute, *mut u64, *mut ()) -> Result<()>,
    attr: ActivityAttribute,
    value: ActivityAttributeSetting,
) -> Result<()> {
    if matches!(
        (attr, value),
        (
            ActivityAttribute::UserDefinedRecords,
            ActivityAttributeSetting::Enabled(true)
        )
    ) {
        note_user_defined_records();
    }

    #[expect(
        deprecated,
        reason = "wraps deprecated upstream API values that remain part of the supported surface"
    )]
    match (attr, value) {
        (
            ActivityAttribute::DeviceBufferSize
            | ActivityAttribute::DeviceBufferSizeCudaDynamicParallelism
            | ActivityAttribute::ProfilingSemaphorePoolSize
            | ActivityAttribute::DeviceBufferSizeDeviceGraphs,
            ActivityAttributeSetting::Bytes(mut value),
        )
        | (
            ActivityAttribute::DeviceBufferPoolLimit
            | ActivityAttribute::ProfilingSemaphorePoolLimit
            | ActivityAttribute::DeviceBufferPreAllocateValue
            | ActivityAttribute::ProfilingSemaphorePreAllocateValue,
            ActivityAttributeSetting::Count(mut value),
        ) => {
            let mut value_size = to_u64(size_of::<usize>(), "activity_attribute_size")?;
            write(attr, &raw mut value_size, (&raw mut value).cast())
        }
        (
            ActivityAttribute::ZeroedOutActivityBuffer
            | ActivityAttribute::MemAllocationTypeHostPinned
            | ActivityAttribute::PerThreadActivityBuffer
            | ActivityAttribute::UserDefinedRecords
            | ActivityAttribute::EnableHes
            | ActivityAttribute::EnableAllocationSourceTracking
            | ActivityAttribute::EnableKernelLatencyTimestamps
            | ActivityAttribute::EnableAllSyncRecords
            | ActivityAttribute::EnableCudaEventDeviceTimestamps
            | ActivityAttribute::EnableKernelLaunchAttributes
            | ActivityAttribute::EnableDeviceGraphTrace
            | ActivityAttribute::EnableMultiSubscriberGraphLevelTrace
            | ActivityAttribute::CigMode,
            ActivityAttributeSetting::Enabled(value),
        ) => {
            let mut value = u8::from(value);
            let mut value_size = to_u64(size_of::<u8>(), "activity_attribute_size")?;
            write(attr, &raw mut value_size, (&raw mut value).cast())
        }
        (ActivityAttribute::ThreadIdType, ActivityAttributeSetting::ThreadIdType(value)) => {
            let mut value = sys::CUpti_ActivityThreadIdType::from(value);
            let mut value_size = to_u64(
                size_of::<sys::CUpti_ActivityThreadIdType>(),
                "activity_attribute_size",
            )?;
            write(attr, &raw mut value_size, (&raw mut value).cast())
        }
        _ => Err(Error::InvalidAttribute {
            name: format!("{attr:?}").into(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_typed_attribute_setter_rejects_mismatched_value_kind_before_ffi() {
        let mut called = false;
        let error = set_attribute_value_with(
            |_, _, _| {
                called = true;
                Ok(())
            },
            ActivityAttribute::DeviceBufferSize,
            ActivityAttributeSetting::Enabled(true),
        )
        .unwrap_err();

        assert!(!called);
        assert!(matches!(error, Error::InvalidAttribute { .. }));
    }

    #[test]
    fn test_typed_attribute_setter_writes_size_t_for_byte_attributes() -> Result<()> {
        let mut observed = None;
        set_attribute_value_with(
            |attr, value_size, value| {
                // SAFETY: `set_attribute_value_with` passes a pointer to its local size.
                let value_size = unsafe { *value_size };
                // SAFETY: byte-size attributes pass a pointer to a local `usize`.
                let value = unsafe { *value.cast::<usize>() };
                observed = Some((attr, value_size, value));
                Ok(())
            },
            ActivityAttribute::DeviceBufferSize,
            ActivityAttributeSetting::Bytes(4096),
        )?;

        assert_eq!(
            observed,
            Some((
                ActivityAttribute::DeviceBufferSize,
                size_of::<usize>() as u64,
                4096,
            ))
        );
        Ok(())
    }

    #[test]
    fn test_typed_attribute_setter_writes_u8_for_bool_attributes() -> Result<()> {
        let mut observed = None;
        set_attribute_value_with(
            |attr, value_size, value| {
                // SAFETY: `set_attribute_value_with` passes a pointer to its local size.
                let value_size = unsafe { *value_size };
                // SAFETY: flag attributes pass a pointer to a local `u8`.
                let value = unsafe { *value.cast::<u8>() };
                observed = Some((attr, value_size, value));
                Ok(())
            },
            ActivityAttribute::EnableAllSyncRecords,
            ActivityAttributeSetting::Enabled(true),
        )?;

        assert_eq!(
            observed,
            Some((
                ActivityAttribute::EnableAllSyncRecords,
                size_of::<u8>() as u64,
                1,
            ))
        );
        Ok(())
    }
}

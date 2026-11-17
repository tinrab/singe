use std::{ffi::CStr, ptr};

use singe_core::string_from_c_chars;
use singe_nvml_sys as sys;

use crate::{
    error::{Error, Result},
    library::EventSet,
    try_ffi,
    types::EventTypes,
    utility::to_usize,
};

/// Event data formats returned by the event-set wait operation.
#[derive(Debug)]
pub enum EventNotification {
    /// NVML event-bit data, with historical event-bit semantics.
    Legacy(LegacyEvent),
    /// Structured GPU Operational Event data and owned context records.
    Operational(OperationalEvent),
}

/// Information about an NVML event-bit notification.
#[derive(Debug)]
pub struct LegacyEvent {
    /// UUID for the GPU where the event occurred.
    /// Empty if unavailable.
    pub uuid: String,
    /// Information about what specific event occurred.
    pub event_types: EventTypes,
    /// Xid code for a critical Xid error, or zero when not applicable.
    pub event_data: u64,
    /// MIG GPU instance ID, if applicable.
    pub gpu_instance_id: Option<u32>,
    /// MIG compute instance ID, if applicable.
    pub compute_instance_id: Option<u32>,
}

/// An owned context payload associated with a structured event.
#[derive(Debug)]
pub struct EventContext {
    /// NVML public interpretation of the context payload.
    pub kind: EventContextType,
    /// Source-defined context payload type identifier carried by the event.
    pub source_event_context_type: u32,
    /// Payload format version for the source event context type.
    pub data_format_version: u16,
    /// Raw payload bytes, excluding alignment padding.
    pub data: Vec<u8>,
    /// Legacy Xid code decoded by NVML when this payload has that interpretation.
    pub legacy_xid: Option<u32>,
}

impl EventSet {
    /// Registers structured GPU Operational Events of at least `min_log_level` and `min_severity` from the GPU with the UUID `uuid`.
    ///
    /// Registering twice has no further effect, and event-bit registrations can share the set.
    /// Requires a Turing or newer GPU on Linux.
    ///
    /// # Errors
    ///
    /// Returns an error if `uuid` with its terminator is longer than NVML's UUID buffer, the entry point is unavailable, or NVML rejects the UUID, the filters, or the caller's permissions.
    pub fn register_operational_events(
        &mut self,
        uuid: &CStr,
        min_log_level: EventLogLevel,
        min_severity: EventSeverity,
    ) -> Result<()> {
        let mut config = sys::nvmlGpuOperationalEventConfig_v1_t {
            minLogLevel: min_log_level.0,
            minSeverity: min_severity.0,
            ..Default::default()
        };
        let bytes = uuid.to_bytes_with_nul();
        if bytes.len() > config.uuid.len() {
            return Err(Error::out_of_range("uuid"));
        }

        for (target, source) in config.uuid.iter_mut().zip(bytes) {
            *target = source.cast_signed();
        }
        // SAFETY: `self` holds a live event-set handle, and `config` is a local struct whose UUID was bounds-checked above.
        unsafe {
            try_ffi!(sys::nvmlEventSetRegisterGpuOperationalEvents_v1(
                self.as_raw(),
                &raw const config
            ))
        }
    }

    /// Waits up to `timeout_ms` for the next event-bit or structured GPU Operational Event.
    ///
    /// Context records are copied out, so they outlive the next wait, which replaces NVML's context storage.
    ///
    /// # Errors
    ///
    /// Returns an error if the wait times out, NVML fails the wait or a context query, an entry point is unavailable, the event has an unknown data type, or NVML reports a context payload larger than the buffer it filled.
    pub fn wait(&mut self, timeout_ms: u32) -> Result<EventNotification> {
        let mut data = sys::nvmlEventSetWait_v3_t {
            timeoutMs: timeout_ms,
            ..Default::default()
        };
        // SAFETY: `self` holds a live event-set handle, and `data` is a writable local struct.
        unsafe {
            try_ffi!(sys::nvmlEventSetWait_v3(self.as_raw(), &raw mut data))?;
        }

        if data.dataType == sys::nvmlEventDataType_t::NVML_EVENT_DATA_TYPE_NVML_EVENT.0 {
            return Ok(EventNotification::Legacy(LegacyEvent {
                uuid: string_from_c_chars(&data.uuid),
                event_types: EventTypes::from_bits_retain(data.eventType),
                event_data: data.eventData,
                gpu_instance_id: (data.gpuInstanceId != sys::NVML_GPU_INSTANCE_ID_ANY)
                    .then_some(data.gpuInstanceId),
                compute_instance_id: (data.computeInstanceId != sys::NVML_COMPUTE_INSTANCE_ID_ANY)
                    .then_some(data.computeInstanceId),
            }));
        }

        if data.dataType != sys::nvmlEventDataType_t::NVML_EVENT_DATA_TYPE_GPU_OPERATIONAL_EVENT.0 {
            return Err(Error::UnknownEnumValue {
                name: "event data type".into(),
                value: data.dataType.into(),
            });
        }

        let mut count = sys::nvmlEventSetGetContextCount_v1_t::default();
        // SAFETY: `self` holds a live event-set handle, and `count` is a writable local struct.
        unsafe {
            try_ffi!(sys::nvmlEventSetGetContextCount_v1(
                self.as_raw(),
                &raw mut count
            ))?;
        }

        let mut contexts = Vec::with_capacity(to_usize(count.count, "context count")?);
        for index in 0..count.count {
            let mut info = sys::nvmlEventSetGetContextInfo_v1_t {
                index,
                ..Default::default()
            };
            // SAFETY: `self` holds a live event-set handle, and `info` is a writable local struct naming a context index below the reported count.
            unsafe {
                try_ffi!(sys::nvmlEventSetGetContextInfo_v1(
                    self.as_raw(),
                    &raw mut info
                ))?;
            }

            let mut payload = vec![0; to_usize(info.dataSize, "context data size")?];
            let mut request = sys::nvmlEventSetGetContextData_v1_t {
                index,
                data: if payload.is_empty() {
                    ptr::null_mut()
                } else {
                    payload.as_mut_ptr().cast()
                },
                dataSize: info.dataSize,
            };
            // SAFETY: `self` holds a live event-set handle, and `request.data` is null for an empty payload or points to the `dataSize` writable bytes of `payload`.
            unsafe {
                try_ffi!(sys::nvmlEventSetGetContextData_v1(
                    self.as_raw(),
                    &raw mut request
                ))?;
            }

            let data_size = to_usize(request.dataSize, "context data size")?;
            if data_size > payload.len() {
                return Err(Error::out_of_range("context data size"));
            }

            payload.truncate(data_size);
            let kind = EventContextType(info.nvmlGpuOperationalEventContextType);
            let legacy_xid = if kind == EventContextType::LEGACY_XID {
                let mut xid = sys::nvmlEventSetGetGpuOperationalEventContextLegacyXid_v1_t {
                    index,
                    ..Default::default()
                };
                // SAFETY: `self` holds a live event-set handle, and `xid` is a writable local struct naming a context of the legacy Xid kind.
                unsafe {
                    try_ffi!(sys::nvmlEventSetGetGpuOperationalEventContextLegacyXid_v1(
                        self.as_raw(),
                        &raw mut xid
                    ))?;
                }
                Some(xid.xidCode)
            } else {
                None
            };
            contexts.push(EventContext {
                kind,
                source_event_context_type: info.sourceEventContextType,
                data_format_version: info.dataFormatVersion,
                data: payload,
                legacy_xid,
            });
        }
        Ok(EventNotification::Operational(OperationalEvent::from_raw(
            data, contexts,
        )))
    }
}

/// Log-level values used by GPU Operational Events.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EventLogLevel(pub u32);

impl EventLogLevel {
    /// Matches all GPU Operational Event log levels.
    pub const ALL: Self =
        Self(sys::nvmlGpuOperationalEventLogLevel_t::NVML_GPU_OPERATIONAL_EVENT_LOG_LEVEL_ALL.0);

    /// High-volume telemetry events.
    pub const TELEMETRY: Self = Self(
        sys::nvmlGpuOperationalEventLogLevel_t::NVML_GPU_OPERATIONAL_EVENT_LOG_LEVEL_TELEMETRY.0,
    );

    /// Diagnostic events.
    pub const DIAG: Self =
        Self(sys::nvmlGpuOperationalEventLogLevel_t::NVML_GPU_OPERATIONAL_EVENT_LOG_LEVEL_DIAG.0);

    /// Notable operational events.
    pub const NOTICE: Self =
        Self(sys::nvmlGpuOperationalEventLogLevel_t::NVML_GPU_OPERATIONAL_EVENT_LOG_LEVEL_NOTICE.0);

    /// Warning events.
    pub const WARNING: Self = Self(
        sys::nvmlGpuOperationalEventLogLevel_t::NVML_GPU_OPERATIONAL_EVENT_LOG_LEVEL_WARNING.0,
    );

    /// Error events.
    pub const ERROR: Self =
        Self(sys::nvmlGpuOperationalEventLogLevel_t::NVML_GPU_OPERATIONAL_EVENT_LOG_LEVEL_ERROR.0);
}

/// Severity values used by Operational Events.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EventSeverity(pub u32);

impl EventSeverity {
    /// Matches all Operational Event severities.
    pub const ALL: Self =
        Self(sys::nvmlOperationalEventSeverity_t::NVML_OPERATIONAL_EVENT_SEVERITY_ALL.0);

    /// Informational event.
    pub const INFORMATIONAL: Self =
        Self(sys::nvmlOperationalEventSeverity_t::NVML_OPERATIONAL_EVENT_SEVERITY_INFORMATIONAL.0);

    /// Corrected error event.
    pub const CORRECTED: Self =
        Self(sys::nvmlOperationalEventSeverity_t::NVML_OPERATIONAL_EVENT_SEVERITY_CORRECTED.0);

    /// Recoverable error event.
    pub const RECOVERABLE: Self =
        Self(sys::nvmlOperationalEventSeverity_t::NVML_OPERATIONAL_EVENT_SEVERITY_RECOVERABLE.0);

    /// Fatal error event.
    pub const FATAL: Self =
        Self(sys::nvmlOperationalEventSeverity_t::NVML_OPERATIONAL_EVENT_SEVERITY_FATAL.0);
}

/// NVML-defined GPU Operational Event context classifications.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EventContextType(pub u32);

impl EventContextType {
    /// No NVML public interpretation is defined for this context payload.
    pub const UNKNOWN: Self = Self(
        sys::nvmlGpuOperationalEventContextType_t::NVML_GPU_OPERATIONAL_EVENT_CONTEXT_TYPE_UNKNOWN
            .0,
    );

    /// The context payload carries a legacy Xid code, decoded into [`EventContext::legacy_xid`].
    pub const LEGACY_XID: Self = Self(sys::nvmlGpuOperationalEventContextType_t::NVML_GPU_OPERATIONAL_EVENT_CONTEXT_TYPE_LEGACY_XID.0);
}

bitflags::bitflags! {
    /// Structured event attribute flags.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct EventAttributes: u32 {
        /// Event reports an uncontained condition.
        const UNCONTAINED = sys::NVML_OPERATIONAL_EVENT_ATTR_UNCONTAINED;
        /// Event reports a latent condition.
        const LATENT = sys::NVML_OPERATIONAL_EVENT_ATTR_LATENT;
        /// Event was propagated from another source.
        const PROPAGATED = sys::NVML_OPERATIONAL_EVENT_ATTR_PROPAGATED;
        /// Event involved a component reset.
        const COMPONENT_RESET = sys::NVML_OPERATIONAL_EVENT_ATTR_COMPONENT_RESET;
        /// Event reports an exceeded threshold.
        const THRESHOLD_EXCEEDED = sys::NVML_OPERATIONAL_EVENT_ATTR_THRESHOLD_EXCEEDED;
        /// Event is the primary event in its group.
        const PRIMARY = sys::NVML_OPERATIONAL_EVENT_ATTR_PRIMARY;
        /// One or more events or associated payloads were dropped before this event was returned.
        const OVERFLOW = sys::NVML_OPERATIONAL_EVENT_ATTR_OVERFLOW;
    }
}

bitflags::bitflags! {
    /// Structured event group attribute flags.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct EventGroupAttributes: u32 {
        /// Event group reports a recovered condition.
        const RECOVERED = sys::NVML_OPERATIONAL_EVENT_GROUP_ATTR_RECOVERED;
        /// Event group reports a previous error condition.
        const PREVERR = sys::NVML_OPERATIONAL_EVENT_GROUP_ATTR_PREVERR;
        /// Event group was generated by simulation or testing.
        const SIMULATED = sys::NVML_OPERATIONAL_EVENT_GROUP_ATTR_SIMULATED;
    }
}

/// Structured GPU Operational Event metadata and context payloads.
#[derive(Debug)]
pub struct OperationalEvent {
    /// UUID for the GPU where the event occurred.
    /// Empty if unavailable.
    pub uuid: String,
    /// Source module signature for structured events.
    pub source_module: String,
    /// Event group identifier.
    pub group_cursor: u64,
    /// Event sequence identifier.
    pub instance_id: u64,
    /// Event timestamp in microseconds, or zero if unavailable.
    pub timestamp_usec: u64,
    /// Event trace identifier.
    pub trace_id: u64,
    /// Severity of the event, possibly a value newer than the constants of [`EventSeverity`].
    pub severity: EventSeverity,
    /// Source-defined event category identifier.
    pub category_id: u32,
    /// Event code defined by [`Self::source_module`].
    pub module_event_code: u32,
    /// Event scope identifier.
    pub scope: u32,
    /// Event originator identifier.
    pub originator: u32,
    /// Module instance identifier.
    pub module_instance: u32,
    /// Chiplet identifier.
    pub chiplet_id: u32,
    /// Log level of the event, possibly a value newer than the constants of [`EventLogLevel`].
    pub log_level: EventLogLevel,
    /// Event attributes, possibly with bits newer than [`EventAttributes`] names.
    ///
    /// [`EventAttributes::OVERFLOW`] means events or payloads were dropped.
    pub attributes: EventAttributes,
    /// Size in bytes of the associated CPER record, or zero if unavailable.
    pub group_cper_size: u32,
    /// Event group attributes, possibly with bits newer than [`EventGroupAttributes`] names.
    pub group_attributes: EventGroupAttributes,
    /// Number of events in the event group.
    pub group_size: u8,
    /// Zero-based index of the event within its group.
    pub group_index: u8,
    /// Owned context records associated with this event.
    pub contexts: Vec<EventContext>,
}

impl OperationalEvent {
    fn from_raw(value: sys::nvmlEventSetWait_v3_t, contexts: Vec<EventContext>) -> Self {
        Self {
            uuid: string_from_c_chars(&value.uuid),
            source_module: string_from_c_chars(&value.sourceModule),
            group_cursor: value.groupCursor,
            instance_id: value.instanceId,
            timestamp_usec: value.timestampUsec,
            trace_id: value.traceId,
            severity: EventSeverity(value.severity),
            category_id: value.categoryId,
            module_event_code: value.moduleEventCode,
            scope: value.scope,
            originator: value.originator,
            module_instance: value.moduleInstance,
            chiplet_id: value.chipletId,
            log_level: EventLogLevel(value.logLevel),
            attributes: EventAttributes::from_bits_retain(value.attributes),
            group_cper_size: value.groupCperSize,
            group_attributes: EventGroupAttributes::from_bits_retain(value.groupAttributes),
            group_size: value.groupSize,
            group_index: value.groupIndex,
            contexts,
        }
    }
}

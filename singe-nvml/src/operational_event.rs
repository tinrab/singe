use std::{ffi::CStr, ptr};

use singe_core::string_from_c_chars;
use singe_nvml_sys as sys;

use crate::{
    error::{Error, Result, Status},
    library::EventSet,
    optional::{
        nvmlEventSetGetContextCount_v1, nvmlEventSetGetContextData_v1,
        nvmlEventSetGetContextInfo_v1, nvmlEventSetGetGpuOperationalEventContextLegacyXid_v1,
        nvmlEventSetRegisterGpuOperationalEvents_v1, nvmlEventSetWait_v3,
    },
    try_ffi,
    types::EventTypes,
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
    /// UUID for the GPU where the event occurred. Empty if unavailable.
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
    /// Registers structured GPU Operational Events for a GPU UUID.
    ///
    /// For Turing or newer fully supported devices on Linux. Registrations are
    /// idempotent. Legacy event-bit subscriptions may coexist on the same set.
    ///
    /// # Errors
    ///
    /// Returns `Status::InvalidArgument` if the UUID, including its terminator, exceeds the native buffer. Also returns an error if the entry point is unavailable or NVML rejects the event set, GPU UUID, filter levels, or caller permissions.
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
            return Err(Status::InvalidArgument.into());
        }
        for (target, source) in config.uuid.iter_mut().zip(bytes) {
            *target = source.cast_signed();
        }
        // SAFETY: `nvmlEventSetRegisterGpuOperationalEvents_v1` receives a live handle and a versioned local struct that outlives the call.
        unsafe {
            try_ffi!(nvmlEventSetRegisterGpuOperationalEvents_v1(
                self.as_raw(),
                &raw const config
            ))
        }
    }

    /// Waits for the next NVML event-bit or structured GPU Operational Event.
    ///
    /// Context records are copied before returning. They remain valid after a
    /// later successful wait replaces NVML's event-set context storage.
    ///
    /// # Errors
    ///
    /// Propagates NVML wait and context-data query errors, including timeout and unavailable entry points. Also rejects an unknown event-data discriminator or a context payload whose returned size exceeds the allocated buffer.
    pub fn wait(&mut self, timeout_ms: u32) -> Result<EventNotification> {
        let mut data = sys::nvmlEventSetWait_v3_t {
            timeoutMs: timeout_ms,
            ..Default::default()
        };
        // SAFETY: `nvmlEventSetWait_v3` receives a live handle and a versioned local struct that outlives the call.
        unsafe {
            try_ffi!(nvmlEventSetWait_v3(self.as_raw(), &raw mut data))?;
        }
        if data.dataType == sys::nvmlEventDataType_t::NVML_EVENT_DATA_TYPE_NVML_EVENT as u32 {
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
        if data.dataType
            != sys::nvmlEventDataType_t::NVML_EVENT_DATA_TYPE_GPU_OPERATIONAL_EVENT as u32
        {
            return Err(Error::UnknownEnumValue {
                name: "event data type".into(),
                value: data.dataType,
            });
        }
        let mut count = sys::nvmlEventSetGetContextCount_v1_t::default();
        // SAFETY: `nvmlEventSetGetContextCount_v1` receives a live handle and a versioned local struct that outlives the call.
        unsafe {
            try_ffi!(nvmlEventSetGetContextCount_v1(
                self.as_raw(),
                &raw mut count
            ))?;
        }
        let mut contexts = Vec::with_capacity(count.count as usize);
        for index in 0..count.count {
            let mut info = sys::nvmlEventSetGetContextInfo_v1_t {
                index,
                ..Default::default()
            };
            // SAFETY: `nvmlEventSetGetContextInfo_v1` receives a live handle and a versioned local struct that outlives the call.
            unsafe {
                try_ffi!(nvmlEventSetGetContextInfo_v1(self.as_raw(), &raw mut info))?;
            }
            let mut payload = vec![0; info.dataSize as usize];
            let mut request = sys::nvmlEventSetGetContextData_v1_t {
                index,
                data: if payload.is_empty() {
                    ptr::null_mut()
                } else {
                    payload.as_mut_ptr().cast()
                },
                dataSize: info.dataSize,
            };
            // SAFETY: `nvmlEventSetGetContextData_v1` receives a live handle and a versioned local struct that outlives the call.
            unsafe {
                try_ffi!(nvmlEventSetGetContextData_v1(
                    self.as_raw(),
                    &raw mut request
                ))?;
            }
            if request.dataSize as usize > payload.len() {
                return Err(Status::InsufficientSize.into());
            }
            payload.truncate(request.dataSize as usize);
            let kind = EventContextType(info.nvmlGpuOperationalEventContextType);
            let legacy_xid = if kind == EventContextType::LEGACY_XID {
                let mut xid = sys::nvmlEventSetGetGpuOperationalEventContextLegacyXid_v1_t {
                    index,
                    ..Default::default()
                };
                // SAFETY: `nvmlEventSetGetGpuOperationalEventContextLegacyXid_v1` receives a live handle and a versioned local struct that outlives the call.
                unsafe {
                    try_ffi!(nvmlEventSetGetGpuOperationalEventContextLegacyXid_v1(
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
        Self(sys::nvmlGpuOperationalEventLogLevel_t::NVML_GPU_OPERATIONAL_EVENT_LOG_LEVEL_ALL as _);

    /// High-volume telemetry events.
    pub const TELEMETRY: Self = Self(
        sys::nvmlGpuOperationalEventLogLevel_t::NVML_GPU_OPERATIONAL_EVENT_LOG_LEVEL_TELEMETRY as _,
    );

    /// Diagnostic events.
    pub const DIAG: Self = Self(
        sys::nvmlGpuOperationalEventLogLevel_t::NVML_GPU_OPERATIONAL_EVENT_LOG_LEVEL_DIAG as _,
    );

    /// Notable operational events.
    pub const NOTICE: Self = Self(
        sys::nvmlGpuOperationalEventLogLevel_t::NVML_GPU_OPERATIONAL_EVENT_LOG_LEVEL_NOTICE as _,
    );

    /// Warning events.
    pub const WARNING: Self = Self(
        sys::nvmlGpuOperationalEventLogLevel_t::NVML_GPU_OPERATIONAL_EVENT_LOG_LEVEL_WARNING as _,
    );

    /// Error events.
    pub const ERROR: Self = Self(
        sys::nvmlGpuOperationalEventLogLevel_t::NVML_GPU_OPERATIONAL_EVENT_LOG_LEVEL_ERROR as _,
    );
}

/// Severity values used by Operational Events.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EventSeverity(pub u32);

impl EventSeverity {
    /// Matches all Operational Event severities.
    pub const ALL: Self =
        Self(sys::nvmlOperationalEventSeverity_t::NVML_OPERATIONAL_EVENT_SEVERITY_ALL as _);

    /// Informational event.
    pub const INFORMATIONAL: Self = Self(
        sys::nvmlOperationalEventSeverity_t::NVML_OPERATIONAL_EVENT_SEVERITY_INFORMATIONAL as _,
    );

    /// Corrected error event.
    pub const CORRECTED: Self =
        Self(sys::nvmlOperationalEventSeverity_t::NVML_OPERATIONAL_EVENT_SEVERITY_CORRECTED as _);

    /// Recoverable error event.
    pub const RECOVERABLE: Self =
        Self(sys::nvmlOperationalEventSeverity_t::NVML_OPERATIONAL_EVENT_SEVERITY_RECOVERABLE as _);

    /// Fatal error event.
    pub const FATAL: Self =
        Self(sys::nvmlOperationalEventSeverity_t::NVML_OPERATIONAL_EVENT_SEVERITY_FATAL as _);
}

/// NVML-defined GPU Operational Event context classifications.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EventContextType(pub u32);

impl EventContextType {
    /// No NVML public interpretation is defined for this context payload.
    pub const UNKNOWN: Self = Self(
        sys::nvmlGpuOperationalEventContextType_t::NVML_GPU_OPERATIONAL_EVENT_CONTEXT_TYPE_UNKNOWN
            as _,
    );

    /// Context payload can be decoded with `nvmlEventSetGetGpuOperationalEventContextLegacyXid_v1`.
    pub const LEGACY_XID: Self = Self(sys::nvmlGpuOperationalEventContextType_t::NVML_GPU_OPERATIONAL_EVENT_CONTEXT_TYPE_LEGACY_XID as _);
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
    /// Structured event attribute flags.
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
    /// UUID for the GPU where the event occurred. Empty if unavailable.
    pub uuid: String,
    /// Source module signature for structured events. Not guaranteed to be NULL-terminated. Empty for NVML event-bit events.
    pub source_module: String,
    /// Structured event group identifier. 0 for NVML event-bit events.
    pub group_cursor: u64,
    /// Structured event sequence identifier. 0 for NVML event-bit events.
    pub instance_id: u64,
    /// Event timestamp in microseconds. 0 if unavailable.
    pub timestamp_usec: u64,
    /// Structured event trace identifier. 0 for NVML event-bit events.
    pub trace_id: u64,
    /// `nvmlOperationalEventSeverity_t` value for structured events. May contain newer severity values not named in this header. `nvmlOperationalEventSeverity_t::NVML_OPERATIONAL_EVENT_SEVERITY_ALL` for NVML event-bit events.
    pub severity: EventSeverity,
    /// Source-defined structured event category identifier. 0 for NVML event-bit events.
    pub category_id: u32,
    /// Source-module-defined event code. Interpret with `sourceModule`. 0 for NVML event-bit events.
    pub module_event_code: u32,
    /// Structured event scope identifier. 0 for NVML event-bit events.
    pub scope: u32,
    /// Structured event originator identifier. 0 for NVML event-bit events.
    pub originator: u32,
    /// Structured event module instance identifier. 0 for NVML event-bit events.
    pub module_instance: u32,
    /// Structured event chiplet identifier. 0 for NVML event-bit events.
    pub chiplet_id: u32,
    /// `nvmlGpuOperationalEventLogLevel_t` value for structured GPU Operational Events. May contain newer log-level values not named in this header. `nvmlGpuOperationalEventLogLevel_t::NVML_GPU_OPERATIONAL_EVENT_LOG_LEVEL_ALL` for NVML event-bit events.
    pub log_level: EventLogLevel,
    /// Bitmask of `NVML_OPERATIONAL_EVENT_ATTR_*` values for structured events. May contain newer bits not named in this header. 0 for NVML event-bit events. May include `NVML_OPERATIONAL_EVENT_ATTR_OVERFLOW` if events or associated payloads were dropped.
    pub attributes: EventAttributes,
    /// Associated CPER record size in bytes. 0 when unavailable.
    pub group_cper_size: u32,
    /// Bitmask of `NVML_OPERATIONAL_EVENT_GROUP_ATTR_*` values for structured events. May contain newer bits not named in this header. 0 for NVML event-bit events.
    pub group_attributes: EventGroupAttributes,
    /// Total number of events in the structured event group. 0 for NVML event-bit events.
    pub group_size: u8,
    /// Zero-based index within the structured event group. 0 for NVML event-bit events.
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

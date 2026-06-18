use std::{
    ffi::{CStr, CString, c_char},
    marker::PhantomData,
    mem::{MaybeUninit, size_of, zeroed},
    panic::{AssertUnwindSafe, catch_unwind},
    ptr,
    sync::Mutex,
};

use singe_cuda::device::Device;
use singe_cupti_sys as sys;

use crate::{
    context::Context,
    error::{Error, Result},
    try_ffi,
    types::{
        DeviceAttribute, DeviceAttributeDeviceClass, DeviceAttributeValue, EventAttribute,
        EventAttributeValue, EventCategory, EventCollectionMethod, EventCollectionMode,
        EventDomainAttribute, EventDomainAttributeValue, EventDomainId, EventGroupAttribute,
        EventGroupAttributeSetting, EventGroupAttributeValue, EventId, EventProfilingScope,
        MetricAttribute, MetricAttributeValue, MetricCategory, MetricEvaluationMode, MetricId,
        MetricPropertyId, MetricValue, MetricValueKind, ReadEventFlags,
    },
    utility::{checked_byte_len, to_u64, to_usize},
};

type KernelReplayUpdateCallback = dyn FnMut(&CStr, i32) + Send + 'static;

static KERNEL_REPLAY_UPDATE_CALLBACK: Mutex<Option<Box<KernelReplayUpdateCallback>>> =
    Mutex::new(None);

fn query_array<T: Copy>(
    mut query: impl FnMut(*mut u64, *mut T) -> sys::CUptiResult,
) -> Result<Vec<T>> {
    let mut size_bytes = 0;
    try_ffi!(query(&mut size_bytes, ptr::null_mut()))?;

    if size_bytes == 0 {
        return Ok(Vec::new());
    }

    let element_size = size_of::<T>();
    let size_bytes = to_usize(size_bytes, "array_size_bytes")?;

    if size_bytes % element_size != 0 {
        return Err(Error::LengthMismatch {
            name: "array_size_bytes".to_string(),
            expected: element_size,
            actual: size_bytes,
        });
    }

    let len = size_bytes / element_size;
    let mut values = vec![unsafe { zeroed() }; len];
    let mut retry_size_bytes = size_bytes as u64;
    try_ffi!(query(&mut retry_size_bytes, values.as_mut_ptr()))?;
    let retry_size_bytes = to_usize(retry_size_bytes, "array_size_bytes")?;
    values.truncate(retry_size_bytes / element_size);
    Ok(values)
}

#[derive(Debug)]
pub struct EventGroup {
    handle: sys::CUpti_EventGroup,
}

#[derive(Debug)]
pub struct EventGroupSets {
    handle: *mut sys::CUpti_EventGroupSets,
}

#[derive(Debug, Clone, Copy)]
pub struct EventGroupSet<'a> {
    handle: *mut sys::CUpti_EventGroupSet,
    _marker: PhantomData<&'a EventGroupSets>,
}

#[derive(Debug, Clone)]
pub struct EventGroupReadAll {
    pub event_ids: Vec<EventId>,
    pub values: Vec<u64>,
}

impl EventGroup {
    fn from_raw(handle: sys::CUpti_EventGroup) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self { handle })
    }

    pub(crate) const fn as_raw(&self) -> sys::CUpti_EventGroup {
        self.handle
    }
}

impl Drop for EventGroup {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::cuptiEventGroupDestroy(self.handle);
        }
    }
}

impl EventGroupSets {
    fn from_raw(handle: *mut sys::CUpti_EventGroupSets) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self { handle })
    }

    pub fn sets(&self) -> Vec<EventGroupSet<'_>> {
        let sets = unsafe { &*self.handle };
        if sets.sets.is_null() || sets.numSets == 0 {
            return Vec::new();
        }
        (0..sets.numSets as usize)
            .map(|index| EventGroupSet {
                handle: unsafe { sets.sets.add(index) },
                _marker: PhantomData,
            })
            .collect()
    }
}

impl Drop for EventGroupSets {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::cuptiEventGroupSetsDestroy(self.handle);
        }
    }
}

impl EventGroupSet<'_> {
    pub(crate) const fn as_raw(self) -> *mut sys::CUpti_EventGroupSet {
        self.handle
    }

    /// Enables every event group in this set and resets their counters.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::Hardware`] or [`crate::error::Status::HardwareBusy`] if profiling hardware cannot be used.
    /// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
    /// - Returns [`crate::error::Status::NotCompatible`] if another enabled event group conflicts with this set.
    /// - Returns [`crate::error::Status::NotReady`] if an event group has no events.
    pub fn enable(self) -> Result<()> {
        unsafe {
            try_ffi!(sys::cuptiEventGroupSetEnable(self.as_raw()))?;
        }
        Ok(())
    }

    /// Disables every event group in this set and stops collecting its events.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::Hardware`] or [`crate::error::Status::HardwareBusy`] if profiling hardware cannot be used.
    /// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
    pub fn disable(self) -> Result<()> {
        unsafe {
            try_ffi!(sys::cuptiEventGroupSetDisable(self.as_raw()))?;
        }
        Ok(())
    }
}

impl Context {
    /// Sets the event collection mode for event groups created in this context.
    ///
    /// This is invalid while kernel replay mode is enabled.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::InvalidContext`] if the CUDA context is invalid.
    /// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
    /// - Returns [`crate::error::Status::InvalidOperation`] if kernel replay mode is enabled.
    /// - Returns [`crate::error::Status::NotSupported`] if the device does not support the requested mode.
    pub fn set_event_collection_mode(&self, mode: EventCollectionMode) -> Result<()> {
        unsafe {
            try_ffi!(sys::cuptiSetEventCollectionMode(self.as_raw(), mode.into()))?;
        }
        Ok(())
    }

    /// Creates an empty event group for this context.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::InvalidContext`] if the CUDA context is invalid.
    /// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
    /// - Returns [`crate::error::Status::OutOfMemory`] if CUPTI cannot allocate required internal storage.
    pub fn create_event_group(&self) -> Result<EventGroup> {
        let mut event_group = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cuptiEventGroupCreate(
                self.as_raw(),
                &mut event_group,
                0
            ))?;
        }
        EventGroup::from_raw(event_group)
    }

    /// Groups events into the passes and event groups required to collect them on this context.
    ///
    /// The grouping depends on the device and event domains; not all requested events can necessarily be collected in one pass.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::InvalidContext`] if the CUDA context is invalid.
    /// - Returns [`crate::error::Status::InvalidEventId`] if the event ID is invalid.
    /// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
    /// - Returns [`crate::error::Status::InvalidParameter`] if the event list cannot be passed to CUPTI.
    pub fn create_event_group_sets(&self, event_ids: &[EventId]) -> Result<EventGroupSets> {
        let mut event_group_sets = ptr::null_mut();
        let event_ids_size = checked_byte_len::<EventId>(event_ids.len(), "event_ids_size_bytes")?;
        let mut event_ids = event_ids
            .iter()
            .copied()
            .map(EventId::as_raw)
            .collect::<Vec<_>>();
        unsafe {
            try_ffi!(sys::cuptiEventGroupSetsCreate(
                self.as_raw(),
                to_u64(event_ids_size, "event_ids_size_bytes")?,
                event_ids.as_mut_ptr(),
                &mut event_group_sets,
            ))?;
        }
        EventGroupSets::from_raw(event_group_sets)
    }

    /// Enables kernel replay mode for this context.
    ///
    /// Kernel replay mode allows more events to be collected for a kernel by replaying it as needed.
    /// CUPTI switches event collection to [`crate::types::EventCollectionMode::Kernel`] while this mode is active.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::InvalidContext`] if the CUDA context is invalid.
    /// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
    /// - Returns [`crate::error::Status::InvalidOperation`] if kernel replay mode conflicts with the current profiling state.
    /// - Returns [`crate::error::Status::NotSupported`] if the device does not support kernel replay mode.
    pub fn enable_kernel_replay_mode(&self) -> Result<()> {
        unsafe {
            try_ffi!(sys::cuptiEnableKernelReplayMode(self.as_raw()))?;
        }
        Ok(())
    }

    /// Disables kernel replay mode for this context and returns to non-replay event collection.
    ///
    /// Disabling replay mode also disables previously enabled event groups and event group sets.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::InvalidContext`] if the CUDA context is invalid.
    /// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
    /// - Returns [`crate::error::Status::InvalidOperation`] if CUPTI cannot leave replay mode in the current state.
    pub fn disable_kernel_replay_mode(&self) -> Result<()> {
        unsafe {
            try_ffi!(sys::cuptiDisableKernelReplayMode(self.as_raw()))?;
        }
        Ok(())
    }

    /// Returns event group sets that must be collected in the same pass for a metric.
    ///
    /// Some metrics do not require fixed grouping; CUPTI may return no required event group sets for those metrics.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::InvalidMetricId`] if the metric ID is invalid.
    /// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
    pub fn metric_required_event_group_sets(&self, metric: MetricId) -> Result<EventGroupSets> {
        let mut event_group_sets = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cuptiMetricGetRequiredEventGroupSets(
                self.as_raw(),
                metric.as_raw(),
                &mut event_group_sets,
            ))?;
        }
        EventGroupSets::from_raw(event_group_sets)
    }

    /// Groups the events required to collect a set of metrics on this context.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::InvalidContext`] if the CUDA context is invalid.
    /// - Returns [`crate::error::Status::InvalidMetricId`] if the metric ID is invalid.
    /// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
    /// - Returns [`crate::error::Status::InvalidParameter`] if the metric list cannot be passed to CUPTI.
    pub fn create_metric_event_group_sets(
        &self,
        metric_ids: &[MetricId],
    ) -> Result<EventGroupSets> {
        let mut event_group_sets = ptr::null_mut();
        let metric_ids_size =
            checked_byte_len::<MetricId>(metric_ids.len(), "metric_ids_size_bytes")?;
        let mut metric_ids = metric_ids
            .iter()
            .copied()
            .map(MetricId::as_raw)
            .collect::<Vec<_>>();
        unsafe {
            try_ffi!(sys::cuptiMetricCreateEventGroupSets(
                self.as_raw(),
                to_u64(metric_ids_size, "metric_ids_size_bytes")?,
                metric_ids.as_mut_ptr(),
                &mut event_group_sets,
            ))?;
        }
        EventGroupSets::from_raw(event_group_sets)
    }
}

fn read_attribute_scalar<T: Copy>(
    mut read: impl FnMut(*mut u64, *mut ()) -> Result<()>,
    name: &'static str,
) -> Result<T> {
    let mut value = MaybeUninit::<T>::uninit();
    let mut value_size = to_u64(size_of::<T>(), name)?;
    read(&mut value_size, value.as_mut_ptr().cast())?;
    Ok(unsafe { value.assume_init() })
}

fn read_attribute_string(
    mut read: impl FnMut(*mut u64, *mut ()) -> Result<()>,
    name: &'static str,
) -> Result<String> {
    let mut buffer = vec![0u8; 4096];
    let mut value_size = to_u64(buffer.len(), name)?;
    read(&mut value_size, buffer.as_mut_ptr().cast())?;
    let len = buffer
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or_else(|| value_size.min(buffer.len() as u64) as usize);
    Ok(String::from_utf8_lossy(&buffer[..len]).into_owned())
}

/// Reads a CUPTI device attribute for a CUDA device.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidDevice`] if the CUDA device is invalid.
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidParameter`] if the requested device attribute is invalid.
/// - Returns [`crate::error::Status::ParameterSizeNotSufficient`] if the decoded value buffer is too small for the requested attribute.
pub fn get_device_attribute_value(
    device: Device,
    attrib: DeviceAttribute,
) -> Result<DeviceAttributeValue> {
    let mut read = |value_size: *mut u64, value: *mut ()| unsafe {
        try_ffi!(sys::cuptiDeviceGetAttribute(
            device.id() as sys::CUdevice,
            attrib.into(),
            value_size,
            value.cast(),
        ))?;
        Ok(())
    };
    match attrib {
        DeviceAttribute::DeviceClass => Ok(DeviceAttributeValue::DeviceClass(
            DeviceAttributeDeviceClass::from(read_attribute_scalar::<
                sys::CUpti_DeviceAttributeDeviceClass,
            >(&mut read, "device_attribute_size")?),
        )),
        DeviceAttribute::ForceInt => Err(Error::InvalidAttribute {
            name: format!("{attrib:?}"),
        }),
        _ => Ok(DeviceAttributeValue::U32(read_attribute_scalar::<u32>(
            &mut read,
            "device_attribute_size",
        )?)),
    }
}

/// Returns the number of event domains available on a CUDA device.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidDevice`] if the CUDA device is invalid.
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
pub fn get_device_event_domain_count(device: Device) -> Result<u32> {
    let mut num_domains = 0;
    unsafe {
        try_ffi!(sys::cuptiDeviceGetNumEventDomains(
            device.id() as sys::CUdevice,
            &mut num_domains,
        ))?;
    }
    Ok(num_domains)
}

/// Returns the event domains available on a CUDA device.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidDevice`] if the CUDA device is invalid.
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
pub fn get_device_event_domains(device: Device) -> Result<Vec<EventDomainId>> {
    let raw = query_array(|array_size_bytes, domain_array| unsafe {
        sys::cuptiDeviceEnumEventDomains(
            device.id() as sys::CUdevice,
            array_size_bytes,
            domain_array,
        )
    })?;
    Ok(raw.into_iter().map(EventDomainId::from).collect())
}

/// Reads an event-domain attribute for a CUDA device and event domain.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidDevice`] if the CUDA device is invalid.
/// - Returns [`crate::error::Status::InvalidEventDomainId`] if the event domain ID is invalid.
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidParameter`] if the requested event-domain attribute is invalid.
/// - Returns [`crate::error::Status::ParameterSizeNotSufficient`] if the decoded value buffer is too small for the requested attribute.
pub fn get_device_event_domain_attribute_value(
    device: Device,
    event_domain: EventDomainId,
    attrib: EventDomainAttribute,
) -> Result<EventDomainAttributeValue> {
    let mut read = |value_size: *mut u64, value: *mut ()| unsafe {
        try_ffi!(sys::cuptiDeviceGetEventDomainAttribute(
            device.id() as sys::CUdevice,
            event_domain.as_raw(),
            attrib.into(),
            value_size,
            value.cast(),
        ))?;
        Ok(())
    };
    event_domain_attribute_value_with(&mut read, attrib)
}

/// Returns the number of event domains available on any CUDA-capable device.
pub fn get_event_domain_count() -> Result<u32> {
    let mut num_domains = 0;
    unsafe {
        try_ffi!(sys::cuptiGetNumEventDomains(&mut num_domains))?;
    }
    Ok(num_domains)
}

/// Returns all event domains available on any CUDA-capable device.
pub fn get_enum_event_domains() -> Result<Vec<EventDomainId>> {
    let raw = query_array(|array_size_bytes, domain_array| unsafe {
        sys::cuptiEnumEventDomains(array_size_bytes, domain_array)
    })?;
    Ok(raw.into_iter().map(EventDomainId::from).collect())
}

/// Reads an event-domain attribute independent of a specific CUDA device.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidEventDomainId`] if the event domain ID is invalid.
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidParameter`] if the requested event-domain attribute is invalid.
/// - Returns [`crate::error::Status::ParameterSizeNotSufficient`] if the decoded value buffer is too small for the requested attribute.
pub fn get_event_domain_attribute_value(
    event_domain: EventDomainId,
    attrib: EventDomainAttribute,
) -> Result<EventDomainAttributeValue> {
    let mut read = |value_size: *mut u64, value: *mut ()| unsafe {
        try_ffi!(sys::cuptiEventDomainGetAttribute(
            event_domain.as_raw(),
            attrib.into(),
            value_size,
            value.cast(),
        ))?;
        Ok(())
    };
    event_domain_attribute_value_with(&mut read, attrib)
}

fn event_domain_attribute_value_with(
    read: &mut impl FnMut(*mut u64, *mut ()) -> Result<()>,
    attrib: EventDomainAttribute,
) -> Result<EventDomainAttributeValue> {
    match attrib {
        EventDomainAttribute::Name => Ok(EventDomainAttributeValue::Name(read_attribute_string(
            read,
            "event_domain_attribute_size",
        )?)),
        EventDomainAttribute::InstanceCount | EventDomainAttribute::TotalInstanceCount => {
            Ok(EventDomainAttributeValue::U32(
                read_attribute_scalar::<u32>(read, "event_domain_attribute_size")?,
            ))
        }
        EventDomainAttribute::CollectionMethod => Ok(EventDomainAttributeValue::CollectionMethod(
            EventCollectionMethod::from(read_attribute_scalar::<sys::CUpti_EventCollectionMethod>(
                read,
                "event_domain_attribute_size",
            )?),
        )),
        EventDomainAttribute::ForceInt => Err(Error::InvalidAttribute {
            name: format!("{attrib:?}"),
        }),
    }
}

/// Returns the number of events in an event domain.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidEventDomainId`] if the event domain ID is invalid.
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
pub fn get_event_domain_event_count(event_domain: EventDomainId) -> Result<u32> {
    let mut num_events = 0;
    unsafe {
        try_ffi!(sys::cuptiEventDomainGetNumEvents(
            event_domain.as_raw(),
            &mut num_events,
        ))?;
    }
    Ok(num_events)
}

/// Returns the events in an event domain.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidEventDomainId`] if the event domain ID is invalid.
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
pub fn get_event_domain_enum_events(event_domain: EventDomainId) -> Result<Vec<EventId>> {
    let raw = query_array(|array_size_bytes, event_array| unsafe {
        sys::cuptiEventDomainEnumEvents(event_domain.as_raw(), array_size_bytes, event_array)
    })?;
    Ok(raw.into_iter().map(EventId::from).collect())
}

/// Reads an event attribute.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidEventId`] if the event ID is invalid.
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidParameter`] if the requested event attribute is invalid.
/// - Returns [`crate::error::Status::ParameterSizeNotSufficient`] if the decoded value buffer is too small for the requested attribute.
pub fn get_event_attribute_value(
    event: EventId,
    attrib: EventAttribute,
) -> Result<EventAttributeValue> {
    let mut read = |value_size: *mut u64, value: *mut ()| unsafe {
        try_ffi!(sys::cuptiEventGetAttribute(
            event.as_raw(),
            attrib.into(),
            value_size,
            value.cast(),
        ))?;
        Ok(())
    };
    match attrib {
        EventAttribute::Name => Ok(EventAttributeValue::Name(read_attribute_string(
            &mut read,
            "event_attribute_size",
        )?)),
        EventAttribute::ShortDescription => Ok(EventAttributeValue::ShortDescription(
            read_attribute_string(&mut read, "event_attribute_size")?,
        )),
        EventAttribute::LongDescription => Ok(EventAttributeValue::LongDescription(
            read_attribute_string(&mut read, "event_attribute_size")?,
        )),
        EventAttribute::Category => Ok(EventAttributeValue::Category(EventCategory::from(
            read_attribute_scalar::<sys::CUpti_EventCategory>(&mut read, "event_attribute_size")?,
        ))),
        EventAttribute::ProfilingScope => Ok(EventAttributeValue::ProfilingScope(
            EventProfilingScope::from(read_attribute_scalar::<sys::CUpti_EventProfilingScope>(
                &mut read,
                "event_attribute_size",
            )?),
        )),
        EventAttribute::ForceInt => Err(Error::InvalidAttribute {
            name: format!("{attrib:?}"),
        }),
    }
}

/// Finds an event ID by name for a CUDA device.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidDevice`] if the CUDA device is invalid.
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidEventName`] if no event with `event_name` exists on the device.
/// - Returns [`crate::error::Status::InvalidParameter`] if CUPTI rejects the event name.
pub fn get_event_id_from_name(device: Device, event_name: &CStr) -> Result<EventId> {
    let mut event = 0u32;
    unsafe {
        try_ffi!(sys::cuptiEventGetIdFromName(
            device.id() as sys::CUdevice,
            event_name.as_ptr(),
            &mut event,
        ))?;
    }
    Ok(EventId::from(event))
}

pub fn get_event_id_from_str(device: Device, event_name: &str) -> Result<EventId> {
    let event_name = CString::new(event_name)?;
    get_event_id_from_name(device, &event_name)
}

pub fn get_event_group_attribute_value(
    event_group: &EventGroup,
    attrib: EventGroupAttribute,
) -> Result<EventGroupAttributeValue> {
    let mut read = |value_size: *mut u64, value: *mut ()| unsafe {
        try_ffi!(sys::cuptiEventGroupGetAttribute(
            event_group.as_raw(),
            attrib.into(),
            value_size,
            value.cast(),
        ))?;
        Ok(())
    };
    match attrib {
        EventGroupAttribute::EventDomainId => Ok(EventGroupAttributeValue::EventDomainId(
            EventDomainId::from(read_attribute_scalar::<sys::CUpti_EventDomainID>(
                &mut read,
                "event_group_attribute_size",
            )?),
        )),
        EventGroupAttribute::ProfileAllDomainInstances => {
            Ok(EventGroupAttributeValue::ProfileAllDomainInstances(
                read_attribute_scalar::<u32>(&mut read, "event_group_attribute_size")? != 0,
            ))
        }
        EventGroupAttribute::NumEvents => Ok(EventGroupAttributeValue::NumEvents(
            read_attribute_scalar::<u32>(&mut read, "event_group_attribute_size")?,
        )),
        EventGroupAttribute::Events => {
            let raw = query_array::<sys::CUpti_EventID>(|array_size_bytes, event_array| unsafe {
                sys::cuptiEventGroupGetAttribute(
                    event_group.as_raw(),
                    attrib.into(),
                    array_size_bytes,
                    event_array.cast(),
                )
            })?;
            Ok(EventGroupAttributeValue::Events(
                raw.into_iter().map(EventId::from).collect(),
            ))
        }
        EventGroupAttribute::InstanceCount => Ok(EventGroupAttributeValue::InstanceCount(
            read_attribute_scalar::<u32>(&mut read, "event_group_attribute_size")?,
        )),
        EventGroupAttribute::ProfilingScope => Ok(EventGroupAttributeValue::ProfilingScope(
            EventProfilingScope::from(read_attribute_scalar::<sys::CUpti_EventProfilingScope>(
                &mut read,
                "event_group_attribute_size",
            )?),
        )),
        EventGroupAttribute::UserData | EventGroupAttribute::ForceInt => {
            Err(Error::InvalidAttribute {
                name: format!("{attrib:?}"),
            })
        }
    }
}

/// Writes a supported event-group attribute.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidParameter`] if the attribute is not writable or the value has the wrong shape.
/// - Returns [`crate::error::Status::ParameterSizeNotSufficient`] if the decoded value buffer is too small for the requested attribute.
pub fn set_event_group_attribute_value(
    event_group: &EventGroup,
    attrib: EventGroupAttribute,
    value: EventGroupAttributeSetting,
) -> Result<()> {
    match (attrib, value) {
        (
            EventGroupAttribute::ProfileAllDomainInstances,
            EventGroupAttributeSetting::ProfileAllDomainInstances(value),
        ) => {
            let mut value = value as u32;
            let value_size = to_u64(size_of::<u32>(), "event_group_attribute_size")?;
            unsafe {
                try_ffi!(sys::cuptiEventGroupSetAttribute(
                    event_group.as_raw(),
                    attrib.into(),
                    value_size,
                    (&raw mut value).cast(),
                ))?;
            }
            Ok(())
        }
        _ => Err(Error::InvalidAttribute {
            name: format!("{attrib:?}"),
        }),
    }
}

/// Adds an event to a disabled event group.
///
/// The event must belong to a compatible event domain, and the device must be able to collect it with the group's existing events.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidEventId`] if the event ID is invalid.
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::OutOfMemory`] if CUPTI cannot allocate required internal storage.
/// - Returns [`crate::error::Status::InvalidOperation`] if the event group is currently enabled.
/// - Returns [`crate::error::Status::InvalidParameter`] if the event group handle is invalid.
/// - Returns [`crate::error::Status::MaxLimitReached`] if the event group is full.
/// - Returns [`crate::error::Status::NotCompatible`] if the event cannot be collected with the group's existing events.
pub fn add_event_group_event(event_group: &EventGroup, event: EventId) -> Result<()> {
    unsafe {
        try_ffi!(sys::cuptiEventGroupAddEvent(
            event_group.as_raw(),
            event.as_raw(),
        ))?;
    }
    Ok(())
}

/// Removes an event from a disabled event group.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidEventId`] if the event ID is invalid.
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidOperation`] if the event group is currently enabled.
/// - Returns [`crate::error::Status::InvalidParameter`] if the event group handle is invalid.
pub fn remove_event_group_event(event_group: &EventGroup, event: EventId) -> Result<()> {
    unsafe {
        try_ffi!(sys::cuptiEventGroupRemoveEvent(
            event_group.as_raw(),
            event.as_raw(),
        ))?;
    }
    Ok(())
}

/// Removes all events from a disabled event group.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidOperation`] if the event group is currently enabled.
/// - Returns [`crate::error::Status::InvalidParameter`] if the event group handle is invalid.
pub fn remove_all_event_group_events(event_group: &EventGroup) -> Result<()> {
    unsafe {
        try_ffi!(sys::cuptiEventGroupRemoveAllEvents(event_group.as_raw()))?;
    }
    Ok(())
}

/// Resets every event counter in an event group to zero.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidParameter`] if the event group handle is invalid.
pub fn reset_all_event_group_events(event_group: &EventGroup) -> Result<()> {
    unsafe {
        try_ffi!(sys::cuptiEventGroupResetAllEvents(event_group.as_raw()))?;
    }
    Ok(())
}

/// Enables one event group and starts collecting its events.
///
/// # Errors
///
/// - Returns [`crate::error::Status::Hardware`] or [`crate::error::Status::HardwareBusy`] if profiling hardware cannot be used.
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidOperation`] if the event group is already enabled or not ready.
/// - Returns [`crate::error::Status::InvalidParameter`] if the event group handle is invalid.
/// - Returns [`crate::error::Status::NotCompatible`] if another enabled event group conflicts with this group.
pub fn enable_event_group(event_group: &EventGroup) -> Result<()> {
    unsafe {
        try_ffi!(sys::cuptiEventGroupEnable(event_group.as_raw()))?;
    }
    Ok(())
}

/// Disables one event group and stops collecting its events.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidOperation`] if the event group is already disabled or invalid in the current state.
/// - Returns [`crate::error::Status::InvalidParameter`] if the event group handle is invalid.
pub fn disable_event_group(event_group: &EventGroup) -> Result<()> {
    unsafe {
        try_ffi!(sys::cuptiEventGroupDisable(event_group.as_raw()))?;
    }
    Ok(())
}

pub fn read_event_group_event(
    event_group: &EventGroup,
    flags: ReadEventFlags,
    event: EventId,
) -> Result<Vec<u64>> {
    let mut value_size_bytes = 0;
    unsafe {
        try_ffi!(sys::cuptiEventGroupReadEvent(
            event_group.as_raw(),
            flags.into(),
            event.as_raw(),
            &mut value_size_bytes,
            ptr::null_mut(),
        ))?;
    }
    let len = to_usize(value_size_bytes, "event_value_buffer_size_bytes")? / size_of::<u64>();
    let mut values = vec![0; len];
    unsafe {
        try_ffi!(sys::cuptiEventGroupReadEvent(
            event_group.as_raw(),
            flags.into(),
            event.as_raw(),
            &mut value_size_bytes,
            values.as_mut_ptr(),
        ))?;
    }
    let len = to_usize(value_size_bytes, "event_value_buffer_size_bytes")? / size_of::<u64>();
    values.truncate(len);
    Ok(values)
}

pub fn read_all_event_group_events(
    event_group: &EventGroup,
    flags: ReadEventFlags,
) -> Result<EventGroupReadAll> {
    let mut value_size_bytes = 0;
    let mut event_id_size_bytes = 0;
    let mut num_event_ids_read = 0;
    unsafe {
        try_ffi!(sys::cuptiEventGroupReadAllEvents(
            event_group.as_raw(),
            flags.into(),
            &mut value_size_bytes,
            ptr::null_mut(),
            &mut event_id_size_bytes,
            ptr::null_mut(),
            &mut num_event_ids_read,
        ))?;
    }
    let value_len = to_usize(value_size_bytes, "event_value_buffer_size_bytes")? / size_of::<u64>();
    let event_id_len = to_usize(event_id_size_bytes, "event_id_array_size_bytes")?
        / size_of::<sys::CUpti_EventID>();
    let mut values = vec![0; value_len];
    let mut event_ids = vec![0; event_id_len];
    unsafe {
        try_ffi!(sys::cuptiEventGroupReadAllEvents(
            event_group.as_raw(),
            flags.into(),
            &mut value_size_bytes,
            values.as_mut_ptr(),
            &mut event_id_size_bytes,
            event_ids.as_mut_ptr(),
            &mut num_event_ids_read,
        ))?;
    }
    let value_len = to_usize(value_size_bytes, "event_value_buffer_size_bytes")? / size_of::<u64>();
    let event_id_len = to_usize(num_event_ids_read, "num_event_ids_read")?;
    values.truncate(value_len);
    event_ids.truncate(event_id_len);
    Ok(EventGroupReadAll {
        event_ids: event_ids.into_iter().map(EventId::from).collect(),
        values,
    })
}

/// Registers a process-wide callback for kernel replay progress updates.
///
/// The callback receives the kernel name and the number of completed replays.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidParameter`] if CUPTI rejects the callback registration.
pub fn subscribe_kernel_replay_update<F>(callback: F) -> Result<()>
where
    F: FnMut(&CStr, i32) + Send + 'static,
{
    *KERNEL_REPLAY_UPDATE_CALLBACK
        .lock()
        .map_err(|_| Error::LockPoisoned {
            name: "kernel replay update callback".into(),
        })? = Some(Box::new(callback));
    unsafe {
        try_ffi!(sys::cuptiKernelReplaySubscribeUpdate(
            Some(kernel_replay_update_trampoline),
            ptr::null_mut(),
        ))?;
    }
    Ok(())
}

unsafe extern "C" fn kernel_replay_update_trampoline(
    kernel_name: *const c_char,
    num_replays_done: i32,
    _custom_data: *mut core::ffi::c_void,
) {
    if kernel_name.is_null() {
        return;
    }

    let _ = catch_unwind(AssertUnwindSafe(|| {
        if let Ok(mut callback) = KERNEL_REPLAY_UPDATE_CALLBACK.lock()
            && let Some(callback) = callback.as_mut()
        {
            callback(unsafe { CStr::from_ptr(kernel_name) }, num_replays_done);
        }
    }));
}

/// Returns the number of metrics available on any CUDA-capable device.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidParameter`] if CUPTI rejects the metric count output parameter.
pub fn get_metric_count() -> Result<u32> {
    let mut num_metrics = 0;
    unsafe {
        try_ffi!(sys::cuptiGetNumMetrics(&mut num_metrics))?;
    }
    Ok(num_metrics)
}

/// Returns all metrics available on any CUDA-capable device.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidParameter`] if CUPTI rejects the request.
pub fn get_enum_metrics() -> Result<Vec<MetricId>> {
    let raw = query_array(|array_size_bytes, metric_array| unsafe {
        sys::cuptiEnumMetrics(array_size_bytes, metric_array)
    })?;
    Ok(raw.into_iter().map(MetricId::from).collect())
}

/// Returns the number of metrics available on a CUDA device.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidDevice`] if the CUDA device is invalid.
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidParameter`] if CUPTI rejects the request.
pub fn get_device_metric_count(device: Device) -> Result<u32> {
    let mut num_metrics = 0;
    unsafe {
        try_ffi!(sys::cuptiDeviceGetNumMetrics(
            device.id() as sys::CUdevice,
            &mut num_metrics,
        ))?;
    }
    Ok(num_metrics)
}

/// Returns the metrics available on a CUDA device.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidDevice`] if the CUDA device is invalid.
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
pub fn get_device_metrics(device: Device) -> Result<Vec<MetricId>> {
    let raw = query_array(|array_size_bytes, metric_array| unsafe {
        sys::cuptiDeviceEnumMetrics(device.id() as sys::CUdevice, array_size_bytes, metric_array)
    })?;
    Ok(raw.into_iter().map(MetricId::from).collect())
}

/// Reads a metric attribute.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidMetricId`] if the metric ID is invalid.
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidParameter`] if the requested metric attribute is invalid.
/// - Returns [`crate::error::Status::ParameterSizeNotSufficient`] if the decoded value buffer is too small for the requested attribute.
pub fn get_metric_attribute_value(
    metric: MetricId,
    attrib: MetricAttribute,
) -> Result<MetricAttributeValue> {
    let mut read = |value_size: *mut u64, value: *mut ()| unsafe {
        try_ffi!(sys::cuptiMetricGetAttribute(
            metric.as_raw(),
            attrib.into(),
            value_size,
            value.cast(),
        ))?;
        Ok(())
    };
    match attrib {
        MetricAttribute::Name => Ok(MetricAttributeValue::Name(read_attribute_string(
            &mut read,
            "metric_attribute_size",
        )?)),
        MetricAttribute::ShortDescription => Ok(MetricAttributeValue::ShortDescription(
            read_attribute_string(&mut read, "metric_attribute_size")?,
        )),
        MetricAttribute::LongDescription => Ok(MetricAttributeValue::LongDescription(
            read_attribute_string(&mut read, "metric_attribute_size")?,
        )),
        MetricAttribute::Category => Ok(MetricAttributeValue::Category(MetricCategory::from(
            read_attribute_scalar::<sys::CUpti_MetricCategory>(&mut read, "metric_attribute_size")?,
        ))),
        MetricAttribute::ValueKind => Ok(MetricAttributeValue::ValueKind(MetricValueKind::from(
            read_attribute_scalar::<sys::CUpti_MetricValueKind>(
                &mut read,
                "metric_attribute_size",
            )?,
        ))),
        MetricAttribute::EvaluationMode => Ok(MetricAttributeValue::EvaluationMode(
            MetricEvaluationMode::from(read_attribute_scalar::<sys::CUpti_MetricEvaluationMode>(
                &mut read,
                "metric_attribute_size",
            )?),
        )),
        MetricAttribute::ForceInt => Err(Error::InvalidAttribute {
            name: format!("{attrib:?}"),
        }),
    }
}

/// Finds a metric ID by name for a CUDA device.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidDevice`] if the CUDA device is invalid.
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidMetricName`] if no metric with `metric_name` exists on the device.
/// - Returns [`crate::error::Status::InvalidParameter`] if CUPTI rejects the metric name.
pub fn get_metric_id_from_name(device: Device, metric_name: &CStr) -> Result<MetricId> {
    let mut metric = 0u32;
    unsafe {
        try_ffi!(sys::cuptiMetricGetIdFromName(
            device.id() as sys::CUdevice,
            metric_name.as_ptr(),
            &mut metric,
        ))?;
    }
    Ok(MetricId::from(metric))
}

pub fn get_metric_id_from_str(device: Device, metric_name: &str) -> Result<MetricId> {
    let metric_name = CString::new(metric_name)?;
    get_metric_id_from_name(device, &metric_name)
}

/// Returns the number of events required to calculate a metric.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidMetricId`] if the metric ID is invalid.
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
pub fn get_metric_event_count(metric: MetricId) -> Result<u32> {
    let mut num_events = 0;
    unsafe {
        try_ffi!(sys::cuptiMetricGetNumEvents(
            metric.as_raw(),
            &mut num_events,
        ))?;
    }
    Ok(num_events)
}

/// Returns the events required to calculate a metric.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidMetricId`] if the metric ID is invalid.
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidParameter`] if CUPTI rejects the request.
pub fn get_metric_enum_events(metric: MetricId) -> Result<Vec<EventId>> {
    let raw = query_array(|array_size_bytes, event_array| unsafe {
        sys::cuptiMetricEnumEvents(metric.as_raw(), array_size_bytes, event_array)
    })?;
    Ok(raw.into_iter().map(EventId::from).collect())
}

/// Returns the number of properties required to calculate a metric.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidMetricId`] if the metric ID is invalid.
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidParameter`] if CUPTI rejects the request.
pub fn get_metric_property_count(metric: MetricId) -> Result<u32> {
    let mut num_properties = 0;
    unsafe {
        try_ffi!(sys::cuptiMetricGetNumProperties(
            metric.as_raw(),
            &mut num_properties,
        ))?;
    }
    Ok(num_properties)
}

/// Returns the properties required to calculate a metric.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidMetricId`] if the metric ID is invalid.
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidParameter`] if CUPTI rejects the request.
pub fn get_metric_enum_properties(metric: MetricId) -> Result<Vec<MetricPropertyId>> {
    let raw = query_array(|array_size_bytes, property_array| unsafe {
        sys::cuptiMetricEnumProperties(metric.as_raw(), array_size_bytes, property_array)
    })?;
    Ok(raw.into_iter().map(MetricPropertyId::from).collect())
}

/// Calculates a metric value from event values collected on a CUDA device.
///
/// `event_ids` and `event_values` must describe the same event samples, and `duration` is the measured interval in nanoseconds.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidDevice`] if the CUDA device is invalid.
/// - Returns [`crate::error::Status::InvalidMetricId`] if the metric ID is invalid.
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidEventValue`] if the event values are not valid for the metric.
/// - Returns [`crate::error::Status::InvalidMetricValue`] if CUPTI cannot calculate the metric value.
/// - Returns [`crate::error::Status::InvalidParameter`] if the event IDs, event values, or CUPTI rejects the request.
pub fn get_metric_value(
    device: Device,
    metric: MetricId,
    event_ids: &[EventId],
    event_values: &[u64],
    duration: u64,
) -> Result<MetricValue> {
    let event_ids_size = checked_byte_len::<EventId>(event_ids.len(), "event_ids_size_bytes")?;
    let event_values_size = checked_byte_len::<u64>(event_values.len(), "event_values_size_bytes")?;
    let mut metric_value = sys::CUpti_MetricValue::default();
    let mut event_ids = event_ids
        .iter()
        .copied()
        .map(EventId::as_raw)
        .collect::<Vec<_>>();
    let mut event_values = event_values.to_vec();
    unsafe {
        try_ffi!(sys::cuptiMetricGetValue(
            device.id() as sys::CUdevice,
            metric.as_raw(),
            to_u64(event_ids_size, "event_ids_size_bytes")?,
            event_ids.as_mut_ptr(),
            to_u64(event_values_size, "event_values_size_bytes")?,
            event_values.as_mut_ptr(),
            duration,
            &mut metric_value,
        ))?;
    }
    metric_value_from_raw(metric, metric_value)
}

/// Calculates a metric value from event values and explicit metric property values.
///
/// `event_ids`, `event_values`, `property_ids`, and `property_values` must match the data required by the metric.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidMetricId`] if the metric ID is invalid.
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidEventValue`] if the event values are not valid for the metric.
/// - Returns [`crate::error::Status::InvalidMetricValue`] if CUPTI cannot calculate the metric value.
/// - Returns [`crate::error::Status::InvalidParameter`] if the event or property arrays are invalid.
pub fn get_metric_value2(
    metric: MetricId,
    event_ids: &[EventId],
    event_values: &[u64],
    property_ids: &[MetricPropertyId],
    property_values: &[u64],
) -> Result<MetricValue> {
    let event_ids_size = checked_byte_len::<EventId>(event_ids.len(), "event_ids_size_bytes")?;
    let event_values_size = checked_byte_len::<u64>(event_values.len(), "event_values_size_bytes")?;
    let property_ids_size =
        checked_byte_len::<MetricPropertyId>(property_ids.len(), "property_ids_size_bytes")?;
    let property_values_size =
        checked_byte_len::<u64>(property_values.len(), "property_values_size_bytes")?;
    let mut property_ids = property_ids
        .iter()
        .copied()
        .map(sys::CUpti_MetricPropertyID::from)
        .collect::<Vec<_>>();
    let mut event_ids = event_ids
        .iter()
        .copied()
        .map(EventId::as_raw)
        .collect::<Vec<_>>();
    let mut event_values = event_values.to_vec();
    let mut property_values = property_values.to_vec();
    let mut metric_value = sys::CUpti_MetricValue::default();
    unsafe {
        try_ffi!(sys::cuptiMetricGetValue2(
            metric.as_raw(),
            to_u64(event_ids_size, "event_ids_size_bytes")?,
            event_ids.as_mut_ptr(),
            to_u64(event_values_size, "event_values_size_bytes")?,
            event_values.as_mut_ptr(),
            to_u64(property_ids_size, "property_ids_size_bytes")?,
            property_ids.as_mut_ptr(),
            to_u64(property_values_size, "property_values_size_bytes")?,
            property_values.as_mut_ptr(),
            &mut metric_value,
        ))?;
    }
    metric_value_from_raw(metric, metric_value)
}

fn metric_value_from_raw(metric: MetricId, value: sys::CUpti_MetricValue) -> Result<MetricValue> {
    let MetricAttributeValue::ValueKind(kind) =
        get_metric_attribute_value(metric, MetricAttribute::ValueKind)?
    else {
        return Err(Error::InvalidAttribute {
            name: format!("{:?}", MetricAttribute::ValueKind),
        });
    };
    let value = unsafe {
        match kind {
            MetricValueKind::Double => MetricValue::Double(value.metricValueDouble),
            MetricValueKind::Uint64 => MetricValue::Uint64(value.metricValueUint64),
            MetricValueKind::Percent => MetricValue::Percent(value.metricValuePercent),
            MetricValueKind::Throughput => MetricValue::Throughput(value.metricValueThroughput),
            MetricValueKind::Int64 => MetricValue::Int64(value.metricValueInt64),
            MetricValueKind::UtilizationLevel => {
                MetricValue::UtilizationLevel(value.metricValueUtilizationLevel.into())
            }
            MetricValueKind::NvtxExtendedPayload => {
                MetricValue::NvtxExtendedPayload(value.metricValueNvtxExtendedPayload)
            }
            MetricValueKind::ForceInt => MetricValue::Uint64(value.metricValueUint64),
        }
    };
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::{ffi::c_void, mem, ptr};

    #[test]
    fn string_attribute_reader_trims_at_nul() -> Result<()> {
        let value = read_attribute_string(
            |value_size, value| {
                let bytes = b"metric-name\0ignored";
                unsafe {
                    bytes
                        .as_ptr()
                        .copy_to_nonoverlapping(value.cast::<u8>(), bytes.len());
                    *value_size = bytes.len() as u64;
                }
                Ok(())
            },
            "test_attribute_size",
        )?;

        assert_eq!(value, "metric-name");
        Ok(())
    }

    #[test]
    fn event_domain_attribute_reader_maps_collection_method() -> Result<()> {
        let value = event_domain_attribute_value_with(
            &mut |value_size, value| {
                let mut raw = sys::CUpti_EventCollectionMethod::CUPTI_EVENT_COLLECTION_METHOD_SM;
                unsafe {
                    *value_size = mem::size_of_val(&raw) as u64;
                    ptr::copy_nonoverlapping(&mut raw as *mut _, value.cast(), 1);
                }
                Ok(())
            },
            EventDomainAttribute::CollectionMethod,
        )?;

        assert_eq!(
            value,
            EventDomainAttributeValue::CollectionMethod(EventCollectionMethod::Sm)
        );
        Ok(())
    }

    #[test]
    fn event_group_setter_rejects_unsupported_attributes_before_ffi() {
        let event_group = EventGroup {
            handle: ptr::NonNull::<c_void>::dangling().as_ptr(),
        };
        let error = set_event_group_attribute_value(
            &event_group,
            EventGroupAttribute::UserData,
            EventGroupAttributeSetting::ProfileAllDomainInstances(true),
        )
        .unwrap_err();

        mem::forget(event_group);
        assert!(matches!(error, Error::InvalidAttribute { .. }));
    }
}

use std::{
    ffi::{CStr, CString},
    marker::PhantomData,
};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cuda_sys::nvtx as sys;

use crate::error::{Error, Result};

// TODO: move to a core version type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Version {
    pub major: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Color(u32);

impl Color {
    pub const fn argb(value: u32) -> Self {
        Self(value)
    }

    pub const fn rgba(red: u8, green: u8, blue: u8, alpha: u8) -> Self {
        Self(((alpha as u32) << 24) | ((red as u32) << 16) | ((green as u32) << 8) | blue as u32)
    }

    pub const fn as_raw(self) -> u32 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Category(u32);

impl Category {
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }

    pub const fn as_raw(self) -> u32 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ColorType {
    Unknown = sys::nvtxColorType_t::NVTX_COLOR_UNKNOWN as _,
    Argb = sys::nvtxColorType_t::NVTX_COLOR_ARGB as _,
}

impl_enum_conversion!(sys::nvtxColorType_t, ColorType);

impl_enum_display!(ColorType, {
    Self::Unknown => "NVTX_COLOR_UNKNOWN",
    Self::Argb => "NVTX_COLOR_ARGB",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum MessageType {
    Unknown = sys::nvtxMessageType_t::NVTX_MESSAGE_UNKNOWN as _,
    Ascii = sys::nvtxMessageType_t::NVTX_MESSAGE_TYPE_ASCII as _,
    Unicode = sys::nvtxMessageType_t::NVTX_MESSAGE_TYPE_UNICODE as _,
    Registered = sys::nvtxMessageType_t::NVTX_MESSAGE_TYPE_REGISTERED as _,
}

impl_enum_conversion!(sys::nvtxMessageType_t, MessageType);

impl_enum_display!(MessageType, {
    Self::Unknown => "NVTX_MESSAGE_UNKNOWN",
    Self::Ascii => "NVTX_MESSAGE_TYPE_ASCII",
    Self::Unicode => "NVTX_MESSAGE_TYPE_UNICODE",
    Self::Registered => "NVTX_MESSAGE_TYPE_REGISTERED",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum PayloadType {
    Unknown = sys::nvtxPayloadType_t::NVTX_PAYLOAD_UNKNOWN as _,
    UnsignedInt64 = sys::nvtxPayloadType_t::NVTX_PAYLOAD_TYPE_UNSIGNED_INT64 as _,
    Int64 = sys::nvtxPayloadType_t::NVTX_PAYLOAD_TYPE_INT64 as _,
    Double = sys::nvtxPayloadType_t::NVTX_PAYLOAD_TYPE_DOUBLE as _,
    UnsignedInt32 = sys::nvtxPayloadType_t::NVTX_PAYLOAD_TYPE_UNSIGNED_INT32 as _,
    Int32 = sys::nvtxPayloadType_t::NVTX_PAYLOAD_TYPE_INT32 as _,
    Float = sys::nvtxPayloadType_t::NVTX_PAYLOAD_TYPE_FLOAT as _,
}

impl_enum_conversion!(sys::nvtxPayloadType_t, PayloadType);

impl_enum_display!(PayloadType, {
    Self::Unknown => "NVTX_PAYLOAD_UNKNOWN",
    Self::UnsignedInt64 => "NVTX_PAYLOAD_TYPE_UNSIGNED_INT64",
    Self::Int64 => "NVTX_PAYLOAD_TYPE_INT64",
    Self::Double => "NVTX_PAYLOAD_TYPE_DOUBLE",
    Self::UnsignedInt32 => "NVTX_PAYLOAD_TYPE_UNSIGNED_INT32",
    Self::Int32 => "NVTX_PAYLOAD_TYPE_INT32",
    Self::Float => "NVTX_PAYLOAD_TYPE_FLOAT",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ResourceGenericType {
    Unknown = sys::nvtxResourceGenericType_t::NVTX_RESOURCE_TYPE_UNKNOWN as _,
    GenericPointer = sys::nvtxResourceGenericType_t::NVTX_RESOURCE_TYPE_GENERIC_POINTER as _,
    GenericHandle = sys::nvtxResourceGenericType_t::NVTX_RESOURCE_TYPE_GENERIC_HANDLE as _,
    GenericThreadNative =
        sys::nvtxResourceGenericType_t::NVTX_RESOURCE_TYPE_GENERIC_THREAD_NATIVE as _,
    GenericThreadPosix =
        sys::nvtxResourceGenericType_t::NVTX_RESOURCE_TYPE_GENERIC_THREAD_POSIX as _,
}

impl_enum_conversion!(sys::nvtxResourceGenericType_t, ResourceGenericType);

impl_enum_display!(ResourceGenericType, {
    Self::Unknown => "NVTX_RESOURCE_TYPE_UNKNOWN",
    Self::GenericPointer => "NVTX_RESOURCE_TYPE_GENERIC_POINTER",
    Self::GenericHandle => "NVTX_RESOURCE_TYPE_GENERIC_HANDLE",
    Self::GenericThreadNative => "NVTX_RESOURCE_TYPE_GENERIC_THREAD_NATIVE",
    Self::GenericThreadPosix => "NVTX_RESOURCE_TYPE_GENERIC_THREAD_POSIX",
});

#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Payload {
    I32(i32),
    I64(i64),
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
}

impl Payload {
    fn encode_type(self) -> sys::nvtxPayloadType_t {
        match self {
            Self::I32(_) => PayloadType::Int32.into(),
            Self::I64(_) => PayloadType::Int64.into(),
            Self::U32(_) => PayloadType::UnsignedInt32.into(),
            Self::U64(_) => PayloadType::UnsignedInt64.into(),
            Self::F32(_) => PayloadType::Float.into(),
            Self::F64(_) => PayloadType::Double.into(),
        }
    }

    fn encode_value(self) -> sys::nvtxEventAttributes_v2_payload_t {
        match self {
            Self::I32(value) => sys::nvtxEventAttributes_v2_payload_t { iValue: value },
            Self::I64(value) => sys::nvtxEventAttributes_v2_payload_t { llValue: value },
            Self::U32(value) => sys::nvtxEventAttributes_v2_payload_t { uiValue: value },
            Self::U64(value) => sys::nvtxEventAttributes_v2_payload_t { ullValue: value },
            Self::F32(value) => sys::nvtxEventAttributes_v2_payload_t { fValue: value },
            Self::F64(value) => sys::nvtxEventAttributes_v2_payload_t { dValue: value },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EventAttributes<'a> {
    message: Option<&'a CStr>,
    category: Option<Category>,
    color: Option<Color>,
    payload: Option<Payload>,
}

impl<'a> EventAttributes<'a> {
    pub const fn new() -> Self {
        Self {
            message: None,
            category: None,
            color: None,
            payload: None,
        }
    }

    pub fn with_message(mut self, message: &'a CStr) -> Self {
        self.message = Some(message);
        self
    }

    pub fn with_category(mut self, category: Category) -> Self {
        self.category = Some(category);
        self
    }

    pub fn with_color(mut self, color: Color) -> Self {
        self.color = Some(color);
        self
    }

    pub fn with_payload(mut self, payload: Payload) -> Self {
        self.payload = Some(payload);
        self
    }

    pub const fn message(&self) -> Option<&'a CStr> {
        self.message
    }

    pub const fn category(&self) -> Option<Category> {
        self.category
    }

    pub const fn color(&self) -> Option<Color> {
        self.color
    }

    pub const fn payload(&self) -> Option<Payload> {
        self.payload
    }

    fn encode(self) -> sys::nvtxEventAttributes_t {
        let mut raw = sys::nvtxEventAttributes_t {
            version: sys::NVTX_VERSION as u16,
            size: size_of::<sys::nvtxEventAttributes_t>() as u16,
            ..Default::default()
        };

        if let Some(category) = self.category {
            raw.category = category.0;
        }

        if let Some(color) = self.color {
            raw.colorType = sys::nvtxColorType_t::from(ColorType::Argb) as i32;
            raw.color = color.0;
        }

        if let Some(payload) = self.payload {
            raw.payloadType = payload.encode_type() as i32;
            raw.payload = payload.encode_value();
        }

        if let Some(message) = self.message {
            raw.messageType = sys::nvtxMessageType_t::from(MessageType::Ascii) as i32;
            raw.message.ascii = message.as_ptr();
        }

        raw
    }
}

impl Default for EventAttributes<'_> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct Event {
    message: CString,
    category: Option<Category>,
    color: Option<Color>,
    payload: Option<Payload>,
}

impl Event {
    pub fn create(message: &str) -> Result<Self> {
        Ok(Self {
            message: CString::new(message)?,
            category: None,
            color: None,
            payload: None,
        })
    }

    pub fn create_from_c_string(message: CString) -> Self {
        Self {
            message,
            category: None,
            color: None,
            payload: None,
        }
    }

    pub fn with_category(mut self, category: Category) -> Self {
        self.category = Some(category);
        self
    }

    pub fn with_color(mut self, color: Color) -> Self {
        self.color = Some(color);
        self
    }

    pub fn with_payload(mut self, payload: Payload) -> Self {
        self.payload = Some(payload);
        self
    }

    pub fn mark(&self) {
        mark_with_attributes(self.attributes());
    }

    pub fn local_range(&self) -> LocalRange {
        LocalRange::from_attributes(self.attributes())
    }

    pub fn range(&self) -> Range {
        Range::from_attributes(self.attributes())
    }

    pub fn domain_mark(&self, domain: &Domain) {
        domain.mark_with_attributes(self.attributes());
    }

    pub fn domain_local_range<'a>(&self, domain: &'a Domain) -> DomainLocalRange<'a> {
        domain.range_with_attributes(self.attributes())
    }

    pub fn domain_range<'a>(&self, domain: &'a Domain) -> DomainRange<'a> {
        domain.start_range_with_attributes(self.attributes())
    }

    pub fn attributes(&self) -> EventAttributes<'_> {
        let mut attributes = EventAttributes::new().with_message(&self.message);

        if let Some(category) = self.category {
            attributes = attributes.with_category(category);
        }

        if let Some(color) = self.color {
            attributes = attributes.with_color(color);
        }

        if let Some(payload) = self.payload {
            attributes = attributes.with_payload(payload);
        }

        attributes
    }
}

#[derive(Debug)]
pub struct Domain {
    handle: sys::nvtxDomainHandle_t,
}

// NVTX domains are process-wide annotation handles. The wrapper owns the handle
// and only passes immutable copies to NVTX entry points.
unsafe impl Send for Domain {}
unsafe impl Sync for Domain {}

impl Domain {
    pub fn create(name: &str) -> Result<Self> {
        let name = CString::new(name)?;
        Self::create_from_c_str(&name)
    }

    pub fn create_from_c_str(name: &CStr) -> Result<Self> {
        let handle = unsafe { sys::nvtxDomainCreateA(name.as_ptr()) };
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self { handle })
    }

    pub fn as_raw(&self) -> sys::nvtxDomainHandle_t {
        self.handle
    }

    pub fn mark(&self, message: &str) -> Result<()> {
        let message = CString::new(message)?;
        self.mark_c_str(&message);
        Ok(())
    }

    pub fn mark_c_str(&self, message: &CStr) {
        self.mark_with_attributes(EventAttributes::new().with_message(message));
    }

    pub fn mark_with_attributes(&self, attributes: EventAttributes<'_>) {
        let raw = attributes.encode();
        unsafe { sys::nvtxDomainMarkEx(self.handle, &raw) };
    }

    pub fn range<'a>(&'a self, message: &str) -> Result<DomainLocalRange<'a>> {
        let message = CString::new(message)?;
        Ok(self.range_c_str(&message))
    }

    pub fn range_c_str<'a>(&'a self, message: &CStr) -> DomainLocalRange<'a> {
        self.range_with_attributes(EventAttributes::new().with_message(message))
    }

    pub fn range_with_attributes<'a>(
        &'a self,
        attributes: EventAttributes<'_>,
    ) -> DomainLocalRange<'a> {
        let raw = attributes.encode();
        unsafe { sys::nvtxDomainRangePushEx(self.handle, &raw) };
        DomainLocalRange {
            domain: self,
            _not_send: PhantomData,
        }
    }

    pub fn start_range(&self, message: &str) -> Result<DomainRange<'_>> {
        let message = CString::new(message)?;
        Ok(self.start_range_c_str(&message))
    }

    pub fn start_range_c_str(&self, message: &CStr) -> DomainRange<'_> {
        self.start_range_with_attributes(EventAttributes::new().with_message(message))
    }

    pub fn start_range_with_attributes(&self, attributes: EventAttributes<'_>) -> DomainRange<'_> {
        let raw = attributes.encode();
        let id = unsafe { sys::nvtxDomainRangeStartEx(self.handle, &raw) };
        DomainRange { domain: self, id }
    }

    pub fn name_category(&self, category: Category, name: &str) -> Result<()> {
        let name = CString::new(name)?;
        unsafe { sys::nvtxDomainNameCategoryA(self.handle, category.0, name.as_ptr()) };
        Ok(())
    }
}

impl Drop for Domain {
    fn drop(&mut self) {
        unsafe { sys::nvtxDomainDestroy(self.handle) };
    }
}

#[derive(Debug)]
pub struct LocalRange {
    _not_send: PhantomData<*mut ()>,
}

impl LocalRange {
    pub fn create(message: &str) -> Result<Self> {
        let message = CString::new(message)?;
        Ok(Self::create_from_c_str(&message))
    }

    pub fn create_from_c_str(message: &CStr) -> Self {
        unsafe { sys::nvtxRangePushA(message.as_ptr()) };
        Self {
            _not_send: PhantomData,
        }
    }

    pub fn from_attributes(attributes: EventAttributes<'_>) -> Self {
        let raw = attributes.encode();
        unsafe { sys::nvtxRangePushEx(&raw) };
        Self {
            _not_send: PhantomData,
        }
    }
}

impl Drop for LocalRange {
    fn drop(&mut self) {
        unsafe { sys::nvtxRangePop() };
    }
}

#[derive(Debug)]
pub struct Range {
    id: sys::nvtxRangeId_t,
}

impl Range {
    pub fn create(message: &str) -> Result<Self> {
        let message = CString::new(message)?;
        Ok(Self::create_from_c_str(&message))
    }

    pub fn create_from_c_str(message: &CStr) -> Self {
        let id = unsafe { sys::nvtxRangeStartA(message.as_ptr()) };
        Self { id }
    }

    pub fn from_attributes(attributes: EventAttributes<'_>) -> Self {
        let raw = attributes.encode();
        let id = unsafe { sys::nvtxRangeStartEx(&raw) };
        Self { id }
    }
}

impl Drop for Range {
    fn drop(&mut self) {
        unsafe { sys::nvtxRangeEnd(self.id) };
    }
}

#[derive(Debug)]
pub struct DomainLocalRange<'a> {
    domain: &'a Domain,
    _not_send: PhantomData<*mut ()>,
}

impl Drop for DomainLocalRange<'_> {
    fn drop(&mut self) {
        unsafe { sys::nvtxDomainRangePop(self.domain.handle) };
    }
}

#[derive(Debug)]
pub struct DomainRange<'a> {
    domain: &'a Domain,
    id: sys::nvtxRangeId_t,
}

impl Drop for DomainRange<'_> {
    fn drop(&mut self) {
        unsafe { sys::nvtxDomainRangeEnd(self.domain.handle, self.id) };
    }
}

pub fn version() -> Version {
    Version {
        major: sys::NVTX_VERSION,
    }
}

pub fn initialize() {
    unsafe { sys::nvtxInitialize(std::ptr::null()) };
}

pub fn mark(message: &str) -> Result<()> {
    Event::create(message)?.mark();
    Ok(())
}

pub fn mark_c_str(message: &CStr) {
    unsafe { sys::nvtxMarkA(message.as_ptr()) };
}

pub fn mark_with_attributes(attributes: EventAttributes<'_>) {
    let raw = attributes.encode();
    unsafe { sys::nvtxMarkEx(&raw) };
}

pub fn name_category(category: Category, name: &str) -> Result<()> {
    let name = CString::new(name)?;
    unsafe { sys::nvtxNameCategoryA(category.0, name.as_ptr()) };
    Ok(())
}

pub fn name_os_thread(thread_id: u32, name: &str) -> Result<()> {
    let name = CString::new(name)?;
    unsafe { sys::nvtxNameOsThreadA(thread_id, name.as_ptr()) };
    Ok(())
}

pub fn scoped_range(message: &str) -> Result<LocalRange> {
    LocalRange::create(message)
}

#[cfg(test)]
mod tests {
    use std::mem;

    use super::*;

    #[test]
    fn encodes_event_attributes() {
        let message = c"work";
        let raw = EventAttributes::new()
            .with_message(message)
            .with_category(Category::from_raw(7))
            .with_color(Color::rgba(1, 2, 3, 4))
            .with_payload(Payload::I64(-42))
            .encode();

        assert_eq!(raw.version, sys::NVTX_VERSION as u16);
        assert_eq!(
            raw.size,
            mem::size_of::<sys::nvtxEventAttributes_t>() as u16
        );
        assert_eq!(raw.category, 7);
        assert_eq!(raw.colorType, sys::nvtxColorType_t::NVTX_COLOR_ARGB as i32);
        assert_eq!(raw.color, 0x0401_0203);
        assert_eq!(
            raw.messageType,
            sys::nvtxMessageType_t::NVTX_MESSAGE_TYPE_ASCII as i32
        );
        assert_eq!(unsafe { raw.message.ascii }, message.as_ptr());
        assert_eq!(
            raw.payloadType,
            sys::nvtxPayloadType_t::NVTX_PAYLOAD_TYPE_INT64 as i32
        );
        assert_eq!(unsafe { raw.payload.llValue }, -42);
    }

    #[test]
    fn owned_event_builds_attributes() {
        let event = Event::create("owned")
            .unwrap()
            .with_category(Category::from_raw(3))
            .with_color(Color::argb(0xff00_00ff))
            .with_payload(Payload::U32(11));

        let attributes = event.attributes();
        let raw = attributes.encode();

        assert_eq!(attributes.message(), Some(c"owned".as_ref()));
        assert_eq!(attributes.category(), Some(Category::from_raw(3)));
        assert_eq!(attributes.color(), Some(Color::argb(0xff00_00ff)));
        assert_eq!(attributes.payload(), Some(Payload::U32(11)));
        assert_eq!(raw.category, 3);
        assert_eq!(raw.color, 0xff00_00ff);
        assert_eq!(
            raw.payloadType,
            sys::nvtxPayloadType_t::NVTX_PAYLOAD_TYPE_UNSIGNED_INT32 as i32
        );
        assert_eq!(unsafe { raw.payload.uiValue }, 11);
    }

    #[test]
    fn enum_wrappers_convert_and_display() {
        assert_eq!(
            ColorType::from(sys::nvtxColorType_t::NVTX_COLOR_ARGB),
            ColorType::Argb
        );
        assert_eq!(
            sys::nvtxMessageType_t::from(MessageType::Ascii),
            sys::nvtxMessageType_t::NVTX_MESSAGE_TYPE_ASCII
        );
        assert_eq!(
            PayloadType::UnsignedInt64.to_string(),
            "NVTX_PAYLOAD_TYPE_UNSIGNED_INT64"
        );
        assert_eq!(
            ResourceGenericType::GenericThreadPosix.to_string(),
            "NVTX_RESOURCE_TYPE_GENERIC_THREAD_POSIX"
        );
    }
}

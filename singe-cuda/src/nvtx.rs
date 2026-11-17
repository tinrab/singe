//! NVTX annotations, ranges, markers, and profiling resources.

use std::{
    ffi::{CStr, CString},
    marker::PhantomData,
    ptr,
};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{LibraryVersion, impl_enum_display};
use singe_cuda_sys::nvtx as sys;

use crate::error::{Error, Result};

/// A 32-bit annotation color in `0xAARRGGBB` order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Color(u32);

impl Color {
    /// Wraps an annotation color encoded as `0xAARRGGBB`.
    #[must_use]
    pub const fn argb(value: u32) -> Self {
        Self(value)
    }

    /// Packs the red, green, blue, and alpha bytes into native `0xAARRGGBB` order.
    #[must_use]
    pub const fn rgba(red: u8, green: u8, blue: u8, alpha: u8) -> Self {
        Self(((alpha as u32) << 24) | ((red as u32) << 16) | ((green as u32) << 8) | blue as u32)
    }

    /// Returns the packed native ARGB color value.
    #[must_use]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
}

/// An NVTX category ID used to group annotation events within a domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Category(u32);

impl Category {
    /// Wraps a category ID without registering a name or selecting a domain.
    #[must_use]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }

    /// Returns the native category ID.
    #[must_use]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
}

singe_core::native_enum! {
    /// Native discriminator selecting whether an annotation includes a color.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum ColorType: sys::nvtxColorType_t {
        /// Color attribute is unused.
        Unknown = NVTX_COLOR_UNKNOWN,
        /// An ARGB color is provided.
        Argb = NVTX_COLOR_ARGB,
    }
}

impl_enum_display!(ColorType, {
    Self::Unknown => "NVTX_COLOR_UNKNOWN",
    Self::Argb => "NVTX_COLOR_ARGB",
});

singe_core::native_enum! {
    /// Native discriminator selecting the annotation message representation.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum MessageType: sys::nvtxMessageType_t {
        /// Message attribute is unused.
        Unknown = NVTX_MESSAGE_UNKNOWN,
        /// A character sequence is used as payload.
        Ascii = NVTX_MESSAGE_TYPE_ASCII,
        /// A wide character sequence is used as payload.
        Unicode = NVTX_MESSAGE_TYPE_UNICODE,
        /// A unique string handle registered with a domain.
        Registered = NVTX_MESSAGE_TYPE_REGISTERED,
    }
}

impl_enum_display!(MessageType, {
    Self::Unknown => "NVTX_MESSAGE_UNKNOWN",
    Self::Ascii => "NVTX_MESSAGE_TYPE_ASCII",
    Self::Unicode => "NVTX_MESSAGE_TYPE_UNICODE",
    Self::Registered => "NVTX_MESSAGE_TYPE_REGISTERED",
});

singe_core::native_enum! {
    /// Native discriminator selecting the numeric annotation payload representation.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum PayloadType: sys::nvtxPayloadType_t {
        /// Payload attribute is unused.
        Unknown = NVTX_PAYLOAD_UNKNOWN,
        /// A 64 bit unsigned integer value is used as payload.
        UnsignedInt64 = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64,
        /// A 64 bit signed integer value is used as payload.
        Int64 = NVTX_PAYLOAD_TYPE_INT64,
        /// A 64 bit floating point value is used as payload.
        Double = NVTX_PAYLOAD_TYPE_DOUBLE,
        /// A 32-bit unsigned integer value is used as payload.
        UnsignedInt32 = NVTX_PAYLOAD_TYPE_UNSIGNED_INT32,
        /// A 32-bit signed integer value is used as payload.
        Int32 = NVTX_PAYLOAD_TYPE_INT32,
        /// A 32 bit floating point value is used as payload.
        Float = NVTX_PAYLOAD_TYPE_FLOAT,
    }
}

impl_enum_display!(PayloadType, {
    Self::Unknown => "NVTX_PAYLOAD_UNKNOWN",
    Self::UnsignedInt64 => "NVTX_PAYLOAD_TYPE_UNSIGNED_INT64",
    Self::Int64 => "NVTX_PAYLOAD_TYPE_INT64",
    Self::Double => "NVTX_PAYLOAD_TYPE_DOUBLE",
    Self::UnsignedInt32 => "NVTX_PAYLOAD_TYPE_UNSIGNED_INT32",
    Self::Int32 => "NVTX_PAYLOAD_TYPE_INT32",
    Self::Float => "NVTX_PAYLOAD_TYPE_FLOAT",
});

singe_core::native_enum! {
    /// Generic resource type for when a resource class is not available.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum ResourceGenericType: sys::nvtxResourceGenericType_t {
        /// No supported NVTX resource identifier type is selected.
        Unknown = NVTX_RESOURCE_TYPE_UNKNOWN,
        /// Generic pointer assumed to have no collisions with other pointers.
        GenericPointer = NVTX_RESOURCE_TYPE_GENERIC_POINTER,
        /// Generic handle assumed to have no collisions with other handles.
        GenericHandle = NVTX_RESOURCE_TYPE_GENERIC_HANDLE,
        /// OS native thread identifier.
        GenericThreadNative = NVTX_RESOURCE_TYPE_GENERIC_THREAD_NATIVE,
        /// POSIX pthread identifier.
        GenericThreadPosix = NVTX_RESOURCE_TYPE_GENERIC_THREAD_POSIX,
    }
}

impl_enum_display!(ResourceGenericType, {
    Self::Unknown => "NVTX_RESOURCE_TYPE_UNKNOWN",
    Self::GenericPointer => "NVTX_RESOURCE_TYPE_GENERIC_POINTER",
    Self::GenericHandle => "NVTX_RESOURCE_TYPE_GENERIC_HANDLE",
    Self::GenericThreadNative => "NVTX_RESOURCE_TYPE_GENERIC_THREAD_NATIVE",
    Self::GenericThreadPosix => "NVTX_RESOURCE_TYPE_GENERIC_THREAD_POSIX",
});

/// A numeric annotation payload and its native representation.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Payload {
    /// A signed 32-bit integer payload.
    I32(i32),
    /// A signed 64-bit integer payload.
    I64(i64),
    /// An unsigned 32-bit integer payload.
    U32(u32),
    /// An unsigned 64-bit integer payload.
    U64(u64),
    /// A 32-bit floating-point payload.
    F32(f32),
    /// A 64-bit floating-point payload.
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

/// Borrowed message and optional metadata for one NVTX marker or range.
///
/// A `&CStr`, `&CString`, or `&Event` converts into attributes with its message and metadata.
/// The message must remain live through the annotation call.
/// Unset color and payload fields use the native unknown selectors, and an unset category uses zero.
#[derive(Debug, Clone, Copy)]
pub struct EventAttributes<'a> {
    message: Option<&'a CStr>,
    category: Option<Category>,
    color: Option<Color>,
    payload: Option<Payload>,
}

impl<'a> EventAttributes<'a> {
    /// Creates attributes with no message, category, color, or payload.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            message: None,
            category: None,
            color: None,
            payload: None,
        }
    }

    /// Sets the borrowed C-string annotation message.
    #[must_use]
    pub fn with_message(mut self, message: &'a CStr) -> Self {
        self.message = Some(message);
        self
    }

    /// Sets the category ID interpreted within the receiving domain.
    #[must_use]
    pub fn with_category(mut self, category: Category) -> Self {
        self.category = Some(category);
        self
    }

    /// Sets the annotation color.
    #[must_use]
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = Some(color);
        self
    }

    /// Sets the numeric annotation payload.
    #[must_use]
    pub fn with_payload(mut self, payload: Payload) -> Self {
        self.payload = Some(payload);
        self
    }

    /// Returns the borrowed annotation message, if set.
    #[must_use]
    pub const fn message(&self) -> Option<&'a CStr> {
        self.message
    }

    /// Returns the annotation category, if set.
    #[must_use]
    pub const fn category(&self) -> Option<Category> {
        self.category
    }

    /// Returns the annotation color, if set.
    #[must_use]
    pub const fn color(&self) -> Option<Color> {
        self.color
    }

    /// Returns the numeric annotation payload, if set.
    #[must_use]
    pub const fn payload(&self) -> Option<Payload> {
        self.payload
    }

    fn encode(self) -> sys::nvtxEventAttributes_t {
        let mut raw = sys::nvtxEventAttributes_t {
            version: u16::try_from(sys::NVTX_VERSION).expect("the NVTX version fits its ABI field"),
            size: u16::try_from(size_of::<sys::nvtxEventAttributes_t>())
                .expect("the NVTX event attributes fit their ABI size field"),
            ..Default::default()
        };

        if let Some(category) = self.category {
            raw.category = category.0;
        }

        if let Some(color) = self.color {
            raw.colorType = sys::nvtxColorType_t::from(ColorType::Argb).0.cast_signed();
            raw.color = color.0;
        }

        if let Some(payload) = self.payload {
            raw.payloadType = payload.encode_type().0.cast_signed();
            raw.payload = payload.encode_value();
        }

        if let Some(message) = self.message {
            raw.messageType = sys::nvtxMessageType_t::from(MessageType::Ascii)
                .0
                .cast_signed();
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

impl<'a> From<&'a CStr> for EventAttributes<'a> {
    fn from(message: &'a CStr) -> Self {
        Self::new().with_message(message)
    }
}

impl<'a> From<&'a CString> for EventAttributes<'a> {
    fn from(message: &'a CString) -> Self {
        Self::new().with_message(message)
    }
}

impl<'a> From<&'a Event> for EventAttributes<'a> {
    fn from(event: &'a Event) -> Self {
        event.attributes()
    }
}

/// An owned C-string message and reusable NVTX annotation metadata.
///
/// A reference converts into [`EventAttributes`] for the marker and range functions.
#[derive(Debug, Clone)]
pub struct Event {
    message: CString,
    category: Option<Category>,
    color: Option<Color>,
    payload: Option<Payload>,
}

impl Event {
    /// Copies `message` into an owned C string with no optional annotation metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if `message` contains an interior NUL byte.
    pub fn create(message: &str) -> Result<Self> {
        Ok(Self::create_from_c_string(CString::new(message)?))
    }

    /// Takes ownership of a C-string message with no optional annotation metadata.
    #[must_use]
    pub fn create_from_c_string(message: CString) -> Self {
        Self {
            message,
            category: None,
            color: None,
            payload: None,
        }
    }

    /// Sets the category used by markers and ranges created from this event.
    #[must_use]
    pub fn with_category(mut self, category: Category) -> Self {
        self.category = Some(category);
        self
    }

    /// Sets the color used by markers and ranges created from this event.
    #[must_use]
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = Some(color);
        self
    }

    /// Sets the numeric payload used by markers and ranges created from this event.
    #[must_use]
    pub fn with_payload(mut self, payload: Payload) -> Self {
        self.payload = Some(payload);
        self
    }

    /// Borrows the message and copies the optional metadata into annotation attributes.
    #[must_use]
    pub fn attributes(&self) -> EventAttributes<'_> {
        EventAttributes {
            message: Some(&self.message),
            category: self.category,
            color: self.color,
            payload: self.payload,
        }
    }
}

/// An owned NVTX annotation domain.
///
/// A domain scopes category names and annotations and maintains its own per-thread range stacks.
/// Dropping this wrapper destroys the native domain; domain range guards borrow it.
#[derive(Debug)]
pub struct Domain {
    handle: sys::nvtxDomainHandle_t,
}

// SAFETY: an NVTX domain handle is process-wide and NVTX entry points are thread-safe, so the owner may move and destroy it on any thread.
unsafe impl Send for Domain {}

// SAFETY: shared methods only pass the handle by value to thread-safe NVTX entry points; per-thread range stacks are kept by NVTX itself.
unsafe impl Sync for Domain {}

impl Domain {
    /// Registers a named NVTX annotation domain.
    ///
    /// # Errors
    ///
    /// Returns an error if `name` contains an interior NUL byte or NVTX returns a null domain handle.
    pub fn create(name: &str) -> Result<Self> {
        let name = CString::new(name)?;
        Self::create_from_c_str(&name)
    }

    /// Registers an NVTX annotation domain using an existing C-string name.
    ///
    /// # Errors
    ///
    /// Returns an error if NVTX returns a null domain handle.
    pub fn create_from_c_str(name: &CStr) -> Result<Self> {
        // SAFETY: `name` is a NUL-terminated string borrowed for the call.
        let handle = unsafe { sys::nvtxDomainCreateA(name.as_ptr()) };
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self { handle })
    }

    /// Returns the borrowed native domain handle without transferring ownership.
    #[must_use]
    pub fn as_raw(&self) -> sys::nvtxDomainHandle_t {
        self.handle
    }

    /// Emits an instantaneous marker in this domain.
    ///
    /// `attributes` is a C-string message, an [`Event`], or [`EventAttributes`] with metadata.
    pub fn mark<'a>(&self, attributes: impl Into<EventAttributes<'a>>) {
        let raw = attributes.into().encode();
        // SAFETY: `self` owns the live domain, and `raw` and the message it points to are borrowed for the call.
        unsafe { sys::nvtxDomainMarkEx(self.handle, &raw const raw) };
    }

    /// Pushes a range onto this domain's current-thread stack and returns its pop-on-drop guard.
    ///
    /// `attributes` is a C-string message, an [`Event`], or [`EventAttributes`] with metadata.
    #[must_use]
    pub fn range<'a>(&self, attributes: impl Into<EventAttributes<'a>>) -> DomainLocalRange<'_> {
        let raw = attributes.into().encode();
        // SAFETY: `self` owns the live domain, and `raw` and the message it points to are borrowed for the call.
        unsafe { sys::nvtxDomainRangePushEx(self.handle, &raw const raw) };
        DomainLocalRange {
            domain: self,
            _not_send: PhantomData,
        }
    }

    /// Starts a process range in this domain and returns a guard that ends its native ID on drop.
    ///
    /// `attributes` is a C-string message, an [`Event`], or [`EventAttributes`] with metadata.
    #[must_use]
    pub fn start_range<'a>(&self, attributes: impl Into<EventAttributes<'a>>) -> DomainRange<'_> {
        let raw = attributes.into().encode();
        // SAFETY: `self` owns the live domain, and `raw` and the message it points to are borrowed for the call.
        let id = unsafe { sys::nvtxDomainRangeStartEx(self.handle, &raw const raw) };
        DomainRange { domain: self, id }
    }

    /// Assigns `name` to a category ID within this domain.
    ///
    /// # Errors
    ///
    /// Returns an error if `name` contains an interior NUL byte.
    pub fn name_category(&self, category: Category, name: &str) -> Result<()> {
        let name = CString::new(name)?;
        // SAFETY: `self` owns the live domain, and `name` is a NUL-terminated string borrowed for the call.
        unsafe { sys::nvtxDomainNameCategoryA(self.handle, category.0, name.as_ptr()) };
        Ok(())
    }
}

impl Drop for Domain {
    fn drop(&mut self) {
        // SAFETY: this wrapper owns the domain, and range guards borrow it, so none outlives the destroy.
        unsafe { sys::nvtxDomainDestroy(self.handle) };
    }
}

/// A guard for one range on the global domain's current-thread stack.
///
/// Dropping the guard pops the most recently pushed range on that thread.
/// Guards must be dropped in reverse push order and cannot move to another thread.
#[derive(Debug)]
pub struct LocalRange {
    _not_send: PhantomData<*mut ()>,
}

impl LocalRange {
    /// Pushes a range onto the global domain's current-thread stack and returns a guard that ends it on drop.
    ///
    /// `attributes` is a C-string message, an [`Event`], or [`EventAttributes`] with metadata.
    #[must_use]
    pub fn push<'a>(attributes: impl Into<EventAttributes<'a>>) -> Self {
        let raw = attributes.into().encode();
        // SAFETY: `raw` and the message it points to are borrowed for the call.
        unsafe { sys::nvtxRangePushEx(&raw const raw) };
        Self {
            _not_send: PhantomData,
        }
    }
}

impl Drop for LocalRange {
    fn drop(&mut self) {
        // SAFETY: the guard is `!Send`, so this pops the range it pushed on the current thread's stack.
        unsafe { sys::nvtxRangePop() };
    }
}

/// A guard for a process range in the global NVTX domain.
///
/// Dropping the guard ends the range by its native ID.
/// These ranges need not be nested and can end on another thread.
#[derive(Debug)]
pub struct Range {
    id: sys::nvtxRangeId_t,
}

impl Range {
    /// Starts a process range in the global NVTX domain and returns a guard that ends it on drop.
    ///
    /// `attributes` is a C-string message, an [`Event`], or [`EventAttributes`] with metadata.
    #[must_use]
    pub fn start<'a>(attributes: impl Into<EventAttributes<'a>>) -> Self {
        let raw = attributes.into().encode();
        // SAFETY: `raw` and the message it points to are borrowed for the call.
        let id = unsafe { sys::nvtxRangeStartEx(&raw const raw) };
        Self { id }
    }
}

impl Drop for Range {
    fn drop(&mut self) {
        // SAFETY: `self.id` is the range this guard started, ended exactly once here.
        unsafe { sys::nvtxRangeEnd(self.id) };
    }
}

/// A guard for a range on a domain's current-thread stack.
///
/// Dropping the guard pops that domain's most recently pushed range on the thread.
/// Guards must be dropped in reverse push order and cannot move to another thread.
#[derive(Debug)]
pub struct DomainLocalRange<'a> {
    domain: &'a Domain,
    _not_send: PhantomData<*mut ()>,
}

impl Drop for DomainLocalRange<'_> {
    fn drop(&mut self) {
        // SAFETY: the borrowed domain is live, and the `!Send` guard pops the range it pushed on this thread.
        unsafe { sys::nvtxDomainRangePop(self.domain.handle) };
    }
}

/// A process-range guard borrowing its NVTX domain.
///
/// Dropping the guard ends its native range ID.
/// These ranges need not be nested and can end on another thread.
#[derive(Debug)]
pub struct DomainRange<'a> {
    domain: &'a Domain,
    id: sys::nvtxRangeId_t,
}

impl Drop for DomainRange<'_> {
    fn drop(&mut self) {
        // SAFETY: the borrowed domain is live, and `self.id` is the range this guard started in it.
        unsafe { sys::nvtxDomainRangeEnd(self.domain.handle, self.id) };
    }
}

/// Returns the compile-time NVTX API version used by this crate.
#[must_use]
pub fn version() -> LibraryVersion {
    LibraryVersion::new(sys::NVTX_VERSION, 0, 0)
}

/// Initializes NVTX before the first annotation call.
///
/// Other NVTX calls initialize the API lazily if this function has not been called.
pub fn initialize() {
    // SAFETY: NVTX accepts a null reserved argument.
    unsafe { sys::nvtxInitialize(ptr::null()) };
}

/// Emits an instantaneous marker in the global NVTX domain.
///
/// `attributes` is a C-string message, an [`Event`], or [`EventAttributes`] with metadata.
pub fn mark<'a>(attributes: impl Into<EventAttributes<'a>>) {
    let raw = attributes.into().encode();
    // SAFETY: `raw` and the message it points to are borrowed for the call.
    unsafe { sys::nvtxMarkEx(&raw const raw) };
}

/// Assigns `name` to a category ID in the global NVTX domain.
///
/// # Errors
///
/// Returns an error if `name` contains an interior NUL byte.
pub fn name_category(category: Category, name: &str) -> Result<()> {
    let name = CString::new(name)?;
    // SAFETY: `name` is a NUL-terminated string borrowed for the call.
    unsafe { sys::nvtxNameCategoryA(category.0, name.as_ptr()) };
    Ok(())
}

/// Assigns `name` to an active operating-system thread in the current process.
///
/// Use the operating system's native numeric thread ID.
/// A POSIX `pthread_t` need not have the required representation.
/// Tool behavior is unspecified for invalid IDs or IDs from another process.
///
/// # Errors
///
/// Returns an error if `name` contains an interior NUL byte.
/// The wrapper does not check that the thread ID is valid.
pub fn name_os_thread(thread_id: u32, name: &str) -> Result<()> {
    let name = CString::new(name)?;
    // SAFETY: `name` is a NUL-terminated string borrowed for the call, and NVTX does not dereference the thread ID.
    unsafe { sys::nvtxNameOsThreadA(thread_id, name.as_ptr()) };
    Ok(())
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_encodes_event_attributes() {
        let message = c"work";
        let raw = EventAttributes::new()
            .with_message(message)
            .with_category(Category::from_raw(7))
            .with_color(Color::rgba(1, 2, 3, 4))
            .with_payload(Payload::I64(-42))
            .encode();

        assert_eq!(
            raw.version,
            u16::try_from(sys::NVTX_VERSION).expect("the NVTX version fits its ABI field")
        );
        assert_eq!(
            raw.size,
            u16::try_from(size_of::<sys::nvtxEventAttributes_t>())
                .expect("the NVTX event attributes fit their ABI size field")
        );
        assert_eq!(raw.category, 7);
        assert_eq!(
            raw.colorType,
            sys::nvtxColorType_t::NVTX_COLOR_ARGB.0.cast_signed()
        );
        assert_eq!(raw.color, 0x0401_0203);
        assert_eq!(
            raw.messageType,
            sys::nvtxMessageType_t::NVTX_MESSAGE_TYPE_ASCII
                .0
                .cast_signed()
        );
        // SAFETY: the message type checked above selects `ascii`.
        assert_eq!(unsafe { raw.message.ascii }, message.as_ptr());
        assert_eq!(
            raw.payloadType,
            sys::nvtxPayloadType_t::NVTX_PAYLOAD_TYPE_INT64
                .0
                .cast_signed()
        );
        // SAFETY: the payload type checked above selects `llValue`.
        assert_eq!(unsafe { raw.payload.llValue }, -42);
    }

    #[test]
    fn test_owned_event_builds_attributes() {
        let event = Event::create("owned")
            .unwrap()
            .with_category(Category::from_raw(3))
            .with_color(Color::argb(0xff00_00ff))
            .with_payload(Payload::U32(11));

        let attributes = event.attributes();
        let raw = attributes.encode();

        assert_eq!(attributes.message(), Some(c"owned"));
        assert_eq!(attributes.category(), Some(Category::from_raw(3)));
        assert_eq!(attributes.color(), Some(Color::argb(0xff00_00ff)));
        assert_eq!(attributes.payload(), Some(Payload::U32(11)));
        assert_eq!(raw.category, 3);
        assert_eq!(raw.color, 0xff00_00ff);
        assert_eq!(
            raw.payloadType,
            sys::nvtxPayloadType_t::NVTX_PAYLOAD_TYPE_UNSIGNED_INT32
                .0
                .cast_signed()
        );
        // SAFETY: the payload type checked above selects `uiValue`.
        assert_eq!(unsafe { raw.payload.uiValue }, 11);
    }

    #[test]
    fn test_enum_wrappers_convert_and_display() {
        assert_eq!(
            ColorType::try_from(sys::nvtxColorType_t::NVTX_COLOR_ARGB).unwrap(),
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

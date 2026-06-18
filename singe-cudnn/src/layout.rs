use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_cudnn_sys as sys;

use singe_core::{impl_enum_conversion, impl_enum_display};

/// Queryable layout requirement reported by a finalized
/// [`BackendDescriptorType::Engine`](crate::descriptor::BackendDescriptorType::Engine)
/// through
/// [`BackendDescriptor::attribute_enum_slice`](crate::descriptor::BackendDescriptor::attribute_enum_slice).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum BackendLayoutType {
    PreferredNchw = sys::cudnnBackendLayoutType_t::CUDNN_LAYOUT_TYPE_PREFERRED_NCHW as _,
    PreferredNhwc = sys::cudnnBackendLayoutType_t::CUDNN_LAYOUT_TYPE_PREFERRED_NHWC as _,
    PreferredPad4ck = sys::cudnnBackendLayoutType_t::CUDNN_LAYOUT_TYPE_PREFERRED_PAD4CK as _,
    PreferredPad8ck = sys::cudnnBackendLayoutType_t::CUDNN_LAYOUT_TYPE_PREFERRED_PAD8CK as _,
}

impl_enum_conversion!(sys::cudnnBackendLayoutType_t, BackendLayoutType);

impl_enum_display!(BackendLayoutType, {
    Self::PreferredNchw => "CUDNN_LAYOUT_TYPE_PREFERRED_NCHW",
    Self::PreferredNhwc => "CUDNN_LAYOUT_TYPE_PREFERRED_NHWC",
    Self::PreferredPad4ck => "CUDNN_LAYOUT_TYPE_PREFERRED_PAD4CK",
    Self::PreferredPad8ck => "CUDNN_LAYOUT_TYPE_PREFERRED_PAD8CK",
});

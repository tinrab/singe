use singe_cudnn_sys as sys;

use crate::metadata::backend_metadata_enum;

backend_metadata_enum! {
    /// Queryable layout requirement reported by a finalized
    /// [`BackendDescriptorType::Engine`](crate::backend::descriptor::BackendDescriptorType::Engine)
    /// through
    /// [`BackendDescriptor::attribute_enum_slice`](crate::backend::descriptor::BackendDescriptor::attribute_enum_slice).
    pub enum BackendLayoutType: sys::cudnnBackendLayoutType_t {
        PreferredNchw => CUDNN_LAYOUT_TYPE_PREFERRED_NCHW => "Prefers NCHW layout.",
        PreferredNhwc => CUDNN_LAYOUT_TYPE_PREFERRED_NHWC => "Prefers NHWC layout.",
        PreferredPad4ck => CUDNN_LAYOUT_TYPE_PREFERRED_PAD4CK => "Prefers padded 4CK layout.",
        PreferredPad8ck => CUDNN_LAYOUT_TYPE_PREFERRED_PAD8CK => "Prefers padded 8CK layout.",
    }
}

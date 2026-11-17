use std::fmt::{self, Display, Formatter};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::impl_enum_display;
use singe_cupti_sys as sys;

pub(crate) type SubscriberHandle = sys::CUpti_SubscriberHandle;

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// Point at which CUPTI invokes an API callback.
    /// Callback payloads expose this as the callback site of the underlying CUPTI callback data.
    pub enum ApiCallbackSite: sys::CUpti_ApiCallbackSite {
        /// The callback is at the entry of the API call.
        Enter = CUPTI_API_ENTER,
        /// The callback is at the exit of the API call.
        Exit = CUPTI_API_EXIT,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_API_CBSITE_FORCE_INT,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// Group of related CUPTI callback IDs.
    pub enum CallbackDomain: sys::CUpti_CallbackDomain {
        /// Invalid domain.
        Invalid = CUPTI_CB_DOMAIN_INVALID,
        /// Domain containing callback points for all driver API functions.
        DriverApi = CUPTI_CB_DOMAIN_DRIVER_API,
        /// Domain containing callback points for all runtime API functions.
        RuntimeApi = CUPTI_CB_DOMAIN_RUNTIME_API,
        /// Domain containing callback points for CUDA resource tracking.
        Resource = CUPTI_CB_DOMAIN_RESOURCE,
        /// Domain containing callback points for CUDA synchronization.
        Synchronize = CUPTI_CB_DOMAIN_SYNCHRONIZE,
        /// Domain containing callback points for NVTX API functions.
        Nvtx = CUPTI_CB_DOMAIN_NVTX,
        /// Domain containing callback points for various states.
        State = CUPTI_CB_DOMAIN_STATE,
        /// Number of entries defined for this CUPTI enum.
        Size = CUPTI_CB_DOMAIN_SIZE,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_CB_DOMAIN_FORCE_INT,
    }
}

impl_enum_display!(CallbackDomain, {
    Self::Invalid => "CUPTI_CB_DOMAIN_INVALID",
    Self::DriverApi => "CUPTI_CB_DOMAIN_DRIVER_API",
    Self::RuntimeApi => "CUPTI_CB_DOMAIN_RUNTIME_API",
    Self::Resource => "CUPTI_CB_DOMAIN_RESOURCE",
    Self::Synchronize => "CUPTI_CB_DOMAIN_SYNCHRONIZE",
    Self::Nvtx => "CUPTI_CB_DOMAIN_NVTX",
    Self::State => "CUPTI_CB_DOMAIN_STATE",
    Self::Size => "CUPTI_CB_DOMAIN_SIZE",
    Self::ForceInt => "CUPTI_CB_DOMAIN_FORCE_INT",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
/// Identifier for a CUPTI callback within a callback domain.
pub struct CallbackId(u32);

impl CallbackId {
    /// Returns the raw value decoded from CUPTI.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }

    pub(crate) const fn as_raw(self) -> sys::CUpti_CallbackId {
        self.0
    }
}

impl From<sys::CUpti_CallbackId> for CallbackId {
    fn from(value: sys::CUpti_CallbackId) -> Self {
        Self(value)
    }
}

impl From<CallbackIdResource> for CallbackId {
    fn from(value: CallbackIdResource) -> Self {
        Self(value.into())
    }
}

impl From<CallbackIdSync> for CallbackId {
    fn from(value: CallbackIdSync) -> Self {
        Self(value.into())
    }
}

impl From<CallbackIdState> for CallbackId {
    fn from(value: CallbackIdState) -> Self {
        Self(value.into())
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// Callback IDs emitted for [`CallbackDomain::Resource`].
    pub enum CallbackIdResource: sys::CUpti_CallbackIdResource {
        /// Invalid resource callback ID.
        Invalid = CUPTI_CBID_RESOURCE_INVALID,
        /// A new context has been created.
        ContextCreated = CUPTI_CBID_RESOURCE_CONTEXT_CREATED,
        /// A context is about to be destroyed.
        ContextDestroyStarting = CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING,
        /// A new stream has been created.
        StreamCreated = CUPTI_CBID_RESOURCE_STREAM_CREATED,
        /// A stream is about to be destroyed.
        StreamDestroyStarting = CUPTI_CBID_RESOURCE_STREAM_DESTROY_STARTING,
        /// The driver has finished initializing.
        CuInitFinished = CUPTI_CBID_RESOURCE_CU_INIT_FINISHED,
        /// A module has been loaded.
        ModuleLoaded = CUPTI_CBID_RESOURCE_MODULE_LOADED,
        /// A module is about to be unloaded.
        ModuleUnloadStarting = CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING,
        /// The current module which is being profiled.
        ModuleProfiled = CUPTI_CBID_RESOURCE_MODULE_PROFILED,
        /// CUDA graph has been created.
        GraphCreated = CUPTI_CBID_RESOURCE_GRAPH_CREATED,
        /// CUDA graph is about to be destroyed.
        GraphDestroyStarting = CUPTI_CBID_RESOURCE_GRAPH_DESTROY_STARTING,
        /// CUDA graph is cloned.
        GraphCloned = CUPTI_CBID_RESOURCE_GRAPH_CLONED,
        /// CUDA graph node is about to be created.
        GraphNodeCreateStarting = CUPTI_CBID_RESOURCE_GRAPHNODE_CREATE_STARTING,
        /// CUDA graph node is created.
        GraphNodeCreated = CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED,
        /// CUDA graph node is about to be destroyed.
        GraphNodeDestroyStarting = CUPTI_CBID_RESOURCE_GRAPHNODE_DESTROY_STARTING,
        /// Dependency on a CUDA graph node is created.
        GraphNodeDependencyCreated = CUPTI_CBID_RESOURCE_GRAPHNODE_DEPENDENCY_CREATED,
        /// Dependency on a CUDA graph node is destroyed.
        GraphNodeDependencyDestroyStarting = CUPTI_CBID_RESOURCE_GRAPHNODE_DEPENDENCY_DESTROY_STARTING,
        /// An executable CUDA graph is about to be created.
        ExecutableGraphCreateStarting = CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATE_STARTING,
        /// An executable CUDA graph is created.
        ExecutableGraphCreated = CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATED,
        /// An executable CUDA graph is about to be destroyed.
        ExecutableGraphDestroyStarting = CUPTI_CBID_RESOURCE_GRAPHEXEC_DESTROY_STARTING,
        /// CUDA graph node is cloned.
        GraphNodeCloned = CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED,
        /// CUDA stream attribute is changed.
        StreamAttributeChanged = CUPTI_CBID_RESOURCE_STREAM_ATTRIBUTE_CHANGED,
        /// CUDA graph node is updated.
        GraphNodeUpdated = CUPTI_CBID_RESOURCE_GRAPH_NODE_UPDATED,
        /// Params are set for the CUDA graph node in the executable graph.
        GraphNodeSetParams = CUPTI_CBID_RESOURCE_GRAPH_NODE_SET_PARAMS,
        /// Number of entries defined for this CUPTI enum.
        Size = CUPTI_CBID_RESOURCE_SIZE,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_CBID_RESOURCE_FORCE_INT,
    }
}

impl_enum_display!(CallbackIdResource, {
    Self::Invalid => "CUPTI_CBID_RESOURCE_INVALID",
    Self::ContextCreated => "CUPTI_CBID_RESOURCE_CONTEXT_CREATED",
    Self::ContextDestroyStarting => "CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING",
    Self::StreamCreated => "CUPTI_CBID_RESOURCE_STREAM_CREATED",
    Self::StreamDestroyStarting => "CUPTI_CBID_RESOURCE_STREAM_DESTROY_STARTING",
    Self::CuInitFinished => "CUPTI_CBID_RESOURCE_CU_INIT_FINISHED",
    Self::ModuleLoaded => "CUPTI_CBID_RESOURCE_MODULE_LOADED",
    Self::ModuleUnloadStarting => "CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING",
    Self::ModuleProfiled => "CUPTI_CBID_RESOURCE_MODULE_PROFILED",
    Self::GraphCreated => "CUPTI_CBID_RESOURCE_GRAPH_CREATED",
    Self::GraphDestroyStarting => "CUPTI_CBID_RESOURCE_GRAPH_DESTROY_STARTING",
    Self::GraphCloned => "CUPTI_CBID_RESOURCE_GRAPH_CLONED",
    Self::GraphNodeCreateStarting => "CUPTI_CBID_RESOURCE_GRAPHNODE_CREATE_STARTING",
    Self::GraphNodeCreated => "CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED",
    Self::GraphNodeDestroyStarting => "CUPTI_CBID_RESOURCE_GRAPHNODE_DESTROY_STARTING",
    Self::GraphNodeDependencyCreated => "CUPTI_CBID_RESOURCE_GRAPHNODE_DEPENDENCY_CREATED",
    Self::GraphNodeDependencyDestroyStarting => "CUPTI_CBID_RESOURCE_GRAPHNODE_DEPENDENCY_DESTROY_STARTING",
    Self::ExecutableGraphCreateStarting => "CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATE_STARTING",
    Self::ExecutableGraphCreated => "CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATED",
    Self::ExecutableGraphDestroyStarting => "CUPTI_CBID_RESOURCE_GRAPHEXEC_DESTROY_STARTING",
    Self::GraphNodeCloned => "CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED",
    Self::StreamAttributeChanged => "CUPTI_CBID_RESOURCE_STREAM_ATTRIBUTE_CHANGED",
    Self::GraphNodeUpdated => "CUPTI_CBID_RESOURCE_GRAPH_NODE_UPDATED",
    Self::GraphNodeSetParams => "CUPTI_CBID_RESOURCE_GRAPH_NODE_SET_PARAMS",
    Self::Size => "CUPTI_CBID_RESOURCE_SIZE",
    Self::ForceInt => "CUPTI_CBID_RESOURCE_FORCE_INT",
});

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// Callback IDs emitted for [`CallbackDomain::Synchronize`].
    pub enum CallbackIdSync: sys::CUpti_CallbackIdSync {
        /// Invalid synchronize callback ID.
        Invalid = CUPTI_CBID_SYNCHRONIZE_INVALID,
        /// Stream synchronization has completed for the stream.
        StreamSynchronized = CUPTI_CBID_SYNCHRONIZE_STREAM_SYNCHRONIZED,
        /// Context synchronization has completed for the context.
        ContextSynchronized = CUPTI_CBID_SYNCHRONIZE_CONTEXT_SYNCHRONIZED,
        /// Number of entries defined for this CUPTI enum.
        Size = CUPTI_CBID_SYNCHRONIZE_SIZE,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_CBID_SYNCHRONIZE_FORCE_INT,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// Callback IDs emitted for [`CallbackDomain::State`].
    pub enum CallbackIdState: sys::CUpti_CallbackIdState {
        /// Invalid state callback ID.
        Invalid = CUPTI_CBID_STATE_INVALID,
        /// High-impact, non-recoverable error notification.
        ///
        /// CUPTI finalizes itself before issuing this callback; subscribers decide whether the application should continue without profiling or terminate.
        FatalError = CUPTI_CBID_STATE_FATAL_ERROR,
        /// Notification of non fatal errors - high impact, but recoverable.
        Error = CUPTI_CBID_STATE_ERROR,
        /// Notification of warnings - low impact, recoverable.
        Warning = CUPTI_CBID_STATE_WARNING,
        /// Number of entries defined for this CUPTI enum.
        Size = CUPTI_CBID_STATE_SIZE,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_CBID_STATE_FORCE_INT,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
/// State of CUPTI multiple-subscriber support.
pub enum MultipleSubscriberState {
    /// Disabled state returned by CUPTI.
    Disabled,
    /// Boolean enabled state returned by CUPTI.
    Enabled,
    /// Unset state returned by CUPTI.
    Unset,
    /// Unknown value decoded from CUPTI.
    Unknown(u8),
}

impl From<u8> for MultipleSubscriberState {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::Disabled,
            1 => Self::Enabled,
            2 => Self::Unset,
            value => Self::Unknown(value),
        }
    }
}

impl From<MultipleSubscriberState> for u8 {
    fn from(value: MultipleSubscriberState) -> Self {
        match value {
            MultipleSubscriberState::Disabled => 0,
            MultipleSubscriberState::Enabled => 1,
            MultipleSubscriberState::Unset => 2,
            MultipleSubscriberState::Unknown(value) => value,
        }
    }
}

impl_enum_display!(ApiCallbackSite, {
    Self::Enter => "CUPTI_API_ENTER",
    Self::Exit => "CUPTI_API_EXIT",
    Self::ForceInt => "CUPTI_API_CBSITE_FORCE_INT",
});

impl_enum_display!(CallbackIdSync, {
    Self::Invalid => "CUPTI_CBID_SYNCHRONIZE_INVALID",
    Self::StreamSynchronized => "CUPTI_CBID_SYNCHRONIZE_STREAM_SYNCHRONIZED",
    Self::ContextSynchronized => "CUPTI_CBID_SYNCHRONIZE_CONTEXT_SYNCHRONIZED",
    Self::Size => "CUPTI_CBID_SYNCHRONIZE_SIZE",
    Self::ForceInt => "CUPTI_CBID_SYNCHRONIZE_FORCE_INT",
});

impl_enum_display!(CallbackIdState, {
    Self::Invalid => "CUPTI_CBID_STATE_INVALID",
    Self::FatalError => "CUPTI_CBID_STATE_FATAL_ERROR",
    Self::Error => "CUPTI_CBID_STATE_ERROR",
    Self::Warning => "CUPTI_CBID_STATE_WARNING",
    Self::Size => "CUPTI_CBID_STATE_SIZE",
    Self::ForceInt => "CUPTI_CBID_STATE_FORCE_INT",
});

impl Display for MultipleSubscriberState {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Disabled => write!(f, "Disabled"),
            Self::Enabled => write!(f, "Enabled"),
            Self::Unset => write!(f, "Unset"),
            Self::Unknown(value) => write!(f, "Unknown({value})"),
        }
    }
}

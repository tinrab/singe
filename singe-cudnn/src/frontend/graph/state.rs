use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    sync::{Arc, Mutex},
};

use bomboni_serde as serde_helpers;
use serde::{Deserialize, Serialize};

use crate::{
    backend::{
        device::{DeviceProperties, DevicePropertiesJson},
        execution::{KernelCache, KernelCacheJson},
        tensor::{TensorId, TensorSpec},
    },
    frontend::{
        graph::{GraphConfig, RuntimeShapeConstraints},
        operation::{AttentionBackwardHeuristicOverride, Operation, TensorAccess},
        plan::PlanCache,
    },
};

/// Runtime-only state attached to a frontend graph.
///
/// Stores construction bookkeeping and process-local objects such as a shared
/// kernel cache or device properties alongside the serializable graph description.
#[derive(Debug, Clone, Default)]
pub struct GraphRuntimeState {
    /// New declaration identities, allowing inference checkpoints to record a count instead of copying all IDs.
    pub(crate) tensor_insertions: Vec<TensorId>,
    /// Operations whose nested score-graph identities have been reserved for automatic allocation.
    pub(crate) indexed_operation_count: usize,
    pub(crate) reserved_tensor_ids: HashSet<TensorId>,
    pub(crate) kernel_cache: Option<Arc<Mutex<KernelCache>>>,
    pub(crate) device_properties: Option<Arc<DeviceProperties>>,
    pub(crate) plan_cache: Option<Arc<Mutex<PlanCache>>>,
}

/// Declarative cuDNN frontend operation graph.
///
/// A graph is a dataflow description of tensor operations that is separate from
/// the backend engines that may execute it. After construction, the graph can be
/// lowered to a cuDNN operation graph and compiled into one or more execution
/// plans selected from heuristic engine configurations.
///
/// This mirrors the high-level C++ frontend graph API model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    pub(crate) config: GraphConfig,
    pub(crate) tensors: BTreeMap<TensorId, TensorSpec>,
    #[serde(default)]
    pub(crate) vector_aligned_tensors: BTreeSet<TensorId>,
    pub(crate) operations: Vec<Operation>,
    pub(crate) storage_views: Vec<StorageView>,
    pub(crate) kernel_cache_enabled: bool,
    pub(crate) kernel_cache_json: Option<KernelCacheJson>,
    pub(crate) device_properties_json: Option<DevicePropertiesJson>,
    #[serde(default)]
    pub(crate) runtime_shape_constraints: BTreeMap<TensorId, RuntimeShapeConstraints>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) attention_backward_heuristic_overrides: Vec<AttentionBackwardHeuristicOverride>,
    #[serde(skip, default)]
    pub(crate) runtime: GraphRuntimeState,
}

/// Binds one tensor as an alias of another tensor at a byte offset.
///
/// This models in-place or aliased graph tensors while preserving the logical
/// tensor IDs used by the frontend graph.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct StorageView {
    /// Frontend identity supplying the aliased device allocation.
    pub source: TensorId,
    /// Frontend identity whose declared layout uses this storage view.
    pub target: TensorId,
    /// Nonnegative byte displacement added to the source address for the target.
    #[serde(with = "serde_helpers::as_string")]
    pub byte_offset: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "native capabilities and tensor properties are independent flags"
)]
pub struct TensorRecord {
    pub access: Option<TensorAccess>,
    pub reads_initial_state: bool,
    pub indexes_memory: bool,
    pub requires_vector_alignment: bool,
    pub requires_int32_filter_offsets: bool,
    pub(crate) tensor: TensorSpec,
    pub(crate) id: TensorId,
}

#[derive(Debug, Clone, Copy)]
pub struct GraphMutationCheckpoint {
    pub(crate) tensor_insertions: usize,
    pub(crate) operations: usize,
    pub(crate) storage_views: usize,
}

use std::{
    collections::{BTreeMap, HashSet},
    sync::{Arc, Mutex},
};

use bomboni_serde as serde_helpers;
use serde::{Deserialize, Serialize};

use crate::{
    data_type::DataType,
    device::DeviceProperties,
    error::{Error, Result},
    execution::{KernelCache, OperationGraph},
    frontend::{
        lower::LoweredGraph,
        operation::Operation,
        plan::{BindingReplacement, PlanCache},
    },
    scalar::ScalarValue,
    tensor::{Shape, TensorId, TensorSpec},
};

/// Runtime-only state attached to a frontend graph.
///
/// This stores process-local objects such as a shared kernel cache or device
/// properties alongside the serializable graph description.
#[derive(Debug, Clone, Default)]
pub struct GraphRuntimeState {
    pub(super) kernel_cache: Option<Arc<Mutex<KernelCache>>>,
    pub(super) device_properties: Option<Arc<DeviceProperties>>,
    pub(super) plan_cache: Option<Arc<Mutex<PlanCache>>>,
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
    pub(crate) name: Option<String>,
    pub(crate) tensors: BTreeMap<TensorId, TensorSpec>,
    pub(crate) operations: Vec<Operation>,
    pub(crate) alias_bindings: Vec<AliasBinding>,
    pub(crate) sm_count_target: Option<i64>,
    pub(crate) sm_version: Option<i32>,
    pub(crate) dynamic_shape_enabled: bool,
    pub(crate) override_shape_enabled: bool,
    pub(crate) kernel_cache_enabled: bool,
    pub(crate) kernel_cache_json: Option<String>,
    pub(crate) device_properties_json: Option<String>,
    #[serde(default)]
    pub(crate) data_type_policy: DataTypePolicy,
    #[serde(default)]
    pub(crate) dynamic_shape_constraints: BTreeMap<TensorId, RuntimeShapeConstraints>,
    #[serde(skip, default)]
    pub(crate) runtime: GraphRuntimeState,
}

/// Lowered representation produced while preparing a [`Graph`] for planning.
pub struct PreparedGraph {
    pub(crate) expanded: Graph,
    pub(crate) lowered: LoweredGraph,
    pub(crate) scalar_bindings: Vec<(TensorId, ScalarValue)>,
    pub(crate) binding_replacements: Vec<BindingReplacement>,
    pub(crate) operation_graph: OperationGraph,
}

/// Binds one tensor as an alias of another tensor at a byte offset.
///
/// This models in-place or aliased graph tensors while preserving the logical
/// tensor IDs used by the frontend graph.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AliasBinding {
    pub source: TensorId,
    pub target: TensorId,
    #[serde(with = "serde_helpers::as_string")]
    pub byte_offset: i64,
}

/// Graph-level default data types used by infer-style frontend helpers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DataTypePolicy {
    pub io: Option<DataType>,
    pub intermediate: Option<DataType>,
    pub compute: Option<DataType>,
}

impl DataTypePolicy {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_io(mut self, data_type: DataType) -> Self {
        self.io = Some(data_type);
        self
    }

    pub fn with_intermediate(mut self, data_type: DataType) -> Self {
        self.intermediate = Some(data_type);
        self
    }

    pub fn with_compute(mut self, data_type: DataType) -> Self {
        self.compute = Some(data_type);
        self
    }
}

/// Inclusive runtime shape bounds accepted for one dynamic-shape tensor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeShapeConstraints {
    pub min_dimensions: Vec<i64>,
    pub max_dimensions: Vec<i64>,
}

impl RuntimeShapeConstraints {
    pub fn create(
        min_dimensions: impl Into<Vec<i64>>,
        max_dimensions: impl Into<Vec<i64>>,
    ) -> Result<Self> {
        let min_dimensions = min_dimensions.into();
        let max_dimensions = max_dimensions.into();
        validate_runtime_shape_constraint_dimensions(&min_dimensions, &max_dimensions)?;
        Ok(Self {
            min_dimensions,
            max_dimensions,
        })
    }

    pub fn contains(&self, dimensions: &[i64]) -> bool {
        dimensions.len() == self.min_dimensions.len()
            && self
                .min_dimensions
                .iter()
                .zip(&self.max_dimensions)
                .zip(dimensions)
                .all(|((min_dimension, max_dimension), dimension)| {
                    *min_dimension <= *dimension && *dimension <= *max_dimension
                })
    }
}

pub(crate) fn validate_runtime_shape_constraint_dimensions(
    min_dimensions: &[i64],
    max_dimensions: &[i64],
) -> Result<()> {
    if min_dimensions.len() != max_dimensions.len() {
        return Err(Error::LengthMismatch {
            name: "dynamic_shape_constraints_max_rank".into(),
            expected: min_dimensions.len(),
            actual: max_dimensions.len(),
        });
    }
    Shape::contiguous(min_dimensions)?;
    Shape::contiguous(max_dimensions)?;
    if min_dimensions
        .iter()
        .zip(max_dimensions)
        .any(|(min_dimension, max_dimension)| min_dimension > max_dimension)
    {
        return Err(Error::FrontendDynamicShapeBoundsInvalid {
            min_dimensions: min_dimensions.to_vec(),
            max_dimensions: max_dimensions.to_vec(),
        });
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub(crate) struct TensorRecord {
    pub(crate) tensor: TensorSpec,
    pub(crate) id: TensorId,
}

#[derive(Debug, Clone)]
pub(crate) struct GraphMutationCheckpoint {
    pub(super) tensor_ids: HashSet<TensorId>,
    pub(super) operation_count: usize,
    pub(super) alias_binding_count: usize,
}

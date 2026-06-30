use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use crate::{
    data_type::DataType,
    device::DeviceProperties,
    error::{Error, Result},
    execution::KernelCache,
    frontend::{plan::PlanCache, support},
    tensor::TensorId,
    version,
};

use super::{
    DataTypePolicy, Graph, GraphConfig, RuntimeShapeConstraints,
    state::{GraphRuntimeState, validate_runtime_shape_constraint_dimensions},
};

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    pub fn new() -> Self {
        Self::with_config(GraphConfig::new())
    }

    pub fn with_config(config: GraphConfig) -> Self {
        Self {
            name: config.name,
            tensors: BTreeMap::new(),
            operations: Vec::new(),
            alias_bindings: Vec::new(),
            sm_count_target: config.sm_count_target,
            sm_version: config.sm_version,
            dynamic_shape_enabled: config.dynamic_shape_enabled,
            override_shape_enabled: config.override_shape_enabled,
            kernel_cache_enabled: false,
            kernel_cache_json: None,
            device_properties_json: None,
            runtime: GraphRuntimeState::default(),
            data_type_policy: config.data_type_policy,
            dynamic_shape_constraints: BTreeMap::new(),
        }
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn sm_count_target(&self) -> Option<i64> {
        self.sm_count_target
    }

    pub fn sm_version(&self) -> Option<i32> {
        self.sm_version
    }

    pub fn dynamic_shape_enabled(&self) -> bool {
        self.dynamic_shape_enabled
    }

    /// Sets inclusive runtime shape bounds for a dynamic tensor.
    ///
    /// These constraints are Rust-side validation metadata. cuDNN still receives
    /// the execution-time dimensions through runtime overrides, while the
    /// frontend rejects shapes outside the recorded bounds before building the
    /// variant pack.
    pub fn record_dynamic_shape_constraints(
        &mut self,
        tensor: TensorId,
        min_dimensions: impl Into<Vec<i64>>,
        max_dimensions: impl Into<Vec<i64>>,
    ) -> Result<()> {
        let tensor_config = self.tensor_config(tensor)?;
        let min_dimensions = min_dimensions.into();
        let max_dimensions = max_dimensions.into();
        if min_dimensions.len() != tensor_config.shape.dimensions().len() {
            return Err(Error::LengthMismatch {
                name: "dynamic_shape_constraints_rank".into(),
                expected: tensor_config.shape.dimensions().len(),
                actual: min_dimensions.len(),
            });
        }
        validate_runtime_shape_constraint_dimensions(&min_dimensions, &max_dimensions)?;
        self.dynamic_shape_constraints.insert(
            tensor,
            RuntimeShapeConstraints {
                min_dimensions,
                max_dimensions,
            },
        );
        Ok(())
    }

    /// Enables dynamic-shape handling and records bounds for one tensor.
    pub fn record_dynamic_shape_tensor(
        &mut self,
        tensor: TensorId,
        min_dimensions: impl Into<Vec<i64>>,
        max_dimensions: impl Into<Vec<i64>>,
    ) -> Result<()> {
        self.record_dynamic_shape_constraints(tensor, min_dimensions, max_dimensions)?;
        self.dynamic_shape_enabled = true;
        Ok(())
    }

    pub fn remove_dynamic_shape_constraints(&mut self, tensor: TensorId) {
        self.dynamic_shape_constraints.remove(&tensor);
    }

    pub fn dynamic_shape_constraints(&self, tensor: TensorId) -> Option<&RuntimeShapeConstraints> {
        self.dynamic_shape_constraints.get(&tensor)
    }

    pub fn override_shape_enabled(&self) -> bool {
        self.override_shape_enabled
    }

    /// Attaches a serialized kernel cache to this graph.
    ///
    /// Kernel caches reduce plan build time by reusing previously compiled
    /// runtime kernels for matching execution plans on dynamic-shape graphs.
    pub fn attach_kernel_cache_json(&mut self, kernel_cache_json: impl Into<String>) {
        self.kernel_cache_enabled = true;
        self.kernel_cache_json = Some(kernel_cache_json.into());
        self.runtime.kernel_cache = None;
    }

    pub fn attach_kernel_cache(&mut self, kernel_cache: KernelCache) -> Result<()> {
        self.attach_shared_kernel_cache(Arc::new(Mutex::new(kernel_cache)))
    }

    pub fn attach_shared_kernel_cache(
        &mut self,
        kernel_cache: Arc<Mutex<KernelCache>>,
    ) -> Result<()> {
        self.kernel_cache_enabled = true;
        let kernel_cache_guard = kernel_cache
            .lock()
            .map_err(|_| Error::FrontendKernelCacheLockPoisoned)?;
        self.kernel_cache_json = if kernel_cache_guard.is_finalized()
            && support::KERNEL_CACHE_SERIALIZATION.is_supported(version()?.raw())
        {
            Some(kernel_cache_guard.json_representation()?)
        } else {
            None
        };
        drop(kernel_cache_guard);
        self.runtime.kernel_cache = Some(kernel_cache);
        Ok(())
    }

    pub fn kernel_cache_enabled(&self) -> bool {
        self.kernel_cache_enabled
    }

    pub fn kernel_cache_json(&self) -> Option<&str> {
        self.kernel_cache_json.as_deref()
    }

    /// Attaches an in-process plan cache shared by graphs with equivalent
    /// topology and compile settings.
    pub fn attach_plan_cache(&mut self, plan_cache: Arc<Mutex<PlanCache>>) {
        self.runtime.plan_cache = Some(plan_cache);
    }

    pub fn shared_plan_cache(&self) -> Option<Arc<Mutex<PlanCache>>> {
        self.runtime.plan_cache.clone()
    }

    /// Attaches serialized cuDNN device properties for deviceless planning.
    ///
    /// Device properties can be serialized from a real device and later used to
    /// query heuristics or create execution plans without that device being
    /// present.
    pub fn attach_device_properties_json(&mut self, device_properties_json: impl Into<String>) {
        self.device_properties_json = Some(device_properties_json.into());
        self.runtime.device_properties = None;
    }

    /// Attaches cuDNN device properties for deviceless planning.
    pub fn attach_device_properties(&mut self, device_properties: DeviceProperties) -> Result<()> {
        self.device_properties_json = if support::DEVICE_PROPERTIES.is_supported(version()?.raw()) {
            Some(device_properties.json_representation()?)
        } else {
            None
        };
        self.runtime.device_properties = Some(Arc::new(device_properties));
        Ok(())
    }

    pub fn device_properties_json(&self) -> Option<&str> {
        self.device_properties_json.as_deref()
    }

    pub fn data_type_policy(&self) -> DataTypePolicy {
        self.data_type_policy
    }

    pub(crate) fn effective_io_data_type(&self, default: DataType) -> DataType {
        self.data_type_policy.effective_io(default)
    }

    pub(crate) fn effective_intermediate_data_type(&self, default: DataType) -> DataType {
        self.data_type_policy.effective_intermediate(default)
    }

    pub(crate) fn effective_compute_data_type(&self, default: DataType) -> DataType {
        self.data_type_policy.effective_compute(default)
    }
}

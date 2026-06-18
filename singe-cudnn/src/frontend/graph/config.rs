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
    DataTypePolicy, Graph, RuntimeShapeConstraints,
    state::{GraphRuntimeState, validate_runtime_shape_constraint_dimensions},
};

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    pub fn new() -> Self {
        Self {
            name: None,
            tensors: BTreeMap::new(),
            operations: Vec::new(),
            alias_bindings: Vec::new(),
            sm_count_target: None,
            sm_version: None,
            dynamic_shape_enabled: false,
            override_shape_enabled: false,
            kernel_cache_enabled: false,
            kernel_cache_json: None,
            device_properties_json: None,
            runtime: GraphRuntimeState::default(),
            data_type_policy: DataTypePolicy::default(),
            dynamic_shape_constraints: BTreeMap::new(),
        }
    }

    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = Some(name.into());
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.set_name(name);
        self
    }

    pub fn clear_name(&mut self) {
        self.name = None;
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn set_sm_count_target(&mut self, sm_count_target: i64) {
        self.sm_count_target = Some(sm_count_target);
    }

    pub fn with_sm_count_target(mut self, sm_count_target: i64) -> Self {
        self.set_sm_count_target(sm_count_target);
        self
    }

    pub fn clear_sm_count_target(&mut self) {
        self.sm_count_target = None;
    }

    pub fn sm_count_target(&self) -> Option<i64> {
        self.sm_count_target
    }

    pub fn set_sm_version(&mut self, sm_version: i32) {
        self.sm_version = Some(sm_version);
    }

    pub fn with_sm_version(mut self, sm_version: i32) -> Self {
        self.set_sm_version(sm_version);
        self
    }

    pub fn clear_sm_version(&mut self) {
        self.sm_version = None;
    }

    pub fn sm_version(&self) -> Option<i32> {
        self.sm_version
    }

    pub fn enable_dynamic_shape(&mut self) {
        self.dynamic_shape_enabled = true;
    }

    pub fn disable_dynamic_shape(&mut self) {
        self.dynamic_shape_enabled = false;
    }

    pub fn with_dynamic_shape(mut self) -> Self {
        self.enable_dynamic_shape();
        self
    }

    pub fn without_dynamic_shape(mut self) -> Self {
        self.disable_dynamic_shape();
        self
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
    pub fn set_dynamic_shape_constraints(
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

    pub fn with_dynamic_shape_constraints(
        mut self,
        tensor: TensorId,
        min_dimensions: impl Into<Vec<i64>>,
        max_dimensions: impl Into<Vec<i64>>,
    ) -> Result<Self> {
        self.set_dynamic_shape_constraints(tensor, min_dimensions, max_dimensions)?;
        Ok(self)
    }

    /// Enables dynamic-shape handling and records bounds for one tensor.
    ///
    /// Prefer this over calling [`Graph::enable_dynamic_shape`] and
    /// [`Graph::set_dynamic_shape_constraints`] separately when a graph expects
    /// runtime shape overrides for specific tensors.
    pub fn enable_dynamic_shape_for_tensor(
        &mut self,
        tensor: TensorId,
        min_dimensions: impl Into<Vec<i64>>,
        max_dimensions: impl Into<Vec<i64>>,
    ) -> Result<()> {
        self.set_dynamic_shape_constraints(tensor, min_dimensions, max_dimensions)?;
        self.enable_dynamic_shape();
        Ok(())
    }

    pub fn with_dynamic_shape_for_tensor(
        mut self,
        tensor: TensorId,
        min_dimensions: impl Into<Vec<i64>>,
        max_dimensions: impl Into<Vec<i64>>,
    ) -> Result<Self> {
        self.enable_dynamic_shape_for_tensor(tensor, min_dimensions, max_dimensions)?;
        Ok(self)
    }

    pub fn clear_dynamic_shape_constraints(&mut self, tensor: TensorId) {
        self.dynamic_shape_constraints.remove(&tensor);
    }

    pub fn dynamic_shape_constraints(&self, tensor: TensorId) -> Option<&RuntimeShapeConstraints> {
        self.dynamic_shape_constraints.get(&tensor)
    }

    pub fn enable_override_shape(&mut self) {
        self.override_shape_enabled = true;
    }

    pub fn disable_override_shape(&mut self) {
        self.override_shape_enabled = false;
    }

    pub fn with_override_shape(mut self) -> Self {
        self.enable_override_shape();
        self
    }

    pub fn without_override_shape(mut self) -> Self {
        self.disable_override_shape();
        self
    }

    pub fn override_shape_enabled(&self) -> bool {
        self.override_shape_enabled
    }

    /// Attaches a serialized kernel cache to this graph.
    ///
    /// Kernel caches reduce plan build time by reusing previously compiled
    /// runtime kernels for matching execution plans on dynamic-shape graphs.
    pub fn set_kernel_cache_json(&mut self, kernel_cache_json: impl Into<String>) {
        self.kernel_cache_enabled = true;
        self.kernel_cache_json = Some(kernel_cache_json.into());
        self.runtime.kernel_cache = None;
    }

    pub fn with_kernel_cache_json(mut self, kernel_cache_json: impl Into<String>) -> Self {
        self.set_kernel_cache_json(kernel_cache_json);
        self
    }

    pub fn set_kernel_cache(&mut self, kernel_cache: KernelCache) -> Result<()> {
        self.set_shared_kernel_cache(Arc::new(Mutex::new(kernel_cache)))
    }

    pub fn with_kernel_cache(mut self, kernel_cache: KernelCache) -> Result<Self> {
        self.set_kernel_cache(kernel_cache)?;
        Ok(self)
    }

    pub fn set_shared_kernel_cache(&mut self, kernel_cache: Arc<Mutex<KernelCache>>) -> Result<()> {
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

    pub fn with_shared_kernel_cache(
        mut self,
        kernel_cache: Arc<Mutex<KernelCache>>,
    ) -> Result<Self> {
        self.set_shared_kernel_cache(kernel_cache)?;
        Ok(self)
    }

    pub fn clear_kernel_cache_json(&mut self) {
        self.kernel_cache_enabled = false;
        self.kernel_cache_json = None;
        self.runtime.kernel_cache = None;
    }

    pub fn kernel_cache_enabled(&self) -> bool {
        self.kernel_cache_enabled
    }

    pub fn kernel_cache_json(&self) -> Option<&str> {
        self.kernel_cache_json.as_deref()
    }

    /// Attaches an in-process plan cache shared by graphs with equivalent
    /// topology and compile settings.
    pub fn set_plan_cache(&mut self, plan_cache: Arc<Mutex<PlanCache>>) {
        self.runtime.plan_cache = Some(plan_cache);
    }

    pub fn with_plan_cache(mut self, plan_cache: Arc<Mutex<PlanCache>>) -> Self {
        self.set_plan_cache(plan_cache);
        self
    }

    pub fn clear_plan_cache(&mut self) {
        self.runtime.plan_cache = None;
    }

    pub fn shared_plan_cache(&self) -> Option<Arc<Mutex<PlanCache>>> {
        self.runtime.plan_cache.clone()
    }

    /// Attaches serialized cuDNN device properties for deviceless planning.
    ///
    /// Device properties can be serialized from a real device and later used to
    /// query heuristics or create execution plans without that device being
    /// present.
    pub fn set_device_properties_json(&mut self, device_properties_json: impl Into<String>) {
        self.device_properties_json = Some(device_properties_json.into());
        self.runtime.device_properties = None;
    }

    pub fn with_device_properties_json(
        mut self,
        device_properties_json: impl Into<String>,
    ) -> Self {
        self.set_device_properties_json(device_properties_json);
        self
    }

    /// Attaches cuDNN device properties for deviceless planning.
    pub fn set_device_properties(&mut self, device_properties: DeviceProperties) -> Result<()> {
        self.device_properties_json = if support::DEVICE_PROPERTIES.is_supported(version()?.raw()) {
            Some(device_properties.json_representation()?)
        } else {
            None
        };
        self.runtime.device_properties = Some(Arc::new(device_properties));
        Ok(())
    }

    pub fn with_device_properties(mut self, device_properties: DeviceProperties) -> Result<Self> {
        self.set_device_properties(device_properties)?;
        Ok(self)
    }

    pub fn clear_device_properties_json(&mut self) {
        self.device_properties_json = None;
        self.runtime.device_properties = None;
    }

    pub fn device_properties_json(&self) -> Option<&str> {
        self.device_properties_json.as_deref()
    }

    pub fn set_data_type_policy(&mut self, policy: DataTypePolicy) {
        self.data_type_policy = policy;
    }

    pub fn with_data_type_policy(mut self, policy: DataTypePolicy) -> Self {
        self.set_data_type_policy(policy);
        self
    }

    pub fn data_type_policy(&self) -> DataTypePolicy {
        self.data_type_policy
    }

    pub fn set_io_data_type(&mut self, data_type: DataType) {
        self.data_type_policy.io = Some(data_type);
    }

    pub fn with_io_data_type(mut self, data_type: DataType) -> Self {
        self.set_io_data_type(data_type);
        self
    }

    pub fn clear_io_data_type(&mut self) {
        self.data_type_policy.io = None;
    }

    pub fn io_data_type(&self) -> Option<DataType> {
        self.data_type_policy.io
    }

    pub fn set_intermediate_data_type(&mut self, data_type: DataType) {
        self.data_type_policy.intermediate = Some(data_type);
    }

    pub fn with_intermediate_data_type(mut self, data_type: DataType) -> Self {
        self.set_intermediate_data_type(data_type);
        self
    }

    pub fn clear_intermediate_data_type(&mut self) {
        self.data_type_policy.intermediate = None;
    }

    pub fn intermediate_data_type(&self) -> Option<DataType> {
        self.data_type_policy.intermediate
    }

    pub fn set_compute_data_type(&mut self, data_type: DataType) {
        self.data_type_policy.compute = Some(data_type);
    }

    pub fn with_compute_data_type(mut self, data_type: DataType) -> Self {
        self.set_compute_data_type(data_type);
        self
    }

    pub fn clear_compute_data_type(&mut self) {
        self.data_type_policy.compute = None;
    }

    pub fn compute_data_type(&self) -> Option<DataType> {
        self.data_type_policy.compute
    }

    pub(crate) fn effective_io_data_type(&self, default: DataType) -> DataType {
        self.data_type_policy.io.unwrap_or(default)
    }

    pub(crate) fn effective_intermediate_data_type(&self, default: DataType) -> DataType {
        self.data_type_policy.intermediate.unwrap_or(default)
    }

    pub(crate) fn effective_compute_data_type(&self, default: DataType) -> DataType {
        self.data_type_policy.compute.unwrap_or(default)
    }
}

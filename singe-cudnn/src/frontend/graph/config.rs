use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
};

use serde::{Deserialize, Serialize};

use crate::{
    backend::{
        device::{DeviceProperties, DevicePropertiesJson, sm_version_for_device},
        execution::{KernelCache, KernelCacheJson},
        tensor::TensorId,
    },
    context::Context,
    data_type::DataType,
    error::Result,
    frontend::{
        error::Error as FrontendError,
        graph::{DataTypePolicy, Graph, RuntimeShapeConstraints, state::GraphRuntimeState},
        plan::PlanCache,
    },
};

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    /// Creates an empty graph with default construction options and no attached runtime caches or device properties.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(GraphConfig::new())
    }

    /// Creates an empty graph using the supplied static options and no attached runtime caches or device properties.
    #[must_use]
    pub fn with_config(config: GraphConfig) -> Self {
        Self {
            config,
            tensors: BTreeMap::new(),
            vector_aligned_tensors: BTreeSet::default(),
            operations: Vec::new(),
            storage_views: Vec::new(),
            kernel_cache_enabled: false,
            kernel_cache_json: None,
            device_properties_json: None,
            runtime: GraphRuntimeState::default(),
            runtime_shape_constraints: BTreeMap::new(),
            attention_backward_heuristic_overrides: Vec::new(),
        }
    }

    pub(crate) fn compilation_sm_version(&self, context: Option<&Context>) -> Result<Option<i32>> {
        if let Some(sm) = self.config.sm_version.filter(|sm| *sm > 0) {
            return Ok(Some(sm));
        }
        if let Some(sm) = self
            .runtime
            .device_properties
            .as_ref()
            .and_then(|properties| properties.sm_version())
        {
            return Ok(Some(sm));
        }
        let Some(context) = context else {
            return Ok(None);
        };
        sm_version_for_device(context.cuda_context().device()).map(Some)
    }

    /// Returns the static construction options of this graph.
    #[must_use]
    pub fn config(&self) -> &GraphConfig {
        &self.config
    }

    /// Returns the optional diagnostic graph name.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.config.name.as_deref()
    }

    /// Sets a diagnostic name, retained by recipes but excluded from compilation-cache identity.
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.config.name = Some(name.into());
    }

    /// Returns the default SM-count target inherited by future planning requests.
    #[must_use]
    pub fn sm_count_target(&self) -> Option<u32> {
        self.config.sm_count_target
    }

    /// Sets the target inherited by future heuristic queries and explicit engine construction.
    /// Zero targets every SM; `None` leaves the native target unset.
    pub fn set_sm_count_target(&mut self, target: Option<u32>) {
        self.config.sm_count_target = target;
    }

    /// Returns the explicit architecture override, encoded as `major * 10 + minor`, or `None` for target resolution.
    #[must_use]
    pub fn sm_version(&self) -> Option<i32> {
        self.config.sm_version
    }

    /// Selects the architecture for future preparation.
    /// `None` restores resolution from attached device properties or the live context.
    pub fn set_sm_version(&mut self, version: Option<i32>) {
        self.config.sm_version = version;
    }

    /// Returns whether future preparation enables dynamic-shape compilation.
    #[must_use]
    pub fn dynamic_shape_enabled(&self) -> bool {
        self.config.dynamic_shape_enabled
    }

    /// Enables or disables dynamic-shape compilation for future preparations.
    /// Recorded bounds, attached caches and the independent override-shape option are retained.
    /// Preparation rejects a kernel cache while dynamic-shape compilation is disabled, matching C++.
    pub fn set_dynamic_shape_enabled(&mut self, enabled: bool) {
        self.config.dynamic_shape_enabled = enabled;
    }

    /// Sets inclusive runtime shape bounds for a dynamic tensor.
    ///
    /// These constraints are Rust-side validation metadata. cuDNN still receives
    /// the execution-time dimensions through runtime overrides, while the
    /// frontend rejects shapes outside the recorded bounds before building the
    /// variant pack.
    ///
    /// # Errors
    ///
    /// Returns an error if `tensor` is unknown, bounds have unequal or incompatible ranks, dimensions are empty, nonpositive, or have unrepresentable packed strides, a minimum exceeds its maximum, or by-value scalar metadata or bounds are invalid.
    pub fn record_runtime_shape_constraints(
        &mut self,
        tensor: TensorId,
        min_dimensions: impl Into<Vec<i64>>,
        max_dimensions: impl Into<Vec<i64>>,
    ) -> Result<()> {
        let tensor_spec = self.tensor_spec(tensor)?;
        let constraints = RuntimeShapeConstraints::create(min_dimensions, max_dimensions)?;
        constraints.validate_for(tensor_spec)?;
        self.runtime_shape_constraints.insert(tensor, constraints);
        Ok(())
    }

    /// Enables dynamic-shape handling and records bounds for one tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if `tensor` is unknown, bounds have unequal or incompatible ranks, dimensions are empty, nonpositive, or have unrepresentable packed strides, a minimum exceeds its maximum, or by-value scalar metadata or bounds are invalid.
    pub fn record_dynamic_shape_tensor(
        &mut self,
        tensor: TensorId,
        min_dimensions: impl Into<Vec<i64>>,
        max_dimensions: impl Into<Vec<i64>>,
    ) -> Result<()> {
        self.record_runtime_shape_constraints(tensor, min_dimensions, max_dimensions)?;
        self.config.dynamic_shape_enabled = true;
        Ok(())
    }

    /// Removes the stored runtime bounds for `tensor`, if present, without changing shape-handling options.
    pub fn remove_runtime_shape_constraints(&mut self, tensor: TensorId) {
        self.runtime_shape_constraints.remove(&tensor);
    }

    /// Borrows recorded inclusive runtime bounds for `tensor`, or `None` when none are recorded.
    #[must_use]
    pub fn runtime_shape_constraints(&self, tensor: TensorId) -> Option<&RuntimeShapeConstraints> {
        self.runtime_shape_constraints.get(&tensor)
    }

    /// Returns whether the independent runtime shape/stride override option is enabled.
    #[must_use]
    pub fn override_shape_enabled(&self) -> bool {
        self.config.override_shape_enabled
    }

    /// Enables or disables the override-shape option independently of dynamic-shape compilation.
    /// Runtime overrides remain available when either option is enabled.
    pub fn set_override_shape_enabled(&mut self, enabled: bool) {
        self.config.override_shape_enabled = enabled;
    }

    /// Attaches a serialized kernel cache to this graph.
    ///
    /// Kernel caches reduce plan build time by reusing previously compiled
    /// runtime kernels for matching execution plans on dynamic-shape graphs.
    pub fn attach_kernel_cache_json(&mut self, kernel_cache_json: KernelCacheJson) {
        self.kernel_cache_enabled = true;
        self.kernel_cache_json = Some(kernel_cache_json);
        self.runtime.kernel_cache = None;
    }

    /// Attaches an owned native cache for future preparation, recording JSON immediately if it is already finalized.
    ///
    /// # Errors
    ///
    /// Returns an error if serializing an already finalized cache fails.
    pub fn attach_kernel_cache(&mut self, kernel_cache: KernelCache) -> Result<()> {
        self.attach_shared_kernel_cache(Arc::new(Mutex::new(kernel_cache)))
    }

    /// Attaches a shared native cache, recording its current JSON snapshot when finalized.
    ///
    /// # Errors
    ///
    /// Returns an error if the cache mutex is poisoned or querying a finalized cache’s JSON fails.
    pub fn attach_shared_kernel_cache(
        &mut self,
        kernel_cache: Arc<Mutex<KernelCache>>,
    ) -> Result<()> {
        let kernel_cache_guard = kernel_cache
            .lock()
            .map_err(|_| FrontendError::KernelCacheLockPoisoned)?;
        let kernel_cache_json = if kernel_cache_guard.is_finalized() {
            Some(kernel_cache_guard.json_representation()?)
        } else {
            None
        };
        drop(kernel_cache_guard);
        self.kernel_cache_enabled = true;
        self.kernel_cache_json = kernel_cache_json;
        self.runtime.kernel_cache = Some(kernel_cache);
        Ok(())
    }

    /// Detaches runtime and serialized kernel-cache state from future preparation.
    /// Previously prepared graphs and plans retain their own cache owners.
    pub fn detach_kernel_cache(&mut self) {
        self.kernel_cache_enabled = false;
        self.kernel_cache_json = None;
        self.runtime.kernel_cache = None;
    }

    /// Returns the shared cache attached in this process, without locking or serializing it.
    /// Serialized-only attachment returns `None`; plans materialize and retain their own cache owners.
    #[must_use]
    pub fn shared_kernel_cache(&self) -> Option<Arc<Mutex<KernelCache>>> {
        self.runtime.kernel_cache.clone()
    }

    /// Returns whether future preparation should use a kernel cache.
    #[must_use]
    pub fn kernel_cache_enabled(&self) -> bool {
        self.kernel_cache_enabled
    }

    /// Borrows the attached serialized cache or the snapshot captured when attaching a finalized runtime cache.
    #[must_use]
    pub fn kernel_cache_json(&self) -> Option<&KernelCacheJson> {
        self.kernel_cache_json.as_ref()
    }

    /// Attaches an in-process plan cache shared by graphs with equivalent
    /// topology and compile settings.
    pub fn attach_plan_cache(&mut self, plan_cache: Arc<Mutex<PlanCache>>) {
        self.runtime.plan_cache = Some(plan_cache);
    }

    /// Clones the attached in-process plan-cache owner without locking it.
    #[must_use]
    pub fn shared_plan_cache(&self) -> Option<Arc<Mutex<PlanCache>>> {
        self.runtime.plan_cache.clone()
    }

    /// Attaches serialized cuDNN device properties for deviceless planning.
    ///
    /// Device properties can be serialized from a real device and later used to
    /// query heuristics or create execution plans without that device being
    /// present.
    pub fn attach_device_properties_json(&mut self, device_properties_json: DevicePropertiesJson) {
        self.device_properties_json = Some(device_properties_json);
        self.runtime.device_properties = None;
    }

    /// Attaches cuDNN device properties for deviceless planning.
    /// Properties collected from a device or handle also supply the default SM target.
    /// Opaque restored properties require an explicit SM version for attention
    /// implementation selection when compiling without a live context.
    ///
    /// # Errors
    ///
    /// Returns an error if querying the native device-property JSON fails.
    pub fn attach_device_properties(&mut self, device_properties: DeviceProperties) -> Result<()> {
        self.device_properties_json = Some(device_properties.json_representation()?);
        self.runtime.device_properties = Some(Arc::new(device_properties));
        Ok(())
    }

    /// Detaches runtime and serialized device properties from future preparation.
    /// The explicit SM version remains independent; existing prepared graphs and plans retain their targets.
    pub fn detach_device_properties(&mut self) {
        self.device_properties_json = None;
        self.runtime.device_properties = None;
    }

    /// Borrows the serialized device properties attached for future preparation.
    #[must_use]
    pub fn device_properties_json(&self) -> Option<&DevicePropertiesJson> {
        self.device_properties_json.as_ref()
    }

    /// Returns default I/O, intermediate-storage, and compute encodings used by future inference and preparation.
    #[must_use]
    pub fn data_type_policy(&self) -> DataTypePolicy {
        self.config.data_type_policy
    }

    /// Replaces precision defaults for future inference and preparation.
    /// Existing tensor encodings and explicit operation precision remain unchanged.
    /// Recorded operations that resolve omitted precision during preparation use this policy.
    /// Prepared graphs and compiled plans retain their own resolved declarations.
    pub fn set_data_type_policy(&mut self, policy: DataTypePolicy) {
        self.config.data_type_policy = policy;
    }

    pub(crate) fn effective_io_data_type(&self, default: DataType) -> DataType {
        self.config.data_type_policy.effective_io(default)
    }

    pub(crate) fn effective_intermediate_data_type(&self, default: DataType) -> DataType {
        self.config.data_type_policy.effective_intermediate(default)
    }

    pub(crate) fn effective_compute_data_type(&self, default: DataType) -> DataType {
        self.config.data_type_policy.effective_compute(default)
    }

    pub(crate) fn resolve_compute_data_type(&self, requested: Option<DataType>) -> DataType {
        self.config.data_type_policy.resolve_compute(requested)
    }
}

/// Static graph construction options.
///
/// Runtime-only handles such as shared kernel and plan caches are intentionally
/// attached to [`Graph`] after construction because they require validation or
/// process-local ownership.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphConfig {
    /// Optional diagnostic graph name.
    pub name: Option<String>,
    /// Default SM target for planning. Zero uses every SM; `None` leaves it unset.
    pub sm_count_target: Option<u32>,
    /// Optional compilation architecture override, encoded as `major * 10 + minor`.
    pub sm_version: Option<i32>,
    /// Enables dynamic-shape compilation during future graph preparation.
    pub dynamic_shape_enabled: bool,
    /// Enables runtime shape and stride overrides independently of dynamic-shape compilation.
    pub override_shape_enabled: bool,
    /// Default I/O, intermediate-storage, and computation encodings.
    pub data_type_policy: DataTypePolicy,
}

impl GraphConfig {
    /// Creates unnamed graph options with no SM override, no runtime shape options, and inherited encoding policies.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the diagnostic graph name and returns the updated options.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets the SM target inherited by heuristic queries and explicit engine construction.
    /// Zero uses every SM; `None` leaves the native target unset.
    /// `None` removes the graph default without changing other graph options.
    #[must_use]
    pub fn with_sm_count_target(mut self, sm_count_target: impl Into<Option<u32>>) -> Self {
        self.sm_count_target = sm_count_target.into();
        self
    }

    /// Declares the compilation target architecture, such as `90` for SM90.
    /// `None` restores target resolution from the live device or attached device properties.
    /// Deviceless automatic attention needs an explicit target or attached device properties.
    #[must_use]
    pub fn with_sm_version(mut self, sm_version: impl Into<Option<i32>>) -> Self {
        self.sm_version = sm_version.into();
        self
    }

    /// Enables or disables dynamic-shape compilation while retaining other construction options.
    #[must_use]
    pub fn with_dynamic_shape(mut self, enabled: bool) -> Self {
        self.dynamic_shape_enabled = enabled;
        self
    }

    /// Enables or disables runtime shape/stride overrides independently of dynamic-shape compilation.
    #[must_use]
    pub fn with_override_shape(mut self, enabled: bool) -> Self {
        self.override_shape_enabled = enabled;
        self
    }

    /// Replaces I/O, intermediate-storage, and compute encoding defaults.
    #[must_use]
    pub fn with_data_type_policy(mut self, policy: DataTypePolicy) -> Self {
        self.data_type_policy = policy;
        self
    }
}

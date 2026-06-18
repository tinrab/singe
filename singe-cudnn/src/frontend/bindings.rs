use std::collections::{BTreeMap, HashMap};

use singe_cuda::{
    graph::{ExecutableGraph, Graph as CudaGraph, GraphNode},
    memory::DeviceMemory,
    types::DevicePtr,
};

use crate::{
    context::Context,
    data_type::DataTypeLike,
    error::{Error, Result},
    execution::{VariantPack, VariantPackOverride},
    frontend::{
        graph::{RuntimeShapeConstraints, TensorRecord},
        plan::{BindingReplacement, CompiledGraph},
    },
    scalar::ScalarValue,
    tensor::{Shape, TensorId},
};

/// Mapping from frontend tensor IDs to device pointers for execution.
pub type BindingMap = HashMap<TensorId, DevicePtr>;

/// Runtime tensor bindings for executing a compiled frontend graph.
///
/// This is the Rust-side builder for the cuDNN variant pack passed to execution
/// and CUDA graph population/update calls. It maps frontend tensor IDs to device
/// pointers and can also own by-value scalar storage.
#[derive(Debug, Default)]
pub struct Bindings<'a> {
    tensors: Option<&'a BTreeMap<TensorId, TensorRecord>>,
    map: BindingMap,
    scalar_storage: HashMap<TensorId, Box<OwnedScalar>>,
    ordered_ids: Vec<TensorId>,
    ordered_pointers: Vec<Option<DevicePtr>>,
}

/// Runtime shape and stride overrides for dynamic-shape execution.
///
/// The override-shape API allows execution-time tensor shapes to differ from
/// the shapes used when building the graph. The override tensor IDs, shapes, and
/// strides are passed with execute/CUDA graph calls.
#[derive(Debug, Default, Clone)]
pub struct RuntimeOverrides {
    overrides: Vec<(TensorId, Vec<i64>, Vec<i64>)>,
}

/// Graph-aware builder for runtime shape and stride overrides.
///
/// Unlike [`RuntimeOverrides::set`], this validates overrides against a compiled graph as they are added.
#[derive(Debug)]
pub struct RuntimeOverrideBuilder<'a> {
    tensors: &'a BTreeMap<TensorId, TensorRecord>,
    constraints: Option<&'a BTreeMap<TensorId, RuntimeShapeConstraints>>,
    overrides: RuntimeOverrides,
}

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
struct OwnedScalar {
    bytes: [u8; 8],
}

impl OwnedScalar {
    fn from_scalar_value(value: ScalarValue) -> Self {
        match value {
            ScalarValue::I32(value) => Self {
                bytes: {
                    let mut bytes = [0_u8; 8];
                    bytes[..4].copy_from_slice(&value.to_ne_bytes());
                    bytes
                },
            },
            ScalarValue::I64(value) => Self {
                bytes: value.to_ne_bytes(),
            },
            ScalarValue::F32(value) => Self {
                bytes: {
                    let mut bytes = [0_u8; 8];
                    bytes[..4].copy_from_slice(&value.to_ne_bytes());
                    bytes
                },
            },
            ScalarValue::F16(value) | ScalarValue::Bf16(value) => Self {
                bytes: {
                    let mut bytes = [0_u8; 8];
                    bytes[..2].copy_from_slice(&value.to_ne_bytes());
                    bytes
                },
            },
            ScalarValue::F8E4M3(value)
            | ScalarValue::F8E5M2(value)
            | ScalarValue::F8UE8M0(value)
            | ScalarValue::F4E2M1(value) => Self {
                bytes: {
                    let mut bytes = [0_u8; 8];
                    bytes[0] = value;
                    bytes
                },
            },
        }
    }

    fn device_ptr(&self) -> DevicePtr {
        DevicePtr::from_raw(self.bytes.as_ptr().cast_mut().cast())
    }
}

impl Clone for Bindings<'_> {
    fn clone(&self) -> Self {
        let mut cloned = Self {
            tensors: self.tensors,
            map: self.map.clone(),
            scalar_storage: self
                .scalar_storage
                .iter()
                .map(|(&id, scalar)| (id, Box::new(**scalar)))
                .collect(),
            ordered_ids: self.ordered_ids.clone(),
            ordered_pointers: self.ordered_pointers.clone(),
        };
        cloned.refresh_scalar_bindings();
        cloned
    }
}

impl<'a> Bindings<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn with_graph_template(
        tensors: &'a BTreeMap<TensorId, TensorRecord>,
        ordered_ids: &[TensorId],
    ) -> Self {
        Self {
            tensors: Some(tensors),
            map: HashMap::new(),
            scalar_storage: HashMap::new(),
            ordered_ids: ordered_ids.to_vec(),
            ordered_pointers: vec![None; ordered_ids.len()],
        }
    }

    pub(crate) fn with_compiled_template(
        compiled: &'a CompiledGraph,
        ordered_ids: &[TensorId],
    ) -> Self {
        Self::with_graph_template(&compiled.tensors, ordered_ids)
    }

    /// Binds a tensor id to a raw device pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure `ptr` is valid for the tensor's data type and
    /// shape for the entire cuDNN execution using these bindings.
    pub unsafe fn set_id(&mut self, id: impl Into<TensorId>, ptr: DevicePtr) -> &mut Self {
        let id: TensorId = id.into();
        self.scalar_storage.remove(&id);
        self.map.insert(id, ptr);
        if let Some(index) = self
            .ordered_ids
            .iter()
            .position(|candidate| *candidate == id)
        {
            self.ordered_pointers[index] = Some(ptr);
        }
        self
    }

    fn validate_tensor_id(&self, id: TensorId) -> Result<()> {
        if let Some(tensors) = self.tensors {
            tensors
                .contains_key(&id)
                .then_some(())
                .ok_or(Error::FrontendTensorNotFound(id))?;
        }
        Ok(())
    }

    fn validate_tensor_binding_type<T: DataTypeLike>(&self, id: TensorId) -> Result<()> {
        if let Some(tensors) = self.tensors {
            let record = tensors.get(&id).ok_or(Error::FrontendTensorNotFound(id))?;
            let actual = T::data_type();
            if actual != record.tensor.data_type {
                return Err(Error::FrontendBindingTypeMismatch {
                    tensor_id: record.id,
                    expected: record.tensor.data_type.type_name().into(),
                    actual: actual.type_name().into(),
                });
            }
        }
        Ok(())
    }

    pub fn set_scalar_id(&mut self, id: impl Into<TensorId>, scalar: ScalarValue) -> &mut Self {
        let id: TensorId = id.into();
        let scalar = Box::new(OwnedScalar::from_scalar_value(scalar));
        let ptr = scalar.device_ptr();
        self.scalar_storage.insert(id, scalar);
        self.map.insert(id, ptr);
        if let Some(index) = self
            .ordered_ids
            .iter()
            .position(|candidate| *candidate == id)
        {
            self.ordered_pointers[index] = Some(ptr);
        }
        self
    }

    pub fn set<T: DataTypeLike>(
        &mut self,
        id: impl Into<TensorId>,
        memory: &mut DeviceMemory<T>,
    ) -> Result<&mut Self> {
        let id: TensorId = id.into();
        self.validate_tensor_id(id)?;
        self.validate_tensor_binding_type::<T>(id)?;
        Ok(unsafe { self.set_id(id, DevicePtr::from_raw(memory.as_mut_ptr().cast())) })
    }

    /// Binds a graph tensor to a raw device pointer after validating the tensor id.
    ///
    /// # Safety
    ///
    /// The caller must ensure `ptr` is valid for the tensor's data type and
    /// shape for the entire cuDNN execution using these bindings.
    pub unsafe fn set_ptr(&mut self, id: impl Into<TensorId>, ptr: DevicePtr) -> Result<&mut Self> {
        let id: TensorId = id.into();
        self.validate_tensor_id(id)?;
        Ok(unsafe { self.set_id(id, ptr) })
    }

    pub fn set_scalar(
        &mut self,
        id: impl Into<TensorId>,
        scalar: ScalarValue,
    ) -> Result<&mut Self> {
        let id: TensorId = id.into();
        let tensors = self.tensors.ok_or(Error::DescriptorMismatch {
            name: "graph-aware bindings".into(),
        })?;
        let record = tensors.get(&id).ok_or(Error::FrontendTensorNotFound(id))?;
        if scalar.data_type() != record.tensor.data_type {
            return Err(Error::FrontendScalarTypeMismatch {
                tensor_id: record.id,
                expected: record.tensor.data_type.type_name().into(),
                actual: scalar.type_name().into(),
            });
        }

        Ok(self.set_scalar_id(record.id, scalar))
    }

    pub fn ptr(&self, id: impl Into<TensorId>) -> Option<DevicePtr> {
        let id: TensorId = id.into();
        self.map.get(&id).copied()
    }

    fn refresh_scalar_bindings(&mut self) {
        for (&id, scalar) in &self.scalar_storage {
            let ptr = scalar.device_ptr();
            self.map.insert(id, ptr);
            if let Some(index) = self
                .ordered_ids
                .iter()
                .position(|candidate| *candidate == id)
            {
                self.ordered_pointers[index] = Some(ptr);
            }
        }
    }
}

impl RuntimeOverrides {
    /// Creates an empty runtime override set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an execution-time shape and stride override for a tensor ID.
    pub fn set(
        &mut self,
        tensor: TensorId,
        shape: impl Into<Vec<i64>>,
        strides: impl Into<Vec<i64>>,
    ) -> &mut Self {
        let shape = shape.into();
        let strides = strides.into();
        if let Some((_, existing_shape, existing_strides)) = self
            .overrides
            .iter_mut()
            .find(|(existing_tensor, _, _)| *existing_tensor == tensor)
        {
            *existing_shape = shape;
            *existing_strides = strides;
        } else {
            self.overrides.push((tensor, shape, strides));
        }
        self
    }

    /// Creates an override set containing one tensor override.
    pub fn with(
        tensor: TensorId,
        shape: impl Into<Vec<i64>>,
        strides: impl Into<Vec<i64>>,
    ) -> Self {
        let mut overrides = Self::new();
        overrides.set(tensor, shape, strides);
        overrides
    }

    /// Adds an execution-time override from a checked [`Shape`].
    pub fn set_shape(&mut self, tensor: TensorId, shape: &Shape) -> &mut Self {
        self.set(
            tensor,
            shape.dimensions().to_vec(),
            shape.strides().to_vec(),
        )
    }

    /// Returns whether no tensor overrides have been recorded.
    pub fn is_empty(&self) -> bool {
        self.overrides.is_empty()
    }

    fn pack_overrides(&self) -> Result<Vec<VariantPackOverride>> {
        self.overrides
            .iter()
            .map(|(tensor_id, shape, strides)| {
                if shape.len() != strides.len() {
                    return Err(Error::LengthMismatch {
                        name: "runtime_override_strides".into(),
                        expected: shape.len(),
                        actual: strides.len(),
                    });
                }

                Ok(VariantPackOverride {
                    id: *tensor_id,
                    dimensions: shape.clone(),
                    strides: strides.clone(),
                })
            })
            .collect()
    }

    fn pack_overrides_with_tensors(
        &self,
        tensors: &BTreeMap<TensorId, TensorRecord>,
        constraints: Option<&BTreeMap<TensorId, RuntimeShapeConstraints>>,
    ) -> Result<Vec<VariantPackOverride>> {
        self.overrides
            .iter()
            .map(|(tensor_id, shape, strides)| {
                let record =
                    validate_runtime_override(tensors, constraints, *tensor_id, shape, strides)?;

                Ok(VariantPackOverride {
                    id: record.id,
                    dimensions: shape.clone(),
                    strides: strides.clone(),
                })
            })
            .collect()
    }
}

impl<'a> RuntimeOverrideBuilder<'a> {
    fn with_constraints(
        tensors: &'a BTreeMap<TensorId, TensorRecord>,
        constraints: &'a BTreeMap<TensorId, RuntimeShapeConstraints>,
    ) -> Self {
        Self {
            tensors,
            constraints: Some(constraints),
            overrides: RuntimeOverrides::new(),
        }
    }

    pub fn set(
        &mut self,
        tensor: TensorId,
        shape: impl Into<Vec<i64>>,
        strides: impl Into<Vec<i64>>,
    ) -> Result<&mut Self> {
        let shape = shape.into();
        let strides = strides.into();
        validate_runtime_override(self.tensors, self.constraints, tensor, &shape, &strides)?;
        self.overrides.set(tensor, shape, strides);
        Ok(self)
    }

    pub fn set_shape(&mut self, tensor: TensorId, shape: &Shape) -> Result<&mut Self> {
        self.set(
            tensor,
            shape.dimensions().to_vec(),
            shape.strides().to_vec(),
        )
    }

    pub fn build(self) -> RuntimeOverrides {
        self.overrides
    }
}

fn validate_runtime_override<'a>(
    tensors: &'a BTreeMap<TensorId, TensorRecord>,
    constraints: Option<&BTreeMap<TensorId, RuntimeShapeConstraints>>,
    tensor_id: TensorId,
    shape: &[i64],
    strides: &[i64],
) -> Result<&'a TensorRecord> {
    let record = tensors
        .get(&tensor_id)
        .ok_or(Error::FrontendTensorNotFound(tensor_id))?;
    if shape.len() != strides.len() {
        return Err(Error::LengthMismatch {
            name: "runtime_override_strides".into(),
            expected: shape.len(),
            actual: strides.len(),
        });
    }
    if shape.len() != record.tensor.shape.dimensions().len() {
        return Err(Error::LengthMismatch {
            name: "runtime_override_rank".into(),
            expected: record.tensor.shape.dimensions().len(),
            actual: shape.len(),
        });
    }
    Shape::contiguous(shape)?.with_strides(strides.to_vec())?;
    if let Some(constraints) = constraints.and_then(|constraints| constraints.get(&tensor_id))
        && !constraints.contains(shape)
    {
        return Err(Error::OutOfRange {
            name: "runtime_override_shape_constraints".into(),
        });
    }
    Ok(record)
}

pub(crate) fn variant_pack_with_replacements_and_overrides_for_ids_and_tensors(
    binding_replacements: &[BindingReplacement],
    bindings: &Bindings<'_>,
    workspace: Option<&mut DeviceMemory<u8>>,
    runtime_overrides: &RuntimeOverrides,
    binding_ids: Option<&[TensorId]>,
    tensors: Option<&BTreeMap<TensorId, TensorRecord>>,
    constraints: Option<&BTreeMap<TensorId, RuntimeShapeConstraints>>,
) -> Result<VariantPack> {
    let mut pack_bindings =
        materialize_bindings_with_replacements(binding_replacements, bindings, binding_ids)?;
    pack_bindings.sort_unstable_by_key(|(id, _)| *id);
    let overrides = match tensors {
        Some(tensors) => runtime_overrides.pack_overrides_with_tensors(tensors, constraints)?,
        None => runtime_overrides.pack_overrides()?,
    };
    unsafe { VariantPack::create_with_overrides(&pack_bindings, workspace, &overrides) }
}

pub(crate) fn materialize_bindings_with_replacements(
    binding_replacements: &[BindingReplacement],
    bindings: &Bindings<'_>,
    binding_ids: Option<&[TensorId]>,
) -> Result<Vec<(TensorId, DevicePtr)>> {
    let mut pointer_map = bindings.map.clone();
    for replacement in binding_replacements {
        let source = pointer_map.get(&replacement.source_id).copied().ok_or(
            Error::FrontendBindingReplacementSourceNotBound {
                source_id: replacement.source_id,
                target_id: replacement.target_id,
                byte_offset: replacement.byte_offset,
            },
        )?;
        let adjusted = adjusted_alias_ptr(source, replacement.byte_offset)?;
        pointer_map.insert(replacement.target_id, adjusted);
    }
    match binding_ids {
        Some(binding_ids) => {
            let mut materialized_ids = binding_ids.to_vec();
            for replacement in binding_replacements {
                if let Some(index) = materialized_ids
                    .iter()
                    .position(|id| *id == replacement.source_id)
                {
                    materialized_ids[index] = replacement.target_id;
                } else if !materialized_ids.contains(&replacement.target_id) {
                    materialized_ids.push(replacement.target_id);
                }
            }
            materialized_ids.sort_unstable();
            materialized_ids.dedup();
            materialized_ids
                .iter()
                .map(|id| {
                    pointer_map
                        .get(id)
                        .copied()
                        .map(|ptr| (*id, ptr))
                        .ok_or(Error::FrontendTensorNotBound(*id))
                })
                .collect()
        }
        None => {
            let mut materialized_ids = bindings.ordered_ids.clone();
            for replacement in binding_replacements {
                if let Some(index) = materialized_ids
                    .iter()
                    .position(|id| *id == replacement.source_id)
                {
                    materialized_ids[index] = replacement.target_id;
                } else if !materialized_ids.contains(&replacement.target_id) {
                    materialized_ids.push(replacement.target_id);
                }
            }
            materialized_ids.sort_unstable();
            materialized_ids.dedup();
            materialized_ids
                .iter()
                .map(|id| {
                    pointer_map
                        .get(id)
                        .copied()
                        .map(|ptr| (*id, ptr))
                        .ok_or(Error::FrontendTensorNotBound(*id))
                })
                .collect()
        }
    }
}

fn adjusted_alias_ptr(source: DevicePtr, byte_offset: i64) -> Result<DevicePtr> {
    let byte_offset = usize::try_from(byte_offset).map_err(|_| Error::OutOfRange {
        name: "alias byte offset".into(),
    })?;
    let raw = source.as_raw().cast::<u8>();
    Ok(DevicePtr::from_raw(unsafe { raw.add(byte_offset) }.cast()))
}

impl CompiledGraph {
    pub(crate) fn variant_pack_with_overrides(
        &self,
        bindings: &Bindings<'_>,
        workspace: Option<&mut DeviceMemory<u8>>,
        runtime_overrides: &RuntimeOverrides,
    ) -> Result<VariantPack> {
        self.validate_runtime_overrides_allowed(runtime_overrides)?;
        variant_pack_with_replacements_and_overrides_for_ids_and_tensors(
            self.binding_replacements(),
            bindings,
            workspace,
            runtime_overrides,
            None,
            Some(&self.tensors),
            Some(&self.graph().dynamic_shape_constraints),
        )
    }

    fn cuda_graph_variant_pack(
        &self,
        bindings: &Bindings<'_>,
        workspace: Option<&mut DeviceMemory<u8>>,
        runtime_overrides: &RuntimeOverrides,
    ) -> Result<VariantPack> {
        self.validate_runtime_overrides_allowed(runtime_overrides)?;
        let binding_ids = self.cuda_graph_binding_ids();
        variant_pack_with_replacements_and_overrides_for_ids_and_tensors(
            self.binding_replacements(),
            bindings,
            workspace,
            runtime_overrides,
            Some(&binding_ids),
            Some(&self.tensors),
            Some(&self.graph().dynamic_shape_constraints),
        )
    }

    /// Creates a binding builder pre-populated with graph-owned scalar values.
    pub fn bindings(&self) -> Bindings<'_> {
        let mut bindings = Bindings::with_compiled_template(self, &self.binding_template_ids);
        for record in self.tensors.values() {
            if let Some(scalar_value) = record.tensor.scalar_value {
                bindings.set_scalar_id(record.id, scalar_value);
            }
        }
        for &(id, scalar_value) in &self.scalar_bindings {
            bindings.set_scalar_id(id, scalar_value);
        }
        bindings
    }

    /// Creates a graph-aware runtime override builder for this compiled graph.
    pub fn runtime_overrides(&self) -> RuntimeOverrideBuilder<'_> {
        RuntimeOverrideBuilder::with_constraints(
            &self.tensors,
            &self.graph().dynamic_shape_constraints,
        )
    }

    /// Executes the default plan with the supplied tensor bindings.
    pub fn execute(
        &self,
        ctx: &Context,
        bindings: &Bindings<'_>,
        workspace: Option<&mut DeviceMemory<u8>>,
    ) -> Result<()> {
        self.execute_at(ctx, bindings, workspace, self.default_plan_index())
    }

    /// Executes the default plan with execution-time shape and stride overrides.
    pub fn execute_with_runtime_overrides(
        &self,
        ctx: &Context,
        bindings: &Bindings<'_>,
        workspace: Option<&mut DeviceMemory<u8>>,
        runtime_overrides: &RuntimeOverrides,
    ) -> Result<()> {
        self.execute_at_with_runtime_overrides(
            ctx,
            bindings,
            workspace,
            self.default_plan_index(),
            runtime_overrides,
        )
    }

    /// Executes a specific plan by index.
    pub fn execute_at(
        &self,
        ctx: &Context,
        bindings: &Bindings<'_>,
        workspace: Option<&mut DeviceMemory<u8>>,
        plan_index: usize,
    ) -> Result<()> {
        self.execute_at_with_runtime_overrides(
            ctx,
            bindings,
            workspace,
            plan_index,
            &RuntimeOverrides::new(),
        )
    }

    /// Executes a specific plan by index with runtime shape and stride overrides.
    pub fn execute_at_with_runtime_overrides(
        &self,
        ctx: &Context,
        bindings: &Bindings<'_>,
        workspace: Option<&mut DeviceMemory<u8>>,
        plan_index: usize,
        runtime_overrides: &RuntimeOverrides,
    ) -> Result<()> {
        let variant_pack = self.cuda_graph_variant_pack(bindings, workspace, runtime_overrides)?;
        self.execution_plan_at(plan_index)?
            .execute(ctx, &variant_pack)
    }

    fn validate_runtime_overrides_allowed(
        &self,
        runtime_overrides: &RuntimeOverrides,
    ) -> Result<()> {
        if !runtime_overrides.is_empty()
            && !self.graph().dynamic_shape_enabled()
            && !self.graph().override_shape_enabled()
        {
            return Err(Error::FrontendRuntimeOverridesRequireShapeEnabledGraph);
        }
        Ok(())
    }

    /// Populates a CUDA graph with the default cuDNN execution plan.
    pub fn populate_cuda_graph(
        &self,
        ctx: &Context,
        bindings: &Bindings<'_>,
        workspace: Option<&mut DeviceMemory<u8>>,
        graph: &mut CudaGraph,
    ) -> Result<()> {
        self.populate_cuda_graph_at(ctx, bindings, workspace, graph, self.default_plan_index())
    }

    /// Populates a CUDA graph with runtime shape and stride overrides.
    pub fn populate_cuda_graph_with_runtime_overrides(
        &self,
        ctx: &Context,
        bindings: &Bindings<'_>,
        workspace: Option<&mut DeviceMemory<u8>>,
        graph: &mut CudaGraph,
        runtime_overrides: &RuntimeOverrides,
    ) -> Result<()> {
        self.populate_cuda_graph_at_with_runtime_overrides(
            ctx,
            bindings,
            workspace,
            graph,
            self.default_plan_index(),
            runtime_overrides,
        )
    }

    /// Appends the default plan to an existing CUDA graph as a child graph node.
    pub fn append_to_cuda_graph(
        &self,
        ctx: &Context,
        bindings: &Bindings<'_>,
        workspace: Option<&mut DeviceMemory<u8>>,
        graph: &mut CudaGraph,
        dependencies: &[GraphNode],
    ) -> Result<GraphNode> {
        self.append_to_cuda_graph_at(
            ctx,
            bindings,
            workspace,
            graph,
            dependencies,
            self.default_plan_index(),
        )
    }

    /// Appends the default plan to a CUDA graph with runtime overrides.
    pub fn append_to_cuda_graph_with_runtime_overrides(
        &self,
        ctx: &Context,
        bindings: &Bindings<'_>,
        workspace: Option<&mut DeviceMemory<u8>>,
        graph: &mut CudaGraph,
        dependencies: &[GraphNode],
        runtime_overrides: &RuntimeOverrides,
    ) -> Result<GraphNode> {
        self.append_to_cuda_graph_at_with_runtime_overrides(
            ctx,
            bindings,
            workspace,
            graph,
            dependencies,
            self.default_plan_index(),
            runtime_overrides,
        )
    }

    /// Populates a fresh cuDNN child graph and appends it to an existing CUDA graph.
    pub fn append_to_cuda_graph_at(
        &self,
        ctx: &Context,
        bindings: &Bindings<'_>,
        workspace: Option<&mut DeviceMemory<u8>>,
        graph: &mut CudaGraph,
        dependencies: &[GraphNode],
        plan_index: usize,
    ) -> Result<GraphNode> {
        self.append_to_cuda_graph_at_with_runtime_overrides(
            ctx,
            bindings,
            workspace,
            graph,
            dependencies,
            plan_index,
            &RuntimeOverrides::new(),
        )
    }

    /// Appends a specific plan to a CUDA graph with runtime overrides.
    pub fn append_to_cuda_graph_at_with_runtime_overrides(
        &self,
        ctx: &Context,
        bindings: &Bindings<'_>,
        workspace: Option<&mut DeviceMemory<u8>>,
        graph: &mut CudaGraph,
        dependencies: &[GraphNode],
        plan_index: usize,
        runtime_overrides: &RuntimeOverrides,
    ) -> Result<GraphNode> {
        let mut child_graph = CudaGraph::create()?;
        self.populate_cuda_graph_at_with_runtime_overrides(
            ctx,
            bindings,
            workspace,
            &mut child_graph,
            plan_index,
            runtime_overrides,
        )?;
        Ok(graph.add_child_graph_node(dependencies, &child_graph)?)
    }

    /// Populates an empty CUDA graph with a specific plan.
    pub fn populate_cuda_graph_at(
        &self,
        ctx: &Context,
        bindings: &Bindings<'_>,
        workspace: Option<&mut DeviceMemory<u8>>,
        graph: &mut CudaGraph,
        plan_index: usize,
    ) -> Result<()> {
        self.populate_cuda_graph_at_with_runtime_overrides(
            ctx,
            bindings,
            workspace,
            graph,
            plan_index,
            &RuntimeOverrides::new(),
        )
    }

    /// Populates an empty CUDA graph with a specific plan and runtime overrides.
    pub fn populate_cuda_graph_at_with_runtime_overrides(
        &self,
        ctx: &Context,
        bindings: &Bindings<'_>,
        workspace: Option<&mut DeviceMemory<u8>>,
        graph: &mut CudaGraph,
        plan_index: usize,
        runtime_overrides: &RuntimeOverrides,
    ) -> Result<()> {
        let variant_pack = self.cuda_graph_variant_pack(bindings, workspace, runtime_overrides)?;
        if !graph.root_nodes()?.is_empty() {
            return Err(Error::FrontendCudaGraphPopulateRequiresEmptyGraph);
        }

        self.execution_plan_at(plan_index)?
            .populate_graph(ctx, &variant_pack, graph)?;
        Ok(())
    }

    /// Updates an existing CUDA graph populated with the default plan.
    pub fn update_cuda_graph(
        &self,
        ctx: &Context,
        bindings: &Bindings<'_>,
        workspace: Option<&mut DeviceMemory<u8>>,
        graph: &CudaGraph,
    ) -> Result<()> {
        self.update_cuda_graph_at(ctx, bindings, workspace, graph, self.default_plan_index())
    }

    /// Updates an existing CUDA graph with runtime shape and stride overrides.
    pub fn update_cuda_graph_with_runtime_overrides(
        &self,
        ctx: &Context,
        bindings: &Bindings<'_>,
        workspace: Option<&mut DeviceMemory<u8>>,
        graph: &CudaGraph,
        runtime_overrides: &RuntimeOverrides,
    ) -> Result<()> {
        self.update_cuda_graph_at_with_runtime_overrides(
            ctx,
            bindings,
            workspace,
            graph,
            self.default_plan_index(),
            runtime_overrides,
        )
    }

    /// Rebuilds the cuDNN child graph referenced by `node` and optionally pushes
    /// the updated child graph into an instantiated executable graph.
    pub fn update_cuda_graph_node(
        &self,
        ctx: &Context,
        bindings: &Bindings<'_>,
        workspace: Option<&mut DeviceMemory<u8>>,
        node: GraphNode,
        exec: Option<&mut ExecutableGraph>,
    ) -> Result<()> {
        self.update_cuda_graph_node_at(
            ctx,
            bindings,
            workspace,
            node,
            exec,
            self.default_plan_index(),
        )
    }

    /// Updates a CUDA child graph node with runtime overrides.
    pub fn update_cuda_graph_node_with_runtime_overrides(
        &self,
        ctx: &Context,
        bindings: &Bindings<'_>,
        workspace: Option<&mut DeviceMemory<u8>>,
        node: GraphNode,
        exec: Option<&mut ExecutableGraph>,
        runtime_overrides: &RuntimeOverrides,
    ) -> Result<()> {
        self.update_cuda_graph_node_at_with_runtime_overrides(
            ctx,
            bindings,
            workspace,
            node,
            exec,
            self.default_plan_index(),
            runtime_overrides,
        )
    }

    /// Updates a CUDA child graph node using a specific plan.
    pub fn update_cuda_graph_node_at(
        &self,
        ctx: &Context,
        bindings: &Bindings<'_>,
        workspace: Option<&mut DeviceMemory<u8>>,
        node: GraphNode,
        exec: Option<&mut ExecutableGraph>,
        plan_index: usize,
    ) -> Result<()> {
        self.update_cuda_graph_node_at_with_runtime_overrides(
            ctx,
            bindings,
            workspace,
            node,
            exec,
            plan_index,
            &RuntimeOverrides::new(),
        )
    }

    /// Updates a CUDA child graph node using a specific plan and runtime overrides.
    pub fn update_cuda_graph_node_at_with_runtime_overrides(
        &self,
        ctx: &Context,
        bindings: &Bindings<'_>,
        workspace: Option<&mut DeviceMemory<u8>>,
        node: GraphNode,
        mut exec: Option<&mut ExecutableGraph>,
        plan_index: usize,
        runtime_overrides: &RuntimeOverrides,
    ) -> Result<()> {
        let child_graph = node.child_graph()?;
        self.update_cuda_graph_at_with_runtime_overrides(
            ctx,
            bindings,
            workspace,
            &child_graph,
            plan_index,
            runtime_overrides,
        )?;
        if let Some(exec) = exec.as_mut() {
            exec.set_child_graph_node(node, &child_graph)?;
        }
        Ok(())
    }

    /// Updates an existing CUDA graph using a specific plan.
    pub fn update_cuda_graph_at(
        &self,
        ctx: &Context,
        bindings: &Bindings<'_>,
        workspace: Option<&mut DeviceMemory<u8>>,
        graph: &CudaGraph,
        plan_index: usize,
    ) -> Result<()> {
        self.update_cuda_graph_at_with_runtime_overrides(
            ctx,
            bindings,
            workspace,
            graph,
            plan_index,
            &RuntimeOverrides::new(),
        )
    }

    /// Updates an existing CUDA graph using a specific plan and runtime overrides.
    pub fn update_cuda_graph_at_with_runtime_overrides(
        &self,
        ctx: &Context,
        bindings: &Bindings<'_>,
        workspace: Option<&mut DeviceMemory<u8>>,
        graph: &CudaGraph,
        plan_index: usize,
        runtime_overrides: &RuntimeOverrides,
    ) -> Result<()> {
        self.validate_runtime_overrides_allowed(runtime_overrides)?;
        let variant_pack =
            self.variant_pack_with_overrides(bindings, workspace, runtime_overrides)?;
        self.execution_plan_at(plan_index)?
            .update_graph(ctx, &variant_pack, graph)
    }
}

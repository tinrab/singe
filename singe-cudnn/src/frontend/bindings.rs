mod overrides;
mod owned;
mod template;

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    mem::{MaybeUninit, size_of},
    sync::Arc,
};

use singe_cuda::{
    context::Context as CudaContext,
    memory::DeviceMemory,
    submission::{Access, AccessMode},
    types::DevicePtr,
    view::{DeviceInput, DeviceOutput, DeviceRegion, DeviceRegionProvider},
};

pub(crate) use crate::frontend::bindings::template::{BindingSource, BindingTemplate};
use crate::{
    backend::{
        execution::{VariantPack, VariantPackOverride, WorkspaceInitialization},
        tensor::{Shape, TensorId, TensorPlacement},
    },
    data_type::{DataType, DataTypeLike},
    error::{Error, Result},
    frontend::{
        error::Error as FrontendError,
        graph::{RuntimeShapeConstraints, TensorRecord},
        plan::{BindingReplacement, CompiledGraph},
        workspace::WorkspaceLayout,
    },
    scalar::{ScalarStorage, ScalarValue},
    utility::to_usize,
};

/// Mapping from frontend tensor IDs to device pointers for execution.
pub type BindingMap = HashMap<TensorId, DevicePtr>;

/// Permitted accesses to registered device storage, independent of initialization.
/// Execution still requires ordered producers, valid values and exclusive writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingAccess {
    /// Permits native reads and rejects native writes when preparing graph execution.
    ReadOnly,
    /// Permits native reads and writes subject to the execution access contract.
    ReadWrite,
}

/// Runtime tensor bindings for executing a compiled frontend graph.
///
/// Maps frontend tensor IDs to device pointers and owns registered by-value scalars.
/// Registering a pointer records binding metadata. It does not establish asynchronous
/// access rights. [`Self::set_owned_region`] retains its allocation, while borrowed
/// setters retain pointer metadata only. The graph lifetime parameter
/// borrows tensor definitions, not the bound device buffers.
///
/// Raw execution and CUDA graph methods are unsafe. Their caller must satisfy these
/// requirements for all actual native accesses, including accesses after a failed call
/// that partially submitted work:
///
/// - Device buffers and workspace belong to the execution context and have the native
///   encoding, alignment and physical capacity required by their effective layouts.
///   This includes runtime overrides, packed storage, aliases and indirect addressing.
/// - Values read at graph entry are initialized and their producers are ordered before execution.
///   Internal producers must write every value consumed later in the graph.
///   Outputs and workspace permit exclusive writes, except for aliases supported by
///   the operation. Conflicting accesses wait until execution completes.
/// - Allocations and required plan code stay alive through execution. Captured addresses
///   stay valid for every replay, including executable snapshots and pending launches.
///   Updating a graph does not release storage still referenced by an older executable.
/// - Host scalar and argument storage stays valid while cuDNN consumes it. Retain any
///   host addresses embedded in a captured graph through its replays.
///
/// Raw execution does not mark typed output storage initialized. Outputs become usable
/// only after successful completion. Owned execution APIs retain the resources supplied
/// to them and resolve outputs through the CUDA submission machinery.
/// Setters and lookups take frontend [`TensorId`] values directly.
/// Runtime integers can be parsed with [`TensorId::try_new`] before binding.
#[derive(Debug, Default)]
pub struct Bindings<'a> {
    tensors: Option<&'a BTreeMap<TensorId, TensorRecord>>,
    replacements: &'a [BindingReplacement],
    map: BindingMap,
    scalar_storage: HashMap<TensorId, Box<ScalarStorage>>,
    memory_info: HashMap<TensorId, BindingMemoryInfo>,
}

/// Established allocation metadata, independent of the caller's asynchronous ownership/access contract.
#[derive(Debug, Clone)]
struct BindingMemoryInfo {
    context: Arc<CudaContext>,
    byte_capacity: usize,
    data_type: DataType,
    access: BindingAccess,
    tracked_access: Option<Access>,
    owner: Option<Arc<owned::RetainedRegion>>,
}

/// Runtime shape and stride overrides for dynamic-shape execution.
///
/// The override-shape API allows execution-time tensor shapes to differ from
/// the shapes used when building the graph. The override tensor IDs, shapes, and
/// strides are passed with execute/CUDA graph calls.
/// Captured tensors inherit source overrides, including index/scalar rank promotion.
/// Explicit native overrides take precedence, and explicit storage views retain independent layouts.
/// Each override owns a checked [`Shape`]. The selected plan validates its native rank,
/// bounds and addressing requirements before execution or a runtime workspace query.
#[derive(Debug, Default, Clone)]
pub struct RuntimeOverrides {
    overrides: BTreeMap<TensorId, Shape>,
}

/// Graph-aware builder for runtime shape and stride overrides.
///
/// Validates declared layouts and source bounds as they are added.
/// Captured layouts are resolved and validated when the overrides are used by a plan.
#[derive(Debug)]
pub struct RuntimeOverrideBuilder<'a> {
    tensors: &'a BTreeMap<TensorId, TensorRecord>,
    replacements: &'a [BindingReplacement],
    constraints: &'a BTreeMap<TensorId, RuntimeShapeConstraints>,
    overrides: RuntimeOverrides,
}

impl Clone for Bindings<'_> {
    fn clone(&self) -> Self {
        let mut cloned = Self {
            tensors: self.tensors,
            replacements: self.replacements,
            map: self.map.clone(),
            scalar_storage: self.scalar_storage.clone(),
            memory_info: self.memory_info.clone(),
        };
        cloned.refresh_scalar_bindings();
        cloned
    }
}

impl<'a> Bindings<'a> {
    /// Creates empty bindings without a graph template or scalar defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of registered pointers, including owned scalar defaults.
    /// Graph declarations without a registration do not contribute to this count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether this set has no registered pointers, including scalar defaults.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Iterates registered frontend identities and pointers without allocating or resolving aliases.
    /// Includes owned scalar pointers and uses no particular iteration order.
    /// Execution preparation resolves workspace and aliases separately from these registrations.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (TensorId, DevicePtr)> + '_ {
        self.map.iter().map(|(&id, &pointer)| (id, pointer))
    }

    pub(crate) fn with_graph_template(
        tensors: &'a BTreeMap<TensorId, TensorRecord>,
        replacements: &'a [BindingReplacement],
    ) -> Self {
        Self {
            tensors: Some(tensors),
            replacements,
            map: HashMap::new(),
            scalar_storage: HashMap::new(),
            memory_info: HashMap::new(),
        }
    }

    /// Replaces borrowed tensor definitions and upgrades owned scalar alignment when required.
    pub(crate) fn with_tensor_metadata<'graph>(
        self,
        tensors: &'graph BTreeMap<TensorId, TensorRecord>,
        replacements: &'graph [BindingReplacement],
    ) -> Result<Bindings<'graph>> {
        let Self {
            map,
            scalar_storage,
            memory_info,
            ..
        } = self;
        let mut bindings = Bindings {
            tensors: Some(tensors),
            replacements,
            map,
            scalar_storage: HashMap::with_capacity(scalar_storage.len()),
            memory_info,
        };
        for (id, mut scalar) in scalar_storage {
            let alignment = bindings.scalar_alignment(id)?;
            if alignment > scalar.byte_alignment() {
                *scalar = ScalarStorage::create(scalar.value(), alignment)?;
            }
            bindings.scalar_storage.insert(id, scalar);
        }
        bindings.refresh_scalar_bindings();
        Ok(bindings)
    }

    /// Binds a tensor id to a raw device pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure `ptr` is valid for the tensor's data type and
    /// shape for the entire cuDNN execution using these bindings.
    pub unsafe fn set_id(&mut self, id: TensorId, ptr: DevicePtr) -> &mut Self {
        self.scalar_storage.remove(&id);
        self.memory_info.remove(&id);
        self.map.insert(id, ptr);
        self
    }

    fn validate_tensor_id(&self, id: TensorId) -> Result<()> {
        if let Some(tensors) = self.tensors {
            tensors
                .contains_key(&id)
                .then_some(())
                .ok_or(FrontendError::TensorNotFound(id))?;
        }
        Ok(())
    }

    fn validate_tensor_binding_type<T: DataTypeLike>(&self, id: TensorId) -> Result<()> {
        if let Some(tensors) = self.tensors {
            let record = tensors.get(&id).ok_or(FrontendError::TensorNotFound(id))?;
            let actual = T::data_type();
            if actual != record.tensor.data_type {
                return Err(FrontendError::BindingTypeMismatch {
                    tensor_id: record.id,
                    expected: record.tensor.data_type.type_name(),
                    actual: actual.type_name(),
                }
                .into());
            }
        }
        Ok(())
    }

    /// Owns a runtime scalar using its declared native encoding and strongest captured-use alignment.
    /// Graph-aware bindings reject unknown IDs, device tensors and compile-time constants.
    /// Detached bindings use sixteen-byte alignment and defer tensor validation until graph execution preparation.
    /// A rejected assignment leaves the previous binding unchanged.
    ///
    /// # Errors
    /// Returns an error for incompatible graph metadata, invalid alignment or scalar allocation failure.
    pub fn set_scalar(
        &mut self,
        id: TensorId,
        scalar: impl Into<ScalarValue>,
    ) -> Result<&mut Self> {
        let scalar = scalar.into();
        if let Some(tensors) = self.tensors {
            let record = tensors.get(&id).ok_or(FrontendError::TensorNotFound(id))?;
            if !record.tensor.is_runtime_scalar() {
                return Err(FrontendError::BindingPlacementMismatch {
                    tensor_id: id,
                    expected: "runtime scalar",
                    actual: record.tensor.placement,
                }
                .into());
            }
            if scalar.data_type() != record.tensor.data_type {
                return Err(FrontendError::ScalarTypeMismatch {
                    tensor_id: id,
                    expected: record.tensor.data_type.type_name(),
                    actual: scalar.type_name(),
                }
                .into());
            }
        }
        let alignment = self.scalar_alignment(id)?;
        let scalar = Box::new(ScalarStorage::create(scalar, alignment)?);
        self.memory_info.remove(&id);
        let ptr = scalar.device_ptr();
        self.scalar_storage.insert(id, scalar);
        self.map.insert(id, ptr);
        Ok(self)
    }

    fn scalar_alignment(&self, id: TensorId) -> Result<usize> {
        let Some(tensors) = self.tensors else {
            return Ok(16);
        };
        let record = tensors.get(&id).ok_or(FrontendError::TensorNotFound(id))?;
        let mut alignment = to_usize(record.tensor.byte_alignment, "scalar byte alignment")?;
        if self.replacements.is_empty() {
            return Ok(alignment);
        }
        for (&native_id, record) in tensors {
            if record.access.is_none() || !record.tensor.is_runtime_scalar() {
                continue;
            }
            let (source, offset) = resolve_binding_source(self.replacements, native_id)?;
            if source != id {
                continue;
            }
            if offset != 0 {
                return Err(Error::DescriptorMismatch {
                    name: "runtime scalar alias byte offset".into(),
                });
            }
            alignment = alignment.max(to_usize(
                record.tensor.byte_alignment,
                "scalar byte alignment",
            )?);
        }
        Ok(alignment)
    }

    /// Binds an ordinary Rust element array with the matching native tensor encoding.
    /// Packed tensor formats require [`Self::set_encoded_bytes`].
    /// Capacity is checked using the effective execution-time layout, allowing smaller runtime shapes.
    ///
    /// # Errors
    ///
    /// Returns an error if a graph-aware tensor identity is unknown, its native encoding differs from the Rust element type, the encoding requires packed byte storage, or its placement is not caller-supplied device storage.
    pub fn set<T: DataTypeLike>(
        &mut self,
        id: TensorId,
        memory: &mut (impl DeviceOutput<T> + ?Sized),
    ) -> Result<&mut Self> {
        self.set_typed_binding::<T>(id, memory, BindingAccess::ReadWrite)
    }

    /// Binds typed input metadata without requiring a mutable device view.
    /// Native writes through this binding, including aliases and captured copies,
    /// are rejected when preparing execution.
    /// This does not retain storage or establish initialized values or producer ordering.
    /// Owned execution derives access declarations from this metadata; retain the allocation separately.
    ///
    /// # Errors
    ///
    /// Returns an error if a graph-aware tensor identity is unknown, its native encoding differs from the Rust element type, the encoding requires packed byte storage, or its placement is not caller-supplied device storage.
    pub fn set_input<T: DataTypeLike>(
        &mut self,
        id: TensorId,
        memory: &(impl DeviceInput<T> + ?Sized),
    ) -> Result<&mut Self> {
        self.set_typed_binding::<T>(id, memory, BindingAccess::ReadOnly)
    }

    fn set_typed_binding<T: DataTypeLike>(
        &mut self,
        id: TensorId,
        memory: &(impl DeviceInput<T> + ?Sized),
        access: BindingAccess,
    ) -> Result<&mut Self> {
        self.validate_tensor_id(id)?;
        self.validate_tensor_binding_type::<T>(id)?;
        if T::data_type().storage_bits() != size_of::<T>() * 8 {
            return Err(FrontendError::PackedBindingRequiresBytes {
                tensor_id: id,
                data_type: T::data_type(),
            }
            .into());
        }
        self.validate_device_placement(id)?;
        let info = binding_memory_info(memory.region(), T::data_type(), access);
        // SAFETY: the pointer and metadata describe this typed view; execution retains its existing ownership/access contract.
        unsafe {
            self.set_id(
                id,
                DevicePtr::from_raw(memory.as_device_ptr().cast_mut().cast()),
            );
        }
        self.memory_info.insert(id, info);
        Ok(self)
    }

    /// Binds bytes encoded in the graph tensor's native cuDNN storage format.
    /// Use this for packed boolean, FP4, INT4 or UINT4 tensors, and for buffers whose encoding is selected at runtime.
    /// Ordinary Rust arrays of scalar values do not have the native sub-byte packing.
    /// Capacity and context are checked against the actual runtime layouts when constructing the variant pack.
    ///
    /// # Errors
    ///
    /// Returns an error if graph metadata is absent, the tensor identity is unknown, the encoding is compute-only, or the tensor placement is not caller-supplied device storage.
    pub fn set_encoded_bytes(
        &mut self,
        id: TensorId,
        memory: &mut (impl DeviceOutput<u8> + ?Sized),
    ) -> Result<&mut Self> {
        self.set_encoded_binding(id, memory, BindingAccess::ReadWrite)
    }

    /// Binds read-only input bytes in the graph tensor's native storage encoding.
    /// Supports packed encodings without requiring a mutable byte view.
    /// Execution retains the same initialization and lifetime contract as [`Self::set_input`].
    ///
    /// # Errors
    ///
    /// Returns an error if graph metadata is absent, the tensor identity is unknown, the encoding is compute-only, or the tensor placement is not caller-supplied device storage.
    pub fn set_encoded_input_bytes(
        &mut self,
        id: TensorId,
        memory: &(impl DeviceInput<u8> + ?Sized),
    ) -> Result<&mut Self> {
        self.set_encoded_binding(id, memory, BindingAccess::ReadOnly)
    }

    fn set_encoded_binding(
        &mut self,
        id: TensorId,
        memory: &(impl DeviceInput<u8> + ?Sized),
        access: BindingAccess,
    ) -> Result<&mut Self> {
        let tensors = self
            .tensors
            .ok_or(FrontendError::GraphAwareBindingsRequired)?;
        let record = tensors.get(&id).ok_or(FrontendError::TensorNotFound(id))?;
        let data_type = record.tensor.data_type;
        if data_type == DataType::FastFloatForFp8 {
            return Err(Error::UnsupportedDataType);
        }
        self.validate_device_placement(id)?;
        let info = binding_memory_info(memory.region(), data_type, access);
        // SAFETY: the byte pointer matches the recorded storage; packing and asynchronous validity are part of execution's contract.
        unsafe {
            self.set_id(
                id,
                DevicePtr::from_raw(memory.as_device_ptr().cast_mut().cast()),
            );
        }
        self.memory_info.insert(id, info);
        Ok(self)
    }

    fn validate_device_placement(&self, id: TensorId) -> Result<()> {
        if let Some(record) = self.tensors.and_then(|tensors| tensors.get(&id))
            && record.tensor.placement != TensorPlacement::Device
        {
            return Err(FrontendError::BindingPlacementMismatch {
                tensor_id: id,
                expected: "device storage",
                actual: record.tensor.placement,
            }
            .into());
        }
        Ok(())
    }

    /// Known byte capacity of a typed, encoded-byte or explicitly sized raw binding.
    #[must_use]
    pub fn byte_capacity(&self, id: TensorId) -> Option<usize> {
        self.memory_info.get(&id).map(|info| info.byte_capacity)
    }

    pub(crate) fn binding_access(&self, id: TensorId, mode: AccessMode) -> Option<Access> {
        self.memory_info
            .get(&id)
            .and_then(|info| info.tracked_access.as_ref())
            .map(|access| access.clone().with_mode(mode))
    }

    pub(crate) fn validate_runtime_binding(
        &self,
        allocation_context: &Arc<CudaContext>,
        source: BindingSource,
        record: &TensorRecord,
        layout: Option<&VariantPackOverride>,
    ) -> Result<()> {
        let BindingSource {
            tensor: binding_id,
            byte_offset,
        } = source;
        if let Some(scalar) = self.scalar_storage.get(&binding_id) {
            if !record.tensor.is_runtime_scalar() {
                return Err(FrontendError::BindingPlacementMismatch {
                    tensor_id: record.id,
                    expected: "runtime scalar",
                    actual: record.tensor.placement,
                }
                .into());
            }
            if scalar.value().data_type() != record.tensor.data_type {
                return Err(FrontendError::ScalarTypeMismatch {
                    tensor_id: record.id,
                    expected: record.tensor.data_type.type_name(),
                    actual: scalar.value().type_name(),
                }
                .into());
            }
        }
        let Some(info) = self.memory_info.get(&binding_id) else {
            return Ok(());
        };
        if info.access == BindingAccess::ReadOnly
            && record
                .access
                .is_some_and(super::operation::TensorAccess::writes)
        {
            return Err(FrontendError::BindingReadOnly {
                binding_id,
                tensor_id: record.id,
            }
            .into());
        }
        if info.context.as_ref() != allocation_context.as_ref() {
            return Err(Error::ContextMismatch {
                name: format!("frontend binding {binding_id}").into(),
            });
        }
        if record.tensor.placement != TensorPlacement::Device {
            return Err(FrontendError::BindingPlacementMismatch {
                tensor_id: record.id,
                expected: "device storage",
                actual: record.tensor.placement,
            }
            .into());
        }
        if info.data_type != record.tensor.data_type {
            return Err(FrontendError::BindingTypeMismatch {
                tensor_id: record.id,
                expected: record.tensor.data_type.type_name(),
                actual: info.data_type.type_name(),
            }
            .into());
        }
        let mut spec = record.tensor.clone();
        if let Some(layout) = layout {
            spec.shape = layout.shape.clone();
        }
        let offset = to_usize(byte_offset, "alias byte offset")?;
        let remaining_capacity = info.byte_capacity.checked_sub(offset).ok_or(
            FrontendError::BindingCapacityMismatch {
                binding_id,
                tensor_id: record.id,
                required: offset,
                actual: info.byte_capacity,
            },
        )?;
        let span = spec.storage_byte_len(Some(remaining_capacity))?;
        let required = offset
            .checked_add(span)
            .ok_or_else(|| Error::out_of_range("binding storage bytes"))?;
        if required > info.byte_capacity {
            return Err(FrontendError::BindingCapacityMismatch {
                binding_id,
                tensor_id: record.id,
                required,
                actual: info.byte_capacity,
            }
            .into());
        }
        Ok(())
    }

    /// Binds a graph tensor to a raw device pointer after validating the tensor id.
    ///
    /// # Safety
    ///
    /// The caller must ensure `ptr` is valid for the tensor's data type and
    /// shape for the entire cuDNN execution using these bindings.
    ///
    /// # Errors
    ///
    /// Returns an error if graph metadata is present and the tensor identity is unknown.
    pub unsafe fn set_ptr(&mut self, id: TensorId, ptr: DevicePtr) -> Result<&mut Self> {
        self.validate_tensor_id(id)?;
        // SAFETY: the caller upholds this function's safety contract.
        Ok(unsafe { self.set_id(id, ptr) })
    }

    /// Binds a raw device buffer with an explicit capacity and CUDA context.
    /// Graph metadata supplies its native encoding. Use this for externally managed
    /// packed/ragged buffers whose physical span cannot be inferred from dimensions.
    ///
    /// # Safety
    /// The pointer must refer to at least `byte_capacity` accessible bytes in `allocation_context`
    /// and satisfy the graph's encoding, access, alignment and aliasing requirements.
    /// The caller retains the allocation and orders its producers through every execution.
    /// Recording metadata here does not retain the underlying allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if graph metadata is absent, the tensor identity is unknown, or its placement is not caller-supplied device storage.
    pub unsafe fn set_raw_buffer(
        &mut self,
        id: TensorId,
        allocation_context: &Arc<CudaContext>,
        ptr: DevicePtr,
        byte_capacity: usize,
    ) -> Result<&mut Self> {
        let tensors = self
            .tensors
            .ok_or(FrontendError::GraphAwareBindingsRequired)?;
        let record = tensors.get(&id).ok_or(FrontendError::TensorNotFound(id))?;
        self.validate_device_placement(id)?;
        let info = BindingMemoryInfo {
            context: Arc::clone(allocation_context),
            byte_capacity,
            data_type: record.tensor.data_type,
            access: BindingAccess::ReadWrite,
            tracked_access: None,
            owner: None,
        };
        // SAFETY: the caller establishes the raw allocation contract above.
        unsafe { self.set_id(id, ptr) };
        self.memory_info.insert(id, info);
        Ok(self)
    }

    /// Removes a registration and releases this binding set's scalar storage or retained region.
    /// Returns whether the tensor was bound; removing an absent tensor has no effect.
    /// Other binding clones and prepared owned executions retain their own storage references.
    /// Removed scalar defaults are not restored automatically.
    pub fn remove(&mut self, id: TensorId) -> bool {
        let removed = self.map.remove(&id).is_some();
        self.scalar_storage.remove(&id);
        self.memory_info.remove(&id);
        removed
    }

    /// Removes all registrations, including scalar defaults, while retaining the graph template.
    /// Releases this binding set's retained storage and access metadata.
    /// Required tensors must be rebound before execution.
    /// Create fresh graph bindings to restore their initial scalar defaults.
    pub fn clear(&mut self) {
        self.map.clear();
        self.scalar_storage.clear();
        self.memory_info.clear();
    }

    /// Returns the registered pointer for this frontend identity, without resolving aliases or workspace bindings.
    #[must_use]
    pub fn ptr(&self, id: TensorId) -> Option<DevicePtr> {
        self.map.get(&id).copied()
    }

    /// Returns a scalar supplied through a typed scalar setter.
    /// Opaque pointer bindings do not provide a readable host value.
    #[must_use]
    pub fn scalar_value(&self, id: TensorId) -> Option<ScalarValue> {
        self.scalar_storage.get(&id).map(|storage| storage.value())
    }

    fn refresh_scalar_bindings(&mut self) {
        for (&id, scalar) in &self.scalar_storage {
            self.map.insert(id, scalar.device_ptr());
        }
    }
}

fn binding_memory_info(
    region: DeviceRegion<'_>,
    data_type: DataType,
    access: BindingAccess,
) -> BindingMemoryInfo {
    BindingMemoryInfo {
        context: Arc::clone(region.context()),
        byte_capacity: region.byte_len(),
        data_type,
        access,
        tracked_access: Some(Access::new(region, AccessMode::Read)),
        owner: None,
    }
}

pub(crate) fn prepare_variant_pack(
    allocation_context: &Arc<CudaContext>,
    binding_replacements: &[BindingReplacement],
    bindings: &Bindings<'_>,
    workspace: Option<&mut DeviceMemory<MaybeUninit<u8>>>,
    runtime_overrides: &RuntimeOverrides,
    template: &BindingTemplate,
    tensors: &BTreeMap<TensorId, TensorRecord>,
    constraints: &BTreeMap<TensorId, RuntimeShapeConstraints>,
    static_layout: &WorkspaceLayout,
) -> Result<VariantPack> {
    let overrides =
        runtime_overrides.resolve_native_layouts(tensors, constraints, binding_replacements)?;
    let layout = static_layout.with_runtime_overrides(tensors, template.ids(), &overrides)?;
    let layout = match workspace.as_ref() {
        Some(workspace) => layout.resolve_address(workspace.region().as_ptr().addr())?,
        None => layout,
    };
    let capacity = workspace.as_ref().map_or(0, |workspace| workspace.len());
    if capacity < layout.byte_len() {
        return Err(Error::InsufficientWorkspaceSize {
            required: layout.byte_len(),
            actual: capacity,
        });
    }
    let workspace_bindings = workspace.as_ref().map(|workspace| {
        // SAFETY: the address belongs to a live device allocation that stays borrowed for the operation that uses it.
        let base = unsafe { DevicePtr::from_raw(workspace.region().as_ptr().cast()) };
        (base, &layout)
    });
    let pack_bindings = template.materialize(bindings, workspace_bindings)?;
    for ((id, pointer), source) in pack_bindings.iter().zip(template.sources()) {
        let record = tensors.get(id).ok_or(FrontendError::TensorNotFound(*id))?;
        bindings.validate_runtime_binding(
            allocation_context,
            source,
            record,
            overrides.iter().find(|layout| layout.id == (*id).into()),
        )?;
        let alignment = to_usize(record.tensor.byte_alignment, "tensor alignment")?;
        if alignment > 1 && !pointer.is_aligned_to(alignment) {
            return Err(FrontendError::BindingAlignmentMismatch {
                tensor_id: *id,
                required: alignment,
            }
            .into());
        }
    }
    let native_bindings = pack_bindings
        .into_iter()
        .map(|(id, ptr)| (id.into(), ptr))
        .collect::<Vec<_>>();
    // SAFETY: every binding points into a live tensor allocation that the caller retains through execution.
    let mut pack = unsafe {
        VariantPack::create_with_workspace_prefix(
            allocation_context,
            &native_bindings,
            workspace,
            layout.byte_len(),
            &overrides,
        )
    }?;
    pack.set_workspace_initializations(
        layout
            .bindings()
            .iter()
            .filter_map(|binding| {
                binding.initializer.as_ref().map(|initializer| {
                    WorkspaceInitialization::new(binding.byte_offset, binding.byte_len, initializer)
                })
            })
            .collect(),
    )?;
    Ok(pack)
}

/// Resolves the same root and byte offset for binding discovery and actual pointer materialization.
pub(crate) fn resolve_binding_source(
    replacements: &[BindingReplacement],
    id: TensorId,
) -> Result<(TensorId, i64)> {
    let mut current = id;
    let mut byte_offset = 0_i64;
    let mut seen = HashSet::new();
    while let Some(alias) = replacements.iter().find(|alias| alias.target_id == current) {
        if !seen.insert(current) {
            return Err(Error::DescriptorMismatch {
                name: "frontend storage view cycle".into(),
            });
        }
        if alias.byte_offset < 0 {
            return Err(Error::out_of_range("alias byte offset"));
        }
        byte_offset = byte_offset
            .checked_add(alias.byte_offset)
            .ok_or_else(|| Error::out_of_range("alias byte offset"))?;
        current = alias.source_id;
    }
    Ok((current, byte_offset))
}

pub(crate) fn adjusted_storage_view_ptr(source: DevicePtr, byte_offset: i64) -> Result<DevicePtr> {
    source
        .checked_byte_add(to_usize(byte_offset, "storage view byte offset")?)
        .ok_or_else(|| Error::out_of_range("storage view device address"))
}

impl CompiledGraph {
    pub(crate) fn variant_pack_with_overrides(
        &self,
        allocation_context: &Arc<CudaContext>,
        bindings: &Bindings<'_>,
        workspace: Option<&mut DeviceMemory<MaybeUninit<u8>>>,
        runtime_overrides: &RuntimeOverrides,
    ) -> Result<VariantPack> {
        runtime_overrides.validate_enabled(self.core.shape_policy.overrides_enabled)?;
        prepare_variant_pack(
            allocation_context,
            self.binding_replacements(),
            bindings,
            workspace,
            runtime_overrides,
            &self.core.binding_template,
            &self.core.artifacts.tensors,
            &self.core.shape_policy.bounds,
            &self.core.workspace_layout,
        )
    }

    /// Creates a binding builder pre-populated with graph-owned scalar values.
    ///
    /// # Errors
    ///
    /// Returns an error if a scalar default references an unknown or incompatible tensor, scalar alignment or alias resolution fails, or aligned host scalar storage cannot be created.
    pub fn bindings(&self) -> Result<Bindings<'_>> {
        let mut bindings = Bindings::with_graph_template(
            &self.core.artifacts.tensors,
            &self.core.artifacts.binding_replacements,
        );
        for record in self.core.artifacts.tensors.values() {
            if let Some(scalar_value) = record.tensor.runtime_scalar_value() {
                bindings.set_scalar(record.id, scalar_value)?;
            }
        }
        for &(id, scalar_value) in &self.core.artifacts.scalar_bindings {
            bindings.set_scalar(id, scalar_value)?;
        }
        Ok(bindings)
    }

    /// Creates a graph-aware runtime override builder for this compiled graph.
    #[must_use]
    pub fn runtime_overrides(&self) -> RuntimeOverrideBuilder<'_> {
        RuntimeOverrideBuilder::new(
            &self.core.artifacts.tensors,
            self.binding_replacements(),
            &self.core.shape_policy.bounds,
        )
    }
}

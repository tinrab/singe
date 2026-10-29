use std::collections::{BTreeMap, HashSet};

use crate::{
    backend::{
        execution::VariantPackOverride,
        tensor::{Shape, TensorId},
    },
    error::{Error, Result},
    frontend::{
        bindings::{RuntimeOverrideBuilder, RuntimeOverrides},
        error::Error as FrontendError,
        graph::{RuntimeShapeConstraints, TensorRecord},
        operation::ConvolutionOperation,
        plan::{AliasTensorKind, BindingReplacement},
    },
};

impl RuntimeOverrides {
    pub(crate) fn validate_enabled(&self, enabled: bool) -> Result<()> {
        if !self.is_empty() && !enabled {
            return Err(FrontendError::RuntimeOverridesRequireShapeEnabledGraph.into());
        }
        Ok(())
    }

    /// Creates an empty runtime override set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds or replaces the checked execution-time layout for a tensor.
    /// Plan-specific rank, bounds and addressing requirements are validated when used.
    pub fn set(&mut self, tensor: TensorId, shape: Shape) -> &mut Self {
        self.overrides.insert(tensor, shape);
        self
    }

    /// Creates an override set containing one tensor override.
    #[must_use]
    pub fn with(tensor: TensorId, shape: Shape) -> Self {
        let mut overrides = Self::new();
        overrides.set(tensor, shape);
        overrides
    }

    /// Returns a caller-specified layout. Inherited capture layouts are resolved by the plan.
    #[must_use]
    pub fn get(&self, tensor: TensorId) -> Option<&Shape> {
        self.overrides.get(&tensor)
    }

    /// Removes a caller-specified override so that use returns to its declared or inherited layout.
    pub fn remove(&mut self, tensor: TensorId) -> Option<Shape> {
        self.overrides.remove(&tensor)
    }

    /// Caller-specified layouts in tensor identity order, before capture propagation.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (TensorId, &Shape)> {
        self.overrides.iter().map(|(&id, shape)| (id, shape))
    }

    /// Returns whether no tensor overrides have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.overrides.is_empty()
    }

    pub(crate) fn resolve_native_layouts(
        &self,
        tensors: &BTreeMap<TensorId, TensorRecord>,
        constraints: &BTreeMap<TensorId, RuntimeShapeConstraints>,
        replacements: &[BindingReplacement],
    ) -> Result<Vec<VariantPackOverride>> {
        if self.is_empty() {
            return Ok(Vec::new());
        }
        let mut layouts = BTreeMap::new();
        for (&id, shape) in &self.overrides {
            validate_runtime_override(tensors, constraints, id, shape)?;
            validate_override_identity(tensors, replacements, id)?;
            layouts.insert(id, shape.clone());
        }
        loop {
            let mut changed = false;
            for replacement in replacements
                .iter()
                .filter(|replacement| replacement.kind == AliasTensorKind::CapturedTensor)
            {
                // Explicit native overrides take precedence over inherited captured layouts.
                if layouts.contains_key(&replacement.target_id) {
                    continue;
                }
                let Some(shape) = layouts.get(&replacement.source_id) else {
                    continue;
                };
                if replacement.byte_offset != 0 {
                    return Err(Error::DescriptorMismatch {
                        name: "captured tensor byte offset".into(),
                    });
                }
                let source = tensors
                    .get(&replacement.source_id)
                    .ok_or(FrontendError::TensorNotFound(replacement.source_id))?;
                let target = tensors
                    .get(&replacement.target_id)
                    .ok_or(FrontendError::TensorNotFound(replacement.target_id))?;
                let shape = captured_layout(source, target, shape)?;
                validate_runtime_override(tensors, constraints, replacement.target_id, &shape)?;
                layouts.insert(replacement.target_id, shape);
                changed = true;
            }
            if !changed {
                break;
            }
        }
        Ok(layouts
            .into_iter()
            .filter_map(|(id, shape)| {
                let record = &tensors[&id];
                record.access.is_some().then(|| VariantPackOverride {
                    id: record.id.into(),
                    shape,
                })
            })
            .collect())
    }
}

impl<'a> RuntimeOverrideBuilder<'a> {
    pub(crate) fn new(
        tensors: &'a BTreeMap<TensorId, TensorRecord>,
        replacements: &'a [BindingReplacement],
        constraints: &'a BTreeMap<TensorId, RuntimeShapeConstraints>,
    ) -> Self {
        Self {
            tensors,
            replacements,
            constraints,
            overrides: RuntimeOverrides::new(),
        }
    }

    /// Adds or replaces a checked layout after validating the plan's rank and bounds.
    /// A rejected replacement leaves the previous override unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is unknown or has no native execution or captured use, ranks differ, a scalar override is invalid, vectorized dimensions or strides violate alignment, convolution filter offsets exceed the native 32-bit limit, or dimensions fall outside recorded runtime bounds.
    pub fn set(&mut self, tensor: TensorId, shape: Shape) -> Result<&mut Self> {
        validate_runtime_override(self.tensors, self.constraints, tensor, &shape)?;
        validate_override_identity(self.tensors, self.replacements, tensor)?;
        self.overrides.set(tensor, shape);
        Ok(self)
    }

    /// Returns the collected overrides without resolving captured-layout inheritance or enabling graph overrides.
    #[must_use]
    pub fn build(self) -> RuntimeOverrides {
        self.overrides
    }
}

fn validate_runtime_override(
    tensors: &BTreeMap<TensorId, TensorRecord>,
    constraints: &BTreeMap<TensorId, RuntimeShapeConstraints>,
    tensor_id: TensorId,
    shape: &Shape,
) -> Result<()> {
    let record = tensors
        .get(&tensor_id)
        .ok_or(FrontendError::TensorNotFound(tensor_id))?;
    if shape.rank() != record.tensor.shape.rank() {
        return Err(Error::LengthMismatch {
            name: "runtime_override_rank".into(),
            expected: record.tensor.shape.rank(),
            actual: shape.rank(),
        });
    }
    if record.tensor.is_by_value()
        && (!shape.has_single_element()
            || record
                .tensor
                .vectorization()?
                .is_some_and(|(count, _)| count != 1))
    {
        return Err(FrontendError::TensorScalarShapeMismatch {
            tensor_id,
            operation: "runtime override",
        }
        .into());
    }
    if record.requires_vector_alignment {
        record
            .tensor
            .validate_vector_stride_alignment(tensor_id, shape)?;
    }
    if record.requires_int32_filter_offsets {
        ConvolutionOperation::validate_forward_filter_size(
            tensor_id,
            shape,
            record.tensor.vectorization()?.map_or(1, |(count, _)| count),
        )?;
    }
    if let Some(constraints) = constraints.get(&tensor_id)
        && !constraints.contains(shape.dimensions())
    {
        return Err(Error::out_of_range("runtime_override_shape_constraints"));
    }
    Ok(())
}

fn validate_override_identity(
    tensors: &BTreeMap<TensorId, TensorRecord>,
    replacements: &[BindingReplacement],
    tensor: TensorId,
) -> Result<()> {
    let mut pending = vec![tensor];
    let mut seen = HashSet::new();
    while let Some(id) = pending.pop() {
        if !seen.insert(id) {
            continue;
        }
        let record = tensors.get(&id).ok_or(FrontendError::TensorNotFound(id))?;
        if record.access.is_some() {
            return Ok(());
        }
        pending.extend(replacements.iter().filter_map(|replacement| {
            (replacement.kind == AliasTensorKind::CapturedTensor && replacement.source_id == id)
                .then_some(replacement.target_id)
        }));
    }
    Err(FrontendError::RuntimeOverrideNotNative(tensor).into())
}

fn captured_layout(source: &TensorRecord, target: &TensorRecord, shape: &Shape) -> Result<Shape> {
    let source_shape = &source.tensor.shape;
    let target_shape = &target.tensor.shape;
    if source_shape.rank() == target_shape.rank() {
        return Ok(shape.clone());
    }
    if source_shape.rank() == 1
        && target_shape.dimensions() == [source_shape.dimensions()[0], 1, 1, 1]
        && target_shape.strides() == [source_shape.strides()[0], 1, 1, 1]
    {
        return Shape::with_strides(
            vec![shape.dimensions()[0], 1, 1, 1],
            vec![shape.strides()[0], 1, 1, 1],
        );
    }
    if source.tensor.is_by_value() && target.tensor.is_by_value() {
        return Shape::contiguous(vec![1; target_shape.rank()]);
    }
    Err(Error::DescriptorMismatch {
        name: "captured runtime tensor rank".into(),
    })
}

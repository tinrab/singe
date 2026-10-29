use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet},
};

use crate::{
    backend::tensor::{Shape, TensorId, TensorSpec},
    error::{Error, Result},
    frontend::{
        error::Error as FrontendError,
        graph::{Graph, RuntimeShapeConstraints, TensorOutputConfig},
        operation::{
            AttentionPrimitiveOperation, ConcatConfig, DiagonalBandMaskConfig, Operation,
            SdpaOperation, SliceConfig, TensorOperation, TransposeConfig,
        },
    },
    utility::to_i64,
};

/// Projects the layouts that fused scalar inputs receive during native preparation.
fn concat_input_shapes<'a>(
    inputs: &[&'a TensorSpec],
    output: Option<&TensorSpec>,
) -> Result<Vec<Cow<'a, Shape>>> {
    let rank = inputs
        .iter()
        .copied()
        .filter(|tensor| !tensor.is_by_value())
        .map(|tensor| tensor.shape.rank())
        .chain(
            output
                .filter(|tensor| !tensor.is_by_value())
                .map(|tensor| tensor.shape.rank()),
        )
        .max()
        .unwrap_or(1);
    inputs
        .iter()
        .map(|&tensor| tensor.shape_with_scalar_rank(rank))
        .collect()
}

impl Graph {
    /// Adds a slice operation with an explicit output tensor.
    ///
    /// Slice extracts a half-open window `[start, limit)` along each dimension.
    /// The output may be virtual or have its own storage and layout.
    /// Missing step entries use one; extra entries must be positive and are otherwise ignored.
    /// Caller-declared output strides and names remain independent of the configuration.
    /// Fused scalar inputs use unit dimensions at the slice-window rank before validation.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensor identities, invalid scalar-rank promotion, invalid slice ranges or supplied steps, incompatible output dimensions or encoding, or unrepresentable slice geometry.
    pub fn slice_into(
        &mut self,
        input: TensorId,
        output: TensorId,
        config: SliceConfig,
    ) -> Result<()> {
        let operation = TensorOperation::Slice {
            input,
            output,
            config,
        };
        operation.validate_graph(self)?;
        self.operations.push(Operation::Tensor(operation));

        Ok(())
    }

    /// Creates a slice result with the requested placement and encoding.
    /// The selected encoding must match the input, as required by C++ and native cuDNN.
    /// Preserves the input vector lanes and vectorized axis.
    /// Slice ranges and strides use descriptor dimensions, which count vectors
    /// on the vectorized axis.
    /// Omitted stride entries use `1`; extra positive entries are ignored.
    /// A singleton output axis retains its source stride if multiplying by the step would overflow.
    /// Placement selects I/O or intermediate encoding defaults; the result must retain input encoding.
    /// A configured scope names the inferred result `name::Y`.
    /// Fused scalar geometry uses the slice-window rank without changing the input declaration.
    ///
    /// # Errors
    ///
    /// Returns an error if the input is unknown, scalar-rank promotion or slice geometry is invalid, output strides or vectorization cannot be represented, or the selected output encoding differs from the input.
    pub fn slice(
        &mut self,
        input: TensorId,
        config: SliceConfig,
        output: TensorOutputConfig,
    ) -> Result<TensorId> {
        let input_tensor = self.tensor_spec(input)?;
        let input_shape = input_tensor.shape_with_scalar_rank(config.slices().len())?;
        let mut output_spec = output.into_spec(
            self.config.data_type_policy,
            input_tensor.data_type,
            config.infer_output_shape(&input_shape)?,
        );
        if let Some((count, axis)) = input_tensor.vectorization()? {
            output_spec = output_spec.with_vectorization(count, axis)?;
        }
        let checkpoint = self.mutation_checkpoint();
        let output = self.inferred_tensor(output_spec, config.name(), "Y");
        if let Err(error) = self.slice_into(input, output, config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(output)
    }

    /// Adds a transpose operation with an explicit output tensor.
    ///
    /// Output strides may differ from the inferred view layout.
    /// Explicit output names are preserved independently of the configured scope.
    /// Fused scalar inputs use unit dimensions at the permutation rank before validation.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensor identities, invalid scalar-rank promotion, an empty or invalid axis permutation, or incompatible output dimensions or encoding.
    pub fn transpose_into(
        &mut self,
        input: TensorId,
        output: TensorId,
        config: TransposeConfig,
    ) -> Result<()> {
        let operation = TensorOperation::Transpose {
            input,
            output,
            config,
        };
        operation.validate_graph(self)?;
        self.operations.push(Operation::Tensor(operation));
        Ok(())
    }

    /// Adds a transpose operation and creates the output tensor.
    ///
    /// Preserves the input vector lanes and moves the vectorized axis with its
    /// corresponding dimension in the permutation.
    /// Fused scalar geometry uses the permutation rank without changing the input declaration.
    /// Placement selects I/O or intermediate encoding defaults; the result must retain input encoding.
    /// A configured scope names the inferred result `name::Y`.
    ///
    /// # Errors
    ///
    /// Returns an error if the input is unknown, scalar-rank promotion or the axis permutation is invalid, output vectorization cannot be represented, or output validation finds incompatible dimensions or encoding.
    pub fn transpose(
        &mut self,
        input: TensorId,
        config: TransposeConfig,
        output: TensorOutputConfig,
    ) -> Result<TensorId> {
        let input_tensor = self.tensor_spec(input)?;
        let input_shape = input_tensor.shape_with_scalar_rank(config.permutation().len())?;
        let mut output_spec = output.into_spec(
            self.config.data_type_policy,
            input_tensor.data_type,
            config.infer_output_shape(&input_shape)?,
        );
        if let Some((count, input_axis)) = input_tensor.vectorization()? {
            let output_axis = config
                .permutation()
                .iter()
                .position(|&axis| axis == input_axis)
                .ok_or_else(|| Error::out_of_range("transpose vector dimension"))?;
            output_spec = output_spec
                .with_vectorization(count, to_i64(output_axis, "transpose vector dimension")?)?;
        }
        let checkpoint = self.mutation_checkpoint();
        let output = self.inferred_tensor(output_spec, config.name(), "Y");
        if let Err(error) = self.transpose_into(input, output, config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(output)
    }

    /// Adds a diagonal band mask operation for attention score masking.
    ///
    /// Diagonal band masks support causal and sliding-window attention score
    /// modification in the frontend SDPA API.
    /// Rank-one sequence-length tensors and their dynamic bounds are promoted to
    /// native four-dimensional layouts after validation succeeds.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensor identities, a nonscalar replacement operand, incompatible replacement encoding or output dimensions, invalid bound-tensor shapes or sequence-length layouts, or failure to promote sequence metadata and its runtime bounds.
    pub fn diagonal_band_mask_into(
        &mut self,
        x: TensorId,
        b: TensorId,
        y: TensorId,
        config: DiagonalBandMaskConfig,
    ) -> Result<()> {
        let sequence = config.sequence_inputs();
        let operation = AttentionPrimitiveOperation::DiagonalBandMask { x, b, y, config };
        operation.validate_graph(self)?;
        self.promote_index_tensors(sequence.tensors())?;
        self.operations
            .push(Operation::AttentionPrimitive(operation));
        Ok(())
    }

    /// Creates a diagonal band mask result with independent placement and encoding.
    /// A configured scope names the inferred result `name::Y`.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensor identities, a nonscalar replacement operand, incompatible replacement encoding or output dimensions, invalid bound-tensor shapes or sequence-length layouts, or failure to promote sequence metadata and its runtime bounds.
    pub fn diagonal_band_mask(
        &mut self,
        x: TensorId,
        b: TensorId,
        config: DiagonalBandMaskConfig,
        output: TensorOutputConfig,
    ) -> Result<TensorId> {
        let x_tensor = self.tensor_spec(x)?;
        let output_spec = output.into_spec(
            self.config.data_type_policy,
            x_tensor.data_type,
            x_tensor.shape.clone(),
        );
        let checkpoint = self.mutation_checkpoint();
        let y = self.inferred_tensor(output_spec, config.name(), "Y");
        if let Err(error) = self.diagonal_band_mask_into(x, b, y, config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(y)
    }

    /// Adds a concatenate operation with an explicit output tensor.
    ///
    /// Concatenate merges one or more tensors along the selected axis and may
    /// optionally request in-place behavior through an input index.
    /// Fused scalar inputs use unit dimensions at the non-scalar operands' rank during validation.
    /// Their original declarations remain unchanged until graph preparation.
    ///
    /// # Errors
    ///
    /// Returns an error for empty or unknown inputs, invalid scalar-rank promotion, an invalid axis or in-place input index, incompatible ranks or non-concatenated dimensions, overflowing output extents, or incompatible output dimensions.
    pub fn concat_into(
        &mut self,
        inputs: &[TensorId],
        output: TensorId,
        config: ConcatConfig,
    ) -> Result<()> {
        let operation = TensorOperation::Concat {
            inputs: inputs.to_vec(),
            output,
            config,
        };
        operation.validate_graph(self)?;
        self.operations.push(Operation::Tensor(operation));
        Ok(())
    }

    /// Adds a concatenate operation and creates the output tensor.
    ///
    /// Preserves the first input's stride order and vectorization.
    /// The concatenation axis and dimensions use descriptor units.
    /// In-place mode requires the selected input to occupy its final region in
    /// the output allocation, with strides matching that region.
    /// Placement and encoding are selected independently of the in-place request.
    /// Fused scalar dimensions and stride order reflect their native rank promotion.
    ///
    /// # Errors
    ///
    /// Returns an error for empty or unknown inputs, invalid scalar-rank promotion, an invalid axis or in-place input index, incompatible ranks or non-concatenated dimensions, overflowing output extents, or incompatible output dimensions. Inferred stride layout and vectorization must also be representable.
    pub fn concat(
        &mut self,
        inputs: &[TensorId],
        config: ConcatConfig,
        output: TensorOutputConfig,
    ) -> Result<TensorId> {
        let input_tensors = inputs
            .iter()
            .map(|tensor| self.tensor_spec(*tensor))
            .collect::<Result<Vec<_>>>()?;
        let input_shapes = concat_input_shapes(&input_tensors, None)?;
        let output_shape = config.infer_output_shape(input_shapes.iter().map(AsRef::as_ref))?;
        let mut output_spec = output.into_spec(
            self.config.data_type_policy,
            input_tensors[0].data_type,
            output_shape,
        );
        if let Some((count, axis)) = input_tensors[0].vectorization()? {
            output_spec = output_spec.with_vectorization(count, axis)?;
        }
        let checkpoint = self.mutation_checkpoint();
        let output = self.inferred_tensor(output_spec, config.name(), "Y");
        if let Err(error) = self.concat_into(inputs, output, config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(output)
    }

    pub(crate) fn validate_concat(
        &self,
        inputs: &[TensorId],
        output: TensorId,
        config: &ConcatConfig,
    ) -> Result<()> {
        let tensors = inputs
            .iter()
            .map(|tensor| self.tensor_spec(*tensor))
            .collect::<Result<Vec<_>>>()?;
        let shapes = concat_input_shapes(&tensors, Some(self.tensor_spec(output)?))?;
        let expected_output = config.infer_output_dimensions(shapes.iter().map(AsRef::as_ref))?;
        self.validate_tensor_dimensions(output, &expected_output, "concat output shape")?;

        config.validate_in_place_index(inputs.len())?;

        Ok(())
    }
}

impl Graph {
    /// Refreshes native index layouts after tensor replacement, import or recipe restoration.
    pub(crate) fn promote_recorded_index_tensors(&mut self) -> Result<()> {
        let mut indices = BTreeSet::new();
        for operation in &self.operations {
            let sequence = match operation {
                Operation::AttentionPrimitive(AttentionPrimitiveOperation::DiagonalBandMask {
                    config,
                    ..
                }) => config.sequence_inputs(),
                Operation::Sdpa(SdpaOperation::Forward { config, .. }) => config.sequence_inputs(),
                Operation::Sdpa(SdpaOperation::Backward { config, .. }) => config.sequence_inputs(),
                Operation::PagedCacheLoad(operation) => {
                    indices.insert(operation.tensors.sequence);
                    continue;
                }
                _ => continue,
            };
            indices.extend(sequence.tensors());
        }
        self.promote_index_tensors(indices)
    }

    /// Promotes index vectors while preserving their stride, placement and dynamic bounds.
    /// A rejected batch leaves every original declaration unchanged.
    pub(crate) fn promote_index_tensors(
        &mut self,
        tensor_ids: impl IntoIterator<Item = TensorId>,
    ) -> Result<()> {
        let mut promotions = Vec::new();
        for id in tensor_ids {
            let tensor = self.tensor_spec(id)?;
            if tensor.shape.rank() == 1 {
                // Validate the original declaration before rank promotion can replace malformed layouts.
                tensor.validate()?;
                let mut promoted = tensor.clone();
                promoted.shape = Shape::with_strides(
                    [tensor.shape.dimensions()[0], 1, 1, 1],
                    [tensor.shape.strides()[0], 1, 1, 1],
                )?;
                let constraints = self
                    .runtime_shape_constraints
                    .get(&id)
                    .map(|bounds| {
                        bounds.validate_for(tensor)?;
                        RuntimeShapeConstraints::create(
                            [bounds.min_dimensions[0], 1, 1, 1],
                            [bounds.max_dimensions[0], 1, 1, 1],
                        )
                    })
                    .transpose()?;
                promotions.push((id, promoted, constraints));
            }
        }
        // Commit only after every reference, promoted layout and bound is valid.
        // Promotion changes shape alone, preserving placement and ragged relationships.
        for (id, tensor, constraints) in promotions {
            self.tensors.insert(id, tensor);
            if let Some(constraints) = constraints {
                self.runtime_shape_constraints.insert(id, constraints);
            }
        }
        Ok(())
    }

    /// Matches fused scalar ranks and unit strides to each native operation while preserving values and runtime bounds.
    pub(crate) fn expand_fused_scalar_shapes(&mut self) -> Result<()> {
        let mut scalar_target_ranks = BTreeMap::<TensorId, usize>::new();

        for operation in &self.operations {
            let uses = match operation {
                Operation::Sdpa(operation) => operation.native_tensor_uses(),
                operation => operation.tensor_uses(self.config.data_type_policy),
            };

            let mut target_rank = 1;
            for &tensor in uses.reads.union(&uses.writes) {
                let tensor = self.tensor_spec(tensor)?;
                if !tensor.is_by_value() {
                    target_rank = target_rank.max(tensor.shape.rank());
                }
            }

            for tensor_id in uses.reads {
                let tensor = self.tensor_spec(tensor_id)?;
                if !tensor.is_by_value() {
                    continue;
                }
                tensor.validate_scalar()?;

                match scalar_target_ranks.insert(tensor_id, target_rank) {
                    Some(existing_rank) if existing_rank != target_rank => {
                        return Err(FrontendError::FusedScalarRankConflict {
                            tensor_id,
                            first_rank: existing_rank,
                            second_rank: target_rank,
                        }
                        .into());
                    }
                    _ => {}
                }
            }
        }

        let mut promotions = Vec::with_capacity(scalar_target_ranks.len());
        for (tensor_index, target_rank) in scalar_target_ranks {
            let tensor = self.tensor_spec(tensor_index)?.clone();
            if tensor.shape.rank() == target_rank
                && tensor.shape.strides().iter().all(|&stride| stride == 1)
            {
                continue;
            }

            let constraints = self
                .runtime_shape_constraints
                .get(&tensor_index)
                .map(|bounds| {
                    bounds.validate_for(&tensor)?;
                    RuntimeShapeConstraints::create(vec![1; target_rank], vec![1; target_rank])
                })
                .transpose()?;
            let tensor = tensor.with_shape(Shape::contiguous(vec![1; target_rank])?);
            tensor.validate_scalar()?;
            promotions.push((tensor_index, tensor, constraints));
        }

        for (tensor_index, tensor, constraints) in promotions {
            self.tensors.insert(tensor_index, tensor);
            if let Some(constraints) = constraints {
                self.runtime_shape_constraints
                    .insert(tensor_index, constraints);
            }
        }

        Ok(())
    }
}

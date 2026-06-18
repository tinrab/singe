use std::ops::Range;

use crate::{
    data_type::DataType,
    error::{Error, Result},
    frontend::{
        graph::Graph,
        infer::{
            infer_concat_output, infer_strided_slice_output, infer_transpose_output,
            slice_byte_offset,
        },
        operation::{ConcatInPlaceMode, DiagonalBandMaskConfig, Operation, PointwiseOperation},
        support,
    },
    math::NanPropagation,
    pointwise::PointwiseMode,
    tensor::{TensorId, TensorSpec},
    utility::{check_range, to_usize},
    version,
};

impl Graph {
    /// Adds a slice operation with an explicit output tensor.
    ///
    /// Slice extracts a half-open window `[start, limit)` along each dimension.
    /// This Rust helper currently records the compatible pointer-offset form.
    ///
    pub fn slice(
        &mut self,
        input: TensorId,
        output: TensorId,
        slices: &[Range<i64>],
    ) -> Result<()> {
        self.strided_slice(input, output, slices, Vec::new())
    }

    /// Adds a strided slice operation with an explicit output tensor.
    ///
    /// Missing stride entries default to `1`, matching C++ cudnn-frontend
    /// `Slice_attributes`.
    pub fn strided_slice(
        &mut self,
        input: TensorId,
        output: TensorId,
        slices: &[Range<i64>],
        slice_strides: impl Into<Vec<i64>>,
    ) -> Result<()> {
        let slice_strides = slice_strides.into();
        let input_tensor = self.tensor_config(input)?.clone();
        let output_tensor = self.tensor_config(output)?.clone();
        if input_tensor.data_type != output_tensor.data_type {
            return Err(Error::FrontendTensorDataTypeMismatch {
                tensor_id: output,
                operation: "slice output".into(),
                expected: input_tensor.data_type,
                actual: output_tensor.data_type,
            });
        }
        let slice_bounds = Self::slice_bounds(slices);
        let effective_strides = Self::effective_slice_strides(slice_bounds.len(), &slice_strides);
        let expected_output =
            infer_strided_slice_output(&input_tensor.shape, &slice_bounds, &effective_strides)?;
        if output_tensor.shape.dimensions() != expected_output.dimensions() {
            return Err(Error::FrontendTensorDimensionsMismatch {
                tensor_id: output,
                operation: "slice output".into(),
                expected: expected_output.dimensions().to_vec(),
                actual: output_tensor.shape.dimensions().to_vec(),
            });
        }

        let byte_offset = slice_byte_offset(
            &input_tensor.shape,
            &slice_bounds,
            input_tensor.data_type.size(),
        )?;
        self.operations.push(Operation::Slice {
            input,
            output,
            starts: slice_bounds.iter().map(|(start, _)| *start).collect(),
            limits: slice_bounds.iter().map(|(_, limit)| *limit).collect(),
            strides: effective_strides,
            byte_offset,
        });

        Ok(())
    }

    /// Adds a slice operation and creates the output tensor.
    pub fn slice_infer(&mut self, input: TensorId, slices: &[Range<i64>]) -> Result<TensorId> {
        self.strided_slice_infer(input, slices, Vec::new())
    }

    /// Adds a strided slice operation and creates the output tensor.
    pub fn strided_slice_infer(
        &mut self,
        input: TensorId,
        slices: &[Range<i64>],
        slice_strides: impl Into<Vec<i64>>,
    ) -> Result<TensorId> {
        let checkpoint = self.mutation_checkpoint();
        let slice_strides = slice_strides.into();
        let input_tensor = self.tensor_config(input)?.clone();
        let slice_bounds = Self::slice_bounds(slices);
        let effective_strides = Self::effective_slice_strides(slice_bounds.len(), &slice_strides);
        let output = self.tensor(TensorSpec::new(
            input_tensor.data_type,
            infer_strided_slice_output(&input_tensor.shape, &slice_bounds, &effective_strides)?,
        ));
        if let Err(error) = self.strided_slice(input, output, slices, effective_strides) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(output)
    }

    fn slice_bounds(slices: &[Range<i64>]) -> Vec<(i64, i64)> {
        slices
            .iter()
            .map(|range| (range.start, range.end))
            .collect()
    }

    fn effective_slice_strides(rank: usize, slice_strides: &[i64]) -> Vec<i64> {
        (0..rank)
            .map(|index| slice_strides.get(index).copied().unwrap_or(1))
            .collect()
    }

    /// Adds a transpose operation with an explicit output tensor.
    ///
    /// The current checked-in cuDNN bindings predate the native backend
    /// transpose descriptor, so this records a frontend transpose node and uses
    /// a zero-offset alias fallback during lowering.
    pub fn transpose(
        &mut self,
        input: TensorId,
        output: TensorId,
        permutation: impl Into<Vec<i64>>,
    ) -> Result<()> {
        let permutation = permutation.into();
        let input_tensor = self.tensor_config(input)?.clone();
        let output_tensor = self.tensor_config(output)?.clone();
        if input_tensor.data_type != output_tensor.data_type {
            return Err(Error::FrontendTensorDataTypeMismatch {
                tensor_id: output,
                operation: "transpose output".into(),
                expected: input_tensor.data_type,
                actual: output_tensor.data_type,
            });
        }
        let expected_output = infer_transpose_output(&input_tensor.shape, &permutation)?;
        if output_tensor.shape.dimensions() != expected_output.dimensions() {
            return Err(Error::FrontendTensorDimensionsMismatch {
                tensor_id: output,
                operation: "transpose output".into(),
                expected: expected_output.dimensions().to_vec(),
                actual: output_tensor.shape.dimensions().to_vec(),
            });
        }
        if output_tensor.shape.strides() != expected_output.strides() {
            return Err(Error::FrontendTensorStridesMismatch {
                tensor_id: output,
                operation: "transpose output".into(),
                expected: expected_output.strides().to_vec(),
                actual: output_tensor.shape.strides().to_vec(),
            });
        }

        self.operations.push(Operation::Transpose {
            input,
            output,
            permutation,
        });
        Ok(())
    }

    /// Adds a transpose operation and creates the output tensor.
    pub fn transpose_infer(
        &mut self,
        input: TensorId,
        permutation: impl Into<Vec<i64>>,
    ) -> Result<TensorId> {
        let checkpoint = self.mutation_checkpoint();
        let permutation = permutation.into();
        let input_tensor = self.tensor_config(input)?.clone();
        let output = self.tensor(TensorSpec::new(
            input_tensor.data_type,
            infer_transpose_output(&input_tensor.shape, &permutation)?,
        ));
        if let Err(error) = self.transpose(input, output, permutation) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(output)
    }

    /// Adds a diagonal band mask operation for attention score masking.
    ///
    /// Diagonal band masks support causal and sliding-window attention score
    /// modification in the frontend SDPA API.
    pub fn diagonal_band_mask(
        &mut self,
        x: TensorId,
        b: TensorId,
        y: TensorId,
        config: DiagonalBandMaskConfig,
    ) -> Result<()> {
        let x_tensor = self.tensor_config(x)?.clone();

        self.validate_tensor_data_type(b, x_tensor.data_type, "diagonal band mask b")?;
        self.validate_tensor_dimensions(
            y,
            x_tensor.shape.dimensions(),
            "diagonal band mask output shape",
        )?;
        self.validate_tensor_element_count(b, 1, "diagonal band mask b shape")?;
        if config.left_bound().is_some() && config.shift_right_bound().is_some() {
            return Err(Error::DescriptorMismatch {
                name: "diagonal band mask bounds".into(),
            });
        }
        if let Some(sequence_length_query) = config.sequence_length_query() {
            self.validate_tensor_data_type(
                sequence_length_query,
                DataType::I32,
                "diagonal band mask seq len q",
            )?;
            self.validate_tensor_dimensions(
                sequence_length_query,
                &[x_tensor.shape.dimensions()[0], 1, 1, 1],
                "diagonal band mask seq len q shape",
            )?;
        }
        if let Some(sequence_length_key_value) = config.sequence_length_key_value() {
            self.validate_tensor_data_type(
                sequence_length_key_value,
                DataType::I32,
                "diagonal band mask seq len kv",
            )?;
            self.validate_tensor_dimensions(
                sequence_length_key_value,
                &[x_tensor.shape.dimensions()[0], 1, 1, 1],
                "diagonal band mask seq len kv shape",
            )?;
        }
        if let Some(left_bound) = config.left_bound() {
            self.validate_tensor_element_count(
                left_bound,
                1,
                "diagonal band mask left bound shape",
            )?;
        }
        if let Some(shift_right_bound) = config.shift_right_bound() {
            self.validate_tensor_element_count(
                shift_right_bound,
                1,
                "diagonal band mask shift right bound shape",
            )?;
        }

        self.operations.push(Operation::DiagonalBandMask {
            x,
            b,
            y,
            comparison_mode: config.comparison_mode(),
            sequence_length_query: config.sequence_length_query(),
            sequence_length_key_value: config.sequence_length_key_value(),
            left_bound: config.left_bound(),
            shift_right_bound: config.shift_right_bound(),
        });
        Ok(())
    }

    /// Adds a diagonal band mask operation and creates the output tensor.
    pub fn diagonal_band_mask_infer(
        &mut self,
        x: TensorId,
        b: TensorId,
        config: DiagonalBandMaskConfig,
    ) -> Result<TensorId> {
        let checkpoint = self.mutation_checkpoint();
        let x_tensor = self.tensor_config(x)?.clone();
        let output_data_type = self.effective_io_data_type(x_tensor.data_type);
        let y = self.tensor(TensorSpec::new(output_data_type, x_tensor.shape.clone()));
        if let Err(error) = self.diagonal_band_mask(x, b, y, config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(y)
    }

    /// Adds a concatenate operation with an explicit output tensor.
    ///
    /// Concatenate merges two or more tensors along the selected axis and may
    /// optionally request in-place behavior through an input index.
    pub fn concat(
        &mut self,
        inputs: &[TensorId],
        output: TensorId,
        axis: i64,
        inplace_index: Option<i64>,
    ) -> Result<()> {
        if inputs.is_empty() {
            return Err(Error::EmptyList {
                name: "inputs".into(),
            });
        }
        self.validate_concat_support_surface_for_version(version()?.raw())?;
        let in_place = ConcatInPlaceMode::from_index(inplace_index);
        self.validate_concat(inputs, output, axis, in_place)?;
        self.operations.push(Operation::Concat {
            inputs: inputs.to_vec(),
            output,
            axis,
            in_place,
        });
        Ok(())
    }

    /// Adds a concatenate operation and creates the output tensor.
    pub fn concat_infer(&mut self, inputs: &[TensorId], axis: i64) -> Result<TensorId> {
        let checkpoint = self.mutation_checkpoint();
        if inputs.is_empty() {
            return Err(Error::EmptyList {
                name: "inputs".into(),
            });
        }
        let input_tensors = inputs
            .iter()
            .map(|tensor| self.tensor_config(*tensor))
            .collect::<Result<Vec<_>>>()?;
        let shapes = input_tensors
            .iter()
            .map(|tensor| &tensor.shape)
            .collect::<Vec<_>>();
        let output_data_type = self.effective_io_data_type(input_tensors[0].data_type);
        let output_dimensions = infer_concat_output(&shapes, axis)?.dimensions().to_vec();
        let output_shape =
            Self::shape_preserving_input_format(&input_tensors[0].shape, output_dimensions)?;
        let output = self.tensor(TensorSpec::new(output_data_type, output_shape));
        if let Err(error) = self.concat(inputs, output, axis, None) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(output)
    }

    pub(crate) fn expand_diagonal_band_mask_for_legacy_runtime(
        &mut self,
        x: TensorId,
        b: TensorId,
        y: TensorId,
        comparison_mode: PointwiseMode,
        sequence_length_query: Option<TensorId>,
        sequence_length_key_value: Option<TensorId>,
        left_bound: Option<TensorId>,
        shift_right_bound: Option<TensorId>,
    ) -> Result<()> {
        let x_tensor = self.tensor_config(x)?.clone();
        let index_shape = x_tensor.shape.clone();

        let row = self.tensor(TensorSpec::new(DataType::I32, index_shape.clone()).virtual_tensor());
        let col = self.tensor(TensorSpec::new(DataType::I32, index_shape.clone()).virtual_tensor());
        self.pointwise(PointwiseOperation::Unary {
            mode: PointwiseMode::GenIndex,
            input: x,
            output: row,
            compute_type: DataType::F32,
            nan_propagation: NanPropagation::NotPropagate,
            alpha1: 1.0,
            axis: Some(2),
        });
        self.pointwise(PointwiseOperation::Unary {
            mode: PointwiseMode::GenIndex,
            input: x,
            output: col,
            compute_type: DataType::F32,
            nan_propagation: NanPropagation::NotPropagate,
            alpha1: 1.0,
            axis: Some(3),
        });

        let compare = if let Some(left_bound) = left_bound {
            let mut col_index =
                self.tensor(TensorSpec::new(DataType::I32, index_shape.clone()).virtual_tensor());
            self.pointwise(PointwiseOperation::Binary {
                mode: PointwiseMode::Add,
                lhs: col,
                rhs: left_bound,
                output: col_index,
                compute_type: DataType::I32,
                nan_propagation: NanPropagation::NotPropagate,
                alpha1: 1.0,
                alpha2: 1.0,
            });
            if let Some(sequence_length_key_value) = sequence_length_key_value {
                let next = self
                    .tensor(TensorSpec::new(DataType::I32, index_shape.clone()).virtual_tensor());
                self.pointwise(PointwiseOperation::Binary {
                    mode: PointwiseMode::Sub,
                    lhs: col_index,
                    rhs: sequence_length_key_value,
                    output: next,
                    compute_type: DataType::I32,
                    nan_propagation: NanPropagation::NotPropagate,
                    alpha1: 1.0,
                    alpha2: 1.0,
                });
                col_index = next;
            }
            if let Some(sequence_length_query) = sequence_length_query {
                let next = self
                    .tensor(TensorSpec::new(DataType::I32, index_shape.clone()).virtual_tensor());
                self.pointwise(PointwiseOperation::Binary {
                    mode: PointwiseMode::Add,
                    lhs: col_index,
                    rhs: sequence_length_query,
                    output: next,
                    compute_type: DataType::I32,
                    nan_propagation: NanPropagation::NotPropagate,
                    alpha1: 1.0,
                    alpha2: 1.0,
                });
                col_index = next;
            }

            let compare = self
                .tensor(TensorSpec::new(DataType::Boolean, index_shape.clone()).virtual_tensor());
            self.pointwise(PointwiseOperation::Binary {
                mode: comparison_mode,
                lhs: col_index,
                rhs: row,
                output: compare,
                compute_type: DataType::Boolean,
                nan_propagation: NanPropagation::NotPropagate,
                alpha1: 1.0,
                alpha2: 1.0,
            });
            compare
        } else {
            let mut row_index = row;
            if let Some(sequence_length_key_value) = sequence_length_key_value {
                let next = self
                    .tensor(TensorSpec::new(DataType::I32, index_shape.clone()).virtual_tensor());
                self.pointwise(PointwiseOperation::Binary {
                    mode: PointwiseMode::Add,
                    lhs: row_index,
                    rhs: sequence_length_key_value,
                    output: next,
                    compute_type: DataType::I32,
                    nan_propagation: NanPropagation::NotPropagate,
                    alpha1: 1.0,
                    alpha2: 1.0,
                });
                row_index = next;
            }
            if let Some(sequence_length_query) = sequence_length_query {
                let next = self
                    .tensor(TensorSpec::new(DataType::I32, index_shape.clone()).virtual_tensor());
                self.pointwise(PointwiseOperation::Binary {
                    mode: PointwiseMode::Sub,
                    lhs: row_index,
                    rhs: sequence_length_query,
                    output: next,
                    compute_type: DataType::I32,
                    nan_propagation: NanPropagation::NotPropagate,
                    alpha1: 1.0,
                    alpha2: 1.0,
                });
                row_index = next;
            }
            if let Some(shift_right_bound) = shift_right_bound {
                let next = self
                    .tensor(TensorSpec::new(DataType::I32, index_shape.clone()).virtual_tensor());
                self.pointwise(PointwiseOperation::Binary {
                    mode: PointwiseMode::Add,
                    lhs: row_index,
                    rhs: shift_right_bound,
                    output: next,
                    compute_type: DataType::I32,
                    nan_propagation: NanPropagation::NotPropagate,
                    alpha1: 1.0,
                    alpha2: 1.0,
                });
                row_index = next;
            }

            let compare = self
                .tensor(TensorSpec::new(DataType::Boolean, index_shape.clone()).virtual_tensor());
            self.pointwise(PointwiseOperation::Binary {
                mode: comparison_mode,
                lhs: row_index,
                rhs: col,
                output: compare,
                compute_type: DataType::Boolean,
                nan_propagation: NanPropagation::NotPropagate,
                alpha1: 1.0,
                alpha2: 1.0,
            });
            compare
        };

        self.pointwise(PointwiseOperation::Ternary {
            mode: PointwiseMode::BinarySelect,
            x,
            b,
            t: compare,
            output: y,
            compute_type: x_tensor.data_type,
            nan_propagation: NanPropagation::NotPropagate,
            alpha1: 1.0,
            alpha2: 1.0,
        });
        Ok(())
    }

    pub(crate) fn validate_concat_support_surface_for_version(
        &self,
        cudnn_version: u64,
    ) -> Result<()> {
        support::CONCAT.require_descriptor_match(cudnn_version)
    }

    fn validate_concat(
        &self,
        inputs: &[TensorId],
        output: TensorId,
        axis: i64,
        in_place: ConcatInPlaceMode,
    ) -> Result<()> {
        let input_tensors = inputs
            .iter()
            .map(|tensor| self.tensor_config(*tensor))
            .collect::<Result<Vec<_>>>()?;
        Self::validate_concat_input_dimensions(inputs, &input_tensors, axis)?;
        let expected_output = infer_concat_output(
            &input_tensors
                .iter()
                .map(|tensor| &tensor.shape)
                .collect::<Vec<_>>(),
            axis,
        )?;

        self.validate_tensor_dimensions(
            output,
            expected_output.dimensions(),
            "concat output shape",
        )?;

        if let Some(inplace_index) = in_place.index() {
            let inplace_index = to_usize(inplace_index, "inplace_index")?;
            check_range!("inplace_index", input_tensors.get(inplace_index).is_some())?;
        }

        Ok(())
    }

    fn validate_concat_input_dimensions(
        inputs: &[TensorId],
        input_tensors: &[&TensorSpec],
        axis: i64,
    ) -> Result<()> {
        let Some(first) = input_tensors.first() else {
            return Ok(());
        };
        let rank = first.shape.dimensions().len();
        let axis = to_usize(axis, "axis")?;
        let first_dimensions = first.shape.dimensions();
        for (&input, input_tensor) in inputs.iter().zip(input_tensors) {
            let actual = input_tensor.shape.dimensions();
            if actual.len() != rank {
                return Err(Error::FrontendTensorDimensionsMismatch {
                    tensor_id: input,
                    operation: "concat input rank".into(),
                    expected: first_dimensions.to_vec(),
                    actual: actual.to_vec(),
                });
            }
            let mut expected = actual.to_vec();
            for (index, expected_dimension) in expected.iter_mut().enumerate() {
                if index != axis {
                    *expected_dimension = first_dimensions[index];
                }
            }
            if actual
                .iter()
                .zip(&expected)
                .enumerate()
                .any(|(index, (actual, expected))| index != axis && actual != expected)
            {
                return Err(Error::FrontendTensorDimensionsMismatch {
                    tensor_id: input,
                    operation: "concat input shape".into(),
                    expected,
                    actual: actual.to_vec(),
                });
            }
        }
        Ok(())
    }
}

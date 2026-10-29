use serde::{Deserialize, Serialize};

use crate::{
    backend::{
        execution::block_scale::{
            BlockScaleDequantizeInputs,
            BlockScaleDequantizeOperation as BackendBlockScaleDequantizeOperation,
            BlockScaleDequantizeOperationConfig,
            BlockScaleQuantizeOperation as BackendBlockScaleQuantizeOperation,
            BlockScaleQuantizeOperationConfig, BlockScaleQuantizeOutputs,
        },
        tensor::{Shape, TensorId},
    },
    data_type::DataType,
    error::{Error, Result},
    frontend::{
        error::Error as FrontendError,
        graph::Graph,
        lower::{LoweredOperation, LoweringContext, tensor_at},
    },
    utility::{check_range, to_i32, to_i64, to_usize},
};

const BLOCK_SCALE_RANK: &str = "block scale rank";

/// Frontend block-scale operation variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum BlockScaleOperation {
    /// Produces quantized values and block scales from an input activation.
    Quantize {
        /// Input activation identity.
        input: TensorId,
        /// Quantized value destination identity.
        output: TensorId,
        /// Block-scale destination identity.
        scale: TensorId,
        /// Compute precision and block geometry or scale-sign settings.
        config: BlockScaleQuantizeConfig,
    },
    /// Combines quantized values with block scales to produce a virtual dequantized tensor.
    Dequantize {
        /// Quantized value tensor identity.
        input: TensorId,
        /// Input block-scale tensor identity.
        scale: TensorId,
        /// Virtual dequantized output tensor identity.
        output: TensorId,
        /// Compute precision and block geometry or scale-sign settings.
        config: BlockScaleDequantizeConfig,
    },
}

impl BlockScaleOperation {
    pub(crate) fn validate_graph(&self, graph: &Graph) -> Result<()> {
        match self {
            Self::Quantize {
                input,
                output,
                scale,
                config,
            } => {
                config.native_block_size()?;
                graph.validate_tensor_dimensions(
                    *output,
                    graph.shape(*input)?.dimensions(),
                    "block scale quantize output shape",
                )?;
                graph.tensor_spec(*scale)?;
            }
            Self::Dequantize {
                input,
                scale,
                output,
                config,
            } => {
                config.validate_block_sizes()?;
                if !graph.tensor_spec(*output)?.is_virtual() {
                    return Err(FrontendError::BindingTypeMismatch {
                        tensor_id: *output,
                        expected: "virtual tensor",
                        actual: "bindable tensor",
                    }
                    .into());
                }
                graph.validate_tensor_dimensions(
                    *output,
                    graph.shape(*input)?.dimensions(),
                    "block scale dequantize output shape",
                )?;
                graph.tensor_spec(*scale)?;
            }
        }
        Ok(())
    }

    pub(crate) fn lower(&self, context: &LoweringContext<'_>) -> Result<LoweredOperation> {
        let tensors = context.backend_tensors();
        match self {
            Self::Quantize {
                input,
                output,
                scale,
                config,
            } => Ok(LoweredOperation::BlockScaleQuantize(
                BackendBlockScaleQuantizeOperation::create(
                    tensor_at(tensors, *input)?,
                    BlockScaleQuantizeOutputs::new(*output, *scale)
                        .try_map(|id| tensor_at(tensors, id))?,
                    BlockScaleQuantizeOperationConfig::new(
                        context.compute_data_type(config.compute_type()),
                        config.native_block_size()?,
                    ),
                )?,
            )),
            Self::Dequantize {
                input,
                scale,
                output,
                config,
            } => Ok(LoweredOperation::BlockScaleDequantize(
                BackendBlockScaleDequantizeOperation::create(
                    BlockScaleDequantizeInputs::new(*input, *scale)
                        .try_map(|id| tensor_at(tensors, id))?,
                    tensor_at(tensors, *output)?,
                    BlockScaleDequantizeOperationConfig::new(
                        context.compute_data_type(config.compute_type()),
                        config.block_sizes().to_vec(),
                    )
                    .with_negative_scale(config.is_negative_scale()),
                )?,
            )),
        }
    }
}

/// Attributes for block scale quantization.
///
/// Block scale quantize produces a quantized output tensor and a scale tensor
/// from a higher precision tensor. MXFP8 and NVFP4 recipes quantize fixed-size
/// blocks along an axis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockScaleQuantizeConfig {
    name: Option<String>,
    compute_type: Option<DataType>,
    block_size: i64,
    axis: Option<i64>,
    transpose: bool,
}

impl BlockScaleQuantizeConfig {
    /// Inherits graph compute precision, defaulting to `F32`.
    /// Lowering always supplies the required native math-precision attribute.
    #[must_use]
    pub fn new(block_size: i64) -> Self {
        Self {
            name: None,
            compute_type: None,
            block_size,
            axis: None,
            transpose: false,
        }
    }

    /// Names inferred results `name::Y` and `name::scale`, matching C++ output scopes.
    /// Explicit tensor names are retained, and an empty scope disables naming.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Returns the nonempty result-name scope, or `None` when unnamed.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref().filter(|name| !name.is_empty())
    }

    /// Overrides graph math precision independently of block geometry and storage encodings.
    #[must_use]
    pub fn with_compute_type(mut self, compute_type: impl Into<Option<DataType>>) -> Self {
        self.compute_type = compute_type.into();
        self
    }

    /// Selects an explicit block axis or restores omitted-axis inference with `None`.
    #[must_use]
    pub fn with_axis(mut self, axis: impl Into<Option<i64>>) -> Self {
        self.axis = axis.into();
        self
    }

    /// Enables or disables transposed stride inference without changing the block axis.
    #[must_use]
    pub fn with_transpose(mut self, transpose: bool) -> Self {
        self.transpose = transpose;
        self
    }

    /// Returns the math-precision override, or `None` to inherit graph policy.
    #[must_use]
    pub fn compute_type(&self) -> Option<DataType> {
        self.compute_type
    }

    /// Returns the requested number of values sharing each quantization scale.
    #[must_use]
    pub fn block_size(&self) -> i64 {
        self.block_size
    }

    /// Infers default value and scale layouts without constructing a graph or native descriptor.
    /// Value dimensions match the input, and scale dimensions cover partial blocks by rounding up.
    /// Non-transposed values retain the input layout; scales use its packed stride order.
    /// Transpose packs both results and rotates an explicit axis into the fastest position.
    /// An omitted axis reduces the last logical dimension without rotating the stride order.
    /// Encoding-specific padding and reordering remain separate tensor declaration choices.
    ///
    /// # Errors
    /// Returns an error for an invalid native block size, out-of-range axis or unrepresentable layout.
    pub fn infer_shapes(&self, input: &Shape) -> Result<BlockScaleQuantizeOutputs<Shape>> {
        let block_size = i64::from(self.native_block_size()?);
        let rank = to_i64(input.rank(), "rank")?;
        let block_axis = self.axis.unwrap_or(-1);
        let block_axis = if block_axis < 0 {
            rank + block_axis
        } else {
            block_axis
        };
        check_range!("axis", block_axis >= 0 && block_axis < rank)?;
        let block_axis = to_usize(block_axis, "axis")?;
        let mut dimensions = input.dimensions().to_vec();
        let dimension = &mut dimensions[block_axis];
        *dimension = *dimension / block_size + i64::from(*dimension % block_size != 0);

        let mut order = input.stride_order();
        if self.transpose && self.axis.is_some() {
            let rotation = order
                .iter()
                .position(|&index| index == block_axis)
                .ok_or(Error::InvalidDataShape)?;
            order.rotate_left(rotation);
        }
        let output = if self.transpose {
            Shape::packed(input.dimensions(), order.iter().copied())?
        } else {
            input.clone()
        };
        let scale = Shape::packed(dimensions, order)?;
        Ok(BlockScaleQuantizeOutputs::new(output, scale))
    }

    pub(crate) fn native_block_size(&self) -> Result<i32> {
        let size = to_i32(self.block_size, "block size")?;
        check_range!("block_size", size > 0)?;
        Ok(size)
    }

    /// An omitted axis reduces the last logical dimension without forcing stride rotation.
    #[must_use]
    pub fn axis(&self) -> Option<i64> {
        self.axis
    }

    /// Returns whether output inference repacks transposed value and scale layouts.
    #[must_use]
    pub fn transpose(&self) -> bool {
        self.transpose
    }
}

/// Attributes for block scale dequantization.
///
/// Block scale dequantize combines quantized values with a block-broadcast scale
/// tensor to produce a dequantized output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockScaleDequantizeConfig {
    name: Option<String>,
    compute_type: Option<DataType>,
    output_data_type: Option<DataType>,
    block_sizes: Vec<i32>,
    is_negative_scale: bool,
}

impl BlockScaleDequantizeConfig {
    pub(crate) fn validate_block_sizes(&self) -> Result<()> {
        if self.block_sizes.is_empty() {
            return Err(Error::EmptyList {
                name: "block_sizes".into(),
            });
        }
        check_range!("block_size", self.block_sizes.iter().all(|&size| size > 0))
    }

    /// Inherits graph compute precision, defaulting to `F32`.
    /// Lowering always supplies the required native math-precision attribute.
    pub fn new(block_sizes: impl Into<Vec<i32>>) -> Self {
        Self {
            name: None,
            compute_type: None,
            output_data_type: None,
            block_sizes: block_sizes.into(),
            is_negative_scale: false,
        }
    }

    /// Names inferred results `name::Y`, matching the C++ dequantization output scope.
    /// Explicit tensor names are retained, and an empty scope disables naming.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Returns the nonempty result-name scope, or `None` when unnamed.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref().filter(|name| !name.is_empty())
    }

    /// Overrides graph math precision independently of block geometry and storage encodings.
    #[must_use]
    pub fn with_compute_type(mut self, compute_type: impl Into<Option<DataType>>) -> Self {
        self.compute_type = compute_type.into();
        self
    }

    /// Sets or clears the native negative-scale flag.
    #[must_use]
    pub fn with_negative_scale(mut self, is_negative_scale: bool) -> Self {
        self.is_negative_scale = is_negative_scale;
        self
    }

    /// Selects the inferred result encoding independently of math precision and graph defaults.
    /// Passing `None` restores inherited result encoding.
    #[must_use]
    pub fn with_output_data_type(mut self, data_type: impl Into<Option<DataType>>) -> Self {
        self.output_data_type = data_type.into();
        self
    }

    /// Returns the dequantized-output encoding override, or `None` for graph inheritance.
    #[must_use]
    pub fn output_data_type(&self) -> Option<DataType> {
        self.output_data_type
    }

    /// Returns the math-precision override, or `None` to inherit graph policy.
    #[must_use]
    pub fn compute_type(&self) -> Option<DataType> {
        self.compute_type
    }

    /// Borrows block extents applied to the trailing input axes.
    #[must_use]
    pub fn block_sizes(&self) -> &[i32] {
        &self.block_sizes
    }

    /// Infers logical scale dimensions for the input without constructing strides or descriptors.
    /// Block sizes apply to the trailing input axes, with each extent rounded up to cover partial blocks.
    /// Explicit scale declarations may use additional encoding-specific padding and independent layouts.
    /// This helper does not restrict the layouts accepted by explicit dequantization construction.
    ///
    /// # Errors
    /// Returns an error for empty or nonpositive block sizes, or more block axes than input axes.
    pub fn infer_scale_dimensions(&self, input: &Shape) -> Result<Vec<i64>> {
        self.validate_block_sizes()?;
        let mut dimensions = input.dimensions().to_vec();
        let rank = dimensions.len();
        let block_rank = self.block_sizes.len();
        if block_rank > rank {
            return Err(Error::LengthMismatch {
                name: BLOCK_SCALE_RANK.into(),
                expected: rank,
                actual: block_rank,
            });
        }
        for (dimension, &block_size) in dimensions[rank - block_rank..]
            .iter_mut()
            .zip(&self.block_sizes)
        {
            let block = i64::from(block_size);
            *dimension = *dimension / block + i64::from(*dimension % block != 0);
        }
        Ok(dimensions)
    }

    /// Returns the native negative-scale flag.
    #[must_use]
    pub fn is_negative_scale(&self) -> bool {
        self.is_negative_scale
    }
}

impl_inherited_compute_type!(BlockScaleQuantizeConfig, BlockScaleDequantizeConfig);

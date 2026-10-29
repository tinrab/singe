use std::result::Result as StdResult;

use crate::{
    backend::{
        attribute::BackendAttributeName,
        descriptor::{BackendDescriptor, BackendDescriptorType},
        tensor::{BackendTensorUid, Tensor, TensorId},
    },
    data_type::DataType,
    error::Result,
    utility::check_range,
};

/// Quantized values and their block scales, shared by frontend IDs, native references and output metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockScaleQuantizeOutputs<T = TensorId> {
    /// Quantized value output or its frontend inference policy.
    pub output: T,
    /// Block-scale output or its independent frontend inference policy.
    pub scale: T,
}

impl<T> BlockScaleQuantizeOutputs<T> {
    /// Records the quantized-value and block-scale destinations.
    pub fn new(output: T, scale: T) -> Self {
        Self { output, scale }
    }

    /// Borrows value and scale output owners for native construction.
    pub fn as_ref(&self) -> BlockScaleQuantizeOutputs<&T> {
        BlockScaleQuantizeOutputs {
            output: &self.output,
            scale: &self.scale,
        }
    }

    /// Maps both outputs while preserving their roles.
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> BlockScaleQuantizeOutputs<U> {
        BlockScaleQuantizeOutputs {
            output: map(self.output),
            scale: map(self.scale),
        }
    }

    /// Maps quantized values and block scales without changing their roles.
    ///
    /// # Errors
    ///
    /// Returns the first tensor-mapping error, processing values before block scales.
    pub fn try_map<U, E>(
        self,
        mut map: impl FnMut(T) -> StdResult<U, E>,
    ) -> StdResult<BlockScaleQuantizeOutputs<U>, E> {
        Ok(BlockScaleQuantizeOutputs {
            output: map(self.output)?,
            scale: map(self.scale)?,
        })
    }
}

/// Quantized values and their block scales, shared by frontend IDs, native references and input metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockScaleDequantizeInputs<T = TensorId> {
    /// Quantized value input.
    pub input: T,
    /// Block-scale input.
    pub scale: T,
}

impl<T> BlockScaleDequantizeInputs<T> {
    /// Records the quantized-value and block-scale operands.
    pub fn new(input: T, scale: T) -> Self {
        Self { input, scale }
    }

    /// Borrows value and scale input owners for native construction.
    pub fn as_ref(&self) -> BlockScaleDequantizeInputs<&T> {
        BlockScaleDequantizeInputs {
            input: &self.input,
            scale: &self.scale,
        }
    }

    /// Maps both inputs while preserving their roles.
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> BlockScaleDequantizeInputs<U> {
        BlockScaleDequantizeInputs {
            input: map(self.input),
            scale: map(self.scale),
        }
    }

    /// Maps values and block scales without changing their roles.
    ///
    /// # Errors
    ///
    /// Returns the first tensor-mapping error, processing values before block scales.
    pub fn try_map<U, E>(
        self,
        mut map: impl FnMut(T) -> StdResult<U, E>,
    ) -> StdResult<BlockScaleDequantizeInputs<U>, E> {
        Ok(BlockScaleDequantizeInputs {
            input: map(self.input)?,
            scale: map(self.scale)?,
        })
    }
}

/// Native quantization attributes; the tensor layouts select the packed block axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockScaleQuantizeOperationConfig {
    /// Native math precision, independent of value and scale storage encodings.
    pub compute_type: DataType,
    /// Positive number of values sharing a quantization scale; native creation validates this.
    pub block_size: i32,
}

impl BlockScaleQuantizeOperationConfig {
    /// Records native math precision and a block size for validation during operation creation.
    #[must_use]
    pub fn new(compute_type: DataType, block_size: i32) -> Self {
        Self {
            compute_type,
            block_size,
        }
    }
}

/// Native dequantization attributes, independent of output encoding and scale layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockScaleDequantizeOperationConfig {
    /// Native math precision, independent of value and scale storage encodings.
    pub compute_type: DataType,
    /// Positive block extents; native creation rejects an empty list or nonpositive entry.
    pub block_sizes: Vec<i32>,
    /// Native negative-scale selection flag.
    pub is_negative_scale: bool,
}

impl BlockScaleDequantizeOperationConfig {
    /// Records native math precision and block extents with the negative-scale flag disabled.
    pub fn new(compute_type: DataType, block_sizes: impl Into<Vec<i32>>) -> Self {
        Self {
            compute_type,
            block_sizes: block_sizes.into(),
            is_negative_scale: false,
        }
    }

    /// Sets or clears the native negative-scale flag while retaining precision and block sizes.
    #[must_use]
    pub fn with_negative_scale(mut self, is_negative_scale: bool) -> Self {
        self.is_negative_scale = is_negative_scale;
        self
    }
}

/// Retained input/value/scale identities and accepted native quantization settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockScaleQuantizeOperationMetadata {
    /// Native identity of the input activation.
    pub input: BackendTensorUid,
    /// Native identities of the quantized values and block-scale outputs.
    pub outputs: BlockScaleQuantizeOutputs<BackendTensorUid>,
    /// Retained math precision and native block size.
    pub config: BlockScaleQuantizeOperationConfig,
}

/// Finalized native block-quantization operation with retained value and scale identities.
#[derive(Debug)]
pub struct BlockScaleQuantizeOperation {
    descriptor: BackendDescriptor,
    metadata: BlockScaleQuantizeOperationMetadata,
}

impl BlockScaleQuantizeOperation {
    /// Creates quantization using the supplied tensor layouts and native block size.
    ///
    /// # Errors
    /// Returns an error for a nonpositive block size or native descriptor failure.
    pub fn create(
        x: &Tensor,
        outputs: BlockScaleQuantizeOutputs<&Tensor>,
        config: BlockScaleQuantizeOperationConfig,
    ) -> Result<Self> {
        let compute_type = config.compute_type;
        let block_size = config.block_size;
        check_range!("block_size", block_size > 0)?;

        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationBlockScaleQuantize)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationBlockScaleQuantizeXDesc,
            x.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationBlockScaleQuantizeYDesc,
            outputs.output.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationBlockScaleQuantizeScaleDesc,
            outputs.scale.descriptor(),
        )?;
        descriptor.set_attribute_enum(
            BackendAttributeName::OperationBlockScaleQuantizeMathPrec,
            compute_type,
        )?;
        descriptor.set_attribute_i32(
            BackendAttributeName::OperationBlockScaleQuantizeBlockSize,
            block_size,
        )?;
        descriptor.finalize()?;

        Ok(Self {
            descriptor,
            metadata: BlockScaleQuantizeOperationMetadata {
                input: x.id(),
                outputs: outputs.map(Tensor::id),
                config,
            },
        })
    }

    /// Retained native attributes without descriptor queries or allocation.
    #[must_use]
    pub fn config(&self) -> BlockScaleQuantizeOperationConfig {
        self.metadata.config
    }

    /// Retained output identities without native descriptor queries.
    #[must_use]
    pub fn output_uids(&self) -> &BlockScaleQuantizeOutputs<BackendTensorUid> {
        &self.metadata.outputs
    }

    /// Retained native operands and configuration without queries or allocation.
    #[must_use]
    pub fn metadata(&self) -> &BlockScaleQuantizeOperationMetadata {
        &self.metadata
    }

    /// Borrows the finalized native block-scale operation descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

/// Retained value/scale/output identities and accepted native dequantization settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockScaleDequantizeOperationMetadata {
    /// Native identities of quantized values and block scales.
    pub inputs: BlockScaleDequantizeInputs<BackendTensorUid>,
    /// Native identity of the dequantized output.
    pub output: BackendTensorUid,
    /// Retained math precision, block sizes, and scale-sign selection.
    pub config: BlockScaleDequantizeOperationConfig,
}

/// Finalized native block-dequantization operation with retained operands and accepted settings.
#[derive(Debug)]
pub struct BlockScaleDequantizeOperation {
    descriptor: BackendDescriptor,
    metadata: BlockScaleDequantizeOperationMetadata,
}

impl BlockScaleDequantizeOperation {
    /// Creates dequantization with explicit compute precision, block sizes and scale-sign selection.
    ///
    /// # Errors
    /// Returns an error for empty or nonpositive block sizes or native descriptor failure.
    pub fn create(
        inputs: BlockScaleDequantizeInputs<&Tensor>,
        y: &Tensor,
        config: BlockScaleDequantizeOperationConfig,
    ) -> Result<Self> {
        let compute_type = config.compute_type;
        let block_sizes = &config.block_sizes;
        let is_negative_scale = config.is_negative_scale;
        check_range!(
            "block_sizes",
            !block_sizes.is_empty() && block_sizes.iter().all(|&block_size| block_size > 0)
        )?;

        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationBlockScaleDequantize)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationBlockScaleDequantizeXDesc,
            inputs.input.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationBlockScaleDequantizeScaleDesc,
            inputs.scale.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationBlockScaleDequantizeYDesc,
            y.descriptor(),
        )?;
        descriptor.set_attribute_enum(
            BackendAttributeName::OperationBlockScaleDequantizeMathPrec,
            compute_type,
        )?;
        descriptor.set_attribute_i32_slice(
            BackendAttributeName::OperationBlockScaleDequantizeBlockSize,
            block_sizes,
        )?;
        descriptor.set_attribute_bool(
            BackendAttributeName::OperationBlockScaleDequantizeNegScale,
            is_negative_scale,
        )?;
        descriptor.finalize()?;

        Ok(Self {
            descriptor,
            metadata: BlockScaleDequantizeOperationMetadata {
                inputs: inputs.map(Tensor::id),
                output: y.id(),
                config,
            },
        })
    }

    /// Retained value/scale native identities without descriptor queries.
    #[must_use]
    pub fn input_uids(&self) -> &BlockScaleDequantizeInputs<BackendTensorUid> {
        &self.metadata.inputs
    }

    /// Retained output native identity without descriptor queries.
    #[must_use]
    pub fn output_uid(&self) -> BackendTensorUid {
        self.metadata.output
    }

    /// Retained native attributes and block sizes without descriptor queries or allocation.
    #[must_use]
    pub fn config(&self) -> &BlockScaleDequantizeOperationConfig {
        &self.metadata.config
    }

    /// Retained native operands and configuration without queries or allocation.
    #[must_use]
    pub fn metadata(&self) -> &BlockScaleDequantizeOperationMetadata {
        &self.metadata
    }

    /// Borrows the finalized native block-scale operation descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

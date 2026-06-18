use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cudnn_sys as sys;

use crate::{
    attribute::{BackendAttributeName, BackendAttributeType},
    descriptor::{BackendDescriptor, BackendDescriptorType},
    error::{Error, Result},
    tensor::Tensor,
};

/// Signaling mode for backend signal operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum SignalMode {
    /// The flag variable is updated with the provided signal value atomically.
    Set = sys::cudnnSignalMode_t::CUDNN_SIGNAL_SET as _,
    /// The operation blocks until the flag variable keeps comparing equal to the provided signal value.
    Wait = sys::cudnnSignalMode_t::CUDNN_SIGNAL_WAIT as _,
}

impl_enum_conversion!(sys::cudnnSignalMode_t, SignalMode);

impl_enum_display!(SignalMode, {
    Self::Set => "CUDNN_SIGNAL_SET",
    Self::Wait => "CUDNN_SIGNAL_WAIT",
});

#[derive(Debug)]
pub struct SignalOperation {
    descriptor: BackendDescriptor,
}

impl SignalOperation {
    pub fn create(
        mode: SignalMode,
        flag: &Tensor,
        value: i64,
        x: Option<&Tensor>,
        y: Option<&Tensor>,
    ) -> Result<Self> {
        if x.is_some() != y.is_some() {
            return Err(Error::DescriptorMismatch {
                name: "signal passthrough".into(),
            });
        }

        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::OperationSignal)?;
        descriptor.set_attribute_enum(
            BackendAttributeName::OperationSignalMode,
            BackendAttributeType::SignalMode,
            mode,
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSignalFlagDesc,
            flag.descriptor(),
        )?;
        descriptor.set_attribute_i64(BackendAttributeName::OperationSignalValue, value)?;
        if let Some(x) = x {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSignalXDesc,
                x.descriptor(),
            )?;
        }
        if let Some(y) = y {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSignalYDesc,
                y.descriptor(),
            )?;
        }
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

#[derive(Debug)]
pub struct ReshapeOperation {
    descriptor: BackendDescriptor,
}

impl ReshapeOperation {
    pub fn create(x: &Tensor, y: &Tensor) -> Result<Self> {
        if x.data_type() != y.data_type() {
            return Err(Error::DescriptorMismatch {
                name: "reshape data type".into(),
            });
        }
        let x_elements = x
            .shape()
            .dimensions()
            .iter()
            .try_fold(1_i64, |product, &dimension| {
                product.checked_mul(dimension).ok_or(Error::OutOfRange {
                    name: "reshape element count".into(),
                })
            })?;
        let y_elements = y
            .shape()
            .dimensions()
            .iter()
            .try_fold(1_i64, |product, &dimension| {
                product.checked_mul(dimension).ok_or(Error::OutOfRange {
                    name: "reshape element count".into(),
                })
            })?;
        if x_elements != y_elements {
            return Err(Error::DescriptorMismatch {
                name: "reshape element count".into(),
            });
        }

        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::OperationReshape)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationReshapeXDesc,
            x.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationReshapeYDesc,
            y.descriptor(),
        )?;
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

#[derive(Debug)]
pub struct ConcatOperation {
    descriptor: BackendDescriptor,
    inplace_index: Option<i64>,
}

impl ConcatOperation {
    pub fn create(
        inputs: &[&Tensor],
        output: &Tensor,
        axis: i64,
        inplace_index: Option<i64>,
    ) -> Result<Self> {
        if inputs.is_empty() {
            return Err(Error::EmptyList {
                name: "inputs".into(),
            });
        }

        let output_rank = output.shape().rank();
        if axis < 0 || axis >= output_rank {
            return Err(Error::OutOfRange {
                name: "axis".into(),
            });
        }
        if let Some(inplace_index) = inplace_index {
            let input_count = i64::try_from(inputs.len()).map_err(|_| Error::OutOfRange {
                name: "input count".into(),
            })?;
            if inplace_index < 0 || inplace_index >= input_count {
                return Err(Error::OutOfRange {
                    name: "inplace_index".into(),
                });
            }
        }
        if inputs
            .iter()
            .any(|input| input.shape().rank() != output.shape().rank())
        {
            return Err(Error::DescriptorMismatch {
                name: "concat ranks".into(),
            });
        }
        let axis = usize::try_from(axis).map_err(|_| Error::OutOfRange {
            name: "axis".into(),
        })?;
        let mut expected_dimensions = output.shape().dimensions().to_vec();
        let mut axis_sum = 0_i64;
        for input in inputs {
            for (index, (&input_dimension, &output_dimension)) in input
                .shape()
                .dimensions()
                .iter()
                .zip(output.shape().dimensions())
                .enumerate()
            {
                if index == axis {
                    axis_sum = axis_sum
                        .checked_add(input_dimension)
                        .ok_or(Error::OutOfRange {
                            name: "concat axis dimension".into(),
                        })?;
                } else if input_dimension != output_dimension {
                    return Err(Error::DescriptorMismatch {
                        name: "concat output shape".into(),
                    });
                }
            }
        }
        expected_dimensions[axis] = axis_sum;
        if output.shape().dimensions() != expected_dimensions {
            return Err(Error::DescriptorMismatch {
                name: "concat output shape".into(),
            });
        }

        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::OperationConcat)?;
        let input_descriptors = inputs
            .iter()
            .map(|tensor| tensor.descriptor())
            .collect::<Vec<_>>();
        descriptor.set_attribute_i64(
            BackendAttributeName::OperationConcatAxis,
            i64::try_from(axis).map_err(|_| Error::OutOfRange {
                name: "axis".into(),
            })?,
        )?;
        descriptor.set_attribute_descriptor_slice(
            BackendAttributeName::OperationConcatInputDescs,
            &input_descriptors,
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationConcatOutputDesc,
            output.descriptor(),
        )?;
        if let Some(inplace_index) = inplace_index {
            descriptor.set_attribute_i64(
                BackendAttributeName::OperationConcatInplaceIndex,
                inplace_index,
            )?;
        }
        descriptor.finalize()?;

        Ok(Self {
            descriptor,
            inplace_index,
        })
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }

    pub fn inplace_index(&self) -> Option<i64> {
        self.inplace_index
    }
}

#![allow(dead_code)]

use std::{marker::PhantomData, mem::size_of};

use cutile::{
    DType,
    cuda_async::{
        device_operation::{IntoDeviceOp, Value, value},
        launch as async_launcher,
    },
    tensor::{
        KernelInput, KernelInputStored, KernelOutput, KernelOutputStored, SpecializationBits,
        compute_spec,
    },
    tile_kernel::DevicePointer,
};

use crate::{
    error::Result,
    utility::{checked_element_count, checked_i32_value},
};

use async_launcher::AsyncKernelLaunch;

#[derive(Debug, Clone)]
pub struct TensorAdapter<'a, T: DType> {
    ptr: DevicePointer<T>,
    shape: Vec<i32>,
    strides: Vec<i32>,
    spec: SpecializationBits,
    _storage: PhantomData<&'a T>,
}

impl<'a, T: DType> TensorAdapter<'a, T> {
    pub fn contiguous_1d(ptr: DevicePointer<T>, len: usize) -> Result<Self> {
        let len = checked_i32_value(len)?;
        Self::create(ptr, vec![len], vec![1])
    }

    pub fn contiguous_2d(ptr: DevicePointer<T>, rows: usize, cols: usize) -> Result<Self> {
        let rows = checked_i32_value(rows)?;
        let cols = checked_i32_value(cols)?;
        Self::create(ptr, vec![rows, cols], vec![cols, 1])
    }

    pub fn contiguous_3d(
        ptr: DevicePointer<T>,
        dim0: usize,
        dim1: usize,
        dim2: usize,
    ) -> Result<Self> {
        let stride0 = checked_i32_value(checked_element_count(dim1, dim2)?)?;
        let stride1 = checked_i32_value(dim2)?;
        let dim0 = checked_i32_value(dim0)?;
        let dim1 = checked_i32_value(dim1)?;
        let dim2 = checked_i32_value(dim2)?;
        Self::create(ptr, vec![dim0, dim1, dim2], vec![stride0, stride1, 1])
    }

    pub fn contiguous_4d(
        ptr: DevicePointer<T>,
        dim0: usize,
        dim1: usize,
        dim2: usize,
        dim3: usize,
    ) -> Result<Self> {
        let stride0 = checked_i32_value(checked_element_count(
            checked_element_count(dim1, dim2)?,
            dim3,
        )?)?;
        let stride1 = checked_i32_value(checked_element_count(dim2, dim3)?)?;
        let stride2 = checked_i32_value(dim3)?;
        let dim0 = checked_i32_value(dim0)?;
        let dim1 = checked_i32_value(dim1)?;
        let dim2 = checked_i32_value(dim2)?;
        let dim3 = checked_i32_value(dim3)?;
        Self::create(
            ptr,
            vec![dim0, dim1, dim2, dim3],
            vec![stride0, stride1, stride2, 1],
        )
    }

    fn create(ptr: DevicePointer<T>, shape: Vec<i32>, strides: Vec<i32>) -> Result<Self> {
        let spec = compute_spec(ptr.cu_deviceptr(), &shape, &strides, size_of::<T>() as i32);
        Ok(Self {
            ptr,
            shape,
            strides,
            spec,
            _storage: PhantomData,
        })
    }

    pub fn partition<const RANK: usize>(
        self,
        partition_shape: [usize; RANK],
    ) -> Result<TensorAdapterPartition<'a, T>> {
        let partition_shape = partition_shape
            .into_iter()
            .map(checked_i32_value)
            .collect::<Result<Vec<_>>>()?;
        let partition_strides = contiguous_strides(&partition_shape);
        Ok(TensorAdapterPartition {
            tensor: self,
            partition_shape,
            partition_strides,
        })
    }
}

#[derive(Debug, Clone)]
pub struct TensorAdapterPartition<'a, T: DType> {
    tensor: TensorAdapter<'a, T>,
    partition_shape: Vec<i32>,
    partition_strides: Vec<i32>,
}

fn contiguous_strides(shape: &[i32]) -> Vec<i32> {
    let mut stride = 1i32;
    let mut strides = Vec::with_capacity(shape.len());
    for dim in shape.iter().rev() {
        strides.push(stride);
        stride = stride.saturating_mul(*dim);
    }
    strides.reverse();
    strides
}

fn grid(shape: &[i32], partition_shape: &[i32]) -> cutile::error::Result<(u32, u32, u32)> {
    let shape = shape.iter().map(|&dim| dim as u32).collect::<Vec<_>>();
    let partition_shape = partition_shape
        .iter()
        .map(|&dim| dim as u32)
        .collect::<Vec<_>>();
    match shape.len() {
        1 => Ok((shape[0].div_ceil(partition_shape[0]), 1, 1)),
        2 => Ok((
            shape[0].div_ceil(partition_shape[0]),
            shape[1].div_ceil(partition_shape[1]),
            1,
        )),
        3 => Ok((
            shape[0].div_ceil(partition_shape[0]),
            shape[1].div_ceil(partition_shape[1]),
            shape[2].div_ceil(partition_shape[2]),
        )),
        _ => Err(cutile::error::tensor_error(
            "mutable tensor must be at most rank 3.",
        )),
    }
}

impl<T: DType> TensorAdapter<'_, T> {
    fn push_tensor_args(&self, launcher: &mut AsyncKernelLaunch) {
        unsafe {
            launcher.push_device_ptr(self.ptr.cu_deviceptr());
        }
        for dim in &self.shape {
            launcher.push_arg(*dim);
        }
        for stride in &self.strides {
            launcher.push_arg(*stride);
        }
    }
}

impl<T: DType> TensorAdapterPartition<'_, T> {
    fn push_partition_args(&self, launcher: &mut AsyncKernelLaunch) {
        self.tensor.push_tensor_args(launcher);
        for dim in &self.partition_shape {
            launcher.push_arg(*dim);
        }
        for stride in &self.partition_strides {
            launcher.push_arg(*stride);
        }
    }
}

impl<T: DType + Sync> KernelInputStored for TensorAdapter<'_, T> {
    fn push_kernel_args(&self, launcher: &mut AsyncKernelLaunch) {
        self.push_tensor_args(launcher);
    }

    fn shape(&self) -> &[i32] {
        &self.shape
    }

    fn strides(&self) -> &[i32] {
        &self.strides
    }

    fn spec(&self) -> &SpecializationBits {
        &self.spec
    }

    fn dtype_str(&self) -> &'static str {
        T::DTYPE.as_str()
    }
}

impl<T: DType + Sync> KernelInput<T> for TensorAdapter<'_, T> {
    type Returned = Self;
    type Stored = Self;

    fn prepare(self) -> Self::Stored {
        self
    }

    fn recover(stored: Self::Stored) -> Self::Returned {
        stored
    }
}

impl<'a, T: DType + Sync> IntoDeviceOp<TensorAdapter<'a, T>> for TensorAdapter<'a, T> {
    type Op = Value<TensorAdapter<'a, T>>;

    fn into_op(self) -> Self::Op {
        value(self)
    }
}

impl<T: DType + Sync> KernelOutputStored<T> for TensorAdapterPartition<'_, T> {
    fn push_kernel_args(&self, launcher: &mut AsyncKernelLaunch) {
        self.push_partition_args(launcher);
    }

    fn grid(&self) -> cutile::error::Result<(u32, u32, u32)> {
        grid(&self.tensor.shape, &self.partition_shape)
    }

    fn dtype_str(&self) -> &'static str {
        T::DTYPE.as_str()
    }

    fn partition_shape_as_i32(&self) -> Vec<i32> {
        self.partition_shape.clone()
    }

    fn strides_hint(&self) -> Vec<i32> {
        self.tensor
            .spec
            .stride_one
            .iter()
            .map(|&is_one| if is_one { 1 } else { -1 })
            .collect()
    }

    fn spec(&self) -> &SpecializationBits {
        &self.tensor.spec
    }

    fn shape_as_i32(&self) -> Vec<i32> {
        self.tensor.shape.clone()
    }
}

impl<T: DType + Sync> KernelOutput<T> for TensorAdapterPartition<'_, T> {
    type Returned = Self;
    type Stored = Self;

    fn prepare(self) -> Self::Stored {
        self
    }

    fn recover(stored: Self::Stored) -> Self::Returned {
        stored
    }
}

impl<'a, T: DType + Sync> IntoDeviceOp<TensorAdapterPartition<'a, T>>
    for TensorAdapterPartition<'a, T>
{
    type Op = Value<TensorAdapterPartition<'a, T>>;

    fn into_op(self) -> Self::Op {
        value(self)
    }
}

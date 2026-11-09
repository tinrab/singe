use std::{
    mem::{ManuallyDrop, MaybeUninit, size_of},
    slice,
};

#[cfg(feature = "cuda")]
use crate::backend::cuda;

use crate::{
    error::{Error, Result},
    observation::check_storage_readback,
    tensor::{Tensor, TensorDataType},
};

impl Tensor {
    /// Copies one element to the host using logical tensor indices.
    /// Negative indices count backward from the end of each axis.
    /// The CUDA path synchronizes the current stream before reading storage.
    ///
    /// # Errors
    ///
    /// Returns an error if capture forbids readback, `T` differs from the tensor data type, the index count or an index is invalid, stride or offset arithmetic fails, stream synchronization or storage copying fails, or the copied bytes are invalid for `T`.
    pub fn get_element<T: TensorDataType>(&self, indices: &[i64]) -> Result<T> {
        check_storage_readback(self.storage().storage_identity())?;
        if self.data_type() != T::data_type() {
            return Err(Error::data_type_mismatch(
                "tensor get_element data type",
                T::data_type(),
                self.data_type(),
            ));
        }

        let rank = usize::try_from(self.shape().rank())
            .map_err(|_| Error::out_of_range("tensor get_element rank".to_string()))?;
        if indices.len() != rank {
            return Err(Error::length_mismatch(
                "tensor get_element indices",
                rank,
                indices.len(),
            ));
        }

        let mut flat_index = 0usize;
        for (i, &idx) in indices.iter().enumerate() {
            let normalized = if idx < 0 {
                self.shape().dimensions()[i] + idx
            } else {
                idx
            };
            if normalized < 0 || normalized >= self.shape().dimensions()[i] {
                return Err(Error::IndexOutOfBounds);
            }

            let normalized = usize::try_from(normalized)
                .map_err(|_| Error::out_of_range("tensor get_element index".to_string()))?;
            let stride =
                usize::try_from(self.layout().strides()[i]).map_err(|_| Error::InvalidStride)?;
            let offset = normalized
                .checked_mul(stride)
                .ok_or_else(|| Error::out_of_range("tensor get_element offset".to_string()))?;
            flat_index = flat_index
                .checked_add(offset)
                .ok_or_else(|| Error::out_of_range("tensor get_element offset".to_string()))?;
        }

        flat_index = flat_index
            .checked_add(self.layout().storage_offset())
            .ok_or_else(|| Error::out_of_range("tensor get_element storage offset".to_string()))?;

        self.synchronize_current_stream_for_host_read()?;
        let mut bytes = vec![0_u8; self.inner.storage.byte_size()];
        self.inner.storage.copy_to_host(&mut bytes)?;
        let start = flat_index * size_of::<T>();
        let values = initialized_values::<T>(size_of::<T>(), |output| {
            output.copy_from_slice(&bytes[start..start + size_of::<T>()]);
            Ok(())
        })?;
        Ok(values[0])
    }

    /// Copies logical elements into a host vector in row-major order.
    /// Strided and broadcast tensors are gathered according to their layout.
    ///
    /// # Errors
    ///
    /// Returns an error if capture forbids readback, `T` differs from the tensor data type, the byte count overflows, synchronization or storage copying fails, or copied representations are invalid for `T`.
    pub fn to_vec<T: TensorDataType>(&self) -> Result<Vec<T>> {
        check_storage_readback(self.storage().storage_identity())?;
        if self.data_type() != T::data_type() {
            return Err(Error::data_type_mismatch(
                "tensor to_vec data type",
                T::data_type(),
                self.data_type(),
            ));
        }

        let element_count = self.shape().element_count();
        if element_count == 0 {
            return Ok(Vec::new());
        }

        let byte_size = element_count
            .checked_mul(T::data_type().size_bytes())
            .ok_or_else(|| Error::out_of_range("tensor to_vec byte size".to_string()))?;
        initialized_values::<T>(byte_size, |output| self.copy_logical_bytes(output))
    }

    /// Copies logical values to contiguous native-endian host bytes.
    /// Strides, offsets, and broadcast views do not require a device compute provider.
    ///
    /// # Errors
    ///
    /// Returns an error if capture forbids readback, the logical byte count overflows, or stream synchronization or storage copying fails.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        check_storage_readback(self.storage().storage_identity())?;
        let mut output = vec![0_u8; self.byte_size()?];
        self.copy_logical_bytes(&mut output)?;
        Ok(output)
    }

    fn copy_logical_bytes(&self, output: &mut [u8]) -> Result<()> {
        if output.is_empty() {
            return Ok(());
        }

        self.synchronize_current_stream_for_host_read()?;
        if self.is_contiguous()
            && self.storage_offset() == 0
            && output.len() == self.inner.storage.byte_size()
        {
            return self.inner.storage.copy_to_host(output);
        }

        let mut storage_bytes = vec![0_u8; self.inner.storage.byte_size()];
        self.inner.storage.copy_to_host(&mut storage_bytes)?;
        let element_size = self.data_type().size_bytes();
        for (linear, element) in output.chunks_exact_mut(element_size).enumerate() {
            let start = self.layout().linear_offset(linear) * element_size;
            element.copy_from_slice(&storage_bytes[start..start + element_size]);
        }
        Ok(())
    }

    #[cfg_attr(
        not(feature = "cuda"),
        expect(
            clippy::unused_self,
            clippy::unnecessary_wraps,
            reason = "only CUDA has streams to wait for"
        )
    )]
    fn synchronize_current_stream_for_host_read(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            if self.device().is_cuda()
                && let Some(stream) = cuda::current_stream(self.device())?
            {
                stream.synchronize()?;
            }
        }
        Ok(())
    }
}

fn initialized_values<T: TensorDataType>(
    byte_size: usize,
    copy: impl FnOnce(&mut [u8]) -> Result<()>,
) -> Result<Vec<T>> {
    let element_size = size_of::<T>();
    if !byte_size.is_multiple_of(element_size) {
        return Err(Error::byte_length_mismatch(
            "tensor storage bytes",
            element_size,
            byte_size,
        ));
    }

    let length = byte_size / element_size;
    // MaybeUninit preserves invalid scalar representations until they have been checked.
    // Zero the staging bytes so the mutable byte slice is initialized before the copy.
    let mut values = vec![MaybeUninit::<T>::zeroed(); length];
    // SAFETY: the staging allocation contains exactly byte_size initialized bytes.
    // TensorDataType is sealed to nonzero-sized scalar types.
    let bytes = unsafe { slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), byte_size) };
    copy(bytes)?;
    T::from_bytes(bytes)?;
    let mut values = ManuallyDrop::new(values);
    // SAFETY: the checked representation, size, and alignment match T; ownership moves once.
    Ok(unsafe {
        Vec::from_raw_parts(
            values.as_mut_ptr().cast::<T>(),
            values.len(),
            values.capacity(),
        )
    })
}

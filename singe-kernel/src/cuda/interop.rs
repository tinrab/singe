use cutile::cuda_async::device_buffer::DevicePointer;
use singe_cuda::view::{DeviceInput, DeviceOutput, DeviceRepr, DeviceSliceMut};

#[allow(dead_code, reason = "only used by some kernel feature combinations")]
pub fn input_pointer<T, S>(input: &S) -> DevicePointer<T>
where
    T: DeviceRepr,
    S: DeviceInput<T> + ?Sized,
{
    // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
    unsafe { DevicePointer::from_cu_deviceptr(input.as_device_ptr() as _) }
}

#[allow(dead_code, reason = "only used by some kernel feature combinations")]
pub fn output_pointer<T>(output: &mut (impl DeviceSliceMut<T> + ?Sized)) -> DevicePointer<T>
where
    T: DeviceRepr,
{
    // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
    unsafe { DevicePointer::from_cu_deviceptr(output.as_mut_ptr() as _) }
}

/// Encodes a write-only destination without asserting initialized input values.
pub fn write_only_pointer<T>(output: &mut (impl DeviceOutput<T> + ?Sized)) -> DevicePointer<T>
where
    T: DeviceRepr,
{
    // SAFETY: DeviceOutput guarantees an aligned CUDA destination of the stated
    // length. Constructing the address does not read or initialize its contents.
    unsafe { DevicePointer::from_cu_deviceptr(output.as_mut_ptr() as _) }
}

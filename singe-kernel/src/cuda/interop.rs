use std::{
    ffi::{c_int, c_void},
    sync::Arc,
};

use cutile::{
    cuda_async::device_buffer::DevicePointer,
    cuda_core::{Device, Stream as CutileStream},
};
use singe_cuda::{
    stream::Stream,
    view::{DeviceRepr, DeviceSlice, DeviceSliceMut},
};

use crate::error::Result;

#[allow(dead_code)]
pub fn borrowed_stream(stream: &Stream) -> Result<Arc<CutileStream>> {
    let context = stream.context();
    context.bind()?;
    let device = context.device();
    let cutile_device = unsafe {
        Device::borrow_raw(
            context.as_raw().cast::<c_void>(),
            device.id() as c_int,
            device.id() as usize,
        )
    };
    Ok(unsafe { CutileStream::borrow_raw(stream.as_raw().cast::<c_void>(), &cutile_device) })
}

#[allow(dead_code)]
pub fn input_pointer<T, S>(input: &S) -> DevicePointer<T>
where
    T: DeviceRepr,
    S: DeviceSlice<T> + ?Sized,
{
    unsafe { DevicePointer::from_cu_deviceptr(input.as_device_ptr() as _) }
}

#[allow(dead_code)]
pub fn output_pointer<T>(output: &mut impl DeviceSliceMut<T>) -> DevicePointer<T>
where
    T: DeviceRepr,
{
    unsafe { DevicePointer::from_cu_deviceptr(output.as_device_mut_ptr() as _) }
}

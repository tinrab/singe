use singe_cuda::{
    memory::{DeviceMemory, MemoryCopyKind},
    types::{Complex32, Complex64},
};

use crate::{
    context::StreamContext,
    error::Result,
    pipeline::SignalPipeline,
    signal::{
        initialization,
        view::{SignalView, SignalViewMut},
    },
    types::{ComplexI16, ComplexI32, ComplexI64},
    utility::validate_same_len,
};

pub trait CopySignal<T> {
    fn copy(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;
}

fn copy_device_to_device<T>(
    source: &SignalView<'_, T>,
    destination: &mut SignalViewMut<'_, T>,
) -> Result<()> {
    validate_same_len(source.len(), destination.len(), "destination")?;
    unsafe {
        DeviceMemory::copy(
            destination.as_mut_ptr(),
            source.as_ptr(),
            source.len(),
            MemoryCopyKind::DeviceToDevice,
        )?;
    }
    Ok(())
}

impl<'a> CopySignal<i8> for SignalPipeline<'a, i8> {
    fn copy(
        _stream_context: &StreamContext,
        source: &SignalView<'_, i8>,
        destination: &mut SignalViewMut<'_, i8>,
    ) -> Result<()> {
        copy_device_to_device(source, destination)
    }
}

impl<'a> CopySignal<u16> for SignalPipeline<'a, u16> {
    fn copy(
        _stream_context: &StreamContext,
        source: &SignalView<'_, u16>,
        destination: &mut SignalViewMut<'_, u16>,
    ) -> Result<()> {
        copy_device_to_device(source, destination)
    }
}

impl<'a> CopySignal<u32> for SignalPipeline<'a, u32> {
    fn copy(
        _stream_context: &StreamContext,
        source: &SignalView<'_, u32>,
        destination: &mut SignalViewMut<'_, u32>,
    ) -> Result<()> {
        copy_device_to_device(source, destination)
    }
}

impl<'a> CopySignal<u8> for SignalPipeline<'a, u8> {
    fn copy(
        stream_context: &StreamContext,
        source: &SignalView<'_, u8>,
        destination: &mut SignalViewMut<'_, u8>,
    ) -> Result<()> {
        initialization::copy_u8(stream_context, source, destination)
    }
}

impl<'a> CopySignal<i16> for SignalPipeline<'a, i16> {
    fn copy(
        stream_context: &StreamContext,
        source: &SignalView<'_, i16>,
        destination: &mut SignalViewMut<'_, i16>,
    ) -> Result<()> {
        initialization::copy_i16(stream_context, source, destination)
    }
}

impl<'a> CopySignal<i32> for SignalPipeline<'a, i32> {
    fn copy(
        stream_context: &StreamContext,
        source: &SignalView<'_, i32>,
        destination: &mut SignalViewMut<'_, i32>,
    ) -> Result<()> {
        initialization::copy_i32(stream_context, source, destination)
    }
}

impl<'a> CopySignal<i64> for SignalPipeline<'a, i64> {
    fn copy(
        stream_context: &StreamContext,
        source: &SignalView<'_, i64>,
        destination: &mut SignalViewMut<'_, i64>,
    ) -> Result<()> {
        initialization::copy_i64(stream_context, source, destination)
    }
}

impl<'a> CopySignal<ComplexI16> for SignalPipeline<'a, ComplexI16> {
    fn copy(
        stream_context: &StreamContext,
        source: &SignalView<'_, ComplexI16>,
        destination: &mut SignalViewMut<'_, ComplexI16>,
    ) -> Result<()> {
        initialization::copy_i16_complex(stream_context, source, destination)
    }
}

impl<'a> CopySignal<ComplexI32> for SignalPipeline<'a, ComplexI32> {
    fn copy(
        stream_context: &StreamContext,
        source: &SignalView<'_, ComplexI32>,
        destination: &mut SignalViewMut<'_, ComplexI32>,
    ) -> Result<()> {
        initialization::copy_i32_complex(stream_context, source, destination)
    }
}

impl<'a> CopySignal<f32> for SignalPipeline<'a, f32> {
    fn copy(
        stream_context: &StreamContext,
        source: &SignalView<'_, f32>,
        destination: &mut SignalViewMut<'_, f32>,
    ) -> Result<()> {
        initialization::copy_f32(stream_context, source, destination)
    }
}

impl<'a> CopySignal<f64> for SignalPipeline<'a, f64> {
    fn copy(
        _stream_context: &StreamContext,
        source: &SignalView<'_, f64>,
        destination: &mut SignalViewMut<'_, f64>,
    ) -> Result<()> {
        copy_device_to_device(source, destination)
    }
}

impl<'a> CopySignal<Complex32> for SignalPipeline<'a, Complex32> {
    fn copy(
        stream_context: &StreamContext,
        source: &SignalView<'_, Complex32>,
        destination: &mut SignalViewMut<'_, Complex32>,
    ) -> Result<()> {
        initialization::copy_f32_complex(stream_context, source, destination)
    }
}

impl<'a> CopySignal<Complex64> for SignalPipeline<'a, Complex64> {
    fn copy(
        stream_context: &StreamContext,
        source: &SignalView<'_, Complex64>,
        destination: &mut SignalViewMut<'_, Complex64>,
    ) -> Result<()> {
        initialization::copy_f64_complex(stream_context, source, destination)
    }
}

impl<'a> CopySignal<ComplexI64> for SignalPipeline<'a, ComplexI64> {
    fn copy(
        stream_context: &StreamContext,
        source: &SignalView<'_, ComplexI64>,
        destination: &mut SignalViewMut<'_, ComplexI64>,
    ) -> Result<()> {
        initialization::copy_i64_complex(stream_context, source, destination)
    }
}

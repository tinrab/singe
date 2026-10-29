use singe_cuda::{
    memory::transfer::copy_device,
    types::{Complex32, Complex64},
    view::DeviceRepr,
};

use crate::{
    context::StreamContext,
    error::Result,
    pipeline::SignalPipeline,
    signal::{
        initialization,
        view::{SignalOutputView, SignalView},
    },
    types::{ComplexI16, ComplexI32, ComplexI64},
};

/// # Safety
/// Uphold the completed native dispatch contract of
/// [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait CopySignal<T> {
    fn copy(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()>;
}

fn copy_device_to_device<T: DeviceRepr>(
    stream_context: &StreamContext,
    source: &SignalView<'_, T>,
    destination: &mut SignalOutputView<'_, T>,
) -> Result<()> {
    stream_context.stream().ensure_not_capturing()?;
    stream_context.prepare_region(source)?;
    stream_context.prepare_region(destination)?;
    stream_context.sync_scope(|| Ok(copy_device(destination, source)?))
}

unsafe impl CopySignal<i8> for SignalPipeline<'_, i8> {
    fn copy(
        stream_context: &StreamContext,
        source: &SignalView<'_, i8>,
        destination: &mut SignalOutputView<'_, i8>,
    ) -> Result<()> {
        copy_device_to_device(stream_context, source, destination)
    }
}

unsafe impl CopySignal<u16> for SignalPipeline<'_, u16> {
    fn copy(
        stream_context: &StreamContext,
        source: &SignalView<'_, u16>,
        destination: &mut SignalOutputView<'_, u16>,
    ) -> Result<()> {
        copy_device_to_device(stream_context, source, destination)
    }
}

unsafe impl CopySignal<u32> for SignalPipeline<'_, u32> {
    fn copy(
        stream_context: &StreamContext,
        source: &SignalView<'_, u32>,
        destination: &mut SignalOutputView<'_, u32>,
    ) -> Result<()> {
        copy_device_to_device(stream_context, source, destination)
    }
}

unsafe impl CopySignal<u8> for SignalPipeline<'_, u8> {
    fn copy(
        stream_context: &StreamContext,
        source: &SignalView<'_, u8>,
        destination: &mut SignalOutputView<'_, u8>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        stream_context
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            .sync_scope(|| unsafe { initialization::copy_u8(stream_context, source, destination) })
    }
}

unsafe impl CopySignal<i16> for SignalPipeline<'_, i16> {
    fn copy(
        stream_context: &StreamContext,
        source: &SignalView<'_, i16>,
        destination: &mut SignalOutputView<'_, i16>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        stream_context
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            .sync_scope(|| unsafe { initialization::copy_i16(stream_context, source, destination) })
    }
}

unsafe impl CopySignal<i32> for SignalPipeline<'_, i32> {
    fn copy(
        stream_context: &StreamContext,
        source: &SignalView<'_, i32>,
        destination: &mut SignalOutputView<'_, i32>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        stream_context
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            .sync_scope(|| unsafe { initialization::copy_i32(stream_context, source, destination) })
    }
}

unsafe impl CopySignal<i64> for SignalPipeline<'_, i64> {
    fn copy(
        stream_context: &StreamContext,
        source: &SignalView<'_, i64>,
        destination: &mut SignalOutputView<'_, i64>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        stream_context
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            .sync_scope(|| unsafe { initialization::copy_i64(stream_context, source, destination) })
    }
}

unsafe impl CopySignal<ComplexI16> for SignalPipeline<'_, ComplexI16> {
    fn copy(
        stream_context: &StreamContext,
        source: &SignalView<'_, ComplexI16>,
        destination: &mut SignalOutputView<'_, ComplexI16>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            initialization::copy_i16_complex(stream_context, source, destination)
        })
    }
}

unsafe impl CopySignal<ComplexI32> for SignalPipeline<'_, ComplexI32> {
    fn copy(
        stream_context: &StreamContext,
        source: &SignalView<'_, ComplexI32>,
        destination: &mut SignalOutputView<'_, ComplexI32>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            initialization::copy_i32_complex(stream_context, source, destination)
        })
    }
}

unsafe impl CopySignal<f32> for SignalPipeline<'_, f32> {
    fn copy(
        stream_context: &StreamContext,
        source: &SignalView<'_, f32>,
        destination: &mut SignalOutputView<'_, f32>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        stream_context
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            .sync_scope(|| unsafe { initialization::copy_f32(stream_context, source, destination) })
    }
}

unsafe impl CopySignal<f64> for SignalPipeline<'_, f64> {
    fn copy(
        stream_context: &StreamContext,
        source: &SignalView<'_, f64>,
        destination: &mut SignalOutputView<'_, f64>,
    ) -> Result<()> {
        copy_device_to_device(stream_context, source, destination)
    }
}

unsafe impl CopySignal<Complex32> for SignalPipeline<'_, Complex32> {
    fn copy(
        stream_context: &StreamContext,
        source: &SignalView<'_, Complex32>,
        destination: &mut SignalOutputView<'_, Complex32>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            initialization::copy_f32_complex(stream_context, source, destination)
        })
    }
}

unsafe impl CopySignal<Complex64> for SignalPipeline<'_, Complex64> {
    fn copy(
        stream_context: &StreamContext,
        source: &SignalView<'_, Complex64>,
        destination: &mut SignalOutputView<'_, Complex64>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            initialization::copy_f64_complex(stream_context, source, destination)
        })
    }
}

unsafe impl CopySignal<ComplexI64> for SignalPipeline<'_, ComplexI64> {
    fn copy(
        stream_context: &StreamContext,
        source: &SignalView<'_, ComplexI64>,
        destination: &mut SignalOutputView<'_, ComplexI64>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            initialization::copy_i64_complex(stream_context, source, destination)
        })
    }
}

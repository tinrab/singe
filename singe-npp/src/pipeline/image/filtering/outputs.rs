use std::{mem::MaybeUninit, sync::Arc};

use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory};

use crate::{
    error::{Error, Result},
    image::{memory::Image, view::C1},
    types::PointPolar,
};

/// Owned Hough output containing the initialized detected-line prefix and device count.
#[derive(Debug)]
pub struct HoughLines {
    /// Initialized detected lines; the complete original allocation remains retained.
    pub lines: DeviceMemory<PointPolar>,
    /// Device buffer containing one initialized detected-line count.
    pub line_count: DeviceMemory<i32>,
}

/// Write-only Hough capacity and count storage for native producers.
#[derive(Debug)]
pub struct HoughLinesOutput {
    /// Uninitialized device line capacity; only the produced prefix may be promoted.
    pub lines: DeviceMemory<MaybeUninit<PointPolar>>,
    /// One uninitialized device count to be written by the native producer.
    pub line_count: DeviceMemory<MaybeUninit<i32>>,
}

impl HoughLinesOutput {
    /// Allocates uninitialized line capacity and one uninitialized device count.
    ///
    /// # Errors
    ///
    /// Returns an error if `max_line_count` is zero or does not fit `i32`, allocation-size
    /// arithmetic fails, or either device allocation fails.
    pub fn create(context: &Arc<CudaContext>, max_line_count: usize) -> Result<Self> {
        if max_line_count == 0 || i32::try_from(max_line_count).is_err() {
            return Err(Error::out_of_range("max line count"));
        }
        Ok(Self {
            lines: DeviceMemory::<PointPolar>::create_uninit(context, max_line_count)?,
            line_count: DeviceMemory::<i32>::create_uninit(context, 1)?,
        })
    }

    /// Reads the completed device count and exposes only the initialized line prefix.
    ///
    /// Retains the complete original line allocation behind that prefix.
    ///
    /// # Errors
    ///
    /// Returns an error if the count buffer does not have exactly one element, copying the count
    /// to the host fails, the count is negative or exceeds line capacity, or constructing the
    /// initialized prefix fails.
    ///
    /// # Safety
    ///
    /// Production must have completed and initialized the device count and that many valid
    /// lines. The count must describe the prefix produced into this exact line allocation.
    pub unsafe fn assume_init(self) -> Result<HoughLines> {
        if self.line_count.len() != 1 {
            return Err(Error::LengthMismatch {
                name: "line count".into(),
                expected: 1,
                actual: self.line_count.len(),
            });
        }
        // SAFETY: the producer initialized the sole count and completed its write.
        let line_count = unsafe { self.line_count.assume_init() };
        let count = usize::try_from(line_count.copy_to_host_vec()?[0])
            .map_err(|_| Error::out_of_range("produced line count"))?;
        if count > self.lines.len() {
            return Err(Error::out_of_range("produced line count"));
        }
        Ok(HoughLines {
            // SAFETY: the count bounds the completed producer's initialized prefix.
            lines: unsafe { self.lines.assume_init_prefix(count)? },
            line_count,
        })
    }
}

/// Owned horizontal and vertical gradients, magnitude, and angle images.
#[derive(Debug)]
pub struct GradientVector<T> {
    /// Horizontal gradient samples.
    pub x: Image<T, C1>,
    /// Vertical gradient samples.
    pub y: Image<T, C1>,
    /// Gradient magnitude samples for the selected native norm.
    pub magnitude: Image<T, C1>,
    /// Gradient angle samples at each output pixel.
    pub angle: Image<f32, C1>,
}

/// Owned contiguous HOG descriptor values and their queried byte count.
#[derive(Debug)]
pub struct HistogramOfGradients {
    /// Contiguous `f32` descriptor windows in requested location order.
    pub descriptors: DeviceMemory<f32>,
    /// Required descriptor output capacity in bytes reported by the native size query.
    pub descriptor_bytes: usize,
}

use std::sync::Arc;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        memory::Image,
        view::{C1, C2},
    },
    pipeline::image::planar::{SemiplanarImage, SubsampledPlanarImage},
    types::Size,
};

impl<T> SubsampledPlanarImage<T> {
    pub(crate) fn prepare_eager(&self, stream_context: &StreamContext) -> Result<()> {
        for image in [&self.y, &self.cb, &self.cr] {
            stream_context
                .stream()
                .defer_drop_until_capture_end(Arc::clone(image.allocation()))?;
        }
        stream_context.stream().ensure_not_capturing()?;
        for image in [&self.y, &self.cb, &self.cr] {
            stream_context.prepare_region(image)?;
        }
        Ok(())
    }

    /// Returns a reference to the full-resolution first plane.
    #[must_use]
    pub fn y(&self) -> &Image<T, C1> {
        &self.y
    }

    /// Returns mutable access to the full-resolution first plane.
    /// Replacing the plane does not revalidate subsampling geometry or CUDA context.
    pub fn y_mut(&mut self) -> &mut Image<T, C1> {
        &mut self.y
    }

    /// Returns a reference to the first separately stored chroma plane.
    #[must_use]
    pub fn cb(&self) -> &Image<T, C1> {
        &self.cb
    }

    /// Returns mutable access to the first separately stored chroma plane.
    /// Replacing the plane does not revalidate subsampling geometry or CUDA context.
    pub fn cb_mut(&mut self) -> &mut Image<T, C1> {
        &mut self.cb
    }

    /// Returns a reference to the second separately stored chroma plane.
    #[must_use]
    pub fn cr(&self) -> &Image<T, C1> {
        &self.cr
    }

    /// Returns mutable access to the second separately stored chroma plane.
    /// Replacing the plane does not revalidate subsampling geometry or CUDA context.
    pub fn cr_mut(&mut self) -> &mut Image<T, C1> {
        &mut self.cr
    }

    /// Consumes the image and returns its owned planes in first-plane, Cb, Cr order.
    ///
    /// Does not copy samples or synchronize pending producers.
    #[must_use]
    pub fn into_planes(self) -> (Image<T, C1>, Image<T, C1>, Image<T, C1>) {
        (self.y, self.cb, self.cr)
    }
}

impl<T> SemiplanarImage<T> {
    pub(crate) fn prepare_eager(&self, stream_context: &StreamContext) -> Result<()> {
        stream_context
            .stream()
            .defer_drop_until_capture_end(Arc::clone(self.y.allocation()))?;
        stream_context
            .stream()
            .defer_drop_until_capture_end(Arc::clone(self.uv.allocation()))?;
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(&self.y)?;
        stream_context.prepare_region(&self.uv)?;
        Ok(())
    }

    /// Returns a reference to the full-resolution first plane.
    #[must_use]
    pub fn y(&self) -> &Image<T, C1> {
        &self.y
    }

    /// Returns mutable access to the full-resolution first plane.
    /// Replacing the plane does not revalidate subsampling geometry or CUDA context.
    pub fn y_mut(&mut self) -> &mut Image<T, C1> {
        &mut self.y
    }

    /// Returns a reference to the interleaved two-component chroma plane.
    #[must_use]
    pub fn uv(&self) -> &Image<T, C2> {
        &self.uv
    }

    /// Returns mutable access to the interleaved two-component chroma plane.
    /// Replacing the plane does not revalidate subsampling geometry or CUDA context.
    pub fn uv_mut(&mut self) -> &mut Image<T, C2> {
        &mut self.uv
    }

    /// Consumes the image and returns its owned planes in first-plane, interleaved-chroma order.
    ///
    /// Does not copy samples or synchronize pending producers.
    #[must_use]
    pub fn into_planes(self) -> (Image<T, C1>, Image<T, C2>) {
        (self.y, self.uv)
    }

    /// Returns the full-resolution first-plane dimensions.
    #[must_use]
    pub fn size(&self) -> Size {
        self.y.size()
    }

    /// Returns the interleaved chroma-plane dimensions.
    #[must_use]
    pub fn chroma_size(&self) -> Size {
        self.uv.size()
    }
}

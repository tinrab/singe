use std::{mem::MaybeUninit, sync::Arc};

use singe_cuda::context::Context;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::{
        Image, SupportedImage,
        video::{VideoFormat, VideoOutputView, VideoView, VideoViewMut, plane_dimensions},
        view::{C1, ImageViewMut},
    },
    types::Size,
};

/// Owned video planes whose stored samples are distinct from the frame's pixel dimensions.
/// Packed formats retain only their actual row bytes as readable samples; native row padding is excluded.
#[derive(Debug)]
pub struct Frame<T> {
    format: VideoFormat,
    size: Size,
    planes: Vec<Image<T, C1>>,
}

impl<T: SupportedImage<C1>> Frame<T> {
    /// Allocates exact stored-sample extents for each video plane without initializing samples.
    ///
    /// The logical frame uses `format` and pixel `size`; plane widths count stored `T` samples
    /// and exclude native pitch padding. Packed pixel groups may contain several pixels.
    ///
    /// # Errors
    ///
    /// Returns an error if the format, dimensions, sample size, or chroma divisibility is invalid,
    /// calculating plane sizes overflows, plane dimensions do not fit native integers,
    /// row bytes are not divisible by the sample size, or a plane allocation fails.
    pub fn create_uninit(
        context: &Arc<Context>,
        format: VideoFormat,
        size: Size,
    ) -> Result<Frame<MaybeUninit<T>>> {
        let dimensions = plane_dimensions::<T>(format, size)?;
        let mut planes = Vec::with_capacity(dimensions.len());
        for (row_bytes, height) in dimensions {
            if !row_bytes.is_multiple_of(size_of::<T>()) {
                return Err(Error::out_of_range("video sample size"));
            }
            let width = i32::try_from(row_bytes / size_of::<T>())
                .map_err(|_| Error::out_of_range("video plane width"))?;
            let height =
                i32::try_from(height).map_err(|_| Error::out_of_range("video plane height"))?;
            planes.push(Image::<T, C1>::create_uninit(
                context,
                Size::new(width, height),
            )?);
        }
        Ok(Frame {
            format,
            size,
            planes,
        })
    }
}

impl<T> Frame<T> {
    /// Adopts owned planes after validating the logical video format and pixel dimensions.
    ///
    /// Does not copy samples or synchronize pending producers. Planes may have larger stored
    /// extents than the minimum required by the logical frame.
    ///
    /// # Errors
    ///
    /// Returns an error if constructing an image view fails, frame geometry or sample-size
    /// requirements are invalid, the plane count or plane extents do not satisfy the format,
    /// a plane byte pitch is too small, or planes belong to different CUDA contexts.
    pub fn from_planes(format: VideoFormat, size: Size, planes: Vec<Image<T, C1>>) -> Result<Self> {
        let _ = VideoView::create(
            format,
            size,
            planes.iter().map(Image::view).collect::<Result<_>>()?,
        )?;
        Ok(Self {
            format,
            size,
            planes,
        })
    }

    /// Returns the CUDA context shared by the validated frame planes.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.planes[0].context()
    }

    /// Returns the logical frame width and height in pixels.
    #[must_use]
    pub fn size(&self) -> Size {
        self.size
    }

    /// Returns the video format that determines component packing and plane geometry.
    #[must_use]
    pub fn format(&self) -> VideoFormat {
        self.format
    }

    /// Returns the stored sample planes in the order required by the video format.
    #[must_use]
    pub fn planes(&self) -> &[Image<T, C1>] {
        &self.planes
    }

    /// Borrows one stored sample plane for native reads and writes.
    ///
    /// The index selects the format-specific stored plane, whose dimensions may differ from
    /// the logical frame dimensions. Does not synchronize pending producers.
    ///
    /// # Errors
    ///
    /// Returns an error if `index` is outside the plane list or constructing a mutable image
    /// view fails.
    pub fn plane_mut(&mut self, index: usize) -> Result<ImageViewMut<'_, T, C1>> {
        self.planes
            .get_mut(index)
            .ok_or_else(|| Error::out_of_range("video plane index"))?
            .view_mut()
    }

    /// Consumes the frame and returns its owned stored-sample planes.
    ///
    /// Does not copy samples or synchronize pending work.
    #[must_use]
    pub fn into_planes(self) -> Vec<Image<T, C1>> {
        self.planes
    }

    /// Borrows the stored planes as a readable video view.
    ///
    /// # Errors
    ///
    /// Returns an error if constructing an image view fails, frame geometry or sample-size
    /// requirements are invalid, the plane count or plane extents do not satisfy the format,
    /// a plane byte pitch is too small, or planes belong to different CUDA contexts.
    pub fn view(&self) -> Result<VideoView<'_, T>> {
        VideoView::create(
            self.format,
            self.size,
            self.planes.iter().map(Image::view).collect::<Result<_>>()?,
        )
    }

    /// Borrows initialized stored planes for video reads and writes.
    ///
    /// # Errors
    ///
    /// Returns an error if constructing an image view fails, frame geometry or sample-size
    /// requirements are invalid, the plane count or plane extents do not satisfy the format,
    /// a plane byte pitch is too small, or planes belong to different CUDA contexts.
    pub fn view_mut(&mut self) -> Result<VideoViewMut<'_, T>> {
        VideoViewMut::create(
            self.format,
            self.size,
            self.planes
                .iter_mut()
                .map(Image::view_mut)
                .collect::<Result<_>>()?,
        )
    }

    /// Borrows the stored planes as exclusive video output storage.
    ///
    /// # Errors
    ///
    /// Returns an error if constructing an image view fails, frame geometry or sample-size
    /// requirements are invalid, the plane count or plane extents do not satisfy the format,
    /// a plane byte pitch is too small, or planes belong to different CUDA contexts.
    pub fn output_view(&mut self) -> Result<VideoOutputView<'_, T>> {
        VideoOutputView::create(
            self.format,
            self.size,
            self.planes
                .iter_mut()
                .map(Image::output_view)
                .collect::<Result<_>>()?,
        )
    }

    pub(crate) fn prepare_eager(&self, context: &StreamContext) -> Result<()> {
        for plane in &self.planes {
            context
                .stream()
                .defer_drop_until_capture_end(Arc::clone(plane.allocation()))?;
        }
        context.stream().ensure_not_capturing()?;
        for plane in &self.planes {
            context.prepare_region(plane)?;
        }
        Ok(())
    }
}

impl<T> Frame<MaybeUninit<T>> {
    /// Borrows uninitialized stored planes as exclusive video output storage.
    ///
    /// Does not initialize samples or submit device work.
    ///
    /// # Errors
    ///
    /// Returns an error if constructing an image view fails, frame geometry or sample-size
    /// requirements are invalid, the plane count or plane extents do not satisfy the format,
    /// a plane byte pitch is too small, or planes belong to different CUDA contexts.
    pub fn output(&mut self) -> Result<VideoOutputView<'_, T>> {
        VideoOutputView::create(
            self.format,
            self.size,
            self.planes
                .iter_mut()
                .map(Image::output)
                .collect::<Result<_>>()?,
        )
    }

    /// Promotes the stored video planes to initialized storage without copying or synchronizing.
    ///
    /// # Safety
    ///
    /// Every stored sample in every plane must contain a valid `T`, and producing work must be
    /// complete. This includes every byte of packed pixel groups but excludes native pitch padding.
    #[must_use]
    pub unsafe fn assume_init(self) -> Frame<T> {
        Frame {
            format: self.format,
            size: self.size,
            planes: self
                .planes
                .into_iter()
                // SAFETY: the caller upholds this function's safety contract.
                .map(|plane| unsafe { plane.assume_init() })
                .collect(),
        }
    }
}

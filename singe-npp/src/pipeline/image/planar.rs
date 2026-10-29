use std::{mem::MaybeUninit, sync::Arc};

use singe_cuda::context::Context as CudaContext;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::{
        SupportedImage,
        memory::Image,
        view::{C1, C2, PlanarImageOutputView, PlanarImageView, PlanarImageViewMut},
    },
    types::Size,
};

/// An owned image with `C` separate full-resolution single-channel planes.
///
/// Native planar views require two through four planes with matching dimensions, byte strides,
/// and CUDA contexts. Plane samples may be `MaybeUninit<T>` until a producer initializes them.
#[derive(Debug)]
pub struct PlanarImage<T, const C: usize> {
    pub(crate) planes: [Image<T, C1>; C],
}

/// An owned full-resolution first plane and two separately stored chroma planes.
///
/// The producing color conversion determines the component order and subsampling factors.
#[derive(Debug)]
pub struct SubsampledPlanarImage<T> {
    pub(crate) y: Image<T, C1>,
    pub(crate) cb: Image<T, C1>,
    pub(crate) cr: Image<T, C1>,
}

/// An owned full-resolution first plane and one interleaved two-component chroma plane.
///
/// The producing color conversion determines chroma component order and subsampling.
#[derive(Debug)]
pub struct SemiplanarImage<T> {
    pub(crate) y: Image<T, C1>,
    pub(crate) uv: Image<T, C2>,
}

mod p4_color;
mod planar_color_methods;
mod planar_color_twist_methods;
mod planar_geometry_helpers;
mod planar_geometry_types;
mod planar_nv12_color_methods;
mod storage;

pub use planar_geometry_types::{
    PlanarRemap, PlanarResize, PlanarResizeSqrPixel, PlanarWarpAffine, PlanarWarpPerspective,
    PlanarWarpQuad,
};

impl<T, const C: usize> PlanarImage<T, C> {
    /// Adopts initialized planes after checking their geometry and actual CUDA contexts.
    ///
    /// Does not copy samples or synchronize pending producers.
    ///
    /// # Errors
    ///
    /// Returns an error if `C` is outside two through four, constructing a plane view fails,
    /// or the planes differ in dimensions, byte stride, or CUDA context.
    pub fn from_planes(planes: [Image<T, C1>; C]) -> Result<Self> {
        let image = Self { planes };
        image.view()?;
        Ok(image)
    }

    pub(crate) fn prepare_eager(&self, stream_context: &StreamContext) -> Result<()> {
        for image in &self.planes {
            stream_context
                .stream()
                .defer_drop_until_capture_end(Arc::clone(image.allocation()))?;
        }
        stream_context.stream().ensure_not_capturing()?;
        for image in &self.planes {
            stream_context.prepare_region(image)?;
        }
        Ok(())
    }

    /// Returns the owned plane array by reference, in its current component order.
    #[must_use]
    pub fn planes(&self) -> &[Image<T, C1>; C] {
        &self.planes
    }

    /// Returns mutable access to the owned plane array.
    ///
    /// Replacing a plane does not revalidate shared geometry or CUDA context.
    /// Subsequent planar-view construction performs those checks.
    pub fn planes_mut(&mut self) -> &mut [Image<T, C1>; C] {
        &mut self.planes
    }

    /// Consumes the image and returns its owned plane array without copying or synchronizing.
    #[must_use]
    pub fn into_planes(self) -> [Image<T, C1>; C] {
        self.planes
    }
}

impl<T, const C: usize> PlanarImage<T, C> {
    /// Allocates `C` uninitialized single-channel images with the requested `size`.
    ///
    /// Each plane uses `context`. This allocation path rejects zero planes; constructing a
    /// native planar view later additionally requires a plane count between two and four.
    ///
    /// # Errors
    ///
    /// Returns an error if `C` is zero, image dimensions or allocation-size arithmetic are invalid,
    /// or a device allocation fails.
    ///
    /// # Panics
    ///
    /// Panics if an internal plane slot is unexpectedly empty after successful construction.
    pub fn create_uninit(
        context: &Arc<CudaContext>,
        size: Size,
    ) -> Result<PlanarImage<MaybeUninit<T>, C>>
    where
        T: SupportedImage<C1>,
    {
        if C == 0 {
            return Err(Error::out_of_range("image plane count"));
        }
        let mut planes = [const { None }; C];
        for plane in &mut planes {
            *plane = Some(Image::<T, C1>::create_uninit(context, size)?);
        }
        Ok(PlanarImage {
            planes: planes.map(|plane| plane.expect("every plane was allocated")),
        })
    }

    /// Borrows the planes as a readable native planar view.
    ///
    /// # Errors
    ///
    /// Returns an error if constructing a plane view fails, `C` is outside two through four,
    /// or planes differ in dimensions, byte stride, or CUDA context.
    ///
    /// # Panics
    ///
    /// Panics if an internal plane slot is unexpectedly empty after successful construction.
    pub fn view(&self) -> Result<PlanarImageView<'_, T, C>> {
        let mut views = [const { None }; C];
        for (image, view) in self.planes.iter().zip(&mut views) {
            *view = Some(image.view()?);
        }
        PlanarImageView::create(views.map(|view| view.expect("every plane was borrowed")))
    }

    /// Borrows the initialized planes for native reads and writes.
    ///
    /// # Errors
    ///
    /// Returns an error if constructing a plane view fails, `C` is outside two through four,
    /// or planes differ in dimensions, byte stride, or CUDA context.
    ///
    /// # Panics
    ///
    /// Panics if an internal plane slot is unexpectedly empty after successful construction.
    pub fn view_mut(&mut self) -> Result<PlanarImageViewMut<'_, T, C>> {
        let mut views = [const { None }; C];
        for (image, view) in self.planes.iter_mut().zip(&mut views) {
            *view = Some(image.view_mut()?);
        }
        PlanarImageViewMut::create(views.map(|view| view.expect("every plane was borrowed")))
    }

    /// Borrows the planes as exclusive native output storage.
    ///
    /// # Errors
    ///
    /// Returns an error if constructing a plane output view fails, `C` is outside two through four,
    /// or planes differ in dimensions, byte stride, or CUDA context.
    ///
    /// # Panics
    ///
    /// Panics if an internal plane slot is unexpectedly empty after successful construction.
    pub fn output_view(&mut self) -> Result<PlanarImageOutputView<'_, T, C>> {
        let mut views = [const { None }; C];
        for (image, view) in self.planes.iter_mut().zip(&mut views) {
            *view = Some(image.output_view()?);
        }
        PlanarImageOutputView::create(views.map(|view| view.expect("every plane was borrowed")))
    }
}

impl<T, const C: usize> PlanarImage<MaybeUninit<T>, C> {
    /// Borrows the uninitialized planes as exclusive native output storage.
    ///
    /// Does not initialize any samples or submit device work.
    ///
    /// # Errors
    ///
    /// Returns an error if constructing a plane output view fails, `C` is outside two through four,
    /// or planes differ in dimensions, byte stride, or CUDA context.
    ///
    /// # Panics
    ///
    /// Panics if an internal plane slot is unexpectedly empty after successful construction.
    pub fn output(&mut self) -> Result<PlanarImageOutputView<'_, T, C>> {
        let mut views = [const { None }; C];
        for (image, view) in self.planes.iter_mut().zip(&mut views) {
            *view = Some(image.output()?);
        }
        PlanarImageOutputView::create(views.map(|view| view.expect("every plane was borrowed")))
    }

    /// Promotes the owned planes to initialized storage without copying or synchronizing.
    ///
    /// # Safety
    ///
    /// Every logical sample in every plane must contain a valid `T`, and producing work must
    /// be complete before this promotion. Row padding may remain uninitialized.
    #[must_use]
    pub unsafe fn assume_init(self) -> PlanarImage<T, C> {
        PlanarImage {
            // SAFETY: the caller upholds this function's safety contract.
            planes: self.planes.map(|plane| unsafe { plane.assume_init() }),
        }
    }
}

impl<T> SubsampledPlanarImage<MaybeUninit<T>> {
    /// Promotes the owned planes to initialized storage without copying or synchronizing.
    ///
    /// # Safety
    ///
    /// Every logical sample in every plane must contain a valid `T`, and producing work must
    /// be complete before this promotion. Row padding may remain uninitialized.
    #[must_use]
    pub unsafe fn assume_init(self) -> SubsampledPlanarImage<T> {
        SubsampledPlanarImage {
            // SAFETY: the caller upholds this function's safety contract.
            y: unsafe { self.y.assume_init() },
            // SAFETY: the caller upholds this function's safety contract.
            cb: unsafe { self.cb.assume_init() },
            // SAFETY: the caller upholds this function's safety contract.
            cr: unsafe { self.cr.assume_init() },
        }
    }
}

impl<T> SemiplanarImage<MaybeUninit<T>> {
    /// Promotes the owned planes to initialized storage without copying or synchronizing.
    ///
    /// # Safety
    ///
    /// Every logical sample in every plane must contain a valid `T`, and producing work must
    /// be complete before this promotion. Row padding may remain uninitialized.
    #[must_use]
    pub unsafe fn assume_init(self) -> SemiplanarImage<T> {
        SemiplanarImage {
            // SAFETY: the caller upholds this function's safety contract.
            y: unsafe { self.y.assume_init() },
            // SAFETY: the caller upholds this function's safety contract.
            uv: unsafe { self.uv.assume_init() },
        }
    }
}

/// Returns plane dimensions after exact horizontal and vertical subsampling.
///
/// # Errors
///
/// Returns an error if either full-resolution dimension or subsampling factor is nonpositive,
/// or a dimension is not divisible by its corresponding factor.
pub fn subsampled_size(
    size: Size,
    horizontal_subsampling: i32,
    vertical_subsampling: i32,
) -> Result<Size> {
    size.validate()?;
    if horizontal_subsampling <= 0 || vertical_subsampling <= 0 {
        return Err(Error::out_of_range("image subsampling"));
    }
    if size.width % horizontal_subsampling != 0 {
        return Err(Error::out_of_range("image width"));
    }
    if size.height % vertical_subsampling != 0 {
        return Err(Error::out_of_range("image height"));
    }

    Ok(Size {
        width: size.width / horizontal_subsampling,
        height: size.height / vertical_subsampling,
    })
}

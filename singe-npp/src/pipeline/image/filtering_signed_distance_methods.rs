use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{C1, ImageOutputView, ImageView},
    },
    pipeline::{
        ImageBacking,
        image::{ImagePipeline, filtering_distance_traits::SignedDistanceTransformPbaImage},
    },
};

mod antialiasing_methods;
mod into_methods;

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
{
    /// Computes a signed Euclidean distance image with the parallel banding algorithm.
    ///
    /// Accepts C1 `f32` or `f64` coverage images: values at or below zero describe exterior
    /// pixels, fractional values between zero and one describe coverage, and values at or
    /// above one describe fully covered interior pixels. Values below `cutoff_value` are
    /// classified as fully outside regions. Distances are generally negative outside and
    /// positive inside. Horizontal and vertical subpixel offsets shift the final distances;
    /// fully covered pixel centers are at `(0.5, 0.5)`.
    /// For `f32` output, converts the `f64` offsets to native `f32`; implementations with
    /// `f64` output retain `f64` offsets.
    /// Allocates temporary device scratch and retains it through native completion.
    /// NPP requires width and height in `64..=32767`; the wrapper checks positive
    /// dimensions but does not enforce that native range.
    /// Requests only the Euclidean distance image; optional Manhattan and Voronoi outputs
    /// are disabled.
    /// Allocates a new source-sized, owned C1 image of type `D`.
    /// NPP receives a mutable source pointer; source storage must be exclusively available
    /// during native use. Prepares image regions, waits for producers on other streams,
    /// rejects capture, and returns after successful stream completion. Writes every
    /// logical output sample; row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, workspace-size queries, scratch allocation, native execution, or
    /// stream synchronization fail. Source and destination dimensions must match and
    /// dimensions must be positive. NPP may reject the supplied transform parameters.
    /// Output allocation or image view construction can also fail.
    pub fn signed_distance_transform_pba<D>(
        self,
        cutoff_value: T,
        horizontal_subpixel_offset: f64,
        vertical_subpixel_offset: f64,
    ) -> Result<ImagePipeline<'a, D, C1>>
    where
        D: Copy + SupportedImage<C1>,
        Self: SignedDistanceTransformPbaImage<T, D>,
    {
        self.signed_distance_transform_pba_operation(
            cutoff_value,
            horizontal_subpixel_offset,
            vertical_subpixel_offset,
            <Self as SignedDistanceTransformPbaImage<T, D>>::signed_distance_transform_pba_image,
        )
    }

    /// Computes a signed Euclidean distance image with the parallel banding algorithm.
    ///
    /// Accepts C1 `f32` or `f64` coverage images: values at or below zero describe exterior
    /// pixels, fractional values between zero and one describe coverage, and values at or
    /// above one describe fully covered interior pixels. Values below `cutoff_value` are
    /// classified as fully outside regions. Distances are generally negative outside and
    /// positive inside. Horizontal and vertical subpixel offsets shift the final distances;
    /// fully covered pixel centers are at `(0.5, 0.5)`.
    /// For `f32` output, converts the `f64` offsets to native `f32`; implementations with
    /// `f64` output retain `f64` offsets.
    /// Allocates temporary device scratch and retains it through native completion.
    /// NPP requires width and height in `64..=32767`; the wrapper checks positive
    /// dimensions but does not enforce that native range.
    /// Uses the native absolute-Manhattan variant, but requests only the Euclidean distance
    /// image. The optional Manhattan and Voronoi outputs are disabled.
    /// Allocates a new source-sized, owned C1 image of type `D`.
    /// NPP receives a mutable source pointer; source storage must be exclusively available
    /// during native use. Prepares image regions, waits for producers on other streams,
    /// rejects capture, and returns after successful stream completion. Writes every
    /// logical output sample; row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, workspace-size queries, scratch allocation, native execution, or
    /// stream synchronization fail. Source and destination dimensions must match and
    /// dimensions must be positive. NPP may reject the supplied transform parameters.
    /// Output allocation or image view construction can also fail.
    pub fn signed_distance_transform_abs_pba<D>(
        self,
        cutoff_value: T,
        horizontal_subpixel_offset: f64,
        vertical_subpixel_offset: f64,
    ) -> Result<ImagePipeline<'a, D, C1>>
    where
        D: Copy + SupportedImage<C1>,
        Self: SignedDistanceTransformPbaImage<T, D>,
    {
        self.signed_distance_transform_pba_operation(
            cutoff_value,
            horizontal_subpixel_offset,
            vertical_subpixel_offset,
            <Self as SignedDistanceTransformPbaImage<T, D>>::signed_distance_transform_abs_pba_image,
        )
    }

    fn signed_distance_transform_pba_operation<D>(
        self,
        cutoff_value: T,
        horizontal_subpixel_offset: f64,
        vertical_subpixel_offset: f64,
        operation: SignedDistanceTransformPbaOperationCallback<D, T>,
    ) -> Result<ImagePipeline<'a, D, C1>>
    where
        D: Copy + SupportedImage<C1>,
    {
        self.ensure_eager()?;
        let mut destination =
            Image::<D, C1>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            operation(
                self.stream_context,
                &source,
                cutoff_value,
                horizontal_subpixel_offset,
                vertical_subpixel_offset,
                &mut destination_view,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}

type SignedDistanceTransformPbaOperationCallback<D, T> = fn(
    &StreamContext,
    &ImageView<'_, T, C1>,
    T,
    f64,
    f64,
    &mut ImageOutputView<'_, D, C1>,
) -> Result<()>;

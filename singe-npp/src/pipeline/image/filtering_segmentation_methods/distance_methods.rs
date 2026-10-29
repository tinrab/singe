use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{C1, ImageOutputView, ImageView},
    },
    pipeline::image::{
        ImageBacking, filtering_segmentation_methods::ImagePipeline,
        filtering_traits::DistanceTransformPbaImage,
    },
};

mod antialiasing_into_methods;

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
{
    /// Computes Euclidean distances to the nearest site with the parallel banding algorithm.
    ///
    /// Source values in the inclusive interval `minimum_site_value..=maximum_site_value`
    /// are sites. NPP requires at least one site and width and height in `64..=32767`; the
    /// wrapper checks positive dimensions but does not inspect pixels or validate that
    /// native range or the site interval. `u16` output truncates distances; floating output
    /// retains the native floating-point result.
    /// Requests only the Euclidean distance image; optional Manhattan and Voronoi outputs
    /// are disabled.
    /// Source and destination dimensions must match.
    /// Owns temporary device scratch through native completion. NPP receives a mutable
    /// source pointer; source storage must be exclusively available during native use.
    /// Writes every logical output sample; row padding remains unwritten.
    /// Prepares image regions and waits for producers on other streams. Rejects stream
    /// capture and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// Workspace queries and device allocation can fail. Source and destination dimensions
    /// must match, and dimensions must be positive.
    pub fn distance_transform_pba_into<D>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        minimum_site_value: T,
        maximum_site_value: T,
        destination: &mut ImageOutputView<'_, D, C1>,
    ) -> Result<()>
    where
        D: Copy,
        Self: DistanceTransformPbaImage<T, D>,
    {
        <Self as DistanceTransformPbaImage<T, D>>::distance_transform_pba_image(
            stream_context,
            source,
            minimum_site_value,
            maximum_site_value,
            destination,
        )
    }

    /// Computes Euclidean distances to the nearest site with the parallel banding algorithm.
    ///
    /// Source values in the inclusive interval `minimum_site_value..=maximum_site_value`
    /// are sites. NPP requires at least one site and width and height in `64..=32767`; the
    /// wrapper checks positive dimensions but does not inspect pixels or validate that
    /// native range or the site interval. `u16` output truncates distances; floating output
    /// retains the native floating-point result.
    /// Uses the native absolute-Manhattan variant, but requests only the Euclidean distance
    /// image. The optional Manhattan and Voronoi outputs are disabled.
    /// Source and destination dimensions must match.
    /// Owns temporary device scratch through native completion. NPP receives a mutable
    /// source pointer; source storage must be exclusively available during native use.
    /// Writes every logical output sample; row padding remains unwritten.
    /// Prepares image regions and waits for producers on other streams. Rejects stream
    /// capture and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// Workspace queries and device allocation can fail. Source and destination dimensions
    /// must match, and dimensions must be positive.
    pub fn distance_transform_abs_pba_into<D>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        minimum_site_value: T,
        maximum_site_value: T,
        destination: &mut ImageOutputView<'_, D, C1>,
    ) -> Result<()>
    where
        D: Copy,
        Self: DistanceTransformPbaImage<T, D>,
    {
        <Self as DistanceTransformPbaImage<T, D>>::distance_transform_abs_pba_image(
            stream_context,
            source,
            minimum_site_value,
            maximum_site_value,
            destination,
        )
    }

    /// Computes Euclidean distances to the nearest site with the parallel banding algorithm.
    ///
    /// Source values in the inclusive interval `minimum_site_value..=maximum_site_value`
    /// are sites. NPP requires at least one site and width and height in `64..=32767`; the
    /// wrapper checks positive dimensions but does not inspect pixels or validate that
    /// native range or the site interval. `u16` output truncates distances; floating output
    /// retains the native floating-point result.
    /// Requests only the Euclidean distance image; optional Manhattan and Voronoi outputs
    /// are disabled.
    /// Allocates a new source-sized, owned C1 image of type `D`.
    /// Owns temporary device scratch through native completion. NPP receives a mutable
    /// source pointer; source storage must be exclusively available during native use.
    /// Writes every logical output sample; row padding remains unwritten.
    /// Prepares image regions and waits for producers on other streams. Rejects stream
    /// capture and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// Workspace queries and device allocation can fail. Source and destination dimensions
    /// must match, and dimensions must be positive.
    /// Output allocation or image view construction can also fail.
    pub fn distance_transform_pba<D>(
        self,
        minimum_site_value: T,
        maximum_site_value: T,
    ) -> Result<ImagePipeline<'a, D, C1>>
    where
        D: Copy + SupportedImage<C1>,
        Self: DistanceTransformPbaImage<T, D>,
    {
        self.distance_transform_pba_operation(
            minimum_site_value,
            maximum_site_value,
            <Self as DistanceTransformPbaImage<T, D>>::distance_transform_pba_image,
        )
    }

    /// Computes Euclidean distances to the nearest site with the parallel banding algorithm.
    ///
    /// Source values in the inclusive interval `minimum_site_value..=maximum_site_value`
    /// are sites. NPP requires at least one site and width and height in `64..=32767`; the
    /// wrapper checks positive dimensions but does not inspect pixels or validate that
    /// native range or the site interval. `u16` output truncates distances; floating output
    /// retains the native floating-point result.
    /// Uses the native absolute-Manhattan variant, but requests only the Euclidean distance
    /// image. The optional Manhattan and Voronoi outputs are disabled.
    /// Allocates a new source-sized, owned C1 image of type `D`.
    /// Owns temporary device scratch through native completion. NPP receives a mutable
    /// source pointer; source storage must be exclusively available during native use.
    /// Writes every logical output sample; row padding remains unwritten.
    /// Prepares image regions and waits for producers on other streams. Rejects stream
    /// capture and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// Workspace queries and device allocation can fail. Source and destination dimensions
    /// must match, and dimensions must be positive.
    /// Output allocation or image view construction can also fail.
    pub fn distance_transform_abs_pba<D>(
        self,
        minimum_site_value: T,
        maximum_site_value: T,
    ) -> Result<ImagePipeline<'a, D, C1>>
    where
        D: Copy + SupportedImage<C1>,
        Self: DistanceTransformPbaImage<T, D>,
    {
        self.distance_transform_pba_operation(
            minimum_site_value,
            maximum_site_value,
            <Self as DistanceTransformPbaImage<T, D>>::distance_transform_abs_pba_image,
        )
    }

    fn distance_transform_pba_operation<D>(
        self,
        minimum_site_value: T,
        maximum_site_value: T,
        operation: DistanceTransformPbaOperationCallback<D, T>,
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
                minimum_site_value,
                maximum_site_value,
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

type DistanceTransformPbaOperationCallback<D, T> =
    fn(&StreamContext, &ImageView<'_, T, C1>, T, T, &mut ImageOutputView<'_, D, C1>) -> Result<()>;

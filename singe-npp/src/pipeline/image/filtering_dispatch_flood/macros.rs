macro_rules! impl_flood_fill_image {
    (
        $ty:ty,
        $layout:ty,
        $value_ty:ty,
        $fill:path,
        $boundary:path,
        $range:path,
        $range_boundary:path,
        $gradient:path,
        $gradient_boundary:path
    ) => {
        unsafe impl<'a> FloodFillImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            type Value = $value_ty;

            fn flood_fill_image(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                seed: Point,
                new_value: Self::Value,
                norm: ImageNormalization,
                connected_region: Option<&mut ConnectedRegion>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe {
                    $fill(
                        stream_context,
                        source_destination,
                        seed,
                        new_value,
                        norm,
                        connected_region,
                    )
                }
            }

            fn flood_fill_boundary_image(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                seed: Point,
                new_value: Self::Value,
                boundary_value: Self::Value,
                norm: ImageNormalization,
                connected_region: Option<&mut ConnectedRegion>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe {
                    $boundary(
                        stream_context,
                        source_destination,
                        seed,
                        new_value,
                        boundary_value,
                        norm,
                        connected_region,
                    )
                }
            }

            fn flood_fill_range_image(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                seed: Point,
                min: Self::Value,
                max: Self::Value,
                new_value: Self::Value,
                norm: ImageNormalization,
                connected_region: Option<&mut ConnectedRegion>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe {
                    $range(
                        stream_context,
                        source_destination,
                        seed,
                        min,
                        max,
                        new_value,
                        norm,
                        connected_region,
                    )
                }
            }

            fn flood_fill_range_boundary_image(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                seed: Point,
                min: Self::Value,
                max: Self::Value,
                new_value: Self::Value,
                boundary_value: Self::Value,
                norm: ImageNormalization,
                connected_region: Option<&mut ConnectedRegion>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe {
                    $range_boundary(
                        stream_context,
                        source_destination,
                        seed,
                        min,
                        max,
                        new_value,
                        boundary_value,
                        norm,
                        connected_region,
                    )
                }
            }

            fn flood_fill_gradient_image(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                seed: Point,
                min: Self::Value,
                max: Self::Value,
                new_value: Self::Value,
                norm: ImageNormalization,
                connected_region: Option<&mut ConnectedRegion>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe {
                    $gradient(
                        stream_context,
                        source_destination,
                        seed,
                        min,
                        max,
                        new_value,
                        norm,
                        connected_region,
                    )
                }
            }

            fn flood_fill_gradient_boundary_image(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                seed: Point,
                min: Self::Value,
                max: Self::Value,
                new_value: Self::Value,
                boundary_value: Self::Value,
                norm: ImageNormalization,
                connected_region: Option<&mut ConnectedRegion>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe {
                    $gradient_boundary(
                        stream_context,
                        source_destination,
                        seed,
                        min,
                        max,
                        new_value,
                        boundary_value,
                        norm,
                        connected_region,
                    )
                }
            }
        }
    };
}

macro_rules! impl_logical_constant_image {
    (
        $ty:ty,
        $layout:ty,
        $constant_ty:ty,
        $and:path,
        $and_in_place:path,
        $or:path,
        $or_in_place:path,
        $xor:path,
        $xor_in_place:path
    ) => {
        impl<'a> LogicalConstantImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            type Constant = $constant_ty;

            fn logical_and_constant_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                constant: Self::Constant,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $and(stream_context, source, constant, destination)
            }

            fn logical_and_constant_image_in_place(
                stream_context: &StreamContext,
                constant: Self::Constant,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $and_in_place(stream_context, constant, source_destination)
            }

            fn logical_or_constant_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                constant: Self::Constant,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $or(stream_context, source, constant, destination)
            }

            fn logical_or_constant_image_in_place(
                stream_context: &StreamContext,
                constant: Self::Constant,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $or_in_place(stream_context, constant, source_destination)
            }

            fn logical_xor_constant_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                constant: Self::Constant,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $xor(stream_context, source, constant, destination)
            }

            fn logical_xor_constant_image_in_place(
                stream_context: &StreamContext,
                constant: Self::Constant,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $xor_in_place(stream_context, constant, source_destination)
            }
        }
    };
}

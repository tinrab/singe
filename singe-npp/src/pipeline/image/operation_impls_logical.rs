macro_rules! impl_logical_not_image {
    ($layout:ty, $logical_not:path, $logical_not_in_place:path) => {
        impl<'a> LogicalNotImage<u8, $layout> for ImagePipeline<'a, u8, $layout> {
            fn logical_not_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, u8, $layout>,
                destination: &mut ImageViewMut<'_, u8, $layout>,
            ) -> Result<()> {
                $logical_not(stream_context, source, destination)
            }

            fn logical_not_image_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, u8, $layout>,
            ) -> Result<()> {
                $logical_not_in_place(stream_context, source_destination)
            }
        }
    };
}

macro_rules! impl_logical_binary_image {
    (
        $ty:ty,
        $layout:ty,
        $and:path,
        $and_in_place:path,
        $or:path,
        $or_in_place:path,
        $xor:path,
        $xor_in_place:path
    ) => {
        impl<'a> LogicalBinaryImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn logical_and_image(
                stream_context: &StreamContext,
                left: &ImageView<'_, $ty, $layout>,
                right: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $and(stream_context, left, right, destination)
            }

            fn logical_and_image_in_place(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $and_in_place(stream_context, source, source_destination)
            }

            fn logical_or_image(
                stream_context: &StreamContext,
                left: &ImageView<'_, $ty, $layout>,
                right: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $or(stream_context, left, right, destination)
            }

            fn logical_or_image_in_place(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $or_in_place(stream_context, source, source_destination)
            }

            fn logical_xor_image(
                stream_context: &StreamContext,
                left: &ImageView<'_, $ty, $layout>,
                right: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $xor(stream_context, left, right, destination)
            }

            fn logical_xor_image_in_place(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $xor_in_place(stream_context, source, source_destination)
            }
        }
    };
}

#[macro_use]
#[path = "operation_impls_logical_constant_macros.rs"]
mod logical_constant_macros;

#[path = "operation_impls_logical_constant.rs"]
mod logical_constant;

#[path = "operation_impls_logical_binary.rs"]
mod logical_binary;

#[path = "operation_impls_logical_not.rs"]
mod logical_not;

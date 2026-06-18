macro_rules! impl_scaled_constant_arithmetic_image {
    (
        $ty:ty,
        $layout:ty,
        $constant_ty:ty,
        $add:path,
        $add_in_place:path,
        $subtract:path,
        $subtract_in_place:path,
        $multiply:path,
        $multiply_in_place:path,
        $divide:path,
        $divide_in_place:path
    ) => {
        impl<'a> ConstantArithmeticImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            type Constant = $constant_ty;

            fn add_constant_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                constant: Self::Constant,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                $add(stream_context, source, constant, destination, scale_factor)
            }

            fn add_constant_image_in_place(
                stream_context: &StreamContext,
                constant: Self::Constant,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                $add_in_place(stream_context, constant, source_destination, scale_factor)
            }

            fn subtract_constant_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                constant: Self::Constant,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                $subtract(stream_context, source, constant, destination, scale_factor)
            }

            fn subtract_constant_image_in_place(
                stream_context: &StreamContext,
                constant: Self::Constant,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                $subtract_in_place(stream_context, constant, source_destination, scale_factor)
            }

            fn multiply_constant_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                constant: Self::Constant,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                $multiply(stream_context, source, constant, destination, scale_factor)
            }

            fn multiply_constant_image_in_place(
                stream_context: &StreamContext,
                constant: Self::Constant,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                $multiply_in_place(stream_context, constant, source_destination, scale_factor)
            }

            fn divide_constant_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                constant: Self::Constant,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                $divide(stream_context, source, constant, destination, scale_factor)
            }

            fn divide_constant_image_in_place(
                stream_context: &StreamContext,
                constant: Self::Constant,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                $divide_in_place(stream_context, constant, source_destination, scale_factor)
            }
        }
    };
}

macro_rules! impl_unscaled_constant_arithmetic_image {
    (
        $ty:ty,
        $layout:ty,
        $constant_ty:ty,
        $add:path,
        $add_in_place:path,
        $subtract:path,
        $subtract_in_place:path,
        $multiply:path,
        $multiply_in_place:path,
        $divide:path,
        $divide_in_place:path
    ) => {
        impl<'a> ConstantArithmeticImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            type Constant = $constant_ty;

            fn add_constant_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                constant: Self::Constant,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $add(stream_context, source, constant, destination)
            }

            fn add_constant_image_in_place(
                stream_context: &StreamContext,
                constant: Self::Constant,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $add_in_place(stream_context, constant, source_destination)
            }

            fn subtract_constant_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                constant: Self::Constant,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $subtract(stream_context, source, constant, destination)
            }

            fn subtract_constant_image_in_place(
                stream_context: &StreamContext,
                constant: Self::Constant,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $subtract_in_place(stream_context, constant, source_destination)
            }

            fn multiply_constant_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                constant: Self::Constant,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $multiply(stream_context, source, constant, destination)
            }

            fn multiply_constant_image_in_place(
                stream_context: &StreamContext,
                constant: Self::Constant,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $multiply_in_place(stream_context, constant, source_destination)
            }

            fn divide_constant_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                constant: Self::Constant,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $divide(stream_context, source, constant, destination)
            }

            fn divide_constant_image_in_place(
                stream_context: &StreamContext,
                constant: Self::Constant,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $divide_in_place(stream_context, constant, source_destination)
            }
        }
    };
}

macro_rules! impl_float_device_constant_arithmetic_image {
    (
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
        impl<'a> DeviceConstantArithmeticImage<f32, $layout> for ImagePipeline<'a, f32, $layout> {
            type Constant = $constant_ty;

            fn add_device_constant_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, f32, $layout>,
                constant: &DeviceMemory<Self::Constant>,
                destination: &mut ImageViewMut<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $add(stream_context, source, constant, destination)
            }

            fn add_device_constant_image_in_place(
                stream_context: &StreamContext,
                constant: &DeviceMemory<Self::Constant>,
                source_destination: &mut ImageViewMut<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $add_in_place(stream_context, constant, source_destination)
            }

            fn subtract_device_constant_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, f32, $layout>,
                constant: &DeviceMemory<Self::Constant>,
                destination: &mut ImageViewMut<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $subtract(stream_context, source, constant, destination)
            }

            fn subtract_device_constant_image_in_place(
                stream_context: &StreamContext,
                constant: &DeviceMemory<Self::Constant>,
                source_destination: &mut ImageViewMut<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $subtract_in_place(stream_context, constant, source_destination)
            }

            fn multiply_device_constant_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, f32, $layout>,
                constant: &DeviceMemory<Self::Constant>,
                destination: &mut ImageViewMut<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $multiply(stream_context, source, constant, destination)
            }

            fn multiply_device_constant_image_in_place(
                stream_context: &StreamContext,
                constant: &DeviceMemory<Self::Constant>,
                source_destination: &mut ImageViewMut<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $multiply_in_place(stream_context, constant, source_destination)
            }

            fn divide_device_constant_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, f32, $layout>,
                constant: &DeviceMemory<Self::Constant>,
                destination: &mut ImageViewMut<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $divide(stream_context, source, constant, destination)
            }

            fn divide_device_constant_image_in_place(
                stream_context: &StreamContext,
                constant: &DeviceMemory<Self::Constant>,
                source_destination: &mut ImageViewMut<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $divide_in_place(stream_context, constant, source_destination)
            }
        }
    };
}

macro_rules! impl_unscaled_device_constant_arithmetic_image {
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
        impl<'a> DeviceConstantArithmeticImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            type Constant = $constant_ty;

            fn add_device_constant_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                constant: &DeviceMemory<Self::Constant>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $add(stream_context, source, constant, destination)
            }

            fn add_device_constant_image_in_place(
                stream_context: &StreamContext,
                constant: &DeviceMemory<Self::Constant>,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $add_in_place(stream_context, constant, source_destination)
            }

            fn subtract_device_constant_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                constant: &DeviceMemory<Self::Constant>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $subtract(stream_context, source, constant, destination)
            }

            fn subtract_device_constant_image_in_place(
                stream_context: &StreamContext,
                constant: &DeviceMemory<Self::Constant>,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $subtract_in_place(stream_context, constant, source_destination)
            }

            fn multiply_device_constant_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                constant: &DeviceMemory<Self::Constant>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $multiply(stream_context, source, constant, destination)
            }

            fn multiply_device_constant_image_in_place(
                stream_context: &StreamContext,
                constant: &DeviceMemory<Self::Constant>,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $multiply_in_place(stream_context, constant, source_destination)
            }

            fn divide_device_constant_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                constant: &DeviceMemory<Self::Constant>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $divide(stream_context, source, constant, destination)
            }

            fn divide_device_constant_image_in_place(
                stream_context: &StreamContext,
                constant: &DeviceMemory<Self::Constant>,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $divide_in_place(stream_context, constant, source_destination)
            }
        }
    };
}

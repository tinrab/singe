macro_rules! impl_scaled_device_constant_arithmetic_image {
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
                scale_factor: i32,
            ) -> Result<()> {
                $add(stream_context, source, constant, destination, scale_factor)
            }

            fn add_device_constant_image_in_place(
                stream_context: &StreamContext,
                constant: &DeviceMemory<Self::Constant>,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                $add_in_place(stream_context, constant, source_destination, scale_factor)
            }

            fn subtract_device_constant_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                constant: &DeviceMemory<Self::Constant>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                $subtract(stream_context, source, constant, destination, scale_factor)
            }

            fn subtract_device_constant_image_in_place(
                stream_context: &StreamContext,
                constant: &DeviceMemory<Self::Constant>,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                $subtract_in_place(stream_context, constant, source_destination, scale_factor)
            }

            fn multiply_device_constant_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                constant: &DeviceMemory<Self::Constant>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                $multiply(stream_context, source, constant, destination, scale_factor)
            }

            fn multiply_device_constant_image_in_place(
                stream_context: &StreamContext,
                constant: &DeviceMemory<Self::Constant>,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                $multiply_in_place(stream_context, constant, source_destination, scale_factor)
            }

            fn divide_device_constant_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                constant: &DeviceMemory<Self::Constant>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                $divide(stream_context, source, constant, destination, scale_factor)
            }

            fn divide_device_constant_image_in_place(
                stream_context: &StreamContext,
                constant: &DeviceMemory<Self::Constant>,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                $divide_in_place(stream_context, constant, source_destination, scale_factor)
            }
        }
    };
}

#[path = "operation_impls_device_constant_arithmetic_i16.rs"]
mod i16_impls;
#[path = "operation_impls_device_constant_arithmetic_i32.rs"]
mod i32_impls;
#[path = "operation_impls_device_constant_arithmetic_u16.rs"]
mod u16_impls;
#[path = "operation_impls_device_constant_arithmetic_u8.rs"]
mod u8_impls;

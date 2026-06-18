macro_rules! impl_color_twist {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            twist: ColorTwistMatrix,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    twist.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_color_twist {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                twist: ColorTwistMatrix,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T, $layout>,
            twist: ColorTwistMatrix,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source, destination, twist)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    twist: ColorTwistMatrix,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, twist)
                }
            }
        )*
    };
}

macro_rules! impl_color_twist_batch {
    ($name:ident, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            sources: &[ImageView<'_, u8, $layout>],
            twists: &[ColorTwistMatrix],
            destinations: &mut [ImageViewMut<'_, u8, $layout>],
            min: f32,
            max: f32,
        ) -> Result<()> {
            let (batch_size, roi, batch_memory, _twist_memory) =
                color_twist_batch_descriptors(sources, twists, destinations)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    min,
                    max,
                    roi.into(),
                    batch_memory.as_ptr().cast_mut().cast(),
                    batch_size,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_color_twist_batch_in_place {
    ($name:ident, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            images: &mut [ImageViewMut<'_, u8, $layout>],
            twists: &[ColorTwistMatrix],
            min: f32,
            max: f32,
        ) -> Result<()> {
            let (batch_size, roi, batch_memory, _twist_memory) =
                color_twist_batch_in_place_descriptors(images, twists)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    min,
                    max,
                    roi.into(),
                    batch_memory.as_ptr().cast_mut().cast(),
                    batch_size,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_color_twist_batch_typed {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            sources: &[ImageView<'_, $ty, $layout>],
            twists: &[ColorTwistMatrix],
            destinations: &mut [ImageViewMut<'_, $ty, $layout>],
            min: f32,
            max: f32,
        ) -> Result<()> {
            let (batch_size, roi, batch_memory, _twist_memory) =
                color_twist_batch_descriptors(sources, twists, destinations)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    min,
                    max,
                    roi.into(),
                    batch_memory.as_ptr().cast_mut().cast(),
                    batch_size,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_color_twist_batch_in_place_typed {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            images: &mut [ImageViewMut<'_, $ty, $layout>],
            twists: &[ColorTwistMatrix],
            min: f32,
            max: f32,
        ) -> Result<()> {
            let (batch_size, roi, batch_memory, _twist_memory) =
                color_twist_batch_in_place_descriptors(images, twists)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    min,
                    max,
                    roi.into(),
                    batch_memory.as_ptr().cast_mut().cast(),
                    batch_size,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_color_twist_batch_with_constants_typed {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            sources: &[ImageView<'_, $ty, C4>],
            twists: &[ColorTwistBatchConstantsMatrix4],
            destinations: &mut [ImageViewMut<'_, $ty, C4>],
            min: f32,
            max: f32,
        ) -> Result<()> {
            let (batch_size, roi, batch_memory, _twist_memory) =
                color_twist_batch_with_constants_descriptors(sources, twists, destinations)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    min,
                    max,
                    roi.into(),
                    batch_memory.as_ptr().cast_mut().cast(),
                    batch_size,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_color_twist_batch_with_constants_in_place_typed {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            images: &mut [ImageViewMut<'_, $ty, C4>],
            twists: &[ColorTwistBatchConstantsMatrix4],
            min: f32,
            max: f32,
        ) -> Result<()> {
            let (batch_size, roi, batch_memory, _twist_memory) =
                color_twist_batch_with_constants_in_place_descriptors(images, twists)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    min,
                    max,
                    roi.into(),
                    batch_memory.as_ptr().cast_mut().cast(),
                    batch_size,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_color_twist_different_layout {
    ($name:ident, $ty:ty, $source_layout:ty, $destination_layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $source_layout>,
            destination: &mut ImageViewMut<'_, $ty, $destination_layout>,
            twist: ColorTwistMatrix,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    twist.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_color_twist_in_place {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, $ty, $layout>,
            twist: ColorTwistMatrix,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    image.as_mut_ptr().cast(),
                    image.step(),
                    image.size().into(),
                    twist.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_color_twist_in_place {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, Layout>,
                twist: ColorTwistMatrix,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, T, $layout>,
            twist: ColorTwistMatrix,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, image, twist)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    image: &mut ImageViewMut<'_, Self, $layout>,
                    twist: ColorTwistMatrix,
                ) -> Result<()> {
                    $direct(stream_context, image, twist)
                }
            }
        )*
    };
}

macro_rules! impl_color_twist_with_constants {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C4>,
            destination: &mut ImageViewMut<'_, $ty, C4>,
            twist: ColorTwistMatrix4,
            constants: ColorTwistConstants4,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    twist.as_ptr().cast(),
                    constants.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_color_twist_with_constants {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C4>,
                destination: &mut ImageViewMut<'_, Self, C4>,
                twist: ColorTwistMatrix4,
                constants: ColorTwistConstants4,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C4>,
            destination: &mut ImageViewMut<'_, T, C4>,
            twist: ColorTwistMatrix4,
            constants: ColorTwistConstants4,
        ) -> Result<()>
        where
            T: $trait,
        {
            T::$method(stream_context, source, destination, twist, constants)
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C4>,
                    destination: &mut ImageViewMut<'_, Self, C4>,
                    twist: ColorTwistMatrix4,
                    constants: ColorTwistConstants4,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, twist, constants)
                }
            }
        )*
    };
}

macro_rules! impl_color_twist_with_constants_in_place {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, $ty, C4>,
            twist: ColorTwistMatrix4,
            constants: ColorTwistConstants4,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    image.as_mut_ptr().cast(),
                    image.step(),
                    image.size().into(),
                    twist.as_ptr().cast(),
                    constants.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_color_twist_with_constants_in_place {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, C4>,
                twist: ColorTwistMatrix4,
                constants: ColorTwistConstants4,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, T, C4>,
            twist: ColorTwistMatrix4,
            constants: ColorTwistConstants4,
        ) -> Result<()>
        where
            T: $trait,
        {
            T::$method(stream_context, image, twist, constants)
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    image: &mut ImageViewMut<'_, Self, C4>,
                    twist: ColorTwistMatrix4,
                    constants: ColorTwistConstants4,
                ) -> Result<()> {
                    $direct(stream_context, image, twist, constants)
                }
            }
        )*
    };
}

macro_rules! impl_color_twist_planar {
    ($name:ident, $ty:ty, $planes:literal, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &PlanarImageView<'_, $ty, $planes>,
            destination: &mut PlanarImageViewMut<'_, $ty, $planes>,
            twist: ColorTwistMatrix,
        ) -> Result<()> {
            validate_same_size(source.planes()[0].size(), destination.planes()[0].size())?;

            let source_planes = source.planes().map(|plane| plane.as_ptr());
            let destination_planes = destination
                .planes_mut()
                .each_mut()
                .map(ImageViewMut::as_mut_ptr);

            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_ptr(),
                    source.planes()[0].step(),
                    destination_planes.as_ptr(),
                    destination.planes()[0].step(),
                    source.planes()[0].size().into(),
                    twist.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_color_twist_planar_in_place {
    ($name:ident, $ty:ty, $planes:literal, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            image: &mut PlanarImageViewMut<'_, $ty, $planes>,
            twist: ColorTwistMatrix,
        ) -> Result<()> {
            let step = image.planes()[0].step();
            let size = image.planes()[0].size();
            let planes = image.planes_mut().each_mut().map(ImageViewMut::as_mut_ptr);

            unsafe {
                try_ffi!(sys::$ffi(
                    planes.as_ptr().cast(),
                    step,
                    size.into(),
                    twist.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_gamma_packed {
    ($name:ident, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, u8, $layout>,
            destination: &mut ImageViewMut<'_, u8, $layout>,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_gamma {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source, destination)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                ) -> Result<()> {
                    $direct(stream_context, source, destination)
                }
            }
        )*
    };
}

macro_rules! impl_gamma_packed_in_place {
    ($name:ident, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, u8, $layout>,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    image.as_mut_ptr().cast(),
                    image.step(),
                    image.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_gamma_in_place {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, Layout>,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, image)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    image: &mut ImageViewMut<'_, Self, $layout>,
                ) -> Result<()> {
                    $direct(stream_context, image)
                }
            }
        )*
    };
}

macro_rules! impl_gamma_planar {
    ($name:ident, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &PlanarImageView<'_, u8, 3>,
            destination: &mut PlanarImageViewMut<'_, u8, 3>,
        ) -> Result<()> {
            validate_same_size(source.planes()[0].size(), destination.planes()[0].size())?;

            let source_planes = source.planes().map(|plane| plane.as_ptr());
            let mut destination_planes = destination
                .planes_mut()
                .each_mut()
                .map(ImageViewMut::as_mut_ptr);

            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_ptr(),
                    source.planes()[0].step(),
                    destination_planes.as_mut_ptr(),
                    destination.planes()[0].step(),
                    source.planes()[0].size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_gamma_planar_in_place {
    ($name:ident, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            image: &mut PlanarImageViewMut<'_, u8, 3>,
        ) -> Result<()> {
            let step = image.planes()[0].step();
            let size = image.planes()[0].size();
            let planes = image.planes_mut().each_mut().map(ImageViewMut::as_mut_ptr);

            unsafe {
                try_ffi!(sys::$ffi(
                    planes.as_ptr().cast(),
                    step,
                    size.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

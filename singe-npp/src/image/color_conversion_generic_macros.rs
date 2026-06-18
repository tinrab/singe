macro_rules! impl_color_convert_same_layout {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
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

macro_rules! impl_generic_color_convert_same_layout {
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

macro_rules! impl_color_convert_different_layout {
    ($name:ident, $ty:ty, $source_layout:ty, $destination_layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $source_layout>,
            destination: &mut ImageViewMut<'_, $ty, $destination_layout>,
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

macro_rules! impl_color_convert_different_layout_constant_alpha {
    ($name:ident, $ty:ty, $source_layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $source_layout>,
            destination: &mut ImageViewMut<'_, $ty, C4>,
            alpha: u8,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    alpha,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_color_convert_batch_same_layout {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            sources: &[ImageView<'_, $ty, $layout>],
            destinations: &mut [ImageViewMut<'_, $ty, $layout>],
        ) -> Result<()> {
            let (batch_size, roi, source_descriptors, destination_descriptors) =
                color_batch_descriptors(sources, destinations)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source_descriptors.as_ptr().cast(),
                    destination_descriptors.as_mut_ptr().cast(),
                    batch_size,
                    roi.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_color_convert_batch_same_layout_advanced {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            sources: &[ImageView<'_, $ty, $layout>],
            destinations: &mut [ImageViewMut<'_, $ty, $layout>],
        ) -> Result<()> {
            let (batch_size, max_roi, source_descriptors, destination_descriptors) =
                color_batch_advanced_descriptors(sources, destinations)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source_descriptors.as_ptr().cast(),
                    destination_descriptors.as_mut_ptr().cast(),
                    batch_size,
                    max_roi.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_color_convert_batch_planar_to_packed {
    ($name:ident, $ty:ty, $planes:literal, $packed_layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            sources: &[PlanarImageView<'_, $ty, $planes>],
            destinations: &mut [ImageViewMut<'_, $ty, $packed_layout>],
        ) -> Result<()> {
            let (batch_size, roi, source_batch_list, destination_descriptors, _source_descriptors) =
                color_planar_batch_descriptors(sources, destinations)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source_batch_list.as_ptr().cast(),
                    destination_descriptors.as_mut_ptr().cast(),
                    batch_size,
                    roi.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_color_convert_batch_planar_to_packed_advanced {
    ($name:ident, $ty:ty, $planes:literal, $packed_layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            sources: &[PlanarImageView<'_, $ty, $planes>],
            destinations: &mut [ImageViewMut<'_, $ty, $packed_layout>],
        ) -> Result<()> {
            let (
                batch_size,
                max_roi,
                source_batch_list,
                destination_descriptors,
                _source_descriptors,
            ) = color_planar_batch_advanced_descriptors(sources, destinations)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source_batch_list.as_ptr().cast(),
                    destination_descriptors.as_mut_ptr().cast(),
                    batch_size,
                    max_roi.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_subsampled_color_convert_batch_planar_to_packed {
    ($name:ident, $ty:ty, $packed_layout:ty, $ffi:ident, $horizontal_subsampling:literal, $vertical_subsampling:literal) => {
        pub fn $name(
            stream_context: &StreamContext,
            source_plane_0: &[ImageView<'_, $ty, C1>],
            source_plane_1: &[ImageView<'_, $ty, C1>],
            source_plane_2: &[ImageView<'_, $ty, C1>],
            destinations: &mut [ImageViewMut<'_, $ty, $packed_layout>],
        ) -> Result<()> {
            let (batch_size, roi, source_batch_list, destination_descriptors, _source_descriptors) =
                subsampled_color_batch_descriptors(
                    source_plane_0,
                    source_plane_1,
                    source_plane_2,
                    destinations,
                    $horizontal_subsampling,
                    $vertical_subsampling,
                )?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source_batch_list.as_ptr().cast(),
                    destination_descriptors.as_mut_ptr().cast(),
                    batch_size,
                    roi.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_subsampled_color_convert_batch_planar_to_packed_advanced {
    ($name:ident, $ty:ty, $packed_layout:ty, $ffi:ident, $horizontal_subsampling:literal, $vertical_subsampling:literal) => {
        pub fn $name(
            stream_context: &StreamContext,
            source_plane_0: &[ImageView<'_, $ty, C1>],
            source_plane_1: &[ImageView<'_, $ty, C1>],
            source_plane_2: &[ImageView<'_, $ty, C1>],
            destinations: &mut [ImageViewMut<'_, $ty, $packed_layout>],
        ) -> Result<()> {
            let (
                batch_size,
                max_roi,
                source_batch_list,
                destination_descriptors,
                _source_descriptors,
            ) = subsampled_color_batch_advanced_descriptors(
                source_plane_0,
                source_plane_1,
                source_plane_2,
                destinations,
                $horizontal_subsampling,
                $vertical_subsampling,
            )?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source_batch_list.as_ptr().cast(),
                    destination_descriptors.as_mut_ptr().cast(),
                    batch_size,
                    max_roi.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_color_convert_packed_to_planar {
    ($name:ident, $ty:ty, $packed_layout:ty, $planes:literal, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $packed_layout>,
            destination: &mut PlanarImageViewMut<'_, $ty, $planes>,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.planes()[0].size())?;

            let mut destination_planes = destination
                .planes_mut()
                .each_mut()
                .map(ImageViewMut::as_mut_ptr);

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination_planes.as_mut_ptr(),
                    destination.planes()[0].step(),
                    source.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_color_convert_planar_to_packed {
    ($name:ident, $ty:ty, $planes:literal, $packed_layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &PlanarImageView<'_, $ty, $planes>,
            destination: &mut ImageViewMut<'_, $ty, $packed_layout>,
        ) -> Result<()> {
            validate_same_size(source.planes()[0].size(), destination.size())?;

            let source_planes = source.planes().map(|plane| plane.as_ptr());

            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_ptr().cast_mut(),
                    source.planes()[0].step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_color_convert_planar_to_packed_constant_alpha {
    ($name:ident, $ty:ty, $planes:literal, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &PlanarImageView<'_, $ty, $planes>,
            destination: &mut ImageViewMut<'_, $ty, C4>,
            alpha: u8,
        ) -> Result<()> {
            validate_same_size(source.planes()[0].size(), destination.size())?;

            let source_planes = source.planes().map(|plane| plane.as_ptr());

            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_ptr().cast_mut(),
                    source.planes()[0].step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.planes()[0].size().into(),
                    alpha,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_color_convert_planar_to_planar {
    ($name:ident, $ty:ty, $planes:literal, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &PlanarImageView<'_, $ty, $planes>,
            destination: &mut PlanarImageViewMut<'_, $ty, $planes>,
        ) -> Result<()> {
            validate_same_size(source.planes()[0].size(), destination.planes()[0].size())?;

            let source_planes = source.planes().map(|plane| plane.as_ptr());
            let mut destination_planes = destination
                .planes_mut()
                .each_mut()
                .map(ImageViewMut::as_mut_ptr);

            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_ptr().cast_mut(),
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

macro_rules! impl_color_convert_planar_to_planar_different_planes {
    ($name:ident, $ty:ty, $source_planes:literal, $destination_planes:literal, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &PlanarImageView<'_, $ty, $source_planes>,
            destination: &mut PlanarImageViewMut<'_, $ty, $destination_planes>,
        ) -> Result<()> {
            validate_same_size(source.planes()[0].size(), destination.planes()[0].size())?;

            let source_planes = source.planes().map(|plane| plane.as_ptr());
            let destination_planes = destination
                .planes_mut()
                .each_mut()
                .map(ImageViewMut::as_mut_ptr);

            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_ptr().cast_mut(),
                    source.planes()[0].step(),
                    destination_planes.as_ptr().cast_mut(),
                    destination.planes()[0].step(),
                    source.planes()[0].size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_mirror {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            axis: Axis,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    axis.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_mirror_in_place {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            image: &mut ImageViewMut<'_, $ty, $layout>,
            axis: Axis,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    image.as_mut_ptr().cast(),
                    image.step(),
                    image.size().into(),
                    axis.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_mirror_batch {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            axis: Axis,
            sources: &[ImageView<'_, $ty, $layout>],
            destinations: &mut [ImageViewMut<'_, $ty, $layout>],
        ) -> Result<()> {
            let (batch_size, roi, descriptors) = mirror_batch_descriptors(sources, destinations)?;
            let batch_size = i32::try_from(batch_size).map_err(|_| Error::OutOfRange {
                name: "batch size".into(),
            })?;

            unsafe {
                try_ffi!(sys::$ffi(
                    roi.into(),
                    axis.into(),
                    descriptors.as_mut_ptr().cast(),
                    batch_size,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_mirror_batch_in_place {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            axis: Axis,
            images: &mut [ImageViewMut<'_, $ty, $layout>],
        ) -> Result<()> {
            let (batch_size, roi, descriptors) = mirror_batch_in_place_descriptors(images)?;
            let batch_size = i32::try_from(batch_size).map_err(|_| Error::OutOfRange {
                name: "batch size".into(),
            })?;

            unsafe {
                try_ffi!(sys::$ffi(
                    roi.into(),
                    axis.into(),
                    descriptors.as_mut_ptr().cast(),
                    batch_size,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_resize_sqr_pixel {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            resize: &ResizeSqrPixel,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.size().into(),
                    source.step(),
                    resize.source_roi.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    resize.destination_roi.into(),
                    resize.x_factor,
                    resize.y_factor,
                    resize.x_shift,
                    resize.y_shift,
                    i32::from(resize.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_resize_sqr_pixel_planar {
    ($name:ident, $ty:ty, $planes:literal, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            resize: &ResizeSqrPixel,
            source: &PlanarImageView<'_, $ty, $planes>,
            destination: &mut PlanarImageViewMut<'_, $ty, $planes>,
        ) -> Result<()> {
            let source_planes = source.planes().map(|plane| plane.as_ptr());
            let mut destination_planes = destination
                .planes_mut()
                .each_mut()
                .map(ImageViewMut::as_mut_ptr);

            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_ptr().cast(),
                    source.planes()[0].size().into(),
                    source.planes()[0].step(),
                    resize.source_roi.into(),
                    destination_planes.as_mut_ptr().cast(),
                    destination.planes()[0].step(),
                    resize.destination_roi.into(),
                    resize.x_factor,
                    resize.y_factor,
                    resize.x_shift,
                    resize.y_shift,
                    i32::from(resize.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_resize {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            resize: &Resize,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    resize.source_roi.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    resize.destination_roi.into(),
                    i32::from(resize.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_resize_planar {
    ($name:ident, $ty:ty, $planes:literal, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            resize: &Resize,
            source: &PlanarImageView<'_, $ty, $planes>,
            destination: &mut PlanarImageViewMut<'_, $ty, $planes>,
        ) -> Result<()> {
            let mut source_planes = source.planes().map(|plane| plane.as_ptr());
            let mut destination_planes = destination
                .planes_mut()
                .each_mut()
                .map(ImageViewMut::as_mut_ptr);

            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_mut_ptr().cast(),
                    source.planes()[0].step(),
                    source.planes()[0].size().into(),
                    resize.source_roi.into(),
                    destination_planes.as_mut_ptr().cast(),
                    destination.planes()[0].step(),
                    destination.planes()[0].size().into(),
                    resize.destination_roi.into(),
                    i32::from(resize.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_resize_batch {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            resize: &Resize,
            sources: &[ImageView<'_, $ty, $layout>],
            destinations: &mut [ImageViewMut<'_, $ty, $layout>],
        ) -> Result<()> {
            let (batch_size, smallest_source_size, smallest_destination_size, descriptors) =
                resize_batch_descriptors(sources, destinations)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    smallest_source_size.into(),
                    resize.source_roi.into(),
                    smallest_destination_size.into(),
                    resize.destination_roi.into(),
                    i32::from(resize.interpolation),
                    descriptors.as_mut_ptr().cast(),
                    batch_size,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_resize_batch_advanced {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            rois: &[ResizeBatchAdvanced],
            interpolation: InterpolationMode,
            sources: &[ImageView<'_, $ty, $layout>],
            destinations: &mut [ImageViewMut<'_, $ty, $layout>],
        ) -> Result<()> {
            let (
                batch_size,
                max_destination_width,
                max_destination_height,
                sources,
                destinations,
                rois,
            ) = resize_batch_advanced_descriptors(rois, sources, destinations)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    max_destination_width,
                    max_destination_height,
                    sources.as_mut_ptr().cast(),
                    destinations.as_mut_ptr().cast(),
                    rois.as_mut_ptr().cast(),
                    batch_size,
                    i32::from(interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_resize_operation {
    (
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                resize: &Resize,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            resize: &Resize,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, resize, source, destination)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    resize: &Resize,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                ) -> Result<()> {
                    $direct(stream_context, resize, source, destination)
                }
            }
        )*
    };
}

macro_rules! impl_generic_resize_sqr_pixel_operation {
    (
        $trait:ident,
        $method:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct:ident),* $(,)?]
    ) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                resize: &ResizeSqrPixel,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            resize: &ResizeSqrPixel,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, resize, source, destination)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    resize: &ResizeSqrPixel,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                ) -> Result<()> {
                    $direct(stream_context, resize, source, destination)
                }
            }
        )*
    };
}

macro_rules! impl_rotate {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            rotate: &Rotate,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.size().into(),
                    source.step(),
                    rotate.source_roi.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    rotate.destination_roi.into(),
                    rotate.angle,
                    rotate.x_shift,
                    rotate.y_shift,
                    i32::from(rotate.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_remap {
    ($name:ident, $ty:ty, $map_ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            remap: &Remap,
            source: &ImageView<'_, $ty, $layout>,
            x_map: &ImageView<'_, $map_ty, C1>,
            y_map: &ImageView<'_, $map_ty, C1>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
        ) -> Result<()> {
            validate_same_size(x_map.size(), destination.size())?;
            validate_same_size(y_map.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.size().into(),
                    source.step(),
                    remap.source_roi.into(),
                    x_map.as_ptr().cast(),
                    x_map.step(),
                    y_map.as_ptr().cast(),
                    y_map.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    i32::from(remap.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_remap_planar {
    ($name:ident, $ty:ty, $map_ty:ty, $planes:literal, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            remap: &Remap,
            source: &PlanarImageView<'_, $ty, $planes>,
            x_map: &ImageView<'_, $map_ty, C1>,
            y_map: &ImageView<'_, $map_ty, C1>,
            destination: &mut PlanarImageViewMut<'_, $ty, $planes>,
        ) -> Result<()> {
            validate_same_size(x_map.size(), destination.planes()[0].size())?;
            validate_same_size(y_map.size(), destination.planes()[0].size())?;
            validate_same_size(source.planes()[0].size(), destination.planes()[0].size())?;

            let source_planes = source.planes().map(|plane| plane.as_ptr());
            let mut destination_planes = destination
                .planes_mut()
                .each_mut()
                .map(ImageViewMut::as_mut_ptr);

            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_ptr().cast(),
                    source.planes()[0].size().into(),
                    source.planes()[0].step(),
                    remap.source_roi.into(),
                    x_map.as_ptr().cast(),
                    x_map.step(),
                    y_map.as_ptr().cast(),
                    y_map.step(),
                    destination_planes.as_mut_ptr().cast(),
                    destination.planes()[0].step(),
                    destination.planes()[0].size().into(),
                    i32::from(remap.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_warp_affine {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            warp: &WarpAffine,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.size().into(),
                    source.step(),
                    warp.source_roi.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    warp.destination_roi.into(),
                    warp.coefficients.as_ptr().cast(),
                    i32::from(warp.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_warp_affine_batch {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            warp: &WarpAffineBatch,
            coefficients: &[AffineCoefficients],
            sources: &[ImageView<'_, $ty, $layout>],
            destinations: &mut [ImageViewMut<'_, $ty, $layout>],
        ) -> Result<()> {
            let (batch_size, smallest_source_size, batch_list, _coefficients) =
                warp_affine_batch_descriptors(coefficients, sources, destinations)?;

            unsafe {
                try_ffi!(sys::nppiWarpAffineBatchInit_Ctx(
                    batch_list.as_mut_ptr().cast(),
                    batch_size,
                    stream_context.as_raw(),
                ))?;
                try_ffi!(sys::$ffi(
                    smallest_source_size.into(),
                    warp.source_roi.into(),
                    warp.destination_roi.into(),
                    i32::from(warp.interpolation),
                    batch_list.as_mut_ptr().cast(),
                    batch_size,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_warp_affine_planar {
    ($name:ident, $ty:ty, $planes:literal, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            warp: &WarpAffine,
            source: &PlanarImageView<'_, $ty, $planes>,
            destination: &mut PlanarImageViewMut<'_, $ty, $planes>,
        ) -> Result<()> {
            let mut source_planes = source.planes().map(|plane| plane.as_ptr());
            let mut destination_planes = destination
                .planes_mut()
                .each_mut()
                .map(ImageViewMut::as_mut_ptr);

            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_mut_ptr().cast(),
                    source.planes()[0].size().into(),
                    source.planes()[0].step(),
                    warp.source_roi.into(),
                    destination_planes.as_mut_ptr().cast(),
                    destination.planes()[0].step(),
                    warp.destination_roi.into(),
                    warp.coefficients.as_ptr().cast(),
                    i32::from(warp.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_warp_affine_quad {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            warp: &WarpQuad,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.size().into(),
                    source.step(),
                    warp.source_roi.into(),
                    warp.source_quadrangle.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    warp.destination_roi.into(),
                    warp.destination_quadrangle.as_ptr().cast(),
                    i32::from(warp.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_warp_affine_quad_planar {
    ($name:ident, $ty:ty, $planes:literal, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            warp: &WarpQuad,
            source: &PlanarImageView<'_, $ty, $planes>,
            destination: &mut PlanarImageViewMut<'_, $ty, $planes>,
        ) -> Result<()> {
            let mut source_planes = source.planes().map(|plane| plane.as_ptr());
            let mut destination_planes = destination
                .planes_mut()
                .each_mut()
                .map(ImageViewMut::as_mut_ptr);

            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_mut_ptr().cast(),
                    source.planes()[0].size().into(),
                    source.planes()[0].step(),
                    warp.source_roi.into(),
                    warp.source_quadrangle.as_ptr().cast(),
                    destination_planes.as_mut_ptr().cast(),
                    destination.planes()[0].step(),
                    warp.destination_roi.into(),
                    warp.destination_quadrangle.as_ptr().cast(),
                    i32::from(warp.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_warp_perspective {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            warp: &WarpPerspective,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.size().into(),
                    source.step(),
                    warp.source_roi.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    warp.destination_roi.into(),
                    warp.coefficients.as_ptr().cast(),
                    i32::from(warp.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_warp_perspective_batch {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            warp: &WarpPerspectiveBatch,
            coefficients: &[PerspectiveCoefficients],
            sources: &[ImageView<'_, $ty, $layout>],
            destinations: &mut [ImageViewMut<'_, $ty, $layout>],
        ) -> Result<()> {
            let (batch_size, smallest_source_size, batch_list, _coefficients) =
                warp_perspective_batch_descriptors(coefficients, sources, destinations)?;

            unsafe {
                try_ffi!(sys::nppiWarpPerspectiveBatchInit_Ctx(
                    batch_list.as_mut_ptr().cast(),
                    batch_size,
                    stream_context.as_raw(),
                ))?;
                try_ffi!(sys::$ffi(
                    smallest_source_size.into(),
                    warp.source_roi.into(),
                    warp.destination_roi.into(),
                    i32::from(warp.interpolation),
                    batch_list.as_mut_ptr().cast(),
                    batch_size,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_warp_perspective_planar {
    ($name:ident, $ty:ty, $planes:literal, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            warp: &WarpPerspective,
            source: &PlanarImageView<'_, $ty, $planes>,
            destination: &mut PlanarImageViewMut<'_, $ty, $planes>,
        ) -> Result<()> {
            let mut source_planes = source.planes().map(|plane| plane.as_ptr());
            let mut destination_planes = destination
                .planes_mut()
                .each_mut()
                .map(ImageViewMut::as_mut_ptr);

            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_mut_ptr().cast(),
                    source.planes()[0].size().into(),
                    source.planes()[0].step(),
                    warp.source_roi.into(),
                    destination_planes.as_mut_ptr().cast(),
                    destination.planes()[0].step(),
                    warp.destination_roi.into(),
                    warp.coefficients.as_ptr().cast(),
                    i32::from(warp.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_warp_perspective_quad {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            warp: &WarpQuad,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.size().into(),
                    source.step(),
                    warp.source_roi.into(),
                    warp.source_quadrangle.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    warp.destination_roi.into(),
                    warp.destination_quadrangle.as_ptr().cast(),
                    i32::from(warp.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_warp_perspective_quad_planar {
    ($name:ident, $ty:ty, $planes:literal, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            warp: &WarpQuad,
            source: &PlanarImageView<'_, $ty, $planes>,
            destination: &mut PlanarImageViewMut<'_, $ty, $planes>,
        ) -> Result<()> {
            let mut source_planes = source.planes().map(|plane| plane.as_ptr());
            let mut destination_planes = destination
                .planes_mut()
                .each_mut()
                .map(ImageViewMut::as_mut_ptr);

            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_mut_ptr().cast(),
                    source.planes()[0].size().into(),
                    source.planes()[0].step(),
                    warp.source_roi.into(),
                    warp.source_quadrangle.as_ptr().cast(),
                    destination_planes.as_mut_ptr().cast(),
                    destination.planes()[0].step(),
                    warp.destination_roi.into(),
                    warp.destination_quadrangle.as_ptr().cast(),
                    i32::from(warp.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_warp_perspective_back {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            warp: &WarpPerspective,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.size().into(),
                    source.step(),
                    warp.source_roi.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    warp.destination_roi.into(),
                    warp.coefficients.as_ptr().cast(),
                    i32::from(warp.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_warp_perspective_back_planar {
    ($name:ident, $ty:ty, $planes:literal, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            warp: &WarpPerspective,
            source: &PlanarImageView<'_, $ty, $planes>,
            destination: &mut PlanarImageViewMut<'_, $ty, $planes>,
        ) -> Result<()> {
            let mut source_planes = source.planes().map(|plane| plane.as_ptr());
            let mut destination_planes = destination
                .planes_mut()
                .each_mut()
                .map(ImageViewMut::as_mut_ptr);

            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_mut_ptr().cast(),
                    source.planes()[0].size().into(),
                    source.planes()[0].step(),
                    warp.source_roi.into(),
                    destination_planes.as_mut_ptr().cast(),
                    destination.planes()[0].step(),
                    warp.destination_roi.into(),
                    warp.coefficients.as_ptr().cast(),
                    i32::from(warp.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_warp_affine_back {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            warp: &WarpAffine,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.size().into(),
                    source.step(),
                    warp.source_roi.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    warp.destination_roi.into(),
                    warp.coefficients.as_ptr().cast(),
                    i32::from(warp.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_warp_affine_back_planar {
    ($name:ident, $ty:ty, $planes:literal, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            warp: &WarpAffine,
            source: &PlanarImageView<'_, $ty, $planes>,
            destination: &mut PlanarImageViewMut<'_, $ty, $planes>,
        ) -> Result<()> {
            validate_same_size(source.planes()[0].size(), destination.planes()[0].size())?;

            let mut source_planes = source.planes().map(|plane| plane.as_ptr());
            let mut destination_planes = destination
                .planes_mut()
                .each_mut()
                .map(ImageViewMut::as_mut_ptr);

            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_mut_ptr().cast(),
                    source.planes()[0].size().into(),
                    source.planes()[0].step(),
                    warp.source_roi.into(),
                    destination_planes.as_mut_ptr().cast(),
                    destination.planes()[0].step(),
                    warp.destination_roi.into(),
                    warp.coefficients.as_ptr().cast(),
                    i32::from(warp.interpolation),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_integer_kernel_filter_image {
    ($ty:ty, $layout:ty, $filter:path, $filter_border:path) => {
        unsafe impl<'a> IntegerKernelFilterImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn filter_kernel_image(
                stream_context: &StreamContext,
                source: &$crate::image::view::ImageWindow<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                kernel: &[i32],
                kernel_size: Size,
                anchor: Point,
                divisor: i32,
            ) -> Result<()> {
                source.require_halo(kernel_size, anchor)?;
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $filter(
                        stream_context,
                        &source.view(),
                        destination,
                        kernel,
                        kernel_size,
                        anchor,
                        divisor,
                    )
                })
            }

            fn filter_kernel_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                kernel: &[i32],
                kernel_size: Size,
                anchor: Point,
                divisor: i32,
                border_type: BorderType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $filter_border(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        kernel,
                        kernel_size,
                        anchor,
                        divisor,
                        border_type,
                    )
                })
            }
        }
    };
}

macro_rules! impl_float_kernel_filter_image {
    ($ty:ty, $layout:ty, $filter:path, $filter_border:path) => {
        unsafe impl<'a> FloatKernelFilterImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn filter_kernel32f_image(
                stream_context: &StreamContext,
                source: &$crate::image::view::ImageWindow<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                kernel: &[f32],
                kernel_size: Size,
                anchor: Point,
            ) -> Result<()> {
                source.require_halo(kernel_size, anchor)?;
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $filter(
                        stream_context,
                        &source.view(),
                        destination,
                        kernel,
                        kernel_size,
                        anchor,
                    )
                })
            }

            fn filter_kernel32f_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                kernel: &[f32],
                kernel_size: Size,
                anchor: Point,
                border_type: BorderType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $filter_border(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        kernel,
                        kernel_size,
                        anchor,
                        border_type,
                    )
                })
            }
        }
    };
}

macro_rules! impl_double_kernel_filter_image {
    ($ty:ty, $layout:ty, $filter:path) => {
        unsafe impl<'a> DoubleKernelFilterImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn filter_kernel64f_image(
                stream_context: &StreamContext,
                source: &$crate::image::view::ImageWindow<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                kernel: &[f64],
                kernel_size: Size,
                anchor: Point,
            ) -> Result<()> {
                source.require_halo(kernel_size, anchor)?;
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $filter(
                        stream_context,
                        &source.view(),
                        destination,
                        kernel,
                        kernel_size,
                        anchor,
                    )
                })
            }
        }
    };
}

macro_rules! impl_typed_float_kernel_filter_image {
    (
        $source_ty:ty,
        $source_layout:ty,
        $destination_ty:ty,
        $destination_layout:ty,
        $filter:path,
        $filter_border:path
    ) => {
        unsafe impl<'a>
            TypedFloatKernelFilterImage<
                $source_ty,
                $source_layout,
                $destination_ty,
                $destination_layout,
            > for ImagePipeline<'a, $source_ty, $source_layout>
        {
            fn filter_kernel32f_to_image(
                stream_context: &StreamContext,
                source: &$crate::image::view::ImageWindow<'_, $source_ty, $source_layout>,
                destination: &mut ImageOutputView<'_, $destination_ty, $destination_layout>,
                kernel: &[f32],
                kernel_size: Size,
                anchor: Point,
            ) -> Result<()> {
                source.require_halo(kernel_size, anchor)?;
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $filter(
                        stream_context,
                        &source.view(),
                        destination,
                        kernel,
                        kernel_size,
                        anchor,
                    )
                })
            }

            fn filter_kernel32f_border_to_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, $source_layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, $destination_ty, $destination_layout>,
                kernel: &[f32],
                kernel_size: Size,
                anchor: Point,
                border_type: BorderType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $filter_border(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        kernel,
                        kernel_size,
                        anchor,
                        border_type,
                    )
                })
            }
        }
    };
}

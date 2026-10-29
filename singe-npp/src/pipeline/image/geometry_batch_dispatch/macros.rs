macro_rules! impl_mirror_batch_image {
    ($ty:ty, $layout:ty, $mirror:path, $mirror_in_place:path) => {
        impl<'a> ImagePipeline<'a, $ty, $layout> {
            /// Mirrors a batch of images along `axis` into output views.
            ///
            /// The common ROI uses the smallest image width and height in the batch.
            /// Prepares image producer streams, rejects stream capture, and returns after successful stream completion.
            /// The native helper owns temporary device descriptors through completion.
            /// Only pixels covered by the transformed source intersection are written.
            /// AC4 operations leave alpha untouched; uncovered pixels, ignored alpha, and row padding do not become initialized by this call.
            ///
            /// # Errors
            ///
            /// Returns an error if capture is active, a participating image belongs to another CUDA context, or CUDA context binding fails.
            /// Producer-stream synchronization, native count conversion, descriptor allocation or upload, NPP execution, and stream synchronization can also fail.
            /// Source and destination batches must have equal lengths of at least two.
            /// Each source/destination pair must have matching image sizes.
            #[doc = concat!("Uses [`", stringify!($mirror), "`].")]
            pub fn mirror_batch(
                stream_context: &StreamContext,
                axis: $crate::types::Axis,
                sources: &[ImageView<'_, $ty, $layout>],
                destinations: &mut [ImageOutputView<'_, $ty, $layout>],
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                for image in sources.iter() {
                    stream_context.prepare_region(image)?;
                }
                for image in destinations.iter() {
                    stream_context.prepare_region(image)?;
                }
                // SAFETY: the native helper completes borrowed use on every path.
                unsafe { $mirror(stream_context, axis, sources, destinations) }
            }

            /// Mirrors a batch of initialized images in place along `axis`.
            ///
            /// The common ROI uses the smallest image width and height in the batch.
            /// Prepares image producer streams, rejects stream capture, and returns after successful stream completion.
            /// The native helper owns temporary device descriptors through completion.
            /// Pixels outside the common ROI and alpha ignored by AC4 retain their existing values.
            ///
            /// # Errors
            ///
            /// Returns an error if capture is active, a participating image belongs to another CUDA context, or CUDA context binding fails.
            /// Producer-stream synchronization, native count conversion, descriptor allocation or upload, NPP execution, and stream synchronization can also fail.
            /// The batch must contain at least two images.
            #[doc = concat!("Uses [`", stringify!($mirror_in_place), "`].")]
            pub fn mirror_batch_in_place(
                stream_context: &StreamContext,
                axis: $crate::types::Axis,
                images: &mut [ImageViewMut<'_, $ty, $layout>],
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                for image in images.iter() {
                    stream_context.prepare_region(image)?;
                }
                // SAFETY: the native helper completes borrowed use on every path.
                unsafe { $mirror_in_place(stream_context, axis, images) }
            }
        }
    };
}

macro_rules! impl_resize_batch_image {
    ($ty:ty, $layout:ty, $resize:path, $resize_advanced:path) => {
        impl<'a> ImagePipeline<'a, $ty, $layout> {
            /// Resizes a batch using one shared source ROI, destination ROI, and interpolation mode.
            ///
            /// Native source and destination bounds use the smallest width and height in their respective batches.
            /// Prepares image producer streams, rejects stream capture, and returns after successful stream completion.
            /// The native helper owns temporary device descriptors through completion.
            /// Only pixels covered by the transformed source intersection are written.
            /// AC4 operations leave alpha untouched; uncovered pixels, ignored alpha, and row padding do not become initialized by this call.
            ///
            /// # Errors
            ///
            /// Returns an error if capture is active, a participating image belongs to another CUDA context, or CUDA context binding fails.
            /// Producer-stream synchronization, native count conversion, descriptor allocation or upload, NPP execution, and stream synchronization can also fail.
            /// Source and destination batches must have equal lengths of at least two.
            /// The source ROI must have positive dimensions and representable endpoints.
            /// The destination ROI must lie within each corresponding destination image.
            #[doc = concat!("Uses [`", stringify!($resize), "`].")]
            pub fn resize_batch(
                stream_context: &StreamContext,
                resize: &geometry::Resize,
                sources: &[ImageView<'_, $ty, $layout>],
                destinations: &mut [ImageOutputView<'_, $ty, $layout>],
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                for image in sources.iter() {
                    stream_context.prepare_region(image)?;
                }
                for image in destinations.iter() {
                    stream_context.prepare_region(image)?;
                }
                // SAFETY: the native helper completes borrowed use on every path.
                unsafe { $resize(stream_context, resize, sources, destinations) }
            }

            /// Resizes image pairs using their individual ROI pairs and the supplied interpolation mode.
            ///
            /// Each descriptor carries its own image dimensions.
            /// The largest destination ROI width and height determine the native launch bounds.
            /// Prepares image producer streams, rejects stream capture, and returns after successful stream completion.
            /// The native helper owns temporary device descriptors through completion.
            /// Only pixels covered by the transformed source intersection are written.
            /// AC4 operations leave alpha untouched; uncovered pixels, ignored alpha, and row padding do not become initialized by this call.
            ///
            /// # Errors
            ///
            /// Returns an error if capture is active, a participating image belongs to another CUDA context, or CUDA context binding fails.
            /// Producer-stream synchronization, native count conversion, descriptor allocation or upload, NPP execution, and stream synchronization can also fail.
            /// Source, destination, and ROI-pair batches must have equal lengths of at least two.
            /// The source ROI must have positive dimensions and representable endpoints.
            /// The destination ROI must lie within each corresponding destination image.
            #[doc = concat!("Uses [`", stringify!($resize_advanced), "`].")]
            pub fn resize_batch_advanced(
                stream_context: &StreamContext,
                rois: &[geometry::ResizeBatchAdvanced],
                interpolation: $crate::types::InterpolationMode,
                sources: &[ImageView<'_, $ty, $layout>],
                destinations: &mut [ImageOutputView<'_, $ty, $layout>],
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                for image in sources.iter() {
                    stream_context.prepare_region(image)?;
                }
                for image in destinations.iter() {
                    stream_context.prepare_region(image)?;
                }
                // SAFETY: the native helper completes borrowed use on every path.
                unsafe {
                    $resize_advanced(stream_context, rois, interpolation, sources, destinations)
                }
            }
        }
    };
}

macro_rules! impl_resize_batch_advanced_image {
    ($ty:ty, $layout:ty, $resize_advanced:path) => {
        impl<'a> ImagePipeline<'a, $ty, $layout> {
            /// Resizes image pairs using their individual ROI pairs and the supplied interpolation mode.
            ///
            /// Each descriptor carries its own image dimensions.
            /// The largest destination ROI width and height determine the native launch bounds.
            /// Prepares image producer streams, rejects stream capture, and returns after successful stream completion.
            /// The native helper owns temporary device descriptors through completion.
            /// Only pixels covered by the transformed source intersection are written.
            /// AC4 operations leave alpha untouched; uncovered pixels, ignored alpha, and row padding do not become initialized by this call.
            ///
            /// # Errors
            ///
            /// Returns an error if capture is active, a participating image belongs to another CUDA context, or CUDA context binding fails.
            /// Producer-stream synchronization, native count conversion, descriptor allocation or upload, NPP execution, and stream synchronization can also fail.
            /// Source, destination, and ROI-pair batches must have equal lengths of at least two.
            /// The source ROI must have positive dimensions and representable endpoints.
            /// The destination ROI must lie within each corresponding destination image.
            #[doc = concat!("Uses [`", stringify!($resize_advanced), "`].")]
            pub fn resize_batch_advanced(
                stream_context: &StreamContext,
                rois: &[geometry::ResizeBatchAdvanced],
                interpolation: $crate::types::InterpolationMode,
                sources: &[ImageView<'_, $ty, $layout>],
                destinations: &mut [ImageOutputView<'_, $ty, $layout>],
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                for image in sources.iter() {
                    stream_context.prepare_region(image)?;
                }
                for image in destinations.iter() {
                    stream_context.prepare_region(image)?;
                }
                // SAFETY: the native helper completes borrowed use on every path.
                unsafe {
                    $resize_advanced(stream_context, rois, interpolation, sources, destinations)
                }
            }
        }
    };
}

macro_rules! impl_warp_batch_image {
    ($ty:ty, $layout:ty, $affine:path, $perspective:path) => {
        impl<'a> ImagePipeline<'a, $ty, $layout> {
            /// Warps image pairs with one affine coefficient matrix per pair.
            ///
            /// Uses shared source and destination ROIs and the configured interpolation mode.
            /// Native source bounds use the smallest source width and height in the batch.
            /// Prepares image producer streams, rejects stream capture, and returns after successful stream completion.
            /// The native helper owns temporary device descriptors through completion.
            /// Only pixels covered by the transformed source intersection are written.
            /// AC4 operations leave alpha untouched; uncovered pixels, ignored alpha, and row padding do not become initialized by this call.
            ///
            /// # Errors
            ///
            /// Returns an error if capture is active, a participating image belongs to another CUDA context, or CUDA context binding fails.
            /// Producer-stream synchronization, native count conversion, descriptor allocation or upload, NPP execution, and stream synchronization can also fail.
            /// Source, destination, and coefficient batches must have equal lengths of at least two.
            /// The source ROI must have positive dimensions and representable endpoints.
            /// The destination ROI must lie within each corresponding destination image.
            /// Native batch initialization can also fail.
            #[doc = concat!("Uses [`", stringify!($affine), "`].")]
            pub fn warp_affine_batch(
                stream_context: &StreamContext,
                warp: &geometry::WarpAffineBatch,
                coefficients: &[$crate::types::AffineCoefficients],
                sources: &[ImageView<'_, $ty, $layout>],
                destinations: &mut [ImageOutputView<'_, $ty, $layout>],
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                for image in sources.iter() {
                    stream_context.prepare_region(image)?;
                }
                for image in destinations.iter() {
                    stream_context.prepare_region(image)?;
                }
                // SAFETY: the native helper completes borrowed use on every path.
                unsafe { $affine(stream_context, warp, coefficients, sources, destinations) }
            }

            /// Warps image pairs with one perspective coefficient matrix per pair.
            ///
            /// Uses shared source and destination ROIs and the configured interpolation mode.
            /// Native source bounds use the smallest source width and height in the batch.
            /// Prepares image producer streams, rejects stream capture, and returns after successful stream completion.
            /// The native helper owns temporary device descriptors through completion.
            /// Only pixels covered by the transformed source intersection are written.
            /// AC4 operations leave alpha untouched; uncovered pixels, ignored alpha, and row padding do not become initialized by this call.
            ///
            /// # Errors
            ///
            /// Returns an error if capture is active, a participating image belongs to another CUDA context, or CUDA context binding fails.
            /// Producer-stream synchronization, native count conversion, descriptor allocation or upload, NPP execution, and stream synchronization can also fail.
            /// Source, destination, and coefficient batches must have equal lengths of at least two.
            /// The source ROI must have positive dimensions and representable endpoints.
            /// The destination ROI must lie within each corresponding destination image.
            /// Native batch initialization can also fail.
            #[doc = concat!("Uses [`", stringify!($perspective), "`].")]
            pub fn warp_perspective_batch(
                stream_context: &StreamContext,
                warp: &geometry::WarpPerspectiveBatch,
                coefficients: &[$crate::types::PerspectiveCoefficients],
                sources: &[ImageView<'_, $ty, $layout>],
                destinations: &mut [ImageOutputView<'_, $ty, $layout>],
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                for image in sources.iter() {
                    stream_context.prepare_region(image)?;
                }
                for image in destinations.iter() {
                    stream_context.prepare_region(image)?;
                }
                // SAFETY: the native helper completes borrowed use on every path.
                unsafe { $perspective(stream_context, warp, coefficients, sources, destinations) }
            }
        }
    };
}

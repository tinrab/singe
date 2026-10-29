use crate::{
    context::StreamContext,
    error::Result,
    image::{
        geometry,
        view::{C1, ImageView, PlanarImageOutputView, PlanarImageView},
    },
    pipeline::image::planar::{PlanarImage, PlanarRemap, PlanarResize, PlanarResizeSqrPixel},
};

macro_rules! impl_planar_resize_geometry {
    ($ty:ty, $planes:literal, $resize:path) => {
        impl PlanarImage<$ty, $planes> {
            /// Resizes the source ROI into the destination ROI using `resize.interpolation`.
            ///
            /// Prepares participating image regions, waiting for their producers on other streams, rejects capture, and returns after successful stream completion.
            /// Writes into the supplied initialized `destination` and returns that destination. Uncovered logical pixels retain their previous destination values; row padding remains unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding or region-context validation fails, preparing a region or waiting for its producer fails, capture is active, or stream synchronization fails.
            /// Source ROI dimensions must be positive and their endpoints representable.
            /// Each destination ROI must lie within its destination image.
            /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
            /// Constructing plane views or output views can also fail if stored plane geometry or contexts no longer agree.
            #[doc = concat!("Uses [`", stringify!($resize), "`].")]
            pub fn resize(
                self,
                stream_context: &StreamContext,
                resize: geometry::Resize,
                destination: Self,
            ) -> Result<Self> {
                self.transform_geometry(
                    stream_context,
                    destination,
                    &resize,
                    $resize as PlanarResize<$ty, $planes>,
                )
            }
        }
    };
}

macro_rules! impl_planar_resize_sqr_pixel_geometry {
    ($ty:ty, $planes:literal, $resize:path) => {
        impl PlanarImage<$ty, $planes> {
            /// Resizes all planes using the scale factors, shifts, and interpolation in `resize`.
            ///
            /// Prepares participating image regions, waiting for their producers on other streams, rejects capture, and returns after successful stream completion.
            /// Writes into the supplied initialized `destination` and returns that destination. Uncovered logical pixels retain their previous destination values; row padding remains unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding or region-context validation fails, preparing a region or waiting for its producer fails, capture is active, or stream synchronization fails.
            /// Source ROI dimensions must be positive and their endpoints representable.
            /// Each destination ROI must lie within its destination image.
            /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
            /// Constructing plane views or output views can also fail if stored plane geometry or contexts no longer agree.
            #[doc = concat!("Uses [`", stringify!($resize), "`].")]
            pub fn resize_sqr_pixel(
                self,
                stream_context: &StreamContext,
                resize: geometry::ResizeSqrPixel,
                destination: Self,
            ) -> Result<Self> {
                self.transform_geometry(
                    stream_context,
                    destination,
                    &resize,
                    $resize as PlanarResizeSqrPixel<$ty, $planes>,
                )
            }
        }
    };
}

macro_rules! impl_planar_remap_geometry {
    ($ty:ty, $map_ty:ty, $planes:literal, $remap:path) => {
        impl PlanarImage<$ty, $planes> {
            /// Samples all source planes at coordinates from `x_map` and `y_map`, using the configured interpolation.
            ///
            /// Prepares participating image and coordinate-map regions, waiting for their producers on other streams, rejects capture, and returns after successful stream completion.
            /// Writes into the supplied initialized `destination` and returns that destination. Uncovered logical pixels retain their previous destination values; row padding remains unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding or region-context validation fails, preparing a region or waiting for its producer fails, capture is active, or stream synchronization fails.
            /// Source ROI dimensions must be positive and their endpoints representable.
            /// Both coordinate maps must have the destination image size, and source and destination plane sizes must match.
            /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
            /// Constructing plane views or output views can also fail if stored plane geometry or contexts no longer agree.
            #[doc = concat!("Uses [`", stringify!($remap), "`].")]
            pub fn remap(
                self,
                stream_context: &StreamContext,
                remap: geometry::Remap,
                x_map: &ImageView<'_, $map_ty, C1>,
                y_map: &ImageView<'_, $map_ty, C1>,
                destination: Self,
            ) -> Result<Self> {
                self.remap_geometry(
                    stream_context,
                    &remap,
                    x_map,
                    y_map,
                    destination,
                    $remap as PlanarRemap<$ty, $map_ty, $planes>,
                )
            }
        }
    };
}

macro_rules! impl_planar_resize_sqr_pixel_geometry_into {
    ($ty:ty, $planes:literal, $resize:path) => {
        impl PlanarImage<$ty, $planes> {
            /// Resizes all planes using the scale factors, shifts, and interpolation in `resize`.
            ///
            /// Prepares participating image regions, waiting for their producers on other streams, rejects capture, and returns after successful stream completion.
            /// Pixels outside the transformed source intersection and row padding retain their previous initialization state; this call does not establish initialization of those samples.
            /// Retain equal dimensions and byte strides within each source and destination plane set, as required by the native planar layout.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding or region-context validation fails, preparing a region or waiting for its producer fails, capture is active, or stream synchronization fails.
            /// Source ROI dimensions must be positive and their endpoints representable.
            /// Each destination ROI must lie within its destination image.
            /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
            #[doc = concat!("Uses [`", stringify!($resize), "`].")]
            pub fn resize_sqr_pixel_into(
                stream_context: &StreamContext,
                resize: &geometry::ResizeSqrPixel,
                source: &PlanarImageView<'_, $ty, $planes>,
                destination: &mut PlanarImageOutputView<'_, $ty, $planes>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                for plane in source.planes() {
                    stream_context.prepare_region(plane)?;
                }
                for plane in destination.planes() {
                    stream_context.prepare_region(plane)?;
                }
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $resize(stream_context, resize, source, destination) })
            }
        }
    };
}

macro_rules! impl_planar_remap_geometry_into {
    ($ty:ty, $map_ty:ty, $planes:literal, $remap:path) => {
        impl PlanarImage<$ty, $planes> {
            /// Samples all source planes at coordinates from `x_map` and `y_map`, using the configured interpolation.
            ///
            /// Prepares participating image and coordinate-map regions, waiting for their producers on other streams, rejects capture, and returns after successful stream completion.
            /// Pixels outside the transformed source intersection and row padding retain their previous initialization state; this call does not establish initialization of those samples.
            /// Retain equal dimensions and byte strides within each source and destination plane set, as required by the native planar layout.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding or region-context validation fails, preparing a region or waiting for its producer fails, capture is active, or stream synchronization fails.
            /// Source ROI dimensions must be positive and their endpoints representable.
            /// Both coordinate maps must have the destination image size, and source and destination plane sizes must match.
            /// Returns an error if NPP rejects the geometry, interpolation, or other operation parameters.
            #[doc = concat!("Uses [`", stringify!($remap), "`].")]
            pub fn remap_into(
                stream_context: &StreamContext,
                remap: &geometry::Remap,
                source: &PlanarImageView<'_, $ty, $planes>,
                x_map: &ImageView<'_, $map_ty, C1>,
                y_map: &ImageView<'_, $map_ty, C1>,
                destination: &mut PlanarImageOutputView<'_, $ty, $planes>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(x_map)?;
                stream_context.prepare_region(y_map)?;
                for plane in source.planes() {
                    stream_context.prepare_region(plane)?;
                }
                for plane in destination.planes() {
                    stream_context.prepare_region(plane)?;
                }
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $remap(stream_context, remap, source, x_map, y_map, destination)
                })
            }
        }
    };
}

impl_planar_resize_geometry!(u8, 3, geometry::resize_u8_p3);

impl_planar_resize_geometry!(u8, 4, geometry::resize_u8_p4);

impl_planar_resize_geometry!(u16, 3, geometry::resize_u16_p3);

impl_planar_resize_geometry!(u16, 4, geometry::resize_u16_p4);

impl_planar_resize_geometry!(i16, 3, geometry::resize_i16_p3);

impl_planar_resize_geometry!(i16, 4, geometry::resize_i16_p4);

impl_planar_resize_geometry!(f32, 3, geometry::resize_f32_p3);

impl_planar_resize_geometry!(f32, 4, geometry::resize_f32_p4);

impl_planar_resize_sqr_pixel_geometry!(u8, 3, geometry::resize_sqr_pixel_u8_p3);

impl_planar_resize_sqr_pixel_geometry!(u8, 4, geometry::resize_sqr_pixel_u8_p4);

impl_planar_resize_sqr_pixel_geometry!(u16, 3, geometry::resize_sqr_pixel_u16_p3);

impl_planar_resize_sqr_pixel_geometry!(u16, 4, geometry::resize_sqr_pixel_u16_p4);

impl_planar_resize_sqr_pixel_geometry!(i16, 3, geometry::resize_sqr_pixel_i16_p3);

impl_planar_resize_sqr_pixel_geometry!(i16, 4, geometry::resize_sqr_pixel_i16_p4);

impl_planar_resize_sqr_pixel_geometry!(f32, 3, geometry::resize_sqr_pixel_f32_p3);

impl_planar_resize_sqr_pixel_geometry!(f32, 4, geometry::resize_sqr_pixel_f32_p4);

impl_planar_resize_sqr_pixel_geometry_into!(f64, 3, geometry::resize_sqr_pixel_f64_p3);

impl_planar_resize_sqr_pixel_geometry_into!(f64, 4, geometry::resize_sqr_pixel_f64_p4);

impl_planar_remap_geometry!(u8, f32, 3, geometry::remap_u8_p3);

impl_planar_remap_geometry!(u8, f32, 4, geometry::remap_u8_p4);

impl_planar_remap_geometry!(u16, f32, 3, geometry::remap_u16_p3);

impl_planar_remap_geometry!(u16, f32, 4, geometry::remap_u16_p4);

impl_planar_remap_geometry!(i16, f32, 3, geometry::remap_i16_p3);

impl_planar_remap_geometry!(i16, f32, 4, geometry::remap_i16_p4);

impl_planar_remap_geometry!(f32, f32, 3, geometry::remap_f32_p3);

impl_planar_remap_geometry!(f32, f32, 4, geometry::remap_f32_p4);

impl_planar_remap_geometry_into!(f64, f64, 3, geometry::remap_f64_p3);

impl_planar_remap_geometry_into!(f64, f64, 4, geometry::remap_f64_p4);

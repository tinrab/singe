use crate::{
    context::StreamContext,
    error::Result,
    image::geometry,
    pipeline::image::planar::{
        PlanarImage, PlanarWarpAffine, PlanarWarpPerspective, PlanarWarpQuad,
    },
};

mod f64_affine_geometry;

macro_rules! impl_planar_warp_geometry {
    ($ty:ty, $planes:literal, $affine:path, $affine_back:path, $affine_quad:path, $perspective:path, $perspective_back:path, $perspective_quad:path) => {
        impl PlanarImage<$ty, $planes> {
            /// Applies an affine warp to every planar image channel.
            ///
            /// Consumes the source and supplied initialized destination and returns that destination.
            /// Pixels outside the mapped source intersection retain their previous destination values.
            /// Prepares all plane producer streams, rejects capture, and waits for successful stream completion.
            ///
            /// # Errors
            ///
            /// Returns an error if source or destination planes have inconsistent dimensions, byte strides, or CUDA contexts.
            /// The source ROI must have positive dimensions and representable endpoints.
            /// The destination ROI must lie within each corresponding destination image.
            /// Capture checks, producer synchronization, CUDA context binding, NPP geometry or interpolation validation, and stream synchronization can also fail.
            #[doc = concat!("Uses [`", stringify!($affine), "`].")]
            pub fn warp_affine(
                self,
                stream_context: &StreamContext,
                warp: geometry::WarpAffine,
                destination: Self,
            ) -> Result<Self> {
                self.transform_geometry(
                    stream_context,
                    destination,
                    &warp,
                    $affine as PlanarWarpAffine<$ty, $planes>,
                )
            }

            /// Applies a backwards affine warp to every planar image channel.
            ///
            /// Consumes the source and supplied initialized destination and returns that destination.
            /// Pixels outside the mapped source intersection retain their previous destination values.
            /// Prepares all plane producer streams, rejects capture, and waits for successful stream completion.
            ///
            /// # Errors
            ///
            /// Returns an error if source or destination planes have inconsistent dimensions, byte strides, or CUDA contexts.
            /// The source ROI must have positive dimensions and representable endpoints.
            /// The destination ROI must lie within each corresponding destination image.
            /// Capture checks, producer synchronization, CUDA context binding, NPP geometry or interpolation validation, and stream synchronization can also fail.
            #[doc = concat!("Uses [`", stringify!($affine_back), "`].")]
            pub fn warp_affine_back(
                self,
                stream_context: &StreamContext,
                warp: geometry::WarpAffine,
                destination: Self,
            ) -> Result<Self> {
                self.transform_geometry(
                    stream_context,
                    destination,
                    &warp,
                    $affine_back as PlanarWarpAffine<$ty, $planes>,
                )
            }

            /// Applies an affine warp between the supplied quadrangles to every planar image channel.
            ///
            /// Consumes the source and supplied initialized destination and returns that destination.
            /// Pixels outside the mapped source intersection retain their previous destination values.
            /// Prepares all plane producer streams, rejects capture, and waits for successful stream completion.
            ///
            /// # Errors
            ///
            /// Returns an error if source or destination planes have inconsistent dimensions, byte strides, or CUDA contexts.
            /// The source ROI must have positive dimensions and representable endpoints.
            /// The destination ROI must lie within each corresponding destination image.
            /// Capture checks, producer synchronization, CUDA context binding, NPP geometry or interpolation validation, and stream synchronization can also fail.
            #[doc = concat!("Uses [`", stringify!($affine_quad), "`].")]
            pub fn warp_affine_quad(
                self,
                stream_context: &StreamContext,
                warp: geometry::WarpQuad,
                destination: Self,
            ) -> Result<Self> {
                self.transform_geometry(
                    stream_context,
                    destination,
                    &warp,
                    $affine_quad as PlanarWarpQuad<$ty, $planes>,
                )
            }

            /// Applies a perspective warp to every planar image channel.
            ///
            /// Consumes the source and supplied initialized destination and returns that destination.
            /// Pixels outside the mapped source intersection retain their previous destination values.
            /// Prepares all plane producer streams, rejects capture, and waits for successful stream completion.
            ///
            /// # Errors
            ///
            /// Returns an error if source or destination planes have inconsistent dimensions, byte strides, or CUDA contexts.
            /// The source ROI must have positive dimensions and representable endpoints.
            /// The destination ROI must lie within each corresponding destination image.
            /// Capture checks, producer synchronization, CUDA context binding, NPP geometry or interpolation validation, and stream synchronization can also fail.
            #[doc = concat!("Uses [`", stringify!($perspective), "`].")]
            pub fn warp_perspective(
                self,
                stream_context: &StreamContext,
                warp: geometry::WarpPerspective,
                destination: Self,
            ) -> Result<Self> {
                self.transform_geometry(
                    stream_context,
                    destination,
                    &warp,
                    $perspective as PlanarWarpPerspective<$ty, $planes>,
                )
            }

            /// Applies a backwards perspective warp to every planar image channel.
            ///
            /// Consumes the source and supplied initialized destination and returns that destination.
            /// Pixels outside the mapped source intersection retain their previous destination values.
            /// Prepares all plane producer streams, rejects capture, and waits for successful stream completion.
            ///
            /// # Errors
            ///
            /// Returns an error if source or destination planes have inconsistent dimensions, byte strides, or CUDA contexts.
            /// The source ROI must have positive dimensions and representable endpoints.
            /// The destination ROI must lie within each corresponding destination image.
            /// Capture checks, producer synchronization, CUDA context binding, NPP geometry or interpolation validation, and stream synchronization can also fail.
            #[doc = concat!("Uses [`", stringify!($perspective_back), "`].")]
            pub fn warp_perspective_back(
                self,
                stream_context: &StreamContext,
                warp: geometry::WarpPerspective,
                destination: Self,
            ) -> Result<Self> {
                self.transform_geometry(
                    stream_context,
                    destination,
                    &warp,
                    $perspective_back as PlanarWarpPerspective<$ty, $planes>,
                )
            }

            /// Applies a perspective warp between the supplied quadrangles to every planar image channel.
            ///
            /// Consumes the source and supplied initialized destination and returns that destination.
            /// Pixels outside the mapped source intersection retain their previous destination values.
            /// Prepares all plane producer streams, rejects capture, and waits for successful stream completion.
            ///
            /// # Errors
            ///
            /// Returns an error if source or destination planes have inconsistent dimensions, byte strides, or CUDA contexts.
            /// The source ROI must have positive dimensions and representable endpoints.
            /// The destination ROI must lie within each corresponding destination image.
            /// Capture checks, producer synchronization, CUDA context binding, NPP geometry or interpolation validation, and stream synchronization can also fail.
            #[doc = concat!("Uses [`", stringify!($perspective_quad), "`].")]
            pub fn warp_perspective_quad(
                self,
                stream_context: &StreamContext,
                warp: geometry::WarpQuad,
                destination: Self,
            ) -> Result<Self> {
                self.transform_geometry(
                    stream_context,
                    destination,
                    &warp,
                    $perspective_quad as PlanarWarpQuad<$ty, $planes>,
                )
            }
        }
    };
}

mod f32_geometry;
mod i32_geometry;

impl_planar_warp_geometry!(
    u8,
    3,
    geometry::warp_affine_u8_p3,
    geometry::warp_affine_back_u8_p3,
    geometry::warp_affine_quad_u8_p3,
    geometry::warp_perspective_u8_p3,
    geometry::warp_perspective_back_u8_p3,
    geometry::warp_perspective_quad_u8_p3
);

impl_planar_warp_geometry!(
    u8,
    4,
    geometry::warp_affine_u8_p4,
    geometry::warp_affine_back_u8_p4,
    geometry::warp_affine_quad_u8_p4,
    geometry::warp_perspective_u8_p4,
    geometry::warp_perspective_back_u8_p4,
    geometry::warp_perspective_quad_u8_p4
);

impl_planar_warp_geometry!(
    u16,
    3,
    geometry::warp_affine_u16_p3,
    geometry::warp_affine_back_u16_p3,
    geometry::warp_affine_quad_u16_p3,
    geometry::warp_perspective_u16_p3,
    geometry::warp_perspective_back_u16_p3,
    geometry::warp_perspective_quad_u16_p3
);

impl_planar_warp_geometry!(
    u16,
    4,
    geometry::warp_affine_u16_p4,
    geometry::warp_affine_back_u16_p4,
    geometry::warp_affine_quad_u16_p4,
    geometry::warp_perspective_u16_p4,
    geometry::warp_perspective_back_u16_p4,
    geometry::warp_perspective_quad_u16_p4
);

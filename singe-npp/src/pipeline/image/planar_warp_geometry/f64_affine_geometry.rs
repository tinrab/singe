use crate::{
    context::StreamContext,
    error::Result,
    image::{
        geometry,
        view::{PlanarImageOutputView, PlanarImageView},
    },
    pipeline::image::planar_warp_geometry::PlanarImage,
};

macro_rules! impl_planar_warp_affine_geometry_into {
    ($ty:ty, $planes:literal, $affine:path) => {
        impl PlanarImage<$ty, $planes> {
            /// Applies an affine warp from planar input views into planar output views.
            ///
            /// Prepares all plane producer streams, rejects capture, and waits for successful stream completion.
            /// Each set of planes must retain equal dimensions and byte strides.
            /// Only pixels covered by the transformed source intersection are written; other output pixels and row padding retain their previous initialization state.
            ///
            /// # Errors
            ///
            /// Returns an error if a participating plane belongs to another CUDA context or CUDA context binding fails.
            /// The source ROI must have positive dimensions and representable endpoints.
            /// The destination ROI must lie within each corresponding destination image.
            /// Capture checks, producer synchronization, NPP geometry or interpolation validation, and stream synchronization can also fail.
            #[doc = concat!("Uses [`", stringify!($affine), "`].")]
            pub fn warp_affine_into(
                stream_context: &StreamContext,
                warp: &geometry::WarpAffine,
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
                    .sync_scope(|| unsafe { $affine(stream_context, warp, source, destination) })
            }
        }
    };
}

impl_planar_warp_affine_geometry_into!(f64, 3, geometry::warp_affine_f64_p3);

impl_planar_warp_affine_geometry_into!(f64, 4, geometry::warp_affine_f64_p4);

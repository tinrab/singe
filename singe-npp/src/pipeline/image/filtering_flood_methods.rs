use crate::{
    error::{Error, Result},
    image::{SupportedImage, view::ChannelLayout},
    pipeline::image::{CopyImage, ImagePipeline, filtering::FloodFillImage},
    types::{ConnectedRegion, ImageNormalization, Point},
};

mod range_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: CopyImage<T, L> + FloodFillImage<T, L>,
    <Self as FloodFillImage<T, L>>::Value: Copy,
{
    /// Fills a seed-connected image region with `new_value`.
    ///
    /// Eligible original samples match the original seed pixel value in every channel.
    /// The provided implementations accept C1 or C3 `u8`, `u16`, or `u32` images. C1 uses
    /// scalar values and C3 uses three-element arrays. `ImageNormalization::L1` selects
    /// four-way connectivity; `ImageNormalization::Infinity` selects eight-way connectivity.
    /// The wrapper forwards connectivity and any range parameters without validating their
    /// native restrictions.
    /// Reuses an owned pipeline image in place or copies borrowed input into a new owned
    /// image. Optional `connected_region` receives completed native region information
    /// in host memory. Its bounding-box fields `x`, `y`, `width`, and `height` contain left,
    /// top, right, and bottom coordinates respectively.
    /// Owns temporary device scratch, prepares the image region, waits for its producers
    /// on other streams, rejects capture, and returns after successful stream completion.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image-region
    /// preparation, producer waits, scratch-size queries, allocation, native flood filling,
    /// or synchronization fail. Image dimensions must be positive and `seed` must lie
    /// inside the image. Copying borrowed input or image view construction can also fail.
    /// NPP may reject connectivity, range, or fill parameters.
    pub fn flood_fill(
        self,
        seed: Point,
        new_value: <Self as FloodFillImage<T, L>>::Value,
        norm: ImageNormalization,
        connected_region: Option<&mut ConnectedRegion>,
    ) -> Result<Self> {
        self.flood_fill_operation(
            seed,
            new_value,
            None,
            None,
            None,
            norm,
            connected_region,
            |stream_context,
             image,
             seed,
             new_value,
             _min,
             _max,
             _boundary_value,
             norm,
             connected_region| {
                <Self as FloodFillImage<T, L>>::flood_fill_image(
                    stream_context,
                    image,
                    seed,
                    new_value,
                    norm,
                    connected_region,
                )
            },
        )
    }

    /// Fills a seed-connected image region with `new_value`.
    ///
    /// Eligible original samples match the original seed pixel value in every channel.
    /// Colors the filled region's boundary with `boundary_value`.
    /// The provided implementations accept C1 or C3 `u8`, `u16`, or `u32` images. C1 uses
    /// scalar values and C3 uses three-element arrays. `ImageNormalization::L1` selects
    /// four-way connectivity; `ImageNormalization::Infinity` selects eight-way connectivity.
    /// The wrapper forwards connectivity and any range parameters without validating their
    /// native restrictions.
    /// Reuses an owned pipeline image in place or copies borrowed input into a new owned
    /// image. Optional `connected_region` receives completed native region information
    /// in host memory. Its bounding-box fields `x`, `y`, `width`, and `height` contain left,
    /// top, right, and bottom coordinates respectively.
    /// Owns temporary device scratch, prepares the image region, waits for its producers
    /// on other streams, rejects capture, and returns after successful stream completion.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image-region
    /// preparation, producer waits, scratch-size queries, allocation, native flood filling,
    /// or synchronization fail. Image dimensions must be positive and `seed` must lie
    /// inside the image. Copying borrowed input or image view construction can also fail.
    /// NPP may reject connectivity, range, or fill parameters.
    pub fn flood_fill_boundary(
        self,
        seed: Point,
        new_value: <Self as FloodFillImage<T, L>>::Value,
        boundary_value: <Self as FloodFillImage<T, L>>::Value,
        norm: ImageNormalization,
        connected_region: Option<&mut ConnectedRegion>,
    ) -> Result<Self> {
        self.flood_fill_operation(
            seed,
            new_value,
            None,
            None,
            Some(boundary_value),
            norm,
            connected_region,
            |stream_context,
             image,
             seed,
             new_value,
             _min,
             _max,
             boundary_value,
             norm,
             connected_region| {
                <Self as FloodFillImage<T, L>>::flood_fill_boundary_image(
                    stream_context,
                    image,
                    seed,
                    new_value,
                    boundary_value.ok_or(Error::MissingParameter {
                        name: "boundary_value".into(),
                    })?,
                    norm,
                    connected_region,
                )
            },
        )
    }
}

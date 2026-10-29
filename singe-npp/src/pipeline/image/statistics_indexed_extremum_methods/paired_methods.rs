use singe_cuda::{
    memory::DeviceMemory,
    view::{DeviceOutputView, DeviceRepr},
};

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView},
    pipeline::image::{
        ImagePipeline,
        statistics::{
            ChannelPairIndexedExtremumStatisticImage, ImageIndexedMinMax,
            PairIndexedExtremumStatisticImage,
        },
        statistics_indexed_extremum_methods::image_indexed_min_max,
    },
    types::Point,
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: DeviceRepr,
    L: ChannelLayout,
    Self: PairIndexedExtremumStatisticImage<T, L>,
{
    /// Finds the minimum and maximum values and their source pixel coordinates.
    /// Coordinates are zero-based x/y positions relative to the supplied source view.
    /// The provided implementation processes a C1 image, returning one value and one point
    /// for each extremum.
    /// Each supplied value and `Point` output view must contain exactly one element.
    /// Prepares source and result regions, waits for producers on other streams,
    /// rejects capture, queries and allocates device scratch, and returns after successful
    /// stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native scratch-size queries or allocation, execution, or stream synchronization fail.
    /// All result views must have the exact element counts described above.
    pub fn min_max_index_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        min: &mut DeviceOutputView<'_, T>,
        max: &mut DeviceOutputView<'_, T>,
        min_index: &mut DeviceOutputView<'_, Point>,
        max_index: &mut DeviceOutputView<'_, Point>,
    ) -> Result<()> {
        <Self as PairIndexedExtremumStatisticImage<T, L>>::min_max_index(
            stream_context,
            source,
            min,
            max,
            min_index,
            max_index,
        )
    }

    /// Finds the minimum and maximum values and their source pixel coordinates.
    /// Coordinates are zero-based x/y positions relative to the supplied source view.
    /// The provided implementation processes a C1 image, returning one value and one point
    /// for each extremum.
    /// Allocates owned device results. Downloads the native `Point` results to the host and
    /// uploads separate device x/y arrays for the returned `ImageIndexedMinMax`.
    /// Prepares source and result regions, waits for producers on other streams,
    /// rejects capture, queries and allocates device scratch, and returns after successful
    /// stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native scratch-size queries or allocation, execution, or stream synchronization fail.
    /// Result allocation or source view construction can also fail.
    /// Point downloads and coordinate uploads can also fail.
    pub fn min_max_index(self) -> Result<ImageIndexedMinMax<T>> {
        self.ensure_eager()?;
        let mut min = DeviceMemory::<T>::create_uninit(self.stream_context.cuda_context(), 1)?;
        let mut max = DeviceMemory::<T>::create_uninit(self.stream_context.cuda_context(), 1)?;
        let mut min_index =
            DeviceMemory::<Point>::create_uninit(self.stream_context.cuda_context(), 1)?;
        let mut max_index =
            DeviceMemory::<Point>::create_uninit(self.stream_context.cuda_context(), 1)?;
        {
            let source = self.view()?;
            <Self as PairIndexedExtremumStatisticImage<T, L>>::min_max_index(
                self.stream_context,
                &source,
                &mut DeviceOutputView::from_output(&mut min),
                &mut DeviceOutputView::from_output(&mut max),
                &mut DeviceOutputView::from_output(&mut min_index),
                &mut DeviceOutputView::from_output(&mut max_index),
            )?;
        }
        image_indexed_min_max(
            // SAFETY: the completed operation above wrote every element.
            unsafe { min.assume_init() },
            // SAFETY: the completed operation above wrote every element.
            unsafe { max.assume_init() },
            // SAFETY: the completed operation above wrote every element.
            unsafe { min_index.assume_init() },
            // SAFETY: the completed operation above wrote every element.
            unsafe { max_index.assume_init() },
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: DeviceRepr,
    L: ChannelLayout,
    Self: ChannelPairIndexedExtremumStatisticImage<T, L>,
{
    /// Finds the minimum and maximum values and their source pixel coordinates in one selected channel.
    /// Coordinates are zero-based x/y positions relative to the supplied source view.
    /// The provided implementation selects a C3 channel using zero-based `channel` in
    /// `0..3`, returning one value and one point for each extremum.
    /// Each supplied value and `Point` output view must contain exactly one element.
    /// Prepares source and result regions, waits for producers on other streams,
    /// rejects capture, queries and allocates device scratch, and returns after successful
    /// stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native scratch-size queries or allocation, execution, or stream synchronization fail.
    /// All result views must have the exact element counts described above.
    /// `channel` must be less than three.
    pub fn min_max_index_channel_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        channel: usize,
        min: &mut DeviceOutputView<'_, T>,
        max: &mut DeviceOutputView<'_, T>,
        min_index: &mut DeviceOutputView<'_, Point>,
        max_index: &mut DeviceOutputView<'_, Point>,
    ) -> Result<()> {
        <Self as ChannelPairIndexedExtremumStatisticImage<T, L>>::min_max_index_channel(
            stream_context,
            source,
            channel,
            min,
            max,
            min_index,
            max_index,
        )
    }

    /// Finds the minimum and maximum values and their source pixel coordinates in one selected channel.
    /// Coordinates are zero-based x/y positions relative to the supplied source view.
    /// The provided implementation selects a C3 channel using zero-based `channel` in
    /// `0..3`, returning one value and one point for each extremum.
    /// Allocates owned device results. Downloads the native `Point` results to the host and
    /// uploads separate device x/y arrays for the returned `ImageIndexedMinMax`.
    /// Prepares source and result regions, waits for producers on other streams,
    /// rejects capture, queries and allocates device scratch, and returns after successful
    /// stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native scratch-size queries or allocation, execution, or stream synchronization fail.
    /// Result allocation or source view construction can also fail.
    /// Point downloads and coordinate uploads can also fail.
    /// `channel` must be less than three.
    pub fn min_max_index_channel(self, channel: usize) -> Result<ImageIndexedMinMax<T>> {
        self.ensure_eager()?;
        let mut min = DeviceMemory::<T>::create_uninit(self.stream_context.cuda_context(), 1)?;
        let mut max = DeviceMemory::<T>::create_uninit(self.stream_context.cuda_context(), 1)?;
        let mut min_index =
            DeviceMemory::<Point>::create_uninit(self.stream_context.cuda_context(), 1)?;
        let mut max_index =
            DeviceMemory::<Point>::create_uninit(self.stream_context.cuda_context(), 1)?;
        {
            let source = self.view()?;
            <Self as ChannelPairIndexedExtremumStatisticImage<T, L>>::min_max_index_channel(
                self.stream_context,
                &source,
                channel,
                &mut DeviceOutputView::from_output(&mut min),
                &mut DeviceOutputView::from_output(&mut max),
                &mut DeviceOutputView::from_output(&mut min_index),
                &mut DeviceOutputView::from_output(&mut max_index),
            )?;
        }
        image_indexed_min_max(
            // SAFETY: the completed operation above wrote every element.
            unsafe { min.assume_init() },
            // SAFETY: the completed operation above wrote every element.
            unsafe { max.assume_init() },
            // SAFETY: the completed operation above wrote every element.
            unsafe { min_index.assume_init() },
            // SAFETY: the completed operation above wrote every element.
            unsafe { max_index.assume_init() },
        )
    }
}

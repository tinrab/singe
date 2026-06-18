use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageView, ImageViewMut},
};

pub(in crate::pipeline::image) type DeviceConstantOperation<T, L, C> =
    for<'source, 'destination> fn(
        &StreamContext,
        &ImageView<'source, T, L>,
        &DeviceMemory<C>,
        &mut ImageViewMut<'destination, T, L>,
    ) -> Result<()>;

pub(in crate::pipeline::image) type DeviceConstantInPlaceOperation<T, L, C> =
    for<'destination> fn(
        &StreamContext,
        &DeviceMemory<C>,
        &mut ImageViewMut<'destination, T, L>,
    ) -> Result<()>;

pub(in crate::pipeline::image) type DeviceConstantScaleOperation<T, L, C> =
    for<'source, 'destination> fn(
        &StreamContext,
        &ImageView<'source, T, L>,
        &DeviceMemory<C>,
        &mut ImageViewMut<'destination, T, L>,
        i32,
    ) -> Result<()>;

pub(in crate::pipeline::image) type DeviceConstantScaleInPlaceOperation<T, L, C> =
    for<'destination> fn(
        &StreamContext,
        &DeviceMemory<C>,
        &mut ImageViewMut<'destination, T, L>,
        i32,
    ) -> Result<()>;

use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageOutputView, ImageView, ImageViewMut},
};

pub type DeviceConstantOperation<T, L, C> = for<'source, 'destination> fn(
    &StreamContext,
    &ImageView<'source, T, L>,
    &DeviceMemory<C>,
    &mut ImageOutputView<'destination, T, L>,
) -> Result<()>;

pub type DeviceConstantInPlaceOperation<T, L, C> = for<'destination> fn(
    &StreamContext,
    &DeviceMemory<C>,
    &mut ImageViewMut<'destination, T, L>,
) -> Result<()>;

pub type DeviceConstantScaleOperation<T, L, C> = for<'source, 'destination> fn(
    &StreamContext,
    &ImageView<'source, T, L>,
    &DeviceMemory<C>,
    &mut ImageOutputView<'destination, T, L>,
    i32,
) -> Result<()>;

pub type DeviceConstantScaleInPlaceOperation<T, L, C> = for<'destination> fn(
    &StreamContext,
    &DeviceMemory<C>,
    &mut ImageViewMut<'destination, T, L>,
    i32,
) -> Result<()>;

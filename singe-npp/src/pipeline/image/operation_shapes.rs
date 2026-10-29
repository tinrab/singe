use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageOutputView, ImageView, ImageViewMut},
    types::RoundMode,
};

mod device_constant_shapes;

pub use device_constant_shapes::{
    DeviceConstantInPlaceOperation, DeviceConstantOperation, DeviceConstantScaleInPlaceOperation,
    DeviceConstantScaleOperation,
};

pub type BinaryImageOperation<T, L> = for<'left, 'right, 'destination> fn(
    &StreamContext,
    &ImageView<'left, T, L>,
    &ImageView<'right, T, L>,
    &mut ImageOutputView<'destination, T, L>,
) -> Result<()>;

pub type BinaryImageInPlaceOperation<T, L> = for<'source, 'destination> fn(
    &StreamContext,
    &ImageView<'source, T, L>,
    &mut ImageViewMut<'destination, T, L>,
) -> Result<()>;

pub type RoundedBinaryImageOperation<T, L> = for<'left, 'right, 'destination> fn(
    &StreamContext,
    &ImageView<'left, T, L>,
    &ImageView<'right, T, L>,
    &mut ImageOutputView<'destination, T, L>,
    i32,
    RoundMode,
) -> Result<()>;

pub type RoundedBinaryImageInPlaceOperation<T, L> = for<'source, 'destination> fn(
    &StreamContext,
    &ImageView<'source, T, L>,
    &mut ImageViewMut<'destination, T, L>,
    i32,
    RoundMode,
) -> Result<()>;

pub type ConstantShiftImageOperation<T, L, C> = for<'source, 'destination> fn(
    &StreamContext,
    &ImageView<'source, T, L>,
    C,
    &mut ImageOutputView<'destination, T, L>,
) -> Result<()>;

pub type ConstantOperation<T, L, C> = for<'source, 'destination> fn(
    &StreamContext,
    &ImageView<'source, T, L>,
    C,
    &mut ImageOutputView<'destination, T, L>,
) -> Result<()>;

pub type ConstantInPlaceOperation<T, L, C> =
    for<'destination> fn(&StreamContext, C, &mut ImageViewMut<'destination, T, L>) -> Result<()>;

pub type ConstantScaleOperation<T, L, C> = for<'source, 'destination> fn(
    &StreamContext,
    &ImageView<'source, T, L>,
    C,
    &mut ImageOutputView<'destination, T, L>,
    i32,
) -> Result<()>;

pub type ConstantScaleInPlaceOperation<T, L, C> = for<'destination> fn(
    &StreamContext,
    C,
    &mut ImageViewMut<'destination, T, L>,
    i32,
) -> Result<()>;

pub type ScaledUnaryImageOperation<T, L> = for<'source, 'destination> fn(
    &StreamContext,
    &ImageView<'source, T, L>,
    &mut ImageOutputView<'destination, T, L>,
    i32,
) -> Result<()>;

pub type ScaledUnaryPowerImageOperation<T, L> = for<'source, 'destination> fn(
    &StreamContext,
    &ImageView<'source, T, L>,
    &mut ImageOutputView<'destination, T, L>,
    i32,
) -> Result<()>;

pub type ScaledBinaryImageOperation<T, L> = for<'left, 'right, 'destination> fn(
    &StreamContext,
    &ImageView<'left, T, L>,
    &ImageView<'right, T, L>,
    &mut ImageOutputView<'destination, T, L>,
    i32,
) -> Result<()>;

pub type ScaledBinaryImageInPlaceOperation<T, L> = for<'source, 'destination> fn(
    &StreamContext,
    &ImageView<'source, T, L>,
    &mut ImageViewMut<'destination, T, L>,
    i32,
) -> Result<()>;

pub type OutOfPlaceBinaryImageOperation<T, L> = for<'left, 'right, 'destination> fn(
    &StreamContext,
    &ImageView<'left, T, L>,
    &ImageView<'right, T, L>,
    &mut ImageOutputView<'destination, T, L>,
) -> Result<()>;

pub type ScaledSubtractImageOperation<T, L> = for<'left, 'right, 'destination> fn(
    &StreamContext,
    &ImageView<'left, T, L>,
    &ImageView<'right, T, L>,
    &mut ImageOutputView<'destination, T, L>,
    i32,
) -> Result<()>;

pub type ScaledSubtractImageInPlaceOperation<T, L> = for<'source, 'destination> fn(
    &StreamContext,
    &ImageView<'source, T, L>,
    &mut ImageViewMut<'destination, T, L>,
    i32,
) -> Result<()>;

pub type UnscaledBinaryImageOperation<T, L> = for<'left, 'right, 'destination> fn(
    &StreamContext,
    &ImageView<'left, T, L>,
    &ImageView<'right, T, L>,
    &mut ImageOutputView<'destination, T, L>,
) -> Result<()>;

pub type UnscaledBinaryImageInPlaceOperation<T, L> = for<'source, 'destination> fn(
    &StreamContext,
    &ImageView<'source, T, L>,
    &mut ImageViewMut<'destination, T, L>,
) -> Result<()>;

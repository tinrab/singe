use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageView, ImageViewMut},
    types::RoundMode,
};

#[path = "operation_device_constant_shapes.rs"]
mod device_constant_shapes;

pub(super) use device_constant_shapes::{
    DeviceConstantInPlaceOperation, DeviceConstantOperation, DeviceConstantScaleInPlaceOperation,
    DeviceConstantScaleOperation,
};

pub(super) type BinaryImageOperation<T, L> = for<'left, 'right, 'destination> fn(
    &StreamContext,
    &ImageView<'left, T, L>,
    &ImageView<'right, T, L>,
    &mut ImageViewMut<'destination, T, L>,
) -> Result<()>;

pub(super) type BinaryImageInPlaceOperation<T, L> = for<'source, 'destination> fn(
    &StreamContext,
    &ImageView<'source, T, L>,
    &mut ImageViewMut<'destination, T, L>,
) -> Result<()>;

pub(super) type RoundedBinaryImageOperation<T, L> =
    for<'left, 'right, 'destination> fn(
        &StreamContext,
        &ImageView<'left, T, L>,
        &ImageView<'right, T, L>,
        &mut ImageViewMut<'destination, T, L>,
        i32,
        RoundMode,
    ) -> Result<()>;

pub(super) type RoundedBinaryImageInPlaceOperation<T, L> =
    for<'source, 'destination> fn(
        &StreamContext,
        &ImageView<'source, T, L>,
        &mut ImageViewMut<'destination, T, L>,
        i32,
        RoundMode,
    ) -> Result<()>;

pub(super) type ConstantShiftImageOperation<T, L, C> = for<'source, 'destination> fn(
    &StreamContext,
    &ImageView<'source, T, L>,
    C,
    &mut ImageViewMut<'destination, T, L>,
) -> Result<()>;

pub(super) type ConstantOperation<T, L, C> = for<'source, 'destination> fn(
    &StreamContext,
    &ImageView<'source, T, L>,
    C,
    &mut ImageViewMut<'destination, T, L>,
) -> Result<()>;

pub(super) type ConstantInPlaceOperation<T, L, C> =
    for<'destination> fn(&StreamContext, C, &mut ImageViewMut<'destination, T, L>) -> Result<()>;

pub(super) type ConstantScaleOperation<T, L, C> = for<'source, 'destination> fn(
    &StreamContext,
    &ImageView<'source, T, L>,
    C,
    &mut ImageViewMut<'destination, T, L>,
    i32,
) -> Result<()>;

pub(super) type ConstantScaleInPlaceOperation<T, L, C> = for<'destination> fn(
    &StreamContext,
    C,
    &mut ImageViewMut<'destination, T, L>,
    i32,
) -> Result<()>;

pub(super) type ScaledUnaryImageOperation<T, L> = for<'source, 'destination> fn(
    &StreamContext,
    &ImageView<'source, T, L>,
    &mut ImageViewMut<'destination, T, L>,
    i32,
) -> Result<()>;

pub(super) type ScaledUnaryPowerImageOperation<T, L> = for<'source, 'destination> fn(
    &StreamContext,
    &ImageView<'source, T, L>,
    &mut ImageViewMut<'destination, T, L>,
    i32,
) -> Result<()>;

pub(super) type ScaledBinaryImageOperation<T, L> = for<'left, 'right, 'destination> fn(
    &StreamContext,
    &ImageView<'left, T, L>,
    &ImageView<'right, T, L>,
    &mut ImageViewMut<'destination, T, L>,
    i32,
)
    -> Result<()>;

pub(super) type ScaledBinaryImageInPlaceOperation<T, L> =
    for<'source, 'destination> fn(
        &StreamContext,
        &ImageView<'source, T, L>,
        &mut ImageViewMut<'destination, T, L>,
        i32,
    ) -> Result<()>;

pub(super) type OutOfPlaceBinaryImageOperation<T, L> =
    for<'left, 'right, 'destination> fn(
        &StreamContext,
        &ImageView<'left, T, L>,
        &ImageView<'right, T, L>,
        &mut ImageViewMut<'destination, T, L>,
    ) -> Result<()>;

pub(super) type ScaledSubtractImageOperation<T, L> =
    for<'left, 'right, 'destination> fn(
        &StreamContext,
        &ImageView<'left, T, L>,
        &ImageView<'right, T, L>,
        &mut ImageViewMut<'destination, T, L>,
        i32,
    ) -> Result<()>;

pub(super) type ScaledSubtractImageInPlaceOperation<T, L> =
    for<'source, 'destination> fn(
        &StreamContext,
        &ImageView<'source, T, L>,
        &mut ImageViewMut<'destination, T, L>,
        i32,
    ) -> Result<()>;

pub(super) type UnscaledBinaryImageOperation<T, L> =
    for<'left, 'right, 'destination> fn(
        &StreamContext,
        &ImageView<'left, T, L>,
        &ImageView<'right, T, L>,
        &mut ImageViewMut<'destination, T, L>,
    ) -> Result<()>;

pub(super) type UnscaledBinaryImageInPlaceOperation<T, L> =
    for<'source, 'destination> fn(
        &StreamContext,
        &ImageView<'source, T, L>,
        &mut ImageViewMut<'destination, T, L>,
    ) -> Result<()>;

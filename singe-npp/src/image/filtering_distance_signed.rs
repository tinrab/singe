use super::*;

pub trait SignedDistanceTransformPbaToF32C1: DataTypeLike {
    type Manhattan: DataTypeLike;

    fn signed_distance_transform_pba_to_f32(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        cutoff_value: Self,
        subpixel_x_shift: f32,
        subpixel_y_shift: f32,
        destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageViewMut<'_, f32, C1>>,
    ) -> Result<()>;
}

impl SignedDistanceTransformPbaToF32C1 for f32 {
    type Manhattan = i16;

    fn signed_distance_transform_pba_to_f32(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        cutoff_value: Self,
        subpixel_x_shift: f32,
        subpixel_y_shift: f32,
        destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageViewMut<'_, f32, C1>>,
    ) -> Result<()> {
        signed_distance_transform_pba_f32_c1(
            stream_context,
            source,
            cutoff_value,
            subpixel_x_shift,
            subpixel_y_shift,
            destination_voronoi,
            destination_voronoi_indices,
            destination_voronoi_manhattan,
            destination_transform,
        )
    }
}

pub fn signed_distance_transform_pba_to_f32<T: SignedDistanceTransformPbaToF32C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    cutoff_value: T,
    subpixel_x_shift: f32,
    subpixel_y_shift: f32,
    destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageViewMut<'_, f32, C1>>,
) -> Result<()> {
    T::signed_distance_transform_pba_to_f32(
        stream_context,
        source,
        cutoff_value,
        subpixel_x_shift,
        subpixel_y_shift,
        destination_voronoi,
        destination_voronoi_indices,
        destination_voronoi_manhattan,
        destination_transform,
    )
}

pub trait SignedDistanceTransformAbsPbaToF32C1: DataTypeLike {
    type Manhattan: DataTypeLike;

    fn signed_distance_transform_abs_pba_to_f32(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        cutoff_value: Self,
        subpixel_x_shift: f32,
        subpixel_y_shift: f32,
        destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageViewMut<'_, f32, C1>>,
    ) -> Result<()>;
}

impl SignedDistanceTransformAbsPbaToF32C1 for f32 {
    type Manhattan = u16;

    fn signed_distance_transform_abs_pba_to_f32(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        cutoff_value: Self,
        subpixel_x_shift: f32,
        subpixel_y_shift: f32,
        destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageViewMut<'_, f32, C1>>,
    ) -> Result<()> {
        signed_distance_transform_abs_pba_f32_c1(
            stream_context,
            source,
            cutoff_value,
            subpixel_x_shift,
            subpixel_y_shift,
            destination_voronoi,
            destination_voronoi_indices,
            destination_voronoi_manhattan,
            destination_transform,
        )
    }
}

pub fn signed_distance_transform_abs_pba_to_f32<T: SignedDistanceTransformAbsPbaToF32C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    cutoff_value: T,
    subpixel_x_shift: f32,
    subpixel_y_shift: f32,
    destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageViewMut<'_, f32, C1>>,
) -> Result<()> {
    T::signed_distance_transform_abs_pba_to_f32(
        stream_context,
        source,
        cutoff_value,
        subpixel_x_shift,
        subpixel_y_shift,
        destination_voronoi,
        destination_voronoi_indices,
        destination_voronoi_manhattan,
        destination_transform,
    )
}

pub trait SignedDistanceTransformPbaToF64C1: DataTypeLike {
    type Manhattan: DataTypeLike;

    fn signed_distance_transform_pba_to_f64(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        cutoff_value: Self,
        subpixel_x_shift: f64,
        subpixel_y_shift: f64,
        destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageViewMut<'_, f64, C1>>,
    ) -> Result<()>;

    fn signed_distance_transform_pba_to_f64_antialiasing(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        cutoff_value: Self,
        subpixel_x_shift: f64,
        subpixel_y_shift: f64,
        destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageViewMut<'_, f64, C1>>,
    ) -> Result<()>;
}

macro_rules! impl_signed_distance_transform_pba_to_f64 {
    ($source_ty:ty, $manhattan_ty:ty, $name:ident, $antialiasing_name:ident) => {
        impl SignedDistanceTransformPbaToF64C1 for $source_ty {
            type Manhattan = $manhattan_ty;

            fn signed_distance_transform_pba_to_f64(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                cutoff_value: Self,
                subpixel_x_shift: f64,
                subpixel_y_shift: f64,
                destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
                destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
                destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
                destination_transform: Option<&mut ImageViewMut<'_, f64, C1>>,
            ) -> Result<()> {
                $name(
                    stream_context,
                    source,
                    cutoff_value,
                    subpixel_x_shift,
                    subpixel_y_shift,
                    destination_voronoi,
                    destination_voronoi_indices,
                    destination_voronoi_manhattan,
                    destination_transform,
                )
            }

            fn signed_distance_transform_pba_to_f64_antialiasing(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                cutoff_value: Self,
                subpixel_x_shift: f64,
                subpixel_y_shift: f64,
                destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
                destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
                destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
                destination_transform: Option<&mut ImageViewMut<'_, f64, C1>>,
            ) -> Result<()> {
                $antialiasing_name(
                    stream_context,
                    source,
                    cutoff_value,
                    subpixel_x_shift,
                    subpixel_y_shift,
                    destination_voronoi,
                    destination_voronoi_indices,
                    destination_voronoi_manhattan,
                    destination_transform,
                )
            }
        }
    };
}

impl_signed_distance_transform_pba_to_f64!(
    f32,
    i16,
    signed_distance_transform_pba_f32_to_f64_c1,
    signed_distance_transform_pba_f32_to_f64_c1_antialiasing
);
impl_signed_distance_transform_pba_to_f64!(
    f64,
    i16,
    signed_distance_transform_pba_f64_c1,
    signed_distance_transform_pba_f64_c1_antialiasing
);

pub fn signed_distance_transform_pba_to_f64<T: SignedDistanceTransformPbaToF64C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    cutoff_value: T,
    subpixel_x_shift: f64,
    subpixel_y_shift: f64,
    destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageViewMut<'_, f64, C1>>,
) -> Result<()> {
    T::signed_distance_transform_pba_to_f64(
        stream_context,
        source,
        cutoff_value,
        subpixel_x_shift,
        subpixel_y_shift,
        destination_voronoi,
        destination_voronoi_indices,
        destination_voronoi_manhattan,
        destination_transform,
    )
}

pub fn signed_distance_transform_pba_to_f64_antialiasing<T: SignedDistanceTransformPbaToF64C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    cutoff_value: T,
    subpixel_x_shift: f64,
    subpixel_y_shift: f64,
    destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageViewMut<'_, f64, C1>>,
) -> Result<()> {
    T::signed_distance_transform_pba_to_f64_antialiasing(
        stream_context,
        source,
        cutoff_value,
        subpixel_x_shift,
        subpixel_y_shift,
        destination_voronoi,
        destination_voronoi_indices,
        destination_voronoi_manhattan,
        destination_transform,
    )
}

pub trait SignedDistanceTransformAbsPbaToF64C1: DataTypeLike {
    type Manhattan: DataTypeLike;

    fn signed_distance_transform_abs_pba_to_f64(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        cutoff_value: Self,
        subpixel_x_shift: f64,
        subpixel_y_shift: f64,
        destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageViewMut<'_, f64, C1>>,
    ) -> Result<()>;

    fn signed_distance_transform_abs_pba_to_f64_antialiasing(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        cutoff_value: Self,
        subpixel_x_shift: f64,
        subpixel_y_shift: f64,
        destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageViewMut<'_, f64, C1>>,
    ) -> Result<()>;
}

macro_rules! impl_signed_distance_transform_abs_pba_to_f64 {
    ($source_ty:ty, $manhattan_ty:ty, $name:ident, $antialiasing_name:ident) => {
        impl SignedDistanceTransformAbsPbaToF64C1 for $source_ty {
            type Manhattan = $manhattan_ty;

            fn signed_distance_transform_abs_pba_to_f64(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                cutoff_value: Self,
                subpixel_x_shift: f64,
                subpixel_y_shift: f64,
                destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
                destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
                destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
                destination_transform: Option<&mut ImageViewMut<'_, f64, C1>>,
            ) -> Result<()> {
                $name(
                    stream_context,
                    source,
                    cutoff_value,
                    subpixel_x_shift,
                    subpixel_y_shift,
                    destination_voronoi,
                    destination_voronoi_indices,
                    destination_voronoi_manhattan,
                    destination_transform,
                )
            }

            fn signed_distance_transform_abs_pba_to_f64_antialiasing(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                cutoff_value: Self,
                subpixel_x_shift: f64,
                subpixel_y_shift: f64,
                destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
                destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
                destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
                destination_transform: Option<&mut ImageViewMut<'_, f64, C1>>,
            ) -> Result<()> {
                $antialiasing_name(
                    stream_context,
                    source,
                    cutoff_value,
                    subpixel_x_shift,
                    subpixel_y_shift,
                    destination_voronoi,
                    destination_voronoi_indices,
                    destination_voronoi_manhattan,
                    destination_transform,
                )
            }
        }
    };
}

impl_signed_distance_transform_abs_pba_to_f64!(
    f32,
    u16,
    signed_distance_transform_abs_pba_f32_to_f64_c1,
    signed_distance_transform_abs_pba_f32_to_f64_c1_antialiasing
);
impl_signed_distance_transform_abs_pba_to_f64!(
    f64,
    u16,
    signed_distance_transform_abs_pba_f64_c1,
    signed_distance_transform_abs_pba_f64_c1_antialiasing
);

pub fn signed_distance_transform_abs_pba_to_f64<T: SignedDistanceTransformAbsPbaToF64C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    cutoff_value: T,
    subpixel_x_shift: f64,
    subpixel_y_shift: f64,
    destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageViewMut<'_, f64, C1>>,
) -> Result<()> {
    T::signed_distance_transform_abs_pba_to_f64(
        stream_context,
        source,
        cutoff_value,
        subpixel_x_shift,
        subpixel_y_shift,
        destination_voronoi,
        destination_voronoi_indices,
        destination_voronoi_manhattan,
        destination_transform,
    )
}

pub fn signed_distance_transform_abs_pba_to_f64_antialiasing<
    T: SignedDistanceTransformAbsPbaToF64C1,
>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    cutoff_value: T,
    subpixel_x_shift: f64,
    subpixel_y_shift: f64,
    destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageViewMut<'_, f64, C1>>,
) -> Result<()> {
    T::signed_distance_transform_abs_pba_to_f64_antialiasing(
        stream_context,
        source,
        cutoff_value,
        subpixel_x_shift,
        subpixel_y_shift,
        destination_voronoi,
        destination_voronoi_indices,
        destination_voronoi_manhattan,
        destination_transform,
    )
}

pub trait SignedDistanceTransformPbaC1<Destination>: DataTypeLike
where
    Destination: DataTypeLike,
{
    type Manhattan: DataTypeLike;
    type Shift: DataTypeLike;

    fn signed_distance_transform_pba_c1(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        cutoff_value: Self,
        subpixel_x_shift: Self::Shift,
        subpixel_y_shift: Self::Shift,
        destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageViewMut<'_, Destination, C1>>,
    ) -> Result<()>;
}

macro_rules! impl_signed_distance_transform_pba_c1 {
    ($destination:ty, $shift:ty, $source_trait:ident, $source_method:ident) => {
        impl<T> SignedDistanceTransformPbaC1<$destination> for T
        where
            T: $source_trait,
        {
            type Manhattan = <T as $source_trait>::Manhattan;
            type Shift = $shift;

            fn signed_distance_transform_pba_c1(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                cutoff_value: Self,
                subpixel_x_shift: Self::Shift,
                subpixel_y_shift: Self::Shift,
                destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
                destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
                destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
                destination_transform: Option<&mut ImageViewMut<'_, $destination, C1>>,
            ) -> Result<()> {
                T::$source_method(
                    stream_context,
                    source,
                    cutoff_value,
                    subpixel_x_shift,
                    subpixel_y_shift,
                    destination_voronoi,
                    destination_voronoi_indices,
                    destination_voronoi_manhattan,
                    destination_transform,
                )
            }
        }
    };
}

impl_signed_distance_transform_pba_c1!(
    f32,
    f32,
    SignedDistanceTransformPbaToF32C1,
    signed_distance_transform_pba_to_f32
);
impl_signed_distance_transform_pba_c1!(
    f64,
    f64,
    SignedDistanceTransformPbaToF64C1,
    signed_distance_transform_pba_to_f64
);

pub fn signed_distance_transform_pba<T, Destination>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    cutoff_value: T,
    subpixel_x_shift: T::Shift,
    subpixel_y_shift: T::Shift,
    destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageViewMut<'_, Destination, C1>>,
) -> Result<()>
where
    T: SignedDistanceTransformPbaC1<Destination>,
    Destination: DataTypeLike,
{
    T::signed_distance_transform_pba_c1(
        stream_context,
        source,
        cutoff_value,
        subpixel_x_shift,
        subpixel_y_shift,
        destination_voronoi,
        destination_voronoi_indices,
        destination_voronoi_manhattan,
        destination_transform,
    )
}

pub fn signed_distance_transform_pba_antialiasing<T>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    cutoff_value: T,
    subpixel_x_shift: f64,
    subpixel_y_shift: f64,
    destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageViewMut<'_, f64, C1>>,
) -> Result<()>
where
    T: SignedDistanceTransformPbaToF64C1,
{
    signed_distance_transform_pba_to_f64_antialiasing(
        stream_context,
        source,
        cutoff_value,
        subpixel_x_shift,
        subpixel_y_shift,
        destination_voronoi,
        destination_voronoi_indices,
        destination_voronoi_manhattan,
        destination_transform,
    )
}

pub trait SignedDistanceTransformAbsPbaC1<Destination>: DataTypeLike
where
    Destination: DataTypeLike,
{
    type Manhattan: DataTypeLike;
    type Shift: DataTypeLike;

    fn signed_distance_transform_abs_pba_c1(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        cutoff_value: Self,
        subpixel_x_shift: Self::Shift,
        subpixel_y_shift: Self::Shift,
        destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageViewMut<'_, Destination, C1>>,
    ) -> Result<()>;
}

macro_rules! impl_signed_distance_transform_abs_pba_c1 {
    ($destination:ty, $shift:ty, $source_trait:ident, $source_method:ident) => {
        impl<T> SignedDistanceTransformAbsPbaC1<$destination> for T
        where
            T: $source_trait,
        {
            type Manhattan = <T as $source_trait>::Manhattan;
            type Shift = $shift;

            fn signed_distance_transform_abs_pba_c1(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                cutoff_value: Self,
                subpixel_x_shift: Self::Shift,
                subpixel_y_shift: Self::Shift,
                destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
                destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
                destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
                destination_transform: Option<&mut ImageViewMut<'_, $destination, C1>>,
            ) -> Result<()> {
                T::$source_method(
                    stream_context,
                    source,
                    cutoff_value,
                    subpixel_x_shift,
                    subpixel_y_shift,
                    destination_voronoi,
                    destination_voronoi_indices,
                    destination_voronoi_manhattan,
                    destination_transform,
                )
            }
        }
    };
}

impl_signed_distance_transform_abs_pba_c1!(
    f32,
    f32,
    SignedDistanceTransformAbsPbaToF32C1,
    signed_distance_transform_abs_pba_to_f32
);
impl_signed_distance_transform_abs_pba_c1!(
    f64,
    f64,
    SignedDistanceTransformAbsPbaToF64C1,
    signed_distance_transform_abs_pba_to_f64
);

pub fn signed_distance_transform_abs_pba<T, Destination>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    cutoff_value: T,
    subpixel_x_shift: T::Shift,
    subpixel_y_shift: T::Shift,
    destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageViewMut<'_, Destination, C1>>,
) -> Result<()>
where
    T: SignedDistanceTransformAbsPbaC1<Destination>,
    Destination: DataTypeLike,
{
    T::signed_distance_transform_abs_pba_c1(
        stream_context,
        source,
        cutoff_value,
        subpixel_x_shift,
        subpixel_y_shift,
        destination_voronoi,
        destination_voronoi_indices,
        destination_voronoi_manhattan,
        destination_transform,
    )
}

pub fn signed_distance_transform_abs_pba_antialiasing<T>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    cutoff_value: T,
    subpixel_x_shift: f64,
    subpixel_y_shift: f64,
    destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageViewMut<'_, f64, C1>>,
) -> Result<()>
where
    T: SignedDistanceTransformAbsPbaToF64C1,
{
    signed_distance_transform_abs_pba_to_f64_antialiasing(
        stream_context,
        source,
        cutoff_value,
        subpixel_x_shift,
        subpixel_y_shift,
        destination_voronoi,
        destination_voronoi_indices,
        destination_voronoi_manhattan,
        destination_transform,
    )
}

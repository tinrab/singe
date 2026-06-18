use super::*;

pub trait DistanceTransformPbaToU16C1: DataTypeLike {
    type Manhattan: DataTypeLike;

    fn distance_transform_pba_to_u16(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        minimum_site_value: Self,
        maximum_site_value: Self,
        destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageViewMut<'_, u16, C1>>,
    ) -> Result<()>;
}

macro_rules! impl_distance_transform_pba_to_u16 {
    ($source_ty:ty, $manhattan_ty:ty, $name:ident) => {
        impl DistanceTransformPbaToU16C1 for $source_ty {
            type Manhattan = $manhattan_ty;

            fn distance_transform_pba_to_u16(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                minimum_site_value: Self,
                maximum_site_value: Self,
                destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
                destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
                destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
                destination_transform: Option<&mut ImageViewMut<'_, u16, C1>>,
            ) -> Result<()> {
                $name(
                    stream_context,
                    source,
                    minimum_site_value,
                    maximum_site_value,
                    destination_voronoi,
                    destination_voronoi_indices,
                    destination_voronoi_manhattan,
                    destination_transform,
                )
            }
        }
    };
}

impl_distance_transform_pba_to_u16!(u8, i16, distance_transform_pba_u8_to_u16_c1);
impl_distance_transform_pba_to_u16!(i8, i16, distance_transform_pba_i8_to_u16_c1);
impl_distance_transform_pba_to_u16!(u16, i16, distance_transform_pba_u16_to_u16_c1);
impl_distance_transform_pba_to_u16!(i16, i16, distance_transform_pba_i16_to_u16_c1);

pub fn distance_transform_pba_to_u16<T: DistanceTransformPbaToU16C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    minimum_site_value: T,
    maximum_site_value: T,
    destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageViewMut<'_, u16, C1>>,
) -> Result<()> {
    T::distance_transform_pba_to_u16(
        stream_context,
        source,
        minimum_site_value,
        maximum_site_value,
        destination_voronoi,
        destination_voronoi_indices,
        destination_voronoi_manhattan,
        destination_transform,
    )
}

pub trait DistanceTransformAbsPbaToU16C1: DataTypeLike {
    type Manhattan: DataTypeLike;

    fn distance_transform_abs_pba_to_u16(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        minimum_site_value: Self,
        maximum_site_value: Self,
        destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageViewMut<'_, u16, C1>>,
    ) -> Result<()>;
}

macro_rules! impl_distance_transform_abs_pba_to_u16 {
    ($source_ty:ty, $manhattan_ty:ty, $name:ident) => {
        impl DistanceTransformAbsPbaToU16C1 for $source_ty {
            type Manhattan = $manhattan_ty;

            fn distance_transform_abs_pba_to_u16(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                minimum_site_value: Self,
                maximum_site_value: Self,
                destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
                destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
                destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
                destination_transform: Option<&mut ImageViewMut<'_, u16, C1>>,
            ) -> Result<()> {
                $name(
                    stream_context,
                    source,
                    minimum_site_value,
                    maximum_site_value,
                    destination_voronoi,
                    destination_voronoi_indices,
                    destination_voronoi_manhattan,
                    destination_transform,
                )
            }
        }
    };
}

impl_distance_transform_abs_pba_to_u16!(u8, u16, distance_transform_abs_pba_u8_to_u16_c1);
impl_distance_transform_abs_pba_to_u16!(i8, u16, distance_transform_abs_pba_i8_to_u16_c1);
impl_distance_transform_abs_pba_to_u16!(u16, u16, distance_transform_abs_pba_u16_to_u16_c1);
impl_distance_transform_abs_pba_to_u16!(i16, u16, distance_transform_abs_pba_i16_to_u16_c1);

pub fn distance_transform_abs_pba_to_u16<T: DistanceTransformAbsPbaToU16C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    minimum_site_value: T,
    maximum_site_value: T,
    destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageViewMut<'_, u16, C1>>,
) -> Result<()> {
    T::distance_transform_abs_pba_to_u16(
        stream_context,
        source,
        minimum_site_value,
        maximum_site_value,
        destination_voronoi,
        destination_voronoi_indices,
        destination_voronoi_manhattan,
        destination_transform,
    )
}

pub trait DistanceTransformPbaToF32C1: DataTypeLike {
    type Manhattan: DataTypeLike;

    fn distance_transform_pba_to_f32(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        minimum_site_value: Self,
        maximum_site_value: Self,
        destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageViewMut<'_, f32, C1>>,
    ) -> Result<()>;
}

macro_rules! impl_distance_transform_pba_to_f32 {
    ($source_ty:ty, $manhattan_ty:ty, $name:ident) => {
        impl DistanceTransformPbaToF32C1 for $source_ty {
            type Manhattan = $manhattan_ty;

            fn distance_transform_pba_to_f32(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                minimum_site_value: Self,
                maximum_site_value: Self,
                destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
                destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
                destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
                destination_transform: Option<&mut ImageViewMut<'_, f32, C1>>,
            ) -> Result<()> {
                $name(
                    stream_context,
                    source,
                    minimum_site_value,
                    maximum_site_value,
                    destination_voronoi,
                    destination_voronoi_indices,
                    destination_voronoi_manhattan,
                    destination_transform,
                )
            }
        }
    };
}

impl_distance_transform_pba_to_f32!(u8, i16, distance_transform_pba_u8_to_f32_c1);
impl_distance_transform_pba_to_f32!(i8, i16, distance_transform_pba_i8_to_f32_c1);
impl_distance_transform_pba_to_f32!(u16, i16, distance_transform_pba_u16_to_f32_c1);
impl_distance_transform_pba_to_f32!(i16, i16, distance_transform_pba_i16_to_f32_c1);

pub fn distance_transform_pba_to_f32<T: DistanceTransformPbaToF32C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    minimum_site_value: T,
    maximum_site_value: T,
    destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageViewMut<'_, f32, C1>>,
) -> Result<()> {
    T::distance_transform_pba_to_f32(
        stream_context,
        source,
        minimum_site_value,
        maximum_site_value,
        destination_voronoi,
        destination_voronoi_indices,
        destination_voronoi_manhattan,
        destination_transform,
    )
}

pub trait DistanceTransformAbsPbaToF32C1: DataTypeLike {
    type Manhattan: DataTypeLike;

    fn distance_transform_abs_pba_to_f32(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        minimum_site_value: Self,
        maximum_site_value: Self,
        destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageViewMut<'_, f32, C1>>,
    ) -> Result<()>;
}

macro_rules! impl_distance_transform_abs_pba_to_f32 {
    ($source_ty:ty, $manhattan_ty:ty, $name:ident) => {
        impl DistanceTransformAbsPbaToF32C1 for $source_ty {
            type Manhattan = $manhattan_ty;

            fn distance_transform_abs_pba_to_f32(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                minimum_site_value: Self,
                maximum_site_value: Self,
                destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
                destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
                destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
                destination_transform: Option<&mut ImageViewMut<'_, f32, C1>>,
            ) -> Result<()> {
                $name(
                    stream_context,
                    source,
                    minimum_site_value,
                    maximum_site_value,
                    destination_voronoi,
                    destination_voronoi_indices,
                    destination_voronoi_manhattan,
                    destination_transform,
                )
            }
        }
    };
}

impl_distance_transform_abs_pba_to_f32!(u8, u16, distance_transform_abs_pba_u8_to_f32_c1);
impl_distance_transform_abs_pba_to_f32!(i8, u16, distance_transform_abs_pba_i8_to_f32_c1);
impl_distance_transform_abs_pba_to_f32!(u16, u16, distance_transform_abs_pba_u16_to_f32_c1);
impl_distance_transform_abs_pba_to_f32!(i16, u16, distance_transform_abs_pba_i16_to_f32_c1);

pub fn distance_transform_abs_pba_to_f32<T: DistanceTransformAbsPbaToF32C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    minimum_site_value: T,
    maximum_site_value: T,
    destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageViewMut<'_, f32, C1>>,
) -> Result<()> {
    T::distance_transform_abs_pba_to_f32(
        stream_context,
        source,
        minimum_site_value,
        maximum_site_value,
        destination_voronoi,
        destination_voronoi_indices,
        destination_voronoi_manhattan,
        destination_transform,
    )
}

pub trait DistanceTransformPbaToF64C1: DataTypeLike {
    type Manhattan: DataTypeLike;

    fn distance_transform_pba_to_f64(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        minimum_site_value: Self,
        maximum_site_value: Self,
        destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageViewMut<'_, f64, C1>>,
    ) -> Result<()>;

    fn distance_transform_pba_to_f64_antialiasing(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        minimum_site_value: Self,
        maximum_site_value: Self,
        destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageViewMut<'_, f64, C1>>,
    ) -> Result<()>;
}

macro_rules! impl_distance_transform_pba_to_f64 {
    ($source_ty:ty, $manhattan_ty:ty, $name:ident, $antialiasing_name:ident) => {
        impl DistanceTransformPbaToF64C1 for $source_ty {
            type Manhattan = $manhattan_ty;

            fn distance_transform_pba_to_f64(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                minimum_site_value: Self,
                maximum_site_value: Self,
                destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
                destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
                destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
                destination_transform: Option<&mut ImageViewMut<'_, f64, C1>>,
            ) -> Result<()> {
                $name(
                    stream_context,
                    source,
                    minimum_site_value,
                    maximum_site_value,
                    destination_voronoi,
                    destination_voronoi_indices,
                    destination_voronoi_manhattan,
                    destination_transform,
                )
            }

            fn distance_transform_pba_to_f64_antialiasing(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                minimum_site_value: Self,
                maximum_site_value: Self,
                destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
                destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
                destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
                destination_transform: Option<&mut ImageViewMut<'_, f64, C1>>,
            ) -> Result<()> {
                $antialiasing_name(
                    stream_context,
                    source,
                    minimum_site_value,
                    maximum_site_value,
                    destination_voronoi,
                    destination_voronoi_indices,
                    destination_voronoi_manhattan,
                    destination_transform,
                )
            }
        }
    };
}

impl_distance_transform_pba_to_f64!(
    u8,
    i16,
    distance_transform_pba_u8_to_f64_c1,
    distance_transform_pba_u8_to_f64_c1_antialiasing
);
impl_distance_transform_pba_to_f64!(
    i8,
    i16,
    distance_transform_pba_i8_to_f64_c1,
    distance_transform_pba_i8_to_f64_c1_antialiasing
);
impl_distance_transform_pba_to_f64!(
    u16,
    i16,
    distance_transform_pba_u16_to_f64_c1,
    distance_transform_pba_u16_to_f64_c1_antialiasing
);
impl_distance_transform_pba_to_f64!(
    i16,
    i16,
    distance_transform_pba_i16_to_f64_c1,
    distance_transform_pba_i16_to_f64_c1_antialiasing
);
impl_distance_transform_pba_to_f64!(
    f32,
    i16,
    distance_transform_pba_f32_to_f64_c1,
    distance_transform_pba_f32_to_f64_c1_antialiasing
);
impl_distance_transform_pba_to_f64!(
    f64,
    i16,
    distance_transform_pba_f64_c1,
    distance_transform_pba_f64_c1_antialiasing
);

pub fn distance_transform_pba_to_f64<T: DistanceTransformPbaToF64C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    minimum_site_value: T,
    maximum_site_value: T,
    destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageViewMut<'_, f64, C1>>,
) -> Result<()> {
    T::distance_transform_pba_to_f64(
        stream_context,
        source,
        minimum_site_value,
        maximum_site_value,
        destination_voronoi,
        destination_voronoi_indices,
        destination_voronoi_manhattan,
        destination_transform,
    )
}

pub fn distance_transform_pba_to_f64_antialiasing<T: DistanceTransformPbaToF64C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    minimum_site_value: T,
    maximum_site_value: T,
    destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageViewMut<'_, f64, C1>>,
) -> Result<()> {
    T::distance_transform_pba_to_f64_antialiasing(
        stream_context,
        source,
        minimum_site_value,
        maximum_site_value,
        destination_voronoi,
        destination_voronoi_indices,
        destination_voronoi_manhattan,
        destination_transform,
    )
}

pub trait DistanceTransformAbsPbaToF64C1: DataTypeLike {
    type Manhattan: DataTypeLike;

    fn distance_transform_abs_pba_to_f64(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        minimum_site_value: Self,
        maximum_site_value: Self,
        destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageViewMut<'_, f64, C1>>,
    ) -> Result<()>;

    fn distance_transform_abs_pba_to_f64_antialiasing(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        minimum_site_value: Self,
        maximum_site_value: Self,
        destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageViewMut<'_, f64, C1>>,
    ) -> Result<()>;
}

macro_rules! impl_distance_transform_abs_pba_to_f64 {
    ($source_ty:ty, $manhattan_ty:ty, $name:ident, $antialiasing_name:ident) => {
        impl DistanceTransformAbsPbaToF64C1 for $source_ty {
            type Manhattan = $manhattan_ty;

            fn distance_transform_abs_pba_to_f64(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                minimum_site_value: Self,
                maximum_site_value: Self,
                destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
                destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
                destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
                destination_transform: Option<&mut ImageViewMut<'_, f64, C1>>,
            ) -> Result<()> {
                $name(
                    stream_context,
                    source,
                    minimum_site_value,
                    maximum_site_value,
                    destination_voronoi,
                    destination_voronoi_indices,
                    destination_voronoi_manhattan,
                    destination_transform,
                )
            }

            fn distance_transform_abs_pba_to_f64_antialiasing(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                minimum_site_value: Self,
                maximum_site_value: Self,
                destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
                destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
                destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
                destination_transform: Option<&mut ImageViewMut<'_, f64, C1>>,
            ) -> Result<()> {
                $antialiasing_name(
                    stream_context,
                    source,
                    minimum_site_value,
                    maximum_site_value,
                    destination_voronoi,
                    destination_voronoi_indices,
                    destination_voronoi_manhattan,
                    destination_transform,
                )
            }
        }
    };
}

impl_distance_transform_abs_pba_to_f64!(
    u8,
    u16,
    distance_transform_abs_pba_u8_to_f64_c1,
    distance_transform_abs_pba_u8_to_f64_c1_antialiasing
);
impl_distance_transform_abs_pba_to_f64!(
    i8,
    u16,
    distance_transform_abs_pba_i8_to_f64_c1,
    distance_transform_abs_pba_i8_to_f64_c1_antialiasing
);
impl_distance_transform_abs_pba_to_f64!(
    u16,
    u16,
    distance_transform_abs_pba_u16_to_f64_c1,
    distance_transform_abs_pba_u16_to_f64_c1_antialiasing
);
impl_distance_transform_abs_pba_to_f64!(
    i16,
    u16,
    distance_transform_abs_pba_i16_to_f64_c1,
    distance_transform_abs_pba_i16_to_f64_c1_antialiasing
);
impl_distance_transform_abs_pba_to_f64!(
    f32,
    u16,
    distance_transform_abs_pba_f32_to_f64_c1,
    distance_transform_abs_pba_f32_to_f64_c1_antialiasing
);
impl_distance_transform_abs_pba_to_f64!(
    f64,
    u16,
    distance_transform_abs_pba_f64_c1,
    distance_transform_abs_pba_f64_c1_antialiasing
);

pub fn distance_transform_abs_pba_to_f64<T: DistanceTransformAbsPbaToF64C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    minimum_site_value: T,
    maximum_site_value: T,
    destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageViewMut<'_, f64, C1>>,
) -> Result<()> {
    T::distance_transform_abs_pba_to_f64(
        stream_context,
        source,
        minimum_site_value,
        maximum_site_value,
        destination_voronoi,
        destination_voronoi_indices,
        destination_voronoi_manhattan,
        destination_transform,
    )
}

pub fn distance_transform_abs_pba_to_f64_antialiasing<T: DistanceTransformAbsPbaToF64C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    minimum_site_value: T,
    maximum_site_value: T,
    destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageViewMut<'_, f64, C1>>,
) -> Result<()> {
    T::distance_transform_abs_pba_to_f64_antialiasing(
        stream_context,
        source,
        minimum_site_value,
        maximum_site_value,
        destination_voronoi,
        destination_voronoi_indices,
        destination_voronoi_manhattan,
        destination_transform,
    )
}

pub trait DistanceTransformPbaC1<Destination>: DataTypeLike
where
    Destination: DataTypeLike,
{
    type Manhattan: DataTypeLike;

    fn distance_transform_pba_c1(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        minimum_site_value: Self,
        maximum_site_value: Self,
        destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageViewMut<'_, Destination, C1>>,
    ) -> Result<()>;
}

macro_rules! impl_distance_transform_pba_c1 {
    ($destination:ty, $source_trait:ident, $source_method:ident) => {
        impl<T> DistanceTransformPbaC1<$destination> for T
        where
            T: $source_trait,
        {
            type Manhattan = <T as $source_trait>::Manhattan;

            fn distance_transform_pba_c1(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                minimum_site_value: Self,
                maximum_site_value: Self,
                destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
                destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
                destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
                destination_transform: Option<&mut ImageViewMut<'_, $destination, C1>>,
            ) -> Result<()> {
                T::$source_method(
                    stream_context,
                    source,
                    minimum_site_value,
                    maximum_site_value,
                    destination_voronoi,
                    destination_voronoi_indices,
                    destination_voronoi_manhattan,
                    destination_transform,
                )
            }
        }
    };
}

impl_distance_transform_pba_c1!(
    u16,
    DistanceTransformPbaToU16C1,
    distance_transform_pba_to_u16
);
impl_distance_transform_pba_c1!(
    f32,
    DistanceTransformPbaToF32C1,
    distance_transform_pba_to_f32
);
impl_distance_transform_pba_c1!(
    f64,
    DistanceTransformPbaToF64C1,
    distance_transform_pba_to_f64
);

pub fn distance_transform_pba<T, Destination>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    minimum_site_value: T,
    maximum_site_value: T,
    destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageViewMut<'_, Destination, C1>>,
) -> Result<()>
where
    T: DistanceTransformPbaC1<Destination>,
    Destination: DataTypeLike,
{
    T::distance_transform_pba_c1(
        stream_context,
        source,
        minimum_site_value,
        maximum_site_value,
        destination_voronoi,
        destination_voronoi_indices,
        destination_voronoi_manhattan,
        destination_transform,
    )
}

pub fn distance_transform_pba_antialiasing<T>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    minimum_site_value: T,
    maximum_site_value: T,
    destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageViewMut<'_, f64, C1>>,
) -> Result<()>
where
    T: DistanceTransformPbaToF64C1,
{
    distance_transform_pba_to_f64_antialiasing(
        stream_context,
        source,
        minimum_site_value,
        maximum_site_value,
        destination_voronoi,
        destination_voronoi_indices,
        destination_voronoi_manhattan,
        destination_transform,
    )
}

pub trait DistanceTransformAbsPbaC1<Destination>: DataTypeLike
where
    Destination: DataTypeLike,
{
    type Manhattan: DataTypeLike;

    fn distance_transform_abs_pba_c1(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        minimum_site_value: Self,
        maximum_site_value: Self,
        destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
        destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
        destination_transform: Option<&mut ImageViewMut<'_, Destination, C1>>,
    ) -> Result<()>;
}

macro_rules! impl_distance_transform_abs_pba_c1 {
    ($destination:ty, $source_trait:ident, $source_method:ident) => {
        impl<T> DistanceTransformAbsPbaC1<$destination> for T
        where
            T: $source_trait,
        {
            type Manhattan = <T as $source_trait>::Manhattan;

            fn distance_transform_abs_pba_c1(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                minimum_site_value: Self,
                maximum_site_value: Self,
                destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
                destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
                destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, Self::Manhattan, C2>>,
                destination_transform: Option<&mut ImageViewMut<'_, $destination, C1>>,
            ) -> Result<()> {
                T::$source_method(
                    stream_context,
                    source,
                    minimum_site_value,
                    maximum_site_value,
                    destination_voronoi,
                    destination_voronoi_indices,
                    destination_voronoi_manhattan,
                    destination_transform,
                )
            }
        }
    };
}

impl_distance_transform_abs_pba_c1!(
    u16,
    DistanceTransformAbsPbaToU16C1,
    distance_transform_abs_pba_to_u16
);
impl_distance_transform_abs_pba_c1!(
    f32,
    DistanceTransformAbsPbaToF32C1,
    distance_transform_abs_pba_to_f32
);
impl_distance_transform_abs_pba_c1!(
    f64,
    DistanceTransformAbsPbaToF64C1,
    distance_transform_abs_pba_to_f64
);

pub fn distance_transform_abs_pba<T, Destination>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    minimum_site_value: T,
    maximum_site_value: T,
    destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageViewMut<'_, Destination, C1>>,
) -> Result<()>
where
    T: DistanceTransformAbsPbaC1<Destination>,
    Destination: DataTypeLike,
{
    T::distance_transform_abs_pba_c1(
        stream_context,
        source,
        minimum_site_value,
        maximum_site_value,
        destination_voronoi,
        destination_voronoi_indices,
        destination_voronoi_manhattan,
        destination_transform,
    )
}

pub fn distance_transform_abs_pba_antialiasing<T>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    minimum_site_value: T,
    maximum_site_value: T,
    destination_voronoi: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_indices: Option<&mut ImageViewMut<'_, i16, C2>>,
    destination_voronoi_manhattan: Option<&mut ImageViewMut<'_, T::Manhattan, C2>>,
    destination_transform: Option<&mut ImageViewMut<'_, f64, C1>>,
) -> Result<()>
where
    T: DistanceTransformAbsPbaToF64C1,
{
    distance_transform_abs_pba_to_f64_antialiasing(
        stream_context,
        source,
        minimum_site_value,
        maximum_site_value,
        destination_voronoi,
        destination_voronoi_indices,
        destination_voronoi_manhattan,
        destination_transform,
    )
}

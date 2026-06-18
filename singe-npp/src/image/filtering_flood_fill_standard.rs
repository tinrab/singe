use super::*;

pub trait FloodFillC1InPlace: DataTypeLike {
    fn flood_fill_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, Self, C1>,
        seed: Point,
        new_value: Self,
        norm: ImageNormalization,
        connected_region: Option<&mut ConnectedRegion>,
    ) -> Result<()>;
}

macro_rules! impl_flood_fill_c1_dispatch {
    ($pixel_ty:ty, $name:ident) => {
        impl FloodFillC1InPlace for $pixel_ty {
            fn flood_fill_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, Self, C1>,
                seed: Point,
                new_value: Self,
                norm: ImageNormalization,
                connected_region: Option<&mut ConnectedRegion>,
            ) -> Result<()> {
                $name(
                    stream_context,
                    source_destination,
                    seed,
                    new_value,
                    norm,
                    connected_region,
                )
            }
        }
    };
}

impl_flood_fill_c1_dispatch!(u8, flood_fill_u8_c1_in_place);
impl_flood_fill_c1_dispatch!(u16, flood_fill_u16_c1_in_place);
impl_flood_fill_c1_dispatch!(u32, flood_fill_u32_c1_in_place);

pub fn flood_fill_in_place<T: FloodFillC1InPlace>(
    stream_context: &StreamContext,
    source_destination: &mut ImageViewMut<'_, T, C1>,
    seed: Point,
    new_value: T,
    norm: ImageNormalization,
    connected_region: Option<&mut ConnectedRegion>,
) -> Result<()> {
    T::flood_fill_in_place(
        stream_context,
        source_destination,
        seed,
        new_value,
        norm,
        connected_region,
    )
}

pub trait FloodFillC3InPlace: DataTypeLike {
    fn flood_fill_color_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, Self, C3>,
        seed: Point,
        new_values: [Self; 3],
        norm: ImageNormalization,
        connected_region: Option<&mut ConnectedRegion>,
    ) -> Result<()>;
}

macro_rules! impl_flood_fill_c3_dispatch {
    ($pixel_ty:ty, $name:ident) => {
        impl FloodFillC3InPlace for $pixel_ty {
            fn flood_fill_color_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, Self, C3>,
                seed: Point,
                new_values: [Self; 3],
                norm: ImageNormalization,
                connected_region: Option<&mut ConnectedRegion>,
            ) -> Result<()> {
                $name(
                    stream_context,
                    source_destination,
                    seed,
                    new_values,
                    norm,
                    connected_region,
                )
            }
        }
    };
}

impl_flood_fill_c3_dispatch!(u8, flood_fill_u8_c3_in_place);
impl_flood_fill_c3_dispatch!(u16, flood_fill_u16_c3_in_place);
impl_flood_fill_c3_dispatch!(u32, flood_fill_u32_c3_in_place);

pub fn flood_fill_color_in_place<T: FloodFillC3InPlace>(
    stream_context: &StreamContext,
    source_destination: &mut ImageViewMut<'_, T, C3>,
    seed: Point,
    new_values: [T; 3],
    norm: ImageNormalization,
    connected_region: Option<&mut ConnectedRegion>,
) -> Result<()> {
    T::flood_fill_color_in_place(
        stream_context,
        source_destination,
        seed,
        new_values,
        norm,
        connected_region,
    )
}

pub trait FloodFillBoundaryC1InPlace: DataTypeLike {
    fn flood_fill_boundary_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, Self, C1>,
        seed: Point,
        new_value: Self,
        boundary_value: Self,
        norm: ImageNormalization,
        connected_region: Option<&mut ConnectedRegion>,
    ) -> Result<()>;
}

macro_rules! impl_flood_fill_boundary_c1_dispatch {
    ($pixel_ty:ty, $name:ident) => {
        impl FloodFillBoundaryC1InPlace for $pixel_ty {
            fn flood_fill_boundary_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, Self, C1>,
                seed: Point,
                new_value: Self,
                boundary_value: Self,
                norm: ImageNormalization,
                connected_region: Option<&mut ConnectedRegion>,
            ) -> Result<()> {
                $name(
                    stream_context,
                    source_destination,
                    seed,
                    new_value,
                    boundary_value,
                    norm,
                    connected_region,
                )
            }
        }
    };
}

impl_flood_fill_boundary_c1_dispatch!(u8, flood_fill_boundary_u8_c1_in_place);
impl_flood_fill_boundary_c1_dispatch!(u16, flood_fill_boundary_u16_c1_in_place);
impl_flood_fill_boundary_c1_dispatch!(u32, flood_fill_boundary_u32_c1_in_place);

pub fn flood_fill_boundary_in_place<T: FloodFillBoundaryC1InPlace>(
    stream_context: &StreamContext,
    source_destination: &mut ImageViewMut<'_, T, C1>,
    seed: Point,
    new_value: T,
    boundary_value: T,
    norm: ImageNormalization,
    connected_region: Option<&mut ConnectedRegion>,
) -> Result<()> {
    T::flood_fill_boundary_in_place(
        stream_context,
        source_destination,
        seed,
        new_value,
        boundary_value,
        norm,
        connected_region,
    )
}

pub trait FloodFillBoundaryC3InPlace: DataTypeLike {
    fn flood_fill_boundary_color_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, Self, C3>,
        seed: Point,
        new_values: [Self; 3],
        boundary_values: [Self; 3],
        norm: ImageNormalization,
        connected_region: Option<&mut ConnectedRegion>,
    ) -> Result<()>;
}

macro_rules! impl_flood_fill_boundary_c3_dispatch {
    ($pixel_ty:ty, $name:ident) => {
        impl FloodFillBoundaryC3InPlace for $pixel_ty {
            fn flood_fill_boundary_color_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, Self, C3>,
                seed: Point,
                new_values: [Self; 3],
                boundary_values: [Self; 3],
                norm: ImageNormalization,
                connected_region: Option<&mut ConnectedRegion>,
            ) -> Result<()> {
                $name(
                    stream_context,
                    source_destination,
                    seed,
                    new_values,
                    boundary_values,
                    norm,
                    connected_region,
                )
            }
        }
    };
}

impl_flood_fill_boundary_c3_dispatch!(u8, flood_fill_boundary_u8_c3_in_place);
impl_flood_fill_boundary_c3_dispatch!(u16, flood_fill_boundary_u16_c3_in_place);
impl_flood_fill_boundary_c3_dispatch!(u32, flood_fill_boundary_u32_c3_in_place);

pub fn flood_fill_boundary_color_in_place<T: FloodFillBoundaryC3InPlace>(
    stream_context: &StreamContext,
    source_destination: &mut ImageViewMut<'_, T, C3>,
    seed: Point,
    new_values: [T; 3],
    boundary_values: [T; 3],
    norm: ImageNormalization,
    connected_region: Option<&mut ConnectedRegion>,
) -> Result<()> {
    T::flood_fill_boundary_color_in_place(
        stream_context,
        source_destination,
        seed,
        new_values,
        boundary_values,
        norm,
        connected_region,
    )
}

pub trait FloodFillRangeC1InPlace: DataTypeLike {
    fn flood_fill_range_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, Self, C1>,
        seed: Point,
        min: Self,
        max: Self,
        new_value: Self,
        norm: ImageNormalization,
        connected_region: Option<&mut ConnectedRegion>,
    ) -> Result<()>;
}

macro_rules! impl_flood_fill_range_c1_dispatch {
    ($pixel_ty:ty, $name:ident) => {
        impl FloodFillRangeC1InPlace for $pixel_ty {
            fn flood_fill_range_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, Self, C1>,
                seed: Point,
                min: Self,
                max: Self,
                new_value: Self,
                norm: ImageNormalization,
                connected_region: Option<&mut ConnectedRegion>,
            ) -> Result<()> {
                $name(
                    stream_context,
                    source_destination,
                    seed,
                    min,
                    max,
                    new_value,
                    norm,
                    connected_region,
                )
            }
        }
    };
}

impl_flood_fill_range_c1_dispatch!(u8, flood_fill_range_u8_c1_in_place);
impl_flood_fill_range_c1_dispatch!(u16, flood_fill_range_u16_c1_in_place);
impl_flood_fill_range_c1_dispatch!(u32, flood_fill_range_u32_c1_in_place);

pub fn flood_fill_range_in_place<T: FloodFillRangeC1InPlace>(
    stream_context: &StreamContext,
    source_destination: &mut ImageViewMut<'_, T, C1>,
    seed: Point,
    min: T,
    max: T,
    new_value: T,
    norm: ImageNormalization,
    connected_region: Option<&mut ConnectedRegion>,
) -> Result<()> {
    T::flood_fill_range_in_place(
        stream_context,
        source_destination,
        seed,
        min,
        max,
        new_value,
        norm,
        connected_region,
    )
}

pub trait FloodFillRangeC3InPlace: DataTypeLike {
    fn flood_fill_range_color_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, Self, C3>,
        seed: Point,
        min: [Self; 3],
        max: [Self; 3],
        new_values: [Self; 3],
        norm: ImageNormalization,
        connected_region: Option<&mut ConnectedRegion>,
    ) -> Result<()>;
}

macro_rules! impl_flood_fill_range_c3_dispatch {
    ($pixel_ty:ty, $name:ident) => {
        impl FloodFillRangeC3InPlace for $pixel_ty {
            fn flood_fill_range_color_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, Self, C3>,
                seed: Point,
                min: [Self; 3],
                max: [Self; 3],
                new_values: [Self; 3],
                norm: ImageNormalization,
                connected_region: Option<&mut ConnectedRegion>,
            ) -> Result<()> {
                $name(
                    stream_context,
                    source_destination,
                    seed,
                    min,
                    max,
                    new_values,
                    norm,
                    connected_region,
                )
            }
        }
    };
}

impl_flood_fill_range_c3_dispatch!(u8, flood_fill_range_u8_c3_in_place);
impl_flood_fill_range_c3_dispatch!(u16, flood_fill_range_u16_c3_in_place);
impl_flood_fill_range_c3_dispatch!(u32, flood_fill_range_u32_c3_in_place);

pub fn flood_fill_range_color_in_place<T: FloodFillRangeC3InPlace>(
    stream_context: &StreamContext,
    source_destination: &mut ImageViewMut<'_, T, C3>,
    seed: Point,
    min: [T; 3],
    max: [T; 3],
    new_values: [T; 3],
    norm: ImageNormalization,
    connected_region: Option<&mut ConnectedRegion>,
) -> Result<()> {
    T::flood_fill_range_color_in_place(
        stream_context,
        source_destination,
        seed,
        min,
        max,
        new_values,
        norm,
        connected_region,
    )
}

pub trait FloodFillRangeBoundaryC1InPlace: DataTypeLike {
    fn flood_fill_range_boundary_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, Self, C1>,
        seed: Point,
        min: Self,
        max: Self,
        new_value: Self,
        boundary_value: Self,
        norm: ImageNormalization,
        connected_region: Option<&mut ConnectedRegion>,
    ) -> Result<()>;
}

macro_rules! impl_flood_fill_range_boundary_c1_dispatch {
    ($pixel_ty:ty, $name:ident) => {
        impl FloodFillRangeBoundaryC1InPlace for $pixel_ty {
            fn flood_fill_range_boundary_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, Self, C1>,
                seed: Point,
                min: Self,
                max: Self,
                new_value: Self,
                boundary_value: Self,
                norm: ImageNormalization,
                connected_region: Option<&mut ConnectedRegion>,
            ) -> Result<()> {
                $name(
                    stream_context,
                    source_destination,
                    seed,
                    min,
                    max,
                    new_value,
                    boundary_value,
                    norm,
                    connected_region,
                )
            }
        }
    };
}

impl_flood_fill_range_boundary_c1_dispatch!(u8, flood_fill_range_boundary_u8_c1_in_place);
impl_flood_fill_range_boundary_c1_dispatch!(u16, flood_fill_range_boundary_u16_c1_in_place);
impl_flood_fill_range_boundary_c1_dispatch!(u32, flood_fill_range_boundary_u32_c1_in_place);

pub fn flood_fill_range_boundary_in_place<T: FloodFillRangeBoundaryC1InPlace>(
    stream_context: &StreamContext,
    source_destination: &mut ImageViewMut<'_, T, C1>,
    seed: Point,
    min: T,
    max: T,
    new_value: T,
    boundary_value: T,
    norm: ImageNormalization,
    connected_region: Option<&mut ConnectedRegion>,
) -> Result<()> {
    T::flood_fill_range_boundary_in_place(
        stream_context,
        source_destination,
        seed,
        min,
        max,
        new_value,
        boundary_value,
        norm,
        connected_region,
    )
}

pub trait FloodFillRangeBoundaryC3InPlace: DataTypeLike {
    fn flood_fill_range_boundary_color_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, Self, C3>,
        seed: Point,
        min: [Self; 3],
        max: [Self; 3],
        new_values: [Self; 3],
        boundary_values: [Self; 3],
        norm: ImageNormalization,
        connected_region: Option<&mut ConnectedRegion>,
    ) -> Result<()>;
}

macro_rules! impl_flood_fill_range_boundary_c3_dispatch {
    ($pixel_ty:ty, $name:ident) => {
        impl FloodFillRangeBoundaryC3InPlace for $pixel_ty {
            fn flood_fill_range_boundary_color_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, Self, C3>,
                seed: Point,
                min: [Self; 3],
                max: [Self; 3],
                new_values: [Self; 3],
                boundary_values: [Self; 3],
                norm: ImageNormalization,
                connected_region: Option<&mut ConnectedRegion>,
            ) -> Result<()> {
                $name(
                    stream_context,
                    source_destination,
                    seed,
                    min,
                    max,
                    new_values,
                    boundary_values,
                    norm,
                    connected_region,
                )
            }
        }
    };
}

impl_flood_fill_range_boundary_c3_dispatch!(u8, flood_fill_range_boundary_u8_c3_in_place);
impl_flood_fill_range_boundary_c3_dispatch!(u16, flood_fill_range_boundary_u16_c3_in_place);
impl_flood_fill_range_boundary_c3_dispatch!(u32, flood_fill_range_boundary_u32_c3_in_place);

pub fn flood_fill_range_boundary_color_in_place<T: FloodFillRangeBoundaryC3InPlace>(
    stream_context: &StreamContext,
    source_destination: &mut ImageViewMut<'_, T, C3>,
    seed: Point,
    min: [T; 3],
    max: [T; 3],
    new_values: [T; 3],
    boundary_values: [T; 3],
    norm: ImageNormalization,
    connected_region: Option<&mut ConnectedRegion>,
) -> Result<()> {
    T::flood_fill_range_boundary_color_in_place(
        stream_context,
        source_destination,
        seed,
        min,
        max,
        new_values,
        boundary_values,
        norm,
        connected_region,
    )
}

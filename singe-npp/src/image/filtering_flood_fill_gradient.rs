use super::*;

pub trait FloodFillGradientC1InPlace: DataTypeLike {
    fn flood_fill_gradient_in_place(
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

macro_rules! impl_flood_fill_gradient_c1_dispatch {
    ($pixel_ty:ty, $name:ident) => {
        impl FloodFillGradientC1InPlace for $pixel_ty {
            fn flood_fill_gradient_in_place(
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

impl_flood_fill_gradient_c1_dispatch!(u8, flood_fill_gradient_u8_c1_in_place);
impl_flood_fill_gradient_c1_dispatch!(u16, flood_fill_gradient_u16_c1_in_place);
impl_flood_fill_gradient_c1_dispatch!(u32, flood_fill_gradient_u32_c1_in_place);

pub fn flood_fill_gradient_in_place<T: FloodFillGradientC1InPlace>(
    stream_context: &StreamContext,
    source_destination: &mut ImageViewMut<'_, T, C1>,
    seed: Point,
    min: T,
    max: T,
    new_value: T,
    norm: ImageNormalization,
    connected_region: Option<&mut ConnectedRegion>,
) -> Result<()> {
    T::flood_fill_gradient_in_place(
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

pub trait FloodFillGradientC3InPlace: DataTypeLike {
    fn flood_fill_gradient_color_in_place(
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

macro_rules! impl_flood_fill_gradient_c3_dispatch {
    ($pixel_ty:ty, $name:ident) => {
        impl FloodFillGradientC3InPlace for $pixel_ty {
            fn flood_fill_gradient_color_in_place(
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

impl_flood_fill_gradient_c3_dispatch!(u8, flood_fill_gradient_u8_c3_in_place);
impl_flood_fill_gradient_c3_dispatch!(u16, flood_fill_gradient_u16_c3_in_place);
impl_flood_fill_gradient_c3_dispatch!(u32, flood_fill_gradient_u32_c3_in_place);

pub fn flood_fill_gradient_color_in_place<T: FloodFillGradientC3InPlace>(
    stream_context: &StreamContext,
    source_destination: &mut ImageViewMut<'_, T, C3>,
    seed: Point,
    min: [T; 3],
    max: [T; 3],
    new_values: [T; 3],
    norm: ImageNormalization,
    connected_region: Option<&mut ConnectedRegion>,
) -> Result<()> {
    T::flood_fill_gradient_color_in_place(
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

pub trait FloodFillGradientBoundaryC1InPlace: DataTypeLike {
    fn flood_fill_gradient_boundary_in_place(
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

macro_rules! impl_flood_fill_gradient_boundary_c1_dispatch {
    ($pixel_ty:ty, $name:ident) => {
        impl FloodFillGradientBoundaryC1InPlace for $pixel_ty {
            fn flood_fill_gradient_boundary_in_place(
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

impl_flood_fill_gradient_boundary_c1_dispatch!(u8, flood_fill_gradient_boundary_u8_c1_in_place);
impl_flood_fill_gradient_boundary_c1_dispatch!(u16, flood_fill_gradient_boundary_u16_c1_in_place);
impl_flood_fill_gradient_boundary_c1_dispatch!(u32, flood_fill_gradient_boundary_u32_c1_in_place);

pub fn flood_fill_gradient_boundary_in_place<T: FloodFillGradientBoundaryC1InPlace>(
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
    T::flood_fill_gradient_boundary_in_place(
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

pub trait FloodFillGradientBoundaryC3InPlace: DataTypeLike {
    fn flood_fill_gradient_boundary_color_in_place(
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

macro_rules! impl_flood_fill_gradient_boundary_c3_dispatch {
    ($pixel_ty:ty, $name:ident) => {
        impl FloodFillGradientBoundaryC3InPlace for $pixel_ty {
            fn flood_fill_gradient_boundary_color_in_place(
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

impl_flood_fill_gradient_boundary_c3_dispatch!(u8, flood_fill_gradient_boundary_u8_c3_in_place);
impl_flood_fill_gradient_boundary_c3_dispatch!(u16, flood_fill_gradient_boundary_u16_c3_in_place);
impl_flood_fill_gradient_boundary_c3_dispatch!(u32, flood_fill_gradient_boundary_u32_c3_in_place);

pub fn flood_fill_gradient_boundary_color_in_place<T: FloodFillGradientBoundaryC3InPlace>(
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
    T::flood_fill_gradient_boundary_color_in_place(
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

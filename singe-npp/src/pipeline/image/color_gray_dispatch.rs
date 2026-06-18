use crate::{
    context::StreamContext,
    error::Result,
    image::view::{AC4, C1, C3, ImageView, ImageViewMut},
    types::{BayerGridPosition, ImageNormalization, Rectangle},
};

pub trait RgbToGrayImage<T, L> {
    fn rgb_to_gray_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, C1>,
    ) -> Result<()>;
}

pub trait ColorToGrayImage<T, L, const COEFFICIENTS: usize> {
    fn color_to_gray_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, C1>,
        coefficients: &[f32; COEFFICIENTS],
    ) -> Result<()>;
}

pub trait GradientColorToGrayImage<T> {
    fn gradient_color_to_gray_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C3>,
        destination: &mut ImageViewMut<'_, T, C1>,
        normalization: ImageNormalization,
    ) -> Result<()>;
}

pub trait CfaToRgbImage<T> {
    fn cfa_to_rgb_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        source_roi: Rectangle,
        destination: &mut ImageViewMut<'_, T, C3>,
        grid: BayerGridPosition,
    ) -> Result<()>;
}

pub trait CfaToRgbaImage<T> {
    fn cfa_to_rgba_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        source_roi: Rectangle,
        destination: &mut ImageViewMut<'_, T, AC4>,
        grid: BayerGridPosition,
        alpha: T,
    ) -> Result<()>;
}

macro_rules! impl_rgb_to_gray_image {
    ($ty:ty, $layout:ty, $convert:path) => {
        impl<'a> RgbToGrayImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn rgb_to_gray_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, C1>,
            ) -> Result<()> {
                $convert(stream_context, source, destination)
            }
        }
    };
}

macro_rules! impl_color_to_gray_image {
    ($ty:ty, $layout:ty, $coefficients:literal, $convert:path) => {
        impl<'a> ColorToGrayImage<$ty, $layout, $coefficients> for ImagePipeline<'a, $ty, $layout> {
            fn color_to_gray_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, C1>,
                coefficients: &[f32; $coefficients],
            ) -> Result<()> {
                $convert(stream_context, source, destination, coefficients)
            }
        }
    };
}

macro_rules! impl_gradient_color_to_gray_image {
    ($ty:ty, $convert:path) => {
        impl<'a> GradientColorToGrayImage<$ty> for ImagePipeline<'a, $ty, C3> {
            fn gradient_color_to_gray_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C3>,
                destination: &mut ImageViewMut<'_, $ty, C1>,
                normalization: ImageNormalization,
            ) -> Result<()> {
                $convert(stream_context, source, destination, normalization)
            }
        }
    };
}

macro_rules! impl_cfa_to_rgb_image {
    ($ty:ty, $cfa_to_rgb:path) => {
        impl<'a> CfaToRgbImage<$ty> for ImagePipeline<'a, $ty, C1> {
            fn cfa_to_rgb_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                source_roi: Rectangle,
                destination: &mut ImageViewMut<'_, $ty, C3>,
                grid: BayerGridPosition,
            ) -> Result<()> {
                $cfa_to_rgb(stream_context, source, source_roi, destination, grid)
            }
        }
    };
}

macro_rules! impl_cfa_to_rgba_image {
    ($ty:ty, $cfa_to_rgba:path) => {
        impl<'a> CfaToRgbaImage<$ty> for ImagePipeline<'a, $ty, C1> {
            fn cfa_to_rgba_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                source_roi: Rectangle,
                destination: &mut ImageViewMut<'_, $ty, AC4>,
                grid: BayerGridPosition,
                alpha: $ty,
            ) -> Result<()> {
                $cfa_to_rgba(stream_context, source, source_roi, destination, grid, alpha)
            }
        }
    };
}

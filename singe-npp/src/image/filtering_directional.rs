use super::*;

macro_rules! impl_directional_filter_dispatch {
    ($trait_name:ident, $method_name:ident, $pixel_ty:ty, $layout:ty, $name:ident) => {
        impl $trait_name<$layout> for $pixel_ty {
            fn $method_name(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                destination: &mut ImageViewMut<'_, Self, $layout>,
            ) -> Result<()> {
                $name(stream_context, source, destination)
            }
        }
    };
}

macro_rules! impl_directional_filter_border_dispatch {
    ($trait_name:ident, $method_name:ident, $pixel_ty:ty, $layout:ty, $name:ident) => {
        impl $trait_name<$layout> for $pixel_ty {
            fn $method_name(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, Self, $layout>,
                border_type: BorderType,
            ) -> Result<()> {
                $name(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    border_type,
                )
            }
        }
    };
}

macro_rules! impl_directional_filter_typed_c1_dispatch {
    ($trait_name:ident, $method_name:ident, $source_ty:ty, $destination_ty:ty, $name:ident) => {
        impl $trait_name for $source_ty {
            type Output = $destination_ty;

            fn $method_name(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                destination: &mut ImageViewMut<'_, Self::Output, C1>,
            ) -> Result<()> {
                $name(stream_context, source, destination)
            }
        }
    };
}

macro_rules! impl_directional_filter_typed_border_c1_dispatch {
    ($trait_name:ident, $method_name:ident, $source_ty:ty, $destination_ty:ty, $name:ident) => {
        impl $trait_name for $source_ty {
            type Output = $destination_ty;

            fn $method_name(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, Self::Output, C1>,
                border_type: BorderType,
            ) -> Result<()> {
                $name(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    border_type,
                )
            }
        }
    };
}

macro_rules! impl_filter_sobel_typed_c1_dispatch {
    ($trait_name:ident, $method_name:ident, $source_ty:ty, $destination_ty:ty, $name:ident) => {
        impl $trait_name for $source_ty {
            type Output = $destination_ty;

            fn $method_name(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                destination: &mut ImageViewMut<'_, Self::Output, C1>,
                mask_size: MaskSize,
            ) -> Result<()> {
                $name(stream_context, source, destination, mask_size)
            }
        }
    };
}

macro_rules! impl_filter_sobel_typed_border_c1_dispatch {
    ($trait_name:ident, $method_name:ident, $source_ty:ty, $destination_ty:ty, $name:ident) => {
        impl $trait_name for $source_ty {
            type Output = $destination_ty;

            fn $method_name(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, Self::Output, C1>,
                mask_size: MaskSize,
                border_type: BorderType,
            ) -> Result<()> {
                $name(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    mask_size,
                    border_type,
                )
            }
        }
    };
}

#[path = "filtering_directional_prewitt.rs"]
mod prewitt;
pub use prewitt::*;

#[path = "filtering_directional_scharr.rs"]
mod scharr;
pub use scharr::*;

#[path = "filtering_directional_roberts.rs"]
mod roberts;
pub use roberts::*;

#[path = "filtering_directional_sobel.rs"]
mod sobel;
pub use sobel::*;

#[path = "filtering_directional_laplace.rs"]
mod laplace;
pub use laplace::*;

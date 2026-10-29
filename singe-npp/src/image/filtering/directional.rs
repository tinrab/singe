macro_rules! impl_directional_filter_dispatch {
    ($trait_name:ident, $method_name:ident, $pixel_ty:ty, $layout:ty, $name:ident) => {
        unsafe impl $trait_name<$layout> for $pixel_ty {
            unsafe fn $method_name(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                destination: &mut ImageOutputView<'_, Self, $layout>,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $name(stream_context, source, destination) }
            }
        }
    };
}

macro_rules! impl_directional_filter_border_dispatch {
    ($trait_name:ident, $method_name:ident, $pixel_ty:ty, $layout:ty, $name:ident) => {
        unsafe impl $trait_name<$layout> for $pixel_ty {
            unsafe fn $method_name(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, Self, $layout>,
                border_type: BorderType,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    $name(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        border_type,
                    )
                }
            }
        }
    };
}

macro_rules! impl_directional_filter_typed_c1_dispatch {
    ($trait_name:ident, $method_name:ident, $source_ty:ty, $destination_ty:ty, $name:ident) => {
        unsafe impl $trait_name for $source_ty {
            type Output = $destination_ty;

            unsafe fn $method_name(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                destination: &mut ImageOutputView<'_, Self::Output, C1>,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $name(stream_context, source, destination) }
            }
        }
    };
}

macro_rules! impl_directional_filter_typed_border_c1_dispatch {
    ($trait_name:ident, $method_name:ident, $source_ty:ty, $destination_ty:ty, $name:ident) => {
        unsafe impl $trait_name for $source_ty {
            type Output = $destination_ty;

            unsafe fn $method_name(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, Self::Output, C1>,
                border_type: BorderType,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    $name(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        border_type,
                    )
                }
            }
        }
    };
}

macro_rules! impl_filter_sobel_typed_c1_dispatch {
    ($trait_name:ident, $method_name:ident, $source_ty:ty, $destination_ty:ty, $name:ident) => {
        unsafe impl $trait_name for $source_ty {
            type Output = $destination_ty;

            unsafe fn $method_name(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                destination: &mut ImageOutputView<'_, Self::Output, C1>,
                mask_size: MaskSize,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $name(stream_context, source, destination, mask_size) }
            }
        }
    };
}

macro_rules! impl_filter_sobel_typed_border_c1_dispatch {
    ($trait_name:ident, $method_name:ident, $source_ty:ty, $destination_ty:ty, $name:ident) => {
        unsafe impl $trait_name for $source_ty {
            type Output = $destination_ty;

            unsafe fn $method_name(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, Self::Output, C1>,
                mask_size: MaskSize,
                border_type: BorderType,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
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
        }
    };
}

mod prewitt;

pub use prewitt::*;

mod scharr;

pub use scharr::*;

mod roberts;

pub use roberts::*;

mod sobel;

pub use sobel::*;

mod laplace;

pub use laplace::*;

use singe_cuda::memory::DeviceMemory;
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    try_ffi,
    types::{DataTypeLike, Size},
    utility::to_usize,
};

use super::statistics_validation::*;

macro_rules! impl_template_match_full_scaled {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            template: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            scale_factor: i32,
        ) -> Result<()> {
            validate_template_full_destination_size(
                source.size(),
                template.size(),
                destination.size(),
            )?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    template.as_ptr().cast(),
                    template.step(),
                    template.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_template_match_full {
    ($name:ident, $source_ty:ty, $destination_ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $layout>,
            template: &ImageView<'_, $source_ty, $layout>,
            destination: &mut ImageViewMut<'_, $destination_ty, $layout>,
        ) -> Result<()> {
            validate_template_full_destination_size(
                source.size(),
                template.size(),
                destination.size(),
            )?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    template.as_ptr().cast(),
                    template.step(),
                    template.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_template_match_same_scaled {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            template: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            scale_factor: i32,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    template.as_ptr().cast(),
                    template.step(),
                    template.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_template_match_same {
    ($name:ident, $source_ty:ty, $destination_ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $layout>,
            template: &ImageView<'_, $source_ty, $layout>,
            destination: &mut ImageViewMut<'_, $destination_ty, $layout>,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    template.as_ptr().cast(),
                    template.step(),
                    template.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_template_match_valid_scaled {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            template: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            scale_factor: i32,
        ) -> Result<()> {
            validate_template_valid_destination_size(
                source.size(),
                template.size(),
                destination.size(),
            )?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    template.as_ptr().cast(),
                    template.step(),
                    template.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_template_match_valid {
    ($name:ident, $source_ty:ty, $destination_ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $layout>,
            template: &ImageView<'_, $source_ty, $layout>,
            destination: &mut ImageViewMut<'_, $destination_ty, $layout>,
        ) -> Result<()> {
            validate_template_valid_destination_size(
                source.size(),
                template.size(),
                destination.size(),
            )?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    template.as_ptr().cast(),
                    template.step(),
                    template.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_template_match_norm_level_full_scaled {
    ($buffer_size_name:ident, $name:ident, $ty:ty, $layout:ty, $buffer_size_ffi:ident, $ffi:ident) => {
        pub fn $buffer_size_name(stream_context: &StreamContext, roi: Size) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            template: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            scale_factor: i32,
        ) -> Result<()> {
            validate_template_full_destination_size(
                source.size(),
                template.size(),
                destination.size(),
            )?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    template.as_ptr().cast(),
                    template.step(),
                    template.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    scale_factor,
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_template_match_norm_level_full {
    (
        $buffer_size_name:ident,
        $name:ident,
        $source_ty:ty,
        $destination_ty:ty,
        $layout:ty,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        pub fn $buffer_size_name(stream_context: &StreamContext, roi: Size) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $layout>,
            template: &ImageView<'_, $source_ty, $layout>,
            destination: &mut ImageViewMut<'_, $destination_ty, $layout>,
        ) -> Result<()> {
            validate_template_full_destination_size(
                source.size(),
                template.size(),
                destination.size(),
            )?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    template.as_ptr().cast(),
                    template.step(),
                    template.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_template_match_norm_level_same_scaled {
    ($buffer_size_name:ident, $name:ident, $ty:ty, $layout:ty, $buffer_size_ffi:ident, $ffi:ident) => {
        pub fn $buffer_size_name(stream_context: &StreamContext, roi: Size) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            template: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            scale_factor: i32,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    template.as_ptr().cast(),
                    template.step(),
                    template.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    scale_factor,
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_template_match_norm_level_same {
    (
        $buffer_size_name:ident,
        $name:ident,
        $source_ty:ty,
        $destination_ty:ty,
        $layout:ty,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        pub fn $buffer_size_name(stream_context: &StreamContext, roi: Size) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $layout>,
            template: &ImageView<'_, $source_ty, $layout>,
            destination: &mut ImageViewMut<'_, $destination_ty, $layout>,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    template.as_ptr().cast(),
                    template.step(),
                    template.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_template_match_norm_level_valid_scaled {
    ($buffer_size_name:ident, $name:ident, $ty:ty, $layout:ty, $buffer_size_ffi:ident, $ffi:ident) => {
        pub fn $buffer_size_name(stream_context: &StreamContext, roi: Size) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            template: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            scale_factor: i32,
        ) -> Result<()> {
            validate_template_valid_destination_size(
                source.size(),
                template.size(),
                destination.size(),
            )?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    template.as_ptr().cast(),
                    template.step(),
                    template.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    scale_factor,
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_template_match_norm_level_valid {
    (
        $buffer_size_name:ident,
        $name:ident,
        $source_ty:ty,
        $destination_ty:ty,
        $layout:ty,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        pub fn $buffer_size_name(stream_context: &StreamContext, roi: Size) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $layout>,
            template: &ImageView<'_, $source_ty, $layout>,
            destination: &mut ImageViewMut<'_, $destination_ty, $layout>,
        ) -> Result<()> {
            validate_template_valid_destination_size(
                source.size(),
                template.size(),
                destination.size(),
            )?;
            let required_bytes = $buffer_size_name(stream_context, source.size())?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    template.as_ptr().cast(),
                    template.step(),
                    template.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_advanced_template_match_buffer_size {
    ($name:ident, $ffi:ident) => {
        pub fn $name(
            source_size: Size,
            template_size: Size,
            destination_element_bytes: usize,
            channels: i32,
        ) -> Result<usize> {
            let destination_element_bytes =
                i32::try_from(destination_element_bytes).map_err(|_| Error::OutOfRange {
                    name: "destination element bytes".into(),
                })?;
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$ffi(
                    source_size.into(),
                    template_size.into(),
                    destination_element_bytes,
                    channels,
                    &raw mut bytes,
                ))?;
            }
            to_usize(bytes, "buffer size")
        }
    };
}

macro_rules! impl_template_match_norm_level_advanced_full {
    (
        $name:ident,
        $source_ty:ty,
        $destination_ty:ty,
        $layout:ty,
        $channels:expr,
        $norm_level_buffer_size_name:ident,
        $advanced_buffer_size_name:ident,
        $ffi:ident
    ) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $layout>,
            template: &ImageView<'_, $source_ty, $layout>,
            destination: &mut ImageViewMut<'_, $destination_ty, $layout>,
        ) -> Result<()> {
            validate_template_full_destination_size(
                source.size(),
                template.size(),
                destination.size(),
            )?;
            let norm_level_bytes = $norm_level_buffer_size_name(stream_context, source.size())?;
            let advanced_bytes = $advanced_buffer_size_name(
                source.size(),
                template.size(),
                size_of::<$destination_ty>(),
                $channels,
            )?;
            let norm_level_scratch = DeviceMemory::<u8>::create(norm_level_bytes)?;
            let advanced_scratch = DeviceMemory::<u8>::create(advanced_bytes)?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    template.as_ptr().cast(),
                    template.step(),
                    template.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    norm_level_scratch.as_mut_ptr().cast(),
                    advanced_scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_template_match_norm_level_advanced_same {
    (
        $name:ident,
        $source_ty:ty,
        $destination_ty:ty,
        $layout:ty,
        $channels:expr,
        $norm_level_buffer_size_name:ident,
        $advanced_buffer_size_name:ident,
        $ffi:ident
    ) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $layout>,
            template: &ImageView<'_, $source_ty, $layout>,
            destination: &mut ImageViewMut<'_, $destination_ty, $layout>,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            let norm_level_bytes = $norm_level_buffer_size_name(stream_context, source.size())?;
            let advanced_bytes = $advanced_buffer_size_name(
                source.size(),
                template.size(),
                size_of::<$destination_ty>(),
                $channels,
            )?;
            let norm_level_scratch = DeviceMemory::<u8>::create(norm_level_bytes)?;
            let advanced_scratch = DeviceMemory::<u8>::create(advanced_bytes)?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    template.as_ptr().cast(),
                    template.step(),
                    template.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    norm_level_scratch.as_mut_ptr().cast(),
                    advanced_scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_template_match_norm_level_advanced_valid {
    (
        $name:ident,
        $source_ty:ty,
        $destination_ty:ty,
        $layout:ty,
        $channels:expr,
        $norm_level_buffer_size_name:ident,
        $advanced_buffer_size_name:ident,
        $ffi:ident
    ) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $layout>,
            template: &ImageView<'_, $source_ty, $layout>,
            destination: &mut ImageViewMut<'_, $destination_ty, $layout>,
        ) -> Result<()> {
            validate_template_valid_destination_size(
                source.size(),
                template.size(),
                destination.size(),
            )?;
            let norm_level_bytes = $norm_level_buffer_size_name(stream_context, source.size())?;
            let advanced_bytes = $advanced_buffer_size_name(
                source.size(),
                template.size(),
                size_of::<$destination_ty>(),
                $channels,
            )?;
            let norm_level_scratch = DeviceMemory::<u8>::create(norm_level_bytes)?;
            let advanced_scratch = DeviceMemory::<u8>::create(advanced_bytes)?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    template.as_ptr().cast(),
                    template.step(),
                    template.size().into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    norm_level_scratch.as_mut_ptr().cast(),
                    advanced_scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_template_match_scaled {
    ($trait_name:ident, $name:ident, $layout:ty, [$(($ty:ty, $direct:ident)),+ $(,)?]) => {
        pub trait $trait_name<Layout>: DataTypeLike {
            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                template: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                scale_factor: i32,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            impl $trait_name<$layout> for $ty {
                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    template: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    scale_factor: i32,
                ) -> Result<()> {
                    $direct(stream_context, source, template, destination, scale_factor)
                }
            }
        )+

        pub fn $name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            template: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T, $layout>,
            scale_factor: i32,
        ) -> Result<()> {
            T::dispatch(stream_context, source, template, destination, scale_factor)
        }
    };
}

macro_rules! impl_generic_template_match {
    ($trait_name:ident, $name:ident, $layout:ty, [$(($ty:ty, $destination_ty:ty, $direct:ident)),+ $(,)?]) => {
        pub trait $trait_name<Layout>: DataTypeLike {
            type Destination: DataTypeLike;

            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                template: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self::Destination, Layout>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            impl $trait_name<$layout> for $ty {
                type Destination = $destination_ty;

                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    template: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self::Destination, $layout>,
                ) -> Result<()> {
                    $direct(stream_context, source, template, destination)
                }
            }
        )+

        pub fn $name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            template: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T::Destination, $layout>,
        ) -> Result<()> {
            T::dispatch(stream_context, source, template, destination)
        }
    };
}

macro_rules! impl_generic_template_match_buffered_scaled {
    ($trait_name:ident, $name:ident, $buffer_size_name:ident, $layout:ty, [$(($ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        pub trait $trait_name<Layout>: DataTypeLike {
            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                template: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                scale_factor: i32,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            impl $trait_name<$layout> for $ty {
                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    template: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    scale_factor: i32,
                ) -> Result<()> {
                    $direct(stream_context, source, template, destination, scale_factor)
                }
            }
        )+

        pub fn $buffer_size_name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        pub fn $name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            template: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T, $layout>,
            scale_factor: i32,
        ) -> Result<()> {
            T::dispatch(stream_context, source, template, destination, scale_factor)
        }
    };
}

macro_rules! impl_generic_template_match_buffered {
    ($trait_name:ident, $name:ident, $buffer_size_name:ident, $layout:ty, [$(($ty:ty, $destination_ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        pub trait $trait_name<Layout>: DataTypeLike {
            type Destination: DataTypeLike;

            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                template: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self::Destination, Layout>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            impl $trait_name<$layout> for $ty {
                type Destination = $destination_ty;

                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    template: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self::Destination, $layout>,
                ) -> Result<()> {
                    $direct(stream_context, source, template, destination)
                }
            }
        )+

        pub fn $buffer_size_name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        pub fn $name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            template: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T::Destination, $layout>,
        ) -> Result<()> {
            T::dispatch(stream_context, source, template, destination)
        }
    };
}

macro_rules! impl_generic_template_match_advanced {
    ($trait_name:ident, $name:ident, $layout:ty, [$(($ty:ty, $destination_ty:ty, $direct:ident)),+ $(,)?]) => {
        pub trait $trait_name<Layout>: DataTypeLike {
            type Destination: DataTypeLike;

            fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                template: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self::Destination, Layout>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            impl $trait_name<$layout> for $ty {
                type Destination = $destination_ty;

                fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    template: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self::Destination, $layout>,
                ) -> Result<()> {
                    $direct(stream_context, source, template, destination)
                }
            }
        )+

        pub fn $name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            template: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T::Destination, $layout>,
        ) -> Result<()> {
            T::dispatch(stream_context, source, template, destination)
        }
    };
}

#[path = "statistics_template_basic.rs"]
mod basic;
pub use basic::*;

#[path = "statistics_template_correlation.rs"]
mod correlation;
pub use correlation::*;

#[path = "statistics_template_norm_level.rs"]
mod norm_level;
pub use norm_level::*;

#[path = "statistics_template_advanced.rs"]
mod advanced;
pub use advanced::*;

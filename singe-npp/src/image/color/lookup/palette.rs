use singe_cuda::memory::DeviceMemory;
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color_descriptors::{
            validate_lookup_table_palette, validate_lookup_table_palette_channels,
            validate_lookup_table_palette_entries, validate_same_size,
        },
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView},
    },
    try_ffi,
    types::DataTypeLike,
};

macro_rules! impl_lookup_table_palette_c1 {
    ($doc:literal, $name:ident, $ty:ty, $max_bits:literal, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            destination: &mut ImageOutputView<'_, $ty, C1>,
            table: &DeviceMemory<$ty>,
            bit_size: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;
            stream_context.validate_region(table)?;

            validate_same_size(source.size(), destination.size())?;
            let bit_size = validate_lookup_table_palette(table.len(), bit_size, $max_bits)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    table.as_ptr().cast(),
                    bit_size,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_lookup_table_palette_packed {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $channels:literal, $max_bits:literal, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
            tables: &[&DeviceMemory<$ty>; $channels],
            bit_size: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;
            let bit_size = validate_lookup_table_palette_channels(tables, bit_size, $max_bits)?;
            let mut table_pointers = tables.map(DeviceMemory::as_ptr);

            // SAFETY: the `ffi` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    table_pointers.as_mut_ptr().cast(),
                    bit_size,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_lookup_table_palette_swap {
    ($doc:literal, $name:ident, $ty:ty, $max_bits:literal, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C3>,
            destination: &mut ImageOutputView<'_, $ty, C4>,
            alpha: i32,
            tables: &[&DeviceMemory<$ty>; 3],
            bit_size: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;
            let bit_size = validate_lookup_table_palette_channels(tables, bit_size, $max_bits)?;
            let mut table_pointers = tables.map(DeviceMemory::as_ptr);

            // SAFETY: the `ffi` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    alpha,
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    table_pointers.as_mut_ptr().cast(),
                    bit_size,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_lookup_table_palette_c1_to {
    ($doc:literal,
        $name:ident,
        $source_ty:ty,
        $destination_ty:ty,
        $destination_layout:ty,
        $table_ty:ty,
        $max_bits:literal,
        $values_per_index:literal,
        $ffi:ident
    ) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, C1>,
            destination: &mut ImageOutputView<'_, $destination_ty, $destination_layout>,
            table: &DeviceMemory<$table_ty>,
            bit_size: i32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;
            stream_context.validate_region(table)?;

            validate_same_size(source.size(), destination.size())?;
            let bit_size = validate_lookup_table_palette_entries(
                table.len(),
                bit_size,
                $max_bits,
                $values_per_index,
            )?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    table.as_ptr().cast(),
                    bit_size,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_lookup_table_palette_c1 {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        /// Dispatches a palette lookup for a supported native element type and layout.
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element representation, palette indexing, channel order, and documented execution contract.
        pub unsafe trait $trait: DataTypeLike + Sized {
            #[doc = $doc]
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                destination: &mut ImageOutputView<'_, Self, C1>,
                table: &DeviceMemory<Self>,
                bit_size: i32,
            ) -> Result<()>;
        }

        #[doc = $doc]
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C1>,
            destination: &mut ImageOutputView<'_, T, C1>,
            table: &DeviceMemory<T>,
            bit_size: i32,
        ) -> Result<()>
        where
            T: $trait,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, destination, table, bit_size)
            }
}

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    destination: &mut ImageOutputView<'_, Self, C1>,
                    table: &DeviceMemory<Self>,
                    bit_size: i32,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, table, bit_size)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_lookup_table_palette_packed {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, $layout:ty, $channels:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        /// Dispatches a palette lookup for a supported native element type and layout.
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element representation, palette indexing, channel order, and documented execution contract.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                tables: &[&DeviceMemory<Self>; $channels],
                bit_size: i32,
            ) -> Result<()>;
        }

        #[doc = $doc]
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T, $layout>,
            tables: &[&DeviceMemory<T>; $channels],
            bit_size: i32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, destination, tables, bit_size)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    tables: &[&DeviceMemory<Self>; $channels],
                    bit_size: i32,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, tables, bit_size)
                    }
}
            }
        )*
    };
}

macro_rules! impl_generic_lookup_table_palette_swap {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        /// Dispatches a palette lookup for a supported native element type and layout.
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element representation, palette indexing, channel order, and documented execution contract.
        pub unsafe trait $trait: DataTypeLike + Sized {
            #[doc = $doc]
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C3>,
                destination: &mut ImageOutputView<'_, Self, C4>,
                alpha: i32,
                tables: &[&DeviceMemory<Self>; 3],
                bit_size: i32,
            ) -> Result<()>;
        }

        #[doc = $doc]
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C3>,
            destination: &mut ImageOutputView<'_, T, C4>,
            alpha: i32,
            tables: &[&DeviceMemory<T>; 3],
            bit_size: i32,
        ) -> Result<()>
        where
            T: $trait,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, destination, alpha, tables, bit_size)
            }
}

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C3>,
                    destination: &mut ImageOutputView<'_, Self, C4>,
                    alpha: i32,
                    tables: &[&DeviceMemory<Self>; 3],
                    bit_size: i32,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, alpha, tables, bit_size)
                    }
}
            }
        )*
    };
}

impl_lookup_table_palette_c1!(
    "One channel 8-bit unsigned bit range restricted palette look-up-table color conversion.\n\nUses the least significant `bit_size` bits of each source sample as the palette index. Each palette table must contain at least `2^bit_size` elements.\nEnqueues work on the selected stream without waiting for completion. Row padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, an image belongs to another CUDA context, source and destination sizes differ, `bit_size` is outside `1..=8`, a palette is too short, a required-length calculation overflows, or NPP rejects the operation.\nThe palette allocation must belong to the stream context; its context is checked.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples and palette entries must be initialized and ready for native execution.\nKeep source and palettes read-only and destination storage exclusively writable and disjoint from them during native use.\nRetain all image and palette allocations through queued completion, including submission errors and every captured replay.\nNative pixel layouts, byte strides, alignment, and allocation bounds must be valid.",
    lookup_table_palette_u8_c1,
    u8,
    8,
    nppiLUTPalette_8u_C1R_Ctx
);

impl_lookup_table_palette_c1_to!(
    "One channel 8-bit unsigned bit range restricted 24-bit palette look-up-table color conversion with 24-bit destination output per pixel.\n\nUses the least significant `bit_size` bits of each source sample as the palette index. The palette must contain at least `3 * 2^bit_size` elements, with consecutive output components for each index.\nEnqueues work on the selected stream without waiting for completion. Row padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, an image belongs to another CUDA context, source and destination sizes differ, `bit_size` is outside `1..=8`, a palette is too short, a required-length calculation overflows, or NPP rejects the operation.\nThe palette allocation must belong to the stream context; its context is checked.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples and palette entries must be initialized and ready for native execution.\nKeep source and palettes read-only and destination storage exclusively writable and disjoint from them during native use.\nRetain all image and palette allocations through queued completion, including submission errors and every captured replay.\nNative pixel layouts, byte strides, alignment, and allocation bounds must be valid.",
    lookup_table_palette_u8_to_u8_c3,
    u8,
    u8,
    C3,
    u8,
    8,
    3,
    nppiLUTPalette_8u24u_C1R_Ctx
);

impl_lookup_table_palette_c1_to!(
    "One channel 8-bit unsigned bit range restricted 32-bit palette look-up-table color conversion with 32-bit destination output per pixel.\n\nUses the least significant `bit_size` bits of each source sample as the palette index. The palette must contain at least `1 * 2^bit_size` elements, with consecutive output components for each index.\nEnqueues work on the selected stream without waiting for completion. Row padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, an image belongs to another CUDA context, source and destination sizes differ, `bit_size` is outside `1..=8`, a palette is too short, a required-length calculation overflows, or NPP rejects the operation.\nThe palette allocation must belong to the stream context; its context is checked.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples and palette entries must be initialized and ready for native execution.\nKeep source and palettes read-only and destination storage exclusively writable and disjoint from them during native use.\nRetain all image and palette allocations through queued completion, including submission errors and every captured replay.\nNative pixel layouts, byte strides, alignment, and allocation bounds must be valid.",
    lookup_table_palette_u8_to_u32_c1,
    u8,
    u32,
    C1,
    u32,
    8,
    1,
    nppiLUTPalette_8u32u_C1R_Ctx
);

impl_lookup_table_palette_packed!(
    "Three channel 8-bit unsigned bit range restricted palette look-up-table color conversion.\n\nUses the least significant `bit_size` bits of each source sample as the palette index. Each palette table must contain at least `2^bit_size` elements.\nEnqueues work on the selected stream without waiting for completion. Row padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, an image belongs to another CUDA context, source and destination sizes differ, `bit_size` is outside `1..=8`, a palette is too short, a required-length calculation overflows, or NPP rejects the operation.\nContexts of the per-channel palette allocations are not checked.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples and palette entries must be initialized and ready for native execution.\nKeep source and palettes read-only and destination storage exclusively writable and disjoint from them during native use.\nRetain all image and palette allocations through queued completion, including submission errors and every captured replay.\nNative pixel layouts, byte strides, alignment, and allocation bounds must be valid.\nEvery per-channel palette allocation must belong to the selected CUDA context; establish producer ordering for those tables.",
    lookup_table_palette_u8_c3,
    u8,
    C3,
    3,
    8,
    nppiLUTPalette_8u_C3R_Ctx
);

impl_lookup_table_palette_packed!(
    "Four channel 8-bit unsigned bit range restricted palette look-up-table color conversion.\n\nUses the least significant `bit_size` bits of each source sample as the palette index. Each palette table must contain at least `2^bit_size` elements.\nEnqueues work on the selected stream without waiting for completion. Row padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, an image belongs to another CUDA context, source and destination sizes differ, `bit_size` is outside `1..=8`, a palette is too short, a required-length calculation overflows, or NPP rejects the operation.\nContexts of the per-channel palette allocations are not checked.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples and palette entries must be initialized and ready for native execution.\nKeep source and palettes read-only and destination storage exclusively writable and disjoint from them during native use.\nRetain all image and palette allocations through queued completion, including submission errors and every captured replay.\nNative pixel layouts, byte strides, alignment, and allocation bounds must be valid.\nEvery per-channel palette allocation must belong to the selected CUDA context; establish producer ordering for those tables.",
    lookup_table_palette_u8_c4,
    u8,
    C4,
    4,
    8,
    nppiLUTPalette_8u_C4R_Ctx
);

impl_lookup_table_palette_packed!(
    "Four channel 8-bit unsigned bit range restricted palette look-up-table color conversion, not affecting Alpha.\n\nUses the least significant `bit_size` bits of each source sample as the palette index. Each palette table must contain at least `2^bit_size` elements. Leaves the last destination alpha channel unwritten.\nEnqueues work on the selected stream without waiting for completion. Row padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, an image belongs to another CUDA context, source and destination sizes differ, `bit_size` is outside `1..=8`, a palette is too short, a required-length calculation overflows, or NPP rejects the operation.\nContexts of the per-channel palette allocations are not checked.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples and palette entries must be initialized and ready for native execution.\nKeep source and palettes read-only and destination storage exclusively writable and disjoint from them during native use.\nRetain all image and palette allocations through queued completion, including submission errors and every captured replay.\nNative pixel layouts, byte strides, alignment, and allocation bounds must be valid.\nEvery per-channel palette allocation must belong to the selected CUDA context; establish producer ordering for those tables.",
    lookup_table_palette_u8_ac4,
    u8,
    AC4,
    3,
    8,
    nppiLUTPalette_8u_AC4R_Ctx
);

impl_lookup_table_palette_swap!(
    "Three channel 8-bit unsigned source bit range restricted palette look-up-table color conversion to four channel 8-bit unsigned destination output with alpha. Uses the native C3-to-C4 palette-swap channel order. Writes destination alpha only when `alpha` is in the unsigned sample range; otherwise leaves it untouched.\n\nUses the least significant `bit_size` bits of each source sample as the palette index. Each palette table must contain at least `2^bit_size` elements.\nEnqueues work on the selected stream without waiting for completion. Row padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, an image belongs to another CUDA context, source and destination sizes differ, `bit_size` is outside `1..=8`, a palette is too short, a required-length calculation overflows, or NPP rejects the operation.\nContexts of the per-channel palette allocations are not checked.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples and palette entries must be initialized and ready for native execution.\nKeep source and palettes read-only and destination storage exclusively writable and disjoint from them during native use.\nRetain all image and palette allocations through queued completion, including submission errors and every captured replay.\nNative pixel layouts, byte strides, alignment, and allocation bounds must be valid.\nEvery per-channel palette allocation must belong to the selected CUDA context; establish producer ordering for those tables.",
    lookup_table_palette_swap_u8_c3_to_c4,
    u8,
    8,
    nppiLUTPaletteSwap_8u_C3A0C4R_Ctx
);

impl_lookup_table_palette_c1!(
    "One channel 16-bit unsigned bit range restricted palette look-up-table color conversion.\n\nUses the least significant `bit_size` bits of each source sample as the palette index. Each palette table must contain at least `2^bit_size` elements.\nEnqueues work on the selected stream without waiting for completion. Row padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, an image belongs to another CUDA context, source and destination sizes differ, `bit_size` is outside `1..=16`, a palette is too short, a required-length calculation overflows, or NPP rejects the operation.\nThe palette allocation must belong to the stream context; its context is checked.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples and palette entries must be initialized and ready for native execution.\nKeep source and palettes read-only and destination storage exclusively writable and disjoint from them during native use.\nRetain all image and palette allocations through queued completion, including submission errors and every captured replay.\nNative pixel layouts, byte strides, alignment, and allocation bounds must be valid.",
    lookup_table_palette_u16_c1,
    u16,
    16,
    nppiLUTPalette_16u_C1R_Ctx
);

impl_lookup_table_palette_c1_to!(
    "One channel 16-bit unsigned bit range restricted 8-bit unsigned palette look-up-table color conversion with 8-bit unsigned destination output per pixel.\n\nUses the least significant `bit_size` bits of each source sample as the palette index. The palette must contain at least `1 * 2^bit_size` elements, with consecutive output components for each index.\nEnqueues work on the selected stream without waiting for completion. Row padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, an image belongs to another CUDA context, source and destination sizes differ, `bit_size` is outside `1..=16`, a palette is too short, a required-length calculation overflows, or NPP rejects the operation.\nThe palette allocation must belong to the stream context; its context is checked.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples and palette entries must be initialized and ready for native execution.\nKeep source and palettes read-only and destination storage exclusively writable and disjoint from them during native use.\nRetain all image and palette allocations through queued completion, including submission errors and every captured replay.\nNative pixel layouts, byte strides, alignment, and allocation bounds must be valid.",
    lookup_table_palette_u16_to_u8_c1,
    u16,
    u8,
    C1,
    u8,
    16,
    1,
    nppiLUTPalette_16u8u_C1R_Ctx
);

impl_lookup_table_palette_c1_to!(
    "One channel 16-bit unsigned bit range restricted 24-bit unsigned palette look-up-table color conversion with 24-bit unsigned destination output per pixel.\n\nUses the least significant `bit_size` bits of each source sample as the palette index. The palette must contain at least `3 * 2^bit_size` elements, with consecutive output components for each index.\nEnqueues work on the selected stream without waiting for completion. Row padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, an image belongs to another CUDA context, source and destination sizes differ, `bit_size` is outside `1..=16`, a palette is too short, a required-length calculation overflows, or NPP rejects the operation.\nThe palette allocation must belong to the stream context; its context is checked.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples and palette entries must be initialized and ready for native execution.\nKeep source and palettes read-only and destination storage exclusively writable and disjoint from them during native use.\nRetain all image and palette allocations through queued completion, including submission errors and every captured replay.\nNative pixel layouts, byte strides, alignment, and allocation bounds must be valid.",
    lookup_table_palette_u16_to_u8_c3,
    u16,
    u8,
    C3,
    u8,
    16,
    3,
    nppiLUTPalette_16u24u_C1R_Ctx
);

impl_lookup_table_palette_c1_to!(
    "One channel 16-bit unsigned bit range restricted 32-bit palette look-up-table color conversion with 32-bit unsigned destination output per pixel.\n\nUses the least significant `bit_size` bits of each source sample as the palette index. The palette must contain at least `1 * 2^bit_size` elements, with consecutive output components for each index.\nEnqueues work on the selected stream without waiting for completion. Row padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, an image belongs to another CUDA context, source and destination sizes differ, `bit_size` is outside `1..=16`, a palette is too short, a required-length calculation overflows, or NPP rejects the operation.\nThe palette allocation must belong to the stream context; its context is checked.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples and palette entries must be initialized and ready for native execution.\nKeep source and palettes read-only and destination storage exclusively writable and disjoint from them during native use.\nRetain all image and palette allocations through queued completion, including submission errors and every captured replay.\nNative pixel layouts, byte strides, alignment, and allocation bounds must be valid.",
    lookup_table_palette_u16_to_u32_c1,
    u16,
    u32,
    C1,
    u32,
    16,
    1,
    nppiLUTPalette_16u32u_C1R_Ctx
);

impl_lookup_table_palette_packed!(
    "Three channel 16-bit unsigned bit range restricted palette look-up-table color conversion.\n\nUses the least significant `bit_size` bits of each source sample as the palette index. Each palette table must contain at least `2^bit_size` elements.\nEnqueues work on the selected stream without waiting for completion. Row padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, an image belongs to another CUDA context, source and destination sizes differ, `bit_size` is outside `1..=16`, a palette is too short, a required-length calculation overflows, or NPP rejects the operation.\nContexts of the per-channel palette allocations are not checked.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples and palette entries must be initialized and ready for native execution.\nKeep source and palettes read-only and destination storage exclusively writable and disjoint from them during native use.\nRetain all image and palette allocations through queued completion, including submission errors and every captured replay.\nNative pixel layouts, byte strides, alignment, and allocation bounds must be valid.\nEvery per-channel palette allocation must belong to the selected CUDA context; establish producer ordering for those tables.",
    lookup_table_palette_u16_c3,
    u16,
    C3,
    3,
    16,
    nppiLUTPalette_16u_C3R_Ctx
);

impl_lookup_table_palette_packed!(
    "Four channel 16-bit unsigned bit range restricted palette look-up-table color conversion.\n\nUses the least significant `bit_size` bits of each source sample as the palette index. Each palette table must contain at least `2^bit_size` elements.\nEnqueues work on the selected stream without waiting for completion. Row padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, an image belongs to another CUDA context, source and destination sizes differ, `bit_size` is outside `1..=16`, a palette is too short, a required-length calculation overflows, or NPP rejects the operation.\nContexts of the per-channel palette allocations are not checked.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples and palette entries must be initialized and ready for native execution.\nKeep source and palettes read-only and destination storage exclusively writable and disjoint from them during native use.\nRetain all image and palette allocations through queued completion, including submission errors and every captured replay.\nNative pixel layouts, byte strides, alignment, and allocation bounds must be valid.\nEvery per-channel palette allocation must belong to the selected CUDA context; establish producer ordering for those tables.",
    lookup_table_palette_u16_c4,
    u16,
    C4,
    4,
    16,
    nppiLUTPalette_16u_C4R_Ctx
);

impl_lookup_table_palette_packed!(
    "Four channel 16-bit unsigned bit range restricted palette look-up-table color conversion, not affecting Alpha.\n\nUses the least significant `bit_size` bits of each source sample as the palette index. Each palette table must contain at least `2^bit_size` elements. Leaves the last destination alpha channel unwritten.\nEnqueues work on the selected stream without waiting for completion. Row padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, an image belongs to another CUDA context, source and destination sizes differ, `bit_size` is outside `1..=16`, a palette is too short, a required-length calculation overflows, or NPP rejects the operation.\nContexts of the per-channel palette allocations are not checked.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples and palette entries must be initialized and ready for native execution.\nKeep source and palettes read-only and destination storage exclusively writable and disjoint from them during native use.\nRetain all image and palette allocations through queued completion, including submission errors and every captured replay.\nNative pixel layouts, byte strides, alignment, and allocation bounds must be valid.\nEvery per-channel palette allocation must belong to the selected CUDA context; establish producer ordering for those tables.",
    lookup_table_palette_u16_ac4,
    u16,
    AC4,
    3,
    16,
    nppiLUTPalette_16u_AC4R_Ctx
);

impl_lookup_table_palette_swap!(
    "Three channel 16-bit unsigned source bit range restricted palette look-up-table color conversion to four channel 16-bit unsigned destination output with alpha. Uses the native C3-to-C4 palette-swap channel order. Writes destination alpha only when `alpha` is in the unsigned sample range; otherwise leaves it untouched.\n\nUses the least significant `bit_size` bits of each source sample as the palette index. Each palette table must contain at least `2^bit_size` elements.\nEnqueues work on the selected stream without waiting for completion. Row padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, an image belongs to another CUDA context, source and destination sizes differ, `bit_size` is outside `1..=16`, a palette is too short, a required-length calculation overflows, or NPP rejects the operation.\nContexts of the per-channel palette allocations are not checked.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples and palette entries must be initialized and ready for native execution.\nKeep source and palettes read-only and destination storage exclusively writable and disjoint from them during native use.\nRetain all image and palette allocations through queued completion, including submission errors and every captured replay.\nNative pixel layouts, byte strides, alignment, and allocation bounds must be valid.\nEvery per-channel palette allocation must belong to the selected CUDA context; establish producer ordering for those tables.",
    lookup_table_palette_swap_u16_c3_to_c4,
    u16,
    16,
    nppiLUTPaletteSwap_16u_C3A0C4R_Ctx
);

impl_generic_lookup_table_palette_c1!(
    "Maps source samples through a device palette using the least significant `bit_size` bits as the palette index.\n\nUses the least significant `bit_size` bits of each source sample as the palette index. Each palette table must contain at least `2^bit_size` elements.\nEnqueues work on the selected stream without waiting for completion. Row padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, an image belongs to another CUDA context, source and destination sizes differ, `bit_size` is outside the range from one through the source element width in bits, a palette is too short, a required-length calculation overflows, or NPP rejects the operation.\nThe palette allocation must belong to the stream context; its context is checked.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples and palette entries must be initialized and ready for native execution.\nKeep source and palettes read-only and destination storage exclusively writable and disjoint from them during native use.\nRetain all image and palette allocations through queued completion, including submission errors and every captured replay.\nNative pixel layouts, byte strides, alignment, and allocation bounds must be valid.",

    LookupTablePaletteC1,
    lookup_table_palette,
    lookup_table_palette_c1,
    [
        u8 => lookup_table_palette_u8_c1,
        u16 => lookup_table_palette_u16_c1,
    ]

);

impl_generic_lookup_table_palette_packed!(
    "Maps source samples through a device palette using the least significant `bit_size` bits as the palette index.\n\nUses the least significant `bit_size` bits of each source sample as the palette index. Each palette table must contain at least `2^bit_size` elements.\nEnqueues work on the selected stream without waiting for completion. Row padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, an image belongs to another CUDA context, source and destination sizes differ, `bit_size` is outside the range from one through the source element width in bits, a palette is too short, a required-length calculation overflows, or NPP rejects the operation.\nContexts of the per-channel palette allocations are not checked.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples and palette entries must be initialized and ready for native execution.\nKeep source and palettes read-only and destination storage exclusively writable and disjoint from them during native use.\nRetain all image and palette allocations through queued completion, including submission errors and every captured replay.\nNative pixel layouts, byte strides, alignment, and allocation bounds must be valid.\nEvery per-channel palette allocation must belong to the selected CUDA context; establish producer ordering for those tables.",

    LookupTablePaletteC3,
    lookup_table_palette,
    lookup_table_palette_c3,
    C3,
    3,
    [
        u8 => lookup_table_palette_u8_c3,
        u16 => lookup_table_palette_u16_c3,
    ]

);

impl_generic_lookup_table_palette_packed!(
    "Maps source samples through a device palette using the least significant `bit_size` bits as the palette index.\n\nUses the least significant `bit_size` bits of each source sample as the palette index. Each palette table must contain at least `2^bit_size` elements.\nEnqueues work on the selected stream without waiting for completion. Row padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, an image belongs to another CUDA context, source and destination sizes differ, `bit_size` is outside the range from one through the source element width in bits, a palette is too short, a required-length calculation overflows, or NPP rejects the operation.\nContexts of the per-channel palette allocations are not checked.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples and palette entries must be initialized and ready for native execution.\nKeep source and palettes read-only and destination storage exclusively writable and disjoint from them during native use.\nRetain all image and palette allocations through queued completion, including submission errors and every captured replay.\nNative pixel layouts, byte strides, alignment, and allocation bounds must be valid.\nEvery per-channel palette allocation must belong to the selected CUDA context; establish producer ordering for those tables.",

    LookupTablePaletteC4,
    lookup_table_palette,
    lookup_table_palette_c4,
    C4,
    4,
    [
        u8 => lookup_table_palette_u8_c4,
        u16 => lookup_table_palette_u16_c4,
    ]

);

impl_generic_lookup_table_palette_packed!(
    "Maps source samples through a device palette using the least significant `bit_size` bits as the palette index.\n\nUses the least significant `bit_size` bits of each source sample as the palette index. Each palette table must contain at least `2^bit_size` elements. Leaves the last destination alpha channel unwritten.\nEnqueues work on the selected stream without waiting for completion. Row padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, an image belongs to another CUDA context, source and destination sizes differ, `bit_size` is outside the range from one through the source element width in bits, a palette is too short, a required-length calculation overflows, or NPP rejects the operation.\nContexts of the per-channel palette allocations are not checked.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples and palette entries must be initialized and ready for native execution.\nKeep source and palettes read-only and destination storage exclusively writable and disjoint from them during native use.\nRetain all image and palette allocations through queued completion, including submission errors and every captured replay.\nNative pixel layouts, byte strides, alignment, and allocation bounds must be valid.\nEvery per-channel palette allocation must belong to the selected CUDA context; establish producer ordering for those tables.",

    LookupTablePaletteAC4,
    lookup_table_palette,
    lookup_table_palette_ac4,
    AC4,
    3,
    [
        u8 => lookup_table_palette_u8_ac4,
        u16 => lookup_table_palette_u16_ac4,
    ]

);

impl_generic_lookup_table_palette_swap!(
    "Maps three source channels through device palettes using the native C3-to-C4 palette-swap channel order. Writes destination alpha only when `alpha` is in the unsigned sample range; otherwise leaves it untouched.\n\nUses the least significant `bit_size` bits of each source sample as the palette index. Each palette table must contain at least `2^bit_size` elements.\nEnqueues work on the selected stream without waiting for completion. Row padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, an image belongs to another CUDA context, source and destination sizes differ, `bit_size` is outside the range from one through the source element width in bits, a palette is too short, a required-length calculation overflows, or NPP rejects the operation.\nContexts of the per-channel palette allocations are not checked.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples and palette entries must be initialized and ready for native execution.\nKeep source and palettes read-only and destination storage exclusively writable and disjoint from them during native use.\nRetain all image and palette allocations through queued completion, including submission errors and every captured replay.\nNative pixel layouts, byte strides, alignment, and allocation bounds must be valid.\nEvery per-channel palette allocation must belong to the selected CUDA context; establish producer ordering for those tables.",

    LookupTablePaletteSwapC3ToC4,
    lookup_table_palette_swap,
    lookup_table_palette_swap_c3_to_c4,
    [
        u8 => lookup_table_palette_swap_u8_c3_to_c4,
        u16 => lookup_table_palette_swap_u16_c3_to_c4,
    ]

);

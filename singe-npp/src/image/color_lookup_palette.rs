use super::*;

macro_rules! impl_lookup_table_palette_c1 {
    ($name:ident, $ty:ty, $max_bits:literal, $ffi:ident) => {
        pub(crate) fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            destination: &mut ImageViewMut<'_, $ty, C1>,
            table: &DeviceMemory<$ty>,
            bit_size: i32,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            let bit_size = validate_lookup_table_palette(table.len(), bit_size, $max_bits)?;

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
    ($name:ident, $ty:ty, $layout:ty, $channels:literal, $max_bits:literal, $ffi:ident) => {
        pub(crate) fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            tables: &[&DeviceMemory<$ty>; $channels],
            bit_size: i32,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            let bit_size = validate_lookup_table_palette_channels(tables, bit_size, $max_bits)?;
            let mut table_pointers = tables.map(DeviceMemory::as_ptr);

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
    ($name:ident, $ty:ty, $max_bits:literal, $ffi:ident) => {
        pub(crate) fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C3>,
            destination: &mut ImageViewMut<'_, $ty, C4>,
            alpha: i32,
            tables: &[&DeviceMemory<$ty>; 3],
            bit_size: i32,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            let bit_size = validate_lookup_table_palette_channels(tables, bit_size, $max_bits)?;
            let mut table_pointers = tables.map(DeviceMemory::as_ptr);

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
    (
        $name:ident,
        $source_ty:ty,
        $destination_ty:ty,
        $destination_layout:ty,
        $table_ty:ty,
        $max_bits:literal,
        $values_per_index:literal,
        $ffi:ident
    ) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, C1>,
            destination: &mut ImageViewMut<'_, $destination_ty, $destination_layout>,
            table: &DeviceMemory<$table_ty>,
            bit_size: i32,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            let bit_size = validate_lookup_table_palette_entries(
                table.len(),
                bit_size,
                $max_bits,
                $values_per_index,
            )?;

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
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                destination: &mut ImageViewMut<'_, Self, C1>,
                table: &DeviceMemory<Self>,
                bit_size: i32,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C1>,
            destination: &mut ImageViewMut<'_, T, C1>,
            table: &DeviceMemory<T>,
            bit_size: i32,
        ) -> Result<()>
        where
            T: $trait,
        {
            T::$method(stream_context, source, destination, table, bit_size)
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    destination: &mut ImageViewMut<'_, Self, C1>,
                    table: &DeviceMemory<Self>,
                    bit_size: i32,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, table, bit_size)
                }
            }
        )*
    };
}

macro_rules! impl_generic_lookup_table_palette_packed {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, $channels:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageViewMut<'_, Self, Layout>,
                tables: &[&DeviceMemory<Self>; $channels],
                bit_size: i32,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageViewMut<'_, T, $layout>,
            tables: &[&DeviceMemory<T>; $channels],
            bit_size: i32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source, destination, tables, bit_size)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    tables: &[&DeviceMemory<Self>; $channels],
                    bit_size: i32,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, tables, bit_size)
                }
            }
        )*
    };
}

macro_rules! impl_generic_lookup_table_palette_swap {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C3>,
                destination: &mut ImageViewMut<'_, Self, C4>,
                alpha: i32,
                tables: &[&DeviceMemory<Self>; 3],
                bit_size: i32,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C3>,
            destination: &mut ImageViewMut<'_, T, C4>,
            alpha: i32,
            tables: &[&DeviceMemory<T>; 3],
            bit_size: i32,
        ) -> Result<()>
        where
            T: $trait,
        {
            T::$method(stream_context, source, destination, alpha, tables, bit_size)
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C3>,
                    destination: &mut ImageViewMut<'_, Self, C4>,
                    alpha: i32,
                    tables: &[&DeviceMemory<Self>; 3],
                    bit_size: i32,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, alpha, tables, bit_size)
                }
            }
        )*
    };
}

impl_lookup_table_palette_c1!(lookup_table_palette_u8_c1, u8, 8, nppiLUTPalette_8u_C1R_Ctx);
impl_lookup_table_palette_c1_to!(
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
    lookup_table_palette_u8_c3,
    u8,
    C3,
    3,
    8,
    nppiLUTPalette_8u_C3R_Ctx
);
impl_lookup_table_palette_packed!(
    lookup_table_palette_u8_c4,
    u8,
    C4,
    4,
    8,
    nppiLUTPalette_8u_C4R_Ctx
);
impl_lookup_table_palette_packed!(
    lookup_table_palette_u8_ac4,
    u8,
    AC4,
    3,
    8,
    nppiLUTPalette_8u_AC4R_Ctx
);
impl_lookup_table_palette_swap!(
    lookup_table_palette_swap_u8_c3_to_c4,
    u8,
    8,
    nppiLUTPaletteSwap_8u_C3A0C4R_Ctx
);
impl_lookup_table_palette_c1!(
    lookup_table_palette_u16_c1,
    u16,
    16,
    nppiLUTPalette_16u_C1R_Ctx
);
impl_lookup_table_palette_c1_to!(
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
    lookup_table_palette_u16_c3,
    u16,
    C3,
    3,
    16,
    nppiLUTPalette_16u_C3R_Ctx
);
impl_lookup_table_palette_packed!(
    lookup_table_palette_u16_c4,
    u16,
    C4,
    4,
    16,
    nppiLUTPalette_16u_C4R_Ctx
);
impl_lookup_table_palette_packed!(
    lookup_table_palette_u16_ac4,
    u16,
    AC4,
    3,
    16,
    nppiLUTPalette_16u_AC4R_Ctx
);
impl_lookup_table_palette_swap!(
    lookup_table_palette_swap_u16_c3_to_c4,
    u16,
    16,
    nppiLUTPaletteSwap_16u_C3A0C4R_Ctx
);
impl_generic_lookup_table_palette_c1!(
    LookupTablePaletteC1,
    lookup_table_palette,
    lookup_table_palette_c1,
    [
        u8 => lookup_table_palette_u8_c1,
        u16 => lookup_table_palette_u16_c1,
    ]
);
impl_generic_lookup_table_palette_packed!(
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
    LookupTablePaletteSwapC3ToC4,
    lookup_table_palette_swap,
    lookup_table_palette_swap_c3_to_c4,
    [
        u8 => lookup_table_palette_swap_u8_c3_to_c4,
        u16 => lookup_table_palette_swap_u16_c3_to_c4,
    ]
);

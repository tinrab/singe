use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::view::{AC4, C1, C3, C4, ImageView, MaskViewMut},
    try_ffi,
    types::{ComparisonOperation, DataTypeLike, Size},
};

macro_rules! impl_generic_compare {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source1: &ImageView<'_, Self, Layout>,
                source2: &ImageView<'_, Self, Layout>,
                destination: &mut MaskViewMut<'_>,
                operation: ComparisonOperation,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source1: &ImageView<'_, T, $layout>,
            source2: &ImageView<'_, T, $layout>,
            destination: &mut MaskViewMut<'_>,
            operation: ComparisonOperation,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source1, source2, destination, operation)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source1: &ImageView<'_, Self, $layout>,
                    source2: &ImageView<'_, Self, $layout>,
                    destination: &mut MaskViewMut<'_>,
                    operation: ComparisonOperation,
                ) -> Result<()> {
                    $direct(stream_context, source1, source2, destination, operation)
                }
            }
        )*
    };
}

macro_rules! impl_generic_compare_constant_scalar {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                constant: Self,
                destination: &mut MaskViewMut<'_>,
                operation: ComparisonOperation,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            constant: T,
            destination: &mut MaskViewMut<'_>,
            operation: ComparisonOperation,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source, constant, destination, operation)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    constant: Self,
                    destination: &mut MaskViewMut<'_>,
                    operation: ComparisonOperation,
                ) -> Result<()> {
                    $direct(stream_context, source, constant, destination, operation)
                }
            }
        )*
    };
}

macro_rules! impl_generic_compare_constant_array {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, $channels:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                constants: [Self; $channels],
                destination: &mut MaskViewMut<'_>,
                operation: ComparisonOperation,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            constants: [T; $channels],
            destination: &mut MaskViewMut<'_>,
            operation: ComparisonOperation,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source, constants, destination, operation)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    constants: [Self; $channels],
                    destination: &mut MaskViewMut<'_>,
                    operation: ComparisonOperation,
                ) -> Result<()> {
                    $direct(stream_context, source, constants, destination, operation)
                }
            }
        )*
    };
}

macro_rules! impl_generic_compare_equal_epsilon {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source1: &ImageView<'_, Self, Layout>,
                source2: &ImageView<'_, Self, Layout>,
                destination: &mut MaskViewMut<'_>,
                epsilon: Self,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source1: &ImageView<'_, T, $layout>,
            source2: &ImageView<'_, T, $layout>,
            destination: &mut MaskViewMut<'_>,
            epsilon: T,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source1, source2, destination, epsilon)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source1: &ImageView<'_, Self, $layout>,
                    source2: &ImageView<'_, Self, $layout>,
                    destination: &mut MaskViewMut<'_>,
                    epsilon: Self,
                ) -> Result<()> {
                    $direct(stream_context, source1, source2, destination, epsilon)
                }
            }
        )*
    };
}

macro_rules! impl_generic_compare_equal_epsilon_constant_scalar {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                constant: Self,
                destination: &mut MaskViewMut<'_>,
                epsilon: Self,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            constant: T,
            destination: &mut MaskViewMut<'_>,
            epsilon: T,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source, constant, destination, epsilon)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    constant: Self,
                    destination: &mut MaskViewMut<'_>,
                    epsilon: Self,
                ) -> Result<()> {
                    $direct(stream_context, source, constant, destination, epsilon)
                }
            }
        )*
    };
}

macro_rules! impl_generic_compare_equal_epsilon_constant_array {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, $channels:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait<Layout>: DataTypeLike + Sized {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                constants: [Self; $channels],
                destination: &mut MaskViewMut<'_>,
                epsilon: Self,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            constants: [T; $channels],
            destination: &mut MaskViewMut<'_>,
            epsilon: T,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            T::$method(stream_context, source, constants, destination, epsilon)
        }

        $(
            impl $trait<$layout> for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    constants: [Self; $channels],
                    destination: &mut MaskViewMut<'_>,
                    epsilon: Self,
                ) -> Result<()> {
                    $direct(stream_context, source, constants, destination, epsilon)
                }
            }
        )*
    };
}

macro_rules! impl_compare_c1 {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source1: &ImageView<'_, $ty, C1>,
            source2: &ImageView<'_, $ty, C1>,
            destination: &mut MaskViewMut<'_>,
            operation: ComparisonOperation,
        ) -> Result<()> {
            validate_same_size(source1.size(), source2.size())?;
            validate_same_size(source1.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source1.as_ptr().cast(),
                    source1.step(),
                    source2.as_ptr().cast(),
                    source2.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source1.size().into(),
                    operation.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

impl_compare_c1!(compare_u8_c1, u8, nppiCompare_8u_C1R_Ctx);
impl_compare_c1!(compare_u16_c1, u16, nppiCompare_16u_C1R_Ctx);
impl_compare_c1!(compare_i16_c1, i16, nppiCompare_16s_C1R_Ctx);
impl_compare_c1!(compare_f32_c1, f32, nppiCompare_32f_C1R_Ctx);

macro_rules! impl_compare_packed {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source1: &ImageView<'_, $ty, $layout>,
            source2: &ImageView<'_, $ty, $layout>,
            destination: &mut MaskViewMut<'_>,
            operation: ComparisonOperation,
        ) -> Result<()> {
            validate_same_size(source1.size(), source2.size())?;
            validate_same_size(source1.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source1.as_ptr().cast(),
                    source1.step(),
                    source2.as_ptr().cast(),
                    source2.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source1.size().into(),
                    operation.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

impl_compare_packed!(compare_u8_c3, u8, C3, nppiCompare_8u_C3R_Ctx);
impl_compare_packed!(compare_u8_c4, u8, C4, nppiCompare_8u_C4R_Ctx);
impl_compare_packed!(compare_u8_ac4, u8, AC4, nppiCompare_8u_AC4R_Ctx);
impl_compare_packed!(compare_u16_c3, u16, C3, nppiCompare_16u_C3R_Ctx);
impl_compare_packed!(compare_u16_c4, u16, C4, nppiCompare_16u_C4R_Ctx);
impl_compare_packed!(compare_u16_ac4, u16, AC4, nppiCompare_16u_AC4R_Ctx);
impl_compare_packed!(compare_i16_c3, i16, C3, nppiCompare_16s_C3R_Ctx);
impl_compare_packed!(compare_i16_c4, i16, C4, nppiCompare_16s_C4R_Ctx);
impl_compare_packed!(compare_i16_ac4, i16, AC4, nppiCompare_16s_AC4R_Ctx);
impl_compare_packed!(compare_f32_c3, f32, C3, nppiCompare_32f_C3R_Ctx);
impl_compare_packed!(compare_f32_c4, f32, C4, nppiCompare_32f_C4R_Ctx);
impl_compare_packed!(compare_f32_ac4, f32, AC4, nppiCompare_32f_AC4R_Ctx);

impl_generic_compare!(CompareC1, compare, compare_c1, C1, [
    u8 => compare_u8_c1,
    u16 => compare_u16_c1,
    i16 => compare_i16_c1,
    f32 => compare_f32_c1,
]);
impl_generic_compare!(CompareC3, compare, compare_c3, C3, [
    u8 => compare_u8_c3,
    u16 => compare_u16_c3,
    i16 => compare_i16_c3,
    f32 => compare_f32_c3,
]);
impl_generic_compare!(CompareC4, compare, compare_c4, C4, [
    u8 => compare_u8_c4,
    u16 => compare_u16_c4,
    i16 => compare_i16_c4,
    f32 => compare_f32_c4,
]);
impl_generic_compare!(CompareAc4, compare, compare_ac4, AC4, [
    u8 => compare_u8_ac4,
    u16 => compare_u16_ac4,
    i16 => compare_i16_ac4,
    f32 => compare_f32_ac4,
]);

macro_rules! impl_compare_constant_c1 {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            constant: $ty,
            destination: &mut MaskViewMut<'_>,
            operation: ComparisonOperation,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    constant,
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    operation.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

impl_compare_constant_c1!(compare_constant_u8_c1, u8, nppiCompareC_8u_C1R_Ctx);
impl_compare_constant_c1!(compare_constant_u16_c1, u16, nppiCompareC_16u_C1R_Ctx);
impl_compare_constant_c1!(compare_constant_i16_c1, i16, nppiCompareC_16s_C1R_Ctx);
impl_compare_constant_c1!(compare_constant_f32_c1, f32, nppiCompareC_32f_C1R_Ctx);

macro_rules! impl_compare_constant_packed {
    ($name:ident, $ty:ty, $layout:ty, $channels:expr, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            constants: [$ty; $channels],
            destination: &mut MaskViewMut<'_>,
            operation: ComparisonOperation,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    constants.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    operation.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

impl_compare_constant_packed!(compare_constant_u8_c3, u8, C3, 3, nppiCompareC_8u_C3R_Ctx);
impl_compare_constant_packed!(compare_constant_u8_c4, u8, C4, 4, nppiCompareC_8u_C4R_Ctx);
impl_compare_constant_packed!(
    compare_constant_u8_ac4,
    u8,
    AC4,
    3,
    nppiCompareC_8u_AC4R_Ctx
);
impl_compare_constant_packed!(
    compare_constant_u16_c3,
    u16,
    C3,
    3,
    nppiCompareC_16u_C3R_Ctx
);
impl_compare_constant_packed!(
    compare_constant_u16_c4,
    u16,
    C4,
    4,
    nppiCompareC_16u_C4R_Ctx
);
impl_compare_constant_packed!(
    compare_constant_u16_ac4,
    u16,
    AC4,
    3,
    nppiCompareC_16u_AC4R_Ctx
);
impl_compare_constant_packed!(
    compare_constant_i16_c3,
    i16,
    C3,
    3,
    nppiCompareC_16s_C3R_Ctx
);
impl_compare_constant_packed!(
    compare_constant_i16_c4,
    i16,
    C4,
    4,
    nppiCompareC_16s_C4R_Ctx
);
impl_compare_constant_packed!(
    compare_constant_i16_ac4,
    i16,
    AC4,
    3,
    nppiCompareC_16s_AC4R_Ctx
);
impl_compare_constant_packed!(
    compare_constant_f32_c3,
    f32,
    C3,
    3,
    nppiCompareC_32f_C3R_Ctx
);
impl_compare_constant_packed!(
    compare_constant_f32_c4,
    f32,
    C4,
    4,
    nppiCompareC_32f_C4R_Ctx
);
impl_compare_constant_packed!(
    compare_constant_f32_ac4,
    f32,
    AC4,
    3,
    nppiCompareC_32f_AC4R_Ctx
);

impl_generic_compare_constant_scalar!(
    CompareConstantC1,
    compare_constant,
    compare_constant_c1,
    C1,
    [
        u8 => compare_constant_u8_c1,
        u16 => compare_constant_u16_c1,
        i16 => compare_constant_i16_c1,
        f32 => compare_constant_f32_c1,
    ]
);
impl_generic_compare_constant_array!(
    CompareConstantC3,
    compare_constant,
    compare_constant_c3,
    C3,
    3,
    [
        u8 => compare_constant_u8_c3,
        u16 => compare_constant_u16_c3,
        i16 => compare_constant_i16_c3,
        f32 => compare_constant_f32_c3,
    ]
);
impl_generic_compare_constant_array!(
    CompareConstantC4,
    compare_constant,
    compare_constant_c4,
    C4,
    4,
    [
        u8 => compare_constant_u8_c4,
        u16 => compare_constant_u16_c4,
        i16 => compare_constant_i16_c4,
        f32 => compare_constant_f32_c4,
    ]
);
impl_generic_compare_constant_array!(
    CompareConstantAc4,
    compare_constant,
    compare_constant_ac4,
    AC4,
    3,
    [
        u8 => compare_constant_u8_ac4,
        u16 => compare_constant_u16_ac4,
        i16 => compare_constant_i16_ac4,
        f32 => compare_constant_f32_ac4,
    ]
);

macro_rules! impl_compare_equal_epsilon {
    ($name:ident, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source1: &ImageView<'_, f32, $layout>,
            source2: &ImageView<'_, f32, $layout>,
            destination: &mut MaskViewMut<'_>,
            epsilon: f32,
        ) -> Result<()> {
            validate_same_size(source1.size(), source2.size())?;
            validate_same_size(source1.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source1.as_ptr().cast(),
                    source1.step(),
                    source2.as_ptr().cast(),
                    source2.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source1.size().into(),
                    epsilon,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

impl_compare_equal_epsilon!(
    compare_equal_epsilon_f32_c1,
    C1,
    nppiCompareEqualEps_32f_C1R_Ctx
);
impl_compare_equal_epsilon!(
    compare_equal_epsilon_f32_c3,
    C3,
    nppiCompareEqualEps_32f_C3R_Ctx
);
impl_compare_equal_epsilon!(
    compare_equal_epsilon_f32_c4,
    C4,
    nppiCompareEqualEps_32f_C4R_Ctx
);
impl_compare_equal_epsilon!(
    compare_equal_epsilon_f32_ac4,
    AC4,
    nppiCompareEqualEps_32f_AC4R_Ctx
);

impl_generic_compare_equal_epsilon!(
    CompareEqualEpsilonC1,
    compare_equal_epsilon,
    compare_equal_epsilon_c1,
    C1,
    [f32 => compare_equal_epsilon_f32_c1]
);
impl_generic_compare_equal_epsilon!(
    CompareEqualEpsilonC3,
    compare_equal_epsilon,
    compare_equal_epsilon_c3,
    C3,
    [f32 => compare_equal_epsilon_f32_c3]
);
impl_generic_compare_equal_epsilon!(
    CompareEqualEpsilonC4,
    compare_equal_epsilon,
    compare_equal_epsilon_c4,
    C4,
    [f32 => compare_equal_epsilon_f32_c4]
);
impl_generic_compare_equal_epsilon!(
    CompareEqualEpsilonAc4,
    compare_equal_epsilon,
    compare_equal_epsilon_ac4,
    AC4,
    [f32 => compare_equal_epsilon_f32_ac4]
);

macro_rules! impl_compare_equal_epsilon_constant_c1 {
    ($name:ident, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, f32, C1>,
            constant: f32,
            destination: &mut MaskViewMut<'_>,
            epsilon: f32,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    constant,
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    epsilon,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_compare_equal_epsilon_constant_packed {
    ($name:ident, $layout:ty, $channels:expr, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, f32, $layout>,
            constants: [f32; $channels],
            destination: &mut MaskViewMut<'_>,
            epsilon: f32,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    constants.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    epsilon,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

impl_compare_equal_epsilon_constant_c1!(
    compare_equal_epsilon_constant_f32_c1,
    nppiCompareEqualEpsC_32f_C1R_Ctx
);
impl_compare_equal_epsilon_constant_packed!(
    compare_equal_epsilon_constant_f32_c3,
    C3,
    3,
    nppiCompareEqualEpsC_32f_C3R_Ctx
);
impl_compare_equal_epsilon_constant_packed!(
    compare_equal_epsilon_constant_f32_c4,
    C4,
    4,
    nppiCompareEqualEpsC_32f_C4R_Ctx
);
impl_compare_equal_epsilon_constant_packed!(
    compare_equal_epsilon_constant_f32_ac4,
    AC4,
    3,
    nppiCompareEqualEpsC_32f_AC4R_Ctx
);

impl_generic_compare_equal_epsilon_constant_scalar!(
    CompareEqualEpsilonConstantC1,
    compare_equal_epsilon_constant,
    compare_equal_epsilon_constant_c1,
    C1,
    [f32 => compare_equal_epsilon_constant_f32_c1]
);
impl_generic_compare_equal_epsilon_constant_array!(
    CompareEqualEpsilonConstantC3,
    compare_equal_epsilon_constant,
    compare_equal_epsilon_constant_c3,
    C3,
    3,
    [f32 => compare_equal_epsilon_constant_f32_c3]
);
impl_generic_compare_equal_epsilon_constant_array!(
    CompareEqualEpsilonConstantC4,
    compare_equal_epsilon_constant,
    compare_equal_epsilon_constant_c4,
    C4,
    4,
    [f32 => compare_equal_epsilon_constant_f32_c4]
);
impl_generic_compare_equal_epsilon_constant_array!(
    CompareEqualEpsilonConstantAc4,
    compare_equal_epsilon_constant,
    compare_equal_epsilon_constant_ac4,
    AC4,
    3,
    [f32 => compare_equal_epsilon_constant_f32_ac4]
);

fn validate_same_size(source: Size, destination: Size) -> Result<()> {
    if source == destination {
        return Ok(());
    }
    Err(Error::SizeMismatch {
        name: "image size".into(),
        expected: source,
        actual: destination,
    })
}

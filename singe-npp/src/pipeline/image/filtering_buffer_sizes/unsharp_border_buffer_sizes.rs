use crate::{
    error::Result,
    image::{
        filtering,
        view::{AC4, C1, C3, C4},
    },
    pipeline::image::filtering_buffer_sizes::ImagePipeline,
};

macro_rules! impl_filter_unsharp_border_buffer_size {
    ($ty:ty, $layout:ty, $buffer_size:path) => {
        impl<'a> ImagePipeline<'a, $ty, $layout> {
            /// Returns the scratch capacity in bytes for unsharp-mask sharpening with `radius` and `sigma`.
            ///
            /// `radius` is the Gaussian radius in pixels excluding the center pixel.
            /// `sigma` is the Gaussian standard deviation in pixels.
            ///
            /// # Errors
            ///
            /// Returns an error if `radius` or `sigma` does not compare greater than zero, NPP rejects the query, or the returned size cannot be represented as `usize`.
            #[doc = concat!("Uses [`", stringify!($buffer_size), "`].")]
            pub fn filter_unsharp_border_buffer_size(radius: f32, sigma: f32) -> Result<usize> {
                $buffer_size(radius, sigma)
            }
        }
    };
}

impl_filter_unsharp_border_buffer_size!(u8, C1, filtering::filter_unsharp_border_u8_c1_buffer_size);

impl_filter_unsharp_border_buffer_size!(u8, C3, filtering::filter_unsharp_border_u8_c3_buffer_size);

impl_filter_unsharp_border_buffer_size!(u8, C4, filtering::filter_unsharp_border_u8_c4_buffer_size);

impl_filter_unsharp_border_buffer_size!(
    u8,
    AC4,
    filtering::filter_unsharp_border_u8_ac4_buffer_size
);

impl_filter_unsharp_border_buffer_size!(
    u16,
    C1,
    filtering::filter_unsharp_border_u16_c1_buffer_size
);

impl_filter_unsharp_border_buffer_size!(
    u16,
    C3,
    filtering::filter_unsharp_border_u16_c3_buffer_size
);

impl_filter_unsharp_border_buffer_size!(
    u16,
    C4,
    filtering::filter_unsharp_border_u16_c4_buffer_size
);

impl_filter_unsharp_border_buffer_size!(
    u16,
    AC4,
    filtering::filter_unsharp_border_u16_ac4_buffer_size
);

impl_filter_unsharp_border_buffer_size!(
    i16,
    C1,
    filtering::filter_unsharp_border_i16_c1_buffer_size
);

impl_filter_unsharp_border_buffer_size!(
    i16,
    C3,
    filtering::filter_unsharp_border_i16_c3_buffer_size
);

impl_filter_unsharp_border_buffer_size!(
    i16,
    C4,
    filtering::filter_unsharp_border_i16_c4_buffer_size
);

impl_filter_unsharp_border_buffer_size!(
    i16,
    AC4,
    filtering::filter_unsharp_border_i16_ac4_buffer_size
);

impl_filter_unsharp_border_buffer_size!(
    f32,
    C1,
    filtering::filter_unsharp_border_f32_c1_buffer_size
);

impl_filter_unsharp_border_buffer_size!(
    f32,
    C3,
    filtering::filter_unsharp_border_f32_c3_buffer_size
);

impl_filter_unsharp_border_buffer_size!(
    f32,
    C4,
    filtering::filter_unsharp_border_f32_c4_buffer_size
);

impl_filter_unsharp_border_buffer_size!(
    f32,
    AC4,
    filtering::filter_unsharp_border_f32_ac4_buffer_size
);

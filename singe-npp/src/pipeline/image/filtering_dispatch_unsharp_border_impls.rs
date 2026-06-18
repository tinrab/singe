use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
    types::{BorderType, Point},
};

use super::super::{ImagePipeline, filtering_traits::*};

impl_unsharp_border_filter_image!(u8, C1, filtering::filter_unsharp_border_u8_c1);
impl_unsharp_border_filter_image!(u8, C3, filtering::filter_unsharp_border_u8_c3);
impl_unsharp_border_filter_image!(u8, C4, filtering::filter_unsharp_border_u8_c4);
impl_unsharp_border_filter_image!(u8, AC4, filtering::filter_unsharp_border_u8_ac4);
impl_unsharp_border_filter_image!(u16, C1, filtering::filter_unsharp_border_u16_c1);
impl_unsharp_border_filter_image!(u16, C3, filtering::filter_unsharp_border_u16_c3);
impl_unsharp_border_filter_image!(u16, C4, filtering::filter_unsharp_border_u16_c4);
impl_unsharp_border_filter_image!(u16, AC4, filtering::filter_unsharp_border_u16_ac4);
impl_unsharp_border_filter_image!(i16, C1, filtering::filter_unsharp_border_i16_c1);
impl_unsharp_border_filter_image!(i16, C4, filtering::filter_unsharp_border_i16_c4);
impl_unsharp_border_filter_image!(i16, AC4, filtering::filter_unsharp_border_i16_ac4);
impl_unsharp_border_filter_image!(f32, C1, filtering::filter_unsharp_border_f32_c1);
impl_unsharp_border_filter_image!(f32, C3, filtering::filter_unsharp_border_f32_c3);
impl_unsharp_border_filter_image!(f32, C4, filtering::filter_unsharp_border_f32_c4);
impl_unsharp_border_filter_image!(f32, AC4, filtering::filter_unsharp_border_f32_ac4);

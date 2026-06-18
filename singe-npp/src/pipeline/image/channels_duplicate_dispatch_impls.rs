use crate::{
    context::StreamContext,
    error::Result,
    image::{
        exchange,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
};

use super::super::ImagePipeline;
use super::{DuplicateToAC4Image, DuplicateToC3Image, DuplicateToC4Image};

impl_duplicate_to_c3_image!(u8, exchange::duplicate_channel_c1_to_c3);
impl_duplicate_to_c4_image!(u8, exchange::duplicate_channel_c1_to_c4);
impl_duplicate_to_ac4_image!(u8, exchange::duplicate_channel_c1_to_ac4);
impl_duplicate_to_c3_image!(u16, exchange::duplicate_channel_c1_to_c3);
impl_duplicate_to_c4_image!(u16, exchange::duplicate_channel_c1_to_c4);
impl_duplicate_to_ac4_image!(u16, exchange::duplicate_channel_c1_to_ac4);
impl_duplicate_to_c3_image!(i16, exchange::duplicate_channel_c1_to_c3);
impl_duplicate_to_c4_image!(i16, exchange::duplicate_channel_c1_to_c4);
impl_duplicate_to_ac4_image!(i16, exchange::duplicate_channel_c1_to_ac4);
impl_duplicate_to_c3_image!(i32, exchange::duplicate_channel_c1_to_c3);
impl_duplicate_to_c4_image!(i32, exchange::duplicate_channel_c1_to_c4);
impl_duplicate_to_ac4_image!(i32, exchange::duplicate_channel_c1_to_ac4);
impl_duplicate_to_c3_image!(f32, exchange::duplicate_channel_c1_to_c3);
impl_duplicate_to_c4_image!(f32, exchange::duplicate_channel_c1_to_c4);
impl_duplicate_to_ac4_image!(f32, exchange::duplicate_channel_c1_to_ac4);

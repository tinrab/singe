use singe_cuda::types::{Complex32, f16};

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        exchange,
        view::{AC4, C1, C2, C3, C4, ImageViewMut},
    },
    types::{ComplexI16, ComplexI32},
};

use super::super::ImagePipeline;
use super::SetImage;

impl_set_image!(u8, C1, u8, exchange::set_c1);
impl_set_image!(u8, C2, [u8; 2], exchange::set_c2);
impl_set_image!(u8, C3, [u8; 3], exchange::set_c3);
impl_set_image!(u8, C4, [u8; 4], exchange::set_c4);
impl_set_image!(u8, AC4, [u8; 3], exchange::set_ac4);
impl_set_image!(i8, C1, i8, exchange::set_c1);
impl_set_image!(i8, C2, [i8; 2], exchange::set_c2);
impl_set_image!(i8, C3, [i8; 3], exchange::set_c3);
impl_set_image!(i8, C4, [i8; 4], exchange::set_c4);
impl_set_image!(i8, AC4, [i8; 3], exchange::set_ac4);
impl_set_image!(u16, C1, u16, exchange::set_c1);
impl_set_image!(u16, C2, [u16; 2], exchange::set_c2);
impl_set_image!(u16, C3, [u16; 3], exchange::set_c3);
impl_set_image!(u16, C4, [u16; 4], exchange::set_c4);
impl_set_image!(u16, AC4, [u16; 3], exchange::set_ac4);
impl_set_image!(f16, C1, f32, exchange::set_c1);
impl_set_image!(f16, C2, [f32; 2], exchange::set_c2);
impl_set_image!(f16, C3, [f32; 3], exchange::set_c3);
impl_set_image!(f16, C4, [f32; 4], exchange::set_c4);
impl_set_image!(i16, C1, i16, exchange::set_c1);
impl_set_image!(i16, C2, [i16; 2], exchange::set_c2);
impl_set_image!(i16, C3, [i16; 3], exchange::set_c3);
impl_set_image!(i16, C4, [i16; 4], exchange::set_c4);
impl_set_image!(i16, AC4, [i16; 3], exchange::set_ac4);
impl_set_image!(i32, C1, i32, exchange::set_c1);
impl_set_image!(i32, C2, [i32; 2], exchange::set_c2);
impl_set_image!(i32, C3, [i32; 3], exchange::set_c3);
impl_set_image!(i32, C4, [i32; 4], exchange::set_c4);
impl_set_image!(i32, AC4, [i32; 3], exchange::set_ac4);
impl_set_image!(u32, C1, u32, exchange::set_c1);
impl_set_image!(u32, C2, [u32; 2], exchange::set_c2);
impl_set_image!(u32, C3, [u32; 3], exchange::set_c3);
impl_set_image!(u32, C4, [u32; 4], exchange::set_c4);
impl_set_image!(u32, AC4, [u32; 3], exchange::set_ac4);
impl_set_image!(f32, C1, f32, exchange::set_c1);
impl_set_image!(f32, C2, [f32; 2], exchange::set_c2);
impl_set_image!(f32, C3, [f32; 3], exchange::set_c3);
impl_set_image!(f32, C4, [f32; 4], exchange::set_c4);
impl_set_image!(f32, AC4, [f32; 3], exchange::set_ac4);
impl_set_image!(ComplexI16, C1, ComplexI16, exchange::set_c1);
impl_set_image!(ComplexI16, C2, [ComplexI16; 2], exchange::set_c2);
impl_set_image!(ComplexI16, C3, [ComplexI16; 3], exchange::set_c3);
impl_set_image!(ComplexI16, C4, [ComplexI16; 4], exchange::set_c4);
impl_set_image!(ComplexI16, AC4, [ComplexI16; 3], exchange::set_ac4);
impl_set_image!(ComplexI32, C1, ComplexI32, exchange::set_c1);
impl_set_image!(ComplexI32, C2, [ComplexI32; 2], exchange::set_c2);
impl_set_image!(ComplexI32, C3, [ComplexI32; 3], exchange::set_c3);
impl_set_image!(ComplexI32, C4, [ComplexI32; 4], exchange::set_c4);
impl_set_image!(ComplexI32, AC4, [ComplexI32; 3], exchange::set_ac4);
impl_set_image!(Complex32, C1, Complex32, exchange::set_c1);
impl_set_image!(Complex32, C2, [Complex32; 2], exchange::set_c2);
impl_set_image!(Complex32, C3, [Complex32; 3], exchange::set_c3);
impl_set_image!(Complex32, C4, [Complex32; 4], exchange::set_c4);
impl_set_image!(Complex32, AC4, [Complex32; 3], exchange::set_ac4);

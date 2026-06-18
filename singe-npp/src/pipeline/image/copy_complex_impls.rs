use singe_cuda::types::Complex32;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        exchange,
        view::{AC4, C1, C2, C3, C4, ImageView, ImageViewMut},
    },
    pipeline::ImagePipeline,
    types::{ComplexI16, ComplexI32},
};

use super::CopyImage;

impl_copy_image!(ComplexI16, C1, exchange::copy_c1);
impl_copy_image!(ComplexI16, C2, exchange::copy_c2);
impl_copy_image!(ComplexI16, C3, exchange::copy_c3);
impl_copy_image!(ComplexI16, C4, exchange::copy_c4);
impl_copy_image!(ComplexI16, AC4, exchange::copy_ac4);
impl_copy_image!(ComplexI32, C1, exchange::copy_c1);
impl_copy_image!(ComplexI32, C2, exchange::copy_c2);
impl_copy_image!(ComplexI32, C3, exchange::copy_c3);
impl_copy_image!(ComplexI32, C4, exchange::copy_c4);
impl_copy_image!(ComplexI32, AC4, exchange::copy_ac4);
impl_copy_image!(Complex32, C1, exchange::copy_c1);
impl_copy_image!(Complex32, C2, exchange::copy_c2);
impl_copy_image!(Complex32, C3, exchange::copy_c3);
impl_copy_image!(Complex32, C4, exchange::copy_c4);
impl_copy_image!(Complex32, AC4, exchange::copy_ac4);

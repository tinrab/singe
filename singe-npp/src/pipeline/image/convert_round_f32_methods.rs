use crate::{error::Result, image::view::C1, types::RoundMode};

use super::super::super::ImagePipeline;

impl<'a> ImagePipeline<'a, f32, C1> {
    pub fn convert_scaled_to_u8(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<ImagePipeline<'a, u8, C1>> {
        self.scaled_round_convert_output::<u8>(round_mode, scale_factor)
    }

    pub fn convert_scaled_to_i8(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<ImagePipeline<'a, i8, C1>> {
        self.scaled_round_convert_output::<i8>(round_mode, scale_factor)
    }

    pub fn convert_scaled_to_u16(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<ImagePipeline<'a, u16, C1>> {
        self.scaled_round_convert_output::<u16>(round_mode, scale_factor)
    }

    pub fn convert_scaled_to_i16(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<ImagePipeline<'a, i16, C1>> {
        self.scaled_round_convert_output::<i16>(round_mode, scale_factor)
    }

    pub fn convert_scaled_to_u32(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<ImagePipeline<'a, u32, C1>> {
        self.scaled_round_convert_output::<u32>(round_mode, scale_factor)
    }

    pub fn convert_scaled_to_i32(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<ImagePipeline<'a, i32, C1>> {
        self.scaled_round_convert_output::<i32>(round_mode, scale_factor)
    }
}

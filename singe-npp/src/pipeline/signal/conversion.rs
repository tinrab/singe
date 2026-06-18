use crate::{
    context::StreamContext,
    error::Result,
    pipeline::{SignalAllocator, Workspace},
    signal::{
        conversion,
        view::{SignalView, SignalViewMut},
    },
    types::RoundMode,
};

use super::{SignalBacking, SignalPipeline};

pub trait ConvertSignal<T, U> {
    fn convert_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, U>,
    ) -> Result<()>;
}

pub trait ScaledConvertSignal<T, U> {
    fn convert_signal_scaled(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, U>,
        scale_factor: i32,
    ) -> Result<()>;
}

pub trait ScaledRoundConvertSignal<T, U> {
    fn convert_signal_scaled_round(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, U>,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<()>;
}

macro_rules! impl_convert_signal {
    ($source:ty, $destination:ty, $convert:path) => {
        impl<'a> ConvertSignal<$source, $destination> for SignalPipeline<'a, $source> {
            fn convert_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $source>,
                destination: &mut SignalViewMut<'_, $destination>,
            ) -> Result<()> {
                $convert(stream_context, source, destination)
            }
        }
    };
}

macro_rules! impl_scaled_convert_signal {
    ($source:ty, $destination:ty, $convert:path) => {
        impl<'a> ScaledConvertSignal<$source, $destination> for SignalPipeline<'a, $source> {
            fn convert_signal_scaled(
                stream_context: &StreamContext,
                source: &SignalView<'_, $source>,
                destination: &mut SignalViewMut<'_, $destination>,
                scale_factor: i32,
            ) -> Result<()> {
                $convert(stream_context, source, destination, scale_factor)
            }
        }
    };
}

macro_rules! impl_scaled_round_convert_signal {
    ($source:ty, $destination:ty, $convert:path) => {
        impl<'a> ScaledRoundConvertSignal<$source, $destination> for SignalPipeline<'a, $source> {
            fn convert_signal_scaled_round(
                stream_context: &StreamContext,
                source: &SignalView<'_, $source>,
                destination: &mut SignalViewMut<'_, $destination>,
                round_mode: RoundMode,
                scale_factor: i32,
            ) -> Result<()> {
                $convert(
                    stream_context,
                    source,
                    destination,
                    round_mode,
                    scale_factor,
                )
            }
        }
    };
}

impl_convert_signal!(i8, f32, conversion::convert_i8_to_f32);
impl_convert_signal!(i8, i16, conversion::convert_i8_to_i16);
impl_convert_signal!(u8, f32, conversion::convert_u8_to_f32);
impl_convert_signal!(i16, f32, conversion::convert_i16_to_f32);
impl_convert_signal!(i16, i32, conversion::convert_i16_to_i32);
impl_convert_signal!(u16, f32, conversion::convert_u16_to_f32);
impl_convert_signal!(i32, i16, conversion::convert_i32_to_i16);
impl_convert_signal!(i32, f32, conversion::convert_i32_to_f32);
impl_convert_signal!(i32, f64, conversion::convert_i32_to_f64);
impl_convert_signal!(f32, f64, conversion::convert_f32_to_f64);
impl_convert_signal!(i64, f64, conversion::convert_i64_to_f64);
impl_convert_signal!(f64, f32, conversion::convert_f64_to_f32);

impl_scaled_convert_signal!(i16, f32, conversion::convert_i16_to_f32_scaled);
impl_scaled_convert_signal!(i16, f64, conversion::convert_i16_to_f64_scaled);
impl_scaled_convert_signal!(i32, i16, conversion::convert_i32_to_i16_scaled);
impl_scaled_convert_signal!(i32, f32, conversion::convert_i32_to_f32_scaled);
impl_scaled_convert_signal!(i32, f64, conversion::convert_i32_to_f64_scaled);

impl_scaled_round_convert_signal!(i16, i8, conversion::convert_i16_to_i8_scaled);
impl_scaled_round_convert_signal!(f32, i8, conversion::convert_f32_to_i8_scaled);
impl_scaled_round_convert_signal!(f32, u8, conversion::convert_f32_to_u8_scaled);
impl_scaled_round_convert_signal!(f32, i16, conversion::convert_f32_to_i16_scaled);
impl_scaled_round_convert_signal!(f32, u16, conversion::convert_f32_to_u16_scaled);
impl_scaled_round_convert_signal!(f32, i32, conversion::convert_f32_to_i32_scaled);
impl_scaled_round_convert_signal!(i64, i32, conversion::convert_i64_to_i32_scaled);
impl_scaled_round_convert_signal!(f64, i16, conversion::convert_f64_to_i16_scaled);
impl_scaled_round_convert_signal!(f64, i32, conversion::convert_f64_to_i32_scaled);
impl_scaled_round_convert_signal!(f64, i64, conversion::convert_f64_to_i64_scaled);

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
{
    pub fn convert_to<U>(self) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy,
        Workspace: SignalAllocator<U>,
        Self: ConvertSignal<T, U>,
    {
        let mut destination = self.workspace.signal::<U>(self.len())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as ConvertSignal<T, U>>::convert_signal(
                self.stream_context,
                &source,
                &mut destination_view,
            )?;
        }

        Ok(SignalPipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: SignalBacking::Owned(destination),
        })
    }

    pub fn convert_to_scaled<U>(self, scale_factor: i32) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy,
        Workspace: SignalAllocator<U>,
        Self: ScaledConvertSignal<T, U>,
    {
        let mut destination = self.workspace.signal::<U>(self.len())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as ScaledConvertSignal<T, U>>::convert_signal_scaled(
                self.stream_context,
                &source,
                &mut destination_view,
                scale_factor,
            )?;
        }

        Ok(SignalPipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: SignalBacking::Owned(destination),
        })
    }

    pub fn convert_to_scaled_round<U>(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy,
        Workspace: SignalAllocator<U>,
        Self: ScaledRoundConvertSignal<T, U>,
    {
        let mut destination = self.workspace.signal::<U>(self.len())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as ScaledRoundConvertSignal<T, U>>::convert_signal_scaled_round(
                self.stream_context,
                &source,
                &mut destination_view,
                round_mode,
                scale_factor,
            )?;
        }

        Ok(SignalPipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: SignalBacking::Owned(destination),
        })
    }

    pub fn convert_into<U>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, U>,
    ) -> Result<()>
    where
        U: Copy,
        Self: ConvertSignal<T, U>,
    {
        <Self as ConvertSignal<T, U>>::convert_signal(stream_context, source, destination)
    }

    pub fn convert_scaled_into<U>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, U>,
        scale_factor: i32,
    ) -> Result<()>
    where
        U: Copy,
        Self: ScaledConvertSignal<T, U>,
    {
        <Self as ScaledConvertSignal<T, U>>::convert_signal_scaled(
            stream_context,
            source,
            destination,
            scale_factor,
        )
    }

    pub fn convert_scaled_round_into<U>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, U>,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<()>
    where
        U: Copy,
        Self: ScaledRoundConvertSignal<T, U>,
    {
        <Self as ScaledRoundConvertSignal<T, U>>::convert_signal_scaled_round(
            stream_context,
            source,
            destination,
            round_mode,
            scale_factor,
        )
    }
}

impl<'a> SignalPipeline<'a, u8> {
    pub fn convert_to_f32(self) -> Result<SignalPipeline<'a, f32>> {
        self.convert_to()
    }
}

impl<'a> SignalPipeline<'a, i8> {
    pub fn convert_to_f32(self) -> Result<SignalPipeline<'a, f32>> {
        self.convert_to()
    }

    pub fn convert_to_i16(self) -> Result<SignalPipeline<'a, i16>> {
        self.convert_to()
    }
}

impl<'a> SignalPipeline<'a, u16> {
    pub fn convert_to_f32(self) -> Result<SignalPipeline<'a, f32>> {
        self.convert_to()
    }
}

impl<'a> SignalPipeline<'a, i16> {
    pub fn convert_to_f32(self) -> Result<SignalPipeline<'a, f32>> {
        self.convert_to()
    }

    pub fn convert_to_i32(self) -> Result<SignalPipeline<'a, i32>> {
        self.convert_to()
    }

    pub fn convert_to_f32_scaled(self, scale_factor: i32) -> Result<SignalPipeline<'a, f32>> {
        self.convert_to_scaled(scale_factor)
    }

    pub fn convert_to_f64_scaled(self, scale_factor: i32) -> Result<SignalPipeline<'a, f64>> {
        self.convert_to_scaled(scale_factor)
    }

    pub fn convert_to_i8_scaled(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i8>> {
        self.convert_to_scaled_round(round_mode, scale_factor)
    }
}

impl<'a> SignalPipeline<'a, i32> {
    pub fn convert_to_i16(self) -> Result<SignalPipeline<'a, i16>> {
        self.convert_to()
    }

    pub fn convert_to_f32(self) -> Result<SignalPipeline<'a, f32>> {
        self.convert_to()
    }

    pub fn convert_to_f64(self) -> Result<SignalPipeline<'a, f64>> {
        self.convert_to()
    }

    pub fn convert_to_i16_scaled(self, scale_factor: i32) -> Result<SignalPipeline<'a, i16>> {
        self.convert_to_scaled(scale_factor)
    }

    pub fn convert_to_f32_scaled(self, scale_factor: i32) -> Result<SignalPipeline<'a, f32>> {
        self.convert_to_scaled(scale_factor)
    }

    pub fn convert_to_f64_scaled(self, scale_factor: i32) -> Result<SignalPipeline<'a, f64>> {
        self.convert_to_scaled(scale_factor)
    }
}

impl<'a> SignalPipeline<'a, f32> {
    pub fn convert_to_f64(self) -> Result<SignalPipeline<'a, f64>> {
        self.convert_to()
    }

    pub fn convert_to_u8_scaled(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, u8>> {
        self.convert_to_scaled_round(round_mode, scale_factor)
    }

    pub fn convert_to_i8_scaled(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i8>> {
        self.convert_to_scaled_round(round_mode, scale_factor)
    }

    pub fn convert_to_i16_scaled(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i16>> {
        self.convert_to_scaled_round(round_mode, scale_factor)
    }

    pub fn convert_to_u16_scaled(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, u16>> {
        self.convert_to_scaled_round(round_mode, scale_factor)
    }

    pub fn convert_to_i32_scaled(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i32>> {
        self.convert_to_scaled_round(round_mode, scale_factor)
    }
}

impl<'a> SignalPipeline<'a, i64> {
    pub fn convert_to_f64(self) -> Result<SignalPipeline<'a, f64>> {
        self.convert_to()
    }

    pub fn convert_to_i32_scaled(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i32>> {
        self.convert_to_scaled_round(round_mode, scale_factor)
    }
}

impl<'a> SignalPipeline<'a, f64> {
    pub fn convert_to_f32(self) -> Result<SignalPipeline<'a, f32>> {
        self.convert_to()
    }

    pub fn convert_to_i16_scaled(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i16>> {
        self.convert_to_scaled_round(round_mode, scale_factor)
    }

    pub fn convert_to_i32_scaled(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i32>> {
        self.convert_to_scaled_round(round_mode, scale_factor)
    }

    pub fn convert_to_i64_scaled(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i64>> {
        self.convert_to_scaled_round(round_mode, scale_factor)
    }
}

use singe_cuda::types::{Complex32, Complex64};

use crate::{
    context::StreamContext,
    error::Result,
    pipeline::{SignalAllocator, Workspace},
    signal::{
        arithmetic, initialization,
        view::{SignalView, SignalViewMut},
    },
    types::{ComplexI16, DataTypeLike},
};

use super::{SignalBacking, SignalPipeline};

pub struct Absolute;
pub struct Arctangent;
pub struct BitNot;
pub struct Cauchy;
pub struct CauchyDerivative;
pub struct CubeRoot;
pub struct CubeRootToI16Scaled;
pub struct ExponentScaled;
pub struct ExponentToF64;
pub struct NaturalLogarithmToF32;
pub struct NaturalLogarithmToI16Scaled;
pub struct NaturalLogarithmScaled;
pub struct SquareScaled;
pub struct Square;
pub struct SquareRootToI16Scaled;
pub struct SquareRootScaled;
pub struct SquareRoot;
pub struct TenTimesLog10Scaled;

pub trait UnaryTransformSignal<T, O> {
    fn unary_transform_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    fn unary_transform_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;
}

pub trait ScaledUnaryTransformSignal<T, O> {
    fn scaled_unary_transform_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()>;

    fn scaled_unary_transform_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()>;
}

pub trait UnaryOutputSignal<T, U, O> {
    fn unary_output_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, U>,
    ) -> Result<()>;
}

pub trait ScaledUnaryOutputSignal<T, U, O> {
    fn scaled_unary_output_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, U>,
        scale_factor: i32,
    ) -> Result<()>;
}

pub trait ParameterizedUnaryTransformSignal<T, P, O> {
    fn parameterized_unary_transform_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        parameter: P,
    ) -> Result<()>;
}

macro_rules! impl_unary_transform_signal {
    ($operation_ty:ty, $ty:ty, $operation:path, $operation_in_place:path) => {
        impl<'a> UnaryTransformSignal<$ty, $operation_ty> for SignalPipeline<'a, $ty> {
            fn unary_transform_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $operation(stream_context, source, destination)
            }

            fn unary_transform_signal_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $operation_in_place(stream_context, signal)
            }
        }
    };
}

macro_rules! impl_scaled_unary_transform_signal {
    ($operation_ty:ty, $ty:ty, $operation:path, $operation_in_place:path) => {
        impl<'a> ScaledUnaryTransformSignal<$ty, $operation_ty> for SignalPipeline<'a, $ty> {
            fn scaled_unary_transform_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                scale_factor: i32,
            ) -> Result<()> {
                $operation(stream_context, source, destination, scale_factor)
            }

            fn scaled_unary_transform_signal_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
                scale_factor: i32,
            ) -> Result<()> {
                $operation_in_place(stream_context, signal, scale_factor)
            }
        }
    };
}

macro_rules! impl_unary_output_signal {
    ($operation_ty:ty, $source:ty, $destination:ty, $operation:path) => {
        impl<'a> UnaryOutputSignal<$source, $destination, $operation_ty>
            for SignalPipeline<'a, $source>
        {
            fn unary_output_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $source>,
                destination: &mut SignalViewMut<'_, $destination>,
            ) -> Result<()> {
                $operation(stream_context, source, destination)
            }
        }
    };
}

macro_rules! impl_scaled_unary_output_signal {
    ($operation_ty:ty, $source:ty, $destination:ty, $operation:path) => {
        impl<'a> ScaledUnaryOutputSignal<$source, $destination, $operation_ty>
            for SignalPipeline<'a, $source>
        {
            fn scaled_unary_output_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $source>,
                destination: &mut SignalViewMut<'_, $destination>,
                scale_factor: i32,
            ) -> Result<()> {
                $operation(stream_context, source, destination, scale_factor)
            }
        }
    };
}

macro_rules! impl_parameterized_unary_transform_signal {
    ($operation_ty:ty, $ty:ty, $parameter_ty:ty, $operation_in_place:path) => {
        impl<'a> ParameterizedUnaryTransformSignal<$ty, $parameter_ty, $operation_ty>
            for SignalPipeline<'a, $ty>
        {
            fn parameterized_unary_transform_signal_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
                parameter: $parameter_ty,
            ) -> Result<()> {
                $operation_in_place(stream_context, signal, parameter)
            }
        }
    };
}

impl_unary_transform_signal!(
    Absolute,
    i16,
    arithmetic::absolute_i16,
    arithmetic::absolute_i16_in_place
);
impl_unary_transform_signal!(
    Absolute,
    i32,
    arithmetic::absolute_i32,
    arithmetic::absolute_i32_in_place
);
impl_unary_transform_signal!(
    Absolute,
    f32,
    arithmetic::absolute_f32,
    arithmetic::absolute_f32_in_place
);
impl_unary_transform_signal!(
    Absolute,
    f64,
    arithmetic::absolute_f64,
    arithmetic::absolute_f64_in_place
);

impl_unary_transform_signal!(
    Arctangent,
    f32,
    arithmetic::arctangent_f32,
    arithmetic::arctangent_f32_in_place
);
impl_unary_transform_signal!(
    Arctangent,
    f64,
    arithmetic::arctangent_f64,
    arithmetic::arctangent_f64_in_place
);

impl_unary_output_signal!(CubeRoot, f32, f32, arithmetic::cube_root_f32);
impl_unary_output_signal!(ExponentToF64, f32, f64, arithmetic::exponent_f32_to_f64);
impl_unary_output_signal!(
    NaturalLogarithmToF32,
    f64,
    f32,
    arithmetic::natural_logarithm_f64_to_f32
);

impl_parameterized_unary_transform_signal!(Cauchy, f32, f32, arithmetic::cauchy_f32_in_place);
impl_parameterized_unary_transform_signal!(
    CauchyDerivative,
    f32,
    f32,
    arithmetic::cauchy_derivative_f32_in_place
);

impl_unary_transform_signal!(BitNot, u8, arithmetic::not_u8, arithmetic::not_u8_in_place);
impl_unary_transform_signal!(
    BitNot,
    u16,
    arithmetic::not_u16,
    arithmetic::not_u16_in_place
);
impl_unary_transform_signal!(
    BitNot,
    u32,
    arithmetic::not_u32,
    arithmetic::not_u32_in_place
);

impl_scaled_unary_transform_signal!(
    ExponentScaled,
    i16,
    arithmetic::exponent_i16_scaled,
    arithmetic::exponent_i16_scaled_in_place
);
impl_scaled_unary_transform_signal!(
    ExponentScaled,
    i32,
    arithmetic::exponent_i32_scaled,
    arithmetic::exponent_i32_scaled_in_place
);
impl_scaled_unary_transform_signal!(
    ExponentScaled,
    i64,
    arithmetic::exponent_i64_scaled,
    arithmetic::exponent_i64_scaled_in_place
);

impl_scaled_unary_transform_signal!(
    NaturalLogarithmScaled,
    i16,
    arithmetic::natural_logarithm_i16_scaled,
    arithmetic::natural_logarithm_i16_scaled_in_place
);
impl_scaled_unary_transform_signal!(
    NaturalLogarithmScaled,
    i32,
    arithmetic::natural_logarithm_i32_scaled,
    arithmetic::natural_logarithm_i32_scaled_in_place
);

impl_scaled_unary_transform_signal!(
    TenTimesLog10Scaled,
    i32,
    arithmetic::ten_times_log10_i32_scaled,
    arithmetic::ten_times_log10_i32_scaled_in_place
);

impl_scaled_unary_output_signal!(
    NaturalLogarithmToI16Scaled,
    i32,
    i16,
    arithmetic::natural_logarithm_i32_to_i16_scaled
);
impl_scaled_unary_output_signal!(
    CubeRootToI16Scaled,
    i32,
    i16,
    arithmetic::cube_root_i32_to_i16_scaled
);
impl_scaled_unary_output_signal!(
    SquareRootToI16Scaled,
    i32,
    i16,
    arithmetic::square_root_i32_to_i16_scaled
);
impl_scaled_unary_output_signal!(
    SquareRootToI16Scaled,
    i64,
    i16,
    arithmetic::square_root_i64_to_i16_scaled
);

impl_scaled_unary_transform_signal!(
    SquareScaled,
    u8,
    arithmetic::square_u8_scaled,
    arithmetic::square_u8_scaled_in_place
);
impl_scaled_unary_transform_signal!(
    SquareScaled,
    u16,
    arithmetic::square_u16_scaled,
    arithmetic::square_u16_scaled_in_place
);
impl_scaled_unary_transform_signal!(
    SquareScaled,
    i16,
    arithmetic::square_i16_scaled,
    arithmetic::square_i16_scaled_in_place
);
impl_scaled_unary_transform_signal!(
    SquareScaled,
    ComplexI16,
    arithmetic::square_i16_complex_scaled,
    arithmetic::square_i16_complex_scaled_in_place
);
impl_unary_transform_signal!(
    Square,
    Complex32,
    arithmetic::square_f32_complex,
    arithmetic::square_f32_complex_in_place
);
impl_unary_transform_signal!(
    Square,
    Complex64,
    arithmetic::square_f64_complex,
    arithmetic::square_f64_complex_in_place
);

impl_scaled_unary_transform_signal!(
    SquareRootScaled,
    u8,
    arithmetic::square_root_u8_scaled,
    arithmetic::square_root_u8_scaled_in_place
);
impl_scaled_unary_transform_signal!(
    SquareRootScaled,
    u16,
    arithmetic::square_root_u16_scaled,
    arithmetic::square_root_u16_scaled_in_place
);
impl_scaled_unary_transform_signal!(
    SquareRootScaled,
    i16,
    arithmetic::square_root_i16_scaled,
    arithmetic::square_root_i16_scaled_in_place
);
impl_scaled_unary_transform_signal!(
    SquareRootScaled,
    ComplexI16,
    arithmetic::square_root_i16_complex_scaled,
    arithmetic::square_root_i16_complex_scaled_in_place
);
impl_unary_transform_signal!(
    SquareRoot,
    Complex32,
    arithmetic::square_root_f32_complex,
    arithmetic::square_root_f32_complex_in_place
);
impl_unary_transform_signal!(
    SquareRoot,
    Complex64,
    arithmetic::square_root_f64_complex,
    arithmetic::square_root_f64_complex_in_place
);
impl_scaled_unary_transform_signal!(
    SquareRootScaled,
    i64,
    arithmetic::square_root_i64_scaled,
    arithmetic::square_root_i64_scaled_in_place
);

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: UnaryTransformSignal<T, Absolute>,
{
    pub fn absolute(self) -> Result<Self> {
        self.unary_transform::<Absolute>()
    }
}

impl<'a> SignalPipeline<'a, f32> {
    pub fn cube_root(self) -> Result<Self> {
        self.unary_output::<f32, CubeRoot>()
    }

    pub fn cauchy(self, parameter: f32) -> Result<Self> {
        self.parameterized_unary_transform::<Cauchy>(parameter)
    }

    pub fn cauchy_derivative(self, parameter: f32) -> Result<Self> {
        self.parameterized_unary_transform::<CauchyDerivative>(parameter)
    }

    pub fn exponent_to_f64(self) -> Result<SignalPipeline<'a, f64>> {
        self.unary_output::<f64, ExponentToF64>()
    }
}

impl<'a> SignalPipeline<'a, f64> {
    pub fn natural_logarithm_to_f32(self) -> Result<SignalPipeline<'a, f32>> {
        self.unary_output::<f32, NaturalLogarithmToF32>()
    }
}

impl<'a> SignalPipeline<'a, i32> {
    pub fn natural_logarithm_to_i16_scaled(
        self,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i16>> {
        self.scaled_unary_output::<i16, NaturalLogarithmToI16Scaled>(scale_factor)
    }

    pub fn cube_root_to_i16_scaled(self, scale_factor: i32) -> Result<SignalPipeline<'a, i16>> {
        self.scaled_unary_output::<i16, CubeRootToI16Scaled>(scale_factor)
    }

    pub fn square_root_to_i16_scaled(self, scale_factor: i32) -> Result<SignalPipeline<'a, i16>> {
        self.scaled_unary_output::<i16, SquareRootToI16Scaled>(scale_factor)
    }
}

impl<'a> SignalPipeline<'a, i64> {
    pub fn square_root_to_i16_scaled(self, scale_factor: i32) -> Result<SignalPipeline<'a, i16>> {
        self.scaled_unary_output::<i16, SquareRootToI16Scaled>(scale_factor)
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: ScaledUnaryTransformSignal<T, ExponentScaled>,
{
    pub fn exponent_scaled(self, scale_factor: i32) -> Result<Self> {
        self.scaled_unary_transform::<ExponentScaled>(scale_factor)
    }
}

impl<'a> SignalPipeline<'a, f32> {
    pub fn parameterized_unary_transform_in_place<O>(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, f32>,
        parameter: f32,
    ) -> Result<()>
    where
        Self: ParameterizedUnaryTransformSignal<f32, f32, O>,
    {
        <Self as ParameterizedUnaryTransformSignal<
            f32,
            f32,
            O,
        >>::parameterized_unary_transform_signal_in_place(stream_context, signal, parameter)
    }

    fn parameterized_unary_transform<O>(mut self, parameter: f32) -> Result<Self>
    where
        Self: ParameterizedUnaryTransformSignal<f32, f32, O>,
    {
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut()?;
                <Self as ParameterizedUnaryTransformSignal<
                    f32,
                    f32,
                    O,
                >>::parameterized_unary_transform_signal_in_place(
                    self.stream_context,
                    &mut signal_view,
                    parameter,
                )?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self.workspace.signal::<f32>(source.len())?;
                let mut destination_view = destination.view_mut()?;
                initialization::copy_f32(self.stream_context, source, &mut destination_view)?;
                <Self as ParameterizedUnaryTransformSignal<
                    f32,
                    f32,
                    O,
                >>::parameterized_unary_transform_signal_in_place(
                    self.stream_context,
                    &mut destination_view,
                    parameter,
                )?;
                self.backing = SignalBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: ScaledUnaryTransformSignal<T, NaturalLogarithmScaled>,
{
    pub fn natural_logarithm_scaled(self, scale_factor: i32) -> Result<Self> {
        self.scaled_unary_transform::<NaturalLogarithmScaled>(scale_factor)
    }
}

impl<'a> SignalPipeline<'a, i32> {
    pub fn ten_times_log10_scaled(self, scale_factor: i32) -> Result<Self> {
        self.scaled_unary_transform::<TenTimesLog10Scaled>(scale_factor)
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: ScaledUnaryTransformSignal<T, SquareScaled>,
{
    pub fn square_scaled(self, scale_factor: i32) -> Result<Self> {
        self.scaled_unary_transform::<SquareScaled>(scale_factor)
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: ScaledUnaryTransformSignal<T, SquareRootScaled>,
{
    pub fn square_root_scaled(self, scale_factor: i32) -> Result<Self> {
        self.scaled_unary_transform::<SquareRootScaled>(scale_factor)
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: UnaryTransformSignal<T, Arctangent>,
{
    pub fn arctangent(self) -> Result<Self> {
        self.unary_transform::<Arctangent>()
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: UnaryTransformSignal<T, BitNot>,
{
    pub fn bit_not(self) -> Result<Self> {
        self.unary_transform::<BitNot>()
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
{
    pub fn unary_transform_into<O>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>
    where
        Self: UnaryTransformSignal<T, O>,
    {
        <Self as UnaryTransformSignal<T, O>>::unary_transform_signal(
            stream_context,
            source,
            destination,
        )
    }

    pub fn unary_transform_in_place<O>(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
    ) -> Result<()>
    where
        Self: UnaryTransformSignal<T, O>,
    {
        <Self as UnaryTransformSignal<T, O>>::unary_transform_signal_in_place(
            stream_context,
            signal,
        )
    }

    pub fn unary_output_into<U, O>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, U>,
    ) -> Result<()>
    where
        U: Copy,
        Self: UnaryOutputSignal<T, U, O>,
    {
        <Self as UnaryOutputSignal<T, U, O>>::unary_output_signal(
            stream_context,
            source,
            destination,
        )
    }

    pub fn scaled_unary_output_into<U, O>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, U>,
        scale_factor: i32,
    ) -> Result<()>
    where
        U: Copy,
        Self: ScaledUnaryOutputSignal<T, U, O>,
    {
        <Self as ScaledUnaryOutputSignal<T, U, O>>::scaled_unary_output_signal(
            stream_context,
            source,
            destination,
            scale_factor,
        )
    }

    pub fn scaled_unary_transform_into<O>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()>
    where
        Self: ScaledUnaryTransformSignal<T, O>,
    {
        <Self as ScaledUnaryTransformSignal<T, O>>::scaled_unary_transform_signal(
            stream_context,
            source,
            destination,
            scale_factor,
        )
    }

    pub fn scaled_unary_transform_in_place<O>(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()>
    where
        Self: ScaledUnaryTransformSignal<T, O>,
    {
        <Self as ScaledUnaryTransformSignal<T, O>>::scaled_unary_transform_signal_in_place(
            stream_context,
            signal,
            scale_factor,
        )
    }

    pub fn exponent_to<U>(self) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        Workspace: SignalAllocator<U>,
        T: arithmetic::ExponentTo<U>,
    {
        let mut destination = self.workspace.signal::<U>(self.len())?;
        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            arithmetic::exponent_to(self.stream_context, &source, &mut destination_view)?;
        }

        Ok(SignalPipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: SignalBacking::Owned(destination),
        })
    }

    pub fn natural_logarithm_to<U>(self) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        Workspace: SignalAllocator<U>,
        T: arithmetic::NaturalLogarithmTo<U>,
    {
        let mut destination = self.workspace.signal::<U>(self.len())?;
        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            arithmetic::natural_logarithm_to(self.stream_context, &source, &mut destination_view)?;
        }

        Ok(SignalPipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: SignalBacking::Owned(destination),
        })
    }

    pub fn natural_logarithm_scaled_to<U>(self, scale_factor: i32) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        Workspace: SignalAllocator<U>,
        T: arithmetic::NaturalLogarithmScaledTo<U>,
    {
        let mut destination = self.workspace.signal::<U>(self.len())?;
        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            arithmetic::natural_logarithm_scaled_to(
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

    pub fn natural_logarithm_to_scaled<U>(self, scale_factor: i32) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        Workspace: SignalAllocator<U>,
        T: arithmetic::NaturalLogarithmScaledTo<U>,
    {
        self.natural_logarithm_scaled_to::<U>(scale_factor)
    }

    pub fn cube_root_scaled_to<U>(self, scale_factor: i32) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        Workspace: SignalAllocator<U>,
        T: arithmetic::CubeRootScaledTo<U>,
    {
        let mut destination = self.workspace.signal::<U>(self.len())?;
        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            arithmetic::cube_root_scaled_to(
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

    pub fn cube_root_to_scaled<U>(self, scale_factor: i32) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        Workspace: SignalAllocator<U>,
        T: arithmetic::CubeRootScaledTo<U>,
    {
        self.cube_root_scaled_to::<U>(scale_factor)
    }

    pub fn square_root_scaled_to<U>(self, scale_factor: i32) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        Workspace: SignalAllocator<U>,
        T: arithmetic::SquareRootScaledTo<U>,
    {
        let mut destination = self.workspace.signal::<U>(self.len())?;
        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            arithmetic::square_root_scaled_to(
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

    pub fn square_root_to_scaled<U>(self, scale_factor: i32) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        Workspace: SignalAllocator<U>,
        T: arithmetic::SquareRootScaledTo<U>,
    {
        self.square_root_scaled_to::<U>(scale_factor)
    }

    fn unary_transform<O>(mut self) -> Result<Self>
    where
        Self: UnaryTransformSignal<T, O>,
    {
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut()?;
                <Self as UnaryTransformSignal<T, O>>::unary_transform_signal_in_place(
                    self.stream_context,
                    &mut signal_view,
                )?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self.workspace.signal::<T>(source.len())?;
                let mut destination_view = destination.view_mut()?;
                <Self as UnaryTransformSignal<T, O>>::unary_transform_signal(
                    self.stream_context,
                    source,
                    &mut destination_view,
                )?;
                self.backing = SignalBacking::Owned(destination);
            }
        }

        Ok(self)
    }

    fn unary_output<U, O>(self) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy,
        Workspace: SignalAllocator<U>,
        Self: UnaryOutputSignal<T, U, O>,
    {
        let mut destination = self.workspace.signal::<U>(self.len())?;
        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as UnaryOutputSignal<T, U, O>>::unary_output_signal(
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

    fn scaled_unary_output<U, O>(self, scale_factor: i32) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy,
        Workspace: SignalAllocator<U>,
        Self: ScaledUnaryOutputSignal<T, U, O>,
    {
        let mut destination = self.workspace.signal::<U>(self.len())?;
        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as ScaledUnaryOutputSignal<T, U, O>>::scaled_unary_output_signal(
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

    fn scaled_unary_transform<O>(mut self, scale_factor: i32) -> Result<Self>
    where
        Self: ScaledUnaryTransformSignal<T, O>,
    {
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut()?;
                <Self as ScaledUnaryTransformSignal<T, O>>::scaled_unary_transform_signal_in_place(
                    self.stream_context,
                    &mut signal_view,
                    scale_factor,
                )?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self.workspace.signal::<T>(source.len())?;
                let mut destination_view = destination.view_mut()?;
                <Self as ScaledUnaryTransformSignal<T, O>>::scaled_unary_transform_signal(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    scale_factor,
                )?;
                self.backing = SignalBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}

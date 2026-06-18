use crate::{
    context::StreamContext,
    error::Result,
    pipeline::{SignalAllocator, Workspace},
    signal::{
        arithmetic,
        view::{SignalView, SignalViewMut},
    },
    types::{ComplexI16, ComplexI32, DataTypeLike},
};

use super::{SignalBacking, SignalPipeline};

type ScaledConstantSignalOperation<T> =
    fn(&StreamContext, &SignalView<'_, T>, T, &mut SignalViewMut<'_, T>, i32) -> Result<()>;
type ScaledConstantSignalToOperation<T> =
    fn(&StreamContext, &SignalView<'_, T>, T, &mut SignalViewMut<'_, T>, i32) -> Result<()>;

impl<'a> SignalPipeline<'a, f32> {
    pub fn multiply_constant_f32_to_i16_low_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, f32>,
        value: f32,
        destination: &mut SignalViewMut<'_, i16>,
    ) -> Result<()> {
        arithmetic::multiply_constant_f32_to_i16_low(stream_context, source, value, destination)
    }

    pub fn multiply_constant_f32_to_i16_scaled_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, f32>,
        value: f32,
        destination: &mut SignalViewMut<'_, i16>,
        scale_factor: i32,
    ) -> Result<()> {
        arithmetic::multiply_constant_f32_to_i16_scaled(
            stream_context,
            source,
            value,
            destination,
            scale_factor,
        )
    }
}

impl<'a> SignalPipeline<'a, f64> {
    pub fn multiply_constant_f64_to_i64_scaled_in_place(
        stream_context: &StreamContext,
        destination: &mut SignalViewMut<'_, i64>,
        value: f64,
        scale_factor: i32,
    ) -> Result<()> {
        arithmetic::multiply_constant_f64_to_i64_scaled_in_place(
            stream_context,
            destination,
            value,
            scale_factor,
        )
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
{
    pub fn multiply_constant_low_to_into<U>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalViewMut<'_, U>,
    ) -> Result<()>
    where
        U: DataTypeLike,
        T: arithmetic::MultiplyConstantLowTo<U>,
    {
        arithmetic::multiply_constant_low_to(stream_context, source, value, destination)
    }

    pub fn multiply_constant_to_low_into<U>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalViewMut<'_, U>,
    ) -> Result<()>
    where
        U: DataTypeLike,
        T: arithmetic::MultiplyConstantLowTo<U>,
    {
        Self::multiply_constant_low_to_into(stream_context, source, value, destination)
    }

    pub fn multiply_constant_scaled_to_into<U>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalViewMut<'_, U>,
        scale_factor: i32,
    ) -> Result<()>
    where
        U: DataTypeLike,
        T: arithmetic::MultiplyConstantScaledTo<U>,
    {
        arithmetic::multiply_constant_scaled_to(
            stream_context,
            source,
            value,
            destination,
            scale_factor,
        )
    }

    pub fn multiply_constant_to_scaled_into<U>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalViewMut<'_, U>,
        scale_factor: i32,
    ) -> Result<()>
    where
        U: DataTypeLike,
        T: arithmetic::MultiplyConstantScaledTo<U>,
    {
        Self::multiply_constant_scaled_to_into(
            stream_context,
            source,
            value,
            destination,
            scale_factor,
        )
    }

    pub fn multiply_constant_scaled_to_in_place<Value>(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: Value,
        scale_factor: i32,
    ) -> Result<()>
    where
        Value: DataTypeLike,
        T: arithmetic::MultiplyConstantScaledToInPlace<Value>,
    {
        arithmetic::multiply_constant_scaled_to_in_place(
            stream_context,
            signal,
            value,
            scale_factor,
        )
    }

    pub fn multiply_constant_to_scaled_in_place<Value>(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: Value,
        scale_factor: i32,
    ) -> Result<()>
    where
        Value: DataTypeLike,
        T: arithmetic::MultiplyConstantScaledToInPlace<Value>,
    {
        Self::multiply_constant_scaled_to_in_place(stream_context, signal, value, scale_factor)
    }

    pub fn multiply_constant_low_to<U>(self, value: T) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        Workspace: SignalAllocator<U>,
        T: arithmetic::MultiplyConstantLowTo<U>,
    {
        let mut destination = self.workspace.signal::<U>(self.len())?;
        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            arithmetic::multiply_constant_low_to(
                self.stream_context,
                &source,
                value,
                &mut destination_view,
            )?;
        }

        Ok(SignalPipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: SignalBacking::Owned(destination),
        })
    }

    pub fn multiply_constant_scaled_to<U>(
        self,
        value: T,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        Workspace: SignalAllocator<U>,
        T: arithmetic::MultiplyConstantScaledTo<U>,
    {
        let mut destination = self.workspace.signal::<U>(self.len())?;
        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            arithmetic::multiply_constant_scaled_to(
                self.stream_context,
                &source,
                value,
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
}

#[path = "scaled_constant_traits.rs"]
mod scaled_constant_traits;

use self::scaled_constant_traits::{
    ConstantBitwiseSignal, ConstantShiftSignal, DivideScaledConstantSignal, ScaledConstantSignal,
};

#[macro_use]
#[path = "scaled_constant_macros.rs"]
mod scaled_constant_macros;

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: ScaledConstantSignal<T>,
{
    pub fn add_constant_scaled(self, value: T, scale_factor: i32) -> Result<Self> {
        self.scaled_constant(
            value,
            scale_factor,
            <Self as ScaledConstantSignal<T>>::add_constant_signal_scaled,
            <Self as ScaledConstantSignal<T>>::add_constant_signal_scaled_in_place,
        )
    }

    pub fn subtract_constant_scaled(self, value: T, scale_factor: i32) -> Result<Self> {
        self.scaled_constant(
            value,
            scale_factor,
            <Self as ScaledConstantSignal<T>>::subtract_constant_signal_scaled,
            <Self as ScaledConstantSignal<T>>::subtract_constant_signal_scaled_in_place,
        )
    }

    pub fn subtract_from_constant_scaled(self, value: T, scale_factor: i32) -> Result<Self> {
        self.scaled_constant(
            value,
            scale_factor,
            <Self as ScaledConstantSignal<T>>::subtract_from_constant_signal_scaled,
            <Self as ScaledConstantSignal<T>>::subtract_from_constant_signal_scaled_in_place,
        )
    }

    pub fn multiply_constant_scaled(self, value: T, scale_factor: i32) -> Result<Self> {
        self.scaled_constant(
            value,
            scale_factor,
            <Self as ScaledConstantSignal<T>>::multiply_constant_signal_scaled,
            <Self as ScaledConstantSignal<T>>::multiply_constant_signal_scaled_in_place,
        )
    }

    fn scaled_constant(
        mut self,
        value: T,
        scale_factor: i32,
        operation: ScaledConstantSignalOperation<T>,
        operation_in_place: fn(&StreamContext, &mut SignalViewMut<'_, T>, T, i32) -> Result<()>,
    ) -> Result<Self> {
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut()?;
                operation_in_place(self.stream_context, &mut signal_view, value, scale_factor)?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self.workspace.signal::<T>(source.len())?;
                let mut destination_view = destination.view_mut()?;
                operation(
                    self.stream_context,
                    source,
                    value,
                    &mut destination_view,
                    scale_factor,
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
    Self: ScaledConstantSignal<T>,
{
    pub fn add_constant_scaled_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ScaledConstantSignal<T>>::add_constant_signal_scaled(
            stream_context,
            source,
            value,
            destination,
            scale_factor,
        )
    }

    pub fn add_constant_scaled_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ScaledConstantSignal<T>>::add_constant_signal_scaled_in_place(
            stream_context,
            signal,
            value,
            scale_factor,
        )
    }

    pub fn subtract_constant_scaled_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ScaledConstantSignal<T>>::subtract_constant_signal_scaled(
            stream_context,
            source,
            value,
            destination,
            scale_factor,
        )
    }

    pub fn subtract_constant_scaled_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ScaledConstantSignal<T>>::subtract_constant_signal_scaled_in_place(
            stream_context,
            signal,
            value,
            scale_factor,
        )
    }

    pub fn subtract_from_constant_scaled_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ScaledConstantSignal<T>>::subtract_from_constant_signal_scaled(
            stream_context,
            source,
            value,
            destination,
            scale_factor,
        )
    }

    pub fn subtract_from_constant_scaled_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ScaledConstantSignal<T>>::subtract_from_constant_signal_scaled_in_place(
            stream_context,
            signal,
            value,
            scale_factor,
        )
    }

    pub fn multiply_constant_scaled_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ScaledConstantSignal<T>>::multiply_constant_signal_scaled(
            stream_context,
            source,
            value,
            destination,
            scale_factor,
        )
    }

    pub fn multiply_constant_scaled_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ScaledConstantSignal<T>>::multiply_constant_signal_scaled_in_place(
            stream_context,
            signal,
            value,
            scale_factor,
        )
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: ConstantBitwiseSignal<T>,
{
    pub fn bit_and_constant(self, value: T) -> Result<Self> {
        self.constant_bitwise(
            value,
            <Self as ConstantBitwiseSignal<T>>::and_constant_signal,
            <Self as ConstantBitwiseSignal<T>>::and_constant_signal_in_place,
        )
    }

    pub fn bit_or_constant(self, value: T) -> Result<Self> {
        self.constant_bitwise(
            value,
            <Self as ConstantBitwiseSignal<T>>::or_constant_signal,
            <Self as ConstantBitwiseSignal<T>>::or_constant_signal_in_place,
        )
    }

    pub fn bit_xor_constant(self, value: T) -> Result<Self> {
        self.constant_bitwise(
            value,
            <Self as ConstantBitwiseSignal<T>>::xor_constant_signal,
            <Self as ConstantBitwiseSignal<T>>::xor_constant_signal_in_place,
        )
    }

    fn constant_bitwise(
        mut self,
        value: T,
        operation: fn(
            &StreamContext,
            &SignalView<'_, T>,
            T,
            &mut SignalViewMut<'_, T>,
        ) -> Result<()>,
        operation_in_place: fn(&StreamContext, &mut SignalViewMut<'_, T>, T) -> Result<()>,
    ) -> Result<Self> {
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut()?;
                operation_in_place(self.stream_context, &mut signal_view, value)?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self.workspace.signal::<T>(source.len())?;
                let mut destination_view = destination.view_mut()?;
                operation(self.stream_context, source, value, &mut destination_view)?;
                self.backing = SignalBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Self: ConstantBitwiseSignal<T>,
{
    pub fn bit_and_constant_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as ConstantBitwiseSignal<T>>::and_constant_signal(
            stream_context,
            source,
            value,
            destination,
        )
    }

    pub fn bit_and_constant_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()> {
        <Self as ConstantBitwiseSignal<T>>::and_constant_signal_in_place(
            stream_context,
            signal,
            value,
        )
    }

    pub fn bit_or_constant_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as ConstantBitwiseSignal<T>>::or_constant_signal(
            stream_context,
            source,
            value,
            destination,
        )
    }

    pub fn bit_or_constant_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()> {
        <Self as ConstantBitwiseSignal<T>>::or_constant_signal_in_place(
            stream_context,
            signal,
            value,
        )
    }

    pub fn bit_xor_constant_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as ConstantBitwiseSignal<T>>::xor_constant_signal(
            stream_context,
            source,
            value,
            destination,
        )
    }

    pub fn bit_xor_constant_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()> {
        <Self as ConstantBitwiseSignal<T>>::xor_constant_signal_in_place(
            stream_context,
            signal,
            value,
        )
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: ConstantShiftSignal<T>,
{
    pub fn left_shift_constant(self, shift: i32) -> Result<Self> {
        self.constant_shift(
            shift,
            <Self as ConstantShiftSignal<T>>::left_shift_constant_signal,
            <Self as ConstantShiftSignal<T>>::left_shift_constant_signal_in_place,
        )
    }

    pub fn right_shift_constant(self, shift: i32) -> Result<Self> {
        self.constant_shift(
            shift,
            <Self as ConstantShiftSignal<T>>::right_shift_constant_signal,
            <Self as ConstantShiftSignal<T>>::right_shift_constant_signal_in_place,
        )
    }

    fn constant_shift(
        mut self,
        shift: i32,
        operation: fn(
            &StreamContext,
            &SignalView<'_, T>,
            i32,
            &mut SignalViewMut<'_, T>,
        ) -> Result<()>,
        operation_in_place: fn(&StreamContext, &mut SignalViewMut<'_, T>, i32) -> Result<()>,
    ) -> Result<Self> {
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut()?;
                operation_in_place(self.stream_context, &mut signal_view, shift)?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self.workspace.signal::<T>(source.len())?;
                let mut destination_view = destination.view_mut()?;
                operation(self.stream_context, source, shift, &mut destination_view)?;
                self.backing = SignalBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Self: ConstantShiftSignal<T>,
{
    pub fn left_shift_constant_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        shift: i32,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as ConstantShiftSignal<T>>::left_shift_constant_signal(
            stream_context,
            source,
            shift,
            destination,
        )
    }

    pub fn left_shift_constant_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        shift: i32,
    ) -> Result<()> {
        <Self as ConstantShiftSignal<T>>::left_shift_constant_signal_in_place(
            stream_context,
            signal,
            shift,
        )
    }

    pub fn right_shift_constant_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        shift: i32,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as ConstantShiftSignal<T>>::right_shift_constant_signal(
            stream_context,
            source,
            shift,
            destination,
        )
    }

    pub fn right_shift_constant_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        shift: i32,
    ) -> Result<()> {
        <Self as ConstantShiftSignal<T>>::right_shift_constant_signal_in_place(
            stream_context,
            signal,
            shift,
        )
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: DivideScaledConstantSignal<T>,
{
    pub fn divide_constant_scaled(self, value: T, scale_factor: i32) -> Result<Self> {
        self.divide_scaled_constant(
            value,
            scale_factor,
            <Self as DivideScaledConstantSignal<T>>::divide_constant_signal_scaled,
            <Self as DivideScaledConstantSignal<T>>::divide_constant_signal_scaled_in_place,
        )
    }

    fn divide_scaled_constant(
        mut self,
        value: T,
        scale_factor: i32,
        operation: ScaledConstantSignalToOperation<T>,
        operation_in_place: fn(&StreamContext, &mut SignalViewMut<'_, T>, T, i32) -> Result<()>,
    ) -> Result<Self> {
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut()?;
                operation_in_place(self.stream_context, &mut signal_view, value, scale_factor)?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self.workspace.signal::<T>(source.len())?;
                let mut destination_view = destination.view_mut()?;
                operation(
                    self.stream_context,
                    source,
                    value,
                    &mut destination_view,
                    scale_factor,
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
    Self: DivideScaledConstantSignal<T>,
{
    pub fn divide_constant_scaled_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as DivideScaledConstantSignal<T>>::divide_constant_signal_scaled(
            stream_context,
            source,
            value,
            destination,
            scale_factor,
        )
    }

    pub fn divide_constant_scaled_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as DivideScaledConstantSignal<T>>::divide_constant_signal_scaled_in_place(
            stream_context,
            signal,
            value,
            scale_factor,
        )
    }
}

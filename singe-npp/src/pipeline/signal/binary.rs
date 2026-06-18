use singe_cuda::types::{Complex32, Complex64};

use crate::{
    context::StreamContext,
    error::{Error, Result},
    pipeline::{SignalAllocator, Workspace},
    signal::{
        arithmetic,
        view::{SignalView, SignalViewMut},
    },
    types::{ComplexI16, ComplexI32, RoundMode},
};

use super::{CopySignal, SignalBacking, SignalPipeline};

#[path = "binary_traits.rs"]
mod binary_traits;

use self::binary_traits::{
    AddHeterogeneousInPlaceSignal, AddProductSignal, BinarySignal, DivideByScaledSignal,
    DivideScaledBinarySignal, DivideSignal, HeterogeneousBinarySignal, MixedBinarySignal,
    MultiplyHeterogeneousInPlaceSignal, MultiplyLowScaledSignal, ScaledAddProductSignal,
    ScaledBinarySignal, ScaledHeterogeneousBinarySignal, ScaledHeterogeneousInPlaceSignal,
    ScaledMixedBinarySignal,
};

#[macro_use]
#[path = "binary_macros.rs"]
mod binary_macros;

impl_binary_signal!(
    i16,
    arithmetic::add_i16,
    arithmetic::add_i16_in_place,
    arithmetic::subtract_i16,
    arithmetic::subtract_i16_in_place,
    arithmetic::multiply_i16,
    arithmetic::multiply_i16_in_place
);

impl_binary_signal!(
    f32,
    arithmetic::add_f32,
    arithmetic::add_f32_in_place,
    arithmetic::subtract_f32,
    arithmetic::subtract_f32_in_place,
    arithmetic::multiply_f32,
    arithmetic::multiply_f32_in_place
);

impl_binary_signal!(
    f64,
    arithmetic::add_f64,
    arithmetic::add_f64_in_place,
    arithmetic::subtract_f64,
    arithmetic::subtract_f64_in_place,
    arithmetic::multiply_f64,
    arithmetic::multiply_f64_in_place
);
impl_binary_signal!(
    Complex32,
    arithmetic::add_f32_complex,
    arithmetic::add_f32_complex_in_place,
    arithmetic::subtract_f32_complex,
    arithmetic::subtract_f32_complex_in_place,
    arithmetic::multiply_f32_complex,
    arithmetic::multiply_f32_complex_in_place
);
impl_binary_signal!(
    Complex64,
    arithmetic::add_f64_complex,
    arithmetic::add_f64_complex_in_place,
    arithmetic::subtract_f64_complex,
    arithmetic::subtract_f64_complex_in_place,
    arithmetic::multiply_f64_complex,
    arithmetic::multiply_f64_complex_in_place
);

impl_divide_signal!(f32, arithmetic::divide_f32, arithmetic::divide_f32_in_place);
impl_divide_signal!(f64, arithmetic::divide_f64, arithmetic::divide_f64_in_place);
impl_divide_signal!(
    Complex32,
    arithmetic::divide_f32_complex,
    arithmetic::divide_f32_complex_in_place
);
impl_divide_signal!(
    Complex64,
    arithmetic::divide_f64_complex,
    arithmetic::divide_f64_complex_in_place
);

impl_scaled_binary_signal!(
    u8,
    arithmetic::add_u8_scaled,
    arithmetic::add_u8_scaled_in_place,
    arithmetic::subtract_u8_scaled,
    arithmetic::subtract_u8_scaled_in_place,
    arithmetic::multiply_u8_scaled,
    arithmetic::multiply_u8_scaled_in_place
);
impl_divide_scaled_binary_signal!(
    u8,
    arithmetic::divide_u8_scaled,
    arithmetic::divide_u8_scaled_in_place
);

impl_scaled_binary_signal!(
    u16,
    arithmetic::add_u16_scaled,
    arithmetic::add_u16_scaled_in_place,
    arithmetic::subtract_u16_scaled,
    arithmetic::subtract_u16_scaled_in_place,
    arithmetic::multiply_u16_scaled,
    arithmetic::multiply_u16_scaled_in_place
);
impl_divide_scaled_binary_signal!(
    u16,
    arithmetic::divide_u16_scaled,
    arithmetic::divide_u16_scaled_in_place
);

impl_scaled_binary_signal!(
    i16,
    arithmetic::add_i16_scaled,
    arithmetic::add_i16_scaled_in_place,
    arithmetic::subtract_i16_scaled,
    arithmetic::subtract_i16_scaled_in_place,
    arithmetic::multiply_i16_scaled,
    arithmetic::multiply_i16_scaled_in_place
);
impl_divide_scaled_binary_signal!(
    i16,
    arithmetic::divide_i16_scaled,
    arithmetic::divide_i16_scaled_in_place
);

impl_scaled_binary_signal!(
    i32,
    arithmetic::add_i32_scaled,
    arithmetic::add_i32_scaled_in_place,
    arithmetic::subtract_i32_scaled,
    arithmetic::subtract_i32_scaled_in_place,
    arithmetic::multiply_i32_scaled,
    arithmetic::multiply_i32_scaled_in_place
);
impl_divide_scaled_binary_signal!(
    i32,
    arithmetic::divide_i32_scaled,
    arithmetic::divide_i32_scaled_in_place
);
impl_scaled_binary_signal!(
    ComplexI16,
    arithmetic::add_i16_complex_scaled,
    arithmetic::add_i16_complex_scaled_in_place,
    arithmetic::subtract_i16_complex_scaled,
    arithmetic::subtract_i16_complex_scaled_in_place,
    arithmetic::multiply_i16_complex_scaled,
    arithmetic::multiply_i16_complex_scaled_in_place
);
impl_divide_scaled_binary_signal!(
    ComplexI16,
    arithmetic::divide_i16_complex_scaled,
    arithmetic::divide_i16_complex_scaled_in_place
);
impl_scaled_binary_signal!(
    ComplexI32,
    arithmetic::add_i32_complex_scaled,
    arithmetic::add_i32_complex_scaled_in_place,
    arithmetic::subtract_i32_complex_scaled,
    arithmetic::subtract_i32_complex_scaled_in_place,
    arithmetic::multiply_i32_complex_scaled,
    arithmetic::multiply_i32_complex_scaled_in_place
);

impl_mixed_binary_signal_without_subtract!(
    u8,
    u16,
    arithmetic::add_u8_to_u16,
    arithmetic::multiply_u8_to_u16
);

impl_mixed_binary_signal!(
    i16,
    f32,
    arithmetic::add_i16_to_f32,
    arithmetic::subtract_i16_to_f32,
    arithmetic::multiply_i16_to_f32
);

impl_scaled_mixed_binary_signal!(i16, i32, arithmetic::multiply_i16_to_i32_scaled);

impl_scaled_heterogeneous_binary_signal!(u16, i16, i16, arithmetic::multiply_u16_with_i16_scaled);
impl_scaled_heterogeneous_binary_signal!(
    i32,
    ComplexI32,
    ComplexI32,
    arithmetic::multiply_i32_with_i32_complex_scaled
);

impl<'a> MultiplyLowScaledSignal<i32> for SignalPipeline<'a, i32> {
    fn multiply_low_signal_scaled(
        stream_context: &StreamContext,
        left: &SignalView<'_, i32>,
        right: &SignalView<'_, i32>,
        destination: &mut SignalViewMut<'_, i32>,
        scale_factor: i32,
    ) -> Result<()> {
        arithmetic::multiply_low_i32_scaled(stream_context, left, right, destination, scale_factor)
    }
}

impl<'a> DivideByScaledSignal<i32, i16, i16> for SignalPipeline<'a, i32> {
    fn divide_by_signal_scaled(
        stream_context: &StreamContext,
        left: &SignalView<'_, i32>,
        right: &SignalView<'_, i16>,
        destination: &mut SignalViewMut<'_, i16>,
        scale_factor: i32,
    ) -> Result<()> {
        arithmetic::divide_i32_by_i16_scaled(stream_context, left, right, destination, scale_factor)
    }
}

impl<'a> ScaledHeterogeneousInPlaceSignal<ComplexI32, i32> for SignalPipeline<'a, ComplexI32> {
    fn multiply_heterogeneous_signal_scaled_in_place(
        stream_context: &StreamContext,
        source: &SignalView<'_, i32>,
        destination: &mut SignalViewMut<'_, ComplexI32>,
        scale_factor: i32,
    ) -> Result<()> {
        arithmetic::multiply_i32_complex_by_i32_scaled_in_place(
            stream_context,
            source,
            destination,
            scale_factor,
        )
    }
}

impl<'a> AddHeterogeneousInPlaceSignal<i32, i16> for SignalPipeline<'a, i32> {
    fn add_heterogeneous_signal_in_place(
        stream_context: &StreamContext,
        source: &SignalView<'_, i16>,
        destination: &mut SignalViewMut<'_, i32>,
    ) -> Result<()> {
        arithmetic::add_i16_to_i32_in_place(stream_context, source, destination)
    }
}

impl<'a> MultiplyHeterogeneousInPlaceSignal<Complex32, f32> for SignalPipeline<'a, Complex32> {
    fn multiply_heterogeneous_signal_in_place(
        stream_context: &StreamContext,
        source: &SignalView<'_, f32>,
        destination: &mut SignalViewMut<'_, Complex32>,
    ) -> Result<()> {
        arithmetic::multiply_f32_complex_by_f32_in_place(stream_context, source, destination)
    }
}

impl<'a> HeterogeneousBinarySignal<f32, Complex32, Complex32> for SignalPipeline<'a, f32> {
    fn multiply_heterogeneous_signal(
        stream_context: &StreamContext,
        left: &SignalView<'_, f32>,
        right: &SignalView<'_, Complex32>,
        destination: &mut SignalViewMut<'_, Complex32>,
    ) -> Result<()> {
        arithmetic::multiply_f32_with_f32_complex(stream_context, left, right, destination)
    }
}

impl_add_product_signal!(f32, arithmetic::add_product_f32);
impl_add_product_signal!(f64, arithmetic::add_product_f64);
impl_add_product_signal!(Complex32, arithmetic::add_product_f32_complex);
impl_add_product_signal!(Complex64, arithmetic::add_product_f64_complex);

impl_scaled_add_product_signal!(i16, i16, arithmetic::add_product_i16_scaled);
impl_scaled_add_product_signal!(i32, i32, arithmetic::add_product_i32_scaled);
impl_scaled_add_product_signal!(i16, i32, arithmetic::add_product_i16_to_i32_scaled);

impl<'a> SignalPipeline<'a, u32> {
    pub fn add_u32_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, u32>,
        right: &SignalView<'_, u32>,
        destination: &mut SignalViewMut<'_, u32>,
    ) -> Result<()> {
        arithmetic::add_u32(stream_context, left, right, destination)
    }
}

impl<'a> SignalPipeline<'a, i64> {
    pub fn add_i64_scaled_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, i64>,
        right: &SignalView<'_, i64>,
        destination: &mut SignalViewMut<'_, i64>,
        scale_factor: i32,
    ) -> Result<()> {
        arithmetic::add_i64_scaled(stream_context, left, right, destination, scale_factor)
    }
}

impl<'a> SignalPipeline<'a, i32> {
    pub fn multiply_low_i32_scaled_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, i32>,
        right: &SignalView<'_, i32>,
        destination: &mut SignalViewMut<'_, i32>,
        scale_factor: i32,
    ) -> Result<()> {
        arithmetic::multiply_low_i32_scaled(stream_context, left, right, destination, scale_factor)
    }

    pub fn divide_i32_by_i16_scaled_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, i32>,
        right: &SignalView<'_, i16>,
        destination: &mut SignalViewMut<'_, i16>,
        scale_factor: i32,
    ) -> Result<()> {
        arithmetic::divide_i32_by_i16_scaled(stream_context, left, right, destination, scale_factor)
    }

    pub fn multiply_i32_complex_by_i32_scaled_in_place(
        stream_context: &StreamContext,
        source: &SignalView<'_, i32>,
        destination: &mut SignalViewMut<'_, ComplexI32>,
        scale_factor: i32,
    ) -> Result<()> {
        arithmetic::multiply_i32_complex_by_i32_scaled_in_place(
            stream_context,
            source,
            destination,
            scale_factor,
        )
    }
}

impl<'a> SignalPipeline<'a, i16> {
    pub fn add_i16_to_i32_in_place(
        stream_context: &StreamContext,
        source: &SignalView<'_, i16>,
        destination: &mut SignalViewMut<'_, i32>,
    ) -> Result<()> {
        arithmetic::add_i16_to_i32_in_place(stream_context, source, destination)
    }
}

impl<'a> SignalPipeline<'a, f32> {
    pub fn multiply_f32_with_f32_complex_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, f32>,
        right: &SignalView<'_, Complex32>,
        destination: &mut SignalViewMut<'_, Complex32>,
    ) -> Result<()> {
        arithmetic::multiply_f32_with_f32_complex(stream_context, left, right, destination)
    }

    pub fn multiply_f32_complex_by_f32_in_place(
        stream_context: &StreamContext,
        source: &SignalView<'_, f32>,
        destination: &mut SignalViewMut<'_, Complex32>,
    ) -> Result<()> {
        arithmetic::multiply_f32_complex_by_f32_in_place(stream_context, source, destination)
    }
}

#[path = "binary_bitwise.rs"]
mod binary_bitwise;

use self::binary_bitwise::BitwiseSignal;

#[path = "binary_basic_methods.rs"]
mod binary_basic_methods;

#[path = "binary_mixed_methods.rs"]
mod binary_mixed_methods;

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: ScaledBinarySignal<T>,
{
    pub fn add_scaled(self, other: &SignalView<'_, T>, scale_factor: i32) -> Result<Self> {
        self.scaled_binary(
            other,
            scale_factor,
            <Self as ScaledBinarySignal<T>>::add_signal_scaled,
            <Self as ScaledBinarySignal<T>>::add_signal_scaled_in_place,
        )
    }

    pub fn subtract_scaled(self, other: &SignalView<'_, T>, scale_factor: i32) -> Result<Self> {
        self.scaled_binary(
            other,
            scale_factor,
            <Self as ScaledBinarySignal<T>>::subtract_signal_scaled,
            <Self as ScaledBinarySignal<T>>::subtract_signal_scaled_in_place,
        )
    }

    pub fn multiply_scaled(self, other: &SignalView<'_, T>, scale_factor: i32) -> Result<Self> {
        self.scaled_binary(
            other,
            scale_factor,
            <Self as ScaledBinarySignal<T>>::multiply_signal_scaled,
            <Self as ScaledBinarySignal<T>>::multiply_signal_scaled_in_place,
        )
    }

    fn scaled_binary(
        mut self,
        other: &SignalView<'_, T>,
        scale_factor: i32,
        operation: fn(
            &StreamContext,
            &SignalView<'_, T>,
            &SignalView<'_, T>,
            &mut SignalViewMut<'_, T>,
            i32,
        ) -> Result<()>,
        operation_in_place: fn(
            &StreamContext,
            &SignalView<'_, T>,
            &mut SignalViewMut<'_, T>,
            i32,
        ) -> Result<()>,
    ) -> Result<Self> {
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut()?;
                operation_in_place(self.stream_context, other, &mut signal_view, scale_factor)?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self.workspace.signal::<T>(source.len())?;
                let mut destination_view = destination.view_mut()?;
                operation(
                    self.stream_context,
                    source,
                    other,
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
    Workspace: SignalAllocator<T>,
    Self: DivideScaledBinarySignal<T>,
{
    pub fn divide_scaled(self, other: &SignalView<'_, T>, scale_factor: i32) -> Result<Self> {
        self.scaled_binary_divide(
            other,
            scale_factor,
            <Self as DivideScaledBinarySignal<T>>::divide_signal_scaled,
            <Self as DivideScaledBinarySignal<T>>::divide_signal_scaled_in_place,
        )
    }

    fn scaled_binary_divide(
        mut self,
        other: &SignalView<'_, T>,
        scale_factor: i32,
        operation: fn(
            &StreamContext,
            &SignalView<'_, T>,
            &SignalView<'_, T>,
            &mut SignalViewMut<'_, T>,
            i32,
        ) -> Result<()>,
        operation_in_place: fn(
            &StreamContext,
            &SignalView<'_, T>,
            &mut SignalViewMut<'_, T>,
            i32,
        ) -> Result<()>,
    ) -> Result<Self> {
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut()?;
                operation_in_place(self.stream_context, other, &mut signal_view, scale_factor)?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self.workspace.signal::<T>(source.len())?;
                let mut destination_view = destination.view_mut()?;
                operation(
                    self.stream_context,
                    source,
                    other,
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
    Workspace: SignalAllocator<T>,
    Self: BitwiseSignal<T>,
{
    pub fn bit_and_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as BitwiseSignal<T>>::and_signal(stream_context, left, right, destination)
    }

    pub fn bit_and_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as BitwiseSignal<T>>::and_signal_in_place(stream_context, signal, destination)
    }

    pub fn bit_or_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as BitwiseSignal<T>>::or_signal(stream_context, left, right, destination)
    }

    pub fn bit_or_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as BitwiseSignal<T>>::or_signal_in_place(stream_context, signal, destination)
    }

    pub fn bit_xor_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as BitwiseSignal<T>>::xor_signal(stream_context, left, right, destination)
    }

    pub fn bit_xor_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as BitwiseSignal<T>>::xor_signal_in_place(stream_context, signal, destination)
    }

    pub fn bit_and(self, other: &SignalView<'_, T>) -> Result<Self> {
        self.bitwise(
            other,
            <Self as BitwiseSignal<T>>::and_signal,
            <Self as BitwiseSignal<T>>::and_signal_in_place,
        )
    }

    pub fn bit_or(self, other: &SignalView<'_, T>) -> Result<Self> {
        self.bitwise(
            other,
            <Self as BitwiseSignal<T>>::or_signal,
            <Self as BitwiseSignal<T>>::or_signal_in_place,
        )
    }

    pub fn bit_xor(self, other: &SignalView<'_, T>) -> Result<Self> {
        self.bitwise(
            other,
            <Self as BitwiseSignal<T>>::xor_signal,
            <Self as BitwiseSignal<T>>::xor_signal_in_place,
        )
    }

    fn bitwise(
        mut self,
        other: &SignalView<'_, T>,
        operation: fn(
            &StreamContext,
            &SignalView<'_, T>,
            &SignalView<'_, T>,
            &mut SignalViewMut<'_, T>,
        ) -> Result<()>,
        operation_in_place: fn(
            &StreamContext,
            &SignalView<'_, T>,
            &mut SignalViewMut<'_, T>,
        ) -> Result<()>,
    ) -> Result<Self> {
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut()?;
                operation_in_place(self.stream_context, other, &mut signal_view)?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self.workspace.signal::<T>(source.len())?;
                let mut destination_view = destination.view_mut()?;
                operation(self.stream_context, source, other, &mut destination_view)?;
                self.backing = SignalBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}

impl<'a> SignalPipeline<'a, f32> {
    pub fn divide(self, other: &SignalView<'_, f32>) -> Result<Self> {
        self.binary(
            other,
            <Self as DivideSignal<f32>>::divide_signal,
            <Self as DivideSignal<f32>>::divide_signal_in_place,
        )
    }
}

impl<'a> SignalPipeline<'a, f64> {
    pub fn divide(self, other: &SignalView<'_, f64>) -> Result<Self> {
        self.binary(
            other,
            <Self as DivideSignal<f64>>::divide_signal,
            <Self as DivideSignal<f64>>::divide_signal_in_place,
        )
    }
}

impl<'a> SignalPipeline<'a, Complex32> {
    pub fn divide(self, other: &SignalView<'_, Complex32>) -> Result<Self> {
        self.binary(
            other,
            <Self as DivideSignal<Complex32>>::divide_signal,
            <Self as DivideSignal<Complex32>>::divide_signal_in_place,
        )
    }
}

impl<'a> SignalPipeline<'a, Complex64> {
    pub fn divide(self, other: &SignalView<'_, Complex64>) -> Result<Self> {
        self.binary(
            other,
            <Self as DivideSignal<Complex64>>::divide_signal,
            <Self as DivideSignal<Complex64>>::divide_signal_in_place,
        )
    }
}

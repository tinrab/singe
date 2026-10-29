use crate::{
    context::StreamContext,
    error::Result,
    pipeline::{
        signal::{
            SignalBacking, SignalPipeline,
            binary::binary_traits::{
                AddHeterogeneousInPlaceSignal, DivideByScaledSignal, HeterogeneousBinarySignal,
                MixedBinarySignal, MultiplyHeterogeneousInPlaceSignal, MultiplyLowScaledSignal,
                ScaledHeterogeneousBinarySignal, ScaledHeterogeneousInPlaceSignal,
                ScaledMixedBinarySignal,
            },
        },
        workspace::{SignalAllocator, Workspace},
    },
    signal::view::{SignalOutputView, SignalView, SignalViewMut},
};

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
{
    fn mixed_binary<U>(
        self,
        other: &SignalView<'_, T>,
        operation: MixedBinaryCallback<T, U>,
    ) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy,
        Workspace: SignalAllocator<U>,
        Self: MixedBinarySignal<T, U>,
    {
        self.ensure_eager()?;
        let mut destination = self
            .workspace
            .signal_uninit::<U>(self.stream_context.cuda_context(), self.len())?;
        {
            let source = self.view();
            let mut destination_view = destination.output();
            operation(self.stream_context, &source, other, &mut destination_view)?;
        }

        Ok(SignalPipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: SignalBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
{
    /// Multiplies `left` and `right` element by element into `destination`.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    /// Uses the native low multiplication routine.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn multiply_low_scaled_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scale_factor: i32,
    ) -> Result<()>
    where
        Self: MultiplyLowScaledSignal<T>,
    {
        <Self as MultiplyLowScaledSignal<T>>::multiply_low_signal_scaled(
            stream_context,
            left,
            right,
            destination,
            scale_factor,
        )
    }

    /// Computes the element-wise quotient `left / right` into `destination`.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn divide_by_scaled_into<U, V>(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, U>,
        destination: &mut SignalOutputView<'_, V>,
        scale_factor: i32,
    ) -> Result<()>
    where
        U: Copy,
        V: Copy,
        Self: DivideByScaledSignal<T, U, V>,
    {
        <Self as DivideByScaledSignal<T, U, V>>::divide_by_signal_scaled(
            stream_context,
            left,
            right,
            destination,
            scale_factor,
        )
    }

    /// Multiplies `destination` by `source` element by element, updating `destination` in place.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn multiply_complex_by_scaled_in_place<U>(
        stream_context: &StreamContext,
        source: &SignalView<'_, U>,
        destination: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()>
    where
        U: Copy,
        Self: ScaledHeterogeneousInPlaceSignal<T, U>,
    {
        <Self as ScaledHeterogeneousInPlaceSignal<T, U>>::multiply_heterogeneous_signal_scaled_in_place(
            stream_context,
            source,
            destination,
            scale_factor,
        )
    }

    /// Adds `source` to `destination` element by element, updating `destination` in place.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn add_to_in_place<U>(
        stream_context: &StreamContext,
        source: &SignalView<'_, U>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>
    where
        U: Copy,
        Self: AddHeterogeneousInPlaceSignal<T, U>,
    {
        <Self as AddHeterogeneousInPlaceSignal<T, U>>::add_heterogeneous_signal_in_place(
            stream_context,
            source,
            destination,
        )
    }

    /// Multiplies `left` and `right` element by element into `destination`.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn multiply_with_complex_into<U, V>(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, U>,
        destination: &mut SignalOutputView<'_, V>,
    ) -> Result<()>
    where
        U: Copy,
        V: Copy,
        Self: HeterogeneousBinarySignal<T, U, V>,
    {
        <Self as HeterogeneousBinarySignal<T, U, V>>::multiply_heterogeneous_signal(
            stream_context,
            left,
            right,
            destination,
        )
    }

    /// Multiplies `destination` by `source` element by element, updating `destination` in place.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn multiply_complex_by_in_place<U>(
        stream_context: &StreamContext,
        source: &SignalView<'_, U>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>
    where
        U: Copy,
        Self: MultiplyHeterogeneousInPlaceSignal<T, U>,
    {
        <Self as MultiplyHeterogeneousInPlaceSignal<T, U>>::multiply_heterogeneous_signal_in_place(
            stream_context,
            source,
            destination,
        )
    }

    /// Adds `other` to the pipeline element by element.
    /// Returns an owned pipeline containing the result.
    ///
    /// Allocates output through the workspace using the destination element type.
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the pipeline source producer can also fail.
    pub fn add_to<U>(self, other: &SignalView<'_, T>) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy,
        Workspace: SignalAllocator<U>,
        Self: MixedBinarySignal<T, U>,
    {
        self.ensure_eager()?;
        self.mixed_binary(other, <Self as MixedBinarySignal<T, U>>::add_mixed_signal)
    }

    /// Subtracts `other` from each pipeline element.
    /// Returns an owned pipeline containing the result.
    ///
    /// Allocates output through the workspace using the destination element type.
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the pipeline source producer can also fail.
    /// The provided `u8` to `u16` mixed subtraction dispatch returns `Error::UnsupportedOperation`.
    pub fn subtract_to<U>(self, other: &SignalView<'_, T>) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy,
        Workspace: SignalAllocator<U>,
        Self: MixedBinarySignal<T, U>,
    {
        self.ensure_eager()?;
        self.mixed_binary(
            other,
            <Self as MixedBinarySignal<T, U>>::subtract_mixed_signal,
        )
    }

    /// Multiplies the pipeline value by `other` element by element.
    /// Returns an owned pipeline containing the result.
    ///
    /// Allocates output through the workspace using the destination element type.
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the pipeline source producer can also fail.
    pub fn multiply_to<U>(self, other: &SignalView<'_, T>) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy,
        Workspace: SignalAllocator<U>,
        Self: MixedBinarySignal<T, U>,
    {
        self.ensure_eager()?;
        self.mixed_binary(
            other,
            <Self as MixedBinarySignal<T, U>>::multiply_mixed_signal,
        )
    }

    /// Adds `left` and `right` element by element into `destination`.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn add_mixed_into<U>(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, U>,
    ) -> Result<()>
    where
        U: Copy,
        Self: MixedBinarySignal<T, U>,
    {
        <Self as MixedBinarySignal<T, U>>::add_mixed_signal(
            stream_context,
            left,
            right,
            destination,
        )
    }

    /// Computes the element-wise difference `left - right` into `destination`.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// The provided `u8` to `u16` mixed subtraction dispatch returns `Error::UnsupportedOperation`.
    pub fn subtract_mixed_into<U>(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, U>,
    ) -> Result<()>
    where
        U: Copy,
        Self: MixedBinarySignal<T, U>,
    {
        <Self as MixedBinarySignal<T, U>>::subtract_mixed_signal(
            stream_context,
            left,
            right,
            destination,
        )
    }

    /// Multiplies `left` and `right` element by element into `destination`.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn multiply_mixed_into<U>(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, U>,
    ) -> Result<()>
    where
        U: Copy,
        Self: MixedBinarySignal<T, U>,
    {
        <Self as MixedBinarySignal<T, U>>::multiply_mixed_signal(
            stream_context,
            left,
            right,
            destination,
        )
    }

    /// Multiplies `left` and `right` element by element into `destination`.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn multiply_mixed_scaled_into<U>(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, U>,
        scale_factor: i32,
    ) -> Result<()>
    where
        U: Copy,
        Self: ScaledMixedBinarySignal<T, U>,
    {
        <Self as ScaledMixedBinarySignal<T, U>>::multiply_mixed_signal_scaled(
            stream_context,
            left,
            right,
            destination,
            scale_factor,
        )
    }

    /// Multiplies `left` and `right` element by element into `destination`.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn multiply_heterogeneous_scaled_into<U, V>(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, U>,
        destination: &mut SignalOutputView<'_, V>,
        scale_factor: i32,
    ) -> Result<()>
    where
        U: Copy,
        V: Copy,
        Self: ScaledHeterogeneousBinarySignal<T, U, V>,
    {
        <Self as ScaledHeterogeneousBinarySignal<T, U, V>>::multiply_heterogeneous_signal_scaled(
            stream_context,
            left,
            right,
            destination,
            scale_factor,
        )
    }
}

impl<'a> SignalPipeline<'a, u8> {
    /// Adds `other` to the pipeline element by element.
    /// Returns an owned pipeline containing the result.
    ///
    /// Allocates output through the workspace using the destination element type.
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the pipeline source producer can also fail.
    pub fn add_to_u16(self, other: &SignalView<'_, u8>) -> Result<SignalPipeline<'a, u16>> {
        self.ensure_eager()?;
        self.mixed_binary(
            other,
            <Self as MixedBinarySignal<u8, u16>>::add_mixed_signal,
        )
    }

    /// Multiplies the pipeline value by `other` element by element.
    /// Returns an owned pipeline containing the result.
    ///
    /// Allocates output through the workspace using the destination element type.
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the pipeline source producer can also fail.
    pub fn multiply_to_u16(self, other: &SignalView<'_, u8>) -> Result<SignalPipeline<'a, u16>> {
        self.ensure_eager()?;
        self.mixed_binary(
            other,
            <Self as MixedBinarySignal<u8, u16>>::multiply_mixed_signal,
        )
    }
}

impl<'a> SignalPipeline<'a, i16> {
    /// Adds `other` to the pipeline element by element.
    /// Returns an owned pipeline containing the result.
    ///
    /// Allocates output through the workspace using the destination element type.
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the pipeline source producer can also fail.
    pub fn add_to_f32(self, other: &SignalView<'_, i16>) -> Result<SignalPipeline<'a, f32>> {
        self.ensure_eager()?;
        self.mixed_binary(
            other,
            <Self as MixedBinarySignal<i16, f32>>::add_mixed_signal,
        )
    }

    /// Subtracts `other` from each pipeline element.
    /// Returns an owned pipeline containing the result.
    ///
    /// Allocates output through the workspace using the destination element type.
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the pipeline source producer can also fail.
    pub fn subtract_to_f32(self, other: &SignalView<'_, i16>) -> Result<SignalPipeline<'a, f32>> {
        self.ensure_eager()?;
        self.mixed_binary(
            other,
            <Self as MixedBinarySignal<i16, f32>>::subtract_mixed_signal,
        )
    }

    /// Multiplies the pipeline value by `other` element by element.
    /// Returns an owned pipeline containing the result.
    ///
    /// Allocates output through the workspace using the destination element type.
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the pipeline source producer can also fail.
    pub fn multiply_to_f32(self, other: &SignalView<'_, i16>) -> Result<SignalPipeline<'a, f32>> {
        self.ensure_eager()?;
        self.mixed_binary(
            other,
            <Self as MixedBinarySignal<i16, f32>>::multiply_mixed_signal,
        )
    }

    /// Multiplies the pipeline value by `other` element by element.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    /// Returns an owned pipeline containing the result.
    ///
    /// Allocates output through the workspace using the destination element type.
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the pipeline source producer can also fail.
    pub fn multiply_to_i32_scaled(
        self,
        other: &SignalView<'_, i16>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i32>> {
        self.ensure_eager()?;
        self.scaled_mixed_binary(
            other,
            scale_factor,
            <Self as ScaledMixedBinarySignal<i16, i32>>::multiply_mixed_signal_scaled,
        )
    }
}

impl<'a> SignalPipeline<'a, u16> {
    /// Multiplies the pipeline value by `other` element by element.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    /// Returns an owned pipeline containing the result.
    ///
    /// Allocates output through the workspace using the destination element type.
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the pipeline source producer can also fail.
    pub fn multiply_with_i16_scaled(
        self,
        other: &SignalView<'_, i16>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i16>> {
        self.ensure_eager()?;
        self.scaled_heterogeneous_binary(
            other,
            scale_factor,
            <Self as ScaledHeterogeneousBinarySignal<u16, i16, i16>>::multiply_heterogeneous_signal_scaled,
        )
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
{
    /// Multiplies the pipeline value by `other` element by element.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    /// Returns an owned pipeline containing the result.
    ///
    /// Allocates output through the workspace using the destination element type.
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the pipeline source producer can also fail.
    pub fn multiply_to_scaled<U>(
        self,
        other: &SignalView<'_, T>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy,
        Workspace: SignalAllocator<U>,
        Self: ScaledMixedBinarySignal<T, U>,
    {
        self.ensure_eager()?;
        self.scaled_mixed_binary(
            other,
            scale_factor,
            <Self as ScaledMixedBinarySignal<T, U>>::multiply_mixed_signal_scaled,
        )
    }

    fn scaled_mixed_binary<U>(
        self,
        other: &SignalView<'_, T>,
        scale_factor: i32,
        operation: ScaledMixedBinaryCallback<T, U>,
    ) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy,
        Workspace: SignalAllocator<U>,
        Self: ScaledMixedBinarySignal<T, U>,
    {
        self.ensure_eager()?;
        let mut destination = self
            .workspace
            .signal_uninit::<U>(self.stream_context.cuda_context(), self.len())?;
        {
            let source = self.view();
            let mut destination_view = destination.output();
            operation(
                self.stream_context,
                &source,
                other,
                &mut destination_view,
                scale_factor,
            )?;
        }

        Ok(SignalPipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: SignalBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
{
    /// Multiplies the pipeline value by `other` element by element.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    /// Returns an owned pipeline containing the result.
    ///
    /// Allocates output through the workspace using the destination element type.
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the pipeline source producer can also fail.
    pub fn multiply_with_scaled<U, V>(
        self,
        other: &SignalView<'_, U>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, V>>
    where
        U: Copy,
        V: Copy,
        Workspace: SignalAllocator<V>,
        Self: ScaledHeterogeneousBinarySignal<T, U, V>,
    {
        self.ensure_eager()?;
        self.scaled_heterogeneous_binary(
            other,
            scale_factor,
            <Self as ScaledHeterogeneousBinarySignal<T, U, V>>::multiply_heterogeneous_signal_scaled,
        )
    }

    fn scaled_heterogeneous_binary<U, V>(
        self,
        other: &SignalView<'_, U>,
        scale_factor: i32,
        operation: ScaledHeterogeneousBinaryCallback<T, U, V>,
    ) -> Result<SignalPipeline<'a, V>>
    where
        U: Copy,
        V: Copy,
        Workspace: SignalAllocator<V>,
        Self: ScaledHeterogeneousBinarySignal<T, U, V>,
    {
        self.ensure_eager()?;
        let mut destination = self
            .workspace
            .signal_uninit::<V>(self.stream_context.cuda_context(), self.len())?;
        {
            let source = self.view();
            let mut destination_view = destination.output();
            operation(
                self.stream_context,
                &source,
                other,
                &mut destination_view,
                scale_factor,
            )?;
        }

        Ok(SignalPipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: SignalBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}

type MixedBinaryCallback<T, U> = fn(
    &StreamContext,
    &SignalView<'_, T>,
    &SignalView<'_, T>,
    &mut SignalOutputView<'_, U>,
) -> Result<()>;

type ScaledMixedBinaryCallback<T, U> = fn(
    &StreamContext,
    &SignalView<'_, T>,
    &SignalView<'_, T>,
    &mut SignalOutputView<'_, U>,
    i32,
) -> Result<()>;

type ScaledHeterogeneousBinaryCallback<T, U, V> = fn(
    &StreamContext,
    &SignalView<'_, T>,
    &SignalView<'_, U>,
    &mut SignalOutputView<'_, V>,
    i32,
) -> Result<()>;

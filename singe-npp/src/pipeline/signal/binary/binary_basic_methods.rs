use crate::{
    context::StreamContext,
    error::Result,
    pipeline::{
        signal::{
            SignalBacking, SignalPipeline,
            binary::binary_traits::{
                AddProductSignal, BinarySignal, DivideScaledBinarySignal, DivideSignal,
                ScaledAddProductSignal, ScaledBinarySignal,
            },
            copy::CopySignal,
        },
        workspace::{SignalAllocator, Workspace},
    },
    signal::view::{SignalOutputView, SignalView, SignalViewMut},
};

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: BinarySignal<T>,
{
    /// Adds `other` to the pipeline element by element.
    /// Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the pipeline source producer can also fail.
    pub fn add_signal(self, other: &SignalView<'_, T>) -> Result<Self> {
        self.ensure_eager()?;
        self.binary(
            other,
            <Self as BinarySignal<T>>::add_signal,
            <Self as BinarySignal<T>>::add_signal_in_place,
        )
    }

    /// Subtracts `other` from each pipeline element.
    /// Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the pipeline source producer can also fail.
    pub fn subtract(self, other: &SignalView<'_, T>) -> Result<Self> {
        self.ensure_eager()?;
        self.binary(
            other,
            <Self as BinarySignal<T>>::subtract_signal,
            <Self as BinarySignal<T>>::subtract_signal_in_place,
        )
    }

    /// Multiplies the pipeline value by `other` element by element.
    /// Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the pipeline source producer can also fail.
    pub fn multiply(self, other: &SignalView<'_, T>) -> Result<Self> {
        self.ensure_eager()?;
        self.binary(
            other,
            <Self as BinarySignal<T>>::multiply_signal,
            <Self as BinarySignal<T>>::multiply_signal_in_place,
        )
    }

    pub(crate) fn binary(
        mut self,
        other: &SignalView<'_, T>,
        operation: BinaryCallback<T>,
        operation_in_place: fn(
            &StreamContext,
            &SignalView<'_, T>,
            &mut SignalViewMut<'_, T>,
        ) -> Result<()>,
    ) -> Result<Self> {
        self.ensure_eager()?;
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut();
                operation_in_place(self.stream_context, other, &mut signal_view)?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self
                    .workspace
                    .signal_uninit::<T>(self.stream_context.cuda_context(), source.len())?;
                let mut destination_view = destination.output();
                operation(self.stream_context, source, other, &mut destination_view)?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = SignalBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Self: BinarySignal<T>,
{
    /// Adds `left` and `right` element by element into `destination`.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn add_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        <Self as BinarySignal<T>>::add_signal(stream_context, left, right, destination)
    }

    /// Adds `signal` to `destination` element by element, updating `destination` in place.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn add_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as BinarySignal<T>>::add_signal_in_place(stream_context, signal, destination)
    }

    /// Computes the element-wise difference `left - right` into `destination`.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn subtract_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        <Self as BinarySignal<T>>::subtract_signal(stream_context, left, right, destination)
    }

    /// Computes the element-wise difference `destination - signal` in place.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn subtract_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as BinarySignal<T>>::subtract_signal_in_place(stream_context, signal, destination)
    }

    /// Multiplies `left` and `right` element by element into `destination`.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn multiply_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        <Self as BinarySignal<T>>::multiply_signal(stream_context, left, right, destination)
    }

    /// Multiplies `destination` by `signal` element by element, updating `destination` in place.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn multiply_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as BinarySignal<T>>::multiply_signal_in_place(stream_context, signal, destination)
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Self: DivideSignal<T>,
{
    /// Computes the element-wise quotient `left / right` into `destination`.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn divide_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        <Self as DivideSignal<T>>::divide_signal(stream_context, left, right, destination)
    }

    /// Computes the element-wise quotient `destination / signal` in place.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn divide_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as DivideSignal<T>>::divide_signal_in_place(stream_context, signal, destination)
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Self: ScaledBinarySignal<T>,
{
    /// Adds `left` and `right` element by element into `destination`.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn add_scaled_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ScaledBinarySignal<T>>::add_signal_scaled(
            stream_context,
            left,
            right,
            destination,
            scale_factor,
        )
    }

    /// Adds `signal` to `destination` element by element, updating `destination` in place.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn add_scaled_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ScaledBinarySignal<T>>::add_signal_scaled_in_place(
            stream_context,
            signal,
            destination,
            scale_factor,
        )
    }

    /// Computes the element-wise difference `left - right` into `destination`.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn subtract_scaled_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ScaledBinarySignal<T>>::subtract_signal_scaled(
            stream_context,
            left,
            right,
            destination,
            scale_factor,
        )
    }

    /// Computes the element-wise difference `destination - signal` in place.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn subtract_scaled_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ScaledBinarySignal<T>>::subtract_signal_scaled_in_place(
            stream_context,
            signal,
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
    pub fn multiply_scaled_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ScaledBinarySignal<T>>::multiply_signal_scaled(
            stream_context,
            left,
            right,
            destination,
            scale_factor,
        )
    }

    /// Multiplies `destination` by `signal` element by element, updating `destination` in place.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn multiply_scaled_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ScaledBinarySignal<T>>::multiply_signal_scaled_in_place(
            stream_context,
            signal,
            destination,
            scale_factor,
        )
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: AddProductSignal<T> + CopySignal<T>,
{
    /// Adds the element-wise product `left * right` to the pipeline accumulator.
    /// Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal. A borrowed accumulator is copied into initialized workspace storage before accumulation.
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the pipeline source producer can also fail.
    pub fn add_product(
        mut self,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
    ) -> Result<Self> {
        self.ensure_eager()?;
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut();
                <Self as AddProductSignal<T>>::add_product_signal(
                    self.stream_context,
                    left,
                    right,
                    &mut signal_view,
                )?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self
                    .workspace
                    .signal_uninit::<T>(self.stream_context.cuda_context(), source.len())?;
                let mut destination_view = destination.output();
                <Self as CopySignal<T>>::copy(self.stream_context, source, &mut destination_view)?;
                // SAFETY: the completed copy established every destination value.
                let mut destination_view = unsafe { destination_view.assume_init() };
                <Self as AddProductSignal<T>>::add_product_signal(
                    self.stream_context,
                    left,
                    right,
                    &mut destination_view,
                )?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = SignalBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Self: DivideScaledBinarySignal<T>,
{
    /// Computes the element-wise quotient `left / right` into `destination`.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn divide_scaled_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as DivideScaledBinarySignal<T>>::divide_signal_scaled(
            stream_context,
            left,
            right,
            destination,
            scale_factor,
        )
    }

    /// Computes the element-wise quotient `destination / signal` in place.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn divide_scaled_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as DivideScaledBinarySignal<T>>::divide_signal_scaled_in_place(
            stream_context,
            signal,
            destination,
            scale_factor,
        )
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Self: AddProductSignal<T>,
{
    /// Adds the element-wise product `left * right` to the initialized `destination`.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn add_product_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as AddProductSignal<T>>::add_product_signal(stream_context, left, right, destination)
    }
}

impl<U> SignalPipeline<'_, U>
where
    U: Copy,
{
    /// Adds the element-wise product `left * right` to the initialized `destination`.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn add_product_scaled_into<T>(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, U>,
        scale_factor: i32,
    ) -> Result<()>
    where
        T: Copy,
        Self: ScaledAddProductSignal<T, U>,
    {
        <Self as ScaledAddProductSignal<T, U>>::add_product_signal_scaled(
            stream_context,
            left,
            right,
            destination,
            scale_factor,
        )
    }
}

impl<U> SignalPipeline<'_, U>
where
    U: Copy,
    Workspace: SignalAllocator<U>,
    Self: CopySignal<U>,
{
    fn add_product_scaled_with<T>(
        mut self,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        scale_factor: i32,
        operation: AddProductScaledWithCallback<T, U>,
    ) -> Result<Self>
    where
        T: Copy,
        Self: ScaledAddProductSignal<T, U>,
    {
        self.ensure_eager()?;
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut();
                operation(
                    self.stream_context,
                    left,
                    right,
                    &mut signal_view,
                    scale_factor,
                )?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self
                    .workspace
                    .signal_uninit::<U>(self.stream_context.cuda_context(), source.len())?;
                let mut destination_view = destination.output();
                <Self as CopySignal<U>>::copy(self.stream_context, source, &mut destination_view)?;
                // SAFETY: the completed copy initialized the accumulator.
                let mut destination_view = unsafe { destination_view.assume_init() };
                operation(
                    self.stream_context,
                    left,
                    right,
                    &mut destination_view,
                    scale_factor,
                )?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = SignalBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}

impl SignalPipeline<'_, i16> {
    /// Adds the element-wise product `left * right` to the pipeline accumulator.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    /// Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal. A borrowed accumulator is copied into initialized workspace storage before accumulation.
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the pipeline source producer can also fail.
    pub fn add_product_scaled(
        self,
        left: &SignalView<'_, i16>,
        right: &SignalView<'_, i16>,
        scale_factor: i32,
    ) -> Result<Self> {
        self.ensure_eager()?;
        self.add_product_scaled_with(
            left,
            right,
            scale_factor,
            <Self as ScaledAddProductSignal<i16, i16>>::add_product_signal_scaled,
        )
    }
}

impl SignalPipeline<'_, i32> {
    /// Adds the element-wise product `left * right` to the pipeline accumulator.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    /// Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal. A borrowed accumulator is copied into initialized workspace storage before accumulation.
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the pipeline source producer can also fail.
    pub fn add_product_scaled(
        self,
        left: &SignalView<'_, i32>,
        right: &SignalView<'_, i32>,
        scale_factor: i32,
    ) -> Result<Self> {
        self.ensure_eager()?;
        self.add_product_scaled_with(
            left,
            right,
            scale_factor,
            <Self as ScaledAddProductSignal<i32, i32>>::add_product_signal_scaled,
        )
    }

    /// Adds the element-wise product `left * right` to the pipeline accumulator.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    /// Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal. A borrowed accumulator is copied into initialized workspace storage before accumulation.
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the pipeline source producer can also fail.
    pub fn add_product_i16_scaled(
        self,
        left: &SignalView<'_, i16>,
        right: &SignalView<'_, i16>,
        scale_factor: i32,
    ) -> Result<Self> {
        self.ensure_eager()?;
        self.add_product_scaled_with(
            left,
            right,
            scale_factor,
            <Self as ScaledAddProductSignal<i16, i32>>::add_product_signal_scaled,
        )
    }
}

type BinaryCallback<T> = fn(
    &StreamContext,
    &SignalView<'_, T>,
    &SignalView<'_, T>,
    &mut SignalOutputView<'_, T>,
) -> Result<()>;

type AddProductScaledWithCallback<T, U> = fn(
    &StreamContext,
    &SignalView<'_, T>,
    &SignalView<'_, T>,
    &mut SignalViewMut<'_, U>,
    i32,
) -> Result<()>;

use super::*;

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: BinarySignal<T>,
{
    pub fn add_signal(self, other: &SignalView<'_, T>) -> Result<Self> {
        self.binary(
            other,
            <Self as BinarySignal<T>>::add_signal,
            <Self as BinarySignal<T>>::add_signal_in_place,
        )
    }

    pub fn subtract(self, other: &SignalView<'_, T>) -> Result<Self> {
        self.binary(
            other,
            <Self as BinarySignal<T>>::subtract_signal,
            <Self as BinarySignal<T>>::subtract_signal_in_place,
        )
    }

    pub fn multiply(self, other: &SignalView<'_, T>) -> Result<Self> {
        self.binary(
            other,
            <Self as BinarySignal<T>>::multiply_signal,
            <Self as BinarySignal<T>>::multiply_signal_in_place,
        )
    }

    pub(super) fn binary(
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

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Self: BinarySignal<T>,
{
    pub fn add_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as BinarySignal<T>>::add_signal(stream_context, left, right, destination)
    }

    pub fn add_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as BinarySignal<T>>::add_signal_in_place(stream_context, signal, destination)
    }

    pub fn subtract_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as BinarySignal<T>>::subtract_signal(stream_context, left, right, destination)
    }

    pub fn subtract_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as BinarySignal<T>>::subtract_signal_in_place(stream_context, signal, destination)
    }

    pub fn multiply_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as BinarySignal<T>>::multiply_signal(stream_context, left, right, destination)
    }

    pub fn multiply_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as BinarySignal<T>>::multiply_signal_in_place(stream_context, signal, destination)
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Self: DivideSignal<T>,
{
    pub fn divide_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as DivideSignal<T>>::divide_signal(stream_context, left, right, destination)
    }

    pub fn divide_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as DivideSignal<T>>::divide_signal_in_place(stream_context, signal, destination)
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Self: ScaledBinarySignal<T>,
{
    pub fn add_scaled_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
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

    pub fn subtract_scaled_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
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

    pub fn multiply_scaled_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
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

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: AddProductSignal<T> + CopySignal<T>,
{
    pub fn add_product(
        mut self,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
    ) -> Result<Self> {
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut()?;
                <Self as AddProductSignal<T>>::add_product_signal(
                    self.stream_context,
                    left,
                    right,
                    &mut signal_view,
                )?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self.workspace.signal::<T>(source.len())?;
                let mut destination_view = destination.view_mut()?;
                <Self as CopySignal<T>>::copy(self.stream_context, source, &mut destination_view)?;
                <Self as AddProductSignal<T>>::add_product_signal(
                    self.stream_context,
                    left,
                    right,
                    &mut destination_view,
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
    Self: DivideScaledBinarySignal<T>,
{
    pub fn divide_scaled_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
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

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Self: AddProductSignal<T>,
{
    pub fn add_product_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as AddProductSignal<T>>::add_product_signal(stream_context, left, right, destination)
    }
}

impl<'a, U> SignalPipeline<'a, U>
where
    U: Copy,
{
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

impl<'a, U> SignalPipeline<'a, U>
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
        operation: fn(
            &StreamContext,
            &SignalView<'_, T>,
            &SignalView<'_, T>,
            &mut SignalViewMut<'_, U>,
            i32,
        ) -> Result<()>,
    ) -> Result<Self>
    where
        T: Copy,
        Self: ScaledAddProductSignal<T, U>,
    {
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut()?;
                operation(
                    self.stream_context,
                    left,
                    right,
                    &mut signal_view,
                    scale_factor,
                )?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self.workspace.signal::<U>(source.len())?;
                let mut destination_view = destination.view_mut()?;
                <Self as CopySignal<U>>::copy(self.stream_context, source, &mut destination_view)?;
                operation(
                    self.stream_context,
                    left,
                    right,
                    &mut destination_view,
                    scale_factor,
                )?;
                self.backing = SignalBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}

impl<'a> SignalPipeline<'a, i16> {
    pub fn add_product_scaled(
        self,
        left: &SignalView<'_, i16>,
        right: &SignalView<'_, i16>,
        scale_factor: i32,
    ) -> Result<Self> {
        self.add_product_scaled_with(
            left,
            right,
            scale_factor,
            <Self as ScaledAddProductSignal<i16, i16>>::add_product_signal_scaled,
        )
    }
}

impl<'a> SignalPipeline<'a, i32> {
    pub fn add_product_scaled(
        self,
        left: &SignalView<'_, i32>,
        right: &SignalView<'_, i32>,
        scale_factor: i32,
    ) -> Result<Self> {
        self.add_product_scaled_with(
            left,
            right,
            scale_factor,
            <Self as ScaledAddProductSignal<i32, i32>>::add_product_signal_scaled,
        )
    }

    pub fn add_product_i16_scaled(
        self,
        left: &SignalView<'_, i16>,
        right: &SignalView<'_, i16>,
        scale_factor: i32,
    ) -> Result<Self> {
        self.add_product_scaled_with(
            left,
            right,
            scale_factor,
            <Self as ScaledAddProductSignal<i16, i32>>::add_product_signal_scaled,
        )
    }
}

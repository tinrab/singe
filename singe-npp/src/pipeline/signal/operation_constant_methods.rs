use super::*;

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: ConstantSignal<T>,
{
    pub fn add_constant(mut self, value: T) -> Result<Self> {
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut()?;
                <Self as ConstantSignal<T>>::add_constant_signal_in_place(
                    self.stream_context,
                    &mut signal_view,
                    value,
                )?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self.workspace.signal::<T>(source.len())?;
                let mut destination_view = destination.view_mut()?;
                <Self as ConstantSignal<T>>::add_constant_signal(
                    self.stream_context,
                    source,
                    value,
                    &mut destination_view,
                )?;
                self.backing = SignalBacking::Owned(destination);
            }
        }

        Ok(self)
    }

    pub fn subtract_constant(mut self, value: T) -> Result<Self> {
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut()?;
                <Self as ConstantSignal<T>>::subtract_constant_signal_in_place(
                    self.stream_context,
                    &mut signal_view,
                    value,
                )?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self.workspace.signal::<T>(source.len())?;
                let mut destination_view = destination.view_mut()?;
                <Self as ConstantSignal<T>>::subtract_constant_signal(
                    self.stream_context,
                    source,
                    value,
                    &mut destination_view,
                )?;
                self.backing = SignalBacking::Owned(destination);
            }
        }

        Ok(self)
    }

    pub fn multiply_constant(mut self, value: T) -> Result<Self> {
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut()?;
                <Self as ConstantSignal<T>>::multiply_constant_signal_in_place(
                    self.stream_context,
                    &mut signal_view,
                    value,
                )?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self.workspace.signal::<T>(source.len())?;
                let mut destination_view = destination.view_mut()?;
                <Self as ConstantSignal<T>>::multiply_constant_signal(
                    self.stream_context,
                    source,
                    value,
                    &mut destination_view,
                )?;
                self.backing = SignalBacking::Owned(destination);
            }
        }

        Ok(self)
    }

    pub fn divide_constant(mut self, value: T) -> Result<Self> {
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut()?;
                <Self as ConstantSignal<T>>::divide_constant_signal_in_place(
                    self.stream_context,
                    &mut signal_view,
                    value,
                )?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self.workspace.signal::<T>(source.len())?;
                let mut destination_view = destination.view_mut()?;
                <Self as ConstantSignal<T>>::divide_constant_signal(
                    self.stream_context,
                    source,
                    value,
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
    Self: ConstantSignal<T>,
{
    pub fn add_constant_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as ConstantSignal<T>>::add_constant_signal(stream_context, source, value, destination)
    }

    pub fn add_constant_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()> {
        <Self as ConstantSignal<T>>::add_constant_signal_in_place(stream_context, signal, value)
    }

    pub fn subtract_constant_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as ConstantSignal<T>>::subtract_constant_signal(
            stream_context,
            source,
            value,
            destination,
        )
    }

    pub fn subtract_constant_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()> {
        <Self as ConstantSignal<T>>::subtract_constant_signal_in_place(
            stream_context,
            signal,
            value,
        )
    }

    pub fn multiply_constant_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as ConstantSignal<T>>::multiply_constant_signal(
            stream_context,
            source,
            value,
            destination,
        )
    }

    pub fn multiply_constant_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()> {
        <Self as ConstantSignal<T>>::multiply_constant_signal_in_place(
            stream_context,
            signal,
            value,
        )
    }

    pub fn divide_constant_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as ConstantSignal<T>>::divide_constant_signal(
            stream_context,
            source,
            value,
            destination,
        )
    }

    pub fn divide_constant_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()> {
        <Self as ConstantSignal<T>>::divide_constant_signal_in_place(stream_context, signal, value)
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: SubtractFromConstantSignal<T>,
{
    pub fn subtract_from_constant(self, value: T) -> Result<Self> {
        self.reverse_constant_arithmetic(
            value,
            <Self as SubtractFromConstantSignal<T>>::subtract_from_constant_signal,
            <Self as SubtractFromConstantSignal<T>>::subtract_from_constant_signal_in_place,
        )
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: DivideIntoConstantSignal<T>,
{
    pub fn divide_into_constant(self, value: T) -> Result<Self> {
        self.reverse_constant_arithmetic(
            value,
            <Self as DivideIntoConstantSignal<T>>::divide_into_constant_signal,
            <Self as DivideIntoConstantSignal<T>>::divide_into_constant_signal_in_place,
        )
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Self: SubtractFromConstantSignal<T>,
{
    pub fn subtract_from_constant_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as SubtractFromConstantSignal<T>>::subtract_from_constant_signal(
            stream_context,
            source,
            value,
            destination,
        )
    }

    pub fn subtract_from_constant_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()> {
        <Self as SubtractFromConstantSignal<T>>::subtract_from_constant_signal_in_place(
            stream_context,
            signal,
            value,
        )
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Self: DivideIntoConstantSignal<T>,
{
    pub fn divide_into_constant_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as DivideIntoConstantSignal<T>>::divide_into_constant_signal(
            stream_context,
            source,
            value,
            destination,
        )
    }

    pub fn divide_into_constant_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()> {
        <Self as DivideIntoConstantSignal<T>>::divide_into_constant_signal_in_place(
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
{
    fn reverse_constant_arithmetic(
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

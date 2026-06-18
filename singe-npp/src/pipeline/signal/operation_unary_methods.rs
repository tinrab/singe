use super::*;

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: UnarySignal<T>,
{
    pub fn exponent(mut self) -> Result<Self> {
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut()?;
                <Self as UnarySignal<T>>::exponent_signal_in_place(
                    self.stream_context,
                    &mut signal_view,
                )?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self.workspace.signal::<T>(source.len())?;
                let mut destination_view = destination.view_mut()?;
                <Self as UnarySignal<T>>::exponent_signal(
                    self.stream_context,
                    source,
                    &mut destination_view,
                )?;
                self.backing = SignalBacking::Owned(destination);
            }
        }

        Ok(self)
    }

    pub fn natural_logarithm(mut self) -> Result<Self> {
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut()?;
                <Self as UnarySignal<T>>::natural_logarithm_signal_in_place(
                    self.stream_context,
                    &mut signal_view,
                )?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self.workspace.signal::<T>(source.len())?;
                let mut destination_view = destination.view_mut()?;
                <Self as UnarySignal<T>>::natural_logarithm_signal(
                    self.stream_context,
                    source,
                    &mut destination_view,
                )?;
                self.backing = SignalBacking::Owned(destination);
            }
        }

        Ok(self)
    }

    pub fn square(mut self) -> Result<Self> {
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut()?;
                <Self as UnarySignal<T>>::square_signal_in_place(
                    self.stream_context,
                    &mut signal_view,
                )?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self.workspace.signal::<T>(source.len())?;
                let mut destination_view = destination.view_mut()?;
                <Self as UnarySignal<T>>::square_signal(
                    self.stream_context,
                    source,
                    &mut destination_view,
                )?;
                self.backing = SignalBacking::Owned(destination);
            }
        }

        Ok(self)
    }

    pub fn square_root(mut self) -> Result<Self> {
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut()?;
                <Self as UnarySignal<T>>::square_root_signal_in_place(
                    self.stream_context,
                    &mut signal_view,
                )?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self.workspace.signal::<T>(source.len())?;
                let mut destination_view = destination.view_mut()?;
                <Self as UnarySignal<T>>::square_root_signal(
                    self.stream_context,
                    source,
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
    Self: UnarySignal<T>,
{
    pub fn exponent_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as UnarySignal<T>>::exponent_signal(stream_context, source, destination)
    }

    pub fn exponent_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as UnarySignal<T>>::exponent_signal_in_place(stream_context, signal)
    }

    pub fn natural_logarithm_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as UnarySignal<T>>::natural_logarithm_signal(stream_context, source, destination)
    }

    pub fn natural_logarithm_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as UnarySignal<T>>::natural_logarithm_signal_in_place(stream_context, signal)
    }

    pub fn square_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as UnarySignal<T>>::square_signal(stream_context, source, destination)
    }

    pub fn square_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as UnarySignal<T>>::square_signal_in_place(stream_context, signal)
    }

    pub fn square_root_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as UnarySignal<T>>::square_root_signal(stream_context, source, destination)
    }

    pub fn square_root_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as UnarySignal<T>>::square_root_signal_in_place(stream_context, signal)
    }
}

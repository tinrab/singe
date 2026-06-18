use super::*;

type ComparisonThresholdCallback<T, L> = fn(
    &StreamContext,
    &SignalView<'_, T>,
    &mut SignalViewMut<'_, T>,
    L,
    ComparisonOperation,
) -> Result<()>;
type ValueThresholdCallback<T> =
    fn(&StreamContext, &SignalView<'_, T>, &mut SignalViewMut<'_, T>, T, T) -> Result<()>;
type ComplexValueThresholdCallback<T, L> =
    fn(&StreamContext, &SignalView<'_, T>, &mut SignalViewMut<'_, T>, L, T) -> Result<()>;

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: ThresholdSignal<T>,
{
    pub fn threshold(mut self, level: T, operation: ComparisonOperation) -> Result<Self> {
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut()?;
                <Self as ThresholdSignal<T>>::threshold_signal_in_place(
                    self.stream_context,
                    &mut signal_view,
                    level,
                    operation,
                )?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self.workspace.signal::<T>(source.len())?;
                let mut destination_view = destination.view_mut()?;
                <Self as ThresholdSignal<T>>::threshold_signal(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    level,
                    operation,
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
    Self: ThresholdSignal<T>,
{
    pub fn threshold_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        level: T,
        operation: ComparisonOperation,
    ) -> Result<()> {
        <Self as ThresholdSignal<T>>::threshold_signal(
            stream_context,
            source,
            destination,
            level,
            operation,
        )
    }

    pub fn threshold_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: T,
        operation: ComparisonOperation,
    ) -> Result<()> {
        <Self as ThresholdSignal<T>>::threshold_signal_in_place(
            stream_context,
            signal,
            level,
            operation,
        )
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
{
    pub fn threshold_complex<L>(self, level: L, operation: ComparisonOperation) -> Result<Self>
    where
        L: Copy,
        Self: ComplexThresholdSignal<T, L>,
    {
        self.complex_threshold(
            level,
            operation,
            <Self as ComplexThresholdSignal<T, L>>::threshold_complex_signal,
            <Self as ComplexThresholdSignal<T, L>>::threshold_complex_signal_in_place,
        )
    }

    fn complex_threshold<L>(
        mut self,
        level: L,
        operation: ComparisonOperation,
        threshold: ComparisonThresholdCallback<T, L>,
        threshold_in_place: fn(
            &StreamContext,
            &mut SignalViewMut<'_, T>,
            L,
            ComparisonOperation,
        ) -> Result<()>,
    ) -> Result<Self>
    where
        L: Copy,
    {
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut()?;
                threshold_in_place(self.stream_context, &mut signal_view, level, operation)?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self.workspace.signal::<T>(source.len())?;
                let mut destination_view = destination.view_mut()?;
                threshold(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    level,
                    operation,
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
{
    pub fn threshold_complex_into<L>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        level: L,
        operation: ComparisonOperation,
    ) -> Result<()>
    where
        L: Copy,
        Self: ComplexThresholdSignal<T, L>,
    {
        <Self as ComplexThresholdSignal<T, L>>::threshold_complex_signal(
            stream_context,
            source,
            destination,
            level,
            operation,
        )
    }

    pub fn threshold_complex_in_place<L>(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: L,
        operation: ComparisonOperation,
    ) -> Result<()>
    where
        L: Copy,
        Self: ComplexThresholdSignal<T, L>,
    {
        <Self as ComplexThresholdSignal<T, L>>::threshold_complex_signal_in_place(
            stream_context,
            signal,
            level,
            operation,
        )
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: FixedThresholdSignal<T>,
{
    pub fn threshold_less(self, level: T) -> Result<Self> {
        self.fixed_threshold(
            level,
            <Self as FixedThresholdSignal<T>>::threshold_less_signal,
            <Self as FixedThresholdSignal<T>>::threshold_less_signal_in_place,
        )
    }

    pub fn threshold_greater(self, level: T) -> Result<Self> {
        self.fixed_threshold(
            level,
            <Self as FixedThresholdSignal<T>>::threshold_greater_signal,
            <Self as FixedThresholdSignal<T>>::threshold_greater_signal_in_place,
        )
    }

    fn fixed_threshold(
        mut self,
        level: T,
        threshold: fn(
            &StreamContext,
            &SignalView<'_, T>,
            &mut SignalViewMut<'_, T>,
            T,
        ) -> Result<()>,
        threshold_in_place: fn(&StreamContext, &mut SignalViewMut<'_, T>, T) -> Result<()>,
    ) -> Result<Self> {
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut()?;
                threshold_in_place(self.stream_context, &mut signal_view, level)?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self.workspace.signal::<T>(source.len())?;
                let mut destination_view = destination.view_mut()?;
                threshold(self.stream_context, source, &mut destination_view, level)?;
                self.backing = SignalBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Self: FixedThresholdSignal<T>,
{
    pub fn threshold_less_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        level: T,
    ) -> Result<()> {
        <Self as FixedThresholdSignal<T>>::threshold_less_signal(
            stream_context,
            source,
            destination,
            level,
        )
    }

    pub fn threshold_less_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: T,
    ) -> Result<()> {
        <Self as FixedThresholdSignal<T>>::threshold_less_signal_in_place(
            stream_context,
            signal,
            level,
        )
    }

    pub fn threshold_greater_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        level: T,
    ) -> Result<()> {
        <Self as FixedThresholdSignal<T>>::threshold_greater_signal(
            stream_context,
            source,
            destination,
            level,
        )
    }

    pub fn threshold_greater_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: T,
    ) -> Result<()> {
        <Self as FixedThresholdSignal<T>>::threshold_greater_signal_in_place(
            stream_context,
            signal,
            level,
        )
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
{
    pub fn threshold_less_complex<L>(self, level: L) -> Result<Self>
    where
        L: Copy,
        Self: ComplexFixedThresholdSignal<T, L>,
    {
        self.complex_fixed_threshold(
            level,
            <Self as ComplexFixedThresholdSignal<T, L>>::threshold_less_complex_signal,
            <Self as ComplexFixedThresholdSignal<T, L>>::threshold_less_complex_signal_in_place,
        )
    }

    pub fn threshold_greater_complex<L>(self, level: L) -> Result<Self>
    where
        L: Copy,
        Self: ComplexFixedThresholdSignal<T, L>,
    {
        self.complex_fixed_threshold(
            level,
            <Self as ComplexFixedThresholdSignal<T, L>>::threshold_greater_complex_signal,
            <Self as ComplexFixedThresholdSignal<T, L>>::threshold_greater_complex_signal_in_place,
        )
    }

    fn complex_fixed_threshold<L>(
        mut self,
        level: L,
        threshold: fn(
            &StreamContext,
            &SignalView<'_, T>,
            &mut SignalViewMut<'_, T>,
            L,
        ) -> Result<()>,
        threshold_in_place: fn(&StreamContext, &mut SignalViewMut<'_, T>, L) -> Result<()>,
    ) -> Result<Self>
    where
        L: Copy,
    {
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut()?;
                threshold_in_place(self.stream_context, &mut signal_view, level)?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self.workspace.signal::<T>(source.len())?;
                let mut destination_view = destination.view_mut()?;
                threshold(self.stream_context, source, &mut destination_view, level)?;
                self.backing = SignalBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
{
    pub fn threshold_less_complex_into<L>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        level: L,
    ) -> Result<()>
    where
        L: Copy,
        Self: ComplexFixedThresholdSignal<T, L>,
    {
        <Self as ComplexFixedThresholdSignal<T, L>>::threshold_less_complex_signal(
            stream_context,
            source,
            destination,
            level,
        )
    }

    pub fn threshold_less_complex_in_place<L>(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: L,
    ) -> Result<()>
    where
        L: Copy,
        Self: ComplexFixedThresholdSignal<T, L>,
    {
        <Self as ComplexFixedThresholdSignal<T, L>>::threshold_less_complex_signal_in_place(
            stream_context,
            signal,
            level,
        )
    }

    pub fn threshold_greater_complex_into<L>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        level: L,
    ) -> Result<()>
    where
        L: Copy,
        Self: ComplexFixedThresholdSignal<T, L>,
    {
        <Self as ComplexFixedThresholdSignal<T, L>>::threshold_greater_complex_signal(
            stream_context,
            source,
            destination,
            level,
        )
    }

    pub fn threshold_greater_complex_in_place<L>(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: L,
    ) -> Result<()>
    where
        L: Copy,
        Self: ComplexFixedThresholdSignal<T, L>,
    {
        <Self as ComplexFixedThresholdSignal<T, L>>::threshold_greater_complex_signal_in_place(
            stream_context,
            signal,
            level,
        )
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: ValueThresholdSignal<T>,
{
    pub fn threshold_less_value(self, level: T, value: T) -> Result<Self> {
        self.value_threshold(
            level,
            value,
            <Self as ValueThresholdSignal<T>>::threshold_less_value_signal,
            <Self as ValueThresholdSignal<T>>::threshold_less_value_signal_in_place,
        )
    }

    pub fn threshold_greater_value(self, level: T, value: T) -> Result<Self> {
        self.value_threshold(
            level,
            value,
            <Self as ValueThresholdSignal<T>>::threshold_greater_value_signal,
            <Self as ValueThresholdSignal<T>>::threshold_greater_value_signal_in_place,
        )
    }

    fn value_threshold(
        mut self,
        level: T,
        value: T,
        threshold: ValueThresholdCallback<T>,
        threshold_in_place: fn(&StreamContext, &mut SignalViewMut<'_, T>, T, T) -> Result<()>,
    ) -> Result<Self> {
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut()?;
                threshold_in_place(self.stream_context, &mut signal_view, level, value)?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self.workspace.signal::<T>(source.len())?;
                let mut destination_view = destination.view_mut()?;
                threshold(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    level,
                    value,
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
    Self: ValueThresholdSignal<T>,
{
    pub fn threshold_less_value_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        level: T,
        value: T,
    ) -> Result<()> {
        <Self as ValueThresholdSignal<T>>::threshold_less_value_signal(
            stream_context,
            source,
            destination,
            level,
            value,
        )
    }

    pub fn threshold_less_value_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: T,
        value: T,
    ) -> Result<()> {
        <Self as ValueThresholdSignal<T>>::threshold_less_value_signal_in_place(
            stream_context,
            signal,
            level,
            value,
        )
    }

    pub fn threshold_greater_value_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        level: T,
        value: T,
    ) -> Result<()> {
        <Self as ValueThresholdSignal<T>>::threshold_greater_value_signal(
            stream_context,
            source,
            destination,
            level,
            value,
        )
    }

    pub fn threshold_greater_value_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: T,
        value: T,
    ) -> Result<()> {
        <Self as ValueThresholdSignal<T>>::threshold_greater_value_signal_in_place(
            stream_context,
            signal,
            level,
            value,
        )
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
{
    pub fn threshold_less_value_complex<L>(self, level: L, value: T) -> Result<Self>
    where
        L: Copy,
        Self: ComplexValueThresholdSignal<T, L>,
    {
        self.complex_value_threshold(
            level,
            value,
            <Self as ComplexValueThresholdSignal<T, L>>::threshold_less_value_complex_signal,
            <Self as ComplexValueThresholdSignal<T, L>>::threshold_less_value_complex_signal_in_place,
        )
    }

    pub fn threshold_greater_value_complex<L>(self, level: L, value: T) -> Result<Self>
    where
        L: Copy,
        Self: ComplexValueThresholdSignal<T, L>,
    {
        self.complex_value_threshold(
            level,
            value,
            <Self as ComplexValueThresholdSignal<T, L>>::threshold_greater_value_complex_signal,
            <Self as ComplexValueThresholdSignal<T, L>>::threshold_greater_value_complex_signal_in_place,
        )
    }

    fn complex_value_threshold<L>(
        mut self,
        level: L,
        value: T,
        threshold: ComplexValueThresholdCallback<T, L>,
        threshold_in_place: fn(&StreamContext, &mut SignalViewMut<'_, T>, L, T) -> Result<()>,
    ) -> Result<Self>
    where
        L: Copy,
    {
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut()?;
                threshold_in_place(self.stream_context, &mut signal_view, level, value)?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self.workspace.signal::<T>(source.len())?;
                let mut destination_view = destination.view_mut()?;
                threshold(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    level,
                    value,
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
{
    pub fn threshold_less_value_complex_into<L>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        level: L,
        value: T,
    ) -> Result<()>
    where
        L: Copy,
        Self: ComplexValueThresholdSignal<T, L>,
    {
        <Self as ComplexValueThresholdSignal<T, L>>::threshold_less_value_complex_signal(
            stream_context,
            source,
            destination,
            level,
            value,
        )
    }

    pub fn threshold_less_value_complex_in_place<L>(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: L,
        value: T,
    ) -> Result<()>
    where
        L: Copy,
        Self: ComplexValueThresholdSignal<T, L>,
    {
        <Self as ComplexValueThresholdSignal<T, L>>::threshold_less_value_complex_signal_in_place(
            stream_context,
            signal,
            level,
            value,
        )
    }

    pub fn threshold_greater_value_complex_into<L>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        level: L,
        value: T,
    ) -> Result<()>
    where
        L: Copy,
        Self: ComplexValueThresholdSignal<T, L>,
    {
        <Self as ComplexValueThresholdSignal<T, L>>::threshold_greater_value_complex_signal(
            stream_context,
            source,
            destination,
            level,
            value,
        )
    }

    pub fn threshold_greater_value_complex_in_place<L>(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: L,
        value: T,
    ) -> Result<()>
    where
        L: Copy,
        Self: ComplexValueThresholdSignal<T, L>,
    {
        <Self as ComplexValueThresholdSignal<T, L>>::threshold_greater_value_complex_signal_in_place(
            stream_context,
            signal,
            level,
            value,
        )
    }
}

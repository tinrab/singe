use crate::{
    context::StreamContext,
    error::Result,
    pipeline::{SignalAllocator, Workspace, workspace::with_temporary_scratch},
    signal::view::{SignalView, SignalViewMut},
    workspace::ScratchBuffer,
};

use super::statistics::{
    AbsoluteExtremumStatisticSignal, AbsoluteIndexedStatisticSignal, ExtremumStatisticSignal,
    IndexedStatisticSignal, MeanStandardDeviationStatisticSignal, PairStatisticSignal,
    SignalIndexedValue, SignalMeanStandardDeviation, SignalMinMax,
};
use super::{SignalBacking, SignalPipeline};

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: ExtremumStatisticSignal<T>,
{
    pub fn max_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as ExtremumStatisticSignal<T>>::max_buffer_size(stream_context, source)
    }

    pub fn max_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as ExtremumStatisticSignal<T>>::max_with_scratch(
            stream_context,
            source,
            destination,
            scratch,
        )
    }

    pub fn max_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        Self::extremum_statistic_into(
            stream_context,
            source,
            destination,
            <Self as ExtremumStatisticSignal<T>>::max_buffer_size,
            <Self as ExtremumStatisticSignal<T>>::max_with_scratch,
        )
    }

    pub fn max(self) -> Result<Self> {
        self.extremum_statistic(
            <Self as ExtremumStatisticSignal<T>>::max_buffer_size,
            <Self as ExtremumStatisticSignal<T>>::max_with_scratch,
        )
    }

    pub fn min_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as ExtremumStatisticSignal<T>>::min_buffer_size(stream_context, source)
    }

    pub fn min_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as ExtremumStatisticSignal<T>>::min_with_scratch(
            stream_context,
            source,
            destination,
            scratch,
        )
    }

    pub fn min_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        Self::extremum_statistic_into(
            stream_context,
            source,
            destination,
            <Self as ExtremumStatisticSignal<T>>::min_buffer_size,
            <Self as ExtremumStatisticSignal<T>>::min_with_scratch,
        )
    }

    pub fn min(self) -> Result<Self> {
        self.extremum_statistic(
            <Self as ExtremumStatisticSignal<T>>::min_buffer_size,
            <Self as ExtremumStatisticSignal<T>>::min_with_scratch,
        )
    }

    fn extremum_statistic_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: fn(
            &StreamContext,
            &SignalView<'_, T>,
            &mut SignalViewMut<'_, T>,
            &mut ScratchBuffer,
        ) -> Result<()>,
    ) -> Result<()> {
        let required_bytes = buffer_size(stream_context, source)?;
        with_temporary_scratch(stream_context, required_bytes, |scratch| {
            statistic(stream_context, source, destination, scratch)
        })
    }

    fn extremum_statistic(
        self,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: fn(
            &StreamContext,
            &SignalView<'_, T>,
            &mut SignalViewMut<'_, T>,
            &mut ScratchBuffer,
        ) -> Result<()>,
    ) -> Result<Self> {
        let mut destination = self.workspace.signal::<T>(1)?;
        let required_bytes = {
            let source = self.view()?;
            buffer_size(self.stream_context, &source)?
        };
        let mut scratch = self.workspace.scratch(required_bytes)?;

        let operation_result = {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            statistic(
                self.stream_context,
                &source,
                &mut destination_view,
                &mut scratch,
            )
        };

        let recycle_result = self
            .workspace
            .recycle_scratch_after(self.stream_context, scratch);
        operation_result?;
        recycle_result?;

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: SignalBacking::Owned(destination),
        })
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: AbsoluteExtremumStatisticSignal<T>,
{
    pub fn max_absolute_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as AbsoluteExtremumStatisticSignal<T>>::max_absolute_buffer_size(
            stream_context,
            source,
        )
    }

    pub fn max_absolute_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as AbsoluteExtremumStatisticSignal<T>>::max_absolute_with_scratch(
            stream_context,
            source,
            destination,
            scratch,
        )
    }

    pub fn max_absolute_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        Self::absolute_extremum_statistic_into(
            stream_context,
            source,
            destination,
            <Self as AbsoluteExtremumStatisticSignal<T>>::max_absolute_buffer_size,
            <Self as AbsoluteExtremumStatisticSignal<T>>::max_absolute_with_scratch,
        )
    }

    pub fn max_absolute(self) -> Result<Self> {
        self.absolute_extremum_statistic(
            <Self as AbsoluteExtremumStatisticSignal<T>>::max_absolute_buffer_size,
            <Self as AbsoluteExtremumStatisticSignal<T>>::max_absolute_with_scratch,
        )
    }

    pub fn min_absolute_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as AbsoluteExtremumStatisticSignal<T>>::min_absolute_buffer_size(
            stream_context,
            source,
        )
    }

    pub fn min_absolute_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as AbsoluteExtremumStatisticSignal<T>>::min_absolute_with_scratch(
            stream_context,
            source,
            destination,
            scratch,
        )
    }

    pub fn min_absolute_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        Self::absolute_extremum_statistic_into(
            stream_context,
            source,
            destination,
            <Self as AbsoluteExtremumStatisticSignal<T>>::min_absolute_buffer_size,
            <Self as AbsoluteExtremumStatisticSignal<T>>::min_absolute_with_scratch,
        )
    }

    pub fn min_absolute(self) -> Result<Self> {
        self.absolute_extremum_statistic(
            <Self as AbsoluteExtremumStatisticSignal<T>>::min_absolute_buffer_size,
            <Self as AbsoluteExtremumStatisticSignal<T>>::min_absolute_with_scratch,
        )
    }

    fn absolute_extremum_statistic_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: fn(
            &StreamContext,
            &SignalView<'_, T>,
            &mut SignalViewMut<'_, T>,
            &mut ScratchBuffer,
        ) -> Result<()>,
    ) -> Result<()> {
        let required_bytes = buffer_size(stream_context, source)?;
        with_temporary_scratch(stream_context, required_bytes, |scratch| {
            statistic(stream_context, source, destination, scratch)
        })
    }

    fn absolute_extremum_statistic(
        self,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: fn(
            &StreamContext,
            &SignalView<'_, T>,
            &mut SignalViewMut<'_, T>,
            &mut ScratchBuffer,
        ) -> Result<()>,
    ) -> Result<Self> {
        let mut destination = self.workspace.signal::<T>(1)?;
        let required_bytes = {
            let source = self.view()?;
            buffer_size(self.stream_context, &source)?
        };
        let mut scratch = self.workspace.scratch(required_bytes)?;

        let operation_result = {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            statistic(
                self.stream_context,
                &source,
                &mut destination_view,
                &mut scratch,
            )
        };

        let recycle_result = self
            .workspace
            .recycle_scratch_after(self.stream_context, scratch);
        operation_result?;
        recycle_result?;

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: SignalBacking::Owned(destination),
        })
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: PairStatisticSignal<T>,
{
    pub fn min_max_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as PairStatisticSignal<T>>::min_max_buffer_size(stream_context, source)
    }

    pub fn min_max_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        min: &mut SignalViewMut<'_, T>,
        max: &mut SignalViewMut<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as PairStatisticSignal<T>>::min_max_with_scratch(
            stream_context,
            source,
            min,
            max,
            scratch,
        )
    }

    pub fn min_max_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        min: &mut SignalViewMut<'_, T>,
        max: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        let required_bytes =
            <Self as PairStatisticSignal<T>>::min_max_buffer_size(stream_context, source)?;
        with_temporary_scratch(stream_context, required_bytes, |scratch| {
            <Self as PairStatisticSignal<T>>::min_max_with_scratch(
                stream_context,
                source,
                min,
                max,
                scratch,
            )
        })
    }

    pub fn min_max(self) -> Result<SignalMinMax<T>> {
        let mut min = self.workspace.signal::<T>(1)?;
        let mut max = self.workspace.signal::<T>(1)?;
        let required_bytes = {
            let source = self.view()?;
            <Self as PairStatisticSignal<T>>::min_max_buffer_size(self.stream_context, &source)?
        };
        let mut scratch = self.workspace.scratch(required_bytes)?;

        let operation_result = {
            let source = self.view()?;
            let mut min_view = min.view_mut()?;
            let mut max_view = max.view_mut()?;
            <Self as PairStatisticSignal<T>>::min_max_with_scratch(
                self.stream_context,
                &source,
                &mut min_view,
                &mut max_view,
                &mut scratch,
            )
        };

        let recycle_result = self
            .workspace
            .recycle_scratch_after(self.stream_context, scratch);
        operation_result?;
        recycle_result?;

        Ok(SignalMinMax { min, max })
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: MeanStandardDeviationStatisticSignal<T>,
{
    pub fn mean_standard_deviation_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as MeanStandardDeviationStatisticSignal<T>>::mean_standard_deviation_buffer_size(
            stream_context,
            source,
        )
    }

    pub fn mean_standard_deviation_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        mean: &mut SignalViewMut<'_, T>,
        standard_deviation: &mut SignalViewMut<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as MeanStandardDeviationStatisticSignal<T>>::mean_standard_deviation_with_scratch(
            stream_context,
            source,
            mean,
            standard_deviation,
            scratch,
        )
    }

    pub fn mean_standard_deviation_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        mean: &mut SignalViewMut<'_, T>,
        standard_deviation: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        let required_bytes =
            <Self as MeanStandardDeviationStatisticSignal<T>>::mean_standard_deviation_buffer_size(
                stream_context,
                source,
            )?;
        with_temporary_scratch(stream_context, required_bytes, |scratch| {
            <Self as MeanStandardDeviationStatisticSignal<T>>::mean_standard_deviation_with_scratch(
                stream_context,
                source,
                mean,
                standard_deviation,
                scratch,
            )
        })
    }

    pub fn mean_standard_deviation(self) -> Result<SignalMeanStandardDeviation<T>> {
        let mut mean = self.workspace.signal::<T>(1)?;
        let mut standard_deviation = self.workspace.signal::<T>(1)?;
        let required_bytes = {
            let source = self.view()?;
            <Self as MeanStandardDeviationStatisticSignal<T>>::mean_standard_deviation_buffer_size(
                self.stream_context,
                &source,
            )?
        };
        let mut scratch = self.workspace.scratch(required_bytes)?;

        let operation_result = {
            let source = self.view()?;
            let mut mean_view = mean.view_mut()?;
            let mut standard_deviation_view = standard_deviation.view_mut()?;
            <Self as MeanStandardDeviationStatisticSignal<T>>::mean_standard_deviation_with_scratch(
                self.stream_context,
                &source,
                &mut mean_view,
                &mut standard_deviation_view,
                &mut scratch,
            )
        };

        let recycle_result = self
            .workspace
            .recycle_scratch_after(self.stream_context, scratch);
        operation_result?;
        recycle_result?;

        Ok(SignalMeanStandardDeviation {
            mean,
            standard_deviation,
        })
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T> + SignalAllocator<i32>,
    Self: IndexedStatisticSignal<T>,
{
    pub fn max_index_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as IndexedStatisticSignal<T>>::max_index_buffer_size(stream_context, source)
    }

    pub fn max_index_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: &mut SignalViewMut<'_, T>,
        index: &mut SignalViewMut<'_, i32>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as IndexedStatisticSignal<T>>::max_index_with_scratch(
            stream_context,
            source,
            value,
            index,
            scratch,
        )
    }

    pub fn max_index_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: &mut SignalViewMut<'_, T>,
        index: &mut SignalViewMut<'_, i32>,
    ) -> Result<()> {
        Self::indexed_statistic_into(
            stream_context,
            source,
            value,
            index,
            <Self as IndexedStatisticSignal<T>>::max_index_buffer_size,
            <Self as IndexedStatisticSignal<T>>::max_index_with_scratch,
        )
    }

    pub fn max_index(self) -> Result<SignalIndexedValue<T>> {
        self.indexed_statistic(
            <Self as IndexedStatisticSignal<T>>::max_index_buffer_size,
            <Self as IndexedStatisticSignal<T>>::max_index_with_scratch,
        )
    }

    pub fn min_index_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as IndexedStatisticSignal<T>>::min_index_buffer_size(stream_context, source)
    }

    pub fn min_index_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: &mut SignalViewMut<'_, T>,
        index: &mut SignalViewMut<'_, i32>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as IndexedStatisticSignal<T>>::min_index_with_scratch(
            stream_context,
            source,
            value,
            index,
            scratch,
        )
    }

    pub fn min_index_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: &mut SignalViewMut<'_, T>,
        index: &mut SignalViewMut<'_, i32>,
    ) -> Result<()> {
        Self::indexed_statistic_into(
            stream_context,
            source,
            value,
            index,
            <Self as IndexedStatisticSignal<T>>::min_index_buffer_size,
            <Self as IndexedStatisticSignal<T>>::min_index_with_scratch,
        )
    }

    pub fn min_index(self) -> Result<SignalIndexedValue<T>> {
        self.indexed_statistic(
            <Self as IndexedStatisticSignal<T>>::min_index_buffer_size,
            <Self as IndexedStatisticSignal<T>>::min_index_with_scratch,
        )
    }

    fn indexed_statistic_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: &mut SignalViewMut<'_, T>,
        index: &mut SignalViewMut<'_, i32>,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: fn(
            &StreamContext,
            &SignalView<'_, T>,
            &mut SignalViewMut<'_, T>,
            &mut SignalViewMut<'_, i32>,
            &mut ScratchBuffer,
        ) -> Result<()>,
    ) -> Result<()> {
        let required_bytes = buffer_size(stream_context, source)?;
        with_temporary_scratch(stream_context, required_bytes, |scratch| {
            statistic(stream_context, source, value, index, scratch)
        })
    }

    fn indexed_statistic(
        self,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: fn(
            &StreamContext,
            &SignalView<'_, T>,
            &mut SignalViewMut<'_, T>,
            &mut SignalViewMut<'_, i32>,
            &mut ScratchBuffer,
        ) -> Result<()>,
    ) -> Result<SignalIndexedValue<T>> {
        let mut value = self.workspace.signal::<T>(1)?;
        let mut index = self.workspace.signal::<i32>(1)?;
        let required_bytes = {
            let source = self.view()?;
            buffer_size(self.stream_context, &source)?
        };
        let mut scratch = self.workspace.scratch(required_bytes)?;

        let operation_result = {
            let source = self.view()?;
            let mut value_view = value.view_mut()?;
            let mut index_view = index.view_mut()?;
            statistic(
                self.stream_context,
                &source,
                &mut value_view,
                &mut index_view,
                &mut scratch,
            )
        };

        let recycle_result = self
            .workspace
            .recycle_scratch_after(self.stream_context, scratch);
        operation_result?;
        recycle_result?;

        Ok(SignalIndexedValue { value, index })
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T> + SignalAllocator<i32>,
    Self: AbsoluteIndexedStatisticSignal<T>,
{
    pub fn max_absolute_index_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as AbsoluteIndexedStatisticSignal<T>>::max_absolute_index_buffer_size(
            stream_context,
            source,
        )
    }

    pub fn max_absolute_index_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: &mut SignalViewMut<'_, T>,
        index: &mut SignalViewMut<'_, i32>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as AbsoluteIndexedStatisticSignal<T>>::max_absolute_index_with_scratch(
            stream_context,
            source,
            value,
            index,
            scratch,
        )
    }

    pub fn max_absolute_index_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: &mut SignalViewMut<'_, T>,
        index: &mut SignalViewMut<'_, i32>,
    ) -> Result<()> {
        Self::absolute_indexed_statistic_into(
            stream_context,
            source,
            value,
            index,
            <Self as AbsoluteIndexedStatisticSignal<T>>::max_absolute_index_buffer_size,
            <Self as AbsoluteIndexedStatisticSignal<T>>::max_absolute_index_with_scratch,
        )
    }

    pub fn max_absolute_index(self) -> Result<SignalIndexedValue<T>> {
        self.absolute_indexed_statistic(
            <Self as AbsoluteIndexedStatisticSignal<T>>::max_absolute_index_buffer_size,
            <Self as AbsoluteIndexedStatisticSignal<T>>::max_absolute_index_with_scratch,
        )
    }

    pub fn min_absolute_index_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as AbsoluteIndexedStatisticSignal<T>>::min_absolute_index_buffer_size(
            stream_context,
            source,
        )
    }

    pub fn min_absolute_index_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: &mut SignalViewMut<'_, T>,
        index: &mut SignalViewMut<'_, i32>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as AbsoluteIndexedStatisticSignal<T>>::min_absolute_index_with_scratch(
            stream_context,
            source,
            value,
            index,
            scratch,
        )
    }

    pub fn min_absolute_index_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: &mut SignalViewMut<'_, T>,
        index: &mut SignalViewMut<'_, i32>,
    ) -> Result<()> {
        Self::absolute_indexed_statistic_into(
            stream_context,
            source,
            value,
            index,
            <Self as AbsoluteIndexedStatisticSignal<T>>::min_absolute_index_buffer_size,
            <Self as AbsoluteIndexedStatisticSignal<T>>::min_absolute_index_with_scratch,
        )
    }

    pub fn min_absolute_index(self) -> Result<SignalIndexedValue<T>> {
        self.absolute_indexed_statistic(
            <Self as AbsoluteIndexedStatisticSignal<T>>::min_absolute_index_buffer_size,
            <Self as AbsoluteIndexedStatisticSignal<T>>::min_absolute_index_with_scratch,
        )
    }

    fn absolute_indexed_statistic_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: &mut SignalViewMut<'_, T>,
        index: &mut SignalViewMut<'_, i32>,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: fn(
            &StreamContext,
            &SignalView<'_, T>,
            &mut SignalViewMut<'_, T>,
            &mut SignalViewMut<'_, i32>,
            &mut ScratchBuffer,
        ) -> Result<()>,
    ) -> Result<()> {
        let required_bytes = buffer_size(stream_context, source)?;
        with_temporary_scratch(stream_context, required_bytes, |scratch| {
            statistic(stream_context, source, value, index, scratch)
        })
    }

    fn absolute_indexed_statistic(
        self,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: fn(
            &StreamContext,
            &SignalView<'_, T>,
            &mut SignalViewMut<'_, T>,
            &mut SignalViewMut<'_, i32>,
            &mut ScratchBuffer,
        ) -> Result<()>,
    ) -> Result<SignalIndexedValue<T>> {
        let mut value = self.workspace.signal::<T>(1)?;
        let mut index = self.workspace.signal::<i32>(1)?;
        let required_bytes = {
            let source = self.view()?;
            buffer_size(self.stream_context, &source)?
        };
        let mut scratch = self.workspace.scratch(required_bytes)?;

        let operation_result = {
            let source = self.view()?;
            let mut value_view = value.view_mut()?;
            let mut index_view = index.view_mut()?;
            statistic(
                self.stream_context,
                &source,
                &mut value_view,
                &mut index_view,
                &mut scratch,
            )
        };

        let recycle_result = self
            .workspace
            .recycle_scratch_after(self.stream_context, scratch);
        operation_result?;
        recycle_result?;

        Ok(SignalIndexedValue { value, index })
    }
}

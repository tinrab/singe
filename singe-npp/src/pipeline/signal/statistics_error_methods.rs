use crate::{
    context::StreamContext,
    error::Result,
    pipeline::{SignalAllocator, Workspace, workspace::with_temporary_scratch},
    signal::{
        memory::Signal,
        view::{SignalView, SignalViewMut},
    },
    types::ZeroCrossingType,
    workspace::ScratchBuffer,
};

use super::SignalPipeline;
use super::statistics::{ErrorMetricStatisticSignal, ZeroCrossingStatisticSignal};

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<f64>,
    Self: ErrorMetricStatisticSignal<T>,
{
    pub fn maximum_error_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as ErrorMetricStatisticSignal<T>>::maximum_error_buffer_size(stream_context, source)
    }

    pub fn maximum_error_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, f64>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as ErrorMetricStatisticSignal<T>>::maximum_error_with_scratch(
            stream_context,
            source,
            other,
            destination,
            scratch,
        )
    }

    pub fn maximum_error_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, f64>,
    ) -> Result<()> {
        Self::error_metric_statistic_into(
            stream_context,
            source,
            other,
            destination,
            <Self as ErrorMetricStatisticSignal<T>>::maximum_error_buffer_size,
            <Self as ErrorMetricStatisticSignal<T>>::maximum_error_with_scratch,
        )
    }

    pub fn maximum_error(self, other: &SignalView<'_, T>) -> Result<Signal<f64>> {
        self.error_metric_statistic(
            other,
            <Self as ErrorMetricStatisticSignal<T>>::maximum_error_buffer_size,
            <Self as ErrorMetricStatisticSignal<T>>::maximum_error_with_scratch,
        )
    }

    pub fn average_error_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as ErrorMetricStatisticSignal<T>>::average_error_buffer_size(stream_context, source)
    }

    pub fn average_error_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, f64>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as ErrorMetricStatisticSignal<T>>::average_error_with_scratch(
            stream_context,
            source,
            other,
            destination,
            scratch,
        )
    }

    pub fn average_error_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, f64>,
    ) -> Result<()> {
        Self::error_metric_statistic_into(
            stream_context,
            source,
            other,
            destination,
            <Self as ErrorMetricStatisticSignal<T>>::average_error_buffer_size,
            <Self as ErrorMetricStatisticSignal<T>>::average_error_with_scratch,
        )
    }

    pub fn average_error(self, other: &SignalView<'_, T>) -> Result<Signal<f64>> {
        self.error_metric_statistic(
            other,
            <Self as ErrorMetricStatisticSignal<T>>::average_error_buffer_size,
            <Self as ErrorMetricStatisticSignal<T>>::average_error_with_scratch,
        )
    }

    pub fn maximum_relative_error_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as ErrorMetricStatisticSignal<T>>::maximum_relative_error_buffer_size(
            stream_context,
            source,
        )
    }

    pub fn maximum_relative_error_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, f64>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as ErrorMetricStatisticSignal<T>>::maximum_relative_error_with_scratch(
            stream_context,
            source,
            other,
            destination,
            scratch,
        )
    }

    pub fn maximum_relative_error_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, f64>,
    ) -> Result<()> {
        Self::error_metric_statistic_into(
            stream_context,
            source,
            other,
            destination,
            <Self as ErrorMetricStatisticSignal<T>>::maximum_relative_error_buffer_size,
            <Self as ErrorMetricStatisticSignal<T>>::maximum_relative_error_with_scratch,
        )
    }

    pub fn maximum_relative_error(self, other: &SignalView<'_, T>) -> Result<Signal<f64>> {
        self.error_metric_statistic(
            other,
            <Self as ErrorMetricStatisticSignal<T>>::maximum_relative_error_buffer_size,
            <Self as ErrorMetricStatisticSignal<T>>::maximum_relative_error_with_scratch,
        )
    }

    pub fn average_relative_error_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as ErrorMetricStatisticSignal<T>>::average_relative_error_buffer_size(
            stream_context,
            source,
        )
    }

    pub fn average_relative_error_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, f64>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as ErrorMetricStatisticSignal<T>>::average_relative_error_with_scratch(
            stream_context,
            source,
            other,
            destination,
            scratch,
        )
    }

    pub fn average_relative_error_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, f64>,
    ) -> Result<()> {
        Self::error_metric_statistic_into(
            stream_context,
            source,
            other,
            destination,
            <Self as ErrorMetricStatisticSignal<T>>::average_relative_error_buffer_size,
            <Self as ErrorMetricStatisticSignal<T>>::average_relative_error_with_scratch,
        )
    }

    pub fn average_relative_error(self, other: &SignalView<'_, T>) -> Result<Signal<f64>> {
        self.error_metric_statistic(
            other,
            <Self as ErrorMetricStatisticSignal<T>>::average_relative_error_buffer_size,
            <Self as ErrorMetricStatisticSignal<T>>::average_relative_error_with_scratch,
        )
    }

    fn error_metric_statistic_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, f64>,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: fn(
            &StreamContext,
            &SignalView<'_, T>,
            &SignalView<'_, T>,
            &mut SignalViewMut<'_, f64>,
            &mut ScratchBuffer,
        ) -> Result<()>,
    ) -> Result<()> {
        let required_bytes = buffer_size(stream_context, source)?;
        with_temporary_scratch(stream_context, required_bytes, |scratch| {
            statistic(stream_context, source, other, destination, scratch)
        })
    }

    fn error_metric_statistic(
        self,
        other: &SignalView<'_, T>,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: fn(
            &StreamContext,
            &SignalView<'_, T>,
            &SignalView<'_, T>,
            &mut SignalViewMut<'_, f64>,
            &mut ScratchBuffer,
        ) -> Result<()>,
    ) -> Result<Signal<f64>> {
        let mut destination = self.workspace.signal::<f64>(1)?;
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
                other,
                &mut destination_view,
                &mut scratch,
            )
        };

        let recycle_result = self
            .workspace
            .recycle_scratch_after(self.stream_context, scratch);
        operation_result?;
        recycle_result?;

        Ok(destination)
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<f32>,
    Self: ZeroCrossingStatisticSignal<T>,
{
    pub fn zero_crossing_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as ZeroCrossingStatisticSignal<T>>::zero_crossing_buffer_size(stream_context, source)
    }

    pub fn zero_crossing_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, f32>,
        zero_crossing_type: ZeroCrossingType,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as ZeroCrossingStatisticSignal<T>>::zero_crossing_with_scratch(
            stream_context,
            source,
            destination,
            zero_crossing_type,
            scratch,
        )
    }

    pub fn zero_crossing_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, f32>,
        zero_crossing_type: ZeroCrossingType,
    ) -> Result<()> {
        let required_bytes = <Self as ZeroCrossingStatisticSignal<T>>::zero_crossing_buffer_size(
            stream_context,
            source,
        )?;
        with_temporary_scratch(stream_context, required_bytes, |scratch| {
            <Self as ZeroCrossingStatisticSignal<T>>::zero_crossing_with_scratch(
                stream_context,
                source,
                destination,
                zero_crossing_type,
                scratch,
            )
        })
    }

    pub fn zero_crossing(self, zero_crossing_type: ZeroCrossingType) -> Result<Signal<f32>> {
        let mut destination = self.workspace.signal::<f32>(1)?;
        let required_bytes = {
            let source = self.view()?;
            <Self as ZeroCrossingStatisticSignal<T>>::zero_crossing_buffer_size(
                self.stream_context,
                &source,
            )?
        };
        let mut scratch = self.workspace.scratch(required_bytes)?;

        let operation_result = {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as ZeroCrossingStatisticSignal<T>>::zero_crossing_with_scratch(
                self.stream_context,
                &source,
                &mut destination_view,
                zero_crossing_type,
                &mut scratch,
            )
        };

        let recycle_result = self
            .workspace
            .recycle_scratch_after(self.stream_context, scratch);
        operation_result?;
        recycle_result?;

        Ok(destination)
    }
}

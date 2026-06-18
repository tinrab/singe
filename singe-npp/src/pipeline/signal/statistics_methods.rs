use singe_cuda::types::{Complex32, Complex64};

use crate::{
    context::StreamContext,
    error::Result,
    pipeline::{SignalAllocator, Workspace, workspace::with_temporary_scratch},
    signal::{
        statistics,
        view::{SignalView, SignalViewMut},
    },
    types::{ComplexI16, ComplexI32, ComplexI64},
    workspace::ScratchBuffer,
};

use super::statistics::{BinaryScalarStatisticSignal, EveryStatisticSignal, ScalarStatisticSignal};
use super::{CopySignal, SignalBacking, SignalPipeline};

#[macro_use]
#[path = "statistics_scalar_methods.rs"]
mod statistics_scalar_methods;

#[path = "statistics_float_complex_methods.rs"]
mod statistics_float_complex_methods;

#[path = "statistics_integer_methods.rs"]
mod statistics_integer_methods;

impl<'a> SignalPipeline<'a, i32>
where
    Workspace: SignalAllocator<i32>,
{
    pub fn count_in_range_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, i32>,
    ) -> Result<usize> {
        statistics::count_in_range_buffer_size(stream_context, source)
    }

    pub fn count_in_range_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, i32>,
        count: &mut SignalViewMut<'_, i32>,
        lower_bound: i32,
        upper_bound: i32,
    ) -> Result<()> {
        statistics::count_in_range_to_device(
            stream_context,
            source,
            count,
            lower_bound,
            upper_bound,
        )
    }

    pub fn count_in_range_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, i32>,
        count: &mut SignalViewMut<'_, i32>,
        lower_bound: i32,
        upper_bound: i32,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        statistics::count_in_range_to_device_with_scratch(
            stream_context,
            source,
            count,
            lower_bound,
            upper_bound,
            scratch,
        )
    }

    pub fn count_in_range(self, lower_bound: i32, upper_bound: i32) -> Result<Self> {
        let mut destination = self.workspace.signal::<i32>(1)?;
        let required_bytes = {
            let source = self.view()?;
            statistics::count_in_range_buffer_size(self.stream_context, &source)?
        };
        let mut scratch = self.workspace.scratch(required_bytes)?;

        let operation_result = {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            statistics::count_in_range_to_device_with_scratch(
                self.stream_context,
                &source,
                &mut destination_view,
                lower_bound,
                upper_bound,
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
    Self: BinaryScalarStatisticSignal<T>,
{
    pub fn norm_diff_inf_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as BinaryScalarStatisticSignal<T>>::norm_diff_inf_buffer_size(stream_context, source)
    }

    pub fn norm_diff_inf_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as BinaryScalarStatisticSignal<T>>::norm_diff_inf_with_scratch(
            stream_context,
            source,
            other,
            destination,
            scratch,
        )
    }

    pub fn norm_diff_inf_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        Self::binary_scalar_statistic_into(
            stream_context,
            source,
            other,
            destination,
            <Self as BinaryScalarStatisticSignal<T>>::norm_diff_inf_buffer_size,
            <Self as BinaryScalarStatisticSignal<T>>::norm_diff_inf_with_scratch,
        )
    }

    pub fn norm_diff_inf(self, other: &SignalView<'_, T>) -> Result<Self> {
        self.binary_scalar_statistic(
            other,
            <Self as BinaryScalarStatisticSignal<T>>::norm_diff_inf_buffer_size,
            <Self as BinaryScalarStatisticSignal<T>>::norm_diff_inf_with_scratch,
        )
    }

    pub fn norm_diff_l1_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as BinaryScalarStatisticSignal<T>>::norm_diff_l1_buffer_size(stream_context, source)
    }

    pub fn norm_diff_l1_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as BinaryScalarStatisticSignal<T>>::norm_diff_l1_with_scratch(
            stream_context,
            source,
            other,
            destination,
            scratch,
        )
    }

    pub fn norm_diff_l1_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        Self::binary_scalar_statistic_into(
            stream_context,
            source,
            other,
            destination,
            <Self as BinaryScalarStatisticSignal<T>>::norm_diff_l1_buffer_size,
            <Self as BinaryScalarStatisticSignal<T>>::norm_diff_l1_with_scratch,
        )
    }

    pub fn norm_diff_l1(self, other: &SignalView<'_, T>) -> Result<Self> {
        self.binary_scalar_statistic(
            other,
            <Self as BinaryScalarStatisticSignal<T>>::norm_diff_l1_buffer_size,
            <Self as BinaryScalarStatisticSignal<T>>::norm_diff_l1_with_scratch,
        )
    }

    pub fn norm_diff_l2_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as BinaryScalarStatisticSignal<T>>::norm_diff_l2_buffer_size(stream_context, source)
    }

    pub fn norm_diff_l2_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as BinaryScalarStatisticSignal<T>>::norm_diff_l2_with_scratch(
            stream_context,
            source,
            other,
            destination,
            scratch,
        )
    }

    pub fn norm_diff_l2_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        Self::binary_scalar_statistic_into(
            stream_context,
            source,
            other,
            destination,
            <Self as BinaryScalarStatisticSignal<T>>::norm_diff_l2_buffer_size,
            <Self as BinaryScalarStatisticSignal<T>>::norm_diff_l2_with_scratch,
        )
    }

    pub fn norm_diff_l2(self, other: &SignalView<'_, T>) -> Result<Self> {
        self.binary_scalar_statistic(
            other,
            <Self as BinaryScalarStatisticSignal<T>>::norm_diff_l2_buffer_size,
            <Self as BinaryScalarStatisticSignal<T>>::norm_diff_l2_with_scratch,
        )
    }

    pub fn dot_product_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as BinaryScalarStatisticSignal<T>>::dot_product_buffer_size(stream_context, source)
    }

    pub fn dot_product_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as BinaryScalarStatisticSignal<T>>::dot_product_with_scratch(
            stream_context,
            source,
            other,
            destination,
            scratch,
        )
    }

    pub fn dot_product_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        Self::binary_scalar_statistic_into(
            stream_context,
            source,
            other,
            destination,
            <Self as BinaryScalarStatisticSignal<T>>::dot_product_buffer_size,
            <Self as BinaryScalarStatisticSignal<T>>::dot_product_with_scratch,
        )
    }

    pub fn dot_product(self, other: &SignalView<'_, T>) -> Result<Self> {
        self.binary_scalar_statistic(
            other,
            <Self as BinaryScalarStatisticSignal<T>>::dot_product_buffer_size,
            <Self as BinaryScalarStatisticSignal<T>>::dot_product_with_scratch,
        )
    }

    fn binary_scalar_statistic_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: fn(
            &StreamContext,
            &SignalView<'_, T>,
            &SignalView<'_, T>,
            &mut SignalViewMut<'_, T>,
            &mut ScratchBuffer,
        ) -> Result<()>,
    ) -> Result<()> {
        let required_bytes = buffer_size(stream_context, source)?;
        with_temporary_scratch(stream_context, required_bytes, |scratch| {
            statistic(stream_context, source, other, destination, scratch)
        })
    }

    fn binary_scalar_statistic(
        self,
        other: &SignalView<'_, T>,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: fn(
            &StreamContext,
            &SignalView<'_, T>,
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

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: SignalBacking::Owned(destination),
        })
    }
}

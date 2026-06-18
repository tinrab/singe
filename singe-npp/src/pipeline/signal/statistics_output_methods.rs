use crate::{
    context::StreamContext,
    error::Result,
    pipeline::{SignalAllocator, Workspace},
    signal::{
        statistics,
        view::{SignalView, SignalViewMut},
    },
    types::DataTypeLike,
    workspace::ScratchBuffer,
};

use super::{SignalBacking, SignalPipeline};

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
{
    pub fn norm_inf_to<U>(self) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::NormInfTo<U>,
        Workspace: SignalAllocator<U>,
    {
        self.statistic_output(
            statistics::norm_inf_buffer_size::<T, U>,
            statistics::norm_inf_to_device_with_scratch::<T, U>,
        )
    }

    pub fn norm_l1_to<U>(self) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::NormL1To<U>,
        Workspace: SignalAllocator<U>,
    {
        self.statistic_output(
            statistics::norm_l1_buffer_size::<T, U>,
            statistics::norm_l1_to_device_with_scratch::<T, U>,
        )
    }

    pub fn norm_l2_to<U>(self) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::NormL2To<U>,
        Workspace: SignalAllocator<U>,
    {
        self.statistic_output(
            statistics::norm_l2_buffer_size::<T, U>,
            statistics::norm_l2_to_device_with_scratch::<T, U>,
        )
    }

    pub fn sum_to_scaled<U>(self, scale_factor: i32) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::SumScaledTo<U>,
        Workspace: SignalAllocator<U>,
    {
        self.scaled_statistic_output(
            scale_factor,
            statistics::sum_scaled_buffer_size::<T, U>,
            statistics::sum_scaled_to_device_with_scratch::<T, U>,
        )
    }

    pub fn sum_to_complex_scaled<U>(self, scale_factor: i32) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::SumScaledTo<U>,
        Workspace: SignalAllocator<U>,
    {
        self.sum_to_scaled::<U>(scale_factor)
    }

    pub fn standard_deviation_to_scaled<U>(self, scale_factor: i32) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::StandardDeviationScaledTo<U>,
        Workspace: SignalAllocator<U>,
    {
        self.scaled_statistic_output(
            scale_factor,
            statistics::standard_deviation_scaled_buffer_size::<T, U>,
            statistics::standard_deviation_scaled_to_device_with_scratch::<T, U>,
        )
    }

    pub fn norm_inf_to_scaled<U>(self, scale_factor: i32) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::NormInfScaledTo<U>,
        Workspace: SignalAllocator<U>,
    {
        self.scaled_statistic_output(
            scale_factor,
            statistics::norm_inf_scaled_buffer_size::<T, U>,
            statistics::norm_inf_scaled_to_device_with_scratch::<T, U>,
        )
    }

    pub fn norm_l1_to_scaled<U>(self, scale_factor: i32) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::NormL1ScaledTo<U>,
        Workspace: SignalAllocator<U>,
    {
        self.scaled_statistic_output(
            scale_factor,
            statistics::norm_l1_scaled_buffer_size::<T, U>,
            statistics::norm_l1_scaled_to_device_with_scratch::<T, U>,
        )
    }

    pub fn norm_l2_to_scaled<U>(self, scale_factor: i32) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::NormL2ScaledTo<U>,
        Workspace: SignalAllocator<U>,
    {
        self.scaled_statistic_output(
            scale_factor,
            statistics::norm_l2_scaled_buffer_size::<T, U>,
            statistics::norm_l2_scaled_to_device_with_scratch::<T, U>,
        )
    }

    pub fn norm_l2_squared_to_scaled<U>(self, scale_factor: i32) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::NormL2SquaredScaledTo<U>,
        Workspace: SignalAllocator<U>,
    {
        self.scaled_statistic_output(
            scale_factor,
            statistics::norm_l2_squared_scaled_buffer_size::<T, U>,
            statistics::norm_l2_squared_scaled_to_device_with_scratch::<T, U>,
        )
    }

    pub(super) fn statistic_output<U>(
        self,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: fn(
            &StreamContext,
            &SignalView<'_, T>,
            &mut SignalViewMut<'_, U>,
            &mut ScratchBuffer,
        ) -> Result<()>,
    ) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy,
        Workspace: SignalAllocator<U>,
    {
        let mut destination = self.workspace.signal::<U>(1)?;
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

        Ok(SignalPipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: SignalBacking::Owned(destination),
        })
    }

    pub(super) fn scaled_statistic_output<U>(
        self,
        scale_factor: i32,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: fn(
            &StreamContext,
            &SignalView<'_, T>,
            &mut SignalViewMut<'_, U>,
            i32,
            &mut ScratchBuffer,
        ) -> Result<()>,
    ) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy,
        Workspace: SignalAllocator<U>,
    {
        let mut destination = self.workspace.signal::<U>(1)?;
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
                scale_factor,
                &mut scratch,
            )
        };

        let recycle_result = self
            .workspace
            .recycle_scratch_after(self.stream_context, scratch);
        operation_result?;
        recycle_result?;

        Ok(SignalPipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: SignalBacking::Owned(destination),
        })
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
{
    pub fn norm_diff_inf_to<U>(self, other: &SignalView<'_, T>) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::NormDiffInfTo<U>,
        Workspace: SignalAllocator<U>,
    {
        self.binary_statistic_output(
            other,
            statistics::norm_diff_inf_buffer_size::<T, U>,
            statistics::norm_diff_inf_to_device_with_scratch::<T, U>,
        )
    }

    pub fn norm_diff_l1_to<U>(self, other: &SignalView<'_, T>) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::NormDiffL1To<U>,
        Workspace: SignalAllocator<U>,
    {
        self.binary_statistic_output(
            other,
            statistics::norm_diff_l1_buffer_size::<T, U>,
            statistics::norm_diff_l1_to_device_with_scratch::<T, U>,
        )
    }

    pub fn norm_diff_l2_to<U>(self, other: &SignalView<'_, T>) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::NormDiffL2To<U>,
        Workspace: SignalAllocator<U>,
    {
        self.binary_statistic_output(
            other,
            statistics::norm_diff_l2_buffer_size::<T, U>,
            statistics::norm_diff_l2_to_device_with_scratch::<T, U>,
        )
    }

    pub fn dot_product_to<U>(self, other: &SignalView<'_, T>) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::DotProductTo<U>,
        Workspace: SignalAllocator<U>,
    {
        self.binary_statistic_output(
            other,
            statistics::dot_product_buffer_size::<T, U>,
            statistics::dot_product_to_device_with_scratch::<T, U>,
        )
    }

    pub fn dot_product_to_buffer_size<U>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>
    where
        U: Copy + DataTypeLike,
        T: statistics::DotProductTo<U>,
    {
        statistics::dot_product_buffer_size::<T, U>(stream_context, source)
    }

    pub fn dot_product_to_with_scratch_into<U>(
        stream_context: &StreamContext,
        source1: &SignalView<'_, T>,
        source2: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, U>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>
    where
        U: Copy + DataTypeLike,
        T: statistics::DotProductTo<U>,
    {
        statistics::dot_product_to_device_with_scratch::<T, U>(
            stream_context,
            source1,
            source2,
            destination,
            scratch,
        )
    }

    pub fn dot_product_to_complex<U>(
        self,
        other: &SignalView<'_, T>,
    ) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::DotProductTo<U>,
        Workspace: SignalAllocator<U>,
    {
        self.dot_product_to::<U>(other)
    }

    pub fn norm_diff_inf_to_scaled<U>(
        self,
        other: &SignalView<'_, T>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::NormDiffInfScaledTo<U>,
        Workspace: SignalAllocator<U>,
    {
        self.scaled_binary_statistic_output(
            other,
            scale_factor,
            statistics::norm_diff_inf_scaled_buffer_size::<T, U>,
            statistics::norm_diff_inf_scaled_to_device_with_scratch::<T, U>,
        )
    }

    pub fn norm_diff_l1_to_scaled<U>(
        self,
        other: &SignalView<'_, T>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::NormDiffL1ScaledTo<U>,
        Workspace: SignalAllocator<U>,
    {
        self.scaled_binary_statistic_output(
            other,
            scale_factor,
            statistics::norm_diff_l1_scaled_buffer_size::<T, U>,
            statistics::norm_diff_l1_scaled_to_device_with_scratch::<T, U>,
        )
    }

    pub fn norm_diff_l2_to_scaled<U>(
        self,
        other: &SignalView<'_, T>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::NormDiffL2ScaledTo<U>,
        Workspace: SignalAllocator<U>,
    {
        self.scaled_binary_statistic_output(
            other,
            scale_factor,
            statistics::norm_diff_l2_scaled_buffer_size::<T, U>,
            statistics::norm_diff_l2_scaled_to_device_with_scratch::<T, U>,
        )
    }

    pub fn norm_diff_l2_squared_to_scaled<U>(
        self,
        other: &SignalView<'_, T>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::NormDiffL2SquaredScaledTo<U>,
        Workspace: SignalAllocator<U>,
    {
        self.scaled_binary_statistic_output(
            other,
            scale_factor,
            statistics::norm_diff_l2_squared_scaled_buffer_size::<T, U>,
            statistics::norm_diff_l2_squared_scaled_to_device_with_scratch::<T, U>,
        )
    }

    pub fn dot_product_to_scaled<U>(
        self,
        other: &SignalView<'_, T>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::DotProductScaledTo<U>,
        Workspace: SignalAllocator<U>,
    {
        self.scaled_binary_statistic_output(
            other,
            scale_factor,
            statistics::dot_product_scaled_buffer_size::<T, U>,
            statistics::dot_product_scaled_to_device_with_scratch::<T, U>,
        )
    }

    pub fn dot_product_to_complex_scaled<U>(
        self,
        other: &SignalView<'_, T>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::DotProductScaledTo<U>,
        Workspace: SignalAllocator<U>,
    {
        self.dot_product_to_scaled::<U>(other, scale_factor)
    }

    pub(super) fn binary_statistic_output<U>(
        self,
        other: &SignalView<'_, T>,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: fn(
            &StreamContext,
            &SignalView<'_, T>,
            &SignalView<'_, T>,
            &mut SignalViewMut<'_, U>,
            &mut ScratchBuffer,
        ) -> Result<()>,
    ) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy,
        Workspace: SignalAllocator<U>,
    {
        let mut destination = self.workspace.signal::<U>(1)?;
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

        Ok(SignalPipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: SignalBacking::Owned(destination),
        })
    }

    pub(super) fn heterogeneous_binary_statistic_output<U, V>(
        self,
        other: &SignalView<'_, U>,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: fn(
            &StreamContext,
            &SignalView<'_, T>,
            &SignalView<'_, U>,
            &mut SignalViewMut<'_, V>,
            &mut ScratchBuffer,
        ) -> Result<()>,
    ) -> Result<SignalPipeline<'a, V>>
    where
        U: Copy,
        V: Copy,
        Workspace: SignalAllocator<V>,
    {
        let mut destination = self.workspace.signal::<V>(1)?;
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

        Ok(SignalPipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: SignalBacking::Owned(destination),
        })
    }

    pub(super) fn scaled_binary_statistic_output<U>(
        self,
        other: &SignalView<'_, T>,
        scale_factor: i32,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: fn(
            &StreamContext,
            &SignalView<'_, T>,
            &SignalView<'_, T>,
            &mut SignalViewMut<'_, U>,
            i32,
            &mut ScratchBuffer,
        ) -> Result<()>,
    ) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy,
        Workspace: SignalAllocator<U>,
    {
        let mut destination = self.workspace.signal::<U>(1)?;
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
                scale_factor,
                &mut scratch,
            )
        };

        let recycle_result = self
            .workspace
            .recycle_scratch_after(self.stream_context, scratch);
        operation_result?;
        recycle_result?;

        Ok(SignalPipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: SignalBacking::Owned(destination),
        })
    }

    pub(super) fn scaled_heterogeneous_binary_statistic_output<U, V>(
        self,
        other: &SignalView<'_, U>,
        scale_factor: i32,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: fn(
            &StreamContext,
            &SignalView<'_, T>,
            &SignalView<'_, U>,
            &mut SignalViewMut<'_, V>,
            i32,
            &mut ScratchBuffer,
        ) -> Result<()>,
    ) -> Result<SignalPipeline<'a, V>>
    where
        U: Copy,
        V: Copy,
        Workspace: SignalAllocator<V>,
    {
        let mut destination = self.workspace.signal::<V>(1)?;
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
                scale_factor,
                &mut scratch,
            )
        };

        let recycle_result = self
            .workspace
            .recycle_scratch_after(self.stream_context, scratch);
        operation_result?;
        recycle_result?;

        Ok(SignalPipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: SignalBacking::Owned(destination),
        })
    }
}

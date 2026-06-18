use super::*;

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: ScalarStatisticSignal<T>,
{
    pub fn sum_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as ScalarStatisticSignal<T>>::sum_buffer_size(stream_context, source)
    }

    pub fn sum_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as ScalarStatisticSignal<T>>::sum_with_scratch(
            stream_context,
            source,
            destination,
            scratch,
        )
    }

    pub fn sum_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        Self::scalar_statistic_into(
            stream_context,
            source,
            destination,
            <Self as ScalarStatisticSignal<T>>::sum_buffer_size,
            <Self as ScalarStatisticSignal<T>>::sum_with_scratch,
        )
    }

    pub fn sum(self) -> Result<Self> {
        self.scalar_statistic(
            <Self as ScalarStatisticSignal<T>>::sum_buffer_size,
            <Self as ScalarStatisticSignal<T>>::sum_with_scratch,
        )
    }

    pub fn sum_natural_logarithm_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as ScalarStatisticSignal<T>>::sum_natural_logarithm_buffer_size(
            stream_context,
            source,
        )
    }

    pub fn sum_natural_logarithm_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as ScalarStatisticSignal<T>>::sum_natural_logarithm_with_scratch(
            stream_context,
            source,
            destination,
            scratch,
        )
    }

    pub fn sum_natural_logarithm_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        Self::scalar_statistic_into(
            stream_context,
            source,
            destination,
            <Self as ScalarStatisticSignal<T>>::sum_natural_logarithm_buffer_size,
            <Self as ScalarStatisticSignal<T>>::sum_natural_logarithm_with_scratch,
        )
    }

    pub fn sum_natural_logarithm(self) -> Result<Self> {
        self.scalar_statistic(
            <Self as ScalarStatisticSignal<T>>::sum_natural_logarithm_buffer_size,
            <Self as ScalarStatisticSignal<T>>::sum_natural_logarithm_with_scratch,
        )
    }

    pub fn mean_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as ScalarStatisticSignal<T>>::mean_buffer_size(stream_context, source)
    }

    pub fn mean_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as ScalarStatisticSignal<T>>::mean_with_scratch(
            stream_context,
            source,
            destination,
            scratch,
        )
    }

    pub fn mean_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        Self::scalar_statistic_into(
            stream_context,
            source,
            destination,
            <Self as ScalarStatisticSignal<T>>::mean_buffer_size,
            <Self as ScalarStatisticSignal<T>>::mean_with_scratch,
        )
    }

    pub fn mean(self) -> Result<Self> {
        self.scalar_statistic(
            <Self as ScalarStatisticSignal<T>>::mean_buffer_size,
            <Self as ScalarStatisticSignal<T>>::mean_with_scratch,
        )
    }

    pub fn standard_deviation_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as ScalarStatisticSignal<T>>::standard_deviation_buffer_size(stream_context, source)
    }

    pub fn standard_deviation_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as ScalarStatisticSignal<T>>::standard_deviation_with_scratch(
            stream_context,
            source,
            destination,
            scratch,
        )
    }

    pub fn standard_deviation_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        Self::scalar_statistic_into(
            stream_context,
            source,
            destination,
            <Self as ScalarStatisticSignal<T>>::standard_deviation_buffer_size,
            <Self as ScalarStatisticSignal<T>>::standard_deviation_with_scratch,
        )
    }

    pub fn standard_deviation(self) -> Result<Self> {
        self.scalar_statistic(
            <Self as ScalarStatisticSignal<T>>::standard_deviation_buffer_size,
            <Self as ScalarStatisticSignal<T>>::standard_deviation_with_scratch,
        )
    }

    pub fn norm_inf_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as ScalarStatisticSignal<T>>::norm_inf_buffer_size(stream_context, source)
    }

    pub fn norm_inf_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as ScalarStatisticSignal<T>>::norm_inf_with_scratch(
            stream_context,
            source,
            destination,
            scratch,
        )
    }

    pub fn norm_inf_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        Self::scalar_statistic_into(
            stream_context,
            source,
            destination,
            <Self as ScalarStatisticSignal<T>>::norm_inf_buffer_size,
            <Self as ScalarStatisticSignal<T>>::norm_inf_with_scratch,
        )
    }

    pub fn norm_inf(self) -> Result<Self> {
        self.scalar_statistic(
            <Self as ScalarStatisticSignal<T>>::norm_inf_buffer_size,
            <Self as ScalarStatisticSignal<T>>::norm_inf_with_scratch,
        )
    }

    pub fn norm_l1_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as ScalarStatisticSignal<T>>::norm_l1_buffer_size(stream_context, source)
    }

    pub fn norm_l1_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as ScalarStatisticSignal<T>>::norm_l1_with_scratch(
            stream_context,
            source,
            destination,
            scratch,
        )
    }

    pub fn norm_l1_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        Self::scalar_statistic_into(
            stream_context,
            source,
            destination,
            <Self as ScalarStatisticSignal<T>>::norm_l1_buffer_size,
            <Self as ScalarStatisticSignal<T>>::norm_l1_with_scratch,
        )
    }

    pub fn norm_l1(self) -> Result<Self> {
        self.scalar_statistic(
            <Self as ScalarStatisticSignal<T>>::norm_l1_buffer_size,
            <Self as ScalarStatisticSignal<T>>::norm_l1_with_scratch,
        )
    }

    pub fn norm_l2_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as ScalarStatisticSignal<T>>::norm_l2_buffer_size(stream_context, source)
    }

    pub fn norm_l2_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as ScalarStatisticSignal<T>>::norm_l2_with_scratch(
            stream_context,
            source,
            destination,
            scratch,
        )
    }

    pub fn norm_l2_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        Self::scalar_statistic_into(
            stream_context,
            source,
            destination,
            <Self as ScalarStatisticSignal<T>>::norm_l2_buffer_size,
            <Self as ScalarStatisticSignal<T>>::norm_l2_with_scratch,
        )
    }

    pub fn norm_l2(self) -> Result<Self> {
        self.scalar_statistic(
            <Self as ScalarStatisticSignal<T>>::norm_l2_buffer_size,
            <Self as ScalarStatisticSignal<T>>::norm_l2_with_scratch,
        )
    }

    fn scalar_statistic_into(
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

    fn scalar_statistic(
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

macro_rules! impl_statistic_output_pipeline_methods {
    (
        $source_ty:ty,
        $destination_ty:ty,
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $npp_buffer_size:path,
        $npp_with_scratch:path
    ) => {
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_ty>,
        ) -> Result<usize> {
            $npp_buffer_size(stream_context, source)
        }

        pub fn $with_scratch_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_ty>,
            destination: &mut SignalViewMut<'_, $destination_ty>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            $npp_with_scratch(stream_context, source, destination, scratch)
        }
    };
}

macro_rules! impl_binary_statistic_output_pipeline_methods {
    (
        $source_ty:ty,
        $destination_ty:ty,
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $npp_buffer_size:path,
        $npp_with_scratch:path
    ) => {
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_ty>,
        ) -> Result<usize> {
            $npp_buffer_size(stream_context, source)
        }

        pub fn $with_scratch_name(
            stream_context: &StreamContext,
            source1: &SignalView<'_, $source_ty>,
            source2: &SignalView<'_, $source_ty>,
            destination: &mut SignalViewMut<'_, $destination_ty>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            $npp_with_scratch(stream_context, source1, source2, destination, scratch)
        }
    };
}

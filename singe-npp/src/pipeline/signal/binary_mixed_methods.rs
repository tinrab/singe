use super::*;

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
{
    fn mixed_binary<U>(
        self,
        other: &SignalView<'_, T>,
        operation: fn(
            &StreamContext,
            &SignalView<'_, T>,
            &SignalView<'_, T>,
            &mut SignalViewMut<'_, U>,
        ) -> Result<()>,
    ) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy,
        Workspace: SignalAllocator<U>,
        Self: MixedBinarySignal<T, U>,
    {
        let mut destination = self.workspace.signal::<U>(self.len())?;
        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            operation(self.stream_context, &source, other, &mut destination_view)?;
        }

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
    pub fn multiply_low_scaled_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()>
    where
        Self: MultiplyLowScaledSignal<T>,
    {
        <Self as MultiplyLowScaledSignal<T>>::multiply_low_signal_scaled(
            stream_context,
            left,
            right,
            destination,
            scale_factor,
        )
    }

    pub fn divide_by_scaled_into<U, V>(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, U>,
        destination: &mut SignalViewMut<'_, V>,
        scale_factor: i32,
    ) -> Result<()>
    where
        U: Copy,
        V: Copy,
        Self: DivideByScaledSignal<T, U, V>,
    {
        <Self as DivideByScaledSignal<T, U, V>>::divide_by_signal_scaled(
            stream_context,
            left,
            right,
            destination,
            scale_factor,
        )
    }

    pub fn multiply_complex_by_scaled_in_place<U>(
        stream_context: &StreamContext,
        source: &SignalView<'_, U>,
        destination: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()>
    where
        U: Copy,
        Self: ScaledHeterogeneousInPlaceSignal<T, U>,
    {
        <Self as ScaledHeterogeneousInPlaceSignal<T, U>>::multiply_heterogeneous_signal_scaled_in_place(
            stream_context,
            source,
            destination,
            scale_factor,
        )
    }

    pub fn add_to_in_place<U>(
        stream_context: &StreamContext,
        source: &SignalView<'_, U>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>
    where
        U: Copy,
        Self: AddHeterogeneousInPlaceSignal<T, U>,
    {
        <Self as AddHeterogeneousInPlaceSignal<T, U>>::add_heterogeneous_signal_in_place(
            stream_context,
            source,
            destination,
        )
    }

    pub fn multiply_with_complex_into<U, V>(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, U>,
        destination: &mut SignalViewMut<'_, V>,
    ) -> Result<()>
    where
        U: Copy,
        V: Copy,
        Self: HeterogeneousBinarySignal<T, U, V>,
    {
        <Self as HeterogeneousBinarySignal<T, U, V>>::multiply_heterogeneous_signal(
            stream_context,
            left,
            right,
            destination,
        )
    }

    pub fn multiply_complex_by_in_place<U>(
        stream_context: &StreamContext,
        source: &SignalView<'_, U>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>
    where
        U: Copy,
        Self: MultiplyHeterogeneousInPlaceSignal<T, U>,
    {
        <Self as MultiplyHeterogeneousInPlaceSignal<T, U>>::multiply_heterogeneous_signal_in_place(
            stream_context,
            source,
            destination,
        )
    }

    pub fn add_to<U>(self, other: &SignalView<'_, T>) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy,
        Workspace: SignalAllocator<U>,
        Self: MixedBinarySignal<T, U>,
    {
        self.mixed_binary(other, <Self as MixedBinarySignal<T, U>>::add_mixed_signal)
    }

    pub fn subtract_to<U>(self, other: &SignalView<'_, T>) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy,
        Workspace: SignalAllocator<U>,
        Self: MixedBinarySignal<T, U>,
    {
        self.mixed_binary(
            other,
            <Self as MixedBinarySignal<T, U>>::subtract_mixed_signal,
        )
    }

    pub fn multiply_to<U>(self, other: &SignalView<'_, T>) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy,
        Workspace: SignalAllocator<U>,
        Self: MixedBinarySignal<T, U>,
    {
        self.mixed_binary(
            other,
            <Self as MixedBinarySignal<T, U>>::multiply_mixed_signal,
        )
    }

    pub fn add_mixed_into<U>(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, U>,
    ) -> Result<()>
    where
        U: Copy,
        Self: MixedBinarySignal<T, U>,
    {
        <Self as MixedBinarySignal<T, U>>::add_mixed_signal(
            stream_context,
            left,
            right,
            destination,
        )
    }

    pub fn subtract_mixed_into<U>(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, U>,
    ) -> Result<()>
    where
        U: Copy,
        Self: MixedBinarySignal<T, U>,
    {
        <Self as MixedBinarySignal<T, U>>::subtract_mixed_signal(
            stream_context,
            left,
            right,
            destination,
        )
    }

    pub fn multiply_mixed_into<U>(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, U>,
    ) -> Result<()>
    where
        U: Copy,
        Self: MixedBinarySignal<T, U>,
    {
        <Self as MixedBinarySignal<T, U>>::multiply_mixed_signal(
            stream_context,
            left,
            right,
            destination,
        )
    }

    pub fn multiply_mixed_scaled_into<U>(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, U>,
        scale_factor: i32,
    ) -> Result<()>
    where
        U: Copy,
        Self: ScaledMixedBinarySignal<T, U>,
    {
        <Self as ScaledMixedBinarySignal<T, U>>::multiply_mixed_signal_scaled(
            stream_context,
            left,
            right,
            destination,
            scale_factor,
        )
    }

    pub fn multiply_heterogeneous_scaled_into<U, V>(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, U>,
        destination: &mut SignalViewMut<'_, V>,
        scale_factor: i32,
    ) -> Result<()>
    where
        U: Copy,
        V: Copy,
        Self: ScaledHeterogeneousBinarySignal<T, U, V>,
    {
        <Self as ScaledHeterogeneousBinarySignal<T, U, V>>::multiply_heterogeneous_signal_scaled(
            stream_context,
            left,
            right,
            destination,
            scale_factor,
        )
    }
}

impl<'a> SignalPipeline<'a, u8> {
    pub fn add_to_u16(self, other: &SignalView<'_, u8>) -> Result<SignalPipeline<'a, u16>> {
        self.mixed_binary(
            other,
            <Self as MixedBinarySignal<u8, u16>>::add_mixed_signal,
        )
    }

    pub fn multiply_to_u16(self, other: &SignalView<'_, u8>) -> Result<SignalPipeline<'a, u16>> {
        self.mixed_binary(
            other,
            <Self as MixedBinarySignal<u8, u16>>::multiply_mixed_signal,
        )
    }
}

impl<'a> SignalPipeline<'a, i16> {
    pub fn add_to_f32(self, other: &SignalView<'_, i16>) -> Result<SignalPipeline<'a, f32>> {
        self.mixed_binary(
            other,
            <Self as MixedBinarySignal<i16, f32>>::add_mixed_signal,
        )
    }

    pub fn subtract_to_f32(self, other: &SignalView<'_, i16>) -> Result<SignalPipeline<'a, f32>> {
        self.mixed_binary(
            other,
            <Self as MixedBinarySignal<i16, f32>>::subtract_mixed_signal,
        )
    }

    pub fn multiply_to_f32(self, other: &SignalView<'_, i16>) -> Result<SignalPipeline<'a, f32>> {
        self.mixed_binary(
            other,
            <Self as MixedBinarySignal<i16, f32>>::multiply_mixed_signal,
        )
    }

    pub fn multiply_to_i32_scaled(
        self,
        other: &SignalView<'_, i16>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i32>> {
        self.scaled_mixed_binary(
            other,
            scale_factor,
            <Self as ScaledMixedBinarySignal<i16, i32>>::multiply_mixed_signal_scaled,
        )
    }
}

impl<'a> SignalPipeline<'a, u16> {
    pub fn multiply_with_i16_scaled(
        self,
        other: &SignalView<'_, i16>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i16>> {
        self.scaled_heterogeneous_binary(
            other,
            scale_factor,
            <Self as ScaledHeterogeneousBinarySignal<u16, i16, i16>>::multiply_heterogeneous_signal_scaled,
        )
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
{
    pub fn multiply_to_scaled<U>(
        self,
        other: &SignalView<'_, T>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy,
        Workspace: SignalAllocator<U>,
        Self: ScaledMixedBinarySignal<T, U>,
    {
        self.scaled_mixed_binary(
            other,
            scale_factor,
            <Self as ScaledMixedBinarySignal<T, U>>::multiply_mixed_signal_scaled,
        )
    }

    fn scaled_mixed_binary<U>(
        self,
        other: &SignalView<'_, T>,
        scale_factor: i32,
        operation: fn(
            &StreamContext,
            &SignalView<'_, T>,
            &SignalView<'_, T>,
            &mut SignalViewMut<'_, U>,
            i32,
        ) -> Result<()>,
    ) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy,
        Workspace: SignalAllocator<U>,
        Self: ScaledMixedBinarySignal<T, U>,
    {
        let mut destination = self.workspace.signal::<U>(self.len())?;
        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            operation(
                self.stream_context,
                &source,
                other,
                &mut destination_view,
                scale_factor,
            )?;
        }

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
    pub fn multiply_with_scaled<U, V>(
        self,
        other: &SignalView<'_, U>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, V>>
    where
        U: Copy,
        V: Copy,
        Workspace: SignalAllocator<V>,
        Self: ScaledHeterogeneousBinarySignal<T, U, V>,
    {
        self.scaled_heterogeneous_binary(
            other,
            scale_factor,
            <Self as ScaledHeterogeneousBinarySignal<T, U, V>>::multiply_heterogeneous_signal_scaled,
        )
    }

    fn scaled_heterogeneous_binary<U, V>(
        self,
        other: &SignalView<'_, U>,
        scale_factor: i32,
        operation: fn(
            &StreamContext,
            &SignalView<'_, T>,
            &SignalView<'_, U>,
            &mut SignalViewMut<'_, V>,
            i32,
        ) -> Result<()>,
    ) -> Result<SignalPipeline<'a, V>>
    where
        U: Copy,
        V: Copy,
        Workspace: SignalAllocator<V>,
        Self: ScaledHeterogeneousBinarySignal<T, U, V>,
    {
        let mut destination = self.workspace.signal::<V>(self.len())?;
        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            operation(
                self.stream_context,
                &source,
                other,
                &mut destination_view,
                scale_factor,
            )?;
        }

        Ok(SignalPipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: SignalBacking::Owned(destination),
        })
    }
}

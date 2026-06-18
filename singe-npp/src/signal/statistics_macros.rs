macro_rules! impl_statistic {
    (
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $name:ident,
        $type:ty,
        $buffer_size:ident,
        $statistic:ident
    ) => {
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $type>,
        ) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size(
                    to_u64(source.len(), "len")?,
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "scratch bytes")
        }

        pub fn $with_scratch_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $type>,
            destination: &mut SignalViewMut<'_, $type>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            validate_scalar_output(destination, "destination")?;
            let required_bytes = $buffer_size_name(stream_context, source)?;
            scratch.require(required_bytes)?;

            unsafe {
                try_ffi!(sys::$statistic(
                    source.as_ptr().cast(),
                    to_u64(source.len(), "len")?,
                    destination.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $type>,
            destination: &mut SignalViewMut<'_, $type>,
        ) -> Result<()> {
            let required_bytes = $buffer_size_name(stream_context, source)?;
            let mut scratch = ScratchBuffer::create(required_bytes)?;
            $with_scratch_name(stream_context, source, destination, &mut scratch)
        }
    };
}

macro_rules! impl_mixed_statistic {
    (
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $name:ident,
        $source_type:ty,
        $destination_type:ty,
        $buffer_size:ident,
        $statistic:ident
    ) => {
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_type>,
        ) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size(
                    to_u64(source.len(), "len")?,
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "scratch bytes")
        }

        pub fn $with_scratch_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_type>,
            destination: &mut SignalViewMut<'_, $destination_type>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            validate_scalar_output(destination, "destination")?;
            let required_bytes = $buffer_size_name(stream_context, source)?;
            scratch.require(required_bytes)?;

            unsafe {
                try_ffi!(sys::$statistic(
                    source.as_ptr().cast(),
                    to_u64(source.len(), "len")?,
                    destination.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_type>,
            destination: &mut SignalViewMut<'_, $destination_type>,
        ) -> Result<()> {
            let required_bytes = $buffer_size_name(stream_context, source)?;
            let mut scratch = ScratchBuffer::create(required_bytes)?;
            $with_scratch_name(stream_context, source, destination, &mut scratch)
        }
    };
}

macro_rules! impl_binary_statistic {
    (
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $name:ident,
        $type:ty,
        $buffer_size:ident,
        $statistic:ident
    ) => {
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $type>,
        ) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size(
                    to_u64(source.len(), "len")?,
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "scratch bytes")
        }

        pub fn $with_scratch_name(
            stream_context: &StreamContext,
            source_1: &SignalView<'_, $type>,
            source_2: &SignalView<'_, $type>,
            destination: &mut SignalViewMut<'_, $type>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            validate_same_len(source_1.len(), source_2.len(), "source")?;
            validate_scalar_output(destination, "destination")?;
            let required_bytes = $buffer_size_name(stream_context, source_1)?;
            scratch.require(required_bytes)?;

            unsafe {
                try_ffi!(sys::$statistic(
                    source_1.as_ptr().cast(),
                    source_2.as_ptr().cast(),
                    to_u64(source_1.len(), "len")?,
                    destination.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }

        pub fn $name(
            stream_context: &StreamContext,
            source_1: &SignalView<'_, $type>,
            source_2: &SignalView<'_, $type>,
            destination: &mut SignalViewMut<'_, $type>,
        ) -> Result<()> {
            let required_bytes = $buffer_size_name(stream_context, source_1)?;
            let mut scratch = ScratchBuffer::create(required_bytes)?;
            $with_scratch_name(
                stream_context,
                source_1,
                source_2,
                destination,
                &mut scratch,
            )
        }
    };
}

macro_rules! impl_mixed_binary_statistic {
    (
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $name:ident,
        $source_type:ty,
        $destination_type:ty,
        $buffer_size:ident,
        $statistic:ident
    ) => {
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_type>,
        ) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size(
                    to_u64(source.len(), "len")?,
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "scratch bytes")
        }

        pub fn $with_scratch_name(
            stream_context: &StreamContext,
            source_1: &SignalView<'_, $source_type>,
            source_2: &SignalView<'_, $source_type>,
            destination: &mut SignalViewMut<'_, $destination_type>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            validate_same_len(source_1.len(), source_2.len(), "source")?;
            validate_scalar_output(destination, "destination")?;
            let required_bytes = $buffer_size_name(stream_context, source_1)?;
            scratch.require(required_bytes)?;

            unsafe {
                try_ffi!(sys::$statistic(
                    source_1.as_ptr().cast(),
                    source_2.as_ptr().cast(),
                    to_u64(source_1.len(), "len")?,
                    destination.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }

        pub fn $name(
            stream_context: &StreamContext,
            source_1: &SignalView<'_, $source_type>,
            source_2: &SignalView<'_, $source_type>,
            destination: &mut SignalViewMut<'_, $destination_type>,
        ) -> Result<()> {
            let required_bytes = $buffer_size_name(stream_context, source_1)?;
            let mut scratch = ScratchBuffer::create(required_bytes)?;
            $with_scratch_name(
                stream_context,
                source_1,
                source_2,
                destination,
                &mut scratch,
            )
        }
    };
}

macro_rules! impl_heterogeneous_binary_statistic {
    (
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $name:ident,
        $source_1_type:ty,
        $source_2_type:ty,
        $destination_type:ty,
        $buffer_size:ident,
        $statistic:ident
    ) => {
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_1_type>,
        ) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size(
                    to_u64(source.len(), "len")?,
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "scratch bytes")
        }

        pub fn $with_scratch_name(
            stream_context: &StreamContext,
            source_1: &SignalView<'_, $source_1_type>,
            source_2: &SignalView<'_, $source_2_type>,
            destination: &mut SignalViewMut<'_, $destination_type>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            validate_same_len(source_1.len(), source_2.len(), "source")?;
            validate_scalar_output(destination, "destination")?;
            let required_bytes = $buffer_size_name(stream_context, source_1)?;
            scratch.require(required_bytes)?;

            unsafe {
                try_ffi!(sys::$statistic(
                    source_1.as_ptr().cast(),
                    source_2.as_ptr().cast(),
                    to_u64(source_1.len(), "len")?,
                    destination.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }

        pub fn $name(
            stream_context: &StreamContext,
            source_1: &SignalView<'_, $source_1_type>,
            source_2: &SignalView<'_, $source_2_type>,
            destination: &mut SignalViewMut<'_, $destination_type>,
        ) -> Result<()> {
            let required_bytes = $buffer_size_name(stream_context, source_1)?;
            let mut scratch = ScratchBuffer::create(required_bytes)?;
            $with_scratch_name(
                stream_context,
                source_1,
                source_2,
                destination,
                &mut scratch,
            )
        }
    };
}

macro_rules! impl_scaled_binary_statistic {
    (
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $name:ident,
        $source_type:ty,
        $destination_type:ty,
        $buffer_size:ident,
        $statistic:ident
    ) => {
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_type>,
        ) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size(
                    to_u64(source.len(), "len")?,
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "scratch bytes")
        }

        pub fn $with_scratch_name(
            stream_context: &StreamContext,
            source_1: &SignalView<'_, $source_type>,
            source_2: &SignalView<'_, $source_type>,
            destination: &mut SignalViewMut<'_, $destination_type>,
            scale_factor: i32,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            validate_same_len(source_1.len(), source_2.len(), "source")?;
            validate_scalar_output(destination, "destination")?;
            let required_bytes = $buffer_size_name(stream_context, source_1)?;
            scratch.require(required_bytes)?;

            unsafe {
                try_ffi!(sys::$statistic(
                    source_1.as_ptr().cast(),
                    source_2.as_ptr().cast(),
                    to_u64(source_1.len(), "len")?,
                    destination.as_mut_ptr().cast(),
                    scale_factor,
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }

        pub fn $name(
            stream_context: &StreamContext,
            source_1: &SignalView<'_, $source_type>,
            source_2: &SignalView<'_, $source_type>,
            destination: &mut SignalViewMut<'_, $destination_type>,
            scale_factor: i32,
        ) -> Result<()> {
            let required_bytes = $buffer_size_name(stream_context, source_1)?;
            let mut scratch = ScratchBuffer::create(required_bytes)?;
            $with_scratch_name(
                stream_context,
                source_1,
                source_2,
                destination,
                scale_factor,
                &mut scratch,
            )
        }
    };
}

macro_rules! impl_scaled_heterogeneous_binary_statistic {
    (
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $name:ident,
        $source_1_type:ty,
        $source_2_type:ty,
        $destination_type:ty,
        $buffer_size:ident,
        $statistic:ident
    ) => {
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_1_type>,
        ) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size(
                    to_u64(source.len(), "len")?,
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "scratch bytes")
        }

        pub fn $with_scratch_name(
            stream_context: &StreamContext,
            source_1: &SignalView<'_, $source_1_type>,
            source_2: &SignalView<'_, $source_2_type>,
            destination: &mut SignalViewMut<'_, $destination_type>,
            scale_factor: i32,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            validate_same_len(source_1.len(), source_2.len(), "source")?;
            validate_scalar_output(destination, "destination")?;
            let required_bytes = $buffer_size_name(stream_context, source_1)?;
            scratch.require(required_bytes)?;

            unsafe {
                try_ffi!(sys::$statistic(
                    source_1.as_ptr().cast(),
                    source_2.as_ptr().cast(),
                    to_u64(source_1.len(), "len")?,
                    destination.as_mut_ptr().cast(),
                    scale_factor,
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }

        pub fn $name(
            stream_context: &StreamContext,
            source_1: &SignalView<'_, $source_1_type>,
            source_2: &SignalView<'_, $source_2_type>,
            destination: &mut SignalViewMut<'_, $destination_type>,
            scale_factor: i32,
        ) -> Result<()> {
            let required_bytes = $buffer_size_name(stream_context, source_1)?;
            let mut scratch = ScratchBuffer::create(required_bytes)?;
            $with_scratch_name(
                stream_context,
                source_1,
                source_2,
                destination,
                scale_factor,
                &mut scratch,
            )
        }
    };
}

macro_rules! impl_scaled_statistic {
    (
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $name:ident,
        $source_type:ty,
        $destination_type:ty,
        $buffer_size:ident,
        $statistic:ident
    ) => {
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_type>,
        ) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size(
                    to_u64(source.len(), "len")?,
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "scratch bytes")
        }

        pub fn $with_scratch_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_type>,
            destination: &mut SignalViewMut<'_, $destination_type>,
            scale_factor: i32,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            validate_scalar_output(destination, "destination")?;
            let required_bytes = $buffer_size_name(stream_context, source)?;
            scratch.require(required_bytes)?;

            unsafe {
                try_ffi!(sys::$statistic(
                    source.as_ptr().cast(),
                    to_u64(source.len(), "len")?,
                    destination.as_mut_ptr().cast(),
                    scale_factor,
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_type>,
            destination: &mut SignalViewMut<'_, $destination_type>,
            scale_factor: i32,
        ) -> Result<()> {
            let required_bytes = $buffer_size_name(stream_context, source)?;
            let mut scratch = ScratchBuffer::create(required_bytes)?;
            $with_scratch_name(
                stream_context,
                source,
                destination,
                scale_factor,
                &mut scratch,
            )
        }
    };
}

macro_rules! impl_error_metric {
    (
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $name:ident,
        $source_type:ty,
        $buffer_size:ident,
        $statistic:ident
    ) => {
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_type>,
        ) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size(
                    to_u64(source.len(), "len")?,
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "scratch bytes")
        }

        pub fn $with_scratch_name(
            stream_context: &StreamContext,
            source_1: &SignalView<'_, $source_type>,
            source_2: &SignalView<'_, $source_type>,
            destination: &mut SignalViewMut<'_, f64>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            validate_same_len(source_1.len(), source_2.len(), "source")?;
            validate_scalar_output(destination, "destination")?;
            let required_bytes = $buffer_size_name(stream_context, source_1)?;
            scratch.require(required_bytes)?;

            unsafe {
                try_ffi!(sys::$statistic(
                    source_1.as_ptr().cast(),
                    source_2.as_ptr().cast(),
                    to_u64(source_1.len(), "len")?,
                    destination.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }

        pub fn $name(
            stream_context: &StreamContext,
            source_1: &SignalView<'_, $source_type>,
            source_2: &SignalView<'_, $source_type>,
            destination: &mut SignalViewMut<'_, f64>,
        ) -> Result<()> {
            let required_bytes = $buffer_size_name(stream_context, source_1)?;
            let mut scratch = ScratchBuffer::create(required_bytes)?;
            $with_scratch_name(
                stream_context,
                source_1,
                source_2,
                destination,
                &mut scratch,
            )
        }
    };
}

macro_rules! impl_mean_stddev {
    (
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $name:ident,
        $type:ty,
        $buffer_size:ident,
        $statistic:ident
    ) => {
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $type>,
        ) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size(
                    to_u64(source.len(), "len")?,
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "scratch bytes")
        }

        pub fn $with_scratch_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $type>,
            mean: &mut SignalViewMut<'_, $type>,
            standard_deviation: &mut SignalViewMut<'_, $type>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            validate_scalar_output(mean, "mean")?;
            validate_scalar_output(standard_deviation, "standard_deviation")?;
            let required_bytes = $buffer_size_name(stream_context, source)?;
            scratch.require(required_bytes)?;

            unsafe {
                try_ffi!(sys::$statistic(
                    source.as_ptr().cast(),
                    to_u64(source.len(), "len")?,
                    mean.as_mut_ptr().cast(),
                    standard_deviation.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $type>,
            mean: &mut SignalViewMut<'_, $type>,
            standard_deviation: &mut SignalViewMut<'_, $type>,
        ) -> Result<()> {
            let required_bytes = $buffer_size_name(stream_context, source)?;
            let mut scratch = ScratchBuffer::create(required_bytes)?;
            $with_scratch_name(
                stream_context,
                source,
                mean,
                standard_deviation,
                &mut scratch,
            )
        }
    };
}

macro_rules! impl_scaled_mean_stddev {
    (
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $name:ident,
        $source_type:ty,
        $destination_type:ty,
        $buffer_size:ident,
        $statistic:ident
    ) => {
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_type>,
        ) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size(
                    to_u64(source.len(), "len")?,
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "scratch bytes")
        }

        pub fn $with_scratch_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_type>,
            mean: &mut SignalViewMut<'_, $destination_type>,
            standard_deviation: &mut SignalViewMut<'_, $destination_type>,
            scale_factor: i32,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            validate_scalar_output(mean, "mean")?;
            validate_scalar_output(standard_deviation, "standard_deviation")?;
            let required_bytes = $buffer_size_name(stream_context, source)?;
            scratch.require(required_bytes)?;

            unsafe {
                try_ffi!(sys::$statistic(
                    source.as_ptr().cast(),
                    to_u64(source.len(), "len")?,
                    mean.as_mut_ptr().cast(),
                    standard_deviation.as_mut_ptr().cast(),
                    scale_factor,
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_type>,
            mean: &mut SignalViewMut<'_, $destination_type>,
            standard_deviation: &mut SignalViewMut<'_, $destination_type>,
            scale_factor: i32,
        ) -> Result<()> {
            let required_bytes = $buffer_size_name(stream_context, source)?;
            let mut scratch = ScratchBuffer::create(required_bytes)?;
            $with_scratch_name(
                stream_context,
                source,
                mean,
                standard_deviation,
                scale_factor,
                &mut scratch,
            )
        }
    };
}

macro_rules! impl_indexed_statistic {
    (
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $name:ident,
        $type:ty,
        $buffer_size:ident,
        $statistic:ident
    ) => {
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $type>,
        ) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size(
                    to_u64(source.len(), "len")?,
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "scratch bytes")
        }

        pub fn $with_scratch_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $type>,
            value: &mut SignalViewMut<'_, $type>,
            index: &mut SignalViewMut<'_, i32>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            validate_scalar_output(value, "value")?;
            validate_scalar_output(index, "index")?;
            let required_bytes = $buffer_size_name(stream_context, source)?;
            scratch.require(required_bytes)?;

            unsafe {
                try_ffi!(sys::$statistic(
                    source.as_ptr().cast(),
                    to_u64(source.len(), "len")?,
                    value.as_mut_ptr().cast(),
                    index.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $type>,
            value: &mut SignalViewMut<'_, $type>,
            index: &mut SignalViewMut<'_, i32>,
        ) -> Result<()> {
            let required_bytes = $buffer_size_name(stream_context, source)?;
            let mut scratch = ScratchBuffer::create(required_bytes)?;
            $with_scratch_name(stream_context, source, value, index, &mut scratch)
        }
    };
}

macro_rules! impl_pair_statistic {
    (
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $name:ident,
        $type:ty,
        $buffer_size:ident,
        $statistic:ident
    ) => {
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $type>,
        ) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size(
                    to_u64(source.len(), "len")?,
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "scratch bytes")
        }

        pub fn $with_scratch_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $type>,
            min: &mut SignalViewMut<'_, $type>,
            max: &mut SignalViewMut<'_, $type>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            validate_scalar_output(min, "min")?;
            validate_scalar_output(max, "max")?;
            let required_bytes = $buffer_size_name(stream_context, source)?;
            scratch.require(required_bytes)?;

            unsafe {
                try_ffi!(sys::$statistic(
                    source.as_ptr().cast(),
                    to_u64(source.len(), "len")?,
                    min.as_mut_ptr().cast(),
                    max.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $type>,
            min: &mut SignalViewMut<'_, $type>,
            max: &mut SignalViewMut<'_, $type>,
        ) -> Result<()> {
            let required_bytes = $buffer_size_name(stream_context, source)?;
            let mut scratch = ScratchBuffer::create(required_bytes)?;
            $with_scratch_name(stream_context, source, min, max, &mut scratch)
        }
    };
}

macro_rules! impl_indexed_pair_statistic {
    (
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $name:ident,
        $type:ty,
        $buffer_size:ident,
        $statistic:ident
    ) => {
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $type>,
        ) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size(
                    to_u64(source.len(), "len")?,
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "scratch bytes")
        }

        pub fn $with_scratch_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $type>,
            min: &mut SignalViewMut<'_, $type>,
            min_index: &mut SignalViewMut<'_, i32>,
            max: &mut SignalViewMut<'_, $type>,
            max_index: &mut SignalViewMut<'_, i32>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            validate_scalar_output(min, "min")?;
            validate_scalar_output(min_index, "min_index")?;
            validate_scalar_output(max, "max")?;
            validate_scalar_output(max_index, "max_index")?;
            let required_bytes = $buffer_size_name(stream_context, source)?;
            scratch.require(required_bytes)?;

            unsafe {
                try_ffi!(sys::$statistic(
                    source.as_ptr().cast(),
                    to_u64(source.len(), "len")?,
                    min.as_mut_ptr().cast(),
                    min_index.as_mut_ptr().cast(),
                    max.as_mut_ptr().cast(),
                    max_index.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $type>,
            min: &mut SignalViewMut<'_, $type>,
            min_index: &mut SignalViewMut<'_, i32>,
            max: &mut SignalViewMut<'_, $type>,
            max_index: &mut SignalViewMut<'_, i32>,
        ) -> Result<()> {
            let required_bytes = $buffer_size_name(stream_context, source)?;
            let mut scratch = ScratchBuffer::create(required_bytes)?;
            $with_scratch_name(
                stream_context,
                source,
                min,
                min_index,
                max,
                max_index,
                &mut scratch,
            )
        }
    };
}

macro_rules! impl_every_statistic_in_place {
    ($name:ident, $type:ty, $statistic:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $type>,
            destination: &mut SignalViewMut<'_, $type>,
        ) -> Result<()> {
            validate_same_len(source.len(), destination.len(), "destination")?;

            unsafe {
                try_ffi!(sys::$statistic(
                    source.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "len")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_zero_crossing {
    (
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $name:ident,
        $source_type:ty,
        $buffer_size:ident,
        $statistic:ident
    ) => {
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_type>,
        ) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size(
                    to_u64(source.len(), "len")?,
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "scratch bytes")
        }

        pub fn $with_scratch_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_type>,
            destination: &mut SignalViewMut<'_, f32>,
            zero_crossing_type: ZeroCrossingType,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            validate_scalar_output(destination, "destination")?;
            let required_bytes = $buffer_size_name(stream_context, source)?;
            scratch.require(required_bytes)?;

            unsafe {
                try_ffi!(sys::$statistic(
                    source.as_ptr().cast(),
                    to_u64(source.len(), "len")?,
                    destination.as_mut_ptr().cast(),
                    zero_crossing_type.into(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_type>,
            destination: &mut SignalViewMut<'_, f32>,
            zero_crossing_type: ZeroCrossingType,
        ) -> Result<()> {
            let required_bytes = $buffer_size_name(stream_context, source)?;
            let mut scratch = ScratchBuffer::create(required_bytes)?;
            $with_scratch_name(
                stream_context,
                source,
                destination,
                zero_crossing_type,
                &mut scratch,
            )
        }
    };
}

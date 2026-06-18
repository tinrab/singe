macro_rules! impl_constant_operation {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $ty>,
            value: $ty,
            destination: &mut SignalViewMut<'_, $ty>,
        ) -> Result<()> {
            validate_same_len(source.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    value.into_npp(),
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "source length")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_constant_operation_in_place {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, $ty>,
            value: $ty,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    value.into_npp(),
                    signal.as_mut_ptr().cast(),
                    to_u64(signal.len(), "signal length")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_scaled_constant_operation {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $ty>,
            value: $ty,
            destination: &mut SignalViewMut<'_, $ty>,
            scale_factor: i32,
        ) -> Result<()> {
            validate_same_len(source.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    value.into_npp(),
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "source length")?,
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_scaled_constant_operation_in_place {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, $ty>,
            value: $ty,
            scale_factor: i32,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    value.into_npp(),
                    signal.as_mut_ptr().cast(),
                    to_u64(signal.len(), "signal length")?,
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_scaled_mixed_constant_operation_in_place {
    ($name:ident, $signal_ty:ty, $value_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, $signal_ty>,
            value: $value_ty,
            scale_factor: i32,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    value.into_npp(),
                    signal.as_mut_ptr(),
                    to_u64(signal.len(), "signal length")?,
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_destination_update_constant_operation {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $ty>,
            value: $ty,
            destination: &mut SignalViewMut<'_, $ty>,
        ) -> Result<()> {
            validate_same_len(source.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    value,
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "source length")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_mixed_constant_operation {
    ($name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_ty>,
            value: $source_ty,
            destination: &mut SignalViewMut<'_, $destination_ty>,
        ) -> Result<()> {
            validate_same_len(source.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    value,
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "source length")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_scaled_mixed_constant_operation {
    ($name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_ty>,
            value: $source_ty,
            destination: &mut SignalViewMut<'_, $destination_ty>,
            scale_factor: i32,
        ) -> Result<()> {
            validate_same_len(source.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    value,
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "source length")?,
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_shift_constant_operation {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $ty>,
            shift: i32,
            destination: &mut SignalViewMut<'_, $ty>,
        ) -> Result<()> {
            validate_same_len(source.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    shift,
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "source length")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_shift_constant_operation_in_place {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, $ty>,
            shift: i32,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    shift,
                    signal.as_mut_ptr(),
                    to_u64(signal.len(), "signal length")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_binary_operation {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            left: &SignalView<'_, $ty>,
            right: &SignalView<'_, $ty>,
            destination: &mut SignalViewMut<'_, $ty>,
        ) -> Result<()> {
            validate_same_len(left.len(), right.len(), "left right length")?;
            validate_same_len(left.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    left.as_ptr().cast(),
                    right.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(left.len(), "left length")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_reversed_binary_operation {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            left: &SignalView<'_, $ty>,
            right: &SignalView<'_, $ty>,
            destination: &mut SignalViewMut<'_, $ty>,
        ) -> Result<()> {
            validate_same_len(left.len(), right.len(), "left right length")?;
            validate_same_len(left.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    right.as_ptr().cast(),
                    left.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(left.len(), "left length")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_binary_operation_in_place {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            signal: &SignalView<'_, $ty>,
            destination: &mut SignalViewMut<'_, $ty>,
        ) -> Result<()> {
            validate_same_len(signal.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    signal.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(signal.len(), "signal length")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_mixed_binary_operation {
    ($name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            left: &SignalView<'_, $source_ty>,
            right: &SignalView<'_, $source_ty>,
            destination: &mut SignalViewMut<'_, $destination_ty>,
        ) -> Result<()> {
            validate_same_len(left.len(), right.len(), "left right length")?;
            validate_same_len(left.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    left.as_ptr().cast(),
                    right.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(left.len(), "left length")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_reversed_mixed_binary_operation {
    ($name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            left: &SignalView<'_, $source_ty>,
            right: &SignalView<'_, $source_ty>,
            destination: &mut SignalViewMut<'_, $destination_ty>,
        ) -> Result<()> {
            validate_same_len(left.len(), right.len(), "left right length")?;
            validate_same_len(left.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    right.as_ptr().cast(),
                    left.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(left.len(), "left length")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_mixed_binary_operation_in_place {
    ($name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            signal: &SignalView<'_, $source_ty>,
            destination: &mut SignalViewMut<'_, $destination_ty>,
        ) -> Result<()> {
            validate_same_len(signal.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    signal.as_ptr(),
                    destination.as_mut_ptr().cast(),
                    to_u64(signal.len(), "signal length")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_heterogeneous_binary_operation {
    ($name:ident, $left_ty:ty, $right_ty:ty, $destination_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            left: &SignalView<'_, $left_ty>,
            right: &SignalView<'_, $right_ty>,
            destination: &mut SignalViewMut<'_, $destination_ty>,
        ) -> Result<()> {
            validate_same_len(left.len(), right.len(), "left right length")?;
            validate_same_len(left.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    left.as_ptr().cast(),
                    right.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(left.len(), "left length")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_heterogeneous_binary_operation_in_place {
    ($name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            signal: &SignalView<'_, $source_ty>,
            destination: &mut SignalViewMut<'_, $destination_ty>,
        ) -> Result<()> {
            validate_same_len(signal.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    signal.as_ptr(),
                    destination.as_mut_ptr().cast(),
                    to_u64(signal.len(), "signal length")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_scaled_binary_operation {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            left: &SignalView<'_, $ty>,
            right: &SignalView<'_, $ty>,
            destination: &mut SignalViewMut<'_, $ty>,
            scale_factor: i32,
        ) -> Result<()> {
            validate_same_len(left.len(), right.len(), "left right length")?;
            validate_same_len(left.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    left.as_ptr().cast(),
                    right.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(left.len(), "left length")?,
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_reversed_scaled_binary_operation {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            left: &SignalView<'_, $ty>,
            right: &SignalView<'_, $ty>,
            destination: &mut SignalViewMut<'_, $ty>,
            scale_factor: i32,
        ) -> Result<()> {
            validate_same_len(left.len(), right.len(), "left right length")?;
            validate_same_len(left.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    right.as_ptr().cast(),
                    left.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(left.len(), "left length")?,
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_scaled_mixed_binary_operation {
    ($name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            left: &SignalView<'_, $source_ty>,
            right: &SignalView<'_, $source_ty>,
            destination: &mut SignalViewMut<'_, $destination_ty>,
            scale_factor: i32,
        ) -> Result<()> {
            validate_same_len(left.len(), right.len(), "left right length")?;
            validate_same_len(left.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    left.as_ptr().cast(),
                    right.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(left.len(), "left length")?,
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_scaled_heterogeneous_binary_operation {
    ($name:ident, $left_ty:ty, $right_ty:ty, $destination_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            left: &SignalView<'_, $left_ty>,
            right: &SignalView<'_, $right_ty>,
            destination: &mut SignalViewMut<'_, $destination_ty>,
            scale_factor: i32,
        ) -> Result<()> {
            validate_same_len(left.len(), right.len(), "left right length")?;
            validate_same_len(left.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    left.as_ptr().cast(),
                    right.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(left.len(), "left length")?,
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_scaled_heterogeneous_binary_operation_in_place {
    ($name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            signal: &SignalView<'_, $source_ty>,
            destination: &mut SignalViewMut<'_, $destination_ty>,
            scale_factor: i32,
        ) -> Result<()> {
            validate_same_len(signal.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    signal.as_ptr(),
                    destination.as_mut_ptr().cast(),
                    to_u64(signal.len(), "signal length")?,
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_reversed_scaled_heterogeneous_binary_operation {
    ($name:ident, $left_ty:ty, $right_ty:ty, $destination_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            left: &SignalView<'_, $left_ty>,
            right: &SignalView<'_, $right_ty>,
            destination: &mut SignalViewMut<'_, $destination_ty>,
            scale_factor: i32,
        ) -> Result<()> {
            validate_same_len(left.len(), right.len(), "left right length")?;
            validate_same_len(left.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    right.as_ptr().cast(),
                    left.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(left.len(), "left length")?,
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_scaled_binary_operation_in_place {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            signal: &SignalView<'_, $ty>,
            destination: &mut SignalViewMut<'_, $ty>,
            scale_factor: i32,
        ) -> Result<()> {
            validate_same_len(signal.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    signal.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(signal.len(), "signal length")?,
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_reversed_scaled_round_binary_operation {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            left: &SignalView<'_, $ty>,
            right: &SignalView<'_, $ty>,
            destination: &mut SignalViewMut<'_, $ty>,
            scale_factor: i32,
            round_mode: RoundMode,
        ) -> Result<()> {
            validate_same_len(left.len(), right.len(), "left right length")?;
            validate_same_len(left.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    right.as_ptr().cast(),
                    left.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(left.len(), "left length")?,
                    round_mode.into(),
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_scaled_round_binary_operation_in_place {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            signal: &SignalView<'_, $ty>,
            destination: &mut SignalViewMut<'_, $ty>,
            scale_factor: i32,
            round_mode: RoundMode,
        ) -> Result<()> {
            validate_same_len(signal.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    signal.as_ptr(),
                    destination.as_mut_ptr().cast(),
                    to_u64(signal.len(), "signal length")?,
                    round_mode.into(),
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_destination_update_operation {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            left: &SignalView<'_, $ty>,
            right: &SignalView<'_, $ty>,
            destination: &mut SignalViewMut<'_, $ty>,
        ) -> Result<()> {
            validate_same_len(left.len(), right.len(), "left right length")?;
            validate_same_len(left.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    left.as_ptr().cast(),
                    right.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(left.len(), "left length")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_scaled_destination_update_operation {
    ($name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            left: &SignalView<'_, $source_ty>,
            right: &SignalView<'_, $source_ty>,
            destination: &mut SignalViewMut<'_, $destination_ty>,
            scale_factor: i32,
        ) -> Result<()> {
            validate_same_len(left.len(), right.len(), "left right length")?;
            validate_same_len(left.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    left.as_ptr().cast(),
                    right.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(left.len(), "left length")?,
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_unary_operation {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $ty>,
            destination: &mut SignalViewMut<'_, $ty>,
        ) -> Result<()> {
            validate_same_len(source.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "source length")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_unary_operation_in_place {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, $ty>,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    signal.as_mut_ptr().cast(),
                    to_u64(signal.len(), "signal length")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_parameterized_unary_operation_in_place {
    ($name:ident, $ty:ty, $parameter_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, $ty>,
            parameter: $parameter_ty,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    signal.as_mut_ptr(),
                    to_u64(signal.len(), "signal length")?,
                    parameter,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_scaled_unary_operation {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $ty>,
            destination: &mut SignalViewMut<'_, $ty>,
            scale_factor: i32,
        ) -> Result<()> {
            validate_same_len(source.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "source length")?,
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_scaled_unary_operation_in_place {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, $ty>,
            scale_factor: i32,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    signal.as_mut_ptr().cast(),
                    to_u64(signal.len(), "signal length")?,
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_scaled_mixed_unary_operation {
    ($name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_ty>,
            destination: &mut SignalViewMut<'_, $destination_ty>,
            scale_factor: i32,
        ) -> Result<()> {
            validate_same_len(source.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "source length")?,
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_mixed_unary_operation {
    ($name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_ty>,
            destination: &mut SignalViewMut<'_, $destination_ty>,
        ) -> Result<()> {
            validate_same_len(source.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "source length")?,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_normalize {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $ty>,
            destination: &mut SignalViewMut<'_, $ty>,
            subtrahend: $ty,
            divisor: $ty,
        ) -> Result<()> {
            validate_same_len(source.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "source length")?,
                    subtrahend,
                    divisor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_complex_normalize {
    ($name:ident, $ty:ty, $divisor_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $ty>,
            destination: &mut SignalViewMut<'_, $ty>,
            subtrahend: $ty,
            divisor: $divisor_ty,
        ) -> Result<()> {
            validate_same_len(source.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "source length")?,
                    subtrahend.into_npp(),
                    divisor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_scaled_normalize {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $ty>,
            destination: &mut SignalViewMut<'_, $ty>,
            subtrahend: $ty,
            divisor: i32,
            scale_factor: i32,
        ) -> Result<()> {
            validate_same_len(source.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "source length")?,
                    subtrahend.into_npp(),
                    divisor,
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_convert {
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

macro_rules! impl_convert_scaled {
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

macro_rules! impl_convert_scaled_round {
    ($name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_ty>,
            destination: &mut SignalViewMut<'_, $destination_ty>,
            round_mode: RoundMode,
            scale_factor: i32,
        ) -> Result<()> {
            validate_same_len(source.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "source length")? as _, // can be `Npp32u`
                    round_mode.into(),
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $ty>,
            destination: &mut SignalViewMut<'_, $ty>,
            level: $ty,
            operation: ComparisonOperation,
        ) -> Result<()> {
            validate_same_len(source.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "source length")?,
                    level,
                    operation.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_in_place {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, $ty>,
            level: $ty,
            operation: ComparisonOperation,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    signal.as_mut_ptr().cast(),
                    to_u64(signal.len(), "signal length")?,
                    level,
                    operation.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_complex {
    ($name:ident, $ty:ty, $level_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $ty>,
            destination: &mut SignalViewMut<'_, $ty>,
            level: $level_ty,
            operation: ComparisonOperation,
        ) -> Result<()> {
            validate_same_len(source.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "source length")?,
                    level,
                    operation.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_complex_in_place {
    ($name:ident, $ty:ty, $level_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, $ty>,
            level: $level_ty,
            operation: ComparisonOperation,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    signal.as_mut_ptr().cast(),
                    to_u64(signal.len(), "signal length")?,
                    level,
                    operation.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_fixed {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $ty>,
            destination: &mut SignalViewMut<'_, $ty>,
            level: $ty,
        ) -> Result<()> {
            validate_same_len(source.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "source length")?,
                    level,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_fixed_in_place {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, $ty>,
            level: $ty,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    signal.as_mut_ptr().cast(),
                    to_u64(signal.len(), "signal length")?,
                    level,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_fixed_complex {
    ($name:ident, $ty:ty, $level_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $ty>,
            destination: &mut SignalViewMut<'_, $ty>,
            level: $level_ty,
        ) -> Result<()> {
            validate_same_len(source.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "source length")?,
                    level,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_fixed_complex_in_place {
    ($name:ident, $ty:ty, $level_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, $ty>,
            level: $level_ty,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    signal.as_mut_ptr().cast(),
                    to_u64(signal.len(), "signal length")?,
                    level,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_value {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $ty>,
            destination: &mut SignalViewMut<'_, $ty>,
            level: $ty,
            value: $ty,
        ) -> Result<()> {
            validate_same_len(source.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "source length")?,
                    level,
                    value,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_value_in_place {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, $ty>,
            level: $ty,
            value: $ty,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    signal.as_mut_ptr().cast(),
                    to_u64(signal.len(), "signal length")?,
                    level,
                    value,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_value_complex {
    ($name:ident, $ty:ty, $level_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $ty>,
            destination: &mut SignalViewMut<'_, $ty>,
            level: $level_ty,
            value: $ty,
        ) -> Result<()> {
            validate_same_len(source.len(), destination.len(), "destination length")?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    to_u64(source.len(), "source length")?,
                    level,
                    value.into_npp(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_threshold_value_complex_in_place {
    ($name:ident, $ty:ty, $level_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, $ty>,
            level: $level_ty,
            value: $ty,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    signal.as_mut_ptr().cast(),
                    to_u64(signal.len(), "signal length")?,
                    level,
                    value.into_npp(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

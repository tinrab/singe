macro_rules! impl_binary_signal {
    (
        $ty:ty,
        $add:path,
        $add_in_place:path,
        $subtract:path,
        $subtract_in_place:path,
        $multiply:path,
        $multiply_in_place:path
    ) => {
        impl<'a> BinarySignal<$ty> for SignalPipeline<'a, $ty> {
            fn add_signal(
                stream_context: &StreamContext,
                left: &SignalView<'_, $ty>,
                right: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $add(stream_context, left, right, destination)
            }

            fn add_signal_in_place(
                stream_context: &StreamContext,
                signal: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $add_in_place(stream_context, signal, destination)
            }

            fn subtract_signal(
                stream_context: &StreamContext,
                left: &SignalView<'_, $ty>,
                right: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $subtract(stream_context, left, right, destination)
            }

            fn subtract_signal_in_place(
                stream_context: &StreamContext,
                signal: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $subtract_in_place(stream_context, signal, destination)
            }

            fn multiply_signal(
                stream_context: &StreamContext,
                left: &SignalView<'_, $ty>,
                right: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $multiply(stream_context, left, right, destination)
            }

            fn multiply_signal_in_place(
                stream_context: &StreamContext,
                signal: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $multiply_in_place(stream_context, signal, destination)
            }
        }
    };
}

macro_rules! impl_divide_signal {
    ($ty:ty, $divide:path, $divide_in_place:path) => {
        impl<'a> DivideSignal<$ty> for SignalPipeline<'a, $ty> {
            fn divide_signal(
                stream_context: &StreamContext,
                left: &SignalView<'_, $ty>,
                right: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $divide(stream_context, left, right, destination)
            }

            fn divide_signal_in_place(
                stream_context: &StreamContext,
                signal: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $divide_in_place(stream_context, signal, destination)
            }
        }
    };
}

macro_rules! impl_scaled_binary_signal {
    (
        $ty:ty,
        $add:path,
        $add_in_place:path,
        $subtract:path,
        $subtract_in_place:path,
        $multiply:path,
        $multiply_in_place:path
    ) => {
        impl<'a> ScaledBinarySignal<$ty> for SignalPipeline<'a, $ty> {
            fn add_signal_scaled(
                stream_context: &StreamContext,
                left: &SignalView<'_, $ty>,
                right: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                scale_factor: i32,
            ) -> Result<()> {
                $add(stream_context, left, right, destination, scale_factor)
            }

            fn add_signal_scaled_in_place(
                stream_context: &StreamContext,
                signal: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                scale_factor: i32,
            ) -> Result<()> {
                $add_in_place(stream_context, signal, destination, scale_factor)
            }

            fn subtract_signal_scaled(
                stream_context: &StreamContext,
                left: &SignalView<'_, $ty>,
                right: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                scale_factor: i32,
            ) -> Result<()> {
                $subtract(stream_context, left, right, destination, scale_factor)
            }

            fn subtract_signal_scaled_in_place(
                stream_context: &StreamContext,
                signal: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                scale_factor: i32,
            ) -> Result<()> {
                $subtract_in_place(stream_context, signal, destination, scale_factor)
            }

            fn multiply_signal_scaled(
                stream_context: &StreamContext,
                left: &SignalView<'_, $ty>,
                right: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                scale_factor: i32,
            ) -> Result<()> {
                $multiply(stream_context, left, right, destination, scale_factor)
            }

            fn multiply_signal_scaled_in_place(
                stream_context: &StreamContext,
                signal: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                scale_factor: i32,
            ) -> Result<()> {
                $multiply_in_place(stream_context, signal, destination, scale_factor)
            }
        }
    };
}

macro_rules! impl_divide_scaled_binary_signal {
    ($ty:ty, $divide:path, $divide_in_place:path) => {
        impl<'a> DivideScaledBinarySignal<$ty> for SignalPipeline<'a, $ty> {
            fn divide_signal_scaled(
                stream_context: &StreamContext,
                left: &SignalView<'_, $ty>,
                right: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                scale_factor: i32,
            ) -> Result<()> {
                $divide(stream_context, left, right, destination, scale_factor)
            }

            fn divide_signal_scaled_in_place(
                stream_context: &StreamContext,
                signal: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                scale_factor: i32,
            ) -> Result<()> {
                $divide_in_place(stream_context, signal, destination, scale_factor)
            }
        }
    };
}

macro_rules! impl_mixed_binary_signal {
    (
        $source_ty:ty,
        $destination_ty:ty,
        $add:path,
        $subtract:path,
        $multiply:path
    ) => {
        impl<'a> MixedBinarySignal<$source_ty, $destination_ty> for SignalPipeline<'a, $source_ty> {
            fn add_mixed_signal(
                stream_context: &StreamContext,
                left: &SignalView<'_, $source_ty>,
                right: &SignalView<'_, $source_ty>,
                destination: &mut SignalViewMut<'_, $destination_ty>,
            ) -> Result<()> {
                $add(stream_context, left, right, destination)
            }

            fn subtract_mixed_signal(
                stream_context: &StreamContext,
                left: &SignalView<'_, $source_ty>,
                right: &SignalView<'_, $source_ty>,
                destination: &mut SignalViewMut<'_, $destination_ty>,
            ) -> Result<()> {
                $subtract(stream_context, left, right, destination)
            }

            fn multiply_mixed_signal(
                stream_context: &StreamContext,
                left: &SignalView<'_, $source_ty>,
                right: &SignalView<'_, $source_ty>,
                destination: &mut SignalViewMut<'_, $destination_ty>,
            ) -> Result<()> {
                $multiply(stream_context, left, right, destination)
            }
        }
    };
}

macro_rules! impl_mixed_binary_signal_without_subtract {
    ($source_ty:ty, $destination_ty:ty, $add:path, $multiply:path) => {
        impl<'a> MixedBinarySignal<$source_ty, $destination_ty> for SignalPipeline<'a, $source_ty> {
            fn add_mixed_signal(
                stream_context: &StreamContext,
                left: &SignalView<'_, $source_ty>,
                right: &SignalView<'_, $source_ty>,
                destination: &mut SignalViewMut<'_, $destination_ty>,
            ) -> Result<()> {
                $add(stream_context, left, right, destination)
            }

            fn subtract_mixed_signal(
                _stream_context: &StreamContext,
                _left: &SignalView<'_, $source_ty>,
                _right: &SignalView<'_, $source_ty>,
                _destination: &mut SignalViewMut<'_, $destination_ty>,
            ) -> Result<()> {
                Err(Error::UnsupportedOperation {
                    name: "mixed signal subtract".into(),
                })
            }

            fn multiply_mixed_signal(
                stream_context: &StreamContext,
                left: &SignalView<'_, $source_ty>,
                right: &SignalView<'_, $source_ty>,
                destination: &mut SignalViewMut<'_, $destination_ty>,
            ) -> Result<()> {
                $multiply(stream_context, left, right, destination)
            }
        }
    };
}

macro_rules! impl_scaled_mixed_binary_signal {
    ($source_ty:ty, $destination_ty:ty, $multiply:path) => {
        impl<'a> ScaledMixedBinarySignal<$source_ty, $destination_ty>
            for SignalPipeline<'a, $source_ty>
        {
            fn multiply_mixed_signal_scaled(
                stream_context: &StreamContext,
                left: &SignalView<'_, $source_ty>,
                right: &SignalView<'_, $source_ty>,
                destination: &mut SignalViewMut<'_, $destination_ty>,
                scale_factor: i32,
            ) -> Result<()> {
                $multiply(stream_context, left, right, destination, scale_factor)
            }
        }
    };
}

macro_rules! impl_scaled_heterogeneous_binary_signal {
    ($left_ty:ty, $right_ty:ty, $destination_ty:ty, $multiply:path) => {
        impl<'a> ScaledHeterogeneousBinarySignal<$left_ty, $right_ty, $destination_ty>
            for SignalPipeline<'a, $left_ty>
        {
            fn multiply_heterogeneous_signal_scaled(
                stream_context: &StreamContext,
                left: &SignalView<'_, $left_ty>,
                right: &SignalView<'_, $right_ty>,
                destination: &mut SignalViewMut<'_, $destination_ty>,
                scale_factor: i32,
            ) -> Result<()> {
                $multiply(stream_context, left, right, destination, scale_factor)
            }
        }
    };
}

macro_rules! impl_add_product_signal {
    ($ty:ty, $add_product:path) => {
        impl<'a> AddProductSignal<$ty> for SignalPipeline<'a, $ty> {
            fn add_product_signal(
                stream_context: &StreamContext,
                left: &SignalView<'_, $ty>,
                right: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $add_product(stream_context, left, right, destination)
            }
        }
    };
}

macro_rules! impl_scaled_add_product_signal {
    ($source_ty:ty, $destination_ty:ty, $add_product:path) => {
        impl<'a> ScaledAddProductSignal<$source_ty, $destination_ty>
            for SignalPipeline<'a, $destination_ty>
        {
            fn add_product_signal_scaled(
                stream_context: &StreamContext,
                left: &SignalView<'_, $source_ty>,
                right: &SignalView<'_, $source_ty>,
                destination: &mut SignalViewMut<'_, $destination_ty>,
                scale_factor: i32,
            ) -> Result<()> {
                $add_product(stream_context, left, right, destination, scale_factor)
            }
        }
    };
}

use std::sync::Arc;

use crate::{
    backend::{
        cpu::{interpreted, kernels as cpu_kernels, random as random_kernels},
        decomposition,
        registry::{
            BatchNormalizationAttributes, ClampAttributes, ContractionAttributes, Conv2dAttributes,
            IndexPutAttributes, LayerNormalizationAttributes, Operation,
            OperationAttributes as Attributes, Pool2dAttributes, ReductionAttributes,
            SortAttributes, TransposeAttributes, registered_operations,
        },
    },
    data_type::DataType,
    dispatch::{
        DispatchKey, Kernel, KernelCall, RegistryBuilder, call as call_operation, call_one,
    },
    error::{Error, Result},
    shape::Shape,
    tensor::Tensor,
};

#[cfg(feature = "cuda")]
use crate::backend::cuda::kernels as cuda_kernels;

/// Returns the attributes of `call` matching `$pattern`, or an attribute error.
macro_rules! attributes {
    ($call:expr, $pattern:pat => $value:expr) => {
        match $call.attributes() {
            $pattern => $value,
            _ => return Err(invalid_attributes()),
        }
    };
}

/// Registers a single-result kernel for `$operation` at `$key`.
macro_rules! single {
    ($builder:expr, $key:expr, $operation:ident, $body:expr) => {
        $builder.kernel(Operation::$operation, $key, Kernel::single($body));
    };
}

macro_rules! unary {
    ($builder:expr, $key:expr, $kernels:ident; $($operation:ident => $method:ident),+ $(,)?) => {
        $(single!($builder, $key, $operation, |call: &KernelCall<'_>| $kernels::$method(call.input(0)?));)+
    };
}

macro_rules! binary {
    ($builder:expr, $key:expr, $kernels:ident; $($operation:ident => $method:ident),+ $(,)?) => {
        $(single!($builder, $key, $operation, |call: &KernelCall<'_>| {
            $kernels::$method(call.input(0)?, call.input(1)?)
        });)+
    };
}

macro_rules! scalar {
    ($builder:expr, $key:expr, $kernels:ident; $($operation:ident => $method:ident),+ $(,)?) => {
        $(single!($builder, $key, $operation, |call: &KernelCall<'_>| {
            let value = attributes!(call, Attributes::Scalar(value) => *value);
            $kernels::$method(call.input(0)?, value)
        });)+
    };
}

macro_rules! reverse_scalar {
    ($builder:expr, $key:expr, $kernels:ident; $($operation:ident => $method:ident),+ $(,)?) => {
        $(single!($builder, $key, $operation, |call: &KernelCall<'_>| {
            let value = attributes!(call, Attributes::Scalar(value) => *value);
            $kernels::$method(value, call.input(0)?)
        });)+
    };
}

macro_rules! reduction {
    ($builder:expr, $key:expr, $kernels:ident; $($operation:ident => $method:ident),+ $(,)?) => {
        $(single!($builder, $key, $operation, |call: &KernelCall<'_>| {
            let (dimensions, keepdim) = attributes!(
                call,
                Attributes::Reduction(ReductionAttributes { dimensions, keepdim }) => (dimensions, *keepdim)
            );
            $kernels::$method(call.input(0)?, dimensions, keepdim)
        });)+
    };
}

macro_rules! dimension {
    ($builder:expr, $key:expr, $kernels:ident; $($operation:ident => $method:ident),+ $(,)?) => {
        $(single!($builder, $key, $operation, |call: &KernelCall<'_>| {
            let dim = attributes!(call, Attributes::Dimension(dim) => *dim);
            along_dimension(call.input(0)?, dim, $kernels::$method)
        });)+
    };
}

/// Registers a kernel for every built-in operation at `$key`, calling the functions of the `$kernels` module.
macro_rules! backend_kernels {
    ($builder:expr, $key:expr, $kernels:ident) => {{
        let builder: &mut RegistryBuilder = $builder;
        let key: DispatchKey = $key;
        binary!(builder, key, $kernels; MatMul => matmul, Embedding => embedding);
        unary!(builder, key, $kernels; Contiguous => contiguous);
        reduction!(builder, key, $kernels; Sum => sum, Mean => mean, Max => max, Min => min, Product => product, LogSumExp => logsumexp);
        dimension!(builder, key, $kernels;
            Softmax => softmax, LogSoftmax => log_softmax, ArgMax => argmax, ArgMin => argmin,
        );

        single!(builder, key, RotaryEmbedding, |call: &KernelCall<'_>| {
            $kernels::rotary_embedding(call.input(0)?, call.input(1)?, call.input(2)?)
        });
        single!(builder, key, Copy, |call: &KernelCall<'_>| {
            let input = call.input(0)?;
            $kernels::copy_to(input, input.device())
        });
        single!(builder, key, CopyTo, |call: &KernelCall<'_>| {
            let device = attributes!(call, Attributes::CopyTo(device) => *device);
            let input = call.input(0)?;
            if input.device() == device {
                Ok(input.alias_unrecorded())
            } else {
                $kernels::copy_to(input, device)
            }
        });
        single!(builder, key, Squeeze, |call: &KernelCall<'_>| {
            let dim = attributes!(call, Attributes::Dimension(dim) => *dim);
            let input = call.input(0)?;
            input.view_unrecorded(input.layout().squeeze(dim)?)
        });
        single!(builder, key, Unsqueeze, |call: &KernelCall<'_>| {
            let dim = attributes!(call, Attributes::Dimension(dim) => *dim);
            let input = call.input(0)?;
            input.view_unrecorded(input.layout().unsqueeze(dim)?)
        });
        single!(builder, key, Transpose, |call: &KernelCall<'_>| {
            let (dim0, dim1) = attributes!(call, Attributes::Transpose(TransposeAttributes { dim0, dim1 }) => (*dim0, *dim1));
            let input = call.input(0)?;
            input.view_unrecorded(input.layout().transpose(dim0, dim1)?)
        });
        single!(builder, key, Permute, |call: &KernelCall<'_>| {
            let order = attributes!(call, Attributes::Permute(order) => order);
            let input = call.input(0)?;
            input.view_unrecorded(input.layout().permute(order)?)
        });
        single!(builder, key, Reshape, |call: &KernelCall<'_>| {
            let shape = attributes!(call, Attributes::Shape(shape) => shape);
            reshape(call.input(0)?, shape)
        });
        single!(builder, key, Flatten, |call: &KernelCall<'_>| {
            reshape(call.input(0)?, call.request().inference().outputs()[0].shape())
        });
        single!(builder, key, Expand, |call: &KernelCall<'_>| {
            let shape = attributes!(call, Attributes::Shape(shape) => shape);
            let input = call.input(0)?;
            Tensor::create_from_storage(
                Arc::clone(input.storage()),
                input.data_type(),
                input.layout().expand(shape)?,
            )
        });
        single!(builder, key, Slice, |call: &KernelCall<'_>| {
            let slice = call.attributes().slice().copied().ok_or_else(invalid_attributes)?;
            let input = call.input(0)?;
            input.strided_view_unrecorded(input.layout().slice(
                slice.dim,
                slice.start,
                slice.length,
                slice.step,
            )?)
        });
        single!(builder, key, Concat, |call: &KernelCall<'_>| {
            let dim = attributes!(call, Attributes::Concat(dim) => *dim);
            $kernels::concat(call.inputs(), dim)
        });
        single!(builder, key, Index, |call: &KernelCall<'_>| {
            let dim = attributes!(call, Attributes::Dimension(dim) => *dim);
            $kernels::index(call.input(0)?, &call.inputs()[1..], dim)
        });
        single!(builder, key, IndexPut, |call: &KernelCall<'_>| {
            let IndexPutAttributes { dim, accumulate } =
                attributes!(call, Attributes::IndexPut(attributes) => *attributes);
            $kernels::index_put(call.input(0)?, call.input(1)?, &call.inputs()[2..], dim, accumulate)
        });
        single!(builder, key, Argsort, |call: &KernelCall<'_>| {
            let SortAttributes { dim, descending } =
                attributes!(call, Attributes::Sort(attributes) => *attributes);
            $kernels::argsort(call.input(0)?, dim, descending)
        });
        single!(builder, key, Cumsum, |call: &KernelCall<'_>| {
            let dim = attributes!(call, Attributes::Dimension(dim) => *dim);
            $kernels::cumsum(call.input(0)?, dim)
        });
        single!(builder, key, Cumprod, |call: &KernelCall<'_>| {
            let dim = attributes!(call, Attributes::Dimension(dim) => *dim);
            $kernels::cumprod(call.input(0)?, dim)
        });
        single!(builder, key, IndexSelect, |call: &KernelCall<'_>| {
            let dim = attributes!(call, Attributes::IndexSelect(dim) => *dim);
            $kernels::index_select(call.input(0)?, dim, call.input(1)?)
        });
        single!(builder, key, Take, |call: &KernelCall<'_>| {
            let dim = attributes!(call, Attributes::IndexSelect(dim) => *dim);
            $kernels::take(call.input(0)?, dim, call.input(1)?)
        });
        single!(builder, key, RepeatKv, |call: &KernelCall<'_>| {
            let repeats = attributes!(call, Attributes::RepeatKv(repeats) => *repeats);
            $kernels::repeat_kv(call.input(0)?, repeats)
        });
        single!(builder, key, Quantize, |call: &KernelCall<'_>| {
            let quantization = call.attributes().quantization().copied().ok_or_else(invalid_attributes)?;
            $kernels::quantize(call.input(0)?, call.input(1)?, call.input(2)?, quantization)
        });
        single!(builder, key, Dequantize, |call: &KernelCall<'_>| {
            let quantization = call.attributes().quantization().copied().ok_or_else(invalid_attributes)?;
            $kernels::dequantize(call.input(0)?, call.input(1)?, call.input(2)?, quantization)
        });
        single!(builder, key, Conv2d, |call: &KernelCall<'_>| {
            let (has_bias, stride, padding, dilation, groups) = attributes!(
                call,
                Attributes::Conv2d(Conv2dAttributes { has_bias, stride, padding, dilation, groups }) => (*has_bias, *stride, *padding, *dilation, *groups)
            );
            $kernels::convolution2d(
                call.input(0)?,
                call.input(1)?,
                call.optional_input(2, has_bias)?,
                stride,
                padding,
                dilation,
                groups,
            )
        });
        single!(builder, key, MaxPool2d, |call: &KernelCall<'_>| {
            let (kernel_size, stride, padding, dilation, ceil_mode) = attributes!(
                call,
                Attributes::Pool2d(Pool2dAttributes { kernel_size, stride, padding, dilation, ceil_mode, .. }) => (*kernel_size, *stride, *padding, *dilation, *ceil_mode)
            );
            $kernels::max_pool2d(call.input(0)?, kernel_size, stride, padding, dilation, ceil_mode)
        });
        single!(builder, key, AvgPool2d, |call: &KernelCall<'_>| {
            let (kernel_size, stride, padding, dilation, ceil_mode, count_include_pad) = attributes!(
                call,
                Attributes::Pool2d(Pool2dAttributes { kernel_size, stride, padding, dilation, ceil_mode, count_include_pad: Some(count_include_pad) }) => (*kernel_size, *stride, *padding, *dilation, *ceil_mode, *count_include_pad)
            );
            $kernels::avg_pool2d(
                call.input(0)?,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
                count_include_pad,
            )
        });
        single!(builder, key, BatchNorm, |call: &KernelCall<'_>| {
            let (has_weight, has_bias, has_running_mean, has_running_var, training, momentum, eps) = attributes!(
                call,
                Attributes::BatchNormalization(BatchNormalizationAttributes { has_weight, has_bias, has_running_mean, has_running_var, training, momentum, eps }) => (*has_weight, *has_bias, *has_running_mean, *has_running_var, *training, *momentum, *eps)
            );
            let weight = call.optional_input(1, has_weight)?;
            let bias_index = 1 + usize::from(has_weight);
            let bias = call.optional_input(bias_index, has_bias)?;
            let mean_index = bias_index + usize::from(has_bias);
            let mean = call.optional_input(mean_index, has_running_mean)?;
            let variance =
                call.optional_input(mean_index + usize::from(has_running_mean), has_running_var)?;
            $kernels::batch_norm(
                call.input(0)?, weight, bias, mean, variance, training, momentum, eps,
            )
        });
        single!(builder, key, LayerNorm, |call: &KernelCall<'_>| {
            let (normalized_shape, has_weight, has_bias, eps) = attributes!(
                call,
                Attributes::LayerNormalization(LayerNormalizationAttributes { normalized_shape, has_weight, has_bias, eps }) => (normalized_shape, *has_weight, *has_bias, *eps)
            );
            let weight = call.optional_input(1, has_weight)?;
            let bias = call.optional_input(1 + usize::from(has_weight), has_bias)?;
            $kernels::layer_norm(call.input(0)?, normalized_shape, weight, bias, eps)
        });
        single!(builder, key, RmsNorm, |call: &KernelCall<'_>| {
            let (normalized_shape, has_weight, has_bias, eps) = attributes!(
                call,
                Attributes::LayerNormalization(LayerNormalizationAttributes { normalized_shape, has_weight, has_bias, eps }) => (normalized_shape, *has_weight, *has_bias, *eps)
            );
            let weight = call.optional_input(1, has_weight)?;
            let bias = call.optional_input(1 + usize::from(has_weight), has_bias)?;
            $kernels::rms_norm(call.input(0)?, normalized_shape, weight, bias, eps)
        });
        single!(builder, key, ScaledDotProductAttention, |call: &KernelCall<'_>| {
            let attention = attributes!(
                call,
                Attributes::ScaledDotProductAttention(attention) => *attention
            );
            $kernels::scaled_dot_product_attention(
                call.input(0)?,
                call.input(1)?,
                call.input(2)?,
                call.optional_input(3, attention.masked)?,
                call.optional_input(3 + usize::from(attention.masked), attention.sinks)?,
                call.optional_input(
                    3 + usize::from(attention.masked) + usize::from(attention.sinks),
                    attention.key_lengths,
                )?,
                attention,
            )
        });
        builder.kernel(
            Operation::PagedAttention,
            key,
            Kernel::new(|call| {
                let attention = attributes!(
                    call,
                    Attributes::ScaledDotProductAttention(attention) => *attention
                );
                let inputs = (0..7)
                    .map(|index| call.input(index))
                    .collect::<Result<Vec<_>>>()?;
                $kernels::paged_attention(&inputs, attention)
            }),
        );
        for (operation, result) in [
            (Operation::PagedAttentionOutput, 0),
            (Operation::PagedAttentionLse, 1),
        ] {
            builder.kernel(
                operation,
                key,
                Kernel::new(move |call| {
                    let attention = attributes!(
                        call,
                        Attributes::ScaledDotProductAttention(attention) => *attention
                    );
                    let inputs = (0..7)
                        .map(|index| call.input(index))
                        .collect::<Result<Vec<_>>>()?;
                    let mut outputs = $kernels::paged_attention(&inputs, attention)?;
                    Ok(vec![outputs.swap_remove(result)])
                }),
            );
        }
        single!(builder, key, GroupedMatMul, |call: &KernelCall<'_>| {
            $kernels::grouped_matmul(call.input(0)?, call.input(1)?, call.input(2)?)
        });
        builder.kernel(
            Operation::PagedMlaAttention,
            key,
            Kernel::new(|call| {
                let attention = attributes!(
                    call,
                    Attributes::ScaledDotProductAttention(attention) => *attention
                );
                let inputs = (0..8)
                    .map(|index| call.input(index))
                    .collect::<Result<Vec<_>>>()?;
                $kernels::paged_mla_attention(&inputs, attention)
            }),
        );
        for (operation, result) in [
            (Operation::PagedMlaAttentionOutput, 0),
            (Operation::PagedMlaAttentionLse, 1),
        ] {
            builder.kernel(
                operation,
                key,
                Kernel::new(move |call| {
                    let attention = attributes!(
                        call,
                        Attributes::ScaledDotProductAttention(attention) => *attention
                    );
                    let inputs = (0..8)
                        .map(|index| call.input(index))
                        .collect::<Result<Vec<_>>>()?;
                    let mut outputs = $kernels::paged_mla_attention(&inputs, attention)?;
                    Ok(vec![outputs.swap_remove(result)])
                }),
            );
        }
        single!(builder, key, Contraction, |call: &KernelCall<'_>| {
            let (a_mode, b_mode, c_mode, d_mode, alpha, beta) = attributes!(
                call,
                Attributes::Contraction(ContractionAttributes { a_mode, b_mode, c_mode, d_mode, alpha, beta }) => (a_mode, b_mode, c_mode, d_mode, *alpha, *beta)
            );
            $kernels::contraction(
                call.input(0)?,
                a_mode,
                call.input(1)?,
                b_mode,
                call.input(2)?,
                c_mode,
                d_mode,
                alpha,
                beta,
            )
        });

        builder.kernel(
            Operation::Assign,
            key,
            Kernel::new(|call| {
                $kernels::assign(call.input(0)?, call.input(1)?)?;
                Ok(Vec::new())
            }),
        );
        builder.kernel(
            Operation::IndexWrite,
            key,
            Kernel::new(|call| {
                let IndexPutAttributes { dim, accumulate } =
                    attributes!(call, Attributes::IndexPut(attributes) => *attributes);
                $kernels::index_write(
                    call.input(0)?,
                    call.input(1)?,
                    &call.inputs()[2..],
                    dim,
                    accumulate,
                )?;
                Ok(Vec::new())
            }),
        );
    }};
}

/// Registers every backend kernel and the composite kernels.
pub fn register_kernels(builder: &mut RegistryBuilder) {
    backend_kernels!(builder, DispatchKey::Cpu, cpu_kernels);
    register_cpu_elementwise_kernels(builder);
    register_interpreted_kernels(builder);
    register_widened_kernels(builder, DispatchKey::Cpu, &CPU_WIDENED);
    register_composed_kernels(builder, DispatchKey::Cpu);
    register_decomposed_fallbacks(
        builder,
        DispatchKey::Cpu,
        &[
            Operation::Contraction,
            Operation::ScaledMatMul,
            Operation::Rfft,
            Operation::Irfft,
            Operation::Conv3d,
            Operation::BatchNorm,
            Operation::RotaryEmbedding,
            Operation::Softmax,
            Operation::LogSoftmax,
            Operation::MatMul,
        ],
    );
    #[cfg(feature = "cuda")]
    {
        backend_kernels!(builder, DispatchKey::Cuda, cuda_kernels);
        single!(builder, DispatchKey::Cuda, Conv3d, |call: &KernelCall<
            '_,
        >| {
            let conv = call
                .attributes()
                .conv3d()
                .copied()
                .ok_or_else(invalid_attributes)?;
            cuda_kernels::convolution3d(
                call.input(0)?,
                call.input(1)?,
                call.optional_input(2, conv.has_bias)?,
                &conv,
            )
        });
        single!(builder, DispatchKey::Cuda, Rfft, |call: &KernelCall<'_>| {
            cuda_kernels::rfft(call.input(0)?)
        });
        single!(builder, DispatchKey::Cuda, Irfft, |call: &KernelCall<
            '_,
        >| {
            let length = attributes!(call, Attributes::Length(length) => *length);
            cuda_kernels::irfft(call.input(0)?, length)
        });
        single!(
            builder,
            DispatchKey::Cuda,
            ScaledMatMul,
            |call: &KernelCall<'_>| {
                let output = attributes!(call, Attributes::Cast(output) => *output);
                cuda_kernels::scaled_matmul(
                    call.input(0)?,
                    call.input(1)?,
                    call.input(2)?,
                    call.input(3)?,
                    output,
                )
            }
        );
        for operation in [
            Operation::TopK,
            Operation::TopKValues,
            Operation::TopKIndices,
        ] {
            builder.kernel(
                operation,
                DispatchKey::Cuda,
                Kernel::new(move |call| {
                    let topk = call
                        .attributes()
                        .topk()
                        .copied()
                        .ok_or_else(invalid_attributes)?;
                    cuda_kernels::topk(operation, call.input(0)?, topk)
                }),
            );
        }

        register_widened_kernels(builder, DispatchKey::Cuda, &[Operation::RotaryEmbedding]);
        register_composed_kernels(builder, DispatchKey::Cuda);
        register_decomposed_fallbacks(
            builder,
            DispatchKey::Cuda,
            &[
                Operation::Contraction,
                Operation::BatchNorm,
                Operation::RotaryEmbedding,
                Operation::Softmax,
                Operation::LogSoftmax,
                Operation::MatMul,
            ],
        );
        // Element-wise operations, and reductions and row operations without a CUDA kernel of their own, run their generated kernel.
        for &operation in registered_operations() {
            if (interpreted::interprets(operation) || operation == Operation::SwiGlu)
                && builder.registered(operation, DispatchKey::Cuda).is_none()
            {
                builder.kernel(
                    operation,
                    DispatchKey::Cuda,
                    Kernel::single(move |call| {
                        let inputs = call.inputs().iter().collect::<Vec<_>>();
                        cuda_kernels::elementwise(operation, call.attributes(), &inputs)
                    }),
                );
            }
        }
    }

    for operation in [
        Operation::Uniform,
        Operation::Normal,
        Operation::Bernoulli,
        Operation::Multinomial,
    ] {
        builder.kernel(
            operation,
            DispatchKey::Cpu,
            Kernel::single(move |call| cpu_random(operation, call)),
        );
        #[cfg(feature = "cuda")]
        builder.kernel(
            operation,
            DispatchKey::Cuda,
            Kernel::single(move |call| {
                let inputs = call.inputs().iter().collect::<Vec<_>>();
                cuda_kernels::random(operation, call.attributes(), &inputs)
            }),
        );
    }

    builder.composite(Operation::SwiGlu, Kernel::single(swiglu_composite));
    builder.composite(Operation::AsStrided, Kernel::single(as_strided_composite));
    builder.composite(
        Operation::StridedScatter,
        Kernel::single(strided_scatter_composite),
    );
    // Every backend pads, ranks values, and multiplies quantized operands as their decompositions do.
    for operation in [
        Operation::Pad,
        Operation::TopK,
        Operation::TopKValues,
        Operation::TopKIndices,
        Operation::QuantizedMatMul,
    ] {
        builder.composite(
            operation,
            Kernel::new(move |call| decomposed(operation, call)),
        );
    }
}

/// Registers the native CPU kernels of element-wise operations, which compute `F32` and `F64`; the kernel IR interpreter wraps them for every other type.
fn register_cpu_elementwise_kernels(builder: &mut RegistryBuilder) {
    let key = DispatchKey::Cpu;
    binary!(builder, key, cpu_kernels;
        Add => add, Sub => sub, Mul => mul, Div => div,
        Equal => equal, NotEqual => not_equal, Less => less, LessEqual => less_equal,
        Greater => greater, GreaterEqual => greater_equal,
        LogicalAnd => logical_and, LogicalOr => logical_or,
        Modulo => modulo, Remainder => remainder, Atan2 => atan2,
        Maximum => maximum, Minimum => minimum,
    );
    unary!(builder, key, cpu_kernels;
        LogicalNot => logical_not, Neg => neg, Abs => abs, Sqrt => sqrt, Exp => exp,
        Log => log, Sin => sin, Cos => cos, Tan => tan, Sinh => sinh, Cosh => cosh,
        Tanh => tanh, Asin => asin, Acos => acos, Atan => atan, Ceil => ceil,
        Floor => floor, Truncate => truncate, Reciprocal => reciprocal, Square => square,
        Rsqrt => rsqrt, Asinh => asinh, Acosh => acosh, Atanh => atanh, Erf => erf,
        Relu => relu, Gelu => gelu, GeluExact => gelu_exact, Silu => silu,
        Sigmoid => sigmoid,
    );
    scalar!(builder, key, cpu_kernels; AddScalar => add_scalar, SubScalar => sub_scalar, Scale => scale);
    reverse_scalar!(builder, key, cpu_kernels; RSubScalar => rsub_scalar, RDivScalar => rdiv_scalar);
    single!(builder, key, Where, |call: &KernelCall<'_>| {
        cpu_kernels::where_op(call.input(0)?, call.input(1)?, call.input(2)?)
    });
    single!(builder, key, Cast, |call: &KernelCall<'_>| {
        let data_type = attributes!(call, Attributes::Cast(data_type) => *data_type);
        cpu_kernels::cast(call.input(0)?, data_type)
    });
    single!(builder, key, Clamp, |call: &KernelCall<'_>| {
        let (min, max) =
            attributes!(call, Attributes::Clamp(ClampAttributes { min, max }) => (*min, *max));
        cpu_kernels::clamp(call.input(0)?, min, max)
    });
    single!(builder, key, Lerp, |call: &KernelCall<'_>| {
        let t = attributes!(call, Attributes::Lerp(t) => *t);
        cpu_kernels::lerp(call.input(0)?, call.input(1)?, t)
    });
}

/// Wraps the CPU kernel of every element-wise operation and reduction so that calls its native kernel does not compute run through the kernel IR interpreter.
fn register_interpreted_kernels(builder: &mut RegistryBuilder) {
    for &operation in registered_operations() {
        if !interpreted::interprets(operation) {
            continue;
        }

        let native = builder.registered(operation, DispatchKey::Cpu);
        builder.kernel(
            operation,
            DispatchKey::Cpu,
            Kernel::new(move |call| match &native {
                Some(native) if interpreted::is_native(call) => native.call(call),
                _ => Ok(vec![interpreted::run(call)?]),
            }),
        );
    }
}

// CPU operations whose native kernels compute only in `F32` and `F64`, which half precision widens to.
const CPU_WIDENED: [Operation; 5] = [
    Operation::Conv2d,
    Operation::MaxPool2d,
    Operation::AvgPool2d,
    Operation::RotaryEmbedding,
    Operation::ScaledDotProductAttention,
];

// Half-precision operands of an operation whose native kernel computes only in `F32` compute in `F32` and round their results, as PyTorch's CPU computes them.
fn register_widened_kernels(
    builder: &mut RegistryBuilder,
    key: DispatchKey,
    operations: &[Operation],
) {
    for &operation in operations {
        let Some(native) = builder.registered(operation, key) else {
            continue;
        };
        builder.kernel(
            operation,
            key,
            Kernel::new(move |call| {
                let half =
                    |input: &Tensor| matches!(input.data_type(), DataType::F16 | DataType::Bf16);
                if !call.inputs().iter().any(half) {
                    return native.call(call);
                }

                let wide = call
                    .inputs()
                    .iter()
                    .map(|input| {
                        if half(input) {
                            input.cast(DataType::F32)
                        } else {
                            Ok(input.clone())
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                let outputs = call_operation(
                    operation,
                    call.attributes().clone(),
                    &wide.iter().collect::<Vec<_>>(),
                )?;
                outputs
                    .iter()
                    .zip(call.request().inference().outputs())
                    .map(|(output, spec)| output.cast(spec.data_type()))
                    .collect()
            }),
        );
    }
}

// Head repeats of types the native `F32` kernels do not hold are an expanded view's copy, which holds every type.
fn register_composed_kernels(builder: &mut RegistryBuilder, key: DispatchKey) {
    if let Some(native) = builder.registered(Operation::RepeatKv, key) {
        builder.kernel(
            Operation::RepeatKv,
            key,
            Kernel::single(move |call| {
                let repeats = attributes!(call, Attributes::RepeatKv(repeats) => *repeats);
                let input = call.input(0)?;
                if input.data_type() == DataType::F32 {
                    return native
                        .call(call)?
                        .pop()
                        .ok_or_else(|| Error::missing_output("register composed kernels"));
                }

                let dimensions = input.shape().dimensions();
                let [batch, heads, sequence, width] = dimensions[..] else {
                    return Err(Error::invalid_argument("repeat kv operand rank"));
                };
                input
                    .unsqueeze(2)?
                    .expand(&Shape::create([batch, heads, repeats, sequence, width])?)?
                    .reshape([batch, heads * repeats, sequence, width])
            }),
        );
    }
}

// Runs the decompositions of calls whose inputs the native kernels of `operations` do not compute.
fn register_decomposed_fallbacks(
    builder: &mut RegistryBuilder,
    key: DispatchKey,
    operations: &[Operation],
) {
    for &operation in operations {
        let native = builder.registered(operation, key);
        builder.kernel(
            operation,
            key,
            Kernel::new(move |call| {
                let data_type = call.input(0)?.data_type();
                match &native {
                    Some(native)
                        if !decomposition::replaces_kernels(
                            operation,
                            call.attributes(),
                            data_type,
                        ) =>
                    {
                        native.call(call)
                    }
                    _ => decomposed(operation, call),
                }
            }),
        );
    }
}

fn decomposed(operation: Operation, call: &KernelCall<'_>) -> Result<Vec<Tensor>> {
    let inputs = call.inputs().iter().collect::<Vec<_>>();
    decomposition::decompose(operation, call.attributes(), &inputs)
}

/// Runs a random operation on the host with the shared Philox engine.
fn cpu_random(operation: Operation, call: &KernelCall<'_>) -> Result<Tensor> {
    let random = call.attributes().random().ok_or_else(invalid_attributes)?;
    match operation {
        Operation::Uniform => random_kernels::uniform(call.input(0)?, random),
        Operation::Normal => random_kernels::normal(call.input(0)?, random),
        Operation::Bernoulli => random_kernels::bernoulli(call.input(0)?, call.input(1)?, random),
        Operation::Multinomial => {
            random_kernels::multinomial(call.input(0)?, call.input(1)?, random)
        }
        _ => Err(Error::unsupported_operation("cpu random kernel")),
    }
}

/// Computes `input * silu(gate)` from its parts.
fn swiglu_composite(call: &KernelCall<'_>) -> Result<Tensor> {
    let gate = call_one(Operation::Silu, Attributes::None, &[call.input(1)?])?;
    call_one(Operation::Mul, Attributes::None, &[call.input(0)?, &gate])
}

/// Views a flat input through strided addressing, copying it first when it is not contiguous.
fn as_strided_composite(call: &KernelCall<'_>) -> Result<Tensor> {
    let strided = call.attributes().strided().ok_or_else(invalid_attributes)?;
    let input = call.input(0)?;
    let base = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()?
    };
    base.strided_view_unrecorded(strided.layout(base.storage_offset())?)
}

/// Copies the flat base, then writes the source at strided positions of the copy.
fn strided_scatter_composite(call: &KernelCall<'_>) -> Result<Tensor> {
    let strided = call.attributes().strided().ok_or_else(invalid_attributes)?;
    let output = call_one(Operation::Copy, Attributes::None, &[call.input(0)?])?;
    let target = output.strided_view_unrecorded(strided.layout(output.storage_offset())?)?;
    call_operation(
        Operation::Assign,
        Attributes::None,
        &[&target, call.input(1)?],
    )?;
    Ok(output)
}

fn invalid_attributes() -> Error {
    Error::unsupported_operation("operation attributes")
}

fn reshape(input: &Tensor, shape: &Shape) -> Result<Tensor> {
    if let Some(layout) = input.layout().reshape_view(shape)? {
        return input.view_unrecorded(layout);
    }

    let contiguous = input.contiguous()?;
    let layout = contiguous
        .layout()
        .reshape_view(shape)?
        .ok_or(Error::InvalidStride)?;
    contiguous.view_unrecorded(layout)
}

/// Runs a kernel that works along `dim`, treating a scalar input as a vector of one element.
///
/// Inference has checked that `dim` is zero or minus one for a scalar; results of the vector shape `[1]` become scalars again, and scalar results stay scalars.
fn along_dimension(
    input: &Tensor,
    dim: i64,
    kernel: impl FnOnce(&Tensor, i64) -> Result<Tensor>,
) -> Result<Tensor> {
    if input.shape().rank() != 0 {
        return kernel(input, dim);
    }

    let vector = input.view_unrecorded(input.layout().unsqueeze(0)?)?;
    let output = kernel(&vector, 0)?;
    if output.shape().rank() == 0 {
        return Ok(output);
    }
    output.view_unrecorded(output.layout().squeeze(0)?)
}

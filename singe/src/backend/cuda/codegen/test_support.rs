//! Helpers that check generated kernels on the GPU against `singe-codegen`'s interpreter, shared by every target's tests.

use std::ptr;

use half::{bf16, f16};
use singe_codegen::{
    abi::Bindings,
    interpret,
    ir::{
        Access, AxisKind, BinaryOp, Combiner, ElementType, Expr, Kernel, OperandId, OperandRole,
        Size, UnaryOp,
    },
    schedule::Schedule,
};

use crate::{
    backend::cuda::{
        CudaStream,
        codegen::{KernelDescription, PlanningContext, plan_kernel},
        runtime::runtime_for_device,
        tensor_region::CudaTensorSource,
    },
    data_type::DataType,
    device::Device,
    shape::Shape,
    tensor::Tensor,
};

/// Runs `kernel` on the GPU under `schedule` with `buffers`, one per operand, and returns every buffer afterwards.
pub fn on_gpu(
    kernel: &Kernel,
    schedule: Schedule,
    bindings: &Bindings,
    buffers: &[Vec<u8>],
) -> Vec<Vec<u8>> {
    let device = Device::Cuda(0);
    let runtime = runtime_for_device(device).unwrap();
    let tensors = buffers
        .iter()
        .map(|bytes| {
            let length = i64::try_from(bytes.len().max(1)).unwrap();
            let mut padded = bytes.clone();
            padded.resize(usize::try_from(length).unwrap(), 0);
            Tensor::from_bytes(
                device,
                DataType::U8,
                Shape::create([length]).unwrap(),
                &padded,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let pointer = |tensor: &Tensor| tensor.tensor_region().unwrap().region().as_ptr();
    let (inputs, outputs): (Vec<_>, Vec<_>) = kernel
        .operands
        .iter()
        .zip(&tensors)
        .partition(|(operand, _)| operand.role == OperandRole::Input);
    let inputs = inputs
        .into_iter()
        .map(|(_, tensor)| pointer(tensor))
        .collect::<Vec<_>>();
    let outputs = outputs
        .into_iter()
        .map(|(_, tensor)| pointer(tensor))
        .collect::<Vec<_>>();
    let generated = plan_kernel(
        kernel,
        schedule,
        bindings,
        &KernelDescription::default(),
        &PlanningContext::default(),
    )
    .unwrap();
    let function = runtime.kernel(&generated).unwrap();
    let stream = CudaStream::create(device).unwrap();
    if !generated.is_empty() {
        let arguments = generated
            .abi
            .arguments(&inputs, &outputs, &[], ptr::null_mut())
            .unwrap();
        // SAFETY: the kernel addresses only elements its accesses cover, which the interpreter checks fit these buffers, and the stream finishes before the tensors drop.
        unsafe {
            function
                .launch_on(&generated.launch_config().unwrap(), arguments, stream.raw())
                .unwrap();
        }
    }
    stream.raw().synchronize().unwrap();
    tensors
        .iter()
        .zip(buffers)
        .map(|(tensor, bytes)| {
            let mut result = tensor.to_bytes().unwrap();
            result.truncate(bytes.len());
            result
        })
        .collect()
}

pub fn on_cpu(kernel: &Kernel, bindings: &Bindings, buffers: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let mut buffers = buffers.to_vec();
    interpret::run(kernel, bindings, &mut buffers).unwrap();
    buffers
}

/// Decodes `bytes` as elements of `element`, widened to `f64`.
pub fn values(element: ElementType, bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(element.size_bytes())
        .map(|chunk| match element {
            ElementType::F32 => f64::from(f32::from_le_bytes(chunk.try_into().unwrap())),
            ElementType::Bf16 => f64::from(bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32()),
            ElementType::F16 => f64::from(f16::from_le_bytes([chunk[0], chunk[1]]).to_f32()),
            ElementType::I64 => i64::from_le_bytes(chunk.try_into().unwrap()) as f64,
            ElementType::I32 => f64::from(i32::from_le_bytes(chunk.try_into().unwrap())),
            ElementType::Bool | ElementType::U8 => f64::from(chunk[0]),
            other => unreachable!("tests decode no {other:?} outputs"),
        })
        .collect()
}

/// Asserts that every output of `kernel` matches between the GPU and the interpreter within `tolerance`, relative to the interpreter's magnitude.
pub fn assert_matches(
    kernel: &Kernel,
    schedule: Schedule,
    bindings: &Bindings,
    buffers: &[Vec<u8>],
    tolerance: f64,
) {
    let gpu = on_gpu(kernel, schedule, bindings, buffers);
    let cpu = on_cpu(kernel, bindings, buffers);
    for (index, operand) in kernel.operands.iter().enumerate() {
        if operand.role != OperandRole::Output {
            continue;
        }
        let actual = values(operand.element, &gpu[index]);
        let expected = values(operand.element, &cpu[index]);
        for (position, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            let close = actual == expected
                || (actual.is_nan() && expected.is_nan())
                || (actual - expected).abs() <= tolerance * expected.abs().max(1.0);
            assert!(
                close,
                "{} under {schedule:?}, element {position}: gpu {actual}, interpreter {expected}",
                kernel.name
            );
        }
    }
}

pub fn f32_buffer(count: usize, seed: usize) -> Vec<u8> {
    (0..count)
        .flat_map(|index| (((index * 37 + seed * 11) % 97) as f32 / 25.0 - 1.9).to_le_bytes())
        .collect()
}

pub fn bf16_buffer(count: usize, seed: usize) -> Vec<u8> {
    f32_buffer(count, seed)
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|&chunk| bf16::from_f32(f32::from_le_bytes(chunk)).to_le_bytes())
        .collect()
}

/// Builds `out[row, column] = f(x[row, column])` for rows of `width` over a parallel row axis and a reduction column axis.
pub fn rows_kernel(name: &str, element: ElementType, weighted: bool) -> (Kernel, OperandId) {
    let mut kernel = Kernel::new(name);
    let x = kernel.add_operand(element, OperandRole::Input);
    let weight = weighted.then(|| kernel.add_operand(element, OperandRole::Input));
    let out = kernel.add_operand(element, OperandRole::Output);
    let rows = kernel.add_size(Size::Argument);
    let width = kernel.add_size(Size::Argument);
    let one = kernel.add_size(Size::Static(1));
    let row = kernel.add_axis(rows, AxisKind::Parallel);
    let column = kernel.add_axis(width, AxisKind::Reduction);
    let matrix = |operand| Access {
        operand,
        base: None,
        indirect: Vec::new(),
        strides: vec![(row, width), (column, one)],
    };
    let value = kernel.load(matrix(x));
    let result = if let Some(weight) = weight {
        // RMS norm: x × rsqrt(mean(x²) + eps) × weight.
        let square = kernel.binary(BinaryOp::Mul, value, value);
        let sum = kernel.reduce(Combiner::Sum, vec![column], square);
        let width_scalar = kernel.add_scalar(ElementType::F32);
        let eps_scalar = kernel.add_scalar(ElementType::F32);
        let width_value = kernel.add_node(ElementType::F32, Expr::Scalar(width_scalar));
        let mean = kernel.binary(BinaryOp::Div, sum, width_value);
        let eps = kernel.add_node(ElementType::F32, Expr::Scalar(eps_scalar));
        let shifted = kernel.binary(BinaryOp::Add, mean, eps);
        let scale = kernel.unary(UnaryOp::Rsqrt, shifted);
        let scaled = kernel.binary(BinaryOp::Mul, value, scale);
        let factor = kernel.load(Access {
            operand: weight,
            base: None,
            indirect: Vec::new(),
            strides: vec![(column, one)],
        });
        kernel.binary(BinaryOp::Mul, scaled, factor)
    } else {
        // Softmax.
        let maximum = kernel.reduce(Combiner::Max, vec![column], value);
        let shifted = kernel.binary(BinaryOp::Sub, value, maximum);
        let exponential = kernel.unary(UnaryOp::Exp, shifted);
        let sum = kernel.reduce(Combiner::Sum, vec![column], exponential);
        kernel.binary(BinaryOp::Div, exponential, sum)
    };
    kernel.store(matrix(out), result);
    (kernel, out)
}

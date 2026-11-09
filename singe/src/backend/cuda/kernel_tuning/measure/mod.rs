//! Times generated kernels on buffers shaped for their launch, replaying CUDA graphs of back-to-back launches between CUDA events.

use std::{ptr, sync::Arc, time::Duration};

use half::{bf16, f16};
use singe_codegen::{
    abi::Bindings,
    ir::{Access, AxisKind, Combiner, ElementType, Kernel, OperandRole, Size},
    schedule::{Schedule, space::operand_lengths},
};
use singe_cuda::{
    device::Device as NativeCudaDevice,
    graph::ExecutableGraph,
    module::{KernelFunction, LaunchConfig},
    stream::StreamCaptureMode,
};

use crate::{
    backend::cuda::{
        CudaEvent, CudaStream, EventTiming,
        codegen::{KernelArtifact, KernelDescription, emit_kernel},
        runtime::{CudaRuntime, runtime_for_device},
        tensor_region::CudaTensorSource,
    },
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    shape::Shape,
    tensor::Tensor,
    utility::{to_i64, to_u64, to_usize},
};

/// The time one kernel took over several launches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timing {
    pub median: Duration,
    pub minimum: Duration,
    /// The distance between the first and third quartiles.
    pub spread: Duration,
}

impl Timing {
    pub(crate) fn of(mut samples: Vec<Duration>) -> Self {
        samples.sort_unstable();
        let at = |fraction: usize| samples[(samples.len() - 1) * fraction / 4];
        Self {
            median: at(2),
            minimum: samples[0],
            spread: at(3).saturating_sub(at(1)),
        }
    }
}

/// Buffers for every launch of one kernel, filled once so every schedule of it sees the same data.
pub struct KernelBench {
    device: Device,
    runtime: Arc<CudaRuntime>,
    stream: CudaStream,
    inputs: Vec<Tensor>,
    /// The byte length of each output buffer.
    output_lengths: Vec<usize>,
    outputs: Vec<Tensor>,
    /// A kernel that reads a buffer twice the L2 cache's size, its input, and its small output.
    flush: (KernelArtifact, Tensor, Tensor),
}

impl KernelBench {
    /// Allocates and fills the operands of `kernel` launched with `bindings` on `device`.
    ///
    /// # Errors
    ///
    /// Returns an error if the kernel gathers or scatters through index tensors, whose buffers depend on data, or the buffers cannot be allocated.
    pub fn create(device: Device, kernel: &Kernel, bindings: &Bindings) -> Result<Self> {
        let lengths = operand_lengths(kernel, &bindings.sizes)
            .ok_or_else(|| Error::unsupported_operation("tuning a kernel with indexed accesses"))?;
        let mut inputs = Vec::new();
        let mut output_lengths = Vec::new();
        for (operand, &length) in kernel.operands.iter().zip(&lengths) {
            let length = to_usize(length, "tuned operand length")?;
            match operand.role {
                OperandRole::Input => {
                    inputs.push(bytes_tensor(device, &fill(operand.element, length))?);
                }
                OperandRole::Output => output_lengths.push(length * operand.element.size_bytes()),
            }
        }

        let outputs = output_lengths
            .iter()
            .map(|&length| bytes_tensor(device, &vec![0; length]))
            .collect::<Result<Vec<_>>>()?;
        let ordinal = device
            .cuda_id()
            .ok_or_else(|| Error::invalid_argument("kernel tuning device"))?;
        let l2 = to_usize(
            NativeCudaDevice::new(ordinal).properties()?.l2_cache_size,
            "l2 cache size",
        )?;
        Ok(Self {
            device,
            runtime: runtime_for_device(device)?,
            stream: CudaStream::create(device)?,
            inputs,
            output_lengths,
            outputs,
            flush: flush_kernel(device, (2 * l2).max(1 << 20))?,
        })
    }

    /// Returns the outputs `generated` writes into zeroed buffers.
    ///
    /// # Errors
    ///
    /// Returns an error if the kernel cannot compile or launch, or its outputs cannot be read.
    pub fn outputs(&mut self, generated: &KernelArtifact) -> Result<Vec<Vec<u8>>> {
        self.outputs = self
            .output_lengths
            .iter()
            .map(|&length| bytes_tensor(self.device, &vec![0; length]))
            .collect::<Result<Vec<_>>>()?;
        self.launch(generated)?;
        self.stream.raw().synchronize()?;
        self.outputs
            .iter()
            .zip(&self.output_lengths)
            .map(|(tensor, &length)| {
                let mut bytes = tensor.to_bytes()?;
                bytes.truncate(length);
                Ok(bytes)
            })
            .collect()
    }

    /// Times each of `kernels` over `rounds` rounds, timing each once per round in turn so drifting clocks affect them alike.
    ///
    /// A CUDA event pair resolves about 2 µs on the development GPU, and stream launches cannot start closer than about 2 µs apart, so each sample replays a CUDA graph of back-to-back launches lasting at least 50 µs and divides by its length, as a compiled graph runs its kernels.
    /// Batched launches find their data in the L2 cache, as a graph's kernels mostly find what the kernel before them wrote; with `cold`, every launch in a batch follows a kernel that reads twice the L2 cache's size, and the sample subtracts a graph of those reads alone.
    /// That subtraction leaves the reads' own variation, so cold samples of kernels much shorter than the reads are noisy.
    ///
    /// # Errors
    ///
    /// Returns an error if a kernel cannot compile, be captured, or launch.
    pub fn time(
        &self,
        kernels: &[KernelArtifact],
        rounds: usize,
        cold: bool,
    ) -> Result<Vec<Timing>> {
        Ok(self
            .time_rounds(kernels, rounds, cold)?
            .into_iter()
            .map(Timing::of)
            .collect())
    }

    /// Returns each kernel's time per launch in each round, in the order of `kernels`.
    ///
    /// # Errors
    ///
    /// Returns an error if a kernel cannot compile, be captured, or launch.
    pub fn time_rounds(
        &self,
        kernels: &[KernelArtifact],
        rounds: usize,
        cold: bool,
    ) -> Result<Vec<Vec<Duration>>> {
        let launches = kernels
            .iter()
            .map(|kernel| self.prepared(kernel))
            .collect::<Result<Vec<_>>>()?;
        let flush = self.prepared(&self.flush.0)?;
        let start = CudaEvent::create(self.device, EventTiming::Recorded)?;
        let end = CudaEvent::create(self.device, EventTiming::Recorded)?;
        let buffers = self.buffers()?;
        let flush_buffers = self.flush_buffers();
        let graph = |launch: Option<&Prepared>, count: usize| {
            self.captured(count, || {
                if cold || launch.is_none() {
                    self.enqueue(&flush, &flush_buffers)?;
                }
                launch.map_or(Ok(()), |launch| self.enqueue(launch, &buffers))
            })
        };
        let replay = |graph: &ExecutableGraph| -> Result<Duration> {
            let hold = self.stream.hold()?;
            start.record(&self.stream)?;
            // SAFETY: the graph's launches address the bench's buffers, which outlive it, and the replay is synchronized before the next one or any read.
            unsafe {
                graph.launch(self.stream.raw())?;
            }

            end.record(&self.stream)?;
            hold.release();
            end.synchronize()?;
            if hold.overran() {
                return Err(Error::unsupported_operation(
                    "a kernel replay that waits for a stream held until it is enqueued",
                ));
            }

            end.elapsed_since(&start)
        };
        // Each kernel's batch length comes from a graph of ten, whose first replay also warms its module and the clocks.
        let mut graphs = Vec::with_capacity(launches.len());
        for launch in &launches {
            let probe = graph(Some(launch), 10)?;
            replay(&probe)?;
            let per_launch = (replay(&probe)?.as_nanos() / 10).max(1);
            let count = BATCH.as_nanos().div_ceil(per_launch).clamp(1, 2000);
            let count = usize::try_from(count).unwrap_or(1);
            let flushes = if cold {
                Some(graph(None, count)?)
            } else {
                None
            };
            let timed = graph(Some(launch), count)?;
            replay(&timed)?;
            graphs.push((timed, flushes, count));
        }

        let mut samples = vec![Vec::with_capacity(rounds); launches.len()];
        for _ in 0..rounds.max(1) {
            for ((timed, flushes, count), samples) in graphs.iter().zip(&mut samples) {
                let mut elapsed = replay(timed)?;
                if let Some(flushes) = flushes {
                    elapsed = elapsed.saturating_sub(replay(flushes)?);
                }

                samples.push(elapsed / u32::try_from(*count).unwrap_or(u32::MAX));
            }
        }
        Ok(samples)
    }

    /// Returns a CUDA graph of `count` runs of `body`, which enqueues work on the bench's stream.
    fn captured(&self, count: usize, body: impl Fn() -> Result<()>) -> Result<ExecutableGraph> {
        let stream = self.stream.raw();
        stream.begin_capture(StreamCaptureMode::ThreadLocal)?;
        let enqueued = (0..count).try_for_each(|_| body());
        let graph = stream.end_capture();
        enqueued?;
        Ok(graph?.instantiate()?)
    }

    fn launch(&self, generated: &KernelArtifact) -> Result<()> {
        let prepared = self.prepared(generated)?;
        self.enqueue(&prepared, &self.buffers()?)
    }

    /// Returns `generated` compiled and ready to launch, or `None` for a kernel with no blocks.
    fn prepared(&self, generated: &KernelArtifact) -> Result<Prepared> {
        if generated.is_empty() {
            return Ok(None);
        }
        Ok(Some((
            self.runtime.kernel(generated)?,
            generated.launch_config()?,
            generated.clone(),
        )))
    }

    /// Returns the addresses of the input and output buffers.
    fn buffers(&self) -> Result<(Vec<*mut u8>, Vec<*mut u8>)> {
        let address = |tensor: &Tensor| Ok(tensor.tensor_region()?.region().as_ptr());
        Ok((
            self.inputs
                .iter()
                .map(address)
                .collect::<Result<Vec<_>>>()?,
            self.outputs
                .iter()
                .map(address)
                .collect::<Result<Vec<_>>>()?,
        ))
    }

    fn flush_buffers(&self) -> (Vec<*mut u8>, Vec<*mut u8>) {
        let address = |tensor: &Tensor| {
            tensor
                .tensor_region()
                .map_or(ptr::null_mut(), |region| region.region().as_ptr())
        };
        (vec![address(&self.flush.1)], vec![address(&self.flush.2)])
    }

    fn enqueue(&self, launch: &Prepared, buffers: &(Vec<*mut u8>, Vec<*mut u8>)) -> Result<()> {
        let Some((function, config, generated)) = launch else {
            return Ok(());
        };
        // A null status word skips every status report.
        let arguments = generated
            .abi
            .arguments(&buffers.0, &buffers.1, &[], ptr::null_mut())?;
        // SAFETY: the buffers span every element the kernel's accesses address, as `operand_lengths` computed them, and they outlive the launch, which the bench synchronizes before reading or dropping them.
        unsafe {
            function.launch_on(config, arguments, self.stream.raw())?;
        }
        Ok(())
    }
}

/// The shortest a timed batch of launches lasts, well above the about 2 µs a CUDA event pair resolves.
const BATCH: Duration = Duration::from_micros(50);

/// A compiled kernel, its launch shape, and its argument layout, or `None` for a kernel with no blocks.
type Prepared = Option<(KernelFunction, LaunchConfig, KernelArtifact)>;

/// Returns a kernel that sums each 1024 words of a `bytes`-byte buffer into one word, that buffer, and the sums.
///
/// Reading leaves the L2 cache holding clean lines, so the kernel timed after it pays only for its own misses, not for writing back the flush's data.
fn flush_kernel(device: Device, bytes: usize) -> Result<(KernelArtifact, Tensor, Tensor)> {
    const ROW: u64 = 1024;
    let mut kernel = Kernel::new("singe_l2_flush");
    let input = kernel.add_operand(ElementType::U32, OperandRole::Input);
    let out = kernel.add_operand(ElementType::U32, OperandRole::Output);
    let rows = kernel.add_size(Size::Argument);
    let width = kernel.add_size(Size::Static(ROW));
    let one = kernel.add_size(Size::Static(1));
    let row = kernel.add_axis(rows, AxisKind::Parallel);
    let column = kernel.add_axis(width, AxisKind::Reduction);
    let value = kernel.load(Access {
        operand: input,
        base: None,
        indirect: Vec::new(),
        strides: vec![(row, width), (column, one)],
    });
    let sum = kernel.reduce(Combiner::Sum, vec![column], value);
    kernel.store(
        Access {
            operand: out,
            base: None,
            indirect: Vec::new(),
            strides: vec![(row, one)],
        },
        sum,
    );
    let words = to_u64(bytes / 4, "flush length")?;
    let row_count = words.div_ceil(ROW);
    let bindings = Bindings {
        sizes: vec![row_count],
        scalars: Vec::new(),
    };
    let generated = emit_kernel(
        &kernel,
        Schedule::Rows {
            threads: 256,
            vector: 4,
            tiles: [1, 1],
        },
        &bindings,
        &KernelDescription::default(),
    )?;
    let length = |words: u64| to_usize(words * 4, "flush length");
    Ok((
        generated,
        bytes_tensor(device, &vec![1; length(row_count * ROW)?])?,
        bytes_tensor(device, &vec![0; length(row_count)?])?,
    ))
}

fn bytes_tensor(device: Device, bytes: &[u8]) -> Result<Tensor> {
    let length = to_i64(bytes.len().max(1), "bench buffer")?;
    let mut padded = bytes.to_vec();
    padded.resize(bytes.len().max(1), 0);
    Tensor::from_bytes(device, DataType::U8, Shape::create([length])?, &padded)
}

/// Returns `length` elements of `element` with deterministic values between about -2 and 2 for floats, and small non-negative integers.
fn fill(element: ElementType, length: usize) -> Vec<u8> {
    let value = |index: usize| ((index * 37 + 11) % 97) as f32 / 25.0 - 1.9;
    let mut bytes = Vec::with_capacity(length * element.size_bytes());
    for index in 0..length {
        match element {
            ElementType::F32 => bytes.extend(value(index).to_le_bytes()),
            ElementType::F64 => bytes.extend(f64::from(value(index)).to_le_bytes()),
            ElementType::F16 => bytes.extend(f16::from_f32(value(index)).to_le_bytes()),
            ElementType::Bf16 => bytes.extend(bf16::from_f32(value(index)).to_le_bytes()),
            ElementType::Bool => bytes.push(u8::from(index % 3 == 0)),
            integer => {
                let small = ((index * 7) % 13) as u64;
                bytes.extend(&small.to_le_bytes()[..integer.size_bytes()]);
            }
        }
    }
    bytes
}

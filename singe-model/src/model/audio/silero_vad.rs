//! Silero VAD v5, a voice activity detector, read from its ONNX file, as `onnx-community/silero-vad` holds it, and the turns it finds in a stream.
//!
//! The file computes either of its two models, for 16 kHz or 8 kHz audio, under an `If` on the sample rate; Singe reads the 16 kHz model's weights and computes it itself.
//! Each call reads a window of 512 samples after the 64 that ended the window before it: a short-time Fourier transform as one convolution, its magnitudes through four convolutions, a step of an LSTM that carries its state between windows, and a sigmoid over a projection of the LSTM's state.

use std::sync::Arc;

use singe::{DataType, Device, Shape, Tensor, shape::spatial::Padding};
use singe_onnx::{
    AttributeValue, Graph, Model, Node, Tensor as OnnxTensor, TensorDataType as OnnxDataType,
    TensorElements,
};

use crate::{
    error::{Error, Result},
    processor::audio::StreamingResampler,
    source::Source,
};

/// The sample rate the detector reads.
pub const SAMPLE_RATE: u32 = 16_000;

/// The samples of a window, 32 ms at [`SAMPLE_RATE`].
pub const WINDOW: usize = 512;

/// A window's length in milliseconds.
const WINDOW_MS: u64 = 32;

/// The samples of the window before that each window reads first.
const CONTEXT: usize = 64;

/// The short-time Fourier transform's hop.
const STFT_HOP: i64 = 128;

/// The samples the short-time Fourier transform reflects past the window's end.
const STFT_PADDING: i64 = 64;

/// The LSTM's width.
const WIDTH: i64 = 128;

/// The frequency bins of the short-time Fourier transform, each with a real and an imaginary row in its basis.
const BINS: i64 = 129;

/// The file in an `onnx-community` snapshot that holds the full-precision model.
const FILE: &str = "onnx/model.onnx";

#[derive(Debug, Clone)]
struct EncoderLayer {
    weight: Tensor,
    bias: Tensor,
    stride: i64,
}

/// Silero VAD's 16 kHz model, on the host.
#[derive(Debug, Clone)]
pub struct SileroVad {
    /// The short-time Fourier transform's basis, `[2 · BINS, 1, 256]`, real rows then imaginary ones.
    basis: Tensor,
    encoder: Vec<EncoderLayer>,
    /// The LSTM's input and hidden weights, transposed to `[WIDTH, 4 · WIDTH]` with ONNX's gate order, input, output, forget, and cell.
    input_weight: Tensor,
    hidden_weight: Tensor,
    /// The input and hidden biases summed, `[1, 4 · WIDTH]`.
    bias: Tensor,
    /// The decoder's projection, `[WIDTH, 1]`, and its bias, `[1]`.
    decoder_weight: Tensor,
    decoder_bias: Tensor,
}

/// What a stream of windows carries from one to the next: the last samples of the window before, and the LSTM's state.
#[derive(Debug, Clone)]
pub struct VadState {
    context: Vec<f32>,
    hidden: Tensor,
    cell: Tensor,
}

impl VadState {
    /// Returns the state of a stream before its first window.
    ///
    /// # Errors
    ///
    /// Returns an error if the state's tensors cannot be created.
    pub fn create() -> Result<Self> {
        Ok(Self {
            context: vec![0.0; CONTEXT],
            hidden: Tensor::zeroed_on(Device::Cpu, DataType::F32, Shape::create([1, WIDTH])?)?,
            cell: Tensor::zeroed_on(Device::Cpu, DataType::F32, Shape::create([1, WIDTH])?)?,
        })
    }
}

impl SileroVad {
    /// Loads the 16 kHz model from `source`'s `onnx/model.onnx`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read, or if it is not Silero VAD v5's graph.
    pub fn load(source: &Source) -> Result<Self> {
        let path = source.directory().join(FILE);
        let file = Model::from_path(&path).map_err(Error::onnx_file(&path))?;
        let graph = sixteen_kilohertz(&file.graph)?;
        let tensor = |suffix: &str| -> Result<Tensor> {
            let found = graph
                .initializers
                .iter()
                .find(|initializer| initializer.name.ends_with(suffix))
                .ok_or_else(|| missing(suffix))?;
            host_tensor(found)
        };
        let encoder = [1, 2, 2, 1]
            .into_iter()
            .enumerate()
            .map(|(index, stride)| {
                Ok(EncoderLayer {
                    weight: tensor(&format!("encoder.{index}.reparam_conv.weight"))?,
                    bias: tensor(&format!("encoder.{index}.reparam_conv.bias"))?,
                    stride,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // The LSTM's weights are constants the export named by the node that made them, so they are found through its inputs.
        let lstm = graph
            .nodes
            .iter()
            .find(|node| node.operator_type == "LSTM")
            .ok_or_else(|| missing("LSTM"))?;
        let input = |index: usize| -> Result<Tensor> {
            let name = lstm
                .inputs
                .get(index)
                .ok_or_else(|| missing("LSTM input"))?;
            let found = graph
                .initializers
                .iter()
                .find(|initializer| &initializer.name == name)
                .ok_or_else(|| missing(name))?;
            host_tensor(found)
        };
        let gates = 4 * WIDTH;
        let input_weight = input(1)?
            .reshape([gates, WIDTH])?
            .transpose(0, 1)?
            .contiguous()?;
        let hidden_weight = input(2)?
            .reshape([gates, WIDTH])?
            .transpose(0, 1)?
            .contiguous()?;
        let biases = input(3)?.reshape([2, gates])?;
        let bias = biases.narrow(0, 0, 1)?.add(&biases.narrow(0, 1, 1)?)?;

        Ok(Self {
            basis: tensor("stft.forward_basis_buffer")?,
            encoder,
            input_weight,
            hidden_weight,
            bias,
            decoder_weight: tensor("decoder.decoder.2.weight")?
                .reshape([1, WIDTH])?
                .transpose(0, 1)?,
            decoder_bias: tensor("decoder.decoder.2.bias")?,
        })
    }

    /// Returns the probability that `window`, [`WINDOW`] samples at [`SAMPLE_RATE`], holds speech, and advances `state` past it.
    ///
    /// # Errors
    ///
    /// Returns an error if `window` is not [`WINDOW`] samples long, or if a tensor operation fails.
    pub fn probability(&self, window: &[f32], state: &mut VadState) -> Result<f32> {
        if window.len() != WINDOW {
            return Err(Error::invalid(format!(
                "a voice activity window of {} samples, rather than {WINDOW}",
                window.len()
            )));
        }

        let samples = [state.context.as_slice(), window].concat();
        state.context = window[WINDOW - CONTEXT..].to_vec();
        let length =
            i64::try_from(samples.len()).map_err(|_| Error::invalid("a window too long"))?;
        let input = Tensor::from_vec(Device::Cpu, Shape::create([1, length])?, samples)?
            .pad_reflect(1, 0, STFT_PADDING)?
            .unsqueeze(1)?;
        let spectrum = input.convolution1d(&self.basis, None, STFT_HOP, Padding::ZERO, 1, 1)?;
        let real = spectrum.narrow(1, 0, BINS)?;
        let imaginary = spectrum.narrow(1, BINS, BINS)?;
        let mut features = real.mul(&real)?.add(&imaginary.mul(&imaginary)?)?.sqrt()?;
        for layer in &self.encoder {
            features = features
                .convolution1d(
                    &layer.weight,
                    Some(&layer.bias),
                    layer.stride,
                    Padding::new(1, 1),
                    1,
                    1,
                )?
                .relu()?;
        }

        let input = features.reshape([1, WIDTH])?;
        let gates = input
            .matmul(&self.input_weight)?
            .add(&state.hidden.matmul(&self.hidden_weight)?)?
            .add(&self.bias)?;
        let gate = |index: i64| gates.narrow(1, index * WIDTH, WIDTH);
        let cell = gate(2)?
            .sigmoid()?
            .mul(&state.cell)?
            .add(&gate(0)?.sigmoid()?.mul(&gate(3)?.tanh()?)?)?;
        let hidden = gate(1)?.sigmoid()?.mul(&cell.tanh()?)?;
        let probability = hidden
            .relu()?
            .matmul(&self.decoder_weight)?
            .add(&self.decoder_bias)?
            .sigmoid()?
            .to_vec::<f32>()?;
        state.hidden = hidden;
        state.cell = cell;
        probability
            .first()
            .copied()
            .ok_or_else(|| Error::invalid("a voice activity probability of no value"))
    }
}

/// Returns the branch of the file's top `If` that the graph takes at 16 kHz: the `then` branch when the condition is `sr` equal to 16000.
fn sixteen_kilohertz(graph: &Graph) -> Result<&Graph> {
    let condition = graph
        .nodes
        .iter()
        .find(|node| node.operator_type == "Equal")
        .ok_or_else(|| missing("the sample rate's comparison"))?;
    let rate = condition
        .inputs
        .iter()
        .find_map(|name| {
            graph
                .initializers
                .iter()
                .chain(graph.nodes.iter().filter_map(constant))
                .find(|tensor| &tensor.name == name)
        })
        .ok_or_else(|| missing("the compared sample rate"))?;
    let branch = if integers(rate)?.first() == Some(&i64::from(SAMPLE_RATE)) {
        "then_branch"
    } else {
        "else_branch"
    };
    graph
        .nodes
        .iter()
        .find(|node| node.operator_type == "If")
        .and_then(|node| {
            node.attributes
                .iter()
                .find(|attribute| attribute.name == branch)
        })
        .and_then(|attribute| match &attribute.value {
            AttributeValue::Graph(graph) => Some(graph),
            _ => None,
        })
        .ok_or_else(|| missing("the 16 kHz branch"))
}

/// Returns the tensor of a `Constant` node's `value`, named by the node's output.
fn constant(node: &Node) -> Option<&OnnxTensor> {
    (node.operator_type == "Constant")
        .then(|| {
            node.attributes
                .iter()
                .find(|attribute| attribute.name == "value")
        })
        .flatten()
        .and_then(|attribute| match &attribute.value {
            AttributeValue::Tensor(tensor) => Some(tensor),
            _ => None,
        })
}

fn integers(tensor: &OnnxTensor) -> Result<Vec<i64>> {
    Ok(match &tensor.data {
        Some(TensorElements::Int64(values)) => values.clone(),
        Some(TensorElements::Raw(bytes)) if tensor.data_type == OnnxDataType::Int64 => bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|bytes| i64::from_le_bytes(*bytes))
            .collect(),
        _ => return Err(missing("an integer constant")),
    })
}

/// Returns an initializer of `FLOAT`s as a host tensor of its shape.
fn host_tensor(tensor: &OnnxTensor) -> Result<Tensor> {
    if tensor.data_type != OnnxDataType::Float {
        return Err(Error::unsupported(format!(
            "Silero VAD's {} of type {:?}, rather than FLOAT",
            tensor.name, tensor.data_type
        )));
    }

    let values = match &tensor.data {
        Some(TensorElements::Float(values)) => values.clone(),
        Some(TensorElements::Raw(bytes)) => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect(),
        _ => return Err(missing(&tensor.name)),
    };
    Ok(Tensor::from_vec(
        Device::Cpu,
        Shape::create(tensor.dimensions.clone())?,
        values,
    )?)
}

fn missing(what: &str) -> Error {
    Error::unsupported(format!(
        "an ONNX file without Silero VAD v5's {what}, which it reads"
    ))
}

/// When speech starts, and how long a silence ends it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TurnSettings {
    /// The probability of speech, in [0, 1], at which a window counts as speech.
    pub threshold: f32,
    /// The silence after speech that ends a turn, in milliseconds.
    pub silence_ms: u64,
}

impl Default for TurnSettings {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            silence_ms: 500,
        }
    }
}

/// Where speech started or stopped, in milliseconds from the stream's start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnEvent {
    /// The first window of speech starts here.
    Started(u64),
    /// The silence after speech, which lasted the settings' `silence_ms`, starts here.
    Stopped(u64),
}

/// A stream's turns: the stream resampled to [`SAMPLE_RATE`] as it arrives, each window's probability of speech, and where speech starts and stops.
#[derive(Debug, Clone)]
pub struct TurnDetector {
    detector: Arc<SileroVad>,
    settings: TurnSettings,
    state: VadState,
    resampler: StreamingResampler,
    /// Resampled samples short of a window.
    pending: Vec<f32>,
    /// The windows read so far.
    windows: u64,
    speaking: bool,
    /// Where the silence since the last window of speech started, while speaking.
    silence: Option<u64>,
}

impl TurnDetector {
    /// Creates a detector of a stream at `rate` hertz, by `settings`.
    ///
    /// # Errors
    ///
    /// Returns an error if the detector's state cannot be created.
    pub fn create(detector: Arc<SileroVad>, rate: u32, settings: TurnSettings) -> Result<Self> {
        Ok(Self {
            detector,
            settings,
            state: VadState::create()?,
            resampler: StreamingResampler::new(rate, SAMPLE_RATE),
            pending: Vec::new(),
            windows: 0,
            speaking: false,
            silence: None,
        })
    }

    /// Returns whether speech has started and not yet stopped.
    #[must_use]
    pub const fn is_speaking(&self) -> bool {
        self.speaking
    }

    /// Forgets the speech in progress, as a turn its listener ended does; the stream's clock runs on.
    pub const fn reset_turn(&mut self) {
        self.speaking = false;
        self.silence = None;
    }

    /// Reads the stream's next `samples` and returns where speech started and stopped in the windows they complete.
    ///
    /// # Errors
    ///
    /// Returns an error if the detector fails.
    pub fn push(&mut self, samples: &[f32]) -> Result<Vec<TurnEvent>> {
        self.pending.extend(self.resampler.push(samples));
        let mut events = Vec::new();
        let (windows, _) = self.pending.as_chunks::<WINDOW>();
        let read = windows.len() * WINDOW;
        for window in windows {
            let probability = self.detector.probability(window, &mut self.state)?;
            let start = self.windows * WINDOW_MS;
            self.windows += 1;
            let speech = probability >= self.settings.threshold;
            if !self.speaking {
                if speech {
                    self.speaking = true;
                    events.push(TurnEvent::Started(start));
                }
            } else if speech {
                self.silence = None;
            } else {
                let silence = *self.silence.get_or_insert(start);
                if start + WINDOW_MS - silence >= self.settings.silence_ms {
                    self.speaking = false;
                    self.silence = None;
                    events.push(TurnEvent::Stopped(silence));
                }
            }
        }

        self.pending.drain(..read);
        Ok(events)
    }
}

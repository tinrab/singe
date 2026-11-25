//! Writes a model as an ONNX file for one task, `model.onnx`, at ONNX's opset 28, with the inputs, outputs, and dynamic axes `optimum-onnx` gives the task, so that code written for `optimum`'s exports runs Singe's. The export holds the model alone; as with `optimum`, the caller copies the tokenizer's or processor's files beside it.
//! A model builds on [`Device::Meta`] with no example input, so exporting runs no kernel and needs no GPU.

use std::{fs, path::Path};

use singe::{
    DataType, Device, Error as SingeError, Result as SingeResult, Shape, Tensor,
    graph::{Graph, GraphInputSpec, GraphProgram},
    nn::{Layer, Model, ModelState, TokenBatch},
    onnx::{WeightPlacement, export_file},
    value::ValueSpec,
};

use crate::{
    error::{Error, Result},
    export::image_size,
    head::{
        AudioClassifier, CtcInput, CtcRecognizer, DepthEstimator, Head, ImageClassifier,
        ImageEncoder, MaskedLm, ObjectDetector, QuestionAnswerer, SemanticSegmenter,
        SentenceEmbedder, SequenceClassifier, TextEncoder, TokenClassifier,
    },
    model,
    processor::image::{ImageTransforms, Resize},
    source::{LoadOptions, Source},
    task::Task,
};

/// The batch and sequence length a model builds with, as `optimum`'s dummy inputs have them; each is a dynamic axis of the export.
const BATCH: i64 = 2;

const SEQUENCE: i64 = 16;

/// The samples of a clip a waveform model builds with.
const SAMPLES: i64 = 16_000;

/// The log-mel frames of a clip a spectrogram model builds with, which a subsampling by eight leaves eight rows of.
const FRAMES: i64 = 64;

/// The settings of an ONNX export.
#[derive(Debug, Clone)]
pub struct ExportOptions {
    /// The type of the weights and of floating inputs, `F32` by default.
    pub data_type: DataType,
    /// Where the weights go: by default in `model.onnx` up to 1.5 GiB of them, and in `model.onnx_data` beside it above.
    pub weights: WeightPlacement,
    /// The shape of the model's main input when the export reads that shape alone, every axis fixed: `[batch, sequence]` for text, `[batch, 3, height, width]` for images, `[batch, samples]` for waveforms, and `[batch, frames, mels]` for spectrograms.
    /// The export then runs only at that shape, and its caller resizes or crops inputs to it.
    /// `None`, the default, keeps the batch and the length or image axes dynamic.
    pub input_shape: Option<Vec<i64>>,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            data_type: DataType::F32,
            weights: WeightPlacement::Automatic,
            input_shape: None,
        }
    }
}

impl ExportOptions {
    const fn load(&self) -> LoadOptions {
        LoadOptions {
            join_projections: false,
            ..LoadOptions::new(Device::Cpu, self.data_type)
        }
    }
}

/// Writes `program`, reading `state`, as `model.onnx` under `directory`, with its weights in `model.onnx_data` when `weights` places them there.
///
/// # Errors
///
/// Returns an error if the directory or the file cannot be written.
pub fn write(
    program: &GraphProgram,
    state: &ModelState,
    directory: &Path,
    weights: WeightPlacement,
) -> Result<()> {
    fs::create_dir_all(directory).map_err(Error::io(directory))?;
    export_file(
        program,
        state.snapshot(),
        &directory.join("model.onnx"),
        weights,
    )?;
    Ok(())
}

/// Returns `source`'s program for `task`, with named inputs, outputs, and dynamic axes, and the state it reads, which [`write`] writes as an ONNX file.
///
/// Text models read `input_ids`, `attention_mask`, and `token_type_ids` when the model reads token types, each `[batch_size, sequence_length]` of `I64`.
/// Vision models read `pixel_values`, `[batch_size, 3, H, W]` at the size the processor gives every image, or with `height` and `width` axes when the processor keeps each image's aspect ratio; waveform models read `input_values`, `[batch_size, sequence_length]`, and spectrogram classifiers `input_values`, `[batch_size, frames, mels]`.
/// Outputs are named as `optimum`'s `_TASK_TO_COMMON_OUTPUTS` names them: `logits` for classifiers, taggers, fill-mask, segmentation, audio classification, and CTC, `last_hidden_state` for feature extraction, with `pooler_output` beside it for an image encoder that pools, `start_logits` and `end_logits` for question answering, `logits` and `pred_boxes` for detection, `predicted_depth` for depth, and `token_embeddings` and `sentence_embedding` for a `sentence-transformers` model.
///
/// # Errors
///
/// Returns an error if exporting does not support `task`, if the model does not load for `task` or does not build on metadata, or if ONNX has no operator for one of the model's operations.
pub fn program(
    source: &Source,
    task: Task,
    options: &ExportOptions,
) -> Result<(GraphProgram, ModelState)> {
    let load = options.load();
    let data_type = options.data_type;
    match task {
        Task::FeatureExtraction => {
            let model: TextEncoder = model::load(source, load)?;
            text(options, &model, &["last_hidden_state"], forward)
        }
        Task::TextClassification | Task::TextRanking | Task::ZeroShotClassification => {
            let model: SequenceClassifier = model::load(source, load)?;
            text(options, model.get_model()?, &["logits"], forward)
        }
        Task::TokenClassification => {
            let model: TokenClassifier = model::load(source, load)?;
            text(options, model.get_model()?, &["logits"], forward)
        }
        Task::FillMask => {
            let model: MaskedLm = model::load(source, load)?;
            text(options, model.get_model()?, &["logits"], forward)
        }
        Task::QuestionAnswering => {
            let model: QuestionAnswerer = model::load(source, load)?;
            text(
                options,
                model.get_model()?,
                &["start_logits", "end_logits"],
                |layer, batch, state| {
                    let scores = layer.forward(batch, state)?;
                    Ok(vec![scores.select(2, 0)?, scores.select(2, 1)?])
                },
            )
        }
        Task::SentenceSimilarity => {
            let model = SentenceEmbedder::load(source, load)?;
            text(
                options,
                model.get_model()?,
                &["token_embeddings", "sentence_embedding"],
                |layer, batch, state| Ok(layer.embeddings(batch, state)?.to_vec()),
            )
        }
        Task::ImageClassification => {
            let ImageClassifier(classifier) = model::load(source, load)?;
            image(
                source,
                classifier.get_model()?,
                options,
                &["logits"],
                forward,
            )
        }
        Task::ImageFeatureExtraction => {
            let model: ImageEncoder = model::load(source, load)?;
            // `pooler_output` is what `pool=True` returns, as `optimum` exports it beside the hidden states.
            image(
                source,
                &model,
                options,
                &["last_hidden_state", "pooler_output"],
                |layer, pixels, state| {
                    let hidden = layer.forward(pixels, state)?;
                    let pooled = layer.pooled(&hidden)?;
                    Ok(vec![hidden, pooled])
                },
            )
        }
        Task::ImageSegmentation => {
            let model: SemanticSegmenter = model::load(source, load)?;
            image(source, model.get_model()?, options, &["logits"], forward)
        }
        Task::DepthEstimation => {
            let model: DepthEstimator = model::load(source, load)?;
            image(
                source,
                model.get_model()?,
                options,
                &["predicted_depth"],
                forward,
            )
        }
        Task::ObjectDetection => {
            let model: ObjectDetector = model::load(source, load)?;
            // Each query's scores are its labels' and no object's, then its box.
            let scores = model.config().label_count()? + 1;
            image(
                source,
                model.get_model()?,
                options,
                &["logits", "pred_boxes"],
                |layer, pixels, state| {
                    let detections = layer.forward(pixels, state)?;
                    Ok(vec![
                        detections.narrow(2, 0, scores)?,
                        detections.narrow(2, scores, 4)?,
                    ])
                },
            )
        }
        Task::AudioClassification => {
            let model: AudioClassifier = model::load(source, load)?;
            let [frames, mels] = [model.features.frames, model.features.mels]
                .map(|count| i64::try_from(count).map_err(|_| SingeError::out_of_range("mels")));
            let spec = input_spec(
                options,
                data_type,
                &[BATCH, frames?, mels?],
                &[(0, "batch_size")],
            )?;
            single(
                model.classifier.get_model()?,
                "input_values",
                spec,
                &["logits"],
            )
        }
        Task::AutomaticSpeechRecognition => {
            if model::default_head(source)? != Head::Ctc {
                return Err(Error::unsupported(
                    "exporting a speech recognizer that generates text as onnx",
                ));
            }

            let model: CtcRecognizer = model::load(source, load)?;
            if let CtcInput::LogMel(front_end) = model.input() {
                // `transformers`' Parakeet reads `input_features`, `[B, frames, mels]`, which the layers read as `[B, mels, frames]`.
                let mels = front_end.options().mels;
                let spec = input_spec(
                    options,
                    data_type,
                    &[BATCH, FRAMES, mels],
                    &[(0, "batch_size"), (1, "sequence_length")],
                )?;
                let graph = build(
                    model.get_model()?,
                    vec![("input_features", spec)],
                    &["logits"],
                    |layer, tensors, state| forward(layer, &tensors[0].transpose(1, 2)?, state),
                )?;
                let state = model.get_model()?.state();
                return Ok((state.program(&graph)?, state.clone()));
            }

            let spec = input_spec(
                options,
                data_type,
                &[BATCH, SAMPLES],
                &[(0, "batch_size"), (1, "sequence_length")],
            )?;
            single(model.get_model()?, "input_values", spec, &["logits"])
        }
        other => Err(Error::unsupported(format!(
            "exporting task {} as onnx",
            other.id()
        ))),
    }
}

fn forward<I, L: Layer<I> + ?Sized>(
    layer: &L,
    input: &I,
    state: &ModelState,
) -> SingeResult<Vec<Tensor>> {
    Ok(vec![layer.forward(input, state)?])
}

/// Builds `model` on metadata over the named `inputs`, with `results` giving its outputs, named `outputs`, from the input tensors.
fn build<L>(
    model: &Model<L>,
    inputs: Vec<(&str, GraphInputSpec)>,
    outputs: &[&str],
    results: impl FnOnce(&L, &[Tensor], &ModelState) -> SingeResult<Vec<Tensor>>,
) -> Result<Graph> {
    let state = model.state().to(Device::Meta, None)?;
    let graph = Graph::build(inputs, state.capture_bindings()?, |tensors| {
        results(model.layer(), tensors, &state)
    })?;
    Ok(graph.with_output_names(outputs.iter().copied())?)
}

/// A text model over `input_ids`, `attention_mask`, and `token_type_ids`, which it leaves out when the model never reads it, as `optimum` does for `DistilBERT` and `RoBERTa`.
fn text<L: Layer<TokenBatch>>(
    options: &ExportOptions,
    model: &Model<L>,
    outputs: &[&str],
    results: impl Fn(&L, &TokenBatch, &ModelState) -> SingeResult<Vec<Tensor>>,
) -> Result<(GraphProgram, ModelState)> {
    let graph = |names: &[&str]| -> Result<Graph> {
        let inputs = names
            .iter()
            .map(|&name| {
                let spec = input_spec(
                    options,
                    DataType::I64,
                    &[BATCH, SEQUENCE],
                    &[(0, "batch_size"), (1, "sequence_length")],
                )?;
                Ok((name, spec))
            })
            .collect::<Result<Vec<_>>>()?;
        build(model, inputs, outputs, |layer, tensors, state| {
            let [tokens, mask, types @ ..] = tensors else {
                return Err(SingeError::missing_input("token batch"));
            };
            // A model that reads no token types gets the ids in their place, which nothing reads.
            let types = types.first().unwrap_or(tokens);
            let batch =
                TokenBatch::create(tokens.clone(), types.clone(), mask.cast(DataType::Bool)?)?;
            results(layer, &batch, state)
        })
    };
    let mut built = graph(&["input_ids", "attention_mask", "token_type_ids"])?;
    if !reads(&built, "token_type_ids") {
        built = graph(&["input_ids", "attention_mask"])?;
    }
    Ok((model.state().program(&built)?, model.state().clone()))
}

/// An input of `data_type` and `shape` whose `axes` are dynamic, or of [`ExportOptions::input_shape`] with every axis fixed when that is set.
fn input_spec(
    options: &ExportOptions,
    data_type: DataType,
    shape: &[i64],
    axes: &[(usize, &str)],
) -> Result<GraphInputSpec> {
    if let Some(fixed) = &options.input_shape {
        if fixed.len() != shape.len() {
            return Err(Error::unsupported(format!(
                "an input shape of {} axes for an input of {}",
                fixed.len(),
                shape.len()
            )));
        }

        return Ok(GraphInputSpec::new(ValueSpec::new(
            data_type,
            Shape::create(fixed.clone())?,
        )));
    }

    axes.iter().try_fold(
        GraphInputSpec::new(ValueSpec::new(data_type, Shape::create(shape.to_vec())?)),
        |spec, &(axis, name)| Ok(spec.with_dimension(axis, name, Some(1), None)?),
    )
}

/// Whether any operation or output of `graph` reads its input `name`.
fn reads(graph: &Graph, name: &str) -> bool {
    graph
        .input_values()
        .filter(|&(_, input)| input == name)
        .any(|(value, _)| {
            graph.output_values().contains(&value)
                || graph
                    .operations()
                    .iter()
                    .any(|operation| operation.inputs().contains(&value))
        })
}

/// An image model over `pixel_values`, `[batch_size, 3, H, W]`, at the size `source`'s processor gives every image.
fn image<L: Layer>(
    source: &Source,
    model: &Model<L>,
    options: &ExportOptions,
    outputs: &[&str],
    results: impl FnOnce(&L, &Tensor, &ModelState) -> SingeResult<Vec<Tensor>>,
) -> Result<(GraphProgram, ModelState)> {
    let processor = ImageTransforms::load(source)?;
    let (size, axes) = match image_size(&processor) {
        Ok(size) => (size, &[(0, "batch_size")][..]),
        Err(_) => (
            nominal_size(processor.resize)?,
            &[(0, "batch_size"), (2, "height"), (3, "width")][..],
        ),
    };
    let spec = input_spec(
        options,
        options.data_type,
        &[BATCH, 3, size[0], size[1]],
        axes,
    )?;

    let graph = build(
        model,
        vec![("pixel_values", spec)],
        outputs,
        |layer, tensors, state| results(layer, &tensors[0], state),
    )?;
    Ok((model.state().program(&graph)?, model.state().clone()))
}

/// The size a model whose processor keeps each image's aspect ratio builds at, which its `height` and `width` axes vary from: the square of the shorter side it resizes to.
fn nominal_size(resize: Option<Resize>) -> Result<[i64; 2]> {
    let round = |side: i64, multiple: i64| side / multiple * multiple;
    match resize {
        Some(Resize::ShortestEdge(side) | Resize::Longest(side)) => Ok([side; 2]),
        Some(Resize::Bounded {
            shortest, multiple, ..
        }) => Ok([round(shortest, multiple.max(1)); 2]),
        Some(
            Resize::Fitted { height, width, .. }
            | Resize::Thumbnail { height, width }
            | Resize::Exact { height, width },
        ) => Ok([height, width]),
        None => Err(Error::unsupported(
            "exporting an image model whose processor does not resize",
        )),
    }
}

/// A model of one input, `name`, whose forward gives its one output.
fn single<L: Layer>(
    model: &Model<L>,
    name: &str,
    spec: GraphInputSpec,
    outputs: &[&str],
) -> Result<(GraphProgram, ModelState)> {
    let graph = build(
        model,
        vec![(name, spec)],
        outputs,
        |layer, tensors, state| forward(layer, &tensors[0], state),
    )?;
    Ok((model.state().program(&graph)?, model.state().clone()))
}

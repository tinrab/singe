// Writes a model as CUDA C++ programs for one task, with what runs them with nothing of Singe: a generation loop for a task that generates, and a runner for one that maps inputs to outputs.

mod tokenizer;

use std::path::Path;

use serde_json::{Map, Value};
use singe::{
    DataType, Device, Error as SingeError, Shape, Tensor,
    backend::cuda::export::{CudaExport, Decoding, GpuArchitecture, Library},
    generation::GenerationOptions,
    graph::Graph,
    nn::{Layer, Model, TokenBatch},
    tensor::SamplingOptions,
};

pub use self::tokenizer::{PromptTemplate, loop_tokenizer};
use crate::{
    error::{Error, Result},
    head::{
        Autoencoder, CausalLm, Head, ImageEncoder, ImageTextToText, LoopLimits, LoopSteps,
        MaskedLm, QuestionAnswerer, SequenceClassifier, SpeechToText, TokenClassifier,
    },
    model,
    processing::{
        chat::{ChatTemplate, Content, Message, Part},
        image::{ImageTransforms, Resize},
    },
    source::{LoadOptions, Source},
    task::{
        Pipeline, Task,
        text::{
            feature_extraction::FeatureExtractionPipeline,
            sentence_similarity::SentenceSimilarityPipeline,
        },
    },
};

// How a model exports: the target and libraries, the type of its weights, and the sizes its programs fix.
#[derive(Debug, Clone)]
pub struct ExportOptions {
    pub architecture: GpuArchitecture,
    pub libraries: Vec<Library>,
    pub data_type: DataType,
    // Where the model loads, whose captures run eagerly on it.
    pub device: Device,
    // The tokens a generating model's sequence holds, prompt included.
    pub capacity: i64,
    // The rows of a generating model's prompt step, beside its one-token step.
    pub prompt_rows: usize,
    // The sequences a generating model's loop runs at once, one per row.
    pub rows: usize,
    // How a generating model chooses each token, or the most likely one for `None`; the loop takes the seed.
    pub sampling: Option<SamplingOptions>,
    // The seconds of audio a speech model's encode step reads.
    pub audio_seconds: i64,
    // The tokens a text encoder's sequence holds.
    pub sequence_length: i64,
    // The side of a square image an autoencoder reads.
    pub image_size: i64,
    // The most patches of an image a vision-language model's encode step reads.
    pub image_patches: Option<i64>,
}

impl ExportOptions {
    #[must_use]
    pub fn new(architecture: GpuArchitecture) -> Self {
        Self {
            architecture,
            libraries: Library::ALL.to_vec(),
            data_type: DataType::Bf16,
            device: Device::Cuda(0),
            capacity: 1024,
            prompt_rows: 64,
            rows: 1,
            sampling: None,
            audio_seconds: 30,
            sequence_length: 128,
            image_size: 256,
            image_patches: None,
        }
    }

    fn export(&self) -> CudaExport {
        CudaExport::new(self.architecture).with_libraries(self.libraries.clone())
    }

    const fn load(&self) -> LoadOptions {
        LoadOptions::new(self.device, self.data_type)
    }

    const fn limits(&self) -> LoopLimits {
        LoopLimits {
            capacity: self.capacity,
            prompt_rows: self.prompt_rows,
            rows: self.rows,
            audio_seconds: self.audio_seconds,
            image_patches: self.image_patches,
        }
    }
}

// What an export wrote, and the files that run it, each of which `model.json` or its own manifest says how to build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Exported {
    // `model_generate`, a generation loop over a program set.
    Generator,
    // `model_run`, which maps input files to output files.
    Runner,
    // `encoder_run` and `decoder_run`, an autoencoder's halves.
    Autoencoder,
}

// The task a model exports for when none is named: the task of the head its weights hold.
pub fn default_task(source: &Source) -> Result<Task> {
    Ok(match model::head(source)? {
        Head::CausalLm => Task::TextGeneration,
        Head::SpeechToText | Head::Ctc => Task::AutomaticSpeechRecognition,
        Head::ImageTextToText => Task::ImageTextToText,
        // A `sentence-transformers` snapshot pools its encoder's states into one vector.
        Head::TextEncoder if source.has("modules.json") => Task::SentenceSimilarity,
        Head::TextEncoder => Task::FeatureExtraction,
        Head::ImageEncoder => Task::ImageFeatureExtraction,
        Head::Autoencoder | Head::SuperResolution => Task::ImageToImage,
        Head::SequenceClassification => Task::TextClassification,
        Head::TokenClassification => Task::TokenClassification,
        Head::QuestionAnswering => Task::QuestionAnswering,
        Head::MaskedLm => Task::FillMask,
        Head::ImageClassification => Task::ImageClassification,
        Head::VideoClassification => Task::VideoClassification,
        Head::ObjectDetection => Task::ObjectDetection,
        Head::ZeroShotImageClassification => Task::ZeroShotImageClassification,
        Head::ZeroShotObjectDetection => Task::ZeroShotObjectDetection,
        Head::DepthEstimation => Task::DepthEstimation,
        Head::ImageSegmentation => Task::ImageSegmentation,
        Head::KeypointDetection => Task::KeypointDetection,
        Head::KeypointMatching => Task::KeypointMatching,
        Head::MaskGeneration => Task::MaskGeneration,
        Head::Seq2SeqLm => Task::Summarization,
        Head::AudioClassification => Task::AudioClassification,
        Head::AudioCodec => Task::AudioToAudio,
        Head::ZeroShotAudioClassification => Task::ZeroShotAudioClassification,
        Head::TextToSpeech => Task::TextToAudio,
        Head::DocumentRetrieval => Task::VisualDocumentRetrieval,
        Head::TimeSeriesForecasting => Task::TimeSeriesForecasting,
        Head::Trajectory => Task::ReinforcementLearning,
        Head::TabularClassification => Task::TabularClassification,
        Head::TabularRegression => Task::TabularRegression,
        Head::Denoiser => Task::UnconditionalImageGeneration,
        Head::ConditionalDenoiser => Task::TextToImage,
    })
}

// Writes `source`'s programs for `task` to `directory`.
pub fn export(
    source: &Source,
    task: Task,
    directory: impl AsRef<Path>,
    options: &ExportOptions,
) -> Result<Exported> {
    let directory = directory.as_ref();
    let load = options.load();
    match task {
        Task::TextGeneration => {
            let model: Box<dyn CausalLm> = model::load(source, load)?;
            let steps = model.loop_steps(&options.limits(), &choices(source, options)?)?;
            let template = prompt_template(source, None)?;
            write_generator(&steps, source, template, directory, options)?;
            Ok(Exported::Generator)
        }
        Task::AutomaticSpeechRecognition => {
            let model: Box<dyn SpeechToText> = model::load(source, load)?;
            let steps = model.loop_steps(&options.limits(), &choices(source, options)?)?;
            write_generator(&steps, source, None, directory, options)?;
            Ok(Exported::Generator)
        }
        Task::ImageTextToText => {
            let model: Box<dyn ImageTextToText> = model::load(source, load)?;
            let steps = model.loop_steps(&options.limits(), &choices(source, options)?)?;
            // A prompt asks about the image, whose rows stand where the template's image token does.
            let template = prompt_template(source, Some(model.image_token()))?;
            write_generator(&steps, source, template, directory, options)?;
            Ok(Exported::Generator)
        }
        Task::FeatureExtraction => {
            let pipeline = FeatureExtractionPipeline::load(source, load)?;
            token_runner(pipeline.model(), directory, options)
        }
        Task::TextClassification | Task::TextRanking | Task::ZeroShotClassification => {
            let model: SequenceClassifier = model::load(source, load)?;
            token_runner(model.model(), directory, options)
        }
        Task::TokenClassification => {
            let model: TokenClassifier = model::load(source, load)?;
            token_runner(model.model(), directory, options)
        }
        Task::QuestionAnswering => {
            let model: QuestionAnswerer = model::load(source, load)?;
            token_runner(model.model(), directory, options)
        }
        Task::FillMask => {
            let model: MaskedLm = model::load(source, load)?;
            token_runner(model.model(), directory, options)
        }
        Task::SentenceSimilarity => {
            let pipeline = SentenceSimilarityPipeline::load(source, load)?;
            token_runner(pipeline.model(), directory, options)
        }
        Task::ImageFeatureExtraction => {
            let model: ImageEncoder = model::load(source, load)?;
            let [height, width] = image_size(&ImageTransforms::load(source)?)?;
            let images = Tensor::zeroed_on(
                options.device,
                options.data_type,
                Shape::create([1, 3, height, width])?,
            )?;
            let (graph, _) = model.capture_graph("images", images)?;
            runner(&graph, directory, options)?;
            Ok(Exported::Runner)
        }
        Task::ImageToImage => {
            autoencoder(source, directory, options)?;
            Ok(Exported::Autoencoder)
        }
        other => Err(Error::UnsupportedConfig(format!(
            "exporting task {}",
            other.id()
        ))),
    }
}

// What a generating model's steps choose and when its loop stops: the source's stop tokens, and the sampling `options` asks for.
fn choices(source: &Source, options: &ExportOptions) -> Result<GenerationOptions> {
    Ok(GenerationOptions {
        stop_tokens: source.stop_tokens()?,
        sampling: options.sampling,
        ..GenerationOptions::default()
    })
}

// The text a prompt goes between in the source's chat template, for a user turn of one prompt, read after an image when `image` names the image token, without thinking.
// A source without a chat template reads prompts as they are.
fn prompt_template(source: &Source, image: Option<i64>) -> Result<Option<PromptTemplate>> {
    const MARKER: &str = "\u{e000}";
    let Some(template) = ChatTemplate::load(source)? else {
        return Ok(None);
    };
    let content = match image {
        Some(_) => Content::Parts(vec![
            Part::Image {
                image: String::new(),
            },
            Part::Text {
                text: MARKER.to_owned(),
            },
        ]),
        None => Content::Text(MARKER.to_owned()),
    };
    let arguments = Map::from_iter([("enable_thinking".to_owned(), Value::Bool(false))]);
    let text = template.render(&[Message::new("user", content)], true, &arguments)?;
    let (before, after) = text
        .split_once(MARKER)
        .ok_or_else(|| Error::unsupported("a chat template that drops the user's text"))?;
    Ok(Some(PromptTemplate {
        before: before.to_owned(),
        after: after.to_owned(),
        placeholder: image,
    }))
}

fn write_generator(
    steps: &LoopSteps,
    source: &Source,
    template: Option<PromptTemplate>,
    directory: &Path,
    options: &ExportOptions,
) -> Result<()> {
    let graphs = steps
        .encode
        .as_ref()
        .map(|encode| ("encode", encode))
        .into_iter()
        .chain([("decode", &steps.step.graph)])
        .chain(
            steps
                .prompt
                .as_ref()
                .map(|prompt| ("prompt", &prompt.graph)),
        )
        .collect::<Vec<_>>();
    options
        .export()
        .with_decoding(Decoding {
            layout: steps.step.layout,
            stop_tokens: steps.generation.stop_tokens.clone(),
            tokenizer: loop_tokenizer(source, template)?,
        })
        .export_steps(&graphs)?
        .write_to(directory, "model")?;
    Ok(())
}

// A batch of one sequence of `options.sequence_length` tokens, which a text encoder's program reads.
fn token_batch(options: &ExportOptions) -> Result<TokenBatch> {
    let tokens = vec![
        0;
        usize::try_from(options.sequence_length)
            .map_err(|_| SingeError::out_of_range("sequence length"))?
    ];
    Ok(TokenBatch::from_sequences(
        options.device,
        &[tokens.as_slice()],
    )?)
}

// The size an image processor gives every image, which an image encoder's program reads.
fn image_size(processor: &ImageTransforms) -> Result<[i64; 2]> {
    let config = processor.config();
    match (config.crop, config.resize) {
        (Some(crop), _) => Ok(crop),
        (None, Some(Resize::Exact { height, width })) => Ok([height, width]),
        _ => Err(Error::unsupported(
            "exporting an image encoder whose processor keeps the aspect ratio",
        )),
    }
}

// An autoencoder's halves, each a program with its runner: `encoder` from images, and `decoder` from the latents it gives.
fn autoencoder(source: &Source, directory: &Path, options: &ExportOptions) -> Result<()> {
    let model: Autoencoder = model::load(source, options.load())?;
    let size = options.image_size;
    let images = Tensor::zeroed_on(
        options.device,
        options.data_type,
        Shape::create([1, 3, size, size])?,
    )?;
    let latents = model.encode(&images)?;
    for (prefix, half, input) in [
        ("encoder", model.encoder(), images),
        ("decoder", model.decoder(), latents),
    ] {
        let (graph, _) = half.capture_graph("input", input)?;
        options
            .export()
            .export(&graph)?
            .write_to(directory, prefix)?;
    }
    Ok(())
}

// A runner of a model over a batch of one sequence of `options.sequence_length` tokens, as an encoder's heads read it.
fn token_runner<L: Layer<TokenBatch>>(
    model: &Model<L>,
    directory: &Path,
    options: &ExportOptions,
) -> Result<Exported> {
    let (graph, _) = model.capture_group(&token_batch(options)?)?;
    runner(&graph, directory, options)?;
    Ok(Exported::Runner)
}

// One program and its runner.
fn runner(graph: &Graph, directory: &Path, options: &ExportOptions) -> Result<()> {
    Ok(options
        .export()
        .export(graph)?
        .write_to(directory, "model")?)
}

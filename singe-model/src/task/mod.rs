//! One module per task, under the Hub's category of tasks, named for its id as the Hugging Face Hub and `transformers`' pipelines name it.
//! Each task has a pipeline: its inputs and outputs, what turns them into the model's and back, and the head it asks the registry for, or the `diffusers` components it loads itself.
//! A task `transformers` adds goes under the category of its nearest Hub task.

pub mod audio;
pub mod diffusion;
pub mod inputs;
pub mod multimodal;
pub mod outputs;
pub mod ranking;
pub mod reinforcement_learning;
pub mod seq2seq;
pub mod tabular;
pub mod text;
pub mod vision;

use std::{any::Any, result::Result as StdResult, slice};

use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;
use singe::Error as SingeError;

use crate::{
    error::{Error, Result},
    head::Head,
    model,
    source::{LoadOptions, Source},
};

/// A task's pipeline: what it reads, how it runs, and what it returns, in the field names `transformers`' pipeline returns.
pub trait Pipeline: TaskPipeline + Sized {
    type Input: DeserializeOwned;

    type Parameters: DeserializeOwned + Default;

    type Output: Serialize;

    fn load(source: &Source, options: LoadOptions) -> Result<Self>;

    fn run_one(&self, input: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output>;

    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        inputs
            .iter()
            .map(|input| self.run_one(input, parameters))
            .collect()
    }
}

pub trait TaskPipeline {
    const TASK: Task;
}

macro_rules! tasks {
    ($($task:ident($id:literal, $pipeline:ty) = [$($head:ident),* $(,)?]),* $(,)?) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum Task {
            $($task),*
        }

        impl Task {
            pub const ALL: &[Self] = &[$(Self::$task),*];

            /// Returns the task's id, as `pipeline(task)` takes it.
            #[must_use]
            pub const fn id(self) -> &'static str {
                match self {
                    $(Self::$task => $id),*
                }
            }

            #[must_use]
            pub const fn heads(self) -> &'static [Head] {
                match self {
                    $(Self::$task => &[$(Head::$head),*]),*
                }
            }
        }

        $(
            impl TaskPipeline for $pipeline {
                const TASK: Task = Task::$task;
            }
        )*

        /// Loads `task`'s pipeline from `source`, to run many times through JSON or, downcast, through its own types.
        pub fn load(task: Task, source: &Source, options: LoadOptions) -> Result<Box<dyn AnyPipeline>> {
            match task {
                $(Task::$task => Ok(Box::new(<$pipeline as Pipeline>::load(source, options)?))),*
            }
        }
    };
}

tasks! {
    FeatureExtraction("feature-extraction", text::feature_extraction::FeatureExtractionPipeline) = [TextEncoder],
    FillMask("fill-mask", text::fill_mask::FillMaskPipeline) = [MaskedLm],
    QuestionAnswering("question-answering", text::question_answering::QuestionAnsweringPipeline) = [QuestionAnswering],
    SentenceSimilarity("sentence-similarity", text::sentence_similarity::SentenceSimilarityPipeline) = [TextEncoder],
    Summarization("summarization", text::summarization::SummarizationPipeline) = [Seq2SeqLm],
    TableQuestionAnswering("table-question-answering", text::table_question_answering::TableQuestionAnsweringPipeline) = [Seq2SeqLm],
    TextClassification("text-classification", text::text_classification::TextClassificationPipeline) = [SequenceClassification],
    TextGeneration("text-generation", text::text_generation::TextGenerationPipeline) = [CausalLm],
    TextRanking("text-ranking", text::text_ranking::TextRankingPipeline) = [SequenceClassification],
    TokenClassification("token-classification", text::token_classification::TokenClassificationPipeline) = [TokenClassification],
    Translation("translation", text::translation::TranslationPipeline) = [Seq2SeqLm],
    ZeroShotClassification("zero-shot-classification", text::zero_shot_classification::ZeroShotClassificationPipeline) = [SequenceClassification],
    DepthEstimation("depth-estimation", vision::depth_estimation::DepthEstimationPipeline) = [DepthEstimation],
    ImageClassification("image-classification", vision::image_classification::ImageClassificationPipeline) = [ImageClassification],
    ImageFeatureExtraction("image-feature-extraction", vision::image_feature_extraction::ImageFeatureExtractionPipeline) = [ImageEncoder],
    ImageSegmentation("image-segmentation", vision::image_segmentation::ImageSegmentationPipeline) = [ImageSegmentation, ConceptSegmentation],
    ImageToImage("image-to-image", vision::image_to_image::ImageToImagePipeline) = [SuperResolution, Autoencoder],
    ImageToText("image-to-text", vision::image_to_text::ImageToTextPipeline) = [Seq2SeqLm],
    ImageToVideo("image-to-video", vision::image_to_video::ImageToVideoPipeline) = [],
    KeypointDetection("keypoint-detection", vision::keypoint_detection::KeypointDetectionPipeline) = [KeypointDetection],
    KeypointMatching("keypoint-matching", vision::keypoint_matching::KeypointMatchingPipeline) = [KeypointMatching],
    MaskGeneration("mask-generation", vision::mask_generation::MaskGenerationPipeline) = [MaskGeneration],
    MaskTracking("mask-tracking", vision::mask_tracking::MaskTrackingPipeline) = [MaskTracking],
    ObjectDetection("object-detection", vision::object_detection::ObjectDetectionPipeline) = [ObjectDetection],
    VideoClassification("video-classification", vision::video_classification::VideoClassificationPipeline) = [VideoClassification],
    VideoFeatureExtraction("video-feature-extraction", vision::video_feature_extraction::VideoFeatureExtractionPipeline) = [VideoFeatureExtraction],
    TextToImage("text-to-image", vision::text_to_image::TextToImagePipeline) = [ConditionalDenoiser, ImageGeneration],
    TextToVideo("text-to-video", vision::text_to_video::TextToVideoPipeline) = [],
    UnconditionalImageGeneration("unconditional-image-generation", vision::unconditional_image_generation::UnconditionalImageGenerationPipeline) = [Denoiser],
    VideoToVideo("video-to-video", vision::video_to_video::VideoToVideoPipeline) = [],
    ZeroShotImageClassification("zero-shot-image-classification", vision::zero_shot_image_classification::ZeroShotImageClassificationPipeline) = [ZeroShotImageClassification],
    ZeroShotObjectDetection("zero-shot-object-detection", vision::zero_shot_object_detection::ZeroShotObjectDetectionPipeline) = [ZeroShotObjectDetection],
    TextTo3d("text-to-3d", vision::text_to_3d::TextTo3dPipeline) = [],
    ImageTo3d("image-to-3d", vision::image_to_3d::ImageTo3dPipeline) = [],
    AudioClassification("audio-classification", audio::audio_classification::AudioClassificationPipeline) = [AudioClassification],
    AudioToAudio("audio-to-audio", audio::audio_to_audio::AudioToAudioPipeline) = [AudioCodec, Vocoder],
    AutomaticSpeechRecognition("automatic-speech-recognition", audio::automatic_speech_recognition::AutomaticSpeechRecognitionPipeline) = [SpeechToText, Ctc],
    TextToAudio("text-to-audio", audio::text_to_audio::TextToAudioPipeline) = [TextToSpeech, TextToMusic, AudioGeneration],
    ZeroShotAudioClassification("zero-shot-audio-classification", audio::zero_shot_audio_classification::ZeroShotAudioClassificationPipeline) = [ZeroShotAudioClassification],
    TabularClassification("tabular-classification", tabular::tabular_classification::TabularClassificationPipeline) = [TabularClassification],
    TabularRegression("tabular-regression", tabular::tabular_regression::TabularRegressionPipeline) = [TabularRegression],
    TimeSeriesForecasting("time-series-forecasting", tabular::time_series_forecasting::TimeSeriesForecastingPipeline) = [TimeSeriesForecasting],
    ReinforcementLearning("reinforcement-learning", reinforcement_learning::reinforcement_learning::ReinforcementLearningPipeline) = [Trajectory],
    ImageTextToText("image-text-to-text", multimodal::image_text_to_text::ImageTextToTextPipeline) = [ImageTextToText],
    AnyToAny("any-to-any", multimodal::any_to_any::AnyToAnyPipeline) = [ImageTextToText],
    AudioTextToText("audio-text-to-text", multimodal::audio_text_to_text::AudioTextToTextPipeline) = [SpeechToText],
    Decision("decision", multimodal::decision::DecisionPipeline) = [CausalLm],
    DocumentQuestionAnswering("document-question-answering", multimodal::document_question_answering::DocumentQuestionAnsweringPipeline) = [Seq2SeqLm],
    ImageTextToImage("image-text-to-image", multimodal::image_text_to_image::ImageTextToImagePipeline) = [ConditionalDenoiser],
    ImageTextToVideo("image-text-to-video", multimodal::image_text_to_video::ImageTextToVideoPipeline) = [],
    VideoTextToText("video-text-to-text", multimodal::video_text_to_text::VideoTextToTextPipeline) = [ImageTextToText],
    VisualDocumentRetrieval("visual-document-retrieval", multimodal::visual_document_retrieval::VisualDocumentRetrievalPipeline) = [DocumentRetrieval],
    VisualQuestionAnswering("visual-question-answering", multimodal::visual_question_answering::VisualQuestionAnsweringPipeline) = [ImageTextToText],
}

impl Task {
    #[must_use]
    pub const fn can_run_external(self) -> bool {
        matches!(
            self,
            Self::FeatureExtraction
                | Self::FillMask
                | Self::QuestionAnswering
                | Self::SentenceSimilarity
                | Self::TextClassification
                | Self::TextRanking
                | Self::TokenClassification
                | Self::ZeroShotClassification
                | Self::ImageClassification
                | Self::ImageFeatureExtraction
                | Self::ImageSegmentation
                | Self::ObjectDetection
                | Self::DepthEstimation
                | Self::AudioClassification
                | Self::AutomaticSpeechRecognition
        )
    }

    /// Returns the task `id` names, as a task id or one of the aliases the Hub or `transformers` gives a task.
    ///
    /// # Errors
    ///
    /// Returns an error if `id` names no task.
    pub fn from_id(id: &str) -> Result<Self> {
        let id = match id {
            "sentiment-analysis" => "text-classification",
            "ner" => "token-classification",
            "text-to-speech" => "text-to-audio",
            "vqa" => "visual-question-answering",
            other => other,
        };
        Self::ALL
            .iter()
            .copied()
            .find(|task| task.id() == id)
            .ok_or_else(|| Error::invalid(format!("task {id}")))
    }

    pub fn default_for(source: &Source) -> Result<Self> {
        // A `diffusers` pipeline's class names its task, as its components' heads do not.
        match source.kind() {
            "StableDiffusionPipeline" => return Ok(Self::TextToImage),
            "StableDiffusionInstructPix2PixPipeline" => return Ok(Self::ImageTextToImage),
            "DDPMPipeline" => return Ok(Self::UnconditionalImageGeneration),
            _ => {}
        }

        let head = model::default_head(source)?;
        if head == Head::TextEncoder && source.has("modules.json") {
            return Ok(Self::SentenceSimilarity);
        }

        Self::ALL
            .iter()
            .copied()
            .find(|task| task.heads().contains(&head))
            .ok_or_else(|| Error::unsupported(format!("head {}, which no task runs", head.id())))
    }
}

pub fn parameters<T: DeserializeOwned + Default>(parameters: &Value) -> Result<T> {
    if parameters.is_null() {
        Ok(T::default())
    } else {
        serde_json::from_value(parameters.clone())
            .map_err(Error::json_value("parse the pipeline's parameters"))
    }
}

/// A loaded pipeline of any task, run on JSON as `transformers`' `pipeline(task)(inputs, **parameters)` takes it.
pub trait AnyPipeline: Any {
    fn task(&self) -> Task;

    /// Runs the pipeline on `inputs`, one input or an array of them, with `parameters`, and returns one output for one input, or an array of one per input.
    ///
    /// # Errors
    ///
    /// Returns an error if `inputs` or `parameters` do not parse as the pipeline's, or if the pipeline fails.
    fn run_json(&self, inputs: &Value, parameters: &Value) -> Result<Value>;

    fn as_any(&self) -> &dyn Any;
}

impl dyn AnyPipeline {
    /// Returns the pipeline as `P`, when it is one.
    #[must_use]
    pub fn downcast<P: Pipeline + 'static>(&self) -> Option<&P> {
        self.as_any().downcast_ref()
    }
}

impl<P: Pipeline + 'static> AnyPipeline for P {
    fn task(&self) -> Task {
        P::TASK
    }

    fn run_json(&self, inputs: &Value, parameters: &Value) -> Result<Value> {
        // An array is one input when it reads as one, such as a conversation's messages, as `transformers` reads a list of messages as one chat.
        let (inputs, many) = match (inputs, serde_json::from_value::<P::Input>(inputs.clone())) {
            (Value::Array(items), Ok(_)) if items.is_empty() => (Vec::new(), true),
            (Value::Array(items), Err(_)) => (
                items
                    .iter()
                    .cloned()
                    .map(serde_json::from_value)
                    .collect::<StdResult<Vec<P::Input>, _>>()
                    .map_err(Error::json_value("parse the pipeline's inputs"))?,
                true,
            ),
            (_, input) => (
                vec![input.map_err(Error::json_value("parse the pipeline's input"))?],
                false,
            ),
        };
        let parameters = self::parameters::<P::Parameters>(parameters)?;
        let mut outputs = if inputs.is_empty() {
            Vec::new()
        } else {
            self.run(&inputs, &parameters)?
                .iter()
                .map(|output| {
                    serde_json::to_value(output)
                        .map_err(Error::json_value("write a pipeline output"))
                })
                .collect::<Result<Vec<_>>>()?
        };
        if many {
            return Ok(Value::Array(outputs));
        }
        outputs.pop().ok_or_else(|| {
            SingeError::missing_output("a pipeline's output for its one input").into()
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub(crate) fn run_alone<P: Pipeline>(
    pipeline: &P,
    input: &P::Input,
    parameters: &P::Parameters,
) -> Result<P::Output> {
    pipeline
        .run(slice::from_ref(input), parameters)?
        .into_iter()
        .next()
        .ok_or_else(|| SingeError::missing_output("a pipeline's output for its one input").into())
}

/// Returns the hypotheses a zero-shot pipeline scores its input against: `template` with each of `labels` in place of its `{}`.
pub(crate) fn hypotheses(template: &str, labels: &[String]) -> Result<Vec<String>> {
    if labels.is_empty() {
        return Err(Error::invalid("zero-shot classification without labels"));
    }

    if !template.contains("{}") {
        return Err(Error::invalid("a hypothesis template without `{}`"));
    }

    Ok(labels
        .iter()
        .map(|label| template.replacen("{}", label, 1))
        .collect())
}

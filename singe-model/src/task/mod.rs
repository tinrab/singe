// One module per task, under the Hub's category of tasks, named for its id as the Hugging Face Hub and `transformers`' pipelines name it.
// Each task has a pipeline: its inputs and outputs, what turns them into the model's and back, and the head it asks the registry for, or the `diffusers` components it loads itself.
// A task `transformers` adds goes under the category of its nearest Hub task.

pub mod audio;
pub mod multimodal;
pub mod reinforcement_learning;
pub mod tabular;
pub mod text;
pub mod vision;

use std::result::Result as StdResult;

use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

use crate::{
    error::{Error, Result},
    head::Head,
    source::{LoadOptions, Source},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Task {
    AnyToAny,
    AudioTextToText,
    DocumentQuestionAnswering,
    VisualDocumentRetrieval,
    ImageTextToText,
    ImageTextToImage,
    ImageTextToVideo,
    VideoTextToText,
    VisualQuestionAnswering,
    Decision,
    FeatureExtraction,
    FillMask,
    QuestionAnswering,
    SentenceSimilarity,
    Summarization,
    TableQuestionAnswering,
    TextClassification,
    TextGeneration,
    TextRanking,
    TokenClassification,
    Translation,
    ZeroShotClassification,
    DepthEstimation,
    ImageClassification,
    ImageFeatureExtraction,
    ImageSegmentation,
    ImageToImage,
    ImageToText,
    ImageToVideo,
    KeypointDetection,
    KeypointMatching,
    MaskGeneration,
    ObjectDetection,
    VideoClassification,
    TextToImage,
    TextToVideo,
    UnconditionalImageGeneration,
    VideoToVideo,
    ZeroShotImageClassification,
    ZeroShotObjectDetection,
    TextTo3d,
    ImageTo3d,
    AudioClassification,
    AudioToAudio,
    AutomaticSpeechRecognition,
    TextToAudio,
    ZeroShotAudioClassification,
    TabularClassification,
    TabularRegression,
    TimeSeriesForecasting,
    ReinforcementLearning,
}

impl Task {
    pub const ALL: [Self; 51] = [
        Self::AnyToAny,
        Self::AudioTextToText,
        Self::DocumentQuestionAnswering,
        Self::VisualDocumentRetrieval,
        Self::ImageTextToText,
        Self::ImageTextToImage,
        Self::ImageTextToVideo,
        Self::VideoTextToText,
        Self::VisualQuestionAnswering,
        Self::Decision,
        Self::FeatureExtraction,
        Self::FillMask,
        Self::QuestionAnswering,
        Self::SentenceSimilarity,
        Self::Summarization,
        Self::TableQuestionAnswering,
        Self::TextClassification,
        Self::TextGeneration,
        Self::TextRanking,
        Self::TokenClassification,
        Self::Translation,
        Self::ZeroShotClassification,
        Self::DepthEstimation,
        Self::ImageClassification,
        Self::ImageFeatureExtraction,
        Self::ImageSegmentation,
        Self::ImageToImage,
        Self::ImageToText,
        Self::ImageToVideo,
        Self::KeypointDetection,
        Self::KeypointMatching,
        Self::MaskGeneration,
        Self::ObjectDetection,
        Self::VideoClassification,
        Self::TextToImage,
        Self::TextToVideo,
        Self::UnconditionalImageGeneration,
        Self::VideoToVideo,
        Self::ZeroShotImageClassification,
        Self::ZeroShotObjectDetection,
        Self::TextTo3d,
        Self::ImageTo3d,
        Self::AudioClassification,
        Self::AudioToAudio,
        Self::AutomaticSpeechRecognition,
        Self::TextToAudio,
        Self::ZeroShotAudioClassification,
        Self::TabularClassification,
        Self::TabularRegression,
        Self::TimeSeriesForecasting,
        Self::ReinforcementLearning,
    ];

    // Returns the task's id, as `pipeline(task)` takes it.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::AnyToAny => "any-to-any",
            Self::AudioTextToText => "audio-text-to-text",
            Self::DocumentQuestionAnswering => "document-question-answering",
            Self::VisualDocumentRetrieval => "visual-document-retrieval",
            Self::ImageTextToText => "image-text-to-text",
            Self::ImageTextToImage => "image-text-to-image",
            Self::ImageTextToVideo => "image-text-to-video",
            Self::VideoTextToText => "video-text-to-text",
            Self::VisualQuestionAnswering => "visual-question-answering",
            Self::Decision => "decision",
            Self::FeatureExtraction => "feature-extraction",
            Self::FillMask => "fill-mask",
            Self::QuestionAnswering => "question-answering",
            Self::SentenceSimilarity => "sentence-similarity",
            Self::Summarization => "summarization",
            Self::TableQuestionAnswering => "table-question-answering",
            Self::TextClassification => "text-classification",
            Self::TextGeneration => "text-generation",
            Self::TextRanking => "text-ranking",
            Self::TokenClassification => "token-classification",
            Self::Translation => "translation",
            Self::ZeroShotClassification => "zero-shot-classification",
            Self::DepthEstimation => "depth-estimation",
            Self::ImageClassification => "image-classification",
            Self::ImageFeatureExtraction => "image-feature-extraction",
            Self::ImageSegmentation => "image-segmentation",
            Self::ImageToImage => "image-to-image",
            Self::ImageToText => "image-to-text",
            Self::ImageToVideo => "image-to-video",
            Self::KeypointDetection => "keypoint-detection",
            Self::KeypointMatching => "keypoint-matching",
            Self::MaskGeneration => "mask-generation",
            Self::ObjectDetection => "object-detection",
            Self::VideoClassification => "video-classification",
            Self::TextToImage => "text-to-image",
            Self::TextToVideo => "text-to-video",
            Self::UnconditionalImageGeneration => "unconditional-image-generation",
            Self::VideoToVideo => "video-to-video",
            Self::ZeroShotImageClassification => "zero-shot-image-classification",
            Self::ZeroShotObjectDetection => "zero-shot-object-detection",
            Self::TextTo3d => "text-to-3d",
            Self::ImageTo3d => "image-to-3d",
            Self::AudioClassification => "audio-classification",
            Self::AudioToAudio => "audio-to-audio",
            Self::AutomaticSpeechRecognition => "automatic-speech-recognition",
            Self::TextToAudio => "text-to-audio",
            Self::ZeroShotAudioClassification => "zero-shot-audio-classification",
            Self::TabularClassification => "tabular-classification",
            Self::TabularRegression => "tabular-regression",
            Self::TimeSeriesForecasting => "time-series-forecasting",
            Self::ReinforcementLearning => "reinforcement-learning",
        }
    }

    // Reads a task id, or one of the aliases the Hub or `transformers` gives a task.
    pub fn from_id(id: &str) -> Result<Self> {
        let id = match id {
            "sentiment-analysis" => "text-classification",
            "ner" => "token-classification",
            "text-to-speech" => "text-to-audio",
            "vqa" => "visual-question-answering",
            other => other,
        };
        Self::ALL
            .into_iter()
            .find(|task| task.id() == id)
            .ok_or_else(|| Error::UnsupportedConfig(format!("task {id}")))
    }

    // The head a task's pipeline asks the registry for, or `None` for a task whose pipeline loads `diffusers` components itself, as video and 3D generation do.
    #[must_use]
    pub const fn head(self) -> Option<Head> {
        match self {
            Self::TextGeneration | Self::Decision => Some(Head::CausalLm),
            Self::FeatureExtraction | Self::SentenceSimilarity => Some(Head::TextEncoder),
            Self::AutomaticSpeechRecognition | Self::AudioTextToText => Some(Head::SpeechToText),
            Self::ImageFeatureExtraction => Some(Head::ImageEncoder),
            Self::ImageTextToText => Some(Head::ImageTextToText),
            Self::ImageToImage => Some(Head::Autoencoder),
            Self::TextClassification | Self::TextRanking | Self::ZeroShotClassification => {
                Some(Head::SequenceClassification)
            }
            Self::TokenClassification => Some(Head::TokenClassification),
            Self::QuestionAnswering => Some(Head::QuestionAnswering),
            Self::FillMask => Some(Head::MaskedLm),
            Self::ImageClassification => Some(Head::ImageClassification),
            Self::VideoClassification => Some(Head::VideoClassification),
            Self::ObjectDetection => Some(Head::ObjectDetection),
            Self::ZeroShotImageClassification => Some(Head::ZeroShotImageClassification),
            Self::ZeroShotObjectDetection => Some(Head::ZeroShotObjectDetection),
            Self::DepthEstimation => Some(Head::DepthEstimation),
            Self::AudioClassification => Some(Head::AudioClassification),
            Self::AudioToAudio => Some(Head::AudioCodec),
            Self::ZeroShotAudioClassification => Some(Head::ZeroShotAudioClassification),
            Self::TextToAudio => Some(Head::TextToSpeech),
            Self::AnyToAny | Self::VideoTextToText | Self::VisualQuestionAnswering => {
                Some(Head::ImageTextToText)
            }
            Self::VisualDocumentRetrieval => Some(Head::DocumentRetrieval),
            Self::TimeSeriesForecasting => Some(Head::TimeSeriesForecasting),
            Self::ReinforcementLearning => Some(Head::Trajectory),
            Self::TabularClassification => Some(Head::TabularClassification),
            Self::TabularRegression => Some(Head::TabularRegression),
            Self::UnconditionalImageGeneration => Some(Head::Denoiser),
            Self::TextToImage | Self::ImageTextToImage => Some(Head::ConditionalDenoiser),
            Self::ImageSegmentation => Some(Head::ImageSegmentation),
            Self::KeypointDetection => Some(Head::KeypointDetection),
            Self::KeypointMatching => Some(Head::KeypointMatching),
            Self::MaskGeneration => Some(Head::MaskGeneration),
            Self::Summarization
            | Self::Translation
            | Self::ImageToText
            | Self::DocumentQuestionAnswering
            | Self::TableQuestionAnswering => Some(Head::Seq2SeqLm),
            _ => None,
        }
    }
}

// A task's pipeline: what it reads, how it runs, and what it returns, in the field names `transformers`' pipeline returns.
pub trait Pipeline: Sized {
    const TASK: Task;

    type Input: DeserializeOwned;

    type Parameters: DeserializeOwned + Default;

    type Output: Serialize;

    fn load(source: &Source, options: LoadOptions) -> Result<Self>;

    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>>;
}

// Runs `task`'s pipeline on `inputs`, one input or an array of them, with `parameters`, as `transformers`' `pipeline(task)(inputs, **parameters)` takes them, and returns one output per input.
pub fn run_json(
    task: Task,
    source: &Source,
    options: LoadOptions,
    inputs: &Value,
    parameters: &Value,
) -> Result<Vec<Value>> {
    match task {
        Task::AutomaticSpeechRecognition => run::<
            audio::automatic_speech_recognition::AutomaticSpeechRecognitionPipeline,
        >(source, options, inputs, parameters),
        Task::FeatureExtraction => run::<text::feature_extraction::FeatureExtractionPipeline>(
            source, options, inputs, parameters,
        ),
        Task::ImageFeatureExtraction => run::<
            vision::image_feature_extraction::ImageFeatureExtractionPipeline,
        >(source, options, inputs, parameters),
        Task::ImageTextToText => run::<multimodal::image_text_to_text::ImageTextToTextPipeline>(
            source, options, inputs, parameters,
        ),
        Task::ImageClassification => {
            run::<vision::image_classification::ImageClassificationPipeline>(
                source, options, inputs, parameters,
            )
        }
        Task::VideoClassification => {
            run::<vision::video_classification::VideoClassificationPipeline>(
                source, options, inputs, parameters,
            )
        }
        Task::ObjectDetection => run::<vision::object_detection::ObjectDetectionPipeline>(
            source, options, inputs, parameters,
        ),
        Task::ZeroShotImageClassification => run::<
            vision::zero_shot_image_classification::ZeroShotImageClassificationPipeline,
        >(source, options, inputs, parameters),
        Task::ZeroShotObjectDetection => run::<
            vision::zero_shot_object_detection::ZeroShotObjectDetectionPipeline,
        >(source, options, inputs, parameters),
        Task::ImageToImage => {
            run::<vision::image_to_image::ImageToImagePipeline>(source, options, inputs, parameters)
        }
        Task::DepthEstimation => run::<vision::depth_estimation::DepthEstimationPipeline>(
            source, options, inputs, parameters,
        ),
        Task::AudioClassification => {
            run::<audio::audio_classification::AudioClassificationPipeline>(
                source, options, inputs, parameters,
            )
        }
        Task::ImageSegmentation => run::<vision::image_segmentation::ImageSegmentationPipeline>(
            source, options, inputs, parameters,
        ),
        Task::KeypointDetection => run::<vision::keypoint_detection::KeypointDetectionPipeline>(
            source, options, inputs, parameters,
        ),
        Task::KeypointMatching => run::<vision::keypoint_matching::KeypointMatchingPipeline>(
            source, options, inputs, parameters,
        ),
        Task::MaskGeneration => run::<vision::mask_generation::MaskGenerationPipeline>(
            source, options, inputs, parameters,
        ),
        Task::Summarization => {
            run::<text::summarization::SummarizationPipeline>(source, options, inputs, parameters)
        }
        Task::Translation => {
            run::<text::translation::TranslationPipeline>(source, options, inputs, parameters)
        }
        Task::ImageToText => {
            run::<vision::image_to_text::ImageToTextPipeline>(source, options, inputs, parameters)
        }
        Task::TableQuestionAnswering => run::<
            text::table_question_answering::TableQuestionAnsweringPipeline,
        >(source, options, inputs, parameters),
        Task::DocumentQuestionAnswering => run::<
            multimodal::document_question_answering::DocumentQuestionAnsweringPipeline,
        >(source, options, inputs, parameters),
        Task::SentenceSimilarity => run::<text::sentence_similarity::SentenceSimilarityPipeline>(
            source, options, inputs, parameters,
        ),
        Task::FillMask => {
            run::<text::fill_mask::FillMaskPipeline>(source, options, inputs, parameters)
        }
        Task::QuestionAnswering => run::<text::question_answering::QuestionAnsweringPipeline>(
            source, options, inputs, parameters,
        ),
        Task::TokenClassification => {
            run::<text::token_classification::TokenClassificationPipeline>(
                source, options, inputs, parameters,
            )
        }
        Task::TextRanking => {
            run::<text::text_ranking::TextRankingPipeline>(source, options, inputs, parameters)
        }
        Task::ZeroShotClassification => run::<
            text::zero_shot_classification::ZeroShotClassificationPipeline,
        >(source, options, inputs, parameters),
        Task::TextClassification => run::<text::text_classification::TextClassificationPipeline>(
            source, options, inputs, parameters,
        ),
        Task::TextGeneration => run::<text::text_generation::TextGenerationPipeline>(
            source, options, inputs, parameters,
        ),
        Task::AudioToAudio => {
            run::<audio::audio_to_audio::AudioToAudioPipeline>(source, options, inputs, parameters)
        }
        Task::ZeroShotAudioClassification => run::<
            audio::zero_shot_audio_classification::ZeroShotAudioClassificationPipeline,
        >(source, options, inputs, parameters),
        Task::TextToAudio => {
            run::<audio::text_to_audio::TextToAudioPipeline>(source, options, inputs, parameters)
        }
        Task::AudioTextToText => run::<multimodal::audio_text_to_text::AudioTextToTextPipeline>(
            source, options, inputs, parameters,
        ),
        Task::AnyToAny => {
            run::<multimodal::any_to_any::AnyToAnyPipeline>(source, options, inputs, parameters)
        }
        Task::Decision => {
            run::<multimodal::decision::DecisionPipeline>(source, options, inputs, parameters)
        }
        Task::VideoTextToText => run::<multimodal::video_text_to_text::VideoTextToTextPipeline>(
            source, options, inputs, parameters,
        ),
        Task::ImageTextToImage => run::<multimodal::image_text_to_image::ImageTextToImagePipeline>(
            source, options, inputs, parameters,
        ),
        Task::ImageTextToVideo => run::<multimodal::image_text_to_video::ImageTextToVideoPipeline>(
            source, options, inputs, parameters,
        ),
        Task::ImageToVideo => {
            run::<vision::image_to_video::ImageToVideoPipeline>(source, options, inputs, parameters)
        }
        Task::VideoToVideo => {
            run::<vision::video_to_video::VideoToVideoPipeline>(source, options, inputs, parameters)
        }
        Task::TextTo3d => {
            run::<vision::text_to_3d::TextTo3dPipeline>(source, options, inputs, parameters)
        }
        Task::ImageTo3d => {
            run::<vision::image_to_3d::ImageTo3dPipeline>(source, options, inputs, parameters)
        }
        Task::TextToVideo => {
            run::<vision::text_to_video::TextToVideoPipeline>(source, options, inputs, parameters)
        }
        Task::TextToImage => {
            run::<vision::text_to_image::TextToImagePipeline>(source, options, inputs, parameters)
        }
        Task::UnconditionalImageGeneration => run::<
            vision::unconditional_image_generation::UnconditionalImageGenerationPipeline,
        >(source, options, inputs, parameters),
        Task::TabularClassification => run::<
            tabular::tabular_classification::TabularClassificationPipeline,
        >(source, options, inputs, parameters),
        Task::TabularRegression => run::<tabular::tabular_regression::TabularRegressionPipeline>(
            source, options, inputs, parameters,
        ),
        Task::ReinforcementLearning => run::<
            reinforcement_learning::reinforcement_learning::ReinforcementLearningPipeline,
        >(source, options, inputs, parameters),
        Task::TimeSeriesForecasting => run::<
            tabular::time_series_forecasting::TimeSeriesForecastingPipeline,
        >(source, options, inputs, parameters),
        Task::VisualDocumentRetrieval => run::<
            multimodal::visual_document_retrieval::VisualDocumentRetrievalPipeline,
        >(source, options, inputs, parameters),
        Task::VisualQuestionAnswering => run::<
            multimodal::visual_question_answering::VisualQuestionAnsweringPipeline,
        >(source, options, inputs, parameters),
    }
}

fn run<P: Pipeline>(
    source: &Source,
    options: LoadOptions,
    inputs: &Value,
    parameters: &Value,
) -> Result<Vec<Value>> {
    // An array is one input when it reads as one, such as a conversation's messages, as `transformers` reads a list of messages as one chat.
    let inputs = match (inputs, serde_json::from_value::<P::Input>(inputs.clone())) {
        (Value::Array(items), Ok(_)) if items.is_empty() => Vec::new(),
        (Value::Array(items), Err(_)) => items
            .iter()
            .cloned()
            .map(serde_json::from_value)
            .collect::<StdResult<Vec<P::Input>, _>>()
            .map_err(Error::json("parse the pipeline's inputs"))?,
        (_, input) => vec![input.map_err(Error::json("parse the pipeline's input"))?],
    };
    let parameters = if parameters.is_null() {
        P::Parameters::default()
    } else {
        serde_json::from_value(parameters.clone())
            .map_err(Error::json("parse the pipeline's parameters"))?
    };

    // No inputs give no outputs, without loading the model.
    if inputs.is_empty() {
        return Ok(Vec::new());
    }

    P::load(source, options)?
        .run(&inputs, &parameters)?
        .iter()
        .map(|output| serde_json::to_value(output).map_err(Error::json("write a pipeline output")))
        .collect()
}

// The hypotheses a zero-shot pipeline scores its input against: `template` with each of `labels` in place of its `{}`.
pub(crate) fn hypotheses(template: &str, labels: &[String]) -> Result<Vec<String>> {
    if labels.is_empty() {
        return Err(Error::unsupported(
            "zero-shot classification without labels",
        ));
    }

    if !template.contains("{}") {
        return Err(Error::unsupported("a hypothesis template without `{}`"));
    }

    Ok(labels
        .iter()
        .map(|label| template.replacen("{}", label, 1))
        .collect())
}

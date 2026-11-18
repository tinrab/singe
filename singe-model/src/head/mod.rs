// What a task asks a model for, as `transformers`' `AutoModelFor*` classes are: a causal language model, a text encoder, and the rest.
// Several tasks share a head, and a model serves each head its weights hold with a loader in its registry entry.
// A head that adds layers on another's outputs, such as a classifier on a text encoder, is written once here and shared by every model whose layout fits.

mod audio;
mod autoencoder;
mod causal_lm;
mod denoiser;
mod encoder;
mod generation_loop;
mod image_encoder;
mod image_text_to_text;
mod keypoint;
mod mask;
mod masked_lm;
mod question_answering;
mod retrieval;
mod seq2seq;
mod sequence_classification;
mod speech_to_text;
mod tabular;
mod text_encoder;
mod time_series;
mod token_classification;
mod trajectory;
mod vision;

pub use self::{
    audio::{
        AudioClassifier, AudioCodec, AudioLayers, CtcRecognizer, FilterBankOptions,
        SpectrogramOptions, Speech, SpeechLayers, TextToSpeech, ZeroShotAudioClassifier,
    },
    autoencoder::Autoencoder,
    causal_lm::CausalLm,
    denoiser::{ConditionalDenoiser, Denoiser},
    encoder::{EncoderHead, EncoderModel, HeadLayouts, encoder_heads, load_encoder_head},
    generation_loop::{LoopLimits, LoopSteps},
    image_encoder::{ImageEncoder, ImageEncoderLayers, PooledLayer},
    image_text_to_text::ImageTextToText,
    keypoint::{
        KeypointConfig, KeypointDetector, KeypointLayers, KeypointMatcher, Keypoints,
        MatchingLayers, PairMatches, extract_keypoints,
    },
    mask::{MaskGenerator, PromptableLayers},
    masked_lm::{MaskedLm, MaskedLmConfig, MaskedLmLayers},
    question_answering::{QuestionAnswerer, QuestionAnswererLayers},
    retrieval::{DocumentRetriever, late_interaction},
    seq2seq::{Seq2SeqGeneration, Seq2SeqInput, Seq2SeqLayers, Seq2SeqLm},
    sequence_classification::{
        ClassifierConfig, ClassifierLayout, ScoreFunction, SequenceClassifier,
        SequenceClassifierLayers, argmax,
    },
    speech_to_text::{SpeechToText, Transcription},
    tabular::{TabularClassifier, TabularRegressor},
    text_encoder::{TextEncoder, TextEncoderLayers},
    time_series::{Forecast, Forecaster},
    token_classification::{TokenClassifier, TokenClassifierLayers},
    trajectory::{Trajectory, TrajectoryModel, TrajectoryPredictions},
    vision::{
        Classifier, ContrastiveEncoder, DepthEstimator, Detection, ImageClassifier, ImageUpscaler,
        ObjectDetector, OpenVocabularyDetector, PixelLayers, QueryDetection, SemanticSegmenter,
        TextEmbeddingLayers, VideoClassifier,
    },
};
use crate::{error::Result, format::Loader};

pub type LoadFn<T> = fn(&mut Loader<'_>) -> Result<T>;

// What loading a head gives, which names the head to load: `model::load::<Box<dyn CausalLm>>` loads a causal language model.
pub trait LoadHead: Sized {
    const HEAD: Head;

    // The loader of this head, if `loader` is one.
    fn loader(loader: &HeadLoader) -> Option<LoadFn<Self>>;
}

// Declares each head once: its name and id, what loading it gives, and the endings of the `transformers` class names whose weights hold it.
// It writes `Head`, `HeadLoader`, and the `LoadHead` of each head's type.
macro_rules! heads {
    ($($head:ident($id:literal): $output:ty = [$($suffix:literal),* $(,)?]),* $(,)?) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum Head {
            $($head),*
        }

        impl Head {
            pub const ALL: &[Self] = &[$(Self::$head),*];

            // The head's name, as a native config records it.
            #[must_use]
            pub const fn id(self) -> &'static str {
                match self {
                    $(Self::$head => $id),*
                }
            }

            // Whether the `transformers` class `class` holds this head's weights.
            #[must_use]
            pub fn names(self, class: &str) -> bool {
                self.class_suffixes().iter().any(|suffix| class.ends_with(suffix))
            }

            #[must_use]
            pub fn from_id(id: &str) -> Option<Self> {
                Self::ALL.iter().copied().find(|head| head.id() == id)
            }

            // The endings of the `transformers` class names, in a config's `architectures`, whose weights hold this head.
            #[must_use]
            pub const fn class_suffixes(self) -> &'static [&'static str] {
                match self {
                    $(Self::$head => &[$($suffix),*]),*
                }
            }
        }

        // A model's loader of one head.
        #[derive(Debug, Clone, Copy)]
        pub enum HeadLoader {
            $($head(LoadFn<$output>)),*
        }

        impl HeadLoader {
            #[must_use]
            pub const fn head(&self) -> Head {
                match self {
                    $(Self::$head(_) => Head::$head),*
                }
            }

            // Loads the head and drops it, leaving what it read in `loader`.
            pub fn run(&self, loader: &mut Loader<'_>) -> Result<()> {
                match self {
                    $(Self::$head(load) => load(loader).map(drop)),*
                }
            }
        }

        $(
            impl LoadHead for $output {
                const HEAD: Head = Head::$head;

                fn loader(loader: &HeadLoader) -> Option<LoadFn<Self>> {
                    let HeadLoader::$head(load) = loader else {
                        return None;
                    };
                    Some(*load)
                }
            }
        )*
    };
}

heads! {
    CausalLm("causal_lm"): Box<dyn CausalLm> = ["ForCausalLM", "LMHeadModel"],
    SpeechToText("speech_to_text"): Box<dyn SpeechToText> = ["ForConditionalGeneration", "ForSpeechSeq2Seq"],
    ImageTextToText("image_text_to_text"): Box<dyn ImageTextToText> = ["ForConditionalGeneration", "ForImageTextToText"],
    TextEncoder("text_encoder"): TextEncoder = ["Model"],
    ImageEncoder("image_encoder"): ImageEncoder = ["Model"],
    Autoencoder("autoencoder"): Autoencoder = [],
    SequenceClassification("sequence_classification"): SequenceClassifier = ["ForSequenceClassification"],
    TokenClassification("token_classification"): TokenClassifier = ["ForTokenClassification"],
    QuestionAnswering("question_answering"): QuestionAnswerer = ["ForQuestionAnswering"],
    MaskedLm("masked_lm"): MaskedLm = ["ForMaskedLM"],
    ImageClassification("image_classification"): ImageClassifier = ["ForImageClassification"],
    VideoClassification("video_classification"): VideoClassifier = ["ForVideoClassification"],
    ObjectDetection("object_detection"): ObjectDetector = ["ForObjectDetection"],
    ZeroShotImageClassification("zero_shot_image_classification"): ContrastiveEncoder = ["Model"],
    ZeroShotObjectDetection("zero_shot_object_detection"): OpenVocabularyDetector = ["ForObjectDetection"],
    DepthEstimation("depth_estimation"): DepthEstimator = ["ForDepthEstimation"],
    ImageSegmentation("image_segmentation"): SemanticSegmenter = ["ForSemanticSegmentation"],
    KeypointDetection("keypoint_detection"): KeypointDetector = ["ForKeypointDetection", "Model"],
    KeypointMatching("keypoint_matching"): KeypointMatcher = ["ForKeypointMatching"],
    SuperResolution("super_resolution"): ImageUpscaler = ["ForImageSuperResolution"],
    MaskGeneration("mask_generation"): MaskGenerator = ["Model"],
    Seq2SeqLm("seq2seq_lm"): Seq2SeqLm = ["ForConditionalGeneration", "VisionEncoderDecoderModel"],
    Ctc("ctc"): CtcRecognizer = ["ForCTC"],
    AudioClassification("audio_classification"): AudioClassifier = ["ForAudioClassification"],
    AudioCodec("audio_codec"): AudioCodec = ["Model"],
    ZeroShotAudioClassification("zero_shot_audio_classification"): ZeroShotAudioClassifier = ["Model"],
    TextToSpeech("text_to_speech"): TextToSpeech = ["Model"],
    DocumentRetrieval("document_retrieval"): Box<dyn DocumentRetriever> = ["ForRetrieval"],
    TimeSeriesForecasting("time_series_forecasting"): Box<dyn Forecaster> = ["ForPrediction"],
    Trajectory("trajectory"): Box<dyn TrajectoryModel> = ["Model"],
    TabularClassification("tabular_classification"): Box<dyn TabularClassifier> = [],
    TabularRegression("tabular_regression"): Box<dyn TabularRegressor> = [],
    Denoiser("denoiser"): Box<dyn Denoiser> = [],
    ConditionalDenoiser("conditional_denoiser"): Box<dyn ConditionalDenoiser> = [],
}

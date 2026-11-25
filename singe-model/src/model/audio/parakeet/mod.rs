//! NVIDIA's Parakeet speech recognizers, as `transformers` writes them: a `FastConformer` encoder of `NeMo`'s log-mel features, read by a CTC head, or by an RNN-T or TDT transducer's prediction and joint networks.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult, Tensor,
    nn::{Conv1d, Conv1dConfig, Layer, ModelState},
    signal::{LogMelOptions, LogMelSpectrogram},
};

use crate::{
    assembly::{ConformerEncoder, ConformerEncoderConfig, TransducerConfig, TransducerLayers},
    error::{Error, Result},
    format::Loader,
    head::{Boxed, CtcInput, CtcRecognizer, HeadLoader},
    model::{ModelEntry, audio::transducer::TransducerModel},
    schema::Schema,
    source::{Format, Source},
};

/// The settings of `transformers`' `ParakeetFeatureExtractor`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ParakeetFeatures {
    pub sample_rate: i64,
    pub mels: i64,
    pub window: i64,
    pub fft: i64,
    pub hop: i64,
    pub preemphasis: Option<f64>,
}

impl ParakeetFeatures {
    /// Returns the options for `NeMo`'s log-mel features with these settings, up to half the sample rate.
    #[must_use]
    pub fn log_mel(&self) -> LogMelOptions {
        LogMelOptions {
            sample_rate: self.sample_rate,
            window: self.window,
            fft: self.fft,
            hop: self.hop,
            max_frequency: self.sample_rate as f64 / 2.0,
            preemphasis: self.preemphasis,
            ..LogMelOptions::nemo(self.mels)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParakeetHead {
    Ctc { vocabulary: i64 },
    Transducer(TransducerConfig),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParakeetConfig {
    pub encoder: ConformerEncoderConfig,
    pub features: ParakeetFeatures,
    pub head: ParakeetHead,
}

pub type Parakeet = TransducerModel<ParakeetConfig, ConformerEncoder>;

/// A CTC Parakeet's layers: the encoder, `encoder`, and a pointwise convolution to each row's scores, `head`.
#[derive(Debug, Clone)]
pub struct ParakeetCtcLayers {
    encoder: ConformerEncoder,
    head: Conv1d,
}

impl ParakeetCtcLayers {
    pub fn create(schema: &mut Schema<'_>, config: &ParakeetConfig) -> SingeResult<Self> {
        let ParakeetHead::Ctc { vocabulary } = config.head else {
            return Err(singe::Error::invalid_argument(
                "a CTC Parakeet with a transducer's config",
            ));
        };
        Ok(Self {
            encoder: ConformerEncoder::create(&mut schema.scope("encoder"), &config.encoder)?,
            head: schema.conv1d(
                "head",
                config.encoder.block.width,
                vocabulary,
                1,
                Conv1dConfig::default(),
                true,
            )?,
        })
    }
}

impl Layer for ParakeetCtcLayers {
    /// Returns each encoder row's scores, `[B, T', V]`, of log-mel features, `[B, mels, T]`.
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        self.encoder
            .forward(input, state)?
            .transpose(1, 2)?
            .apply(&self.head, state)?
            .transpose(1, 2)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "parakeet",
    formats: &[(
        Format::Transformers,
        &["parakeet_tdt", "parakeet_rnnt", "parakeet_ctc"],
    )],
    loaders: &[
        HeadLoader::SpeechToText(|loader| Ok(Box::new(load(loader)?))),
        HeadLoader::Ctc(load_ctc),
    ],
};

/// Loads an RNN-T or TDT Parakeet.
///
/// # Errors
///
/// Returns an error if the snapshot is a CTC Parakeet, or if its config or checkpoint does not load.
pub fn load(loader: &mut Loader<'_>) -> Result<Parakeet> {
    let config: ParakeetConfig = loader.config(transformers::config)?;
    let ParakeetHead::Transducer(transducer) = &config.head else {
        return Err(Error::unsupported(
            "transcribing with a CTC Parakeet's transducer",
        ));
    };
    let data_type = loader.options().data_type;
    let encoder = Schema::model(data_type, |schema| {
        ConformerEncoder::create(&mut schema.scope("encoder"), &config.encoder)
    })?;
    let decoder = Schema::model(data_type, |schema| {
        TransducerLayers::create(&mut schema.scope("decoder"), transducer)
    })?;
    let front_end = config.features.log_mel();
    TransducerModel::load(
        loader,
        config,
        front_end,
        (&encoder, &transformers::encoder_rules()),
        (&decoder, &transformers::decoder_rules()),
    )
}

/// Returns the settings of a Parakeet snapshot's feature extractor, which a served export reads its log-mel features with.
///
/// # Errors
///
/// Returns an error if the snapshot's config does not read as a Parakeet's.
pub fn features(source: &Source) -> Result<ParakeetFeatures> {
    Ok(transformers::config(source)?.features)
}

/// Loads a CTC Parakeet's encoder and head.
///
/// # Errors
///
/// Returns an error if the snapshot is an RNN-T or TDT Parakeet, or if its config or checkpoint does not load.
pub fn load_ctc(loader: &mut Loader<'_>) -> Result<CtcRecognizer> {
    let config: ParakeetConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        ParakeetCtcLayers::create(schema, &config)
            .map(|layers| Boxed::<dyn Layer>(Box::new(layers)))
    })?;
    Ok(CtcRecognizer::new(
        loader.load_whole(&model, &transformers::ctc_rules())?,
        config.features.sample_rate,
        CtcInput::LogMel(LogMelSpectrogram::create(config.features.log_mel())?),
    ))
}

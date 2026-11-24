//! A speech transducer, as `NeMo`'s RNN-T and TDT models are: a log-mel front end, an encoder of the features, and prediction and joint networks that decode the encoder's rows greedily.
//! The encoder has state of its own, `encoder`, and the prediction and joint networks share `decoder`.

use std::fmt::Debug;

use singe::{
    DataType, Error as SingeError, Result as SingeResult, Tensor,
    generation::GenerationOptions,
    nn::{Model, ModelState},
    signal::{LogMelOptions, LogMelSpectrogram},
};

use crate::{
    BatchOptions,
    assembly::{ConformerEncoder, TransducerLayers, TransducerStep, greedy_decode},
    error::{Error, Result},
    format::{Loader, Rules},
    head::{LoopLimits, LoopSteps, SpeechToText, Transcription},
    processor::text::TextTokenizer,
};

/// An encoder of log-mel features, `[B, mels, T]`, to rows, `[B, T', W]`.
pub trait AudioEncoder: Debug + Send + Sync {
    /// Returns how many rows it gives for `frames` frames of features.
    fn rows(&self, frames: i64) -> i64;

    /// Encodes `features`, `[B, mels, T]`, to rows, `[B, T', W]`.
    /// With `lengths`, an `I64` `[B]` of each clip's own frames, it also returns how many rows each clip keeps, `[B]`.
    fn encode_features(
        &self,
        features: &Tensor,
        lengths: Option<&Tensor>,
        state: &ModelState,
    ) -> SingeResult<(Tensor, Option<Tensor>)>;
}

impl AudioEncoder for ConformerEncoder {
    fn rows(&self, frames: i64) -> i64 {
        self.config().subsampling.subsampled(frames)
    }

    fn encode_features(
        &self,
        features: &Tensor,
        lengths: Option<&Tensor>,
        state: &ModelState,
    ) -> SingeResult<(Tensor, Option<Tensor>)> {
        self.encode(features, lengths, state)
    }
}

#[derive(Debug, Clone)]
pub struct TransducerModel<C, E> {
    config: C,
    front_end: LogMelSpectrogram,
    encoder: Model<E>,
    decoder: Model<TransducerLayers>,
}

impl<C, E: AudioEncoder> TransducerModel<C, E> {
    pub(crate) fn load(
        loader: &mut Loader<'_>,
        config: C,
        front_end: LogMelOptions,
        (encoder, encoder_rules): (&Model<E>, &Rules),
        (decoder, decoder_rules): (&Model<TransducerLayers>, &Rules),
    ) -> Result<Self> {
        let encoder = loader.load("encoder", encoder, encoder_rules)?;
        let decoder = loader.load("decoder", decoder, decoder_rules)?;
        loader.finish()?;
        Ok(Self {
            config,
            front_end: LogMelSpectrogram::create(front_end)?,
            encoder,
            decoder,
        })
    }

    #[must_use]
    pub const fn config(&self) -> &C {
        &self.config
    }

    #[must_use]
    pub const fn encoder(&self) -> &Model<E> {
        &self.encoder
    }

    #[must_use]
    pub const fn decoder(&self) -> &Model<TransducerLayers> {
        &self.decoder
    }

    #[must_use]
    pub const fn front_end(&self) -> &LogMelSpectrogram {
        &self.front_end
    }

    /// Returns the log-mel features of one clip's samples, `[N]`, as `[1, mels, F]`, its own `F` frames as [`LogMelSpectrogram::own_frames`] counts them.
    /// A clip alone needs no mask: the frame past its own, which the feature extractor masks, would only be zeroed and hidden from every step the encoder keeps.
    pub fn features(&self, samples: &Tensor) -> SingeResult<Tensor> {
        let &[count] = samples.shape().dimensions() else {
            return Err(SingeError::invalid_argument(
                "a transducer reads one clip's samples, [samples]",
            ));
        };
        let frames = self.front_end.own_frames(count);
        let (features, _) = self.front_end.features(&samples.unsqueeze(0)?, None)?;
        features
            .narrow(2, 0, frames)?
            .cast(self.encoder.state().floating_type()?)
    }

    /// Encodes one clip's samples, `[N]`, to the joint network's projection of its rows, `[T, J]`.
    pub fn encode(&self, samples: &Tensor) -> SingeResult<Tensor> {
        let features = self.features(samples)?;
        let (rows, _) =
            self.encoder
                .layer()
                .encode_features(&features, None, self.encoder.state())?;
        let decoder = &self.decoder;
        decoder.layer().joint.project(
            &rows.select(0, 0)?.cast(decoder.state().floating_type()?)?,
            decoder.state(),
        )
    }

    /// Decodes one clip's samples, `[N]`, greedily to its emissions, at most `max_tokens` of them.
    /// It takes at most the steps `transformers` allows: `steps_per_row` for each row the encoder gives one frame more than the clip's own, less one.
    /// A clip shorter than one frame gives none.
    pub fn decode(&self, samples: &Tensor, max_tokens: usize) -> SingeResult<Vec<TransducerStep>> {
        let frames = samples
            .shape()
            .dimensions()
            .first()
            .map_or(0, |&count| self.front_end.own_frames(count));
        if frames == 0 {
            return Ok(Vec::new());
        }

        let rows = self.encode(samples)?;
        let count = rows.shape().dimensions()[0];
        // `transformers` sizes decoding by the rows of its extractor's frames, one more than the clip's own, times `max_symbols_per_step`, less the start token that the length counts.
        let decoder = self.decoder.layer();
        let max_steps =
            decoder.config.steps_per_row.max(1) * self.encoder.layer().rows(frames + 1) - 1;
        greedy_decode(
            decoder,
            &rows,
            count,
            max_tokens,
            max_steps,
            self.decoder.state(),
        )
    }
}

impl<C: Debug + Send + Sync, E: AudioEncoder> SpeechToText for TransducerModel<C, E> {
    fn sample_rate(&self) -> i64 {
        self.front_end.options().sample_rate
    }

    /// Returns `None`: decoding ends with the clip's rows, so a transducer needs no limit of its own.
    fn default_max_new_tokens(&self) -> Option<usize> {
        None
    }

    /// Transcribes one clip's samples, `[N]`, emitting at most `options.generation.max_new_tokens` tokens.
    /// A transducer names no language, so `language` changes nothing.
    fn transcribe(
        &self,
        tokenizer: &TextTokenizer,
        samples: &Tensor,
        _language: Option<&str>,
        options: &BatchOptions,
    ) -> Result<Transcription> {
        let samples = samples.cast(DataType::F32)?;
        let steps = self.decode(&samples, options.generation.max_new_tokens)?;
        let tokens = steps.iter().map(|step| step.token).collect::<Vec<_>>();
        Ok(Transcription {
            language: None,
            text: tokenizer.decode(&tokens, true)?.trim().to_owned(),
        })
    }

    fn loop_steps(&self, _: &LoopLimits, _: &GenerationOptions) -> Result<LoopSteps> {
        Err(Error::unsupported("exporting a transducer's decoding loop"))
    }
}

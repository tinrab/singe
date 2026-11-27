//! A speech recognizer's session: PCM streamed in, its turns found by the model's voice activity detector or ended by `commit`, and each turn's transcript.

use std::{mem, sync::Arc};

use futures::future::BoxFuture;
use serde_json::{Map, Value};
use singe_model::{
    model::audio::silero_vad::{SileroVad, TurnDetector, TurnEvent, TurnSettings},
    processor::audio::Audio,
    task::{
        AnyPipeline, Pipeline,
        audio::automatic_speech_recognition::{
            AutomaticSpeechRecognitionParameters, AutomaticSpeechRecognitionPipeline,
        },
    },
};
use tokio::task::JoinSet;

use crate::{
    api::realtime::{
        DEFAULT_SAMPLE_RATE, RecognitionEvent, RecognitionRequest, RecognitionSettings,
        TurnDetection,
    },
    error::{ApiError, ApiResult},
    handler::{blocking, downcast},
    realtime::{Incoming, Outgoing, RealtimeSession},
    worker::Worker,
};

/// The most audio a session holds without a turn ending, ten minutes.
const MAX_BUFFERED_SECONDS: usize = 600;

/// The session of a connection to a speech recognizer.
pub struct RecognitionSession {
    worker: Worker<Box<dyn AnyPipeline>>,
    /// The model's voice activity detector, which turn detection needs.
    voice_activity: Option<Arc<SileroVad>>,
    sample_rate: u32,
    language: Option<String>,
    turn_detection: Option<TurnDetection>,
    detector: Option<TurnDetector>,
    /// Audio not yet in a turn.
    buffer: Vec<f32>,
    /// The stream position of `buffer`'s first sample.
    buffer_start: usize,
    /// Where the speech in progress started, in milliseconds.
    speech_start: Option<u64>,
    /// The turns so far.
    turns: u64,
    running: JoinSet<()>,
}

impl RecognitionSession {
    /// Opens a session on `worker`, with turn detection by `voice_activity` when the model has a detector, and the `sample_rate` and `language` the query names.
    ///
    /// # Errors
    ///
    /// Returns a 400 error for a query whose `sample_rate` is not a positive integer.
    pub fn open(
        worker: Worker<Box<dyn AnyPipeline>>,
        voice_activity: Option<Arc<SileroVad>>,
        query: &Map<String, Value>,
    ) -> ApiResult<Self> {
        let sample_rate = match query.get("sample_rate").and_then(Value::as_str) {
            Some(rate) => rate.parse().map_err(|_| {
                ApiError::invalid_field("sample_rate", format!("a sample rate of {rate}"))
            })?,
            None => DEFAULT_SAMPLE_RATE,
        };
        let language = query
            .get("language")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let mut session = Self {
            worker,
            voice_activity,
            sample_rate: DEFAULT_SAMPLE_RATE,
            language: None,
            turn_detection: None,
            detector: None,
            buffer: Vec::new(),
            buffer_start: 0,
            speech_start: None,
            turns: 0,
            running: JoinSet::new(),
        };
        session.configure(RecognitionSettings {
            sample_rate,
            language,
            turn_detection: None,
        })?;
        Ok(session)
    }

    /// Replaces the settings and starts the stream over; turn detection is on by default when the model has a detector.
    fn configure(&mut self, settings: RecognitionSettings) -> ApiResult<()> {
        if settings.sample_rate == 0 {
            return Err(ApiError::invalid_field("sample_rate", "a sample rate of 0"));
        }

        let turn_detection = settings
            .turn_detection
            .unwrap_or_else(|| self.voice_activity.is_some().then(TurnDetection::default));
        let detector = match turn_detection {
            Some(turn_detection) => {
                if !(0.0..=1.0).contains(&turn_detection.threshold) {
                    return Err(ApiError::invalid_field(
                        "turn_detection.threshold",
                        "a threshold outside [0, 1]",
                    ));
                }

                let model = self.voice_activity.as_ref().ok_or_else(|| {
                    ApiError::invalid_field(
                        "turn_detection",
                        "turn detection by a model served without a voice activity detector (`voice_activity`)",
                    )
                })?;
                let turns = TurnSettings {
                    threshold: turn_detection.threshold,
                    silence_ms: turn_detection.silence_ms,
                };
                Some(TurnDetector::create(
                    Arc::clone(model),
                    settings.sample_rate,
                    turns,
                )?)
            }
            None => None,
        };

        self.sample_rate = settings.sample_rate;
        self.language = settings.language;
        self.turn_detection = turn_detection;
        self.detector = detector;
        self.buffer.clear();
        self.buffer_start = 0;
        self.speech_start = None;
        Ok(())
    }

    async fn append(&mut self, bytes: &[u8], outgoing: &Outgoing) -> ApiResult<()> {
        if !bytes.len().is_multiple_of(2) {
            return Err(ApiError::invalid(
                "an odd number of bytes, rather than 16-bit samples",
            ));
        }

        let samples = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&pair| f32::from(i16::from_le_bytes(pair)) / 32768.0)
            .collect::<Vec<_>>();
        self.buffer.extend_from_slice(&samples);

        if let Some(mut detector) = self.detector.take() {
            let (detector, turns) = blocking(move || {
                let turns = detector.push(&samples)?;
                Ok((detector, turns))
            })
            .await?;
            let speaking = detector.is_speaking();
            self.detector = Some(detector);
            for turn in turns {
                match turn {
                    TurnEvent::Started(at_ms) => {
                        self.speech_start = Some(at_ms);
                        outgoing
                            .json(&RecognitionEvent::SpeechStarted { at_ms })
                            .await;
                    }
                    TurnEvent::Stopped(at_ms) => {
                        outgoing
                            .json(&RecognitionEvent::SpeechStopped { at_ms })
                            .await;
                        self.end_turn(at_ms, outgoing);
                    }
                }
            }

            // Between turns only the audio a turn keeps before its speech stays.
            if !speaking && let Some(turn_detection) = self.turn_detection {
                let keep = self.samples(turn_detection.prefix_ms);
                self.drop_buffered(self.buffer.len().saturating_sub(keep));
            }
        }

        let most = MAX_BUFFERED_SECONDS * self.sample_rate as usize;
        if self.buffer.len() > most {
            self.drop_buffered(self.buffer.len() - most);
            return Err(ApiError::invalid(
                "audio over ten minutes without a turn, whose oldest was dropped; commit turns sooner",
            ));
        }
        Ok(())
    }

    /// Ends the turn whose speech the detector found stopping at `end`, in milliseconds.
    fn end_turn(&mut self, end: u64, outgoing: &Outgoing) {
        let prefix = self.turn_detection.map_or(0, |turn| turn.prefix_ms);
        let start = self.speech_start.take().unwrap_or(0).saturating_sub(prefix);
        let first = self.position(start).min(self.buffer.len());
        let last = self.position(end).clamp(first, self.buffer.len());
        let turn = self.buffer[first..last].to_vec();
        self.drop_buffered(last);
        self.transcribe(turn, outgoing);
    }

    fn commit(&mut self, outgoing: &Outgoing) -> ApiResult<()> {
        if self.buffer.is_empty() {
            return Err(ApiError::invalid("an empty turn to commit"));
        }

        if let Some(detector) = &mut self.detector {
            detector.reset_turn();
        }

        self.speech_start = None;
        let turn = mem::take(&mut self.buffer);
        self.buffer_start += turn.len();
        self.transcribe(turn, outgoing);
        Ok(())
    }

    fn clear(&mut self) {
        if let Some(detector) = &mut self.detector {
            detector.reset_turn();
        }

        self.speech_start = None;
        self.drop_buffered(self.buffer.len());
    }

    /// Transcribes a turn in a task of its own, which sends its transcript.
    fn transcribe(&mut self, samples: Vec<f32>, outgoing: &Outgoing) {
        while self.running.try_join_next().is_some() {}
        self.turns += 1;
        let turn = self.turns;
        let clip = Audio {
            samples,
            rate: self.sample_rate,
        };
        let parameters = AutomaticSpeechRecognitionParameters {
            language: self.language.clone(),
            ..AutomaticSpeechRecognitionParameters::default()
        };
        let (worker, outgoing) = (self.worker.clone(), outgoing.clone());
        self.running.spawn(async move {
            let transcribed = worker
                .call(move |pipeline| {
                    Ok(downcast::<AutomaticSpeechRecognitionPipeline>(&**pipeline)?
                        .run_one(&clip, &parameters)?)
                })
                .await;
            match transcribed {
                Ok(transcription) => {
                    let text = transcription.text.trim().to_owned();
                    outgoing
                        .json(&RecognitionEvent::Transcript { turn, text })
                        .await;
                }
                Err(error) => {
                    outgoing.error(Some(&turn.to_string()), &error).await;
                }
            }
        });
    }

    /// Returns the samples in `milliseconds` of audio.
    fn samples(&self, milliseconds: u64) -> usize {
        usize::try_from(milliseconds * u64::from(self.sample_rate) / 1000).unwrap_or(usize::MAX)
    }

    /// Returns the buffer's index of the stream position `milliseconds` from its start.
    fn position(&self, milliseconds: u64) -> usize {
        self.samples(milliseconds).saturating_sub(self.buffer_start)
    }

    /// Drops the buffer's first `count` samples.
    fn drop_buffered(&mut self, count: usize) {
        self.buffer.drain(..count);
        self.buffer_start += count;
    }
}

impl RealtimeSession for RecognitionSession {
    fn receive<'a>(
        &'a mut self,
        message: Incoming,
        outgoing: &'a Outgoing,
    ) -> BoxFuture<'a, ApiResult<()>> {
        Box::pin(async move {
            if let Incoming::Binary(bytes) = &message {
                return self.append(bytes, outgoing).await;
            }

            match message.parse::<RecognitionRequest>()? {
                RecognitionRequest::Configure(settings) => self.configure(settings),
                RecognitionRequest::Commit => self.commit(outgoing),
                RecognitionRequest::Clear => {
                    self.clear();
                    Ok(())
                }
            }
        })
    }
}

//! A voice assistant built on `singe-server`'s realtime layer, served at `/v1/realtime/voice` beside the three models it chains: a speech recognizer hears each turn, a chat model replies, and a speech synthesizer speaks the reply a sentence at a time while the rest generates.
//! Speech during a reply cancels it; the conversation lasts as long as the connection.
//!
//! `cargo run --release -p singe-server --example voice -- RECOGNIZER CHAT SYNTHESIZER DETECTOR`, for example `nvidia/parakeet-tdt-0.6b-v3 Qwen/Qwen3-0.6B facebook/mms-tts-eng onnx-community/silero-vad`.
//!
//! The client sends binary frames of 16-bit little-endian mono PCM at 24 kHz, and receives `{"type": "transcript", "text"}` for each turn, then for each sentence of the reply `{"type": "sentence", "text", "sample_rate"}` and its speech as binary frames, then `{"type": "reply_done"}`, or `{"type": "interrupted"}` when speech cancels the reply.

use std::{
    env,
    error::Error,
    mem,
    sync::{Arc, Mutex, PoisonError},
};

use futures::future::BoxFuture;
use serde_json::{Map, Value, json};
use singe_model::{
    head::SpeechPrompt,
    model::audio::silero_vad::{SileroVad, TurnDetector, TurnEvent, TurnSettings},
    processor::{
        audio::Audio,
        chat::{Content, Message},
    },
    task::{
        AnyPipeline, Pipeline,
        audio::{
            automatic_speech_recognition::{
                AutomaticSpeechRecognitionParameters, AutomaticSpeechRecognitionPipeline,
            },
            text_to_audio::{TextToAudioParameters, TextToAudioPipeline},
        },
    },
};
use singe_server::{
    api::{completion::GenerationFields, now},
    config::{ModelConfig, ServerConfig},
    error::{ApiError, ApiResult},
    generation::{
        engine::TextEngine,
        format::TextFormat,
        output::{Piece, TextGeneration},
    },
    handler::{blocking, downcast, max_tokens},
    realtime::{Incoming, Outgoing, RealtimeRunner, RealtimeSession},
    registry::{Engine, Served, ServedModel},
    server::{app, load_model},
    text::Detokenizer,
    worker::{ModelState, Worker},
};
use tokio::{
    net::TcpListener,
    sync::mpsc,
    task::{AbortHandle, JoinSet},
};

/// The rate of the audio the client sends.
const SAMPLE_RATE: u32 = 24_000;

/// The audio before speech that a turn keeps, in milliseconds.
const PREFIX_MS: u64 = 300;

/// The system message of every reply.
const INSTRUCTIONS: &str = "You are a helpful voice assistant. Answer briefly, in plain spoken sentences, without lists, headings, or other formatting.";

/// The fewest words of a first clause cut at a comma, so speech starts before the first sentence is written.
const FIRST_CLAUSE_WORDS: usize = 4;

type Pipelines = Worker<Box<dyn AnyPipeline>>;

/// The three models a voice session chains, and the detector its turns are found with.
#[derive(Clone)]
struct Voice {
    recognizer: Pipelines,
    detector: Arc<SileroVad>,
    chat: (TextEngine, Arc<TextFormat>),
    synthesizer: Pipelines,
}

impl RealtimeRunner for Voice {
    fn state(&self) -> ModelState {
        [
            self.recognizer.state(),
            self.chat.0.state(),
            self.synthesizer.state(),
        ]
        .into_iter()
        .find(|state| *state != ModelState::Ready)
        .unwrap_or(ModelState::Ready)
    }

    fn open(&self, _query: Map<String, Value>) -> ApiResult<Box<dyn RealtimeSession>> {
        let settings = TurnSettings::default();
        let turns = TurnDetector::create(Arc::clone(&self.detector), SAMPLE_RATE, settings)?;
        Ok(Box::new(VoiceSession {
            voice: self.clone(),
            turns: Some(turns),
            buffer: Vec::new(),
            buffer_start: 0,
            speech_start: 0,
            history: Arc::new(Mutex::new(Vec::new())),
            running: JoinSet::new(),
            reply: None,
        }))
    }
}

struct VoiceSession {
    voice: Voice,
    /// The turn detector, taken while it reads a frame on a blocking thread.
    turns: Option<TurnDetector>,
    /// Audio not yet in a turn, and the stream position of its first sample.
    buffer: Vec<f32>,
    buffer_start: usize,
    /// Where the speech in progress started, in milliseconds.
    speech_start: u64,
    /// The conversation so far: each turn's transcript and what each reply said.
    history: Arc<Mutex<Vec<(&'static str, String)>>>,
    running: JoinSet<()>,
    /// The turn being answered.
    reply: Option<AbortHandle>,
}

impl RealtimeSession for VoiceSession {
    fn receive<'a>(
        &'a mut self,
        message: Incoming,
        outgoing: &'a Outgoing,
    ) -> BoxFuture<'a, ApiResult<()>> {
        Box::pin(async move {
            let Incoming::Binary(bytes) = message else {
                return Err(ApiError::invalid(
                    "a text message; send audio as binary frames",
                ));
            };
            let samples = bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&pair| f32::from(i16::from_le_bytes(pair)) / 32768.0)
                .collect::<Vec<_>>();
            self.buffer.extend_from_slice(&samples);
            let mut turns = self
                .turns
                .take()
                .ok_or_else(|| ApiError::internal("a turn detector lost to a failed frame"))?;
            let (turns, events) = blocking(move || {
                let events = turns.push(&samples)?;
                Ok((turns, events))
            })
            .await?;
            let speaking = turns.is_speaking();
            self.turns = Some(turns);
            for event in events {
                match event {
                    TurnEvent::Started(start) => {
                        self.speech_start = start;
                        if let Some(reply) = self.reply.take()
                            && !reply.is_finished()
                        {
                            reply.abort();
                            outgoing.json(&json!({"type": "interrupted"})).await;
                        }
                    }
                    TurnEvent::Stopped(end) => self.answer(end, outgoing),
                }
            }

            if !speaking {
                let keep = position(PREFIX_MS);
                let dropped = self.buffer.len().saturating_sub(keep);
                self.buffer.drain(..dropped);
                self.buffer_start += dropped;
            }
            Ok(())
        })
    }
}

impl VoiceSession {
    /// Answers the turn whose speech stopped at `end`, in a task of its own.
    fn answer(&mut self, end: u64, outgoing: &Outgoing) {
        let first = position(self.speech_start.saturating_sub(PREFIX_MS))
            .saturating_sub(self.buffer_start)
            .min(self.buffer.len());
        let last = position(end)
            .saturating_sub(self.buffer_start)
            .clamp(first, self.buffer.len());
        let turn = self.buffer[first..last].to_vec();
        self.buffer.drain(..last);
        self.buffer_start += last;

        while self.running.try_join_next().is_some() {}
        let (voice, history, outgoing) = (
            self.voice.clone(),
            Arc::clone(&self.history),
            outgoing.clone(),
        );
        self.reply = Some(self.running.spawn(async move {
            if let Err(error) = turn_reply(turn, &voice, &history, &outgoing).await {
                outgoing.error(None, &error).await;
            }
        }));
    }
}

/// Returns the stream position `milliseconds` from its start.
fn position(milliseconds: u64) -> usize {
    usize::try_from(milliseconds * u64::from(SAMPLE_RATE) / 1000).unwrap_or(usize::MAX)
}

/// Transcribes a turn, then speaks the chat model's reply to the conversation, sentence by sentence as it generates.
async fn turn_reply(
    turn: Vec<f32>,
    voice: &Voice,
    history: &Mutex<Vec<(&'static str, String)>>,
    outgoing: &Outgoing,
) -> ApiResult<()> {
    let clip = Audio {
        samples: turn,
        rate: SAMPLE_RATE,
    };
    let transcript = voice
        .recognizer
        .call(move |pipeline| {
            Ok(downcast::<AutomaticSpeechRecognitionPipeline>(&**pipeline)?
                .run_one(&clip, &AutomaticSpeechRecognitionParameters::default())?)
        })
        .await?
        .text
        .trim()
        .to_owned();
    if transcript.is_empty() {
        return Ok(());
    }

    outgoing
        .json(&json!({"type": "transcript", "text": &transcript}))
        .await;
    let messages = {
        let mut history = history.lock().unwrap_or_else(PoisonError::into_inner);
        history.push(("user", transcript));
        [("system", INSTRUCTIONS.to_owned())]
            .into_iter()
            .chain(history.iter().cloned())
            .map(|(role, text)| Message::new(role, Content::Text(text)))
            .collect::<Vec<_>>()
    };

    let (engine, format) = &voice.chat;
    let mut generation = generate(engine, format, messages).await?;
    let (sentences, mut heard) = mpsc::unbounded_channel();
    let writing = async {
        let mut cutter = SentenceCutter::default();
        while let Some(piece) = generation.next().await? {
            if let Piece::Text { text, .. } = piece {
                for sentence in cutter.push(&text) {
                    sentences.send(sentence).ok();
                }
            }
        }

        if let Some(sentence) = cutter.finish() {
            sentences.send(sentence).ok();
        }

        drop(sentences);
        ApiResult::Ok(())
    };
    let speaking = async {
        while let Some(sentence) = heard.recv().await {
            speak(&sentence, &voice.synthesizer, outgoing).await?;
            // What was spoken stays in the conversation, even if speech then interrupts the rest.
            let mut history = history.lock().unwrap_or_else(PoisonError::into_inner);
            match history.last_mut() {
                Some(("assistant", said)) => {
                    said.push(' ');
                    said.push_str(&sentence);
                }
                _ => history.push(("assistant", sentence)),
            }
        }
        ApiResult::Ok(())
    };
    let (written, spoken) = tokio::join!(writing, speaking);
    written?;
    spoken?;
    outgoing.json(&json!({"type": "reply_done"})).await;
    Ok(())
}

/// Starts the chat model's reply to `messages`, with the model answering at once rather than thinking first.
async fn generate(
    engine: &TextEngine,
    format: &Arc<TextFormat>,
    messages: Vec<Message>,
) -> ApiResult<TextGeneration> {
    let rendering = Arc::clone(format);
    let prompt = blocking(move || {
        let mut arguments = Map::new();
        arguments.insert("enable_thinking".to_owned(), Value::Bool(false));
        let text = rendering.render(&messages, true, &arguments)?;
        rendering.encode(&text, false)
    })
    .await?;
    let limit = max_tokens(None, prompt.len(), engine.limits())?;
    let sampling = GenerationFields::default().sampling(None)?;
    let options = format.sequence(limit, &sampling, &[], false)?;
    let (stop_tokens, min_tokens) = (options.stop_tokens.clone(), options.min_tokens);
    let detokenizer = Detokenizer::new(Arc::clone(&format.tokenizer), true, &prompt)?;
    Ok(TextGeneration::new(
        engine.submit(prompt, options)?,
        detokenizer,
        Vec::new(),
        stop_tokens,
        min_tokens,
    ))
}

/// Speaks `sentence`, sending it and then its speech in frames of 100 ms.
async fn speak(sentence: &str, synthesizer: &Pipelines, outgoing: &Outgoing) -> ApiResult<()> {
    let text = sentence.replace(['*', '#', '`', '_'], "");
    let generated = synthesizer
        .call(move |pipeline| {
            Ok(downcast::<TextToAudioPipeline>(&**pipeline)?
                .run_one(&SpeechPrompt::Text(text), &TextToAudioParameters::default())?)
        })
        .await?;
    let rate = u32::try_from(generated.sampling_rate)
        .map_err(|_| ApiError::internal("a sample rate out of range"))?;
    outgoing
        .json(&json!({"type": "sentence", "text": sentence, "sample_rate": rate}))
        .await;
    for samples in generated.audio.chunks(rate as usize / 10) {
        let pcm = Audio {
            samples: samples.to_vec(),
            rate,
        }
        .to_pcm();
        outgoing.binary(pcm).await;
    }
    Ok(())
}

/// Text streamed in, given back a sentence at a time: up to a `.`, `?`, or `!` before whitespace, past a sentence of one word, or a line break; the first piece also ends at a comma after [`FIRST_CLAUSE_WORDS`] words.
#[derive(Debug, Default)]
struct SentenceCutter {
    pending: String,
    started: bool,
}

impl SentenceCutter {
    fn push(&mut self, text: &str) -> Vec<String> {
        self.pending.push_str(text);
        let mut sentences = Vec::new();
        while let Some(end) = self.boundary() {
            let rest = self.pending.split_off(end);
            let sentence = mem::replace(&mut self.pending, rest);
            let sentence = sentence.trim();
            if !sentence.is_empty() {
                sentences.push(sentence.to_owned());
                self.started = true;
            }
        }
        sentences
    }

    fn boundary(&self) -> Option<usize> {
        let mut characters = self.pending.char_indices().peekable();
        while let Some((index, character)) = characters.next() {
            let end = index + character.len_utf8();
            let spaced = characters
                .peek()
                .is_some_and(|&(_, next)| next.is_whitespace());
            let words = self.pending[..index].split_whitespace().count();
            let ends = match character {
                '\n' => true,
                '.' | '?' | '!' => spaced && words > 1,
                ',' => spaced && !self.started && words >= FIRST_CLAUSE_WORDS,
                _ => false,
            };
            if ends {
                return Some(end);
            }
        }
        None
    }

    fn finish(self) -> Option<String> {
        let rest = self.pending.trim();
        (!rest.is_empty()).then(|| rest.to_owned())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let [recognizer, chat, synthesizer, detector] = env::args()
        .skip(1)
        .collect::<Vec<_>>()
        .try_into()
        .map_err(|_| "expected RECOGNIZER CHAT SYNTHESIZER DETECTOR")?;
    let recognizer = load_model(&ModelConfig {
        voice_activity: Some(detector),
        ..ModelConfig::new(&recognizer)
    })?;
    let chat = load_model(&ModelConfig::new(&chat))?;
    let synthesizer = load_model(&ModelConfig::new(&synthesizer))?;
    let voice = match (&recognizer.engine, &chat.engine, &synthesizer.engine) {
        (Engine::Pipeline(heard), Engine::Text { engine, format }, Engine::Pipeline(spoken)) => {
            Voice {
                recognizer: heard.clone(),
                detector: recognizer
                    .voice_activity
                    .clone()
                    .ok_or("a recognizer without a voice activity detector")?,
                chat: (engine.clone(), Arc::clone(format)),
                synthesizer: spoken.clone(),
            }
        }
        _ => {
            return Err(
                "expected a speech recognizer, a chat model, and a speech synthesizer".into(),
            );
        }
    };
    let served = Served::create(vec![
        recognizer,
        chat,
        synthesizer,
        ServedModel {
            name: "voice".to_owned(),
            task: "voice".to_owned(),
            engine: Engine::Realtime(Arc::new(voice)),
            tokenizer: None,
            created: now(),
            voice_activity: None,
        },
    ])?;
    let listener = TcpListener::bind("127.0.0.1:8000").await?;
    println!("listening on ws://127.0.0.1:8000/v1/realtime/voice");
    axum::serve(listener, app(served, &ServerConfig::default())?).await?;
    Ok(())
}

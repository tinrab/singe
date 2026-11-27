//! A speech synthesizer's session: texts to speak, spoken one after another, each as `audio_start`, binary frames of PCM, and `audio_end`.

use futures::future::BoxFuture;
use singe_model::{
    head::SpeechPrompt,
    processor::audio::Audio,
    task::{AnyPipeline, Pipeline, audio::text_to_audio::TextToAudioPipeline},
};
use tokio::{sync::mpsc, task::JoinSet};

use crate::{
    api::realtime::{Speak, SynthesisEvent, SynthesisRequest},
    error::{ApiError, ApiResult},
    handler::downcast,
    realtime::{Incoming, Outgoing, RealtimeSession},
    worker::Worker,
};

/// The binary frames each second of audio is sent in.
const FRAMES_PER_SECOND: usize = 10;

/// The session of a connection to a speech synthesizer.
pub struct SynthesisSession {
    worker: Worker<Box<dyn AnyPipeline>>,
    /// The requests waiting to be spoken, once the first starts the speaker.
    queue: Option<mpsc::UnboundedSender<Speak>>,
    running: JoinSet<()>,
}

impl SynthesisSession {
    /// Creates the session of a connection to the speech synthesizer on `worker`.
    #[must_use]
    pub fn new(worker: Worker<Box<dyn AnyPipeline>>) -> Self {
        Self {
            worker,
            queue: None,
            running: JoinSet::new(),
        }
    }

    fn speak(&mut self, request: Speak, outgoing: &Outgoing) -> ApiResult<()> {
        if request.text.trim().is_empty() {
            return Err(ApiError::invalid_field("text", "no text to speak"));
        }

        let queue = self.queue.get_or_insert_with(|| {
            let (queue, requests) = mpsc::unbounded_channel();
            self.running
                .spawn(speaker(requests, self.worker.clone(), outgoing.clone()));
            queue
        });
        queue
            .send(request)
            .map_err(|_| ApiError::internal("the session's speaker stopped"))
    }
}

impl RealtimeSession for SynthesisSession {
    fn receive<'a>(
        &'a mut self,
        message: Incoming,
        outgoing: &'a Outgoing,
    ) -> BoxFuture<'a, ApiResult<()>> {
        Box::pin(async move {
            match message.parse::<SynthesisRequest>()? {
                SynthesisRequest::Speak(request) => {
                    let id = request.id.clone();
                    if let Err(error) = self.speak(request, outgoing) {
                        outgoing.error(Some(&id), &error).await;
                    }
                    Ok(())
                }
            }
        })
    }
}

/// Speaks each of `requests` in turn, until the client leaves.
async fn speaker(
    mut requests: mpsc::UnboundedReceiver<Speak>,
    worker: Worker<Box<dyn AnyPipeline>>,
    outgoing: Outgoing,
) {
    while let Some(request) = requests.recv().await {
        let id = request.id.clone();
        let spoken = match speak(request, &worker, &outgoing).await {
            Ok(spoken) => spoken,
            Err(error) => outgoing.error(Some(&id), &error).await,
        };
        if !spoken {
            return;
        }
    }
}

/// Speaks `request`, sending its audio; returns whether the client is still there.
async fn speak(
    request: Speak,
    worker: &Worker<Box<dyn AnyPipeline>>,
    outgoing: &Outgoing,
) -> ApiResult<bool> {
    let (text, parameters) = (request.text, request.parameters);
    let generated = worker
        .call(move |pipeline| {
            Ok(downcast::<TextToAudioPipeline>(&**pipeline)?
                .run_one(&SpeechPrompt::Text(text), &parameters)?)
        })
        .await?;
    let sample_rate = u32::try_from(generated.sampling_rate)
        .map_err(|_| ApiError::internal("a sample rate out of range"))?;
    let id = request.id;
    let start = SynthesisEvent::AudioStart {
        id: id.clone(),
        sample_rate,
    };
    if !outgoing.json(&start).await {
        return Ok(false);
    }

    let frame = (sample_rate as usize / FRAMES_PER_SECOND).max(1);
    for samples in generated.audio.chunks(frame) {
        let pcm = Audio {
            samples: samples.to_vec(),
            rate: sample_rate,
        }
        .to_pcm();
        if !outgoing.binary(pcm).await {
            return Ok(false);
        }
    }

    Ok(outgoing.json(&SynthesisEvent::AudioEnd { id }).await)
}

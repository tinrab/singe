//! Server-sent events as OpenAI streams them: a JSON object per event, then `data: [DONE]`.
//! A task produces the events while the client reads them; when the client leaves, sending fails and the task stops, which drops and so cancels what it was reading.

use std::convert::Infallible;

use axum::response::{
    IntoResponse, Response,
    sse::{Event, KeepAlive, Sse},
};
use serde::Serialize;
use tokio::sync::mpsc;
use tokio_stream::{StreamExt, wrappers::ReceiverStream};

use crate::error::ApiError;

/// The sending side of a [`stream`] response.
#[derive(Debug, Clone)]
pub struct EventSender {
    sender: mpsc::Sender<Event>,
}

impl EventSender {
    /// Sends `value` as an event's data; returns whether the client is still reading.
    pub async fn json<T: Serialize + Sync>(&self, value: &T) -> bool {
        match Event::default().json_data(value) {
            Ok(event) => self.sender.send(event).await.is_ok(),
            Err(error) => {
                tracing::error!("failed to write an event: {error}");
                false
            }
        }
    }

    /// Sends `error` in OpenAI's shape, as a stream's last event; returns whether the client is still reading.
    pub async fn error(&self, error: &ApiError) -> bool {
        self.json(&error.body()).await
    }

    /// Sends `data: [DONE]`, which ends an OpenAI stream; returns whether the client is still reading.
    pub async fn done(&self) -> bool {
        self.sender
            .send(Event::default().data("[DONE]"))
            .await
            .is_ok()
    }
}

/// Returns a response streaming the events `produce` sends, with keep-alive comments.
/// `produce` runs on its own task, up to 32 events ahead of the client.
pub fn stream<F>(produce: impl FnOnce(EventSender) -> F) -> Response
where
    F: Future<Output = ()> + Send + 'static,
{
    let (sender, receiver) = mpsc::channel(32);
    tokio::spawn(produce(EventSender { sender }));
    Sse::new(ReceiverStream::new(receiver).map(Ok::<_, Infallible>))
        .keep_alive(KeepAlive::default())
        .into_response()
}

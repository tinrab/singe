//! `GET /v1/realtime/{model}`: a long-lived WebSocket connection to a model, in place of a request and its response.
//!
//! A connection opens a [`RealtimeSession`], which reads each message the client sends and sends any number back through [`Outgoing`].
//! The server reads and writes apart, so a session's slow work never stops it reading; it answers a session's error with an `error` message and keeps the connection; it cancels a session's work when the client leaves, by dropping the session; and it closes connections with code 1001 when it stops.
//!
//! A model of one's own serves connections through a [`RealtimeRunner`], registered as [`Engine::Realtime`].
//! `singe-model`'s text generators, speech recognizers, and speech synthesizers serve them with no code, in the messages [`crate::api::realtime`] describes.

pub mod recognition;
pub mod synthesis;
pub mod text;

use std::sync::Arc;

use axum::{
    body::Bytes,
    extract::{
        Path, Query, State,
        ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade, close_code},
    },
    response::Response,
};
use futures::{SinkExt, StreamExt, future::BoxFuture};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Map, Value, json};
use singe_model::task::Task;
use tokio::sync::mpsc;

use crate::{
    error::{ApiError, ApiResult},
    registry::{Engine, Served, ServedModel},
    worker::ModelState,
};

pub(crate) const ENDPOINT: &str = "/v1/realtime/<model>";

/// The subprotocol the server answers a browser's handshake with, which carries its API key as a second subprotocol.
const SUBPROTOCOL: &str = "realtime";

/// The messages a session may have waiting to be written before sending one waits.
const OUTGOING_CAPACITY: usize = 256;

/// A model of one's own that serves realtime connections.
pub trait RealtimeRunner: Send + Sync + 'static {
    /// Returns whether the model can take connections, as `/health` and `/v1/models` report it.
    fn state(&self) -> ModelState;

    /// Opens the session of one connection, from the query of its URL.
    ///
    /// # Errors
    ///
    /// Returns an error the connection is refused with, before it upgrades.
    fn open(&self, query: Map<String, Value>) -> ApiResult<Box<dyn RealtimeSession>>;
}

/// The session of one connection.
/// Dropping it, when the client leaves, cancels what it started; work it spawns should end when it drops, as in a [`tokio::task::JoinSet`] it holds.
pub trait RealtimeSession: Send {
    /// Handles one message from the client, sending any number back through `outgoing`.
    /// The next message waits for it, so slow work belongs in tasks of its own.
    ///
    /// # Errors
    ///
    /// Returns an error the client receives as an `error` message; the connection stays open.
    fn receive<'a>(
        &'a mut self,
        message: Incoming,
        outgoing: &'a Outgoing,
    ) -> BoxFuture<'a, ApiResult<()>>;
}

/// A message from the client.
#[derive(Debug, Clone)]
pub enum Incoming {
    /// A text message, parsed as JSON.
    Json(Value),
    /// A binary frame.
    Binary(Bytes),
}

impl Incoming {
    /// Returns the message as `T`, refusing one that is not a JSON message of that shape.
    ///
    /// # Errors
    ///
    /// Returns a 400 error for a binary frame or a JSON message of another shape.
    pub fn parse<T: DeserializeOwned>(self) -> ApiResult<T> {
        match self {
            Self::Json(value) => serde_json::from_value(value).map_err(|error| {
                ApiError::invalid(format!("an unknown or malformed message: {error}"))
            }),
            Self::Binary(_) => Err(ApiError::invalid(
                "a binary frame, which this session does not read",
            )),
        }
    }
}

/// The messages to the client, in order; clones send on the same connection.
/// Each send waits while the client falls behind, and reports whether the client is still there.
#[derive(Debug, Clone)]
pub struct Outgoing {
    sender: mpsc::Sender<Message>,
}

impl Outgoing {
    /// Sends `value` as a JSON text message.
    pub async fn json<T: Serialize + Sync + ?Sized>(&self, value: &T) -> bool {
        match serde_json::to_string(value) {
            Ok(text) => self.send(Message::Text(text.into())).await,
            Err(error) => {
                tracing::error!("failed to write a realtime message: {error}");
                false
            }
        }
    }

    /// Sends `bytes` as a binary frame.
    pub async fn binary(&self, bytes: impl Into<Bytes>) -> bool {
        self.send(Message::Binary(bytes.into())).await
    }

    /// Sends `error` as `{"type": "error", "id", "error"}`, about the client's message or work `id` names, if any.
    pub async fn error(&self, id: Option<&str>, error: &ApiError) -> bool {
        if error.status.is_server_error() {
            tracing::error!(status = error.status.as_u16(), "{}", error.message);
        }

        let mut message = json!({"type": "error", "id": id});
        message["error"] = error.body()["error"].take();
        self.json(&message).await
    }

    async fn send(&self, message: Message) -> bool {
        self.sender.send(message).await.is_ok()
    }
}

/// Serves `GET /v1/realtime/{model}`: upgrades to a WebSocket connection with the model's session.
///
/// # Errors
///
/// Returns a 404 error for an unknown model, 422 for a model that serves no realtime session, 503 while it loads, and any error its runner refuses the connection with.
pub async fn connect(
    State(served): State<Served>,
    Path(model): Path<String>,
    Query(query): Query<Map<String, Value>>,
    upgrade: WebSocketUpgrade,
) -> ApiResult<Response> {
    let model = served.get(&model)?;
    let session = open(&model, query)?;
    let name = model.name.clone();
    Ok(upgrade
        .protocols([SUBPROTOCOL])
        .on_upgrade(move |socket| run(socket, session, served, name)))
}

/// Opens `model`'s session: its runner's, or the built-in one of its task.
fn open(model: &ServedModel, query: Map<String, Value>) -> ApiResult<Box<dyn RealtimeSession>> {
    match model.state() {
        ModelState::Ready => {}
        ModelState::Loading => return Err(ApiError::loading(&model.name)),
        ModelState::Failed(message) => {
            return Err(ApiError::internal(format!(
                "model {} failed to load: {message}",
                model.name
            )));
        }
    }

    let task = Task::from_id(&model.task).ok();
    Ok(match (&model.engine, task) {
        (Engine::Realtime(runner), _) => runner.open(query)?,
        (Engine::Text { engine, format }, _) => Box::new(text::TextSession::new(
            model.name.clone(),
            engine.clone(),
            Arc::clone(format),
        )),
        (Engine::Pipeline(worker), Some(Task::AutomaticSpeechRecognition)) => {
            Box::new(recognition::RecognitionSession::open(
                worker.clone(),
                model.voice_activity.clone(),
                &query,
            )?)
        }
        (Engine::Pipeline(worker), Some(Task::TextToAudio)) => {
            Box::new(synthesis::SynthesisSession::new(worker.clone()))
        }
        _ => return Err(ApiError::wrong_task(&model.name, &model.task, ENDPOINT)),
    })
}

/// Runs a connection until the client leaves or the server stops.
async fn run(
    socket: WebSocket,
    mut session: Box<dyn RealtimeSession>,
    served: Served,
    name: String,
) {
    tracing::info!(model = name, "realtime connection opened");
    let (mut sink, mut stream) = socket.split();
    let (sender, mut messages) = mpsc::channel(OUTGOING_CAPACITY);
    let writer = tokio::spawn(async move {
        while let Some(message) = messages.recv().await {
            if sink.send(message).await.is_err() {
                break;
            }
        }

        sink.close().await.ok();
    });

    let outgoing = Outgoing { sender };
    loop {
        let message = tokio::select! {
            message = stream.next() => message,
            () = served.stopped() => {
                let close = CloseFrame {
                    code: close_code::AWAY,
                    reason: "the server is stopping".into(),
                };
                outgoing.send(Message::Close(Some(close))).await;
                break;
            }
        };
        let incoming = match message {
            Some(Ok(Message::Text(text))) => match serde_json::from_str(text.as_str()) {
                Ok(value) => Incoming::Json(value),
                Err(error) => {
                    let error = ApiError::invalid(format!("a message that is not JSON: {error}"));
                    outgoing.error(None, &error).await;
                    continue;
                }
            },
            Some(Ok(Message::Binary(bytes))) => Incoming::Binary(bytes),
            Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue,
            Some(Ok(Message::Close(_)) | Err(_)) | None => break,
        };
        if let Err(error) = session.receive(incoming, &outgoing).await {
            outgoing.error(None, &error).await;
        }
    }

    // The session's work ends with it, and the writer once nothing can send.
    drop(session);
    drop(outgoing);
    writer.await.ok();
    tracing::info!(model = name, "realtime connection closed");
}

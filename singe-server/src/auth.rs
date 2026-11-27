//! API keys: a request carries one as `Authorization: Bearer KEY`, or a browser's WebSocket handshake as the subprotocol `singe-api-key.KEY`, compared against the server's in constant time through their SHA-256 digests.

use std::sync::Arc;

use axum::{
    extract::{Request, State},
    http::header,
    middleware::Next,
    response::{IntoResponse, Response},
};
use sha2::{Digest, Sha256};

use crate::error::ApiError;

/// The digests of the keys a server accepts.
#[derive(Debug, Clone)]
pub struct ApiKeySet {
    digests: Arc<Vec<[u8; 32]>>,
}

impl ApiKeySet {
    /// Creates the set from keys in plain text, keeping only their digests.
    #[must_use]
    pub fn new(keys: &[String]) -> Self {
        Self {
            digests: Arc::new(keys.iter().map(|key| digest(key)).collect()),
        }
    }

    /// Returns whether `key` is one of the keys.
    #[must_use]
    pub fn accepts(&self, key: &str) -> bool {
        let presented = digest(key);
        // Every digest is compared in full, so the time taken says nothing of which bytes matched.
        let matched = self.digests.iter().fold(0_u8, |matched, digest| {
            let difference = digest
                .iter()
                .zip(&presented)
                .fold(0_u8, |difference, (left, right)| {
                    difference | (left ^ right)
                });
            matched | u8::from(difference == 0)
        });
        matched != 0
    }
}

fn digest(key: &str) -> [u8; 32] {
    Sha256::digest(key.as_bytes()).into()
}

/// The prefix of the WebSocket subprotocol that carries a browser's key, which cannot set headers on a WebSocket.
const KEY_SUBPROTOCOL: &str = "singe-api-key.";

/// Passes a request whose `Authorization: Bearer` header, or whose WebSocket subprotocol `singe-api-key.KEY`, carries an accepted key on to `next`, and answers any other with [`ApiError::unauthorized`].
pub async fn require_key(State(keys): State<ApiKeySet>, request: Request, next: Next) -> Response {
    let headers = request.headers();
    let bearer = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "));
    let subprotocol = headers
        .get(header::SEC_WEBSOCKET_PROTOCOL)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| {
            value
                .split(',')
                .find_map(|protocol| protocol.trim().strip_prefix(KEY_SUBPROTOCOL))
        });
    if bearer
        .or(subprotocol)
        .is_some_and(|key| keys.accepts(key.trim()))
    {
        next.run(request).await
    } else {
        ApiError::unauthorized().into_response()
    }
}

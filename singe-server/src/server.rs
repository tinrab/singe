//! A server of `singe-model` models: each loaded on its own thread while the port already answers, the routers behind keys, CORS, body limits, request ids, and logging, and a graceful stop on SIGINT or SIGTERM.

use std::{future, io, path::Path, sync::Arc, time::Duration};

use axum::{
    Router,
    extract::DefaultBodyLimit,
    http::{HeaderValue, Method},
    middleware,
};
use singe::{DataType, Device};
#[cfg(all(feature = "onnx", feature = "cuda_13_4"))]
use singe_model::function::onnx;
#[cfg(feature = "cuda_13_4")]
use singe_model::function::program;
use singe_model::{
    LoadOptions, Source,
    function::{ExportManifest, Shapes, TensorFunction},
    processor::text::TextTokenizer,
    task::{self, Pipeline, Task, text::text_generation::TextGenerationPipeline},
    tune::Tuning,
};
use tokio::{net::TcpListener, signal, sync::watch, time::timeout};
use tower_http::{
    cors::{AllowOrigin, Any, CorsLayer},
    request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer},
    trace::{DefaultMakeSpan, DefaultOnResponse, TraceLayer},
};
use tracing::Level;

use crate::{
    api::now,
    auth::{ApiKeys, require_key},
    config::{ModelConfig, ModelSource, ServerConfig},
    error::error_chain,
    generation::{
        engine::{TextEngine, TextEngineOptions, TextLimits},
        format::TextFormat,
    },
    handler::{
        audio_router, chat_router, embedding_router, health_router, image_router, model_router,
        task_router, tensor_router, tokenize_router,
    },
    registry::{DuplicateModel, Engine, Served, ServedModel, TENSORS},
    worker::{ModelState, Worker},
};

/// The longest sequence a text generator holds unless configured otherwise.
const DEFAULT_MAX_MODEL_LEN: usize = 4096;

const PAGE_SIZE: usize = 32;

/// Why a server could not start or stopped with a failure.
#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    /// The config has no models or an allowed origin that is not a valid header value.
    #[error("invalid configuration: {0}")]
    Config(String),
    /// A model could not be opened, or failed to load on its thread.
    #[error("model {name}: {message}")]
    Model {
        /// The model's served name.
        name: String,
        /// What went wrong, with its causes.
        message: String,
    },
    /// The listener could not bind.
    #[error("failed to listen on {address}")]
    Bind {
        /// The `host:port` it tried.
        address: String,
        /// The error binding it.
        #[source]
        source: io::Error,
    },
    /// A model's thread could not be spawned, or serving failed.
    #[error("server io")]
    Io(#[from] io::Error),
    /// Two models share a name.
    #[error(transparent)]
    Duplicate(#[from] DuplicateModel),
    /// A text generator's configured limits do not fit together.
    #[error("model {name}'s limits")]
    TextLimits {
        /// The model's served name.
        name: String,
        /// Which limit is wrong.
        #[source]
        source: TextLimitError,
    },
}

/// Opens `config`'s model and starts its engine, which loads it on its own thread.
/// A text-generation model gets a [`TextEngine`]; any other task gets a [`Worker`].
///
/// # Errors
///
/// Returns an error if the model's snapshot, task, device, data type, or text format cannot be resolved, if its task has no pipeline, if a text generator's limits do not fit, or if its thread cannot be spawned.
pub fn load_model(config: &ModelConfig) -> Result<ServedModel, ServerError> {
    let name = config.served_name();
    let failed = |message: String| ServerError::Model {
        name: name.clone(),
        message,
    };
    let source = config
        .get_source()
        .map_err(|error| failed(error.to_string()))?;
    let device = config
        .get_device()
        .map_err(|error| failed(error.to_string()))?;
    let model = match source {
        ModelSource::Snapshot(model) => model,
        ModelSource::Program(directory) => {
            if !directory.join(ExportManifest::FILE).is_file() {
                return Err(failed(format!(
                    "{} has no {}; --program serves what `singe export --target cuda` writes for single-pass tasks and ONNX files",
                    directory.display(),
                    ExportManifest::FILE
                )));
            }

            if device != Device::Cuda(0) {
                return Err(failed(
                    "an exported program on a device other than cuda:0".to_owned(),
                ));
            }

            let manifest =
                ExportManifest::read(directory).map_err(|error| failed(error_chain(&error)))?;
            let directory = directory.to_owned();
            return load_external(config, name, device, Some(&manifest), move || {
                open_program(&directory)
            });
        }
        ModelSource::Onnx(file) => {
            let file = file.to_owned();
            let shapes = Shapes {
                dimensions: config.dims.clone(),
                ranges: config.ranges.clone(),
                inputs: config.inputs.clone(),
            };
            let tuning = config
                .tune
                .get_tuning()
                .map_err(|error| failed(error_chain(&error)))?;
            return load_external(config, name, device, None, move || {
                open_onnx(&file, &shapes, device, tuning.as_ref())
            });
        }
    };
    let model_error = |error: singe_model::Error| failed(error_chain(&error));
    let source = Source::find(model).map_err(model_error)?;
    let task = config
        .task
        .as_deref()
        .map_or_else(|| Task::default_for(&source), Task::from_id)
        .map_err(model_error)?;
    if task.heads().is_empty() {
        return Err(failed(format!("task {} has no pipeline yet", task.id())));
    }

    let options = LoadOptions::new(
        device,
        config.data_type.resolve(&source).map_err(model_error)?,
    );
    let tokenizer = TextTokenizer::load(&source).ok().map(Arc::new);

    let engine = if task == Task::TextGeneration {
        let format = TextFormat::load(&source).map_err(model_error)?;
        let engine_options = text_options(config, format.context_length).map_err(|source| {
            ServerError::TextLimits {
                name: name.clone(),
                source,
            }
        })?;
        let engine = TextEngine::spawn(&name, engine_options, move || {
            Ok(TextGenerationPipeline::load(&source, options)?)
        })?;
        Engine::Text {
            engine,
            format: Arc::new(format),
        }
    } else {
        let worker = Worker::spawn(&name, config.max_queued_requests, move || {
            Ok(task::load(task, &source, options)?)
        })?;
        Engine::Pipeline(worker)
    };
    Ok(ServedModel {
        name,
        task: task.id().to_owned(),
        engine,
        tokenizer,
        created: now(),
    })
}

fn load_external(
    config: &ModelConfig,
    name: String,
    device: Device,
    manifest: Option<&ExportManifest>,
    open: impl FnOnce() -> singe_model::Result<Arc<dyn TensorFunction>> + Send + 'static,
) -> Result<ServedModel, ServerError> {
    // A program exported for a task records the snapshot it came from, whose processor reads its inputs.
    let exported_task = manifest.and_then(|manifest| manifest.task.as_deref());
    let exported_source = manifest
        .filter(|manifest| manifest.task.is_some())
        .map(|manifest| manifest.source.clone());
    let failed = |message: String| ServerError::Model {
        name: name.clone(),
        message,
    };
    let model_error = |error: singe_model::Error| failed(error_chain(&error));
    let processor = config
        .processor
        .clone()
        .or(exported_source)
        .map(|processor| Source::find(&processor))
        .transpose()
        .map_err(model_error)?;
    let task = match (config.task.as_deref().or(exported_task), &processor) {
        (Some(task), _) => Some(Task::from_id(task).map_err(model_error)?),
        (None, Some(processor)) => Some(Task::default_for(processor).map_err(model_error)?),
        (None, None) => None,
    };
    let Some(task) = task else {
        let worker = Worker::spawn(&name, config.max_queued_requests, move || Ok(open()?))?;
        return Ok(ServedModel {
            name,
            task: TENSORS.to_owned(),
            engine: Engine::Tensors(worker),
            tokenizer: None,
            created: now(),
        });
    };

    if !task.can_run_external() {
        let served = Task::ALL
            .iter()
            .filter(|task| task.can_run_external())
            .map(|task| task.id())
            .collect::<Vec<_>>();
        return Err(failed(format!(
            "task {} from an exported program or an ONNX file, which serves {}",
            task.id(),
            served.join(", ")
        )));
    }

    let processor = processor.ok_or_else(|| {
        failed(format!(
            "task {} without a processor: the snapshot whose tokenizer or image processor and config the task reads",
            task.id()
        ))
    })?;
    let options = LoadOptions::new(device, DataType::F32);
    let tokenizer = TextTokenizer::load(&processor).ok().map(Arc::new);
    let worker = Worker::spawn(&name, config.max_queued_requests, move || {
        let source = processor.with_function(open()?);
        Ok(task::load(task, &source, options)?)
    })?;
    Ok(ServedModel {
        name,
        task: task.id().to_owned(),
        engine: Engine::Pipeline(worker),
        tokenizer,
        created: now(),
    })
}

#[cfg(feature = "cuda_13_4")]
fn open_program(directory: &Path) -> singe_model::Result<Arc<dyn TensorFunction>> {
    program::open(directory)
}

#[cfg(not(feature = "cuda_13_4"))]
fn open_program(_directory: &Path) -> singe_model::Result<Arc<dyn TensorFunction>> {
    Err(singe_model::Error::unsupported(
        "serving an exported program in a build without CUDA",
    ))
}

#[cfg(all(feature = "onnx", feature = "cuda_13_4"))]
fn open_onnx(
    file: &Path,
    shapes: &Shapes,
    device: Device,
    tuning: Option<&Tuning>,
) -> singe_model::Result<Arc<dyn TensorFunction>> {
    onnx::open(file, shapes, device, tuning)
}

#[cfg(not(all(feature = "onnx", feature = "cuda_13_4")))]
fn open_onnx(
    _file: &Path,
    _shapes: &Shapes,
    _device: Device,
    _tuning: Option<&Tuning>,
) -> singe_model::Result<Arc<dyn TensorFunction>> {
    Err(singe_model::Error::unsupported(
        "serving an ONNX file in a build without ONNX and CUDA",
    ))
}

fn text_options(
    config: &ModelConfig,
    context: Option<usize>,
) -> Result<TextEngineOptions, TextLimitError> {
    let max_model_len = match (config.max_model_len, context) {
        (Some(length), Some(context)) if length > context => {
            return Err(TextLimitError::BeyondContext { length, context });
        }
        (Some(length), _) => length,
        (None, context) => context
            .unwrap_or(DEFAULT_MAX_MODEL_LEN)
            .min(DEFAULT_MAX_MODEL_LEN),
    };
    if config.max_num_seqs == 0 {
        return Err(TextLimitError::NoSequences);
    }

    let limits = TextLimits {
        max_model_len,
        cache_tokens: 0,
        page_size: PAGE_SIZE,
    };
    let longest = limits.reservation(max_model_len);
    let cache_tokens = config.cache_tokens.unwrap_or(longest * config.max_num_seqs);
    if cache_tokens < longest {
        return Err(TextLimitError::SmallCache {
            cache_tokens,
            max_model_len,
        });
    }

    Ok(TextEngineOptions {
        max_num_seqs: config.max_num_seqs,
        max_model_len,
        cache_tokens: limits.reservation(cache_tokens),
        page_size: PAGE_SIZE,
        compile: !config.eager,
        max_queued_requests: config.max_queued_requests,
        seed: config.seed.unwrap_or(0),
    })
}

/// A text generator's configured limits that do not fit together.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TextLimitError {
    /// `max_model_len` exceeds the model's own context.
    #[error("max_model_len {length} exceeds the model's context of {context}")]
    BeyondContext {
        /// The configured `max_model_len`.
        length: usize,
        /// The model's context length.
        context: usize,
    },
    /// `max_num_seqs` is zero.
    #[error("max_num_seqs of zero")]
    NoSequences,
    /// `cache_tokens` cannot hold one sequence of `max_model_len`.
    #[error("cache_tokens {cache_tokens} hold less than one sequence of {max_model_len}")]
    SmallCache {
        /// The configured `cache_tokens`.
        cache_tokens: usize,
        /// The sequence length in effect.
        max_model_len: usize,
    },
}

/// Builds the application: every router over `served`, with `config`'s API keys on all but `/health`, its CORS origins and body limit, and `x-request-id` and request tracing on all.
///
/// # Errors
///
/// Returns [`ServerError::Config`] if an allowed origin is not a valid header value.
pub fn app(served: Served, config: &ServerConfig) -> Result<Router, ServerError> {
    let mut api = Router::new()
        .merge(chat_router())
        .merge(embedding_router())
        .merge(audio_router())
        .merge(image_router())
        .merge(task_router())
        .merge(tensor_router())
        .merge(model_router())
        .merge(tokenize_router());
    if !config.api_keys.is_empty() {
        api = api.layer(middleware::from_fn_with_state(
            ApiKeys::new(&config.api_keys),
            require_key,
        ));
    }

    let mut app = health_router()
        .merge(api)
        .with_state(served)
        .layer(DefaultBodyLimit::max(config.max_body_size));
    if !config.allowed_origins.is_empty() {
        let origins = if config.allowed_origins.iter().any(|origin| origin == "*") {
            AllowOrigin::from(Any)
        } else {
            AllowOrigin::list(
                config
                    .allowed_origins
                    .iter()
                    .map(|origin| {
                        HeaderValue::from_str(origin)
                            .map_err(|_| ServerError::Config(format!("an allowed origin {origin}")))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
        };
        app = app.layer(
            CorsLayer::new()
                .allow_origin(origins)
                .allow_methods([Method::GET, Method::POST])
                .allow_headers(Any),
        );
    }
    Ok(app
        .layer(PropagateRequestIdLayer::x_request_id())
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(DefaultMakeSpan::new().level(Level::INFO))
                .on_response(DefaultOnResponse::new().level(Level::INFO))
                .on_failure(()),
        )
        .layer(SetRequestIdLayer::x_request_id(MakeRequestUuid)))
}

/// Serves `config` until SIGINT or SIGTERM, or until a model fails to load.
/// The port answers while models load; on stopping, `/health` fails and running requests get `shutdown_grace_seconds` to finish before their connections close.
///
/// # Errors
///
/// Returns an error if the config has no models, a model cannot be opened or fails to load, two models share a name, or the address cannot be bound.
pub async fn serve(config: ServerConfig) -> Result<(), ServerError> {
    if config.models.is_empty() {
        return Err(ServerError::Config("no models to serve".to_owned()));
    }

    let names = config
        .models
        .iter()
        .map(ModelConfig::served_name)
        .collect::<Vec<_>>();
    for (index, name) in names.iter().enumerate() {
        if names[..index].contains(name) {
            return Err(DuplicateModel(name.clone()).into());
        }
    }

    let models = config
        .models
        .iter()
        .map(load_model)
        .collect::<Result<Vec<_>, _>>()?;
    let served = Served::create(models)?;
    let app = app(served.clone(), &config)?;
    let address = format!("{}:{}", config.host, config.port);
    let listener = TcpListener::bind(&address)
        .await
        .map_err(|source| ServerError::Bind {
            address: address.clone(),
            source,
        })?;
    tracing::info!("listening on http://{address}");

    let (failures, mut failed) = watch::channel(None::<(String, String)>);
    for model in served.models() {
        let model = Arc::clone(model);
        let failures = failures.clone();
        tokio::spawn(async move {
            match model.loaded().await {
                ModelState::Ready => {
                    tracing::info!(model = model.name, task = model.task, "loaded");
                }
                ModelState::Failed(message) => {
                    failures.send_replace(Some((model.name.clone(), message)));
                }
                ModelState::Loading => {}
            }
        });
    }

    let (stopping, stop) = watch::channel(false);
    let server = axum::serve(listener, app).with_graceful_shutdown({
        let mut stop = stop.clone();
        async move {
            stop.wait_for(|stopping| *stopping).await.ok();
        }
    });
    let server = server.into_future();
    tokio::pin!(server);
    let failure = tokio::select! {
        result = &mut server => return Ok(result?),
        () = stop_requested() => None,
        failure = failed.wait_for(Option::is_some) => failure.ok().and_then(|failure| failure.clone()),
    };
    tracing::info!("stopping");
    served.stop();
    stopping.send_replace(true);
    let grace = Duration::from_secs(config.shutdown_grace_seconds);
    if timeout(grace, server).await.is_err() {
        tracing::warn!("closing the connections still open after {grace:?}");
    }
    failure.map_or(Ok(()), |(name, message)| {
        Err(ServerError::Model { name, message })
    })
}

/// Resolves on SIGINT, or on SIGTERM where there is one.
async fn stop_requested() {
    let interrupt = async {
        signal::ctrl_c().await.ok();
    };
    #[cfg(unix)]
    let terminate = async {
        match signal::unix::signal(signal::unix::SignalKind::terminate()) {
            Ok(mut terminate) => {
                terminate.recv().await;
            }
            Err(_) => future::pending().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = future::pending::<()>();
    tokio::select! {
        () = interrupt => {}
        () = terminate => {}
    }
}

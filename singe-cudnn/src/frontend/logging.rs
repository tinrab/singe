//! C++ frontend-compatible logging controls, independent of the native cuDNN callback.
//! `CUDNN_FRONTEND_LOG_INFO` enables logging when positive and enables tensor dumps for levels 1 through 9.
//! `CUDNN_FRONTEND_LOG_FILE` selects `stdout`, `stderr`, or a file path; an absent destination disables logging.

mod dump;

use std::{
    env,
    fmt::{self, Debug, Formatter},
    fs::File,
    io::{self, BufWriter, Write},
    mem,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

pub(crate) use crate::frontend::logging::dump::log_variant_pack;
pub use crate::frontend::logging::dump::{
    TensorDump, TensorDumpDisplay, TensorDumpFormat, dump_tensor, format_tensor_bytes,
};
use crate::{
    error::{Error, Result},
    frontend::error::Error as FrontendError,
};

/// Destination for frontend messages and tensor dumps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LogDestination {
    /// Writes frontend diagnostics to standard output.
    Stdout,
    /// Writes frontend diagnostics to standard error.
    Stderr,
    /// Opens or truncates this file when the configuration is installed.
    File(PathBuf),
}

/// Logging configuration; levels follow the C++ frontend environment convention.
#[derive(Debug, Clone)]
pub struct LoggingConfig {
    /// Enables messages when positive; levels 1 through 9 also enable tensor dumps unless overridden.
    pub level: i32,
    /// Writer destination opened when constructing or installing the logger.
    pub destination: LogDestination,
    /// Overrides the level-derived tensor-dump switch when supplied.
    pub tensor_dumps: Option<bool>,
}

impl LoggingConfig {
    /// Selects a destination at level one with level-derived tensor dumping.
    #[must_use]
    pub fn new(destination: LogDestination) -> Self {
        Self {
            level: 1,
            destination,
            tensor_dumps: None,
        }
    }

    /// Reads the C++ frontend environment names without opening any files.
    /// Paths that merely begin with `stdout` or `stderr` remain ordinary file paths.
    ///
    /// # Errors
    ///
    /// Returns an error if a present logging level cannot be parsed as an `i32`, including non-Unicode values. If the destination variable is absent, the level is not read.
    pub fn from_environment() -> Result<Option<Self>> {
        let Some(destination) = env::var_os("CUDNN_FRONTEND_LOG_FILE") else {
            return Ok(None);
        };
        let level = match env::var_os("CUDNN_FRONTEND_LOG_INFO") {
            None => 0,
            Some(value) => {
                let invalid = || FrontendError::LoggingLevelInvalid {
                    value: value.to_string_lossy().into_owned(),
                };
                value
                    .to_str()
                    .ok_or_else(invalid)?
                    .trim()
                    .parse()
                    .map_err(|_| invalid())?
            }
        };
        if level <= 0 {
            return Ok(None);
        }
        let destination = if destination == "stdout" {
            LogDestination::Stdout
        } else if destination == "stderr" {
            LogDestination::Stderr
        } else {
            LogDestination::File(destination.into())
        };
        Ok(Some(Self {
            level,
            destination,
            tensor_dumps: None,
        }))
    }
}

/// A shared message writer with independent logging and tensor-dump switches.
/// A message and its trailing newline are written under one lock.
/// Buffered formatting is flushed after every message.
/// Custom writers can collect frontend diagnostics without changing native cuDNN callbacks.
pub struct Logger {
    enabled: AtomicBool,
    tensor_dumps: AtomicBool,
    writer: Mutex<BufWriter<Box<dyn Write + Send>>>,
}

impl Debug for Logger {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Logger").finish_non_exhaustive()
    }
}

impl Logger {
    /// Uses an application-provided writer, initially enabling messages and tensor dumps.
    pub fn new(writer: impl Write + Send + 'static) -> Self {
        Self {
            enabled: AtomicBool::new(true),
            tensor_dumps: AtomicBool::new(true),
            writer: Mutex::new(BufWriter::new(Box::new(writer))),
        }
    }

    /// Opens a configured destination and applies the level and optional dump switch.
    ///
    /// # Errors
    /// Returns the destination's file-open error.
    pub fn create(config: &LoggingConfig) -> Result<Self> {
        let writer: Box<dyn Write + Send> = match &config.destination {
            LogDestination::Stdout => Box::new(io::stdout()),
            LogDestination::Stderr => Box::new(io::stderr()),
            LogDestination::File(path) => Box::new(File::create(path).map_err(logging_io)?),
        };
        Ok(Self {
            enabled: AtomicBool::new(config.level > 0),
            tensor_dumps: AtomicBool::new(
                config
                    .tensor_dumps
                    .unwrap_or_else(|| (1..10).contains(&config.level)),
            ),
            writer: Mutex::new(BufWriter::new(writer)),
        })
    }

    /// Returns whether messages are enabled.
    pub fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Enables or disables messages without changing the retained tensor-dump preference.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Returns whether both messages and tensor dumps are enabled.
    pub fn tensor_dump_enabled(&self) -> bool {
        self.enabled() && self.tensor_dumps.load(Ordering::Relaxed)
    }

    /// Replaces the tensor-dump preference; dumps still require messages to be enabled.
    pub fn set_tensor_dump_enabled(&self, enabled: bool) {
        self.tensor_dumps.store(enabled, Ordering::Relaxed);
    }

    /// Writes and flushes one C++ frontend-style labeled message.
    ///
    /// # Errors
    /// Returns writer-lock or destination I/O errors.
    pub fn message(&self, message: fmt::Arguments<'_>) -> Result<()> {
        if !self.enabled() {
            return Ok(());
        }
        let mut writer = self
            .writer
            .lock()
            .map_err(|_| FrontendError::LoggingLockPoisoned { name: "writer" })?;
        writer
            .write_fmt(format_args!("[cudnn_frontend] {message}\n"))
            .map_err(logging_io)?;
        writer.flush().map_err(logging_io)
    }

    /// Internal diagnostics must not replace the outcome of the operation being described.
    fn diagnostic(&self, message: fmt::Arguments<'_>) {
        let _ = self.message(message);
    }
}

enum LoggingState {
    Environment,
    Configured(Option<Arc<Logger>>),
}

static STATE: Mutex<LoggingState> = Mutex::new(LoggingState::Environment);

/// Returns the active logger, loading the environment once on first use.
///
/// # Errors
/// Returns invalid environment, file-open or configuration-lock errors.
pub fn logger() -> Result<Option<Arc<Logger>>> {
    let mut state = STATE
        .lock()
        .map_err(|_| FrontendError::LoggingLockPoisoned {
            name: "configuration",
        })?;
    let result = match &*state {
        LoggingState::Configured(logger) => Ok(logger.clone()),
        LoggingState::Environment => {
            let configured = LoggingConfig::from_environment()?
                .as_ref()
                .map(Logger::create)
                .transpose()?
                .map(Arc::new);
            *state = LoggingState::Configured(configured.clone());
            Ok(configured)
        }
    };
    drop(state);
    result
}

/// Installs an application logger, or disables frontend logging with `None`.
/// Existing operations may finish writing to their retained logger.
/// Releases the previous logger outside the configuration lock, allowing writer cleanup to reconfigure logging.
///
/// # Errors
///
/// Returns an error if the global configuration mutex is poisoned.
pub fn set_logger(logger: Option<Arc<Logger>>) -> Result<()> {
    let mut state = STATE
        .lock()
        .map_err(|_| FrontendError::LoggingLockPoisoned {
            name: "configuration",
        })?;
    let previous = mem::replace(&mut *state, LoggingState::Configured(logger));
    drop(state);
    // Custom writer flushing and destruction may call application code.
    drop(previous);
    Ok(())
}

/// Installs a configuration after successfully opening its destination.
/// `None` disables logging without consulting the environment again.
///
/// # Errors
///
/// Returns an error if opening the configured destination fails or the global configuration mutex is poisoned.
pub fn configure(config: Option<&LoggingConfig>) -> Result<()> {
    let logger = config.map(Logger::create).transpose()?.map(Arc::new);
    set_logger(logger)
}

/// Reloads the C++ frontend environment controls explicitly.
///
/// # Errors
///
/// Returns an error for an invalid environment logging level, destination file-open failure, or poisoned configuration mutex.
pub fn configure_from_environment() -> Result<()> {
    configure(LoggingConfig::from_environment()?.as_ref())
}

/// Writes through the active logger when frontend messages are enabled.
/// Explicit calls return logger configuration and writer errors.
/// Library diagnostics use best-effort delivery independently of this API.
///
/// # Errors
///
/// Returns an error if loading the active logger fails, the writer mutex is poisoned, or writing or flushing the message fails.
pub fn message(message: fmt::Arguments<'_>) -> Result<()> {
    if let Some(logger) = logger()? {
        logger.message(message)?;
    }
    Ok(())
}

/// Emits internal diagnostics without turning logger configuration or writer failures into operation errors.
/// Writes a formatted diagnostic to the installed frontend logger, if any.
macro_rules! diagnostic {
    ($($argument:tt)*) => {
        $crate::frontend::logging::write_diagnostic(format_args!($($argument)*))
    };
}

pub(crate) use diagnostic;

pub(crate) fn write_diagnostic(message: fmt::Arguments<'_>) {
    if let Ok(Some(logger)) = logger() {
        logger.diagnostic(message);
    }
}

fn logging_io(source: io::Error) -> Error {
    FrontendError::LoggingIo { source }.into()
}

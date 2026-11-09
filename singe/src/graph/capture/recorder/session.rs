use std::cell::RefCell;

use crate::{
    device::Device,
    dispatch::{DispatchKey, excluded},
    error::{CaptureError, Error, Result},
    graph::{
        CaptureBindings, CaptureOptions, DimensionExpr, Graph, GraphInputSpec,
        capture::recorder::CaptureSession,
    },
    tensor::Tensor,
};

thread_local! {
    static ACTIVE_CAPTURE: RefCell<Option<CaptureSession>> = const { RefCell::new(None) };
}

struct ActiveCaptureGuard;

impl Drop for ActiveCaptureGuard {
    fn drop(&mut self) {
        ACTIVE_CAPTURE.with(|active| {
            if let Ok(mut active) = active.try_borrow_mut() {
                *active = None;
            }
        });
    }
}

impl ActiveCaptureGuard {
    fn ensure_inactive() -> Result<()> {
        ACTIVE_CAPTURE.with(|active| {
            let active = active
                .try_borrow()
                .map_err(|_| Error::from(CaptureError::SessionStillBorrowed))?;
            if active.is_some() {
                return Err(CaptureError::AlreadyActive.into());
            }
            Ok(())
        })
    }

    fn activate(session: CaptureSession) -> Result<Self> {
        ACTIVE_CAPTURE.with(|active| {
            let mut active = active
                .try_borrow_mut()
                .map_err(|_| Error::from(CaptureError::SessionStillBorrowed))?;
            debug_assert!(active.is_none());
            *active = Some(session);
            Ok(Self)
        })
    }

    #[expect(
        clippy::unused_self,
        reason = "consuming the guard ends the active capture and runs cleanup on every return path"
    )]
    fn finish(self) -> Result<CaptureSession> {
        ACTIVE_CAPTURE.with(|active| {
            active
                .try_borrow_mut()
                .map_err(|_| Error::from(CaptureError::SessionStillBorrowed))?
                .take()
                .ok_or_else(|| Error::from(CaptureError::NotActive))
        })
    }
}

/// Returns whether a capture is active on this thread.
pub fn is_active() -> bool {
    ACTIVE_CAPTURE.with(|active| active.try_borrow().map_or(true, |active| active.is_some()))
}

/// Calls `record` with the active capture session, unless capture is inactive or excluded on this thread.
pub fn with_active_session(record: impl FnOnce(&mut CaptureSession) -> Result<()>) -> Result<()> {
    if excluded().contains(DispatchKey::Capture) {
        return Ok(());
    }
    with_session(record)
}

/// Calls `record` with the active capture session, if any, even while the capture key runs.
pub fn with_session(record: impl FnOnce(&mut CaptureSession) -> Result<()>) -> Result<()> {
    ACTIVE_CAPTURE.with(|active| {
        let mut active = active
            .try_borrow_mut()
            .map_err(|_| Error::from(CaptureError::SessionStillBorrowed))?;
        let Some(session) = active.as_mut() else {
            return Ok(());
        };
        record(session)
    })
}

impl Graph {
    /// Captures eager tensor operations into a graph, returning it with the outputs `f` computed eagerly.
    ///
    /// Each input tensor is aliased before calling `f`, so operations recorded during capture refer to graph inputs rather than the caller's original tensor ids.
    /// Tensor operations still execute eagerly while being recorded.
    ///
    /// # Errors
    ///
    /// Returns an error if capture is already active on this thread, input names are duplicated, `f` returns an error, no outputs are returned, or capture state cannot be borrowed.
    pub fn capture<I, S, F>(inputs: I, f: F) -> Result<(Self, Vec<Tensor>)>
    where
        I: IntoIterator<Item = (S, Tensor)>,
        S: Into<String>,
        F: FnOnce(&[Tensor]) -> Result<Vec<Tensor>>,
    {
        Self::capture_with_options(inputs, CaptureBindings::new(), CaptureOptions::new(), f)
    }

    /// Captures with explicit closed-over state and literal constants.
    ///
    /// # Errors
    ///
    /// Returns an error if another capture is active, input or closed-over binding registration fails, the capture session cannot be borrowed or finalized, `f` fails, or it produces no recorded outputs.
    pub fn capture_with_bindings<I, S, F>(
        inputs: I,
        bindings: CaptureBindings,
        f: F,
    ) -> Result<(Self, Vec<Tensor>)>
    where
        I: IntoIterator<Item = (S, Tensor)>,
        S: Into<String>,
        F: FnOnce(&[Tensor]) -> Result<Vec<Tensor>>,
    {
        Self::capture_with_options(inputs, bindings, CaptureOptions::new(), f)
    }

    /// Executes `f` eagerly while recording a graph with explicit closed-over bindings and observation options.
    /// The closure receives tracked input tensors in declaration order.
    /// The active capture is removed when recording finishes or unwinds.
    ///
    /// # Errors
    ///
    /// Returns an error if another capture is active, input or closed-over binding registration fails, the capture session cannot be borrowed or finalized, `f` fails, or it produces no recorded outputs.
    pub fn capture_with_options<I, S, F>(
        inputs: I,
        bindings: CaptureBindings,
        options: CaptureOptions,
        f: F,
    ) -> Result<(Self, Vec<Tensor>)>
    where
        I: IntoIterator<Item = (S, Tensor)>,
        S: Into<String>,
        F: FnOnce(&[Tensor]) -> Result<Vec<Tensor>>,
    {
        ActiveCaptureGuard::ensure_inactive()?;
        let (session, tensors) = CaptureSession::with_inputs(inputs, bindings, options)?;
        let guard = ActiveCaptureGuard::activate(session)?;

        let captured_outputs = f(&tensors);
        let session = guard.finish()?;
        let outputs = captured_outputs?;
        if outputs.is_empty() {
            return Err(CaptureError::MissingOutput.into());
        }

        session.finish(outputs)
    }

    /// Builds a graph from input metadata, without storage or kernels: inputs are `Device::Meta` tensors, so `f` runs only to record what it does.
    ///
    /// Declare run-time axes with [`GraphInputSpec::with_dimension`], and use [`Tensor::dimension`] and [`Tensor::reshape_with`] where a shape depends on them.
    /// Bindings stay as given, so bind `Device::Meta` tensors to keep the build free of storage.
    ///
    /// # Errors
    ///
    /// Returns an error if another capture is active, an input is invalid, a repeated symbol has different representative dimensions, `f` fails, or it produces no outputs.
    pub fn build<I, S, V, F>(inputs: I, bindings: CaptureBindings, f: F) -> Result<Self>
    where
        I: IntoIterator<Item = (S, V)>,
        S: Into<String>,
        V: Into<GraphInputSpec>,
        F: FnOnce(&[Tensor]) -> Result<Vec<Tensor>>,
    {
        let mut options = CaptureOptions::new();
        let inputs = inputs
            .into_iter()
            .map(|(name, spec)| {
                let name = name.into();
                let (spec, dimensions) = spec.into().into_parts(&name);
                options.dimensions.extend(dimensions);
                let tensor =
                    Tensor::zeroed_on(Device::Meta, spec.data_type(), spec.shape().clone())?;
                Ok((name, tensor))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::capture_with_options(inputs, bindings, options, f)?.0)
    }
}

// Returns an expression for one axis of `tensor` under the capture recording this thread, or `None` when nothing records it.
pub fn dimension(tensor: &Tensor, axis: usize) -> Result<Option<DimensionExpr>> {
    let mut expression = None;
    with_active_session(|session| {
        expression = session.dimension(tensor, axis)?;
        Ok(())
    })?;
    Ok(expression)
}

// Evaluates `dimensions` under the capture recording this thread, or as constants when nothing records.
pub fn evaluate_dimensions(dimensions: &[DimensionExpr]) -> Result<Vec<i64>> {
    let mut evaluated = None;
    with_active_session(|session| {
        evaluated = Some(session.evaluate_dimensions(dimensions)?);
        Ok(())
    })?;
    evaluated.map_or_else(
        || {
            dimensions
                .iter()
                .map(|dimension| {
                    dimension.evaluate(&mut |_, _| {
                        Err(Error::invalid_argument(
                            "graph dimension outside its capture",
                        ))
                    })
                })
                .collect()
        },
        Ok,
    )
}

// Records that `output` takes its shape from `dimensions` on each run, under the capture recording this thread.
pub fn record_reshape(output: &Tensor, dimensions: &[DimensionExpr]) -> Result<()> {
    with_active_session(|session| session.record_reshape(output, dimensions))
}

pub fn register_constant(name: String, tensor: &Tensor) -> Result<Tensor> {
    ACTIVE_CAPTURE.with(|active| {
        let mut active = active
            .try_borrow_mut()
            .map_err(|_| Error::from(CaptureError::SessionStillBorrowed))?;
        active
            .as_mut()
            .ok_or_else(|| Error::from(CaptureError::NotActive))?
            .explicit_constant(name, tensor)
    })
}

mod binary;
mod conversion;
mod copy;
mod fill;
mod filtering;
mod normalize;
mod operations;
mod scaled_constant;
mod statistics;
mod statistics_error_methods;
mod statistics_extremum_methods;
mod statistics_methods;
mod statistics_output_methods;
mod unary;

use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    pipeline::{SignalAllocator, Workspace, memory_context::validate_device_memory_stream_context},
    signal::{
        memory::Signal,
        view::{SignalView, SignalViewMut},
    },
};

pub use copy::CopySignal;
pub use statistics::{SignalIndexedValue, SignalMeanStandardDeviation, SignalMinMax};

#[derive(Debug)]
#[non_exhaustive]
pub enum SignalBacking<'a, T> {
    Borrowed(SignalView<'a, T>),
    Owned(Signal<T>),
}

pub struct SignalPipeline<'a, T> {
    stream_context: &'a StreamContext,
    workspace: &'a mut Workspace,
    backing: SignalBacking<'a, T>,
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
{
    pub fn create(
        stream_context: &'a StreamContext,
        workspace: &'a mut Workspace,
        len: usize,
    ) -> Result<Self>
    where
        Workspace: SignalAllocator<T>,
    {
        let signal = workspace.signal::<T>(len)?;
        Ok(Self::from_owned(stream_context, workspace, signal))
    }

    pub fn from_view(
        stream_context: &'a StreamContext,
        workspace: &'a mut Workspace,
        source: SignalView<'a, T>,
    ) -> Self {
        Self {
            stream_context,
            workspace,
            backing: SignalBacking::Borrowed(source),
        }
    }

    pub fn from_owned(
        stream_context: &'a StreamContext,
        workspace: &'a mut Workspace,
        signal: Signal<T>,
    ) -> Self {
        Self {
            stream_context,
            workspace,
            backing: SignalBacking::Owned(signal),
        }
    }

    pub fn from_memory(
        stream_context: &'a StreamContext,
        workspace: &'a mut Workspace,
        memory: &'a DeviceMemory<T>,
        len: usize,
    ) -> Result<Self> {
        validate_device_memory_stream_context(stream_context, memory)?;
        Ok(Self::from_view(
            stream_context,
            workspace,
            SignalView::from_memory(memory, len)?,
        ))
    }

    pub const fn len(&self) -> usize {
        match &self.backing {
            SignalBacking::Borrowed(view) => view.len(),
            SignalBacking::Owned(signal) => signal.len(),
        }
    }

    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub const fn is_owned(&self) -> bool {
        matches!(self.backing, SignalBacking::Owned(_))
    }

    fn view(&self) -> Result<SignalView<'_, T>> {
        match &self.backing {
            SignalBacking::Borrowed(view) => Ok(*view),
            SignalBacking::Owned(signal) => signal.view(),
        }
    }

    pub fn into_backing(self) -> SignalBacking<'a, T> {
        self.backing
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: CopySignal<T>,
{
    pub fn finish(self) -> Result<Signal<T>> {
        match self.backing {
            SignalBacking::Owned(signal) => Ok(signal),
            SignalBacking::Borrowed(source) => {
                let mut destination = self.workspace.signal::<T>(source.len())?;
                let mut destination_view = destination.view_mut()?;
                <Self as CopySignal<T>>::copy(self.stream_context, &source, &mut destination_view)?;
                Ok(destination)
            }
        }
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Self: CopySignal<T>,
{
    pub fn copy_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as CopySignal<T>>::copy(stream_context, source, destination)
    }
}

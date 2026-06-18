use crate::{
    context::StreamContext,
    error::{Error, Result},
    pipeline::{SignalAllocator, Workspace},
    signal::{
        filtering,
        view::{SignalView, SignalViewMut},
    },
    workspace::ScratchBuffer,
};

use super::{SignalBacking, SignalPipeline};

impl<'a> SignalPipeline<'a, i32>
where
    Workspace: SignalAllocator<i32>,
{
    pub fn integral_buffer_size(source: &SignalView<'_, i32>) -> Result<usize> {
        filtering::integral_buffer_size(source)
    }

    pub fn integral_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, i32>,
        destination: &mut SignalViewMut<'_, i32>,
    ) -> Result<()> {
        filtering::integral(stream_context, source, destination)
    }

    pub fn integral_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, i32>,
        destination: &mut SignalViewMut<'_, i32>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        filtering::integral_with_scratch(stream_context, source, destination, scratch)
    }

    pub fn integral(self) -> Result<Self> {
        let destination_len = self.len().checked_add(1).ok_or_else(|| Error::OutOfRange {
            name: "destination length".into(),
        })?;
        let mut destination = self.workspace.signal::<i32>(destination_len)?;
        let required_bytes = {
            let source = self.view()?;
            filtering::integral_buffer_size(&source)?
        };
        let mut scratch = self.workspace.scratch(required_bytes)?;

        let operation_result = {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            filtering::integral_with_scratch(
                self.stream_context,
                &source,
                &mut destination_view,
                &mut scratch,
            )
        };

        let recycle_result = self
            .workspace
            .recycle_scratch_after(self.stream_context, scratch);
        operation_result?;
        recycle_result?;

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: SignalBacking::Owned(destination),
        })
    }
}

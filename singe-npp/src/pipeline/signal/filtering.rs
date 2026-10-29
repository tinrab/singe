use crate::{
    context::StreamContext,
    error::{Error, Result},
    pipeline::{
        SignalAllocator, Workspace,
        signal::{SignalBacking, SignalPipeline},
    },
    signal::{
        filtering,
        view::{SignalOutputView, SignalView},
    },
    workspace::ScratchBuffer,
};

impl SignalPipeline<'_, i32>
where
    Workspace: SignalAllocator<i32>,
{
    /// Returns the required device scratch capacity in bytes for a native `i32` cumulative sum.
    ///
    /// Queries for `source.len() + 1` output elements. Does not read samples, allocate
    /// workspace, or submit stream work.
    ///
    /// # Errors
    ///
    /// Returns an error if adding the extra output element overflows, length or byte-count
    /// conversions fail, or NPP rejects the query.
    pub fn integral_buffer_size(source: &SignalView<'_, i32>) -> Result<usize> {
        filtering::integral_buffer_size(source)
    }

    /// Computes the native `i32` cumulative sum into a signal with one extra element.
    ///
    /// The destination length must equal `source.len() + 1`. Prepares source and output
    /// regions and waits for their producers on other streams. Rejects capture and returns
    /// after successful native execution and stream completion.
    /// Allocates temporary device scratch using the native size query.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits where performed, native execution, or stream synchronization fail.
    /// Adding the extra output element, native length conversions, or workspace-size
    /// queries can fail. The destination length must be exactly one greater than the source.
    /// Temporary scratch allocation can fail.
    pub fn integral_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, i32>,
        destination: &mut SignalOutputView<'_, i32>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        stream_context
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            .sync_scope(|| unsafe { filtering::integral(stream_context, source, destination) })
    }

    /// Computes the native `i32` cumulative sum into a signal with one extra element.
    ///
    /// The destination length must equal `source.len() + 1`. Prepares source and output
    /// regions and waits for their producers on other streams. Rejects capture and returns
    /// after successful native execution and stream completion.
    /// Uses the supplied scratch capacity, which must cover the native size query. Scratch
    /// must belong to the selected CUDA context and be ready for exclusive native use;
    /// this path checks capacity without validating its context or waiting for its producer.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits where performed, native execution, or stream synchronization fail.
    /// Adding the extra output element, native length conversions, or workspace-size
    /// queries can fail. The destination length must be exactly one greater than the source.
    /// Insufficient scratch capacity is rejected.
    pub fn integral_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, i32>,
        destination: &mut SignalOutputView<'_, i32>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            filtering::integral_with_scratch(stream_context, source, destination, scratch)
        })
    }

    /// Computes the native `i32` cumulative sum into a signal with one extra element.
    ///
    /// Obtains a new buffer of length `self.len() + 1` and scratch from the workspace.
    /// Rejects capture, executes the native cumulative sum, waits for stream completion,
    /// and returns scratch to the workspace before exposing initialized output. The source
    /// must already be ready for the selected stream: this path validates its CUDA context
    /// but does not insert producer waits.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits where performed, native execution, or stream synchronization fail.
    /// Adding the extra output element, native length conversions, or workspace-size
    /// queries can fail.
    /// Workspace output allocation, scratch allocation or recycling can also fail.
    pub fn integral(self) -> Result<Self> {
        self.ensure_eager()?;
        let destination_len = self
            .len()
            .checked_add(1)
            .ok_or_else(|| Error::out_of_range("destination length"))?;
        let mut destination = self
            .workspace
            .signal_uninit::<i32>(self.stream_context.cuda_context(), destination_len)?;
        let required_bytes = {
            let source = self.view();
            filtering::integral_buffer_size(&source)?
        };
        let mut scratch = self
            .workspace
            .scratch(self.stream_context.cuda_context(), required_bytes)?;

        let operation_result = {
            let source = self.view();
            let mut destination_view = destination.output();
            self.stream_context.sync_scope(|| {
                // SAFETY: this helper retains its source, destination, and native resources through completion.
                unsafe {
                    filtering::integral_with_scratch(
                        self.stream_context,
                        &source,
                        &mut destination_view,
                        &mut scratch,
                    )
                }
            })
        };

        let recycle_result = self
            .workspace
            .recycle_scratch_after(self.stream_context, scratch);
        operation_result?;
        recycle_result?;

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: SignalBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}

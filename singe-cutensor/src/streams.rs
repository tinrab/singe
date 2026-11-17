use singe_cuda::{
    stream::{AsStream, StreamBinding},
    view::DeviceRegion,
};

use crate::error::Result;

// Waits for every stream in `streams` before returning or unwinding.
struct StreamsDrain<'a> {
    streams: &'a [StreamBinding],
    drained: bool,
}

impl Drop for StreamsDrain<'_> {
    fn drop(&mut self) {
        // Returning or unwinding would release borrowed storage that queued work may still access.
        if !self.drained {
            for stream in self.streams {
                stream.synchronize_or_abort();
            }
        }
    }
}

// Runs `f`, then waits for every stream, even when `f` fails, like `StreamBinding::sync_scope` over several streams.
pub(crate) fn sync_streams<R>(
    streams: &[StreamBinding],
    f: impl FnOnce() -> Result<R>,
) -> Result<R> {
    for stream in streams {
        stream.ensure_not_capturing()?;
    }

    let mut drain = StreamsDrain {
        streams,
        drained: false,
    };
    let result = f();
    let mut completion = Ok(());
    for stream in streams {
        let synchronized = stream.synchronize();
        if completion.is_ok() {
            completion = synchronized;
        }
    }

    drain.drained = completion.is_ok();
    drop(drain);
    match (result, completion) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error.into()),
    }
}

// Orders each stream after the producers of the regions in its own context.
pub(crate) fn join_producers(
    streams: &[StreamBinding],
    regions: &[DeviceRegion<'_>],
) -> Result<()> {
    for stream in streams {
        let local = regions
            .iter()
            .copied()
            .filter(|region| region.context() == stream.context())
            .collect::<Vec<_>>();
        stream.join_regions(&local)?;
    }
    Ok(())
}

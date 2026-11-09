mod cache;
mod clear;
mod stream;

use std::sync::Arc;

use singe_cuda::{
    context::Context,
    event::{Event, EventFlags, EventRecordFlags},
    stream::{Stream, StreamFlags},
};

use crate::{
    backend::cuda::allocator::{CudaAllocator, CudaCachingAllocator},
    data_type::DataType,
    device::Device,
    error::Error,
    storage::cuda::cuda_context_for_device,
    testing::{maybe_lock_device, skip_if_stub},
};

fn caching_allocator() -> Option<CudaCachingAllocator> {
    let context = skip_if_stub(cuda_context_for_device(0))?;
    skip_if_stub(CudaCachingAllocator::create(&context, None))
}

fn caching_allocator_with_stream(stream: Arc<Stream>) -> Option<CudaCachingAllocator> {
    let context = Arc::clone(stream.context());
    skip_if_stub(CudaCachingAllocator::create(&context, Some(stream)))
}

fn default_stream(context: &Arc<Context>) -> Option<Stream> {
    skip_if_stub(
        context
            .create_stream_with_flags(StreamFlags::DEFAULT)
            .map_err(Error::from),
    )
}

fn record_disabled_timing_event(context: &Arc<Context>, stream: &Arc<Stream>) -> Event {
    let event = context
        .create_event_with_flags(EventFlags::DISABLE_TIMING)
        .unwrap();
    event.record(stream, EventRecordFlags::DEFAULT).unwrap();
    event
}

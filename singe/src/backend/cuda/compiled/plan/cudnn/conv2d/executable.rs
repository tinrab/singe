use std::{collections::HashMap, sync::Arc};

use singe_cuda::graph::Graph as NativeCudaGraph;

use crate::{
    backend::cuda::{
        CudaRuntime, cudnn_backend::CudnnBackendExecutable, graph_value::CudaGraphValue,
    },
    error::Result,
    graph::ir::ValueId,
    utility::Locked,
};

pub fn populate_cudnn_conv2d_graph(
    runtime: &Arc<CudaRuntime>,
    mut executable: CudnnBackendExecutable,
    output: &CudaGraphValue,
    cache: &HashMap<ValueId, CudaGraphValue>,
) -> Result<(NativeCudaGraph, CudnnBackendExecutable)> {
    let mut child_graph = runtime.context().create_graph()?;
    let variant_pack = executable.variant_pack(output, cache)?;
    let mut cudnn = runtime.cudnn().locked("cuda runtime cache")?;
    // SAFETY: compiler graph values retain tensor allocations, and the returned executable retains its workspace and plan through graph replay resources.
    unsafe { executable.populate_graph(&mut cudnn, &variant_pack, &mut child_graph) }?;
    drop(cudnn);
    Ok((child_graph, executable))
}

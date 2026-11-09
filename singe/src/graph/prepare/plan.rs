#[cfg(feature = "cuda")]
use std::hash::{Hash, Hasher};

use crate::{
    backend::registry::BackendExecution,
    device::Device,
    error::{Error, Result},
    graph::{
        CompiledGraph, GraphFingerprintOptions, GraphSpecializationKey, NumericalScope,
        ProgramFingerprint,
        prepare::{
            GraphExecutionMode, GraphExecutionPlan, GraphExecutor, GraphPrepareOptions,
            PreparedBackend, PreparedPlan,
        },
    },
    state::StateSnapshot,
};
#[cfg(feature = "cuda")]
use crate::{
    backend::{
        cuda::{self, CudaStream},
        registry::Operation,
    },
    graph::{Determinism, fingerprint::StableHasher, prepare::SharedBuffers},
    tensor::Tensor,
};

impl GraphExecutionPlan {
    /// Creates an exclusive instance with independent workspace and replay buffers.
    ///
    /// # Errors
    ///
    /// Returns an error if the execution mode or operation support is unavailable, the numerical policy cannot be satisfied, fingerprinting fails, or creating CUDA streams or compiled replay resources fails.
    pub fn create_instance(&self) -> Result<GraphExecutor> {
        #[allow(unused_mut, reason = "only cuda has buffers to stop sharing")]
        let mut options = self.options.clone();
        #[cfg(feature = "cuda")]
        {
            options.cuda.shared_arena = None;
        }
        self.create_instance_with(options)
    }

    // Each worker reads its own tensors, such as its own KV caches, in place; the plan's bindings are replaced, not extended.
    #[cfg(feature = "cuda")]
    pub fn create_instance_with_inputs_in_place(
        &self,
        inputs: impl IntoIterator<Item = (String, Tensor)>,
    ) -> Result<GraphExecutor> {
        let mut options = self.options.clone();
        options.cuda = options.cuda.with_only_inputs_in_place(inputs);
        self.create_instance_with(options)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn create_instance_sharing(
        &self,
        inputs: impl IntoIterator<Item = (String, Tensor)>,
        buffers: &SharedBuffers,
    ) -> Result<GraphExecutor> {
        let mut options = self.options.clone();
        options.cuda = options.cuda.with_only_inputs_in_place(inputs);
        self.create_instance_with(options.sharing_buffers(buffers))
    }

    fn create_instance_with(&self, options: GraphPrepareOptions) -> Result<GraphExecutor> {
        let mut instance = GraphExecutor::create(
            self.compiled.clone(),
            options,
            self.program.clone(),
            self.state.clone(),
            self.template.clone(),
        )?;
        instance.active_key = self.active_key.clone();
        instance.cache_stats.analysis_builds = 0;
        Ok(instance)
    }
}

impl PreparedPlan {
    pub(crate) fn create(
        compiled: CompiledGraph,
        options: &GraphPrepareOptions,
        program: ProgramFingerprint,
        state: &StateSnapshot,
    ) -> Result<Self> {
        let _numerics = NumericalScope::enter(options.numerics);
        #[cfg(feature = "cuda")]
        if options.device.is_cuda()
            && options.numerics.determinism == Determinism::Require
            && compiled
                .analysis
                .operation_requests
                .values()
                .any(|request| request.operation() == Operation::Contraction)
        {
            return Err(Error::unsupported_operation(
                "cuTENSOR contraction determinism is not established",
            ));
        }

        if !options.device.is_meta()
            && (options.device == Device::Cpu || options.mode == GraphExecutionMode::Eager)
        {
            let execution = if options.device == Device::Cpu {
                BackendExecution::CpuEager
            } else {
                BackendExecution::CudaEager
            };
            for &id in &compiled.analysis.operation_order {
                if let Some(request) = compiled.analysis.operation_requests.get(&id) {
                    let operation = compiled.ir().required_operation(id)?;
                    request
                        .validate_execution(execution)
                        .map_err(|error| error.with_graph_source(operation.source_location()))?;
                }
            }
        }

        let backend = match (options.device, options.mode) {
            // Meta graphs run in the interpreter, which propagates metadata without kernels.
            (Device::Cpu | Device::Meta, _) => PreparedBackend::Cpu,
            #[cfg(feature = "cuda")]
            (Device::Cuda(_), GraphExecutionMode::Eager) => PreparedBackend::CudaEager {
                stream: CudaStream::create(options.device)?,
            },
            (Device::Cuda(_), _) => {
                #[cfg(feature = "cuda")]
                {
                    let graph = cuda::compile(&compiled, &options.cuda)?;
                    let stream = CudaStream::create(options.device)?;
                    PreparedBackend::Cuda {
                        graph: Box::new(graph),
                        stream,
                    }
                }
                #[cfg(not(feature = "cuda"))]
                return Err(Error::unsupported_operation_with_details(
                    "graph preparation",
                    "cuda support is not enabled",
                ));
            }
        };
        let (native_context, provider_digest) = match &backend {
            PreparedBackend::Cpu => (None, None),
            #[cfg(feature = "cuda")]
            PreparedBackend::CudaEager { stream } => {
                (Some(stream.raw().context().as_raw() as usize), None)
            }
            #[cfg(feature = "cuda")]
            PreparedBackend::Cuda { graph, .. } => {
                let mut hasher = StableHasher::new();
                graph.cache_key().hash(&mut hasher);
                (Some(graph.native_context_identity()), Some(hasher.finish()))
            }
        };
        let specialization = GraphSpecializationKey {
            program,
            state: state.revision(),
            device: options.device,
            execution_mode: options.mode,
            numerics: options.numerics,
            freeze: options.freeze,
            bound_graph: compiled.fingerprint_with_options(
                &GraphFingerprintOptions::with_compile_options(options.compile),
            )?,
            native_context,
            provider_digest,
        };
        Ok(Self {
            compiled,
            backend,
            specialization,
        })
    }
}

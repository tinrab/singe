mod kernels;
mod libraries;
mod matmul;
mod names;

use std::collections::{HashMap, HashSet};

use singe_cuda::architecture::GpuArchitecture;

use crate::{
    backend::{
        cuda::{
            codegen::KernelRecorder,
            compiled::{
                build::{
                    lower::placement::place_by_shared_rules,
                    operation::{contraction_operation, convolution_operation, copy_to_input},
                },
                plan::{
                    fusion::{FusionPlan, plan_fusion},
                    kernel::KernelFeasibility,
                    lowering::{CudaGraphProvider, LoweringPlan, LoweringPolicy},
                    memory::GraphMemoryPlanner,
                },
            },
            export::{
                builder::{
                    matmul::{Epilogue, MatmulLibraries, matmul_libraries},
                    names::identifier,
                },
                program::{
                    BufferId, BufferKind, BufferView, CudaProgram, Library, LibraryCall,
                    OperationProvider, ProgramBuffer, ProgramState, ProgramStep, ProgramValue,
                    StateKind, StepProvider,
                },
            },
            graph_binding::{InPlaceWrites, WritableInputs, graph_users},
            kernel_plan::{KernelInput, KernelPlan, KernelPlanner},
            tuning::{RecordCoverage, TunedChoice, TunedWay, TuningRecords, schedule_key},
        },
        registry::Operation,
    },
    data_type::DataType,
    device::Device,
    error::{Error, ExportFailure, ExportFailureReason, Result},
    graph::{
        NumericalPolicy,
        ir::{Graph, OperationNode, ValueBindingKind, ValueId},
    },
    layout::Layout,
    scalar::Scalar,
    shape::Shape,
    tensor::Tensor,
    utility::to_u64,
};

pub fn span_bytes(data_type: DataType, layout: &Layout) -> Result<u64> {
    let bytes = layout
        .storage_span_element_count()?
        .checked_mul(data_type.size_bytes())
        .ok_or_else(|| Error::out_of_range("program buffer bytes"))?;
    to_u64(bytes, "program buffer bytes")
}

// The bytes of a constant over `layout`'s storage span, with its gaps zeroed, since kernels read it with those strides.
pub fn constant_bytes(tensor: &Tensor, layout: &Layout) -> Result<Vec<u8>> {
    let logical = tensor.contiguous()?.to_bytes()?;
    if layout.is_contiguous() && layout.storage_offset() == 0 {
        return Ok(logical);
    }

    let size = tensor.data_type().size_bytes();
    let span = layout.storage_span_element_count()?;
    let mut bytes = vec![0; span * size];
    let dimensions = layout.dimensions();
    let strides = layout.strides();
    let mut index = vec![0_i64; dimensions.len()];
    for element in logical.chunks_exact(size) {
        let offset = index
            .iter()
            .zip(strides)
            .map(|(&coordinate, &stride)| coordinate * stride)
            .sum::<i64>();
        let start = (usize::try_from(offset).map_err(|_| Error::DimensionOutOfBounds)?
            + layout.storage_offset())
            * size;
        bytes[start..start + size].copy_from_slice(element);
        for axis in (0..index.len()).rev() {
            index[axis] += 1;
            if index[axis] < dimensions[axis] {
                break;
            }

            index[axis] = 0;
        }
    }
    Ok(bytes)
}

struct Builder<'a> {
    ir: &'a Graph,
    policy: &'a LoweringPolicy,
    lowering: LoweringPlan,
    fusion: FusionPlan,
    planner: &'a KernelPlanner,
    memory: GraphMemoryPlanner,
    users: HashMap<ValueId, Vec<ValueId>>,
    outputs: HashSet<ValueId>,
    in_place_writes: InPlaceWrites,
    positions: HashMap<ValueId, usize>,
    buffers: Vec<ProgramBuffer>,
    free: Vec<BufferId>,
    views: HashMap<ValueId, BufferView>,
    siblings: HashMap<ValueId, (BufferView, StepProvider)>,
    placed: HashSet<ValueId>,
    steps: Vec<ProgramStep>,
    status: Option<BufferId>,
    providers: Vec<OperationProvider>,
    failures: Vec<ExportFailure>,
    inputs: Vec<ProgramValue>,
    merged: HashSet<ValueId>,
    library_matmuls: HashMap<ValueId, MatmulLibraries>,
    state: Vec<ProgramState>,
    names: HashSet<String>,
}

impl Builder<'_> {
    // Whether `provider` may run `operation` at the node producing `id`, where a record may have narrowed what is allowed.
    fn allowed(&self, id: ValueId, provider: CudaGraphProvider, operation: Operation) -> bool {
        self.policy.allows(provider, operation)
            && match self.lowering.way(id) {
                Some(TunedWay::Library(library)) => library == provider,
                Some(TunedWay::Generated) => false,
                _ => true,
            }
    }

    fn name(&self, id: ValueId) -> String {
        self.ir
            .required_operation_for_value(id)
            .ok()
            .and_then(OperationNode::opcode)
            .map_or_else(
                || format!("%{}", id.index()),
                |operation| format!("{} %{}", operation.name(), id.index()),
            )
    }

    fn record(&mut self, id: ValueId, provider: StepProvider) {
        let operation = self
            .ir
            .required_operation_for_value(id)
            .ok()
            .and_then(OperationNode::opcode);
        self.providers.push(OperationProvider {
            operation,
            value: id.index(),
            provider,
        });
    }

    fn add_buffer(&mut self, kind: BufferKind, bytes: u64) -> BufferId {
        self.buffers.push(ProgramBuffer { kind, bytes });
        BufferId(self.buffers.len() - 1)
    }

    fn view_of(&self, id: ValueId) -> Result<BufferView> {
        self.views
            .get(&id)
            .cloned()
            .ok_or_else(|| Error::missing_input(format!("program view of {}", self.name(id))))
    }

    // The smallest released buffer that holds `bytes`, or a new one.
    fn internal(&mut self, bytes: u64) -> BufferId {
        let best = self
            .free
            .iter()
            .enumerate()
            .filter(|(_, buffer)| self.buffers[buffer.0].bytes >= bytes)
            .min_by_key(|(_, buffer)| self.buffers[buffer.0].bytes)
            .map(|(position, _)| position);
        match best {
            Some(position) => self.free.swap_remove(position),
            None => self.add_buffer(BufferKind::Internal, bytes),
        }
    }

    fn allocate(&mut self, id: ValueId) -> Result<BufferView> {
        let data_type = self.ir.required_value_data_type(id)?;
        let layout = self.ir.required_value_layout(id)?.clone();
        let buffer = self.internal(span_bytes(data_type, &layout)?);
        Ok(BufferView {
            buffer,
            data_type,
            layout,
        })
    }

    fn release(&mut self, id: ValueId) -> Result<()> {
        let index = *self
            .positions
            .get(&id)
            .ok_or_else(|| Error::missing_node("program release"))?;
        for value in self.memory.released_by(id, index) {
            if let Some(view) = self.views.get(&value)
                && matches!(self.buffers[view.buffer.0].kind, BufferKind::Internal)
                && !self.free.contains(&view.buffer)
            {
                self.free.push(view.buffer);
            }
        }
        Ok(())
    }

    fn scratch(&mut self, plan: &KernelPlan) -> Vec<BufferId> {
        plan.scratch()
            .into_iter()
            .map(|bytes| self.add_buffer(BufferKind::Scratch, bytes))
            .collect()
    }

    fn status_buffer(&mut self) -> BufferId {
        if let Some(status) = self.status {
            return status;
        }

        let status = self.add_buffer(BufferKind::Status, 4);
        self.status = Some(status);
        status
    }

    // Adds a step for each of `plan`'s kernels, which share its scratch buffers.
    fn add_plan(&mut self, plan: &KernelPlan, inputs: &[BufferView], outputs: &[BufferView]) {
        let scratch = self.scratch(plan);
        for kernel in plan.kernels().filter(|kernel| !kernel.is_empty()) {
            let status = kernel.abi.checks().then(|| self.status_buffer());
            self.steps.push(ProgramStep::Kernel {
                artifact: Box::new(kernel.clone()),
                inputs: inputs.to_vec(),
                outputs: outputs.to_vec(),
                scratch: scratch.clone(),
                status,
            });
        }
    }

    // Holds a tensor of the graph as state, so that the program's functions read and replace it.
    fn add_state(
        &mut self,
        id: ValueId,
        graph_name: String,
        kind: StateKind,
        values: Vec<u8>,
    ) -> Result<()> {
        let data_type = self.ir.required_value_data_type(id)?;
        let layout = self.ir.required_value_layout(id)?.clone();
        let bytes = span_bytes(data_type, &layout)?;
        let name = identifier(&graph_name, &mut self.names);
        let buffer = self.add_buffer(BufferKind::State(self.state.len()), bytes);
        self.state.push(ProgramState {
            name,
            graph_name,
            kind,
            data_type,
            layout: layout.clone(),
            values,
        });
        self.views.insert(
            id,
            BufferView {
                buffer,
                data_type,
                layout,
            },
        );
        Ok(())
    }

    fn terminal(&mut self, id: ValueId) -> Result<bool> {
        let kind = self.ir.required_value_binding_kind(id)?;
        let data_type = self.ir.required_value_data_type(id)?;
        let layout = self.ir.required_value_layout(id)?.clone();
        let value = self.ir.required_value(id)?;
        let name = value
            .binding()
            .input_name()
            .or_else(|| value.name())
            .map_or_else(|| format!("constant_{}", id.index()), ToOwned::to_owned);
        match kind {
            ValueBindingKind::OperationOutput => return Ok(false),
            ValueBindingKind::Constant => {
                let values = constant_bytes(self.ir.required_constant_tensor(id)?, &layout)?;
                self.add_state(id, name, StateKind::Constant, values)?;
            }
            ValueBindingKind::InitializerInput | ValueBindingKind::StaticInput => {
                let values = constant_bytes(self.ir.required_initializer_tensor(id)?, &layout)?;
                self.add_state(id, name, StateKind::Initializer, values)?;
            }
            ValueBindingKind::RuntimeInput => {
                if let Some(generator) = self
                    .ir
                    .generators()
                    .iter()
                    .find(|generator| generator.name == name)
                {
                    let state = generator.generator.state()?;
                    let values = [state.seed, state.offset]
                        .iter()
                        .flat_map(|word| word.to_ne_bytes())
                        .collect();
                    let blocks = generator.blocks;
                    self.add_state(id, name, StateKind::Generator { blocks }, values)?;
                    return Ok(true);
                }

                let bytes = span_bytes(data_type, &layout)?;
                let index = self.inputs.len();
                self.inputs.push(ProgramValue {
                    name: identifier(&name, &mut self.names),
                    graph_name: name,
                    data_type,
                    layout: layout.clone(),
                    written: false,
                    count: None,
                });
                let buffer = self.add_buffer(BufferKind::Input(index), bytes);
                self.views.insert(
                    id,
                    BufferView {
                        buffer,
                        data_type,
                        layout,
                    },
                );
            }
        }
        Ok(true)
    }

    fn filled(fill: Scalar, data_type: DataType, shape: &Shape) -> Result<Tensor> {
        fill.full(Device::Cpu, data_type, shape)
    }

    fn place(&mut self, id: ValueId) -> Result<()> {
        let operation = self.ir.required_operation_for_value(id)?;
        let Some(opcode) = operation.opcode() else {
            return Err(Error::missing_node("program operation"));
        };
        let inputs = operation.inputs().to_vec();
        // An allowed library takes what it can run before any generated kernel does.
        let library_first = opcode == Operation::MatMul
            || (matches!(
                opcode,
                Operation::Conv2d | Operation::Conv3d | Operation::ScaledDotProductAttention
            ) && self.allowed(id, CudaGraphProvider::CudnnBackend, opcode));
        if !library_first && self.kernels(id)? {
            return Ok(());
        }

        let placed = match opcode {
            Operation::MatMul => match inputs.as_slice() {
                &[lhs, rhs] => self.matmul(id, lhs, rhs)?,
                _ => false,
            },
            Operation::Conv2d | Operation::Conv3d
                if self.allowed(id, CudaGraphProvider::CudnnBackend, opcode) =>
            {
                let convolution = convolution_operation(self.ir, id)?
                    .ok_or_else(|| Error::missing_node("program convolution"))?;
                // cuDNN's engines fuse an activation only after a bias, and only one that reads the convolution itself, since an activation after an add belongs to the sum.
                let epilogue = match self.epilogue(id, false)? {
                    found if convolution.bias.is_some() && found.add.is_none() => found,
                    _ => Epilogue::default(),
                };
                let output = self.library_output(epilogue.result(id))?;
                let call = LibraryCall::Convolution {
                    input: self.view_of(convolution.input)?,
                    weight: self.view_of(convolution.weight)?,
                    bias: convolution
                        .bias
                        .map(|bias| self.view_of(bias))
                        .transpose()?,
                    activation: epilogue.activation.map(|(_, activation)| activation),
                    output,
                    stride: convolution.window.stride.clone(),
                    padding: convolution.window.padding.clone(),
                    dilation: convolution.window.dilation,
                };
                self.push_library(id, call);
                self.finish_merged(id, &epilogue, StepProvider::Library(Library::Cudnn))?;
                true
            }
            Operation::ScaledDotProductAttention
                if self.allowed(id, CudaGraphProvider::CudnnBackend, opcode) =>
            {
                self.cudnn_attention(id, &inputs)?
            }
            Operation::Contraction if self.allowed(id, CudaGraphProvider::Cutensor, opcode) => {
                let contraction = contraction_operation(self.ir, id)?
                    .ok_or_else(|| Error::missing_node("program contraction"))?;
                let call = LibraryCall::Contraction {
                    a: (self.view_of(contraction.a)?, contraction.a_mode),
                    b: (self.view_of(contraction.b)?, contraction.b_mode),
                    c: (self.view_of(contraction.c)?, contraction.c_mode),
                    output: (self.library_output(id)?, contraction.d_mode),
                    alpha: contraction.alpha,
                    beta: contraction.beta,
                };
                self.finish_library(id, call)?;
                true
            }
            Operation::ScaledMatMul if self.allowed(id, CudaGraphProvider::CublasLt, opcode) => {
                self.scaled_matmul(id, &inputs)?
            }
            Operation::Rfft | Operation::Irfft
                if self.allowed(id, CudaGraphProvider::Cufft, opcode) =>
            {
                self.fft(id, opcode, inputs[0])?
            }
            _ => false,
        };
        if !placed {
            // Later operations read a placeholder, so one export reports every operation it cannot place.
            let view = self.allocate(id)?;
            self.views.insert(id, view);
            self.failures.push(ExportFailure {
                operation: self.name(id),
                reason: ExportFailureReason::NoProvider,
            });
        }
        Ok(())
    }

    fn visit(&mut self, id: ValueId) -> Result<()> {
        // A library call computed this already, and what it alone read frees now that the walk has passed it.
        if self.merged.remove(&id) {
            return self.release(id);
        }

        if self.placed.contains(&id) || self.terminal(id)? {
            return Ok(());
        }

        if place_by_shared_rules(self, id)? {
            return Ok(());
        }

        // A program runs on one device, so a transfer is a copy.
        if let Some(source) = copy_to_input(self.ir, id)? {
            let source = self.view_of(source)?;
            let view = self.allocate(id)?;
            self.views.insert(id, view.clone());
            self.copy(source, view)?;
            self.record(id, StepProvider::Kernel);
            self.release(id)?;
            return Ok(());
        }

        // A result with no elements needs no work.
        if self.ir.required_value_shape(id)?.element_count() == 0 {
            let view = self.allocate(id)?;
            self.views.insert(id, view);
            self.release(id)?;
            return Ok(());
        }
        self.place(id)
    }

    // Copies `source` into `destination`: bytes when both are contiguous, and through a generated kernel otherwise.
    fn copy(&mut self, source: BufferView, destination: BufferView) -> Result<()> {
        if source.layout.element_count() == 0 || source.buffer == destination.buffer {
            return Ok(());
        }

        if source.layout.is_contiguous() && destination.layout.is_contiguous() {
            self.steps.push(ProgramStep::Copy {
                source,
                destination,
            });
            return Ok(());
        }

        let plan = self.planner.assign(
            &KernelInput::new(destination.data_type, destination.layout.clone()),
            &KernelInput::new(source.data_type, source.layout.clone()),
        )?;
        self.add_plan(&plan, &[source], &[destination]);
        Ok(())
    }

    // Copies `value` into a buffer the caller passes.
    fn add_output(
        &mut self,
        name: &str,
        value: ValueId,
        count: Option<usize>,
        names: &mut HashSet<String>,
        outputs: &mut Vec<ProgramValue>,
    ) -> Result<()> {
        let data_type = self.ir.required_value_data_type(value)?;
        let layout = Layout::from_shape(self.ir.required_value_shape(value)?);
        let bytes = span_bytes(data_type, &layout)?;
        let buffer = self.add_buffer(BufferKind::Output(outputs.len()), bytes);
        outputs.push(ProgramValue {
            name: identifier(name, names),
            graph_name: name.to_owned(),
            data_type,
            layout: layout.clone(),
            written: false,
            count,
        });
        let source = self.view_of(value)?;
        self.copy(
            source,
            BufferView {
                buffer,
                data_type,
                layout,
            },
        )
    }

    // Binds what the program returns and writes: outputs copy into the caller's buffers, and each written input's final value copies into that input.
    fn finish(&mut self) -> Result<Vec<ProgramValue>> {
        let ir = self.ir;
        let outputs = ir.output_values_vec();
        let writes = ir.input_writes();
        let counted = ir.counted_outputs();
        // A graph's outputs are its results, then each written input's final value, then the count of each counted result.
        let visible = outputs
            .len()
            .checked_sub(writes.len() + counted.len())
            .ok_or_else(|| Error::missing_output("program outputs"))?;
        let mut program_outputs = Vec::with_capacity(visible + counted.len());
        let mut names = HashSet::<String>::new();
        let results = outputs[..visible]
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                let name = ir
                    .output_name(index)
                    .map_or_else(|| format!("output{index}"), ToOwned::to_owned);
                let count = counted
                    .iter()
                    .position(|&position| position == index)
                    .map(|position| visible + position);
                (name, value, count)
            });
        for (name, value, count) in results {
            self.add_output(&name, value, count, &mut names, &mut program_outputs)?;
        }

        for (name, &value) in writes.iter().zip(&outputs[visible..]) {
            let target = ir
                .input_values()
                .find_map(|(id, input)| (input == name).then_some(id))
                .ok_or_else(|| Error::missing_input("program written input"))?;
            let destination = self.view_of(target)?;
            if let BufferKind::Input(index) = self.buffers[destination.buffer.0].kind {
                self.inputs[index].written = true;
            }

            let source = self.view_of(value)?;
            self.copy(source, destination)?;
        }

        for (&position, &value) in counted.iter().zip(&outputs[visible + writes.len()..]) {
            let name = format!("{}_count", program_outputs[position].name);
            self.add_output(&name, value, None, &mut names, &mut program_outputs)?;
        }
        Ok(program_outputs)
    }
}

pub fn build_program(
    ir: &Graph,
    order: &[ValueId],
    policy: &LoweringPolicy,
    planner: &KernelPlanner,
    recorder: &KernelRecorder,
    target: GpuArchitecture,
) -> Result<CudaProgram> {
    let builder_records = planner.context().records.clone();
    let records_hold = planner.context().records_hold();
    let lowering = LoweringPlan::create(
        ir,
        order,
        policy,
        &KernelFeasibility::new(planner.context()),
    )?;
    let mut library_matmuls = HashMap::new();
    for operation in ir.operations() {
        if operation.opcode() == Some(Operation::MatMul)
            && let (&[lhs, rhs], Some(value)) =
                (operation.inputs(), operation.result_values().next())
        {
            let libraries = matmul_libraries(ir, policy, lowering.way(value), lhs, rhs)?;
            if libraries.any() {
                library_matmuls.insert(value, libraries);
            }
        }
    }

    let library_values = library_matmuls.keys().copied().collect::<HashSet<_>>();
    let users = graph_users(ir, order)?;
    let outputs = ir.output_value_set();
    let fusion = plan_fusion(
        ir,
        order,
        &users,
        &outputs,
        &lowering,
        Some(&library_values),
        planner.context().records(),
    )?;
    let order = fusion.order().to_vec();
    let positions = order
        .iter()
        .enumerate()
        .map(|(index, &id)| (id, index))
        .collect();
    let memory = GraphMemoryPlanner::create(ir, &order)?;
    let in_place_writes =
        InPlaceWrites::analyze(ir, &order, &users, &outputs, WritableInputs::Every)?;
    let mut builder = Builder {
        ir,
        policy,
        lowering,
        fusion,
        planner,
        memory,
        users,
        outputs,
        in_place_writes,
        positions,
        buffers: Vec::new(),
        free: Vec::new(),
        views: HashMap::new(),
        siblings: HashMap::new(),
        placed: HashSet::new(),
        steps: Vec::new(),
        status: None,
        providers: Vec::new(),
        failures: Vec::new(),
        merged: HashSet::new(),
        library_matmuls,
        inputs: Vec::new(),
        state: Vec::new(),
        names: HashSet::new(),
    };
    for &id in &order {
        builder.visit(id)?;
    }

    if !builder.failures.is_empty() {
        return Err(Error::Export {
            failures: builder.failures,
        });
    }

    let program_outputs = builder.finish()?;
    // Requirements a target cannot meet surface here rather than when the program compiles.
    for step in &builder.steps {
        if let ProgramStep::Kernel { artifact, .. } = step
            && let Some(minimum) = artifact.requirements.minimum_capability
            && target.compute_capability() < minimum
        {
            builder.failures.push(ExportFailure {
                operation: artifact.name.clone(),
                reason: ExportFailureReason::ComputeCapability {
                    minimum,
                    target: target.to_string(),
                },
            });
        }
    }

    if !builder.failures.is_empty() {
        return Err(Error::Export {
            failures: builder.failures,
        });
    }

    let tuned_kernels = recorder
        .take()?
        .into_iter()
        .filter_map(|planned| {
            let records = builder_records.as_deref().filter(|_| records_hold)?;
            match records.get(&schedule_key(&planned.kernel, &planned.bindings))? {
                TunedChoice::Schedule(tuned) if tuned != planned.schedule => {
                    Some(format!("{}: {tuned:?}", planned.kernel.name))
                }
                _ => None,
            }
        })
        .collect();
    let record_coverage = builder_records
        .as_deref()
        .map_or_else(RecordCoverage::default, TuningRecords::coverage);
    Ok(CudaProgram {
        target,
        buffers: builder.buffers,
        inputs: builder.inputs,
        state: builder.state,
        outputs: program_outputs,
        steps: builder.steps,
        providers: builder.providers,
        decomposed: Vec::new(),
        tuned_kernels,
        record_coverage,
        state_functions: false,
        dependent_launch: false,
        library_table: false,
        numerics: NumericalPolicy::current(),
        decoding: None,
    })
}

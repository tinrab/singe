use std::slice::from_ref;

use crate::{
    backend::{
        cuda::{
            compiled::{
                build::{
                    lower::{
                        placement::NodePlacement,
                        results::{attention_sibling, plan_kernel_result, result_of},
                    },
                    operation::{contiguous_input, copy_to_input},
                },
                plan::{
                    group::{kernel_input, plan_group_kernel},
                    kernel::{NodeKernelPlans, node_kernel_plans},
                    lowering::CudaGraphProvider,
                },
            },
            export::{
                builder::{Builder, constant_bytes, matmul::Epilogue, span_bytes},
                program::{BufferKind, BufferView, Library, StateKind, StepProvider},
            },
            kernel_plan::{KernelInput, cudnn_paging},
        },
        registry::Operation,
        spec,
    },
    error::{Error, Result},
    graph::ir::ValueId,
    layout::Layout,
};

impl NodePlacement for Builder<'_> {
    fn fusion_group(&mut self, id: ValueId) -> Result<bool> {
        self.group(id)
    }

    fn kernel_result(&mut self, id: ValueId) -> Result<bool> {
        self.kernel_result_of(id)
    }

    fn alias(&mut self, id: ValueId) -> Result<bool> {
        self.alias_of(id)
    }

    fn empty_reduction(&mut self, id: ValueId) -> Result<bool> {
        self.empty_reduction_of(id)
    }
}

impl Builder<'_> {
    pub fn group(&mut self, id: ValueId) -> Result<bool> {
        let Some(group) = self.fusion.led_by(id).cloned() else {
            return Ok(false);
        };
        if group
            .members
            .iter()
            .any(|member| self.placed.contains(member))
        {
            return Ok(false);
        }

        let Some(kernel) = plan_group_kernel(self.planner, self.ir, &group)? else {
            return Ok(false);
        };
        let inputs = kernel
            .inputs
            .iter()
            .map(|&value| self.view_of(value))
            .collect::<Result<Vec<_>>>()?;
        let mut outputs = Vec::with_capacity(kernel.outputs.len());
        for &value in &kernel.outputs {
            // A module's cache write writes its base in place when nothing else reads it, and otherwise a copy made first.
            let operation = self.ir.required_operation_for_value(value)?;
            let scatter =
                (operation.opcode() == Some(Operation::IndexPut)).then(|| operation.inputs()[0]);
            let base = self.in_place_writes.value(value);
            let view = match base {
                Some(base) => self.view_of(base)?,
                None => self.allocate(value)?,
            };
            if let (Some(source), None) = (scatter, base) {
                let source = self.view_of(source)?;
                let assign = self.planner.assign(
                    &KernelInput::new(view.data_type, view.layout.clone()),
                    &KernelInput::new(source.data_type, source.layout.clone()),
                )?;
                self.add_plan(&assign, &[source], from_ref(&view));
            }

            self.views.insert(value, view.clone());
            outputs.push(view);
        }

        self.add_plan(&kernel.plan, &inputs, &outputs);
        for &member in &group.members {
            self.placed.insert(member);
            self.record(member, StepProvider::Fusion);
        }

        for &member in &group.members {
            self.release(member)?;
        }
        Ok(true)
    }

    // Two-result kernels: normalized top-k holds both results, and each of latent attention's is its own node.
    fn kernel_result_of(&mut self, id: ValueId) -> Result<bool> {
        let operation = self.ir.required_operation_for_value(id)?;
        let Some(kept) = operation.opcode().and_then(result_of) else {
            return Ok(false);
        };
        if let Some((view, provider)) = self.siblings.remove(&id) {
            self.views.insert(id, view);
            self.record(id, provider);
            self.release(id)?;
            return Ok(true);
        }

        let metadata = operation
            .inputs()
            .iter()
            .map(|&value| kernel_input(self.ir, value))
            .collect::<Result<Vec<_>>>()?;
        let plan = plan_kernel_result(self.planner, operation, &metadata)?;
        let library = if self.allowed(
            id,
            CudaGraphProvider::CudnnBackend,
            Operation::PagedAttention,
        ) && matches!(
            operation.opcode(),
            Some(Operation::PagedAttentionOutput | Operation::PagedAttentionLse)
        ) {
            let attention = operation
                .attributes()
                .scaled_dot_product_attention()
                .copied()
                .ok_or_else(|| Error::missing_node("kernel results"))?;
            cudnn_paging(&metadata, &attention)?.map(|paging| (paging, attention))
        } else {
            None
        };
        let provider = StepProvider::of(library.is_some().then_some(Library::Cudnn));
        let inputs = operation
            .inputs()
            .iter()
            .map(|&value| self.view_of(value))
            .collect::<Result<Vec<_>>>()?;
        let output = self.allocate(id)?;
        self.views.insert(id, output.clone());
        let is_topk = operation.opcode() == Some(Operation::TopK);
        let sibling = if is_topk {
            operation.result_values().nth(1)
        } else {
            let other = attention_sibling(
                operation
                    .opcode()
                    .ok_or_else(|| Error::missing_node("kernel results"))?,
            );
            let position = self.positions[&id];
            self.users
                .get(&operation.inputs()[0])
                .into_iter()
                .flatten()
                .copied()
                .find(|&user| {
                    self.positions.get(&user).is_some_and(|&at| at > position)
                        && self
                            .ir
                            .required_operation_for_value(user)
                            .is_ok_and(|candidate| {
                                candidate.opcode() == Some(other)
                                    && candidate.inputs() == operation.inputs()
                                    && candidate.attributes() == operation.attributes()
                            })
                })
        };
        let other = if let Some(sibling) = sibling {
            let view = self.allocate(sibling)?;
            if is_topk {
                self.views.insert(sibling, view.clone());
            } else {
                self.siblings.insert(sibling, (view.clone(), provider));
            }
            view
        } else {
            let spec = plan
                .outputs
                .get(1 - kept)
                .ok_or_else(|| Error::missing_output("program kernel result"))?;
            let layout = Layout::from_shape(spec.shape());
            let bytes = span_bytes(spec.data_type(), &layout)?;
            BufferView {
                buffer: self.add_buffer(BufferKind::Internal, bytes),
                data_type: spec.data_type(),
                layout,
            }
        };
        let outputs = if kept == 0 {
            vec![output, other]
        } else {
            vec![other, output]
        };
        if let Some((paging, attention)) = &library {
            self.cudnn_paged_attention(id, (paging, attention), &metadata, &inputs, &outputs)?;
            self.finish_merged(
                id,
                &Epilogue::default(),
                StepProvider::Library(Library::Cudnn),
            )?;
            return Ok(true);
        }

        self.add_plan(&plan, &inputs, &outputs);
        self.record(id, StepProvider::Kernel);
        self.release(id)?;
        Ok(true)
    }

    fn alias_of(&mut self, id: ValueId) -> Result<bool> {
        if let Some(input) = contiguous_input(self.ir, id)? {
            let view = self.view_of(input)?;
            if !(view.layout.is_contiguous() && view.layout.storage_offset() == 0) {
                return Ok(false);
            }
        } else if self.ir.required_value_alias(id)?.is_none()
            || copy_to_input(self.ir, id)?.is_some()
        {
            return Ok(false);
        }

        let Some(alias) = self.ir.required_value_alias(id)? else {
            return Ok(false);
        };
        let source = self.view_of(alias.source())?;
        self.views.insert(
            id,
            BufferView {
                buffer: source.buffer,
                data_type: self.ir.required_value_data_type(id)?,
                layout: self.ir.required_value_layout(id)?.clone(),
            },
        );
        self.record(id, StepProvider::Alias);
        self.release(id)?;
        Ok(true)
    }

    fn empty_reduction_of(&mut self, id: ValueId) -> Result<bool> {
        let operation = self.ir.required_operation_for_value(id)?;
        let (Some(attributes), Some(opcode), [input]) = (
            operation.attributes().reduction(),
            operation.opcode(),
            operation.inputs(),
        ) else {
            return Ok(false);
        };
        let shape = self.ir.required_value_shape(*input)?;
        let problem = spec::reduction_problem(shape, &attributes.dimensions, attributes.keepdim)?;
        let Some(fill) = spec::empty_reduction_fill(shape, &problem, opcode)? else {
            return Ok(false);
        };
        let data_type = self.ir.required_value_data_type(id)?;
        let layout = self.ir.required_value_layout(id)?.clone();
        let tensor = Self::filled(fill, data_type, &problem.output_shape)?;
        let values = constant_bytes(&tensor, &layout)?;
        let name = format!("{}_fill_{}", opcode.name(), id.index());
        self.add_state(id, name, StateKind::Constant, values)?;
        self.release(id)?;
        Ok(true)
    }

    // One node's generated kernels, or `None` when it has none.
    pub fn kernels(&mut self, id: ValueId) -> Result<bool> {
        if !self
            .lowering
            .should_try_provider(id, CudaGraphProvider::Kernel)
        {
            return Ok(false);
        }

        let Some(plans) = node_kernel_plans(self.planner, self.ir, id) else {
            return Ok(false);
        };
        let operation = self.ir.required_operation_for_value(id)?;
        let opcode = operation.opcode();
        match plans {
            NodeKernelPlans::Copies(copies) => {
                let output = self.allocate(id)?;
                self.views.insert(id, output.clone());
                for (plan, input) in &copies {
                    let input = self.view_of(*input)?;
                    self.add_plan(plan, &[input], from_ref(&output));
                }
            }
            NodeKernelPlans::Kernel(plan) => {
                let mut inputs = operation
                    .inputs()
                    .iter()
                    .map(|&value| self.view_of(value))
                    .collect::<Result<Vec<_>>>()?;
                let scatter = matches!(
                    opcode,
                    Some(Operation::IndexPut | Operation::StridedScatter)
                );
                let base = self.in_place_writes.value(id);
                let output = match base {
                    Some(base) => self.view_of(base)?,
                    None => self.allocate(id)?,
                };
                self.views.insert(id, output.clone());
                // A scatter writes into a copy of its first input, unless nothing else reads it.
                if scatter {
                    let source = inputs.remove(0);
                    if base.is_none() {
                        let assign = self.planner.assign(
                            &KernelInput::new(output.data_type, output.layout.clone()),
                            &KernelInput::new(source.data_type, source.layout.clone()),
                        )?;
                        self.add_plan(&assign, &[source], from_ref(&output));
                    }
                }

                self.add_plan(&plan, &inputs, &[output]);
            }
        }

        self.record(id, StepProvider::Kernel);
        self.release(id)?;
        Ok(true)
    }

    // Kernels for `epilogue`, starting from `value`, which `view` holds.
    pub fn kernels_after(
        &mut self,
        value: ValueId,
        view: BufferView,
        epilogue: &Epilogue,
        output: &BufferView,
    ) -> Result<bool> {
        let (mut value, mut view) = (value, view);
        let result = epilogue.result(value);
        for member in epilogue.members() {
            let target = if member == result {
                output.clone()
            } else {
                self.intermediate_view(member)?
            };
            if !self.kernel_into(member, Some((value, &view)), &target)? {
                return Ok(false);
            }

            (value, view) = (member, target);
        }
        Ok(true)
    }

    // One operation's kernel writing `output`, reading `known` where it reads that value.
    pub fn kernel_into(
        &mut self,
        id: ValueId,
        known: Option<(ValueId, &BufferView)>,
        output: &BufferView,
    ) -> Result<bool> {
        let Some(NodeKernelPlans::Kernel(plan)) = node_kernel_plans(self.planner, self.ir, id)
        else {
            return Ok(false);
        };
        let inputs = self
            .ir
            .required_operation_for_value(id)?
            .inputs()
            .iter()
            .map(|&input| match known {
                Some((value, view)) if value == input => Ok(view.clone()),
                _ => self.view_of(input),
            })
            .collect::<Result<Vec<_>>>()?;
        self.add_plan(&plan, &inputs, from_ref(output));
        Ok(true)
    }

    // A buffer of its own for an intermediate value between the steps of one operation.
    pub fn intermediate_view(&mut self, id: ValueId) -> Result<BufferView> {
        let data_type = self.ir.required_value_data_type(id)?;
        let layout = self.ir.required_value_layout(id)?.clone();
        let bytes = span_bytes(data_type, &layout)?;
        let buffer = self.add_buffer(BufferKind::Internal, bytes);
        Ok(BufferView {
            buffer,
            data_type,
            layout,
        })
    }
}

use std::slice::from_ref;

use crate::{
    backend::{
        cuda::{
            cudnn_backend::{
                CudnnPaging, cudnn_attention_scale, cudnn_reads_key_lengths, cudnn_takes_attention,
            },
            export::{
                builder::{Builder, matmul::Epilogue, span_bytes},
                program::{
                    BufferKind, BufferView, CudnnAttentionCall, CudnnLengths, CudnnPagedCall,
                    Library, LibraryCall, ProgramStep, StepProvider,
                },
            },
            kernel_plan::{
                KernelInput, plan_cudnn_paging_indices, plan_cudnn_paging_results,
                plan_cudnn_query_lengths,
            },
            tuning::ForcedLibrary,
        },
        registry::{Operation, OperationAttributes, ScaledDotProductAttentionAttributes},
        spec,
    },
    data_type::DataType,
    error::{Error, Result},
    graph::ir::ValueId,
    layout::Layout,
    scalar::Scalar,
    shape::Shape,
    value::ValueSpec,
};

impl Builder<'_> {
    // Attention through cuDNN when it takes the options, and its generated kernel otherwise.
    pub fn cudnn_attention(&mut self, id: ValueId, inputs: &[ValueId]) -> Result<bool> {
        let attributes = self.ir.required_operation_for_value(id)?.attributes();
        let attention = attributes
            .scaled_dot_product_attention()
            .copied()
            .ok_or_else(|| Error::missing_node("program attention"))?;
        let data_type = self.ir.required_value_data_type(id)?;
        // cuDNN takes key lengths as they are, as compiled graphs pass them.
        let (query, key, value, key_lengths) = match *inputs {
            [query, key, value] => (query, key, value, None),
            [query, key, value, lengths]
                if attention.key_lengths && cudnn_reads_key_lengths(self.ir, lengths)? =>
            {
                (query, key, value, Some(lengths))
            }
            _ => return self.kernels(id),
        };
        let head_dimension = self
            .ir
            .required_value_shape(query)?
            .dimensions()
            .last()
            .copied()
            .unwrap_or(0);
        if !cudnn_takes_attention(data_type, head_dimension, &attention) {
            return self.kernels(id);
        }

        let (query, key, value) = (
            self.view_of(query)?,
            self.view_of(key)?,
            self.view_of(value)?,
        );
        let dim = query.layout.dimensions()[3];
        let lengths = key_lengths
            .map(|lengths| {
                let [batch, _, queries, _] = query.layout.dimensions()[..] else {
                    return Err(Error::missing_input("program attention queries"));
                };
                let plan = plan_cudnn_query_lengths(batch, queries)?;
                let query_lengths = self.zeroed_buffer(&plan.outputs[0])?;
                self.add_plan(&plan, &[], from_ref(&query_lengths));
                Ok(CudnnLengths {
                    query_lengths,
                    key_lengths: self.view_of(lengths)?,
                })
            })
            .transpose()?;
        let output = self.library_output(id)?;
        self.push_library(
            id,
            LibraryCall::CudnnAttention(Box::new(CudnnAttentionCall {
                query,
                key,
                value,
                output,
                scale: cudnn_attention_scale(&attention, dim),
                causal: attention.causal,
                window: attention.window,
                paged: None,
                lengths,
                engine: self
                    .planner
                    .context()
                    .forced_library(self.ir, id)?
                    .as_ref()
                    .and_then(ForcedLibrary::cudnn)
                    .cloned(),
            })),
        );
        self.finish_merged(
            id,
            &Epilogue::default(),
            StepProvider::Library(Library::Cudnn),
        )?;
        Ok(true)
    }

    // A zeroed buffer the program allocates once.
    fn zeroed_buffer(&mut self, spec: &ValueSpec) -> Result<BufferView> {
        let layout = Layout::from_shape(spec.shape());
        let bytes = span_bytes(spec.data_type(), &layout)?;
        Ok(BufferView {
            buffer: self.add_buffer(BufferKind::Scratch, bytes),
            data_type: spec.data_type(),
            layout,
        })
    }

    // cuDNN's attention over a paged cache, between the kernels that adapt the paging tensors to it and its statistics to `lse`.
    pub fn cudnn_paged_attention(
        &mut self,
        id: ValueId,
        (paging, attention): (&CudnnPaging, &ScaledDotProductAttentionAttributes),
        metadata: &[KernelInput],
        inputs: &[BufferView],
        outputs: &[BufferView],
    ) -> Result<bool> {
        let (
            [
                query,
                key,
                value,
                query_offsets,
                page_offsets,
                pages,
                last_page_lengths,
            ],
            [_, _, _, offset_input, page_input, _, length_input],
            [output, _],
        ) = (inputs, metadata, outputs)
        else {
            return Err(Error::missing_input("program paged attention"));
        };
        let indices = plan_cudnn_paging_indices(paging, &metadata[3..])?;
        let views = indices
            .outputs
            .iter()
            .map(|spec| self.zeroed_buffer(spec))
            .collect::<Result<Vec<_>>>()?;
        let [offsets, query_lengths, key_lengths, page_table] = views.as_slice() else {
            return Err(Error::missing_output("program paged attention"));
        };
        self.add_plan(
            &indices,
            &[
                query_offsets.clone(),
                page_offsets.clone(),
                pages.clone(),
                last_page_lengths.clone(),
            ],
            &views,
        );
        let stats = self.zeroed_buffer(&ValueSpec::new(
            DataType::F32,
            Shape::create([paging.sequences, paging.heads, paging.tokens])?,
        ))?;
        self.push_library(
            id,
            LibraryCall::CudnnAttention(Box::new(CudnnAttentionCall {
                query: query.clone(),
                key: key.clone(),
                value: value.clone(),
                output: output.clone(),
                scale: cudnn_attention_scale(attention, paging.dim),
                causal: attention.causal,
                window: attention.window,
                paged: Some(CudnnPagedCall {
                    paging: *paging,
                    offsets: offsets.clone(),
                    query_lengths: query_lengths.clone(),
                    key_lengths: key_lengths.clone(),
                    pages: page_table.clone(),
                    stats: stats.clone(),
                }),
                lengths: None,
                engine: self
                    .planner
                    .context()
                    .forced_library(self.ir, id)?
                    .as_ref()
                    .and_then(ForcedLibrary::cudnn)
                    .cloned(),
            })),
        );
        let results = plan_cudnn_paging_results(
            paging,
            attention,
            output.data_type,
            &[
                offset_input.clone(),
                page_input.clone(),
                length_input.clone(),
                KernelInput::new(DataType::F32, stats.layout.clone()),
            ],
        )?;
        self.add_plan(
            &results,
            &[
                query_offsets.clone(),
                page_offsets.clone(),
                last_page_lengths.clone(),
                stats,
            ],
            outputs,
        );
        Ok(true)
    }

    pub fn library_output(&mut self, id: ValueId) -> Result<BufferView> {
        let view = self.allocate(id)?;
        self.views.insert(id, view.clone());
        Ok(view)
    }

    pub fn finish_library(&mut self, id: ValueId, call: LibraryCall) -> Result<()> {
        self.record(id, StepProvider::Library(call.library()));
        self.push_library(id, call);
        self.release(id)
    }

    // Adds a step calling a library for the operation producing `id`.
    pub fn push_library(&mut self, id: ValueId, call: LibraryCall) {
        let operations = self.name(id);
        self.steps.push(ProgramStep::Library {
            call: Box::new(call),
            operations,
        });
    }

    pub fn fft(&mut self, id: ValueId, operation: Operation, input: ValueId) -> Result<bool> {
        let data_type = self.ir.required_value_data_type(id)?;
        let output_layout = self.ir.required_value_layout(id)?;
        let source = self.view_of(input)?;
        let dimensions = source.layout.dimensions().to_vec();
        let inverse = operation == Operation::Irfft;
        let length = if inverse {
            output_layout.dimensions().last().copied().unwrap_or(0)
        } else {
            dimensions.last().copied().unwrap_or(0)
        };
        if !spec::transforms_with_cufft(length, data_type) || !output_layout.is_contiguous() {
            return Ok(false);
        }

        let signals = if inverse {
            output_layout.element_count()
        } else {
            source.layout.element_count()
        };
        let batch = signals / usize::try_from(length).map_err(|_| Error::DimensionOutOfBounds)?;
        let double = data_type == DataType::F64;
        let spectrum = if inverse {
            // cuFFT overwrites what it reads and leaves its result unnormalized, so it reads a scaled copy of the frequencies the signal has.
            let needed = length / 2 + 1;
            if dimensions[dimensions.len() - 2] < needed {
                return Ok(false);
            }

            let narrowed = source.layout.slice(-2, 0, needed, 1)?;
            #[expect(
                clippy::cast_precision_loss,
                reason = "signal lengths are far below 2^53"
            )]
            let scale = 1.0 / length as f64;
            let plan = self.planner.create(
                Operation::Scale,
                &OperationAttributes::Scalar(Scalar::Float(scale)),
                &[KernelInput::new(data_type, narrowed)],
            )?;
            let mut shape = dimensions;
            let axis = shape.len() - 2;
            shape[axis] = needed;
            let layout = Layout::from_shape(&Shape::create(shape)?);
            let bytes = span_bytes(data_type, &layout)?;
            let copy = BufferView {
                buffer: self.add_buffer(BufferKind::Internal, bytes),
                data_type,
                layout,
            };
            self.add_plan(&plan, &[source], from_ref(&copy));
            copy
        } else {
            if !source.layout.is_contiguous() {
                return Ok(false);
            }
            source
        };
        let output = self.library_output(id)?;
        self.finish_library(
            id,
            LibraryCall::Fft {
                inverse,
                double,
                length,
                batch,
                input: spectrum,
                output,
            },
        )?;
        Ok(true)
    }

    pub fn scaled_matmul(&mut self, id: ValueId, inputs: &[ValueId]) -> Result<bool> {
        let &[lhs, rhs, lhs_scale, rhs_scale] = inputs else {
            return Ok(false);
        };
        let [
            Some(lhs_view),
            Some(rhs_view),
            Some(lhs_scale_view),
            Some(rhs_scale_view),
        ] = [lhs, rhs, lhs_scale, rhs_scale].map(|value| self.views.get(&value).cloned())
        else {
            return Err(Error::missing_input("program scaled matmul"));
        };
        let operand = |view: &BufferView| (view.layout.shape().clone(), view.data_type);
        let operands = [&lhs_view, &rhs_view, &lhs_scale_view, &rhs_scale_view].map(operand);
        let problem = spec::scaled_matmul_problem(
            (&operands[0].0, operands[0].1),
            (&operands[1].0, operands[1].1),
            (&operands[2].0, operands[2].1),
            (&operands[3].0, operands[3].1),
            "program scaled_mm",
        )?;
        if problem.scaling != spec::MatmulScaling::Tensorwise
            || !spec::takes_scaled_matmul(&lhs_view.layout, &rhs_view.layout, &problem)
        {
            return Ok(false);
        }

        let output = self.library_output(id)?;
        self.finish_library(
            id,
            LibraryCall::ScaledMatmul {
                rows: problem.rows,
                depth: problem.depth,
                columns: problem.columns,
                lhs: lhs_view,
                rhs: rhs_view,
                lhs_scale: lhs_scale_view,
                rhs_scale: rhs_scale_view,
                output,
            },
        )?;
        Ok(true)
    }
}

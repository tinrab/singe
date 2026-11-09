use std::{iter::once, slice::from_ref};

use singe_cublas::types::Operation as CublasOperation;

use crate::{
    backend::{
        cuda::{
            compiled::plan::{
                fusion::{FusionGroup, GroupKind},
                group::plan_group_kernel,
                lowering::{CudaGraphProvider, LoweringPolicy},
            },
            cudnn_backend::cudnn_takes_matmul,
            export::{
                builder::Builder,
                program::{
                    Activation, BufferView, GemmOperand, Library, LibraryCall, ProgramStep,
                    StepProvider,
                },
            },
            gemm::{GemmLayout, GemmMatrix, cublaslt_takes_strides},
            provider_selection::Operand,
            tuning::{ForcedLibrary, TunedWay},
        },
        registry::Operation,
        spec,
    },
    data_type::DataType,
    error::{Error, ExportFailure, ExportFailureReason, Result},
    graph::ir::{Graph, OperationNode, ValueId},
};

// What a library can compute inside its call, besides the operation it is called for.
#[derive(Debug, Clone, Copy, Default)]
pub struct Epilogue {
    // The add's value and its other operand.
    pub add: Option<(ValueId, ValueId)>,
    pub activation: Option<(ValueId, Activation)>,
}

impl Epilogue {
    pub const fn is_empty(&self) -> bool {
        self.add.is_none() && self.activation.is_none()
    }

    pub fn result(&self, base: ValueId) -> ValueId {
        self.activation
            .map(|(value, _)| value)
            .or_else(|| self.add.map(|(value, _)| value))
            .unwrap_or(base)
    }

    pub fn members(&self) -> impl Iterator<Item = ValueId> {
        self.add
            .map(|(value, _)| value)
            .into_iter()
            .chain(self.activation.map(|(value, _)| value))
    }
}

// Which allowed libraries can run a matmul.
#[derive(Debug, Clone, Default)]
pub struct MatmulLibraries {
    pub cublas: Option<GemmLayout>,
    pub cublaslt: Option<GemmLayout>,
    pub cudnn: bool,
}

impl MatmulLibraries {
    pub const fn any(&self) -> bool {
        self.cublas.is_some() || self.cublaslt.is_some() || self.cudnn
    }
}

pub fn gemm_call(
    gemm: &GemmLayout,
    lhs: BufferView,
    rhs: BufferView,
    output: BufferView,
) -> LibraryCall {
    LibraryCall::Gemm {
        rows: gemm.rows,
        columns: gemm.columns,
        inner: gemm.inner,
        batch_count: gemm.batch_count,
        lhs_operand: gemm_operand(gemm.lhs),
        rhs_operand: gemm_operand(gemm.rhs),
        lhs,
        rhs,
        output,
    }
}

pub const fn gemm_operand(matrix: GemmMatrix) -> GemmOperand {
    GemmOperand {
        transposed: matches!(matrix.operation, CublasOperation::Transpose),
        leading_dimension: matrix.leading_dimension,
        batch_stride: matrix.batch_stride,
    }
}

// The libraries that run this matmul, unless a tuning record chose otherwise.
pub fn matmul_libraries(
    ir: &Graph,
    policy: &LoweringPolicy,
    way: Option<TunedWay>,
    lhs: ValueId,
    rhs: ValueId,
) -> Result<MatmulLibraries> {
    let operand = |value: ValueId| -> Result<Operand<'_>> {
        Ok(Operand {
            data_type: ir.required_value_data_type(value)?,
            layout: ir.required_value_layout(value)?,
        })
    };
    let (left, right) = (operand(lhs)?, operand(rhs)?);
    let data_type = left.data_type;
    if data_type != right.data_type
        || !matches!(
            data_type,
            DataType::F16 | DataType::Bf16 | DataType::F32 | DataType::F64
        )
    {
        return Ok(MatmulLibraries::default());
    }

    let allowed = |provider| {
        policy.allows(provider, Operation::MatMul)
            && match way {
                Some(TunedWay::Library(library)) => library == provider,
                Some(TunedWay::Generated) => false,
                _ => true,
            }
    };
    let gemm = GemmLayout::create(left.layout, right.layout);
    // A count too large for `i64` reads as one element, which no check takes.
    let elements = |count: usize| i64::try_from(count).unwrap_or(1);
    let cublas = gemm.clone().filter(|_| allowed(CudaGraphProvider::Cublas));
    // A batch dimension one operand repeats has no strided batch in cuBLASLt.
    let cublaslt = gemm.filter(|layout| {
        allowed(CudaGraphProvider::CublasLt)
            && (layout.batch_count == 1
                || (layout.lhs.batch_stride != 0 && layout.rhs.batch_stride != 0))
            && cublaslt_takes_strides(
                &[
                    elements(layout.lhs.leading_dimension),
                    elements(layout.rhs.leading_dimension),
                    layout.lhs.batch_stride,
                    layout.rhs.batch_stride,
                    elements(layout.columns),
                ],
                data_type.size_bytes(),
            )
    });
    let cudnn = allowed(CudaGraphProvider::CudnnBackend)
        && cudnn_takes_matmul(data_type, left.layout, right.layout);
    Ok(MatmulLibraries {
        cublas,
        cublaslt,
        cudnn,
    })
}

impl Builder<'_> {
    pub fn matmul(&mut self, id: ValueId, lhs: ValueId, rhs: ValueId) -> Result<bool> {
        let (lhs_layout, rhs_layout) = (
            self.ir.required_value_layout(lhs)?,
            self.ir.required_value_layout(rhs)?,
        );
        let problem = spec::matmul_problem(lhs_layout.shape(), rhs_layout.shape(), "program")?;
        if problem.is_zero_work() {
            let destination = self.allocate(id)?;
            self.views.insert(id, destination.clone());
            self.steps.push(ProgramStep::Fill { destination });
            self.record(id, StepProvider::Kernel);
            self.release(id)?;
            return Ok(true);
        }

        let Some(libraries) = self.library_matmuls.get(&id).cloned() else {
            return self.kernels(id);
        };
        // What follows joins the call, so every option computes the same operations.
        let epilogue = if libraries.cudnn {
            self.epilogue(id, false)?
        } else if libraries.cublaslt.is_some() {
            self.epilogue(id, true)?
        } else {
            Epilogue::default()
        };
        if epilogue.is_empty()
            && let Some(gemm) = &libraries.cublas
        {
            // cuBLAS has no setup to refuse.
            let call = gemm_call(
                gemm,
                self.view_of(lhs)?,
                self.view_of(rhs)?,
                self.library_output(id)?,
            );
            self.finish_library(id, call)?;
            return Ok(true);
        }

        let output = self.library_output(epilogue.result(id))?;
        let (lhs_view, rhs_view) = (self.view_of(lhs)?, self.view_of(rhs)?);
        let addend = epilogue
            .add
            .map(|(_, addend)| self.view_of(addend))
            .transpose()?;
        let activation = epilogue.activation.map(|(_, activation)| activation);
        let forced = self.planner.context().forced_library(self.ir, id)?;
        if let Some(forced) = &forced {
            forced.records.note_used(&forced.key);
        }

        // The first library the checks allow runs it, with nothing to fall back to, so what a library then refuses fails the program at `model_create`.
        if libraries.cudnn {
            self.push_library(
                id,
                LibraryCall::CudnnMatmul {
                    lhs: lhs_view,
                    rhs: rhs_view,
                    addend,
                    activation,
                    output,
                    engine: forced.as_ref().and_then(ForcedLibrary::cudnn).cloned(),
                },
            );
            self.finish_merged(id, &epilogue, StepProvider::Library(Library::Cudnn))?;
            return Ok(true);
        }

        let contiguous = addend.as_ref().is_none_or(|addend| {
            addend.layout.is_contiguous() && addend.layout.storage_offset() == 0
        });
        if let Some(layout) = libraries.cublaslt.as_ref().filter(|_| contiguous) {
            self.push_library(
                id,
                LibraryCall::CublasLtMatmul {
                    rows: layout.rows,
                    columns: layout.columns,
                    inner: layout.inner,
                    batch_count: layout.batch_count,
                    lhs_operand: gemm_operand(layout.lhs),
                    rhs_operand: gemm_operand(layout.rhs),
                    lhs: lhs_view,
                    rhs: rhs_view,
                    addend,
                    activation,
                    output,
                    config: forced.as_ref().and_then(ForcedLibrary::cublaslt).copied(),
                },
            );
            self.finish_merged(id, &epilogue, StepProvider::Library(Library::CublasLt))?;
            return Ok(true);
        }

        if let Some(gemm) = &libraries.cublas {
            // cuBLAS computes the matmul alone, and kernels what follows.
            let built = self.attempt(|builder| {
                let product = builder.intermediate_view(id)?;
                builder.push_library(id, gemm_call(gemm, lhs_view, rhs_view, product.clone()));
                builder.kernels_after(id, product, &epilogue, &output)
            })?;
            if built {
                self.finish_merged(id, &epilogue, StepProvider::Library(Library::Cublas))?;
                return Ok(true);
            }
        } else if self.attempt(|builder| builder.generated_matmul(id, &epilogue, &output))? {
            self.finish_merged(id, &epilogue, StepProvider::Kernel)?;
            return Ok(true);
        }

        Err(Error::Export {
            failures: vec![ExportFailure {
                operation: self.name(id),
                reason: ExportFailureReason::NoProvider,
            }],
        })
    }

    // Generated kernels for a matmul and what follows it: one fused kernel where fusion can write it, and one each otherwise.
    pub fn generated_matmul(
        &mut self,
        id: ValueId,
        epilogue: &Epilogue,
        output: &BufferView,
    ) -> Result<bool> {
        if self.policy.provider_enabled(CudaGraphProvider::Fusion) && !epilogue.is_empty() {
            let group = FusionGroup {
                kind: GroupKind::Contraction {
                    matmul: id,
                    weight: None,
                },
                members: once(id).chain(epilogue.members()).collect(),
                outputs: vec![epilogue.result(id)],
                inputs: Vec::new(),
                saved_bytes: 0,
            };
            if let Some(kernel) = plan_group_kernel(self.planner, self.ir, &group)? {
                let inputs = kernel
                    .inputs
                    .iter()
                    .map(|&value| self.view_of(value))
                    .collect::<Result<Vec<_>>>()?;
                self.add_plan(&kernel.plan, &inputs, from_ref(output));
                return Ok(true);
            }
        }
        self.generated_chain(id, epilogue, output)
    }

    // A kernel for `id`, then one for each operation of `epilogue`.
    pub fn generated_chain(
        &mut self,
        id: ValueId,
        epilogue: &Epilogue,
        output: &BufferView,
    ) -> Result<bool> {
        let first = if epilogue.is_empty() {
            output.clone()
        } else {
            self.intermediate_view(id)?
        };
        Ok(self.kernel_into(id, None, &first)?
            && self.kernels_after(id, first, epilogue, output)?)
    }

    // Runs `build`, and takes back the steps and buffers it added if it could not write them.
    pub fn attempt(&mut self, build: impl FnOnce(&mut Self) -> Result<bool>) -> Result<bool> {
        let (steps, buffers) = (self.steps.len(), self.buffers.len());
        if build(self)? {
            return Ok(true);
        }

        self.steps.truncate(steps);
        self.buffers.truncate(buffers);
        self.free.retain(|buffer| buffer.0 < buffers);
        if self.status.is_some_and(|status| status.0 >= buffers) {
            self.status = None;
        }
        Ok(false)
    }

    // Records that what the steps just added run `id` and `epilogue`, which then need no steps of their own.
    pub fn finish_merged(
        &mut self,
        id: ValueId,
        epilogue: &Epilogue,
        provider: StepProvider,
    ) -> Result<()> {
        for member in once(id).chain(epilogue.members()) {
            self.record(member, provider);
        }

        for member in epilogue.members() {
            self.placed.insert(member);
            self.merged.insert(member);
        }
        self.release(id)
    }

    // The value's only reader, when nothing else needs the value itself.
    pub fn only_reader(&self, value: ValueId) -> Result<Option<(ValueId, &OperationNode)>> {
        let ir = self.ir;
        if self.outputs.contains(&value) {
            return Ok(None);
        }

        let Some(&[reader]) = self.users.get(&value).map(Vec::as_slice) else {
            return Ok(None);
        };
        Ok(Some((reader, ir.required_operation_for_value(reader)?)))
    }

    // What a library can compute inside its call to `base`, which saves the steps and buffers between them: an add of a ready value of the same shape, then an activation.
    pub fn epilogue(&self, base: ValueId, contiguous_addend: bool) -> Result<Epilogue> {
        let ir = self.ir;
        let data_type = ir.required_value_data_type(base)?;
        let shape = ir.required_value_shape(base)?;
        let mut epilogue = Epilogue::default();
        let mut last = base;
        if let Some((add, operation)) = self.only_reader(last)?
            && operation.opcode() == Some(Operation::Add)
            && let &[a, b] = operation.inputs()
            && a != b
        {
            let addend = if a == last { b } else { a };
            let fits = self.views.get(&addend).is_some_and(|view| {
                view.data_type == data_type
                    && view.layout.shape() == shape
                    && view.layout.strides().iter().all(|&stride| stride > 0)
                    && (!contiguous_addend
                        || (view.layout.is_contiguous() && view.layout.storage_offset() == 0))
            });
            if fits
                && ir.required_value_data_type(add)? == data_type
                && ir.required_value_shape(add)? == shape
            {
                epilogue.add = Some((add, addend));
                last = add;
            }
        }

        if let Some((value, operation)) = self.only_reader(last)? {
            let activation = match operation.opcode() {
                Some(Operation::Relu) => Some(Activation::Relu),
                Some(Operation::Gelu) => Some(Activation::Gelu),
                _ => None,
            };
            if let Some(activation) = activation
                && ir.required_value_data_type(value)? == data_type
            {
                epilogue.activation = Some((value, activation));
            }
        }
        Ok(epilogue)
    }
}

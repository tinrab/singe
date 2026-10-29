use std::{fmt::Debug, marker::PhantomData, sync::Arc};

use crate::{
    ast::{AddressSize, ScalarType, StateSpace},
    emulator::{
        CompileError, CompileErrorKind, CompiledKernel, ParsedModule,
        execution_error::SubmitError,
        layout::{ModuleLayout, ParameterLayout},
        memory::{generic_address, specific},
        runtime::{
            Client, CommandKind, Lease,
            buffer::{AllocationRef, Argument, BufferView, KernelArg, foreign},
            completion::{Retained, Submission},
            device::Stream,
            options::{ExecutionReport, LaunchConfig},
            submit::submit_retained,
            worker::{LaunchWork, Worker},
        },
    },
};

#[derive(Debug)]
pub struct Loaded {
    pub layout: ModuleLayout,
    pub addresses: Vec<u64>,
    pub lease: Arc<Lease>,
}

/// A module loaded on an emulator, with its own global and constant variables.
#[derive(Debug, Clone)]
pub struct Module {
    pub(crate) client: Arc<Client>,
    pub(crate) parsed: ParsedModule,
    pub(crate) loaded: Arc<Loaded>,
}

impl Module {
    /// Compiles the `name` kernel, or reuses a cached compilation, and binds it to this
    /// module.
    ///
    /// # Errors
    ///
    /// Returns an error if `name` is not an entry point, or if the kernel or a function it
    /// calls is invalid or unsupported.
    pub fn kernel(&self, name: &str) -> Result<Kernel, CompileError> {
        self.bind(&self.parsed.compile(name)?)
    }

    /// Binds already compiled code to this module without recompiling it.
    ///
    /// `code` must come from the same [`ParsedModule`] this module was loaded from, or
    /// from a clone of it. Modules parsed separately are distinct even if their text is
    /// identical.
    ///
    /// # Errors
    ///
    /// Returns [`CompileErrorKind::ModuleMismatch`] if `code` was compiled from a
    /// different parsed module.
    pub fn bind(&self, code: &CompiledKernel) -> Result<Kernel, CompileError> {
        if !Arc::ptr_eq(&self.parsed.inner, &code.source.inner) {
            return Err(CompileError::new(
                code.source.syntax().span,
                CompileErrorKind::ModuleMismatch,
            ));
        }
        Ok(Kernel {
            module: self.clone(),
            code: code.clone(),
        })
    }

    /// Returns the bytes of the global or constant variable `name`.
    ///
    /// Use it to read results a kernel stored in a module variable, including after a
    /// fault.
    ///
    /// # Errors
    ///
    /// Returns an error if no global or constant variable is named `name`, the variable is
    /// `.shared` (which has one instance per CTA), or its size does not fit in `usize`.
    pub fn global(&self, name: &str) -> Result<BufferView<u8>, SubmitError> {
        let id = self
            .loaded
            .layout
            .symbols
            .get(name)
            .ok_or_else(|| SubmitError::InvalidArgument(format!("unknown global `{name}`")))?
            .0;
        let variable = &self.loaded.layout.variables[id];
        if variable.space == StateSpace::Shared {
            return Err(SubmitError::InvalidArgument(
                "shared storage is per CTA".into(),
            ));
        }
        Ok(BufferView {
            client: Arc::clone(&self.client),
            allocation: AllocationRef {
                _lease: Arc::clone(&self.loaded.lease),
                address: self.loaded.addresses[id],
                writable: variable.space != StateSpace::Const,
                space: variable.space,
            },
            offset: 0,
            len: usize::try_from(variable.layout.byte_len())
                .map_err(|_| SubmitError::ExecutionStateLimit)?,
            marker: PhantomData,
        })
    }
}

/// A kernel bound to a loaded module.
#[derive(Debug, Clone)]
pub struct Kernel {
    pub(crate) module: Module,
    pub(crate) code: CompiledKernel,
}

impl Kernel {
    /// Launches the kernel on the default stream with `arguments` in parameter order.
    ///
    /// See [`launch_on`](Self::launch_on).
    ///
    /// # Errors
    ///
    /// Returns an error if `config` does not meet the kernel's block requirements, an
    /// argument does not match its parameter, staging storage cannot hold the arguments,
    /// or the command is not admitted. Execution faults are reported through the
    /// submission.
    pub fn launch(
        &self,
        config: LaunchConfig,
        arguments: impl IntoIterator<Item = KernelArg>,
    ) -> Result<Submission<ExecutionReport>, SubmitError> {
        self.launch_on(
            &Stream {
                client: Arc::clone(&self.module.client),
                id: 0,
            },
            config,
            arguments,
        )
    }

    /// Launches the kernel on `stream` with `arguments` in parameter order.
    ///
    /// The launch keeps the module and every buffer passed as a pointer argument alive
    /// until it completes. The submission yields an [`ExecutionReport`]; on a fault, the
    /// [`ExecutionError`](crate::emulator::ExecutionError) carries the report of the partial
    /// launch.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` or a buffer belongs to another emulator, `config` does
    /// not meet the kernel's block requirements, an argument does not match its
    /// parameter, staging storage cannot hold the arguments, or the command is not
    /// admitted. Execution faults are reported through the submission.
    pub fn launch_on(
        &self,
        stream: &Stream,
        config: LaunchConfig,
        arguments: impl IntoIterator<Item = KernelArg>,
    ) -> Result<Submission<ExecutionReport>, SubmitError> {
        foreign(&self.module.client, &stream.client)?;
        self.code
            .launch_requirements()
            .validate_block(config.block)?;
        let arguments: Vec<_> = arguments.into_iter().collect();
        if arguments.len() != self.code.parameters().len() {
            return Err(SubmitError::InvalidArgument("parameter count".into()));
        }
        let mut bytes = Vec::new();
        let mut leases = Vec::new();
        bytes
            .try_reserve_exact(arguments.len())
            .map_err(|_| SubmitError::StagingLimit)?;
        leases
            .try_reserve_exact(arguments.len())
            .map_err(|_| SubmitError::StagingLimit)?;
        let mut staged = bytes
            .capacity()
            .checked_mul(size_of::<Vec<u8>>())
            .and_then(|size| {
                size.checked_add(leases.capacity().checked_mul(size_of::<AllocationRef>())?)
            })
            .ok_or(SubmitError::StagingLimit)?;
        for (arg, parameter) in arguments.into_iter().zip(self.code.parameters()) {
            let value = self.argument_bytes(arg, parameter, &mut leases)?;
            staged = staged
                .checked_add(value.capacity())
                .ok_or(SubmitError::StagingLimit)?;
            bytes.push(value);
        }
        let code = self.code.clone();
        let loaded = Arc::clone(&self.module.loaded);
        let stream_id = stream.id;
        submit_retained(
            &self.module.client,
            stream.id,
            CommandKind::Launch,
            false,
            staged,
            move |staging| {
                let mut work = LaunchWork {
                    code,
                    loaded,
                    config,
                    args: bytes,
                    _leases: leases,
                    staging,
                    engine: None,
                    builder: None,
                    pause: None,
                    stream_id,
                    result_storage: None,
                };
                move |worker: &mut Worker, terminal| {
                    work.poll(worker, terminal).map(|result| Retained {
                        result,
                        storage: work.result_storage.take(),
                    })
                }
            },
        )
    }

    /// Encodes one kernel argument for `parameter`, recording any pointer's allocation lease.
    fn argument_bytes(
        &self,
        arg: KernelArg,
        parameter: &ParameterLayout,
        leases: &mut Vec<AllocationRef>,
    ) -> Result<Vec<u8>, SubmitError> {
        let address_size = self.module.loaded.layout.address_size;
        let value = match arg.value {
            Argument::Scalar(ty, bytes)
                if !parameter.aggregate
                    && parameter.pointer.is_none()
                    && representation(ty, parameter.ty) =>
            {
                bytes
            }
            Argument::Pointer {
                client,
                allocation,
                address,
            } if !parameter.aggregate
                && match address_size {
                    AddressSize::Bits32 => matches!(
                        parameter.ty,
                        ScalarType::U32 | ScalarType::B32 | ScalarType::S32
                    ),
                    AddressSize::Bits64 => matches!(
                        parameter.ty,
                        ScalarType::U64 | ScalarType::B64 | ScalarType::S64
                    ),
                } =>
            {
                foreign(&self.module.client, &client)?;
                if parameter.pointer.is_some_and(|p| {
                    p.state_space.is_some_and(|s| s != allocation.space)
                        || p.alignment
                            .is_some_and(|a| !address.is_multiple_of(u64::from(a)))
                }) {
                    return Err(SubmitError::InvalidArgument("pointer annotation".into()));
                }
                // Unannotated pointers and .ptr without a state space use generic addresses.
                let address = if parameter.pointer.is_some_and(|p| p.state_space.is_some()) {
                    specific(address)
                } else {
                    generic_address(address)
                };
                leases.push(allocation);
                match address_size {
                    AddressSize::Bits32 => u32::try_from(address)
                        .map_err(|_| {
                            SubmitError::InvalidArgument(
                                "pointer exceeds 32-bit module address range".into(),
                            )
                        })?
                        .to_le_bytes()
                        .to_vec(),
                    AddressSize::Bits64 => address.to_le_bytes().to_vec(),
                }
            }
            Argument::Aggregate { bytes, alignment }
                if parameter.aggregate
                    && alignment >= parameter.layout.alignment()
                    && bytes.len() as u64 == parameter.layout.byte_len() =>
            {
                bytes
            }
            _ => {
                return Err(SubmitError::InvalidArgument(format!(
                    "representation of `{}`",
                    parameter.name
                )));
            }
        };
        Ok(value)
    }
}

pub fn representation(actual: ScalarType, expected: ScalarType) -> bool {
    actual == expected
        || matches!(
            (actual, expected),
            (ScalarType::U8 | ScalarType::S8, ScalarType::B8)
                | (
                    ScalarType::U16 | ScalarType::S16 | ScalarType::F16 | ScalarType::Bf16,
                    ScalarType::B16
                )
                | (
                    ScalarType::U32 | ScalarType::S32 | ScalarType::F32,
                    ScalarType::B32
                )
                | (
                    ScalarType::U64 | ScalarType::S64 | ScalarType::F64,
                    ScalarType::B64
                )
        )
}

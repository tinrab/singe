use std::{
    collections::HashMap,
    sync::{OnceLock, RwLock},
};

use crate::{
    autograd,
    backend::{register_kernels, registry::Operation},
    dispatch::{DispatchKey, Kernel, autocast, custom::CustomOperation},
    error::{Error, Result},
    graph::capture,
};

/// Kernels registered for one operation.
#[derive(Debug, Clone, Default)]
pub struct OperatorEntry {
    kernels: [Option<Kernel>; DispatchKey::COUNT],
    composite: Option<Kernel>,
}

impl OperatorEntry {
    pub(crate) fn kernel(&self, key: DispatchKey) -> Option<&Kernel> {
        self.kernels[key.index()].as_ref()
    }

    /// Returns the kernel that runs at backend `key`: its own kernel, or the composite one.
    pub(crate) fn backend_kernel(&self, key: DispatchKey) -> Option<&Kernel> {
        self.kernel(key).or(self.composite.as_ref())
    }
}

/// Collects kernels while the registry is built.
#[derive(Debug, Default)]
pub struct RegistryBuilder {
    entries: HashMap<Operation, OperatorEntry>,
    fallbacks: [Option<Kernel>; DispatchKey::COUNT],
}

impl RegistryBuilder {
    /// Registers `kernel` for `operation` at `key`, replacing any earlier one.
    pub fn kernel(&mut self, operation: Operation, key: DispatchKey, kernel: Kernel) -> &mut Self {
        self.entries.entry(operation).or_default().kernels[key.index()] = Some(kernel);
        self
    }

    /// Returns the kernel registered for `operation` at `key`, if any.
    #[must_use]
    pub fn registered(&self, operation: Operation, key: DispatchKey) -> Option<Kernel> {
        self.entries
            .get(&operation)
            .and_then(|entry| entry.kernels[key.index()].clone())
    }

    /// Registers a composite `kernel` for `operation`, used by any backend without its own kernel.
    ///
    /// A composite kernel computes the operation from other operations through the dispatcher.
    pub fn composite(&mut self, operation: Operation, kernel: Kernel) -> &mut Self {
        self.entries.entry(operation).or_default().composite = Some(kernel);
        self
    }

    /// Registers `kernel` for every operation at frontend `key` that has no kernel of its own there.
    pub fn fallback(&mut self, key: DispatchKey, kernel: Kernel) -> &mut Self {
        self.fallbacks[key.index()] = Some(kernel);
        self
    }
}

#[derive(Debug)]
pub struct Registry {
    entries: HashMap<Operation, OperatorEntry>,
    fallbacks: [Option<Kernel>; DispatchKey::COUNT],
    custom: RwLock<HashMap<CustomOperation, OperatorEntry>>,
}

impl Registry {
    pub(crate) fn get() -> &'static Self {
        static REGISTRY: OnceLock<Registry> = OnceLock::new();
        REGISTRY.get_or_init(|| {
            let mut builder = RegistryBuilder::default();
            register_kernels(&mut builder);
            autocast::register(&mut builder);
            capture::register(&mut builder);
            autograd::register(&mut builder);
            Self {
                entries: builder.entries,
                fallbacks: builder.fallbacks,
                custom: RwLock::default(),
            }
        })
    }

    /// Returns a copy of the kernels registered for `operation`.
    ///
    /// # Errors
    ///
    /// Returns an error if a thread panicked while registering a custom operation.
    pub(crate) fn entry(&self, operation: Operation) -> Result<Option<OperatorEntry>> {
        Ok(match operation {
            Operation::Custom(custom) => self
                .custom
                .read()
                .map_err(|_| Error::invalid_argument("custom operation registry"))?
                .get(&custom)
                .cloned(),
            builtin => self.entries.get(&builtin).cloned(),
        })
    }

    pub(crate) fn fallback(&self, key: DispatchKey) -> Option<&Kernel> {
        self.fallbacks[key.index()].as_ref()
    }

    /// Registers a kernel for a custom operation at `key`, or as its composite kernel when `key` is `None`.
    ///
    /// # Errors
    ///
    /// Returns an error if a thread panicked while registering a custom operation.
    pub(crate) fn register_custom(
        &self,
        operation: CustomOperation,
        key: Option<DispatchKey>,
        kernel: Kernel,
    ) -> Result<()> {
        let mut custom = self
            .custom
            .write()
            .map_err(|_| Error::invalid_argument("custom operation registry"))?;
        let entry = custom.entry(operation).or_default();
        match key {
            Some(key) => entry.kernels[key.index()] = Some(kernel),
            None => entry.composite = Some(kernel),
        }

        drop(custom);
        Ok(())
    }
}

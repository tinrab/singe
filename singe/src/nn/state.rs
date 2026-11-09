use std::{
    collections::{BTreeMap, HashMap, HashSet},
    sync::Arc,
};

use crate::{
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    graph::{CaptureBindings, Graph, GraphProgram},
    state::{StateBindingKind, StateBindingSpec, StateSchema, StateSnapshot},
    tensor::Tensor,
    value::ValueSpec,
};

/// A reference to one named parameter or persistent buffer in a model schema.
/// Cloning a key reuses the binding.
/// `StateBuilder::tie` declares another name.
#[derive(Debug, Clone)]
pub struct StateKey {
    owner: Arc<()>,
    name: String,
}

impl StateKey {
    /// Returns the registered parameter or buffer name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Registers a model's named state before publishing its immutable snapshot.
#[derive(Debug, Default)]
pub struct StateBuilder {
    owner: Arc<()>,
    bindings: BTreeMap<String, StateBindingSpec>,
    tensors: HashMap<String, Tensor>,
}

impl StateBuilder {
    /// Creates an empty state registry with its own key-ownership identity.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `tensor` as a named parameter and returns its state key.
    ///
    /// # Errors
    ///
    /// Returns an error if the name is already registered or contains an empty dot-separated component.
    pub fn parameter(&mut self, name: impl Into<String>, tensor: Tensor) -> Result<StateKey> {
        self.insert(name.into(), tensor, StateBindingKind::Parameter)
    }

    /// Registers `tensor` as a named persistent buffer and returns its state key.
    ///
    /// # Errors
    ///
    /// Returns an error if the name is already registered or contains an empty dot-separated component.
    pub fn buffer(&mut self, name: impl Into<String>, tensor: Tensor) -> Result<StateKey> {
        self.insert(name.into(), tensor, StateBindingKind::Buffer)
    }

    fn insert(&mut self, name: String, tensor: Tensor, kind: StateBindingKind) -> Result<StateKey> {
        self.validate_name(&name)?;
        self.bindings.insert(
            name.clone(),
            StateBindingSpec {
                id: self.bindings.len(),
                name: name.clone(),
                spec: ValueSpec::new(tensor.data_type(), tensor.shape().clone()),
                kind,
            },
        );
        self.tensors.insert(name.clone(), tensor);
        Ok(StateKey {
            owner: Arc::clone(&self.owner),
            name,
        })
    }

    // Returns a view of this builder that prefixes each name it registers with `prefix` and a dot.
    pub fn scope(&mut self, prefix: impl Into<String>) -> StateScope<'_> {
        StateScope {
            builder: self,
            prefix: prefix.into(),
        }
    }

    // Returns a view of this builder that adds no prefix.
    pub fn unscoped(&mut self) -> StateScope<'_> {
        StateScope {
            builder: self,
            prefix: String::new(),
        }
    }

    /// Registers a second schema name for the same logical parameter or buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if the new name is invalid or already registered, the target key belongs to another builder, or the target binding is absent.
    pub fn tie(&mut self, name: impl Into<String>, target: &StateKey) -> Result<StateKey> {
        let name = name.into();
        self.validate_name(&name)?;
        if !Arc::ptr_eq(&self.owner, &target.owner) {
            return Err(Error::invalid_argument("foreign model state key"));
        }

        let mut binding = self
            .bindings
            .get(&target.name)
            .cloned()
            .ok_or_else(|| Error::missing_input("binding of the tied state key"))?;
        binding.name.clone_from(&name);
        self.bindings.insert(name.clone(), binding);
        Ok(StateKey {
            owner: Arc::clone(&self.owner),
            name,
        })
    }

    fn validate_name(&self, name: &str) -> Result<()> {
        if name.split('.').any(str::is_empty) || self.bindings.contains_key(name) {
            return Err(Error::invalid_argument(format!(
                "model state name `{name}`"
            )));
        }
        Ok(())
    }

    /// Consumes the builder and publishes validated immutable model state.
    ///
    /// # Errors
    ///
    /// Returns an error if state bindings are missing or incompatible, tied values conflict, comparing tied tensors fails, or allocating a state identity exhausts its counter.
    pub fn finish(self) -> Result<ModelState> {
        Ok(ModelState {
            owner: self.owner,
            snapshot: StateSnapshot::create(
                Arc::new(StateSchema {
                    bindings: self.bindings,
                }),
                &self.tensors,
            )?,
        })
    }
}

// Registers state under a name prefix, for one layer of a model.
#[derive(Debug)]
pub struct StateScope<'a> {
    builder: &'a mut StateBuilder,
    prefix: String,
}

impl StateScope<'_> {
    fn named(&self, name: &str) -> String {
        if self.prefix.is_empty() {
            name.to_owned()
        } else {
            format!("{}.{name}", self.prefix)
        }
    }

    // Returns a narrower scope, whose names also start with `name`.
    pub fn scope(&mut self, name: impl AsRef<str>) -> StateScope<'_> {
        StateScope {
            prefix: self.named(name.as_ref()),
            builder: self.builder,
        }
    }

    // Registers `tensor` as a named parameter under this scope.
    pub fn parameter(&mut self, name: impl AsRef<str>, tensor: Tensor) -> Result<StateKey> {
        let name = self.named(name.as_ref());
        self.builder.parameter(name, tensor)
    }

    // Registers `tensor` as a named persistent buffer under this scope.
    pub fn buffer(&mut self, name: impl AsRef<str>, tensor: Tensor) -> Result<StateKey> {
        let name = self.named(name.as_ref());
        self.builder.buffer(name, tensor)
    }

    // Registers a second name under this scope for the state of `target`.
    pub fn tie(&mut self, name: impl AsRef<str>, target: &StateKey) -> Result<StateKey> {
        let name = self.named(name.as_ref());
        self.builder.tie(name, target)
    }
}

/// Immutable model state with explicit tied-name groups and publication identity.
#[derive(Debug, Clone)]
pub struct ModelState {
    pub(crate) snapshot: StateSnapshot,
    owner: Arc<()>,
}

impl ModelState {
    pub(crate) fn same_owner(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.owner, &other.owner)
    }

    // Returns the one device this state's tensors are on.
    pub fn device(&self) -> Result<Device> {
        let mut devices = self.bindings().map(|(_, tensor)| tensor.device());
        let device = devices
            .next()
            .ok_or_else(|| Error::missing_input("model state tensor to take a device from"))?;
        if let Some(other) = devices.find(|&other| other != device) {
            return Err(Error::device_mismatch("model state", device, other));
        }
        Ok(device)
    }

    // Returns the floating type this state's weights are in, which caches and tables follow.
    pub fn floating_type(&self) -> Result<DataType> {
        self.bindings()
            .map(|(_, tensor)| tensor.data_type())
            .find(|data_type| data_type.is_float())
            .ok_or_else(|| Error::missing_input("model state floating tensor"))
    }

    // Returns this state with `buffers` added under new names, for what a program binds beyond the checkpoint.
    pub fn with_buffers(&self, buffers: Vec<(String, Tensor)>) -> Result<Self> {
        let mut bindings = self.snapshot.schema.bindings.clone();
        let mut tensors = (*self.snapshot.tensors).clone();
        let first = bindings
            .values()
            .map(StateBindingSpec::id)
            .max()
            .map_or(0, |id| id + 1);
        for (id, (name, tensor)) in (first..).zip(buffers) {
            let binding = StateBindingSpec {
                id,
                name: name.clone(),
                spec: ValueSpec::new(tensor.data_type(), tensor.shape().clone()),
                kind: StateBindingKind::Buffer,
            };
            if bindings.insert(name.clone(), binding).is_some() {
                return Err(Error::invalid_argument(format!(
                    "model state name `{name}`"
                )));
            }

            tensors.insert(name, tensor);
        }
        Ok(Self {
            owner: Arc::clone(&self.owner),
            snapshot: StateSnapshot::create(Arc::new(StateSchema { bindings }), &tensors)?,
        })
    }

    /// Returns the immutable state publication and its binding schema.
    #[must_use]
    pub fn snapshot(&self) -> &StateSnapshot {
        &self.snapshot
    }

    /// Returns the tensor identified by a key from this state's originating builder.
    ///
    /// # Errors
    ///
    /// Returns an error if the key belongs to another builder or its binding or tensor is absent.
    pub fn tensor(&self, key: &StateKey) -> Result<&Tensor> {
        self.binding(key)?;
        self.named_tensor(&key.name)
            .ok_or_else(|| Error::missing_input("tensor of the model state key"))
    }

    pub(crate) fn binding(&self, key: &StateKey) -> Result<&StateBindingSpec> {
        if !Arc::ptr_eq(&self.owner, &key.owner) {
            return Err(Error::invalid_argument("foreign model state key"));
        }
        self.snapshot
            .schema
            .bindings
            .get(&key.name)
            .ok_or_else(|| Error::missing_input("binding of the model state key"))
    }

    /// Returns the tensor for `name`, or `None` when no tensor is bound under that name.
    #[must_use]
    pub fn named_tensor(&self, name: &str) -> Option<&Tensor> {
        self.snapshot.tensor(name)
    }

    /// Iterates schema bindings in name order, pairing each with its tensor.
    ///
    /// Tied names remain separate entries.
    pub fn bindings(&self) -> impl Iterator<Item = (&StateBindingSpec, &Tensor)> {
        self.snapshot
            .schema
            .bindings
            .values()
            .map(|binding| (binding, &self.snapshot.tensors[binding.name()]))
    }

    /// Returns the learned parameters, the tensors bound as `StateBindingKind::Parameter`, in name order.
    ///
    /// Tied names share one tensor, which appears once, under its first name.
    /// Optimizers take these tensors: `Sgd::new(state.parameters(), learning_rate)`.
    #[must_use]
    pub fn parameters(&self) -> Vec<Tensor> {
        self.named_parameters()
            .into_iter()
            .map(|(_, tensor)| tensor)
            .collect()
    }

    /// Returns each learned parameter with the first name it is bound under, in name order.
    #[must_use]
    pub fn named_parameters(&self) -> Vec<(&str, Tensor)> {
        let mut seen = HashSet::new();
        self.bindings()
            .filter(|(binding, _)| binding.kind() == StateBindingKind::Parameter)
            .filter(|(_, tensor)| seen.insert(tensor.id()))
            .map(|(binding, tensor)| (binding.name(), tensor.clone()))
            .collect()
    }

    // Lifts a graph captured with [`Self::capture_bindings`] into a program over this state's schema, declaring the tied names those bindings left out.
    pub fn program(&self, graph: &Graph) -> Result<GraphProgram> {
        let mut graph = graph.clone();
        let mut seen = HashSet::new();
        for (binding, tensor) in self.bindings() {
            if seen.insert(binding.id()) {
                continue;
            }

            graph.add_input(
                binding.name(),
                tensor.data_type(),
                tensor.layout().clone(),
                tensor.device(),
            );
            match binding.kind() {
                StateBindingKind::Parameter => {
                    graph.add_parameter(binding.name(), tensor.clone())?;
                }
                StateBindingKind::Buffer => graph.add_buffer(binding.name(), tensor.clone())?,
                StateBindingKind::Static | StateBindingKind::Default => {
                    return Err(Error::invalid_argument("model state role"));
                }
            }
        }
        graph.program()?.with_state_schema(self.snapshot())
    }

    // Returns explicit capture bindings for this state, one for each logical tensor, so a capture closure may read them directly.
    pub fn capture_bindings(&self) -> Result<CaptureBindings> {
        let mut seen = HashSet::new();
        self.bindings()
            .filter(|(binding, _)| seen.insert(binding.id()))
            .try_fold(
                CaptureBindings::new(),
                |bindings, (binding, tensor)| match binding.kind() {
                    StateBindingKind::Parameter => {
                        Ok(bindings.with_parameter(binding.name(), tensor.clone()))
                    }
                    StateBindingKind::Buffer => {
                        Ok(bindings.with_buffer(binding.name(), tensor.clone()))
                    }
                    StateBindingKind::Static | StateBindingKind::Default => {
                        Err(Error::invalid_argument("model state role"))
                    }
                },
            )
    }

    /// Returns a new publication only after all replacements and ties validate.
    ///
    /// # Errors
    ///
    /// Returns an error if replacement names or logical metadata are invalid, tied values conflict, a required tensor read fails, or state revision numbering is exhausted.
    pub fn replace(&self, replacements: &HashMap<String, Tensor>) -> Result<Self> {
        Ok(Self {
            owner: Arc::clone(&self.owner),
            snapshot: self.snapshot.replace(replacements)?,
        })
    }

    /// Visits each logical binding once.
    /// Floating conversion preserves integer and Bool buffers.
    /// A changed dtype requires recapture of existing graphs.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested floating type is not `F16`, `Bf16`, `F32`, or `F64`, a binding is missing, a tensor cast or device transfer fails, or publishing the converted state fails validation or identity allocation.
    pub fn to(&self, device: Device, floating_dtype: Option<DataType>) -> Result<Self> {
        if floating_dtype.is_some_and(|dtype| {
            !matches!(
                dtype,
                DataType::F16 | DataType::Bf16 | DataType::F32 | DataType::F64
            )
        }) {
            return Err(Error::invalid_argument("model floating dtype"));
        }

        let dtype_changed = self.bindings().any(|(_, tensor)| {
            tensor.data_type().is_float()
                && floating_dtype.is_some_and(|dtype| dtype != tensor.data_type())
        });
        if !dtype_changed {
            return Ok(Self {
                owner: Arc::clone(&self.owner),
                snapshot: self.snapshot.to_device(device)?,
            });
        }

        let mut converted = HashMap::new();
        let mut tensors = HashMap::new();
        let mut bindings = self.snapshot.schema.bindings.clone();
        for binding in bindings.values_mut() {
            let source = self
                .snapshot
                .tensor(binding.name())
                .ok_or_else(|| Error::missing_input("tensor of a model state binding"))?;
            let target = if source.data_type().is_float() {
                floating_dtype.unwrap_or_else(|| source.data_type())
            } else {
                source.data_type()
            };
            let tensor = if let Some(tensor) = converted.get(&binding.id) {
                Tensor::clone(tensor)
            } else {
                // Building a program must not read its weights back from a device.
                let tensor = if device.is_meta() {
                    Tensor::zeroed_on(device, target, source.shape().clone())?
                } else {
                    source.cast(target)?.to_device(device)?
                };
                converted.insert(binding.id, tensor.clone());
                tensor
            };
            binding.spec = ValueSpec::new(target, source.shape().clone());
            tensors.insert(binding.name.clone(), tensor);
        }

        let snapshot = StateSnapshot::create(Arc::new(StateSchema { bindings }), &tensors)?;
        Ok(Self {
            owner: Arc::clone(&self.owner),
            snapshot,
        })
    }
}

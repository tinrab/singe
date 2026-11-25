use std::{
    collections::{BTreeMap, HashMap, HashSet},
    sync::Arc,
};

use crate::{
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    graph::{CaptureBindings, Graph, GraphProgram},
    layout::Layout,
    nn::Linear,
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
    row_groups: Vec<Vec<String>>,
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

    pub fn join_rows(&mut self, linears: &[&Linear]) -> Result<()> {
        let weights = || linears.iter().map(|linear| linear.weight());
        if weights()
            .chain(linears.iter().filter_map(|linear| linear.bias()))
            .any(|key| !Arc::ptr_eq(&self.owner, &key.owner))
        {
            return Err(Error::invalid_argument("foreign model state key"));
        }

        self.row_groups
            .push(weights().map(|key| key.name.clone()).collect());
        if let Some(biases) = linears
            .iter()
            .map(|linear| linear.bias().map(|key| key.name.clone()))
            .collect::<Option<Vec<_>>>()
        {
            self.row_groups.push(biases);
        }
        Ok(())
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
            row_groups: Arc::from(self.row_groups),
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

    pub fn join_rows(&mut self, linears: &[&Linear]) -> Result<()> {
        self.builder.join_rows(linears)
    }
}

/// Immutable model state with explicit tied-name groups and publication identity.
#[derive(Debug, Clone)]
pub struct ModelState {
    pub(crate) snapshot: StateSnapshot,
    owner: Arc<()>,
    row_groups: Arc<[Vec<String>]>,
}

impl ModelState {
    pub(crate) fn same_owner(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.owner, &other.owner)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn row_groups(&self) -> &[Vec<String>] {
        &self.row_groups
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
        // Buffers a program adds, such as a draft head in another type, do not decide it.
        let floats = || {
            self.bindings()
                .filter(|(_, tensor)| tensor.data_type().is_float())
        };
        floats()
            .find(|(binding, _)| binding.kind() != StateBindingKind::Buffer)
            .or_else(|| floats().next())
            .map(|(_, tensor)| tensor.data_type())
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
            row_groups: Arc::clone(&self.row_groups),
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
            row_groups: Arc::clone(&self.row_groups),
        })
    }

    // The layers keep their keys, and memory holds each weight once; a layer whose parts are adjacent views of one tensor multiplies by them together (see `joined_rows`).
    // Groups are replaced one at a time, so memory peaks at one group's copy.
    pub fn join_projection_rows(&self) -> Result<Self> {
        let mut state = self.clone();
        for members in &*self.row_groups {
            let Some(parts) = members
                .iter()
                .map(|member| state.named_tensor(member).cloned())
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            if !joinable(&parts) {
                continue;
            }

            let joined = Tensor::concat(&parts, 0)?;
            let mut replacements = HashMap::new();
            let mut start = 0;
            for (member, part) in members.iter().zip(&parts) {
                let &[rows, ..] = part.shape().dimensions() else {
                    return Err(Error::invalid_argument("joined projection rows"));
                };
                replacements.insert(member.clone(), joined.narrow(0, start, rows)?);
                start += rows;
            }

            state = state.replace(&replacements)?;
        }
        Ok(state)
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
                row_groups: Arc::clone(&self.row_groups),
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
            row_groups: Arc::clone(&self.row_groups),
        })
    }
}

fn stacks_by_rows(first: &Tensor, part: &Tensor) -> bool {
    matches!(first.shape().rank(), 1 | 2)
        && part.is_contiguous()
        && part.shape().dimensions().get(1..) == first.shape().dimensions().get(1..)
}

fn joinable(parts: &[Tensor]) -> bool {
    let Some(first) = parts.first() else {
        return false;
    };
    if joined_rows(&parts.iter().collect::<Vec<_>>()).is_some() {
        return false;
    }

    let mut ids = HashSet::new();
    matches!(
        first.data_type(),
        DataType::F16 | DataType::Bf16 | DataType::F32 | DataType::F64
    ) && parts.iter().all(|part| {
        stacks_by_rows(first, part)
            && part.data_type() == first.data_type()
            && part.device() == first.device()
            && ids.insert(part.id())
    })
}

// The tensor `parts`' rows make together, when they are adjacent row views of one storage in order, as `ModelState::join_projection_rows` binds them; `None` otherwise.
pub(crate) fn joined_rows(parts: &[&Tensor]) -> Option<Tensor> {
    let first = *parts.first()?;
    let row = *first.layout().strides().first()?;
    let mut rows = 0;
    for part in parts {
        let adjacent = stacks_by_rows(first, part)
            && Arc::ptr_eq(part.storage(), first.storage())
            && i64::try_from(part.storage_offset()).ok()?
                == i64::try_from(first.storage_offset()).ok()? + rows * row;
        if !adjacent {
            return None;
        }

        rows += *part.shape().dimensions().first()?;
    }

    let mut dimensions = first.shape().dimensions().to_vec();
    dimensions[0] = rows;
    let layout = Layout::contiguous(dimensions)
        .ok()?
        .with_offset(first.storage_offset());
    first.strided_view_unrecorded(layout).ok()
}

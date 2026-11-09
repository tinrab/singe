use std::{
    collections::{BTreeMap, HashMap, HashSet},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use crate::{
    device::Device,
    error::{Error, Result},
    tensor::Tensor,
    value::ValueSpec,
};

static NEXT_STATE_ID: AtomicU64 = AtomicU64::new(1);

/// The logical role of a named binding.
/// Parameters and buffers are replaceable state; only explicit graph constants participate in constant folding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StateBindingKind {
    /// Replaceable learned parameter state.
    Parameter,
    /// Replaceable persistent model buffer.
    Buffer,
    /// Non-overridable graph initializer state.
    Static,
    /// Initializer value that a runtime input can override.
    Default,
}

/// Which non-overridable state may become constants during preparation.
/// Runtime-overridable defaults are never frozen.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum StateFreezePolicy {
    /// Does not freeze additional state bindings during preparation.
    #[default]
    None,
    /// Permits parameter bindings to become constants during preparation.
    Parameters,
    /// Permits every state role except runtime-overridable defaults to become constants.
    AllState,
}

impl StateFreezePolicy {
    pub(crate) fn freezes(self, kind: StateBindingKind) -> bool {
        match self {
            Self::None => false,
            Self::Parameters => kind == StateBindingKind::Parameter,
            Self::AllState => kind != StateBindingKind::Default,
        }
    }
}

/// One program-local state slot, independent of its currently bound allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateBindingSpec {
    pub(crate) id: usize,
    pub(crate) name: String,
    pub(crate) spec: ValueSpec,
    pub(crate) kind: StateBindingKind,
}

impl StateBindingSpec {
    /// Returns the program-local logical binding identity, shared by explicitly tied names.
    #[must_use]
    pub fn id(&self) -> usize {
        self.id
    }

    /// Returns the name under which the binding appears in the schema.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the required logical data type and shape.
    #[must_use]
    pub fn spec(&self) -> &ValueSpec {
        &self.spec
    }

    /// Returns the role controlling replacement and freezing.
    #[must_use]
    pub fn kind(&self) -> StateBindingKind {
        self.kind
    }
}

#[derive(Debug)]
pub(crate) struct StateSchema {
    pub(crate) bindings: BTreeMap<String, StateBindingSpec>,
}

#[derive(Debug)]
struct StateIdentity {
    id: u64,
    next_revision: AtomicU64,
}

/// Identity and revision of one immutable state publication.
/// Sibling revisions are unique even when several callers replace the same original snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StateRevision {
    identity: u64,
    revision: u64,
}

impl StateRevision {
    /// Returns the identity shared by revisions descended from the same state publication.
    #[must_use]
    pub fn identity(self) -> u64 {
        self.identity
    }

    /// Returns this publication's revision number within its state identity.
    #[must_use]
    pub fn revision(self) -> u64 {
        self.revision
    }
}

/// A validated, immutable set of parameter, buffer, and default bindings.
/// Replacements return another snapshot; old executors keep the old snapshot.
#[derive(Debug, Clone)]
pub struct StateSnapshot {
    pub(crate) schema: Arc<StateSchema>,
    identity: Arc<StateIdentity>,
    revision: u64,
    pub(crate) tensors: Arc<HashMap<String, Tensor>>,
}

impl StateSnapshot {
    pub(crate) fn create(
        schema: Arc<StateSchema>,
        tensors: &HashMap<String, Tensor>,
    ) -> Result<Self> {
        let tensors = normalize_bindings(&schema, tensors, true)?;
        let id = next_counter(&NEXT_STATE_ID, "state identity")?;
        Ok(Self {
            schema,
            identity: Arc::new(StateIdentity {
                id,
                next_revision: AtomicU64::new(1),
            }),
            revision: 0,
            tensors: Arc::new(tensors),
        })
    }

    /// Returns this immutable publication's identity and revision.
    #[must_use]
    pub fn revision(&self) -> StateRevision {
        StateRevision {
            identity: self.identity.id,
            revision: self.revision,
        }
    }

    /// Returns the tensor bound to `name`, or `None` when the name is absent.
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// Iterates bound names and tensor descriptors in unspecified order.
    ///
    /// Explicitly tied names each appear, even when they share a logical binding.
    pub fn bindings(&self) -> impl Iterator<Item = (&str, &Tensor)> {
        self.tensors
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
    }

    /// Moves each tensor descriptor once while preserving logical state identity, revision, and every dtype.
    /// Existing snapshots retain their allocations.
    ///
    /// # Errors
    ///
    /// Returns an error if moving any retained tensor to the target device fails.
    pub fn to_device(&self, device: Device) -> Result<Self> {
        if self
            .tensors
            .values()
            .all(|tensor| tensor.device() == device)
        {
            return Ok(self.clone());
        }

        let mut moved = HashMap::new();
        let mut tensors = HashMap::new();
        for (name, tensor) in self.tensors.iter() {
            let converted = if let Some(converted) = moved.get(&tensor.id()) {
                Tensor::clone(converted)
            } else {
                // Building a program must not read its weights back from a device.
                let converted = if device.is_meta() {
                    tensor.to_meta()?
                } else {
                    tensor.to_device(device)?
                };
                moved.insert(tensor.id(), converted.clone());
                converted
            };
            tensors.insert(name.clone(), converted);
        }
        Ok(Self {
            schema: Arc::clone(&self.schema),
            identity: Arc::clone(&self.identity),
            revision: self.revision,
            tensors: Arc::new(tensors),
        })
    }

    /// Validates all replacements before publishing a new revision.
    /// Unknown names, missing bindings, and changed logical metadata are errors.
    ///
    /// An empty replacement map returns a clone without publishing a new revision.
    /// Replacing any name in a tied group updates every name in that group.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown names, incompatible shapes or data types, conflicting values for tied names, failed tensor reads needed to compare ties, or exhausted revision numbering.
    pub fn replace(&self, replacements: &HashMap<String, Tensor>) -> Result<Self> {
        if replacements.is_empty() {
            return Ok(self.clone());
        }

        let replacements = normalize_bindings(&self.schema, replacements, false)?;
        let mut tensors = (*self.tensors).clone();
        tensors.extend(replacements);
        separate_binding_descriptors(&self.schema, &mut tensors);
        let revision = next_counter(&self.identity.next_revision, "state revision")?;
        Ok(Self {
            schema: Arc::clone(&self.schema),
            identity: Arc::clone(&self.identity),
            revision,
            tensors: Arc::new(tensors),
        })
    }
}

pub(crate) fn normalize_bindings(
    schema: &StateSchema,
    tensors: &HashMap<String, Tensor>,
    require_all: bool,
) -> Result<HashMap<String, Tensor>> {
    for name in tensors.keys() {
        if !schema.bindings.contains_key(name) {
            return Err(Error::UnexpectedInput(name.clone()));
        }
    }

    let mut groups: BTreeMap<usize, (&str, &Tensor)> = BTreeMap::new();
    for (name, binding) in &schema.bindings {
        let Some(tensor) = tensors.get(name) else {
            continue;
        };
        if tensor.data_type() != binding.spec.data_type() {
            return Err(Error::data_type_mismatch(
                name.clone(),
                binding.spec.data_type(),
                tensor.data_type(),
            ));
        }

        if tensor.shape() != binding.spec.shape() {
            return Err(Error::shape_mismatch(
                name.clone(),
                binding.spec.shape().dimensions().to_vec(),
                tensor.shape().dimensions().to_vec(),
            ));
        }
        if let Some((other_name, other)) = groups.get(&binding.id) {
            if tensor.id() != other.id() && tensor.to_bytes()? != other.to_bytes()? {
                return Err(Error::invalid_argument(format!(
                    "conflicting tied state bindings `{other_name}` and `{name}`"
                )));
            }
        } else {
            groups.insert(binding.id, (name, tensor));
        }
    }

    let mut normalized = HashMap::new();
    for (name, binding) in &schema.bindings {
        if let Some((_, tensor)) = groups.get(&binding.id) {
            normalized.insert(name.clone(), Tensor::clone(tensor));
        } else if require_all {
            return Err(Error::missing_input("normalize bindings"));
        }
    }

    separate_binding_descriptors(schema, &mut normalized);
    Ok(normalized)
}

/// Each logical slot needs its own descriptor for unambiguous capture.
/// Only schema-declared ties share a descriptor; allocation sharing remains legal.
fn separate_binding_descriptors(schema: &StateSchema, tensors: &mut HashMap<String, Tensor>) {
    let mut groups = HashMap::new();
    let mut identities = HashSet::new();
    for (name, binding) in &schema.bindings {
        let Some(tensor) = tensors.get(name).cloned() else {
            continue;
        };
        let canonical = groups.entry(binding.id).or_insert_with(|| {
            let tensor = if identities.contains(&tensor.id()) {
                tensor.alias_unrecorded()
            } else {
                tensor
            };
            identities.insert(tensor.id());
            tensor
        });
        tensors.insert(name.clone(), canonical.clone());
    }
}

fn next_counter(counter: &AtomicU64, name: &'static str) -> Result<u64> {
    counter
        .try_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map_err(|_| Error::out_of_range(name))
}

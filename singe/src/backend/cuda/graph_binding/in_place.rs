use std::{
    collections::{HashMap, HashSet},
    hash::BuildHasher,
};

use crate::{
    backend::registry::Operation,
    error::{Error, Result},
    graph::ir::{Graph, ValueBindingKind, ValueId},
    tensor::Tensor,
};

// What a scatter writes in place of a fresh buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScatterTarget {
    // A result of an earlier operation, which only this scatter reads.
    Base(ValueId),
    // An input whose old contents nothing else reads.
    Input(ValueId),
}

impl ScatterTarget {
    #[must_use]
    pub const fn value(self) -> ValueId {
        match self {
            Self::Base(value) | Self::Input(value) => value,
        }
    }
}

// Which inputs a scatter may write where they are, the one thing that differs between what plans graphs.
#[derive(Debug, Clone, Copy)]
pub enum WritableInputs<'a> {
    // None, as when only planning what a graph allocates.
    Nothing,
    // Those a compiled graph holds in place.
    Bound(&'a HashMap<ValueId, Tensor>),
    // Every one, as a program's caller holds each buffer, and tuning writes a copy.
    Every,
}

// What every scatter of a graph writes.
// A scatter this does not name writes a fresh buffer, into which its base is first copied.
#[derive(Debug, Clone, Default)]
pub struct InPlaceWrites {
    targets: HashMap<ValueId, ScatterTarget>,
}

impl InPlaceWrites {
    // Finds what each scatter in `order` can write.
    //
    // # Errors
    //
    // Returns an error if a value the scatters name is missing.
    pub fn analyze<S: BuildHasher, T: BuildHasher>(
        ir: &Graph,
        order: &[ValueId],
        users: &HashMap<ValueId, Vec<ValueId>, S>,
        outputs: &HashSet<ValueId, T>,
        inputs: WritableInputs<'_>,
    ) -> Result<Self> {
        let mut targets = HashMap::new();
        for &id in order {
            if let Some(target) = scatter_target(ir, id, users, outputs, inputs)? {
                targets.insert(id, target);
            }
        }
        Ok(Self { targets })
    }

    #[must_use]
    pub fn target(&self, id: ValueId) -> Option<ScatterTarget> {
        self.targets.get(&id).copied()
    }

    // The value `id` writes where it is, if any.
    #[must_use]
    pub fn value(&self, id: ValueId) -> Option<ValueId> {
        self.target(id).map(ScatterTarget::value)
    }
}

fn scatter_target<S: BuildHasher, T: BuildHasher>(
    ir: &Graph,
    id: ValueId,
    users: &HashMap<ValueId, Vec<ValueId>, S>,
    outputs: &HashSet<ValueId, T>,
    inputs: WritableInputs<'_>,
) -> Result<Option<ScatterTarget>> {
    let Some(operation) = ir.operation_for_value(id) else {
        return Ok(None);
    };
    if !matches!(
        operation.opcode(),
        Some(Operation::IndexPut | Operation::StridedScatter)
    ) {
        return Ok(None);
    }

    let Some(&base) = operation.inputs().first() else {
        return Ok(None);
    };
    // A base anything else reads must keep its old contents.
    if users.get(&base).is_none_or(|readers| readers.len() != 1)
        || ir.required_value_data_type(base)? != ir.required_value_data_type(id)?
        || ir.required_value_layout(base)? != ir.required_value_layout(id)?
        || !ir.required_value_layout(id)?.is_contiguous()
    {
        return Ok(None);
    }

    if result_can_be_written(ir, base, outputs)? {
        return Ok(Some(ScatterTarget::Base(base)));
    }

    let writable = match inputs {
        WritableInputs::Nothing => false,
        WritableInputs::Bound(bound) => bound.contains_key(&base),
        WritableInputs::Every => true,
    };
    Ok((writable && input_can_be_written(ir, id, base)?).then_some(ScatterTarget::Input(base)))
}

// Whether `base` is an operation's only result, with storage of its own, which the graph does not return.
fn result_can_be_written<T: BuildHasher>(
    ir: &Graph,
    base: ValueId,
    outputs: &HashSet<ValueId, T>,
) -> Result<bool> {
    let Some(producer) = ir.operation_for_value(base) else {
        return Ok(false);
    };
    Ok(
        ir.required_value_binding_kind(base)? == ValueBindingKind::OperationOutput
            && ir.required_value_alias(base)?.is_none()
            && producer.result_values().count() == 1
            && !outputs.contains(&base),
    )
}

// Whether `base` is an input the graph writes, whose written value is this scatter's result and nothing the caller keeps.
fn input_can_be_written(ir: &Graph, id: ValueId, base: ValueId) -> Result<bool> {
    if !ir.counted_outputs().is_empty() {
        return Ok(false);
    }

    let outputs = ir.output_values_vec();
    let writes = ir.input_writes();
    let visible = outputs
        .len()
        .checked_sub(writes.len())
        .ok_or_else(|| Error::missing_output("scatter input in place"))?;
    let Some(position) = ir
        .input_values()
        .find(|&(value, _)| value == base)
        .and_then(|(_, name)| writes.iter().position(|write| write == name))
    else {
        return Ok(false);
    };
    let written = outputs[visible + position];
    if alias_root(ir, written)? != id
        || ir.required_value_layout(written)? != ir.required_value_layout(base)?
    {
        return Ok(false);
    }

    // A result the caller keeps cannot live in storage the next run overwrites.
    for &output in &outputs[..visible] {
        if alias_root(ir, output)? == id {
            return Ok(false);
        }
    }
    Ok(true)
}

// Returns the value whose storage `value` views.
fn alias_root(ir: &Graph, mut value: ValueId) -> Result<ValueId> {
    while let Some(alias) = ir.required_value_alias(value)? {
        value = alias.source();
    }
    Ok(value)
}

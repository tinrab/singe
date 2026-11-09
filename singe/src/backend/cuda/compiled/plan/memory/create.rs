use std::collections::{HashMap, HashSet};

use crate::{
    backend::cuda::{
        compiled::plan::memory::{
            GraphMemoryPlanner, graph_alias_group_root, union_cuda_graph_alias_group,
        },
        graph_binding::{
            InPlaceWrites, ScatterTarget, WritableInputs, graph_users, operation_dependencies,
        },
    },
    error::{Error, Result},
    graph::ir::{Graph, ValueBindingKind, ValueId},
};

impl GraphMemoryPlanner {
    pub(crate) fn create(ir: &Graph, execution_order: &[ValueId]) -> Result<Self> {
        let mut positions = HashMap::new();
        let mut dependencies = HashMap::new();
        let mut last_use = HashMap::new();
        let mut alias_parent = HashMap::new();
        let mut primary_values = HashMap::new();
        for (index, &id) in execution_order.iter().enumerate() {
            positions.insert(id, index);
            last_use.insert(id, index);
            dependencies.insert(id, operation_dependencies(ir, id)?);
            let operation = ir.required_operation_for_value(id)?;
            let values = operation
                .results()
                .iter()
                .map(|result| result.value())
                .collect::<Vec<_>>();
            let primary_value = values
                .first()
                .copied()
                .ok_or_else(|| Error::missing_output("create create"))?;
            primary_values.insert(id, primary_value);
            for value in values {
                alias_parent.insert(value, value);
            }
        }

        record_alias_groups_and_last_uses(
            ir,
            execution_order,
            &positions,
            &dependencies,
            &primary_values,
            &mut last_use,
            &mut alias_parent,
        )?;

        let group_last_use =
            graph_memory_group_last_uses(&last_use, &primary_values, &alias_parent);
        let (release_use, release_group) = graph_memory_release_maps(
            execution_order,
            &primary_values,
            &alias_parent,
            &group_last_use,
        );
        let (
            protected_groups,
            protected_input_groups,
            protected_constant_groups,
            protected_output_groups,
        ) = graph_memory_protected_groups(ir, execution_order, &primary_values, &alias_parent)?;

        Ok(Self {
            dependencies,
            release_use,
            release_group,
            protected_groups,
            protected_input_groups,
            protected_constant_groups,
            protected_output_groups,
            released_groups: HashSet::new(),
            group_readers: HashMap::new(),
            free: HashMap::new(),
            reused_buffers: 0,
            allocated_bytes: 0,
            reused_buffer_bytes: 0,
            arena: None,
            arena_taken: Vec::new(),
        })
    }
}

pub fn graph_memory_release_maps(
    execution_order: &[ValueId],
    primary_values: &HashMap<ValueId, ValueId>,
    alias_parent: &HashMap<ValueId, ValueId>,
    group_last_use: &HashMap<ValueId, usize>,
) -> (HashMap<ValueId, usize>, HashMap<ValueId, ValueId>) {
    let mut release_use = HashMap::new();
    let mut release_group = HashMap::new();
    for &id in execution_order {
        let value = primary_values[&id];
        let group = graph_alias_group_root(alias_parent, value);
        release_group.insert(id, group);
        if let Some(&last_use) = group_last_use.get(&group) {
            release_use.insert(id, last_use);
        }
    }
    (release_use, release_group)
}

enum GraphMemoryValueBinding {
    Input,
    Constant,
    OperationOutput,
}

pub fn graph_memory_protected_groups(
    ir: &Graph,
    execution_order: &[ValueId],
    primary_values: &HashMap<ValueId, ValueId>,
    alias_parent: &HashMap<ValueId, ValueId>,
) -> Result<GraphProtectedGroups> {
    let mut protected_groups = HashSet::new();
    let mut protected_input_groups = HashSet::new();
    let mut protected_constant_groups = HashSet::new();
    let mut protected_output_groups = HashSet::new();
    for &id in execution_order {
        let value = primary_values[&id];
        match graph_memory_value_binding(ir.required_value_binding_kind(value)?) {
            GraphMemoryValueBinding::Input => {
                let group = graph_alias_group_root(alias_parent, value);
                protected_groups.insert(group);
                protected_input_groups.insert(group);
            }
            GraphMemoryValueBinding::Constant => {
                let group = graph_alias_group_root(alias_parent, value);
                protected_groups.insert(group);
                protected_constant_groups.insert(group);
            }
            GraphMemoryValueBinding::OperationOutput => {}
        }
    }

    ir.for_each_output_value(|output| {
        let group = graph_alias_group_root(alias_parent, output);
        protected_groups.insert(group);
        protected_output_groups.insert(group);
    });
    Ok((
        protected_groups,
        protected_input_groups,
        protected_constant_groups,
        protected_output_groups,
    ))
}

fn graph_memory_value_binding(kind: ValueBindingKind) -> GraphMemoryValueBinding {
    match kind {
        ValueBindingKind::RuntimeInput
        | ValueBindingKind::InitializerInput
        | ValueBindingKind::StaticInput => GraphMemoryValueBinding::Input,
        ValueBindingKind::Constant => GraphMemoryValueBinding::Constant,
        ValueBindingKind::OperationOutput => GraphMemoryValueBinding::OperationOutput,
    }
}

type GraphProtectedGroups = (
    HashSet<ValueId>,
    HashSet<ValueId>,
    HashSet<ValueId>,
    HashSet<ValueId>,
);

pub fn record_alias_groups_and_last_uses(
    ir: &Graph,
    execution_order: &[ValueId],
    positions: &HashMap<ValueId, usize>,
    dependencies: &HashMap<ValueId, Vec<ValueId>>,
    primary_values: &HashMap<ValueId, ValueId>,
    last_use: &mut HashMap<ValueId, usize>,
    alias_parent: &mut HashMap<ValueId, ValueId>,
) -> Result<()> {
    let users = graph_users(ir, execution_order)?;
    let graph_outputs = ir.output_value_set();
    let writes = InPlaceWrites::analyze(
        ir,
        execution_order,
        &users,
        &graph_outputs,
        WritableInputs::Nothing,
    )?;
    for (index, &id) in execution_order.iter().enumerate() {
        let output = primary_values[&id];
        if let Some(alias) = ir.required_value_alias(output)? {
            let input = alias.source();
            if alias_parent.contains_key(&input) {
                union_cuda_graph_alias_group(alias_parent, output, input);
            }
        }

        // A scatter that writes its base in place holds that buffer until its own last reader.
        if let Some(ScatterTarget::Base(base)) = writes.target(id)
            && alias_parent.contains_key(&base)
        {
            union_cuda_graph_alias_group(alias_parent, output, base);
        }
        for &dependency in dependencies.get(&id).into_iter().flatten() {
            if positions.contains_key(&dependency) {
                let entry = last_use.entry(dependency).or_insert(index);
                *entry = (*entry).max(index);
            }
        }
    }
    Ok(())
}

pub fn graph_memory_group_last_uses(
    last_use: &HashMap<ValueId, usize>,
    primary_values: &HashMap<ValueId, ValueId>,
    alias_parent: &HashMap<ValueId, ValueId>,
) -> HashMap<ValueId, usize> {
    let mut group_last_use = HashMap::new();
    for (&id, &last_use) in last_use {
        let value = primary_values[&id];
        let group = graph_alias_group_root(alias_parent, value);
        let entry = group_last_use.entry(group).or_insert(last_use);
        *entry = (*entry).max(last_use);
    }
    group_last_use
}

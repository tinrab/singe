//! Fusion planning: which operations of a compiled graph share one generated kernel, decided over the whole graph before lowering.
//!
//! Every node starts in a group of its own, except the dequantization chains that a generated matmul reads as it loads its weight, which start in the matmul's group.
//! Merging the group of a value's producer with the group of a reader saves the bytes of the values the kernel no longer writes or reads, as Inductor's scheduler scores fusions (`score_fusion_memory` in `torch/_inductor/scheduler.py`).
//! Each round merges the best-scoring legal pairs, each group at most once, until a round merges nothing, as `fuse_nodes_once` does.
//! A merge is legal when the union is a kernel some lowering writes ([`GroupKind`]) and no path leaves the union and comes back to it, which would make the kernel wait for itself.
//! Lowering then visits the groups in a topological order, each group's members together.

mod module;
mod weight;

use std::{
    cmp::Reverse,
    collections::{BTreeSet, BinaryHeap, HashMap, HashSet, hash_map::Entry},
    fmt,
    hash::BuildHasher,
};

use self::weight::{dequantized_weights, generated};
pub use self::{
    module::{ModuleKind, RecognizedModule, RotaryAttention, RotaryCacheWrite, Rotation},
    weight::DequantizedWeight,
};
use crate::{
    backend::{
        cuda::{
            compiled::plan::lowering::{CudaGraphProvider, LoweringPlan},
            tuning::{TunedChoice, TunedWay, TuningRecords, group_key},
        },
        kernel_lowering::{is_fusible, is_row_operation},
        registry::{Operation, OperationAttributes},
    },
    error::{Error, Result},
    graph::ir::{Graph, ValueId},
    shape::normalize_dim_usize,
};

/// The most operations one group holds, so its kernel stays one the emitters write well.
const MAX_MEMBERS: usize = 64;

/// What kernel a group's lowering writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupKind {
    /// Element-wise operations of one data type and shape, which store each value something outside the group reads.
    Pointwise,
    /// A normalization, softmax, or reduction that computes the group's element-wise members as it loads its input; `stored` when something else also reads that input, so the kernel stores it too.
    Row {
        row: ValueId,
        stored: bool,
    },
    /// A concatenation whose inputs the group's element-wise members compute, each stored into its slice along `axis`.
    Concat {
        concat: ValueId,
        axis: usize,
    },
    /// A generated matmul, with the element-wise members before it computing its left operand as it loads it, the dequantization it computes as it loads its weight, and the element-wise members after it applied to its product.
    Contraction {
        matmul: ValueId,
        weight: Option<DequantizedWeight>,
    },
    // A subgraph a dedicated kernel computes whole, which a recognizer found.
    Module(ModuleKind),
}

/// Operations that share one kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionGroup {
    pub kind: GroupKind,
    /// The operations, in the order lowering visits them.
    pub members: Vec<ValueId>,
    /// The values the kernel stores because something outside the group reads them, or the graph returns them.
    pub outputs: Vec<ValueId>,
    /// The values the kernel reads that the group does not compute.
    pub inputs: Vec<ValueId>,
    /// The bytes of the values the group keeps out of memory, written once and read once each.
    pub saved_bytes: u64,
}

/// The fusion groups of a compiled graph and the order lowering visits its operations in.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FusionPlan {
    groups: Vec<FusionGroup>,
    /// Every operation, each group's members together, after everything they read.
    order: Vec<ValueId>,
    /// The group whose first member each operation is.
    leaders: HashMap<ValueId, usize>,
    /// The operation and data type names of every planned member, for printing.
    names: HashMap<ValueId, String>,
}

impl FusionPlan {
    /// Returns the groups of more than one operation.
    #[must_use]
    pub fn groups(&self) -> &[FusionGroup] {
        &self.groups
    }

    /// Returns every operation in the order lowering visits them.
    #[must_use]
    pub fn order(&self) -> &[ValueId] {
        &self.order
    }

    /// Returns the group whose first member is `id`.
    #[must_use]
    pub fn led_by(&self, id: ValueId) -> Option<&FusionGroup> {
        self.leaders.get(&id).map(|&index| &self.groups[index])
    }

    /// Returns whether `id` is a member of some group.
    #[must_use]
    pub fn is_planned(&self, id: ValueId) -> bool {
        self.groups.iter().any(|group| group.members.contains(&id))
    }
}

impl fmt::Display for FusionPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let list = |values: &[ValueId]| {
            values
                .iter()
                .map(|value| format!("%{}", value.index()))
                .collect::<Vec<_>>()
                .join(", ")
        };
        for (index, group) in self.groups.iter().enumerate() {
            let kind = match &group.kind {
                GroupKind::Pointwise => "pointwise".to_owned(),
                GroupKind::Row { row, stored } => format!(
                    "row %{}{}",
                    row.index(),
                    if *stored { ", storing its input" } else { "" }
                ),
                GroupKind::Concat { concat, axis } => {
                    format!("concat %{} along {axis}", concat.index())
                }
                GroupKind::Contraction { matmul, weight } => format!(
                    "contraction %{}{}",
                    matmul.index(),
                    if weight.is_some() {
                        " with its dequantized weight"
                    } else {
                        ""
                    },
                ),
                GroupKind::Module(kind) => format!("module: {}", kind.name()),
            };
            writeln!(
                f,
                "group {index}: {kind}, saving {} bytes",
                group.saved_bytes
            )?;
            for member in &group.members {
                writeln!(
                    f,
                    "  %{} {}",
                    member.index(),
                    self.names.get(member).map_or("", String::as_str)
                )?;
            }

            writeln!(f, "  inputs {}", list(&group.inputs))?;
            writeln!(f, "  outputs {}", list(&group.outputs))?;
        }
        Ok(())
    }
}

/// The graph a plan groups and what it knows about its values.
struct Planner<'a, S, T> {
    ir: &'a Graph,
    users: &'a HashMap<ValueId, Vec<ValueId>, S>,
    graph_outputs: &'a HashSet<ValueId, T>,
    lowering_plan: &'a LoweringPlan,
    position: HashMap<ValueId, usize>,
    /// The matmuls that run as generated kernels.
    generated: HashSet<ValueId>,
    /// The dequantized weight each generated matmul reads, by matmul.
    weights: HashMap<ValueId, DequantizedWeight>,
    /// The matmul whose weight chain each dequantization, cast, or transpose belongs to.
    weight_members: HashMap<ValueId, ValueId>,
    // Nodes a tuning record chose to run outside any group.
    separate: HashSet<ValueId>,
    // The modules the recognizer found, each already one group that nothing joins.
    modules: Vec<RecognizedModule>,
}

/// Plans the fusion groups of `ir`, whose operations run in `execution_order`, read by `users`, with `graph_outputs` returned.
///
/// # Errors
///
/// Returns an error if the graph's metadata is missing.
pub fn plan_fusion<S: BuildHasher, T: BuildHasher>(
    ir: &Graph,
    execution_order: &[ValueId],
    users: &HashMap<ValueId, Vec<ValueId>, S>,
    graph_outputs: &HashSet<ValueId, T>,
    lowering_plan: &LoweringPlan,
    library_matmuls: Option<&HashSet<ValueId>>,
    records: Option<&TuningRecords>,
) -> Result<FusionPlan> {
    let weights = dequantized_weights(
        ir,
        execution_order,
        users,
        graph_outputs,
        lowering_plan,
        library_matmuls,
    )?;
    let mut generated_matmuls = weights.keys().copied().collect::<HashSet<_>>();
    for &id in execution_order {
        let operation = ir.required_operation_for_value(id)?;
        if operation.opcode() == Some(Operation::MatMul)
            && !generated_matmuls.contains(&id)
            && generated(ir, id, lowering_plan, library_matmuls)?
        {
            generated_matmuls.insert(id);
        }
    }

    let weight_members = weights
        .iter()
        .flat_map(|(&matmul, weight)| weight.members().iter().map(move |&member| (member, matmul)))
        .collect();
    let mut planner = Planner {
        ir,
        users,
        graph_outputs,
        lowering_plan,
        position: execution_order
            .iter()
            .enumerate()
            .map(|(index, &id)| (id, index))
            .collect(),
        generated: generated_matmuls,
        weights,
        weight_members,
        separate: HashSet::new(),
        modules: Vec::new(),
    };
    planner.modules = planner.recognize(execution_order, records)?;
    let plan = planner.plan(execution_order)?;
    let Some(records) = records.filter(|records| !records.is_empty()) else {
        return Ok(plan);
    };
    // Members of a group a record chose to run separately take no part in any group.
    for group in plan.groups() {
        let key = group_key(ir, &group.members)?;
        if records.get(&key) == Some(TunedChoice::Way(TunedWay::Separate)) {
            records.note_used(&key);
            planner.separate.extend(group.members.iter().copied());
        }
    }

    if planner.separate.is_empty() {
        return Ok(plan);
    }

    planner
        .weights
        .retain(|matmul, _| !planner.separate.contains(matmul));
    planner.plan(execution_order)
}

impl<S: BuildHasher, T: BuildHasher> Planner<'_, S, T> {
    fn plan(&self, execution_order: &[ValueId]) -> Result<FusionPlan> {
        // Groups by index; a merged group's slot is emptied.
        let mut groups: Vec<Option<BTreeSet<ValueId>>> = Vec::new();
        let mut group_of = HashMap::new();
        for (&matmul, weight) in &self.weights {
            let members = weight
                .members()
                .iter()
                .copied()
                .chain([matmul])
                .collect::<BTreeSet<_>>();
            for &member in &members {
                group_of.insert(member, groups.len());
            }

            groups.push(Some(members));
        }

        for module in &self.modules {
            for &member in &module.members {
                group_of.insert(member, groups.len());
            }

            groups.push(Some(module.members.clone()));
        }

        for &id in execution_order {
            if let Entry::Vacant(entry) = group_of.entry(id) {
                entry.insert(groups.len());
                groups.push(Some(BTreeSet::from([id])));
            }

            // Lowering visits an operation by its first result, and whatever reads its others waits for the same group.
            let group = group_of[&id];
            for result in self.ir.required_operation_for_value(id)?.result_values() {
                group_of.entry(result).or_insert(group);
            }
        }

        loop {
            let mut pairs = Vec::new();
            let mut seen = HashSet::new();
            for &reader in execution_order {
                for &input in self.ir.required_operation_for_value(reader)?.inputs() {
                    // A reader that reads a value through views pairs with the value's producer, and the views join them.
                    let (producer, views) = self.through_views(input)?;
                    let mut indices = vec![group_of[&producer], group_of[&reader]];
                    indices.extend(views.iter().map(|view| group_of[view]));
                    indices.sort_unstable();
                    indices.dedup();
                    if indices.len() < 2 || !seen.insert(indices.clone()) {
                        continue;
                    }

                    if indices.iter().any(|&index| groups[index].is_none()) {
                        continue;
                    }

                    let mut union = BTreeSet::new();
                    for &index in &indices {
                        union.extend(groups[index].iter().flatten().copied());
                    }

                    // Every view of a member that a member reads, directly or through other views, joins too, with its group, so the union never stores a value only its own views read.
                    loop {
                        let views = union
                            .iter()
                            .flat_map(|&member| self.readers(member))
                            .copied()
                            .filter(|reader| !union.contains(reader))
                            .filter(|&reader| self.view(reader).unwrap_or(false))
                            .filter(|&view| self.view_returns(view, &union))
                            .collect::<Vec<_>>();
                        if views.is_empty() {
                            break;
                        }
                        for view in views {
                            let index = group_of[&view];
                            if !indices.contains(&index) {
                                indices.push(index);
                            }

                            union.extend(groups[index].iter().flatten().copied());
                        }
                    }

                    indices.sort_unstable();
                    if union.iter().any(|member| {
                        self.separate.contains(member)
                            || self
                                .modules
                                .iter()
                                .any(|module| module.members.contains(member))
                    }) {
                        continue;
                    }

                    if union.len() > MAX_MEMBERS
                        || self.classify(&union)?.is_none()
                        || self.leaves_and_returns(&union)
                    {
                        continue;
                    }

                    let parts = indices
                        .iter()
                        .filter_map(|&index| groups[index].as_ref())
                        .collect::<Vec<_>>();
                    let saved = self.saved_bytes(&parts, &union)?;
                    if saved > 0 {
                        let first = union.iter().map(|id| self.position[id]).min();
                        pairs.push((Reverse(saved), first, indices));
                    }
                }
            }

            pairs.sort_unstable();
            let mut touched = HashSet::new();
            for (_, _, indices) in pairs {
                if indices.iter().any(|index| touched.contains(index)) {
                    continue;
                }

                touched.extend(indices.iter().copied());
                let (&first, rest) = indices
                    .split_first()
                    .ok_or_else(|| Error::missing_node("fusion plan"))?;
                for &index in rest {
                    let members = groups[index]
                        .take()
                        .ok_or_else(|| Error::missing_node("fusion plan"))?;
                    for &member in &members {
                        group_of.insert(member, first);
                    }

                    groups[first]
                        .as_mut()
                        .ok_or_else(|| Error::missing_node("fusion plan"))?
                        .extend(members);
                }
            }
            if touched.is_empty() {
                break;
            }
        }
        self.finish(execution_order, &groups, &group_of)
    }

    /// Returns the plan of the merged groups: each group of more than one operation with its kind, and an order in which each group's members come together.
    fn finish(
        &self,
        execution_order: &[ValueId],
        groups: &[Option<BTreeSet<ValueId>>],
        group_of: &HashMap<ValueId, usize>,
    ) -> Result<FusionPlan> {
        let order = self.group_order(execution_order, groups, group_of);
        let mut plan = FusionPlan {
            order,
            ..FusionPlan::default()
        };
        // Groups in the order lowering reaches their first member.
        let mut emitted = HashSet::new();
        for &id in &plan.order.clone() {
            let index = group_of[&id];
            let Some(members) = &groups[index] else {
                continue;
            };
            if members.len() < 2 || !emitted.insert(index) {
                continue;
            }

            let kind = self
                .classify(members)?
                .ok_or_else(|| Error::missing_node("finish"))?;
            let mut ordered = members.iter().copied().collect::<Vec<_>>();
            ordered.sort_by_key(|member| plan.order.iter().position(|id| id == member));
            // Lowering splits a row group at its row: what computes the row's input comes before it, and the epilogue with its side chains after.
            if let &GroupKind::Row { row, .. } = &kind {
                let input = self.ir.required_operation_for_value(row)?.inputs()[0];
                let computes_input = self.computes(input, members)?;
                let (mut front, back): (Vec<_>, Vec<_>) = ordered
                    .iter()
                    .partition(|&&member| member == row || computes_input.contains(&member));
                front.extend(back);
                // A group's members are consecutive in the order, and lowering builds the group at the first one it reaches.
                let start = plan.order.iter().position(|id| members.contains(id));
                if let Some(start) = start
                    && plan.order[start..]
                        .iter()
                        .take(members.len())
                        .all(|id| members.contains(id))
                {
                    plan.order
                        .splice(start..start + members.len(), front.iter().copied());
                    ordered = front;
                }
            }

            let outputs = ordered
                .iter()
                .copied()
                .filter(|&member| self.read_outside(member, members))
                .collect();
            let mut inputs = Vec::new();
            for &member in &ordered {
                for &input in self.ir.required_operation_for_value(member)?.inputs() {
                    if !members.contains(&input) && !inputs.contains(&input) {
                        inputs.push(input);
                    }
                }
            }

            // A value a member reads from another is never read from memory, and never written either unless something outside reads it.
            let mut saved_bytes = 0;
            for &member in &ordered {
                if self
                    .readers(member)
                    .iter()
                    .any(|reader| members.contains(reader))
                {
                    let bytes = self.bytes(member)?;
                    saved_bytes += if self.read_outside(member, members) {
                        bytes
                    } else {
                        2 * bytes
                    };
                }
            }

            for &member in &ordered {
                let operation = self.ir.required_operation_for_value(member)?;
                let name = operation
                    .opcode()
                    .map_or_else(|| "operation".to_owned(), |opcode| opcode.name().to_owned());
                let data_type = self.ir.required_value_data_type(member)?;
                let shape = self.ir.required_value_shape(member)?;
                plan.names.insert(
                    member,
                    format!("{name} {data_type:?} {:?}", shape.dimensions()),
                );
            }

            plan.leaders.insert(ordered[0], plan.groups.len());
            plan.groups.push(FusionGroup {
                kind,
                members: ordered,
                outputs,
                inputs,
                saved_bytes,
            });
        }
        Ok(plan)
    }

    /// Returns every operation of `execution_order`, grouped: a group comes once all it reads is computed, the group with the earliest first member first.
    fn group_order(
        &self,
        execution_order: &[ValueId],
        groups: &[Option<BTreeSet<ValueId>>],
        group_of: &HashMap<ValueId, usize>,
    ) -> Vec<ValueId> {
        let live = (0..groups.len())
            .filter(|&index| groups[index].is_some())
            .collect::<Vec<_>>();
        let mut waiting = HashMap::<usize, usize>::new();
        let mut dependents = HashMap::<usize, HashSet<usize>>::new();
        for &index in &live {
            let mut needs = HashSet::new();
            for member in groups[index].iter().flatten() {
                let Ok(operation) = self.ir.required_operation_for_value(*member) else {
                    continue;
                };
                for input in operation.inputs() {
                    if let Some(&producer) = group_of.get(input)
                        && producer != index
                    {
                        needs.insert(producer);
                    }
                }
            }

            waiting.insert(index, needs.len());
            for producer in needs {
                dependents.entry(producer).or_default().insert(index);
            }
        }

        let first = |index: usize| {
            groups[index]
                .iter()
                .flatten()
                .map(|id| self.position[id])
                .min()
                .unwrap_or(usize::MAX)
        };
        let mut ready = live
            .iter()
            .copied()
            .filter(|index| waiting[index] == 0)
            .map(|index| Reverse((first(index), index)))
            .collect::<BinaryHeap<_>>();
        let mut order = Vec::with_capacity(execution_order.len());
        while let Some(Reverse((_, index))) = ready.pop() {
            let mut members = groups[index].iter().flatten().copied().collect::<Vec<_>>();
            members.sort_by_key(|id| self.position[id]);
            order.extend(members);
            for &dependent in dependents.get(&index).into_iter().flatten() {
                let count = waiting.get_mut(&dependent).expect("every live group waits");
                *count -= 1;
                if *count == 0 {
                    ready.push(Reverse((first(dependent), dependent)));
                }
            }
        }
        order
    }

    /// Returns the value `value` is a view of through slices and transposes, with those views, or `value` itself with none.
    fn through_views(&self, mut value: ValueId) -> Result<(ValueId, Vec<ValueId>)> {
        let mut views = Vec::new();
        while self.view(value)? && !self.graph_outputs.contains(&value) {
            views.push(value);
            value = self.ir.required_operation_for_value(value)?.inputs()[0];
        }
        Ok((value, views))
    }

    /// Returns whether a member of `union` reads view `view`, directly or through other views.
    fn view_returns(&self, view: ValueId, union: &BTreeSet<ValueId>) -> bool {
        self.readers(view).iter().any(|&reader| {
            union.contains(&reader)
                || (self.view(reader).unwrap_or(false) && self.view_returns(reader, union))
        })
    }

    /// Returns whether `id` is a slice with unit step or a transpose, which a row epilogue reads as an index map instead of memory.
    fn view(&self, id: ValueId) -> Result<bool> {
        let operation = self.ir.required_operation_for_value(id)?;
        Ok(match (operation.opcode(), operation.attributes()) {
            (Some(Operation::Slice), OperationAttributes::Slice(slice)) => slice.step == 1,
            (Some(Operation::Transpose), OperationAttributes::Transpose(_)) => true,
            _ => false,
        })
    }

    fn readers(&self, id: ValueId) -> &[ValueId] {
        self.users.get(&id).map_or(&[][..], Vec::as_slice)
    }

    /// Returns whether something outside `members` reads `id`, or the graph returns it.
    fn read_outside(&self, id: ValueId, members: &BTreeSet<ValueId>) -> bool {
        self.graph_outputs.contains(&id)
            || self
                .readers(id)
                .iter()
                .any(|reader| !members.contains(reader))
    }

    /// Returns the bytes of `id`'s value, nothing for a view, which shares its source's memory.
    fn bytes(&self, id: ValueId) -> Result<u64> {
        if self.view(id)? {
            return Ok(0);
        }

        let elements = self.ir.required_value_shape(id)?.element_count();
        let size = self.ir.required_value_data_type(id)?.size_bytes();
        Ok(u64::try_from(elements.saturating_mul(size)).unwrap_or(u64::MAX))
    }

    /// Returns the bytes merging `parts` into `union` keeps out of memory: a value one part computes and another reads is no longer read, and no longer written either when nothing outside the union reads it.
    fn saved_bytes(&self, parts: &[&BTreeSet<ValueId>], union: &BTreeSet<ValueId>) -> Result<u64> {
        let mut saved = 0;
        for (index, part) in parts.iter().enumerate() {
            for &value in *part {
                let read_by_another = self.readers(value).iter().any(|reader| {
                    parts
                        .iter()
                        .enumerate()
                        .any(|(other, part)| other != index && part.contains(reader))
                });
                if read_by_another {
                    let bytes = self.bytes(value)?;
                    saved += if self.read_outside(value, union) {
                        bytes
                    } else {
                        2 * bytes
                    };
                }
            }
        }
        Ok(saved)
    }

    /// Returns whether a path from a member through operations outside `members` reaches another member.
    fn leaves_and_returns(&self, members: &BTreeSet<ValueId>) -> bool {
        let mut visited = HashSet::new();
        let mut pending = members
            .iter()
            .flat_map(|&member| self.readers(member))
            .copied()
            .filter(|reader| !members.contains(reader))
            .collect::<Vec<_>>();
        while let Some(id) = pending.pop() {
            if !visited.insert(id) {
                continue;
            }
            for &reader in self.readers(id) {
                if members.contains(&reader) {
                    return true;
                }

                pending.push(reader);
            }
        }
        false
    }

    /// Returns the kernel a group of `members` is, or `None` when no lowering writes one.
    fn classify(&self, members: &BTreeSet<ValueId>) -> Result<Option<GroupKind>> {
        if let Some(module) = self
            .modules
            .iter()
            .find(|module| module.members == *members)
        {
            return Ok(Some(GroupKind::Module(module.kind.clone())));
        }

        let ir = self.ir;
        let mut pointwise = Vec::new();
        let (mut rows, mut concats, mut matmuls, mut weight_chain, mut views) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for &id in members {
            let operation = ir.required_operation_for_value(id)?;
            let Some(opcode) = operation.opcode() else {
                return Ok(None);
            };
            let fusible = self
                .lowering_plan
                .should_try_provider(id, CudaGraphProvider::Fusion);
            if self.weight_members.contains_key(&id) {
                weight_chain.push(id);
            } else if self.generated.contains(&id) {
                matmuls.push(id);
            } else if self.view(id)? {
                views.push(id);
            } else if fusible && is_row_operation(opcode, operation.attributes()) {
                rows.push(id);
            } else if fusible && opcode == Operation::Concat {
                concats.push(id);
            } else if fusible && is_fusible(opcode) {
                pointwise.push(id);
            } else {
                return Ok(None);
            }
        }
        Ok(
            match (rows.as_slice(), concats.as_slice(), matmuls.as_slice()) {
                (&[row], _, []) if weight_chain.is_empty() => {
                    self.row(row, &pointwise, &views, &concats, members)?
                }
                _ if !views.is_empty() => None,
                ([], [], []) if weight_chain.is_empty() => self.pointwise(&pointwise, members)?,
                ([], &[concat], []) if weight_chain.is_empty() => {
                    self.concat(concat, &pointwise, members)?
                }
                ([], [], &[matmul]) => {
                    self.contraction(matmul, &weight_chain, &pointwise, members)?
                }
                _ => None,
            },
        )
    }

    /// Returns whether `values` all have one data type and shape.
    fn uniform(&self, values: &[ValueId]) -> Result<bool> {
        let Some(&first) = values.first() else {
            return Ok(true);
        };
        let (data_type, shape) = (
            self.ir.required_value_data_type(first)?,
            self.ir.required_value_shape(first)?,
        );
        for &value in &values[1..] {
            if self.ir.required_value_data_type(value)? != data_type
                || self.ir.required_value_shape(value)? != shape
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn same_shape(&self, values: &[ValueId]) -> Result<bool> {
        let Some(&first) = values.first() else {
            return Ok(true);
        };
        let shape = self.ir.required_value_shape(first)?;
        for &value in &values[1..] {
            if self.ir.required_value_shape(value)? != shape {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn pointwise(
        &self,
        pointwise: &[ValueId],
        members: &BTreeSet<ValueId>,
    ) -> Result<Option<GroupKind>> {
        let outputs = pointwise
            .iter()
            .filter(|&&member| self.read_outside(member, members))
            .count();
        // Each step of a point-wise kernel keeps its own data type, so only the shape must agree.
        Ok((self.same_shape(pointwise)? && outputs >= 1).then_some(GroupKind::Pointwise))
    }

    /// A row operation with a prologue of element-wise members that compute its input, and an epilogue of members that read its output, of which there is at least one.
    fn row(
        &self,
        row: ValueId,
        pointwise: &[ValueId],
        views: &[ValueId],
        concats: &[ValueId],
        members: &BTreeSet<ValueId>,
    ) -> Result<Option<GroupKind>> {
        // The epilogue is what reads the row's output, directly or through other members.
        let mut epilogue = BTreeSet::new();
        let mut pending = vec![row];
        while let Some(id) = pending.pop() {
            for &reader in self.readers(id) {
                if members.contains(&reader) && epilogue.insert(reader) {
                    pending.push(reader);
                }
            }
        }

        if views.iter().chain(concats).any(|id| !epilogue.contains(id)) {
            return Ok(None);
        }

        let operation = self.ir.required_operation_for_value(row)?;
        let [input, parameters @ ..] = operation.inputs() else {
            return Ok(None);
        };
        // The prologue computes the row's input; the other members that do not read the row's output are side chains, such as `silu(gate)` in `rms_norm(x) * silu(gate)`, which only the epilogue reads.
        let computes_input = self.computes(*input, members)?;
        let (prologue_members, side): (Vec<_>, Vec<_>) = pointwise
            .iter()
            .copied()
            .filter(|id| !epilogue.contains(id))
            .partition(|id| computes_input.contains(id));
        if !side.is_empty() {
            if epilogue.is_empty() || parameters.iter().any(|parameter| side.contains(parameter)) {
                return Ok(None);
            }

            epilogue.extend(side);
        }

        if !epilogue.is_empty() && !self.epilogue(row, &epilogue, members)? {
            return Ok(None);
        }

        // The epilogue reads the row's output and values from outside the kernel, never the prologue's values, which the kernel holds only at the row's own points.
        for &id in &epilogue {
            let inputs = self.ir.required_operation_for_value(id)?.inputs();
            if inputs.iter().any(|input| prologue_members.contains(input)) {
                return Ok(None);
            }
        }

        if prologue_members.is_empty() {
            let stored = false;
            return Ok((!epilogue.is_empty()).then_some(GroupKind::Row { row, stored }));
        }

        let pointwise = prologue_members.as_slice();
        let prologue = pointwise.iter().copied().collect::<BTreeSet<_>>();
        if !self.uniform(pointwise)?
            || parameters
                .iter()
                .any(|parameter| prologue.contains(parameter))
        {
            return Ok(None);
        }

        // The prologue's one value read outside it is the row's input.
        let outputs = pointwise
            .iter()
            .copied()
            .filter(|&member| self.read_outside(member, &prologue))
            .collect::<Vec<_>>();
        if outputs != [*input] {
            return Ok(None);
        }

        let stored = self.graph_outputs.contains(input)
            || self
                .readers(*input)
                .iter()
                .any(|reader| *reader != row && !prologue.contains(reader));
        Ok(Some(GroupKind::Row { row, stored }))
    }

    // The members `value` is computed from, itself included when it is a member.
    fn computes(&self, value: ValueId, members: &BTreeSet<ValueId>) -> Result<BTreeSet<ValueId>> {
        let mut computed = BTreeSet::new();
        let mut pending = vec![value];
        while let Some(id) = pending.pop() {
            if members.contains(&id) && computed.insert(id) {
                pending.extend_from_slice(self.ir.required_operation_for_value(id)?.inputs());
            }
        }
        Ok(computed)
    }

    /// Returns whether `epilogue` is work a normalization or softmax `row` stores in place of its output: slices along the row's axis, transposes of the axes before it, concatenations along it of epilogue values, and element-wise operations, all of the row's data type and rank, of which one, spanning the whole row, is read outside `members`.
    fn epilogue(
        &self,
        row: ValueId,
        epilogue: &BTreeSet<ValueId>,
        members: &BTreeSet<ValueId>,
    ) -> Result<bool> {
        let ir = self.ir;
        let operation = ir.required_operation_for_value(row)?;
        let shape = ir.required_value_shape(row)?;
        let data_type = ir.required_value_data_type(row)?;
        let rank = shape.rank();
        let last_axis = shape.dimensions().len().saturating_sub(1);
        // The row runs along the last axis: a normalization of one axis, or a softmax along the last.
        let along_last = match (operation.opcode(), operation.attributes()) {
            (_, OperationAttributes::LayerNormalization(normalization)) => {
                normalization.normalized_shape.len() == 1
            }
            (
                Some(Operation::Softmax | Operation::LogSoftmax),
                &OperationAttributes::Dimension(dim),
            ) => rank > 0 && normalize_dim_usize(dim, rank)? == last_axis,
            _ => false,
        };
        // A reduction's epilogue is element-wise work on the reduced value, which the kernel computes after it stores nothing else.
        // Members of the reduced shape run once per reduced point; when the reduction keeps its axes, members of its input's shape read the reduced value broadcast along them, as `x / sqrt(mean(x², 1) + ε)` divides each point by its column's root, and the kernel stores the one read outside along the reduced axes.
        if let Some(reduction) = operation.attributes().reduction()
            && !self.read_outside(row, members)
        {
            let [input] = operation.inputs() else {
                return Ok(false);
            };
            let input_shape = ir.required_value_shape(*input)?;
            let mut outputs = Vec::new();
            for &id in epilogue {
                let member = ir.required_operation_for_value(id)?;
                let member_shape = ir.required_value_shape(id)?;
                if !member.opcode().is_some_and(is_fusible)
                    || (member_shape != shape
                        && !(reduction.keepdim && member_shape == input_shape))
                    || ir.required_value_data_type(id)? != data_type
                {
                    return Ok(false);
                }
                if self.read_outside(id, members) {
                    outputs.push(id);
                }
            }
            return Ok(outputs.len() == 1);
        }

        if !along_last || self.read_outside(row, members) {
            return Ok(false);
        }

        let mut outputs = Vec::new();
        for &id in epilogue {
            let operation = ir.required_operation_for_value(id)?;
            if ir.required_value_shape(id)?.rank() != rank
                || ir.required_value_data_type(id)? != data_type
            {
                return Ok(false);
            }

            let fits = match (operation.opcode(), operation.attributes()) {
                (Some(Operation::Slice), OperationAttributes::Slice(slice)) => {
                    normalize_dim_usize(slice.dim, rank)? == last_axis
                }
                (Some(Operation::Transpose), OperationAttributes::Transpose(transpose)) => {
                    normalize_dim_usize(transpose.dim0, rank)? < last_axis
                        && normalize_dim_usize(transpose.dim1, rank)? < last_axis
                }
                (Some(Operation::Concat), &OperationAttributes::Concat(dim)) => {
                    normalize_dim_usize(dim, rank)? == last_axis
                        && operation
                            .inputs()
                            .iter()
                            .all(|input| epilogue.contains(input))
                }
                _ => true,
            };
            if !fits {
                return Ok(false);
            }
            if self.read_outside(id, members) {
                outputs.push(id);
            }
        }

        let &[last] = outputs.as_slice() else {
            return Ok(false);
        };
        Ok(!self.view(last)?
            && ir.required_value_shape(last)?.dimensions().last() == shape.dimensions().last())
    }

    /// A concatenation whose inputs, all of one data type and shape, the element-wise members compute or the group reads, with nothing outside the group reading the members.
    fn concat(
        &self,
        concat: ValueId,
        pointwise: &[ValueId],
        members: &BTreeSet<ValueId>,
    ) -> Result<Option<GroupKind>> {
        let operation = self.ir.required_operation_for_value(concat)?;
        let &OperationAttributes::Concat(dim) = operation.attributes() else {
            return Ok(None);
        };
        let mut values = operation.inputs().to_vec();
        values.extend_from_slice(pointwise);
        if pointwise.is_empty()
            || !self.uniform(&values)?
            || pointwise
                .iter()
                .any(|&member| self.read_outside(member, members))
        {
            return Ok(None);
        }

        let rank = self.ir.required_value_shape(operation.inputs()[0])?.rank();
        let axis = normalize_dim_usize(dim, rank)?;
        Ok(Some(GroupKind::Concat { concat, axis }))
    }

    /// A generated matmul with all of its weight chain or none, element-wise members that compute its left operand, of one data type and shape, of which only the left operand is read outside them and only by the matmul, and element-wise members of the product's data type and shape that the matmul's product alone feeds, of which one is read outside `members`.
    fn contraction(
        &self,
        matmul: ValueId,
        weight_chain: &[ValueId],
        pointwise: &[ValueId],
        members: &BTreeSet<ValueId>,
    ) -> Result<Option<GroupKind>> {
        let ir = self.ir;
        let weight = self.weights.get(&matmul);
        let chain_whole = weight.map_or(weight_chain.is_empty(), |weight| {
            weight_chain.len() == weight.members().len()
        });
        if !chain_whole
            || weight_chain
                .iter()
                .any(|member| self.weight_members[member] != matmul)
        {
            return Ok(None);
        }

        let operation = ir.required_operation_for_value(matmul)?;
        let &[lhs, rhs] = operation.inputs() else {
            return Ok(None);
        };
        // The epilogue is what the product feeds; the rest computes the left operand.
        let mut epilogue = BTreeSet::new();
        let mut pending = vec![matmul];
        while let Some(id) = pending.pop() {
            for &reader in self.readers(id) {
                if members.contains(&reader) && epilogue.insert(reader) {
                    pending.push(reader);
                }
            }
        }

        let prologue = pointwise
            .iter()
            .copied()
            .filter(|id| !epilogue.contains(id))
            .collect::<Vec<_>>();
        if !prologue.is_empty() {
            let set = prologue.iter().copied().collect::<BTreeSet<_>>();
            let outputs = prologue
                .iter()
                .copied()
                .filter(|&member| self.read_outside(member, &set))
                .collect::<Vec<_>>();
            if outputs != [lhs]
                || set.contains(&rhs)
                || self.readers(lhs) != [matmul]
                || self.graph_outputs.contains(&lhs)
                || !self.uniform(&prologue)?
            {
                return Ok(None);
            }
        }

        if !epilogue.is_empty() {
            let values = epilogue.iter().copied().chain([matmul]).collect::<Vec<_>>();
            let outputs = epilogue
                .iter()
                .filter(|&&member| self.read_outside(member, members))
                .count();
            if outputs != 1 || self.read_outside(matmul, members) || !self.uniform(&values)? {
                return Ok(None);
            }
        }
        Ok(Some(GroupKind::Contraction {
            matmul,
            weight: weight.cloned(),
        }))
    }
}

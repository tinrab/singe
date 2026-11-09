// Modules: subgraphs a dedicated kernel computes whole, which a recognizer finds after graph passes so that no layer has to opt in.
// Each kind has one small matcher, which declines what its kernel cannot take, so those nodes keep their generic route.
// This is separate from `graph/passes/recognize/`, which rewrites primitive spellings into operations for every backend.

use std::{
    collections::{BTreeSet, HashSet},
    hash::BuildHasher,
};

use crate::{
    backend::{
        cuda::{
            compiled::plan::{
                fusion::Planner,
                lowering::{CudaGraphProvider, ModuleKernel},
            },
            tuning::{TunedChoice, TunedWay, TuningRecords, fused_attention_key, group_key},
        },
        registry::{Operation, OperationAttributes, SliceAttributes},
    },
    data_type::DataType,
    error::Result,
    graph::ir::ValueId,
    shape::normalize_dim_usize,
};

// What a module's kernel computes, by the values its members read and produce.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModuleKind {
    RotaryCacheWrite(RotaryCacheWrite),
    RotaryAttention(RotaryAttention),
}

impl ModuleKind {
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::RotaryCacheWrite(_) => "rotary cache write",
            Self::RotaryAttention(_) => "rotary attention",
        }
    }
}

// A head's rotation by its position's angles: `first * cos - second * sin` and `second * cos + first * sin`, concatenated, with an RMS norm first where a graph has one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rotation {
    // What the rotation reads, before any norm.
    pub input: ValueId,
    pub norm: Option<ValueId>,
    pub cos: ValueId,
    pub sin: ValueId,
    pub output: ValueId,
}

// Query and key rotation with the writes of keys and values to a paged cache, or of positional keys and compressed values, for latent attention.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RotaryCacheWrite {
    // Absent when attention's kernel rotates its own query.
    pub query: Option<Rotation>,
    pub key: Rotation,
    pub value: ValueId,
    pub pages: ValueId,
    pub slots: ValueId,
    pub key_write: ValueId,
    pub value_write: ValueId,
}

// Attention over a paged cache with the query's norm and rotation in its kernel, as DeepInfer's `fused_norm_rope_attn_rope_cast` fuses them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RotaryAttention {
    pub query: Rotation,
    pub attention: ValueId,
}

// A module the recognizer found: its members and its kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecognizedModule {
    pub members: BTreeSet<ValueId>,
    pub kind: ModuleKind,
}

struct FoundRotation {
    rotation: Rotation,
    members: Vec<ValueId>,
}

struct CacheWrite {
    id: ValueId,
    base: ValueId,
    values: ValueId,
    pages: ValueId,
    slots: ValueId,
}

impl<S: BuildHasher, T: BuildHasher> Planner<'_, S, T> {
    // Returns the modules of the graph, none of which share a member.
    // A record that chose to run a module's nodes separately declines that module, so generic fusion takes them.
    pub fn recognize(
        &self,
        execution_order: &[ValueId],
        records: Option<&TuningRecords>,
    ) -> Result<Vec<RecognizedModule>> {
        let fused_attention = records.is_some_and(|records| {
            records.get(&fused_attention_key()) == Some(TunedChoice::FusedAttention)
        }) && self
            .lowering_plan
            .allows_module(ModuleKernel::RotaryAttention);
        if !fused_attention
            && !self
                .lowering_plan
                .allows_module(ModuleKernel::RotaryCacheWrite)
        {
            return Ok(Vec::new());
        }

        let mut rotations = Vec::new();
        let mut writes = Vec::new();
        let mut seen = HashSet::new();
        for &id in execution_order {
            let operation = self.ir.required_operation_for_value(id)?;
            if !seen.insert(operation.id()) {
                continue;
            }
            match operation.opcode() {
                Some(Operation::Concat) => rotations.extend(self.rotation(id)?),
                Some(Operation::IndexPut) => writes.extend(self.cache_write(id)?),
                _ => {}
            }
        }

        let mut modules = Vec::new();
        let mut claimed = HashSet::new();
        if fused_attention {
            for &id in execution_order {
                let Some(module) = self.rotary_attention(id, &rotations)? else {
                    continue;
                };
                if let Some(records) = records {
                    records.note_used(&fused_attention_key());
                }

                claimed.extend(module.members.iter().copied());
                modules.push(module);
            }
        }

        if !self
            .lowering_plan
            .allows_module(ModuleKernel::RotaryCacheWrite)
        {
            return Ok(modules);
        }

        for key_write in &writes {
            // Latent attention's one positional key per token reshapes on its way to the cache.
            let reshape = self.head_reshape(key_write.values)?;
            let written = reshape.unwrap_or(key_write.values);
            let Some(key) = rotations
                .iter()
                .find(|found: &&FoundRotation| found.rotation.output == written)
            else {
                continue;
            };
            let reader = if reshape.is_some() {
                key_write.values
            } else {
                key_write.id
            };
            if self.graph_outputs.contains(&key.rotation.output)
                || self.readers(key.rotation.output) != [reader]
                || (reshape.is_some()
                    && (self.graph_outputs.contains(&key_write.values)
                        || self.readers(key_write.values) != [key_write.id]))
                || claimed.contains(&key.rotation.output)
                || claimed.contains(&key_write.id)
            {
                continue;
            }

            // A step's layers share their angles and paging, so any other rotation and write could pair with these: the nearest that forms no cycle does.
            let position = self.position[&key_write.id];
            let distance = |id: ValueId| self.position[&id].abs_diff(position);
            let mut value_writes = writes
                .iter()
                .filter(|write| {
                    write.id != key_write.id
                        && write.base != key_write.base
                        && write.pages == key_write.pages
                        && write.slots == key_write.slots
                        && !claimed.contains(&write.id)
                        && !rotations.iter().any(|found| {
                            found.rotation.output == write.values
                                || self.head_reshape(write.values).ok().flatten()
                                    == Some(found.rotation.output)
                        })
                })
                .collect::<Vec<_>>();
            value_writes.sort_by_key(|write| distance(write.id));
            let mut queries = rotations
                .iter()
                .filter(|found| {
                    found.rotation.output != key.rotation.output
                        && found.rotation.cos == key.rotation.cos
                        && found.rotation.sin == key.rotation.sin
                        && !claimed.contains(&found.rotation.output)
                })
                .collect::<Vec<_>>();
            queries.sort_by_key(|found| distance(found.rotation.output));
            let mut chosen = None;
            // With attention rotating its own queries, what is left of this module writes the cache.
            let queries = queries
                .iter()
                .map(|query| Some(*query))
                .chain(fused_attention.then_some(None))
                .collect::<Vec<_>>();
            'pairs: for value_write in &value_writes {
                for &query in &queries {
                    let module = self.cache_module(query, key, key_write, value_write, reshape)?;
                    if let Some(module) = module {
                        chosen = Some(module);
                        break 'pairs;
                    }
                }
            }

            let Some(module) = chosen else {
                continue;
            };
            if let Some(records) = records {
                let key = group_key(self.ir, &self.in_order(&module.members))?;
                if records.get(&key) == Some(TunedChoice::Way(TunedWay::Separate)) {
                    records.note_used(&key);
                    continue;
                }
            }

            claimed.extend(module.members.iter().copied());
            modules.push(module);
        }
        Ok(modules)
    }

    // Returns the module of the attention at `id`, if its query is a rotation only it reads and nothing reads its log-sum-exp.
    fn rotary_attention(
        &self,
        id: ValueId,
        rotations: &[FoundRotation],
    ) -> Result<Option<RecognizedModule>> {
        let ir = self.ir;
        let operation = ir.required_operation_for_value(id)?;
        if operation.opcode() != Some(Operation::PagedAttentionOutput) {
            return Ok(None);
        }

        let Some(&query) = operation.inputs().first() else {
            return Ok(None);
        };
        let Some(found) = rotations
            .iter()
            .find(|found| found.rotation.output == query)
        else {
            return Ok(None);
        };
        // A log-sum-exp's reader would read this rotation too, since it is an operation of its own over the same inputs.
        if self.graph_outputs.contains(&query)
            || self.readers(query) != [id]
            || !ir
                .required_value_layout(found.rotation.input)?
                .is_contiguous()
            || ir.required_value_data_type(found.rotation.input)?
                != ir.required_value_data_type(query)?
        {
            return Ok(None);
        }

        let mut members = BTreeSet::from([id]);
        members.extend(found.members.iter().copied());
        if members
            .iter()
            .any(|member| self.weight_members.contains_key(member))
        {
            return Ok(None);
        }

        for &member in &found.members {
            if !self.view(member)?
                && !self
                    .lowering_plan
                    .should_try_provider(member, CudaGraphProvider::Fusion)
            {
                return Ok(None);
            }
        }

        if self.leaves_and_returns(&members) {
            return Ok(None);
        }
        Ok(Some(RecognizedModule {
            members,
            kind: ModuleKind::RotaryAttention(RotaryAttention {
                query: found.rotation.clone(),
                attention: id,
            }),
        }))
    }

    // Returns what `id` reshapes, if it reshapes `[T, 1, D]` into `[T, D]`.
    fn head_reshape(&self, id: ValueId) -> Result<Option<ValueId>> {
        let Some(operation) = self.ir.operation_for_value(id) else {
            return Ok(None);
        };
        let (Some(Operation::Reshape), [input, ..]) = (operation.opcode(), operation.inputs())
        else {
            return Ok(None);
        };
        let (from, to) = (
            self.ir.required_value_shape(*input)?.dimensions(),
            self.ir.required_value_shape(id)?.dimensions(),
        );
        Ok((from.len() == 3
            && from[1] == 1
            && to.len() == 2
            && to[0] == from[0]
            && to[1] == from[2])
            .then_some(*input))
    }

    // The members in execution order, as a group's members come.
    pub fn in_order(&self, members: &BTreeSet<ValueId>) -> Vec<ValueId> {
        let mut ordered = members.iter().copied().collect::<Vec<_>>();
        ordered.sort_by_key(|member| self.position[member]);
        ordered
    }

    // Returns the slice of the last axis `id` is, as what it slices and where it starts, if it is one.
    fn head_slice(&self, id: ValueId) -> Result<Option<(ValueId, i64, i64)>> {
        let Some(operation) = self.ir.operation_for_value(id) else {
            return Ok(None);
        };
        let (Some(Operation::Slice), OperationAttributes::Slice(slice)) =
            (operation.opcode(), operation.attributes())
        else {
            return Ok(None);
        };
        let SliceAttributes {
            dim,
            start,
            length,
            step,
        } = *slice;
        let input = operation.inputs()[0];
        let shape = self.ir.required_value_shape(input)?;
        if step != 1
            || shape.dimensions().is_empty()
            || normalize_dim_usize(dim, shape.rank())? + 1 != shape.dimensions().len()
        {
            return Ok(None);
        }
        Ok(Some((input, start, length)))
    }

    // Returns the operands of a multiplication: a slice of the head, and whatever it multiplies.
    fn head_product(&self, id: ValueId) -> Result<Option<(ValueId, ValueId)>> {
        let Some(operation) = self.ir.operation_for_value(id) else {
            return Ok(None);
        };
        let (Some(Operation::Mul), &[lhs, rhs]) = (operation.opcode(), operation.inputs()) else {
            return Ok(None);
        };
        if self.head_slice(lhs)?.is_some() {
            return Ok(Some((lhs, rhs)));
        }

        if self.head_slice(rhs)?.is_some() {
            return Ok(Some((rhs, lhs)));
        }
        Ok(None)
    }

    // Matches `concat(first * cos - second * sin, second * cos + first * sin)` at `id`, in either order of each commutative operation, where nothing else reads its intermediate values.
    fn rotation(&self, id: ValueId) -> Result<Option<FoundRotation>> {
        let ir = self.ir;
        let operation = ir.required_operation_for_value(id)?;
        let (&OperationAttributes::Concat(dim), &[rotated_first, rotated_second]) =
            (operation.attributes(), operation.inputs())
        else {
            return Ok(None);
        };
        let shape = ir.required_value_shape(id)?;
        let rank = shape.rank();
        if rank != 3 || normalize_dim_usize(dim, rank)? != 2 {
            return Ok(None);
        }

        let (Some(subtraction), Some(addition)) = (
            ir.operation_for_value(rotated_first),
            ir.operation_for_value(rotated_second),
        ) else {
            return Ok(None);
        };
        let (Some(Operation::Sub), &[first_cos, second_sin]) =
            (subtraction.opcode(), subtraction.inputs())
        else {
            return Ok(None);
        };
        let (Some(Operation::Add), &[left, right]) = (addition.opcode(), addition.inputs()) else {
            return Ok(None);
        };
        let products = [first_cos, second_sin, left, right];
        let mut pairs = Vec::with_capacity(4);
        for &product in &products {
            let Some(pair) = self.head_product(product)? else {
                return Ok(None);
            };
            pairs.push(pair);
        }

        let ((first, cos), (second, sin)) = (pairs[0], pairs[1]);
        // The addition holds the other two products in either order.
        let (second_cos, first_sin) = ((second, cos), (first, sin));
        if (pairs[2], pairs[3]) != (second_cos, first_sin)
            && (pairs[3], pairs[2]) != (second_cos, first_sin)
        {
            return Ok(None);
        }

        let (Some((source, 0, half)), Some((other, start, length))) =
            (self.head_slice(first)?, self.head_slice(second)?)
        else {
            return Ok(None);
        };
        let dimensions = shape.dimensions();
        if source != other
            || start != half
            || length != half
            || half <= 0
            || dimensions[2] != 2 * half
            || ir.required_value_shape(source)? != shape
        {
            return Ok(None);
        }

        let angle = [dimensions[0], 1, half];
        let data_type = ir.required_value_data_type(id)?;
        if !matches!(
            data_type,
            DataType::F16 | DataType::Bf16 | DataType::F32 | DataType::F64
        ) || ir.required_value_shape(cos)?.dimensions() != angle
            || ir.required_value_shape(sin)?.dimensions() != angle
        {
            return Ok(None);
        }

        let mut members = vec![first, second, rotated_first, rotated_second, id];
        members.extend_from_slice(&products);
        for &member in &members {
            if ir.required_value_data_type(member)? != data_type {
                return Ok(None);
            }
        }

        // Only the rotation itself may be read outside.
        let inside = members.iter().copied().collect::<BTreeSet<_>>();
        for &member in members.iter().filter(|&&member| member != id) {
            if self.read_outside(member, &inside) {
                return Ok(None);
            }
        }

        // A norm whose only readers are this rotation's slices joins it.
        let normalized = self.rotation_norm(source, first, second)?;
        if normalized.is_some() {
            members.push(source);
        }

        let norm = normalized.map(|_| source);
        let input = normalized.unwrap_or(source);
        Ok(Some(FoundRotation {
            rotation: Rotation {
                input,
                norm,
                cos,
                sin,
                output: id,
            },
            members,
        }))
    }

    // Returns what `source` normalizes, if it is an RMS norm of the head that only `first` and `second` read.
    fn rotation_norm(
        &self,
        source: ValueId,
        first: ValueId,
        second: ValueId,
    ) -> Result<Option<ValueId>> {
        let Some(operation) = self.ir.operation_for_value(source) else {
            return Ok(None);
        };
        let (Some(Operation::RmsNorm), OperationAttributes::LayerNormalization(attributes)) =
            (operation.opcode(), operation.attributes())
        else {
            return Ok(None);
        };
        let shape = self.ir.required_value_shape(source)?;
        let head = shape.dimensions()[2];
        let readers = self.readers(source);
        if attributes.has_bias
            || attributes.normalized_shape != [head]
            || self.graph_outputs.contains(&source)
            || readers
                .iter()
                .any(|&reader| reader != first && reader != second)
        {
            return Ok(None);
        }
        Ok(operation.inputs().first().copied())
    }

    // Matches a plain write of `[T, ..]` values at `[T]` pages and slots of a contiguous `[P, S, G, D]` cache.
    fn cache_write(&self, id: ValueId) -> Result<Option<CacheWrite>> {
        let ir = self.ir;
        let operation = ir.required_operation_for_value(id)?;
        let (OperationAttributes::IndexPut(attributes), &[base, values, pages, slots]) =
            (operation.attributes(), operation.inputs())
        else {
            return Ok(None);
        };
        let tokens = ir
            .required_value_shape(values)?
            .dimensions()
            .first()
            .copied();
        let indices = [pages, slots];
        for index in indices {
            if ir.required_value_data_type(index)? != DataType::I64
                || ir
                    .required_value_shape(index)?
                    .dimensions()
                    .first()
                    .copied()
                    != tokens
                || ir.required_value_shape(index)?.rank() != 1
            {
                return Ok(None);
            }
        }

        let base_shape = ir.required_value_shape(base)?.dimensions();
        let value_shape = ir.required_value_shape(values)?.dimensions();
        let fits = attributes.dim == 0
            && !attributes.accumulate
            && matches!((base_shape.len(), value_shape.len()), (4, 3) | (3, 2))
            && base_shape[2..] == value_shape[1..]
            && ir.required_value_data_type(base)? == ir.required_value_data_type(values)?
            && ir.required_value_layout(id)?.is_contiguous();
        Ok(fits.then_some(CacheWrite {
            id,
            base,
            values,
            pages,
            slots,
        }))
    }

    // Returns the module of these parts, unless its kernel cannot take them, the policy keeps one of its nodes from generated kernels, or running them together would make the kernel wait for itself.
    fn cache_module(
        &self,
        query: Option<&FoundRotation>,
        key: &FoundRotation,
        key_write: &CacheWrite,
        value_write: &CacheWrite,
        reshape: Option<ValueId>,
    ) -> Result<Option<RecognizedModule>> {
        let ir = self.ir;
        let key_shape = ir.required_value_shape(key.rotation.output)?.dimensions();
        let data_type = ir.required_value_data_type(key.rotation.output)?;
        let value_shape = ir.required_value_shape(value_write.values)?.dimensions();
        // Both writes have heads in their caches, or neither does.
        let value_fits = if reshape.is_some() {
            value_shape.len() == 2 && value_shape[0] == key_shape[0]
        } else {
            value_shape == key_shape
        };
        if let Some(query) = query {
            let query_shape = ir.required_value_shape(query.rotation.output)?.dimensions();
            if query_shape[0] != key_shape[0]
                || query_shape[2] != key_shape[2]
                || ir.required_value_data_type(query.rotation.output)? != data_type
            {
                return Ok(None);
            }
        }

        if !value_fits || ir.required_value_data_type(value_write.values)? != data_type {
            return Ok(None);
        }

        let mut members = BTreeSet::new();
        if let Some(query) = query {
            members.extend(query.members.iter().copied());
        }

        members.extend(key.members.iter().copied());
        members.extend([key_write.id, value_write.id]);
        if reshape.is_some() {
            members.insert(key_write.values);
        }

        if members
            .iter()
            .any(|member| self.weight_members.contains_key(member))
        {
            return Ok(None);
        }

        for &member in &members {
            let fusible = self.view(member)?
                || (reshape.is_some() && member == key_write.values)
                || self
                    .lowering_plan
                    .should_try_provider(member, CudaGraphProvider::Fusion);
            let generated = (member == key_write.id || member == value_write.id)
                && self
                    .lowering_plan
                    .should_try_provider(member, CudaGraphProvider::Kernel);
            if !fusible && !generated {
                return Ok(None);
            }
        }

        if self.leaves_and_returns(&members) {
            return Ok(None);
        }
        Ok(Some(RecognizedModule {
            members,
            kind: ModuleKind::RotaryCacheWrite(RotaryCacheWrite {
                query: query.map(|query| query.rotation.clone()),
                key: key.rotation.clone(),
                value: value_write.values,
                pages: key_write.pages,
                slots: key_write.slots,
                key_write: key_write.id,
                value_write: value_write.id,
            }),
        }))
    }
}

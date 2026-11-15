use std::{collections::BTreeSet, hash::BuildHasher};

use crate::{
    backend::{
        cuda::compiled::plan::{
            fusion::{GroupKind, Planner},
            lowering::CudaGraphProvider,
        },
        kernel_lowering::{is_fusible, is_row_operation},
        registry::{Operation, OperationAttributes},
    },
    error::Result,
    graph::ir::ValueId,
    shape::normalize_dim_usize,
};

impl<S: BuildHasher, T: BuildHasher> Planner<'_, S, T> {
    fn read_within(&self, value: ValueId, members: &BTreeSet<ValueId>) -> BTreeSet<ValueId> {
        // What reads `value` within `members`, directly or through other members: the epilogue of a row or a product.
        let mut read = BTreeSet::new();
        let mut pending = vec![value];
        while let Some(id) = pending.pop() {
            for &reader in self.readers(id) {
                if members.contains(&reader) && read.insert(reader) {
                    pending.push(reader);
                }
            }
        }
        read
    }

    /// Returns the kernel a group of `members` is, or `None` when no lowering writes one.
    pub fn classify(&self, members: &BTreeSet<ValueId>) -> Result<Option<GroupKind>> {
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
        let mut epilogue = self.read_within(row, members);

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
    pub fn computes(
        &self,
        value: ValueId,
        members: &BTreeSet<ValueId>,
    ) -> Result<BTreeSet<ValueId>> {
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
        let epilogue = self.read_within(matmul, members);

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

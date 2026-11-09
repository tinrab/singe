use crate::{
    backend::cuda::compiled::executable::report::types::GraphCompileAliasGroupReport,
    error::Result, graph::ir::Graph,
};

pub fn compile_report_ir_alias_groups(ir: &Graph) -> Result<Vec<GraphCompileAliasGroupReport>> {
    let outputs = ir.output_value_set();
    let members = ir.alias_group_members()?;
    let mut groups = members
        .into_iter()
        .filter_map(|(root, mut values)| {
            (values.len() > 1).then(|| {
                values.sort_by_key(|value| value.index());
                let contains_input = values.iter().any(|id| ir.value_is_input(*id));
                let contains_constant = values.iter().any(|id| ir.value_is_constant(*id));
                let contains_output = values.iter().any(|id| outputs.contains(id));
                GraphCompileAliasGroupReport::new(
                    root,
                    values,
                    contains_input,
                    contains_constant,
                    contains_output,
                )
            })
        })
        .collect::<Vec<_>>();
    groups.sort_by_key(|group| group.root_value().index());
    Ok(groups)
}

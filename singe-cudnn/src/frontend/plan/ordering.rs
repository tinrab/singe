use std::{collections::HashMap, mem::take};

use crate::{
    error::Result,
    frontend::{error::Error as FrontendError, plan::PlanId},
};

/// Resolves a nonempty selection before any entries or handle state change.
pub fn plan_positions<'a>(
    available: impl IntoIterator<Item = &'a PlanId>,
    requested: &[PlanId],
) -> Result<Vec<usize>> {
    if requested.is_empty() {
        return Err(FrontendError::PlanOrderEmpty.into());
    }
    let available = available
        .into_iter()
        .enumerate()
        .map(|(position, plan)| (plan, position))
        .collect::<HashMap<_, _>>();
    let mut seen = vec![false; available.len()];
    let mut positions = Vec::with_capacity(requested.len());
    for plan in requested {
        let &position = available
            .get(plan)
            .ok_or_else(|| FrontendError::PlanNotFound { plan: plan.clone() })?;
        if seen[position] {
            return Err(FrontendError::PlanOrderDuplicate { plan: plan.clone() }.into());
        }
        seen[position] = true;
        positions.push(position);
    }
    Ok(positions)
}

/// Moves complete entries together, preserving native owners and associated metadata.
pub fn reorder_plan_entries<T>(
    entries: &mut Vec<T>,
    requested: &[PlanId],
    id: impl Fn(&T) -> &PlanId,
) -> Result<()> {
    let positions = plan_positions(entries.iter().map(id), requested)?;
    let mut previous = take(entries).into_iter().map(Some).collect::<Vec<_>>();
    entries.extend(
        positions
            .into_iter()
            .filter_map(|position| previous[position].take()),
    );
    Ok(())
}

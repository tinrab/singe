use crate::graph::{InputDimensionGuard, ir::Graph};

impl Graph {
    /// Returns one symbolic input dimension guard by input name and axis.
    #[must_use]
    pub fn input_dimension_guard(&self, name: &str, axis: usize) -> Option<&InputDimensionGuard> {
        self.input_dimension_guards.get(&(name.to_string(), axis))
    }

    /// Returns symbolic input dimension guards for one graph input.
    pub fn input_dimension_guards_for_input(
        &self,
        name: &str,
    ) -> impl Iterator<Item = &InputDimensionGuard> {
        self.input_dimension_guards
            .values()
            .filter(move |guard| guard.input() == name)
    }

    /// Returns whether this IR carries symbolic input dimension guards.
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn has_input_dimension_guards(&self) -> bool {
        !self.input_dimension_guards.is_empty()
    }

    /// Returns all symbolic input dimension guards in stable input/axis/name order.
    #[must_use]
    pub fn ordered_input_dimension_guards(&self) -> Vec<&InputDimensionGuard> {
        let mut guards = self.input_dimension_guards.values().collect::<Vec<_>>();
        guards.sort_by(|lhs, rhs| {
            lhs.input()
                .cmp(rhs.input())
                .then_with(|| lhs.axis().cmp(&rhs.axis()))
                .then_with(|| lhs.name().cmp(rhs.name()))
        });
        guards
    }
}

use std::cell::Cell;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PlanChoice {
    #[default]
    Timed,
    First,
}

thread_local! {
    static CURRENT: Cell<PlanChoice> = const { Cell::new(PlanChoice::Timed) };
}

impl PlanChoice {
    pub(crate) fn current() -> Self {
        CURRENT.with(Cell::get)
    }
}

pub fn with_plan_choice<T>(choice: PlanChoice, f: impl FnOnce() -> T) -> T {
    struct Restore(PlanChoice);

    impl Drop for Restore {
        fn drop(&mut self) {
            CURRENT.with(|current| current.set(self.0));
        }
    }

    let _restore = Restore(CURRENT.with(|current| current.replace(choice)));
    f()
}

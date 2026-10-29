//! Warp and CTA operations that park a lane until its peers arrive.

use crate::{
    ast::ScalarType,
    emulator::{
        engine::{Engine, Result, Step, Waiting, WarpOperation, product},
        execution_error::ExecutionError,
        ir::CollectiveOperation,
        memory::Memory,
        target::{NAMED_BARRIERS, WARP_SIZE},
    },
};

impl Engine {
    pub fn execute_collective(
        &mut self,
        global: &Memory,
        lane: usize,
        operation: &CollectiveOperation,
        active_mask: u32,
    ) -> Result<Step> {
        match operation {
            CollectiveOperation::MatrixLoad { layout, address } => {
                self.wait_matrix_load(global, lane, *layout, *address)?;
                return Ok(Step::Hold);
            }
            CollectiveOperation::MatrixMultiply(multiply) => {
                self.wait_matrix_multiply(lane, multiply)?;
                return Ok(Step::Hold);
            }
            CollectiveOperation::CtaBarrier { barrier, count } => {
                let barrier = self.u32_source(lane, barrier)? as usize;
                let count = count
                    .as_ref()
                    .map(|c| self.source(lane, c, ScalarType::U32).map(|c| c as usize))
                    .transpose()?
                    .filter(|count| *count != 0);
                if barrier >= NAMED_BARRIERS
                    || count.is_some_and(|c| {
                        c % WARP_SIZE != 0
                            || c > product(self.config.block)
                                .unwrap_or(0)
                                .next_multiple_of(WARP_SIZE)
                    })
                {
                    return Err(ExecutionError::Invalid("barrier number or count".into()));
                }
                self.lanes[lane].waiting = Some(Waiting::Cta {
                    barrier,
                    generation: self.generations[self.lanes[lane].cta][barrier],
                    count,
                });
                self.mark_waiting(lane);
                return Ok(Step::Hold);
            }
            CollectiveOperation::WarpBarrier { mask } => {
                let mask = self.b32_source(lane, mask)?;
                self.wait_warp(lane, mask, WarpOperation::Barrier)?;
                return Ok(Step::Hold);
            }
            CollectiveOperation::Shuffle {
                mode,
                destination,
                valid,
                value,
                lane: source_lane,
                control,
                mask,
            } => {
                let mask = self.b32_source(lane, mask)?;
                let operation = WarpOperation::Shuffle {
                    mode: *mode,
                    destination: *destination,
                    valid: *valid,
                    value: self.source(lane, value, ScalarType::B32)?,
                    lane: self.u32_source(lane, source_lane)?,
                    control: self.u32_source(lane, control)?,
                };
                self.wait_warp(lane, mask, operation)?;
                return Ok(Step::Hold);
            }
            CollectiveOperation::Vote {
                mode,
                destination,
                predicate,
                mask,
            } => {
                let mask = self.b32_source(lane, mask)?;
                let operation = WarpOperation::Vote {
                    mode: *mode,
                    destination: *destination,
                    predicate: self.predicate(lane, *predicate)?,
                };
                self.wait_warp(lane, mask, operation)?;
                return Ok(Step::Hold);
            }
            CollectiveOperation::ReduceAdd {
                destination,
                value,
                mask,
            } => {
                let mask = self.b32_source(lane, mask)?;
                let operation = WarpOperation::ReduceAdd {
                    destination: *destination,
                    value: self.u32_source(lane, value)?,
                };
                self.wait_warp(lane, mask, operation)?;
                return Ok(Step::Hold);
            }
            CollectiveOperation::ActiveMask { destination } => {
                self.write(lane, *destination, u128::from(active_mask), ScalarType::B32);
            }
        }
        Ok(Step::Advance)
    }
}

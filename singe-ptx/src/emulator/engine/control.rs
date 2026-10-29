//! Branches, calls, returns, and lane termination.

use std::sync::atomic::AtomicBool;

use crate::emulator::{
    engine::{Engine, PausedAt, Result, Step, frame_charge},
    execution_error::ExecutionError,
    ir::{ControlOperation, FrameRole},
    memory::Memory,
};

impl Engine {
    pub fn execute_control(
        &mut self,
        global: &mut Memory,
        lane: usize,
        operation: &ControlOperation,
        cancellation: &AtomicBool,
    ) -> Result<Step> {
        match operation {
            ControlOperation::Branch { target } => {
                self.current_mut(lane).pc = target.0;
                return Ok(Step::Hold);
            }
            ControlOperation::Call {
                function,
                arguments,
                returns,
                ..
            } => {
                if self.lanes[lane].frames.len() >= self.options.call_depth {
                    return Err(ExecutionError::CallDepth);
                }
                let charge = frame_charge(self.code.function(*function))?;
                global.reserve_execution(self.grid, charge)?;
                let values: Vec<_> = arguments
                    .iter()
                    .map(|arg| self.object(lane, *arg))
                    .collect();
                let returns = returns.iter().map(|r| self.object(lane, *r)).collect();
                let frame = self.frame(
                    *function,
                    self.actor(lane),
                    &values,
                    returns,
                    cancellation,
                    global,
                )?;
                self.lanes[lane]
                    .frames
                    .try_reserve_exact(1)
                    .map_err(|_| ExecutionError::ExecutionStateLimit)?;
                self.current_mut(lane).pc += 1;
                self.lanes[lane].carry = None;
                self.lanes[lane].frames.push(frame);
                return Ok(Step::Hold);
            }
            ControlOperation::Return => {
                if self.function(lane).noreturn {
                    return Err(ExecutionError::Invalid(
                        "noreturn function returned to its caller".into(),
                    ));
                }
                self.lanes[lane].carry = None;
                let frame = self.current(lane);
                let function = &self.code.function(frame.function);
                let mut copies = Vec::new();
                for ((id, _), address) in function
                    .objects
                    .iter()
                    .enumerate()
                    .filter(|(_, o)| o.role == FrameRole::Return)
                    .zip(&frame.returns)
                {
                    copies.push((
                        frame.objects[id],
                        *address,
                        usize::try_from(function.objects[id].layout.byte_len())
                            .map_err(|_| ExecutionError::ExecutionStateLimit)?,
                    ));
                }
                for (source, destination, size) in copies {
                    global.copy_parameter(
                        source,
                        destination,
                        size,
                        self.actor(lane),
                        cancellation,
                    )?;
                }
                self.pop(lane, global);
                return Ok(Step::Hold);
            }
            ControlOperation::Exit => {
                while !self.lanes[lane].frames.is_empty() {
                    self.pop(lane, global);
                }
                return Ok(Step::Hold);
            }
            ControlOperation::Trap => return Err(ExecutionError::Trap),
            ControlOperation::Breakpoint => {
                if self.breakpoint.is_none() {
                    self.breakpoint = Some(PausedAt {
                        lane,
                        pc: self.current(lane).pc,
                    });
                }
            }
        }
        Ok(Step::Advance)
    }
}

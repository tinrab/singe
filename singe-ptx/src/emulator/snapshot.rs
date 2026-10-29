//! Copies of lane state returned by
//! [`Emulator::inspect_lane`](crate::emulator::Emulator::inspect_lane).

use crate::ast::{ScalarType, Span};

/// A register of an inspected lane: its declaration and value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisterSnapshot {
    pub(crate) name: String,
    pub(crate) ty: ScalarType,
    pub(crate) declaration: Span,
    pub(crate) value: Option<u128>,
}

impl RegisterSnapshot {
    /// Returns the register name. Registers declared as a range, such as `%r<4>`, have
    /// their index appended (`%r2`).
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the declared storage type.
    #[must_use]
    pub fn data_type(&self) -> ScalarType {
        self.ty
    }

    /// Returns the source byte range of the declaration, which tells apart registers
    /// that shadow each other.
    #[must_use]
    pub fn declaration(&self) -> Span {
        self.declaration
    }

    /// Returns the register's bits, or `None` if no instruction has written it yet.
    /// A register is always written whole, so bits above the declared width are zero.
    #[must_use]
    pub fn value(&self) -> Option<u128> {
        self.value
    }
}

/// One call frame of an inspected lane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameSnapshot {
    pub(crate) function: String,
    pub(crate) instruction: usize,
    pub(crate) span: Option<Span>,
    pub(crate) registers: Vec<RegisterSnapshot>,
}

impl FrameSnapshot {
    /// Returns the function name.
    #[must_use]
    pub fn function(&self) -> &str {
        &self.function
    }

    /// Returns the index of the frame's current instruction within its function.
    #[must_use]
    pub fn instruction(&self) -> usize {
        self.instruction
    }

    /// Returns the source byte range of the current instruction, or `None` past the
    /// last instruction.
    #[must_use]
    pub fn span(&self) -> Option<Span> {
        self.span
    }

    /// Returns the frame's registers in declaration order.
    #[must_use]
    pub fn registers(&self) -> &[RegisterSnapshot] {
        &self.registers
    }
}

/// A copy of one lane's state, taken by
/// [`Emulator::inspect_lane`](crate::emulator::Emulator::inspect_lane).
///
/// The copy does not keep the emulator alive and does not change when the launch
/// resumes. An exited lane has no frames.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaneSnapshot {
    pub(crate) cta: [u32; 3],
    pub(crate) lane: [u32; 3],
    pub(crate) carry: Option<bool>,
    pub(crate) frames: Vec<FrameSnapshot>,
}

impl LaneSnapshot {
    /// Returns the selected CTA's coordinates in the grid.
    #[must_use]
    pub fn cta(&self) -> [u32; 3] {
        self.cta
    }

    /// Returns the selected lane's coordinates in its CTA.
    #[must_use]
    pub fn lane(&self) -> [u32; 3] {
        self.lane
    }

    /// Returns the lane's carry flag, or `None` while it is undefined: at lane start
    /// and after a call or return.
    #[must_use]
    pub fn carry(&self) -> Option<bool> {
        self.carry
    }

    /// Returns the call frames, from the entry function to the current function.
    #[must_use]
    pub fn frames(&self) -> &[FrameSnapshot] {
        &self.frames
    }

    /// Returns whether the lane has exited.
    #[must_use]
    pub fn is_exited(&self) -> bool {
        self.frames.is_empty()
    }
}

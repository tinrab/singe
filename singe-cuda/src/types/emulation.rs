use std::fmt::{self, Display, Formatter};

use bitflags::parser;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::impl_enum_display;
use singe_cuda_sys::library_types;

singe_core::native_enum! {
    /// Enum for specifying how to leverage floating-point emulation algorithms
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum EmulationStrategy: library_types::cudaEmulationStrategy {
        /// The default emulation strategy.
        /// For emulated computations, this is equivalent to `CUDA_EMULATION_STRATEGY_PERFORMANT`, unless a library dependent environment variable is set
        Default = CUDA_EMULATION_STRATEGY_DEFAULT,
        /// An emulation strategy which configures libraries to use emulation when it provides a performance benefit
        Performant = CUDA_EMULATION_STRATEGY_PERFORMANT,
        /// An emulation strategy which configures libraries to use emulation whenever possible
        Eager = CUDA_EMULATION_STRATEGY_EAGER,
    }
}

impl_enum_display!(EmulationStrategy, {
    Self::Default => "CUDA_EMULATION_STRATEGY_DEFAULT",
    Self::Performant => "CUDA_EMULATION_STRATEGY_PERFORMANT",
    Self::Eager => "CUDA_EMULATION_STRATEGY_EAGER",
});

singe_core::native_enum! {
    /// Enum to configure the mantissa related parameters for floating-point emulation algorithms
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum EmulationMantissaControl: library_types::cudaEmulationMantissaControl {
        /// The number of retained mantissa bits are computed at runtime to ensure the same or better accuracy than the floating point representation being emulated
        Dynamic = CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC,
        /// The number of retained mantissa bits are known at runtime
        Fixed = CUDA_EMULATION_MANTISSA_CONTROL_FIXED,
    }
}

impl_enum_display!(EmulationMantissaControl, {
    Self::Dynamic => "CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC",
    Self::Fixed => "CUDA_EMULATION_MANTISSA_CONTROL_FIXED",
});

bitflags::bitflags! {
    /// Special input and output values that CUDA emulation algorithms must support.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct EmulationSpecialValuesSupport: u32 {
        /// Places no requirement on emulation algorithms to support special values.
        const NONE = library_types::cudaEmulationSpecialValuesSupport::CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_NONE.0;
        /// Requires support for signed infinity inputs and outputs.
        const INFINITY = library_types::cudaEmulationSpecialValuesSupport::CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_INFINITY.0;
        /// Requires support for NaN inputs and outputs.
        const NAN = library_types::cudaEmulationSpecialValuesSupport::CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_NAN.0;
        /// Requires the default support for both signed infinities and NaNs.
        const DEFAULT = library_types::cudaEmulationSpecialValuesSupport::CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT.0;
    }
}

impl Display for EmulationSpecialValuesSupport {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

impl From<library_types::cudaEmulationSpecialValuesSupport> for EmulationSpecialValuesSupport {
    fn from(value: library_types::cudaEmulationSpecialValuesSupport) -> Self {
        Self::from_bits_retain(value.0)
    }
}

impl From<EmulationSpecialValuesSupport> for library_types::cudaEmulationSpecialValuesSupport {
    fn from(value: EmulationSpecialValuesSupport) -> Self {
        Self(value.bits())
    }
}

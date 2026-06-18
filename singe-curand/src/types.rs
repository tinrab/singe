use std::fmt;

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_curand_sys as sys;

/// Result status returned by cuRAND host API calls.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Status {
    /// The operation completed successfully.
    Success,
    /// The cuRAND header version does not match the linked library version.
    VersionMismatch,
    /// The generator has not been initialized.
    NotInitialized,
    /// cuRAND could not allocate required memory.
    AllocationFailed,
    /// The generator or argument has the wrong type.
    TypeError,
    /// An argument is outside the range accepted by cuRAND.
    OutOfRange,
    /// The requested length is not a multiple of the quasirandom dimension.
    LengthNotMultiple,
    /// The selected operation requires GPU double-precision support.
    DoublePrecisionRequired,
    /// A cuRAND kernel launch failed.
    LaunchFailure,
    /// A previous CUDA failure was already pending when cuRAND was entered.
    PreexistingFailure,
    /// cuRAND could not initialize CUDA.
    InitializationFailed,
    /// The GPU architecture does not support the requested feature.
    ArchitectureMismatch,
    /// cuRAND reported an internal library error.
    InternalError,
    /// cuRAND returned a status code not known to this wrapper.
    Unknown(u32),
}

impl Status {
    pub const fn raw(self) -> u32 {
        match self {
            Self::Success => sys::curandStatus_t::CURAND_STATUS_SUCCESS as _,
            Self::VersionMismatch => sys::curandStatus_t::CURAND_STATUS_VERSION_MISMATCH as _,
            Self::NotInitialized => sys::curandStatus_t::CURAND_STATUS_NOT_INITIALIZED as _,
            Self::AllocationFailed => sys::curandStatus_t::CURAND_STATUS_ALLOCATION_FAILED as _,
            Self::TypeError => sys::curandStatus_t::CURAND_STATUS_TYPE_ERROR as _,
            Self::OutOfRange => sys::curandStatus_t::CURAND_STATUS_OUT_OF_RANGE as _,
            Self::LengthNotMultiple => sys::curandStatus_t::CURAND_STATUS_LENGTH_NOT_MULTIPLE as _,
            Self::DoublePrecisionRequired => {
                sys::curandStatus_t::CURAND_STATUS_DOUBLE_PRECISION_REQUIRED as _
            }
            Self::LaunchFailure => sys::curandStatus_t::CURAND_STATUS_LAUNCH_FAILURE as _,
            Self::PreexistingFailure => sys::curandStatus_t::CURAND_STATUS_PREEXISTING_FAILURE as _,
            Self::InitializationFailed => {
                sys::curandStatus_t::CURAND_STATUS_INITIALIZATION_FAILED as _
            }
            Self::ArchitectureMismatch => sys::curandStatus_t::CURAND_STATUS_ARCH_MISMATCH as _,
            Self::InternalError => sys::curandStatus_t::CURAND_STATUS_INTERNAL_ERROR as _,
            Self::Unknown(code) => code,
        }
    }
}

impl From<sys::curandStatus_t> for Status {
    fn from(status: sys::curandStatus_t) -> Self {
        Self::from(status as u32)
    }
}

impl From<u32> for Status {
    fn from(code: u32) -> Self {
        match code {
            x if x == sys::curandStatus_t::CURAND_STATUS_SUCCESS as u32 => Self::Success,
            x if x == sys::curandStatus_t::CURAND_STATUS_VERSION_MISMATCH as u32 => {
                Self::VersionMismatch
            }
            x if x == sys::curandStatus_t::CURAND_STATUS_NOT_INITIALIZED as u32 => {
                Self::NotInitialized
            }
            x if x == sys::curandStatus_t::CURAND_STATUS_ALLOCATION_FAILED as u32 => {
                Self::AllocationFailed
            }
            x if x == sys::curandStatus_t::CURAND_STATUS_TYPE_ERROR as u32 => Self::TypeError,
            x if x == sys::curandStatus_t::CURAND_STATUS_OUT_OF_RANGE as u32 => Self::OutOfRange,
            x if x == sys::curandStatus_t::CURAND_STATUS_LENGTH_NOT_MULTIPLE as u32 => {
                Self::LengthNotMultiple
            }
            x if x == sys::curandStatus_t::CURAND_STATUS_DOUBLE_PRECISION_REQUIRED as u32 => {
                Self::DoublePrecisionRequired
            }
            x if x == sys::curandStatus_t::CURAND_STATUS_LAUNCH_FAILURE as u32 => {
                Self::LaunchFailure
            }
            x if x == sys::curandStatus_t::CURAND_STATUS_PREEXISTING_FAILURE as u32 => {
                Self::PreexistingFailure
            }
            x if x == sys::curandStatus_t::CURAND_STATUS_INITIALIZATION_FAILED as u32 => {
                Self::InitializationFailed
            }
            x if x == sys::curandStatus_t::CURAND_STATUS_ARCH_MISMATCH as u32 => {
                Self::ArchitectureMismatch
            }
            x if x == sys::curandStatus_t::CURAND_STATUS_INTERNAL_ERROR as u32 => {
                Self::InternalError
            }
            code => Self::Unknown(code),
        }
    }
}

impl TryFrom<Status> for sys::curandStatus_t {
    type Error = Status;

    fn try_from(status: Status) -> Result<Self, Self::Error> {
        match status {
            Status::Success => Ok(Self::CURAND_STATUS_SUCCESS),
            Status::VersionMismatch => Ok(Self::CURAND_STATUS_VERSION_MISMATCH),
            Status::NotInitialized => Ok(Self::CURAND_STATUS_NOT_INITIALIZED),
            Status::AllocationFailed => Ok(Self::CURAND_STATUS_ALLOCATION_FAILED),
            Status::TypeError => Ok(Self::CURAND_STATUS_TYPE_ERROR),
            Status::OutOfRange => Ok(Self::CURAND_STATUS_OUT_OF_RANGE),
            Status::LengthNotMultiple => Ok(Self::CURAND_STATUS_LENGTH_NOT_MULTIPLE),
            Status::DoublePrecisionRequired => Ok(Self::CURAND_STATUS_DOUBLE_PRECISION_REQUIRED),
            Status::LaunchFailure => Ok(Self::CURAND_STATUS_LAUNCH_FAILURE),
            Status::PreexistingFailure => Ok(Self::CURAND_STATUS_PREEXISTING_FAILURE),
            Status::InitializationFailed => Ok(Self::CURAND_STATUS_INITIALIZATION_FAILED),
            Status::ArchitectureMismatch => Ok(Self::CURAND_STATUS_ARCH_MISMATCH),
            Status::InternalError => Ok(Self::CURAND_STATUS_INTERNAL_ERROR),
            Status::Unknown(_) => Err(status),
        }
    }
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Success => f.write_str("CURAND_STATUS_SUCCESS"),
            Self::VersionMismatch => f.write_str("CURAND_STATUS_VERSION_MISMATCH"),
            Self::NotInitialized => f.write_str("CURAND_STATUS_NOT_INITIALIZED"),
            Self::AllocationFailed => f.write_str("CURAND_STATUS_ALLOCATION_FAILED"),
            Self::TypeError => f.write_str("CURAND_STATUS_TYPE_ERROR"),
            Self::OutOfRange => f.write_str("CURAND_STATUS_OUT_OF_RANGE"),
            Self::LengthNotMultiple => f.write_str("CURAND_STATUS_LENGTH_NOT_MULTIPLE"),
            Self::DoublePrecisionRequired => f.write_str("CURAND_STATUS_DOUBLE_PRECISION_REQUIRED"),
            Self::LaunchFailure => f.write_str("CURAND_STATUS_LAUNCH_FAILURE"),
            Self::PreexistingFailure => f.write_str("CURAND_STATUS_PREEXISTING_FAILURE"),
            Self::InitializationFailed => f.write_str("CURAND_STATUS_INITIALIZATION_FAILED"),
            Self::ArchitectureMismatch => f.write_str("CURAND_STATUS_ARCH_MISMATCH"),
            Self::InternalError => f.write_str("CURAND_STATUS_INTERNAL_ERROR"),
            Self::Unknown(code) => write!(f, "unknown curand status ({code})"),
        }
    }
}

/// cuRAND random number generator kind.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum RngType {
    /// Internal test generator type.
    Test = sys::curandRngType_t::CURAND_RNG_TEST as _,
    /// Default pseudorandom generator.
    PseudoDefault = sys::curandRngType_t::CURAND_RNG_PSEUDO_DEFAULT as _,
    /// XORWOW pseudorandom generator.
    PseudoXorwow = sys::curandRngType_t::CURAND_RNG_PSEUDO_XORWOW as _,
    /// MRG32k3a pseudorandom generator.
    PseudoMrg32k3a = sys::curandRngType_t::CURAND_RNG_PSEUDO_MRG32K3A as _,
    /// Mersenne Twister MTGP32 pseudorandom generator.
    PseudoMtgp32 = sys::curandRngType_t::CURAND_RNG_PSEUDO_MTGP32 as _,
    /// Mersenne Twister MT19937 pseudorandom generator.
    PseudoMt19937 = sys::curandRngType_t::CURAND_RNG_PSEUDO_MT19937 as _,
    /// PHILOX-4x32-10 pseudorandom generator.
    PseudoPhilox4_32_10 = sys::curandRngType_t::CURAND_RNG_PSEUDO_PHILOX4_32_10 as _,
    /// Default quasirandom generator.
    QuasiDefault = sys::curandRngType_t::CURAND_RNG_QUASI_DEFAULT as _,
    /// Sobol32 quasirandom generator.
    QuasiSobol32 = sys::curandRngType_t::CURAND_RNG_QUASI_SOBOL32 as _,
    /// Scrambled Sobol32 quasirandom generator.
    QuasiScrambledSobol32 = sys::curandRngType_t::CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 as _,
    /// Sobol64 quasirandom generator.
    QuasiSobol64 = sys::curandRngType_t::CURAND_RNG_QUASI_SOBOL64 as _,
    /// Scrambled Sobol64 quasirandom generator.
    QuasiScrambledSobol64 = sys::curandRngType_t::CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 as _,
}

impl_enum_conversion!(sys::curandRngType_t, RngType);

impl_enum_display!(RngType, {
    Self::Test => "CURAND_RNG_TEST",
    Self::PseudoDefault => "CURAND_RNG_PSEUDO_DEFAULT",
    Self::PseudoXorwow => "CURAND_RNG_PSEUDO_XORWOW",
    Self::PseudoMrg32k3a => "CURAND_RNG_PSEUDO_MRG32K3A",
    Self::PseudoMtgp32 => "CURAND_RNG_PSEUDO_MTGP32",
    Self::PseudoMt19937 => "CURAND_RNG_PSEUDO_MT19937",
    Self::PseudoPhilox4_32_10 => "CURAND_RNG_PSEUDO_PHILOX4_32_10",
    Self::QuasiDefault => "CURAND_RNG_QUASI_DEFAULT",
    Self::QuasiSobol32 => "CURAND_RNG_QUASI_SOBOL32",
    Self::QuasiScrambledSobol32 => "CURAND_RNG_QUASI_SCRAMBLED_SOBOL32",
    Self::QuasiSobol64 => "CURAND_RNG_QUASI_SOBOL64",
    Self::QuasiScrambledSobol64 => "CURAND_RNG_QUASI_SCRAMBLED_SOBOL64",
});

/// Memory ordering used by generated cuRAND results.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum Ordering {
    /// Best ordering for pseudorandom results.
    PseudoBest = sys::curandOrdering_t::CURAND_ORDERING_PSEUDO_BEST as _,
    /// Default pseudorandom ordering, equivalent to [`Ordering::PseudoBest`].
    PseudoDefault = sys::curandOrdering_t::CURAND_ORDERING_PSEUDO_DEFAULT as _,
    /// Fast lower-quality pseudorandom ordering using a specific seeding pattern.
    PseudoSeeded = sys::curandOrdering_t::CURAND_ORDERING_PSEUDO_SEEDED as _,
    /// Legacy pseudorandom ordering, guaranteed stable across cuRAND releases.
    PseudoLegacy = sys::curandOrdering_t::CURAND_ORDERING_PSEUDO_LEGACY as _,
    /// Device-dependent pseudorandom ordering chosen for best performance.
    PseudoDynamic = sys::curandOrdering_t::CURAND_ORDERING_PSEUDO_DYNAMIC as _,
    /// N-dimensional ordering for quasirandom results.
    QuasiDefault = sys::curandOrdering_t::CURAND_ORDERING_QUASI_DEFAULT as _,
}

impl_enum_conversion!(sys::curandOrdering_t, Ordering);

impl_enum_display!(Ordering, {
    Self::PseudoBest => "CURAND_ORDERING_PSEUDO_BEST",
    Self::PseudoDefault => "CURAND_ORDERING_PSEUDO_DEFAULT",
    Self::PseudoSeeded => "CURAND_ORDERING_PSEUDO_SEEDED",
    Self::PseudoLegacy => "CURAND_ORDERING_PSEUDO_LEGACY",
    Self::PseudoDynamic => "CURAND_ORDERING_PSEUDO_DYNAMIC",
    Self::QuasiDefault => "CURAND_ORDERING_QUASI_DEFAULT",
});

/// Direction-vector table used for Sobol quasirandom generation.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum DirectionVectorSet {
    /// 32-bit Joe-Kuo direction vectors for up to 20,000 dimensions.
    JoeKuo6_32 = sys::curandDirectionVectorSet_t::CURAND_DIRECTION_VECTORS_32_JOEKUO6 as _,
    /// Scrambled 32-bit Joe-Kuo direction vectors for up to 20,000 dimensions.
    ScrambledJoeKuo6_32 =
        sys::curandDirectionVectorSet_t::CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 as _,
    /// 64-bit Joe-Kuo direction vectors for up to 20,000 dimensions.
    JoeKuo6_64 = sys::curandDirectionVectorSet_t::CURAND_DIRECTION_VECTORS_64_JOEKUO6 as _,
    /// Scrambled 64-bit Joe-Kuo direction vectors for up to 20,000 dimensions.
    ScrambledJoeKuo6_64 =
        sys::curandDirectionVectorSet_t::CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 as _,
}

impl_enum_conversion!(sys::curandDirectionVectorSet_t, DirectionVectorSet);

impl_enum_display!(DirectionVectorSet, {
    Self::JoeKuo6_32 => "CURAND_DIRECTION_VECTORS_32_JOEKUO6",
    Self::ScrambledJoeKuo6_32 => "CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6",
    Self::JoeKuo6_64 => "CURAND_DIRECTION_VECTORS_64_JOEKUO6",
    Self::ScrambledJoeKuo6_64 => "CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6",
});

/// Algorithm choice for cuRAND discrete distribution generation.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum Method {
    /// Let cuRAND choose the method based on the distribution parameters.
    ChooseBest = sys::curandMethod_t::CURAND_CHOOSE_BEST as _,
    /// Inversion transform method.
    Inversion = sys::curandMethod_t::CURAND_ITR as _,
    /// Knuth method.
    Knuth = sys::curandMethod_t::CURAND_KNUTH as _,
    /// Hybrid inversion-rejection method.
    HybridInversionRejection = sys::curandMethod_t::CURAND_HITR as _,
    /// M1 method.
    M1 = sys::curandMethod_t::CURAND_M1 as _,
    /// M2 method.
    M2 = sys::curandMethod_t::CURAND_M2 as _,
    /// Binary-search method.
    BinarySearch = sys::curandMethod_t::CURAND_BINARY_SEARCH as _,
    /// Discrete Gaussian method.
    DiscreteGauss = sys::curandMethod_t::CURAND_DISCRETE_GAUSS as _,
    /// Rejection method.
    Rejection = sys::curandMethod_t::CURAND_REJECTION as _,
    /// Device API method.
    DeviceApi = sys::curandMethod_t::CURAND_DEVICE_API as _,
    /// Fast rejection method.
    FastRejection = sys::curandMethod_t::CURAND_FAST_REJECTION as _,
    /// Third method.
    Third = sys::curandMethod_t::CURAND_3RD as _,
    /// Definition method.
    Definition = sys::curandMethod_t::CURAND_DEFINITION as _,
    /// Poisson method.
    Poisson = sys::curandMethod_t::CURAND_POISSON as _,
}

impl_enum_conversion!(sys::curandMethod_t, Method);

impl_enum_display!(Method, {
    Self::ChooseBest => "CURAND_CHOOSE_BEST",
    Self::Inversion => "CURAND_ITR",
    Self::Knuth => "CURAND_KNUTH",
    Self::HybridInversionRejection => "CURAND_HITR",
    Self::M1 => "CURAND_M1",
    Self::M2 => "CURAND_M2",
    Self::BinarySearch => "CURAND_BINARY_SEARCH",
    Self::DiscreteGauss => "CURAND_DISCRETE_GAUSS",
    Self::Rejection => "CURAND_REJECTION",
    Self::DeviceApi => "CURAND_DEVICE_API",
    Self::FastRejection => "CURAND_FAST_REJECTION",
    Self::Third => "CURAND_3RD",
    Self::Definition => "CURAND_DEFINITION",
    Self::Poisson => "CURAND_POISSON",
});

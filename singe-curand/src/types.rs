use num_enum::{IntoPrimitive, TryFromPrimitive};

use singe_core::{impl_enum_conversion, impl_enum_display};

use singe_curand_sys as sys;

/// cuRAND random number generator kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
#[repr(u32)]
pub enum RngType {
    /// Internal test generator type.
    Test = sys::curandRngType_t::CURAND_RNG_TEST as _,
    /// Default pseudorandom generator.
    PseudoDefault = sys::curandRngType_t::CURAND_RNG_PSEUDO_DEFAULT as _,
    /// XORWOW pseudorandom generator.
    PseudoXorwow = sys::curandRngType_t::CURAND_RNG_PSEUDO_XORWOW as _,
    /// `MRG32k3a` pseudorandom generator.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
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

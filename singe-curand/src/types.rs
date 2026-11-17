use num_enum::{IntoPrimitive, TryFromPrimitive};

use singe_core::impl_enum_display;

use singe_curand_sys as sys;

singe_core::native_enum! {
    /// cuRAND random number generator kind.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[non_exhaustive]
    #[repr(u32)]
    pub enum RngType: sys::curandRngType_t {
        /// Internal test generator type.
        Test = CURAND_RNG_TEST,
        /// Default pseudorandom generator.
        PseudoDefault = CURAND_RNG_PSEUDO_DEFAULT,
        /// XORWOW pseudorandom generator.
        PseudoXorwow = CURAND_RNG_PSEUDO_XORWOW,
        /// `MRG32k3a` pseudorandom generator.
        PseudoMrg32k3a = CURAND_RNG_PSEUDO_MRG32K3A,
        /// Mersenne Twister MTGP32 pseudorandom generator.
        PseudoMtgp32 = CURAND_RNG_PSEUDO_MTGP32,
        /// Mersenne Twister MT19937 pseudorandom generator.
        PseudoMt19937 = CURAND_RNG_PSEUDO_MT19937,
        /// PHILOX-4x32-10 pseudorandom generator.
        PseudoPhilox4_32_10 = CURAND_RNG_PSEUDO_PHILOX4_32_10,
        /// Default quasirandom generator.
        QuasiDefault = CURAND_RNG_QUASI_DEFAULT,
        /// Sobol32 quasirandom generator.
        QuasiSobol32 = CURAND_RNG_QUASI_SOBOL32,
        /// Scrambled Sobol32 quasirandom generator.
        QuasiScrambledSobol32 = CURAND_RNG_QUASI_SCRAMBLED_SOBOL32,
        /// Sobol64 quasirandom generator.
        QuasiSobol64 = CURAND_RNG_QUASI_SOBOL64,
        /// Scrambled Sobol64 quasirandom generator.
        QuasiScrambledSobol64 = CURAND_RNG_QUASI_SCRAMBLED_SOBOL64,
    }
}

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

singe_core::native_enum! {
    /// Memory ordering used by generated cuRAND results.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[non_exhaustive]
    #[repr(u32)]
    pub enum Ordering: sys::curandOrdering_t {
        /// Best ordering for pseudorandom results.
        PseudoBest = CURAND_ORDERING_PSEUDO_BEST,
        /// Default pseudorandom ordering, equivalent to [`Ordering::PseudoBest`].
        PseudoDefault = CURAND_ORDERING_PSEUDO_DEFAULT,
        /// Fast lower-quality pseudorandom ordering using a specific seeding pattern.
        PseudoSeeded = CURAND_ORDERING_PSEUDO_SEEDED,
        /// Legacy pseudorandom ordering, guaranteed stable across cuRAND releases.
        PseudoLegacy = CURAND_ORDERING_PSEUDO_LEGACY,
        /// Device-dependent pseudorandom ordering chosen for best performance.
        PseudoDynamic = CURAND_ORDERING_PSEUDO_DYNAMIC,
        /// N-dimensional ordering for quasirandom results.
        QuasiDefault = CURAND_ORDERING_QUASI_DEFAULT,
    }
}

impl_enum_display!(Ordering, {
    Self::PseudoBest => "CURAND_ORDERING_PSEUDO_BEST",
    Self::PseudoDefault => "CURAND_ORDERING_PSEUDO_DEFAULT",
    Self::PseudoSeeded => "CURAND_ORDERING_PSEUDO_SEEDED",
    Self::PseudoLegacy => "CURAND_ORDERING_PSEUDO_LEGACY",
    Self::PseudoDynamic => "CURAND_ORDERING_PSEUDO_DYNAMIC",
    Self::QuasiDefault => "CURAND_ORDERING_QUASI_DEFAULT",
});

singe_core::native_enum! {
    /// Direction-vector table used for Sobol quasirandom generation.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[non_exhaustive]
    #[repr(u32)]
    pub enum DirectionVectorSet: sys::curandDirectionVectorSet_t {
        /// 32-bit Joe-Kuo direction vectors for up to 20,000 dimensions.
        JoeKuo6_32 = CURAND_DIRECTION_VECTORS_32_JOEKUO6,
        /// Scrambled 32-bit Joe-Kuo direction vectors for up to 20,000 dimensions.
        ScrambledJoeKuo6_32 = CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6,
        /// 64-bit Joe-Kuo direction vectors for up to 20,000 dimensions.
        JoeKuo6_64 = CURAND_DIRECTION_VECTORS_64_JOEKUO6,
        /// Scrambled 64-bit Joe-Kuo direction vectors for up to 20,000 dimensions.
        ScrambledJoeKuo6_64 = CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6,
    }
}

impl_enum_display!(DirectionVectorSet, {
    Self::JoeKuo6_32 => "CURAND_DIRECTION_VECTORS_32_JOEKUO6",
    Self::ScrambledJoeKuo6_32 => "CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6",
    Self::JoeKuo6_64 => "CURAND_DIRECTION_VECTORS_64_JOEKUO6",
    Self::ScrambledJoeKuo6_64 => "CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6",
});

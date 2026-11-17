macro_rules! impl_float_storage {
    ($(#[$meta:meta])* $name:ident, $bits:ty) => {
        $(#[$meta])*
        #[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[repr(transparent)]
        pub struct $name(
            /// Raw storage bits, preserved without conversion or validation.
            pub $bits,
        );

        impl $name {
            /// Wraps storage bits without conversion, masking, or validation.
            pub const fn from_bits(bits: $bits) -> Self {
                Self(bits)
            }

            /// Returns the unchanged storage bits.
            pub const fn to_bits(self) -> $bits {
                self.0
            }
        }
    };
}

impl_float_storage!(
    /// Raw storage for an FP8 E4M3 encoding with four exponent and three fraction bits.
    f8e4m3,
    u8
);

impl_float_storage!(
    /// Raw storage for an FP8 E5M2 encoding with five exponent and two fraction bits.
    f8e5m2,
    u8
);

impl_float_storage!(
    /// Raw storage for an unsigned FP8 E8M0 encoding with eight exponent bits and no fraction bits.
    f8ue8m0,
    u8
);

impl_float_storage!(
    /// Real as an unsigned `nv_fp8_ue5m3`.
    /// Stores the unsigned UE5M3 encoding without converting it.
    f8ue5m3,
    u8
);

impl_float_storage!(
    /// Byte storage for a six-bit E2M3 floating-point encoding, without masking unused bits.
    f6e2m3,
    u8
);

impl_float_storage!(
    /// Byte storage for a six-bit E3M2 floating-point encoding, without masking unused bits.
    f6e3m2,
    u8
);

impl_float_storage!(
    /// Byte storage for a four-bit E2M1 floating-point encoding, without masking unused bits.
    f4e2m1,
    u8
);

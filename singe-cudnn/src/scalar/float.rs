//! Preserve nonfinite scalar bits in human-readable recipes and cache keys.

macro_rules! float_serde {
    ($module:ident, $float:ty, $bits:ty, $digits:literal, $serialize:ident) => {
        pub mod $module {
            use std::fmt;

            use serde::{
                Deserialize, Deserializer, Serializer,
                de::{Error, Unexpected, Visitor},
            };

            pub fn serialize<S: Serializer>(
                value: &$float,
                serializer: S,
            ) -> Result<S::Ok, S::Error> {
                if serializer.is_human_readable() && !value.is_finite() {
                    serializer.serialize_str(&format!(
                        "0x{:0width$x}",
                        value.to_bits(),
                        width = $digits,
                    ))
                } else {
                    serializer.$serialize(*value)
                }
            }

            pub fn deserialize<'de, D: Deserializer<'de>>(
                deserializer: D,
            ) -> Result<$float, D::Error> {
                if deserializer.is_human_readable() {
                    deserializer.deserialize_any(FloatVisitor)
                } else {
                    <$float>::deserialize(deserializer)
                }
            }

            struct FloatVisitor;

            #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, reason = "Serde numeric visitors convert incoming numbers to the requested IEEE dtype")]
            impl<'de> Visitor<'de> for FloatVisitor {
                type Value = $float;

                fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                    write!(
                        formatter,
                        "a number or a {}-digit hexadecimal IEEE-754 bit pattern",
                        $digits,
                    )
                }

                fn visit_f32<E: Error>(self, value: f32) -> Result<Self::Value, E> {
                    Ok(value as $float)
                }

                fn visit_f64<E: Error>(self, value: f64) -> Result<Self::Value, E> {
                    Ok(value as $float)
                }

                fn visit_i64<E: Error>(self, value: i64) -> Result<Self::Value, E> {
                    Ok(value as $float)
                }

                fn visit_u64<E: Error>(self, value: u64) -> Result<Self::Value, E> {
                    Ok(value as $float)
                }

                fn visit_str<E: Error>(self, value: &str) -> Result<Self::Value, E> {
                    let hex = value
                        .strip_prefix("0x")
                        .filter(|hex| hex.len() == $digits)
                        .ok_or_else(|| E::invalid_value(Unexpected::Str(value), &self))?;
                    let bits = <$bits>::from_str_radix(hex, 16)
                        .map_err(|_| E::invalid_value(Unexpected::Str(value), &self))?;
                    Ok(<$float>::from_bits(bits))
                }
            }
        }
    };
}

float_serde!(f32_bits, f32, u32, 8, serialize_f32);

float_serde!(f64_bits, f64, u64, 16, serialize_f64);

pub mod optional_f32 {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    struct Float(#[serde(with = "crate::scalar::float::f32_bits")] f32);

    #[expect(
        clippy::ref_option,
        clippy::trivially_copy_pass_by_ref,
        reason = "Serde custom serializers require a borrowed field value"
    )]
    pub fn serialize<S: Serializer>(value: &Option<f32>, serializer: S) -> Result<S::Ok, S::Error> {
        value.map(Float).serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<f32>, D::Error> {
        Option::<Float>::deserialize(deserializer).map(|value| value.map(|value| value.0))
    }
}

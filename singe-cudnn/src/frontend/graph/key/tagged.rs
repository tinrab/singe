use std::result::Result as StdResult;

use serde::{
    Serialize, Serializer,
    ser::{self, Error as _},
};

use serde_json::Value;

use crate::{backend::tensor::TENSOR_ID_SERDE_NAME, error::Result};

pub const GRAPH: &str = "\0singe:graph";
const PREFIX: &str = "\0singe:";
const TENSOR: &str = "\0singe:tensor:";
const TEXT: &str = "\0singe:text:";

pub fn is_tensor_id(value: &str) -> bool {
    value.starts_with(TENSOR)
}

pub fn unescape(value: &str) -> Option<&str> {
    value.strip_prefix(TEXT)
}

pub fn to_value(value: &impl Serialize) -> Result<Value> {
    Ok(serde_json::to_value(Tagged(value))?)
}

/// Keeps semantic tensor IDs distinguishable from ordinary numeric strings in a private key projection.
/// Normal recipe serialization still emits the original string representation of `TensorId`.
struct Tagged<'a, T: ?Sized>(&'a T);

impl<T: Serialize + ?Sized> Serialize for Tagged<'_, T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> StdResult<S::Ok, S::Error> {
        self.0.serialize(TagSerializer(serializer))
    }
}

struct TagSerializer<S>(S);

struct Compound<C>(C);

macro_rules! forward_values {
    ($($method:ident($value_type:ty);)*) => {$(
        fn $method(self, value: $value_type) -> StdResult<Self::Ok, Self::Error> {
            self.0.$method(value)
        }
    )*};
}

impl<S: Serializer> Serializer for TagSerializer<S> {
    type Ok = S::Ok;

    type Error = S::Error;

    type SerializeSeq = Compound<S::SerializeSeq>;

    type SerializeTuple = Compound<S::SerializeTuple>;

    type SerializeTupleStruct = Compound<S::SerializeTupleStruct>;

    type SerializeTupleVariant = Compound<S::SerializeTupleVariant>;

    type SerializeMap = Compound<S::SerializeMap>;

    type SerializeStruct = Compound<S::SerializeStruct>;

    type SerializeStructVariant = Compound<S::SerializeStructVariant>;

    forward_values! {
        serialize_bool(bool);
        serialize_i8(i8);
        serialize_i16(i16);
        serialize_i32(i32);
        serialize_i64(i64);
        serialize_i128(i128);
        serialize_u8(u8);
        serialize_u16(u16);
        serialize_u32(u32);
        serialize_u64(u64);
        serialize_u128(u128);
        serialize_f32(f32);
        serialize_f64(f64);
        serialize_char(char);
        serialize_bytes(&[u8]);
    }

    fn serialize_str(self, value: &str) -> StdResult<Self::Ok, Self::Error> {
        if value.starts_with(PREFIX) {
            self.0.serialize_str(&format!("{TEXT}{value}"))
        } else {
            self.0.serialize_str(value)
        }
    }

    fn serialize_none(self) -> StdResult<Self::Ok, Self::Error> {
        self.0.serialize_none()
    }

    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> StdResult<Self::Ok, Self::Error> {
        self.0.serialize_some(&Tagged(value))
    }

    fn serialize_unit(self) -> StdResult<Self::Ok, Self::Error> {
        self.0.serialize_unit()
    }

    fn serialize_unit_struct(self, name: &'static str) -> StdResult<Self::Ok, Self::Error> {
        self.0.serialize_unit_struct(name)
    }

    fn serialize_unit_variant(
        self,
        name: &'static str,
        index: u32,
        variant: &'static str,
    ) -> StdResult<Self::Ok, Self::Error> {
        self.0.serialize_unit_variant(name, index, variant)
    }

    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        name: &'static str,
        value: &T,
    ) -> StdResult<Self::Ok, Self::Error> {
        if name == TENSOR_ID_SERDE_NAME {
            let Value::String(id) = serde_json::to_value(value).map_err(S::Error::custom)? else {
                return Err(S::Error::custom("tensor id must serialize as a string"));
            };
            self.0.serialize_str(&format!("{TENSOR}{id}"))
        } else {
            self.0.serialize_newtype_struct(name, &Tagged(value))
        }
    }

    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        name: &'static str,
        index: u32,
        variant: &'static str,
        value: &T,
    ) -> StdResult<Self::Ok, Self::Error> {
        self.0
            .serialize_newtype_variant(name, index, variant, &Tagged(value))
    }

    fn serialize_seq(self, length: Option<usize>) -> StdResult<Self::SerializeSeq, Self::Error> {
        self.0.serialize_seq(length).map(Compound)
    }

    fn serialize_tuple(self, length: usize) -> StdResult<Self::SerializeTuple, Self::Error> {
        self.0.serialize_tuple(length).map(Compound)
    }

    fn serialize_tuple_struct(
        self,
        name: &'static str,
        length: usize,
    ) -> StdResult<Self::SerializeTupleStruct, Self::Error> {
        self.0.serialize_tuple_struct(name, length).map(Compound)
    }

    fn serialize_tuple_variant(
        self,
        name: &'static str,
        index: u32,
        variant: &'static str,
        length: usize,
    ) -> StdResult<Self::SerializeTupleVariant, Self::Error> {
        self.0
            .serialize_tuple_variant(name, index, variant, length)
            .map(Compound)
    }

    fn serialize_map(self, length: Option<usize>) -> StdResult<Self::SerializeMap, Self::Error> {
        self.0.serialize_map(length).map(Compound)
    }

    fn serialize_struct(
        self,
        name: &'static str,
        length: usize,
    ) -> StdResult<Self::SerializeStruct, Self::Error> {
        let graph = name == "Graph";
        let mut compound = self.0.serialize_struct(name, length + usize::from(graph))?;
        if graph {
            ser::SerializeStruct::serialize_field(&mut compound, GRAPH, &true)?;
        }
        Ok(Compound(compound))
    }

    fn serialize_struct_variant(
        self,
        name: &'static str,
        index: u32,
        variant: &'static str,
        length: usize,
    ) -> StdResult<Self::SerializeStructVariant, Self::Error> {
        self.0
            .serialize_struct_variant(name, index, variant, length)
            .map(Compound)
    }

    fn is_human_readable(&self) -> bool {
        self.0.is_human_readable()
    }
}

macro_rules! sequence {
    ($trait:ident, $method:ident) => {
        impl<C: ser::$trait> ser::$trait for Compound<C> {
            type Ok = C::Ok;
            type Error = C::Error;

            fn $method<T: Serialize + ?Sized>(&mut self, value: &T) -> StdResult<(), Self::Error> {
                self.0.$method(&Tagged(value))
            }

            fn end(self) -> StdResult<Self::Ok, Self::Error> {
                self.0.end()
            }
        }
    };
}

sequence!(SerializeSeq, serialize_element);

sequence!(SerializeTuple, serialize_element);

sequence!(SerializeTupleStruct, serialize_field);

sequence!(SerializeTupleVariant, serialize_field);

impl<C: ser::SerializeMap> ser::SerializeMap for Compound<C> {
    type Ok = C::Ok;

    type Error = C::Error;

    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> StdResult<(), Self::Error> {
        self.0.serialize_key(&Tagged(key))
    }

    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> StdResult<(), Self::Error> {
        self.0.serialize_value(&Tagged(value))
    }

    fn end(self) -> StdResult<Self::Ok, Self::Error> {
        self.0.end()
    }
}

macro_rules! structure {
    ($trait:ident) => {
        impl<C: ser::$trait> ser::$trait for Compound<C> {
            type Ok = C::Ok;
            type Error = C::Error;

            fn serialize_field<T: Serialize + ?Sized>(
                &mut self,
                key: &'static str,
                value: &T,
            ) -> StdResult<(), Self::Error> {
                self.0.serialize_field(key, &Tagged(value))
            }

            fn skip_field(&mut self, key: &'static str) -> StdResult<(), Self::Error> {
                self.0.skip_field(key)
            }

            fn end(self) -> StdResult<Self::Ok, Self::Error> {
                self.0.end()
            }
        }
    };
}

structure!(SerializeStruct);

structure!(SerializeStructVariant);

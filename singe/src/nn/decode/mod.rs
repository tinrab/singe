// What a cached decoder is made of: one step's tensors, the pages that place them, the caches blocks extend, and the model trait over them.

mod cache;
mod step;

use std::fmt::Debug;

pub use self::{
    cache::{
        CacheSize, DecodeCache, EncoderDecoderCache, EncoderDecoderCacheSpec, KvCache, KvCacheSpec,
        LatentKvCache, LatentKvCacheSpec, caches_from_inputs, named_cache_inputs,
    },
    step::{DecodeStep, PageLayout, PageTable},
};
use crate::{
    data_type::DataType,
    device::Device,
    error::Result,
    nn::{AttentionInput, ModelState},
    tensor::Tensor,
};

// A fixed set of named tensors a compiled step binds, as inputs or as buffers.
pub trait TensorGroup: Debug + Clone + Send + Sync + 'static {
    fn named_tensors(&self) -> Vec<(String, Tensor)>;
}

impl TensorGroup for () {
    fn named_tensors(&self) -> Vec<(String, Tensor)> {
        Vec::new()
    }
}

impl<T: TensorGroup> TensorGroup for Vec<T> {
    fn named_tensors(&self) -> Vec<(String, Tensor)> {
        self.iter()
            .enumerate()
            .flat_map(|(index, group)| {
                group
                    .named_tensors()
                    .into_iter()
                    .map(move |(name, tensor)| (format!("{index}.{name}"), tensor))
            })
            .collect()
    }
}

// A [`TensorGroup`] a capture closure can rebuild from its aliased inputs, in `named_tensors` order.
pub trait InputGroup: TensorGroup {
    // The number of tensors in the group.
    const COUNT: usize;

    fn from_tensors(tensors: &[Tensor]) -> Result<Self>;
}

// Declares a struct of tensors with its [`TensorGroup`] and [`InputGroup`], so each field is listed once.
macro_rules! input_group {
    ($(#[$meta:meta])* $vis:vis struct $name:ident { $($field:ident),+ $(,)? }) => {
        $(#[$meta])*
        #[derive(Debug, Clone)]
        $vis struct $name {
            $($field: $crate::tensor::Tensor,)+
        }

        impl $crate::nn::TensorGroup for $name {
            fn named_tensors(&self) -> Vec<(String, $crate::tensor::Tensor)> {
                vec![$((stringify!($field).to_owned(), self.$field.clone()),)+]
            }
        }

        impl $crate::nn::InputGroup for $name {
            const COUNT: usize = [$(stringify!($field)),+].len();

            fn from_tensors(tensors: &[$crate::tensor::Tensor]) -> $crate::error::Result<Self> {
                let [$($field),+] = tensors else {
                    return Err($crate::error::Error::missing_input(stringify!($name)));
                };
                Ok(Self { $($field: $field.clone(),)+ })
            }
        }
    };
}

pub(crate) use input_group;

// A layer a cached decoder can run: what it computes and caches, and one step's computation.
//
// `Model<L>` pairs it with its state, so a model holds no state of its own, and another crate can implement this for a layer of its own.
pub trait DecoderModel: Debug + Send + Sync + 'static {
    type Cache: DecodeCache;

    // What the model computes for a capacity and looks up by position, such as a rotary table, or `()` for nothing.
    type Tables: TensorGroup;

    // Computes the tables for `capacity` positions on `device`, in the type of the model's weights.
    // On `Device::Meta` they hold shapes and no values.
    fn tables(&self, data_type: DataType, device: Device, capacity: i64) -> Result<Self::Tables>;

    // Allocates each block's cache of `size` on `device`, in the type of the model's weights.
    fn caches(
        &self,
        data_type: DataType,
        device: Device,
        size: CacheSize,
    ) -> Result<Vec<Self::Cache>>;

    // Runs `step`, returning the logits of each sequence's last token, `[S, V]`, and extending `caches`.
    fn forward(
        &self,
        step: &DecodeStep,
        tables: &Self::Tables,
        caches: &[Self::Cache],
        state: &ModelState,
    ) -> Result<Tensor>;
}

// A block or attention layer of a decoder: it reads and extends a cache of one kind.
pub trait DecodingLayer: Debug + Send + Sync + 'static {
    type Cache: DecodeCache;

    // Runs the layer on one step's `input`.
    fn decode(&self, input: &AttentionInput<'_, Self::Cache>, state: &ModelState)
    -> Result<Tensor>;
}

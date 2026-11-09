mod attention;
mod contiguous;
mod decompose;
mod fold;
mod merge;
mod pipeline;
mod recognize;
mod remap;
mod shape;
mod simplify;
mod validation;

#[cfg(feature = "onnx")]
pub use self::attention::decompose_all_attention;
#[cfg(feature = "cuda")]
pub use self::merge::merged_projection_bytes;
use self::{
    attention::{decompose_scaled_dot_product_attention, fold_grouped_query_attention},
    contiguous::{elide_contiguous_copies, materialize_contiguous_operands},
    decompose::decompose_operations,
    fold::fold_small_constants,
    merge::merge_sibling_projections,
    recognize::recognize_row_operations,
    remap::prune_dead_nodes_with_map,
    shape::infer_value_metadata,
    simplify::remove_redundant_ops_with_map,
};
pub use self::{
    pipeline::{GraphPassKind, GraphPassPipeline},
    remap::remapped_node,
};

#[cfg(test)]
mod tests;

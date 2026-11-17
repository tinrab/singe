//! Protobuf types generated from the ONNX schema by `cargo xtask gen-onnx-proto`.

/// Raw ONNX protobuf module generated from `proto/onnx.proto`.
#[allow(
    clippy::absolute_paths,
    reason = "generated protobuf code uses fully qualified paths"
)]
#[allow(
    clippy::derive_partial_eq_without_eq,
    clippy::doc_markdown,
    clippy::must_use_candidate,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::useless_conversion,
    clippy::used_underscore_items,
    reason = "protobuf docs preserve upstream layout, generated adapters use uniform conversions, and generated enums name their protobuf values, which start with an underscore"
)]
#[allow(missing_docs)]
#[allow(unused_qualifications)]
pub mod onnx;

/// Structure-only ONNX protobuf module generated from `proto/onnx_structure.proto`.
#[allow(
    clippy::absolute_paths,
    reason = "generated protobuf code uses fully qualified paths"
)]
#[allow(
    clippy::derive_partial_eq_without_eq,
    clippy::doc_markdown,
    clippy::must_use_candidate,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::useless_conversion,
    reason = "protobuf docs preserve upstream layout and generated adapters use uniform conversions"
)]
#[allow(missing_docs)]
#[allow(unused_qualifications)]
pub mod onnx_structure;

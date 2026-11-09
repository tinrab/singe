use crate::{backend::registry::Operation as RegistryOperation, graph::test_support::*};

fn infer_and_validate_value_metadata(graph: &Graph) -> Graph {
    let pipeline = GraphPassPipeline::new(vec![
        GraphPassKind::InferValueMetadata,
        GraphPassKind::ValidateValueMetadata,
        GraphPassKind::ValidateNodeContracts,
        GraphPassKind::ValidateOutputs,
    ]);
    let (normalized, _) = pipeline.normalize(graph).unwrap();

    normalized
}

fn infer_value_metadata_error(graph: &Graph) -> Error {
    let pipeline = GraphPassPipeline::new(vec![GraphPassKind::InferValueMetadata]);

    pipeline.normalize(graph).unwrap_err()
}

mod algebra;
mod elementwise;
mod metadata;
mod neural;
mod reduction;
mod shape_ops;
mod views;

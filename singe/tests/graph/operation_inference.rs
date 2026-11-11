use serde_json::Value;

use singe::{
    DataType, Shape,
    backend::registry::{Operation, operation_schema},
    value::ValueSpec,
};

#[test]
fn test_promotion_table_matches_all_supported_pytorch_dtype_and_rank_categories() {
    let promotions: Vec<Value> =
        serde_json::from_str(include_str!("../references/conformance/promotions.json")).unwrap();
    assert_eq!(promotions.len(), 144);
    let data_type = |value: &Value| match value.as_str().unwrap() {
        "bool" => DataType::Bool,
        "int64" => DataType::I64,
        "float16" => DataType::F16,
        "bfloat16" => DataType::Bf16,
        "float32" => DataType::F32,
        "float64" => DataType::F64,
        other => panic!("unknown promotion dtype {other}"),
    };
    for case in &promotions {
        let actual = data_type(&case["lhs_dtype"])
            .promote(
                case["lhs_rank"].as_i64().unwrap(),
                data_type(&case["rhs_dtype"]),
                case["rhs_rank"].as_i64().unwrap(),
            )
            .unwrap();
        assert_eq!(actual, data_type(&case["result"]), "promotion {case}");
    }
}

#[test]
fn test_inference_rejects_invalid_broadcasts_and_reads_numbers_as_truth_values() {
    let numeric = ValueSpec::new(DataType::F32, Shape::create([2]).unwrap());
    let logical = operation_schema(Operation::LogicalAnd)
        .binary_request(numeric.clone(), numeric.clone())
        .unwrap()
        .infer()
        .unwrap();
    assert_eq!(logical.computation_data_type(), DataType::Bool);
    assert_eq!(logical.outputs()[0].data_type(), DataType::Bool);
    let incompatible = ValueSpec::new(DataType::F32, Shape::create([3]).unwrap());
    assert!(
        operation_schema(Operation::Add)
            .binary_request(numeric, incompatible)
            .unwrap()
            .infer()
            .is_err()
    );
}

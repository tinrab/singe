use std::{collections::HashMap, slice};

use serde_json::Value;

use singe::{
    DataType, Device, Shape, Tensor,
    backend::registry::{Operation, operation_schema},
    graph::Graph,
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
fn test_binary_inference_has_no_storage_or_device_dependency() {
    let dimensional = ValueSpec::new(DataType::F32, Shape::create([2, 3]).unwrap());
    let scalar = ValueSpec::new(DataType::F64, Shape::scalar());
    let arithmetic = operation_schema(Operation::Add)
        .binary_request(dimensional.clone(), scalar.clone())
        .unwrap()
        .infer()
        .unwrap();
    assert_eq!(arithmetic.computation_data_type(), DataType::F32);
    assert_eq!(arithmetic.outputs(), slice::from_ref(&dimensional));
    let comparison = operation_schema(Operation::Less)
        .binary_request(dimensional, scalar)
        .unwrap()
        .infer()
        .unwrap();
    assert_eq!(comparison.computation_data_type(), DataType::F32);
    assert_eq!(
        comparison.outputs(),
        &[ValueSpec::new(
            DataType::Bool,
            Shape::create([2, 3]).unwrap()
        )]
    );
    let integers = ValueSpec::new(DataType::I64, Shape::scalar());
    let division = operation_schema(Operation::Div)
        .binary_request(integers.clone(), integers)
        .unwrap()
        .infer()
        .unwrap();
    assert_eq!(
        division.outputs(),
        &[ValueSpec::new(DataType::F32, Shape::scalar())]
    );
}

#[test]
fn test_promotion_distinguishes_rank_zero_from_dimensioned_float_tensors() {
    let input =
        Tensor::from_vec(Device::Cpu, Shape::create([2]).unwrap(), vec![1.0_f32, 2.0]).unwrap();
    let scalar = Tensor::scalar(Device::Cpu, 0.25_f64).unwrap();
    let vector =
        Tensor::from_vec(Device::Cpu, Shape::create([1]).unwrap(), vec![0.25_f64]).unwrap();
    let (captured, _) = Graph::capture(
        [
            ("x", input.clone()),
            ("scalar", scalar.clone()),
            ("vector", vector.clone()),
        ],
        |inputs| {
            Ok(vec![
                inputs[0].add(&inputs[1])?,
                inputs[0].add(&inputs[2])?,
                inputs[0].less(&inputs[2])?,
            ])
        },
    )
    .unwrap();
    let bindings = HashMap::from([
        ("x".into(), input),
        ("scalar".into(), scalar),
        ("vector".into(), vector),
    ]);
    for outputs in [
        captured.run(&bindings).unwrap(),
        captured.compile().unwrap().run(&bindings).unwrap(),
    ] {
        assert_eq!(outputs[0].data_type(), DataType::F32);
        assert_eq!(outputs[0].to_vec::<f32>().unwrap(), [1.25, 2.25]);
        assert_eq!(outputs[1].data_type(), DataType::F64);
        assert_eq!(outputs[1].to_vec::<f64>().unwrap(), [1.25, 2.25]);
        assert_eq!(outputs[2].to_vec::<bool>().unwrap(), [false, false]);
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

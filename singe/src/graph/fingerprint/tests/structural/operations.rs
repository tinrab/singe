use super::*;

#[test]
fn test_graph_fingerprint_changes_for_operation_parameters_outputs_and_constants() {
    let shape = Shape::create(vec![2]).unwrap();

    let scale_two = scalar_graph(&shape, 2.0);
    let scale_three = scalar_graph(&shape, 3.0);

    let mut output_input = scale_two.clone();
    output_input.set_outputs(vec![ValueId::new(0)]).unwrap();

    let const_one = constant_graph(&shape, vec![1.0_f32, 2.0]);
    let const_two = constant_graph(&shape, vec![1.0_f32, 3.0]);

    let base = scale_two.fingerprint().unwrap();
    assert_ne!(base, scale_three.fingerprint().unwrap());
    assert_ne!(base, output_input.fingerprint().unwrap());
    assert_ne!(
        const_one.fingerprint().unwrap(),
        const_two.fingerprint().unwrap()
    );
    assert!(const_one.fingerprint().unwrap().is_cacheable());
}

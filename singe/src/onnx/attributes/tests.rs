use singe_onnx::Attribute;

#[test]
fn test_attribute_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<Attribute>();
}

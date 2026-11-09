use std::fs::File;

#[test]
fn test_file_model_source_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<File>();
}

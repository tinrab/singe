use super::*;

pub fn assert_core_capture_outputs(outputs: &[Tensor]) {
    assert_close!(
        &outputs[0].to_vec::<f32>().unwrap(),
        &[19.0, 22.0, 43.0, 50.0],
        1e-6,
    );
    assert_close!(
        &outputs[1].to_vec::<f32>().unwrap(),
        &[1.25, 2.75, 2.25, 5.75],
        1e-6,
    );
    assert_eq!(outputs[2].to_vec::<i32>().unwrap(), [1, 2]);
    assert_close!(&outputs[3].to_vec::<f32>().unwrap(), &[1.0, 2.0], 1e-6,);
    assert_close!(&outputs[4].to_vec::<f32>().unwrap(), &[2.0, 4.0], 1e-6,);
    assert_close!(
        &outputs[5].to_vec::<f32>().unwrap(),
        &[-1.0, 1.0, 2.0, -1.0],
        1e-6,
    );
    assert_close!(&outputs[6].to_vec::<f32>().unwrap(), &[1.25, 1.75], 1e-6,);
}

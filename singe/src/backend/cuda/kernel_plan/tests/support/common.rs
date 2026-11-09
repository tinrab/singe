use super::*;

pub fn assert_unsupported_target<T>(result: Result<T, Error>, target: UnsupportedTarget) {
    let Err(error) = result else {
        panic!("expected unsupported {target:?}, got ok");
    };
    match error {
        Error::Unsupported { target: actual, .. } => assert_eq!(actual, target),
        error => panic!("expected unsupported {target:?}, got {error:?}"),
    }
}

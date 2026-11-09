macro_rules! tensor_scalar_operator_forwarders {
    ($($method:ident => $function:ident),+ $(,)?) => {
        $(
            #[doc = concat!("Applies the `", stringify!($method), "` scalar operation.")]
            ///
            /// This records a graph operation when called inside graph capture.
            ///
            /// # Errors
            ///
            /// Returns an error if the backend does not support the operation for this tensor.
            pub fn $method(&self, scalar: impl Into<$crate::scalar::Scalar>) -> Result<Self> {
                $function(self, scalar.into())
            }
        )+
    };
}

mod activation;
mod basic;
mod conversion;
mod numeric;
mod shape;
mod unary;

pub use self::{activation::GeluApproximation, conversion::scalar_parameters};

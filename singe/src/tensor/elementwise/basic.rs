use crate::{
    error::Result,
    tensor::{
        Tensor,
        operators::{
            add_tensor_scalar, add_tensors, div_tensor_scalar, div_tensors, mul_tensor_scalar,
            mul_tensors, neg_tensor, pow_tensor_scalar, rdiv_tensor_scalar, rsub_tensor_scalar,
            sub_tensor_scalar, sub_tensors,
        },
    },
};

macro_rules! tensor_basic_unary_operator {
    ($method:ident => $function:ident) => {
        #[doc = concat!("Applies the `", stringify!($method), "` operation elementwise.")]
        ///
        /// This records a graph operation when called inside graph capture.
        ///
        /// # Errors
        ///
        /// Returns an error if the backend does not support this operation for this tensor.
        pub fn $method(&self) -> Result<Self> {
            $function(self)
        }
    };
}

macro_rules! tensor_basic_binary_operators {
    ($($method:ident => $function:ident),+ $(,)?) => {
        $(
            #[doc = concat!("Applies the `", stringify!($method), "` operation elementwise.")]
            ///
            /// This records a graph operation when called inside graph capture.
            ///
            /// # Errors
            ///
            /// Returns an error if shapes, devices, data types, or backend support are invalid.
            pub fn $method(&self, other: &Self) -> Result<Self> {
                $function(self, other)
            }
        )+
    };
}

impl Tensor {
    tensor_basic_unary_operator!(neg => neg_tensor);

    tensor_basic_binary_operators!(
        add => add_tensors,
        sub => sub_tensors,
        mul => mul_tensors,
        div => div_tensors,
    );

    tensor_scalar_operator_forwarders!(
        add_scalar => add_tensor_scalar,
        sub_scalar => sub_tensor_scalar,
        rsub_scalar => rsub_tensor_scalar,
        mul_scalar => mul_tensor_scalar,
        div_scalar => div_tensor_scalar,
        rdiv_scalar => rdiv_tensor_scalar,
        pow_scalar => pow_tensor_scalar,
    );
}

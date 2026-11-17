use std::{
    fmt::{self, Formatter},
    marker::PhantomData,
    mem::{MaybeUninit, align_of, size_of},
    ptr,
};

use crate::{
    memory::{DeviceMemory, ManagedMemory},
    view::{DeviceInputView, DeviceRange, DeviceRepr, DeviceSliceMut, DeviceView, DeviceViewMut},
};

/// Dynamically built CUDA kernel argument list.
///
/// Use this builder when the argument list depends on runtime conditions:
///
/// ```ignore
/// let mut params = KernelParameters::new();
/// params.arg(&input_ptr).arg(&output_ptr).push(len);
/// // SAFETY: the buffers match the kernel's parameters and outlive the launch.
/// unsafe { function.launch_async(&config, params, &stream)? };
/// ```
///
/// For fixed argument lists, passing a tuple of references directly to [`KernelFunction::launch_async`](super::KernelFunction::launch_async) avoids building a dynamic list:
///
/// ```ignore
/// // SAFETY: the buffers match the kernel's parameters and outlive the launch.
/// unsafe { function.launch_async(&config, (&input_ptr, &mut output_ptr, &len), &stream)? };
/// ```
///
/// Borrowed arguments are tied to the lifetime of this list.
/// Owned scalar and pointer arguments pushed with [`Self::push`] or [`Self::owned_arg`] are stored inline when they fit, so ordinary kernel launches do not allocate per argument.
pub struct KernelParameters<'a> {
    arguments: Vec<KernelParameter<'a>>,
}

const INLINE_KERNEL_ARGUMENTS: usize = 16;
const INLINE_KERNEL_ARGUMENT_BYTES: usize = 16;

mod private {
    #[expect(
        unnameable_types,
        reason = "sealed supertrait that other crates must not name or implement"
    )]
    pub trait Sealed {}
}

/// Appends a value to a CUDA kernel parameter list.
///
/// Implementations convert Rust wrapper types into the value a CUDA kernel sees at the ABI boundary, such as a scalar or device pointer.
pub trait PushKernelArg {
    /// Appends this value in the kernel ABI representation selected by its implementation.
    fn push_to(self, params: &mut KernelParameters<'_>);
}

/// Kernel launch arguments accepted by launch and graph-node APIs.
///
/// This sealed trait is implemented for [`KernelParameters`], `()`, and tuples of shared or mutable references up to 16 elements.
pub trait KernelLaunchArgs<'a>: private::Sealed {
    #[doc(hidden)]
    fn with_encoded_arguments<R>(self, f: impl FnOnce(EncodedKernelArgs<'_>) -> R) -> R;
}

/// Encoded CUDA kernel arguments for one launch or graph-node update call.
///
/// This is an implementation detail of [`KernelLaunchArgs`].
/// CUDA receives a temporary array of pointers to encoded argument values; callers should use [`KernelParameters`] or generated launch methods instead of constructing raw argument arrays directly.
#[derive(Debug)]
#[doc(hidden)]
pub struct EncodedKernelArgs<'a> {
    pointers: &'a mut [*mut ()],
}

trait KernelTupleArgument<'a> {
    fn into_kernel_argument_ptr(self) -> *mut ();
}

enum KernelParameter<'a> {
    Borrowed {
        ptr: *mut (),
        _marker: PhantomData<&'a ()>,
    },
    Owned(OwnedKernelArgument),
}

impl fmt::Debug for KernelParameter<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Borrowed { ptr, .. } => f.debug_tuple("Borrowed").field(ptr).finish(),
            Self::Owned(value) => f.debug_tuple("Owned").field(value).finish(),
        }
    }
}

enum OwnedKernelArgument {
    Inline(InlineKernelArgument),
    Boxed(Box<dyn KernelArgumentStorage>),
}

impl fmt::Debug for OwnedKernelArgument {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Inline(value) => f.debug_tuple("Inline").field(value).finish(),
            Self::Boxed(_) => f.debug_tuple("Boxed").finish_non_exhaustive(),
        }
    }
}

trait KernelArgumentStorage {
    fn as_mut_ptr(&mut self) -> *mut ();
}

impl<T> KernelArgumentStorage for T {
    fn as_mut_ptr(&mut self) -> *mut () {
        ptr::from_mut(self).cast()
    }
}

#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct InlineKernelArgument {
    bytes: [MaybeUninit<u8>; INLINE_KERNEL_ARGUMENT_BYTES],
}

impl fmt::Debug for InlineKernelArgument {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("InlineKernelArgument")
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for KernelParameters<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("KernelParameters")
            .field("arguments", &self.arguments.len())
            .finish()
    }
}

impl<'a> KernelParameters<'a> {
    /// Creates an empty kernel argument list.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            arguments: Vec::new(),
        }
    }

    /// Appends a borrowed argument value without converting its Rust representation.
    ///
    /// The value stays borrowed for this list's lifetime.
    /// If it contains a device pointer, the borrow covers the pointer value rather than the pointed-to allocation.
    pub fn arg<T: 'a>(&mut self, value: &'a T) -> &mut Self {
        self.arguments.push(KernelParameter::Borrowed {
            ptr: ptr::from_ref(value).cast_mut().cast::<()>(),
            _marker: PhantomData,
        });
        self
    }

    /// Appends an exclusively borrowed argument value without converting its Rust representation.
    ///
    /// CUDA reads the value as a kernel argument; this does not request a write back to the host argument slot.
    /// A pointer argument still needs the pointee lifetime and access guarantees required by the launch.
    pub fn arg_mut<T: 'a>(&mut self, value: &'a mut T) -> &mut Self {
        self.arguments.push(KernelParameter::Borrowed {
            ptr: ptr::from_mut(value).cast::<()>(),
            _marker: PhantomData,
        });
        self
    }

    /// Pushes a copied kernel argument whose storage is owned by this list.
    ///
    /// Small scalar and pointer values are stored inline.
    /// Larger values fall back to heap storage, while keeping the argument pointee stable until CUDA has copied it during launch or graph-node creation.
    pub fn owned_arg<T: Copy + 'static>(&mut self, value: T) -> &mut Self {
        let value = OwnedKernelArgument::from_value(value);
        self.arguments.push(KernelParameter::Owned(value));
        self
    }

    /// Appends an argument using its [`PushKernelArg`] implementation to encode the kernel ABI value.
    pub fn push<A: PushKernelArg>(&mut self, arg: A) -> &mut Self {
        arg.push_to(self);
        self
    }

    /// Encodes a device input address, including deferred graph inputs.
    ///
    /// This copies metadata only.
    /// The launch contract determines when the pointed-to values must be initialized and how access is ordered.
    pub fn device_input<T: DeviceRepr, S: DeviceRange<T> + ?Sized>(
        &mut self,
        slice: &S,
    ) -> &mut Self {
        // Kernels take the device address for slice-like wrappers; length is a
        // separate scalar argument when the kernel needs it.
        self.owned_arg(slice.as_device_ptr())
    }

    /// Appends a copied device address for a mutable slice.
    ///
    /// This does not append the slice length or retain the allocation.
    /// Keep the allocation alive and preserve exclusive writable access for every execution that uses the copied pointer.
    pub fn device_slice_mut<T: DeviceRepr, S: DeviceSliceMut<T> + ?Sized>(
        &mut self,
        slice: &mut S,
    ) -> &mut Self {
        self.owned_arg(slice.as_mut_ptr())
    }

    fn raw_pointers(&mut self) -> RawKernelPointers {
        RawKernelPointers::from_parameters(self.arguments.as_mut_slice())
    }
}

impl KernelParameter<'_> {
    fn as_mut_ptr(&mut self) -> *mut () {
        match self {
            Self::Borrowed { ptr, .. } => *ptr,
            Self::Owned(value) => value.as_mut_ptr(),
        }
    }
}

impl OwnedKernelArgument {
    fn from_value<T: Copy + 'static>(value: T) -> Self {
        InlineKernelArgument::from_value(value)
            .map_or_else(|| Self::Boxed(Box::new(value)), Self::Inline)
    }

    fn as_mut_ptr(&mut self) -> *mut () {
        match self {
            Self::Inline(value) => value.as_mut_ptr(),
            Self::Boxed(value) => value.as_mut().as_mut_ptr(),
        }
    }
}

impl InlineKernelArgument {
    /// Stores `value` inline, or returns `None` if it does not fit the inline storage.
    fn from_value<T: Copy>(value: T) -> Option<Self> {
        if size_of::<T>() > INLINE_KERNEL_ARGUMENT_BYTES || align_of::<T>() > align_of::<Self>() {
            return None;
        }

        let mut storage = Self {
            bytes: [MaybeUninit::uninit(); INLINE_KERNEL_ARGUMENT_BYTES],
        };
        // SAFETY: the checks above guarantee the storage is large enough and aligned for `T`.
        unsafe {
            ptr::write(storage.as_mut_ptr().cast::<T>(), value);
        }
        Some(storage)
    }

    fn as_mut_ptr(&mut self) -> *mut () {
        self.bytes.as_mut_ptr().cast()
    }
}

enum RawKernelPointers {
    Inline {
        pointers: [*mut (); INLINE_KERNEL_ARGUMENTS],
        len: usize,
    },
    Heap(Vec<*mut ()>),
}

impl RawKernelPointers {
    fn from_parameters(parameters: &mut [KernelParameter<'_>]) -> Self {
        if parameters.len() <= INLINE_KERNEL_ARGUMENTS {
            let mut pointers = [ptr::null_mut(); INLINE_KERNEL_ARGUMENTS];
            for (dst, parameter) in pointers.iter_mut().zip(&mut *parameters) {
                *dst = parameter.as_mut_ptr();
            }
            Self::Inline {
                pointers,
                len: parameters.len(),
            }
        } else {
            Self::Heap(
                parameters
                    .iter_mut()
                    .map(KernelParameter::as_mut_ptr)
                    .collect(),
            )
        }
    }

    fn as_mut_slice(&mut self) -> &mut [*mut ()] {
        match self {
            Self::Inline { pointers, len } => &mut pointers[..*len],
            Self::Heap(pointers) => pointers.as_mut_slice(),
        }
    }
}

impl EncodedKernelArgs<'_> {
    pub(crate) fn as_mut_ptr(&mut self) -> *mut *mut () {
        self.pointers.as_mut_ptr()
    }
}

impl<'a> KernelLaunchArgs<'a> for KernelParameters<'a> {
    fn with_encoded_arguments<R>(mut self, f: impl FnOnce(EncodedKernelArgs<'_>) -> R) -> R {
        let mut pointers = self.raw_pointers();
        f(EncodedKernelArgs {
            pointers: pointers.as_mut_slice(),
        })
    }
}

impl private::Sealed for KernelParameters<'_> {}

impl<'a> KernelLaunchArgs<'a> for &mut KernelParameters<'a> {
    fn with_encoded_arguments<R>(self, f: impl FnOnce(EncodedKernelArgs<'_>) -> R) -> R {
        let mut pointers = self.raw_pointers();
        f(EncodedKernelArgs {
            pointers: pointers.as_mut_slice(),
        })
    }
}

impl private::Sealed for &mut KernelParameters<'_> {}

impl KernelLaunchArgs<'_> for () {
    fn with_encoded_arguments<R>(self, f: impl FnOnce(EncodedKernelArgs<'_>) -> R) -> R {
        let mut pointers: [*mut (); 0] = [];
        f(EncodedKernelArgs {
            pointers: &mut pointers,
        })
    }
}

impl private::Sealed for () {}

macro_rules! impl_kernel_arguments_for_tuple {
    ($($arg:ident),+ $(,)?) => {
        impl<'a, $($arg),+> private::Sealed for ($($arg,)+)
        where
            $($arg: KernelTupleArgument<'a>,)+
        {
        }

        impl<'a, $($arg),+> KernelLaunchArgs<'a> for ($($arg,)+)
        where
            $($arg: KernelTupleArgument<'a>,)+
        {
            fn with_encoded_arguments<R>(self, f: impl FnOnce(EncodedKernelArgs<'_>) -> R) -> R {
                #[allow(non_snake_case, reason = "tuple fields are bound to the macro's uppercase type parameter names")]
                let ($($arg,)+) = self;
                let mut pointers = [
                    $($arg.into_kernel_argument_ptr(),)+
                ];
                f(EncodedKernelArgs {
                    pointers: &mut pointers,
                })
            }
        }
    };
}

impl<'a, T: 'a> KernelTupleArgument<'a> for &'a T {
    fn into_kernel_argument_ptr(self) -> *mut () {
        ptr::from_ref(self).cast_mut().cast()
    }
}

impl<'a, T: 'a> KernelTupleArgument<'a> for &'a mut T {
    fn into_kernel_argument_ptr(self) -> *mut () {
        ptr::from_mut(self).cast()
    }
}

impl_kernel_arguments_for_tuple!(A);

impl_kernel_arguments_for_tuple!(A, B);

impl_kernel_arguments_for_tuple!(A, B, C);

impl_kernel_arguments_for_tuple!(A, B, C, D);

impl_kernel_arguments_for_tuple!(A, B, C, D, E);

impl_kernel_arguments_for_tuple!(A, B, C, D, E, F);

impl_kernel_arguments_for_tuple!(A, B, C, D, E, F, G);

impl_kernel_arguments_for_tuple!(A, B, C, D, E, F, G, H);

impl_kernel_arguments_for_tuple!(A, B, C, D, E, F, G, H, I);

impl_kernel_arguments_for_tuple!(A, B, C, D, E, F, G, H, I, J);

impl_kernel_arguments_for_tuple!(A, B, C, D, E, F, G, H, I, J, K);

impl_kernel_arguments_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L);

impl_kernel_arguments_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M);

impl_kernel_arguments_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N);

impl_kernel_arguments_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O);

impl_kernel_arguments_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P);

macro_rules! impl_push_scalar {
    ($($ty:ty),* $(,)?) => {
        $(
            impl PushKernelArg for $ty {
                fn push_to(self, params: &mut KernelParameters<'_>) {
                    params.owned_arg(self);
                }
            }
        )*
    };
}

impl_push_scalar!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64,
);

impl<T: DeviceRepr> PushKernelArg for &DeviceMemory<T> {
    fn push_to(self, params: &mut KernelParameters<'_>) {
        params.device_input(self);
    }
}

impl<T: DeviceRepr> PushKernelArg for &mut DeviceMemory<T> {
    fn push_to(self, params: &mut KernelParameters<'_>) {
        params.device_slice_mut(self);
    }
}

impl<T: DeviceRepr> PushKernelArg for &ManagedMemory<T> {
    fn push_to(self, params: &mut KernelParameters<'_>) {
        params.device_input(self);
    }
}

impl<T: DeviceRepr> PushKernelArg for &mut ManagedMemory<T> {
    fn push_to(self, params: &mut KernelParameters<'_>) {
        params.device_slice_mut(self);
    }
}

impl<T: DeviceRepr> PushKernelArg for DeviceInputView<'_, T> {
    fn push_to(self, params: &mut KernelParameters<'_>) {
        params.device_input(&self);
    }
}

impl<T: DeviceRepr> PushKernelArg for &DeviceInputView<'_, T> {
    fn push_to(self, params: &mut KernelParameters<'_>) {
        params.device_input(self);
    }
}

impl<T: DeviceRepr> PushKernelArg for DeviceView<'_, T> {
    fn push_to(self, params: &mut KernelParameters<'_>) {
        params.owned_arg(self.as_ptr());
    }
}

impl<T: DeviceRepr> PushKernelArg for &DeviceView<'_, T> {
    fn push_to(self, params: &mut KernelParameters<'_>) {
        params.owned_arg(self.as_ptr());
    }
}

impl<T: DeviceRepr> PushKernelArg for &DeviceViewMut<'_, T> {
    fn push_to(self, params: &mut KernelParameters<'_>) {
        params.owned_arg(self.as_ptr());
    }
}

impl<T: DeviceRepr> PushKernelArg for &mut DeviceViewMut<'_, T> {
    fn push_to(self, params: &mut KernelParameters<'_>) {
        params.owned_arg(self.as_mut_ptr());
    }
}

impl Default for KernelParameters<'_> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct LargeArgument {
        words: [u64; 3],
    }

    #[test]
    fn test_boxed_owned_kernel_argument_points_to_inner_value() {
        let mut argument = OwnedKernelArgument::from_value(LargeArgument { words: [1, 2, 3] });
        assert!(matches!(argument, OwnedKernelArgument::Boxed(_)));

        let expected = match &mut argument {
            OwnedKernelArgument::Boxed(value) => value.as_mut().as_mut_ptr(),
            OwnedKernelArgument::Inline(_) => unreachable!(),
        };

        assert_eq!(argument.as_mut_ptr(), expected);
    }
}

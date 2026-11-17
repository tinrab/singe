//! Raw ownership of cuSPARSE data and analysis descriptors.

use std::sync::Arc;

use singe_cuda::{context::Context as CudaContext, view::DeviceRegion};

/// A transferred opaque native resource and the CUDA context needed to destroy it.
/// External analysis workspaces remain subject to the raw execution contract.
#[derive(Debug)]
pub struct RawResource<H> {
    /// Native resource handle transferred to the caller.
    pub handle: H,
    /// CUDA context required to use and destroy the native resource.
    pub cuda_context: Arc<CudaContext>,
}

/// A transferred native descriptor and its actual CUDA context and buffer borrows.
///
/// Destroy `handle` with its corresponding cuSPARSE destroy function before releasing these fields, or transfer the complete value back to its wrapper.
/// Regions describe available storage, including pending values and stride gaps.
/// Descriptors adopted without storage metadata have an empty region list; their original unsafe lifetime and access obligations still apply.
#[derive(Debug)]
pub struct RawDescriptor<'a, H> {
    /// Native descriptor handle transferred to the caller.
    pub handle: H,
    /// Actual CUDA context associated with the descriptor and its storage.
    pub cuda_context: Arc<CudaContext>,
    /// Borrowed storage ranges retained by the descriptor, including absent optional arrays.
    pub regions: Vec<Option<DeviceRegion<'a>>>,
}

macro_rules! analysis_descriptor {
    (
        $(#[$meta:meta])*
        $name:ident {
            raw: $raw:ty,
            object: $object:literal,
            create: $create:path,
            destroy: $destroy:path $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Debug)]
        pub struct $name {
            handle: $raw,
            cuda_context: ::std::sync::Arc<::singe_cuda::context::Context>,
        }

        // SAFETY: cuSPARSE lets any thread use and destroy the descriptor one call at a time, and `&self` exposes only its handle and context, so the owner moves between threads; it is not `Sync`.
        unsafe impl Send for $name {}

        impl $name {
            /// Creates native analysis state in the CUDA context of `context`.
            ///
            /// # Errors
            ///
            /// Returns an error if the context cannot be bound, cuSPARSE cannot create the descriptor, or it returns a null handle.
            pub fn create(context: &$crate::context::Context) -> $crate::error::Result<Self> {
                context.bind()?;
                let mut handle = ::std::ptr::null_mut();
                // SAFETY: the output location is valid for writes and the context is bound.
                unsafe {
                    $crate::error::try_ffi!($create(&raw mut handle))?;
                }

                if handle.is_null() {
                    return Err($crate::error::Error::NullHandle);
                }

                Ok(Self {
                    handle,
                    cuda_context: ::std::sync::Arc::clone(context.cuda_context()),
                })
            }

            /// Adopts a native descriptor and the CUDA context it was created in.
            ///
            /// # Safety
            ///
            /// `raw.handle` must be a live, uniquely owned descriptor created in `raw.cuda_context`.
            /// Preserve all analysis bindings and workspaces through their final dependent operation.
            ///
            /// # Errors
            ///
            /// Returns an error if the handle is null.
            pub unsafe fn from_raw(
                raw: $crate::descriptor::RawResource<$raw>,
            ) -> $crate::error::Result<Self> {
                if raw.handle.is_null() {
                    return Err($crate::error::Error::NullHandle);
                }

                Ok(Self {
                    handle: raw.handle,
                    cuda_context: raw.cuda_context,
                })
            }

            /// Transfers the native descriptor and its CUDA context to the caller without destroying it.
            #[must_use]
            pub fn into_raw(self) -> $crate::descriptor::RawResource<$raw> {
                let this = ::std::mem::ManuallyDrop::new(self);
                $crate::descriptor::RawResource {
                    handle: this.handle,
                    // SAFETY: `this` is never dropped, so the context moves out exactly once.
                    cuda_context: unsafe { ::std::ptr::read(&raw const this.cuda_context) },
                }
            }

            /// Returns the native descriptor without transferring ownership.
            #[must_use]
            pub fn as_raw(&self) -> $raw {
                self.handle
            }

            /// Returns the CUDA context the descriptor was created in.
            #[must_use]
            pub fn cuda_context(&self) -> &::std::sync::Arc<::singe_cuda::context::Context> {
                &self.cuda_context
            }

            pub(crate) fn ensure_context(
                &self,
                context: &$crate::context::Context,
            ) -> $crate::error::Result<()> {
                if self.cuda_context.as_ref() != context.cuda_context().as_ref() {
                    return Err($crate::error::Error::context_mismatch($object));
                }
                Ok(())
            }
        }

        impl Drop for $name {
            fn drop(&mut self) {
                if let Err(error) = self.cuda_context.bind() {
                    ::singe_core::report_cleanup_failure(
                        "bind the cuda context to destroy",
                        $object,
                        &error,
                    );
                    return;
                }

                // SAFETY: the descriptor is live, uniquely owned, and its context is bound.
                if let Err(error) = unsafe { $crate::error::try_ffi!($destroy(self.handle)) } {
                    ::singe_core::report_cleanup_failure("destroy", $object, &error);
                }
            }
        }
    };
}

pub(crate) use analysis_descriptor;

//! Exact-signature access to header-declared APIs absent from some library builds.
#![expect(non_snake_case, reason = "names mirror the upstream C API")]

use std::{ffi::c_int, sync::OnceLock};

use libloading::Library;
use singe_npp_sys as sys;

fn library() -> Option<&'static Library> {
    static LIBRARY: OnceLock<Option<Library>> = OnceLock::new();
    // Keep the library loaded for the lifetime of every cached function pointer.
    LIBRARY
        // SAFETY: loading the NPP library runs only its initializers, which have no preconditions.
        .get_or_init(|| unsafe {
            Library::new(if cfg!(target_os = "windows") {
                "nppicc64_13.dll"
            } else {
                "libnppicc.so.13"
            })
            .ok()
        })
        .as_ref()
}

macro_rules! optional_api {
    ($name:ident($($argument:ident: $ty:ty),* $(,)?) -> $result:ty) => {
        pub unsafe fn $name($($argument: $ty),*) -> $result {
            type Function = unsafe extern "C" fn($($ty),*) -> $result;
            static FUNCTION: OnceLock<Option<Function>> = OnceLock::new();
            let Some(function) = FUNCTION.get_or_init(|| {
                let library = library()?;
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { library.get::<Function>(concat!(stringify!($name), "\0").as_bytes()).ok().map(|symbol| *symbol) }
            }) else { return sys::NppStatus::NPP_NOT_SUPPORTED_MODE_ERROR; };
            // SAFETY: the caller upholds this function's safety contract.
            unsafe { function($($argument),*) }
        }
    };
}

optional_api!(
    nppiI444_8u_ToUYVP_10u_P3C3R_Ctx(
        pSrc: *const *const sys::Npp8u,
        nSrcStep: c_int,
        oSrcOffset: sys::NppiPoint,
        pDst: *mut sys::Npp8u,
        nDstStep: c_int,
        oSizeROI: sys::NppiSize,
        nppStreamCtx: sys::NppStreamContext,
    ) -> sys::NppStatus
);

optional_api!(
    nppiNV12_8u_ToUYVP_10u_P2C3R_Ctx(
        pSrc: *const *const sys::Npp8u,
        nSrcStep: *const c_int,
        oSrcOffset: sys::NppiPoint,
        pDst: *mut sys::Npp8u,
        nDstStep: c_int,
        oSizeROI: sys::NppiSize,
        nppStreamCtx: sys::NppStreamContext,
    ) -> sys::NppStatus
);

optional_api!(
    nppiNV16_8u_ToUYVP_10u_P2C3R_Ctx(
        pSrc: *const *const sys::Npp8u,
        nSrcStep: *const c_int,
        oSrcOffset: sys::NppiPoint,
        pDst: *mut sys::Npp8u,
        nDstStep: c_int,
        oSizeROI: sys::NppiSize,
        nppStreamCtx: sys::NppStreamContext,
    ) -> sys::NppStatus
);

optional_api!(
    nppiP010_16u_ToUYVP_10u_P3C3R_Ctx(
        pSrc: *const *const sys::Npp16u,
        nSrcStep: c_int,
        oSrcOffset: sys::NppiPoint,
        pDst: *mut sys::Npp8u,
        nDstStep: c_int,
        oSizeROI: sys::NppiSize,
        nppStreamCtx: sys::NppStreamContext,
    ) -> sys::NppStatus
);

optional_api!(
    nppiP208_8u_ToUYVP_10u_C3C3R_Ctx(
        pSrc: *const sys::Npp8u,
        nSrcStep: c_int,
        oSrcOffset: sys::NppiPoint,
        pDst: *mut sys::Npp8u,
        nDstStep: c_int,
        oSizeROI: sys::NppiSize,
        nppStreamCtx: sys::NppStreamContext,
    ) -> sys::NppStatus
);

optional_api!(
    nppiP210_16u_ToUYVP_10u_P2C3R_Ctx(
        pSrc: *const *const sys::Npp16u,
        nSrcStep: *const c_int,
        oSrcOffset: sys::NppiPoint,
        pDst: *mut sys::Npp8u,
        nDstStep: c_int,
        oSizeROI: sys::NppiSize,
        nppStreamCtx: sys::NppStreamContext,
    ) -> sys::NppStatus
);

optional_api!(
    nppiUYVP_10u_ToI444_8u_C3P3R_Ctx(
        pSrc: *const sys::Npp8u,
        nSrcStep: c_int,
        oSrcOffset: sys::NppiPoint,
        pDst: *const *mut sys::Npp8u,
        nDstStep: c_int,
        oSizeROI: sys::NppiSize,
        nppStreamCtx: sys::NppStreamContext,
    ) -> sys::NppStatus
);

optional_api!(
    nppiUYVP_10u_ToNV16_8u_C3P2R_Ctx(
        pSrc: *const sys::Npp8u,
        nSrcStep: c_int,
        oSrcOffset: sys::NppiPoint,
        pDst: *mut *mut sys::Npp8u,
        nDstStep: *const c_int,
        oSizeROI: sys::NppiSize,
        nppStreamCtx: sys::NppStreamContext,
    ) -> sys::NppStatus
);

optional_api!(
    nppiUYVP_10u_ToP208_8u_C3C3R_Ctx(
        pSrc: *const sys::Npp8u,
        srcStep: c_int,
        oSrcOffset: sys::NppiPoint,
        pDst: *mut sys::Npp8u,
        dstStep: c_int,
        oSizeROI: sys::NppiSize,
        nppStreamCtx: sys::NppStreamContext,
    ) -> sys::NppStatus
);

optional_api!(
    nppiUYVP_10u_ToV210_10u_C3C3R_Ctx(
        pSrc: *const sys::Npp8u,
        nSrcStep: c_int,
        oSrcOffset: sys::NppiPoint,
        pDst: *mut sys::Npp8u,
        nDstStep: c_int,
        oSizeROI: sys::NppiSize,
        nppStreamCtx: sys::NppStreamContext,
    ) -> sys::NppStatus
);

optional_api!(
    nppiUYVP_10u_ToY410_10u_C3C3R_Ctx(
        pSrc: *const sys::Npp8u,
        nSrcStep: c_int,
        oSrcOffset: sys::NppiPoint,
        pDst: *mut sys::Npp8u,
        nDstStep: c_int,
        oSizeROI: sys::NppiSize,
        nppStreamCtx: sys::NppStreamContext,
    ) -> sys::NppStatus
);

optional_api!(
    nppiUYVP_10u_ToYUV444_8u_C3C3R_Ctx(
        pSrc: *const sys::Npp8u,
        nSrcStep: c_int,
        oSrcOffset: sys::NppiPoint,
        pDst: *mut sys::Npp8u,
        nDstStep: c_int,
        oSizeROI: sys::NppiSize,
        nppStreamCtx: sys::NppStreamContext,
    ) -> sys::NppStatus
);

optional_api!(
    nppiV210_10u_ToUYVP_10u_C3C3R_Ctx(
        pSrc: *const sys::Npp8u,
        nSrcStep: c_int,
        oSrcOffset: sys::NppiPoint,
        pDst: *mut sys::Npp8u,
        nDstStep: c_int,
        oSizeROI: sys::NppiSize,
        nppStreamCtx: sys::NppStreamContext,
    ) -> sys::NppStatus
);

optional_api!(
    nppiY410_10u_ToUYVP_10u_C3C3R_Ctx(
        pSrc: *const sys::Npp16u,
        nSrcStep: c_int,
        oSrcOffset: sys::NppiPoint,
        pDst: *mut sys::Npp8u,
        nDstStep: c_int,
        oSizeROI: sys::NppiSize,
        nppStreamCtx: sys::NppStreamContext,
    ) -> sys::NppStatus
);

optional_api!(
    nppiYUV444_8u_ToUYVP_10u_C3C3R_Ctx(
        pSrc: *const sys::Npp8u,
        nSrcStep: c_int,
        oSrcOffset: sys::NppiPoint,
        pDst: *mut sys::Npp8u,
        nDstStep: c_int,
        oSizeROI: sys::NppiSize,
        nppStreamCtx: sys::NppStreamContext,
    ) -> sys::NppStatus
);

//! Process-global encryption for NCCL-owned TCP sockets.

use std::{ffi::CStr, fmt, ptr};

use crate::{
    error::{Error, Result},
    sys, try_ffi,
};

/// Configures process-global encryption for NCCL-owned TCP sockets.
/// All processes that communicate with each other must use the same PSK and keep it unchanged for the job.
#[derive(Clone, Copy, Default)]
pub enum Encryption<'a> {
    /// Use plaintext TCP sockets.
    #[default]
    Disabled,
    /// Request TLS encryption using the supplied PSK.
    /// The key must contain at least 32 bytes, excluding its terminating nul.
    Psk(&'a CStr),
}

impl fmt::Debug for Encryption<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Disabled => f.write_str("Disabled"),
            Self::Psk(_) => f.write_str("Psk([redacted])"),
        }
    }
}

/// Configures process-global encryption for NCCL-owned TCP sockets.
///
/// Applications should call this function before [`crate::unique_id`] or any other NCCL operation that may create network connections.
/// All processes that communicate with each other must use the same PSK and keep it unchanged for the job.
/// NCCL copies the key during this call.
/// TLS requires an NCCL build with the OpenSSL 3 backend and its runtime libraries.
///
/// # Errors
/// Returns an error if the key is shorter than 32 bytes or NCCL rejects the configuration.
pub fn configure(encryption: Encryption<'_>) -> Result<()> {
    let (mode, psk) = match encryption {
        Encryption::Disabled => (
            sys::ncclEncryptionMode_t::NCCL_ENCRYPTION_MODE_NONE,
            ptr::null(),
        ),
        Encryption::Psk(key) => {
            if key.to_bytes().len() < 32 {
                return Err(Error::out_of_range("encryption key length"));
            }
            (
                sys::ncclEncryptionMode_t::NCCL_ENCRYPTION_MODE_PSK,
                key.as_ptr(),
            )
        }
    };
    let config = sys::ncclEncryptionConfig_t {
        size: size_of::<sys::ncclEncryptionConfig_t>() as _,
        magic: sys::NCCL_API_MAGIC,
        version: sys::NCCL_VERSION_CODE,
        mode: mode as _,
        psk,
    };
    // SAFETY: `ncclSetEncryption` receives pointers into buffers borrowed for the call.
    unsafe { try_ffi!(sys::ncclSetEncryption(&raw const config)) }
}

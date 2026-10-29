//! Discovery and selection of producer-authored embedded artifact bundles.
//! Loading and typed preparation reuse the ordinary context, module and kernel APIs.

use std::{cmp::Reverse, env, ffi::CString, fs, path::Path, sync::Arc};

pub use oxide_artifacts::{
    ArtifactCompileOptions, ArtifactDebugPolicy, ArtifactEntryKind, ArtifactPayloadKind,
    OwnedArtifactBundle, OwnedArtifactEntry, OwnedArtifactPayload,
};

use crate::{
    context::Context,
    device::DeviceAttribute,
    error::{Error, Result, Status},
    kernel::launch::{KernelSignature, PreparedLaunch},
    module::{LaunchConfig, Module, ModuleImage},
};

/// Original producer metadata retained by the native module and its prepared functions.
#[derive(Debug, Clone)]
pub struct ArtifactMetadata {
    /// Producer-assigned artifact bundle name.
    pub name: String,
    /// Producer target string, such as `sm_90` or `compute_90`.
    pub target: String,
    /// Producer compilation policy, including FMA contraction and debug information.
    pub compile_options: ArtifactCompileOptions,
    /// Producer-declared kernel symbols and their entry metadata.
    pub entries: Vec<OwnedArtifactEntry>,
}

impl ArtifactMetadata {
    /// Copies the bundle's producer metadata without retaining its code payloads.
    #[must_use]
    pub fn from_bundle(bundle: &OwnedArtifactBundle) -> Self {
        Self {
            name: bundle.name.clone(),
            target: bundle.target.clone(),
            compile_options: bundle.compile_options,
            entries: bundle.entries.clone(),
        }
    }

    /// Returns the first producer entry whose symbol equals `symbol`, or `None` if no entry matches.
    #[must_use]
    pub fn entry(&self, symbol: &str) -> Option<&OwnedArtifactEntry> {
        self.entries.iter().find(|entry| entry.symbol == symbol)
    }
}

/// Device-code format of a precompiled image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ImageKind {
    /// PTX assembly, finalized by the driver or native linker before execution.
    Ptx,
    /// A compiled CUDA device binary.
    Cubin,
    /// A CUDA fat binary containing one or more device-code images.
    Fatbin,
}

/// Precompiled bytes and producer metadata, using the existing module-image storage.
#[derive(Debug, Clone)]
pub struct PrecompiledImage<'a> {
    /// Declared device-code format of `image`.
    pub kind: ImageKind,
    /// Borrowed or owned precompiled device-code bytes.
    pub image: ModuleImage<'a>,
    /// Producer compilation policy and entry descriptions retained with the loaded module.
    pub metadata: ArtifactMetadata,
}

impl<'a> PrecompiledImage<'a> {
    /// Combines the image bytes, declared format, and producer metadata without validating them.
    #[must_use]
    pub fn new(kind: ImageKind, image: ModuleImage<'a>, metadata: ArtifactMetadata) -> Self {
        Self {
            kind,
            image,
            metadata,
        }
    }
}

/// Owned producer artifact bundles discovered in a host object file or artifact section.
///
/// Loading selects candidate PTX or cubin payloads for a context's device and retains the selected bundle's producer metadata with the module.
#[derive(Debug)]
pub struct ArtifactLibrary {
    bundles: Vec<OwnedArtifactBundle>,
}

impl ArtifactLibrary {
    /// Stores the supplied artifact bundles without parsing or validating their payloads.
    #[must_use]
    pub fn new(bundles: Vec<OwnedArtifactBundle>) -> Self {
        Self { bundles }
    }

    /// Reads embedded artifact bundles from the current executable.
    ///
    /// # Errors
    ///
    /// Returns an error if the current executable path cannot be determined, its contents cannot be read, or object-file or artifact-section parsing fails.
    pub fn from_current_exe() -> Result<Self> {
        Self::from_binary_path(env::current_exe()?)
    }

    /// Reads embedded artifact bundles from the host object file at `path`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or its object-file or artifact-section contents cannot be parsed.
    pub fn from_binary_path(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_binary_bytes(&fs::read(path)?)
    }

    /// Reads artifact bundles from the embedded `.oxart` sections of a host object image.
    ///
    /// A valid object without artifact sections produces an empty library. Returned bundles own their metadata and payload bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the object file, a section name or contents, or an embedded artifact bundle cannot be parsed.
    pub fn from_binary_bytes(bytes: &[u8]) -> Result<Self> {
        Ok(Self::new(
            oxide_artifacts::read_artifact_bundles_from_object_bytes(bytes)?,
        ))
    }

    /// Parses artifact bundles from the contents of an artifact section.
    ///
    /// Accepts concatenated bundles and trailing zero padding. Returned bundles own their metadata and payload bytes.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed or truncated bundle data, unsupported artifact versions, invalid metadata, or invalid payload descriptions.
    pub fn from_section(bytes: &[u8]) -> Result<Self> {
        Ok(Self::new(
            oxide_artifacts::parse_artifact_section(bytes)?
                .into_iter()
                .map(Into::into)
                .collect(),
        ))
    }

    /// Borrows all stored bundles in their discovery order.
    #[must_use]
    pub fn bundles(&self) -> &[OwnedArtifactBundle] {
        &self.bundles
    }

    /// Tries native binaries before PTX, preferring newer and more specific targets.
    /// Target metadata only filters candidates; CUDA and nvJitLink determine image compatibility.
    ///
    /// # Errors
    ///
    /// Returns an error if the device compute capability cannot be queried or converted. Returns `Error::NoCompatibleArtifact` if no named bundle has an eligible PTX or cubin payload. Image-compatibility failures try the next candidate; if every candidate fails, the last such error is returned. Other loading or policy-finalization failures return immediately.
    pub fn load(&self, context: &Arc<Context>, name: &str) -> Result<Module> {
        let capability = capability(context)?;
        let mut candidates = Vec::new();
        for bundle in self.bundles.iter().filter(|bundle| bundle.name == name) {
            let Some(target) = Target::parse(&bundle.target) else {
                continue;
            };
            for payload in &bundle.payloads {
                let kind = match payload.kind {
                    ArtifactPayloadKind::Ptx => ImageKind::Ptx,
                    ArtifactPayloadKind::Cubin => ImageKind::Cubin,
                    _ => continue,
                };
                if target.is_candidate(capability, kind) {
                    let score = (
                        u8::from(kind == ImageKind::Cubin),
                        target.capability,
                        target.specificity(),
                    );
                    candidates.push((score, bundle, payload, kind));
                }
            }
        }
        candidates.sort_by_key(|(score, ..)| Reverse(*score));
        let mut last_error = None;
        for (_, bundle, payload, kind) in candidates {
            let image = PrecompiledImage::new(
                kind,
                ModuleImage::new(&payload.bytes),
                ArtifactMetadata::from_bundle(bundle),
            );
            match context.load_precompiled(&image) {
                Ok(module) => return Ok(module),
                Err(error) if can_try_next_image(&error) => last_error = Some(error),
                Err(error) => return Err(error),
            }
        }
        Err(last_error.unwrap_or_else(|| Error::NoCompatibleArtifact {
            name: name.to_owned().into(),
            major: capability.0,
            minor: capability.1,
        }))
    }
}

fn capability(context: &Context) -> Result<(u32, u32)> {
    let major = context
        .device()
        .attribute(DeviceAttribute::ComputeCapabilityMajor)?;
    let minor = context
        .device()
        .attribute(DeviceAttribute::ComputeCapabilityMinor)?;
    Ok((
        u32::try_from(major).map_err(|_| Error::InvalidValue)?,
        u32::try_from(minor).map_err(|_| Error::InvalidValue)?,
    ))
}

#[derive(Debug, Clone, Copy)]
struct Target {
    capability: (u32, u32),
    suffix: Option<char>,
}

impl Target {
    fn parse(target: &str) -> Option<Self> {
        let target = target
            .strip_prefix("sm_")
            .or_else(|| target.strip_prefix("compute_"))?;
        let digits = target.trim_end_matches(['a', 'f']);
        if digits.is_empty() || !digits.bytes().all(|digit| digit.is_ascii_digit()) {
            return None;
        }
        let code: u32 = digits.parse().ok()?;
        let suffix = match &target[digits.len()..] {
            "" => None,
            "a" => Some('a'),
            "f" => Some('f'),
            _ => return None,
        };
        Some(Self {
            capability: (code / 10, code % 10),
            suffix,
        })
    }

    fn is_candidate(self, device: (u32, u32), kind: ImageKind) -> bool {
        match self.suffix {
            Some('a') => self.capability == device,
            Some('f') => self.capability.0 == device.0 && self.capability <= device,
            _ => {
                self.capability <= device
                    && (kind == ImageKind::Ptx || self.capability.0 == device.0)
            }
        }
    }

    fn specificity(self) -> u8 {
        match self.suffix {
            Some('a') => 2,
            Some('f') => 1,
            _ => 0,
        }
    }
}

fn can_try_next_image(error: &Error) -> bool {
    match error {
        Error::Cuda {
            code:
                Status::InvalidImage
                | Status::InvalidPtx
                | Status::NoBinaryForGpu
                | Status::UnsupportedPtxVersion,
            ..
        } => true,
        #[cfg(feature = "nvjitlink_13_4")]
        Error::NvJitLink { code, .. } => {
            use crate::nvjitlink::Status as LinkStatus;
            matches!(
                code,
                LinkStatus::InvalidInput
                    | LinkStatus::UnrecognizedInput
                    | LinkStatus::IncorrectInputType
                    | LinkStatus::PtxCompile
                    | LinkStatus::ArchMismatch
                    | LinkStatus::UnsupportedArch
                    | LinkStatus::UnrecognizedArch
                    | LinkStatus::OutdatedLibrary
            )
        }
        Error::UnsupportedArchitecture { .. } => true,
        _ => false,
    }
}

impl Context {
    /// Loads a precompiled image while preserving producer policy and entry metadata.
    /// PTX policies requiring finalization use the existing nvJitLink integration.
    /// CUBIN and executable fatbin policies describe compilation already performed by the producer.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidValue` for an empty image, or a NUL-byte error for PTX with an interior NUL. Also propagates context binding and module loading errors. PTX with a nondefault compilation policy requires `nvjitlink_13_4`; without it, returns `Error::Compiler`. With the feature enabled, device capability queries, target architecture parsing, native linker creation or finalization, and artifact-name conversion can also fail.
    pub fn load_precompiled(self: &Arc<Self>, image: &PrecompiledImage<'_>) -> Result<Module> {
        if image.image.as_bytes().is_empty() {
            return Err(Error::InvalidValue);
        }
        let terminated;
        let bytes = if image.kind == ImageKind::Ptx {
            let bytes = image.image.as_bytes();
            terminated = CString::new(bytes.strip_suffix(&[0]).unwrap_or(bytes))?;
            terminated.as_bytes_with_nul()
        } else {
            image.image.as_bytes()
        };
        let policy = image.metadata.compile_options;
        if image.kind == ImageKind::Ptx && policy != ArtifactCompileOptions::default() {
            load_with_policy(self, image, bytes)
        } else {
            self.load_module_with_metadata(
                &ModuleImage::new(bytes),
                Some(Arc::new(image.metadata.clone())),
            )
        }
    }
}

#[cfg(feature = "nvjitlink_13_4")]
fn load_with_policy(
    context: &Arc<Context>,
    image: &PrecompiledImage<'_>,
    bytes: &[u8],
) -> Result<Module> {
    use crate::{
        architecture::GpuArchitecture,
        nvjitlink::{InputType, LinkOptions, Linker},
    };
    let (major, minor) = capability(context)?;
    let suffix = Target::parse(&image.metadata.target)
        .and_then(|target| target.suffix)
        .map_or(String::new(), |suffix| suffix.to_string());
    let architecture: GpuArchitecture = format!("sm_{major}{minor}{suffix}").parse()?;
    let mut options = LinkOptions::new(architecture);
    let policy = image.metadata.compile_options;
    options.raw_options.push(format!(
        "-fma={}",
        u8::from(policy.fma_contraction_enabled())
    ));
    match policy.debug_policy() {
        ArtifactDebugPolicy::None => (),
        ArtifactDebugPolicy::LineTables => options.line_info = true,
        ArtifactDebugPolicy::Full => options.raw_options.push("-g".into()),
    }
    let mut linker = Linker::create(options)?;
    linker.add_data(InputType::Ptx, bytes, &image.metadata.name)?;
    let binary = linker.complete()?;
    context.load_module_with_metadata(
        &ModuleImage::new(binary.as_bytes()),
        Some(Arc::new(image.metadata.clone())),
    )
}

#[cfg(not(feature = "nvjitlink_13_4"))]
fn load_with_policy(_: &Arc<Context>, _: &PrecompiledImage<'_>, _: &[u8]) -> Result<Module> {
    Err(Error::Compiler {
        message: "artifact compile policy requires the nvjitlink_13_4 feature".into(),
    })
}

impl Module {
    /// Returns the retained producer metadata, or `None` for modules loaded without it.
    #[must_use]
    pub fn artifact_metadata(&self) -> Option<&ArtifactMetadata> {
        self.inner.artifact_metadata.as_deref()
    }

    /// Binds an existing image entry to the ordinary typed launch contract.
    ///
    /// # Safety
    /// `K` must describe this symbol's ABI, indexing and semantic requirements.
    /// Producer entry metadata does not prove field layout, bounds, initialization or alias permission.
    ///
    /// # Errors
    ///
    /// Propagates errors from looking up `symbol` and validating the typed launch. Validation can fail if native parameter sizes or offsets disagree with `K`, parameter layout overflows, the launch violates the signature's requirements or device limits, or CUDA cannot query the required kernel or device attributes.
    pub unsafe fn prepare<K: KernelSignature>(
        &self,
        symbol: &str,
        config: LaunchConfig,
    ) -> Result<PreparedLaunch<K>> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { PreparedLaunch::create(&self.function(symbol)?, config) }
    }
}

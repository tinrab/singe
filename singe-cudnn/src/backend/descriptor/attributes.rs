use std::ptr;

use singe_core::string_from_c_chars;

use singe_cudnn_sys as sys;

use crate::{
    backend::{
        attribute::{BackendAttributeName, BackendAttributeType, BackendEnum},
        descriptor::{BackendDescriptor, BackendDescriptorType},
    },
    error::{Error, Result},
    math::Fraction,
    utility::to_i64,
};

macro_rules! impl_set_attribute_scalar {
    ($method:ident, $value_ty:ty, $attribute_ty:ident) => {
        /// Sets an attribute on this backend descriptor.
        ///
        /// The `name` selects the attribute, and this method supplies the cuDNN attribute type
        /// from the Rust value being passed. The descriptor must not be finalized.
        ///
        /// # Errors
        ///
        /// Returns an error if the descriptor has already been finalized, if
        /// cuDNN rejects the attribute name, value type, element count, or value,
        /// or if the value is not supported by the current cuDNN version.
        pub fn $method(&mut self, name: BackendAttributeName, value: $value_ty) -> Result<()> {
            self.set_attribute_raw(
                name,
                BackendAttributeType::$attribute_ty,
                1,
                (&raw const value).cast(),
            )
        }
    };
}

macro_rules! impl_set_attribute_slice {
    ($method:ident, $value_ty:ty, $attribute_ty:ident) => {
        /// Sets an attribute on this backend descriptor.
        ///
        /// The `name` selects the attribute, and this method supplies the cuDNN attribute type
        /// from the Rust value being passed. The descriptor must not be finalized.
        ///
        /// # Errors
        ///
        /// Returns an error if the descriptor has already been finalized, if
        /// cuDNN rejects the attribute name, value type, element count, or value,
        /// or if the value is not supported by the current cuDNN version.
        pub fn $method(&mut self, name: BackendAttributeName, values: &[$value_ty]) -> Result<()> {
            self.set_attribute_slice_raw(
                name,
                BackendAttributeType::$attribute_ty,
                values,
                "values",
            )
        }
    };
}

macro_rules! impl_attribute_scalar {
    ($method:ident, $value_ty:ty, $attribute_ty:ident) => {
        /// Returns an attribute from this finalized backend descriptor.
        ///
        /// The `name` selects the attribute, and this method supplies the expected cuDNN
        /// attribute type from the Rust value or return type.
        ///
        /// # Errors
        ///
        /// Returns an error if the descriptor has not been finalized, if cuDNN
        /// does not return the requested attribute value, or if cuDNN rejects
        /// the attribute name or expected type for this descriptor.
        pub fn $method(&self, name: BackendAttributeName) -> Result<$value_ty> {
            self.attribute_raw(name, BackendAttributeType::$attribute_ty)
        }
    };
}

impl BackendDescriptor {
    /// Sets an array of backend enums using their associated native attribute category.
    /// Values are encoded as native-width integers, independently of the Rust enum's layout.
    /// Open metadata enums preserve their unknown IDs through the same conversion as scalar writes.
    ///
    /// # Errors
    /// Returns an error for an empty array, an unrepresentable element count,
    /// a finalized descriptor or a native attribute rejection.
    pub fn set_attribute_enum_slice<E: BackendEnum>(
        &mut self,
        name: BackendAttributeName,
        values: &[E],
    ) -> Result<()> {
        let values = values.iter().copied().map(Into::into).collect::<Vec<u32>>();
        self.set_attribute_slice_raw(name, E::ATTRIBUTE_TYPE, &values, "enum values")
    }

    /// Sets an attribute on this backend descriptor.
    ///
    /// The `name` selects the attribute, and this method supplies the cuDNN attribute type
    /// from the Rust value being passed. The descriptor must not be finalized.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor has already been finalized, if
    /// cuDNN rejects the attribute name, value type, element count, or value,
    /// or if the value is not supported by the current cuDNN version.
    pub fn set_attribute_enum<E: BackendEnum>(
        &mut self,
        name: BackendAttributeName,
        value: E,
    ) -> Result<()> {
        let raw_value: u32 = value.into();
        self.set_attribute_raw(name, E::ATTRIBUTE_TYPE, 1, (&raw const raw_value).cast())
    }

    /// Sets an attribute on this backend descriptor.
    ///
    /// The `name` selects the attribute, and this method supplies the cuDNN attribute type
    /// from the Rust value being passed. The descriptor must not be finalized.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor has already been finalized, if
    /// cuDNN rejects the attribute name, value type, element count, or value,
    /// or if the value is not supported by the current cuDNN version.
    pub fn set_attribute_bool(&mut self, name: BackendAttributeName, value: bool) -> Result<()> {
        let value_i8 = i8::from(value);
        self.set_attribute_raw(
            name,
            BackendAttributeType::Boolean,
            1,
            (&raw const value_i8).cast(),
        )
    }

    impl_set_attribute_scalar!(set_attribute_i64, i64, Int64);

    impl_set_attribute_scalar!(set_attribute_i32, i32, Int32);

    impl_set_attribute_scalar!(set_attribute_f32, f32, Float);

    impl_set_attribute_scalar!(set_attribute_f64, f64, Double);

    impl_set_attribute_slice!(set_attribute_i64_slice, i64, Int64);

    impl_set_attribute_slice!(set_attribute_i32_slice, i32, Int32);

    /// Sets exact fractional attributes without exposing native struct storage.
    /// Numerators and denominators are preserved; operation-specific validation remains separate.
    ///
    /// # Errors
    /// Returns an error for an empty array, an unrepresentable element count,
    /// a finalized descriptor or a native attribute rejection.
    pub fn set_attribute_fraction_slice(
        &mut self,
        name: BackendAttributeName,
        values: &[Fraction],
    ) -> Result<()> {
        let values = values
            .iter()
            .copied()
            .map(Fraction::raw)
            .collect::<Vec<_>>();
        self.set_attribute_slice_raw(name, BackendAttributeType::Fraction, &values, "fractions")
    }

    /// Sets an attribute on this backend descriptor.
    ///
    /// The `name` selects the attribute, and this method supplies the cuDNN attribute type
    /// from the Rust value being passed. The descriptor must not be finalized.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor has already been finalized, if
    /// cuDNN rejects the attribute name, value type, element count, or value,
    /// or if the value is not supported by the current cuDNN version.
    pub fn set_attribute_descriptor(
        &mut self,
        name: BackendAttributeName,
        descriptor_to_set: &Self,
    ) -> Result<()> {
        let inner_desc = descriptor_to_set.as_raw();
        self.set_attribute_raw(
            name,
            BackendAttributeType::BackendDescriptor,
            1,
            (&raw const inner_desc).cast(),
        )
    }

    /// Sets an attribute on this backend descriptor.
    ///
    /// The `name` selects the attribute, and this method supplies the cuDNN attribute type
    /// from the Rust value being passed. The descriptor must not be finalized.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor has already been finalized, if
    /// cuDNN rejects the attribute name, value type, element count, or value,
    /// or if the value is not supported by the current cuDNN version.
    pub fn set_attribute_descriptor_slice(
        &mut self,
        name: BackendAttributeName,
        descriptors: &[&Self],
    ) -> Result<()> {
        if descriptors.is_empty() {
            return Err(Error::EmptyList {
                name: "descriptors".into(),
            });
        }
        let inner_descriptors: Vec<_> = descriptors.iter().map(|d| d.as_raw()).collect();
        self.set_attribute_raw(
            name,
            BackendAttributeType::BackendDescriptor,
            i64::try_from(inner_descriptors.len())
                .map_err(|_| Error::out_of_range("descriptor count"))?,
            inner_descriptors.as_ptr().cast(),
        )
    }

    /// Sets an attribute using a UTF-8 string as a raw char buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if `value` is empty, its byte count does not fit `i64`, the descriptor is already finalized, or cuDNN rejects the attribute assignment.
    pub fn set_attribute_char_string(
        &mut self,
        name: BackendAttributeName,
        value: &str,
    ) -> Result<()> {
        self.set_attribute_char_buffer(name, value.as_bytes())
    }

    /// Sets an attribute on this backend descriptor.
    ///
    /// The `name` selects the attribute, and this method supplies the cuDNN attribute type
    /// from the Rust value being passed. The descriptor must not be finalized.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor has already been finalized, if
    /// cuDNN rejects the attribute name, value type, element count, or value,
    /// or if the value is not supported by the current cuDNN version.
    pub fn set_attribute_char_buffer(
        &mut self,
        name: BackendAttributeName,
        value: &[u8],
    ) -> Result<()> {
        if value.is_empty() {
            return Err(Error::EmptyList {
                name: "value".into(),
            });
        }

        self.set_attribute_raw(
            name,
            BackendAttributeType::Char,
            to_i64(value.len(), "char attribute bytes")?,
            value.as_ptr().cast(),
        )
    }

    /// Sets an attribute on this backend descriptor.
    ///
    /// The `name` selects the attribute, and this method supplies the cuDNN attribute type
    /// from the Rust value being passed. The descriptor must not be finalized.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor has already been finalized, if
    /// cuDNN rejects the attribute name, value type, element count, or value,
    /// or if the value is not supported by the current cuDNN version.
    ///
    /// # Safety
    /// Supplied handles or pointees must satisfy this attribute's native validity,
    /// access and lifetime requirements through descriptor use and queued replay.
    /// Preserve their owners until native consumers have finished with them.
    pub unsafe fn set_attribute_handle(
        &mut self,
        name: BackendAttributeName,
        handle: sys::cudnnHandle_t,
    ) -> Result<()> {
        self.set_attribute_raw(
            name,
            BackendAttributeType::Handle,
            1,
            (&raw const handle).cast(),
        )
    }

    /// Sets an attribute on this backend descriptor.
    ///
    /// The `name` selects the attribute, and this method supplies the cuDNN attribute type
    /// from the Rust value being passed. The descriptor must not be finalized.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor has already been finalized, if
    /// cuDNN rejects the attribute name, value type, element count, or value,
    /// or if the value is not supported by the current cuDNN version.
    ///
    /// # Safety
    /// Supplied handles or pointees must satisfy this attribute's native validity,
    /// access and lifetime requirements through descriptor use and queued replay.
    /// Preserve their owners until native consumers have finished with them.
    pub unsafe fn set_attribute_void_ptr_slice(
        &mut self,
        name: BackendAttributeName,
        pointers: &[*mut ()],
    ) -> Result<()> {
        self.set_attribute_slice_raw(name, BackendAttributeType::VoidPtr, pointers, "pointers")
    }

    /// Sets an attribute on this backend descriptor.
    ///
    /// The `name` selects the attribute, and this method supplies the cuDNN attribute type
    /// from the Rust value being passed. The descriptor must not be finalized.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor has already been finalized, if
    /// cuDNN rejects the attribute name, value type, element count, or value,
    /// or if the value is not supported by the current cuDNN version.
    ///
    /// # Safety
    /// Supplied handles or pointees must satisfy this attribute's native validity,
    /// access and lifetime requirements through descriptor use and queued replay.
    /// Preserve their owners until native consumers have finished with them.
    pub unsafe fn set_attribute_void_ptr(
        &mut self,
        name: BackendAttributeName,
        pointer: *mut (),
    ) -> Result<()> {
        let pointers = [pointer];
        self.set_attribute_slice_raw(name, BackendAttributeType::VoidPtr, &pointers, "pointers")
    }

    /// Sets a `CUDNN_TYPE_VOID_PTR` attribute whose payload is the object pointer itself rather than an array of void pointers.
    ///
    /// # Safety
    /// `pointer` must address the native host object payload expected by this
    /// attribute, and its transitive storage must remain live for every native use.
    pub(crate) unsafe fn set_attribute_void_ptr_payload(
        &mut self,
        name: BackendAttributeName,
        pointer: *mut (),
    ) -> Result<()> {
        self.set_attribute_raw(name, BackendAttributeType::VoidPtr, 1, pointer.cast_const())
    }

    impl_attribute_scalar!(attribute_i64, i64, Int64);

    impl_attribute_scalar!(attribute_i32, i32, Int32);

    /// Returns an attribute from this finalized backend descriptor.
    ///
    /// The `name` selects the attribute, and this method supplies the expected cuDNN
    /// attribute type from the Rust value or return type.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor has not been finalized, if cuDNN
    /// does not return the requested attribute value, or if cuDNN rejects
    /// the attribute name or expected type for this descriptor.
    pub fn attribute_bool(&self, name: BackendAttributeName) -> Result<bool> {
        self.attribute_raw::<i8>(name, BackendAttributeType::Boolean)
            .map(|value| value != 0)
    }

    impl_attribute_scalar!(attribute_f64, f64, Double);

    impl_attribute_scalar!(attribute_f32, f32, Float);

    /// Returns an attribute from this finalized backend descriptor.
    ///
    /// The `name` selects the attribute, and this method supplies the expected cuDNN
    /// attribute type from the Rust value or return type.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor has not been finalized, if cuDNN
    /// does not return the requested attribute value, or if cuDNN rejects
    /// the attribute name or expected type for this descriptor.
    pub fn attribute_i64_slice(&self, name: BackendAttributeName) -> Result<Vec<i64>> {
        self.attribute_raw_vec(name, BackendAttributeType::Int64, 0_i64, "values")
    }

    /// Returns exact native fractional attributes without normalization or conversion to floating point.
    ///
    /// # Errors
    /// Returns descriptor-state, native attribute-query or inconsistent element-count errors.
    pub fn attribute_fraction_slice(&self, name: BackendAttributeName) -> Result<Vec<Fraction>> {
        Ok(self
            .attribute_raw_vec(
                name,
                BackendAttributeType::Fraction,
                Fraction::whole(0).raw(),
                "fractions",
            )?
            .into_iter()
            .map(|value| Fraction::new(value.numerator, value.denominator))
            .collect())
    }

    /// Returns an attribute from this finalized backend descriptor.
    ///
    /// The `name` selects the attribute, and this method supplies the expected cuDNN
    /// attribute type from the Rust value or return type.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor has not been finalized, if cuDNN
    /// does not return the requested attribute value, or if cuDNN rejects
    /// the attribute name or expected type for this descriptor.
    ///
    /// # Safety
    /// The attribute must return an array of native void pointers. Object payloads,
    /// such as variant-pack shape/stride overrides, require `attribute_unchecked`
    /// with their actual native object storage rather than a pointer array.
    pub unsafe fn attribute_void_ptr_slice(
        &self,
        name: BackendAttributeName,
    ) -> Result<Vec<*mut ()>> {
        self.attribute_raw_vec(
            name,
            BackendAttributeType::VoidPtr,
            ptr::null_mut::<()>(),
            "values",
        )
    }

    /// Returns an attribute from this finalized backend descriptor.
    ///
    /// The `name` selects the attribute, and this method supplies the expected cuDNN
    /// attribute type from the Rust value or return type.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor has not been finalized, if cuDNN
    /// does not return the requested attribute value, or if cuDNN rejects
    /// the attribute name or expected type for this descriptor.
    pub fn attribute_char_string(&self, name: BackendAttributeName) -> Result<String> {
        let values =
            self.attribute_raw_vec(name, BackendAttributeType::Char, 0_i8, "char_values")?;
        Ok(string_from_c_chars(&values))
    }

    /// Reads the exact native character payload, including any trailing NUL bytes.
    /// Retries a growing payload on cuDNN's size-insufficient response, up to eight reads.
    ///
    /// # Errors
    /// Returns finalization, native attribute, invalid count or persistent size-race errors.
    pub fn attribute_char_buffer(&self, name: BackendAttributeName) -> Result<Vec<u8>> {
        Ok(self
            .attribute_raw_vec(name, BackendAttributeType::Char, 0_i8, "char_values")?
            .into_iter()
            .map(|byte| byte as u8)
            .collect())
    }

    /// Returns an attribute from this finalized backend descriptor.
    ///
    /// The `name` selects the attribute, and this method supplies the expected cuDNN
    /// attribute type from the Rust value or return type.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor has not been finalized, if cuDNN
    /// does not return the requested attribute value, or if cuDNN rejects
    /// the attribute name or expected type for this descriptor.
    pub fn attribute_enum_slice<E: BackendEnum>(
        &self,
        name: BackendAttributeName,
    ) -> Result<Vec<E>> {
        self.attribute_raw_vec(name, E::ATTRIBUTE_TYPE, 0u32, "enum values")?
            .into_iter()
            .map(|value| {
                E::try_from(value)
                    .map_err(|_| Error::DescriptorInvalidAttributeValue(name, value.to_string()))
            })
            .collect()
    }

    /// Retrieves owned descriptors of the specified native class.
    /// cuDNN populates preallocated objects rather than returning borrowed pointers.
    /// `maximum` bounds the returned prefix; `None` retrieves every descriptor.
    /// A zero maximum returns no descriptors without querying the attribute.
    /// Retries a growing result up to eight reads when more descriptors are needed.
    ///
    /// # Errors
    /// Returns native query or allocation errors, or an error if a complete result keeps growing beyond the retry limit.
    ///
    /// # Safety
    /// `descriptor_type` must be the class required by `name`; cuDNN writes native
    /// class-specific state into each destination. Use `attribute_unchecked` when
    /// supplying an externally established destination ABI.
    pub unsafe fn attribute_descriptor_slice(
        &self,
        name: BackendAttributeName,
        descriptor_type: BackendDescriptorType,
        maximum: Option<usize>,
    ) -> Result<Vec<Self>> {
        self.check_finalized()?;
        if maximum == Some(0) {
            return Ok(Vec::new());
        }
        let available = self.attribute_count(name, BackendAttributeType::BackendDescriptor)?;
        let mut count = maximum.map_or(available, |maximum| maximum.min(available));
        if count == 0 {
            return Ok(Vec::new());
        }
        let mut retries = 0;
        loop {
            let mut descriptors = (0..count)
                .map(|_| Self::create(descriptor_type))
                .collect::<Result<Vec<_>>>()?;
            let mut handles = descriptors.iter().map(Self::as_raw).collect::<Vec<_>>();
            // SAFETY: all destination objects are live and have the native class promised by the caller.
            // cuDNN writes at most the requested count, while reporting the attribute's total count.
            // Owners survive native failure and are released before a larger read is attempted.
            let available = unsafe {
                self.attribute_unchecked(
                    name,
                    BackendAttributeType::BackendDescriptor,
                    count,
                    handles.as_mut_ptr().cast(),
                )?
            };
            let required = maximum.map_or(available, |maximum| maximum.min(available));
            if required > count {
                if retries == 7 {
                    return Err(Error::InvalidBackendDescriptorCount {
                        name: "descriptors".into(),
                    });
                }
                count = required;
                retries += 1;
                continue;
            }
            descriptors.truncate(required);
            descriptors.iter_mut().for_each(Self::mark_finalized);
            return Ok(descriptors);
        }
    }

    /// Retrieves one owned descriptor of the specified native class.
    ///
    /// # Safety
    /// `descriptor_type` must be the native class required by `name`.
    ///
    /// # Errors
    ///
    /// Returns an error if this descriptor is not finalized, destination descriptor creation or the native query fails, or the attribute does not report exactly one descriptor.
    pub unsafe fn attribute_descriptor(
        &self,
        name: BackendAttributeName,
        descriptor_type: BackendDescriptorType,
    ) -> Result<Self> {
        self.check_finalized()?;
        let mut descriptor = Self::create(descriptor_type)?;
        let mut handle = descriptor.as_raw();
        // SAFETY: this is one live destination of the class promised by the caller.
        let count = unsafe {
            self.attribute_unchecked(
                name,
                BackendAttributeType::BackendDescriptor,
                1,
                (&raw mut handle).cast(),
            )?
        };
        if count != 1 {
            return Err(Error::DescriptorAttributeNotFound(name));
        }
        descriptor.mark_finalized();
        Ok(descriptor)
    }

    /// Reads one native enum value using the attribute category associated with `E`.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor is not finalized, the native query fails or does not return one value, or `E` cannot decode the reported value.
    pub fn attribute_enum<E: BackendEnum>(&self, name: BackendAttributeName) -> Result<E> {
        let value: u32 = self.attribute_raw(name, E::ATTRIBUTE_TYPE)?;
        E::try_from(value)
            .map_err(|_| Error::DescriptorInvalidAttributeValue(name, value.to_string()))
    }
}

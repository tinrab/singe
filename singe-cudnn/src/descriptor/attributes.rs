use std::ptr;

use singe_core::string_from_c_chars;
use singe_cudnn_sys as sys;

use crate::{
    attribute::{BackendAttributeName, BackendAttributeType},
    descriptor::{BackendDescriptor, BackendDescriptorType},
    error::{Error, Result},
    try_ffi,
    utility::{to_i64, to_usize},
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
        /// attribute type from its Rust return type or `attribute_type` argument.
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
    pub fn set_attribute_enum<E: Copy + Into<u32>>(
        &mut self,
        name: BackendAttributeName,
        ty: BackendAttributeType,
        value: E,
    ) -> Result<()> {
        let raw_value: u32 = value.into();
        self.set_attribute_raw(name, ty, 1, (&raw const raw_value).cast())
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
    impl_set_attribute_slice!(set_attribute_fraction_slice, sys::cudnnFraction_t, Fraction);

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
        descriptor_to_set: &BackendDescriptor,
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
        descriptors: &[&BackendDescriptor],
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
            inner_descriptors.len() as _,
            inner_descriptors.as_ptr().cast(),
        )
    }

    /// Sets an attribute using a UTF-8 string as a raw char buffer.
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

        let values = value.iter().map(|byte| *byte as i8).collect::<Vec<_>>();
        self.set_attribute_raw(
            name,
            BackendAttributeType::Char,
            values.len() as _,
            values.as_ptr().cast(),
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
    pub fn set_attribute_handle(
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
    pub fn set_attribute_void_ptr_slice(
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
    pub fn set_attribute_void_ptr(
        &mut self,
        name: BackendAttributeName,
        pointer: *mut (),
    ) -> Result<()> {
        let pointers = [pointer];
        self.set_attribute_void_ptr_slice(name, &pointers)
    }

    /// Sets a `CUDNN_TYPE_VOID_PTR` attribute whose payload is the object pointer itself rather than an array of void pointers.
    pub(crate) fn set_attribute_void_ptr_payload(
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
    /// attribute type from its Rust return type or `attribute_type` argument.
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
    /// attribute type from its Rust return type or `attribute_type` argument.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor has not been finalized, if cuDNN
    /// does not return the requested attribute value, or if cuDNN rejects
    /// the attribute name or expected type for this descriptor.
    pub fn attribute_i64_slice(&self, name: BackendAttributeName) -> Result<Vec<i64>> {
        self.attribute_raw_vec(name, BackendAttributeType::Int64, 0_i64, "values")
    }

    /// Returns an attribute from this finalized backend descriptor.
    ///
    /// The `name` selects the attribute, and this method supplies the expected cuDNN
    /// attribute type from its Rust return type or `attribute_type` argument.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor has not been finalized, if cuDNN
    /// does not return the requested attribute value, or if cuDNN rejects
    /// the attribute name or expected type for this descriptor.
    pub fn attribute_void_ptr_slice(&self, name: BackendAttributeName) -> Result<Vec<*mut ()>> {
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
    /// attribute type from its Rust return type or `attribute_type` argument.
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

    /// Returns an attribute from this finalized backend descriptor.
    ///
    /// The `name` selects the attribute, and this method supplies the expected cuDNN
    /// attribute type from its Rust return type or `attribute_type` argument.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor has not been finalized, if cuDNN
    /// does not return the requested attribute value, or if cuDNN rejects
    /// the attribute name or expected type for this descriptor.
    pub fn attribute_enum_slice<E: TryFrom<u32> + Copy>(
        &self,
        name: BackendAttributeName,
        attribute_type: BackendAttributeType,
    ) -> Result<Vec<E>> {
        let count = self.attribute_count(name, attribute_type)?;
        if count == 0 {
            return Ok(Vec::new());
        }

        let count = to_usize(count, "backend attribute count")?;
        let mut values_raw = vec![0u32; count];
        let mut count_written = 0;
        unsafe {
            try_ffi!(sys::cudnnBackendGetAttribute(
                self.handle,
                name.into(),
                attribute_type.into(),
                to_i64(count, "backend attribute count")?,
                &raw mut count_written,
                values_raw.as_mut_ptr() as _,
            ))?;
        }
        debug_assert!(
            count_written as usize == count,
            "expected to read {count} values, but only read {count_written}",
        );

        let mut values = Vec::with_capacity(count);
        for value in values_raw {
            values
                .push(E::try_from(value).map_err(|_| {
                    Error::DescriptorInvalidAttributeValue(name, value.to_string())
                })?);
        }

        Ok(values)
    }

    /// Returns an attribute from this finalized backend descriptor.
    ///
    /// The `name` selects the attribute, and this method supplies the expected cuDNN
    /// attribute type from its Rust return type or `attribute_type` argument.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor has not been finalized, if cuDNN
    /// does not return the requested attribute value, or if cuDNN rejects
    /// the attribute name or expected type for this descriptor.
    pub fn attribute_descriptor_slice(
        &self,
        name: BackendAttributeName,
        attribute_type: BackendAttributeType,
    ) -> Result<Vec<sys::cudnnBackendDescriptor_t>> {
        self.attribute_raw_vec(name, attribute_type, ptr::null_mut(), "descriptors")
    }

    /// Returns an attribute from this finalized backend descriptor.
    ///
    /// The `name` selects the attribute, and this method supplies the expected cuDNN
    /// attribute type from its Rust return type or `attribute_type` argument.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor has not been finalized, if cuDNN
    /// does not return the requested attribute value, or if cuDNN rejects
    /// the attribute name or expected type for this descriptor.
    pub fn attribute_descriptor(
        &self,
        name: BackendAttributeName,
        attribute_type: BackendAttributeType,
    ) -> Result<sys::cudnnBackendDescriptor_t> {
        let value: sys::cudnnBackendDescriptor_t = self.attribute_raw(name, attribute_type)?;
        if value.is_null() {
            return Err(Error::DescriptorAttributeNotFound(name));
        }

        Ok(value)
    }

    /// Returns an attribute from this finalized backend descriptor.
    ///
    /// The `name` selects the attribute, and this method supplies the expected cuDNN
    /// attribute type from its Rust return type or `attribute_type` argument.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor has not been finalized, if cuDNN
    /// does not return the requested attribute value, or if cuDNN rejects
    /// the attribute name or expected type for this descriptor.
    pub fn attribute_descriptor_of_type(
        &self,
        name: BackendAttributeName,
        descriptor_type: BackendDescriptorType,
    ) -> Result<BackendDescriptor> {
        if !self.finalized {
            return Err(Error::DescriptorRequiredFinalized);
        }

        let mut descriptor = BackendDescriptor::create(descriptor_type)?;
        let mut raw_descriptor = descriptor.as_raw();
        let mut count_written = 0_i64;
        unsafe {
            try_ffi!(sys::cudnnBackendGetAttribute(
                self.handle,
                name.into(),
                BackendAttributeType::BackendDescriptor.into(),
                1,
                &raw mut count_written,
                (&raw mut raw_descriptor).cast(),
            ))?;
        }
        if count_written != 1 {
            return Err(Error::DescriptorAttributeNotFound(name));
        }

        descriptor.mark_finalized();

        Ok(descriptor)
    }

    pub fn attribute_enum<E: TryFrom<u32> + Copy>(
        &self,
        name: BackendAttributeName,
        attribute_type: BackendAttributeType,
    ) -> Result<E> {
        self.attribute_enum_slice(name, attribute_type)?
            .into_iter()
            .next()
            .ok_or(Error::DescriptorAttributeNotFound(name))
    }
}

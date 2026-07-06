//! The castable key type for the cast slot maps.
//!
//! [`CastKey<T, K>`] stores a `slotmap` key alongside `T`'s pointer metadata.
//! It is the bare key of [`UnsafeCastMapG`](crate::unsafe_cast_map::UnsafeCastMapG);
//! the checked [`CastMapG`](crate::cast_map::CastMapG) uses the *same* key type
//! and validates lookups against each slot's stored concrete
//! [`TypeId`](std::any::TypeId) (see [`ConcreteTypeId`](crate::type_tagged_ptr::ConcreteTypeId)).
//!
//! `CastKey` is not a `slotmap::Key`; the map wrappers convert at the boundary
//! via [`inner_key`](CastKey::inner_key).
//!
//! Because `slotmap`'s keys are `Copy`, a `CastKey` simply holds the key by
//! value. For dyn dispatch on a key, see [`DynKey`](crate::dyn_key::DynKey) /
//! [`as_dyn`](CastKey::as_dyn).

use std::ops::Receiver;
use std::ptr::Pointee;

use slotmap::{DefaultKey, Key, KeyData};

use crate::dyn_key::DynKey;

// ─── CastKey<T, K> ───────────────────────────────────────────────────────────

/// A key parameterized over `T: ?Sized` that stores the backing `slotmap` key
/// (`K`, defaults to [`DefaultKey`]) plus `T`'s pointer metadata.
///
/// # Sizes (64-bit)
/// - `CastKey<SizedType>`: the size of `K` (metadata is `()`).
/// - `CastKey<dyn Trait>`: `K` + a vtable pointer.
pub struct CastKey<T: ?Sized + Pointee, K: Key = DefaultKey>
where
    <T as Pointee>::Metadata: Copy,
{
    pub(crate) key: K,
    pub(crate) metadata: <T as Pointee>::Metadata,
}

impl<T: ?Sized + Pointee, K: Key> Clone for CastKey<T, K>
where
    <T as Pointee>::Metadata: Copy,
{
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized + Pointee, K: Key> Copy for CastKey<T, K> where <T as Pointee>::Metadata: Copy {}

impl<T: ?Sized + Pointee, K: Key> std::fmt::Debug for CastKey<T, K>
where
    <T as Pointee>::Metadata: Copy,
{
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CastKey").field("key", &self.key).finish()
    }
}

impl<T: ?Sized + Pointee, K: Key> PartialEq for CastKey<T, K>
where
    <T as Pointee>::Metadata: Copy,
{
    /// Equality is on the backing key only; pointer metadata is not compared.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

// A receiver without `Deref`: lets traits take `self: CastKey<Self>` by
// value (`arbitrary_self_types`). Static dispatch only — dyn dispatch needs
// the pointer-shaped `DynKey`.
impl<T: ?Sized + Pointee, K: Key> Receiver for CastKey<T, K>
where
    <T as Pointee>::Metadata: Copy,
{
    type Target = T;
}


impl<T: ?Sized + Pointee, K: Key> Eq for CastKey<T, K> where <T as Pointee>::Metadata: Copy {}

impl<T: ?Sized + Pointee, K: Key> std::hash::Hash for CastKey<T, K>
where
    <T as Pointee>::Metadata: Copy,
{
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.key.hash(state);
    }
}

impl<T: ?Sized + Pointee, K: Key> CastKey<T, K>
where
    <T as Pointee>::Metadata: Copy,
{
    /// Returns the backing `slotmap` [`KeyData`].
    #[inline]
    pub fn key_data(&self) -> KeyData {
        self.key.data()
    }

    /// Returns the pointer metadata for `T`.
    #[inline]
    pub fn metadata(&self) -> <T as Pointee>::Metadata {
        self.metadata
    }

    /// Strips the pointer metadata, producing the backing `slotmap` key.
    #[inline]
    pub fn inner_key(&self) -> K {
        self.key
    }

    /// Borrows this key into its dyn-dispatchable form, usable as a trait
    /// method receiver (`fn m(self: DynKey<Self>, ..)`).
    #[inline]
    pub fn as_dyn(&self) -> DynKey<'_, T, K> {
        DynKey::new(self)
    }

    /// Upcasts the key's metadata from `T` to `U` where `T: Unsize<U>`
    /// (e.g. `CastKey<Dog>` to `CastKey<dyn Any>`) without needing a data
    /// pointer.
    #[inline]
    pub fn upcast<U: ?Sized + Pointee>(self) -> CastKey<U, K>
    where
        T: std::marker::Unsize<U>,
        <U as Pointee>::Metadata: Copy,
    {
        let dummy: *const T = std::ptr::from_raw_parts(std::ptr::null::<()>(), self.metadata);
        let upcast: *const U = dummy;
        CastKey {
            key: self.key,
            metadata: std::ptr::metadata(upcast),
        }
    }

    /// Builds a cast key from raw parts.
    #[inline]
    pub fn from_raw_parts(key: K, metadata: <T as Pointee>::Metadata) -> Self {
        Self { key, metadata }
    }
}

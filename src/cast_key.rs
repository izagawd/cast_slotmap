//! Castable key types for the cast slot maps.
//!
//! - [`CastKey<T, K>`] stores a `slotmap` key alongside `T`'s pointer metadata.
//!   It is the bare key used by [`UnsafeCastMap`](crate::unsafe_cast_map::UnsafeCastMap).
//! - [`StableCastKey<T, K>`] wraps a `CastKey` and adds a [`MapId`](crate::map_id::MapId),
//!   making it safe to use with [`CastMap`](crate::cast_map::CastMap)
//!   (cross-map misuse returns `None` instead of being unsound).
//!
//! Neither is a `slotmap::Key`; the map wrappers convert at the boundary via
//! [`inner_key`](CastKey::inner_key).
//!
//! Because `slotmap`'s keys are `Copy`, a `CastKey` simply holds the key by
//! value (unlike `stable_gen_map`, which stores raw `KeyData`).

use std::ptr::Pointee;

use slotmap::{DefaultKey, Key, KeyData};

use crate::map_id::MapId;

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

    /// Build a cast key from raw parts.
    ///
    /// # Safety
    /// `metadata` must be valid for the value identified by `key`.
    #[inline]
    pub unsafe fn from_parts(key: K, metadata: <T as Pointee>::Metadata) -> Self {
        Self { key, metadata }
    }
}

// ─── StableCastKey<T, K> ─────────────────────────────────────────────────────

/// A [`CastKey`] paired with a [`MapId`] so that cross-map misuse is caught at
/// runtime (returns `None`) rather than being unsound.
pub struct StableCastKey<T: ?Sized + Pointee, K: Key = DefaultKey>
where
    <T as Pointee>::Metadata: Copy,
{
    pub(crate) map_id: MapId,
    pub(crate) inner: CastKey<T, K>,
}

impl<T: ?Sized + Pointee, K: Key> Clone for StableCastKey<T, K>
where
    <T as Pointee>::Metadata: Copy,
{
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized + Pointee, K: Key> Copy for StableCastKey<T, K> where <T as Pointee>::Metadata: Copy {}

impl<T: ?Sized + Pointee, K: Key> std::fmt::Debug for StableCastKey<T, K>
where
    <T as Pointee>::Metadata: Copy,
{
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StableCastKey")
            .field("key", &self.inner.key)
            .field("map_id", &self.map_id)
            .finish()
    }
}

impl<T: ?Sized + Pointee, K: Key> PartialEq for StableCastKey<T, K>
where
    <T as Pointee>::Metadata: Copy,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner.key == other.inner.key && self.map_id == other.map_id
    }
}

impl<T: ?Sized + Pointee, K: Key> Eq for StableCastKey<T, K> where <T as Pointee>::Metadata: Copy {}

impl<T: ?Sized + Pointee, K: Key> std::hash::Hash for StableCastKey<T, K>
where
    <T as Pointee>::Metadata: Copy,
{
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inner.key.hash(state);
        self.map_id.hash(state);
    }
}

impl<T: ?Sized + Pointee, K: Key> StableCastKey<T, K>
where
    <T as Pointee>::Metadata: Copy,
{
    /// Constructs a `StableCastKey` from its raw cast key and map id.
    ///
    /// # Safety
    /// `map_id` must identify the map that owns the slot addressed by
    /// `cast_key.inner_key()`, and the metadata must be valid for `T`.
    #[inline]
    pub unsafe fn from_parts(cast_key: CastKey<T, K>, map_id: MapId) -> Self {
        StableCastKey {
            inner: cast_key,
            map_id,
        }
    }

    /// Returns the backing `slotmap` [`KeyData`].
    #[inline]
    pub fn key_data(&self) -> KeyData {
        self.inner.key.data()
    }

    /// Returns the pointer metadata for `T`.
    #[inline]
    pub fn metadata(&self) -> <T as Pointee>::Metadata {
        self.inner.metadata
    }

    /// Returns the map identity this key is bound to.
    #[inline]
    pub fn map_id(&self) -> MapId {
        self.map_id
    }

    /// Strips the metadata and map id, producing the backing `slotmap` key.
    #[inline]
    pub fn inner_key(&self) -> K {
        self.inner.inner_key()
    }

    /// Returns the underlying [`CastKey`] without the map id.
    #[inline]
    pub fn cast_key(&self) -> CastKey<T, K> {
        self.inner
    }

    /// Upcasts the key's metadata from `T` to `U` where `T: Unsize<U>`.
    /// The map id is preserved.
    #[inline]
    pub fn upcast<U: ?Sized + Pointee>(self) -> StableCastKey<U, K>
    where
        T: std::marker::Unsize<U>,
        <U as Pointee>::Metadata: Copy,
    {
        StableCastKey {
            inner: self.inner.upcast(),
            map_id: self.map_id,
        }
    }
}

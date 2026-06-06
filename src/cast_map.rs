//! Safe wrapper around [`UnsafeCastMapG`](crate::unsafe_cast_map::UnsafeCastMapG)
//! that binds keys to the map that created them via a [`MapId`].
//!
//! [`CastMapG`] is generic over the backing `slotmap` map `M`
//! ([`slotmap::SlotMap`] or [`slotmap::DenseSlotMap`]); the backing key and
//! stored pointer types are read off `M` as `M::Key` and `M::Value`. The
//! concrete maps are exposed as aliases: [`CastMap`] (sparse) and
//! [`DenseCastMap`] (dense), plus the `Box`-storing [`BoxCastMap`] /
//! [`BoxDenseCastMap`].
//!
//! `detach` / `reattach` are deliberately **not** offered here: reattaching a
//! value of a different type would invalidate a key's cached pointer metadata,
//! which the [`MapId`] check cannot catch. They live on the unsafe
//! [`UnsafeCastMapG`](crate::unsafe_cast_map::UnsafeCastMapG), reachable via
//! [`inner_mut`](CastMapG::inner_mut) if you accept that `unsafe` contract.
//!
//! Every [`StableCastKey`](crate::cast_key::StableCastKey) carries the map's
//! identity. Keyed lookups check the id before touching pointer metadata, so a
//! key from one map used on a different one returns `None` instead of being unsound.
//!
//! Soundness, briefly: a `StableCastKey` that passes both the [`MapId`] check
//! and `slotmap`'s slot-version check provably refers to the exact value that
//! minted it, so the metadata stored in the key is valid for that value.

use std::any::Any;
use std::ops::{Deref, DerefMut};
use std::ptr::Pointee;

use slotmap::{DenseSlotMap, Key, SlotMap};

use crate::slotmap_trait::{MTarget, SlotMapTrait};
use crate::cast_key::{CastKey, StableCastKey};
use crate::map_id::MapId;
use crate::retype_ptr::RetypePtr;
use stable_deref_trait::StableDeref;
use crate::unsafe_cast_map::{self, UnsafeCastMapG};

// ─── helpers ─────────────────────────────────────────────────────────────────

#[inline]
fn stabilize<T: ?Sized + Pointee, K: Key>(key: CastKey<T, K>, map_id: MapId) -> StableCastKey<T, K>
where
    <T as Pointee>::Metadata: Copy,
{
    StableCastKey { inner: key, map_id }
}

// ─── CastMapG ──────────────────────────────────────────────────────────────────

/// A safe wrapper around [`UnsafeCastMapG`] that checks a per-map [`MapId`] on
/// every keyed access.
///
/// Its one type parameter is the backing `slotmap` map `M`; the stored smart
/// pointer is `M::Value` (e.g. `Box<dyn Any>`) and the output type is
/// `<M::Value as Deref>::Target`. Use the aliases [`CastMap`] / [`DenseCastMap`]
/// (or the `Box` forms [`BoxCastMap`] / [`BoxDenseCastMap`]) rather than naming
/// `M` directly.
pub struct CastMapG<M> {
    inner: UnsafeCastMapG<M>,
    map_id: MapId,
}

// ─── Clone ───────────────────────────────────────────────────────────────────

impl<M> Clone for CastMapG<M>
where
    M: SlotMapTrait + Clone,
{
    /// Clones the map. The clone receives a **fresh** map identity, so keys from
    /// the original are not valid on the clone (use iteration to obtain new
    /// keys for the cloned data).
    #[inline]
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            map_id: MapId::next(),
        }
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        self.inner.clone_from(&source.inner);
        self.map_id = MapId::next();
    }
}

impl<M> Default for CastMapG<M>
where
    M: SlotMapTrait,
    M::Value: StableDeref,
{
    fn default() -> Self {
        Self::new()
    }
}

// ─── Basic methods ─────────────────────────────────────────────────────────────

impl<M> CastMapG<M>
where
    M: SlotMapTrait,
    M::Value: StableDeref,
{
    /// Creates a new, empty map with a fresh [`MapId`].
    #[inline]
    pub fn new() -> Self {
        Self {
            inner: UnsafeCastMapG::new(),
            map_id: MapId::next(),
        }
    }

    /// Creates a new map with the given pre-allocated capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: UnsafeCastMapG::with_capacity(capacity),
            map_id: MapId::next(),
        }
    }

    /// Returns this map's unique identity.
    #[inline]
    pub fn map_id(&self) -> MapId {
        self.map_id
    }

    /// Returns true if the map is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the number of occupied elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns how many slots the backing storage can hold before reallocating.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Reserves capacity for at least `additional` more elements.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional);
    }

    /// Tries to reserve capacity for at least `additional` more elements.
    #[inline]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), std::collections::TryReserveError> {
        self.inner.try_reserve(additional)
    }

    /// Removes all elements from the map. Outstanding keys are invalidated.
    #[inline]
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Returns whether the key is still live in *this* map. A key from another
    /// map (mismatched [`MapId`]) returns `false`.
    #[inline]
    pub fn contains_key<T: ?Sized + Pointee>(&self, key: StableCastKey<T, M::Key>) -> bool
    where
        <T as Pointee>::Metadata: Copy,
    {
        key.map_id == self.map_id && self.inner.contains_key(key.inner)
    }

    // ── inner accessors ───────────────────────────────────────────────────

    /// Consumes this map and returns the underlying [`UnsafeCastMapG`].
    #[inline]
    pub fn inner(self) -> UnsafeCastMapG<M> {
        self.inner
    }

    /// Returns a shared reference to the underlying [`UnsafeCastMapG`].
    #[inline]
    pub fn inner_ref(&self) -> &UnsafeCastMapG<M> {
        &self.inner
    }

    /// Returns a mutable reference to the underlying [`UnsafeCastMapG`].
    #[inline]
    pub fn inner_mut(&mut self) -> &mut UnsafeCastMapG<M> {
        &mut self.inner
    }

    // ── backing-key access (no map-id check) ───────────────────────────────

    /// Shared-reference lookup using the backing `slotmap` key directly.
    #[inline]
    pub fn get_by_inner_key(&self, key: M::Key) -> Option<&MTarget<M>> {
        self.inner.get_by_inner_key(key)
    }

    /// Removes an element by its backing `slotmap` key.
    #[inline]
    pub fn remove_by_inner_key(&mut self, key: M::Key) -> Option<M::Value> {
        self.inner.remove_by_inner_key(key)
    }

    /// Shared iterator over output references only.
    #[inline]
    pub fn values(&self) -> impl Iterator<Item = &MTarget<M>> + '_ {
        self.inner.values()
    }
}

// ── backing-key access requiring `&mut Output` ───────────────────────────────

impl<M> CastMapG<M>
where
    M: SlotMapTrait,
    M::Value: StableDeref + DerefMut,
{
    /// Mutable-reference lookup using the backing `slotmap` key directly.
    #[inline]
    pub fn get_by_inner_key_mut(&mut self, key: M::Key) -> Option<&mut MTarget<M>> {
        self.inner.get_by_inner_key_mut(key)
    }

    /// Mutable iterator over output references only.
    #[inline]
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut MTarget<M>> + '_ {
        self.inner.values_mut()
    }

    /// Mutable disjoint lookup by backing `slotmap` keys. Returns `None` if any
    /// key is invalid or two keys alias the same slot. Backing keys bypass the
    /// map-id check, exactly like
    /// [`get_by_inner_key_mut`](Self::get_by_inner_key_mut).
    #[inline]
    pub fn get_disjoint_mut_by_inner_key<const N: usize>(
        &mut self,
        keys: [M::Key; N],
    ) -> Option<[&mut MTarget<M>; N]> {
        self.inner.get_disjoint_mut_by_inner_key(keys)
    }

    /// Like [`get_disjoint_mut_by_inner_key`](Self::get_disjoint_mut_by_inner_key)
    /// but without validity or uniqueness checks.
    ///
    /// # Safety
    /// Every key must be valid for this map and no two keys may alias one slot.
    #[inline]
    pub unsafe fn get_disjoint_unchecked_mut_by_inner_key<const N: usize>(
        &mut self,
        keys: [M::Key; N],
    ) -> [&mut MTarget<M>; N] {
        self.inner.get_disjoint_unchecked_mut_by_inner_key(keys)
    }
}

// ─── Core operations (require pointer metadata) ──────────────────────────────

impl<M> CastMapG<M>
where
    M: SlotMapTrait,
    M::Value: StableDeref,
    MTarget<M>: Pointee,
    <MTarget<M> as Pointee>::Metadata: Copy,
{
    /// Attempts to downcast a `StableCastKey<dyn Any>` to a concrete type.
    /// Returns `None` if the key belongs to a different map or the type doesn't
    /// match.
    #[inline]
    pub fn downcast_key<Concrete: 'static>(
        &self,
        key: StableCastKey<dyn Any, M::Key>,
    ) -> Option<StableCastKey<Concrete, M::Key>> {
        if key.map_id != self.map_id {
            return None;
        }
        let inner = unsafe { self.inner.downcast_key::<Concrete>(key.inner) }?;
        Some(stabilize(inner, self.map_id))
    }

    // ── insert ───────────────────────────────────────────────────────────

    /// Inserts a value and returns its [`StableCastKey`].
    #[inline]
    pub fn insert(&mut self, value: M::Value) -> StableCastKey<MTarget<M>, M::Key> {
        let key = self.inner.insert(value);
        stabilize(key, self.map_id)
    }

    /// Inserts a value produced by `func`, which receives the backing key.
    #[inline]
    pub fn insert_with_key(
        &mut self,
        func: impl FnOnce(M::Key) -> M::Value,
    ) -> StableCastKey<MTarget<M>, M::Key> {
        let key = self.inner.insert_with_key(func);
        stabilize(key, self.map_id)
    }

    /// Like [`insert_with_key`](Self::insert_with_key) but the closure may fail.
    #[inline]
    pub fn try_insert_with_key<E>(
        &mut self,
        func: impl FnOnce(M::Key) -> Result<M::Value, E>,
    ) -> Result<StableCastKey<MTarget<M>, M::Key>, E> {
        let key = self.inner.try_insert_with_key(func)?;
        Ok(stabilize(key, self.map_id))
    }

    // ── insert_sized ─────────────────────────────────────────────────────

    /// Inserts a concrete-typed smart pointer, returning a typed [`StableCastKey`].
    #[inline]
    pub fn insert_sized<ConcretePtr>(
        &mut self,
        value: ConcretePtr,
    ) -> StableCastKey<ConcretePtr::Target, M::Key>
    where
        ConcretePtr: std::ops::CoerceUnsized<M::Value> + Deref,
        ConcretePtr::Target: Sized,
    {
        let key = self.inner.insert_sized(value);
        stabilize(key, self.map_id)
    }

    /// Like [`insert_sized`](Self::insert_sized) but the closure receives a
    /// typed key.
    #[inline]
    pub fn insert_sized_with_key<ConcretePtr>(
        &mut self,
        func: impl FnOnce(StableCastKey<ConcretePtr::Target, M::Key>) -> ConcretePtr,
    ) -> StableCastKey<ConcretePtr::Target, M::Key>
    where
        ConcretePtr: std::ops::CoerceUnsized<M::Value> + Deref,
        ConcretePtr::Target: Sized,
    {
        let map_id = self.map_id;
        let key = self
            .inner
            .insert_sized_with_key(|ck| func(stabilize(ck, map_id)));
        stabilize(key, self.map_id)
    }

    /// Fallible version of [`insert_sized_with_key`](Self::insert_sized_with_key).
    #[inline]
    pub fn try_insert_sized_with_key<ConcretePtr, E>(
        &mut self,
        func: impl FnOnce(StableCastKey<ConcretePtr::Target, M::Key>) -> Result<ConcretePtr, E>,
    ) -> Result<StableCastKey<ConcretePtr::Target, M::Key>, E>
    where
        ConcretePtr: std::ops::CoerceUnsized<M::Value> + Deref,
        ConcretePtr::Target: Sized,
    {
        let map_id = self.map_id;
        let key = self
            .inner
            .try_insert_sized_with_key(|ck| func(stabilize(ck, map_id)))?;
        Ok(stabilize(key, self.map_id))
    }

    // ── insert_as ─────────────────────────────────────────────────────────

    /// Inserts a smart pointer, preserving the source pointer's metadata.
    #[inline]
    pub fn insert_as<SourcePtr>(
        &mut self,
        value: SourcePtr,
    ) -> StableCastKey<SourcePtr::Target, M::Key>
    where
        SourcePtr: std::ops::CoerceUnsized<M::Value> + Deref,
        SourcePtr::Target: Pointee<Metadata: Copy>,
    {
        let key = self.inner.insert_as(value);
        stabilize(key, self.map_id)
    }

    /// Like [`insert_as`](Self::insert_as) but the closure receives the backing key.
    #[inline]
    pub fn insert_as_with_key<SourcePtr>(
        &mut self,
        func: impl FnOnce(M::Key) -> SourcePtr,
    ) -> StableCastKey<SourcePtr::Target, M::Key>
    where
        SourcePtr: std::ops::CoerceUnsized<M::Value> + Deref,
        SourcePtr::Target: Pointee<Metadata: Copy>,
    {
        let key = self.inner.insert_as_with_key(func);
        stabilize(key, self.map_id)
    }

    /// Fallible version of [`insert_as_with_key`](Self::insert_as_with_key).
    #[inline]
    pub fn try_insert_as_with_key<SourcePtr, E>(
        &mut self,
        func: impl FnOnce(M::Key) -> Result<SourcePtr, E>,
    ) -> Result<StableCastKey<SourcePtr::Target, M::Key>, E>
    where
        SourcePtr: std::ops::CoerceUnsized<M::Value> + Deref,
        SourcePtr::Target: Pointee<Metadata: Copy>,
    {
        let key = self.inner.try_insert_as_with_key(func)?;
        Ok(stabilize(key, self.map_id))
    }

    // ── cast_key_of ───────────────────────────────────────────────────────

    /// Converts a backing `slotmap` key into a [`StableCastKey`] by reading
    /// pointer metadata from the stored value. Returns `None` if the key is
    /// stale.
    #[inline]
    pub fn cast_key_of(&self, key: M::Key) -> Option<StableCastKey<MTarget<M>, M::Key>> {
        let key = self.inner.cast_key_of(key)?;
        Some(stabilize(key, self.map_id))
    }

    // ── typed lookups (shared) ─────────────────────────────────────────────

    /// Typed lookup by [`StableCastKey`]. Returns `None` if the key belongs to
    /// a different map or the slot is no longer occupied.
    #[inline]
    pub fn get<T: ?Sized + Pointee>(&self, key: StableCastKey<T, M::Key>) -> Option<&T>
    where
        <T as Pointee>::Metadata: Copy,
    {
        if key.map_id != self.map_id {
            return None;
        }
        unsafe { self.inner.get(key.inner) }
    }

    /// Shared-reference lookup without bounds, version, or map-id checks.
    ///
    /// # Safety
    /// - The key's slot must be occupied with the matching version.
    /// - The key's pointer metadata must be valid for the data in that slot.
    #[inline]
    pub unsafe fn get_unchecked<T: ?Sized + Pointee>(&self, key: StableCastKey<T, M::Key>) -> &T
    where
        <T as Pointee>::Metadata: Copy,
    {
        self.inner.get_unchecked(key.inner)
    }

    /// Removes an element by its [`StableCastKey`]. Returns `None` if the key
    /// belongs to a different map.
    #[inline]
    pub fn remove<'a, T: ?Sized + Pointee>(
        &mut self,
        key: StableCastKey<T, M::Key>,
    ) -> Option<<M::Value as RetypePtr<'a>>::Retyped<T>>
    where
        <T as Pointee>::Metadata: Copy,
        M::Value: RetypePtr<'a>,
    {
        if key.map_id != self.map_id {
            return None;
        }
        unsafe { self.inner.remove(key.inner) }
    }

    // ── iterators ─────────────────────────────────────────────────────────

    /// Lazy iterator over all [`StableCastKey`]s.
    #[inline]
    pub fn keys(&self) -> impl Iterator<Item = StableCastKey<MTarget<M>, M::Key>> + '_ {
        let map_id = self.map_id;
        self.inner.keys().map(move |ck| stabilize(ck, map_id))
    }

    /// Shared iterator over all occupied `(StableCastKey, &output)` pairs.
    #[inline]
    pub fn iter(&self) -> Iter<'_, M> {
        Iter {
            inner: self.inner.iter(),
            map_id: self.map_id,
        }
    }

    // ── drain ─────────────────────────────────────────────────────────────

    /// Draining iterator. Removes all elements and yields them.
    #[inline]
    pub fn drain(&mut self) -> Drain<'_, M> {
        Drain {
            inner: self.inner.drain(),
            map_id: self.map_id,
        }
    }
}

// ─── Core operations requiring `&mut Output` ─────────────────────────────────

impl<M> CastMapG<M>
where
    M: SlotMapTrait,
    M::Value: StableDeref + DerefMut,
    MTarget<M>: Pointee,
    <MTarget<M> as Pointee>::Metadata: Copy,
{
    /// Mutable typed lookup by [`StableCastKey`].
    #[inline]
    pub fn get_mut<T: ?Sized + Pointee>(&mut self, key: StableCastKey<T, M::Key>) -> Option<&mut T>
    where
        <T as Pointee>::Metadata: Copy,
    {
        if key.map_id != self.map_id {
            return None;
        }
        unsafe { self.inner.get_mut(key.inner) }
    }

    /// Mutable-reference lookup without bounds, version, or map-id checks.
    ///
    /// # Safety
    /// - The key's slot must be occupied with the matching version.
    /// - The key's pointer metadata must be valid for the data in that slot.
    #[inline]
    pub unsafe fn get_unchecked_mut<T: ?Sized + Pointee>(
        &mut self,
        key: StableCastKey<T, M::Key>,
    ) -> &mut T
    where
        <T as Pointee>::Metadata: Copy,
    {
        self.inner.get_unchecked_mut(key.inner)
    }

    /// Retains only elements for which `f(key, &mut output)` returns `true`.
    #[inline]
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(StableCastKey<MTarget<M>, M::Key>, &mut MTarget<M>) -> bool,
    {
        let map_id = self.map_id;
        self.inner.retain(|ck, val| f(stabilize(ck, map_id), val));
    }

    /// Mutable iterator over all occupied `(StableCastKey, &mut output)` pairs.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, M> {
        IterMut {
            inner: self.inner.iter_mut(),
            map_id: self.map_id,
        }
    }

    /// Cross-typed mutable disjoint lookup. All keys must share the pointee type
    /// `T`. Returns `None` if any key belongs to a different map, is invalid, or
    /// two keys alias the same slot.
    #[inline]
    pub fn get_disjoint_mut<T: ?Sized + Pointee, const N: usize>(
        &mut self,
        keys: [StableCastKey<T, M::Key>; N],
    ) -> Option<[&mut T; N]>
    where
        <T as Pointee>::Metadata: Copy,
    {
        if keys.iter().any(|k| k.map_id != self.map_id) {
            return None;
        }
        let inner = keys.map(|k| k.inner);
        unsafe { self.inner.get_disjoint_mut(inner) }
    }

    /// Cross-typed mutable disjoint lookup without validity, uniqueness, or
    /// map-id checks.
    ///
    /// # Safety
    /// - Every key must be valid for this map and no two keys may alias one slot.
    /// - Each key's pointer metadata must be valid for the data in its slot.
    #[inline]
    pub unsafe fn get_disjoint_unchecked_mut<T: ?Sized + Pointee, const N: usize>(
        &mut self,
        keys: [StableCastKey<T, M::Key>; N],
    ) -> [&mut T; N]
    where
        <T as Pointee>::Metadata: Copy,
    {
        let inner = keys.map(|k| k.inner);
        self.inner.get_disjoint_unchecked_mut(inner)
    }
}

// ─── Index / IndexMut ──────────────────────────────────────────────────────────

impl<M> std::ops::Index<StableCastKey<MTarget<M>, M::Key>> for CastMapG<M>
where
    M: SlotMapTrait,
    M::Value: StableDeref,
    MTarget<M>: Pointee,
    <MTarget<M> as Pointee>::Metadata: Copy,
{
    type Output = MTarget<M>;

    #[inline]
    fn index(&self, key: StableCastKey<MTarget<M>, M::Key>) -> &Self::Output {
        self.get(key).expect("invalid StableCastKey for this map")
    }
}

impl<M> std::ops::IndexMut<StableCastKey<MTarget<M>, M::Key>> for CastMapG<M>
where
    M: SlotMapTrait,
    M::Value: StableDeref + DerefMut,
    MTarget<M>: Pointee,
    <MTarget<M> as Pointee>::Metadata: Copy,
{
    #[inline]
    fn index_mut(&mut self, key: StableCastKey<MTarget<M>, M::Key>) -> &mut Self::Output {
        self.get_mut(key)
            .expect("invalid StableCastKey for this map")
    }
}

// ─── Iter (shared) ───────────────────────────────────────────────────────────

pub struct Iter<'a, M>
where
    M: SlotMapTrait + 'a,
    M::Value: 'a,
{
    inner: unsafe_cast_map::Iter<'a, M>,
    map_id: MapId,
}

impl<'a, M> Iterator for Iter<'a, M>
where
    M: SlotMapTrait + 'a,
    M::Value: StableDeref + 'a,
    MTarget<M>: Pointee,
    <MTarget<M> as Pointee>::Metadata: Copy,
{
    type Item = (StableCastKey<MTarget<M>, M::Key>, &'a MTarget<M>);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (ck, r) = self.inner.next()?;
        Some((stabilize(ck, self.map_id), r))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

// ─── IterMut ─────────────────────────────────────────────────────────────────

pub struct IterMut<'a, M>
where
    M: SlotMapTrait + 'a,
    M::Value: 'a,
{
    inner: unsafe_cast_map::IterMut<'a, M>,
    map_id: MapId,
}

impl<'a, M> Iterator for IterMut<'a, M>
where
    M: SlotMapTrait + 'a,
    M::Value: StableDeref + DerefMut + 'a,
    MTarget<M>: Pointee,
    <MTarget<M> as Pointee>::Metadata: Copy,
{
    type Item = (StableCastKey<MTarget<M>, M::Key>, &'a mut MTarget<M>);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (ck, val) = self.inner.next()?;
        Some((stabilize(ck, self.map_id), val))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

// ─── Drain ───────────────────────────────────────────────────────────────────

pub struct Drain<'a, M>
where
    M: SlotMapTrait + 'a,
    M::Value: 'a,
{
    inner: unsafe_cast_map::Drain<'a, M>,
    map_id: MapId,
}

impl<'a, M> Iterator for Drain<'a, M>
where
    M: SlotMapTrait + 'a,
    M::Value: StableDeref + 'a,
    MTarget<M>: Pointee,
    <MTarget<M> as Pointee>::Metadata: Copy,
{
    type Item = (StableCastKey<MTarget<M>, M::Key>, M::Value);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (ck, val) = self.inner.next()?;
        Some((stabilize(ck, self.map_id), val))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

// ─── IntoIter (owning) ───────────────────────────────────────────────────────

pub struct IntoIter<M>
where
    M: SlotMapTrait,
{
    inner: unsafe_cast_map::IntoIter<M>,
    map_id: MapId,
}

impl<M> Iterator for IntoIter<M>
where
    M: SlotMapTrait,
    M::Value: StableDeref,
    MTarget<M>: Pointee,
    <MTarget<M> as Pointee>::Metadata: Copy,
{
    type Item = (StableCastKey<MTarget<M>, M::Key>, M::Value);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (ck, val) = self.inner.next()?;
        Some((stabilize(ck, self.map_id), val))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<M> IntoIterator for CastMapG<M>
where
    M: SlotMapTrait,
    M::Value: StableDeref,
    MTarget<M>: Pointee,
    <MTarget<M> as Pointee>::Metadata: Copy,
{
    type Item = (StableCastKey<MTarget<M>, M::Key>, M::Value);
    type IntoIter = IntoIter<M>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            inner: self.inner.into_iter(),
            map_id: self.map_id,
        }
    }
}

impl<'a, M> IntoIterator for &'a CastMapG<M>
where
    M: SlotMapTrait + 'a,
    M::Value: StableDeref + 'a,
    MTarget<M>: Pointee,
    <MTarget<M> as Pointee>::Metadata: Copy,
{
    type Item = (StableCastKey<MTarget<M>, M::Key>, &'a MTarget<M>);
    type IntoIter = Iter<'a, M>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, M> IntoIterator for &'a mut CastMapG<M>
where
    M: SlotMapTrait + 'a,
    M::Value: StableDeref + DerefMut + 'a,
    MTarget<M>: Pointee,
    <MTarget<M> as Pointee>::Metadata: Copy,
{
    type Item = (StableCastKey<MTarget<M>, M::Key>, &'a mut MTarget<M>);
    type IntoIter = IterMut<'a, M>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

// ─── Type aliases ──────────────────────────────────────────────────────────────

/// Safe castable-key map backed by [`slotmap::SlotMap`] (sparse storage).
pub type CastMap<K, Ptr> = CastMapG<SlotMap<K, Ptr>>;

/// Safe castable-key map backed by [`slotmap::DenseSlotMap`] (contiguous
/// storage, fast iteration).
pub type DenseCastMap<K, Ptr> = CastMapG<DenseSlotMap<K, Ptr>>;

/// Convenience alias: [`CastMap`] storing `Box<T>` (e.g. `dyn Any`).
pub type BoxCastMap<K, T> = CastMap<K, Box<T>>;

/// Convenience alias: [`DenseCastMap`] storing `Box<T>` (e.g. `dyn Any`).
pub type BoxDenseCastMap<K, T> = DenseCastMap<K, Box<T>>;

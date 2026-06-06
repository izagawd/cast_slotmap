//! Low-level cast map generic over the backing `slotmap` map, without per-map
//! identity checks.
//!
//! [`UnsafeCastMapG`] is the single source of truth for the cast logic — the
//! pointer-metadata reconstruction behind the typed `get` / `get_mut` /
//! `remove` family. Its one type parameter is the backing map `M`
//! ([`SlotMapTrait`](crate::slotmap_trait::SlotMapTrait)); the backing key and stored
//! pointer types are read off `M` as `M::Key` and `M::Value`. The same code
//! serves both [`slotmap::SlotMap`] and [`slotmap::DenseSlotMap`], exposed as the
//! [`UnsafeCastMap`] and [`UnsafeDenseCastMap`] type aliases.
//!
//! Typed lookups go through [`CastKey`], but `get`, `get_mut`, `remove`, and
//! `downcast_key` are **`unsafe`**: the caller must ensure the key's pointer
//! metadata is valid for the data stored at that slot. For a safe wrapper that
//! checks a per-map [`MapId`](crate::map_id::MapId), see
//! [`CastMapG`](crate::cast_map::CastMapG) (and its aliases).
//!
//! ## Relationship to `slotmap`
//! Every method forwards to the backing `slotmap` map through the
//! [`SlotMapTrait`](crate::slotmap_trait::SlotMapTrait) trait. Mutating methods
//! (`insert*`, `remove`, `reserve`, `clear`, `retain`, `drain`) take `&mut self`
//! because that is `slotmap`'s signature. `detach` / `reattach` are exposed here
//! (both `slotmap` maps support them) but **not** on the checked
//! [`CastMapG`](crate::cast_map::CastMapG): reattaching a different concrete type
//! under an existing key would leave that key's cached pointer metadata stale,
//! so it is left to the caller's `unsafe` discipline for now. There is
//! intentionally **no** `get_slot`,
//! `get_by_index_only`, `reset`, or `unsafe_clone` / `clone_mut` family:
//! `slotmap` exposes none of those. `iter` is a plain safe shared iterator
//! because `slotmap`'s `get` borrows `&self` while `insert` borrows `&mut self`,
//! so a live reference can never coexist with an insert.

use std::any::{Any, TypeId};
use std::collections::TryReserveError;
use std::ops::{Deref, DerefMut};
use std::ptr::Pointee;

use slotmap::{DenseSlotMap, Key, SlotMap};

use crate::slotmap_trait::{MTarget, SlotMapTrait};
use crate::cast_key::CastKey;
use crate::retype_ptr::RetypePtr;
use stable_deref_trait::StableDeref;

// ─── Conversion helper ───────────────────────────────────────────────────────

/// Build a cast key from a `slotmap` key and a reference (for pointer metadata).
#[inline]
fn to_castable<K: Key, O: ?Sized + Pointee>(key: K, reference: &O) -> CastKey<O, K>
where
    <O as Pointee>::Metadata: Copy,
{
    let metadata = std::ptr::metadata(reference as *const O);
    CastKey { key, metadata }
}

// ─── UnsafeCastMapG ────────────────────────────────────────────────────────────

/// A `slotmap` wrapper, generic over the backing map `M`, that supports
/// typed lookups via [`CastKey`].
///
/// The backing key type is `M::Key` and the stored smart pointer is `M::Value`
/// (which must implement [`StableDeref`] so pointer-metadata casts are sound);
/// the map's "output" type is `<M::Value as Deref>::Target`. `M` is the backing
/// `slotmap` map ([`slotmap::SlotMap`] or [`slotmap::DenseSlotMap`]); see the
/// [`UnsafeCastMap`] / [`UnsafeDenseCastMap`] aliases.
pub struct UnsafeCastMapG<M> {
    pub(crate) inner: M,
}

// ─── Clone ───────────────────────────────────────────────────────────────────

impl<M> Clone for UnsafeCastMapG<M>
where
    M: SlotMapTrait + Clone,
{
    /// Cloning preserves every slot's key and version, so keys valid on the
    /// original stay valid on the clone (the safe
    /// [`CastMapG`](crate::cast_map::CastMapG) layer, by contrast, mints a fresh
    /// [`MapId`](crate::map_id::MapId) on clone).
    #[inline]
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        self.inner.clone_from(&source.inner);
    }
}

impl<M> Default for UnsafeCastMapG<M>
where
    M: SlotMapTrait,
    M::Value: StableDeref,
{
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// ─── Basic methods (no pointer metadata needed) ──────────────────────────────

impl<M> UnsafeCastMapG<M>
where
    M: SlotMapTrait,
    M::Value: StableDeref,
{
    /// Creates a new, empty map.
    #[inline]
    pub fn new() -> Self {
        Self { inner: M::empty() }
    }

    /// Creates a new map with the given pre-allocated capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: M::with_capacity(capacity),
        }
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
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.inner.try_reserve(additional)
    }

    /// Removes all elements from the map. Outstanding keys are invalidated
    /// (`slotmap` bumps slot versions on clear).
    #[inline]
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Returns whether the backing key is still live (delegates to the map's
    /// `contains_key`).
    #[inline]
    pub fn contains_key<T: ?Sized + Pointee>(&self, key: CastKey<T, M::Key>) -> bool
    where
        <T as Pointee>::Metadata: Copy,
    {
        self.inner.contains_key(key.inner_key())
    }

    // ── backing-key access ───────────────────────────────────────────────

    /// Shared-reference lookup using the backing `slotmap` key directly.
    #[inline]
    pub fn get_by_inner_key(&self, key: M::Key) -> Option<&MTarget<M>> {
        self.inner.get(key).map(|p| &**p)
    }

    /// Removes an element by its backing `slotmap` key, returning the pointer.
    #[inline]
    pub fn remove_by_inner_key(&mut self, key: M::Key) -> Option<M::Value> {
        self.inner.remove(key)
    }

    /// Shared iterator over output references only.
    #[inline]
    pub fn values(&self) -> impl Iterator<Item = &MTarget<M>> + '_ {
        self.inner.values().map(|p| &**p)
    }
}

// ── backing-key access requiring `&mut Output` ───────────────────────────────

impl<M> UnsafeCastMapG<M>
where
    M: SlotMapTrait,
    M::Value: StableDeref + DerefMut,
{
    /// Mutable-reference lookup using the backing `slotmap` key directly.
    #[inline]
    pub fn get_by_inner_key_mut(&mut self, key: M::Key) -> Option<&mut MTarget<M>> {
        self.inner.get_mut(key).map(|p| &mut **p)
    }

    /// Mutable iterator over output references only.
    #[inline]
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut MTarget<M>> + '_ {
        self.inner.values_mut().map(|p| &mut **p)
    }

    /// Mutable disjoint lookup by backing `slotmap` keys, yielding erased output
    /// references. Returns `None` if any key is invalid or two keys alias the
    /// same slot.
    #[inline]
    pub fn get_disjoint_mut_by_inner_key<const N: usize>(
        &mut self,
        keys: [M::Key; N],
    ) -> Option<[&mut MTarget<M>; N]> {
        let stored = self.inner.get_disjoint_mut(keys)?;
        Some(stored.map(|p| &mut **p))
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
        let stored = self.inner.get_disjoint_unchecked_mut(keys);
        stored.map(|p| &mut **p)
    }
}

// ─── Core operations (require pointer metadata) ──────────────────────────────

impl<M> UnsafeCastMapG<M>
where
    M: SlotMapTrait,
    M::Value: StableDeref,
    MTarget<M>: Pointee,
    <MTarget<M> as Pointee>::Metadata: Copy,
{
    /// Attempts to downcast a `CastKey<dyn Any, ..>` to a concrete-typed key.
    /// Returns `None` if the actual type doesn't match `Concrete`.
    ///
    /// # Safety
    /// The key's metadata must be a valid vtable for `dyn Any` pointing at the
    /// data stored in that slot.
    #[inline]
    pub unsafe fn downcast_key<Concrete: 'static>(
        &self,
        key: CastKey<dyn Any, M::Key>,
    ) -> Option<CastKey<Concrete, M::Key>> {
        let stored = self.inner.get(key.inner_key())?;
        let base: &MTarget<M> = &**stored;
        let data_as_any: &dyn Any =
            &*std::ptr::from_raw_parts(base as *const MTarget<M> as *const (), key.metadata());
        if data_as_any.type_id() == TypeId::of::<Concrete>() {
            Some(CastKey {
                key: key.key,
                metadata: (),
            })
        } else {
            None
        }
    }

    // ── insert ───────────────────────────────────────────────────────────

    /// Inserts a smart pointer, returning a key with metadata.
    #[inline]
    pub fn insert(&mut self, value: M::Value) -> CastKey<MTarget<M>, M::Key> {
        self.insert_with_key(|_| value)
    }

    /// Inserts a smart pointer produced by `func`, which receives the backing
    /// key that will identify the inserted element.
    #[inline]
    pub fn insert_with_key(
        &mut self,
        func: impl FnOnce(M::Key) -> M::Value,
    ) -> CastKey<MTarget<M>, M::Key> {
        self.try_insert_with_key(|key| Ok::<_, ()>(func(key)))
            .unwrap()
    }

    /// Like [`insert_with_key`](Self::insert_with_key) but the closure may
    /// return `Err`, in which case nothing is inserted.
    #[inline]
    pub fn try_insert_with_key<E>(
        &mut self,
        func: impl FnOnce(M::Key) -> Result<M::Value, E>,
    ) -> Result<CastKey<MTarget<M>, M::Key>, E> {
        let inner_key = self.inner.try_insert_with_key(func)?;
        let reference = self
            .inner
            .get(inner_key)
            .expect("just-inserted key is live");
        Ok(to_castable::<M::Key, MTarget<M>>(inner_key, &**reference))
    }

    // ── cast_key_of ──────────────────────────────────────────────────────

    /// Converts a backing `slotmap` key into a full [`CastKey`] by reading the
    /// stored value's pointer metadata. Returns `None` if the key is stale.
    #[inline]
    pub fn cast_key_of(&self, key: M::Key) -> Option<CastKey<MTarget<M>, M::Key>> {
        let reference = self.inner.get(key)?;
        Some(to_castable::<M::Key, MTarget<M>>(key, &**reference))
    }

    // ── typed lookups (shared) ─────────────────────────────────────────────

    /// Cross-typed shared-reference lookup. Reconstructs a fat pointer to `T`
    /// from the stored output's data pointer and the key's metadata.
    ///
    /// # Safety
    /// The key's pointer metadata must be valid for the data stored at that
    /// slot (e.g. for a trait object, the correct vtable for the concrete type).
    #[inline]
    pub unsafe fn get<T: ?Sized + Pointee>(&self, key: CastKey<T, M::Key>) -> Option<&T>
    where
        <T as Pointee>::Metadata: Copy,
    {
        let stored = self.inner.get(key.inner_key())?;
        let base: &MTarget<M> = &**stored;
        let data_ptr: *const () = (base as *const MTarget<M>).cast();
        let fat_ptr: *const T = std::ptr::from_raw_parts(data_ptr, key.metadata());
        Some(&*fat_ptr)
    }

    /// Shared-reference lookup without bounds or version checks.
    ///
    /// # Safety
    /// - The key's slot must be occupied with the matching version.
    /// - The key's pointer metadata must be valid for the data in that slot.
    #[inline]
    pub unsafe fn get_unchecked<T: ?Sized + Pointee>(&self, key: CastKey<T, M::Key>) -> &T
    where
        <T as Pointee>::Metadata: Copy,
    {
        let stored = self.inner.get_unchecked(key.inner_key());
        let base: &MTarget<M> = &**stored;
        let data_ptr: *const () = (base as *const MTarget<M>).cast();
        let fat_ptr: *const T = std::ptr::from_raw_parts(data_ptr, key.metadata());
        &*fat_ptr
    }

    /// Removes an element by its [`CastKey`], returning the owned smart pointer
    /// re-typed to `T`.
    ///
    /// # Safety
    /// The key's pointer metadata must be valid for the data stored at that slot.
    #[inline]
    pub unsafe fn remove<'a, T: ?Sized + Pointee>(
        &mut self,
        key: CastKey<T, M::Key>,
    ) -> Option<<M::Value as RetypePtr<'a>>::Retyped<T>>
    where
        <T as Pointee>::Metadata: Copy,
        M::Value: RetypePtr<'a>,
    {
        let stored = self.inner.remove(key.inner_key())?;
        Some(stored.retype::<T>(key.metadata()))
    }

    // ── iterators ────────────────────────────────────────────────────────

    /// Lazy iterator over all [`CastKey`]s.
    #[inline]
    pub fn keys(&self) -> impl Iterator<Item = CastKey<MTarget<M>, M::Key>> + '_ {
        self.inner
            .iter()
            .map(|(k, p)| to_castable::<M::Key, MTarget<M>>(k, &**p))
    }

    /// Shared iterator over all occupied `(CastKey, &output)` pairs (safe).
    #[inline]
    pub fn iter(&self) -> Iter<'_, M> {
        Iter {
            inner: self.inner.iter(),
        }
    }

    // ── drain ────────────────────────────────────────────────────────────

    /// Draining iterator. Removes all elements and yields them.
    #[inline]
    pub fn drain(&mut self) -> Drain<'_, M> {
        Drain {
            inner: self.inner.drain(),
        }
    }
}

// ─── Core operations requiring `&mut Output` ─────────────────────────────────

impl<M> UnsafeCastMapG<M>
where
    M: SlotMapTrait,
    M::Value: StableDeref + DerefMut,
    MTarget<M>: Pointee,
    <MTarget<M> as Pointee>::Metadata: Copy,
{
    /// Cross-typed mutable-reference lookup.
    ///
    /// # Safety
    /// The key's pointer metadata must be valid for the data stored at that slot.
    #[inline]
    pub unsafe fn get_mut<T: ?Sized + Pointee>(&mut self, key: CastKey<T, M::Key>) -> Option<&mut T>
    where
        <T as Pointee>::Metadata: Copy,
    {
        let stored = self.inner.get_mut(key.inner_key())?;
        let base: &mut MTarget<M> = &mut **stored;
        let data_ptr: *mut () = (base as *mut MTarget<M>).cast();
        let fat_ptr: *mut T = std::ptr::from_raw_parts_mut(data_ptr, key.metadata());
        Some(&mut *fat_ptr)
    }

    /// Mutable-reference lookup without bounds or version checks.
    ///
    /// # Safety
    /// - The key's slot must be occupied with the matching version.
    /// - The key's pointer metadata must be valid for the data in that slot.
    #[inline]
    pub unsafe fn get_unchecked_mut<T: ?Sized + Pointee>(
        &mut self,
        key: CastKey<T, M::Key>,
    ) -> &mut T
    where
        <T as Pointee>::Metadata: Copy,
    {
        let stored = self.inner.get_unchecked_mut(key.inner_key());
        let base: &mut MTarget<M> = &mut **stored;
        let data_ptr: *mut () = (base as *mut MTarget<M>).cast();
        let fat_ptr: *mut T = std::ptr::from_raw_parts_mut(data_ptr, key.metadata());
        &mut *fat_ptr
    }

    /// Retains only elements for which `f(key, &mut output)` returns `true`.
    #[inline]
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(CastKey<MTarget<M>, M::Key>, &mut MTarget<M>) -> bool,
    {
        self.inner.retain(|inner_key, stored| {
            let patched = to_castable::<M::Key, MTarget<M>>(inner_key, &**stored);
            f(patched, &mut **stored)
        })
    }

    /// Mutable iterator over all occupied `(CastKey, &mut output)` pairs (safe).
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, M> {
        IterMut {
            inner: self.inner.iter_mut(),
        }
    }

    /// Cross-typed mutable disjoint lookup. All keys must share the pointee type
    /// `T`; each fat pointer to `T` is rebuilt from that key's own metadata.
    /// Returns `None` if any key is invalid or two keys alias the same slot.
    ///
    /// # Safety
    /// Each key's pointer metadata must be valid for the data in its slot.
    #[inline]
    pub unsafe fn get_disjoint_mut<T: ?Sized + Pointee, const N: usize>(
        &mut self,
        keys: [CastKey<T, M::Key>; N],
    ) -> Option<[&mut T; N]>
    where
        <T as Pointee>::Metadata: Copy,
    {
        let metadata = keys.map(|k| k.metadata());
        let inner_keys = keys.map(|k| k.inner_key());
        let stored = self.inner.get_disjoint_mut(inner_keys)?;
        let mut i = 0;
        let out = stored.map(|p| {
            let meta = metadata[i];
            i += 1;
            let base: &mut MTarget<M> = &mut **p;
            let data_ptr: *mut () = (base as *mut MTarget<M>).cast();
            unsafe { &mut *std::ptr::from_raw_parts_mut(data_ptr, meta) }
        });
        Some(out)
    }

    /// Like [`get_disjoint_mut`](Self::get_disjoint_mut) but without validity or
    /// uniqueness checks.
    ///
    /// # Safety
    /// - Every key must be valid for this map and no two keys may alias one slot.
    /// - Each key's pointer metadata must be valid for the data in its slot.
    #[inline]
    pub unsafe fn get_disjoint_unchecked_mut<T: ?Sized + Pointee, const N: usize>(
        &mut self,
        keys: [CastKey<T, M::Key>; N],
    ) -> [&mut T; N]
    where
        <T as Pointee>::Metadata: Copy,
    {
        let metadata = keys.map(|k| k.metadata());
        let inner_keys = keys.map(|k| k.inner_key());
        let stored = self.inner.get_disjoint_unchecked_mut(inner_keys);
        let mut i = 0;
        stored.map(|p| {
            let meta = metadata[i];
            i += 1;
            let base: &mut MTarget<M> = &mut **p;
            let data_ptr: *mut () = (base as *mut MTarget<M>).cast();
            unsafe { &mut *std::ptr::from_raw_parts_mut(data_ptr, meta) }
        })
    }
}

// ─── detach / reattach (backing key) ────────────────────────────

impl<M> UnsafeCastMapG<M>
where
    M: SlotMapTrait,
    M::Value: StableDeref,
{
    /// Detaches an element by its backing `slotmap` key, returning the stored
    /// pointer but keeping the slot reservable so the key can be reused with
    /// [`reattach_by_inner_key`](Self::reattach_by_inner_key). Forwards to
    /// `slotmap`'s `detach`, which both `SlotMap` and `DenseSlotMap` provide.
    #[inline]
    pub fn detach_by_inner_key(&mut self, key: M::Key) -> Option<M::Value> {
        self.inner.detach(key)
    }

    /// Reattaches an already-erased `value` (e.g. a `Box<dyn Any>`) at a slot
    /// previously freed with [`detach_by_inner_key`](Self::detach_by_inner_key),
    /// reusing `key`. Use [`reattach`](Self::reattach) to pass a [`CastKey`]
    /// rather than a raw backing key.
    ///
    /// Reattaching a value whose concrete type differs from the original leaves
    /// any [`CastKey`] previously minted for that slot with stale pointer
    /// metadata; using such a key with the `unsafe` typed accessors is then
    /// undefined behavior. This hazard is why `reattach` lives only on the
    /// unsafe map and not (yet) on the checked
    /// [`CastMapG`](crate::cast_map::CastMapG).
    ///
    /// # Panics
    /// Panics if `key` is not in a detached state, or if the map is full
    /// (mirrors `slotmap`'s `reattach`).
    #[inline]
    pub fn reattach_by_inner_key(&mut self, key: M::Key, value: M::Value) {
        self.inner.reattach(key, value);
    }
}

// ─── detach / reattach (cast key) ────────────────────────────────

impl<M> UnsafeCastMapG<M>
where
    M: SlotMapTrait,
    M::Value: StableDeref,
    MTarget<M>: Pointee,
    <MTarget<M> as Pointee>::Metadata: Copy,
{
    /// Detaches an element by its [`CastKey`], returning the owned smart pointer
    /// re-typed to `T` (so a `Box<dyn Any>` map hands back a `Box<T>`). Unlike
    /// [`remove`](Self::remove) the slot stays reservable: the same key can be
    /// reused with [`reattach`](Self::reattach) or
    /// [`reattach_by_inner_key`](Self::reattach_by_inner_key) (erased pointer).
    ///
    /// # Safety
    /// The key's pointer metadata must be valid for the data stored at that slot.
    #[inline]
    pub unsafe fn detach<'a, T: ?Sized + Pointee>(
        &mut self,
        key: CastKey<T, M::Key>,
    ) -> Option<<M::Value as RetypePtr<'a>>::Retyped<T>>
    where
        <T as Pointee>::Metadata: Copy,
        M::Value: RetypePtr<'a>,
    {
        let stored = self.inner.detach(key.inner_key())?;
        Some(stored.retype::<T>(key.metadata()))
    }

    /// Reattaches `value` at a slot freed with [`detach`](Self::detach), reusing
    /// `key`. `value` is the pointer the backing `slotmap` stores directly
    /// (`M::Value`, e.g. `Box<dyn Any>`); a concrete pointer like `Box<Dog>`
    /// unsizes to it implicitly at the call site. `key` is the erased-target
    /// [`CastKey`] the map itself issues — `CastKey<MTarget<M>, M::Key>`, as
    /// returned by [`insert`](Self::insert), [`keys`](Self::keys), or
    /// [`cast_key_of`](Self::cast_key_of); a concrete-typed key (e.g. from
    /// [`downcast_key`](Self::downcast_key)) reaches it via
    /// [`CastKey::upcast`](crate::cast_key::CastKey::upcast).
    ///
    /// Reattaching a value of a different concrete type than the slot last held
    /// leaves any retained [`CastKey`] with stale metadata; using such a key with
    /// the `unsafe` typed accessors is then undefined behavior — the same hazard
    /// as [`reattach_by_inner_key`](Self::reattach_by_inner_key), and why neither
    /// is offered on the checked [`CastMapG`](crate::cast_map::CastMapG).
    ///
    /// # Panics
    /// Panics if `key` is not detached or the map is full.
    #[inline]
    pub fn reattach(&mut self, key: CastKey<MTarget<M>, M::Key>, value: M::Value) {
        self.inner.reattach(key.inner_key(), value);
    }
}

// ─── Iter (shared) ───────────────────────────────────────────────────────────

/// Shared iterator over `(CastKey, &Target)` pairs.
pub struct Iter<'a, M>
where
    M: SlotMapTrait + 'a,
    M::Value: 'a,
{
    inner: <M as SlotMapTrait>::Iter<'a>,
}

impl<'a, M> Iterator for Iter<'a, M>
where
    M: SlotMapTrait + 'a,
    M::Value: StableDeref + 'a,
    MTarget<M>: Pointee,
    <MTarget<M> as Pointee>::Metadata: Copy,
{
    type Item = (CastKey<MTarget<M>, M::Key>, &'a MTarget<M>);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (k, p) = self.inner.next()?;
        let r: &'a MTarget<M> = &**p;
        Some((to_castable::<M::Key, MTarget<M>>(k, r), r))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

// ─── IterMut ─────────────────────────────────────────────────────────────────

/// Mutable iterator over `(CastKey, &mut Target)` pairs.
pub struct IterMut<'a, M>
where
    M: SlotMapTrait + 'a,
    M::Value: 'a,
{
    inner: <M as SlotMapTrait>::IterMut<'a>,
}

impl<'a, M> Iterator for IterMut<'a, M>
where
    M: SlotMapTrait + 'a,
    M::Value: StableDeref + DerefMut + 'a,
    MTarget<M>: Pointee,
    <MTarget<M> as Pointee>::Metadata: Copy,
{
    type Item = (CastKey<MTarget<M>, M::Key>, &'a mut MTarget<M>);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (k, stored) = self.inner.next()?;
        let patched = to_castable::<M::Key, MTarget<M>>(k, &**stored);
        Some((patched, &mut **stored))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

// ─── Drain ───────────────────────────────────────────────────────────────────

/// Draining iterator over `(CastKey, value)`, emptying the map.
pub struct Drain<'a, M>
where
    M: SlotMapTrait + 'a,
    M::Value: 'a,
{
    inner: <M as SlotMapTrait>::Drain<'a>,
}

impl<'a, M> Iterator for Drain<'a, M>
where
    M: SlotMapTrait + 'a,
    M::Value: StableDeref + 'a,
    MTarget<M>: Pointee,
    <MTarget<M> as Pointee>::Metadata: Copy,
{
    type Item = (CastKey<MTarget<M>, M::Key>, M::Value);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (k, value) = self.inner.next()?;
        let patched = to_castable::<M::Key, MTarget<M>>(k, &*value);
        Some((patched, value))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

// ─── IntoIter (owning) ───────────────────────────────────────────────────────

/// Owning iterator over `(CastKey, value)` pairs.
pub struct IntoIter<M>
where
    M: SlotMapTrait,
{
    inner: <M as SlotMapTrait>::IntoIter,
}

impl<M> Iterator for IntoIter<M>
where
    M: SlotMapTrait,
    M::Value: StableDeref,
    MTarget<M>: Pointee,
    <MTarget<M> as Pointee>::Metadata: Copy,
{
    type Item = (CastKey<MTarget<M>, M::Key>, M::Value);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (k, value) = self.inner.next()?;
        let patched = to_castable::<M::Key, MTarget<M>>(k, &*value);
        Some((patched, value))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<M> IntoIterator for UnsafeCastMapG<M>
where
    M: SlotMapTrait,
    M::Value: StableDeref,
    MTarget<M>: Pointee,
    <MTarget<M> as Pointee>::Metadata: Copy,
{
    type Item = (CastKey<MTarget<M>, M::Key>, M::Value);
    type IntoIter = IntoIter<M>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            inner: self.inner.into_pairs(),
        }
    }
}

impl<'a, M> IntoIterator for &'a UnsafeCastMapG<M>
where
    M: SlotMapTrait + 'a,
    M::Value: StableDeref + 'a,
    MTarget<M>: Pointee,
    <MTarget<M> as Pointee>::Metadata: Copy,
{
    type Item = (CastKey<MTarget<M>, M::Key>, &'a MTarget<M>);
    type IntoIter = Iter<'a, M>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, M> IntoIterator for &'a mut UnsafeCastMapG<M>
where
    M: SlotMapTrait + 'a,
    M::Value: StableDeref + DerefMut + 'a,
    MTarget<M>: Pointee,
    <MTarget<M> as Pointee>::Metadata: Copy,
{
    type Item = (CastKey<MTarget<M>, M::Key>, &'a mut MTarget<M>);
    type IntoIter = IterMut<'a, M>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

// ─── Type aliases ──────────────────────────────────────────────────────────────

/// Raw castable-key map backed by [`slotmap::SlotMap`] (sparse storage).
pub type UnsafeCastMap<K, Ptr> = UnsafeCastMapG<SlotMap<K, Ptr>>;

/// Raw castable-key map backed by [`slotmap::DenseSlotMap`] (contiguous storage,
/// fast iteration).
pub type UnsafeDenseCastMap<K, Ptr> = UnsafeCastMapG<DenseSlotMap<K, Ptr>>;

/// Convenience alias: [`UnsafeCastMap`] storing `Box<T>` (e.g. `dyn Any`).
pub type UnsafeBoxCastMap<K, T> = UnsafeCastMap<K, Box<T>>;

/// Convenience alias: [`UnsafeDenseCastMap`] storing `Box<T>` (e.g. `dyn Any`).
pub type UnsafeBoxDenseCastMap<K, T> = UnsafeDenseCastMap<K, Box<T>>;

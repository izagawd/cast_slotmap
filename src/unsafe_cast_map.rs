//! Low-level cast map generic over the backing `slotmap` map; its typed
//! accessors are `unsafe` (the caller upholds the metadata contract spelled out
//! below).
//!
//! [`UnsafeCastMapG`] is the single source of truth for the cast logic — the
//! pointer-metadata reconstruction behind the typed `get` / `get_mut` /
//! `remove` family. Its one type parameter is the backing map `M`
//! ([`SlotMapTrait`](crate::slotmap_trait::SlotMapTrait)); the backing key and stored
//! pointer types are read off `M` as `M::Key` and `M::Value`. The same code
//! serves both [`slotmap::SlotMap`] and [`slotmap::DenseSlotMap`], exposed as the
//! [`UnsafeCastMap`] and [`UnsafeDenseCastMap`] type aliases.
//!
//! Typed lookups go through [`CastKey`], but `get`, `get_mut`, and `remove`
//! are **`unsafe`**: the caller must ensure the key's pointer
//! metadata is valid for the data stored at that slot. For a safe wrapper that
//! validates each lookup against the slot's stored concrete type id (see
//! [`ConcreteTypeId`](crate::cast_box::ConcreteTypeId)), see
//! [`CastMapG`](crate::cast_map::CastMapG) (and its aliases).
//!
//! ## Relationship to `slotmap`
//! Every method forwards to the backing `slotmap` map through the
//! [`SlotMapTrait`](crate::slotmap_trait::SlotMapTrait) trait. `detach` /
//! `reattach` are exposed here
//! (both `slotmap` maps support them) but **not** on the checked
//! [`CastMapG`](crate::cast_map::CastMapG): reattaching a different concrete type
//! under an existing key would leave that key's cached pointer metadata stale,
//! so it is left to the caller's `unsafe` discipline.

use std::collections::TryReserveError;
use std::ops::{Deref, DerefMut};
use std::ptr::Pointee;

use slotmap::{DenseSlotMap, Key, SlotMap};

use crate::cast_key::CastKey;
use crate::retype_ptr::RetypePtr;
use crate::slotmap_trait::{MTarget, SlotMapTrait};
use stable_deref_trait::StableDeref;

// ─── Conversion helper ───────────────────────────────────────────────────────

/// Build a cast key from a `slotmap` key and a reference (for pointer metadata).
#[inline]
fn to_castable<K: Key, O: ?Sized + Pointee>(key: K, reference: &O) -> CastKey<O, K>
where
    <O as Pointee>::Metadata: Copy,
{
    let metadata = std::ptr::metadata(reference as *const O);
    unsafe { CastKey::from_raw_parts(key, metadata) }
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
    /// original stay valid on the clone. (The checked
    /// [`CastMapG`](crate::cast_map::CastMapG) layer behaves the same way:
    /// its lookups are validated by slot version and stored type id, so
    /// cloning it carries no extra caveats.)
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

    // ── inner accessors ───────────────────────────────────────────────────

    /// Consumes this map and returns the backing `slotmap` map.
    ///
    /// # Safety
    /// Keys minted by this map cache pointer metadata for the values as
    /// stored; anything done to the backing map directly (or to values moved
    /// out of it) that changes which concrete type lives under a still-held
    /// key makes that key's metadata stale, and using it with the typed
    /// accessors is undefined behavior.
    #[inline]
    pub unsafe fn inner(self) -> M {
        self.inner
    }

    /// Returns a shared reference to the backing `slotmap` map.
    ///
    /// # Safety
    /// See [`inner`](Self::inner).
    #[inline]
    pub unsafe fn inner_ref(&self) -> &M {
        &self.inner
    }

    /// Returns a mutable reference to the backing `slotmap` map.
    ///
    /// # Safety
    /// See [`inner`](Self::inner).
    #[inline]
    pub unsafe fn inner_mut(&mut self) -> &mut M {
        &mut self.inner
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
    /// Every key must address a live slot, and no two keys may alias one slot.
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
    // ── insert ───────────────────────────────────────────────────────────

    /// Inserts a smart pointer, returning the output-typed [`CastKey`]
    /// (`CastKey<MTarget<M>, M::Key>`, metadata read from the stored value).
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

    // ── insert_sized ─────────────────────────────────────────────────────

    /// Inserts a concrete-typed smart pointer (coerced into `M::Value` on the
    /// way in), returning a [`CastKey`] whose metadata is for
    /// `ConcretePtr::Target` (not the erased output type).
    #[inline]
    pub fn insert_sized<ConcretePtr>(
        &mut self,
        value: ConcretePtr,
    ) -> CastKey<ConcretePtr::Target, M::Key>
    where
        ConcretePtr: std::ops::CoerceUnsized<M::Value> + Deref,
        ConcretePtr::Target: Sized,
    {
        self.insert_sized_with_key(|_| value)
    }

    /// Inserts a concrete smart pointer produced by `func`, which receives the
    /// fully-typed [`CastKey`] the value will live under.
    #[inline]
    pub fn insert_sized_with_key<ConcretePtr>(
        &mut self,
        func: impl FnOnce(CastKey<ConcretePtr::Target, M::Key>) -> ConcretePtr,
    ) -> CastKey<ConcretePtr::Target, M::Key>
    where
        ConcretePtr: std::ops::CoerceUnsized<M::Value> + Deref,
        ConcretePtr::Target: Sized,
    {
        self.try_insert_sized_with_key(|key| Ok::<_, ()>(func(key)))
            .unwrap()
    }

    /// Like [`insert_sized_with_key`](Self::insert_sized_with_key) but the
    /// closure may return `Err`, in which case nothing is inserted.
    #[inline]
    pub fn try_insert_sized_with_key<ConcretePtr, E>(
        &mut self,
        func: impl FnOnce(CastKey<ConcretePtr::Target, M::Key>) -> Result<ConcretePtr, E>,
    ) -> Result<CastKey<ConcretePtr::Target, M::Key>, E>
    where
        ConcretePtr: std::ops::CoerceUnsized<M::Value> + Deref,
        ConcretePtr::Target: Sized,
    {
        let mut saved_key: Option<CastKey<ConcretePtr::Target, M::Key>> = None;

        self.inner
            .try_insert_with_key(|inner_key| -> Result<M::Value, E> {
                // SAFETY: `()` metadata is trivially valid for the sized
                // `ConcretePtr::Target` about to occupy this slot.
                let typed_key = unsafe { CastKey::from_raw_parts(inner_key, ()) };
                saved_key = Some(typed_key);
                let concrete: ConcretePtr = func(typed_key)?;
                Ok(concrete)
            })?;

        Ok(saved_key.unwrap())
    }

    // ── insert_as ────────────────────────────────────────────────────────

    /// Inserts a smart pointer whose (possibly unsized) target differs from
    /// the map's output type, returning a key typed with the *source* type
    /// (e.g. insert a `CastBox<dyn Foo>` into a `dyn Any` map, keeping a
    /// `CastKey<dyn Foo>`).
    #[inline]
    pub fn insert_as<SourcePtr>(
        &mut self,
        value: SourcePtr,
    ) -> CastKey<SourcePtr::Target, M::Key>
    where
        SourcePtr: std::ops::CoerceUnsized<M::Value> + StableDeref,
        SourcePtr::Target: Pointee<Metadata: Copy>,
    {
        self.insert_as_with_key(|_| value)
    }

    /// Inserts a smart pointer produced by `func`, returning a key typed with
    /// the source `SourcePtr::Target`. The closure receives the backing key
    /// (not a typed [`CastKey`] — the metadata does not exist until the value
    /// does).
    ///
    /// `SourcePtr: StableDeref` (not just `Deref`) because the key's metadata
    /// is read through the source pointer's deref *before* the coercion: the
    /// deref must describe the same, stable allocation the map ends up
    /// owning.
    #[inline]
    pub fn insert_as_with_key<SourcePtr>(
        &mut self,
        func: impl FnOnce(M::Key) -> SourcePtr,
    ) -> CastKey<SourcePtr::Target, M::Key>
    where
        SourcePtr: std::ops::CoerceUnsized<M::Value> + StableDeref,
        SourcePtr::Target: Pointee<Metadata: Copy>,
    {
        self.try_insert_as_with_key(|key| Ok::<_, ()>(func(key)))
            .unwrap()
    }

    /// Like [`insert_as_with_key`](Self::insert_as_with_key) but the closure
    /// may return `Err`, in which case nothing is inserted.
    #[inline]
    pub fn try_insert_as_with_key<SourcePtr, E>(
        &mut self,
        func: impl FnOnce(M::Key) -> Result<SourcePtr, E>,
    ) -> Result<CastKey<SourcePtr::Target, M::Key>, E>
    where
        SourcePtr: std::ops::CoerceUnsized<M::Value> + StableDeref,
        SourcePtr::Target: Pointee<Metadata: Copy>,
    {
        let mut saved_metadata: Option<<SourcePtr::Target as Pointee>::Metadata> = None;

        let inner_key = self
            .inner
            .try_insert_with_key(|inner_key| -> Result<M::Value, E> {
                let concrete: SourcePtr = func(inner_key)?;
                // Source-typed metadata, read before the coercion erases it;
                // the coercion never changes the allocation's address.
                saved_metadata =
                    Some(std::ptr::metadata(&*concrete as *const SourcePtr::Target));
                Ok(concrete)
            })?;

        // SAFETY: `metadata` was read from the exact value now living under
        // `inner_key`.
        Ok(unsafe { CastKey::from_raw_parts(inner_key, saved_metadata.unwrap()) })
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

    /// Shared-reference lookup typed by the key's `T`, which may differ from
    /// the map's output type. Reconstructs the `&T` from the stored value's
    /// data pointer plus the key's metadata (a vtable, a slice length, or `()`
    /// for sized `T`).
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
        let typed_ptr: *const T = std::ptr::from_raw_parts(data_ptr, key.metadata());
        Some(&*typed_ptr)
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
        let typed_ptr: *const T = std::ptr::from_raw_parts(data_ptr, key.metadata());
        &*typed_ptr
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
    /// Mutable-reference lookup typed by the key's `T`, which may differ from
    /// the map's output type; see [`get`](Self::get).
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
        let typed_ptr: *mut T = std::ptr::from_raw_parts_mut(data_ptr, key.metadata());
        Some(&mut *typed_ptr)
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
        let typed_ptr: *mut T = std::ptr::from_raw_parts_mut(data_ptr, key.metadata());
        &mut *typed_ptr
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

    /// Mutable disjoint lookup typed by the keys' `T`, which may differ from
    /// the map's output type. All keys must share the pointee type `T`; each
    /// `&mut T` is rebuilt from that key's own metadata.
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
    /// - Every key must address a live slot, and no two keys may alias one slot.
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
    /// unsafe map and not on the checked
    /// [`CastMapG`](crate::cast_map::CastMapG).
    ///
    /// # Panics
    /// Panics if `key` is not in a detached state (and, for dense storage, if
    /// the map is full) — mirrors `slotmap`'s `reattach`.
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
    /// unsizes to it implicitly at the call site. `key` is the output-typed
    /// [`CastKey`] the map itself issues — `CastKey<MTarget<M>, M::Key>`, as
    /// returned by [`insert`](Self::insert), [`keys`](Self::keys), or
    /// [`cast_key_of`](Self::cast_key_of); a concrete-typed key reaches it
    /// via [`CastKey::upcast`](crate::cast_key::CastKey::upcast).
    ///
    /// Reattaching a value of a different concrete type than the slot last held
    /// leaves any retained [`CastKey`] with stale metadata; using such a key with
    /// the `unsafe` typed accessors is then undefined behavior — the same hazard
    /// as [`reattach_by_inner_key`](Self::reattach_by_inner_key), and why neither
    /// is offered on the checked [`CastMapG`](crate::cast_map::CastMapG).
    ///
    /// # Panics
    /// Panics if `key` is not detached (and, for dense storage, if the map is
    /// full).
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

/// Raw castable-key map backed by [`slotmap::DenseSlotMap`]: values are stored
/// contiguously for fast iteration, at the cost of one extra indirection per
/// lookup, and `remove` swaps the last element into the vacated position.
pub type UnsafeDenseCastMap<K, Ptr> = UnsafeCastMapG<DenseSlotMap<K, Ptr>>;

/// Convenience alias: [`UnsafeCastMap`] storing `Box<T>`
pub type UnsafeBoxCastMap<K, T> = UnsafeCastMap<K, Box<T>>;

/// Convenience alias: [`UnsafeDenseCastMap`] storing `Box<T>`
pub type UnsafeBoxDenseCastMap<K, T> = UnsafeDenseCastMap<K, Box<T>>;

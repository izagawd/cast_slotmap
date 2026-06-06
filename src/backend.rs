//! The storage backend the cast-map layer delegates to.
//!
//! [`SlotMapTrait`] captures the slice of `slotmap`'s API that
//! [`UnsafeCastMapG`](crate::unsafe_cast_map::UnsafeCastMapG) /
//! [`CastMapG`](crate::cast_map::CastMapG) build on, so the cast logic (the
//! pointer-metadata reconstruction, the `MapId` checks) is written **once**,
//! generic over the backend, instead of being duplicated per map kind.
//!
//! Two maps are provided: [`slotmap::SlotMap`] and [`slotmap::DenseSlotMap`].
//! Both support the whole surface — including `detach` / `reattach`, which
//! `slotmap` offers on both — so a single [`SlotMapTrait`] covers them with no
//! sub-trait. (`slotmap::HopSlotMap`, which lacks `detach` / `reattach`, would
//! therefore need those carved into an optional sub-trait before it could be
//! supported.)
//!
//! The method names here intentionally differ from `slotmap`'s where they would
//! otherwise collide with an inherent method during delegation (`empty` /
//! `with_capacity` for the constructors, `into_pairs` for the owning iterator).

use std::collections::TryReserveError;
use std::ops::Deref;

use slotmap::{DenseSlotMap, Key, SlotMap};

/// The dereferenced output type of map `M`'s stored pointer
/// (`<M::Value as Deref>::Target`). A convenience alias to keep the cast-map
/// signatures readable now that the key/value types are associated.
pub(crate) type MTarget<M> = <<M as SlotMapTrait>::Value as Deref>::Target;

/// The subset of a `slotmap` map the cast layer relies on.
///
/// Implemented for [`slotmap::SlotMap`] and [`slotmap::DenseSlotMap`]. The
/// associated iterator types let the cast-map iterators wrap whichever concrete
/// iterator the backend produces.
pub trait SlotMapTrait: Sized {
    /// The backing `slotmap` key type.
    type Key: Key;
    /// The stored value type (for the cast layer, a smart pointer such as
    /// `Box<dyn Any>`).
    type Value;

    /// Shared iterator yielding `(key, &value)`.
    type Iter<'a>: Iterator<Item = (Self::Key, &'a Self::Value)>
    where
        Self: 'a,
        Self::Value: 'a;
    /// Mutable iterator yielding `(key, &mut value)`.
    type IterMut<'a>: Iterator<Item = (Self::Key, &'a mut Self::Value)>
    where
        Self: 'a,
        Self::Value: 'a;
    /// Shared iterator over values.
    type Values<'a>: Iterator<Item = &'a Self::Value>
    where
        Self: 'a,
        Self::Value: 'a;
    /// Mutable iterator over values.
    type ValuesMut<'a>: Iterator<Item = &'a mut Self::Value>
    where
        Self: 'a,
        Self::Value: 'a;
    /// Draining iterator yielding `(key, value)`.
    type Drain<'a>: Iterator<Item = (Self::Key, Self::Value)>
    where
        Self: 'a;
    /// Owning iterator yielding `(key, value)`.
    type IntoIter: Iterator<Item = (Self::Key, Self::Value)>;

    /// Creates an empty backend (`slotmap`'s `with_key`).
    fn empty() -> Self;
    /// Creates an empty backend with capacity (`slotmap`'s `with_capacity_and_key`).
    fn with_capacity(capacity: usize) -> Self;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn capacity(&self) -> usize;
    fn reserve(&mut self, additional: usize);
    fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError>;
    fn clear(&mut self);
    fn contains_key(&self, key: Self::Key) -> bool;
    fn get(&self, key: Self::Key) -> Option<&Self::Value>;
    /// # Safety
    /// `key` must address an occupied slot with the matching version.
    unsafe fn get_unchecked(&self, key: Self::Key) -> &Self::Value;
    fn get_mut(&mut self, key: Self::Key) -> Option<&mut Self::Value>;
    /// # Safety
    /// `key` must address an occupied slot with the matching version.
    unsafe fn get_unchecked_mut(&mut self, key: Self::Key) -> &mut Self::Value;
    fn get_disjoint_mut<const N: usize>(
        &mut self,
        keys: [Self::Key; N],
    ) -> Option<[&mut Self::Value; N]>;
    /// # Safety
    /// Every key must be valid and no two keys may alias the same slot.
    unsafe fn get_disjoint_unchecked_mut<const N: usize>(
        &mut self,
        keys: [Self::Key; N],
    ) -> [&mut Self::Value; N];
    fn remove(&mut self, key: Self::Key) -> Option<Self::Value>;
    fn retain<F: FnMut(Self::Key, &mut Self::Value) -> bool>(&mut self, f: F);
    /// Inserts the value produced by `f`, passing it the key it will live under;
    /// nothing is inserted if `f` returns `Err`.
    fn try_insert_with_key<F, E>(&mut self, f: F) -> Result<Self::Key, E>
    where
        F: FnOnce(Self::Key) -> Result<Self::Value, E>;
    fn values(&self) -> Self::Values<'_>;
    fn values_mut(&mut self) -> Self::ValuesMut<'_>;
    fn iter(&self) -> Self::Iter<'_>;
    fn iter_mut(&mut self) -> Self::IterMut<'_>;
    fn drain(&mut self) -> Self::Drain<'_>;
    /// Consumes the backend into its owning `(key, value)` iterator.
    fn into_pairs(self) -> Self::IntoIter;

    /// Temporarily removes a value, leaving the slot reservable for
    /// [`reattach`](Self::reattach). Both [`slotmap::SlotMap`] and
    /// [`slotmap::DenseSlotMap`] support this.
    fn detach(&mut self, key: Self::Key) -> Option<Self::Value>;
    /// Reattaches a value at a detached slot, reusing `detached_key`.
    ///
    /// # Panics
    /// Panics if `detached_key` is not currently in a detached state (and, for
    /// dense storage, if the map is full).
    fn reattach(&mut self, detached_key: Self::Key, value: Self::Value);
}

// ─── SlotMap ─────────────────────────────────────────────────────────────────

impl<K: Key, V> SlotMapTrait for SlotMap<K, V> {
    type Key = K;
    type Value = V;
    type Iter<'a>
        = slotmap::basic::Iter<'a, K, V>
    where
        Self: 'a,
        V: 'a;
    type IterMut<'a>
        = slotmap::basic::IterMut<'a, K, V>
    where
        Self: 'a,
        V: 'a;
    type Values<'a>
        = slotmap::basic::Values<'a, K, V>
    where
        Self: 'a,
        V: 'a;
    type ValuesMut<'a>
        = slotmap::basic::ValuesMut<'a, K, V>
    where
        Self: 'a,
        V: 'a;
    type Drain<'a>
        = slotmap::basic::Drain<'a, K, V>
    where
        Self: 'a;
    type IntoIter = slotmap::basic::IntoIter<K, V>;

    #[inline]
    fn empty() -> Self {
        SlotMap::with_key()
    }
    #[inline]
    fn with_capacity(capacity: usize) -> Self {
        SlotMap::with_capacity_and_key(capacity)
    }
    #[inline]
    fn len(&self) -> usize {
        self.len()
    }
    #[inline]
    fn is_empty(&self) -> bool {
        self.is_empty()
    }
    #[inline]
    fn capacity(&self) -> usize {
        self.capacity()
    }
    #[inline]
    fn reserve(&mut self, additional: usize) {
        self.reserve(additional);
    }
    #[inline]
    fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.try_reserve(additional)
    }
    #[inline]
    fn clear(&mut self) {
        self.clear();
    }
    #[inline]
    fn contains_key(&self, key: K) -> bool {
        self.contains_key(key)
    }
    #[inline]
    fn get(&self, key: K) -> Option<&V> {
        self.get(key)
    }
    #[inline]
    unsafe fn get_unchecked(&self, key: K) -> &V {
        self.get_unchecked(key)
    }
    #[inline]
    fn get_mut(&mut self, key: K) -> Option<&mut V> {
        self.get_mut(key)
    }
    #[inline]
    unsafe fn get_unchecked_mut(&mut self, key: K) -> &mut V {
        self.get_unchecked_mut(key)
    }
    #[inline]
    fn get_disjoint_mut<const N: usize>(&mut self, keys: [K; N]) -> Option<[&mut V; N]> {
        self.get_disjoint_mut(keys)
    }
    #[inline]
    unsafe fn get_disjoint_unchecked_mut<const N: usize>(&mut self, keys: [K; N]) -> [&mut V; N] {
        self.get_disjoint_unchecked_mut(keys)
    }
    #[inline]
    fn remove(&mut self, key: K) -> Option<V> {
        self.remove(key)
    }
    #[inline]
    fn retain<F: FnMut(K, &mut V) -> bool>(&mut self, f: F) {
        self.retain(f);
    }
    #[inline]
    fn try_insert_with_key<F, E>(&mut self, f: F) -> Result<K, E>
    where
        F: FnOnce(K) -> Result<V, E>,
    {
        self.try_insert_with_key(f)
    }
    #[inline]
    fn values(&self) -> Self::Values<'_> {
        self.values()
    }
    #[inline]
    fn values_mut(&mut self) -> Self::ValuesMut<'_> {
        self.values_mut()
    }
    #[inline]
    fn iter(&self) -> Self::Iter<'_> {
        self.iter()
    }
    #[inline]
    fn iter_mut(&mut self) -> Self::IterMut<'_> {
        self.iter_mut()
    }
    #[inline]
    fn drain(&mut self) -> Self::Drain<'_> {
        self.drain()
    }
    #[inline]
    fn into_pairs(self) -> Self::IntoIter {
        self.into_iter()
    }
    #[inline]
    fn detach(&mut self, key: K) -> Option<V> {
        self.detach(key)
    }
    #[inline]
    fn reattach(&mut self, detached_key: K, value: V) {
        self.reattach(detached_key, value);
    }
}

// ─── DenseSlotMap ──────────────────────────────────────────────────────────────

impl<K: Key, V> SlotMapTrait for DenseSlotMap<K, V> {
    type Key = K;
    type Value = V;
    type Iter<'a>
        = slotmap::dense::Iter<'a, K, V>
    where
        Self: 'a,
        V: 'a;
    type IterMut<'a>
        = slotmap::dense::IterMut<'a, K, V>
    where
        Self: 'a,
        V: 'a;
    type Values<'a>
        = slotmap::dense::Values<'a, K, V>
    where
        Self: 'a,
        V: 'a;
    type ValuesMut<'a>
        = slotmap::dense::ValuesMut<'a, K, V>
    where
        Self: 'a,
        V: 'a;
    type Drain<'a>
        = slotmap::dense::Drain<'a, K, V>
    where
        Self: 'a;
    type IntoIter = slotmap::dense::IntoIter<K, V>;

    #[inline]
    fn empty() -> Self {
        DenseSlotMap::with_key()
    }
    #[inline]
    fn with_capacity(capacity: usize) -> Self {
        DenseSlotMap::with_capacity_and_key(capacity)
    }
    #[inline]
    fn len(&self) -> usize {
        self.len()
    }
    #[inline]
    fn is_empty(&self) -> bool {
        self.is_empty()
    }
    #[inline]
    fn capacity(&self) -> usize {
        self.capacity()
    }
    #[inline]
    fn reserve(&mut self, additional: usize) {
        self.reserve(additional);
    }
    #[inline]
    fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.try_reserve(additional)
    }
    #[inline]
    fn clear(&mut self) {
        self.clear();
    }
    #[inline]
    fn contains_key(&self, key: K) -> bool {
        self.contains_key(key)
    }
    #[inline]
    fn get(&self, key: K) -> Option<&V> {
        self.get(key)
    }
    #[inline]
    unsafe fn get_unchecked(&self, key: K) -> &V {
        self.get_unchecked(key)
    }
    #[inline]
    fn get_mut(&mut self, key: K) -> Option<&mut V> {
        self.get_mut(key)
    }
    #[inline]
    unsafe fn get_unchecked_mut(&mut self, key: K) -> &mut V {
        self.get_unchecked_mut(key)
    }
    #[inline]
    fn get_disjoint_mut<const N: usize>(&mut self, keys: [K; N]) -> Option<[&mut V; N]> {
        self.get_disjoint_mut(keys)
    }
    #[inline]
    unsafe fn get_disjoint_unchecked_mut<const N: usize>(&mut self, keys: [K; N]) -> [&mut V; N] {
        self.get_disjoint_unchecked_mut(keys)
    }
    #[inline]
    fn remove(&mut self, key: K) -> Option<V> {
        self.remove(key)
    }
    #[inline]
    fn retain<F: FnMut(K, &mut V) -> bool>(&mut self, f: F) {
        self.retain(f);
    }
    #[inline]
    fn try_insert_with_key<F, E>(&mut self, f: F) -> Result<K, E>
    where
        F: FnOnce(K) -> Result<V, E>,
    {
        self.try_insert_with_key(f)
    }
    #[inline]
    fn values(&self) -> Self::Values<'_> {
        self.values()
    }
    #[inline]
    fn values_mut(&mut self) -> Self::ValuesMut<'_> {
        self.values_mut()
    }
    #[inline]
    fn iter(&self) -> Self::Iter<'_> {
        self.iter()
    }
    #[inline]
    fn iter_mut(&mut self) -> Self::IterMut<'_> {
        self.iter_mut()
    }
    #[inline]
    fn drain(&mut self) -> Self::Drain<'_> {
        self.drain()
    }
    #[inline]
    fn into_pairs(self) -> Self::IntoIter {
        self.into_iter()
    }
    #[inline]
    fn detach(&mut self, key: K) -> Option<V> {
        self.detach(key)
    }
    #[inline]
    fn reattach(&mut self, detached_key: K, value: V) {
        self.reattach(detached_key, value);
    }
}

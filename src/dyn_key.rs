//! [`DynKey`]: a [`CastKey`](crate::cast_key::CastKey) borrowed into a shape
//! that can be a **method receiver on trait objects**.
//!
//! A dyn-dispatch receiver must be exactly the size and shape of a pointer,
//! and `CastKey` cannot guarantee that: *pointer* size varies by target
//! (32- vs 64-bit) while the key is a fixed 8 bytes — and `slotmap` plans to
//! let users pick the size of their keys — so the key cannot be relied on to
//! fit in, or match, a pointer.
//! `DynKey` re-expresses the same information as a single `*const T`:
//! the *metadata* half carries the key's pointer metadata (the vtable for
//! `dyn` targets), and the *address* half smuggles the backing `slotmap` key.
//! That makes `DynKey` layout-compatible with dyn dispatch, so traits can
//! declare methods as `fn m(self: DynKey<Self>, ...)` and be called through
//! `DynKey<dyn Trait>`.
//!
//! Which representation the address holds is decided by compile-time checks
//! of the actual types involved, in three tiers (`u64` being the packed
//! form's type, confirmed at compile time against `as_ffi` / `from_ffi`):
//! - **`size_of::<u64>() == size_of::<usize>()`:** the address is the
//!   key's packed [`KeyData`] via [`KeyData::as_ffi`], relying only on its
//!   documented guarantee — round-tripping through [`KeyData::from_ffi`] —
//!   never the key's byte layout, which could contain padding.
//! - **`size_of::<u64>() < size_of::<usize>()`** (e.g. a 128-bit
//!   target): same packing; the `as_ffi` value zero-extends into the wider
//!   address.
//! - **Otherwise:** the address is a real pointer to the borrowed key's
//!   backing `K` field, which the `'a` borrow keeps alive. Only `K` is read
//!   back (its type is the same for every `T`); the metadata always travels
//!   in the fat pointer itself, where unsizing coercions keep it correct.
//!
//! The smuggled address is **never dereferenced** on the first two paths.
//!
//! Why a raw pointer and not [`NonNull`](std::ptr::NonNull): `NonNull` would
//! demand a nonzero address, but on the packed paths, the address is whatever
//! [`KeyData::as_ffi`] returns, and its *only* documented guarantee is the
//! [`KeyData::from_ffi`] round-trip — nothing promises the value is nonzero.
//! (Today it happens never to be `0`, since every `KeyData` carries a
//! `NonZeroU32` version, but that is an implementation detail, not a
//! contract.) A raw pointer carries any address, `0` included, so construction
//! is infallible — no runtime check, no panic path, and no dependence on
//! `as_ffi`'s bit layout at all. The trade-off is that `Option<DynKey>` gets
//! no pointer niche, so it is larger than a bare `DynKey`.

use std::marker::{PhantomData, Unsize};
use std::ops::{CoerceUnsized, DispatchFromDyn, Receiver};
use std::ptr::{self, Pointee};

use slotmap::{DefaultKey, Key, KeyData};

use crate::cast_key::CastKey;

// Compile-time confirmation that the packed form really is `u64`: these fail
// to compile if `as_ffi` / `from_ffi` ever change signature. The tiers below
// compare `u64` — the type that actually crosses — against the pointer size.
const _: fn(KeyData) -> u64 = KeyData::as_ffi;
const _: fn(u64) -> KeyData = KeyData::from_ffi;

/// Is a packed key (a `u64`, confirmed above) exactly the size of a pointer
/// address? Decided from the types themselves, per target.
#[inline]
const fn equals_ptr() -> bool {
    size_of::<u64>() == size_of::<usize>()
}

/// Is a packed key (a `u64`, confirmed above) smaller than a pointer address
/// (e.g. a 128-bit target)? Decided from the types themselves, per target.
#[inline]
const fn smaller_than_ptr() -> bool {
    size_of::<u64>() < size_of::<usize>()
}

/// A borrowed, dyn-dispatchable form of a [`CastKey`].
///
/// Obtain one with [`CastKey::as_dyn`] (or `From<&CastKey>`); recover the key
/// with [`DynKey::key`] (or `Into<CastKey>`). Use it as a trait-method
/// receiver:
///
/// ```ignore
/// trait Component {
///     fn tick(self: DynKey<'_, Self>, world: &mut World);
/// }
/// let dk: DynKey<'_, dyn Component> = key.as_dyn();
/// dk.tick(&mut world); // virtual call through the key's metadata
/// ```
pub struct DynKey<'a, T: ?Sized, K: Key = DefaultKey> {
    /// Address = packed `KeyData` (or a pointer to the borrowed key's `K`
    /// field when packing does not fit); metadata = the `CastKey`'s pointer
    /// metadata. Raw — and thus allowed to be null — on purpose: the packed
    /// value has no nonzero guarantee (see the module docs). Never
    /// dereferenced as a `T`.
    ptr: *const T,
    _borrow: PhantomData<&'a K>,
}

impl<'a, T: ?Sized, K: Key> Clone for DynKey<'a, T, K> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, T: ?Sized, K: Key> Copy for DynKey<'a, T, K> {}

// SAFETY: a `DynKey` is semantically a `&'a CastKey<T, K>` (on the borrow
// path) or a by-value copy of the key's bits (on the packed path); it is
// `Copy`, never hands out `&mut`, and never dereferences `ptr` as a `T`.
// Sending or sharing it across threads therefore only permits *reading* the
// borrowed `CastKey`, so both impls require exactly `CastKey<T, K>: Sync`.
unsafe impl<'a, T: ?Sized + Pointee, K: Key> Send for DynKey<'a, T, K>
where
    <T as Pointee>::Metadata: Copy,
    CastKey<T, K>: Sync,
{
}
unsafe impl<'a, T: ?Sized + Pointee, K: Key> Sync for DynKey<'a, T, K>
where
    <T as Pointee>::Metadata: Copy,
    CastKey<T, K>: Sync,
{
}

// Dyn-dispatch machinery: `DynKey` is a single (fat) pointer plus 1-ZSTs, the
// exact shape `DispatchFromDyn` requires of a receiver.
impl<'a, T: ?Sized + Unsize<U>, U: ?Sized, K: Key> CoerceUnsized<DynKey<'a, U, K>>
    for DynKey<'a, T, K>
{
}
impl<'a, T: ?Sized + Unsize<U>, U: ?Sized, K: Key> DispatchFromDyn<DynKey<'a, U, K>>
    for DynKey<'a, T, K>
{
}

// A receiver without `Deref`: the key alone cannot reach the value (that needs
// the map), so only dispatch — not `*dyn_key` — is offered.
impl<'a, T: ?Sized, K: Key> Receiver for DynKey<'a, T, K> {
    type Target = T;
}

impl<'a, T: ?Sized + Pointee, K: Key> DynKey<'a, T, K>
where
    <T as Pointee>::Metadata: Copy,
{
    /// Borrows a [`CastKey`] into its dyn-dispatchable form.
    #[inline]
    pub fn new(key: &'a CastKey<T, K>) -> Self {
        let thin: *const () = if const { equals_ptr() || smaller_than_ptr() } {
            // Packed `as_ffi` value as the address (zero-extends if usize is
            // wider); pure data, never dereferenced, 0 is fine — see module docs.
            ptr::without_provenance(key.inner_key().data().as_ffi() as usize)
        } else {
            // The key does not fit in a pointer on this target: point at the
            // borrowed key's backing `K` field. `K` is the same type for
            // every `T`, so `key()` can read it back even after an unsizing
            // coercion changes `T` (reading the whole `CastKey<T, K>` could
            // not: its layout differs per `T`). Valid for 'a; provenance is
            // preserved through from/to_raw_parts.
            ptr::from_ref(&key.key).cast()
        };
        Self {
            ptr: ptr::from_raw_parts(thin, key.metadata()),
            _borrow: PhantomData,
        }
    }

    /// Recovers the [`CastKey`] this `DynKey` was made from.
    #[inline]
    pub fn key(self) -> CastKey<T, K> {
        let (thin, metadata) = self.ptr.to_raw_parts();
        if const { equals_ptr() || smaller_than_ptr() } {
            // `from_ffi(as_ffi(k)) == k` is the documented round-trip
            // guarantee; nothing about the value's layout is relied on.
            let key = K::from(KeyData::from_ffi(thin.addr() as u64));
            // SAFETY: `key`/`metadata` round-trip the exact values of the
            // `CastKey` given to `new`, whose construction already vouched
            // for the metadata.
            unsafe { CastKey::from_raw_parts(key, metadata) }
        } else {
            // SAFETY: on this path `thin` points at the `K` field of the
            // `CastKey` borrowed by `new`, still alive for 'a; `K` is `Copy`
            // and its type does not depend on `T`. The metadata comes from
            // the fat pointer, which unsizing coercions keep correct for the
            // current `T`.
            let k = unsafe { thin.cast::<K>().read() };
            // SAFETY: `k` and `metadata` describe the same borrowed key.
            unsafe { CastKey::from_raw_parts(k, metadata) }
        }
    }
}

impl<'a, T: ?Sized + Pointee, K: Key> From<&'a CastKey<T, K>> for DynKey<'a, T, K>
where
    <T as Pointee>::Metadata: Copy,
{
    #[inline]
    fn from(key: &'a CastKey<T, K>) -> Self {
        Self::new(key)
    }
}

impl<'a, T: ?Sized + Pointee, K: Key> From<DynKey<'a, T, K>> for CastKey<T, K>
where
    <T as Pointee>::Metadata: Copy,
{
    #[inline]
    fn from(key: DynKey<'a, T, K>) -> Self {
        key.key()
    }
}

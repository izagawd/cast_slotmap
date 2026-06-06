//! Castable-key wrappers over the [`slotmap`] crate's
//! [`SlotMap`](slotmap::SlotMap) and [`DenseSlotMap`](slotmap::DenseSlotMap).
//!
//! Store type-erased heterogeneous values (e.g. `Box<dyn Any>`) and hand back
//! typed [`CastKey`]s, so `map.get(key)` returns a correctly typed `&T` with no
//! `downcast_ref` at the call site.
//!
//! Two axes, four maps. The **identity** axis is raw vs. checked; the
//! **storage** axis is basic vs. dense:
//!
//! - [`UnsafeCastMap`] — the low-level map over [`slotmap::SlotMap`]. Lookups
//!   are typed via [`CastKey`], but `get` / `get_mut` / `remove` /
//!   `downcast_key` are `unsafe`: they rebuild the typed reference from the
//!   key's cached metadata without checking it still matches the value in the
//!   slot, so using a key whose slot holds a different type is undefined
//!   behavior.
//! - [`CastMap`] — the safe, recommended API over [`slotmap::SlotMap`]. Each map
//!   gets a unique [`MapId`] on creation and every [`StableCastKey`] carries it,
//!   so a key from map A used on map B returns `None` instead of being unsound.
//! - [`UnsafeDenseCastMap`] / [`DenseCastMap`] — the same raw/checked pair over
//!   [`slotmap::DenseSlotMap`], which stores values contiguously for fast
//!   iteration (see [`UnsafeDenseCastMap`] for the storage trade-offs). The
//!   cast-key API is identical to the basic maps'.
//!
//! All four maps support disjoint mutable access via `get_disjoint_mut` (typed,
//! by [`CastKey`]) and `get_disjoint_mut_by_inner_key` (by backing key), each
//! with an `unchecked` companion.
//!
//! Under the hood there are really just **two** generic types,
//! [`UnsafeCastMapG`] and [`CastMapG`], each parameterized over a backing
//! `slotmap` map `M` implementing [`SlotMapTrait`]. The four maps above are type
//! aliases that pin `M` to `SlotMap` or `DenseSlotMap`.
//!
//! For the common `Box` case use the aliases [`BoxCastMap`] / [`UnsafeBoxCastMap`]
//! (and [`BoxDenseCastMap`] / [`UnsafeBoxDenseCastMap`]), typically with
//! `dyn Any`: `BoxCastMap<DefaultKey, dyn Any>`.
//!
//! # A `SlotMap`, not a stable-reference arena
//! `slotmap::SlotMap::insert` takes `&mut self` (it is not a stable-reference,
//! interior-mutability arena; `DenseSlotMap` is the same), so every mutating
//! method here — `insert*`, `remove`, `reserve`, `clear`, `retain`, `drain` —
//! takes `&mut self`. Because `get` borrows `&self` while `insert` borrows
//! `&mut self`, references and inserts can never coexist; consequently the
//! shared [`iter`](CastMap::iter) is plain safe (no `unsafe_iter`), `Clone` is a
//! normal forward (no `unsafe_clone`/`clone_mut`), and there is no `get_slot`,
//! `get_by_index_only`, or `reset` — `slotmap` exposes no such operations.
//! `clear` is the native way to invalidate all keys.
//!
//! # Nightly
//! Pointer-metadata reconstruction relies on the unstable `ptr_metadata`,
//! `coerce_unsized`, and `unsize` features, so this crate requires a **nightly**
//! toolchain. It is single-threaded in spirit, mirroring `slotmap::SlotMap`'s
//! own `Send`/`Sync` behavior (which depends on the stored value).
//!
//! # Example
//! ```ignore
//! #![feature(ptr_metadata, coerce_unsized, unsize)]
//! use cast_slotmap::{BoxCastMap, DefaultKey, StableCastKey};
//! use std::any::Any;
//!
//! struct Dog { name: String }
//!
//! let mut map: BoxCastMap<DefaultKey, dyn Any> = BoxCastMap::new();
//!
//! // Insert a concrete type into a `dyn Any` map; the key comes back erased.
//! let dyn_key: StableCastKey<dyn Any> = map.insert(Box::new(Dog { name: "Rex".into() }));
//!
//! // Downcast the erased key to a concrete `Dog`-typed key.
//! let dog_key: StableCastKey<Dog> = map.downcast_key::<Dog>(dyn_key).unwrap();
//!
//! assert_eq!(map.get(dog_key).unwrap().name, "Rex");
//! ```
#![feature(ptr_metadata)]
#![feature(coerce_unsized)]
#![feature(unsize)]

pub mod slotmap_trait;
pub mod cast_key;
pub mod cast_map;
pub mod map_id;
pub mod retype_ptr;
pub mod unsafe_cast_map;

// Re-export the slotmap items callers need so they don't have to depend on
// `slotmap` directly for the common path.
pub use slotmap::{new_key_type, DefaultKey, Key, KeyData};

#[doc(inline)]
pub use cast_key::{CastKey, StableCastKey};
#[doc(inline)]
pub use slotmap_trait::SlotMapTrait;
#[doc(inline)]
pub use cast_map::{BoxCastMap, BoxDenseCastMap, CastMap, CastMapG, DenseCastMap};
#[doc(inline)]
pub use map_id::MapId;
#[doc(inline)]
pub use retype_ptr::RetypePtr;
#[doc(no_inline)]
pub use stable_deref_trait::StableDeref;
#[doc(inline)]
pub use unsafe_cast_map::{
    UnsafeBoxCastMap, UnsafeBoxDenseCastMap, UnsafeCastMap, UnsafeCastMapG, UnsafeDenseCastMap,
};

#[cfg(test)]
mod tests;

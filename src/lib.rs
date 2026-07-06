//! Castable-key wrappers over the [`slotmap`] crate's
//! [`SlotMap`](slotmap::SlotMap) and [`DenseSlotMap`](slotmap::DenseSlotMap).
//!
//! Store type-erased heterogeneous values (e.g. `TypeTaggedBox<dyn Any>`) and hand
//! back typed [`CastKey`]s, so `map.get(key)` returns a correctly typed `&T`
//! with no `downcast_ref` at the call site.
//!
//! Two axes, four maps. The **checking** axis is raw vs. type-id-checked; the
//! **storage** axis is basic vs. dense:
//!
//! - [`UnsafeCastMap`] — the low-level map over [`slotmap::SlotMap`]. Lookups
//!   are typed via [`CastKey`], but `get` / `get_mut` / `remove`
//!   are `unsafe`: they rebuild the typed reference from the
//!   key's cached metadata without checking it still matches the value in the
//!   slot, so using a key whose slot holds a different type is undefined
//!   behavior.
//! - [`CastMap`] — the safe, recommended API over [`slotmap::SlotMap`]. Values
//!   are stored in a box that records its concrete [`TypeId`](std::any::TypeId)
//!   (such as [`TypeTaggedBox`], an alias of [`TypeTaggedPtr`]`<Box<T>>`); every keyed lookup recovers the type id implied by
//!   the key's metadata ([`type_id_from_meta`]) and compares it to the slot's.
//!   A stale, mistyped, or foreign key returns `None` instead of being unsound.
//! - [`UnsafeDenseCastMap`] / [`DenseCastMap`] — the same raw/checked pair over
//!   [`slotmap::DenseSlotMap`], which stores values contiguously for fast
//!   iteration. The cast-key API is identical to the basic maps'.
//!
//! All four maps support disjoint mutable access via `get_disjoint_mut` (typed,
//! by [`CastKey`]) and `get_disjoint_mut_by_inner_key` (by backing key), each
//! with an `unchecked` companion.
//!
//! Under the hood there are really just **two** generic types,
//! [`UnsafeCastMapG`] and [`CastMapG`], each parameterized over a backing
//! `slotmap` map `M` implementing [`SlotMapTrait`]. The four maps above are
//! type aliases that pin `M` to `SlotMap` or `DenseSlotMap`.
//!
//! For the common case use the aliases [`BoxCastMap`] / [`BoxDenseCastMap`]
//! (which store [`TypeTaggedBox`]) — typically with `dyn Any`:
//! `BoxCastMap<DefaultKey, dyn Any>`. The raw maps have [`UnsafeBoxCastMap`] /
//! [`UnsafeBoxDenseCastMap`], storing plain `Box`.
//!
//! # `AnyHaver` and key types
//! Checked lookups require `T: AnyHaver`, an **`unsafe` trait** that recovers a
//! concrete [`TypeId`](std::any::TypeId) from pointer metadata alone. All
//! `'static` sized types get it via a blanket impl; trait-object keys get it by
//! declaring it as a supertrait (`trait Foo: AnyHaver`), which puts the lookup
//! in `dyn Foo`'s vtable. `dyn Any` has no such supertrait, so
//! `map.get(dyn_any_key)` is a **compile error** — recover a typed key with
//! [`downcast_key`](CastMapG::downcast_key) or read type-erased through
//! [`get_by_inner_key`](CastMapG::get_by_inner_key) instead.
//!
//! # Dyn-dispatchable keys
//! [`DynKey`] (via [`CastKey::as_dyn`]) reshapes a borrowed key into a valid
//! trait-object method receiver, so traits can declare
//! `fn m(self: DynKey<Self>, ..)` and be dispatched through `DynKey<dyn Trait>`
//! using the vtable already cached in the key — no map access needed for the
//! dispatch itself.
//!
//! # Nightly
//! Pointer-metadata reconstruction and the dyn-dispatchable key rely on the
//! unstable `ptr_metadata`, `coerce_unsized`, `unsize`, `dispatch_from_dyn`,
//! `arbitrary_self_types`, and `arbitrary_self_types_pointers` features, so
//! this crate requires a **nightly** toolchain.
//!
//! # Example
//! ```ignore
//! use cast_slotmap::{BoxCastMap, TypeTaggedBox, CastKey, DefaultKey};
//! use std::any::Any;
//!
//! struct Dog { name: String }
//!
//! let mut map: BoxCastMap<DefaultKey, dyn Any> = BoxCastMap::new();
//!
//! // Insert a concrete type into a `dyn Any` map; the key comes back typed.
//! let dog_key: CastKey<Dog> = map.insert_sized(TypeTaggedBox::new(Dog { name: "Rex".into() }));
//!
//! assert_eq!(map.get(dog_key).unwrap().name, "Rex");
//!
//! // Or insert erased and recover the typed key later.
//! let dyn_key: CastKey<dyn Any> = map.insert(TypeTaggedBox::new(Dog { name: "Ax".into() }));
//! let typed: CastKey<Dog> = map.downcast_key::<Dog>(dyn_key.inner_key()).unwrap();
//! ```
#![feature(ptr_metadata)]
#![feature(coerce_unsized)]
#![feature(unsize)]
#![feature(dispatch_from_dyn)]
#![feature(arbitrary_self_types)]
#![feature(arbitrary_self_types_pointers)]

pub mod any_haver;
pub mod cast_key;
pub mod cast_map;
pub mod dyn_key;
pub mod retype_ptr;
pub mod slotmap_trait;
pub mod type_tagged_ptr;
pub mod unsafe_cast_map;

// Re-export the slotmap items callers need so they don't have to depend on
// `slotmap` directly for the common path.
pub use slotmap::{new_key_type, DefaultKey, Key, KeyData};

#[doc(inline)]
pub use any_haver::{type_id_from_meta, AnyHaver};
#[doc(inline)]
pub use cast_key::CastKey;
#[doc(inline)]
pub use cast_map::{BoxCastMap, BoxDenseCastMap, CastMap, CastMapG, DenseCastMap};
#[doc(inline)]
pub use dyn_key::DynKey;
#[doc(inline)]
pub use retype_ptr::RetypePtr;
#[doc(inline)]
pub use slotmap_trait::SlotMapTrait;
#[doc(inline)]
pub use type_tagged_ptr::{ConcreteTypeId, TypeTaggedBox, TypeTaggedPtr};
#[doc(no_inline)]
pub use stable_deref_trait::StableDeref;
#[doc(inline)]
pub use unsafe_cast_map::{
    UnsafeBoxCastMap, UnsafeBoxDenseCastMap, UnsafeCastMap, UnsafeCastMapG, UnsafeDenseCastMap,
};

#[cfg(test)]
mod tests;

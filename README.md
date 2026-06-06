# cast_slotmap

Castable-key wrappers over the [`slotmap`](https://crates.io/crates/slotmap)
crate's `SlotMap` and `DenseSlotMap`. This is to `slotmap` what
`stable_gen_map`'s `StableCastMap` / `UnsafeCastMap` are to its `GenMap`: store
type-erased heterogeneous values (e.g. `Box<dyn Any>`) and get back **typed**
keys, so `map.get(key)` returns a correctly typed `&T` with no `downcast_ref` at
the call site.

> **Nightly only.** Pointer-metadata reconstruction uses the unstable
> `ptr_metadata`, `coerce_unsized`, and `unsize` features.

## The maps

Two axes — **identity** (raw vs. checked) and **storage** (basic vs. dense):

- **`UnsafeCastMap<K, Ptr>`** — raw, over `slotmap::SlotMap`. Typed lookups via
  `CastKey`, but `get` / `get_mut` / `remove` / `downcast_key` are `unsafe`
  (no per-map identity check).
- **`CastMap<K, Ptr>`** — the safe, recommended API over `slotmap::SlotMap`.
  Each map gets a unique `MapId`; every `StableCastKey` carries it, so a key
  from map A used on map B returns `None` instead of being unsound.
- **`UnsafeDenseCastMap`** / **`DenseCastMap`** — the same raw/checked pair over
  `slotmap::DenseSlotMap`, which stores values contiguously for fast iteration
  (one extra indirection per lookup; `remove` reorders the survivors). The
  cast-key API is identical.

These four are thin **type aliases**. Internally there are just two generic
types — `UnsafeCastMapG<M>` and `CastMapG<M>` — parameterized
over a backing map `M: SlotMapTrait` (implemented for both `SlotMap` and
`DenseSlotMap`).

For the common case use the `Box` aliases — `BoxCastMap<K, T>` /
`UnsafeBoxCastMap<K, T>` and `BoxDenseCastMap<K, T>` /
`UnsafeBoxDenseCastMap<K, T>` — typically with `dyn Any`.

```rust
#![feature(ptr_metadata, coerce_unsized, unsize)]
use cast_slotmap::{BoxCastMap, DefaultKey, StableCastKey};
use std::any::Any;

struct Dog { name: String }

let mut map: BoxCastMap<DefaultKey, dyn Any> = BoxCastMap::new();

// Insert a concrete type into a `dyn Any` map; the key comes back erased.
let dyn_key: StableCastKey<dyn Any> = map.insert(Box::new(Dog { name: "Rex".into() }));

// Downcast the erased key to a concrete `Dog`-typed key.
let dog_key: StableCastKey<Dog> = map.downcast_key::<Dog>(dyn_key).unwrap();

assert_eq!(map.get(dog_key).unwrap().name, "Rex");
```

## Design: this is a `SlotMap`, not a stable-reference arena

`slotmap::SlotMap::insert` takes `&mut self` (and so does `DenseSlotMap`). So
this crate **mirrors slotmap's mutability model**, rather than copying
`stable_gen_map`'s `&self`-insert API:

- Every mutating method — `insert*`, `remove`, `reserve`, `clear`, `retain`,
  `drain` — takes `&mut self`.
- Each public method **delegates to the underlying `slotmap::SlotMap`** method.
- Because `get` borrows `&self` while `insert` borrows `&mut self`, references
  and inserts can never coexist. Consequently:
  - `iter` is a plain **safe** shared iterator (there is no `unsafe_iter`),
  - `Clone` is a normal forward (no `unsafe_clone` / `clone_mut`),
  - there is **no** `get_slot`, `get_by_index_only`, or `reset` — `slotmap`
    exposes no such operations, and faking them with `unsafe` would mean
    reaching past its public API. `clear` is the native way to invalidate every
    outstanding key.

## What carries over from the cast-map design

The *key*-level machinery is unchanged, because it is about keys, not about how
the underlying map mutates: `insert` / `insert_with_key` / `try_insert_with_key`,
`downcast_key`, `CastKey::upcast`, typed `get<T>` /
`get_unchecked<T>` / `remove<T>` (via `RetypePtr`), `cast_key_of`,
`get_by_inner_key`(`_mut`), `remove_by_inner_key`, `keys`,
`values`(`_mut`), `iter`(`_mut`), `drain`, `retain`, `Index`/`IndexMut`, and
`IntoIterator` (owned / `&` / `&mut`).

All four maps also offer disjoint mutable access — `get_disjoint_mut` (typed) and
`get_disjoint_mut_by_inner_key`, each with an `unchecked` companion.

`MapId` and `RetypePtr` are reimplemented locally, so the only dependencies are
[`slotmap`](https://crates.io/crates/slotmap) and
[`stable_deref_trait`](https://crates.io/crates/stable_deref_trait) — the latter
supplies the `StableDeref` bound (the same trait `elsa`, `owning_ref`, etc. use),
so any `StableDeref` smart pointer works as the stored `Ptr`, not just the std
ones.

## Layout

| file | contents |
| --- | --- |
| `cast_key.rs` | `CastKey<T, K>`, `StableCastKey<T, K>` (generic over `slotmap::Key`) |
| `slotmap_trait.rs` | `SlotMapTrait` trait + impls for `SlotMap` and `DenseSlotMap` |
| `unsafe_cast_map.rs` | generic `UnsafeCastMapG<M>` + iterators; `UnsafeCastMap` / `UnsafeDenseCastMap` aliases |
| `cast_map.rs` | generic `CastMapG<M>` (safe, `MapId`-checked) + iterators; `CastMap` / `DenseCastMap` aliases |
| `map_id.rs` | `MapId` |
| `retype_ptr.rs` | `RetypePtr` (Box / Rc / Arc / &T / &mut T) |

The `StableDeref` bound comes from the `stable_deref_trait` crate.

## License

MIT.

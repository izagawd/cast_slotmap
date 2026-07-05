# cast_slotmap

Castable-key wrappers over the [`slotmap`](https://crates.io/crates/slotmap)
crate's `SlotMap` and `DenseSlotMap`: store type-erased heterogeneous values
(e.g. `CastBox<dyn Any>`) and get back **typed** keys, so `map.get(key)`
returns a correctly typed `&T` with no `downcast_ref` at the call site.

> **Nightly only.** Pointer-metadata reconstruction and the dyn-dispatchable
> key use the unstable `ptr_metadata`, `coerce_unsized`, `unsize`,
> `dispatch_from_dyn`, `arbitrary_self_types`, and
> `arbitrary_self_types_pointers` features.

## The maps

Two axes — **checking** (raw vs. type-id-checked) and **storage** (basic vs.
dense):

- **`UnsafeCastMap<K, Ptr>`** — the low-level map over `slotmap::SlotMap`.
  Lookups are typed through a `CastKey<T>`, which caches the pointer metadata
  (for a `dyn` type, its vtable) needed to rebuild a `&T` from the erased value.
  The catch: `get` / `get_mut` / `remove` / `downcast_key` are `unsafe` because
  they **trust that metadata blindly** — they rebuild the `&T` straight from the
  key's cached metadata without checking it still matches the value actually in
  that slot. If the slot now holds a *different* type than the key describes, the
  method reinterprets those bytes as `T` — dispatching through the wrong vtable,
  reading past the end of the value, and so on. That's undefined behavior, not a
  `None`. (The lookup itself is fine — `slotmap`'s versioning returns `None` for
  a stale key; it's the type mismatch that bites. Using a key from a *different*
  `UnsafeCastMap` is the easy way to trip it: nothing stops its lookup from
  landing on a live slot that holds some other type.) Reach for it only when you
  can guarantee the key's type still matches the value in its slot.
- **`CastMap<K, Ptr>`** — the safe, recommended API over `slotmap::SlotMap`.
  Values live in a box that records its concrete `TypeId` (`CastBox`, or any
  box implementing `ConcreteTypeId`); every keyed lookup recovers the type id
  implied by the key's metadata (`type_id_from_meta`) and compares it to the
  slot's. A stale, mistyped, or foreign key returns `None` instead of being
  unsound — **no per-map identity needed**. Consequence: a key from map A used
  on map B is memory-safe and resolves iff B's slot holds the same version *and*
  type; keep keys with their map if you need identity.
- **`UnsafeDenseCastMap`** / **`DenseCastMap`** — the same raw/checked pair over
  `slotmap::DenseSlotMap`, which stores values contiguously for fast iteration
  (one extra indirection per lookup; `remove` reorders the survivors). The
  cast-key API is identical.

These four are thin **type aliases**. Internally there are just two generic
types — `UnsafeCastMapG<M>` and `CastMapG<M>` — parameterized over a backing
map `M: SlotMapTrait` (implemented for both `SlotMap` and `DenseSlotMap`).

For the common case use the box aliases — the checked `BoxCastMap<K, T>` /
`BoxDenseCastMap<K, T>` store `CastBox` (which supplies the type id); the raw
`UnsafeBoxCastMap<K, T>` / `UnsafeBoxDenseCastMap<K, T>` store plain `Box`.

```rust
#![feature(ptr_metadata, coerce_unsized, unsize, dispatch_from_dyn,
           arbitrary_self_types, arbitrary_self_types_pointers)]
use cast_slotmap::{BoxCastMap, CastBox, CastKey, DefaultKey};
use std::any::Any;

struct Dog { name: String }

let mut map: BoxCastMap<DefaultKey, dyn Any> = BoxCastMap::new();

// Insert a concrete type into a `dyn Any` map; the key comes back typed.
let dog_key: CastKey<Dog> = map.insert_sized(CastBox::new(Dog { name: "Rex".into() }));
assert_eq!(map.get(dog_key).unwrap().name, "Rex");

// Or insert erased and recover the typed key later.
let dyn_key: CastKey<dyn Any> = map.insert(CastBox::new(Dog { name: "Ax".into() }));
let typed: CastKey<Dog> = map.downcast_key::<Dog>(dyn_key).unwrap();
```

## `AnyHaver`: the key-side type check

Checked lookups take `T: AnyHaver`, an **`unsafe` trait** whose one method
recovers the concrete `TypeId` from pointer metadata alone (callable on a null
data pointer — only the vtable is consulted). All `'static` **sized** types get
it via a blanket impl. Trait-object keys get it by supertrait:

```rust
trait Component: AnyHaver { /* … */ }   // puts the lookup in dyn Component's vtable
```

`dyn Any` has no such supertrait, so `map.get(dyn_any_key)` is a **compile
error** rather than a silent miss — use `downcast_key` to recover a typed key,
or `get_by_inner_key` for type-erased access. Implementing `AnyHaver` manually
is `unsafe`: returning a wrong `TypeId` would make the checked lookups unsound.

## `DynKey`: dyn-dispatchable keys

`CastKey<dyn Trait>` is `Sized`, so `&CastKey<dyn Trait>` can't dispatch. 
`CastKey::as_dyn()` borrows the key as a `DynKey<'_, T>` — a single fat
`NonNull` whose metadata half is the key's vtable and whose address half packs
the backing `slotmap` key (`KeyData::as_ffi`; always nonzero since the version
is `NonZeroU32`, so `Option<DynKey>` stays pointer-sized; on 32-bit targets it
falls back to pointing at the borrowed key). That makes it a valid trait-object
**method receiver**:

```rust
trait Component: AnyHaver {
    fn tick(self: DynKey<'_, Self>, world: &mut World);
}

let key: CastKey<dyn Component> = component_key.upcast();
key.as_dyn().tick(&mut world);   // virtual call through the key's own vtable
```

Inside the method, `self.key()` returns the `CastKey<Self>` to resolve against
the map. The dispatch itself never touches the map.

## Design: this is a `SlotMap`, not a stable-reference arena

`slotmap::SlotMap::insert` takes `&mut self` (and so does `DenseSlotMap`), so
this crate **mirrors that mutability model** rather than offering a
`&self`-insert, stable-reference API:

- Every mutating method — `insert*`, `remove`, `reserve`, `clear`, `retain`,
  `drain` — takes `&mut self`.
- Each public method **delegates to the underlying `slotmap::SlotMap`** method.
- Because `get` borrows `&self` while `insert` borrows `&mut self`, references
  and inserts can never coexist. Consequently:
  - `iter` is a plain **safe** shared iterator (there is no `unsafe_iter`),
  - `Clone` is a normal forward — and since checking is by version + type id,
    keys from the original remain valid on the clone,
  - there is **no** `get_slot`, `get_by_index_only`, or `reset` — `slotmap`
    exposes no such operations, and faking them with `unsafe` would mean
    reaching past its public API. `clear` is the native way to invalidate every
    outstanding key.

## Key-level API

The key-level API is independent of how the underlying map mutates: `insert` /
`insert_with_key` / `try_insert_with_key`, the typed-key `insert_sized` family,
the source-typed `insert_as` family,
`downcast_key`, `CastKey::upcast`, `CastKey::as_dyn`, typed `get<T>` /
`get_unchecked<T>` / `remove<T>` (via `RetypePtr`), `cast_key_of`,
`get_by_inner_key`(`_mut`), `remove_by_inner_key`, `keys`,
`values`(`_mut`), `iter`(`_mut`), `drain`, `retain`, `Index`/`IndexMut`, and
`IntoIterator` (owned / `&` / `&mut`).

All four maps also offer disjoint mutable access — `get_disjoint_mut` (typed) and
`get_disjoint_mut_by_inner_key`, each with an `unchecked` companion.

The only dependencies are
[`slotmap`](https://crates.io/crates/slotmap) and
[`stable_deref_trait`](https://crates.io/crates/stable_deref_trait) — the latter
supplies the `StableDeref` bound (the same trait `elsa`, `owning_ref`, etc. use),
so any `StableDeref` smart pointer works as the stored `Ptr`, not just the std
ones (checked maps additionally need it to implement `ConcreteTypeId`).

## License

MIT.

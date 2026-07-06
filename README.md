# cast_slotmap

Castable-key wrappers over the [`slotmap`](https://crates.io/crates/slotmap)
crate's `SlotMap` and `DenseSlotMap`: store type-erased heterogeneous values
(e.g. `TypeTaggedBox<dyn Any>`) and get back **typed** keys, so `map.get(key)`
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
  (for a `dyn` type, its vtable) needed to rebuild a `&(mut) T` from the erased value.
  The catch: `get` / `get_mut` / `remove` are `unsafe` because
  they **trust that metadata blindly** — they rebuild the `&(mut) T` straight from the
  key's cached metadata without checking it still matches the value actually in
  that slot. If the slot now holds a *different* type than the key describes, the
  method reinterprets those bytes as `T` — dispatching through the wrong vtable,
  reading past the end of the value, and so on. That's undefined behavior, not a
  `None`. Reach for it only when you can guarantee the key's type still matches the value in its slot.
- **`CastMap<K, Ptr>`** — the safe, recommended API over `slotmap::SlotMap`.
  Values live behind a type-tagged pointer that records their concrete
  `TypeId` (`TypeTaggedBox`, or any stored pointer implementing
  `ConcreteTypeId` — an `unsafe` trait, since a wrong type id
  would make the checked lookups unsound); every keyed lookup recovers the type id
  implied by the key's metadata, and compares it to the
  slot's. A mistyped key returns `None` instead of being
  unsound
- **`UnsafeDenseCastMap`** / **`DenseCastMap`** — the same raw/checked pair over
  `slotmap::DenseSlotMap`, which stores values contiguously for faster iteration

These four are thin **type aliases**. Internally there are just two generic
types — `UnsafeCastMapG<M>` and `CastMapG<M>` — parameterized over a backing
map `M: SlotMapTrait` (implemented for both `SlotMap` and `DenseSlotMap`).

For the common case use the box aliases — the checked `BoxCastMap<K, T>` /
`BoxDenseCastMap<K, T>` store `TypeTaggedBox` (which supplies the type id); the raw
`UnsafeBoxCastMap<K, T>` / `UnsafeBoxDenseCastMap<K, T>` store plain `Box`.
(`TypeTaggedBox<T>` is an alias of `TypeTaggedPtr<Box<T>>`, the generic form that
pairs any smart pointer — `Rc`, `Arc`, `&T`, `&mut T`, ... — with its
pointee's concrete `TypeId`.)

```rust
#![feature(ptr_metadata, coerce_unsized, unsize, dispatch_from_dyn,
           arbitrary_self_types, arbitrary_self_types_pointers)]
use cast_slotmap::{BoxCastMap, TypeTaggedBox, CastKey, DefaultKey};
use std::any::Any;

struct Dog { name: String }

let mut map: BoxCastMap<DefaultKey, dyn Any> = BoxCastMap::new();

// Insert a concrete type into a `dyn Any` map; the key comes back typed.
let dog_key: CastKey<Dog> = map.insert_sized(TypeTaggedBox::new(Dog { name: "Rex".into() }));
assert_eq!(map.get(dog_key).unwrap().name, "Rex");

// Or insert erased and recover the typed key later.
let dyn_key: CastKey<dyn Any> = map.insert(TypeTaggedBox::new(Dog { name: "Ax".into() }));
let typed: CastKey<Dog> = map.downcast_key::<Dog>(dyn_key.inner_key()).unwrap();
```

## `AnyHaver`: the key-side type check

Checked lookups take `T: AnyHaver`, an **`unsafe` trait** whose one method
recovers the concrete `TypeId` from pointer metadata alone. All `'static` **sized** types get
it via a blanket impl. Trait-object keys get it by supertrait:

```rust
trait Component: AnyHaver { /* … */ }   // puts the lookup in dyn Component's vtable
```

`dyn Any` has no such supertrait, so `map.get(dyn_any_key)` is a **compile
error** rather than a silent miss — use `downcast_key` to recover a typed key,
or `get_by_inner_key` for type-erased access. Implementing `AnyHaver` manually
is `unsafe`: returning a wrong `TypeId` would make the checked lookups unsound.

## `DynKey`: dyn-dispatchable keys

A dyn-dispatch receiver must be exactly the size and shape of a pointer, and
`CastKey` cannot guarantee that: *pointer* size varies by target (32- vs
64-bit) while the key is a fixed 8 bytes — and `slotmap` plans to let users
pick the size of their keys — so the key cannot be relied on to fit in, or
match, a pointer. `CastKey::as_dyn` instead
borrows the key as a `DynKey<'_, T>` — a single fat
`NonNull` whose metadata half is the key's vtable and whose address half packs
the backing `slotmap` key — its `u64` `as_ffi` form — when
`size_of::<u64>() <= size_of::<usize>()` (checked per target at compile time;
nonzero is verified at runtime, not assumed from `as_ffi`'s layout), falling
back to pointing at the borrowed key when it doesn't fit. That makes it a valid trait-object **method receiver**:

```rust
trait Component: AnyHaver {
    fn tick(self: DynKey<'_, Self>, world: &mut World);
}

let key: CastKey<dyn Component> = component_key.upcast();
key.as_dyn().tick(&mut world);   // virtual call through the key's own vtable
```

Inside the method, `self.key()` returns the `CastKey<Self>` to resolve against
the map. The dispatch itself never touches the map.

## License

MIT.

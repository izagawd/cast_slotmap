# cast_slotmap

Wrappers over the [`slotmap`](https://crates.io/crates/slotmap) crate's `SlotMap` and `DenseSlotMap`. You store erased values (like `TypeTaggedBox<dyn Any>`) but get **typed** keys back, so `map.get(key)` returns a correctly typed `&T` with no `downcast_ref` at the call site.

> **Nightly only.** This crate uses the unstable `ptr_metadata`, `coerce_unsized`, `unsize`, `dispatch_from_dyn`, `arbitrary_self_types`, and `arbitrary_self_types_pointers` features.

## The maps

There are four maps, split along two choices: checked or unchecked lookups, and normal or dense storage.

* **`UnsafeCastMap<K, Ptr>`**: the low level map over `slotmap::SlotMap`. A `CastKey<T>` caches the pointer metadata (for a `dyn` type, its vtable) needed to rebuild a `&T` or `&mut T` from the erased value. The catch: `get`, `get_mut` and `remove` are `unsafe` because they trust that metadata blindly. They never check that the slot still holds the type the key describes. If the slot now holds a different type, its bytes get read as if they were a `T`: wrong vtable, reads past the end of the value, and so on. That is undefined behavior, not a `None`. Only use this map when you can guarantee the key's type still matches the value in its slot.
* **`CastMap<K, Ptr>`**: the safe map, and the recommended one. Each value lives behind a pointer that also records its concrete `TypeId` (`TypeTaggedBox`, or any stored pointer that implements `ConcreteTypeId`, an `unsafe` trait, since a wrong type id would break the safety of the checked lookups). Every keyed lookup works out the type id the key implies and compares it to the slot's. A key of the wrong type simply returns `None`.
* **`UnsafeDenseCastMap`** / **`DenseCastMap`**: the same unchecked and checked pair, but over `slotmap::DenseSlotMap`, which stores values next to each other in memory for faster iteration.

These four are thin **type aliases**. Under the hood there are only two generic types, `UnsafeCastMapG<M>` and `CastMapG<M>`, generic over a backing map `M: SlotMapTrait` (implemented for both `SlotMap` and `DenseSlotMap`).

For the common case, use the box aliases. The checked `BoxCastMap<K, T>` and `BoxDenseCastMap<K, T>` store `TypeTaggedBox` (which supplies the type id). The raw `UnsafeBoxCastMap<K, T>` and `UnsafeBoxDenseCastMap<K, T>` store a plain `Box`. (`TypeTaggedBox<T>` is an alias of `TypeTaggedPtr<Box<T>>`, the generic form that pairs any smart pointer, such as `Rc`, `Arc`, `&T` or `&mut T`, with the concrete `TypeId` of the value it points to.)

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

## `AnyHaver`: the type check on the key side

Checked lookups need `T: AnyHaver`, an **`unsafe` trait** with one method that gets the concrete `TypeId` using only pointer metadata. Every `'static` **sized** type gets it from a blanket impl. Trait objects get it through a supertrait:

```rust
trait Component: AnyHaver { /* … */ }   // puts the lookup in dyn Component's vtable
```

`dyn Any` has no such supertrait, so `map.get(dyn_any_key)` fails to compile instead of silently missing. Use `downcast_key` to get a typed key back, or `get_by_inner_key` for erased access. Implementing `AnyHaver` by hand is `unsafe`: returning a wrong `TypeId` would break the safety of the checked lookups.

## `DynKey`: keys that work with dyn dispatch

A method receiver for dyn dispatch must have exactly the size and shape of a pointer, and `CastKey` cannot promise that. Pointer size depends on the target (32 vs 64 bit) while the key is always 8 bytes, and `slotmap` plans to let users pick the size of their keys, so the key cannot be trusted to fit in, or match, a pointer. Instead, `CastKey::as_dyn` borrows the key as a `DynKey<'_, T>`: a single fat `NonNull`. Its metadata half is the key's vtable. Its address half packs the backing `slotmap` key when the size of the key is <= the size of a memory address, which is checked per target at compile time. When it does not fit, it points at the borrowed key instead. That makes it a valid **method receiver** for trait objects:

```rust
trait Component: AnyHaver {
    fn tick(self: DynKey<'_, Self>, world: &mut World);
}

let key: CastKey<dyn Component> = component_key.upcast();
key.as_dyn().tick(&mut world);   // virtual call through the key's own vtable
```

Inside the method, `self.key()` returns the `CastKey<Self>` to look things up in the map. The dispatch itself never touches the map.

## License

MIT.
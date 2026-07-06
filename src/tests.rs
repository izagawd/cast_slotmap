//! Tests for the cast slot maps. These exercise the public surface and the
//! type-id / version-based soundness checks specific to this crate.

use std::any::Any;

use crate::any_haver::{type_id_from_meta, AnyHaver};
use crate::type_tagged_ptr::TypeTaggedBox;
use crate::cast_key::CastKey;
use crate::cast_map::BoxCastMap;
use crate::dyn_key::DynKey;
use crate::unsafe_cast_map::UnsafeBoxCastMap;
use crate::DefaultKey;

#[derive(Debug, PartialEq)]
struct Dog {
    name: String,
}

#[derive(Debug, PartialEq)]
struct Cat {
    lives: u32,
}

type AnyMap = BoxCastMap<DefaultKey, dyn Any>;

// ─── insert + downcast → typed get ───────────────────────────────────────────

#[test]
fn insert_then_downcast_typed_get() {
    let mut map: AnyMap = AnyMap::new();
    // `insert` erases the concrete `Dog` and hands back the erased-target key;
    // `downcast_key` recovers a `Dog`-typed key.
    let dyn_key = map.insert(TypeTaggedBox::new(Dog { name: "Rex".into() }));
    let key = map.downcast_key::<Dog>(dyn_key.inner_key()).unwrap();
    assert_eq!(map.get(key).unwrap().name, "Rex");
    assert_eq!(map.len(), 1);
    assert!(!map.is_empty());
}

#[test]
fn typed_get_and_downcast_via_upcast() {
    let mut map: AnyMap = AnyMap::new();
    // `dyn Any` does not implement `AnyHaver` (no supertrait), so a checked
    // `map.get(dyn_key)` would not compile. `insert_sized` yields the typed
    // key for the checked get; `upcast` supplies the erased key for
    // `downcast_key`.
    let key: CastKey<Dog> = map.insert_sized(TypeTaggedBox::new(Dog { name: "Fido".into() }));
    assert_eq!(map.get(key).unwrap().name, "Fido");

    let dyn_key: CastKey<dyn Any> = key.upcast();
    let recovered: CastKey<Dog> = map.downcast_key::<Dog>(dyn_key.inner_key()).unwrap();
    assert_eq!(map.get(recovered).unwrap().name, "Fido");
}

// ─── insert_sized: typed key straight from insertion ─────────────────────────

#[test]
fn insert_sized_gives_typed_key() {
    let mut map: AnyMap = AnyMap::new();
    let key: CastKey<Dog> = map.insert_sized(TypeTaggedBox::new(Dog { name: "Sz".into() }));
    assert_eq!(map.get(key).unwrap().name, "Sz");

    // The same slot is reachable through the backing key, type-erased.
    assert!(map.get_by_inner_key(key.inner_key()).unwrap().is::<Dog>());
}

// ─── upcast + downcast_key ───────────────────────────────────────────────────

#[test]
fn downcast_key_right_and_wrong_type() {
    let mut map: AnyMap = AnyMap::new();
    let dyn_key = map.insert(TypeTaggedBox::new(Dog { name: "Spot".into() }));

    let recovered: CastKey<Dog> = map.downcast_key::<Dog>(dyn_key.inner_key()).unwrap();
    assert_eq!(map.get(recovered).unwrap().name, "Spot");

    assert!(map.downcast_key::<Cat>(dyn_key.inner_key()).is_none());
}

// ─── get_mut ─────────────────────────────────────────────────────────────────

#[test]
fn get_mut_mutates() {
    let mut map: AnyMap = AnyMap::new();
    let kx = map.insert(TypeTaggedBox::new(Cat { lives: 9 }));
    let key = map.downcast_key::<Cat>(kx.inner_key()).unwrap();
    map.get_mut(key).unwrap().lives -= 1;
    assert_eq!(map.get(key).unwrap().lives, 8);
}

// ─── remove returns a re-typed TypeTaggedBox ───────────────────────────────────────

#[test]
fn remove_returns_boxed_concrete() {
    let mut map: AnyMap = AnyMap::new();
    let kx = map.insert(TypeTaggedBox::new(Dog { name: "Bud".into() }));
    let key = map.downcast_key::<Dog>(kx.inner_key()).unwrap();

    let removed: TypeTaggedBox<Dog> = map.remove(key).unwrap();
    assert_eq!(removed.name, "Bud");

    // The key is now stale (slotmap bumps the slot version on remove).
    assert!(map.get(key).is_none());
    assert!(map.is_empty());
}

// ─── removing through a mistyped key is rejected ─────────────────────────────

#[test]
fn remove_wrong_type_is_rejected() {
    let mut map: AnyMap = AnyMap::new();
    let dog_key: CastKey<Dog> = map.insert_sized(TypeTaggedBox::new(Dog { name: "Kept".into() }));

    // Forge a `Cat`-typed key naming the same slot; the type check refuses it.
    // SAFETY of the test's premise: `from_raw_parts` is the unsafe key-construction
    // path — the map must remain safe against exactly this.
    let wrong: CastKey<Cat> = unsafe { CastKey::from_raw_parts(dog_key.inner_key(), ()) };
    assert!(map.remove(wrong).is_none());
    assert!(map.get(wrong).is_none());
    assert_eq!(map.get(dog_key).unwrap().name, "Kept");
}

// ─── version invalidation ────────────────────────────────────────────────────

#[test]
fn stale_key_after_remove_reinsert() {
    let mut map: AnyMap = AnyMap::new();
    let k1x = map.insert(TypeTaggedBox::new(Cat { lives: 1 }));
    let k1 = map.downcast_key::<Cat>(k1x.inner_key()).unwrap();
    let _ = map.remove(k1).unwrap();
    let k2x = map.insert(TypeTaggedBox::new(Cat { lives: 2 }));
    let k2 = map.downcast_key::<Cat>(k2x.inner_key()).unwrap();

    // Old key does not alias the new value.
    assert!(map.get(k1).is_none());
    assert_eq!(map.get(k2).unwrap().lives, 2);
}

// ─── clear invalidates ───────────────────────────────────────────────────────

#[test]
fn clear_invalidates_keys() {
    let mut map: AnyMap = AnyMap::new();
    let kx = map.insert(TypeTaggedBox::new(Dog { name: "Rex".into() }));
    let key = map.downcast_key::<Dog>(kx.inner_key()).unwrap();
    map.clear();
    assert!(map.get(key).is_none());
    assert!(map.is_empty());
}

// ─── cross-map keys under the type-id model ──────────────────────────────────
//
// A foreign key is memory-safe, resolving iff the slot it names is live and
// holds a value of the key's type.

#[test]
fn cross_map_wrong_type_is_rejected() {
    let mut a: AnyMap = AnyMap::new();
    let mut b: AnyMap = AnyMap::new();

    let ka: CastKey<Dog> = a.insert_sized(TypeTaggedBox::new(Dog { name: "A".into() }));
    // Same slot index + version in `b`, but holding a Cat.
    let _kb: CastKey<Cat> = b.insert_sized(TypeTaggedBox::new(Cat { lives: 9 }));

    assert!(b.get(ka).is_none());
    assert!(!b.contains_key(ka));
    assert!(b.remove(ka).is_none());
    assert!(b.downcast_key::<Dog>(ka.inner_key()).is_none());

    // Original still works.
    assert_eq!(a.get(ka).unwrap().name, "A");
}

#[test]
fn cross_map_same_type_resolves() {
    let mut a: AnyMap = AnyMap::new();
    let mut b: AnyMap = AnyMap::new();

    let ka: CastKey<Dog> = a.insert_sized(TypeTaggedBox::new(Dog { name: "A".into() }));
    let _kb: CastKey<Dog> = b.insert_sized(TypeTaggedBox::new(Dog { name: "B".into() }));

    // Documented semantics: same slot, same version, same type — the foreign
    // key resolves (to *b*'s value). Safe, if surprising; keep keys with their
    // map if you need identity.
    assert_eq!(b.get(ka).unwrap().name, "B");
}

// ─── contains_key ────────────────────────────────────────────────────────────

#[test]
fn contains_key_tracks_liveness() {
    let mut map: AnyMap = AnyMap::new();
    let kx = map.insert(TypeTaggedBox::new(Cat { lives: 9 }));
    let key = map.downcast_key::<Cat>(kx.inner_key()).unwrap();
    assert!(map.contains_key(key));
    let _ = map.remove(key);
    assert!(!map.contains_key(key));
}

// ─── iter / values / keys ────────────────────────────────────────────────────

#[test]
fn iter_and_values_and_keys() {
    let mut map: AnyMap = AnyMap::new();
    map.insert(TypeTaggedBox::new(Dog { name: "a".into() }));
    map.insert(TypeTaggedBox::new(Cat { lives: 1 }));
    map.insert(TypeTaggedBox::new(Cat { lives: 2 }));

    assert_eq!(map.iter().count(), 3);
    assert_eq!(map.values().count(), 3);
    assert_eq!(map.keys().count(), 3);

    let cat_lives_sum: u32 = map
        .values()
        .filter_map(|v| v.downcast_ref::<Cat>())
        .map(|c| c.lives)
        .sum();
    assert_eq!(cat_lives_sum, 3);
}

// ─── iter_mut / values_mut ───────────────────────────────────────────────────

#[test]
fn iter_mut_mutates_all() {
    let mut map: AnyMap = AnyMap::new();
    map.insert(TypeTaggedBox::new(Cat { lives: 1 }));
    map.insert(TypeTaggedBox::new(Cat { lives: 1 }));

    for (_k, v) in map.iter_mut() {
        if let Some(c) = v.downcast_mut::<Cat>() {
            c.lives += 10;
        }
    }

    let total: u32 = map
        .values()
        .filter_map(|v| v.downcast_ref::<Cat>())
        .map(|c| c.lives)
        .sum();
    assert_eq!(total, 22);
}

// ─── retain ──────────────────────────────────────────────────────────────────

#[test]
fn retain_keeps_matching() {
    let mut map: AnyMap = AnyMap::new();
    map.insert(TypeTaggedBox::new(Cat { lives: 1 }));
    map.insert(TypeTaggedBox::new(Cat { lives: 9 }));
    map.insert(TypeTaggedBox::new(Dog { name: "z".into() }));

    map.retain(|_k, v| v.downcast_ref::<Cat>().map_or(false, |c| c.lives > 5));
    assert_eq!(map.len(), 1);
    assert!(map.values().next().unwrap().is::<Cat>());
}

// ─── drain ───────────────────────────────────────────────────────────────────

#[test]
fn drain_empties_and_yields() {
    let mut map: AnyMap = AnyMap::new();
    map.insert(TypeTaggedBox::new(Cat { lives: 3 }));
    map.insert(TypeTaggedBox::new(Cat { lives: 4 }));

    let drained: Vec<TypeTaggedBox<dyn Any>> = map.drain().map(|(_k, v)| v).collect();
    assert_eq!(drained.len(), 2);
    assert!(map.is_empty());
}

// ─── typed get of a sized value inside a dyn Any map ─────────────────────────
// (`Index`/`IndexMut` themselves require the map's *output* type to be
// `AnyHaver`, which `dyn Any` is not — square-bracket indexing is therefore
// exercised on the sized-output map in `index_mut_mutates` and
// `index_panics_on_stale_key` instead.)

#[test]
fn index_reads() {
    let mut map: BoxCastMap<DefaultKey, dyn Any> = BoxCastMap::new();
    let key: CastKey<u32> = map.insert_sized(TypeTaggedBox::new(41u32));
    // Index is generic over the key's type: a concrete-typed index into a
    // `dyn Any` map yields the concrete reference.
    assert_eq!(map[key], 41);
}

// ─── capacity / with_capacity ────────────────────────────────────────────────

#[test]
fn with_capacity_reserves() {
    let map: AnyMap = AnyMap::with_capacity(16);
    assert!(map.capacity() >= 16);
    assert!(map.is_empty());
}

// ─── Clone keeps keys valid ──────────────────────────────────────────────────

#[test]
fn clone_keys_stay_valid() {
    // A map is `Clone` iff its stored pointer type is. `TypeTaggedBox` is not
    // `Clone` (its inner `Box<T: ?Sized>` isn't), so the checked box maps are
    // never `Clone`; the unsafe map storing `Box<u32>` is, and demonstrates
    // the shared semantics: checking is by slot version (plus, on the checked
    // layer, stored type id), so keys handed out by the original remain valid
    // on the clone.
    let mut map: UnsafeBoxCastMap<DefaultKey, u32> = UnsafeBoxCastMap::new();
    let key: CastKey<u32> = map.insert(Box::new(7u32));

    let clone = map.clone();
    // SAFETY: same slot layout in the clone; the key was issued for a `u32`
    // slot and the clone stores the same type.
    assert_eq!(*unsafe { clone.get(key) }.unwrap(), 7);
    assert_eq!(*unsafe { map.get(key) }.unwrap(), 7);
}

// ─── insert_with_key sees its own key ────────────────────────────────────────

#[test]
fn insert_with_key_threads_key() {
    let mut map: AnyMap = AnyMap::new();
    // The closure receives the backing key and returns the value to store; the
    // returned key is the erased-target key.
    let mut captured = None;
    let dyn_key = map.insert_with_key(|k| {
        captured = Some(k);
        let boxed: TypeTaggedBox<dyn Any> = TypeTaggedBox::new(123u32);
        boxed
    });
    assert!(captured.is_some());

    let key = map.downcast_key::<u32>(dyn_key.inner_key()).unwrap();
    assert_eq!(*map.get(key).unwrap(), 123);
}

#[test]
fn insert_sized_with_key_threads_typed_key() {
    let mut map: AnyMap = AnyMap::new();
    let mut captured: Option<CastKey<Dog>> = None;
    let key = map.insert_sized_with_key(|k| {
        captured = Some(k);
        TypeTaggedBox::new(Dog { name: "WK".into() })
    });
    assert_eq!(captured.unwrap(), key);
    assert_eq!(map.get(key).unwrap().name, "WK");
}

// ─── DynKey: round-trip + dyn dispatch through the key's metadata ────────────

trait Pet: AnyHaver + Any {
    fn speak(self: DynKey<'_, Self>, map: &AnyMap) -> String;
}

impl Pet for Dog {
    fn speak(self: DynKey<'_, Self>, map: &AnyMap) -> String {
        format!("woof {}", map.get(self.key()).unwrap().name)
    }
}

impl Pet for Cat {
    fn speak(self: DynKey<'_, Self>, map: &AnyMap) -> String {
        format!("meow x{}", map.get(self.key()).unwrap().lives)
    }
}

#[test]
fn dyn_key_is_send_sync_when_castkey_is() {
    // Compile-time assertion: `DefaultKey` is `Sync`, so `CastKey<T>` is, so
    // `DynKey` must be `Send + Sync`. Fails to compile if the bounds regress.
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<DynKey<'static, Dog>>();
    assert_send_sync::<DynKey<'static, dyn Pet>>();
}

#[test]
fn dyn_key_round_trips() {
    let mut map: AnyMap = AnyMap::new();
    let key: CastKey<Dog> = map.insert_sized(TypeTaggedBox::new(Dog { name: "RT".into() }));

    // Sized target: metadata is `()`; only the smuggled key must round-trip.
    let back = key.as_dyn().key();
    assert_eq!(back, key);
    assert_eq!(map.get(back).unwrap().name, "RT");

    // Dyn target: the vtable metadata must survive the round-trip too.
    let pet_key: CastKey<dyn Pet> = key.upcast();
    let pet_back = pet_key.as_dyn().key();
    assert_eq!(pet_back, pet_key);
    assert_eq!(
        type_id_from_meta::<dyn Pet>(pet_back.metadata()),
        type_id_from_meta::<dyn Pet>(pet_key.metadata()),
    );
}

#[test]
fn dyn_key_coerced_round_trips() {
    let mut map: AnyMap = AnyMap::new();
    let dog: CastKey<Dog> = map.insert_sized(TypeTaggedBox::new(Dog { name: "Co".into() }));

    // Safe unsizing coercion of the DynKey itself (not of the CastKey);
    // `key()` must still recover a correct `CastKey<dyn Pet>` afterwards.
    let dk: DynKey<'_, dyn Pet> = dog.as_dyn();
    let back: CastKey<dyn Pet> = dk.key();
    assert_eq!(back, dog.upcast::<dyn Pet>());
    assert_eq!(back.as_dyn().speak(&map), "woof Co");
}

#[test]
fn dyn_key_dispatches_virtually() {
    let mut map: AnyMap = AnyMap::new();
    let dog: CastKey<Dog> = map.insert_sized(TypeTaggedBox::new(Dog { name: "Rex".into() }));
    let cat: CastKey<Cat> = map.insert_sized(TypeTaggedBox::new(Cat { lives: 9 }));

    // Erase to `dyn Pet` keys; dispatch selects the concrete impl through the
    // vtable carried in the key itself — the map is only consulted *inside*
    // the methods.
    let pets: [CastKey<dyn Pet>; 2] = [dog.upcast(), cat.upcast()];
    let spoken: Vec<String> = pets.iter().map(|k| k.as_dyn().speak(&map)).collect();
    assert_eq!(spoken, ["woof Rex", "meow x9"]);
}

// ─── try_insert_with_key error path leaves the map untouched ─────────────────

#[test]
fn try_insert_error_is_noop() {
    let mut map: BoxCastMap<DefaultKey, u32> = BoxCastMap::new();
    let res: Result<CastKey<u32>, &str> = map.try_insert_with_key(|_k| Err("nope"));
    assert_eq!(res.err(), Some("nope"));
    assert!(map.is_empty());
}

#[test]
fn get_disjoint_mut_basic() {
    let mut map: BoxCastMap<DefaultKey, u32> = BoxCastMap::new();
    let k1: CastKey<u32> = map.insert(TypeTaggedBox::new(10u32));
    let k2: CastKey<u32> = map.insert(TypeTaggedBox::new(20u32));

    let [a, b] = map.get_disjoint_mut([k1, k2]).unwrap();
    *a += 1;
    *b += 2;
    assert_eq!(*map.get(k1).unwrap(), 11);
    assert_eq!(*map.get(k2).unwrap(), 22);

    // aliasing keys are rejected
    assert!(map.get_disjoint_mut([k1, k1]).is_none());

    // disjoint access by backing key, too
    let [c] = map.get_disjoint_mut_by_inner_key([k2.inner_key()]).unwrap();
    *c += 100;
    assert_eq!(*map.get(k2).unwrap(), 122);
}

// ─── unsized non-`dyn` target: slice length metadata round-trips ─────────────

#[test]
fn slice_target_metadata_roundtrip() {
    // The key carries slice *length* metadata (a `usize`), not a vtable — the
    // one `Pointee::Metadata` kind the `dyn Any` and sized tests don't
    // exercise. This lives on the unsafe map: the checked map's type-id test
    // compares against the *concrete* stored type (e.g. `[u32; 3]`), which a
    // `CastKey<[u32]>` can never name, so slice targets are an unsafe-map
    // use case.
    let mut map: UnsafeBoxCastMap<DefaultKey, [u32]> = UnsafeBoxCastMap::new();
    let key: CastKey<[u32]> = map.insert(vec![1u32, 2, 3].into_boxed_slice());
    // SAFETY: key was just issued for this slot; metadata (the length) is valid.
    assert_eq!(unsafe { map.get(key) }.unwrap(), &[1, 2, 3]);

    // SAFETY: same key, slot unchanged.
    unsafe { map.get_mut(key) }.unwrap()[0] = 9;
    assert_eq!(unsafe { map.get(key) }.unwrap(), &[9, 2, 3]);
}

// ─── owning IntoIterator yields the cast keys + values ───────────────────────

#[test]
fn owned_into_iterator_yields_all() {
    let mut map: AnyMap = AnyMap::new();
    map.insert(TypeTaggedBox::new(Cat { lives: 1 }));
    map.insert(TypeTaggedBox::new(Cat { lives: 2 }));

    let sum: u32 = map
        .into_iter()
        .filter_map(|(_k, v)| v.downcast_ref::<Cat>().map(|c| c.lives))
        .sum();
    assert_eq!(sum, 3);
}

// ─── IndexMut mutates in place ───────────────────────────────────────────────

#[test]
fn index_mut_mutates() {
    let mut map: BoxCastMap<DefaultKey, u32> = BoxCastMap::new();
    let key: CastKey<u32> = map.insert(TypeTaggedBox::new(5u32));
    map[key] += 10;
    assert_eq!(map[key], 15);
}

// ─── UnsafeCastMap direct usage ──────────────────────────────────────────────

#[test]
fn unsafe_map_typed_roundtrip() {
    let mut map: UnsafeBoxCastMap<DefaultKey, dyn Any> = UnsafeBoxCastMap::new();
    let key: CastKey<Dog> = map.insert_sized(Box::new(Dog { name: "U".into() }));

    // SAFETY: the slot still holds the `Dog` `key` was made for, so its metadata is valid.
    let d: &Dog = unsafe { map.get(key).unwrap() };
    assert_eq!(d.name, "U");

    // SAFETY: same key, still valid.
    let removed: Box<Dog> = unsafe { map.remove(key).unwrap() };
    assert_eq!(removed.name, "U");
    assert!(map.is_empty());
}

#[test]
fn unsafe_map_detach_reattach() {
    let mut map: UnsafeBoxCastMap<DefaultKey, dyn Any> = UnsafeBoxCastMap::new();
    let key: CastKey<Dog> = map.insert_sized(Box::new(Dog { name: "Rex".into() }));

    // SAFETY: the slot still holds the `Dog` `key` was made for, so its metadata is valid.
    let mut dog: Box<Dog> = unsafe { map.detach(key).unwrap() };
    assert_eq!(dog.name, "Rex");
    // SAFETY: the slot is detached, so the lookup misses without using metadata.
    assert!(unsafe { map.get(key) }.is_none());
    assert!(map.is_empty());

    // Reattach the (mutated) value. `reattach` takes the map's erased-target
    // key, so upcast the typed `CastKey<Dog>` to `CastKey<dyn Any>`; the
    // `Box<Dog>` value unsizes to `Box<dyn Any>` at the call site.
    dog.name = "Max".into();
    map.reattach(key.upcast::<dyn Any>(), dog);
    // SAFETY: a `Dog` is back in the slot, so `key`'s metadata is still correct.
    assert_eq!(unsafe { map.get(key) }.unwrap().name, "Max");
    assert_eq!(map.len(), 1);

    // Backing-key detach/reattach with an already-erased pointer.
    let ik = key.inner_key();
    let erased: Box<dyn Any> = map.detach_by_inner_key(ik).unwrap();
    assert_eq!(erased.downcast_ref::<Dog>().unwrap().name, "Max");
    map.reattach_by_inner_key(ik, Box::new(Dog { name: "Zed".into() }) as Box<dyn Any>);
    // SAFETY: still a `Dog`, so `key`'s metadata remains valid.
    assert_eq!(unsafe { map.get(key) }.unwrap().name, "Zed");
}

// ─── insert_as: source-typed keys for already-unsized pointers ───────────────

#[test]
fn insert_as_keeps_source_typed_key() {
    let mut map: AnyMap = AnyMap::new();
    // `TypeTaggedBox::new(Dog)` unsizes to `TypeTaggedBox<dyn Pet>` first; `insert_as`
    // then coerces that into the map's `TypeTaggedBox<dyn Any>` (trait upcasting,
    // possible because `Pet: Any`), while the returned key keeps the *source*
    // `dyn Pet` typing — something `insert_sized` (Sized targets only) and
    // `insert` (erased-target key) cannot express.
    let pet_box: TypeTaggedBox<dyn Pet> = TypeTaggedBox::new(Dog { name: "As".into() });
    let key: CastKey<dyn Pet> = map.insert_as(pet_box);

    // The checked lookup validates the key's `dyn Pet` vtable (concrete type
    // `Dog`) against the slot's stored type id, and virtual dispatch through
    // the key's own metadata still works.
    assert_eq!(key.as_dyn().speak(&map), "woof As");

    // `remove` re-types the stored box back to the source type.
    let removed: TypeTaggedBox<dyn Pet> = map.remove(key).unwrap();
    assert!(map.is_empty());
    drop(removed);
}

#[test]
fn insert_as_with_key_threads_backing_key() {
    let mut map: AnyMap = AnyMap::new();
    // Unlike `insert_sized_with_key`, the closure receives only the backing
    // `slotmap` key: the pointer metadata does not exist until the value is
    // constructed, so a typed `CastKey` cannot be built up front.
    let mut captured = None;
    let key: CastKey<dyn Pet> = map.insert_as_with_key(|k| {
        captured = Some(k);
        let boxed: TypeTaggedBox<dyn Pet> = TypeTaggedBox::new(Cat { lives: 3 });
        boxed
    });
    assert_eq!(captured.unwrap(), key.inner_key());
    assert_eq!(key.as_dyn().speak(&map), "meow x3");
}

// ─── fallible insert closures leave the map untouched ────────────────────────

#[test]
fn try_insert_sized_error_is_noop() {
    let mut map: AnyMap = AnyMap::new();
    let res: Result<CastKey<Dog>, &str> =
        map.try_insert_sized_with_key(|_k| Err::<TypeTaggedBox<Dog>, _>("nope"));
    assert_eq!(res.err(), Some("nope"));
    assert!(map.is_empty());
}

#[test]
fn try_insert_as_error_is_noop() {
    let mut map: AnyMap = AnyMap::new();
    let res: Result<CastKey<dyn Pet>, &str> =
        map.try_insert_as_with_key(|_k| Err::<TypeTaggedBox<dyn Pet>, _>("nope"));
    assert_eq!(res.err(), Some("nope"));
    assert!(map.is_empty());
}

// ─── cast_key_of: backing key → erased-target CastKey ────────────────────────

#[test]
fn cast_key_of_live_and_stale() {
    let mut map: AnyMap = AnyMap::new();
    let key: CastKey<Cat> = map.insert_sized(TypeTaggedBox::new(Cat { lives: 5 }));

    // Live slot: metadata is re-read from the stored value.
    let erased = map.cast_key_of(key.inner_key()).unwrap();
    assert_eq!(erased.inner_key(), key.inner_key());
    assert_eq!(map.downcast_key::<Cat>(erased.inner_key()).unwrap(), key);

    // Stale after remove: slotmap's version check rejects the backing key.
    let _ = map.remove(key).unwrap();
    assert!(map.cast_key_of(key.inner_key()).is_none());
}

// ─── get_disjoint_mut rejects a mistyped key ─────────────────────────────────

#[test]
fn get_disjoint_mut_wrong_type_is_rejected() {
    let mut map: BoxCastMap<DefaultKey, u32> = BoxCastMap::new();
    let k: CastKey<u32> = map.insert(TypeTaggedBox::new(7u32));

    // Forge a `Cat`-typed key naming the same live slot; the per-key type-id
    // pre-check must refuse it before any mutable borrow is handed out.
    let wrong: CastKey<Cat> = unsafe { CastKey::from_raw_parts(k.inner_key(), ()) };
    assert!(map.get_disjoint_mut([wrong]).is_none());

    // The honest key is unaffected.
    let [v] = map.get_disjoint_mut([k]).unwrap();
    *v += 1;
    assert_eq!(*map.get(k).unwrap(), 8);
}

// ─── downcast_key on a version-stale key ─────────────────────────────────────

#[test]
fn downcast_key_stale_returns_none() {
    let mut map: AnyMap = AnyMap::new();
    let dog: CastKey<Dog> = map.insert_sized(TypeTaggedBox::new(Dog { name: "St".into() }));
    let dyn_key: CastKey<dyn Any> = dog.upcast();

    let _ = map.remove(dog).unwrap();
    // The slot version was bumped by the removal, so the erased key no longer
    // resolves — even though a `Dog` used to live there.
    assert!(map.downcast_key::<Dog>(dyn_key.inner_key()).is_none());
}

// ─── backing-key mutation helpers ────────────────────────────────────────────

#[test]
fn inner_key_mut_helpers_work() {
    let mut map: BoxCastMap<DefaultKey, u32> = BoxCastMap::new();
    let k: CastKey<u32> = map.insert(TypeTaggedBox::new(10u32));

    *map.get_by_inner_key_mut(k.inner_key()).unwrap() += 1;
    for v in map.values_mut() {
        *v += 1;
    }
    assert_eq!(*map.get(k).unwrap(), 12);

    let removed: TypeTaggedBox<u32> = map.remove_by_inner_key(k.inner_key()).unwrap();
    assert_eq!(*removed, 12);
    assert!(map.is_empty());
}

// ─── Index panics on an invalid key ──────────────────────────────────────────

#[test]
#[should_panic(expected = "invalid CastKey")]
fn index_panics_on_stale_key() {
    let mut map: BoxCastMap<DefaultKey, u32> = BoxCastMap::new();
    let key: CastKey<u32> = map.insert(TypeTaggedBox::new(1u32));
    let _ = map.remove(key);
    let _ = &map[key];
}

// ─────────────────────────────────────────────────────────────────────────────
// Dense variants: identical behaviour, backed by DenseSlotMap.
// ─────────────────────────────────────────────────────────────────────────────

mod dense {
    use std::any::Any;

    use super::{Cat, Dog};
    use crate::type_tagged_ptr::TypeTaggedBox;
    use crate::cast_key::CastKey;
    use crate::cast_map::BoxDenseCastMap;
    use crate::unsafe_cast_map::UnsafeBoxDenseCastMap;
    use crate::DefaultKey;

    type AnyMap = BoxDenseCastMap<DefaultKey, dyn Any>;

    #[test]
    fn insert_then_downcast_typed_get() {
        let mut map: AnyMap = AnyMap::new();
        let dyn_key = map.insert(TypeTaggedBox::new(Dog { name: "Rex".into() }));
        let key = map.downcast_key::<Dog>(dyn_key.inner_key()).unwrap();
        assert_eq!(map.get(key).unwrap().name, "Rex");
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn typed_get_and_downcast_key() {
        let mut map: AnyMap = AnyMap::new();
        let key: CastKey<Dog> = map.insert_sized(TypeTaggedBox::new(Dog { name: "Fido".into() }));
        assert_eq!(map.get(key).unwrap().name, "Fido");

        let dyn_key: CastKey<dyn Any> = key.upcast();
        let recovered: CastKey<Dog> = map.downcast_key::<Dog>(dyn_key.inner_key()).unwrap();
        assert_eq!(map.get(recovered).unwrap().name, "Fido");
        assert!(map.downcast_key::<Cat>(dyn_key.inner_key()).is_none());
    }

    #[test]
    fn get_mut_then_remove_invalidates() {
        let mut map: AnyMap = AnyMap::new();
        let kx = map.insert(TypeTaggedBox::new(Cat { lives: 9 }));
        let key = map.downcast_key::<Cat>(kx.inner_key()).unwrap();
        map.get_mut(key).unwrap().lives -= 1;
        assert_eq!(map.get(key).unwrap().lives, 8);

        let removed: TypeTaggedBox<Cat> = map.remove(key).unwrap();
        assert_eq!(removed.lives, 8);
        assert!(map.get(key).is_none()); // stale after remove
        assert!(map.is_empty());
    }

    #[test]
    fn cross_map_wrong_type_is_rejected() {
        let mut a: AnyMap = AnyMap::new();
        let mut b: AnyMap = AnyMap::new();

        let ka: CastKey<Dog> = a.insert_sized(TypeTaggedBox::new(Dog { name: "A".into() }));
        let _kb: CastKey<Cat> = b.insert_sized(TypeTaggedBox::new(Cat { lives: 1 }));

        assert!(b.get(ka).is_none());
        assert!(!b.contains_key(ka));
        assert_eq!(a.get(ka).unwrap().name, "A");
    }

    #[test]
    fn iter_values_keys_retain() {
        let mut map: AnyMap = AnyMap::new();
        map.insert(TypeTaggedBox::new(Dog { name: "a".into() }));
        map.insert(TypeTaggedBox::new(Cat { lives: 1 }));
        map.insert(TypeTaggedBox::new(Cat { lives: 9 }));

        assert_eq!(map.iter().count(), 3);
        assert_eq!(map.values().count(), 3);
        assert_eq!(map.keys().count(), 3);

        map.retain(|_k, v| v.downcast_ref::<Cat>().map_or(false, |c| c.lives > 5));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn unsafe_dense_map_roundtrip() {
        let mut map: UnsafeBoxDenseCastMap<DefaultKey, dyn Any> = UnsafeBoxDenseCastMap::new();
        let key: CastKey<Dog> = map.insert_sized(Box::new(Dog { name: "U".into() }));

        // SAFETY: the slot still holds the `Dog` `key` was made for, so its metadata is valid.
        let d: &Dog = unsafe { map.get(key).unwrap() };
        assert_eq!(d.name, "U");

        // SAFETY: same key, still valid.
        let removed: Box<Dog> = unsafe { map.remove(key).unwrap() };
        assert_eq!(removed.name, "U");
        assert!(map.is_empty());
    }

    #[test]
    fn insert_as_roundtrip() {
        use super::Pet;

        let mut map: AnyMap = AnyMap::new();
        let pet_box: TypeTaggedBox<dyn Pet> = TypeTaggedBox::new(Dog { name: "D".into() });
        let key: CastKey<dyn Pet> = map.insert_as(pet_box);
        assert_eq!(map.len(), 1);

        let removed: TypeTaggedBox<dyn Pet> = map.remove(key).unwrap();
        assert!(map.is_empty());
        drop(removed);
    }

    #[test]
    fn get_disjoint_mut_typed() {
        let mut map: BoxDenseCastMap<DefaultKey, u32> = BoxDenseCastMap::new();
        let k1: CastKey<u32> = map.insert(TypeTaggedBox::new(1u32));
        let k2: CastKey<u32> = map.insert(TypeTaggedBox::new(2u32));
        let k3: CastKey<u32> = map.insert(TypeTaggedBox::new(3u32));

        let [a, b, c] = map.get_disjoint_mut([k1, k2, k3]).unwrap();
        *a += 10;
        *b += 20;
        *c += 30;
        assert_eq!(*map.get(k1).unwrap(), 11);
        assert_eq!(*map.get(k2).unwrap(), 22);
        assert_eq!(*map.get(k3).unwrap(), 33);

        // aliasing keys are rejected
        assert!(map.get_disjoint_mut([k1, k1]).is_none());
    }
}

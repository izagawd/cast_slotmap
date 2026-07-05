//! Tests for the cast slot maps. These exercise the public surface and the
//! type-id / version-based soundness checks specific to this crate, on top of
//! `slotmap`'s `&mut self` mutation model.

use std::any::Any;

use crate::any_haver::{type_id_from_meta, AnyHaver};
use crate::cast_box::CastBox;
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
    let dyn_key = map.insert(CastBox::new(Dog { name: "Rex".into() }));
    let key = map.downcast_key::<Dog>(dyn_key).unwrap();
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
    let key: CastKey<Dog> = map.insert_sized(CastBox::new(Dog { name: "Fido".into() }));
    assert_eq!(map.get(key).unwrap().name, "Fido");

    let dyn_key: CastKey<dyn Any> = key.upcast();
    let recovered: CastKey<Dog> = map.downcast_key::<Dog>(dyn_key).unwrap();
    assert_eq!(map.get(recovered).unwrap().name, "Fido");
}

// ─── insert_sized: typed key straight from insertion ─────────────────────────

#[test]
fn insert_sized_gives_typed_key() {
    let mut map: AnyMap = AnyMap::new();
    let key: CastKey<Dog> = map.insert_sized(CastBox::new(Dog { name: "Sz".into() }));
    assert_eq!(map.get(key).unwrap().name, "Sz");

    // The same slot is reachable through the backing key, type-erased.
    assert!(map.get_by_inner_key(key.inner_key()).unwrap().is::<Dog>());
}

// ─── upcast + downcast_key ───────────────────────────────────────────────────

#[test]
fn downcast_key_right_and_wrong_type() {
    let mut map: AnyMap = AnyMap::new();
    let dyn_key = map.insert(CastBox::new(Dog { name: "Spot".into() }));

    let recovered: CastKey<Dog> = map.downcast_key::<Dog>(dyn_key).unwrap();
    assert_eq!(map.get(recovered).unwrap().name, "Spot");

    assert!(map.downcast_key::<Cat>(dyn_key).is_none());
}

// ─── get_mut ─────────────────────────────────────────────────────────────────

#[test]
fn get_mut_mutates() {
    let mut map: AnyMap = AnyMap::new();
    let kx = map.insert(CastBox::new(Cat { lives: 9 }));
    let key = map.downcast_key::<Cat>(kx).unwrap();
    map.get_mut(key).unwrap().lives -= 1;
    assert_eq!(map.get(key).unwrap().lives, 8);
}

// ─── remove returns a re-typed CastBox ───────────────────────────────────────

#[test]
fn remove_returns_boxed_concrete() {
    let mut map: AnyMap = AnyMap::new();
    let kx = map.insert(CastBox::new(Dog { name: "Bud".into() }));
    let key = map.downcast_key::<Dog>(kx).unwrap();

    let removed: CastBox<Dog> = map.remove(key).unwrap();
    assert_eq!(removed.name, "Bud");

    // The key is now stale (slotmap bumps the slot version on remove).
    assert!(map.get(key).is_none());
    assert!(map.is_empty());
}

// ─── removing through a mistyped key is rejected ─────────────────────────────

#[test]
fn remove_wrong_type_is_rejected() {
    let mut map: AnyMap = AnyMap::new();
    let dog_key: CastKey<Dog> = map.insert_sized(CastBox::new(Dog { name: "Kept".into() }));

    // Forge a `Cat`-typed key naming the same slot; the type check refuses it.
    // SAFETY of the test's premise: `from_raw_parts` is the unsafe minting
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
    let k1x = map.insert(CastBox::new(Cat { lives: 1 }));
    let k1 = map.downcast_key::<Cat>(k1x).unwrap();
    let _ = map.remove(k1).unwrap();
    let k2x = map.insert(CastBox::new(Cat { lives: 2 }));
    let k2 = map.downcast_key::<Cat>(k2x).unwrap();

    // Old key does not alias the new value.
    assert!(map.get(k1).is_none());
    assert_eq!(map.get(k2).unwrap().lives, 2);
}

// ─── clear invalidates ───────────────────────────────────────────────────────

#[test]
fn clear_invalidates_keys() {
    let mut map: AnyMap = AnyMap::new();
    let kx = map.insert(CastBox::new(Dog { name: "Rex".into() }));
    let key = map.downcast_key::<Dog>(kx).unwrap();
    map.clear();
    assert!(map.get(key).is_none());
    assert!(map.is_empty());
}

// ─── cross-map keys under the type-id model ──────────────────────────────────
//
// There is no per-map identity: a foreign key is memory-safe, resolving iff
// the slot it names is live and holds a value of the key's type.

#[test]
fn cross_map_wrong_type_is_rejected() {
    let mut a: AnyMap = AnyMap::new();
    let mut b: AnyMap = AnyMap::new();

    let ka: CastKey<Dog> = a.insert_sized(CastBox::new(Dog { name: "A".into() }));
    // Same slot index + version in `b`, but holding a Cat.
    let _kb: CastKey<Cat> = b.insert_sized(CastBox::new(Cat { lives: 9 }));

    assert!(b.get(ka).is_none());
    assert!(!b.contains_key(ka));
    assert!(b.remove(ka).is_none());
    assert!(b.downcast_key::<Dog>(ka.upcast()).is_none());

    // Original still works.
    assert_eq!(a.get(ka).unwrap().name, "A");
}

#[test]
fn cross_map_same_type_resolves() {
    let mut a: AnyMap = AnyMap::new();
    let mut b: AnyMap = AnyMap::new();

    let ka: CastKey<Dog> = a.insert_sized(CastBox::new(Dog { name: "A".into() }));
    let _kb: CastKey<Dog> = b.insert_sized(CastBox::new(Dog { name: "B".into() }));

    // Documented semantics: same slot, same version, same type — the foreign
    // key resolves (to *b*'s value). Safe, if surprising; keep keys with their
    // map if you need identity.
    assert_eq!(b.get(ka).unwrap().name, "B");
}

// ─── contains_key ────────────────────────────────────────────────────────────

#[test]
fn contains_key_tracks_liveness() {
    let mut map: AnyMap = AnyMap::new();
    let kx = map.insert(CastBox::new(Cat { lives: 9 }));
    let key = map.downcast_key::<Cat>(kx).unwrap();
    assert!(map.contains_key(key));
    let _ = map.remove(key);
    assert!(!map.contains_key(key));
}

// ─── iter / values / keys ────────────────────────────────────────────────────

#[test]
fn iter_and_values_and_keys() {
    let mut map: AnyMap = AnyMap::new();
    map.insert(CastBox::new(Dog { name: "a".into() }));
    map.insert(CastBox::new(Cat { lives: 1 }));
    map.insert(CastBox::new(Cat { lives: 2 }));

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
    map.insert(CastBox::new(Cat { lives: 1 }));
    map.insert(CastBox::new(Cat { lives: 1 }));

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
    map.insert(CastBox::new(Cat { lives: 1 }));
    map.insert(CastBox::new(Cat { lives: 9 }));
    map.insert(CastBox::new(Dog { name: "z".into() }));

    map.retain(|_k, v| v.downcast_ref::<Cat>().map_or(false, |c| c.lives > 5));
    assert_eq!(map.len(), 1);
    assert!(map.values().next().unwrap().is::<Cat>());
}

// ─── drain ───────────────────────────────────────────────────────────────────

#[test]
fn drain_empties_and_yields() {
    let mut map: AnyMap = AnyMap::new();
    map.insert(CastBox::new(Cat { lives: 3 }));
    map.insert(CastBox::new(Cat { lives: 4 }));

    let drained: Vec<CastBox<dyn Any>> = map.drain().map(|(_k, v)| v).collect();
    assert_eq!(drained.len(), 2);
    assert!(map.is_empty());
}

// ─── Index (sized output map) ────────────────────────────────────────────────
// `Index` requires the map's output type itself to be `AnyHaver`, which
// `dyn Any` is not — so indexing is exercised on a sized-output map.

#[test]
fn index_reads() {
    let mut map: BoxCastMap<DefaultKey, dyn Any> = BoxCastMap::new();
    let key: CastKey<u32> = map.insert_sized(CastBox::new(41u32));
    assert_eq!(*map.get(key).unwrap(), 41);
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
    // `CastBox<u32>` derefs to `u32`; the map is `Clone` iff `M` is, which a
    // `CastBox<dyn Any>` map is not — exactly as `dyn Any` is not `Clone`.
    // (This uses the unsafe map because `CastBox` itself is not `Clone`;
    // cloning of checked maps is exercised where the value type allows it.)
    let mut map: UnsafeBoxCastMap<DefaultKey, u32> = UnsafeBoxCastMap::new();
    let key: CastKey<u32> = map.insert(Box::new(7u32));

    let clone = map.clone();
    // SAFETY: same slot layout in the clone; the key was minted for a `u32`
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
        let boxed: CastBox<dyn Any> = CastBox::new(123u32);
        boxed
    });
    assert!(captured.is_some());

    let key = map.downcast_key::<u32>(dyn_key).unwrap();
    assert_eq!(*map.get(key).unwrap(), 123);
}

#[test]
fn insert_sized_with_key_threads_typed_key() {
    let mut map: AnyMap = AnyMap::new();
    let mut captured: Option<CastKey<Dog>> = None;
    let key = map.insert_sized_with_key(|k| {
        captured = Some(k);
        CastBox::new(Dog { name: "WK".into() })
    });
    assert_eq!(captured.unwrap(), key);
    assert_eq!(map.get(key).unwrap().name, "WK");
}

// ─── DynKey: round-trip + dyn dispatch through the key's metadata ────────────

trait Pet: AnyHaver {
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
fn dyn_key_round_trips() {
    let mut map: AnyMap = AnyMap::new();
    let key: CastKey<Dog> = map.insert_sized(CastBox::new(Dog { name: "RT".into() }));

    // Sized target: metadata is `()`, address packs the KeyData.
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
fn dyn_key_dispatches_virtually() {
    let mut map: AnyMap = AnyMap::new();
    let dog: CastKey<Dog> = map.insert_sized(CastBox::new(Dog { name: "Rex".into() }));
    let cat: CastKey<Cat> = map.insert_sized(CastBox::new(Cat { lives: 9 }));

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
    let k1: CastKey<u32> = map.insert(CastBox::new(10u32));
    let k2: CastKey<u32> = map.insert(CastBox::new(20u32));

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
    // SAFETY: key was just minted for this slot; metadata (the length) is valid.
    assert_eq!(unsafe { map.get(key) }.unwrap(), &[1, 2, 3]);

    // SAFETY: same key, slot unchanged.
    unsafe { map.get_mut(key) }.unwrap()[0] = 9;
    assert_eq!(unsafe { map.get(key) }.unwrap(), &[9, 2, 3]);
}

// ─── owning IntoIterator yields the cast keys + values ───────────────────────

#[test]
fn owned_into_iterator_yields_all() {
    let mut map: AnyMap = AnyMap::new();
    map.insert(CastBox::new(Cat { lives: 1 }));
    map.insert(CastBox::new(Cat { lives: 2 }));

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
    let key: CastKey<u32> = map.insert(CastBox::new(5u32));
    map[key] += 10;
    assert_eq!(map[key], 15);
}

// ─── UnsafeCastMap direct usage ──────────────────────────────────────────────

#[test]
fn unsafe_map_typed_roundtrip() {
    let mut map: UnsafeBoxCastMap<DefaultKey, dyn Any> = UnsafeBoxCastMap::new();
    let dyn_key = map.insert(Box::new(Dog { name: "U".into() }));
    // SAFETY: just-inserted key; the stored value really is a `Dog`.
    let key = unsafe { map.downcast_key::<Dog>(dyn_key) }.unwrap();

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
    let dyn_key = map.insert(Box::new(Dog { name: "Rex".into() }));
    // SAFETY: just-inserted key; the stored value really is a `Dog`.
    let key = unsafe { map.downcast_key::<Dog>(dyn_key) }.unwrap();

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

// ─────────────────────────────────────────────────────────────────────────────
// Dense variants: identical behaviour, backed by DenseSlotMap.
// ─────────────────────────────────────────────────────────────────────────────

mod dense {
    use std::any::Any;

    use super::{Cat, Dog};
    use crate::cast_box::CastBox;
    use crate::cast_key::CastKey;
    use crate::cast_map::BoxDenseCastMap;
    use crate::unsafe_cast_map::UnsafeBoxDenseCastMap;
    use crate::DefaultKey;

    type AnyMap = BoxDenseCastMap<DefaultKey, dyn Any>;

    #[test]
    fn insert_then_downcast_typed_get() {
        let mut map: AnyMap = AnyMap::new();
        let dyn_key = map.insert(CastBox::new(Dog { name: "Rex".into() }));
        let key = map.downcast_key::<Dog>(dyn_key).unwrap();
        assert_eq!(map.get(key).unwrap().name, "Rex");
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn typed_get_and_downcast_key() {
        let mut map: AnyMap = AnyMap::new();
        let key: CastKey<Dog> = map.insert_sized(CastBox::new(Dog { name: "Fido".into() }));
        assert_eq!(map.get(key).unwrap().name, "Fido");

        let dyn_key: CastKey<dyn Any> = key.upcast();
        let recovered: CastKey<Dog> = map.downcast_key::<Dog>(dyn_key).unwrap();
        assert_eq!(map.get(recovered).unwrap().name, "Fido");
        assert!(map.downcast_key::<Cat>(dyn_key).is_none());
    }

    #[test]
    fn get_mut_then_remove_invalidates() {
        let mut map: AnyMap = AnyMap::new();
        let kx = map.insert(CastBox::new(Cat { lives: 9 }));
        let key = map.downcast_key::<Cat>(kx).unwrap();
        map.get_mut(key).unwrap().lives -= 1;
        assert_eq!(map.get(key).unwrap().lives, 8);

        let removed: CastBox<Cat> = map.remove(key).unwrap();
        assert_eq!(removed.lives, 8);
        assert!(map.get(key).is_none()); // stale after remove
        assert!(map.is_empty());
    }

    #[test]
    fn cross_map_wrong_type_is_rejected() {
        let mut a: AnyMap = AnyMap::new();
        let mut b: AnyMap = AnyMap::new();

        let ka: CastKey<Dog> = a.insert_sized(CastBox::new(Dog { name: "A".into() }));
        let _kb: CastKey<Cat> = b.insert_sized(CastBox::new(Cat { lives: 1 }));

        assert!(b.get(ka).is_none());
        assert!(!b.contains_key(ka));
        assert_eq!(a.get(ka).unwrap().name, "A");
    }

    #[test]
    fn iter_values_keys_retain() {
        let mut map: AnyMap = AnyMap::new();
        map.insert(CastBox::new(Dog { name: "a".into() }));
        map.insert(CastBox::new(Cat { lives: 1 }));
        map.insert(CastBox::new(Cat { lives: 9 }));

        assert_eq!(map.iter().count(), 3);
        assert_eq!(map.values().count(), 3);
        assert_eq!(map.keys().count(), 3);

        map.retain(|_k, v| v.downcast_ref::<Cat>().map_or(false, |c| c.lives > 5));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn unsafe_dense_map_roundtrip() {
        let mut map: UnsafeBoxDenseCastMap<DefaultKey, dyn Any> = UnsafeBoxDenseCastMap::new();
        let dyn_key = map.insert(Box::new(Dog { name: "U".into() }));
        // SAFETY: just-inserted key; the stored value really is a `Dog`.
        let key = unsafe { map.downcast_key::<Dog>(dyn_key) }.unwrap();

        // SAFETY: the slot still holds the `Dog` `key` was made for, so its metadata is valid.
        let d: &Dog = unsafe { map.get(key).unwrap() };
        assert_eq!(d.name, "U");

        // SAFETY: same key, still valid.
        let removed: Box<Dog> = unsafe { map.remove(key).unwrap() };
        assert_eq!(removed.name, "U");
        assert!(map.is_empty());
    }

    #[test]
    fn get_disjoint_mut_typed() {
        let mut map: BoxDenseCastMap<DefaultKey, u32> = BoxDenseCastMap::new();
        let k1: CastKey<u32> = map.insert(CastBox::new(1u32));
        let k2: CastKey<u32> = map.insert(CastBox::new(2u32));
        let k3: CastKey<u32> = map.insert(CastBox::new(3u32));

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

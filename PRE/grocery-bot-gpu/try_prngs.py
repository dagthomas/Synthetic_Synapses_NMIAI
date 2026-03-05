#!/usr/bin/env python3
"""Try various PRNG algorithms to crack orders from map_id."""
import random, hashlib, uuid

map_id = '05ddc283-9097-4314-824c-90b3269a3d95'
map_seed = 7003
types = ['bread','milk','cereal','rice','yogurt','eggs','cheese','cream','oats','pasta','butter','flour']
captured = [
    ['butter','pasta','yogurt'],
    ['milk','cheese','butter','cream'],
    ['butter','eggs','cereal'],
    ['eggs','butter','oats','flour'],
    ['flour','cheese','milk'],
]

def gen_py(seed_val, count=5):
    rng = random.Random(seed_val)
    orders = []
    for _ in range(count):
        n = rng.randint(3, 5)
        orders.append([rng.choice(types) for _ in range(n)])
    return orders

def match(g, c):
    for i in range(min(len(g), len(c))):
        if g[i] != c[i]:
            return i
    return min(len(g), len(c))

u = uuid.UUID(map_id)

# === Python random with various seeds ===
print("=== Python random ===")
string_seeds = {
    'map_id': map_id,
    'no_hyphens': map_id.replace('-', ''),
    'map_seed_str': str(map_seed),
    'id:seed': f'{map_id}:{map_seed}',
    'seed:id': f'{map_seed}:{map_id}',
    'hard:id': f'hard:{map_id}',
    'id:hard': f'{map_id}:hard',
    'id:hard:seed': f'{map_id}:hard:{map_seed}',
    'bytes': u.bytes,
    'bytes_le': u.bytes_le,
    'uuid_int': u.int,
    'lower64': u.int & 0xFFFFFFFFFFFFFFFF,
    'upper64': u.int >> 64,
    'lower32': u.int & 0xFFFFFFFF,
    'time_low': u.time_low,
    'map_seed_int': map_seed,
}
for label, s in string_seeds.items():
    orders = gen_py(s)
    m = match(orders, captured)
    flag = f" *** MATCH={m} ***" if m > 0 else ""
    print(f"  {label:20s}: {orders[0]}{flag}")

# === XorShift128+ ===
print("\n=== XorShift128+ ===")
class XS128:
    def __init__(self, s0, s1):
        self.s0 = s0 & 0xFFFFFFFFFFFFFFFF
        self.s1 = s1 & 0xFFFFFFFFFFFFFFFF
    def nxt(self):
        s1 = self.s0
        s0 = self.s1
        self.s0 = s0
        s1 ^= (s1 << 23) & 0xFFFFFFFFFFFFFFFF
        s1 ^= s1 >> 17
        s1 ^= s0
        s1 ^= s0 >> 26
        self.s1 = s1
        return (self.s0 + self.s1) & 0xFFFFFFFFFFFFFFFF

s0 = u.int >> 64
s1 = u.int & 0xFFFFFFFFFFFFFFFF
for label, a, b in [('hi/lo', s0, s1), ('lo/hi', s1, s0), ('seed/0', map_seed, 0)]:
    rng = XS128(a, b)
    orders = []
    for _ in range(5):
        n = 3 + int(rng.nxt() % 3)
        orders.append([types[int(rng.nxt() % len(types))] for _ in range(n)])
    m = match(orders, captured)
    flag = f" *** MATCH={m} ***" if m > 0 else ""
    print(f"  {label:20s}: {orders[0]}{flag}")

# === Mulberry32 ===
print("\n=== Mulberry32 ===")
def mb32(seed):
    st = [seed & 0xFFFFFFFF]
    def n():
        st[0] = (st[0] + 0x6D2B79F5) & 0xFFFFFFFF
        t = st[0]
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t = (t ^ ((t ^ (t >> 7)) * (t | 61))) & 0xFFFFFFFF
        return (t ^ (t >> 14)) & 0xFFFFFFFF
    return n

for label, seed in [('lower32', u.int & 0xFFFFFFFF), ('upper32', u.int >> 96),
                     ('map_seed', map_seed), ('time_low', u.time_low)]:
    rng_fn = mb32(seed)
    orders = []
    for _ in range(5):
        n = 3 + int(rng_fn() % 3)
        orders.append([types[int(rng_fn() % len(types))] for _ in range(n)])
    m = match(orders, captured)
    flag = f" *** MATCH={m} ***" if m > 0 else ""
    print(f"  {label:20s}: {orders[0]}{flag}")

# === SplitMix64 ===
print("\n=== SplitMix64 ===")
def sm64(seed):
    st = [seed & 0xFFFFFFFFFFFFFFFF]
    def n():
        st[0] = (st[0] + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = st[0]
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF
    return n

for label, seed in [('uuid_int', u.int), ('lower64', u.int & 0xFFFFFFFFFFFFFFFF),
                     ('upper64', u.int >> 64), ('map_seed', map_seed)]:
    rng_fn = sm64(seed)
    orders = []
    for _ in range(5):
        n = 3 + int(rng_fn() % 3)
        orders.append([types[int(rng_fn() % len(types))] for _ in range(n)])
    m = match(orders, captured)
    flag = f" *** MATCH={m} ***" if m > 0 else ""
    print(f"  {label:20s}: {orders[0]}{flag}")

# === Java LCG ===
print("\n=== Java LCG ===")
class JavaRandom:
    def __init__(self, seed):
        self.seed = (seed ^ 0x5DEECE66D) & ((1 << 48) - 1)
    def _next(self, bits):
        self.seed = (self.seed * 0x5DEECE66D + 0xB) & ((1 << 48) - 1)
        return self.seed >> (48 - bits)
    def nextInt(self, bound):
        if (bound & (bound - 1)) == 0:
            return (bound * self._next(31)) >> 31
        while True:
            bits = self._next(31)
            val = bits % bound
            if bits - val + (bound - 1) >= 0:
                return val

for label, seed in [('lower32', u.int & 0xFFFFFFFF), ('map_seed', map_seed),
                     ('lower64', u.int & 0xFFFFFFFFFFFFFFFF), ('time_low', u.time_low)]:
    rng = JavaRandom(seed)
    orders = []
    for _ in range(5):
        n = 3 + rng.nextInt(3)
        orders.append([types[rng.nextInt(len(types))] for _ in range(n)])
    m = match(orders, captured)
    flag = f" *** MATCH={m} ***" if m > 0 else ""
    print(f"  {label:20s}: {orders[0]}{flag}")

# === Hash-per-order ===
print("\n=== Hash-per-order (SHA256) ===")
for template in ['{map_id}:{i}', '{i}:{map_id}', '{map_id}:{map_seed}:{i}',
                  '{map_seed}:{map_id}:{i}', 'order:{map_id}:{i}']:
    orders = []
    for i in range(5):
        key = template.format(map_id=map_id, map_seed=map_seed, i=i)
        h = hashlib.sha256(key.encode()).digest()
        n = 3 + h[0] % 3
        items = [types[h[j+1] % len(types)] for j in range(n)]
        orders.append(items)
    m = match(orders, captured)
    flag = f" *** MATCH={m} ***" if m > 0 else ""
    print(f"  {template:30s}: {orders[0]}{flag}")

# === Try with different type orderings ===
print("\n=== Alternate type orderings (Python random, map_id string seed) ===")
alt_types_lists = {
    'sorted': sorted(types),
    'reverse': list(reversed(types)),
    'alpha_orig': ['bread','butter','cereal','cheese','cream','eggs','flour','milk','oats','pasta','rice','yogurt'],
}
for tname, tlist in alt_types_lists.items():
    orders = gen_py(map_id)  # but with alt types we need to regenerate
    rng = random.Random(map_id)
    orders = []
    for _ in range(5):
        n = rng.randint(3, 5)
        orders.append([tlist[rng.randrange(len(tlist))] for _ in range(n)])
    # Wait, that changes the RNG calls. Need to use choice.
    rng = random.Random(map_id)
    orders = []
    for _ in range(5):
        n = rng.randint(3, 5)
        orders.append([rng.choice(tlist) for _ in range(n)])
    m = match(orders, captured)
    flag = f" *** MATCH={m} ***" if m > 0 else ""
    print(f"  {tname:20s}: {orders[0]}{flag}")

print(f"\nTarget order 0: {captured[0]}")
print("Done.")

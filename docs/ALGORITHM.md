# Kernel Algorithm Overview

## What the Kernel Does

The kernel performs a **parallel tree traversal** on a batch of 256 elements, repeated for 16 rounds.

### Data Structures

**Tree** (implicit binary tree):
- Perfect balanced binary tree with height 10
- `n_nodes = 2^11 - 1 = 2047` nodes
- Each node has a random 32-bit value
- Child indices: `left = 2*idx + 1`, `right = 2*idx + 2`

**Batch**:
- 256 elements, each with:
  - `idx`: current position in tree (starts at 0 = root)
  - `val`: accumulated value (random initial)

### Per-Element Logic

```python
for round in range(16):
    for i in range(256):
        idx = indices[i]
        val = values[i]

        # 1. XOR with tree node
        node_val = tree[idx]
        val = val ^ node_val

        # 2. Hash (6 stages, see below)
        val = myhash(val)

        # 3. Compute next index
        if val % 2 == 0:
            idx = 2 * idx + 1  # go left
        else:
            idx = 2 * idx + 2  # go right

        # 4. Bounds check (wrap to root if past leaves)
        if idx >= n_nodes:
            idx = 0

        # 5. Store results
        indices[i] = idx
        values[i] = val
```

### Hash Function

6-stage hash, each stage:
```python
HASH_STAGES = [
    ("+", 0x7ED55D16, "+", "<<", 12),  # val = (val + const1) + (val << 12)
    ("^", 0xC761C23C, "^", ">>", 19),  # val = (val ^ const2) ^ (val >> 19)
    ("+", 0x165667B1, "+", "<<", 5),   # val = (val + const3) + (val << 5)
    ("+", 0xD3A2646C, "^", "<<", 9),   # val = (val + const4) ^ (val << 9)
    ("+", 0xFD7046C5, "+", "<<", 3),   # val = (val + const5) + (val << 3)
    ("^", 0xB55A4F09, "^", ">>", 16),  # val = (val ^ const6) ^ (val >> 16)
]

for op1, const1, op2, op3, const2 in HASH_STAGES:
    tmp1 = op1(val, const1)   # e.g., val + 0x7ED55D16
    tmp2 = op3(val, const2)   # e.g., val << 12
    val = op2(tmp1, tmp2)     # e.g., tmp1 + tmp2
```

**Key observation**: Within each stage, `tmp1` and `tmp2` can be computed in parallel, but `val` depends on both.

---

## Vectorization Strategy

Process 8 elements at once (VLEN=8):
- 256 elements ÷ 8 = 32 "chunks" per round
- 16 rounds × 32 chunks = 512 iterations total

### Memory Access Pattern

**Contiguous** (can use vload/vstore):
- Load 8 consecutive `idx` values
- Load 8 consecutive `val` values
- Store 8 consecutive results

**Non-contiguous** (gather required):
- Tree access: `tree[idx[0]], tree[idx[1]], ..., tree[idx[7]]`
- Each `idx` points to a different tree location
- Must use 8 scalar loads (gather pattern)

### Gather Pattern

```python
# Compute 8 addresses (can be parallel)
for i in range(8):
    gather_addr[i] = forest_values_p + idx[i]

# Load 8 values (2 loads per cycle, so 4 cycles minimum)
for i in range(8):
    node_val[i] = mem[gather_addr[i]]
```

---

## Pipeline Stages

Each chunk iteration has these stages:

| Stage | Operations | Engine Usage |
|-------|------------|--------------|
| 1. Address | Compute memory addresses | ALU ×2 |
| 2. VLoad | Load idx, val vectors | LOAD ×2 |
| 3. Gather Addr | Compute 8 gather addresses | ALU ×8 |
| 4. Gather | 8 scalar loads | LOAD ×2 × 4 cycles |
| 5. XOR | val ^= node_val | VALU ×1 |
| 6. Hash | 6 stages × 3 ops | VALU ×18 |
| 7. Index | Compute next index | VALU ×6 |
| 8. Store | Write idx, val | STORE ×2 |

---

## Why It's Hard to Optimize

1. **Gather is slow**: 8 loads at 2/cycle = 4 cycles minimum
2. **Hash has dependencies**: Each stage depends on previous
3. **Round-to-round dependency**: Can't process round N+1 until round N's stores complete
4. **Memory bandwidth**: Lots of load/store operations

### Theoretical Minimum

If we could perfectly overlap everything:
- 512 iterations × 8 gather loads = 4096 load ops
- 4096 ÷ 2 loads/cycle = **2048 cycles** (gather-limited)

But we also need loads for idx/val, and stores, so realistic minimum is higher.

---

## Current Approach

**Pair processing with quad buffering**:
1. Process chunks in pairs (A+B or C+D)
2. Interleave their hash computations
3. While hashing pair N, load pair N+1
4. Alternate between buffer pairs

This achieves **5,723 cycles** (2.8x above theoretical gather minimum).

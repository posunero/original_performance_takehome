# Optimization Status and Next Steps

## Current Performance

| Metric | Value |
|--------|-------|
| **Current cycles** | 5,296 |
| **Baseline** | 147,734 |
| **Speedup** | 27.9x |
| **Target** | < 2,164 |

## Problem Summary

The kernel performs a parallel tree traversal:
1. Load index and value for each batch element
2. Gather tree node value at that index
3. XOR value with node value
4. Hash the result (6 stages)
5. Compute next index: `idx = 2*idx + (1 if val%2==0 else 2)`
6. Clamp index to bounds
7. Store updated index and value
8. Repeat for 16 rounds × 32 chunks = 512 iterations

**Workload**: 256 batch elements, 16 rounds, VLEN=8 → 32 chunks per round

---

## Implemented Optimizations

### 1. Basic Vectorization (147,734 → ~50,000 cycles)
- Process 8 elements per iteration using VLEN=8
- Use vload/vstore instead of scalar load/store
- Broadcast constants to vectors

### 2. VLIW Packing (→ ~20,000 cycles)
- Automatic dependency-aware instruction bundling
- Pack independent operations into same cycle
- Maximize slot utilization

### 3. Gather Pattern (~15,000 cycles)
- Tree access requires non-contiguous loads (gather)
- Compute 8 addresses, then 8 scalar loads
- 8 scalar loads = 4 cycles minimum (2 load slots)

### 4. vselect Elimination (→ 9,301 cycles)
Original:
```python
# if val%2==0: add=1, else add=2
ops.append(("valu", ("%", v_tmp1, v_val, v_two)))
ops.append(("valu", ("==", v_tmp1, v_tmp1, v_zero)))
ops.append(("flow", ("vselect", v_tmp3, v_tmp1, v_one, v_two)))  # SLOW!
```

Optimized:
```python
# add = 1 + (val & 1)  → 1 when even (val&1=0), 2 when odd (val&1=1)
ops.append(("valu", ("&", v_tmp1, v_val, v_one)))   # val & 1
ops.append(("valu", ("+", v_tmp1, v_tmp1, v_one)))  # 1 + (val & 1)
```

Also for bounds check:
```python
# Original: vselect(idx < n_nodes, idx, 0)
# Optimized: idx * (idx < n_nodes)
ops.append(("valu", ("<", v_tmp1, v_idx, v_n_nodes)))
ops.append(("valu", ("*", v_idx, v_idx, v_tmp1)))
```

### 5. Pair-Based Hash Interleaving (→ 6,998 cycles)
Hash has 6 stages, each with internal dependency:
```
Stage S: op1 = val OP1 const1  (parallel with op2)
         op2 = val OP3 const2  (parallel with op1)
         val = op1 OP2 op2     (depends on op1, op2)
```

Interleave two chunks' hash stages:
```
Cycle 1: A.s0.op1, A.s0.op2, B.s0.op1, B.s0.op2  (4 VALU ops)
Cycle 2: A.s0.final, B.s0.final                   (2 VALU ops)
Cycle 3: A.s1.op1, A.s1.op2, B.s1.op1, B.s1.op2  (4 VALU ops)
...
```

### 6. Quad Buffering with Load Overlap (→ 5,723 cycles)
Use 4 buffer sets (A, B, C, D) to overlap:
- While computing hash for pair (A, B)
- Load data for next pair (C, D)

### 7. Triple Processing with 6 Buffers (→ 5,296 cycles)
Upgraded from pair (2 chunks) to triple (3 chunks) processing:
- Use 6 buffer sets (A, B, C, D, E, F)
- Hash parallel ops now use all 6 valu slots (3 × 2 = 6)
- Hash final ops use 3 valu slots

Current pipeline:
```
Triple 0 (A,B,C): XOR → Hash stages 0-5 → Index → Store
                  ↓ during hash stages ↓
                  Load triple 1 (D,E,F): vloads + gathers interleaved

Triple 1 (D,E,F): XOR → Hash stages 0-5 → Index → Store
                  ↓ during hash stages ↓
                  Load triple 0 (A,B,C): ...
```

### 8. Depth 0 Broadcast Optimization
For rounds where tree depth = 0 (rounds 0 and 11):
- All 256 indices are 0 (root node)
- Skip gather loads, just broadcast tree[0]
- Saves 8 gather loads per chunk for depth 0 rounds

---

## Current Code Structure

```python
def build_kernel(self, forest_height, n_nodes, batch_size, rounds):
    # 1. Allocate scratch for scalars, constants
    # 2. Allocate 6 buffers (A, B, C, D, E, F) each with:
    #    - v_idx, v_val, v_node_val (8 elements each)
    #    - v_tmp1, v_tmp2, v_tmp3 (8 elements each)
    #    - tmp_idx_addr, tmp_val_addr (scalars)
    #    - gather_addrs[8] (8 scalar addresses)
    # 3. Broadcast vector constants
    # 4. Prologue: Load first three chunks into A, B, C
    # 5. Main loop (512 iterations, processing triples):
    #    - XOR all 3 chunks (3 valu ops)
    #    - Start address computation for next triple
    #    - Hash stages 0-5:
    #      - Parallel ops for 3 chunks (6 valu - full utilization!)
    #      - Interleaved loads for next triple
    #      - Final ops for 3 chunks (3 valu)
    #    - Depth 0 broadcast (if applicable)
    #    - Index computation for 3 chunks
    #    - Stores for 3 chunks
```

---

## Bottleneck Analysis

### Current breakdown (per triple of chunks):
```
XOR + addr start:           ~1-2 cycles (3 valu ops)
Hash stages (6 × 3 chunks): ~12 cycles
  - Parallel: 6 valu (3 chunks × 2 ops) - full utilization!
  - Final: 3 valu (3 chunks × 1 op)
  - Includes overlapped loads for next triple
Index computation (×3):     ~5 cycles (18 valu, dependency-limited)
Stores (×3):                ~3 cycles (6 vstores ÷ 2 slots)
─────────────────────────────────────────
Total per triple:           ~21 cycles
× 171 triples:              ~3,591 cycles (theoretical)
+ prologue/edge cases:      ~1,700 cycles
─────────────────────────────────────────
Actual:                     ~5,296 cycles
```

### VALU Utilization (improved with triple processing)
```
                    Pair (v6)    Triple (v7)
Hash parallel:      4 valu       6 valu (100% utilization!)
Hash final:         2 valu       3 valu
Index cycle 1:      4 valu       6 valu (100% utilization!)
Index cycles 2-5:   2 valu       3 valu
```

### Load Operation Analysis
Per triple (3 chunks × 8 elements):
- 6 vloads (idx, val for 3 chunks) = 6 load ops
- 24 scalar gather loads = 24 load ops
- **Total: 30 load ops per triple**

171 triples × 30 = 5,130 load ops
At 2 loads/cycle: **2,565 cycles minimum just for loads**

Loads are overlapped with hash computation using otherwise-idle load slots.

---

## Next Optimizations to Explore

### Option A: Keep idx/val in Scratch Across Rounds

**Insight**: Currently we vload/vstore idx and val every iteration. If we keep all 256 idx and 256 val values in scratch:
- Load all at start (64 vloads = 32 cycles)
- Process 16 rounds using scratch (no memory access for idx/val)
- Store all at end (64 vstores = 32 cycles)

**Savings**: Eliminate 512 iterations × (2 vloads + 2 vstores) = 2048 memory ops
**Potential**: ~1000 cycle reduction

**Challenge**: Need 512 scratch words for idx+val, plus existing allocations

### Option B: Loop with Jump Instructions

**Current**: Fully unrolled loop (512 iterations → huge instruction count)
**Alternative**: Use `cond_jump_rel` to create actual loop

```python
# Loop header
loop_start = len(instrs)

# ... loop body ...

# Decrement counter and jump
instrs.append(("alu", ("-", counter, counter, one_const)))
instrs.append(("flow", ("cond_jump_rel", counter, -(len(instrs) - loop_start))))
```

**Benefit**: Reduces instruction memory, enables optimizations impossible with unrolling

### Option C: Deeper Pipeline with Triple Buffering

Process 3 chunks simultaneously:
- Chunk N-2: Finishing index+stores
- Chunk N-1: Hash computation
- Chunk N: Loading

**Challenge**: Register pressure, complexity

### Option D: Exploit Hash Parallelism Better

Hash stage final ops only use 2 VALU slots. Could interleave:
- Hash final ops for pair N
- Hash parallel ops for pair N+1
- Index ops for pair N-1

---

## Scratch Memory Budget

| Allocation | Size | Purpose |
|------------|------|---------|
| Scalar temps | 3 | tmp1, tmp2, tmp3 |
| Init vars | 7 | rounds, n_nodes, etc. |
| Scalar consts | ~10 | 0, 1, 2, base_i values |
| Quad buffers | 4 × 51 = 204 | v_idx(8), v_val(8), v_node_val(8), v_tmp1-3(24), addrs(11) |
| Vector consts | 4 × 8 = 32 | v_zero, v_one, v_two, v_n_nodes |
| Hash consts | ~6 × 8 = 48 | v_hash constants |
| **Total** | ~304 | Out of 1536 available |

**Available for expansion**: ~1200 scratch words

This is enough for Option A (keep all idx/val in scratch: 512 words)

---

## Test Commands

```bash
# Quick correctness test
python -m pytest tests/submission_tests.py::CorrectnessTests -v

# Performance test
python -m pytest tests/submission_tests.py::SpeedTests -v

# Debug with trace
python perf_takehome.py Tests.test_kernel_trace
# Then open trace.json in https://ui.perfetto.dev/
```

---

## Target Thresholds

| Target | Cycles | Status |
|--------|--------|--------|
| Baseline | 147,734 | ✅ PASSED (25.8x better) |
| Updated starting | < 18,532 | ✅ PASSED |
| Opus 4 many hours | < 2,164 | ❌ Need ~2.6x more improvement |
| Best human 2hr | < 1,790 | ❌ Need ~3.2x more |
| Opus 4.5 improved | < 1,363 | ❌ Need ~4.2x more |

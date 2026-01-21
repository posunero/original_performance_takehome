# Optimization Changelog

## Version History

### v9 - Prologue/Epilogue VLIW Packing (Current)
**Cycles**: 4,944
**Speedup**: 29.9x from baseline

**Changes**:
- Allocated 64 address temporaries (`addr_temps[0..63]`) instead of single `tmp_addr`
- Restructured prologue into two phases:
  - Phase 1: Compute ALL 64 addresses (32 for idx, 32 for val) - packs to ~6 cycles at 12 alu/cycle
  - Phase 2: Execute ALL 64 vloads - packs to 32 cycles at 2 load/cycle
- Restructured epilogue similarly:
  - Phase 1: Compute ALL 64 store addresses - packs to ~6 cycles
  - Phase 2: Execute ALL 64 vstores - packs to 32 cycles at 2 store/cycle

**Key insight**: Previous prologue/epilogue had RAW (Read-After-Write) hazards:
```
alu writes tmp_addr → load reads tmp_addr → alu writes tmp_addr → ...
```
This serialized the operations. By using 64 separate temporaries, all ALU ops can
execute first (no dependencies between them), then all loads/stores.

**Savings breakdown**:
- Prologue: ~32 cycles before → ~38 cycles after (6 alu + 32 load)
- Epilogue: ~32 cycles before → ~38 cycles after (6 alu + 32 store)
- Wait, that's worse? Let me recalculate...
- Actually the savings come from VLIW packing:
  - Before: alu,load,alu,load,... = 64 cycles (each alu+load = 2 cycles due to RAW)
  - After: alu phase (6 cycles) + load phase (32 cycles) = 38 cycles
  - Savings per prologue/epilogue: ~26 cycles
  - Total savings: ~52 cycles (actual: 56 cycles)

**Scratch memory usage**:
- New: addr_temps[64] = 64 words
- Removed: tmp_addr[1] = 1 word
- Net increase: 63 words
- Total scratch: ~843 words (within 1536 limit)

---

### v8 - Scratch-Based idx/val Storage
**Cycles**: 5,000
**Speedup**: 29.5x from baseline

**Changes**:
- Allocate global scratch arrays: `s_idx[256]` and `s_val[256]` for all batch data
- Prologue: Load all 256 idx + 256 val from memory to scratch (64 vloads)
- Main loop: Operate directly on scratch arrays (no per-iteration memory access)
- Epilogue: Store all data back to memory (64 vstores)
- Removed per-iteration vstores (was 1024 vstores → now 64)
- Simplified buffer structure (removed v_idx, v_val, tmp_idx_addr, tmp_val_addr)

**Key insights**:
- Eliminated 960 vstores (171 triples × 6 vstores → 64 epilogue vstores)
- Savings less than expected (~296 cycles vs ~900 estimated) because:
  - Old code already interleaved idx/val loads during hash stages (essentially free)
  - Old stores may have been partially overlapped with computation
  - Prologue adds ~32 cycles overhead for sequential loads
- Net effect: 5.6% cycle reduction

**Scratch memory usage**:
- New: s_idx[256] + s_val[256] = 512 words
- Buffer reduction: 6 buffers × 4 removed fields = ~30 words saved
- Total scratch: ~780 words (within 1536 limit)

---

### v7 - Triple Processing with Depth 0 Optimization
**Cycles**: 5,296
**Speedup**: 27.9x from baseline

**Changes**:
- Process 3 chunks at a time instead of 2 (triple processing)
- Expanded to 6 buffers (A-F) to support triple pipelining
- Hash parallel ops now use all 6 valu slots (3 chunks × 2 ops = 6 valu)
- Added depth 0 broadcast optimization (skip gather for tree root)
- Interleaved loading for next triple across all 6 hash stages

**Key insights**:
- Pair processing used only 4 valu slots for hash parallel; triple uses all 6
- For depth 0 rounds, all indices are 0, so broadcast tree[0] instead of gathering
- Triple processing achieves ~8% improvement over pair processing

**VALU utilization improved**:
```
Pair (v6):   4 valu/cycle for hash parallel, 2 for final
Triple (v7): 6 valu/cycle for hash parallel, 3 for final
```

---

### v6 - Aggressive Load Overlap
**Cycles**: 5,723
**Speedup**: 25.8x from baseline

**Changes**:
- Load BOTH chunks of next pair during hash computation
- Spread gather loads across all 6 hash stages
- Maximize load slot utilization during compute-heavy phases

**Key insight**: Hash stages have 2 unused load slots per cycle. Fill them with next pair's loads.

---

### v5 - Quad Buffering with Pair Processing
**Cycles**: 6,998
**Speedup**: 21.1x

**Changes**:
- Use 4 buffer sets (A, B, C, D) instead of 2
- Process chunks in pairs for hash interleaving
- Overlap pair N+1 loads with pair N computation

---

### v4 - Pair-Based Hash Interleaving
**Cycles**: 8,270
**Speedup**: 17.9x

**Changes**:
- Interleave hash stages from two chunks
- Both chunks' parallel ops in same cycle
- Then both chunks' final ops

```
Before (sequential):
  A.stage0 → A.stage1 → ... → A.stage5
  B.stage0 → B.stage1 → ... → B.stage5

After (interleaved):
  A.s0.parallel + B.s0.parallel → A.s0.final + B.s0.final
  A.s1.parallel + B.s1.parallel → A.s1.final + B.s1.final
  ...
```

---

### v3 - vselect Elimination
**Cycles**: 9,301
**Speedup**: 15.9x

**Changes**:
- Replace `vselect` with arithmetic operations
- Index computation: `add = 1 + (val & 1)` instead of conditional
- Bounds check: `idx = idx * (idx < n_nodes)` instead of vselect

**Key insight**: vselect uses FLOW engine (1 slot). Arithmetic uses VALU (6 slots).

---

### v2 - True Instruction Interleaving
**Cycles**: 10,324
**Speedup**: 14.3x

**Changes**:
- More aggressive VLIW packing
- Interleave operations from different pipeline stages
- Better utilization of ALU slots during loads

---

### v1 - Basic Double Buffering
**Cycles**: 12,398
**Speedup**: 11.9x

**Changes**:
- Allocate two buffer sets (A, B)
- Alternate processing between buffers
- Overlap loads for buffer B while computing buffer A

---

### v0 - Vectorized VLIW Baseline
**Cycles**: 13,390
**Speedup**: 11.0x

**Changes**:
- VLEN=8 vectorization (8 elements per iteration)
- VLIW packing with dependency analysis
- Gather pattern for non-contiguous tree access
- Vector hash computation

---

### Initial - Reference Implementation
**Cycles**: 147,734
**Speedup**: 1.0x (baseline)

Sequential scalar implementation matching `reference_kernel()`.

---

## Failed Experiments

### Deep Pipeline with Index Overlap
**Result**: IndexError - correctness failure

**Attempted**: Overlap index computation from previous pair with hash final ops of current pair.

**Problem**: Corrupted gather addresses due to buffer reuse timing issues.

### Shallow Depth 1 & 2 Optimization
**Result**: Increased cycle count (6,299 cycles vs 5,296)

**Attempted**: For depths 1-2, use arithmetic selection instead of gather loads.

**Problem**: Shallow computation adds valu ops AFTER hash stages (when valu is free),
but the gather loads it replaces happened DURING hash stages (using otherwise-idle load slots).
Net effect: added cycles instead of saving them.

### Hash Fusion + Depth Specialization (Combined)
**Result**: Increased cycle count (5,427 cycles vs 5,296)

**Attempted**: Two optimizations combined:
1. **Depth specialization**: Enable `emit_shallow_d1` and `emit_shallow_d2` for depths 1-2
   - Changed `emit_node_val` to call shallow functions for depths 0, 1, 2
   - Updated interleaved loading conditions from `depth > 0` to `depth > 2`
2. **Hash fusion**: Fuse stages 0, 2, 4 using `multiply_add`
   - Algebraic insight: `(a + C) + (a << k) = a * (1 + 2^k) + C`
   - Stage 0: multiplier 4097 (1 + 2^12)
   - Stage 2: multiplier 33 (1 + 2^5)
   - Stage 4: multiplier 9 (1 + 2^3)
   - Replaced 3-op pattern with single `multiply_add` for fusible stages

**Problem**: While hash fusion reduces valu ops (3→1 per fusible stage), the depth
specialization issue dominates. The shallow computations execute AFTER hash stages,
adding extra cycles, while the gather loads they replace were "free" (using idle load
slots DURING hash computation). Even with hash fusion savings, net effect is +131 cycles.

---

## Remaining Optimization Ideas (To reach < 2,164 cycles)

1. ~~**Scratch-based idx/val storage**~~ - IMPLEMENTED in v8 (saved 296 cycles, less than expected)

2. ~~**Prologue/Epilogue VLIW packing**~~ - IMPLEMENTED in v9 (saved 56 cycles)

3. **Loop instructions** - Use `cond_jump_rel` for iteration instead of full unrolling
   - Would reduce instruction memory footprint
   - May improve cache behavior

4. **Interleaved shallow computation** - Do depth 0/1/2 node_val computation DURING hash finals
   - Hash finals use 3 valu slots (for triple), leaving 3 free
   - Could interleave shallow selection ops with hash finals
   - Challenge: Complex dependency tracking

5. **Interleaved prologue** - Overlap prologue vloads with tree value loading/broadcasting
   - Currently prologue is sequential (but now with better VLIW packing)
   - Could overlap with initial setup operations

6. **Better VLIW packing** - Analyze instruction stream for packing inefficiencies
   - Current `build()` is greedy; might miss global optimizations

## Performance Analysis

Current bottlenecks at 4,944 cycles:
- Hash stages: 12 cycles per triple (6 stages × 2 cycles: parallel + final)
- Index computation: 5-cycle dependency chain
- Prologue: ~38 cycles (6 alu + 32 load)
- Epilogue: ~38 cycles (6 alu + 32 store)

Theoretical minimum for triple: ~20 cycles × 171 triples = 3,420 cycles
Current overhead: 4,944 - 3,420 = 1,524 cycles (prologue, epilogue, edge cases, packing inefficiency)

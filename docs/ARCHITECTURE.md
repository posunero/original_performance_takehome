# VLIW SIMD Architecture Reference

This document describes the custom VLIW SIMD architecture used in the performance optimization challenge.

## Overview

The architecture simulates a **VLIW (Very Large Instruction Word)** processor with **SIMD (Single Instruction Multiple Data)** capabilities. It features:

- Multiple parallel execution engines
- Vector operations on 8 elements (VLEN=8)
- Scratch memory (1536 words) acting as registers/cache
- Main memory for input/output data

## Execution Model

### VLIW Bundles

Each instruction is a "bundle" that can contain multiple operations across different engines. All operations in a bundle execute **in the same cycle**. Effects (writes to scratch/memory) take place **after** all inputs are read.

Example bundle:
```python
{"valu": [("*", 4, 0, 0), ("+", 8, 4, 0)], "load": [("load", 16, 17)]}
```

### Slot Limits

Each engine has a maximum number of operations per cycle:

| Engine | Slots/Cycle | Purpose |
|--------|-------------|---------|
| `alu`  | 12          | Scalar arithmetic |
| `valu` | 6           | Vector arithmetic |
| `load` | 2           | Memory → Scratch |
| `store`| 2           | Scratch → Memory |
| `flow` | 1           | Control flow, select |
| `debug`| 64          | Debug operations (free) |

### Dependencies

Within a bundle:
- Operations **cannot** read values written by other operations in the same bundle
- The VLIW packer automatically separates dependent operations into different cycles

---

## Instruction Set Reference

### ALU Engine (Scalar)

Format: `("op", dest, src1, src2)`

All operands are scratch addresses. Result written to `dest`.

| Op | Description | Formula |
|----|-------------|---------|
| `+` | Addition | `dest = src1 + src2` |
| `-` | Subtraction | `dest = src1 - src2` |
| `*` | Multiplication | `dest = src1 * src2` |
| `//` | Integer division | `dest = src1 // src2` |
| `cdiv` | Ceiling division | `dest = (src1 + src2 - 1) // src2` |
| `^` | XOR | `dest = src1 ^ src2` |
| `&` | AND | `dest = src1 & src2` |
| `\|` | OR | `dest = src1 \| src2` |
| `<<` | Left shift | `dest = src1 << src2` |
| `>>` | Right shift | `dest = src1 >> src2` |
| `%` | Modulo | `dest = src1 % src2` |
| `<` | Less than | `dest = 1 if src1 < src2 else 0` |
| `==` | Equality | `dest = 1 if src1 == src2 else 0` |

All results are masked to 32 bits (`% 2^32`).

---

### VALU Engine (Vector)

Operates on VLEN=8 consecutive scratch addresses.

#### Vector Binary Operations
Format: `("op", v_dest, v_src1, v_src2)`

Same operations as ALU, but applied element-wise:
```python
for i in range(8):
    scratch[v_dest + i] = op(scratch[v_src1 + i], scratch[v_src2 + i])
```

#### vbroadcast
Format: `("vbroadcast", v_dest, scalar_src)`

Broadcasts a scalar to all vector elements:
```python
for i in range(8):
    scratch[v_dest + i] = scratch[scalar_src]
```

#### multiply_add
Format: `("multiply_add", v_dest, v_a, v_b, v_c)`

Fused multiply-add:
```python
for i in range(8):
    scratch[v_dest + i] = (scratch[v_a + i] * scratch[v_b + i] + scratch[v_c + i]) % 2^32
```

---

### LOAD Engine

#### load
Format: `("load", dest, addr_scratch)`

Loads from memory at address stored in scratch:
```python
scratch[dest] = mem[scratch[addr_scratch]]
```

#### load_offset
Format: `("load_offset", dest, addr_scratch, offset)`

Loads with offset (useful for vector-like operations):
```python
scratch[dest + offset] = mem[scratch[addr_scratch + offset]]
```

#### vload
Format: `("vload", v_dest, addr_scratch)`

Vector load of 8 consecutive memory locations:
```python
addr = scratch[addr_scratch]
for i in range(8):
    scratch[v_dest + i] = mem[addr + i]
```

#### const
Format: `("const", dest, immediate_value)`

Loads an immediate constant into scratch:
```python
scratch[dest] = immediate_value % 2^32
```

---

### STORE Engine

#### store
Format: `("store", addr_scratch, src)`

Stores scratch value to memory:
```python
mem[scratch[addr_scratch]] = scratch[src]
```

#### vstore
Format: `("vstore", addr_scratch, v_src)`

Vector store of 8 consecutive values:
```python
addr = scratch[addr_scratch]
for i in range(8):
    mem[addr + i] = scratch[v_src + i]
```

---

### FLOW Engine

#### select
Format: `("select", dest, cond, src_true, src_false)`

Scalar conditional select:
```python
scratch[dest] = scratch[src_true] if scratch[cond] != 0 else scratch[src_false]
```

#### vselect
Format: `("vselect", v_dest, v_cond, v_true, v_false)`

Vector conditional select (element-wise):
```python
for i in range(8):
    scratch[v_dest + i] = scratch[v_true + i] if scratch[v_cond + i] != 0 else scratch[v_false + i]
```

**Note**: `vselect` is expensive - only 1 per cycle. Prefer arithmetic alternatives when possible.

#### add_imm
Format: `("add_imm", dest, src, immediate)`

Add immediate value:
```python
scratch[dest] = (scratch[src] + immediate) % 2^32
```

#### Control Flow

| Instruction | Format | Description |
|-------------|--------|-------------|
| `halt` | `("halt",)` | Stop execution |
| `pause` | `("pause",)` | Pause until next `run()` call |
| `jump` | `("jump", pc)` | Unconditional jump |
| `cond_jump` | `("cond_jump", cond, pc)` | Jump if `scratch[cond] != 0` |
| `cond_jump_rel` | `("cond_jump_rel", cond, offset)` | Relative jump if condition |
| `jump_indirect` | `("jump_indirect", addr)` | Jump to `scratch[addr]` |

#### Misc

| Instruction | Format | Description |
|-------------|--------|-------------|
| `coreid` | `("coreid", dest)` | Get core ID (always 0 in single-core) |
| `trace_write` | `("trace_write", val)` | Write to trace buffer |

---

## Memory Layout

The problem uses a flat memory image with this structure:

| Address | Content |
|---------|---------|
| 0 | `rounds` - number of iterations |
| 1 | `n_nodes` - tree size |
| 2 | `batch_size` - input batch size |
| 3 | `forest_height` - tree height |
| 4 | `forest_values_p` - pointer to tree values |
| 5 | `inp_indices_p` - pointer to input indices |
| 6 | `inp_values_p` - pointer to input values |
| 7+ | Tree values, then indices, then values |

---

## Scratch Space

- 1536 words total (SCRATCH_SIZE)
- Acts as registers and local cache
- Allocated sequentially via `alloc_scratch()`
- Named allocations tracked for debugging

---

## Performance Considerations

### Bottlenecks (in order of severity)

1. **LOAD engine (2 slots)**: Most limiting factor
   - vload counts as 1 slot (loads 8 values)
   - Scalar load counts as 1 slot (loads 1 value)
   - Gather pattern: 8 scalar loads = 4 cycles minimum

2. **FLOW engine (1 slot)**: vselect is expensive
   - Only 1 vselect per cycle
   - Use arithmetic alternatives: `result = value * condition`

3. **STORE engine (2 slots)**: Usually not limiting
   - vstore counts as 1 slot

4. **VALU engine (6 slots)**: Rarely limiting
   - Hash computation needs 3 ops/stage, 6 stages = 18 ops
   - But internal dependencies create serialization

5. **ALU engine (12 slots)**: Almost never limiting

### Key Optimizations

1. **Vectorization**: Process 8 elements per vload/vstore
2. **Software pipelining**: Overlap loads with computation
3. **Arithmetic conditionals**: Replace vselect with multiplication
4. **VLIW packing**: Maximize parallel operations per cycle

---

## Tracing

Enable tracing to generate `trace.json` for visualization in Perfetto:
```python
do_kernel_test(10, 16, 256, trace=True)
```

Open `trace.json` in https://ui.perfetto.dev/ to visualize:
- Each engine's slot utilization
- Instruction timing
- Scratch variable changes

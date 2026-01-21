"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def _get_slot_deps(self, engine: str, slot: tuple) -> tuple[set, set]:
        """Returns (reads, writes) - sets of scratch addresses read/written by this slot."""
        reads = set()
        writes = set()

        if engine == "alu":
            _, dest, a1, a2 = slot
            writes.add(dest)
            reads.add(a1)
            reads.add(a2)
        elif engine == "valu":
            if slot[0] == "vbroadcast":
                _, dest, src = slot
                for i in range(VLEN):
                    writes.add(dest + i)
                reads.add(src)
            elif slot[0] == "multiply_add":
                _, dest, a, b, c = slot
                for i in range(VLEN):
                    writes.add(dest + i)
                    reads.add(a + i)
                    reads.add(b + i)
                    reads.add(c + i)
            else:
                _, dest, a1, a2 = slot
                for i in range(VLEN):
                    writes.add(dest + i)
                    reads.add(a1 + i)
                    reads.add(a2 + i)
        elif engine == "load":
            if slot[0] == "load":
                _, dest, addr = slot
                writes.add(dest)
                reads.add(addr)
            elif slot[0] == "vload":
                _, dest, addr = slot
                for i in range(VLEN):
                    writes.add(dest + i)
                reads.add(addr)
            elif slot[0] == "const":
                _, dest, _ = slot
                writes.add(dest)
        elif engine == "store":
            if slot[0] == "store":
                _, addr, src = slot
                reads.add(addr)
                reads.add(src)
            elif slot[0] == "vstore":
                _, addr, src = slot
                reads.add(addr)
                for i in range(VLEN):
                    reads.add(src + i)
        elif engine == "flow":
            if slot[0] == "vselect":
                _, dest, cond, a, b = slot
                for i in range(VLEN):
                    writes.add(dest + i)
                    reads.add(cond + i)
                    reads.add(a + i)
                    reads.add(b + i)

        return reads, writes

    def build(self, slots: list[tuple[str, tuple]], vliw: bool = True):
        """Dependency-aware VLIW instruction packing."""
        if not vliw:
            return [{engine: [slot]} for engine, slot in slots]

        instrs = []
        i = 0
        while i < len(slots):
            bundle = defaultdict(list)
            bundle_reads = set()
            bundle_writes = set()

            j = i
            while j < len(slots):
                engine, slot = slots[j]
                if engine == "debug":
                    bundle[engine].append(slot)
                    j += 1
                    continue
                if len(bundle[engine]) >= SLOT_LIMITS.get(engine, 1):
                    break
                reads, writes = self._get_slot_deps(engine, slot)
                if reads & bundle_writes or writes & bundle_writes:
                    break
                bundle[engine].append(slot)
                bundle_reads |= reads
                bundle_writes |= writes
                j += 1

            if bundle:
                instrs.append(dict(bundle))
            i = j

        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Highly optimized kernel with:
        1. Vectorization (VLEN=8 elements per operation)
        2. VLIW pipelining (pack independent ops into same cycle)
        3. Quad buffering for deep pipelining
        4. Pair-based hash interleaving
        5. Aggressive load overlap during hash computation
        6. Arithmetic-based conditional elimination
        """
        # Scalar temporaries
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")

        # Memory layout info
        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        # Scalar constants
        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Global scratch arrays for ALL batch idx/val (eliminates per-iteration memory access)
        s_idx = self.alloc_scratch("s_idx", batch_size)  # 256 words for all batch indices
        s_val = self.alloc_scratch("s_val", batch_size)  # 256 words for all batch values

        # Six buffers for triple processing (3 chunks at a time)
        # Reduced buffer layout: (v_node_val, v_tmp1, v_tmp2, gather_addrs, v_tmp3)
        # idx/val now accessed directly from s_idx/s_val using chunk offset
        buffers = []
        for buf_name in ['A', 'B', 'C', 'D', 'E', 'F']:
            v_node_val = self.alloc_scratch(f"v_node_val_{buf_name}", VLEN)
            v_tmp1 = self.alloc_scratch(f"v_tmp1_{buf_name}", VLEN)
            v_tmp2 = self.alloc_scratch(f"v_tmp2_{buf_name}", VLEN)
            gather_addrs = [self.alloc_scratch(f"gather_addr_{buf_name}_{i}") for i in range(VLEN)]
            v_tmp3 = self.alloc_scratch(f"v_tmp3_{buf_name}", VLEN)  # Extra temp for depth 2
            buffers.append((v_node_val, v_tmp1, v_tmp2, gather_addrs, v_tmp3))

        buf_A, buf_B, buf_C, buf_D, buf_E, buf_F = buffers

        # Vector constants
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)

        self.add("valu", ("vbroadcast", v_zero, zero_const))
        self.add("valu", ("vbroadcast", v_one, one_const))
        self.add("valu", ("vbroadcast", v_two, two_const))
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))

        # Hash constants
        v_hash_const_map = {}
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            for val in [val1, val3]:
                if val not in v_hash_const_map:
                    v_addr = self.alloc_scratch(f"v_hash_{val}", VLEN)
                    scalar_addr = self.scratch_const(val)
                    self.add("valu", ("vbroadcast", v_addr, scalar_addr))
                    v_hash_const_map[val] = v_addr

        # Fused multipliers for hash stages 0, 2, 4: a = (a + C) + (a << k) = a * (1 + 2^k) + C
        FUSED_HASH_STAGES = {0: 4097, 2: 33, 4: 9}  # 1 + 2^12, 1 + 2^5, 1 + 2^3
        v_fused_mult = {}
        for stage, mult in FUSED_HASH_STAGES.items():
            v_addr = self.alloc_scratch(f"v_fused_mult_{stage}", VLEN)
            scalar_addr = self.scratch_const(mult)
            self.add("valu", ("vbroadcast", v_addr, scalar_addr))
            v_fused_mult[stage] = v_addr

        # Pre-compute base_i constants
        n_chunks = batch_size // VLEN
        base_i_consts = {}
        for chunk in range(n_chunks):
            base_i = chunk * VLEN
            base_i_consts[base_i] = self.scratch_const(base_i)

        # Shallow depth node values (tree[0..6] for depths 0, 1, 2)
        three_const = self.scratch_const(3)
        v_three = self.alloc_scratch("v_three", VLEN)
        self.add("valu", ("vbroadcast", v_three, three_const))

        scalar_node_vals = [self.alloc_scratch(f"sn_{i}") for i in range(7)]
        v_node_d1_1 = self.alloc_scratch("v_node_d1_1", VLEN)
        v_node_d1_2 = self.alloc_scratch("v_node_d1_2", VLEN)
        v_node_d2 = [self.alloc_scratch(f"v_node_d2_{i}", VLEN) for i in range(4)]

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting pipelined vectorized loop"))

        total_iterations = rounds * n_chunks
        instrs = []

        # Load shallow tree values in prologue
        for i in range(7):
            c = self.scratch_const(i) if i not in base_i_consts else base_i_consts[i]
            instrs.append(("alu", ("+", tmp1, self.scratch["forest_values_p"], c)))
            instrs.append(("load", ("load", scalar_node_vals[i], tmp1)))

        # Broadcast shallow node values
        instrs.append(("valu", ("vbroadcast", v_node_d1_1, scalar_node_vals[1])))
        instrs.append(("valu", ("vbroadcast", v_node_d1_2, scalar_node_vals[2])))
        for i in range(4):
            instrs.append(("valu", ("vbroadcast", v_node_d2[i], scalar_node_vals[3 + i])))

        def get_depth(iter_num):
            """Get tree depth for this iteration."""
            return (iter_num // n_chunks) % (forest_height + 1)

        def get_scratch_addrs(chunk):
            """Get scratch addresses for idx/val for a given chunk."""
            base_i = chunk * VLEN
            return s_idx + base_i, s_val + base_i

        def emit_gather_addrs(buf, chunk):
            v_node_val, v_tmp1, v_tmp2, gather_addrs, v_tmp3 = buf
            v_idx, v_val = get_scratch_addrs(chunk)
            ops = []
            for i in range(VLEN):
                ops.append(("alu", ("+", gather_addrs[i], self.scratch["forest_values_p"], v_idx + i)))
            return ops

        def emit_gather_loads(buf):
            v_node_val, v_tmp1, v_tmp2, gather_addrs, v_tmp3 = buf
            ops = []
            for i in range(VLEN):
                ops.append(("load", ("load", v_node_val + i, gather_addrs[i])))
            return ops

        def emit_shallow_d0(buf):
            """Depth 0: all idx=0, just broadcast tree[0]."""
            v_node_val, v_tmp1, v_tmp2, gather_addrs, v_tmp3 = buf
            return [("valu", ("vbroadcast", v_node_val, scalar_node_vals[0]))]

        def emit_shallow_d1(buf, chunk):
            """Depth 1: idx ∈ {1,2}, select tree[1] or tree[2]."""
            v_node_val, v_tmp1, v_tmp2, gather_addrs, v_tmp3 = buf
            v_idx, v_val = get_scratch_addrs(chunk)
            ops = []
            # (idx & 1) = 1 if idx==1, 0 if idx==2
            ops.append(("valu", ("&", v_tmp1, v_idx, v_one)))
            # node_val = tmp1 * tree[1] + (1-tmp1) * tree[2]
            ops.append(("valu", ("*", v_node_val, v_tmp1, v_node_d1_1)))
            ops.append(("valu", ("-", v_tmp2, v_one, v_tmp1)))
            ops.append(("valu", ("multiply_add", v_node_val, v_tmp2, v_node_d1_2, v_node_val)))
            return ops

        def emit_shallow_d2(buf, chunk):
            """Depth 2: idx ∈ {3,4,5,6}, select among tree[3..6]."""
            v_node_val, v_tmp1, v_tmp2, gather_addrs, v_tmp3 = buf
            v_idx, v_val = get_scratch_addrs(chunk)
            ops = []
            # selector = idx - 3 (0..3)
            ops.append(("valu", ("-", v_tmp1, v_idx, v_three)))
            # b0 = selector & 1, b1 = selector >> 1
            ops.append(("valu", ("&", v_tmp2, v_tmp1, v_one)))  # b0 in v_tmp2
            ops.append(("valu", (">>", v_tmp1, v_tmp1, v_one)))  # b1 in v_tmp1
            # Build using arithmetic selection:
            # val_low = tree[3]*(1-b0) + tree[4]*b0
            # val_high = tree[5]*(1-b0) + tree[6]*b0
            # result = val_low*(1-b1) + val_high*b1
            # Compute 1-b0 in v_node_val (temp)
            ops.append(("valu", ("-", v_node_val, v_one, v_tmp2)))  # 1-b0
            # val_low: (1-b0)*tree[3] + b0*tree[4]
            ops.append(("valu", ("*", v_node_val, v_node_val, v_node_d2[0])))  # (1-b0)*tree[3]
            ops.append(("valu", ("multiply_add", v_node_val, v_tmp2, v_node_d2[1], v_node_val)))  # + b0*tree[4] -> val_low
            # val_high in v_tmp3: (1-b0)*tree[5] + b0*tree[6]
            ops.append(("valu", ("-", v_tmp3, v_one, v_tmp2)))  # 1-b0
            ops.append(("valu", ("*", v_tmp3, v_tmp3, v_node_d2[2])))  # (1-b0)*tree[5]
            ops.append(("valu", ("multiply_add", v_tmp3, v_tmp2, v_node_d2[3], v_tmp3)))  # + b0*tree[6] -> val_high
            # Final blend: val_low*(1-b1) + b1*val_high
            ops.append(("valu", ("-", v_tmp2, v_one, v_tmp1)))  # 1-b1 (reuse v_tmp2, done with b0)
            ops.append(("valu", ("*", v_node_val, v_node_val, v_tmp2)))  # val_low*(1-b1)
            ops.append(("valu", ("multiply_add", v_node_val, v_tmp1, v_tmp3, v_node_val)))  # + b1*val_high
            return ops

        def emit_hash_stage_parallel_ops(buf, chunk):
            v_node_val, v_tmp1, v_tmp2, gather_addrs, v_tmp3 = buf
            v_idx, v_val = get_scratch_addrs(chunk)
            stages = []
            for stage, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                if stage in FUSED_HASH_STAGES:
                    # Fused: single multiply_add replaces all 3 ops
                    # a = (a + C) + (a << k) = a * (1 + 2^k) + C
                    stages.append([
                        ("valu", ("multiply_add", v_val, v_val, v_fused_mult[stage], v_hash_const_map[val1]))
                    ])
                else:
                    # Non-fusible: keep 2 parallel ops (XOR stages)
                    stages.append([
                        ("valu", (op1, v_tmp1, v_val, v_hash_const_map[val1])),
                        ("valu", (op3, v_tmp2, v_val, v_hash_const_map[val3]))
                    ])
            return stages

        def emit_hash_stage_final_op(buf, chunk, stage):
            if stage in FUSED_HASH_STAGES:
                return None  # Already done in parallel phase via multiply_add
            v_node_val, v_tmp1, v_tmp2, gather_addrs, v_tmp3 = buf
            v_idx, v_val = get_scratch_addrs(chunk)
            op1, val1, op2, op3, val3 = HASH_STAGES[stage]
            return ("valu", (op2, v_val, v_tmp1, v_tmp2))

        def emit_idx_ops(buf, chunk):
            """Index computation using arithmetic instead of vselect."""
            v_node_val, v_tmp1, v_tmp2, gather_addrs, v_tmp3 = buf
            v_idx, v_val = get_scratch_addrs(chunk)
            ops = []
            # val & 1: 0 if even, 1 if odd
            # Add amount: 1 + (val & 1) = 1 if even, 2 if odd
            ops.append(("valu", ("&", v_tmp1, v_val, v_one)))
            ops.append(("valu", ("+", v_tmp1, v_tmp1, v_one)))
            ops.append(("valu", ("*", v_idx, v_idx, v_two)))
            ops.append(("valu", ("+", v_idx, v_idx, v_tmp1)))
            # Bounds check: idx * (idx < n_nodes) - wraps to 0 if out of bounds
            ops.append(("valu", ("<", v_tmp1, v_idx, v_n_nodes)))
            ops.append(("valu", ("*", v_idx, v_idx, v_tmp1)))
            return ops

        def emit_node_val(buf, iter_num):
            """Emit node value computation - shallow for depths 0-2, gather otherwise."""
            chunk = iter_num % n_chunks
            depth = get_depth(iter_num)
            if depth == 0:
                return emit_shallow_d0(buf)
            elif depth == 1:
                return emit_shallow_d1(buf, chunk)
            elif depth == 2:
                return emit_shallow_d2(buf, chunk)
            else:
                ops = emit_gather_addrs(buf, chunk)
                ops.extend(emit_gather_loads(buf))
                return ops

        # Triple processing: process 3 chunks at a time for better valu utilization
        # 6 valu for hash parallel (3 chunks * 2 ops) = full utilization vs 4 for pairs

        # PROLOGUE: Load ALL idx/val from memory into scratch arrays (32 vloads each)
        # This eliminates per-iteration memory access!
        # Allocate 64 address temporaries to avoid RAW hazards between alu and load
        addr_temps = [self.alloc_scratch(f"addr_temp_{i}") for i in range(64)]

        # Phase 1: Compute ALL addresses (64 ALU ops, packs to ~6 cycles at 12 alu/cycle)
        for chunk in range(n_chunks):
            base_i = chunk * VLEN
            instrs.append(("alu", ("+", addr_temps[chunk], self.scratch["inp_indices_p"], base_i_consts[base_i])))
            instrs.append(("alu", ("+", addr_temps[32 + chunk], self.scratch["inp_values_p"], base_i_consts[base_i])))

        # Phase 2: Execute ALL loads (64 vloads, packs to 32 cycles at 2 load/cycle)
        for chunk in range(n_chunks):
            base_i = chunk * VLEN
            instrs.append(("load", ("vload", s_idx + base_i, addr_temps[chunk])))
            instrs.append(("load", ("vload", s_val + base_i, addr_temps[32 + chunk])))

        # Prologue: Compute node_val for first triple
        instrs.extend(emit_node_val(buf_A, 0))
        if total_iterations > 1:
            instrs.extend(emit_node_val(buf_B, 1))
        if total_iterations > 2:
            instrs.extend(emit_node_val(buf_C, 2))

        iter_idx = 0
        while iter_idx < total_iterations:
            has_second = iter_idx + 1 < total_iterations
            has_third = iter_idx + 2 < total_iterations
            has_next_triple = iter_idx + 3 < total_iterations

            # Current chunks being processed
            chunk0 = iter_idx % n_chunks
            chunk1 = (iter_idx + 1) % n_chunks
            chunk2 = (iter_idx + 2) % n_chunks

            triple_idx = (iter_idx // 3) % 2
            if triple_idx == 0:
                buf0, buf1, buf2 = buf_A, buf_B, buf_C
                buf_next0, buf_next1, buf_next2 = buf_D, buf_E, buf_F
            else:
                buf0, buf1, buf2 = buf_D, buf_E, buf_F
                buf_next0, buf_next1, buf_next2 = buf_A, buf_B, buf_C

            # Get scratch addresses for current chunks' values
            v_idx0, v_val0 = get_scratch_addrs(chunk0)
            v_idx1, v_val1 = get_scratch_addrs(chunk1) if has_second else (0, 0)
            v_idx2, v_val2 = get_scratch_addrs(chunk2) if has_third else (0, 0)

            next_chunk0 = (iter_idx + 3) % n_chunks if has_next_triple else 0
            next_chunk1 = (iter_idx + 4) % n_chunks if iter_idx + 4 < total_iterations else 0
            next_chunk2 = (iter_idx + 5) % n_chunks if iter_idx + 5 < total_iterations else 0
            depth_next0 = get_depth(iter_idx + 3) if has_next_triple else 99
            depth_next1 = get_depth(iter_idx + 4) if iter_idx + 4 < total_iterations else 99
            depth_next2 = get_depth(iter_idx + 5) if iter_idx + 5 < total_iterations else 99

            # XOR for all 3 chunks: val ^= node_val (operates directly on scratch arrays)
            instrs.append(("valu", ("^", v_val0, v_val0, buf0[0])))
            if has_second:
                instrs.append(("valu", ("^", v_val1, v_val1, buf1[0])))
            if has_third:
                instrs.append(("valu", ("^", v_val2, v_val2, buf2[0])))

            # Hash stages with gather interleaving for next triple (no more idx/val loading!)
            for stage in range(6):
                # Hash parallel ops for all 3 chunks (6 valu - full utilization!)
                hp0 = emit_hash_stage_parallel_ops(buf0, chunk0)
                instrs.extend(hp0[stage])
                if has_second:
                    hp1 = emit_hash_stage_parallel_ops(buf1, chunk1)
                    instrs.extend(hp1[stage])
                if has_third:
                    hp2 = emit_hash_stage_parallel_ops(buf2, chunk2)
                    instrs.extend(hp2[stage])

                # Interleaved gather address computation and loads for next triple's node_val
                # (idx/val are already in scratch, no need to load them!)
                if stage == 0:
                    # Compute gather addresses for next triple if depth > 2
                    if has_next_triple and depth_next0 > 2:
                        v_idx_next0, _ = get_scratch_addrs(next_chunk0)
                        for i in range(VLEN):
                            instrs.append(("alu", ("+", buf_next0[3][i], self.scratch["forest_values_p"], v_idx_next0 + i)))
                elif stage == 1:
                    if has_next_triple and depth_next0 > 2:
                        instrs.append(("load", ("load", buf_next0[0] + 0, buf_next0[3][0])))
                        instrs.append(("load", ("load", buf_next0[0] + 1, buf_next0[3][1])))
                    if iter_idx + 4 < total_iterations and depth_next1 > 2:
                        v_idx_next1, _ = get_scratch_addrs(next_chunk1)
                        for i in range(VLEN):
                            instrs.append(("alu", ("+", buf_next1[3][i], self.scratch["forest_values_p"], v_idx_next1 + i)))
                elif stage == 2:
                    if has_next_triple and depth_next0 > 2:
                        instrs.append(("load", ("load", buf_next0[0] + 2, buf_next0[3][2])))
                        instrs.append(("load", ("load", buf_next0[0] + 3, buf_next0[3][3])))
                    if iter_idx + 4 < total_iterations and depth_next1 > 2:
                        instrs.append(("load", ("load", buf_next1[0] + 0, buf_next1[3][0])))
                        instrs.append(("load", ("load", buf_next1[0] + 1, buf_next1[3][1])))
                    if iter_idx + 5 < total_iterations and depth_next2 > 2:
                        v_idx_next2, _ = get_scratch_addrs(next_chunk2)
                        for i in range(VLEN):
                            instrs.append(("alu", ("+", buf_next2[3][i], self.scratch["forest_values_p"], v_idx_next2 + i)))
                elif stage == 3:
                    if has_next_triple and depth_next0 > 2:
                        instrs.append(("load", ("load", buf_next0[0] + 4, buf_next0[3][4])))
                        instrs.append(("load", ("load", buf_next0[0] + 5, buf_next0[3][5])))
                    if iter_idx + 4 < total_iterations and depth_next1 > 2:
                        instrs.append(("load", ("load", buf_next1[0] + 2, buf_next1[3][2])))
                        instrs.append(("load", ("load", buf_next1[0] + 3, buf_next1[3][3])))
                    if iter_idx + 5 < total_iterations and depth_next2 > 2:
                        instrs.append(("load", ("load", buf_next2[0] + 0, buf_next2[3][0])))
                        instrs.append(("load", ("load", buf_next2[0] + 1, buf_next2[3][1])))
                elif stage == 4:
                    if has_next_triple and depth_next0 > 2:
                        instrs.append(("load", ("load", buf_next0[0] + 6, buf_next0[3][6])))
                        instrs.append(("load", ("load", buf_next0[0] + 7, buf_next0[3][7])))
                    if iter_idx + 4 < total_iterations and depth_next1 > 2:
                        instrs.append(("load", ("load", buf_next1[0] + 4, buf_next1[3][4])))
                        instrs.append(("load", ("load", buf_next1[0] + 5, buf_next1[3][5])))
                    if iter_idx + 5 < total_iterations and depth_next2 > 2:
                        instrs.append(("load", ("load", buf_next2[0] + 2, buf_next2[3][2])))
                        instrs.append(("load", ("load", buf_next2[0] + 3, buf_next2[3][3])))
                elif stage == 5:
                    if iter_idx + 4 < total_iterations and depth_next1 > 2:
                        instrs.append(("load", ("load", buf_next1[0] + 6, buf_next1[3][6])))
                        instrs.append(("load", ("load", buf_next1[0] + 7, buf_next1[3][7])))
                    if iter_idx + 5 < total_iterations and depth_next2 > 2:
                        instrs.append(("load", ("load", buf_next2[0] + 4, buf_next2[3][4])))
                        instrs.append(("load", ("load", buf_next2[0] + 5, buf_next2[3][5])))

                # Hash final ops for all 3 chunks (None for fused stages)
                final_op = emit_hash_stage_final_op(buf0, chunk0, stage)
                if final_op:
                    instrs.append(final_op)
                if has_second:
                    final_op = emit_hash_stage_final_op(buf1, chunk1, stage)
                    if final_op:
                        instrs.append(final_op)
                if has_third:
                    final_op = emit_hash_stage_final_op(buf2, chunk2, stage)
                    if final_op:
                        instrs.append(final_op)

            # Finish loading remaining gathers for next triple (skip for depths 0-2)
            if iter_idx + 5 < total_iterations and depth_next2 > 2:
                instrs.append(("load", ("load", buf_next2[0] + 6, buf_next2[3][6])))
                instrs.append(("load", ("load", buf_next2[0] + 7, buf_next2[3][7])))

            # Shallow node val for depths 0-2 chunks of next triple
            if has_next_triple and depth_next0 == 0:
                instrs.extend(emit_shallow_d0(buf_next0))
            elif has_next_triple and depth_next0 == 1:
                instrs.extend(emit_shallow_d1(buf_next0, next_chunk0))
            elif has_next_triple and depth_next0 == 2:
                instrs.extend(emit_shallow_d2(buf_next0, next_chunk0))
            if iter_idx + 4 < total_iterations and depth_next1 == 0:
                instrs.extend(emit_shallow_d0(buf_next1))
            elif iter_idx + 4 < total_iterations and depth_next1 == 1:
                instrs.extend(emit_shallow_d1(buf_next1, next_chunk1))
            elif iter_idx + 4 < total_iterations and depth_next1 == 2:
                instrs.extend(emit_shallow_d2(buf_next1, next_chunk1))
            if iter_idx + 5 < total_iterations and depth_next2 == 0:
                instrs.extend(emit_shallow_d0(buf_next2))
            elif iter_idx + 5 < total_iterations and depth_next2 == 1:
                instrs.extend(emit_shallow_d1(buf_next2, next_chunk2))
            elif iter_idx + 5 < total_iterations and depth_next2 == 2:
                instrs.extend(emit_shallow_d2(buf_next2, next_chunk2))

            # Index computation for all 3 chunks (operates directly on scratch arrays)
            instrs.extend(emit_idx_ops(buf0, chunk0))
            if has_second:
                instrs.extend(emit_idx_ops(buf1, chunk1))
            if has_third:
                instrs.extend(emit_idx_ops(buf2, chunk2))

            # NO PER-ITERATION STORES! Data stays in scratch arrays

            # Advance by 3 (or fewer for last iterations)
            if has_third:
                iter_idx += 3
            elif has_second:
                iter_idx += 2
            else:
                iter_idx += 1

        # EPILOGUE: Store ALL idx/val from scratch back to memory (32 vstores each)
        # Phase 1: Compute ALL store addresses (64 ALU ops, packs to ~6 cycles)
        for chunk in range(n_chunks):
            base_i = chunk * VLEN
            instrs.append(("alu", ("+", addr_temps[chunk], self.scratch["inp_indices_p"], base_i_consts[base_i])))
            instrs.append(("alu", ("+", addr_temps[32 + chunk], self.scratch["inp_values_p"], base_i_consts[base_i])))

        # Phase 2: Execute ALL stores (64 vstores, packs to 32 cycles at 2 store/cycle)
        for chunk in range(n_chunks):
            base_i = chunk * VLEN
            instrs.append(("store", ("vstore", addr_temps[chunk], s_idx + base_i)))
            instrs.append(("store", ("vstore", addr_temps[32 + chunk], s_val + base_i)))

        body_instrs = self.build(instrs)
        self.instrs.extend(body_instrs)
        self.instrs.append({"flow": [("pause",)]})


BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


if __name__ == "__main__":
    unittest.main()

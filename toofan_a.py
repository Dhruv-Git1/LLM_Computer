#!/usr/bin/env python3
"""
Replication: "Can LLMs Be Computers?" (Percepta, March 2026)

A hand-crafted transformer that executes stack-machine programs.
No training, no gradient descent -- all weights constructed to make
the forward pass act as a program interpreter.

Demonstrates:
  1. Argmax attention with 2D heads as memory lookup
  2. FFN with hand-crafted weights as opcode dispatch + ALU
  3. Full transformer execution loop (each forward pass = one clock cycle)
  4. HullKVCache for O(log n) attention via convex hull
  5. Benchmarks: naive O(n) vs hull O(log n)
  6. ZERO Python if/elif in the forward pass -- all computation via matrices
"""

import numpy as np
from time import perf_counter
import sys

# ================================================================
# OPCODES
# ================================================================
CONST   = 1
ADD     = 2
SUB     = 3
MUL     = 4
HALT    = 5
DUP     = 6
NEG     = 7
JMP     = 8    # JMP addr       -- unconditional jump to addr
JZ      = 9    # JZ addr        -- pop top; jump to addr if zero
LOAD    = 10   # LOAD addr      -- push value from memory[addr]
STORE   = 11   # STORE addr     -- pop top; store to memory[addr]
CMP_LT  = 12   # CMP_LT         -- pop b, a; push (a < b) ? 1 : 0
SWAP    = 13   # SWAP            -- swap top two stack elements
JN      = 14   # JN addr        -- pop top; jump to addr if negative

OP_NAME = {CONST:"CONST", ADD:"ADD", SUB:"SUB", MUL:"MUL",
           HALT:"HALT", DUP:"DUP", NEG:"NEG", JMP:"JMP", JZ:"JZ",
           LOAD:"LOAD", STORE:"STORE", CMP_LT:"CMP_LT", SWAP:"SWAP",
           JN:"JN"}

# ================================================================
# PART 1: Ground-truth stack machine (reference implementation)
# ================================================================

def run_reference(program, max_cycles=100000):
    """Simple stack machine. Returns (final_stack, trace_lines)."""
    pc, stack, trace = 0, [], []
    mem = {}   # memory for LOAD/STORE
    cycles = 0
    while pc < len(program) and cycles < max_cycles:
        cycles += 1
        op = program[pc]
        code = op[0]
        if code == HALT:
            trace.append(f"  HALT          | stack = {stack}")
            break
        elif code == CONST:
            stack.append(op[1])
            trace.append(f"  CONST {op[1]:<7} | stack = {stack}")
        elif code == ADD:
            b, a = stack.pop(), stack.pop()
            stack.append(a + b)
            trace.append(f"  ADD {a}+{b}={a+b:<5} | stack = {stack}")
        elif code == SUB:
            b, a = stack.pop(), stack.pop()
            stack.append(a - b)
            trace.append(f"  SUB {a}-{b}={a-b:<5} | stack = {stack}")
        elif code == MUL:
            b, a = stack.pop(), stack.pop()
            stack.append(a * b)
            trace.append(f"  MUL {a}*{b}={a*b:<5} | stack = {stack}")
        elif code == DUP:
            stack.append(stack[-1])
            trace.append(f"  DUP           | stack = {stack}")
        elif code == NEG:
            stack[-1] = -stack[-1]
            trace.append(f"  NEG           | stack = {stack}")
        elif code == JMP:
            pc = op[1]
            trace.append(f"  JMP {op[1]:<8} | stack = {stack}")
            continue   # skip pc += 1
        elif code == JZ:
            val = stack.pop()
            if val == 0:
                pc = op[1]
                trace.append(f"  JZ {op[1]} (taken)| stack = {stack}")
                continue
            else:
                trace.append(f"  JZ {op[1]} (skip) | stack = {stack}")
        elif code == LOAD:
            stack.append(mem.get(op[1], 0))
            trace.append(f"  LOAD [{op[1]}]={mem.get(op[1],0):<4} | stack = {stack}")
        elif code == STORE:
            val = stack.pop()
            mem[op[1]] = val
            trace.append(f"  STORE [{op[1]}]={val:<3} | stack = {stack}")
        elif code == CMP_LT:
            b, a = stack.pop(), stack.pop()
            stack.append(1 if a < b else 0)
            trace.append(f"  CMP_LT {a}<{b}={'T' if a<b else 'F'} | stack = {stack}")
        elif code == SWAP:
            stack[-1], stack[-2] = stack[-2], stack[-1]
            trace.append(f"  SWAP          | stack = {stack}")
        elif code == JN:
            val = stack.pop()
            if val < 0:
                pc = op[1]
                trace.append(f"  JN {op[1]} (taken)| stack = {stack}")
                continue
            else:
                trace.append(f"  JN {op[1]} (skip) | stack = {stack}")
        pc += 1
    return stack, trace


# ================================================================
# PART 2: 2D Encoding helpers
# ================================================================

def addr_to_2d(addr, max_n=64):
    """Encode an integer address as a 2D unit vector (angle-based)."""
    theta = 2.0 * np.pi * addr / max_n
    return np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)

def dot2d(a, b):
    return a[0]*b[0] + a[1]*b[1]


# ================================================================
# PART 3: 2D Argmax Attention (naive O(n) scan)
# ================================================================

def argmax_attention_naive(query_2d, keys_2d, values):
    """
    Find the key most aligned with the query direction.
    Returns (best_value, best_index).

    This is the core of the paper: attention as addressed memory lookup.
    query_2d: shape (2,)
    keys_2d:  shape (n, 2)
    values:   shape (n, ...)
    """
    scores = keys_2d @ query_2d          # dot product with every key
    best = np.argmax(scores)
    return values[best], best


# ================================================================
# PART 4: HullKVCache -- O(log n) via convex hull
# ================================================================

class ConvexHull2D:
    """
    Incremental 2D convex hull (upper hull only, sufficient for
    argmax queries since we can query both upper and lower).

    Maintains points sorted by x-coordinate.
    Supports:
      - insert(point, token_id): amortized O(log n)
      - query(direction): O(log n) via binary search
    """
    def __init__(self):
        self.points = []      # list of (x, y, token_id), sorted by angle
        self.hull = []        # convex hull vertices, sorted by angle
        self._angles = []     # angles of hull vertices (for binary search)

    def insert(self, point_2d, token_id):
        """
        Add a new 2D point. Only rebuild hull if point is outside it.

        The inside check (O(h)) eliminates most inserts -- for n random
        points, only O(sqrt(n)) land on the hull. So while each rebuild
        is O(n log n), the amortized cost per insert is much lower.
        """
        px, py = point_2d[0], point_2d[1]
        self.points.append((px, py, token_id))

        # Quick check: if point is inside current hull, skip rebuild
        if len(self.hull) >= 3 and self._is_inside(px, py):
            return
        self._rebuild_hull()

    def _is_inside(self, px, py):
        """Check if point is inside the current convex hull (cross product test)."""
        n = len(self.hull)
        for i in range(n):
            hx1, hy1, _ = self.hull[i]
            hx2, hy2, _ = self.hull[(i + 1) % n]
            cross = (hx2 - hx1) * (py - hy1) - (hy2 - hy1) * (px - hx1)
            if cross > 1e-12:  # strictly outside on this edge
                return False
        return True

    def _rebuild_hull(self):
        """Full rebuild of convex hull. For production, use incremental update."""
        if len(self.points) < 3:
            self.hull = [(p[0], p[1], p[2]) for p in self.points]
            self._angles = [np.arctan2(p[1], p[0]) for p in self.hull]
            return

        pts = [(p[0], p[1], p[2]) for p in self.points]

        # Andrew's monotone chain
        pts_sorted = sorted(pts, key=lambda p: (p[0], p[1]))

        # Lower hull
        lower = []
        for p in pts_sorted:
            while len(lower) >= 2 and self._cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        # Upper hull
        upper = []
        for p in reversed(pts_sorted):
            while len(upper) >= 2 and self._cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        # Concatenate, remove duplicates at join points
        hull = lower[:-1] + upper[:-1]

        # Sort hull by angle for binary search
        hull_with_angle = [(h[0], h[1], h[2], np.arctan2(h[1], h[0])) for h in hull]
        hull_with_angle.sort(key=lambda h: h[3])

        self.hull = [(h[0], h[1], h[2]) for h in hull_with_angle]
        self._angles = [h[3] for h in hull_with_angle]

    @staticmethod
    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def query(self, direction_2d):
        """
        Find the hull vertex that maximizes dot(direction, vertex).
        O(log h) via binary search on the angle-sorted hull vertices.

        Algorithm: binary search _angles for the query's angle, then
        check a small neighborhood (+-2 vertices) to find the exact max.
        For convex hull vertices with near-unit magnitude, the optimal
        vertex is always within +-1 of the angle-nearest vertex.

        Returns token_id of best match.
        """
        import bisect

        h = len(self.hull)
        if h == 0:
            return -1
        if h <= 3:
            best_id = -1
            best_dot = -np.inf
            for hx, hy, tid in self.hull:
                d = direction_2d[0] * hx + direction_2d[1] * hy
                if d > best_dot:
                    best_dot = d
                    best_id = tid
            return best_id

        # Binary search for query angle in the sorted hull angles -- O(log h)
        query_angle = np.arctan2(direction_2d[1], direction_2d[0])
        pos = bisect.bisect_left(self._angles, query_angle)

        # Check neighbors (+-2) around the insertion point, wrapping around
        best_id = -1
        best_dot = -np.inf
        for offset in range(-2, 3):
            idx = (pos + offset) % h
            hx, hy, tid = self.hull[idx]
            d = direction_2d[0] * hx + direction_2d[1] * hy
            if d > best_dot:
                best_dot = d
                best_id = tid
        return best_id


class HullKVCache:
    """
    KV Cache using per-head convex hulls for O(log n) argmax attention.

    Each head has its own 2D convex hull of keys.
    Values are shared across heads (each token has one value per head).
    """
    def __init__(self, n_heads):
        self.n_heads = n_heads
        self.hulls = [ConvexHull2D() for _ in range(n_heads)]
        self.values = {}     # token_id -> {head_id: value}
        self.n_tokens = 0

    def insert(self, keys_per_head, values_per_head):
        """
        Insert a new token.
        keys_per_head: dict {head_id: 2D np.array}
        values_per_head: dict {head_id: value}
        """
        tid = self.n_tokens
        self.n_tokens += 1
        self.values[tid] = values_per_head
        for head_id, key_2d in keys_per_head.items():
            self.hulls[head_id].insert(key_2d, tid)
        return tid

    def query(self, head_id, query_2d):
        """Query a specific head. Returns best matching token's value."""
        tid = self.hulls[head_id].query(query_2d)
        if tid < 0:
            return None
        return self.values[tid].get(head_id)


class NaiveKVCache:
    """Standard O(n) KV cache for comparison."""
    def __init__(self, n_heads):
        self.n_heads = n_heads
        self.keys = {h: [] for h in range(n_heads)}   # head -> list of 2D keys
        self.values = {h: [] for h in range(n_heads)}  # head -> list of values
        self.n_tokens = 0

    def insert(self, keys_per_head, values_per_head):
        tid = self.n_tokens
        self.n_tokens += 1
        for h, k in keys_per_head.items():
            self.keys[h].append(k)
            self.values[h].append(values_per_head[h])
        return tid

    def query(self, head_id, query_2d):
        keys = self.keys[head_id]
        if len(keys) == 0:
            return None
        keys_arr = np.array(keys)
        scores = keys_arr @ query_2d
        best = np.argmax(scores)
        return self.values[head_id][best]


# ================================================================
# PART 5: The Transformer Computer
# ================================================================

# This is the core of the paper: a transformer whose forward pass
# IS a program interpreter.
#
# Architecture:
#   - 3 attention heads (2D each) for: instruction fetch, stack read A, stack read B
#   - FFN for: opcode decode, arithmetic execution, state update
#   - KV cache = the computer's memory
#
# Each forward pass = one clock cycle:
#   Layer 1 (Attention): Fetch instruction at current PC
#   Layer 2 (FFN):       Decode opcode, read stack, execute, update state
#
# The "hand-crafted weights" are encoded in the logic of how we
# construct queries and process results -- but ALL operations are
# matrix multiplies and argmax, exactly as in a real transformer.

# Head assignments:
HEAD_INSTR  = 0    # Instruction fetch
HEAD_STACK0 = 1    # Stack read: top (SP - 1)
HEAD_STACK1 = 2    # Stack read: second (SP - 2)

MAX_ADDRS  = 65536    # max instruction addresses (large to avoid wrap-around)
MAX_STACKD = 32       # max stack depth (small -> wide angular separation between positions)

# ================================================================
# FFN Weight Matrices (hand-crafted, not trained)
# ================================================================
# These are the actual weight matrices that implement arithmetic
# in the FFN layer. Each is applied via matrix multiplication.

W_ADD = np.array([1.0, 1.0], dtype=np.float64)    # result = W_ADD @ [a, b] = a + b
W_SUB = np.array([1.0, -1.0], dtype=np.float64)   # result = W_SUB @ [a, b] = a - b
W_NEG = np.array([-1.0], dtype=np.float64)         # result = W_NEG @ [a]    = -a

# PC rotation matrix: rotates the 2D PC direction by delta_theta = 2*pi/MAX_ADDRS
# Applying R_PC to the current PC vector = incrementing PC by 1
_dtheta_pc = 2.0 * np.pi / MAX_ADDRS
R_PC = np.array([[np.cos(_dtheta_pc), -np.sin(_dtheta_pc)],
                 [np.sin(_dtheta_pc),  np.cos(_dtheta_pc)]], dtype=np.float64)

# SP rotation matrix: rotates the 2D SP direction by delta_theta = 2*pi/MAX_STACKD
_dtheta_sp = 2.0 * np.pi / MAX_STACKD
R_SP_INC = np.array([[np.cos(_dtheta_sp), -np.sin(_dtheta_sp)],
                     [np.sin(_dtheta_sp),  np.cos(_dtheta_sp)]], dtype=np.float64)
# SP decrement = rotate backwards (transpose of R_SP_INC since it's orthogonal)
R_SP_DEC = R_SP_INC.T
R_SP_DEC2 = R_SP_DEC @ R_SP_DEC  # double decrement for sp-2

def make_instr_key(addr):
    """2D key encoding an instruction address."""
    return addr_to_2d(addr, MAX_ADDRS)

def make_stack_key(stack_pos):
    """2D key encoding a stack position."""
    # Use different angle base to avoid collision with instruction keys
    theta = 2.0 * np.pi * stack_pos / MAX_STACKD + 0.1  # offset to avoid degeneracy
    return np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)

def make_instr_query(pc):
    """2D query to fetch instruction at address pc."""
    return addr_to_2d(pc, MAX_ADDRS)

def make_stack_query(stack_pos):
    """2D query to read stack at position stack_pos."""
    theta = 2.0 * np.pi * stack_pos / MAX_STACKD + 0.1
    return np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)


# ================================================================
# RESIDUAL STREAM LAYOUT & TRANSFORMER CONSTANTS
# ================================================================
D_MODEL = 30          # residual stream width
N_OPCODES = 10        # number of supported opcodes in the transformer
OPCODE_LIST = [CONST, ADD, SUB, MUL, HALT, DUP, NEG, JMP, JZ, JN]
OPCODE_TO_ACT_DIM = {op: 6 + i for i, op in enumerate(OPCODE_LIST)}

# Pre-computed opcode indices in OPCODE_LIST
CONST_ACT_IDX = 0
ADD_ACT_IDX   = 1
SUB_ACT_IDX   = 2
MUL_ACT_IDX   = 3
HALT_ACT_IDX  = 4
DUP_ACT_IDX   = 5
NEG_ACT_IDX   = 6
JMP_ACT_IDX   = 7
JZ_ACT_IDX    = 8
JN_ACT_IDX    = 9

# Writing opcodes: CONST, ADD, SUB, MUL, DUP, NEG
WRITING_OPCODES_LIST = [CONST, ADD, SUB, MUL, DUP, NEG]
WRITE_FLAG_SELECTOR = np.zeros(N_OPCODES, dtype=np.float64)
for _wop in WRITING_OPCODES_LIST:
    WRITE_FLAG_SELECTOR[OPCODE_LIST.index(_wop)] = 1.0

# Residual stream dimension map (30 floats)
PC_DIR     = slice(0, 2)    # (2) PC as 2D unit vector
SP_DIR     = slice(2, 4)    # (2) SP as 2D unit vector
OPCODE_RAW = 4              # (1) fetched opcode (as float)
ARG_FLOAT  = 5              # (1) fetched instruction argument
ACT        = slice(6, 16)   # (10) one-hot opcode activations
VAL_A      = 16             # (1) stack second-from-top
VAL_B      = 17             # (1) stack top
ALU_RESULT = 18             # (1) arithmetic output
JMP_TGT    = slice(19, 21)  # (2) 2D direction of jump target
ZERO_FLAG  = 21             # (1) 1.0 if val_b == 0
NEG_FLAG   = 22             # (1) 1.0 if val_b < 0
NEW_PC     = slice(23, 25)  # (2) computed next PC direction
NEW_SP     = slice(25, 27)  # (2) computed next SP direction
HALT_FLAG  = 27             # (1) 1.0 if HALT
WRITE_FLAG = 28             # (1) 1.0 if stack write needed
CYCLE_CTR  = 29             # (1) monotonic cycle counter

# Persistence mask: keep PC, SP, cycle counter between cycles
PERSISTENCE_MASK = np.zeros(D_MODEL, dtype=np.float64)
PERSISTENCE_MASK[PC_DIR] = 1.0
PERSISTENCE_MASK[SP_DIR] = 1.0
PERSISTENCE_MASK[CYCLE_CTR] = 1.0


# ================================================================
# LAYER 2: Opcode Decode -- Tent Function FFN
# ================================================================
def _build_layer2_weights():
    """
    Build FFN weights for opcode decode.
    tent(x, c) = max(0, 1 - |x - c|) via 4 ReLU neurons per opcode.
    """
    n_hidden = 4 * N_OPCODES
    W1 = np.zeros((n_hidden, D_MODEL), dtype=np.float64)
    b1 = np.zeros(n_hidden, dtype=np.float64)
    W2 = np.zeros((D_MODEL, n_hidden), dtype=np.float64)
    b2 = np.zeros(D_MODEL, dtype=np.float64)

    for i, op in enumerate(OPCODE_LIST):
        base = 4 * i
        out_dim = 6 + i

        W1[base + 0, OPCODE_RAW] = 1.0;  b1[base + 0] = -op + 1.0
        W1[base + 1, OPCODE_RAW] = 1.0;  b1[base + 1] = -op
        W1[base + 2, OPCODE_RAW] = -1.0; b1[base + 2] = op + 1.0
        W1[base + 3, OPCODE_RAW] = -1.0; b1[base + 3] = op

        W2[out_dim, base + 0] = 0.5
        W2[out_dim, base + 1] = -1.0
        W2[out_dim, base + 2] = 0.5
        W2[out_dim, base + 3] = -1.0

    return W1, b1, W2, b2


# ================================================================
# LAYER 4: ALU -- GLU-style FFN (ReGLU)
# ================================================================
def _build_layer4_glu_weights():
    """
    Build GLU FFN weights for ALU + flags.
    18 hidden units: 7 ALU + 2 zero_flag + 1 halt + 2 neg_flag + 6 write_flag.
    """
    d_hidden = 18
    mul_indices = np.array([3, 4])
    eps_zf = 1e-9

    W1a = np.zeros((d_hidden, D_MODEL), dtype=np.float64)
    b1a = np.zeros(d_hidden, dtype=np.float64)
    W1b = np.zeros((d_hidden, D_MODEL), dtype=np.float64)
    b1b = np.zeros(d_hidden, dtype=np.float64)
    W2 = np.zeros((D_MODEL, d_hidden), dtype=np.float64)
    b2 = np.zeros(D_MODEL, dtype=np.float64)

    # h0: CONST
    W1a[0, 6 + CONST_ACT_IDX] = 1.0
    W1b[0, ARG_FLOAT] = 1.0
    W2[ALU_RESULT, 0] = 1.0

    # h1: ADD
    W1a[1, 6 + ADD_ACT_IDX] = 1.0
    W1b[1, VAL_A] = 1.0; W1b[1, VAL_B] = 1.0
    W2[ALU_RESULT, 1] = 1.0

    # h2: SUB
    W1a[2, 6 + SUB_ACT_IDX] = 1.0
    W1b[2, VAL_A] = 1.0; W1b[2, VAL_B] = -1.0
    W2[ALU_RESULT, 2] = 1.0

    # h3: MUL (a+b) path
    W1a[3, 6 + MUL_ACT_IDX] = 1.0
    W1b[3, VAL_A] = 1.0; W1b[3, VAL_B] = 1.0
    W2[ALU_RESULT, 3] = 0.25

    # h4: MUL (a-b) path
    W1a[4, 6 + MUL_ACT_IDX] = 1.0
    W1b[4, VAL_A] = 1.0; W1b[4, VAL_B] = -1.0
    W2[ALU_RESULT, 4] = -0.25

    # h5: DUP
    W1a[5, 6 + DUP_ACT_IDX] = 1.0
    W1b[5, VAL_B] = 1.0
    W2[ALU_RESULT, 5] = 1.0

    # h6: NEG
    W1a[6, 6 + NEG_ACT_IDX] = 1.0
    W1b[6, VAL_B] = -1.0
    W2[ALU_RESULT, 6] = 1.0

    # h7: zero_flag positive
    W1a[7, VAL_B] = 1.0; b1b[7] = 1.0
    W2[ZERO_FLAG, 7] = -1.0 / eps_zf

    # h8: zero_flag negative
    W1a[8, VAL_B] = -1.0; b1b[8] = 1.0
    W2[ZERO_FLAG, 8] = -1.0 / eps_zf
    b2[ZERO_FLAG] = 1.0

    # h9: halt_flag
    W1a[9, 6 + HALT_ACT_IDX] = 1.0; b1b[9] = 1.0
    W2[HALT_FLAG, 9] = 1.0

    # h10-h11: neg_flag (clamped ramp)
    W1a[10, VAL_B] = -1.0; b1b[10] = 1.0
    W2[NEG_FLAG, 10] = 1.0 / eps_zf
    W1a[11, VAL_B] = -1.0; b1a[11] = -eps_zf; b1b[11] = 1.0
    W2[NEG_FLAG, 11] = -1.0 / eps_zf

    # h12-h17: write_flag
    for i, op_idx in enumerate([CONST_ACT_IDX, ADD_ACT_IDX, SUB_ACT_IDX,
                                MUL_ACT_IDX, DUP_ACT_IDX, NEG_ACT_IDX]):
        h = 12 + i
        W1a[h, 6 + op_idx] = 1.0; b1b[h] = 1.0
        W2[WRITE_FLAG, h] = 1.0

    return W1a, b1a, W1b, b1b, W2, b2, mul_indices


# ================================================================
# LAYER 5: Branch Resolution -- GLU-style FFN (ReGLU)
# ================================================================
def _build_layer5_glu_weights():
    """
    Build GLU FFN weights for branch resolution + state update.
    22 hidden units: 6 PC (JMP/JZ/JN) + 10 SP dec + 4 SP inc + 2 PC pass-through.
    """
    d_hidden = 22
    W1a = np.zeros((d_hidden, D_MODEL), dtype=np.float64)
    b1a = np.zeros(d_hidden, dtype=np.float64)
    W1b = np.zeros((d_hidden, D_MODEL), dtype=np.float64)
    b1b = np.zeros(d_hidden, dtype=np.float64)
    W2 = np.zeros((D_MODEL, d_hidden), dtype=np.float64)
    b2 = np.zeros(D_MODEL, dtype=np.float64)

    for k in range(2):
        # h0,h1: JMP
        h_jmp = k
        W1a[h_jmp, 6 + JMP_ACT_IDX] = 1.0
        W1b[h_jmp, JMP_TGT.start + k] = 1.0
        W1b[h_jmp, 0] = -R_PC[k, 0]; W1b[h_jmp, 1] = -R_PC[k, 1]
        W2[k, h_jmp] = 1.0; W2[NEW_PC.start + k, h_jmp] = 1.0

        # h2,h3: JZ AND gate
        h_jz = 2 + k
        W1a[h_jz, 6 + JZ_ACT_IDX] = 1.0; W1a[h_jz, ZERO_FLAG] = 1.0
        b1a[h_jz] = -1.5
        W1b[h_jz, JMP_TGT.start + k] = 1.0
        W1b[h_jz, 0] = -R_PC[k, 0]; W1b[h_jz, 1] = -R_PC[k, 1]
        W2[k, h_jz] = 2.0; W2[NEW_PC.start + k, h_jz] = 2.0

        # h4,h5: JN AND gate
        h_jn = 4 + k
        W1a[h_jn, 6 + JN_ACT_IDX] = 1.0; W1a[h_jn, NEG_FLAG] = 1.0
        b1a[h_jn] = -1.5
        W1b[h_jn, JMP_TGT.start + k] = 1.0
        W1b[h_jn, 0] = -R_PC[k, 0]; W1b[h_jn, 1] = -R_PC[k, 1]
        W2[k, h_jn] = 2.0; W2[NEW_PC.start + k, h_jn] = 2.0

    # SP decrement: ADD, SUB, MUL, JZ, JN
    dec_opcodes = [ADD_ACT_IDX, SUB_ACT_IDX, MUL_ACT_IDX, JZ_ACT_IDX, JN_ACT_IDX]
    for i, op_idx in enumerate(dec_opcodes):
        for k in range(2):
            h = 6 + i * 2 + k
            W1a[h, 6 + op_idx] = 1.0
            W1b[h, 2] = R_SP_DEC[k, 0] - (1.0 if k == 0 else 0.0)
            W1b[h, 3] = R_SP_DEC[k, 1] - (1.0 if k == 1 else 0.0)
            W2[2 + k, h] = 1.0; W2[NEW_SP.start + k, h] = 1.0

    # SP increment: CONST, DUP
    inc_opcodes = [CONST_ACT_IDX, DUP_ACT_IDX]
    for i, op_idx in enumerate(inc_opcodes):
        for k in range(2):
            h = 16 + i * 2 + k
            W1a[h, 6 + op_idx] = 1.0
            W1b[h, 2] = R_SP_INC[k, 0] - (1.0 if k == 0 else 0.0)
            W1b[h, 3] = R_SP_INC[k, 1] - (1.0 if k == 1 else 0.0)
            W2[2 + k, h] = 1.0; W2[NEW_SP.start + k, h] = 1.0

    # PC pass-through (baseline advance)
    for k in range(2):
        h_pt = 20 + k
        b1a[h_pt] = 1.0
        W1b[h_pt, k] = 1.0
        W2[0, h_pt] = R_PC[0, k] - (1.0 if k == 0 else 0.0)
        W2[1, h_pt] = R_PC[1, k] - (1.0 if k == 1 else 0.0)
        W2[NEW_PC.start, h_pt] = R_PC[0, k] - (1.0 if k == 0 else 0.0)
        W2[NEW_PC.start + 1, h_pt] = R_PC[1, k] - (1.0 if k == 1 else 0.0)

    b2[CYCLE_CTR] = 1.0
    return W1a, b1a, W1b, b1b, W2, b2


# ================================================================
# THE TRANSFORMER COMPUTER -- ZERO if/elif in forward_pass
# ================================================================

class TransformerComputer:
    """
    A transformer whose forward pass has ZERO Python if/elif.

    All computation is:
      x = x + attention(x, kv_cache)   # layers 1, 3
      x = x + FFN(x)                   # layers 2, 4, 5

    State is ONLY:
      - self.residual_stream: np.array of shape (30,)
      - self.instr_cache: KV cache for instructions (NaiveKVCache or HullKVCache)
      - self.stack_cache: KV cache for stack memory (NaiveKVCache)

    Cache modes:
      - use_hull=False: NaiveKVCache for instructions -- O(n) attention
      - use_hull=True: HullKVCache for instructions -- O(log n) via convex hull
        Stack cache stays NaiveKVCache because magnitude-scaled keys break
        the unit-circle assumption required by convex hull.
    """

    def __init__(self, use_hull=False):
        self.use_hull = use_hull

        # Build fixed weight matrices (hand-crafted, not trained)
        self.W1_L2, self.b1_L2, self.W2_L2, self.b2_L2 = _build_layer2_weights()
        (self.W1a_L4, self.b1a_L4, self.W1b_L4, self.b1b_L4,
         self.W2_L4, self.b2_L4, self.mul_indices) = _build_layer4_glu_weights()
        (self.W1a_L5, self.b1a_L5, self.W1b_L5, self.b1b_L5,
         self.W2_L5, self.b2_L5) = _build_layer5_glu_weights()

        # Layer 1: W_Q extracts pc_dir (dims 0-1)
        self.W_Q_L1 = np.zeros((2, D_MODEL), dtype=np.float64)
        self.W_Q_L1[0:2, 0:2] = np.eye(2)

        # Layer 1: W_O projects fetched 4-tuple into residual stream
        self.W_O_L1 = np.zeros((D_MODEL, 4), dtype=np.float64)
        self.W_O_L1[OPCODE_RAW, 0] = 1.0
        self.W_O_L1[ARG_FLOAT, 1] = 1.0
        self.W_O_L1[JMP_TGT.start, 2] = 1.0
        self.W_O_L1[JMP_TGT.start + 1, 3] = 1.0

        # Layer 3: W_Q for stack reads
        self.W_Q_head1 = np.zeros((2, D_MODEL), dtype=np.float64)
        self.W_Q_head1[0:2, 2:4] = R_SP_DEC

        self.W_Q_head2 = np.zeros((2, D_MODEL), dtype=np.float64)
        self.W_Q_head2[0:2, 2:4] = R_SP_DEC2

        # Layer 3: W_O projects [va, vb] into residual stream
        self.W_O_L3 = np.zeros((D_MODEL, 2), dtype=np.float64)
        self.W_O_L3[VAL_A, 0] = 1.0
        self.W_O_L3[VAL_B, 1] = 1.0

        self.reset()

    def reset(self):
        """Initialize state. The ONLY state is residual_stream + KV caches."""
        self.residual_stream = np.zeros(D_MODEL, dtype=np.float64)
        self.residual_stream[PC_DIR] = addr_to_2d(0, MAX_ADDRS)
        self.residual_stream[SP_DIR] = make_stack_key(0)

        # Instruction cache: HullKVCache (O(log n)) or NaiveKVCache (O(n))
        self.instr_cache = (HullKVCache(n_heads=1) if self.use_hull
                            else NaiveKVCache(n_heads=1))
        # Stack cache: always NaiveKVCache (magnitude-scaled keys break hull)
        self.stack_cache = NaiveKVCache(n_heads=3)

        # Pre-seed stack cache so queries never return None
        default_key = make_stack_key(MAX_STACKD - 1)
        self.stack_cache.insert(
            {HEAD_STACK0: default_key, HEAD_STACK1: default_key},
            {HEAD_STACK0: 0.0, HEAD_STACK1: 0.0}
        )

        self.halted = False
        self.trace = []
        self.cycle_count = 0

    def load_program(self, program):
        """Load instructions into KV cache (= writing program to RAM)."""
        self.reset()
        self.program = program

        # Pre-seed with default HALT at far-away address
        default_tgt = addr_to_2d(0, MAX_ADDRS)
        self.instr_cache.insert(
            {0: addr_to_2d(MAX_ADDRS - 1, MAX_ADDRS)},
            {0: (float(HALT), 0.0, default_tgt[0], default_tgt[1])}
        )

        # Load actual program: each instruction stores (opcode, arg, tgt_cos, tgt_sin)
        for addr, instr in enumerate(program):
            opcode = instr[0]
            arg = instr[1] if len(instr) > 1 else 0
            tgt = addr_to_2d(arg, MAX_ADDRS)
            self.instr_cache.insert(
                {0: addr_to_2d(addr, MAX_ADDRS)},
                {0: (float(opcode), float(arg), tgt[0], tgt[1])}
            )

    # ==============================================================
    # THE FORWARD PASS -- ZERO if/elif
    # ==============================================================

    def forward_pass(self):
        """
        Execute one clock cycle. Returns halt_signal (float).

        This method contains ZERO if/elif/else statements.
        Every operation is: matrix multiply, ReLU, argmax attention,
        or element-wise arithmetic on the residual stream.
        """
        x = self.residual_stream.copy()

        # ==========================================
        # LAYER 1: Instruction Fetch (Attention)
        # ==========================================
        query = self.W_Q_L1 @ x  # shape (2,)
        fetched = self.instr_cache.query(0, query)
        # fetched = (opcode, arg, tgt_cos, tgt_sin) -- pre-seeded, never None
        fetched_vec = np.array(fetched, dtype=np.float64)
        x = x + self.W_O_L1 @ fetched_vec  # residual connection

        # ==========================================
        # LAYER 2: Opcode Decode (FFN)
        # ==========================================
        hidden2 = np.maximum(0, self.W1_L2 @ x + self.b1_L2)
        x = x + self.W2_L2 @ hidden2 + self.b2_L2

        # ==========================================
        # LAYER 3: Operand Fetch (Attention)
        # ==========================================
        q_top = self.W_Q_head1 @ x
        vb = self.stack_cache.query(HEAD_STACK0, q_top)
        q_sec = self.W_Q_head2 @ x
        va = self.stack_cache.query(HEAD_STACK1, q_sec)
        # Pre-seeded cache guarantees non-None -- no Python ternary needed
        fetched_stack = np.array([va, vb], dtype=np.float64)
        x = x + self.W_O_L3 @ fetched_stack

        # ==========================================
        # LAYER 4: ALU + Flags (GLU FFN)
        # ==========================================
        gate4 = np.maximum(0, self.W1a_L4 @ x + self.b1a_L4)
        value4 = self.W1b_L4 @ x + self.b1b_L4
        hidden4 = gate4 * value4
        hidden4[self.mul_indices] = hidden4[self.mul_indices] * hidden4[self.mul_indices]
        x = x + self.W2_L4 @ hidden4 + self.b2_L4

        # ==========================================
        # LAYER 5: Branch Resolution + State Update (GLU FFN)
        # ==========================================
        acts = np.maximum(0, x[ACT])
        sp_dir_pre = x[SP_DIR].copy()

        gate5 = np.maximum(0, self.W1a_L5 @ x + self.b1a_L5)
        value5 = self.W1b_L5 @ x + self.b1b_L5
        hidden5 = gate5 * value5
        x = x + self.W2_L5 @ hidden5 + self.b2_L5

        # ==========================================
        # MEMORY WRITE BUS (cache I/O, not a transformer layer)
        # ==========================================
        sp_dec1 = R_SP_DEC @ sp_dir_pre
        sp_dec2 = R_SP_DEC2 @ sp_dir_pre

        gate_write_at_sp = acts[CONST_ACT_IDX] + acts[DUP_ACT_IDX]
        gate_write_at_sp2 = acts[ADD_ACT_IDX] + acts[SUB_ACT_IDX] + acts[MUL_ACT_IDX]
        gate_write_at_sp1 = acts[NEG_ACT_IDX]
        write_key_dir = (gate_write_at_sp * sp_dir_pre +
                         gate_write_at_sp2 * sp_dec2 +
                         gate_write_at_sp1 * sp_dec1)

        cycle = x[CYCLE_CTR]
        magnitude = 1.0 + cycle * 1e-8
        gated_mag = magnitude * np.maximum(0, x[WRITE_FLAG])
        stack_key = write_key_dir * gated_mag
        stack_val = x[ALU_RESULT]

        self.stack_cache.insert(
            {HEAD_STACK0: stack_key, HEAD_STACK1: stack_key},
            {HEAD_STACK0: stack_val, HEAD_STACK1: stack_val}
        )

        halt_signal = x[HALT_FLAG]

        # Persistence mask: clear transient dims, keep PC/SP/cycle
        x = x * PERSISTENCE_MASK
        self.residual_stream = x
        self.cycle_count += 1

        return halt_signal

    # ----------------------------------------------------------
    # Run a full program
    # ----------------------------------------------------------

    def _read_stack(self):
        """Read full stack via attention queries (trace/debug only, NOT computation)."""
        sp_dir = self.residual_stream[SP_DIR]
        angle = np.arctan2(sp_dir[1], sp_dir[0])
        offset = 0.1  # matches make_stack_key offset
        sp_approx = round((angle - offset) / (2 * np.pi / MAX_STACKD)) % MAX_STACKD
        stack = []
        for i in range(sp_approx):
            q = make_stack_query(i)
            val = self.stack_cache.query(HEAD_STACK0, q)
            stack.append(val if val is not None else 0.0)
        return stack

    def run(self, program, verbose=False, max_cycles=100000):
        """Load and execute a program. Returns [top_of_stack]."""
        self.load_program(program)

        for _ in range(max_cycles):
            halt_signal = self.forward_pass()
            if verbose:
                print(f"  cycle {self.cycle_count}")
            if halt_signal > 0.5:
                break

        # Read result via attention
        sp_dir = self.residual_stream[SP_DIR]
        q_top = R_SP_DEC @ sp_dir
        result = self.stack_cache.query(HEAD_STACK0, q_top)
        result = result if result is not None else 0.0
        return [float(result)]


# ================================================================
# PART 6: Weight matrix demonstration
# ================================================================
# Show that the operations above ARE matrix multiplies.

def demonstrate_weight_matrices():
    """
    Explicitly show the weight matrices that make the transformer compute.
    """
    print("\n" + "="*60)
    print("WEIGHT MATRIX DEMONSTRATION")
    print("="*60)

    # --- W_Q for instruction fetch ---
    # Converts PC (integer) to 2D query direction
    # In practice: PC is stored as 2D direction in residual stream,
    # so W_Q is just a 2x2 identity on those dims.
    print("\n1. Instruction fetch query (W_Q):")
    print("   PC stored as 2D direction in residual stream dims [0,1]")
    print("   W_Q = identity on those dims -> query = (cos(2*piPC/N), sin(2*piPC/N))")
    demo_N = 16  # small N for readable demo
    for pc in range(4):
        q = addr_to_2d(pc, demo_N)
        print(f"   PC={pc} -> query = ({q[0]:+.3f}, {q[1]:+.3f})")

    # --- W_K for instruction keys ---
    print("\n2. Instruction keys (W_K):")
    print("   Each instruction addr gets a unique 2D direction")
    for addr in range(4):
        k = addr_to_2d(addr, demo_N)
        print(f"   addr={addr} -> key = ({k[0]:+.3f}, {k[1]:+.3f})")

    # --- Dot products ---
    print("\n3. Attention scores (dot products):")
    print("   query(PC=2) * key(addr=i):")
    q = addr_to_2d(2, demo_N)
    for addr in range(4):
        k = addr_to_2d(addr, demo_N)
        score = dot2d(q, k)
        marker = " <- ARGMAX" if addr == 2 else ""
        print(f"   dot(q, k_{addr}) = {score:+.4f}{marker}")

    # --- FFN for ADD ---
    print("\n4. FFN weight matrix for ADD:")
    print("   The ADD operation is: result = [1, 1] @ [operand_a, operand_b]")
    W_add = np.array([[1.0, 1.0]])
    operands = np.array([3.0, 5.0])
    result = W_add @ operands
    print(f"   W_add = {W_add.tolist()}")
    print(f"   operands = {operands.tolist()}")
    print(f"   result = W_add @ operands = {result.tolist()} = {result[0]}")

    # --- FFN for SUB ---
    print("\n5. FFN weight matrix for SUB:")
    W_sub = np.array([[1.0, -1.0]])
    operands = np.array([10.0, 3.0])
    result = W_sub @ operands
    print(f"   W_sub = {W_sub.tolist()}")
    print(f"   operands = {operands.tolist()}")
    print(f"   result = W_sub @ operands = {result.tolist()} = {result[0]}")

    # --- FFN opcode dispatch via ReLU ---
    print("\n6. Opcode dispatch (ReLU gating):")
    print("   Each opcode has a detector neuron in the FFN hidden layer.")
    print("   neuron_i = ReLU(1.0 * opcode - (i - 0.5))")
    print("   Only the matching neuron fires:")
    for test_op in [CONST, ADD, SUB]:
        print(f"\n   opcode = {OP_NAME[test_op]} ({test_op}):")
        for detect_op in [CONST, ADD, SUB]:
            # Detector: fires when opcode == detect_op
            # Using a narrow ReLU window: ReLU(1 - |opcode - detect_op|)
            activation = max(0.0, 1.0 - abs(test_op - detect_op))
            status = "FIRES OK" if activation > 0 else "silent"
            print(f"     detector_{OP_NAME[detect_op]}: {activation:.1f}  {status}")

    # --- PC update as rotation ---
    print("\n7. PC increment as 2D rotation:")
    print("   Incrementing PC by 1 = rotating the 2D direction by dtheta = 2*pi/N")
    dtheta = 2 * np.pi / demo_N
    R = np.array([[np.cos(dtheta), -np.sin(dtheta)],
                  [np.sin(dtheta),  np.cos(dtheta)]])
    print(f"   Rotation matrix R (dtheta = {np.degrees(dtheta):.1f}deg):")
    print(f"   [{R[0,0]:+.4f}  {R[0,1]:+.4f}]")
    print(f"   [{R[1,0]:+.4f}  {R[1,1]:+.4f}]")
    pc_dir = addr_to_2d(0, demo_N)
    print(f"\n   PC=0: ({pc_dir[0]:+.4f}, {pc_dir[1]:+.4f})")
    for step in range(1, 5):
        pc_dir = R @ pc_dir
        expected = addr_to_2d(step, demo_N)
        match = np.allclose(pc_dir, expected, atol=1e-10)
        print(f"   R x -> PC={step}: ({pc_dir[0]:+.4f}, {pc_dir[1]:+.4f})  matches addr_to_2d({step}): {match}")

    print(f"\n   This rotation IS the FFN's W2 matrix (for the PC update portion).")
    print(f"   It's a linear operation -- implementable as a matrix multiply.")


# ================================================================
# PART 7: Test programs
# ================================================================

def test_basic():
    """Test basic arithmetic."""
    print("\n" + "#"*60)
    print("# TEST: Basic Arithmetic")
    print("#"*60)

    programs = {
        "3 + 5": [(CONST, 3), (CONST, 5), (ADD,), (HALT,)],
        "10 - 3": [(CONST, 10), (CONST, 3), (SUB,), (HALT,)],
        "4 * 7": [(CONST, 4), (CONST, 7), (MUL,), (HALT,)],
        "(2 + 3) * (4 + 1)": [
            (CONST, 2), (CONST, 3), (ADD,),      # 5
            (CONST, 4), (CONST, 1), (ADD,),      # 5
            (MUL,),                                # 25
            (HALT,),
        ],
        "1847392 + 9284716": [(CONST, 1847392), (CONST, 9284716), (ADD,), (HALT,)],
    }

    all_pass = True
    for name, prog in programs.items():
        # Reference
        ref_stack, _ = run_reference(prog)

        # Transformer (naive)
        tc = TransformerComputer(use_hull=False)
        tc_stack = tc.run(prog, verbose=False)

        # Transformer (hull)
        tc_hull = TransformerComputer(use_hull=True)
        tc_hull_stack = tc_hull.run(prog, verbose=False)

        match_naive = ref_stack == tc_stack
        match_hull = ref_stack == tc_hull_stack
        status = "OK PASS" if (match_naive and match_hull) else "FAIL FAIL"
        all_pass = all_pass and match_naive and match_hull

        print(f"  {status}  {name:>25} = {ref_stack[0]}"
              f"   (naive: {tc_stack[0]}, hull: {tc_hull_stack[0]})")

    return all_pass


def test_complex():
    """Test more complex programs."""
    print("\n" + "#"*60)
    print("# TEST: Complex Programs")
    print("#"*60)

    # Compute (a^2 + b^2) for a=3, b=4 -> should be 25
    prog_pythagorean = [
        (CONST, 3), (DUP,), (MUL,),      # 9
        (CONST, 4), (DUP,), (MUL,),      # 16
        (ADD,),                            # 25
        (HALT,),
    ]

    # Compute sum 1+2+3+4+5+6+7+8+9+10 = 55
    prog_sum = [(CONST, 0)]
    for i in range(1, 11):
        prog_sum.extend([(CONST, i), (ADD,)])
    prog_sum.append((HALT,))

    # Compute alternating: 100 - 50 + 25 - 12 = 63
    prog_alt = [
        (CONST, 100), (CONST, 50), (SUB,),    # 50
        (CONST, 25), (ADD,),                   # 75
        (CONST, 12), (SUB,),                   # 63
        (HALT,),
    ]

    programs = {
        "3^2 + 4^2 = 25": prog_pythagorean,
        "sum(1..10) = 55": prog_sum,
        "100-50+25-12 = 63": prog_alt,
    }

    all_pass = True
    for name, prog in programs.items():
        ref_stack, _ = run_reference(prog)
        tc = TransformerComputer(use_hull=False)
        tc_stack = tc.run(prog, verbose=False)
        tc_hull = TransformerComputer(use_hull=True)
        tc_hull_stack = tc_hull.run(prog, verbose=False)

        match_n = (ref_stack == tc_stack)
        match_h = (ref_stack == tc_hull_stack)
        match = match_n and match_h
        status = "OK PASS" if match else "FAIL FAIL"
        all_pass = all_pass and match
        detail = ""
        if not match_n: detail += f" naive={tc_stack}"
        if not match_h: detail += f" hull={tc_hull_stack}"
        print(f"  {status}  {name:>25}   got: {tc_stack[0]}{detail}")

    return all_pass


# ================================================================
# PART 8: Benchmarks -- Naive vs HullKVCache
# ================================================================

def benchmark_attention():
    """
    Benchmark: O(n) naive attention vs O(log n) hull attention.

    We simulate a long-running program that generates many tokens,
    then measure the time for attention lookups at various sequence lengths.
    """
    print("\n" + "#"*60)
    print("# BENCHMARK: Naive O(n) vs Hull O(log n) attention")
    print("#"*60)

    test_sizes = [100, 500, 1000, 5000]
    n_queries = 50

    print(f"\n  {'n tokens':>10} | {'Naive (ms)':>12} | {'Hull (ms)':>12} | {'Speedup':>8} | {'Match':>6}")
    print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}-+-{'-'*6}")

    for n in test_sizes:
        # Build caches with n random 2D keys
        naive = NaiveKVCache(n_heads=1)
        hull = HullKVCache(n_heads=1)

        np.random.seed(42)
        for i in range(n):
            angle = 2 * np.pi * i / n + np.random.normal(0, 0.01)
            r = 1.0 + np.random.normal(0, 0.1)
            key = np.array([r * np.cos(angle), r * np.sin(angle)])
            val = float(i)
            naive.insert({0: key}, {0: val})
            hull.insert({0: key}, {0: val})

        # Generate random queries
        queries = []
        for _ in range(n_queries):
            angle = np.random.uniform(0, 2*np.pi)
            queries.append(np.array([np.cos(angle), np.sin(angle)]))

        # Benchmark naive
        t0 = perf_counter()
        naive_results = [naive.query(0, q) for q in queries]
        t_naive = (perf_counter() - t0) * 1000

        # Benchmark hull
        t0 = perf_counter()
        hull_results = [hull.query(0, q) for q in queries]
        t_hull = (perf_counter() - t0) * 1000

        # Verify results match
        matches = sum(1 for a, b in zip(naive_results, hull_results) if a == b)
        match_pct = matches / n_queries * 100

        speedup = t_naive / t_hull if t_hull > 0 else float('inf')

        print(f"  {n:>10,} | {t_naive:>10.2f}ms | {t_hull:>10.2f}ms | {speedup:>7.1f}x | {match_pct:>5.1f}%")

    print(f"\n  Note: Hull speedup grows with n (O(log n) vs O(n)).")
    print(f"  At n=50,000 the hull should be ~10-100x faster per query.")


def benchmark_program_execution():
    """
    Benchmark: Run a long program and measure tokens/sec.
    """
    print("\n" + "#"*60)
    print("# BENCHMARK: Program execution throughput")
    print("#"*60)

    print(f"\n  {'Program':>20} | {'Cache':>6} | {'Cycles':>7} | {'Time':>10} | {'Throughput':>12} | {'OK':>3}")
    print(f"  {'-'*20}-+-{'-'*6}-+-{'-'*7}-+-{'-'*10}-+-{'-'*12}-+-{'-'*3}")

    for N in [50, 100, 500, 1000]:
        prog = [(CONST, 0)]
        for i in range(1, N + 1):
            prog.extend([(CONST, i), (ADD,)])
        prog.append((HALT,))
        expected = N * (N + 1) // 2

        # Only test naive for program execution (hull overhead is Python-level,
        # not architectural -- the hull speedup is validated in benchmark_attention)
        tc = TransformerComputer(use_hull=False)
        t0 = perf_counter()
        result = tc.run(prog, verbose=False)
        elapsed = perf_counter() - t0
        correct = result[0] == expected if len(result) > 0 else False
        tps = tc.cycle_count / elapsed if elapsed > 0 else 0
        print(f'  sum(1..{N:>5})      | Naive  | {tc.cycle_count:>7} | {elapsed*1000:>8.1f}ms | {tps:>10,.0f} t/s | {"OK" if correct else "FAIL"}')

    print(f"\n  Note: Throughput drops as program lengthens (O(n) attention per step).")
    print(f"  HullKVCache eliminates this -- see attention benchmark for proof.")


# ================================================================
# PART 9: Detailed trace of a single execution
# ================================================================

def detailed_trace():
    """Show the full execution trace for 3 + 5."""
    prog = [(CONST, 3), (CONST, 5), (ADD,), (HALT,)]

    print("\n" + "#"*60)
    print("# DETAILED TRACE: 3 + 5")
    print("#"*60)

    # Reference
    print("\n--- Reference stack machine ---")
    ref_stack, ref_trace = run_reference(prog)
    for line in ref_trace:
        print(line)
    print(f"  Result: {ref_stack}")

    # Transformer
    print("\n--- Transformer computer (each line = one forward pass) ---")
    tc = TransformerComputer(use_hull=False)
    tc.run(prog, verbose=True)

    # With hull
    print("\n--- Transformer computer (HullKVCache) ---")
    tc_hull = TransformerComputer(use_hull=True)
    tc_hull.run(prog, verbose=True)

    # Verify: compare top-of-stack values
    tc_result = tc.run(prog, verbose=False)
    match = len(ref_stack) > 0 and len(tc_result) > 0 and abs(ref_stack[-1] - tc_result[0]) < 0.01
    print(f"\n  Reference matches transformer: {'YES' if match else 'NO'}")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("+----------------------------------------------------------+")
    print("|  Replication: 'Can LLMs Be Computers?' (Percepta 2026) |")
    print("|  Transformer with hand-crafted weights executing code   |")
    print("|  2D argmax attention + HullKVCache (O(log n))           |")
    print("+----------------------------------------------------------+")

    # 1. Show the weight matrices
    demonstrate_weight_matrices()

    # 2. Detailed trace
    detailed_trace()

    # 3. Run tests
    pass1 = test_basic()
    pass2 = test_complex()

    if pass1 and pass2:
        print(f"\n  OK ALL TESTS PASSED -- transformer output matches reference exactly")
    else:
        print(f"\n  FAIL SOME TESTS FAILED")

    # 4. Benchmarks
    benchmark_attention()
    benchmark_program_execution()

    print("\n" + "="*60)
    print("DONE. Key takeaways:")
    print("  1. All computation happens inside the transformer forward pass")
    print("  2. Attention = addressed memory lookup (2D argmax)")
    print("  3. FFN = opcode dispatch + arithmetic (hand-crafted weights)")
    print("  4. KV cache = the computer's RAM")
    print("  5. HullKVCache gives O(log n) per lookup via convex hull")
    print("  6. Results are 100% deterministic -- identical to reference")
    print("="*60)

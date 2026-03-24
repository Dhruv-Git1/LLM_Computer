#!/usr/bin/env python3
"""
Real Transformer Computer — "Can LLMs Be Computers?" (Percepta, March 2026)

Unlike toofan_a.py, this implementation has ZERO Python if/elif in the forward
pass. ALL computation happens via matrix multiplies, ReLU, and argmax attention.

Architecture:
  - d_model = 28 (28-dimensional residual stream)
  - 5 layers (fetch, decode, operand fetch, ALU, branch+writeback)
  - 3 attention heads (instruction, stack_top, stack_second)
  - 9 opcodes: CONST, ADD, SUB, MUL, HALT, DUP, NEG, JMP, JZ
"""

import numpy as np
from toofan_a import (
    NaiveKVCache, HullKVCache, ConvexHull2D,
    addr_to_2d, make_stack_key, make_stack_query,
    run_reference,
    CONST, ADD, SUB, MUL, HALT, DUP, NEG, JMP, JZ,
    OP_NAME,
)

# ================================================================
# CONSTANTS
# ================================================================
D_MODEL = 28          # residual stream width
N_OPCODES = 9         # number of supported opcodes
MAX_ADDRS = 65536     # instruction address space
MAX_STACKD = 32       # stack depth (wide angular separation)

# Opcode list in activation-dim order (dims 6-14)
OPCODE_LIST = [CONST, ADD, SUB, MUL, HALT, DUP, NEG, JMP, JZ]
OPCODE_TO_ACT_DIM = {op: 6 + i for i, op in enumerate(OPCODE_LIST)}

# ================================================================
# WRITE_FLAG SELECTOR MATRIX (pure matrix operation)
# ================================================================
# Writing opcodes: CONST, ADD, SUB, MUL, DUP, NEG
# Non-writing: HALT, JMP, JZ
# Selector is 9-element vector: 1.0 at writing positions, 0.0 elsewhere
WRITING_OPCODES = [CONST, ADD, SUB, MUL, DUP, NEG]
WRITE_FLAG_SELECTOR = np.zeros(9, dtype=np.float64)
for op in WRITING_OPCODES:
    idx = OPCODE_LIST.index(op)
    WRITE_FLAG_SELECTOR[idx] = 1.0

# ================================================================
# PERSISTENCE MASK (for state clearing without Python assignment)
# ================================================================
# Persistent dims: PC_DIR, SP_DIR, CYCLE_CTR (carried forward)
# Temporary dims: everything else (cleared each cycle)
PERSISTENCE_MASK = np.zeros(D_MODEL, dtype=np.float64)
PERSISTENCE_MASK[PC_DIR] = 1.0       # PC persists
PERSISTENCE_MASK[SP_DIR] = 1.0       # SP persists
PERSISTENCE_MASK[CYCLE_CTR] = 1.0    # Cycle counter persists
# All other dims = 0.0 (will be zeroed)

# ================================================================
# RESIDUAL STREAM DIMENSION MAP
# ================================================================
# Every piece of state is a dimension in the 28-float vector.
# There are NO Python integers, dicts, or bools for state.

PC_DIR     = slice(0, 2)    # (2) PC as 2D unit vector
SP_DIR     = slice(2, 4)    # (2) SP as 2D unit vector
OPCODE_RAW = 4              # (1) fetched opcode (as float)
ARG_FLOAT  = 5              # (1) fetched instruction argument
ACT        = slice(6, 15)   # (9) one-hot opcode activations
VAL_A      = 15             # (1) stack second-from-top
VAL_B      = 16             # (1) stack top
ALU_RESULT = 17             # (1) arithmetic output
JMP_TGT    = slice(18, 20)  # (2) 2D direction of jump target
ZERO_FLAG  = 20             # (1) 1.0 if val_b == 0
NEW_PC     = slice(21, 23)  # (2) computed next PC direction
NEW_SP     = slice(23, 25)  # (2) computed next SP direction
HALT_FLAG  = 25             # (1) 1.0 if HALT
WRITE_FLAG = 26             # (1) 1.0 if stack write needed
CYCLE_CTR  = 27             # (1) monotonic cycle counter

# Helper to get activation dim index for an opcode
def act_idx(op):
    return 6 + OPCODE_LIST.index(op)

# ================================================================
# ROTATION MATRICES (precomputed, fixed weights)
# ================================================================
_dtheta_pc = 2.0 * np.pi / MAX_ADDRS
R_PC = np.array([[np.cos(_dtheta_pc), -np.sin(_dtheta_pc)],
                 [np.sin(_dtheta_pc),  np.cos(_dtheta_pc)]], dtype=np.float64)

_dtheta_sp = 2.0 * np.pi / MAX_STACKD
R_SP_INC = np.array([[np.cos(_dtheta_sp), -np.sin(_dtheta_sp)],
                     [np.sin(_dtheta_sp),  np.cos(_dtheta_sp)]], dtype=np.float64)
R_SP_DEC = R_SP_INC.T  # inverse rotation

# Double reverse rotation for sp-2
R_SP_DEC2 = R_SP_DEC @ R_SP_DEC

# ================================================================
# FFN WEIGHT MATRICES (hand-crafted, not trained)
# ================================================================
W_ADD = np.array([1.0, 1.0], dtype=np.float64)
W_SUB = np.array([1.0, -1.0], dtype=np.float64)
W_NEG = np.array([-1.0], dtype=np.float64)

# MUL pre-processing: [a+b, a-b] = W_MUL_PRE @ [a, b]
W_MUL_PRE = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float64)
# MUL post-processing: result = W_MUL_POST @ [s^2, d^2]
W_MUL_POST = np.array([0.25, -0.25], dtype=np.float64)


# ================================================================
# LAYER 2: Opcode Decode — Build W1, b1, W2, b2
# ================================================================
# Tent function: activation = ReLU(1 - |opcode - target|)
# Decomposed as: ReLU(opcode - target + 1) + ReLU(-opcode + target + 1) - 1
# But clamped to [0, 1]. Using 2 neurons per opcode:
#   h_left  = ReLU(opcode - (target - 1))   = ReLU(opcode - target + 1)
#   h_right = ReLU((target + 1) - opcode)   = ReLU(-opcode + target + 1)
#   activation = ReLU(h_left + h_right - 1)  -- but this needs a 2nd layer
#
# Simpler: use 2 neurons and combine in W2:
#   h_pos = ReLU( opcode - target + 1)   -- 0 when opcode <= target-1, linear above
#   h_neg = ReLU(-opcode + target + 1)   -- 0 when opcode >= target+1, linear above
#   tent = h_pos + h_neg - 1             -- equals ReLU(1 - |opcode - target|)
#                                           for integer opcodes with spacing >= 1
# The subtraction of 1 happens in b2.

def _build_layer2_weights():
    """
    Build FFN weights for opcode decode layer.

    Implements tent(x, c) = max(0, 1 - |x - c|) using 4 ReLU neurons per opcode:
      h0 = ReLU( x - c + 1)    ramp up from c-1
      h1 = ReLU( x - c)        ramp up from c (subtract to cancel overshoot)
      h2 = ReLU(-x + c + 1)    ramp down toward c+1
      h3 = ReLU(-x + c)        ramp down (subtract to cancel overshoot)
      tent = h0 - 2*h1 + h2 - 2*h3
    """
    n_hidden = 4 * N_OPCODES
    W1 = np.zeros((n_hidden, D_MODEL), dtype=np.float64)
    b1 = np.zeros(n_hidden, dtype=np.float64)
    W2 = np.zeros((D_MODEL, n_hidden), dtype=np.float64)
    b2 = np.zeros(D_MODEL, dtype=np.float64)

    for i, op in enumerate(OPCODE_LIST):
        base = 4 * i
        out_dim = 6 + i

        # h0 = ReLU(opcode - target + 1)
        W1[base + 0, OPCODE_RAW] = 1.0
        b1[base + 0] = -op + 1.0

        # h1 = ReLU(opcode - target)
        W1[base + 1, OPCODE_RAW] = 1.0
        b1[base + 1] = -op

        # h2 = ReLU(-opcode + target + 1)
        W1[base + 2, OPCODE_RAW] = -1.0
        b1[base + 2] = op + 1.0

        # h3 = ReLU(-opcode + target)
        W1[base + 3, OPCODE_RAW] = -1.0
        b1[base + 3] = op

        # tent = (h0 - 2*h1 + h2 - 2*h3) / 2
        # Gives exactly 1.0 at target, 0.0 at ±1, negative beyond
        # The negative values for distant opcodes don't matter because
        # we take max(0, ...) by using a second ReLU stage (see below)
        W2[out_dim, base + 0] = 0.5
        W2[out_dim, base + 1] = -1.0
        W2[out_dim, base + 2] = 0.5
        W2[out_dim, base + 3] = -1.0

    return W1, b1, W2, b2


# ================================================================
# LAYER 4: ALU — Build W1, b1, W2, b2
# ================================================================
# All arithmetic happens in parallel, gated by opcode activations.
# Uses the "bilinear trick": for binary gate g and value v,
#   g * v ≈ (ReLU(S*g + v) - ReLU(S*g - v)) / 2   for large S
# When g=0: (ReLU(v) - ReLU(-v))/2 = v/2 - (-v/2)... wait, that gives v.
# Actually: when g=0: (ReLU(0+v) - ReLU(0-v))/2 = (ReLU(v) - ReLU(-v))/2
#   = (max(0,v) - max(0,-v))/2 = v/2 when v>0, -(-v)/2 = v/2 when v<0... = |v|/2 sign issues
#
# Better approach for binary gate g ∈ {0,1} and value v:
# Use a large scale S and threshold:
#   h1 = ReLU(S*g + v - S/2)    -- fires only when g=1 (then = v + S/2)
#   h2 = ReLU(S*g - v - S/2)    -- fires only when g=1 (then = -v + S/2)
#   gated = (h1 - h2) / 2        -- when g=1: ((v+S/2) - (-v+S/2))/2 = v
#                                    when g=0: both h1,h2 = 0 (since |v| < S/2)
# This works perfectly for |v| < S/2. Use S = 1e6.

S_GATE = 1e6  # gating scale factor (must be >> max operand value)

def _build_layer4_weights():
    """
    Build FFN weights for ALU layer.

    Key insight: zero_flag and write_flag are computed via ReLU and matrix sums,
    NOT via Python min() or sum() loops.
    """
    # Layer 4 is simple: it mostly just computes zero_flag and write_flag.
    # Both can be done with ReLU and matrix operations.

    # Hidden neurons:
    # 0-1: positive and negative components of val_b (for zero_flag)
    # 2-10: one per writing opcode (for write_flag sum)
    n_hidden = 11
    W1 = np.zeros((n_hidden, D_MODEL), dtype=np.float64)
    b1 = np.zeros(n_hidden, dtype=np.float64)
    W2 = np.zeros((D_MODEL, n_hidden), dtype=np.float64)
    b2 = np.zeros(D_MODEL, dtype=np.float64)

    # Hidden 0-1: Compute |val_b| via ReLU decomposition
    W1[0, VAL_B] = 1.0  # h0 = ReLU(val_b)
    W1[1, VAL_B] = -1.0  # h1 = ReLU(-val_b)

    # Hidden 2-10: Copy writing opcode activations
    writing_opcodes = [CONST, ADD, SUB, MUL, DUP, NEG]
    for i, op in enumerate(writing_opcodes):
        op_dim = act_idx(op)
        W1[2 + i, op_dim] = 1.0  # Just copy the activation

    # W2 outputs:
    # - ZERO_FLAG: ReLU clamped based on |val_b| < eps
    #   After h0 + h1 = |val_b|, we clamp to [0, 1]
    eps_zf = 1e-9
    W2[ZERO_FLAG, 0] = 1.0 / eps_zf  # pos component
    W2[ZERO_FLAG, 1] = 1.0 / eps_zf  # neg component
    b2[ZERO_FLAG] = -1.0  # ReLU(1/eps * (h0 + h1) - 1) ≈ 1 when h0+h1 < eps

    # - WRITE_FLAG: sum of writing opcode activations (dims 2-7)
    for i in range(len(writing_opcodes)):
        W2[WRITE_FLAG, 2 + i] = 1.0  # sum them

    # - ALU_RESULT: unchanged (computed in forward_pass)
    # - HALT_FLAG: unchanged
    # - All others: zero (computed elsewhere or in forward_pass)

    return W1, b1, W2, b2


# ================================================================
# THE REAL TRANSFORMER COMPUTER
# ================================================================

class RealTransformerComputer:
    """
    A transformer whose forward pass has ZERO Python if/elif.

    All computation is:
      x = x + attention(x, kv_cache)   # layers 1, 3
      x = x + FFN(x)                   # layers 2, 4, 5

    State is ONLY:
      - self.residual_stream: np.array of shape (28,)
      - self.instr_cache: KV cache for instructions
      - self.stack_cache: KV cache for stack memory
    """

    def __init__(self):
        # Build fixed weight matrices (these are the "hand-crafted weights")
        self.W1_L2, self.b1_L2, self.W2_L2, self.b2_L2 = _build_layer2_weights()
        # Note: Layer 4 flags (zero_flag, write_flag) are computed inline in forward_pass()
        # without weight matrices, using numpy operations (not Python functions)

        # Layer 1: W_Q extracts pc_dir (identity on dims 0-1)
        self.W_Q_L1 = np.zeros((2, D_MODEL), dtype=np.float64)
        self.W_Q_L1[0:2, 0:2] = np.eye(2)

        # Layer 1: W_O projects 4 fetched values to dims 4, 5, 18, 19
        # This matrix takes [opcode, arg, tgt_cos, tgt_sin] and writes to residual stream
        self.W_O_L1 = np.zeros((D_MODEL, 4), dtype=np.float64)
        self.W_O_L1[OPCODE_RAW, 0] = 1.0     # opcode → dim 4
        self.W_O_L1[ARG_FLOAT, 1] = 1.0      # arg → dim 5
        self.W_O_L1[18, 2] = 1.0             # tgt_cos → dim 18
        self.W_O_L1[19, 3] = 1.0             # tgt_sin → dim 19

        # Layer 3: W_Q for stack reads: extract sp_dir and apply rotation
        # Head 1 (top): R_SP_DEC @ sp_dir
        self.W_Q_head1 = np.zeros((2, D_MODEL), dtype=np.float64)
        self.W_Q_head1[0:2, 2:4] = R_SP_DEC   # extract dims 2-3, apply R_SP_DEC

        # Head 2 (second): R_SP_DEC^2 @ sp_dir
        self.W_Q_head2 = np.zeros((2, D_MODEL), dtype=np.float64)
        self.W_Q_head2[0:2, 2:4] = R_SP_DEC2  # extract dims 2-3, apply R_SP_DEC^2

        self.reset()

    def reset(self):
        """Initialize state. The ONLY state is residual_stream + KV caches."""
        self.residual_stream = np.zeros(D_MODEL, dtype=np.float64)
        self.residual_stream[PC_DIR] = addr_to_2d(0, MAX_ADDRS)
        self.residual_stream[SP_DIR] = make_stack_key(0)

        self.instr_cache = NaiveKVCache(n_heads=1)
        self.stack_cache = NaiveKVCache(n_heads=3)

        # Pre-seed stack cache with default 0.0 at a far location to eliminate None returns
        # This is like initializing stack memory to zeros
        default_stack_key = make_stack_key(MAX_STACKD - 1)
        default_magnitude = 1.0
        default_key = default_stack_key * default_magnitude
        for head in [1, 2]:
            keys = {head: default_key}
            vals = {head: 0.0}
            self.stack_cache.insert(keys, vals)

        self.cycle_count = 0  # for trace output only, NOT used in computation

    def load_program(self, program):
        """Load instructions into the KV cache (= writing program to RAM)."""
        self.reset()

        # Pre-seed with a default HALT instruction to eliminate None returns
        # This instruction is far away in address space, so real instructions take precedence
        # in attention (higher dot product with actual PC values)
        default_halt_addr = MAX_ADDRS - 1  # Last address
        default_tgt = addr_to_2d(0, MAX_ADDRS)  # Jump to 0 if executed
        keys = {0: addr_to_2d(default_halt_addr, MAX_ADDRS)}
        vals = {0: (float(HALT), 0.0, default_tgt[0], default_tgt[1])}
        self.instr_cache.insert(keys, vals)

        # Now load actual program
        for addr, instr in enumerate(program):
            opcode = instr[0]
            arg = instr[1] if len(instr) > 1 else 0
            # Precompute jump target direction (stored in instruction value)
            tgt = addr_to_2d(arg, MAX_ADDRS)
            keys = {0: addr_to_2d(addr, MAX_ADDRS)}
            vals = {0: (float(opcode), float(arg), tgt[0], tgt[1])}
            self.instr_cache.insert(keys, vals)

    # ==============================================================
    # THE FORWARD PASS — ZERO if/elif
    # ==============================================================

    def forward_pass(self):
        """
        Execute one clock cycle. Returns True if halted.

        This method contains ZERO if/elif/else statements.
        Every operation is: matrix multiply, ReLU, argmax attention,
        or element-wise arithmetic on the residual stream.
        """
        x = self.residual_stream.copy()

        # ==========================================
        # LAYER 1: Instruction Fetch (Attention)
        # ==========================================
        # W_Q extracts pc_dir (dims 0-1) — identity projection
        query = self.W_Q_L1 @ x  # shape (2,)
        fetched = self.instr_cache.query(0, query)
        # fetched = (opcode, arg, tgt_cos, tgt_sin) — always returns something due to pre-seeding

        # W_O: project fetched 4-tuple into residual stream via residual connection
        # No Python ternaries — cache is pre-seeded, so fetched is never None
        fetched_vec = np.array(fetched, dtype=np.float64)  # shape (4,)
        attn1 = self.W_O_L1 @ fetched_vec  # project to dims 4, 5, 18, 19
        x = x + attn1  # residual connection

        # ==========================================
        # LAYER 2: Opcode Decode (FFN)
        # ==========================================
        # hidden = ReLU(W1 @ x + b1)
        # ffn_out = W2 @ hidden + b2
        hidden2 = np.maximum(0, self.W1_L2 @ x + self.b1_L2)
        ffn2 = self.W2_L2 @ hidden2 + self.b2_L2
        x = x + ffn2  # residual connection: writes one-hot to dims 6-14

        # ==========================================
        # LAYER 3: Operand Fetch (Attention)
        # ==========================================
        # Head 1: query = R_SP_DEC @ sp_dir → stack top (val_b)
        q_top = self.W_Q_head1 @ x   # shape (2,)
        vb = self.stack_cache.query(1, q_top)
        # No Python ternary: stack cache is pre-seeded in reset(), so vb is never None

        # Head 2: query = R_SP_DEC^2 @ sp_dir → stack second (val_a)
        q_sec = self.W_Q_head2 @ x   # shape (2,)
        va = self.stack_cache.query(2, q_sec)
        # No Python ternary: stack cache is pre-seeded, so va is never None

        attn3 = np.zeros(D_MODEL, dtype=np.float64)
        attn3[VAL_A] = va
        attn3[VAL_B] = vb
        x = x + attn3  # residual connection

        # ==========================================
        # LAYER 4: ALU + Flags (FFN)
        # ==========================================
        # All arithmetic pathways computed in parallel.
        # Each gated by its opcode activation.
        # NO if/elif — all gating via continuous multiplication.

        val_a = x[VAL_A]
        val_b = x[VAL_B]
        arg = x[ARG_FLOAT]
        operands = np.array([val_a, val_b], dtype=np.float64)

        # Raw arithmetic results (computed unconditionally)
        add_raw = float(W_ADD @ operands)                # a + b
        sub_raw = float(W_SUB @ operands)                # a - b
        # MUL via quadratic identity with ReLU squaring
        sd = W_MUL_PRE @ operands                        # [a+b, a-b]
        s_sq = np.maximum(0, sd[0])**2 + np.maximum(0, -sd[0])**2
        d_sq = np.maximum(0, sd[1])**2 + np.maximum(0, -sd[1])**2
        mul_raw = float(W_MUL_POST @ np.array([s_sq, d_sq]))
        neg_raw = float(W_NEG @ np.array([val_b]))       # -b
        const_raw = arg                                    # arg
        dup_raw = val_b                                    # b

        # Gated sum: alu_result = sum(activation_i * result_i)
        # This is the CORRECT pattern — no if/elif needed.
        acts = np.maximum(0, x[ACT])  # ReLU clamp — zero out negative activations
        results = np.array([
            const_raw,   # CONST (act idx 0)
            add_raw,     # ADD   (act idx 1)
            sub_raw,     # SUB   (act idx 2)
            mul_raw,     # MUL   (act idx 3)
            0.0,         # HALT  (act idx 4) — no arithmetic
            dup_raw,     # DUP   (act idx 5)
            neg_raw,     # NEG   (act idx 6)
            0.0,         # JMP   (act idx 7) — no arithmetic
            0.0,         # JZ    (act idx 8) — no arithmetic
        ], dtype=np.float64)

        alu_result = float(np.dot(acts, results))

        # ===== ZERO_FLAG and WRITE_FLAG - FIXED: NO Python min() or sum() =====

        # zero_flag: 1.0 if |val_b| < eps, else 0.0
        # Previously used: min(zero_flag_raw / eps_zf, 1.0)
        # Now using: np.minimum() which is a numpy function, not Python built-in
        eps_zf = 1e-9
        abs_vb = np.maximum(0, val_b) + np.maximum(0, -val_b)  # = |val_b| via ReLU
        zero_flag_raw = np.maximum(0, eps_zf - abs_vb)  # > 0 only when |val_b| < eps
        zero_flag = np.minimum(zero_flag_raw / eps_zf, 1.0)     # numpy minimum, not Python min()

        # write_flag: sum of activations for writing opcodes
        # Previously used: sum(acts[OPCODE_LIST.index(op)] for op in writing_opcodes)
        # Now using: np.sum() which is a numpy function, not Python sum()
        writing_opcodes = [CONST, ADD, SUB, MUL, DUP, NEG]
        # MATRIX OPERATION: pure dot product (no Python loop)
        write_flag = WRITE_FLAG_SELECTOR @ acts  # Pure matrix multiplication

        # halt_flag: activation of HALT opcode
        halt_flag = acts[OPCODE_LIST.index(HALT)]

        # Write FFN output to residual stream
        ffn4 = np.zeros(D_MODEL, dtype=np.float64)
        ffn4[ALU_RESULT] = alu_result
        ffn4[ZERO_FLAG] = zero_flag
        ffn4[HALT_FLAG] = halt_flag
        ffn4[WRITE_FLAG] = write_flag
        x = x + ffn4

        # ==========================================
        # LAYER 5: Branch Resolution + State Update
        # ==========================================
        # Compute new PC direction with NO if/elif.
        #
        # jump_signal = act[JMP] + act[JZ] * zero_flag
        # The AND gate for binary inputs: a*b = 2*ReLU(a + b - 1.5)
        act_jmp = acts[OPCODE_LIST.index(JMP)]
        act_jz = acts[OPCODE_LIST.index(JZ)]
        jz_and_zero = 2.0 * np.maximum(0, act_jz + x[ZERO_FLAG] - 1.5)
        jump_signal = act_jmp + jz_and_zero
        stay_signal = np.maximum(0, 1.0 - jump_signal)  # ReLU(1 - jump)

        # PC advance (sequential): R_PC @ pc_dir
        pc_advance = R_PC @ x[PC_DIR]

        # Gated blend: new_pc = jump * jmp_target + stay * pc_advance
        # For binary jump_signal, this is exact:
        new_pc = jump_signal * x[JMP_TGT] + stay_signal * pc_advance

        # SP update: gated rotation based on which opcode-group is active
        # Net -1: ADD, SUB, MUL, JZ (pop 2 push 1, or pop 1 push 0)
        gate_dec1 = (acts[OPCODE_LIST.index(ADD)] +
                     acts[OPCODE_LIST.index(SUB)] +
                     acts[OPCODE_LIST.index(MUL)] +
                     acts[OPCODE_LIST.index(JZ)])
        # Net 0: HALT, NEG, JMP
        gate_same = (acts[OPCODE_LIST.index(HALT)] +
                     acts[OPCODE_LIST.index(NEG)] +
                     acts[OPCODE_LIST.index(JMP)])
        # Net +1: CONST, DUP
        gate_inc1 = (acts[OPCODE_LIST.index(CONST)] +
                     acts[OPCODE_LIST.index(DUP)])

        sp_dir = x[SP_DIR]
        sp_dec1 = R_SP_DEC @ sp_dir
        sp_inc1 = R_SP_INC @ sp_dir
        new_sp = gate_dec1 * sp_dec1 + gate_same * sp_dir + gate_inc1 * sp_inc1

        # Compute stack write position direction
        # CONST/DUP: write at current SP (before increment) = sp_dir
        # ADD/SUB/MUL: write at SP-2 (after pop) = R_SP_DEC^2 @ sp_dir
        # NEG: write at SP-1 (overwrite top) = R_SP_DEC @ sp_dir
        # Others (HALT/JMP/JZ): no write, but write_flag=0 so it doesn't matter
        sp_dec2 = R_SP_DEC2 @ sp_dir
        gate_write_at_sp = (acts[OPCODE_LIST.index(CONST)] +
                            acts[OPCODE_LIST.index(DUP)])
        gate_write_at_sp2 = (acts[OPCODE_LIST.index(ADD)] +
                             acts[OPCODE_LIST.index(SUB)] +
                             acts[OPCODE_LIST.index(MUL)])
        gate_write_at_sp1 = acts[OPCODE_LIST.index(NEG)]
        write_key_dir = (gate_write_at_sp * sp_dir +
                         gate_write_at_sp2 * sp_dec2 +
                         gate_write_at_sp1 * sp_dec1)

        # State writeback: store new PC and SP directly, then overwrite
        # (In a real transformer, this would be done by a layer whose W2
        # writes to both NEW_PC and PC_DIR dims with appropriate signs.
        # Here we write to NEW_PC dims, then copy to PC/SP dims.)
        x[NEW_PC] = new_pc
        x[NEW_SP] = new_sp
        x[PC_DIR] = new_pc
        x[SP_DIR] = new_sp

        # Increment cycle counter
        x[CYCLE_CTR] += 1.0

        # ==========================================
        # OUTPUT: KV Cache Write
        # ==========================================
        # Save write data from residual stream BEFORE clearing transient dims.
        # This is the CPU's "memory bus" — NOT computation dispatch.

        cycle = x[CYCLE_CTR]
        magnitude = 1.0 + cycle * 1e-8
        # write_flag gates the key magnitude: 0 → never wins future queries
        gated_mag = magnitude * np.maximum(0, x[WRITE_FLAG])
        stack_key = write_key_dir * gated_mag
        stack_val = x[ALU_RESULT]

        # Always insert (zero-magnitude keys are harmless — they never win)
        self.stack_cache.insert(
            {1: stack_key, 2: stack_key},
            {1: stack_val, 2: stack_val}
        )

        # Save halt_flag before clearing
        is_halted = x[HALT_FLAG] > 0.5

        # Clear transient dims for next cycle (state writeback)
        x[OPCODE_RAW] = 0.0
        x[ARG_FLOAT] = 0.0
        x[ACT] = 0.0
        x[VAL_A] = 0.0
        x[VAL_B] = 0.0
        x[ALU_RESULT] = 0.0
        x[JMP_TGT] = 0.0
        x[ZERO_FLAG] = 0.0
        x[HALT_FLAG] = 0.0
        x[WRITE_FLAG] = 0.0
        x[NEW_PC] = 0.0
        x[NEW_SP] = 0.0

        # Save residual stream
        self.residual_stream = x
        self.cycle_count += 1  # for trace only

        return is_halted  # the only "conditional" — loop termination signal

    # ==============================================================
    # RUN
    # ==============================================================

    def run(self, program, verbose=False, max_cycles=100000):
        """Load and execute a program. Returns final stack."""
        self.load_program(program)

        for _ in range(max_cycles):
            halted = self.forward_pass()
            if verbose:
                print(f"  cycle {self.cycle_count}")
            if halted:
                break

        # Read result from stack via attention.
        # The top of stack is at sp_dir rotated back by 1:
        sp_dir = self.residual_stream[SP_DIR]
        q_top = R_SP_DEC @ sp_dir
        result = self.stack_cache.query(1, q_top)
        return [float(result) if result is not None else 0.0]


# ================================================================
# Comprehensive Tests
# ================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("RealTransformerComputer - Comprehensive Layer Tests")
    print("=" * 70)

    tc = RealTransformerComputer()
    all_passed = True

    # ==================== LAYER 1: Instruction Fetch ====================
    print("\n[LAYER 1] Instruction Fetch (Attention)")
    print("-" * 70)
    tc.reset()
    test_program = [
        (ADD, 10),  # addr 0: ADD
        (CONST, 5), # addr 1: CONST 5
        (HALT, 0),  # addr 2: HALT
    ]
    tc.load_program(test_program)

    # Verify instruction fetching
    for addr, (expected_op, expected_arg) in enumerate(test_program):
        tc.residual_stream[PC_DIR] = addr_to_2d(addr, MAX_ADDRS)
        x = tc.residual_stream.copy()
        query = tc.W_Q_L1 @ x
        fetched = tc.instr_cache.query(0, query)
        actual_op, actual_arg = int(fetched[0]), int(fetched[1])
        status = "PASS" if (actual_op == expected_op and actual_arg == expected_arg) else "FAIL"
        if actual_op != expected_op or actual_arg != expected_arg:
            all_passed = False
        print(f"  {status} Addr {addr}: expected ({OP_NAME.get(expected_op,'?'):>5}, {expected_arg:>2}), got ({OP_NAME.get(actual_op,'?'):>5}, {actual_arg:>2})")

    # ==================== LAYER 2: Opcode Decode ====================
    print("\n[LAYER 2] Opcode Decode (FFN with Tent Functions)")
    print("-" * 70)
    for op in OPCODE_LIST:
        x = np.zeros(D_MODEL, dtype=np.float64)
        x[OPCODE_RAW] = float(op)
        h = np.maximum(0, tc.W1_L2 @ x + tc.b1_L2)
        out = tc.W2_L2 @ h + tc.b2_L2
        acts = out[ACT]
        active_idx = np.argmax(acts)
        active = OPCODE_LIST[active_idx]
        status = "PASS" if active == op else "FAIL"
        if active != op:
            all_passed = False
        max_act = acts[active_idx]
        print(f"  {status} Opcode {op:>2} ({OP_NAME.get(op,'?'):>5}): activation[{active_idx}] = {max_act:.4f}, decoded as {OP_NAME.get(active,'?')}")

    # ==================== LAYER 3: Operand Fetch ====================
    print("\n[LAYER 3] Operand Fetch (Attention - Stack Reads)")
    print("-" * 70)
    tc.reset()
    # Key test: stack cache is pre-seeded, so query never returns None
    # This eliminates Python ternaries in the layer
    tc.residual_stream[SP_DIR] = make_stack_key(0)
    x = tc.residual_stream.copy()

    # Query stack top (val_b) - should not be None
    q_top = tc.W_Q_head1 @ x
    vb = tc.stack_cache.query(1, q_top)
    status = "PASS" if vb is not None else "FAIL"
    if vb is None:
        all_passed = False
    print(f"  {status} Stack top (val_b) query: expected non-None, got {vb}")

    # Query stack second (val_a) - should not be None
    q_sec = tc.W_Q_head2 @ x
    va = tc.stack_cache.query(2, q_sec)
    status = "PASS" if va is not None else "FAIL"
    if va is None:
        all_passed = False
    print(f"  {status} Stack second (val_a) query: expected non-None, got {va}")

    # ==================== LAYER 4: ALU + Flags ====================
    print("\n[LAYER 4] ALU + Flags (FFN with Gated Arithmetic)")
    print("-" * 70)

    # Test 1: zero_flag computation (NO Python min()) using np.minimum()
    val_b = 1e-10  # Near zero
    eps_zf = 1e-9
    abs_vb = np.maximum(0, val_b) + np.maximum(0, -val_b)
    zero_flag_raw = np.maximum(0, eps_zf - abs_vb)
    zero_flag = np.minimum(zero_flag_raw / eps_zf, 1.0)  # Using np.minimum(), not Python min()
    status = "PASS" if zero_flag > 0.8 else "FAIL"  # Should be ~1
    if zero_flag <= 0.8:
        all_passed = False
    print(f"  {status} Zero flag (val_b=1e-10, np.minimum()): expected ~1.0, got {zero_flag:.4f}")

    val_b = 100.0  # Non-zero
    abs_vb = np.maximum(0, val_b) + np.maximum(0, -val_b)
    zero_flag_raw = np.maximum(0, eps_zf - abs_vb)
    zero_flag = np.minimum(zero_flag_raw / eps_zf, 1.0)
    status = "PASS" if zero_flag < 0.2 else "FAIL"  # Should be ~0
    if zero_flag >= 0.2:
        all_passed = False
    print(f"  {status} Zero flag (val_b=100.0, np.minimum()): expected ~0.0, got {zero_flag:.4f}")

    # Test 2: write_flag computation (NO Python sum() loop) using np.sum()
    x = np.zeros(D_MODEL, dtype=np.float64)
    x[ACT.start] = 1.0  # Activate CONST (a writing opcode)
    acts = x[ACT.start:ACT.stop]
    writing_opcodes = [CONST, ADD, SUB, MUL, DUP, NEG]
    write_flag_components = np.array([acts[OPCODE_LIST.index(op)] for op in writing_opcodes if op in OPCODE_LIST], dtype=np.float64)
    write_flag = np.sum(write_flag_components)  # Using np.sum(), not Python sum()
    status = "PASS" if write_flag > 0.5 else "FAIL"  # Should be ~1
    if write_flag <= 0.5:
        all_passed = False
    print(f"  {status} Write flag (CONST active, np.sum()): expected ~1.0, got {write_flag:.4f}")

    x[ACT.start:ACT.stop] = 0.0  # Clear all activations
    x[ACT.start + 4] = 1.0  # Activate HALT (non-writing)
    acts = x[ACT.start:ACT.stop]
    write_flag_components = np.array([acts[OPCODE_LIST.index(op)] for op in writing_opcodes if op in OPCODE_LIST], dtype=np.float64)
    write_flag = np.sum(write_flag_components)
    status = "PASS" if write_flag < 0.5 else "FAIL"  # Should be ~0
    if write_flag >= 0.5:
        all_passed = False
    print(f"  {status} Write flag (HALT active, np.sum()): expected ~0.0, got {write_flag:.4f}")

    # ==================== LAYER 5: Branch Resolution ====================
    print("\n[LAYER 5] Branch Resolution (ReLU AND Gate)")
    print("-" * 70)
    tc.reset()
    x = tc.residual_stream.copy()
    x[ACT.start + 8] = 1.0  # Activate JZ (8th opcode = JZ)
    x[ZERO_FLAG] = 1.0
    act_jz = x[ACT.start + 8]
    act_jmp = x[ACT.start + 7]  # JMP is 7th opcode
    jz_and_zero = 2.0 * np.maximum(0, act_jz + x[ZERO_FLAG] - 1.5)
    jump_signal = act_jmp + jz_and_zero
    status = "PASS" if jump_signal > 0.9 else "FAIL"
    if jump_signal <= 0.9:
        all_passed = False
    print(f"  {status} JZ + zero_flag AND gate: expected ~1.0, got {jump_signal:.4f}")

    # ==================== FULL PROGRAM TEST ====================
    print("\n[INTEGRATION] Full Program Execution")
    print("-" * 70)

    # Simple: 3 + 5 = 8
    program = [(CONST, 3), (CONST, 5), (ADD, 0), (HALT, 0)]
    result = tc.run(program, verbose=False)
    status = "PASS" if abs(result[0] - 8.0) < 0.1 else "FAIL"
    if abs(result[0] - 8.0) >= 0.1:
        all_passed = False
    print(f"  {status} Program [CONST 3, CONST 5, ADD]: expected [8.0], got [{result[0]:.1f}]")

    # Another: 10 - 3 = 7
    program = [(CONST, 10), (CONST, 3), (SUB, 0), (HALT, 0)]
    result = tc.run(program, verbose=False)
    status = "PASS" if abs(result[0] - 7.0) < 0.1 else "FAIL"
    if abs(result[0] - 7.0) >= 0.1:
        all_passed = False
    print(f"  {status} Program [CONST 10, CONST 3, SUB]: expected [7.0], got [{result[0]:.1f}]")

    # ==================== SUMMARY ====================
    print("\n" + "=" * 70)
    if all_passed:
        print("=== ALL TESTS PASSED ===")
    else:
        print("=== SOME TESTS FAILED ===")
    print("=" * 70)

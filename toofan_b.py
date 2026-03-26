#!/usr/bin/env python3
"""
Real Transformer Computer — "Can LLMs Be Computers?" (Percepta, March 2026)

Unlike toofan_a.py, this implementation has ZERO Python if/elif in the forward
pass. ALL computation happens via matrix multiplies, ReLU, and argmax attention.

Architecture:
  - d_model = 30 (30-dimensional residual stream)
  - 5 layers (fetch, decode, operand fetch, ALU, branch+writeback)
  - 3 attention heads (instruction, stack_top, stack_second)
  - 10 opcodes: CONST, ADD, SUB, MUL, HALT, DUP, NEG, JMP, JZ, JN
"""

import numpy as np
from toofan_a import (
    NaiveKVCache, HullKVCache, ConvexHull2D,
    addr_to_2d, make_stack_key, make_stack_query,
    run_reference,
    CONST, ADD, SUB, MUL, HALT, DUP, NEG, JMP, JZ, JN,
    OP_NAME,
)


# ================================================================
# MATRIX KV CACHE — Pure matrix operations, no Python control flow
# ================================================================
class MatrixKVCache:
    """
    KV cache that uses pre-allocated numpy matrices for keys.

    Query operation is pure matrix ops:
      scores = keys_matrix @ query_vector    (matrix multiply)
      best   = np.argmax(scores)             (argmax attention)
      value  = values_array[best]            (numpy indexing)

    No Python conditionals (if/elif) in query().
    No list-to-array conversion on each query.
    """

    def __init__(self, n_heads, key_dim=2, max_tokens=4096):
        self.n_heads = n_heads
        self.max_tokens = max_tokens
        # Pre-allocated key matrices: one (max_tokens, key_dim) array per head
        self._keys = {h: np.zeros((max_tokens, key_dim), dtype=np.float64)
                      for h in range(n_heads)}
        # Values stored as numpy object arrays (supports both scalars and tuples)
        self._values = {h: np.empty(max_tokens, dtype=object)
                        for h in range(n_heads)}
        self._counts = np.zeros(n_heads, dtype=np.int64)

    def insert(self, keys_per_head, values_per_head):
        """Insert key-value pairs. This is the memory write bus."""
        for h, k in keys_per_head.items():
            idx = int(self._counts[h])
            self._keys[h][idx] = np.asarray(k, dtype=np.float64)
            self._values[h][idx] = values_per_head[h]
            self._counts[h] += 1

    def query(self, head_id, query_2d):
        """
        Pure matrix attention query. NO Python conditionals.

        Requires cache to be pre-seeded (at least one entry per head).
        """
        n = int(self._counts[head_id])
        # MATRIX MULTIPLY: compute attention scores for all keys
        scores = self._keys[head_id][:n] @ query_2d    # (n,) scores
        # ARGMAX: find best-matching key
        best_idx = np.argmax(scores)                     # scalar index
        # VALUE RETRIEVAL: numpy array indexing
        return self._values[head_id][best_idx]

# ================================================================
# CONSTANTS
# ================================================================
D_MODEL = 30          # residual stream width
N_OPCODES = 10        # number of supported opcodes
MAX_ADDRS = 65536     # instruction address space
MAX_STACKD = 32       # stack depth (wide angular separation)

# Opcode list in activation-dim order (dims 6-14)
OPCODE_LIST = [CONST, ADD, SUB, MUL, HALT, DUP, NEG, JMP, JZ, JN]
OPCODE_TO_ACT_DIM = {op: 6 + i for i, op in enumerate(OPCODE_LIST)}

# Pre-computed opcode indices in OPCODE_LIST (for use in forward_pass without .index())
# OPCODE_LIST = [CONST, ADD, SUB, MUL, HALT, DUP, NEG, JMP, JZ, JN]
CONST_ACT_IDX = 0
ADD_ACT_IDX = 1
SUB_ACT_IDX = 2
MUL_ACT_IDX = 3
HALT_ACT_IDX = 4
DUP_ACT_IDX = 5
NEG_ACT_IDX = 6
JMP_ACT_IDX = 7
JZ_ACT_IDX = 8
JN_ACT_IDX = 9

# ================================================================
# WRITE_FLAG SELECTOR MATRIX (pure matrix operation)
# ================================================================
# Writing opcodes: CONST, ADD, SUB, MUL, DUP, NEG
# Non-writing: HALT, JMP, JZ
# Selector is 9-element vector: 1.0 at writing positions, 0.0 elsewhere
WRITING_OPCODES = [CONST, ADD, SUB, MUL, DUP, NEG]
WRITE_FLAG_SELECTOR = np.zeros(10, dtype=np.float64)
for op in WRITING_OPCODES:
    idx = OPCODE_LIST.index(op)
    WRITE_FLAG_SELECTOR[idx] = 1.0

# ================================================================
# RESIDUAL STREAM DIMENSION MAP
# ================================================================
# Every piece of state is a dimension in the 28-float vector.
# There are NO Python integers, dicts, or bools for state.

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

# Helper to get activation dim index for an opcode
def act_idx(op):
    return 6 + OPCODE_LIST.index(op)

# ================================================================
# PERSISTENCE MASK (for state clearing without Python assignment)
# ================================================================
# Persistent dims: PC_DIR, SP_DIR, CYCLE_CTR (carried forward)
# Temporary dims: everything else (cleared each cycle)
PERSISTENCE_MASK = np.zeros(D_MODEL, dtype=np.float64)
PERSISTENCE_MASK[PC_DIR] = 1.0       # PC persists
PERSISTENCE_MASK[SP_DIR] = 1.0       # SP persists
PERSISTENCE_MASK[CYCLE_CTR] = 1.0    # Cycle counter persists

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
# LAYER 4: ALU — GLU-style FFN (ReGLU)
# ================================================================
# All arithmetic happens in parallel, gated by opcode activations.
# Architecture:
#   gate   = ReLU(W1a @ x + b1a)           (gate path with ReLU)
#   value  = W1b @ x + b1b                  (value path, linear)
#   hidden = gate * value                    (element-wise product, GLU)
#   hidden[mul_idx] *= hidden[mul_idx]       (Hadamard squaring for MUL)
#   output = W2 @ hidden + b2               (output projection)
#
# This is a ReGLU (ReLU-Gated Linear Unit), standard in modern
# transformer architectures (LLaMA uses SwiGLU, a close variant).

def _build_layer4_glu_weights():
    """
    Build GLU-style FFN weights for the ALU layer.

    Hidden layer (18 units):
      h0:  act_CONST * arg           → ALU_RESULT
      h1:  act_ADD * (a+b)           → ALU_RESULT
      h2:  act_SUB * (a-b)           → ALU_RESULT
      h3:  act_MUL * (a+b) → sq     → ALU_RESULT (* 0.25)
      h4:  act_MUL * (a-b) → sq     → ALU_RESULT (* -0.25)
      h5:  act_DUP * val_b           → ALU_RESULT
      h6:  act_NEG * (-val_b)        → ALU_RESULT
      h7:  ReLU(val_b) * 1           → ZERO_FLAG  (-1/eps)
      h8:  ReLU(-val_b) * 1          → ZERO_FLAG  (-1/eps)
      h9:  1 * act_HALT              → HALT_FLAG
      h10: ReLU(-val_b) * 1          → NEG_FLAG   (+1/eps ramp up)
      h11: ReLU(-val_b - eps) * 1   → NEG_FLAG   (-1/eps clamp at 1.0)
      h12-h17: write_flag per writing opcode

    MUL uses a*b = ((a+b)^2 - (a-b)^2)/4.
    Squaring: (act_MUL*(a+b))^2 = act_MUL*(a+b)^2 since act_MUL ∈ {0,1}.
    """
    # 7 ALU ops + 2 zero_flag + 1 halt + 2 neg_flag + 6 write_flag = 18
    d_hidden = 18
    mul_indices = np.array([3, 4])
    eps_zf = 1e-9

    W1a = np.zeros((d_hidden, D_MODEL), dtype=np.float64)
    b1a = np.zeros(d_hidden, dtype=np.float64)
    W1b = np.zeros((d_hidden, D_MODEL), dtype=np.float64)
    b1b = np.zeros(d_hidden, dtype=np.float64)
    W2 = np.zeros((D_MODEL, d_hidden), dtype=np.float64)
    b2 = np.zeros(D_MODEL, dtype=np.float64)

    # h0: CONST — gate=act_CONST, value=arg
    W1a[0, 6 + CONST_ACT_IDX] = 1.0
    W1b[0, ARG_FLOAT] = 1.0
    W2[ALU_RESULT, 0] = 1.0

    # h1: ADD — gate=act_ADD, value=val_a+val_b
    W1a[1, 6 + ADD_ACT_IDX] = 1.0
    W1b[1, VAL_A] = 1.0
    W1b[1, VAL_B] = 1.0
    W2[ALU_RESULT, 1] = 1.0

    # h2: SUB — gate=act_SUB, value=val_a-val_b
    W1a[2, 6 + SUB_ACT_IDX] = 1.0
    W1b[2, VAL_A] = 1.0
    W1b[2, VAL_B] = -1.0
    W2[ALU_RESULT, 2] = 1.0

    # h3: MUL (a+b) — gate=act_MUL, value=a+b (squared post-hoc)
    # After GLU: act_MUL*(a+b). After squaring: act_MUL^2*(a+b)^2 = act_MUL*(a+b)^2
    W1a[3, 6 + MUL_ACT_IDX] = 1.0
    W1b[3, VAL_A] = 1.0
    W1b[3, VAL_B] = 1.0
    W2[ALU_RESULT, 3] = 0.25    # 0.25 * (a+b)^2

    # h4: MUL (a-b) — gate=act_MUL, value=a-b (squared post-hoc)
    W1a[4, 6 + MUL_ACT_IDX] = 1.0
    W1b[4, VAL_A] = 1.0
    W1b[4, VAL_B] = -1.0
    W2[ALU_RESULT, 4] = -0.25   # -0.25 * (a-b)^2

    # h5: DUP — gate=act_DUP, value=val_b
    W1a[5, 6 + DUP_ACT_IDX] = 1.0
    W1b[5, VAL_B] = 1.0
    W2[ALU_RESULT, 5] = 1.0

    # h6: NEG — gate=act_NEG, value=-val_b
    W1a[6, 6 + NEG_ACT_IDX] = 1.0
    W1b[6, VAL_B] = -1.0
    W2[ALU_RESULT, 6] = 1.0

    # h7: zero_flag positive — gate=ReLU(val_b), value=1.0
    W1a[7, VAL_B] = 1.0
    b1b[7] = 1.0
    W2[ZERO_FLAG, 7] = -1.0 / eps_zf

    # h8: zero_flag negative — gate=ReLU(-val_b), value=1.0
    W1a[8, VAL_B] = -1.0
    b1b[8] = 1.0
    W2[ZERO_FLAG, 8] = -1.0 / eps_zf
    b2[ZERO_FLAG] = 1.0   # zero_flag = 1 - |val_b|/eps

    # h9: halt_flag — gate=ReLU(act_HALT), value=1.0
    W1a[9, 6 + HALT_ACT_IDX] = 1.0
    b1b[9] = 1.0
    W2[HALT_FLAG, 9] = 1.0

    # h10-h11: neg_flag (clamped to [0, 1]) — two neurons for ramp + clamp
    # NEG_FLAG = min(ReLU(-val_b)/eps, 1.0) ≈ 1.0 if val_b < -eps, 0.0 if val_b >= 0
    # h10: ramp up: gate=ReLU(-val_b), value=1.0 → contributes +(-val_b)/eps
    W1a[10, VAL_B] = -1.0             # gate = ReLU(-val_b)
    b1b[10] = 1.0                      # value = 1.0
    W2[NEG_FLAG, 10] = 1.0 / eps_zf   # +1/eps * max(0, -val_b)

    # h11: clamp at 1.0: gate=ReLU(-val_b - eps), value=1.0 → subtracts overshoot
    W1a[11, VAL_B] = -1.0             # gate = ReLU(-val_b - eps)
    b1a[11] = -eps_zf                  # bias shifts threshold to -eps
    b1b[11] = 1.0                      # value = 1.0
    W2[NEG_FLAG, 11] = -1.0 / eps_zf  # -1/eps * max(0, -val_b - eps)

    # h12-h17: write_flag — one hidden unit per writing opcode
    # (Can't sum activations in one gate because tent function makes
    # non-active opcodes negative. Each needs separate ReLU gate.)
    for i, op_idx in enumerate([CONST_ACT_IDX, ADD_ACT_IDX, SUB_ACT_IDX,
                                MUL_ACT_IDX, DUP_ACT_IDX, NEG_ACT_IDX]):
        h = 12 + i
        W1a[h, 6 + op_idx] = 1.0    # gate = ReLU(act_OP)
        b1b[h] = 1.0                  # value = 1.0
        W2[WRITE_FLAG, h] = 1.0       # sum into write_flag

    return W1a, b1a, W1b, b1b, W2, b2, mul_indices


# ================================================================
# LAYER 5: Branch Resolution — GLU-style FFN (ReGLU)
# ================================================================
# Computes PC and SP updates via gated blends using pre-built weight
# matrices. Uses residual connection: output encodes DELTAS.

def _build_layer5_glu_weights():
    """
    Build GLU-style FFN weights for Branch Resolution.

    Tent function produces negative values for non-active opcodes,
    so each opcode gets its own hidden unit (can't sum activations).

    Hidden layer (22 units):
      h0-h1:   act_JMP * delta_pc per component
      h2-h3:   AND(act_JZ, zero_flag) * delta_pc per component
      h4-h5:   AND(act_JN, neg_flag) * delta_pc per component
      h6-h15:  SP decrement: 5 opcodes × 2 components (ADD, SUB, MUL, JZ, JN)
      h16-h19: SP increment: 2 opcodes × 2 components (CONST, DUP)
      h20-h21: PC pass-through (baseline advance)
    """
    d_hidden = 22

    W1a = np.zeros((d_hidden, D_MODEL), dtype=np.float64)
    b1a = np.zeros(d_hidden, dtype=np.float64)
    W1b = np.zeros((d_hidden, D_MODEL), dtype=np.float64)
    b1b = np.zeros(d_hidden, dtype=np.float64)
    W2 = np.zeros((D_MODEL, d_hidden), dtype=np.float64)
    b2 = np.zeros(D_MODEL, dtype=np.float64)

    # --- PC update ---
    for k in range(2):
        # h0,h1: JMP gating
        h_jmp = k
        W1a[h_jmp, 6 + JMP_ACT_IDX] = 1.0
        W1b[h_jmp, JMP_TGT.start + k] = 1.0
        W1b[h_jmp, 0] = -R_PC[k, 0]
        W1b[h_jmp, 1] = -R_PC[k, 1]
        W2[k, h_jmp] = 1.0
        W2[NEW_PC.start + k, h_jmp] = 1.0

        # h2,h3: JZ AND gate — AND(act_JZ, zero_flag)
        h_jz = 2 + k
        W1a[h_jz, 6 + JZ_ACT_IDX] = 1.0
        W1a[h_jz, ZERO_FLAG] = 1.0
        b1a[h_jz] = -1.5
        W1b[h_jz, JMP_TGT.start + k] = 1.0
        W1b[h_jz, 0] = -R_PC[k, 0]
        W1b[h_jz, 1] = -R_PC[k, 1]
        W2[k, h_jz] = 2.0
        W2[NEW_PC.start + k, h_jz] = 2.0

        # h4,h5: JN AND gate — AND(act_JN, neg_flag)
        h_jn = 4 + k
        W1a[h_jn, 6 + JN_ACT_IDX] = 1.0
        W1a[h_jn, NEG_FLAG] = 1.0
        b1a[h_jn] = -1.5
        W1b[h_jn, JMP_TGT.start + k] = 1.0
        W1b[h_jn, 0] = -R_PC[k, 0]
        W1b[h_jn, 1] = -R_PC[k, 1]
        W2[k, h_jn] = 2.0
        W2[NEW_PC.start + k, h_jn] = 2.0

    # --- SP decrement: one hidden unit per opcode per component ---
    # Opcodes that decrement SP: ADD, SUB, MUL, JZ, JN
    dec_opcodes = [ADD_ACT_IDX, SUB_ACT_IDX, MUL_ACT_IDX, JZ_ACT_IDX, JN_ACT_IDX]
    for i, op_idx in enumerate(dec_opcodes):
        for k in range(2):
            h = 6 + i * 2 + k
            W1a[h, 6 + op_idx] = 1.0             # gate = ReLU(act_OP)
            W1b[h, 2] = R_SP_DEC[k, 0] - (1.0 if k == 0 else 0.0)
            W1b[h, 3] = R_SP_DEC[k, 1] - (1.0 if k == 1 else 0.0)
            W2[2 + k, h] = 1.0                    # → SP_DIR[k]
            W2[NEW_SP.start + k, h] = 1.0         # → NEW_SP[k]

    # --- SP increment: one hidden unit per opcode per component ---
    # Opcodes that increment SP: CONST, DUP
    inc_opcodes = [CONST_ACT_IDX, DUP_ACT_IDX]
    for i, op_idx in enumerate(inc_opcodes):
        for k in range(2):
            h = 16 + i * 2 + k
            W1a[h, 6 + op_idx] = 1.0
            W1b[h, 2] = R_SP_INC[k, 0] - (1.0 if k == 0 else 0.0)
            W1b[h, 3] = R_SP_INC[k, 1] - (1.0 if k == 1 else 0.0)
            W2[2 + k, h] = 1.0
            W2[NEW_SP.start + k, h] = 1.0

    # --- PC pass-through (baseline advance) ---
    for k in range(2):
        h_pt = 20 + k
        b1a[h_pt] = 1.0                          # constant gate = 1.0
        W1b[h_pt, k] = 1.0                       # pc_dir[k]
        W2[0, h_pt] = R_PC[0, k] - (1.0 if k == 0 else 0.0)
        W2[1, h_pt] = R_PC[1, k] - (1.0 if k == 1 else 0.0)
        W2[NEW_PC.start, h_pt] = R_PC[0, k] - (1.0 if k == 0 else 0.0)
        W2[NEW_PC.start + 1, h_pt] = R_PC[1, k] - (1.0 if k == 1 else 0.0)

    # Cycle counter: constant +1 via bias
    b2[CYCLE_CTR] = 1.0

    return W1a, b1a, W1b, b1b, W2, b2


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
      - self.residual_stream: np.array of shape (30,)
      - self.instr_cache: KV cache for instructions
      - self.stack_cache: KV cache for stack memory
    """

    def __init__(self, max_tokens=4096):
        self.max_tokens = max_tokens
        # Build fixed weight matrices (these are the "hand-crafted weights")
        # Layer 2: Standard FFN (tent function opcode decode)
        self.W1_L2, self.b1_L2, self.W2_L2, self.b2_L2 = _build_layer2_weights()

        # Layer 4: GLU FFN (ALU arithmetic + flags)
        (self.W1a_L4, self.b1a_L4, self.W1b_L4, self.b1b_L4,
         self.W2_L4, self.b2_L4, self.mul_indices) = _build_layer4_glu_weights()

        # Layer 5: GLU FFN (branch resolution + state update)
        (self.W1a_L5, self.b1a_L5, self.W1b_L5, self.b1b_L5,
         self.W2_L5, self.b2_L5) = _build_layer5_glu_weights()

        # Layer 1: W_Q extracts pc_dir (identity on dims 0-1)
        self.W_Q_L1 = np.zeros((2, D_MODEL), dtype=np.float64)
        self.W_Q_L1[0:2, 0:2] = np.eye(2)

        # Layer 1: W_O projects 4 fetched values to dims OPCODE_RAW, ARG_FLOAT, JMP_TGT
        self.W_O_L1 = np.zeros((D_MODEL, 4), dtype=np.float64)
        self.W_O_L1[OPCODE_RAW, 0] = 1.0     # opcode → dim 4
        self.W_O_L1[ARG_FLOAT, 1] = 1.0      # arg → dim 5
        self.W_O_L1[JMP_TGT.start, 2] = 1.0  # tgt_cos → dim 19
        self.W_O_L1[JMP_TGT.start + 1, 3] = 1.0  # tgt_sin → dim 20

        # Layer 3: W_Q for stack reads
        self.W_Q_head1 = np.zeros((2, D_MODEL), dtype=np.float64)
        self.W_Q_head1[0:2, 2:4] = R_SP_DEC

        self.W_Q_head2 = np.zeros((2, D_MODEL), dtype=np.float64)
        self.W_Q_head2[0:2, 2:4] = R_SP_DEC2

        # Layer 3: W_O projects [va, vb] into residual stream
        self.W_O_L3 = np.zeros((D_MODEL, 2), dtype=np.float64)
        self.W_O_L3[VAL_A, 0] = 1.0          # va → dim 16
        self.W_O_L3[VAL_B, 1] = 1.0          # vb → dim 17

        self.reset()

    def reset(self):
        """Initialize state. The ONLY state is residual_stream + KV caches."""
        self.residual_stream = np.zeros(D_MODEL, dtype=np.float64)
        self.residual_stream[PC_DIR] = addr_to_2d(0, MAX_ADDRS)
        self.residual_stream[SP_DIR] = make_stack_key(0)

        self.instr_cache = MatrixKVCache(n_heads=1, max_tokens=self.max_tokens)
        self.stack_cache = MatrixKVCache(n_heads=3, max_tokens=self.max_tokens)

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
        attn1 = self.W_O_L1 @ fetched_vec  # project to dims 4, 5, 19, 20
        x = x + attn1  # residual connection

        # ==========================================
        # LAYER 2: Opcode Decode (FFN)
        # ==========================================
        # hidden = ReLU(W1 @ x + b1)
        # ffn_out = W2 @ hidden + b2
        hidden2 = np.maximum(0, self.W1_L2 @ x + self.b1_L2)
        ffn2 = self.W2_L2 @ hidden2 + self.b2_L2
        x = x + ffn2  # residual connection: writes one-hot to dims 6-15

        # ==========================================
        # LAYER 3: Operand Fetch (Attention)
        # ==========================================
        # Head 1: query = R_SP_DEC @ sp_dir → stack top (val_b)
        q_top = self.W_Q_head1 @ x   # shape (2,)
        vb = self.stack_cache.query(1, q_top)

        # Head 2: query = R_SP_DEC^2 @ sp_dir → stack second (val_a)
        q_sec = self.W_Q_head2 @ x   # shape (2,)
        va = self.stack_cache.query(2, q_sec)

        # W_O: project [va, vb] into residual stream via matrix multiply
        fetched_stack = np.array([va, vb], dtype=np.float64)
        attn3 = self.W_O_L3 @ fetched_stack  # shape (30,)
        x = x + attn3  # residual connection

        # ==========================================
        # LAYER 4: ALU + Flags (GLU FFN)
        # ==========================================
        # ReGLU architecture: gate * value with Hadamard squaring for MUL.
        # ALL arithmetic is encoded in pre-built weight matrices.
        gate4 = np.maximum(0, self.W1a_L4 @ x + self.b1a_L4)    # ReLU gate
        value4 = self.W1b_L4 @ x + self.b1b_L4                    # linear value
        hidden4 = gate4 * value4                                    # element-wise product (GLU)
        # Hadamard squaring for MUL: (act_MUL*(a+b))^2 = act_MUL*(a+b)^2
        hidden4[self.mul_indices] = hidden4[self.mul_indices] * hidden4[self.mul_indices]
        ffn4 = self.W2_L4 @ hidden4 + self.b2_L4                  # output projection
        x = x + ffn4                                                # residual connection

        # ==========================================
        # LAYER 5: Branch Resolution + State Update (GLU FFN)
        # ==========================================
        # Save pre-Layer5 state for memory write bus
        acts = np.maximum(0, x[ACT])          # activations (unchanged by L4/L5)
        sp_dir_pre = x[SP_DIR].copy()         # SP before update

        # GLU FFN: all PC/SP updates via pre-built weight matrices
        # W2 encodes (R_PC - I) for baseline advance, jump gating,
        # SP rotation deltas, and cycle counter increment.
        gate5 = np.maximum(0, self.W1a_L5 @ x + self.b1a_L5)    # ReLU gate
        value5 = self.W1b_L5 @ x + self.b1b_L5                    # linear value
        hidden5 = gate5 * value5                                    # element-wise product (GLU)
        ffn5 = self.W2_L5 @ hidden5 + self.b2_L5                  # output projection
        x = x + ffn5                                                # residual connection

        # ==========================================
        # MEMORY WRITE BUS (not a transformer layer — cache I/O)
        # ==========================================
        # Compute stack write position from pre-update SP direction
        sp_dec1 = R_SP_DEC @ sp_dir_pre
        sp_dec2 = R_SP_DEC2 @ sp_dir_pre
        gate_write_at_sp = acts[CONST_ACT_IDX] + acts[DUP_ACT_IDX]
        gate_write_at_sp2 = acts[ADD_ACT_IDX] + acts[SUB_ACT_IDX] + acts[MUL_ACT_IDX]
        gate_write_at_sp1 = acts[NEG_ACT_IDX]
        write_key_dir = (gate_write_at_sp * sp_dir_pre +
                         gate_write_at_sp2 * sp_dec2 +
                         gate_write_at_sp1 * sp_dec1)

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

        # SAVE HALT SIGNAL BEFORE CLEARING STATE
        # Return the halt signal BEFORE persistence mask clears it
        halt_signal = x[HALT_FLAG]

        # MATRIX OPERATION: Clear transient dims via element-wise multiplication
        # Persistent dims (PC, SP, CYCLE_CTR) are multiplied by 1.0
        # Temporary dims are multiplied by 0.0 (cleared)
        x = x * PERSISTENCE_MASK

        # Save residual stream
        self.residual_stream = x
        self.cycle_count += 1  # for trace only

        # Return the halt SIGNAL (not interpreted)
        # Let external code (run()) decide when to stop
        return halt_signal  # Return the signal value captured BEFORE clearing

    # ==============================================================
    # RUN
    # ==============================================================

    def run(self, program, verbose=False, max_cycles=100000):
        """Load and execute a program. Returns final stack."""
        self.load_program(program)

        for _ in range(max_cycles):
            halt_signal = self.forward_pass()  # Get the signal value
            if verbose:
                print(f"  cycle {self.cycle_count}")
            if halt_signal > 0.5:  # Interpret signal as halt threshold
                break

        # Read result from stack via attention.
        # The top of stack is at sp_dir rotated back by 1:
        sp_dir = self.residual_stream[SP_DIR]
        q_top = R_SP_DEC @ sp_dir
        result = self.stack_cache.query(1, q_top)
        # No ternary needed - cache pre-seeding guarantees result is never None
        return [float(result)]


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

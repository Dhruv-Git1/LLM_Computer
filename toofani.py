#!/usr/bin/env python3
"""
Toofani: PyTorch + CUDA Transformer Computer
=============================================
A hand-crafted transformer that executes stack-machine programs on GPU.
No training, no gradient descent -- all weights constructed analytically.

6 opcodes: CONST, ADD, SUB, MUL, DUP, HALT
5 layers: fetch, decode, operand fetch, ALU, state update
22-dim residual stream, pure matrix ops in forward pass (zero if/elif)

Based on: "Can LLMs Be Computers?" (Percepta, March 2026)
"""

import torch
import math
import time

# ================================================================
# OPCODES
# ================================================================
CONST = 1
ADD   = 2
SUB   = 3
MUL   = 4
HALT  = 5
DUP   = 6

N_OPCODES = 6
OPCODE_LIST = [CONST, ADD, SUB, MUL, HALT, DUP]
OP_NAME = {CONST: "CONST", ADD: "ADD", SUB: "SUB",
           MUL: "MUL", HALT: "HALT", DUP: "DUP"}

# Activation indices within ACT slice
CONST_IDX = 0
ADD_IDX   = 1
SUB_IDX   = 2
MUL_IDX   = 3
HALT_IDX  = 4
DUP_IDX   = 5

# ================================================================
# RESIDUAL STREAM LAYOUT (22 dimensions)
# ================================================================
D_MODEL = 22

PC_DIR     = slice(0, 2)    # (2) program counter as 2D unit vector
SP_DIR     = slice(2, 4)    # (2) stack pointer as 2D unit vector
OPCODE_RAW = 4              # (1) fetched opcode float
ARG_FLOAT  = 5              # (1) fetched argument
ACT        = slice(6, 12)   # (6) one-hot opcode activations
VAL_A      = 12             # (1) stack second-from-top
VAL_B      = 13             # (1) stack top
ALU_RESULT = 14             # (1) arithmetic output
HALT_FLAG  = 15             # (1) 1.0 if HALT
WRITE_FLAG = 16             # (1) 1.0 if stack write needed
NEW_PC     = slice(17, 19)  # (2) next PC direction
NEW_SP     = slice(19, 21)  # (2) next SP direction
CYCLE_CTR  = 21             # (1) monotonic cycle counter

# ================================================================
# CONSTANTS
# ================================================================
MAX_ADDRS  = 65536   # instruction address space
MAX_STACKD = 32      # stack depth


def _get_device(device=None):
    """Resolve device string to torch.device."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================================
# ADDRESS ENCODING
# ================================================================
def addr_to_2d(addr, n=MAX_ADDRS, device=None):
    """Encode integer address as 2D unit vector on the circle."""
    dev = _get_device(device)
    theta = 2.0 * math.pi * addr / n
    return torch.tensor([math.cos(theta), math.sin(theta)],
                        dtype=torch.float64, device=dev)


def make_stack_key(depth, device=None):
    """Encode stack depth as 2D unit vector."""
    dev = _get_device(device)
    theta = 2.0 * math.pi * depth / MAX_STACKD
    return torch.tensor([math.cos(theta), math.sin(theta)],
                        dtype=torch.float64, device=dev)


# ================================================================
# ROTATION MATRICES (precomputed, fixed)
# ================================================================
def _build_rotations(device):
    """Build PC and SP rotation matrices on device."""
    dev = _get_device(device)

    dtheta_pc = 2.0 * math.pi / MAX_ADDRS
    R_PC = torch.tensor([[math.cos(dtheta_pc), -math.sin(dtheta_pc)],
                         [math.sin(dtheta_pc),  math.cos(dtheta_pc)]],
                        dtype=torch.float64, device=dev)

    dtheta_sp = 2.0 * math.pi / MAX_STACKD
    R_SP_INC = torch.tensor([[math.cos(dtheta_sp), -math.sin(dtheta_sp)],
                             [math.sin(dtheta_sp),  math.cos(dtheta_sp)]],
                            dtype=torch.float64, device=dev)
    R_SP_DEC = R_SP_INC.T
    R_SP_DEC2 = R_SP_DEC @ R_SP_DEC

    return R_PC, R_SP_INC, R_SP_DEC, R_SP_DEC2


# ================================================================
# PERSISTENCE MASK
# ================================================================
def _build_persistence_mask(device):
    dev = _get_device(device)
    mask = torch.zeros(D_MODEL, dtype=torch.float64, device=dev)
    mask[PC_DIR] = 1.0
    mask[SP_DIR] = 1.0
    mask[CYCLE_CTR] = 1.0
    return mask


# ================================================================
# KV CACHE (PyTorch, GPU-compatible)
# ================================================================
class TorchKVCache:
    """
    KV cache using pre-allocated torch tensors.
    Query = matrix multiply + argmax (pure tensor ops).
    """

    def __init__(self, n_heads, key_dim=2, val_dim=1, max_tokens=4096, device=None):
        self.dev = _get_device(device)
        self.n_heads = n_heads
        self.max_tokens = max_tokens
        self.val_dim = val_dim

        # Pre-allocate on device
        self._keys = {h: torch.zeros(max_tokens, key_dim,
                                     dtype=torch.float64, device=self.dev)
                      for h in range(n_heads)}
        if val_dim == 1:
            self._values = {h: torch.zeros(max_tokens,
                                           dtype=torch.float64, device=self.dev)
                            for h in range(n_heads)}
        else:
            self._values = {h: torch.zeros(max_tokens, val_dim,
                                           dtype=torch.float64, device=self.dev)
                            for h in range(n_heads)}
        self._counts = {h: 0 for h in range(n_heads)}

    def insert(self, head, key, value):
        """Insert a key-value pair into a single head."""
        idx = self._counts[head]
        self._keys[head][idx] = key
        self._values[head][idx] = value
        self._counts[head] += 1

    def query(self, head, query_vec):
        """
        Pure tensor attention: scores = keys @ query, best = argmax(scores).
        """
        n = self._counts[head]
        scores = self._keys[head][:n] @ query_vec   # (n,)
        best = torch.argmax(scores)                   # scalar
        return self._values[head][best]


# ================================================================
# WEIGHT BUILDERS
# ================================================================

def _build_layer2_weights(device):
    """
    Opcode decode: tent function via 4 ReLU neurons per opcode.
    tent(x, c) = max(0, 1 - |x - c|)
    """
    dev = _get_device(device)
    n_hidden = 4 * N_OPCODES  # 24

    W1 = torch.zeros(n_hidden, D_MODEL, dtype=torch.float64, device=dev)
    b1 = torch.zeros(n_hidden, dtype=torch.float64, device=dev)
    W2 = torch.zeros(D_MODEL, n_hidden, dtype=torch.float64, device=dev)
    b2 = torch.zeros(D_MODEL, dtype=torch.float64, device=dev)

    for i, op in enumerate(OPCODE_LIST):
        base = 4 * i
        out_dim = 6 + i  # ACT slice starts at 6

        W1[base + 0, OPCODE_RAW] = 1.0;  b1[base + 0] = -op + 1.0
        W1[base + 1, OPCODE_RAW] = 1.0;  b1[base + 1] = -op
        W1[base + 2, OPCODE_RAW] = -1.0; b1[base + 2] = op + 1.0
        W1[base + 3, OPCODE_RAW] = -1.0; b1[base + 3] = op

        W2[out_dim, base + 0] = 0.5
        W2[out_dim, base + 1] = -1.0
        W2[out_dim, base + 2] = 0.5
        W2[out_dim, base + 3] = -1.0

    return W1, b1, W2, b2


def _build_layer4_glu_weights(device):
    """
    ALU layer (ReGLU): all arithmetic in parallel, gated by opcode activations.

    Hidden units (12):
      h0:  act_CONST * arg          -> ALU_RESULT
      h1:  act_ADD * (a+b)          -> ALU_RESULT
      h2:  act_SUB * (a-b)          -> ALU_RESULT
      h3:  act_MUL * (a+b) -> sq   -> ALU_RESULT (* 0.25)
      h4:  act_MUL * (a-b) -> sq   -> ALU_RESULT (* -0.25)
      h5:  act_DUP * val_b          -> ALU_RESULT
      h6:  act_HALT * 1.0           -> HALT_FLAG
      h7-h11: write_flag (one per writing opcode: CONST, ADD, SUB, MUL, DUP)
    """
    dev = _get_device(device)
    d_hidden = 12
    mul_indices = torch.tensor([3, 4], dtype=torch.long, device=dev)

    W1a = torch.zeros(d_hidden, D_MODEL, dtype=torch.float64, device=dev)
    b1a = torch.zeros(d_hidden, dtype=torch.float64, device=dev)
    W1b = torch.zeros(d_hidden, D_MODEL, dtype=torch.float64, device=dev)
    b1b = torch.zeros(d_hidden, dtype=torch.float64, device=dev)
    W2  = torch.zeros(D_MODEL, d_hidden, dtype=torch.float64, device=dev)
    b2  = torch.zeros(D_MODEL, dtype=torch.float64, device=dev)

    # h0: CONST -> gate=act_CONST, value=arg
    W1a[0, 6 + CONST_IDX] = 1.0
    W1b[0, ARG_FLOAT] = 1.0
    W2[ALU_RESULT, 0] = 1.0

    # h1: ADD -> gate=act_ADD, value=a+b
    W1a[1, 6 + ADD_IDX] = 1.0
    W1b[1, VAL_A] = 1.0
    W1b[1, VAL_B] = 1.0
    W2[ALU_RESULT, 1] = 1.0

    # h2: SUB -> gate=act_SUB, value=a-b
    W1a[2, 6 + SUB_IDX] = 1.0
    W1b[2, VAL_A] = 1.0
    W1b[2, VAL_B] = -1.0
    W2[ALU_RESULT, 2] = 1.0

    # h3: MUL (a+b) path -> squared post-hoc, *0.25
    W1a[3, 6 + MUL_IDX] = 1.0
    W1b[3, VAL_A] = 1.0
    W1b[3, VAL_B] = 1.0
    W2[ALU_RESULT, 3] = 0.25

    # h4: MUL (a-b) path -> squared post-hoc, *-0.25
    W1a[4, 6 + MUL_IDX] = 1.0
    W1b[4, VAL_A] = 1.0
    W1b[4, VAL_B] = -1.0
    W2[ALU_RESULT, 4] = -0.25

    # h5: DUP -> gate=act_DUP, value=val_b
    W1a[5, 6 + DUP_IDX] = 1.0
    W1b[5, VAL_B] = 1.0
    W2[ALU_RESULT, 5] = 1.0

    # h6: HALT flag -> gate=act_HALT, value=1.0
    W1a[6, 6 + HALT_IDX] = 1.0
    b1b[6] = 1.0
    W2[HALT_FLAG, 6] = 1.0

    # h7-h11: write_flag (one per writing opcode)
    writing = [CONST_IDX, ADD_IDX, SUB_IDX, MUL_IDX, DUP_IDX]
    for i, op_idx in enumerate(writing):
        h = 7 + i
        W1a[h, 6 + op_idx] = 1.0
        b1b[h] = 1.0
        W2[WRITE_FLAG, h] = 1.0

    return W1a, b1a, W1b, b1b, W2, b2, mul_indices


def _build_layer5_glu_weights(device, R_PC, R_SP_INC, R_SP_DEC):
    """
    State update layer (no branching needed -- just PC advance + SP update).

    Hidden units (12):
      h0-h1:   PC pass-through (baseline advance by R_PC)
      h2-h9:   SP decrement: 3 opcodes x 2 components (ADD, SUB, MUL)
               SP increment: 2 opcodes x 2 components (CONST, DUP)
               (but we use 4 pairs for dec and 2 pairs for inc = 12 total with PC)
    Actually let's be precise:
      h0-h1:   PC advance (constant gate=1, value=pc_dir)
      h2-h3:   SP dec for ADD
      h4-h5:   SP dec for SUB
      h6-h7:   SP dec for MUL
      h8-h9:   SP inc for CONST
      h10-h11: SP inc for DUP
    """
    dev = _get_device(device)
    d_hidden = 12

    W1a = torch.zeros(d_hidden, D_MODEL, dtype=torch.float64, device=dev)
    b1a = torch.zeros(d_hidden, dtype=torch.float64, device=dev)
    W1b = torch.zeros(d_hidden, D_MODEL, dtype=torch.float64, device=dev)
    b1b = torch.zeros(d_hidden, dtype=torch.float64, device=dev)
    W2  = torch.zeros(D_MODEL, d_hidden, dtype=torch.float64, device=dev)
    b2  = torch.zeros(D_MODEL, dtype=torch.float64, device=dev)

    # h0-h1: PC advance (always: gate=1.0 via bias, value=pc_dir[k])
    for k in range(2):
        h = k
        b1a[h] = 1.0                  # constant gate = 1.0
        W1b[h, k] = 1.0               # value = pc_dir[k]
        # delta = R_PC @ pc_dir - pc_dir, projected per component
        W2[k, h] = R_PC[0, k].item() - (1.0 if k == 0 else 0.0)
        W2[1, h] = R_PC[1, k].item() - (1.0 if k == 1 else 0.0) if k == 0 else W2[1, h]
        # Let me redo this more carefully
    # Actually, let me rewrite the PC advance cleanly:
    # We want: new_pc = R_PC @ pc_dir
    # Residual: delta = new_pc - pc_dir = (R_PC - I) @ pc_dir
    # Two hidden units, each carrying one component of pc_dir:
    #   h0: gate=1, value=pc_dir[0] -> output adds (R_PC - I)[:,0] * pc_dir[0]
    #   h1: gate=1, value=pc_dir[1] -> output adds (R_PC - I)[:,1] * pc_dir[1]
    W1a[:] = 0; b1a[:] = 0; W1b[:] = 0; b1b[:] = 0; W2[:] = 0; b2[:] = 0

    for k in range(2):
        h = k
        b1a[h] = 1.0               # constant gate
        W1b[h, k] = 1.0            # value = pc_dir[k]
        # Output: (R_PC - I) column k -> PC_DIR dims
        W2[0, h] = R_PC[0, k].item() - (1.0 if k == 0 else 0.0)
        W2[1, h] = R_PC[1, k].item() - (0.0 if k == 0 else 1.0)
        # Also write to NEW_PC
        W2[NEW_PC.start, h]     = R_PC[0, k].item() - (1.0 if k == 0 else 0.0)
        W2[NEW_PC.start + 1, h] = R_PC[1, k].item() - (0.0 if k == 0 else 1.0)

    # SP decrement opcodes: ADD, SUB, MUL
    dec_opcodes = [ADD_IDX, SUB_IDX, MUL_IDX]
    for i, op_idx in enumerate(dec_opcodes):
        for k in range(2):
            h = 2 + i * 2 + k
            W1a[h, 6 + op_idx] = 1.0   # gate = ReLU(act_OP)
            # value = (R_SP_DEC - I) column k applied to sp_dir[k]
            W1b[h, 2] = R_SP_DEC[k, 0].item() - (1.0 if k == 0 else 0.0)
            W1b[h, 3] = R_SP_DEC[k, 1].item() - (0.0 if k == 0 else 1.0)
            W2[2 + k, h] = 1.0                # -> SP_DIR[k]
            W2[NEW_SP.start + k, h] = 1.0     # -> NEW_SP[k]

    # SP increment opcodes: CONST, DUP
    inc_opcodes = [CONST_IDX, DUP_IDX]
    for i, op_idx in enumerate(inc_opcodes):
        for k in range(2):
            h = 8 + i * 2 + k
            W1a[h, 6 + op_idx] = 1.0
            W1b[h, 2] = R_SP_INC[k, 0].item() - (1.0 if k == 0 else 0.0)
            W1b[h, 3] = R_SP_INC[k, 1].item() - (0.0 if k == 0 else 1.0)
            W2[2 + k, h] = 1.0
            W2[NEW_SP.start + k, h] = 1.0

    # Cycle counter increment
    b2[CYCLE_CTR] = 1.0

    return W1a, b1a, W1b, b1b, W2, b2


# ================================================================
# REFERENCE INTERPRETER (for verification)
# ================================================================
def run_reference(program, max_cycles=100000):
    """Simple Python stack machine. Returns final stack top as [float]."""
    pc, stack = 0, []
    cycles = 0
    while pc < len(program) and cycles < max_cycles:
        cycles += 1
        op, arg = program[pc][0], program[pc][1] if len(program[pc]) > 1 else 0
        if op == HALT:
            break
        elif op == CONST:
            stack.append(arg)
        elif op == ADD:
            b, a = stack.pop(), stack.pop()
            stack.append(a + b)
        elif op == SUB:
            b, a = stack.pop(), stack.pop()
            stack.append(a - b)
        elif op == MUL:
            b, a = stack.pop(), stack.pop()
            stack.append(a * b)
        elif op == DUP:
            stack.append(stack[-1])
        pc += 1
    return [float(stack[-1])] if stack else [0.0]


# ================================================================
# THE TRANSFORMER COMPUTER
# ================================================================
class ToofaniComputer:
    """
    PyTorch + CUDA transformer CPU.

    Forward pass has ZERO if/elif -- all computation via:
      - torch.matmul (matrix multiply)
      - torch.relu (ReLU activation)
      - torch.argmax (attention)
      - element-wise ops (*, +)
    """

    def __init__(self, max_tokens=4096, device=None):
        self.dev = _get_device(device)
        self.max_tokens = max_tokens

        # Build rotation matrices
        self.R_PC, self.R_SP_INC, self.R_SP_DEC, self.R_SP_DEC2 = \
            _build_rotations(self.dev)

        # Build persistence mask
        self.PERSIST = _build_persistence_mask(self.dev)

        # Layer 1 weights: instruction fetch attention
        self.W_Q_L1 = torch.zeros(2, D_MODEL, dtype=torch.float64, device=self.dev)
        self.W_Q_L1[0, 0] = 1.0
        self.W_Q_L1[1, 1] = 1.0

        self.W_O_L1 = torch.zeros(D_MODEL, 2, dtype=torch.float64, device=self.dev)
        self.W_O_L1[OPCODE_RAW, 0] = 1.0
        self.W_O_L1[ARG_FLOAT, 1] = 1.0

        # Layer 2 weights: opcode decode
        self.W1_L2, self.b1_L2, self.W2_L2, self.b2_L2 = \
            _build_layer2_weights(self.dev)

        # Layer 3 weights: operand fetch attention
        self.W_Q_head1 = torch.zeros(2, D_MODEL, dtype=torch.float64, device=self.dev)
        self.W_Q_head1[0:2, 2:4] = self.R_SP_DEC

        self.W_Q_head2 = torch.zeros(2, D_MODEL, dtype=torch.float64, device=self.dev)
        self.W_Q_head2[0:2, 2:4] = self.R_SP_DEC2

        self.W_O_L3 = torch.zeros(D_MODEL, 2, dtype=torch.float64, device=self.dev)
        self.W_O_L3[VAL_A, 0] = 1.0
        self.W_O_L3[VAL_B, 1] = 1.0

        # Layer 4 weights: ALU
        self.W1a_L4, self.b1a_L4, self.W1b_L4, self.b1b_L4, \
            self.W2_L4, self.b2_L4, self.mul_idx = \
            _build_layer4_glu_weights(self.dev)

        # Layer 5 weights: state update
        self.W1a_L5, self.b1a_L5, self.W1b_L5, self.b1b_L5, \
            self.W2_L5, self.b2_L5 = \
            _build_layer5_glu_weights(self.dev, self.R_PC,
                                      self.R_SP_INC, self.R_SP_DEC)

        self.reset()

    def reset(self):
        """Initialize residual stream + KV caches."""
        self.x = torch.zeros(D_MODEL, dtype=torch.float64, device=self.dev)
        self.x[PC_DIR] = addr_to_2d(0, MAX_ADDRS, self.dev)
        self.x[SP_DIR] = make_stack_key(0, self.dev)

        # Instruction cache: head 0, stores (opcode, arg) as 2-element values
        self.instr_cache = TorchKVCache(1, key_dim=2, val_dim=2,
                                        max_tokens=self.max_tokens,
                                        device=self.dev)
        # Stack cache: heads 1,2 for val_b and val_a reads, scalar values
        self.stack_cache = TorchKVCache(3, key_dim=2, val_dim=1,
                                        max_tokens=self.max_tokens,
                                        device=self.dev)

        # Pre-seed stack cache (eliminates None checks)
        default_key = make_stack_key(MAX_STACKD - 1, self.dev)
        zero_val = torch.tensor(0.0, dtype=torch.float64, device=self.dev)
        self.stack_cache.insert(1, default_key, zero_val)
        self.stack_cache.insert(2, default_key, zero_val)

        self.cycle_count = 0

    def load_program(self, program):
        """Load instructions into KV cache (= writing program to RAM)."""
        self.reset()

        # Pre-seed with HALT at far address
        far_key = addr_to_2d(MAX_ADDRS - 1, MAX_ADDRS, self.dev)
        halt_val = torch.tensor([float(HALT), 0.0],
                                dtype=torch.float64, device=self.dev)
        self.instr_cache.insert(0, far_key, halt_val)

        # Load actual program
        for addr, instr in enumerate(program):
            opcode = instr[0]
            arg = instr[1] if len(instr) > 1 else 0
            key = addr_to_2d(addr, MAX_ADDRS, self.dev)
            val = torch.tensor([float(opcode), float(arg)],
                               dtype=torch.float64, device=self.dev)
            self.instr_cache.insert(0, key, val)

    # ==============================================================
    # FORWARD PASS -- ZERO if/elif
    # ==============================================================
    def forward_pass(self):
        """
        One clock cycle. Returns halt signal (float).
        Pure tensor ops: matmul, relu, argmax, element-wise.
        """
        x = self.x.clone()

        # --- LAYER 1: Instruction Fetch (Attention) ---
        query = self.W_Q_L1 @ x                          # (2,)
        fetched = self.instr_cache.query(0, query)        # (2,) = [opcode, arg]
        attn1 = self.W_O_L1 @ fetched                    # (D_MODEL,)
        x = x + attn1

        # --- LAYER 2: Opcode Decode (FFN) ---
        h2 = torch.relu(self.W1_L2 @ x + self.b1_L2)
        x = x + self.W2_L2 @ h2 + self.b2_L2

        # --- LAYER 3: Operand Fetch (Attention) ---
        q_top = self.W_Q_head1 @ x                       # (2,)
        vb = self.stack_cache.query(1, q_top)             # scalar
        q_sec = self.W_Q_head2 @ x                        # (2,)
        va = self.stack_cache.query(2, q_sec)             # scalar
        stack_vec = torch.stack([va, vb])                 # (2,)
        x = x + self.W_O_L3 @ stack_vec

        # --- LAYER 4: ALU + Flags (ReGLU FFN) ---
        gate4 = torch.relu(self.W1a_L4 @ x + self.b1a_L4)
        val4  = self.W1b_L4 @ x + self.b1b_L4
        hidden4 = gate4 * val4                             # GLU
        # Hadamard squaring for MUL: ((a+b)^2 - (a-b)^2) / 4
        hidden4[self.mul_idx] = hidden4[self.mul_idx] * hidden4[self.mul_idx]
        x = x + self.W2_L4 @ hidden4 + self.b2_L4

        # --- LAYER 5: State Update (ReGLU FFN) ---
        # Save SP direction before update for memory write bus
        acts = torch.relu(x[ACT])
        sp_dir_pre = x[SP_DIR].clone()

        gate5 = torch.relu(self.W1a_L5 @ x + self.b1a_L5)
        val5  = self.W1b_L5 @ x + self.b1b_L5
        hidden5 = gate5 * val5
        x = x + self.W2_L5 @ hidden5 + self.b2_L5

        # --- MEMORY WRITE BUS ---
        sp_dec1 = self.R_SP_DEC @ sp_dir_pre
        sp_dec2 = self.R_SP_DEC2 @ sp_dir_pre
        # CONST/DUP write at sp (before inc), ADD/SUB/MUL write at sp-2
        gate_at_sp  = acts[CONST_IDX] + acts[DUP_IDX]
        gate_at_sp2 = acts[ADD_IDX] + acts[SUB_IDX] + acts[MUL_IDX]
        write_key_dir = gate_at_sp * sp_dir_pre + gate_at_sp2 * sp_dec2

        cycle = x[CYCLE_CTR]
        magnitude = 1.0 + cycle * 1e-8
        gated_mag = magnitude * torch.relu(x[WRITE_FLAG])
        stack_key = write_key_dir * gated_mag
        stack_val = x[ALU_RESULT]

        self.stack_cache.insert(1, stack_key, stack_val)
        self.stack_cache.insert(2, stack_key, stack_val)

        # Capture halt signal before clearing
        halt_signal = x[HALT_FLAG].item()

        # Persistence mask: clear transient dims
        x = x * self.PERSIST
        self.x = x
        self.cycle_count += 1

        return halt_signal

    # ==============================================================
    # RUN
    # ==============================================================
    def run(self, program, max_cycles=100000):
        """Load and execute program. Returns [top_of_stack]."""
        self.load_program(program)

        for _ in range(max_cycles):
            halt = self.forward_pass()
            if halt > 0.5:
                break

        # Read top of stack via attention
        sp_dir = self.x[SP_DIR]
        q_top = self.R_SP_DEC @ sp_dir
        result = self.stack_cache.query(1, q_top)
        return [float(result.item())]


# ================================================================
# TESTS
# ================================================================
def run_tests(device=None):
    dev = _get_device(device)
    print("=" * 60)
    print(f"Toofani Transformer Computer -- Tests")
    print(f"Device: {dev}")
    if dev.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(dev)}")
    print("=" * 60)

    tc = ToofaniComputer(device=dev)
    passed = 0
    failed = 0

    def check(name, program, expected):
        nonlocal passed, failed
        result = tc.run(program)
        ref = run_reference(program)
        ok = abs(result[0] - expected) < 0.5 and abs(ref[0] - expected) < 0.5
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  {status} {name}: expected {expected}, got {result[0]:.4f} (ref={ref[0]:.4f})")
        return ok

    # --- Basic arithmetic ---
    print("\n[1] Basic Arithmetic")
    print("-" * 60)
    check("3 + 5 = 8",
          [(CONST, 3), (CONST, 5), (ADD, 0), (HALT, 0)], 8.0)
    check("10 - 3 = 7",
          [(CONST, 10), (CONST, 3), (SUB, 0), (HALT, 0)], 7.0)
    check("4 * 5 = 20",
          [(CONST, 4), (CONST, 5), (MUL, 0), (HALT, 0)], 20.0)
    check("7 + 0 = 7",
          [(CONST, 7), (CONST, 0), (ADD, 0), (HALT, 0)], 7.0)
    check("0 * 100 = 0",
          [(CONST, 0), (CONST, 100), (MUL, 0), (HALT, 0)], 0.0)
    check("100 - 100 = 0",
          [(CONST, 100), (CONST, 100), (SUB, 0), (HALT, 0)], 0.0)

    # --- Negative numbers ---
    print("\n[2] Negative Numbers")
    print("-" * 60)
    check("3 - 7 = -4",
          [(CONST, 3), (CONST, 7), (SUB, 0), (HALT, 0)], -4.0)
    check("-3 + -5 = -8",
          [(CONST, -3), (CONST, -5), (ADD, 0), (HALT, 0)], -8.0)
    check("-4 * 5 = -20",
          [(CONST, -4), (CONST, 5), (MUL, 0), (HALT, 0)], -20.0)
    check("-3 * -7 = 21",
          [(CONST, -3), (CONST, -7), (MUL, 0), (HALT, 0)], 21.0)

    # --- DUP ---
    print("\n[3] DUP Instruction")
    print("-" * 60)
    check("DUP 5 then ADD = 10",
          [(CONST, 5), (DUP, 0), (ADD, 0), (HALT, 0)], 10.0)
    check("DUP 7 then MUL = 49 (7^2)",
          [(CONST, 7), (DUP, 0), (MUL, 0), (HALT, 0)], 49.0)
    check("DUP 3 then SUB = 0",
          [(CONST, 3), (DUP, 0), (SUB, 0), (HALT, 0)], 0.0)

    # --- Multi-step ---
    print("\n[4] Multi-step Programs")
    print("-" * 60)
    check("(3 + 5) * 2 = 16",
          [(CONST, 3), (CONST, 5), (ADD, 0),
           (CONST, 2), (MUL, 0), (HALT, 0)], 16.0)
    check("(10 - 3) * (2 + 1) = 21",
          [(CONST, 10), (CONST, 3), (SUB, 0),
           (CONST, 2), (CONST, 1), (ADD, 0),
           (MUL, 0), (HALT, 0)], 21.0)
    check("3 * 5 - 2 = 13",
          [(CONST, 3), (CONST, 5), (MUL, 0),
           (CONST, 2), (SUB, 0), (HALT, 0)], 13.0)
    check("(1 + 2) * (3 + 4) = 21",
          [(CONST, 1), (CONST, 2), (ADD, 0),
           (CONST, 3), (CONST, 4), (ADD, 0),
           (MUL, 0), (HALT, 0)], 21.0)

    # --- Large values ---
    print("\n[5] Large Values")
    print("-" * 60)
    check("1000 + 2000 = 3000",
          [(CONST, 1000), (CONST, 2000), (ADD, 0), (HALT, 0)], 3000.0)
    check("50 * 50 = 2500",
          [(CONST, 50), (CONST, 50), (MUL, 0), (HALT, 0)], 2500.0)
    check("999 - 1000 = -1",
          [(CONST, 999), (CONST, 1000), (SUB, 0), (HALT, 0)], -1.0)

    # --- x^2 via DUP+MUL ---
    print("\n[6] x^2 via DUP+MUL")
    print("-" * 60)
    for x_val in [0, 1, 2, 3, 5, 10, -3]:
        check(f"{x_val}^2 = {x_val**2}",
              [(CONST, x_val), (DUP, 0), (MUL, 0), (HALT, 0)],
              float(x_val ** 2))

    # --- Chained operations ---
    print("\n[7] Chained Operations")
    print("-" * 60)
    check("((2+3)*4)-5 = 15",
          [(CONST, 2), (CONST, 3), (ADD, 0),
           (CONST, 4), (MUL, 0),
           (CONST, 5), (SUB, 0), (HALT, 0)], 15.0)
    check("2*3*4 = 24",
          [(CONST, 2), (CONST, 3), (MUL, 0),
           (CONST, 4), (MUL, 0), (HALT, 0)], 24.0)
    check("1+2+3+4+5 = 15",
          [(CONST, 1), (CONST, 2), (ADD, 0),
           (CONST, 3), (ADD, 0),
           (CONST, 4), (ADD, 0),
           (CONST, 5), (ADD, 0), (HALT, 0)], 15.0)

    # --- Summary ---
    total = passed + failed
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("ALL TESTS PASSED")
    print("=" * 60)

    return failed == 0


def run_benchmark(device=None):
    """Benchmark: time N programs on the given device."""
    dev = _get_device(device)
    tc = ToofaniComputer(device=dev)

    # Benchmark program: chain of arithmetic (10 instructions)
    program = [
        (CONST, 7), (CONST, 3), (ADD, 0),    # 10
        (CONST, 2), (MUL, 0),                 # 20
        (DUP, 0), (ADD, 0),                   # 40
        (CONST, 8), (SUB, 0),                 # 32
        (HALT, 0),
    ]

    # Warmup
    for _ in range(5):
        tc.run(program)

    # Sync CUDA before timing
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)

    n_runs = 200
    t0 = time.perf_counter()
    for _ in range(n_runs):
        tc.run(program)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elapsed = time.perf_counter() - t0

    cycles_per_run = 9  # 9 instructions before HALT
    total_cycles = n_runs * cycles_per_run
    print(f"\nBenchmark on {dev}:")
    print(f"  {n_runs} program runs, {total_cycles} total cycles")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  {total_cycles / elapsed:.0f} cycles/sec")
    print(f"  {elapsed / n_runs * 1000:.2f} ms/program")


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    import sys

    # Determine device
    if "--cpu" in sys.argv:
        device = "cpu"
    elif "--cuda" in sys.argv:
        device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()

    ok = run_tests(device)
    if ok:
        run_benchmark(device)

    # If CUDA available, also benchmark CPU for comparison
    if device == "cuda":
        print()
        run_benchmark("cpu")

# Replication plan: "Can LLMs Be Computers?"

**Goal**: Build a transformer from scratch whose hand-crafted weights execute a simple program (addition of two integers), then extend to a minimal WASM subset, then add HullKVCache for O(log n) attention.

**Status of original code**: No public code released by Percepta as of March 2026.

**Key prior work to study first** (these have open-source code):
- **RASP** (Weiss et al., 2021): Programming language that maps to transformer operations. Code: `github.com/tech-srl/RASP`
- **Tracr** (Lindner et al., 2023): Compiler that converts RASP programs → actual transformer weights in JAX. Code: `github.com/google-deepmind/tracr`
- **Looped Transformers as Programmable Computers** (Giannou et al., 2023): Constructs transformer weights to simulate a one-instruction computer (SUBLEQ). Paper: `arxiv.org/abs/2301.13196`

---

## Phase 0: Prerequisites (1-2 weeks)

### 0.1 — Math foundations to be comfortable with
- How multi-head attention works: Q, K, V projection, dot-product scores, softmax vs argmax
- Residual connections: output = input + layer_output (additive, not replacement)
- FFN as function approximator: ReLU(xW₁ + b₁)W₂ + b₂
- How specific weight values make matmul do specific things (routing, addition, conditional gating)
- Convex hull in 2D: definition, incremental construction, binary search for extreme point in a direction

### 0.2 — Tools to set up
- Python 3.10+, PyTorch
- A basic transformer implementation you understand line-by-line (no libraries — write your own ~200 line transformer)
- A WASM reference: the WebAssembly spec's list of opcodes (start with i32 integer subset only)

### 0.3 — Study Tracr
- Install Tracr (`pip install tracr`)
- Run its examples: compile a simple RASP program (e.g., token counting, sorting) into transformer weights
- Inspect the resulting weight matrices — understand how RASP's `select` becomes attention and `aggregate` becomes value readout
- This gives you a concrete feel for "compiling programs into weights"

---

## Phase 1: Toy computer — add two numbers (2-3 weeks)

**Goal**: A 3-layer transformer with d_model=8 and 4 heads (2D each) that executes: `push A, push B, add, halt`. Four clock cycles, four tokens generated.

### 1.1 — Define the state encoding
Decide how to pack the computer's state into 8 dimensions:
```
dim 0-1: instruction address (which step we're on)
dim 2-3: stack pointer
dim 4-5: top of stack value
dim 6-7: second stack value / scratch
```

### 1.2 — Define the token vocabulary
This is a tiny closed vocabulary:
```
Token 0: PUSH_A (encodes "push the first number")
Token 1: PUSH_B (encodes "push the second number")
Token 2: ADD
Token 3: HALT
Token 4-N: numeric output tokens (the result)
```

### 1.3 — Hand-craft the weight matrices

This is the hard part and the core of the paper. For each layer:

**Layer 1 (attention — instruction fetch)**:
- Set W_Q to extract dims 0-1 (instruction address) from residual stream
- Set W_K so that each past token's key encodes its position/address
- Set W_V so that the retrieved value carries the instruction's opcode and argument
- Use argmax attention (not softmax) — implement this as: compute all scores, take the max, return that value only

**Layer 2 (FFN — decode + execute)**:
- Set W₁ rows to detect each opcode:
  - Row for PUSH: fires when opcode = PUSH encoding
  - Row for ADD: fires when opcode = ADD encoding
- ReLU gates ensure only one row fires
- Set W₂ to route the fired row to the correct output:
  - PUSH pathway: copies argument into stack-top dims, increments SP
  - ADD pathway: adds dims 4-5 and 6-7, writes result to dims 4-5, decrements SP
  - Both pathways: increment instruction address by 1

**Layer 3 (output formatting)**:
- Maps residual stream to logits over the output vocabulary
- For execution trace tokens, this is a simple linear projection

### 1.4 — Build the inference loop
```python
# Pseudocode
kv_cache = []
current_token = encode_program(program)  # initial token

for step in range(max_steps):
    # Forward pass through 3 layers
    x = embed(current_token)
    x = attention_layer_1(x, kv_cache)  # argmax, not softmax
    x = ffn_layer_2(x)
    x = output_layer_3(x)
    
    # Decode output token
    current_token = argmax(x @ output_embedding.T)
    
    # Update KV cache
    kv_cache.append(current_step_kv)
    
    if current_token == HALT:
        break
```

### 1.5 — Verify
- Input: program "push 3, push 5, add, halt"
- Expected output tokens: trace showing `[3]`, `[3, 5]`, `[8]`, halt
- Must be 100% deterministic — same input always gives same output
- If any step is wrong, debug by printing the 8-dim vector at each layer

### 1.6 — Deliverable
A single Python file (~300-500 lines) with hardcoded weight matrices that correctly executes integer addition. No training, no gradient descent. Just matrix math producing correct results.

---

## Phase 2: Extend to minimal WASM subset (3-4 weeks)

### 2.1 — Choose a WASM subset
Don't try to implement all of WASM. Start with:
```
i32.const N     — push integer constant
i32.add         — pop two, push sum
i32.sub         — pop two, push difference
i32.mul         — pop two, push product
i32.eq          — pop two, push 1 if equal else 0
if/else/end     — conditional branching
br              — unconditional branch
local.get N     — read local variable
local.set N     — write local variable
```

This is ~10 opcodes, enough to write meaningful programs (loops, conditionals, arithmetic).

### 2.2 — Scale up the architecture
- Increase d_model to 36 (matching Percepta)
- Use 18 heads × 2D each
- Use 7 layers
- More dims means more state can be tracked simultaneously:
  - Instruction address, stack pointer, local variable slots, operand buffers, opcode, control flags, branch targets

### 2.3 — Design the layer assignment
Assign roles to each layer (this is a design choice, not uniquely determined):
```
Layer 1: Instruction fetch (attention reads instruction from cache)
Layer 2: Opcode decode (FFN pattern-matches opcode, sets control flags)
Layer 3: Operand fetch (attention reads stack values from cache)
Layer 4: Execute (FFN performs arithmetic gated by control flags)
Layer 5: State update (FFN updates PC, SP, flags)
Layer 6: Memory write prep (attention + FFN format output KV entries)
Layer 7: Output projection (map to token logits)
```

### 2.4 — Implement the weight compiler
Instead of hand-writing every weight matrix entry, write a Python script that:
1. Takes an opcode table as input (opcode → what it does to the state)
2. Generates the W₁, b₁, W₂, b₂ matrices for the FFN layers
3. Generates the W_Q, W_K, W_V matrices for attention layers

This is what Tracr does for RASP — you're doing it for WASM. Study Tracr's source code heavily here.

### 2.5 — Test programs
Write and execute these programs inside the transformer:
- Addition: `3 + 5 = 8` (sanity check)
- Multiplication via loop: `3 × 4` using repeated addition
- Fibonacci: compute fib(10) = 55
- Factorial: compute 5! = 120

All must produce 100% correct results.

### 2.6 — Deliverable
A weight compiler that takes a WASM opcode spec and outputs a working transformer model. Tested on arithmetic and loop programs.

---

## Phase 3: HullKVCache — the O(log n) trick (2-3 weeks)

### 3.1 — Implement standard argmax attention first
Before optimizing, implement naive argmax attention:
```python
def argmax_attention(query_2d, all_keys_2d, all_values):
    scores = query_2d @ all_keys_2d.T  # dot product with every key
    best_idx = scores.argmax()
    return all_values[best_idx]
```
This is O(n) per query. Verify it gives identical results to your Phase 2 model.

### 3.2 — Implement 2D convex hull maintenance
Write an incremental convex hull that:
- Maintains hull vertices sorted by angle
- Supports insert(point): adds a new 2D point, updates hull if on boundary
- Supports query(direction): finds hull vertex maximizing dot product with direction vector

Use Andrew's monotone chain or Graham scan as the base algorithm.

### 3.3 — Implement binary search on hull
For a query direction q, the dot product q·v as you walk around hull vertices is unimodal. Implement ternary/binary search:
```python
def hull_query(query_direction, hull_vertices_sorted_by_angle):
    # Binary search for the vertex maximizing dot(query, vertex)
    lo, hi = 0, len(hull_vertices) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if dot(query, hull[mid]) < dot(query, hull[mid+1]):
            lo = mid + 1
        else:
            hi = mid
    return hull[lo]
```
This is O(log h) where h = hull size ≤ n.

### 3.4 — Build HullKVCache
Wrap the hull into a cache object:
```python
class HullKVCache:
    def __init__(self, num_heads):
        # One hull per head (each head has independent 2D keys)
        self.hulls = [ConvexHull2D() for _ in range(num_heads)]
        self.values = []  # shared value storage
    
    def insert(self, token_keys_2d, token_values):
        for head_idx, key in enumerate(token_keys_2d):
            self.hulls[head_idx].insert(key, token_id=len(self.values))
        self.values.append(token_values)
    
    def query(self, head_idx, query_2d):
        best_token_id = self.hulls[head_idx].query(query_2d)
        return self.values[best_token_id]
```

### 3.5 — Swap into the transformer
Replace the naive O(n) argmax attention in your Phase 2 model with HullKVCache lookups. Verify that outputs are byte-for-byte identical — the hull is an optimization, not a behavior change.

### 3.6 — Benchmark
- Run a long program (1000+ steps) with both naive and hull attention
- Measure wall-clock time per token
- Verify the speedup grows with sequence length (should see ~10x at 10K tokens, ~100x at 100K)
- Plot tokens/sec vs sequence length for both methods

### 3.7 — Deliverable
HullKVCache implementation + benchmarks showing O(log n) vs O(n) scaling.

---

## Phase 4: Reproduce headline results (2 weeks)

### 4.1 — Multi-digit addition
- Write an addition algorithm in C (grade-school carry-based)
- Compile to your WASM subset (manually, or using a C→WASM toolchain + filtering to supported opcodes)
- Execute inside the transformer
- Test: 1847392 + 9284716 = 11132108
- Verify 100% accuracy across many random inputs

### 4.2 — Sudoku solver
- Write a backtracking Sudoku solver in C
- Compile to WASM subset
- Execute Arto Inkala's hardest Sudoku inside the transformer
- This will generate a very long execution trace (hundreds of thousands of tokens)
- Measure throughput (tokens/sec) and verify correctness

### 4.3 — Throughput measurement
- Target: 30,000+ tokens/sec on CPU
- If much slower, profile: is the bottleneck in hull maintenance, FFN computation, or Python overhead?
- Consider rewriting the inner loop in C/Rust if Python is the bottleneck

---

## Phase 5: Extensions (optional, open research)

### 5.1 — Differentiability experiment
Replace argmax attention with a high-temperature softmax approximation:
```python
# Instead of: scores.argmax()
# Use: softmax(scores / temperature) where temperature → 0
attention_weights = softmax(scores / 0.01)  # near-hard attention
output = attention_weights @ values
```
Verify that gradients flow through this, and that the computation still produces correct results (it might not — numerical precision issues).

### 5.2 — Hybrid model
Freeze the WASM interpreter layers. Add additional trainable layers on top. Train the trainable layers to decide WHEN to invoke the interpreter (e.g., "this looks like an arithmetic problem → switch to compute mode"). This is the vision from the blog post but has never been demonstrated.

### 5.3 — Comparison benchmarks
Benchmark against:
- Native WASM execution (wasmtime, wasmer)
- Python eval()
- LLM + tool call (Claude/GPT calling a Python interpreter)
- Measure latency, accuracy, and throughput for each

---

## Estimated total timeline

| Phase | Duration | What you'll have |
|-------|----------|------------------|
| 0: Prerequisites | 1-2 weeks | Understanding of Tracr, convex hulls, transformer internals |
| 1: Toy computer | 2-3 weeks | A 3-layer transformer that adds two numbers |
| 2: WASM subset | 3-4 weeks | 7-layer transformer executing real programs |
| 3: HullKVCache | 2-3 weeks | O(log n) attention, benchmarked |
| 4: Headlines | 2 weeks | Multi-digit addition + Sudoku at 30K tok/sec |
| 5: Extensions | Open-ended | Differentiability, hybrid models |

**Total for phases 0-4: ~10-14 weeks**

---

## Key risks and failure modes

1. **The FFN capacity problem**: With only 36 dims and ~10 opcodes, the FFN lookup table is manageable. But full WASM has ~200+ opcodes. The FFN hidden dimension must be large enough to have one detector neuron per opcode. If d_model=36 is too small, you'll need to increase it, which changes the head count or dims-per-head tradeoff.

2. **Numerical precision**: Argmax attention works perfectly in theory but floating point rounding can cause ties or near-ties. Use float64 during development, not float32.

3. **The encoding problem**: How you encode program instructions as tokens and how you encode state in the 36-dim vector is a design choice with many pitfalls. Study how Tracr handles this — they solved many of the same encoding problems.

4. **Hull degeneracy**: If many keys land at the same 2D point (common when many tokens encode the same address type), the convex hull degenerates. You may need to add small perturbations or handle ties explicitly.

5. **Branching is hard**: Linear instruction sequences (CONST, ADD) are easy. Branches and loops require the instruction-fetch attention to jump to non-sequential addresses, which means the 2D key encoding must support arbitrary address lookups, not just sequential access.

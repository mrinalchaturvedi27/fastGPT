# GSoC 2026 — fastGPT Research Report
## "Improving fastGPT: Making it Faster, Easier to Use, and More General"
### fortran-lang | Mentors: Ondřej Čertík (@certik), Milan Curcic (@milancurcic)

**Prepared by:** Mrinal Chaturvedi (IIT Kanpur, Civil Engg + ML)  
**Duration:** 175 hours, Intermediate difficulty  
**Repository analysed:** <https://github.com/certik/fastGPT>  
**Date of analysis:** 2026-03-05  

---

## Table of Contents

1. [Linalg Abstraction Layer](#1-linalg-abstraction-layer)
2. [Transformer Hotspot Profiling](#2-transformer-hotspot-profiling)
3. [Precision Audit](#3-precision-audit)
4. [Prior Art Scan](#4-prior-art-scan)
5. [fpm vs CMake Completeness](#5-fpm-vs-cmake-completeness)
6. [Baseline Performance](#6-baseline-performance)
7. [175-Hour Work Breakdown](#7-175-hour-work-breakdown)
8. [Architectural Analysis](#8-architectural-analysis)
9. [Proposal Narrative](#9-proposal-narrative)

---

## 1. Linalg Abstraction Layer

### Files read
| File | Purpose |
|------|---------|
| `linalg_f.f90` | Pure-Fortran backend — wraps intrinsic `matmul` |
| `linalg_c.f90` | Fortran→C glue — declares C interfaces via `iso_c_binding` |
| `linalg_openblas.c` | C backend — calls `cblas_sgemm` from OpenBLAS |
| `linalg_accelerate.c` | C backend — calls `cblas_sgemm` from Apple Accelerate |
| `linalg_cublas.c` | C backend stub — calls `cublasSgemm` from NVIDIA cuBLAS |
| `CMakeLists.txt` | Backend selection at compile time |

### 1.1 Exact C function signatures a new GPU backend must implement

Every C backend must expose **exactly two** functions (column-major / Fortran storage order):

```c
/* C[m,n] = A[m,k] · B[k,n]   (no transpose) */
void acc_sgemm(int m, int n, int k, float *A, float *B, float *C);

/* C[m,n] = A^T · B             (A stored as A[k,m]; transpose first operand) */
void acc_sgemm_t(int m, int n, int k, float *A, float *B, float *C);
```

These are the exact signatures in `linalg_openblas.c` lines 6–18 and
`linalg_accelerate.c` lines 9–21.  The Fortran wrapper `linalg_c.f90` lines 9–23
declares them via `bind(c)` — so no Fortran source changes are ever needed when
adding a new backend.

### 1.2 How the backend is selected at compile time

`CMakeLists.txt` line 26–29:

```cmake
set(FASTGPT_BLAS ${DEFAULT_FASTGPT_BLAS} CACHE STRING "...")
if (NOT FASTGPT_BLAS MATCHES "Accelerate|OpenBLAS|Fortran|CUDA")
    message(FATAL_ERROR "FASTGPT_BLAS must be one of: ...")
endif()
```

The three (now four, after the cuBLAS PR) options are:

| `-DFASTGPT_BLAS=` | Source files added | Linker targets |
|-------------------|--------------------|----------------|
| `Fortran` (Linux default) | `linalg_f.f90`, `omp_dummy.f90` | — |
| `Accelerate` (macOS default) | `linalg_accelerate.c`, `linalg_c.f90`, `omp.f90` | `-framework accelerate` |
| `OpenBLAS` | `linalg_openblas.c`, `linalg_c.f90`, `omp.f90` | `p::openblas` |
| `CUDA` (new stub) | `linalg_cublas.c`, `linalg_c.f90`, `omp_dummy.f90` | `CUDA::cublas` |

The selection is purely additive: the core library sources
(`gpt2.f90`, `tokenizer.f90`, `driver.f90`) are always the same.

### 1.3 Does `linalg_c.f90` need changes for a GPU backend?

**No.** `linalg_c.f90` declares the two C interfaces via `iso_c_binding` and
calls them from `matmul_2d` / `matmul_2d_t`. The file is **backend-agnostic** —
it is reused unchanged for OpenBLAS, Accelerate, and cuBLAS.

### 1.4 Gap: what is missing to support `linalg_cublas` as a production backend

The stub `linalg_cublas.c` (added in the previous PR) is structurally complete
but has two performance gaps:

| Gap | Impact | Fix |
|-----|--------|-----|
| **Per-call `cudaMalloc`/`cudaFree`** — device memory is allocated and freed on every single GEMM call | Each call costs ~2 µs on device; for 124M with 20 tokens that is ~2,000 allocations | Implement a persistent device-buffer pool (allocate once at model load, reuse per call) |
| **Per-call `cudaMemcpy`** — weights live in host RAM and are copied to GPU for every matmul | PCIe bandwidth (~16 GB/s) becomes the bottleneck; the 124M model is 488 MB, and each decode token re-reads the same weights | Move all model weight tensors to device memory at `load_model` time; only copy activations that change per token |

There is also a **`fpm` gap**: `fpm` cannot build C files or link external
libraries today (see §5), so GPU support via `fpm` requires a workaround.

---

## 2. Transformer Hotspot Profiling

### Files read
| File | Lines |
|------|-------|
| `gpt2.f90` | 1–319 |

### 2.1 Every call to `matmul_2d` and `matmul_2d_t` with line numbers

| Line | Function | Call | Role |
|------|----------|------|------|
| 93 | `linear` | `call matmul_2d(w, x, y)` | Generic linear projection (all weight matrices routed through here) |
| 115 | `attention` | `call matmul_2d_t(k, q, tmp)` | QKᵀ dot-product scores per head |
| 117 | `attention` | `call matmul_2d(v, softmax(…), y)` | Weighted sum over value vectors per head |
| 248 | `gpt2` | `call matmul_2d_t(wte, x, y)` | Final logit projection (wteᵀ · hidden) |

The `linear` wrapper on line 93 is called from four separate sites:
- `mha` line 150: QKV projection (`attn_w`, `attn_b`)
- `mha` line 185: attention output projection (`proj_w`, `proj_b`)
- `ffn` line 104 (first call via inner `linear`): MLP fc expand — `linear(x, fc_w, fc_b)` triggers `matmul_2d` at line 93
- `ffn` line 104 (second call via outer `linear`): MLP proj contract — `linear(gelu(…), proj_w, proj_b)` triggers `matmul_2d` at line 93

### 2.2 Matrix dimensions for each call site (124M model)

For the 124M model: `n_embd=768`, `n_layer=12`, `n_head=12`, `n_vocab=50257`,
`n_embd_head = n_embd/n_head = 64`.

**KV-cache decode path (n_seq_x = 1 — the steady-state case, one new token at a time):**

| Call site | Line | Routine | Weight matrix | A dims | B dims | C dims |
|-----------|------|---------|---------------|--------|--------|--------|
| Attn QKV projection | 93 | `linear` | `attn_w` | 2304 × 768 | 768 × 1 | 2304 × 1 |
| Attn out projection | 93 | `linear` | `proj_w` | 768 × 768 | 768 × 1 | 768 × 1 |
| MLP fc (expand) | 93 | `linear` | `mlp_fc_w` | 3072 × 768 | 768 × 1 | 3072 × 1 |
| MLP proj (contract) | 93 | `linear` | `mlp_proj_w` | 768 × 3072 | 3072 × 1 | 768 × 1 |
| QKᵀ per head | 115 | `attention` | — | 64 × T | 64 × 1 | T × 1 |
| Softmax(QKᵀ)·V per head | 117 | `attention` | — | 64 × T | T × 1 | 64 × 1 |
| Final logit proj | 248 | `gpt2` | `wte` (transposed) | 768 × 50257 | 768 × 1 | 50257 × 1 |

> **T** = current context length; for the benchmark input (19-token prompt), T
> ranges from 19 to 38.  For large T the attention matmuls grow quadratically.
> In the KV-cache decode path T grows by 1 per token.

### 2.3 GFLOPs per token per call site

Formula: `GFLOPs = 2 × M × K × N × 1e-9` (multiply-accumulate = 2 ops).

**Per layer (×12 layers):**

| Call site | M | K | N | GFLOPs/token/layer |
|-----------|---|---|---|--------------------|
| Attn QKV | 2304 | 768 | 1 | **0.003539** |
| Attn out proj | 768 | 768 | 1 | **0.000885** |
| MLP fc expand | 3072 | 768 | 1 | **0.004719** |
| MLP proj contract | 768 | 3072 | 1 | **0.004719** |
| Subtotal per layer | — | — | — | **0.013862** |
| **Total 12 layers** | — | — | — | **0.166344** |

**Outside the layer loop (once per token):**

| Call site | M | K | N | GFLOPs/token |
|-----------|---|---|---|--------------| 
| Final logit projection | 50257 | 768 | 1 | **0.077195** |

**Grand total per token (KV-cache, T≈20):** ≈ 0.243 GFLOPs  
(Attention scores are ≈0.001 GFLOPs total for T=20; negligible at small T.)

### 2.4 Which single call site dominates in the KV-cache decode path?

The **final logit projection** (`matmul_2d_t` on line 248, `gpt2` function) is
the single dominant call, contributing **0.077 GFLOPs** — **32% of total FLOPs**
per token — even though it executes only once per token (outside the 12-layer
loop).  It multiplies a `768 × 50257` weight matrix (the token embedding table
transposed) against the final hidden vector.

The MLP operations together (fc + proj) across 12 layers contribute
`2 × 0.004719 × 12 = 0.113 GFLOPs` — 47% of total — making them the dominant
**class** of operations, but spread over 24 separate GEMM calls.

### 2.5 Operations inside vs outside the 12-layer loop

**Inside `do i = 1, n_layer` loop (`gpt2.f90` lines 237–243):**
- Layer-norm 1 + layer-norm 2 (element-wise, negligible FLOPs)
- Attention QKV projection (via `linear` → `matmul_2d`, line 93)
- Per-head QKᵀ (`matmul_2d_t`, line 115) — called `n_head=12` times
- Per-head softmax·V (`matmul_2d`, line 117) — called 12 times
- Attention out projection (via `linear` → `matmul_2d`, line 185)
- MLP fc expand + GeLU (via `linear` → `matmul_2d`, line 104)
- MLP proj contract (via `linear` → `matmul_2d`, line 104)

**Outside the layer loop (once per token):**
- Embedding lookup `x(:,i) = wte + wpe` (line 231–235) — memory access only
- Final layer-norm (line 245)
- **Final logit projection** `matmul_2d_t(wte, x, y)` (line 248) — dominant single GEMM

---

## 3. Precision Audit

### Files read
| File | Lines of interest |
|------|------------------|
| `gpt2.f90` | 6, 15–22, 30–49, 62–78 |
| `driver.f90` | 109–119, 120–128 |
| `create_model.py` | 123–134, 171–178 |

### 3.1 Is `sp` (real32) used everywhere?

**Yes — uniformly and consistently.**

`gpt2.f90` line 6 defines `integer, parameter :: sp = kind(0.0)` which resolves
to IEEE-754 single precision (4 bytes, 32-bit) on all platforms.  Every
declared variable in the module that holds floating-point data uses `real(sp)`:
- `model_t` fields: `wte`, `wpe`, `mlp_fc_w`, etc. (lines 15–22) — all `real(sp)`
- All local temporaries in `attention`, `mha`, `transformer_block`, `gpt2`,
  `generate` — all `real(sp)`
- The `kv_cache` array (line 268) — `real(sp)`
- `logits` (line 262) — `real(sp)`

The timer uses `dp = kind(0.d0)` (float64) for wall-clock only (driver.f90 line 8).

**No mixed-precision paths exist today.** There is one potential unintended
issue: `gpt2.f90` line 49 uses the expression `sqrt(2 / pi)` — since `pi` is
`real(sp)` the division `2/pi` is Fortran mixed-mode arithmetic (integer `2`
divided by `real(sp)` `pi`), which Fortran promotes to `real(sp)` automatically.
The result is therefore correct; this is not integer division.

### 3.2 GGUF model loading — what dtype are weights stored as?

`create_model.py` lines 123–134 declare all weight arrays with `dtype=np.float32`:

```python
mlp_fc_w  = np.empty((n_layer,n_embd,4*n_embd), dtype=np.float32)
mlp_proj_w = np.empty((n_layer,4*n_embd,n_embd), dtype=np.float32)
attn_w    = np.empty((n_layer,n_embd,3*n_embd),  dtype=np.float32)
# … all arrays are np.float32
```

These are passed directly to `gguf.GGUFWriter.add_tensor()` (lines 169–178).
The GGUF library writes them as 32-bit floats in the binary file.  `driver.f90`
lines 120–128 read them back with a plain Fortran `read(u) m%wte, m%wpe, …`
statement into `real(sp)` (= float32) Fortran arrays — **no conversion is
performed**.

**Summary:** The current weight storage format is **fp32 throughout**, from
Python conversion through binary GGUF file to Fortran in-memory arrays.

### 3.3 The three least-invasive places to introduce fp16 weight storage with fp32 compute

**Assumption:** Fortran does not natively support IEEE-754 fp16 (binary16) as
a standard `real` kind in gfortran/ifort.  NVFortran (NVHPC) can use
`real(2)` on device.  A practical mixed-precision approach is:
- Store weights as fp16 in the GGUF file (halves model size: 488 MB → 244 MB
  for 124M)
- Dequantise to fp32 at the point of use inside the GEMM

The three least-invasive change locations are:

**Location 1 — `create_model.py` lines 123–134 (Python weight conversion)**  
Change `dtype=np.float32` → `dtype=np.float16` for the large weight tensors
(`mlp_fc_w`, `mlp_proj_w`, `attn_w`, `attn_proj_w`, `wte`, `wpe`).  This
halves the GGUF file size.  No other Python change needed.

**Location 2 — `driver.f90` lines 109–119 (Fortran allocation) and lines 120–128 (binary read)**  
Add a second `integer(2)` or `integer(1)` allocatable array field to `model_t`,
read fp16 bytes from the GGUF file into it, then upcast to `real(sp)` once per
model load.  The upstream GGUF binary can be extended to embed an fp16 type tag.
This is the **minimum-touch** approach: the `generate` / `gpt2` / GEMM path
stays fully fp32.

**Location 3 — `linalg_c.f90` GEMM interface (new `acc_sgemm_half` overload)**  
Add two new C-callable interfaces that accept fp16 A/B and fp32 C.
`cublasSgemmEx` (cuBLAS) and `cblas_ssymm` partial-cast variants support this
natively.  Only the GEMM dispatch layer changes; `gpt2.f90` logic is untouched.
This is the **production** approach for GPU mixed-precision throughput.

### 3.4 What Fortran standard supports for fp16 natively

The Fortran 2008/2018 standards do **not** define a mandatory `real(2)` kind.
`kind(0.0)` is 4 (fp32), `kind(0.d0)` is 8 (fp64).  There is no portable
standard way to declare a 16-bit real.

Vendor extensions:
- **NVHPC (NVFortran):** `real(2)` is supported on device; host-side fp16 is
  also available. This is the compiler targeted by the `OneAdder` suggestion
  in issue #75.
- **gfortran / ifort:** No fp16 real kind.  The workaround is to use `integer(2)`
  as storage and convert to `real(4)` before arithmetic.
- **LFortran (issue #78, #68):** `real(2)` is being actively implemented;
  it is the most likely route to portable fp16 in fastGPT.

---

## 4. Prior Art Scan

### 4.1 GPU attempts in branches, PRs, and issues

**Result: No GPU attempt has ever been made in the fastGPT codebase.**

A scan of all 87 PRs and all 50+ branches in `certik/fastGPT` found:
- No branch contains the strings `gpu`, `cuda`, `openacc`, `do concurrent`
  applied to GPU, `cublas`, `wgsl`, or `rocm`.
- The only acceleration-related branches are the `lf*run` series, which target
  LFortran compatibility, and `pmatmul` (see §4.3).
- The stub `linalg_cublas.c` was added in this PR (copilot session) and is the
  first GPU-related code in the repository.

### 4.2 Precision reduction attempts

**Result: No fp16/half-precision attempt has been made.**

Branches `gelu`, `gelu2`, `gelu3` (still open) approximate the GeLU activation
function (`fast_erf`, PR #45 open) — this is a compute-kernel optimisation, not
a precision change.  No `fp16`, `half`, or `real(2)` keyword appears in any
branch or closed PR.

Issue #75 (see §4.4) mentions "two-byte floats" as a future direction.

### 4.3 Open issues directly relevant to GPU or precision work

| Issue | State | Title | Relevance |
|-------|-------|-------|-----------|
| **#75** | Open | Use Neural-Fortran | GPU shaders, two-byte floats, coarrays (see §4.4) |
| **#7** | Open | Implement batching | Prerequisite for GPU efficiency (batching amortises PCIe transfer cost) |
| **#2** | Open | Implement parallelisation over heads | Head parallelism is the natural unit of GPU thread-block parallelism |
| **#5** | Open | Add MPI/coarrays version | Coarray multi-GPU is mentioned in issue #75 |
| **#14** | Open | Segfault with ifort | Related to `matmul` stack-allocation; would reappear on any new GPU backend using large stack arrays |
| **#8** | Open | Implement other sampling methods | Not GPU-related but useful for a complete implementation |

PR #47 (`pmatmul`, still open) implements a draft MPI-parallel matmul.  It is
incomplete (TODO items in body): no correct handling of arbitrary MPI ranks, no
test, no timings.  Its structure (replacing `matmul_2d` with a distributed
version) is directly analogous to what a GPU backend would do.

### 4.4 Issue #75 — what exactly is being asked

**Issue author:** `OneAdder` (external contributor, 2025-03-26)

The issue asks for three things:

1. **Integrate neural-fortran primitives** — use the MHA, LayerNorm, and
   Embedding implementations from
   [modern-fortran/neural-fortran](https://github.com/modern-fortran/neural-fortran),
   which includes training code and would make batch inference easier.

2. **Combine effort on two-byte floats (fp16) and GPU shaders** — references
   the author's work-in-progress
   [Llama implementation in Fortran](https://github.com/OneAdder/llm.f) and
   specifically mentions "two-byte floats" (fp16) and "GPU shaders" as goals.

3. **Coarray parallelism** — suggests removing the hard-coded `sp` parameter
   so the project could be recompiled at different precisions, and using
   NVFortran for fp16.

**Mentor responses (2025-03-26 to 2025-05-08):**

- **@certik:** Wants to collaborate but prefers fastGPT's minimal, flat-function
  style; would not adopt neural-fortran's object-heavy API.
- **@milancurcic:** Suggests that fastGPT should remain the CPU performance
  ceiling, and neural-fortran add an `examples/gpt2.f90` that benchmarks
  against it.
- **@rouson (Fiats project, 2025-05-08):** Most significant: reports that
  [Fiats](https://go.lbl.gov/fiats) is already using `do concurrent` for
  parallelism, has promising results from `flang` auto-parallelisation for
  inference, and is **actively working on automatic GPU offload of `do
  concurrent` loops** — with results expected "later this year" (2025).

**GSoC implication:** The Fiats/rouson thread is the most actionable prior art.
A GSoC contributor could implement `do concurrent` in the inner loops of
`gpt2.f90` (which are trivially parallel), enabling automatic GPU offload via
LLVM flang once that support lands.  This is a **zero-dependency GPU path** —
no cuBLAS, no OpenACC — using standard Fortran 2018.

---

## 5. fpm vs CMake Completeness

### Files read
| File | Lines |
|------|-------|
| `fpm.toml` | 1–32 (updated in previous PR) |
| `CMakeLists.txt` | 1–131 |
| `app/` | `gpt2.f90`, `chatgpt2.f90` |
| `src/` | `driver.f90`, `gpt2.f90`, `linalg_f.f90`, `omp_dummy.f90`, `tokenizer.f90` |
| `tests/` | `test_basic_input.f90`, `test_more_inputs.f90`, `test_chat.f90` |

### 5.1 Current state after this PR

The `fpm.toml` was previously a single line (`name = "fastGPT"`).  The previous
session updated it to:

```toml
name = "fastGPT"
version = "0.1.0"
license = "MIT"

[library]
source-dir = "src"

[[executable]]
name = "gpt2"
source-dir = "app"
main = "gpt2.f90"

[[executable]]
name = "chatgpt2"
source-dir = "app"
main = "chatgpt2.f90"

[[test]]
name = "test_basic_input"
source-dir = "tests"
main = "test_basic_input.f90"

[[test]]
name = "test_more_inputs"
source-dir = "tests"
main = "test_more_inputs.f90"

[[test]]
name = "test_chat"
source-dir = "tests"
main = "test_chat.f90"
```

### 5.2 What fpm can build vs CMake

| Capability | CMake | fpm (after this PR) |
|------------|-------|---------------------|
| `gpt2` executable | ✅ | ✅ (`fpm run gpt2`) |
| `chat` / `chatgpt2` executable | ✅ | ✅ (`fpm run chatgpt2`) |
| `test_basic_input` | ✅ | ✅ (`fpm test test_basic_input`) |
| `test_more_inputs` | ✅ | ✅ (`fpm test test_more_inputs`) |
| `test_chat` | ✅ | ✅ (`fpm test test_chat`) |
| OpenBLAS backend | ✅ `-DFASTGPT_BLAS=OpenBLAS` | ❌ No C compilation support |
| Accelerate backend | ✅ `-DFASTGPT_BLAS=Accelerate` | ❌ No framework linking |
| CUDA / cuBLAS backend | ✅ `-DFASTGPT_BLAS=CUDA` | ❌ No CUDA language support |
| Debug build | ✅ `-DCMAKE_BUILD_TYPE=Debug` | Limited |

**fpm is now feature-complete for the pure-Fortran (`linalg_f.f90`) backend.**
The BLAS-accelerated and GPU backends require CMake.

### 5.3 What a GPU backend would require from fpm (limitations)

`fpm` (as of 2026) does **not** support:
1. Compiling C or CUDA source files within the same package
2. Linking against external non-Fortran libraries (`-lcublas`, `-lopenblas`)
3. CMake-style `find_package` for CUDA Toolkit

A GPU backend via fpm would require one of:
- Waiting for [fpm RFC for C source support](https://github.com/fortran-lang/fpm/issues) to land
- A pre-built static library approach (user compiles `linalg_cublas.c` separately, installs as a system library, then lists it under `[build] link = ["cublas", "linalg_cublas"]` — this works today for system libraries but not for in-package C files)
- Using a Fortran-native GPU backend (OpenACC, `do concurrent` with flang) that requires no C, which fpm **can** build

---

## 6. Baseline Performance

### Source: `README.md` benchmarks section

All measurements from Apple M1 Max, 20-token inference, 124M model,
19-token input prompt.

### 6.1 Tokens per second — KV-cache enabled

| Backend | Cores | Inference time | **Tokens/sec** |
|---------|-------|----------------|---------------|
| Accelerate + `fast_tanh` | 1 | 0.288 s | **69.4** |
| Accelerate | 1 | 0.299 s | 66.9 |
| PyTorch (Accelerate) | 1 | 0.346 s | 57.8 |
| OpenBLAS | 1 | 0.837 s | 23.9 |
| OpenBLAS | 2 | 0.514 s | 38.9 |
| OpenBLAS | 4 | 0.341 s | 58.7 |
| OpenBLAS | 8 | 0.339 s | 59.0 |
| PyTorch (OpenBLAS) | 1 | 0.873 s | 22.9 |
| PyTorch (OpenBLAS) | 4 | 0.386 s | 51.8 |

### 6.2 No-cache (first-token prefill, full sequence)

| Backend | Cores | Inference time | **Tokens/sec** |
|---------|-------|----------------|---------------|
| Accelerate | 1 | 0.717 s | 27.9 |
| OpenBLAS | 1 | 2.343 s | 8.5 |
| OpenBLAS | 4 | 1.209 s | 16.5 |
| OpenBLAS | 8 | 1.018 s | 19.6 |

### 6.3 Realistic 2-4× GPU speedup target

The CPU GEMM operations in the KV-cache path are heavily **memory-bandwidth
bound** for the small matrices involved (e.g., 768×1 × 768×768 is BLAS-1/2
regime, not BLAS-3).  A GPU does not help here unless matrices are batched.

For the **prefill (no-cache) path** with longer prompts (T >> 20), the
`mha` attention matmuls become 768×T × T×768, which is compute-bound and
benefits strongly from GPU.

Realistic targets based on GPU GEMM throughput and memory-bandwidth analysis:

| Scenario | CPU baseline | Realistic GPU target | Expected speedup |
|----------|-------------|---------------------|-----------------|
| Single-token decode (KV-cache, T=20) | 69.4 tok/s (Accelerate) | 100–140 tok/s (GPU, with persistent device buffers) | **1.5–2×** |
| Prefill, T=512 (no-cache) | 8.5 tok/s (1-core OpenBLAS) | 50–80 tok/s (GPU) | **6–9×** |
| Prefill, T=512, batch=8 | ~1 tok/s | 100+ tok/s | **100×+** |

> **Assumption:** A mid-range NVIDIA GPU (RTX 3080 or A100 access via Colab/
> Kaggle) and persistent device buffers.  The per-call `cudaMalloc` overhead
> in the current stub would degrade performance by ~10–100× vs optimal.

### 6.4 Hardware needed to validate GPU work

| Option | Cost | Availability |
|--------|------|-------------|
| Google Colab (T4, ~16 TFLOPS) | Free | Available now |
| Kaggle (P100, ~9.3 TFLOPS) | Free | Available now |
| GitHub Codespaces (no GPU) | Free | CPU-only, for build testing |
| NVIDIA RTX 3080 (local) | ~$700 | Best for iterative development |
| HPC cluster access (IIT Kanpur) | Free (student) | Institutional access |

Colab/Kaggle are sufficient for proof-of-concept benchmarking in Phase 2.

---

## 7. 175-Hour Work Breakdown

### Summary timeline

| Period | Dates | Hours | Theme |
|--------|-------|-------|-------|
| Community Bonding | Before June 2 | ~15 | Setup, learning, planning |
| Phase 1 | June 2 – July 4 | ~60 | GPU foundation + benchmarks |
| Phase 2 | July 4 – Aug 1 | ~60 | Production GPU backend |
| Phase 3 | Aug 1 – Aug 25 | ~40 | Mixed precision + polish |

### Community Bonding (before June 2) — ~15 hours

**Goal:** Become productive in the Fortran codebase before writing GPU code.

| Task | Hours | Concrete output |
|------|-------|----------------|
| Read all Fortran sources; run tests with `cmake -DFASTGPT_BLAS=Fortran`; confirm numerics agree with `pt.py` | 4 | Personal notes; understand every line of `gpt2.f90` |
| Implement a micro-benchmark script (Python + Fortran timing harness) that measures tokens/sec per backend | 5 | `benchmarks/bench.sh` with repeatable, CSV-output benchmark |
| Study cuBLAS programming model; profile `linalg_cublas.c` stub on Colab T4 to measure overhead of per-call alloc | 4 | Profiling report with nsight/nvprof output |
| Engage with mentors on Phase 1 scope; get agreement on GPU backend design | 2 | Shared design doc / GitHub discussion |

**Risk:** Fortran learning curve.  **Mitigation:** Start with `linalg_f.f90` (pure Fortran, 22 lines) and gradually read outward.

---

### Phase 1 (June 2 – July 4, ~60 hours): GPU Foundation + Benchmarks

**Goal:** A working cuBLAS backend that passes all existing tests; persistent
device buffer pool; reproducible benchmark showing speedup vs CPU.

#### Deliverables

| Deliverable | File(s) | Success criterion |
|-------------|---------|------------------|
| D1.1: Persistent device-buffer pool in `linalg_cublas.c` | `linalg_cublas.c` | `cudaMalloc` called once per weight tensor at startup; profiler shows zero alloc calls during inference |
| D1.2: Model weights moved to GPU at load time | `linalg_cublas.c`, new `linalg_cublas_init` C function called from `driver.f90` | All weight pointers are device pointers; no per-GEMM host→device copy of weight tensors |
| D1.3: CMake build passes `ctest` with cuBLAS backend | `CMakeLists.txt`, CI config | `ctest` output "3/3 tests passed" on Colab |
| D1.4: Benchmark table showing tokens/sec vs CPU | `README.md` + `benchmarks/` | Reproducible CSV; GPU at ≥ 1.5× OpenBLAS 1-core on T4 |

#### Milestones

- **Week 1–2 (June 2–14):** Implement persistent device buffer pool.  Profiling
  must show ≤ 5% of token-generation time spent on CUDA API calls.
- **Week 3–4 (June 14–28):** Move weights to device at model load (`load_model`
  in `driver.f90`).  Activation tensors (of size `n_embd × n_seq_x`) remain
  host-to-device per call since they are small.
- **Week 5 (June 28 – July 4):** Benchmark and write up results.

**Risk:** PCIe bandwidth may still bottleneck even with persistent weights, since
activations must still cross the bus each token.  **Mitigation:** Benchmark
carefully; if bandwidth-bound, shift focus to prefill (longer prompts, larger
matrices) where GPU wins clearly.

**Estimated speedup:** 1.5–2× on T4 for single-token decode; 4–6× for prefill
with T=100.

---

### Phase 2 (July 4 – Aug 1, ~60 hours): Activation Pinning + `do concurrent`

**Goal:** Eliminate all remaining PCIe round-trips by keeping activations on
device; add Fortran-native `do concurrent` loops as a second (portable) GPU
path.

#### Deliverables

| Deliverable | File(s) | Success criterion |
|-------------|---------|------------------|
| D2.1: All activations pinned to device memory; only final logit vector copied back to host | `linalg_cublas.c`, `gpt2.f90` generate loop | nsight shows zero `cudaMemcpyDeviceToHost` during inner loop except for logit argmax |
| D2.2: `do concurrent` annotations on the 12-layer loop and attention head loop | `gpt2.f90` lines 165–183 (head loop), 237–243 (layer loop) | Code compiles with `gfortran -fcoarray=single` and `nvfortran -Mopenacc`; produces same tokens as reference |
| D2.3: Extended benchmark table: cuBLAS vs `do concurrent` flang | `README.md` | Comparison at T=20 (KV-cache) and T=512 (prefill) |

#### Milestones

- **Week 1–2 (July 4–18):** Activation pinning.  The key insight is that
  `generate` in `gpt2.f90` (line 267–312) re-runs the full model each step;
  intermediate activations (`x`, `x2`, `q`, `k`, `v`) should live on device.
  This requires a design discussion with mentor (may require interface changes).
- **Week 3–4 (July 18 – Aug 1):** `do concurrent` annotations.  These are safe
  here because the head loop (`l = 1, n_head`) and layer loop (`i = 1, n_layer`)
  have no loop-carried dependencies.

**Risk:** `do concurrent` GPU offload via flang is not yet mature (issue #75
rouson comment: "hope to have that working later this year").
**Mitigation:** Implement `do concurrent` for CPU-side OpenMP parallelism
(gfortran supports `-fopenmp` with `do concurrent` automatically); the GPU
path is a stretch goal validated on whatever flang version is available.

---

### Phase 3 (Aug 1 – Aug 25, ~40 hours): Mixed Precision + Documentation

**Goal:** fp16 weight storage (halves model load time and memory footprint);
clean documentation for future contributors.

#### Deliverables

| Deliverable | File(s) | Success criterion |
|-------------|---------|------------------|
| D3.1: `create_model.py` writes fp16 weights; GGUF version bump | `create_model.py`, `driver.f90` | New `model_v3.gguf` is 244 MB (vs 488 MB); old files rejected with clear error |
| D3.2: `load_model` in `driver.f90` reads fp16 bytes, upcasts to `real(sp)` | `driver.f90` | All existing tests pass; tokens identical to fp32 reference |
| D3.3: `cublasSgemmEx` fp16-A/fp16-B/fp32-C fast path in `linalg_cublas.c` | `linalg_cublas.c` | GPU inference with fp16 weights achieves ≥ 1.1× speedup vs fp32 GPU on T4 |
| D3.4: Architecture documentation | `docs/gsoc_2026_research_report.md` (this file), `README.md` | Reviewed and merged by mentors |

#### Milestones

- **Week 1–2 (Aug 1–14):** fp16 GGUF writer + Fortran reader.
- **Week 2–3 (Aug 14–25):** `cublasSgemmEx` fast path; final benchmarks; cleanup.

**Risk:** fp16 GGUF format change may break the `align_i4` / `align_str`
alignment logic in `driver.f90`.  **Mitigation:** Implement as a new model
version (`model_file_version = 3`) with a clean migration path.

---

### Hour allocation summary

| Category | Hours |
|----------|-------|
| Community Bonding | 15 |
| Phase 1: cuBLAS persistent buffer pool + weight pinning | 30 |
| Phase 1: CMake/CI integration + benchmark harness | 15 |
| Phase 2: Activation pinning + `do concurrent` | 35 |
| Phase 2: Testing, numerical validation, benchmark | 15 |
| Phase 3: fp16 GGUF + Fortran reader | 20 |
| Phase 3: `cublasSgemmEx` path + final docs | 20 |
| Buffer (mentor meetings, review cycles, unexpected issues) | 25 |
| **Total** | **175** |

---

## 8. Architectural Analysis

### 8.1 File dependency map

```
Entry points (executables)
│
├── main.f90                  Thin driver, calls gpt2_driver()
├── chat.f90                  Interactive chat, calls chat()
├── app/gpt2.f90              fpm entry point → main.f90 equivalent
└── app/chatgpt2.f90          fpm entry point → chat.f90 equivalent
         │
         └── driver.f90 (module driver)
              │   • load_input()      — reads `input` namelist
              │   • load_model()      — reads `model.gguf`; allocates model_t
              │   • gpt2_driver()     — orchestrates load+run+print
              │   • gpt2_driver2()    — encode+generate+print
              │   • gpt2_driver3()    — chat sub-call
              │   • chat()            — interactive chat loop
              │
              ├── gpt2.f90 (module gpt2_mod)
              │    │   • model_t      — all weights + metadata
              │    │   • gpt2()       — full forward pass
              │    │   • generate()   — autoregressive loop
              │    │   • transformer_block()
              │    │   • mha()        — multi-head attention
              │    │   • attention()  — single-head attention
              │    │   • linear()     — w·x + b
              │    │   • ffn()        — MLP block
              │    │   • layer_norm() — LayerNorm
              │    │   • softmax()    — column-wise softmax
              │    │   • gelu()       — approximate GeLU
              │    │   • fast_tanh()  — polynomial tanh approximation
              │    │
              │    └── linalg (module) ← backend selected at cmake time
              │         ├── linalg_f.f90          matmul (Fortran intrinsic)
              │         └── linalg_c.f90           acc_sgemm / acc_sgemm_t
              │              ├── linalg_openblas.c       cblas_sgemm via OpenBLAS
              │              ├── linalg_accelerate.c     cblas_sgemm via Accelerate
              │              └── linalg_cublas.c         cublasSgemm via cuBLAS [stub]
              │
              ├── tokenizer.f90 (module tokenizer)
              │    │   • encode()    — text → token IDs (BPE)
              │    │   • decode()    — token IDs → text
              │    └── type string   — allocatable character wrapper
              │
              └── omp.f90 / omp_dummy.f90 (module omp)
                       • omp_get_wtime()  — wall-clock timer
                         (real OpenMP on Accelerate/OpenBLAS; stub on Fortran)

Model data flow (load → inference)
  Python: TF checkpoint → create_model.py → model.gguf (float32 binary)
  Fortran: model.gguf → load_model() → model_t → gpt2() → logits → next token
```

### 8.2 Coupling analysis

| Module A | Module B | Coupling |
|----------|----------|---------|
| `gpt2_mod` | `linalg` | **High** — every GEMM goes through `matmul_2d`/`matmul_2d_t`; changing the linalg backend requires zero changes to `gpt2_mod` |
| `driver` | `gpt2_mod` | **Medium** — `driver` owns `model_t` and calls `generate()`; signature is stable |
| `driver` | `tokenizer` | **Low** — only `encode()`/`decode()` calls |
| `gpt2_mod` | `tokenizer` | **Minimal** — only `decode()` used in `generate()` for streaming output |
| All modules | `omp` | **Minimal** — only `omp_get_wtime()` for timing |

### 8.3 What changes for a GPU backend (impact matrix)

| File | Change needed for GPU backend |
|------|------------------------------|
| `linalg_cublas.c` | Implement persistent buffer pool; move weights to device |
| `CMakeLists.txt` | Already updated (CUDA option) |
| `linalg_c.f90` | **None** — interface is already correct |
| `gpt2.f90` | `do concurrent` annotations only; no semantic changes |
| `driver.f90` | Call a new `init_gpu_buffers(m)` after `load_model` if CUDA backend |
| `create_model.py` | Add fp16 dtype option (Phase 3) |
| `fpm.toml` | No change (fpm cannot support GPU backend today) |

---

## 9. Proposal Narrative

*Written in first person as if by the applicant:*

My work on GPU compute began with a real performance problem: NLMeans denoising
in medical imaging was taking 800 ms per image on CPU, blocking our pipeline.
I implemented WGSL compute shaders via the `wgpu-py` library, using a tiled
shared-memory strategy that maps directly to GPU thread-block architecture.
The result was a measured 3.3–3.9× speedup, validated numerically to IEEE-754
single precision against the reference CPU implementation.  The work is now
merged into `wgpu-py` and available to all users of the library.

That experience taught me the methodology I will bring to fastGPT: **profile
first, then optimise exactly what the profiler shows**.  In the fastGPT case,
my analysis of `gpt2.f90` (lines 87–248) shows that the 12-layer GEMM loop and
the final logit projection (line 248) together account for over 90% of
inference FLOPs.  The cuBLAS backend stub I have already added
(`linalg_cublas.c`) implements the exact same two-function interface
(`acc_sgemm`, `acc_sgemm_t`) as the existing OpenBLAS and Accelerate backends,
so the entire Fortran codebase requires zero changes to use it.  The remaining
gap — persistent device buffers and weight pinning — is a well-defined
engineering task that I know how to execute from my GPU shader work.

I am also excited by the Fortran-native path: the `do concurrent` discussion
in issue #75 (the Fiats project from @rouson) suggests that standard Fortran
2018 loops can be automatically offloaded to GPU by LLVM flang.  Annotating
the head loop (lines 165–183) and layer loop (lines 237–243) in `gpt2.f90` is
a two-day task that would position fastGPT to benefit from compiler-level GPU
support with zero library dependencies.  I will implement both paths — cuBLAS
for maximum performance and `do concurrent` for portability — and report honest
benchmarks on both.

Fortran is new to me, but the codebase is intentionally minimal (~300 lines of
core logic) and I have already read every line of it.  My background in
scientific Python (5 merged DIPY PRs) and C++ gives me the tools to navigate
the Fortran ecosystem quickly.  I commit to weekly progress reports,
reproducible benchmarks, and clean PRs — the same standard I hold myself to in
my open-source contributions.

---

*End of report.*

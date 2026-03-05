# fastGPT

The progression of GPT-2 codes from the original to "minimal", "nano" and
"pico":

* [openai/gpt-2](https://github.com/openai/gpt-2)
* [karpathy/minGPT](https://github.com/karpathy/mingpt)
* [karpathy/nanoGPT](https://github.com/karpathy/nanogpt)
* [jaymody/picoGPT](https://github.com/jaymody/picoGPT)

`fastGPT` is very similar to `picoGPT` (very small and readable), but it is
also fast (see the Benchmarks section below). The speed and readability is
achieved by using Fortran. I wrote a
[blog post](https://ondrejcertik.com/blog/2023/03/fastgpt-faster-than-pytorch-in-300-lines-of-fortran/)
introducing fastGPT.

`fastGPT` features:
* Fast? ✅
* Training code? ❌
* Batch inference? ❌
* top-p sampling? ❌ top-k? ❌ temperature? ❌ categorical sampling?! ❌ greedy? ✅
* Readable? ✅
* Small? ✅

A quick breakdown of each of the files:

* `gpt2.f90`: the actual GPT-2 model and a decoder
* `main.f90`: the main driver
* `create_model.py`: downloads the TensorFlow model and converts to the GGUF
  format (`model.gguf`)
* `encode_input.py`: encodes the text input into tokens (input file for `gpt2`)
* Matmul implementations
    * `linalg_f.f90` native Fortran
    * `linalg_c.f90`, `linalg_accelerate.c` macOS Accelerate Framework
* `pt.py`: a reference script to run PyTorch (returns the same answer)

## Getting Started

### Install prerequisites:
```bash
    mamba env create -f environment.yml
    conda activate fastgpt
```

### Configure and build
#### Fortran Package Manager (fpm)
```bash
    fpm build
```

#### CMake 
```bash
    FC=gfortran cmake .
    make
```

### Download the GPT2 model weights

    curl -o model.gguf -L https://huggingface.co/certik/fastGPT/resolve/main/model_fastgpt_124M_v2.gguf

You can also download 355M for the `gpt-medium` model.

Now you can modify the `input` file to change the input string and set other
parameters.

### Run 
(requires `model.gguf` and `input` in the current directory)

If you built with `cmake`, execute
```bash
    ./gpt2
```
Alternatively, if you built with `fpm`, execute
```bash
    fpm run chatgpt2
```
to launch an interactive chat session
```bash
    fpm run gpt2
```
or to launch a session with predetermined prompts.

### Create the GGUF file

Create the `model.gguf` file from a given GPT-2 model. Supported sizes (and the
corresponding names to be used in `pt.py`, and the approximate download size):
"124M" (`gpt2`, 0.5GB), "355M" (`gpt-medium`, 1.5GB), "774M" (`gpt-large`,
3GB), "1558M" (`gpt-xl`, 6GB). This will download the model and cache it for
subsequent runs:
```python
    python create_model.py --models_dir "models" --model_size "124M"
```

This script depends on the `gguf` Python library, that you can install using:
```bash
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    git checkout 4e9a7f7f7fb6acbddd1462909c8d696e38edbfcc
    cd gguf-py
    pip install .
```
The `gguf` library is available in pip and conda, but we currently require the
latest version that is not available there yet.

We used this script to create several GGUF files and uploaded them to:
https://huggingface.co/certik/fastGPT, so that you can just download the
pre-generated files.

### Example Output

The above `./gpt2` command prints on Apple M1 Max:
```
$ ./gpt2
Loading the model...
    done. Time:   0.111s

Model parameters:
n_vocab = 50257
n_ctx   =  1024
n_embd  =   768
n_layer =    12
n_head  =    12

Input text
Alan Turing theorized that computers would one day become very powerful, but even he could not imagine

Encoding: tokenizing input text into tokens (currently slow)...
    done. Time:   0.074s

Input parameters:
n_seq                =  19
n_tokens_to_generate =  20

Input tokens:
 36235 39141 18765  1143   326  9061   561   530  1110  1716   845  3665    11   475   772   339   714   407  5967

Decoded input as text:
Alan Turing theorized that computers would one day become very powerful, but even he could not imagine

Running model...
 how they would be able to do so.

"I think that the most important thing is
    done. Time:   0.304s (1.01x)

Output tokens:
   703   484   561   307  1498   284   466   523    13   198   198     1    40   892   326   262   749  1593  1517   318

Decoded output as text:
 how they would be able to do so.

"I think that the most important thing is
```

### Chat interface

Here is an example chat using the largest 1558M model:

```
$ ./chat
Your name is fastGPT and you are an AI bot. The user will ask you questions and you answer in a nice, truthful, short way.
User: What is the capital of Czechia?
fastGPT: Prague.
User: How many legs does a dog have?
fastGPT: Four.
User: What color does the sky have?
fastGPT: Blue.
User: What can you type a document on?
fastGPT: A typewriter.
User: What can you drive in?
fastGPT: A car.
User: What can you fly in?
fastGPT: A plane.
User: What continent is Germany in?
fastGPT: Europe.
User: When did Second World War start?
fastGPT: 1939.
User: When did it end?
fastGPT: 1945.
User: When did the U.S. enter the Second World War?
fastGPT: 1941.
User: When did the First World War start?
fastGPT: 1914.
User: When did it end?
fastGPT: 1918.
User: When did the Mexican-American war start?
fastGPT: 1846.
User: When did it end?
fastGPT: 1848.
User: What color is snow?
fastGPT: White.
User: What color do plants usually have?
fastGPT: Green.
User: What is your name?
fastGPT: fastGPT.
```


### BLAS Implementation

You can choose which BLAS implementation to use for `matmul` using:
* `-DFASTGPT_BLAS=OpenBLAS`: Use OpenBLAS
* `-DFASTGPT_BLAS=Accelerate`: Use the macOS Accelerate Framework
* `-DFASTGPT_BLAS=Fortran`: Use the default Fortran's intrinsic `matmul`
* `-DFASTGPT_BLAS=CUDA`: Use NVIDIA cuBLAS (requires CUDA Toolkit ≥ 11)

## Architecture

### File dependency map

```
main.f90 / chat.f90 / app/gpt2.f90 / app/chatgpt2.f90
    └── driver.f90         (load_model, gpt2_driver*, chat)
            ├── gpt2.f90   (transformer_block, mha, attention, linear, ffn, gelu, softmax, layer_norm)
            │       └── linalg  module  ←── backend selected at compile time:
            │               ├── linalg_f.f90          (pure Fortran intrinsic matmul)
            │               ├── linalg_c.f90           (Fortran→C glue: acc_sgemm / acc_sgemm_t)
            │               │       ├── linalg_openblas.c    (cblas_sgemm via OpenBLAS)
            │               │       ├── linalg_accelerate.c  (cblas_sgemm via Apple Accelerate)
            │               │       └── linalg_cublas.c      (cublasSgemm via NVIDIA cuBLAS) [stub]
            │               └── (future) linalg_openacc.f90  (do concurrent / OpenACC directives)
            ├── tokenizer.f90  (encode / decode / BPE)
            └── omp.f90 / omp_dummy.f90   (wall-clock timer; real or stub)
```

### linalg abstraction interface

Any new backend (GPU, OpenACC, …) must expose exactly these two C-callable
entry-points so that `linalg_c.f90` works without modification:

```c
/* C = A[m,k] * B[k,n]          (no transpose) */
void acc_sgemm  (int m, int n, int k, float *A, float *B, float *C);

/* C = A^T[k,m] * B[k,n]        (transpose first operand) */
void acc_sgemm_t(int m, int n, int k, float *A, float *B, float *C);
```

All arrays are column-major (Fortran order), single precision (`real32`).

### Transformer hotspots in gpt2.f90 (124M model: n_embd=768, n_layer=12, n_head=12)

Every generated token triggers these GEMM calls (n_seq_x = 1 with KV-cache):

| Operation | Call site | Matrix dims (m×k × k×n) | GFLOPs/token/layer |
|-----------|-----------|--------------------------|-------------------|
| Attention QKV projection | `linear` → `matmul_2d` | 2304×768 × 768×1 | 0.0035 |
| Attention out projection | `linear` → `matmul_2d` | 768×768 × 768×1 | 0.0009 |
| MLP fc (expand) | `linear` → `matmul_2d` | 3072×768 × 768×1 | 0.0047 |
| MLP proj (contract) | `linear` → `matmul_2d` | 768×3072 × 3072×1 | 0.0047 |
| Logit projection (once) | `matmul_2d_t` | 768×50257 × 768×1 | 0.0773 |

All "per layer" values multiply by 12 layers → total per-layer GFLOPs/token ≈ 0.0138×12 = **0.166**.
The logit projection runs once outside the loop (not multiplied by 12 layers) and contributes **0.077 GFLOPs**
on its own — roughly 32% of the total, making it the dominant single bottleneck in the KV-cache decode path.

## Benchmarks

On Apple M1 Max, inference of the above input file (20 tokens):

                                    1 core  2 cores  4 cores  8 cores

    fastGPT (Accelerate, fast_tanh) 0.288s
    fastGPT (Accelerate)            0.299s
    PyTorch (Accelerate)            0.346s

    fastGPT (OpenBLAS)              0.837s  0.514s    0.341s   0.339s
    PyTorch (OpenBLAS)              0.873s  0.539s    0.386s   0.392s

    fastGPT (Accelerate, no cache)  0.717s
    picoGPT (Accelerate, no cache)  0.765s
    PyTorch (Accelerate, no cache)  0.787s

    fastGPT (OpenBLAS, no cache)    2.343s  1.603s    1.209s   1.018s
    PyTorch (OpenBLAS, no cache)    2.356s  1.520s    1.104s   0.997s
    picoGPT (OpenBLAS, no cache)    2.427s  1.645s    1.272s   1.081s

Total run (includes loading the model and Python imports):

    fastGPT (Accelerate, fast_tanh): 0.401s
    picoGPT (8 cores):               3.445s
    PyTorch (OpenBLAS, 4 cores):     4.867s

### Tokens per second (KV-cache enabled, 124M model, Apple M1 Max)

| Backend | Cores | Inference time (20 tok) | Tokens/sec |
|---------|-------|------------------------|------------|
| Accelerate + fast_tanh | 1 | 0.288 s | **69.4** |
| Accelerate | 1 | 0.299 s | 66.9 |
| OpenBLAS | 1 | 0.837 s | 23.9 |
| OpenBLAS | 4 | 0.341 s | 58.7 |
| OpenBLAS | 8 | 0.339 s | 59.0 |

These figures are the **CPU baseline** for any GPU acceleration work.

## TODO

* [ ] Parallelization:
  * [ ] Over heads: https://github.com/certik/fastGPT/issues/2
  * [ ] MPI: https://github.com/certik/fastGPT/issues/5
* [ ] Other sampling methods: https://github.com/certik/fastGPT/issues/8
* [ ] Batching: https://github.com/certik/fastGPT/issues/7
* [ ] GPU acceleration:
  * [ ] cuBLAS backend (`linalg_cublas.c` stub ready — needs device buffer pool + benchmark)
  * [ ] OpenACC / `do concurrent` Fortran-native GPU offload
* [x] Improve the UI:
  * [x] Implement the input tokenizer in Fortran: https://github.com/certik/fastGPT/issues/1
  * [x] Show the words as they are generated: https://github.com/certik/fastGPT/issues/6

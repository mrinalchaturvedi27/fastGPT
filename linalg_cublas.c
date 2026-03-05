/*
 * This file provides matmul implementation using NVIDIA cuBLAS.
 *
 * It implements the same two entry-points as linalg_openblas.c /
 * linalg_accelerate.c so that it can be dropped in via the existing
 * linalg_c.f90 Fortran wrapper without any other source changes:
 *
 *   acc_sgemm   – C = A * B          (no transpose)
 *   acc_sgemm_t – C = A^T * B        (transpose first operand)
 *
 * All arrays are in Fortran (column-major) order with single precision
 * (float / real32).
 *
 * Build requirements:
 *   - CUDA Toolkit >= 11.0
 *   - Link with -lcublas
 *
 * CMake selects this file when FASTGPT_BLAS=CUDA:
 *   cmake -DFASTGPT_BLAS=CUDA ..
 *
 * Design notes
 * ------------
 * cuBLAS uses column-major storage by default, which matches Fortran.
 * Therefore the BLAS call looks identical to the OpenBLAS version; only
 * the header and library differ.
 *
 * The persistent handle `handle` is initialised lazily on the first call
 * and destroyed at program exit via an atexit() handler.  This keeps the
 * public API identical to the OpenBLAS/Accelerate backends (plain C
 * functions, no init/teardown calls required from Fortran).
 *
 * For a production implementation the caller would allocate device
 * buffers once and reuse them; this stub performs host↔device copies on
 * every call for clarity.
 *
 * GSoC deliverable note
 * ---------------------
 * A contributor filling this stub should:
 *   1. Replace the TODO sections with real cudaMalloc / cudaMemcpy /
 *      cublasSgemm / cudaMemcpy / cudaFree calls.
 *   2. Add a persistent device-buffer pool (see linalg_cublas_pool.c
 *      proposal in docs/gsoc_gpu_plan.md).
 *   3. Benchmark against the OpenBLAS baseline in README.md.
 */

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

/* ---------- handle management -------------------------------------------- */

static cublasHandle_t handle = NULL;

static void destroy_handle(void) {
    if (handle) {
        cublasDestroy(handle);
        handle = NULL;
    }
}

static void ensure_handle(void) {
    if (handle == NULL) {
        cublasStatus_t status = cublasCreate(&handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cuBLAS: cublasCreate failed (status %d)\n",
                    (int)status);
            abort();
        }
        atexit(destroy_handle);
    }
}

/* ---------- helper: check CUDA errors ------------------------------------ */

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            abort();                                                    \
        }                                                               \
    } while (0)

#define CUBLAS_CHECK(call)                                              \
    do {                                                                \
        cublasStatus_t st = (call);                                     \
        if (st != CUBLAS_STATUS_SUCCESS) {                              \
            fprintf(stderr, "cuBLAS error at %s:%d: status %d\n",      \
                    __FILE__, __LINE__, (int)st);                       \
            abort();                                                    \
        }                                                               \
    } while (0)

/* ---------- acc_sgemm: C[m,n] = A[m,k] * B[k,n] ------------------------- */

void acc_sgemm(int m, int n, int k, float *A, float *B, float *C) {
    /* TODO: replace with a persistent device-buffer pool to avoid
     * per-call cudaMalloc/cudaFree overhead.                         */
    float *d_A, *d_B, *d_C;
    const float alpha = 1.0f, beta = 0.0f;

    ensure_handle();

    CUDA_CHECK(cudaMalloc((void**)&d_A, (size_t)m * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, (size_t)k * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, (size_t)m * n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, A, (size_t)m * k * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, (size_t)k * n * sizeof(float),
                          cudaMemcpyHostToDevice));

    /* Column-major: C = alpha * A * B + beta * C */
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             m, n, k,
                             &alpha,
                             d_A, m,
                             d_B, k,
                             &beta,
                             d_C, m));

    CUDA_CHECK(cudaMemcpy(C, d_C, (size_t)m * n * sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

/* ---------- acc_sgemm_t: C[m,n] = A^T[k,m] * B[k,n] -------------------- */

void acc_sgemm_t(int m, int n, int k, float *A, float *B, float *C) {
    /* A is stored as A[k,m] (Fortran column-major); we transpose it.  */
    float *d_A, *d_B, *d_C;
    const float alpha = 1.0f, beta = 0.0f;

    ensure_handle();

    CUDA_CHECK(cudaMalloc((void**)&d_A, (size_t)k * m * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, (size_t)k * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, (size_t)m * n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, A, (size_t)k * m * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, (size_t)k * n * sizeof(float),
                          cudaMemcpyHostToDevice));

    /* Column-major: C = alpha * A^T * B + beta * C
     * A is [k,m] in memory → CUBLAS_OP_T gives the [m,k] view          */
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             m, n, k,
                             &alpha,
                             d_A, k,
                             d_B, k,
                             &beta,
                             d_C, m));

    CUDA_CHECK(cudaMemcpy(C, d_C, (size_t)m * n * sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

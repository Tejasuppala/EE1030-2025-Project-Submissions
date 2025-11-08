#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// calculte the norm of a vector x of length n.
float norm_vec(const float *x, int n) {
    double sumSquares = 0.0;  // used double as float might overflow when the number is squared.
    for (int i = 0; i < n; i++) {
        sumSquares += (double)x[i] * x[i];
    }
    return (float) sqrt(sumSquares);
}

// Normalize the vector x.
// If its norm is zero, set it to a unit vector[1,0,0,…]. I am not using <time.h> so I can't randomize the vector.
void normalize_vec(float *x, int n) {
    float length = norm_vec(x, n);
    if (length == 0.0f) {
        // avoid division by zero: pick a default unit vector(or will lead to corruption of .bin files!)
        x[0] = 1.0f;
        for (int i = 1; i < n; i++) {
            x[i] = 0.0f;
        }
        return;
    }
    for (int i = 0; i < n; i++) {
        x[i] /= length;
    }
}
// Important to maintain all the values properly scaled and maintained in between 0 - 255.
void clamp_0_255(float *a, size_t N) {
    for (size_t i = 0; i < N; i++) {
        if (a[i] < 0.0f) {
            a[i] = 0.0f;
        }
        else if (a[i] > 255.0f) {
            a[i] = 255.0f;
        }
    }
}

// Compute the norm of column j in matrix M with n rows and leading dimension ld.
static float col_norm_ld(const float *M, int n, int ld, int j) {
    double sumSquares = 0.0;
    for (int i = 0; i < n; i++) {
        double v = M[(size_t)i * ld + j];
        sumSquares += v * v;
    }
    return (float) sqrt(sumSquares);
}

// Scale column j in matrix M (n rows, leading dimension ld) by factor alpha.
static void col_scale_ld(float *M, int n, int ld, int j, float alpha) {
    for (int i = 0; i < n; i++) {
        M[(size_t)i * ld + j] *= alpha;
    }
}

/* Matrix‐multiplication:
   A (m×n), V (n×b), Y (m×b) → Y = A * V */

/*VERY IMPORTANT, sometimes using noraml int type for i will lead to negative 
index or overflow of the data type SO THIS WHY SIZE_T is used*/

#define BLOCK 64 // 64 is usually used as block value.
//BLOCKED MULTIPLICATION IS MUCH FASTER (without using other libraries) ACCORDING TO OTHER SOURCES FOR LARGE MATRICES.
void matmul_A_V_blocked(const float *A, const float *V, float *Y,int m, int n, int b)
{
    // Initialize Y to zero
    for (int i = 0; i < m*b; i++) Y[i] = 0.0f;

    for (int ii = 0; ii < m; ii += BLOCK)
        for (int kk = 0; kk < n; kk += BLOCK)
            for (int jj = 0; jj < b; jj += BLOCK)
                for (int i = ii; i < ii + BLOCK && i < m; i++)
                    for (int k = kk; k < kk + BLOCK && k < n; k++) {
                        float aVal = A[i*n + k];
                        const float *Vk = V + k*b;
                        float *Yi = Y + i*b;
                        for (int j = jj; j < jj + BLOCK && j < b; j++) {
                            Yi[j] += aVal * Vk[j];
                        }
                    }
}

/* Matrix‐multiplication:
   A^T (n×m), Y (m×b), Z (n×b) → Z = A^T * Y
*/

void matmul_AT_Y_blocked(const float *A, const float *Y, float *Z, int m, int n, int b)
{
    // Initialize Z to zero
    for (int i = 0; i < n*b; i++) Z[i] = 0.0f;

    for (int jj = 0; jj < n; jj += BLOCK)          // block over n
        for (int ii = 0; ii < m; ii += BLOCK)      // block over m
            for (int tt = 0; tt < b; tt += BLOCK)  // block over b
                for (int j = jj; j < jj + BLOCK && j < n; j++)
                    for (int i = ii; i < ii + BLOCK && i < m; i++) {

                        float a = A[i*n + j];   // A[i,j]

                        const float *Yi = Y + i*b;
                        float *Zj = Z + j*b;

                        for (int t = tt; t < tt + BLOCK && t < b; t++) {
                            Zj[t] += a * Yi[t];
                        }
                    }
}


/* Block‐view of a matrix: allows specifying rows, cols, and a leading dimension ld.
   This supports non-contiguous column storage (e.g., when working with sub‐blocks).
*/
//Using a struct saves so much time
typedef struct {
    float *data;
    int rows;
    int cols;
    int ld;
} Block;

static float getB(Block B, int i, int j) {
    return B.data[(size_t)i * B.ld + j];
}
static void setB(Block B, int i, int j, float v) {
    B.data[(size_t)i * B.ld + j] = v;
}

/* Modified Gram–Schmidt QR factorization on block Z (n × b, leading dimension Z.ld).
   If R is non‐NULL then it’s the b×b upper‐triangular factor.
*/
void qr_for_Block(Block Z, float *R /* b×b or NULL */) {
    int n = Z.rows, b = Z.cols, ld = Z.ld;
    for (int j = 0; j < b; j++) {
        // subtract projections onto all previous q_k (columns k < j)
        for (int k = 0; k < j; k++) {
            double rkj = 0.0;
            for (int i = 0; i < n; i++) {
                rkj += (double)getB(Z, i, k) * getB(Z, i, j);
            }
            if (R) {
                R[k * b + j] = (float)rkj;
            }
            float alpha = (float)rkj;
            for (int i = 0; i < n; i++) {
                setB(Z, i, j, getB(Z, i, j) - alpha * getB(Z, i, k));
            }
        }
        // compute norm of column j
        double rjj = 0.0;
        for (int i = 0; i < n; i++) {
            double v = getB(Z, i, j);
            rjj += v * v;
        }
        rjj = sqrt(rjj);
        if (R) {
            R[j * b + j] = (float)rjj;
        }
        float invNorm = (rjj == 0.0) ? 0.0f : (float)(1.0 / rjj);
        // normalize column j
        for (int i = 0; i < n; i++) {
            setB(Z, i, j, getB(Z, i, j) * invNorm);
        }
    }
}

// Orthogonalize V (n×b, ldV) **against** Prev (n×p, ldP) using MGS.
void orth_against_prev(Block V, Block Prev) {
    if (Prev.cols <= 0) {
        return;
    }
    int n = V.rows, b = V.cols, ldV = V.ld, ldP = Prev.ld;
    for (int j = 0; j < b; j++) {
        // for each column j of V, subtract projections onto each column k of Prev
        for (int k = 0; k < Prev.cols; k++) {
            double dot = 0.0;
            for (int i = 0; i < n; i++) {
                dot += (double)Prev.data[(size_t)i * ldP + k] * V.data[(size_t)i * ldV + j];
            }
            float alpha = (float)dot;
            for (int i = 0; i < n; i++) {
                V.data[(size_t)i * ldV + j] -= alpha * Prev.data[(size_t)i * ldP + k];
            }
        }
        // normalize the adjusted column j of V
        float norm = col_norm_ld(V.data, n, ldV, j);
        float inv = (norm == 0.0f) ? 0.0f : 1.0f / norm;
        col_scale_ld(V.data, n, ldV, j, inv);
    }
}

/* ---------- Block‐power + QR update step for a rank‐b approximation ---------- */
void svd_block_step(float *A_res, int m, int n, int b,
                    float *A_app,        // m×n (approximation)
                    float *workV,        // n×b (ld = b)
                    float *workY,        // m×b (ld = b)
                    float *workZ,        // n×b (ld = b)
                    int max_iter, float tol,
                    const float *PrevV, int prev_cols, int prev_ld)
{
    Block V   = { workV,   n,     b,     b };
    Block Z   = { workZ,   n,     b,     b };
    Block Y   = { workY,   m,     b,     b };
    Block Prev = { (float*)PrevV, n, prev_cols, prev_ld };

    // Initialize V with small “random-ish” values
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < b; j++) {
            setB(V, i, j, 1.0f + 1e-3f * (float)(j + (i % 7)));
        }
    }
    // Orthogonalize V against previous subspace Prev
    orth_against_prev(V, Prev);

    // allocate space to store old V for convergence check
    float *Vold = (float*) malloc((size_t)n * b * sizeof(float));
    for (int it = 0; it < max_iter; it++) {
        // copy current V into Vold
        for (int i = 0; i < n * b; i++) {
            Vold[i] = V.data[i];
        }

        // Y = A_res * V
        matmul_A_V_blocked(A_res, V.data, Y.data, m, n, b);
        // Z = A_res^T * Y
        matmul_AT_Y_blocked(A_res, Y.data, Z.data, m, n, b);
        // V = orthonormalize Z
        qr_for_Block(Z, NULL);
        for (int i = 0; i < n * b; i++) {
            V.data[i] = Z.data[i];
        }

        // compute change = || V – Vold ||
        double diffSum = 0.0;
        for (int i = 0; i < n * b; i++) {
            double d = (double)V.data[i] - (double)Vold[i];
            diffSum += d * d;
        }
        if ((float) sqrt(diffSum) < tol) {
            break;
        }
    }
    free(Vold);

    // Now update the low‐rank approximation A_app and residual A_res
    for (int j = 0; j < b; j++) {
        // Compute y_j = A_res * v_j
        for (int i = 0; i < m; i++) {
            double sum = 0.0;
            const float *rowA = A_res + (size_t)i * n;
            for (int k = 0; k < n; k++) {
                sum += (double)rowA[k] * V.data[(size_t)k * b + j];
            }
            Y.data[(size_t)i * b + j] = (float)sum;
        }

        float sigma = col_norm_ld(Y.data, m, b, j);
        if (sigma == 0.0f) {
            continue;
        }

        // A_app += sigma * (u * v_j^T), A_res -= sigma * (u * v_j^T)
        for (int i = 0; i < m; i++) {
            float ui = Y.data[(size_t)i * b + j] / sigma;
            float sig_ui = sigma * ui;
            float *rowApp = A_app + (size_t)i * n;
            float *rowRes = A_res + (size_t)i * n;
            for (int k = 0; k < n; k++) {
                float vkj = V.data[(size_t)k * b + j];
                float increment = sig_ui * vkj;
                rowApp[k] += increment;
                rowRes[k] -= increment;
            }
        }
    }
}

/* ---------- I/O helpers ---------- */

// Read `count` floats from binary file at `path`.
float* read_bin(const char *path, size_t count) {
    FILE *f = fopen(path, "rb");
    float *buf = (float*) malloc(count * sizeof(float));
    size_t got = fread(buf, sizeof(float), count, f);
    fclose(f);
    return buf;
}

// Write `count` floats from `buf` into binary file at `path`.
void write_bin(const char *path, const float *buf, size_t count) {
    FILE *f = fopen(path, "wb");
    size_t written = fwrite(buf, sizeof(float), count, f);
    fclose(f);
}

/* ---------- main entry: block-power + QR for rank-k approx ---------- */
int main(void) {
    // Read shapes from "shape.txt" (three channels).
    FILE *shape = fopen("../hybrid_c_python/data/shape.txt", "r");
    int dim[2][3];
    for (int i = 0; i < 3; i++) {
        fscanf(shape, "%d %d", &dim[0][i], &dim[1][i]);
    }
    fclose(shape);

    // Read k (target rank) from "k.txt"
    int k = 0;
    FILE *kf = fopen("../hybrid_c_python/data/k.txt", "r");
    fscanf(kf, "%d", &k);
    fclose(kf);

    //I dont want to repeat the same code for the three channels, when i can just write a for loop.
    const char *in_file[3]  = {"../hybrid_c_python/data/a.bin",   "../hybrid_c_python/data/b.bin",   "../hybrid_c_python/data/c.bin"};
    const char *out_file[3] = {"../hybrid_c_python/data/a_out.bin", "../hybrid_c_python/data/a_out.bin", "../hybrid_c_python/data/a_out.bin"};

    for (int ch = 0; ch < 3; ch++) {
        int m = dim[0][ch];
        int n = dim[1][ch];
        int r = (m < n ? m : n);
        int kk = k; //I don't want to change k.
        if (kk > r) {
            kk = r;
        }

        size_t count = (size_t)m * n;
        float *elements   = read_bin(in_file[ch], count);

        float *A_res = (float*) malloc(count * sizeof(float));
        float *A_app = (float*) calloc(count, sizeof(float));
        for (size_t i = 0; i < count; i++) {
            A_res[i] = elements[i];
        }

        // Workspace for block algorithm
        const int b_default = 16;
        int bmax = (kk < b_default ? kk : b_default);
        if (bmax < 1) bmax = 1;

        float *V = (float*) malloc((size_t)n * bmax * sizeof(float)); // n×b
        float *Y = (float*) malloc((size_t)m * bmax * sizeof(float)); // m×b
        float *Z = (float*) malloc((size_t)n * bmax * sizeof(float)); // n×b

        // Store previous V columns to orthogonalize against
        int   pld       = bmax;
        float *PrevV    = (float*) calloc((size_t)n * pld, sizeof(float));
        int   prev_cols = 0;

        int remaining = kk;
        while (remaining > 0) {
            int curr_b = (remaining < bmax ? remaining : bmax);
            int   max_iter = 80;
            float tol      = 1e-4f;

            svd_block_step(A_res, m, n, curr_b, A_app, V, Y, Z, max_iter, tol,
                           PrevV, prev_cols, pld);

            // Append current V (first curr_b columns) into PrevV; keep up to pld columns
            if (prev_cols + curr_b > pld) {
                int overflow = prev_cols + curr_b - pld;
                if (overflow < prev_cols) {
                    // shift existing columns left
                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j < prev_cols - overflow; j++) {
                            PrevV[(size_t)i * pld + j] = PrevV[(size_t)i * pld + (j + overflow)];
                        }
                    }
                    prev_cols -= overflow;
                } else {
                    prev_cols = 0;
                }
            }
            for (int j = 0; j < curr_b; j++) {
                for (int i = 0; i < n; i++) {
                    PrevV[(size_t)i * pld + (prev_cols + j)] = V[(size_t)i * curr_b + j];
                }
            }
            prev_cols += curr_b;
            remaining -= curr_b;
        }
        clamp_0_255(A_app, count);
        write_bin(out_file[ch], A_app, count);
// obviously need to free all of the allocated meory.
        free(PrevV);
        free(V);
        free(Y);
        free(Z);
        free(A_res);
        free(A_app);
        free(elements);
    }

    printf("Ready for reconstruction.\n");
    return 0;
}

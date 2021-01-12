#include <iostream>
#include <chrono>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <immintrin.h>
#include <cassert>
#include "mkl.h"

using namespace std;
const int ALIGN = 32;
const int BATCH_SIZE = 256*3;

template<int BatchSize, int InputSize, int OutputSize, int Lda=InputSize, int Ldc=OutputSize>
class PW {
  public:
    PW() {
        if (jitter == NULL) {
            mkl_cblas_jit_create_sgemm(&jitter, MKL_ROW_MAJOR, MKL_NOTRANS, MKL_NOTRANS,
                BatchSize, OutputSize, InputSize, 1.0, Lda, OutputSize, 1.0, Ldc);

            sgemm_kernel = mkl_jit_get_sgemm_ptr(jitter);
        }

        weight = (float*) aligned_alloc(ALIGN, InputSize*OutputSize*sizeof(float)); 
        for (int i = 0; i < InputSize; i++) {
            for (int j = 0; j < OutputSize; j++) {
                weight[i*OutputSize+j] = (rand()%4) - 2;
            }
        }
    }

    void run(float *inp, float *out) {
        sgemm_kernel(jitter, inp, weight, out);
    }


    float *weight;
    static void *jitter;
    static sgemm_jit_kernel_t sgemm_kernel;
};

template <int BatchSize, int InputSize, int OutputSize, int Lda, int Ldc>
void* PW<BatchSize, InputSize, OutputSize, Lda, Ldc>::jitter = NULL;

template <int BatchSize, int InputSize, int OutputSize, int Lda, int Ldc>
sgemm_jit_kernel_t PW<BatchSize, InputSize, OutputSize, Lda, Ldc>::sgemm_kernel = 0;


#define cast_uint32_t static_cast<uint32_t>

static inline float
fasterpow2 (float p)
{
//  float clipp = (p < -126) ? -126.0f : p;
  float clipp = p;
  union { uint32_t i; float f; } v = { cast_uint32_t ( (1 << 23) * (clipp + 126.94269504f) ) };
  return v.f;
}

static inline float
fasterexp (float p)
{
  return fasterpow2 (1.442695040f * p);
}

static inline float turboexp(float p) {
  union { uint32_t i; float f; } v = { cast_uint32_t    (12102203.15410432f * p +
          1064872507.15410442f) };
  return v.f;
}

static inline float
fasterswish (float x)
{
  return x / (1.0f + turboexp (-x));
}

void dw11(float *data, float *mat, float *out, int m, int c) {
    assert(rf == 11);
//    memset(out, 0, m*c*sizeof(float));
    int BS = 128;
    for (int ib = 0; ib < m; ib += BS) {
        for (int k = 0; k < c; k+=8) {
            __m256 rs[] = {
                _mm256_load_ps(mat + 0*c+k),
                _mm256_load_ps(mat + 1*c+k),
                _mm256_load_ps(mat + 2*c+k),
                _mm256_load_ps(mat + 3*c+k),
                _mm256_load_ps(mat + 4*c+k),
                _mm256_load_ps(mat + 5*c+k),
                _mm256_load_ps(mat + 6*c+k),
                _mm256_load_ps(mat + 7*c+k),
                _mm256_load_ps(mat + 8*c+k),
                _mm256_load_ps(mat + 9*c+k),
                _mm256_load_ps(mat + 10*c+k)};
            for (int i = ib; i < min(ib+BS, m); i++) {
                __m256 v = _mm256_set1_ps(0);
                for (int j = 0; j < 11; j++) {
                    int i2 = i - 5 + j;
                    v = _mm256_fmadd_ps(_mm256_load_ps(data + (i2*c+k)),
                                        rs[j],
                                        v);
                }
                _mm256_store_ps(out + (i*c+k), v);
            }
        }
    }
}

const int RECEPTIVE_FIELD = 11;
template<int BatchSize, int Channels>
class Block {
  public:
    float *residualb,
          *dw[5], *pwb[5];
    

    PW<BatchSize/3, Channels, 2*Channels> in_pw;
    PW<BatchSize/3, 2*Channels, 2*Channels> pw[3];
    PW<BatchSize/3, Channels, Channels, 2*Channels, 3*Channels> expand[3];
    PW<BatchSize, Channels, Channels> residual;

    Block() {
        residualb = (float*) aligned_alloc(ALIGN, Channels*sizeof(float));
        for (int i = 0; i < 5; i++) {
            dw[i] = (float*) aligned_alloc(ALIGN, 2*Channels*RECEPTIVE_FIELD*sizeof(float)); 
            pwb[i] = (float*) aligned_alloc(ALIGN, 2*Channels*sizeof(float)); 
        }

        for (int i = 0; i < Channels; i++) {
            residualb[i] = (float) (rand() % 4) - 2;
        }
        for (int i = 0; i < 2*Channels; i++) {
            for (int k = 0; k < 5; k++) {
                pwb[k][i] = (float) (rand() % 4) - 2;
            }
            for (int j = 0; j < RECEPTIVE_FIELD; j++) {
                for (int k = 0; k < 5; k++) {
                    dw[k][i*RECEPTIVE_FIELD+j] = (float) (rand() % 4) - 2;
                }
            }        
        }
    }

    // Assumes, that all buffers are padded on both sides with zeros
    // (since we are lazy to deal with special cases around borders)
    void calc(float *input, float *output, float* buf2) {
        for (int i = 0; i < BatchSize; i++) {
            memcpy(output + i*Channels, residualb, Channels*sizeof(float));
        }
        residual.run(input, output);

        //Pool
        // Zero is special case
        for (int i = 0; i < Channels; i++) {
            input[i] += input[Channels+i];
            input[i] += input[Channels*2+i];
        }

        for (int i = 1; i < BatchSize / 3; i++) {
            memset(input + i*Channels, 0, Channels*sizeof(float));
            for (int j = 0; j < Channels; j++) {
                input[i*Channels+j] += input[(i*3)*Channels+j];
                input[i*Channels+j] += input[(i*3+1)*Channels+j];
                input[i*Channels+j] += input[(i*3+2)*Channels+j];
            }
        }
        memset(input + (BatchSize/3)*Channels, 0, Channels*11*sizeof(float));
        in_pw.run(input, buf2);
        // TODO: bias
        dw11(buf2, dw[0], input, BatchSize/3, 2*Channels);
        for (int i = 0; i < BatchSize/3 * 2 * Channels; i++) {
            input[i] = fasterswish(input[i]);
        }

        for (int k = 1; k < 4; k++) {
            dw11(input, dw[k], buf2, BatchSize / 3, Channels*2);
            for (int i = 0; i < BatchSize/3; i++) {
                memcpy(input + i*2*Channels, pwb[k], 2*Channels*sizeof(float));
            }
            pw[k-1].run(buf2, input);
            for (int i = 0; i < BatchSize/3 * 2 * Channels; i++) {
                input[i] = fasterswish(input[i]);
            }
        }
        dw11(input, dw[4], buf2, BatchSize / 3, Channels * 2);
        expand[0].run(buf2, output);
        expand[1].run(buf2 + 40, output + 80);
        expand[2].run(buf2 + 80, output + 160);
/*        pw[4].run(buf2, output);*/
        for (int i = 0; i < BatchSize * Channels; i++) {
            output[i] = fasterswish(output[i]);
        }
    }
};

template<int BatchSize, int Channels>
class BlockC {
  public:
    float *dw, *pwb;
    PW<BatchSize, Channels, Channels> pw;

    BlockC() {
        dw = (float*) aligned_alloc(ALIGN, Channels*RECEPTIVE_FIELD*sizeof(float)); 
        
        pwb = (float*) aligned_alloc(ALIGN, Channels*sizeof(float)); 

        for (int i = 0; i < Channels; i++) {
            pwb[i] = (float) (rand() % 4) - 2;
            for (int j = 0; j < RECEPTIVE_FIELD; j++) {
                dw[i*RECEPTIVE_FIELD+j] = (float) (rand() % 4) - 2;
            }        
        }
    }

    // Assumes, that all buffers are padded on both sides with zeros
    // (since we are lazy to deal with special cases around borders)
    void calc(float *input, float *output, float* buf2) {
        dw11(input, dw, buf2, BatchSize, Channels);
        for (int i = 0; i < BatchSize; i++) {
            memcpy(output + i*Channels, pwb, Channels*sizeof(float));
        }
        pw.run(buf2, output);
        for (int i = 0; i < BatchSize * Channels; i++) {
            output[i] = fasterswish(output[i]);
        }
    }
};

template<int BatchSize, int Channels>
class BlockC2 {
  public:
    PW<BatchSize,Channels, Channels/2> pw[7];
    float *pwb;


    BlockC2() {
        pwb = (float*) aligned_alloc(ALIGN, Channels*sizeof(float)); 

        for (int i = 0; i < Channels; i++) {
            pwb[i] = (float) (rand() % 4) - 2;
        }
    }

    // Assumes, that all buffers are padded on both sides with zeros
    // (since we are lazy to deal with special cases around borders)
    void calc(float *input, float *output, float* buf2) {
        for (int i = 0; i < BatchSize; i++) {
            memcpy(output + i*Channels/2, pwb, Channels/2*sizeof(float));
        }
        for (int k = 0; k < 7; k++) {
            pw[k].run(input + (k-3)*Channels, output);
        }
        for (int i = 0; i < BatchSize * Channels; i++) {
            output[i] = fasterswish(output[i]);
        }
    }
};
// inpx (seq_size * 3, 1)
// inp_im2col (seq_size_out, 80)
// w (9, 80)
// b (80)
void im2col(float *inpx, float* inp_im2col, int seq_size_out) {
    int rf = 9;
    for (int i = 0; i < seq_size_out; i++) {
        int base_point = i*3;
        for (int j = 0; j < rf; j++) {
            int i2 = base_point - rf/2 + j;
            if (i2 < 0 || i2 >= seq_size_out*3) {

            } else {
                inp_im2col[i*9+j] = inpx[i2];
            }

        }
    }
}

int main() {
    int pad = 16;
    int channels = 80;

    PW<BATCH_SIZE, 9, 80> pw_in;
    float* ib = (float*) aligned_alloc(ALIGN, channels*sizeof(float));
    PW<BATCH_SIZE, 40, 5> pw_out;
    for (int i = 0; i < channels; i++) {
        ib[i] = (float) (rand() % 4) - 2;
    }

    Block<BATCH_SIZE, 80> block1;
    Block<BATCH_SIZE, 80> block2;
    Block<BATCH_SIZE, 80> block3;
    Block<BATCH_SIZE, 80> block4;
    Block<BATCH_SIZE, 80> block5;
    BlockC<BATCH_SIZE, 80> blockc;
    BlockC2<BATCH_SIZE, 80> blockc2;

    float *inpx = (float*) aligned_alloc(ALIGN, (3*BATCH_SIZE)*1*sizeof(float));
    float *inp_im2col = (float*) aligned_alloc(ALIGN, (BATCH_SIZE)*9*sizeof(float));

    float *inp = (float*) aligned_alloc(ALIGN, (BATCH_SIZE+2*pad)*80*sizeof(float));
    float *out = (float*) aligned_alloc(ALIGN, (BATCH_SIZE+2*pad)*80*sizeof(float));
    float *buf2 = (float*) aligned_alloc(ALIGN, (BATCH_SIZE+2*pad)*80*sizeof(float));
    float *bufd2s = (float*) aligned_alloc(ALIGN, (BATCH_SIZE/3+2*pad)*160*sizeof(float));
    printf("%llx %llx %llx\n", inp, out, buf2);
    
    int reps = 100;
    for (int j = 0; j < 50; j++) {
        memset(inpx, 0, (3*BATCH_SIZE)*1*sizeof(float));
        memset(inp_im2col, 0, (BATCH_SIZE)*9*sizeof(float));
        memset(inp, 0, (BATCH_SIZE+2*pad)*80*sizeof(float));
        memset(out, 0, (BATCH_SIZE+2*pad)*80*sizeof(float));
        memset(buf2, 0, (BATCH_SIZE+2*pad)*80*sizeof(float));
        memset(bufd2s, 0, (BATCH_SIZE/3+2*pad)*160*sizeof(float));
        auto start = std::chrono::system_clock::now();
        for (int i = 0; i < reps; i++) {
            im2col(inpx, inp_im2col, BATCH_SIZE);
            for (int j = 0; j < BATCH_SIZE; j++) {
                memcpy(inp + (j+pad)*channels, ib, channels*sizeof(float));
            }
//            sgemm_kernel_in(jitter_in, inp_im2col, iw, inp + pad*channels);
            pw_in.run(inp_im2col, inp + pad*channels);

            block1.calc(inp + pad*80, out + pad*80, bufd2s + pad*160);
            block2.calc(out + pad*80, inp + pad*80, bufd2s + pad*160);
            block3.calc(inp + pad*80, out + pad*80, bufd2s + pad*160);
            block4.calc(out + pad*80, inp + pad*80, bufd2s + pad*160);
            block5.calc(inp + pad*80, out + pad*80, bufd2s + pad*160);
            blockc.calc(out + pad*80, inp + pad*80, buf2 + pad*80);
            blockc2.calc(inp + pad*80, out + pad*80, buf2 + pad*80);
            pw_out.run(out + pad*channels, inp + pad*channels);
        }
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        cout << "tx " << elapsed_seconds.count() << " " << reps * BATCH_SIZE * 3 / elapsed_seconds.count() << endl;
    }
}

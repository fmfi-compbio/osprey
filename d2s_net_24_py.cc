#include <iostream>
#include <chrono>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <immintrin.h>
#include <cassert>
#include <cmath>
#include "mkl.h"
#include "loading.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace std;
const int ALIGN = 32;
const int BATCH_SIZE = 512*3;
const int CHANNELS = 96;
const int PAD = 16;
const int STEPPING_PAD = 28;

// mat: [in, out]
template<int BatchSize, int InputSize, int OutputSize, int Lda=InputSize, int Ldc=OutputSize>
class PW {
  public:
    PW(const vector<float>& data) {
        if (jitter == NULL) {
            mkl_cblas_jit_create_sgemm(&jitter, MKL_ROW_MAJOR, MKL_NOTRANS, MKL_NOTRANS,
                BatchSize, OutputSize, InputSize, 1.0, Lda, OutputSize, 1.0, Ldc);

            sgemm_kernel = mkl_jit_get_sgemm_ptr(jitter);
        }

        weight = (float*) aligned_alloc(ALIGN, InputSize*OutputSize*sizeof(float)); 
        for (int i = 0; i < InputSize; i++) {
            for (int j = 0; j < OutputSize; j++) {
                weight[i*OutputSize+j] = data[i*OutputSize+j];
            }
        }
    }

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

static inline void fasterswisharr(float *f, int l) {
    for (int i = 0; i < l; i++) {
        f[i] = f[i] / (1 + expf(-f[i]));
    }
    return;
    __m256 c1 = _mm256_set1_ps(-12102203.15410432f);
    __m256 c2 = _mm256_set1_ps(1064872507.15410442f);
    __m256 one = _mm256_set1_ps(1.0f);

    for (int i = 0; i < l; i+=8) {
        __m256 v = _mm256_load_ps(f + i);
        __m256 r = _mm256_mul_ps(v, c1);
        __m256 r2 = _mm256_add_ps(r, c2);
        union { __m256i i; __m256 f; } r3 = { _mm256_cvtps_epi32(r2) }; 
        __m256 r4 = _mm256_add_ps(r3.f, one);
//        __m256 r5 = _mm256_rcp_ps(r4);
//        __m256 r6 = _mm256_mul_ps(v, r5);
        __m256 r6 = _mm256_div_ps(v, r4);
       _mm256_store_ps(f + i, r6);
    }
}

class DW11 {
  public:
    int channels_;
    float *weight;
    DW11(int channels, map<string, vector<float>>&data, string prefix) : channels_(channels) {
        weight = (float*) aligned_alloc(ALIGN, channels*11*sizeof(float));
        for (int i = 0; i < channels*11; i++) {
            weight[i] = data[prefix][i];
        }
    }

    // mat: [pos, channels]
    void run(float* data, float*out, int m) {
        int BS = 128;
        for (int ib = 0; ib < m; ib += BS) {
            for (int k = 0; k < channels_; k+=8) {
                __m256 rs[] = {
                    _mm256_load_ps(weight + 0*channels_+k),
                    _mm256_load_ps(weight + 1*channels_+k),
                    _mm256_load_ps(weight + 2*channels_+k),
                    _mm256_load_ps(weight + 3*channels_+k),
                    _mm256_load_ps(weight + 4*channels_+k),
                    _mm256_load_ps(weight + 5*channels_+k),
                    _mm256_load_ps(weight + 6*channels_+k),
                    _mm256_load_ps(weight + 7*channels_+k),
                    _mm256_load_ps(weight + 8*channels_+k),
                    _mm256_load_ps(weight + 9*channels_+k),
                    _mm256_load_ps(weight + 10*channels_+k)};
                for (int i = ib; i < min(ib+BS, m); i++) {
                    __m256 v = _mm256_set1_ps(0);
                    for (int j = 0; j < 11; j++) {
                        int i2 = i - 5 + j;
                        v = _mm256_fmadd_ps(_mm256_load_ps(data + (i2*channels_+k)),
                                            rs[j],
                                            v);
                    }
                    _mm256_store_ps(out + (i*channels_+k), v);
                }
            }
        }
    }
};


const int RECEPTIVE_FIELD = 11;
template<int BatchSize, int Channels>
class Block4 {
  public:
    float *residualb,
          *pwb[4];
    

    PW<BatchSize/3, Channels, 2*Channels> in_pw;
    PW<BatchSize/3, 2*Channels, 2*Channels> pw[2];
    PW<BatchSize, Channels, Channels> residual;
    PW<BatchSize/3, Channels, Channels, 2*Channels, 3*Channels> expand[3];
    DW11 dw[4];

    Block4(map<string, vector<float>>& data, string prefix) : in_pw(data[prefix+".conv.1.weight"]),
    pw { PW<BatchSize/3, 2*Channels, 2*Channels>(data[prefix+".conv.6.pointwise.weight"]),
         PW<BatchSize/3, 2*Channels, 2*Channels>(data[prefix+".conv.10.pointwise.weight"]),
    }, residual(data[prefix+".residual.0.conv.weight"]),
    expand {
        PW<BatchSize/3, Channels, Channels, 2*Channels, 3*Channels>(
                data[prefix+".conv.15.convs.0.weight"]),
        PW<BatchSize/3, Channels, Channels, 2*Channels, 3*Channels>(
                data[prefix+".conv.15.convs.1.weight"]),
        PW<BatchSize/3, Channels, Channels, 2*Channels, 3*Channels>(
                data[prefix+".conv.15.convs.2.weight"])
    }, dw {
        DW11(Channels*2, data, prefix+".conv.2.weight"),
        DW11(Channels*2, data, prefix+".conv.6.depthwise.weight"),
        DW11(Channels*2, data, prefix+".conv.10.depthwise.weight"),
        DW11(Channels*2, data, prefix+".conv.14.weight"),
    }

    {
        residualb = (float*) aligned_alloc(ALIGN, Channels*sizeof(float));
        for (int i = 0; i < 4; i++) {
            pwb[i] = (float*) aligned_alloc(ALIGN, 2*Channels*sizeof(float)); 
        }

        for (int i = 0; i < Channels; i++) {
            residualb[i] = data[prefix+".residual.0.conv.bias"][i];
            residualb[i] += data[prefix+".conv.15.convs.0.bias"][i];
        }

        for (int i = 0; i < 2*Channels; i++) {
            pwb[0][i] = data[prefix+".conv.2.bias"][i];
            pwb[1][i] = data[prefix+".conv.6.pointwise.bias"][i];
            pwb[2][i] = data[prefix+".conv.10.pointwise.bias"][i];
        }
    }

    // Assumes, that all buffers are padded on both sides with zeros
    // (since we are lazy to deal with special cases around borders)
    void calc(float *input, float *output, float* buf2) {
        for (int i = 0; i < BatchSize; i++) {
            memcpy(output + i*Channels, residualb, Channels*sizeof(float));
        }
        residual.run(input, output);

        // TODO: fuse /3 into next conv weight
        //Pool
        // Zero is special case
        for (int i = 0; i < Channels; i++) {
            input[i] /= 3;
            input[i] += input[Channels+i]/3;
            input[i] += input[Channels*2+i]/3;
        }

        for (int i = 1; i < BatchSize / 3; i++) {
            memset(input + i*Channels, 0, Channels*sizeof(float));
            for (int j = 0; j < Channels; j++) {
                input[i*Channels+j] += input[(i*3)*Channels+j]/3;
                input[i*Channels+j] += input[(i*3+1)*Channels+j]/3;
                input[i*Channels+j] += input[(i*3+2)*Channels+j]/3;
            }
        }

        memset(input + (BatchSize/3)*Channels, 0, (BatchSize/3+11)*Channels*sizeof(float));
        memset(buf2, 0, (BatchSize/3)*2*Channels*sizeof(float));
        in_pw.run(input, buf2);


        dw[0].run(buf2, input, BatchSize/3);
        for (int i = 0; i < BatchSize/3; i++) {
            for (int j = 0; j < 2*Channels; j++) {
                input[i*2*Channels+j] += pwb[0][j];
            }
        }

        fasterswisharr(input, BatchSize/3 * 2 * Channels);

        for (int k = 1; k < 3; k++) {
            dw[k].run(input, buf2, BatchSize / 3);
            for (int i = 0; i < BatchSize/3; i++) {
                memcpy(input + i*2*Channels, pwb[k], 2*Channels*sizeof(float));
            }
            pw[k-1].run(buf2, input);
            fasterswisharr(input, BatchSize/3 * 2 * Channels);

        }

        dw[3].run(input, buf2, BatchSize / 3);

        expand[0].run(buf2, output);
        expand[1].run(buf2 + Channels / 2, output + Channels);
        expand[2].run(buf2 + Channels, output + 2*Channels);
        fasterswisharr(output, BatchSize * Channels);
    }
};



template<int BatchSize, int Channels>
class Block5 {
  public:
    float *residualb,
          *pwb[5];
    

    PW<BatchSize/3, Channels, 2*Channels> in_pw;
    PW<BatchSize/3, 2*Channels, 2*Channels> pw[3];
    PW<BatchSize, Channels, Channels> residual;
    PW<BatchSize/3, Channels, Channels, 2*Channels, 3*Channels> expand[3];
    DW11 dw[5];

    Block5(map<string, vector<float>>& data, string prefix) : in_pw(data[prefix+".conv.1.weight"]),
    pw { PW<BatchSize/3, 2*Channels, 2*Channels>(data[prefix+".conv.6.pointwise.weight"]),
         PW<BatchSize/3, 2*Channels, 2*Channels>(data[prefix+".conv.10.pointwise.weight"]),
         PW<BatchSize/3, 2*Channels, 2*Channels>(data[prefix+".conv.14.pointwise.weight"]),
    }, residual(data[prefix+".residual.0.conv.weight"]),
    expand {
        PW<BatchSize/3, Channels, Channels, 2*Channels, 3*Channels>(
                data[prefix+".conv.19.convs.0.weight"]),
        PW<BatchSize/3, Channels, Channels, 2*Channels, 3*Channels>(
                data[prefix+".conv.19.convs.1.weight"]),
        PW<BatchSize/3, Channels, Channels, 2*Channels, 3*Channels>(
                data[prefix+".conv.19.convs.2.weight"])
    }, dw {
        DW11(Channels*2, data, prefix+".conv.2.weight"),
        DW11(Channels*2, data, prefix+".conv.6.depthwise.weight"),
        DW11(Channels*2, data, prefix+".conv.10.depthwise.weight"),
        DW11(Channels*2, data, prefix+".conv.14.depthwise.weight"),
        DW11(Channels*2, data, prefix+".conv.18.weight"),
    } {
        residualb = (float*) aligned_alloc(ALIGN, Channels*sizeof(float));
        for (int i = 0; i < 5; i++) {
            pwb[i] = (float*) aligned_alloc(ALIGN, 2*Channels*sizeof(float)); 
        }

        for (int i = 0; i < Channels; i++) {
            residualb[i] = data[prefix+".residual.0.conv.bias"][i];
            residualb[i] += data[prefix+".conv.19.convs.0.bias"][i];
        }

        for (int i = 0; i < 2*Channels; i++) {
            pwb[0][i] = data[prefix+".conv.2.bias"][i];
            pwb[1][i] = data[prefix+".conv.6.pointwise.bias"][i];
            pwb[2][i] = data[prefix+".conv.10.pointwise.bias"][i];
            pwb[3][i] = data[prefix+".conv.14.pointwise.bias"][i];
        }
    }

    // Assumes, that all buffers are padded on both sides with zeros
    // (since we are lazy to deal with special cases around borders)
    void calc(float *input, float *output, float* buf2) {
        for (int i = 0; i < BatchSize; i++) {
            memcpy(output + i*Channels, residualb, Channels*sizeof(float));
        }
        residual.run(input, output);

        // TODO: fuse /3 into next conv weight
        //Pool
        // Zero is special case
        for (int i = 0; i < Channels; i++) {
            input[i] /= 3;
            input[i] += input[Channels+i]/3;
            input[i] += input[Channels*2+i]/3;
        }

        for (int i = 1; i < BatchSize / 3; i++) {
            memset(input + i*Channels, 0, Channels*sizeof(float));
            for (int j = 0; j < Channels; j++) {
                input[i*Channels+j] += input[(i*3)*Channels+j]/3;
                input[i*Channels+j] += input[(i*3+1)*Channels+j]/3;
                input[i*Channels+j] += input[(i*3+2)*Channels+j]/3;
            }
        }

        memset(input + (BatchSize/3)*Channels, 0, (BatchSize/3+11)*Channels*sizeof(float));
        memset(buf2, 0, (BatchSize/3)*2*Channels*sizeof(float));
        in_pw.run(input, buf2);

        dw[0].run(buf2, input, BatchSize/3);
        for (int i = 0; i < BatchSize/3; i++) {
            for (int j = 0; j < 2*Channels; j++) {
                input[i*2*Channels+j] += pwb[0][j];
            }
        }

        fasterswisharr(input, BatchSize/3 * 2 * Channels);

        for (int k = 1; k < 4; k++) {
            dw[k].run(input, buf2, BatchSize / 3);
            for (int i = 0; i < BatchSize/3; i++) {
                memcpy(input + i*2*Channels, pwb[k], 2*Channels*sizeof(float));
            }
            pw[k-1].run(buf2, input);
            fasterswisharr(input, BatchSize/3 * 2 * Channels);

        }

        dw[4].run(input, buf2, BatchSize / 3);

        expand[0].run(buf2, output);
        expand[1].run(buf2 + Channels/2, output + Channels);
        expand[2].run(buf2 + Channels, output + 2*Channels);
        fasterswisharr(output, BatchSize * Channels);
    }
};

template<int BatchSize, int Channels>
class Block6 {
  public:
    float *residualb,
          *pwb[6];
    

    PW<BatchSize/3, Channels, 2*Channels> in_pw;
    PW<BatchSize/3, 2*Channels, 2*Channels> pw[4];
    PW<BatchSize, Channels, Channels> residual;
    PW<BatchSize/3, Channels, Channels, 2*Channels, 3*Channels> expand[3];
    DW11 dw[6];

    Block6(map<string, vector<float>>& data, string prefix) : in_pw(data[prefix+".conv.1.weight"]),
    pw { PW<BatchSize/3, 2*Channels, 2*Channels>(data[prefix+".conv.6.pointwise.weight"]),
         PW<BatchSize/3, 2*Channels, 2*Channels>(data[prefix+".conv.10.pointwise.weight"]),
         PW<BatchSize/3, 2*Channels, 2*Channels>(data[prefix+".conv.14.pointwise.weight"]),
         PW<BatchSize/3, 2*Channels, 2*Channels>(data[prefix+".conv.18.pointwise.weight"]),
    }, residual(data[prefix+".residual.0.conv.weight"]),
    expand {
        PW<BatchSize/3, Channels, Channels, 2*Channels, 3*Channels>(
                data[prefix+".conv.23.convs.0.weight"]),
        PW<BatchSize/3, Channels, Channels, 2*Channels, 3*Channels>(
                data[prefix+".conv.23.convs.1.weight"]),
        PW<BatchSize/3, Channels, Channels, 2*Channels, 3*Channels>(
                data[prefix+".conv.23.convs.2.weight"])
    }, dw {
        DW11(Channels*2, data, prefix+".conv.2.weight"),
        DW11(Channels*2, data, prefix+".conv.6.depthwise.weight"),
        DW11(Channels*2, data, prefix+".conv.10.depthwise.weight"),
        DW11(Channels*2, data, prefix+".conv.14.depthwise.weight"),
        DW11(Channels*2, data, prefix+".conv.18.depthwise.weight"),
        DW11(Channels*2, data, prefix+".conv.22.weight"),
    }{
        residualb = (float*) aligned_alloc(ALIGN, Channels*sizeof(float));
        for (int i = 0; i < 6; i++) {
            pwb[i] = (float*) aligned_alloc(ALIGN, 2*Channels*sizeof(float)); 
        }

        for (int i = 0; i < Channels; i++) {
            residualb[i] = data[prefix+".residual.0.conv.bias"][i];
            residualb[i] += data[prefix+".conv.23.convs.0.bias"][i];
        }

        for (int i = 0; i < 2*Channels; i++) {
            pwb[0][i] = data[prefix+".conv.2.bias"][i];
            pwb[1][i] = data[prefix+".conv.6.pointwise.bias"][i];
            pwb[2][i] = data[prefix+".conv.10.pointwise.bias"][i];
            pwb[3][i] = data[prefix+".conv.14.pointwise.bias"][i];
            pwb[4][i] = data[prefix+".conv.18.pointwise.bias"][i];
        }
    }

    // Assumes, that all buffers are padded on both sides with zeros
    // (since we are lazy to deal with special cases around borders)
    void calc(float *input, float *output, float* buf2) {
        for (int i = 0; i < BatchSize; i++) {
            memcpy(output + i*Channels, residualb, Channels*sizeof(float));
        }
        residual.run(input, output);

        // TODO: fuse /3 into next conv weight
        //Pool
        // Zero is special case
        for (int i = 0; i < Channels; i++) {
            input[i] /= 3;
            input[i] += input[Channels+i]/3;
            input[i] += input[Channels*2+i]/3;
        }

        for (int i = 1; i < BatchSize / 3; i++) {
            memset(input + i*Channels, 0, Channels*sizeof(float));
            for (int j = 0; j < Channels; j++) {
                input[i*Channels+j] += input[(i*3)*Channels+j]/3;
                input[i*Channels+j] += input[(i*3+1)*Channels+j]/3;
                input[i*Channels+j] += input[(i*3+2)*Channels+j]/3;
            }
        }

        memset(input + (BatchSize/3)*Channels, 0, (BatchSize/3+11)*Channels*sizeof(float));
        memset(buf2, 0, (BatchSize/3)*2*Channels*sizeof(float));
        in_pw.run(input, buf2);


        dw[0].run(buf2, input, BatchSize/3);
        for (int i = 0; i < BatchSize/3; i++) {
            for (int j = 0; j < 2*Channels; j++) {
                input[i*2*Channels+j] += pwb[0][j];
            }
        }

        fasterswisharr(input, BatchSize/3 * 2 * Channels);

        for (int k = 1; k < 5; k++) {
            dw[k].run(input, buf2, BatchSize / 3);
            for (int i = 0; i < BatchSize/3; i++) {
                memcpy(input + i*2*Channels, pwb[k], 2*Channels*sizeof(float));
            }
            pw[k-1].run(buf2, input);
            fasterswisharr(input, BatchSize/3 * 2 * Channels);
        }

        dw[5].run(input, buf2, BatchSize / 3);

        expand[0].run(buf2, output);
        expand[1].run(buf2 + Channels/2, output + Channels);
        expand[2].run(buf2 + Channels, output + Channels*2);
        fasterswisharr(output, BatchSize * Channels);
    }
};


template<int BatchSize, int Channels>
class BlockC {
  public:
    float *pwb;
    PW<BatchSize, Channels, Channels> pw;
    DW11 dw;

    BlockC(map<string, vector<float>>& data, string prefix) :
        pw(data[prefix+".conv.0.pointwise.weight"]), dw(Channels, data, prefix+".conv.0.depthwise.weight") {
        
        pwb = (float*) aligned_alloc(ALIGN, Channels*sizeof(float)); 

        for (int i = 0; i < Channels; i++) {
            pwb[i] = data[prefix+".conv.0.pointwise.bias"][i];
        }
    }

    // Assumes, that all buffers are padded on both sides with zeros
    // (since we are lazy to deal with special cases around borders)
    void calc(float *input, float *output, float* buf2) {
        dw.run(input, buf2, BatchSize);
        for (int i = 0; i < BatchSize; i++) {
            memcpy(output + i*Channels, pwb, Channels*sizeof(float));
        }
        pw.run(buf2, output);
        fasterswisharr(output, BatchSize * Channels);
    }
};

template<int BatchSize, int Channels>
class BlockC2 {
  public:
    PW<BatchSize,Channels, Channels/2> pw[7];
    float *pwb;


    BlockC2(map<string, vector<float>>& data, string prefix) {
        pwb = (float*) aligned_alloc(ALIGN, Channels/2*sizeof(float)); 

        for (int i = 0; i < Channels/2; i++) {
            pwb[i] = data[prefix+".conv.0.conv.bias"][i];
        }
        for (int i = 0; i < 7; i++) {
            for (int j = 0; j < Channels*Channels/2; j++) {
                pw[i].weight[j] = data[prefix+".conv.0.conv.weight"][j+i*Channels*Channels/2];

            }
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
        fasterswisharr(output, BatchSize * Channels);
    }
};

template<int BatchSize, int Channels>
class BlockS {
  public:
    PW<BatchSize/2, Channels, Channels*2, Channels*2, Channels*2> pw;
    float *pwb;


    BlockS(map<string, vector<float>>& data, string prefix) :
        pw(data[prefix+".conv.0.conv.weight"]) {
        pwb = (float*) aligned_alloc(ALIGN, Channels*2*sizeof(float)); 

        for (int i = 0; i < Channels*2; i++) {
            pwb[i] = data[prefix+".conv.0.conv.bias"][i];
        }
    }

    // Assumes, that all buffers are padded on both sides with zeros
    // (since we are lazy to deal with special cases around borders)
    void calc(float *input, float *output, float* buf2) {
        for (int i = 0; i < BatchSize/2; i++) {
            memcpy(output + i*Channels*2, pwb, Channels*2*sizeof(float));
        }
        pw.run(input, output);
        fasterswisharr(output, BatchSize * Channels);
    }
};

// inpx (seq_size * 3, 1)
// inp_im2col (seq_size_out, 80)
// w (9, 80)
// b (80)
void im2col(float *inpx, float* inp_im2col, int seq_size_out) {
    int rf = 9;
    for (int i = 0; i < seq_size_out; i++) {
        int base_point = i;
        for (int j = 0; j < rf; j++) {
            int i2 = base_point - rf/2 + j;
            if (i2 < 0 || i2 >= seq_size_out) {

            } else {
                inp_im2col[i*9+j] = inpx[i2];
            }

        }
    }
}


class Caller {
  public:
    map<string, vector<float>> weights;
    PW<BATCH_SIZE*4, 9, CHANNELS/4> pw_in;
    PW<BATCH_SIZE, CHANNELS/2, 5> pw_out;

    Block4<BATCH_SIZE*4, CHANNELS/4> block1;
    BlockS<BATCH_SIZE*4, CHANNELS/4> blocks1;
    Block5<BATCH_SIZE*2, CHANNELS/2> block2;
    BlockS<BATCH_SIZE*2, CHANNELS/2> blocks2;

    Block6<BATCH_SIZE, CHANNELS> blocks[3];

    BlockC<BATCH_SIZE, CHANNELS> blockc;
    BlockC2<BATCH_SIZE, CHANNELS> blockc2;
    float *ib, *ob, *inp_im2col, *inp, *out, *buf2, *bufd2s;

    Caller() : weights(load_weight("net24.txt")),
        pw_in(weights["e.encoder.0.conv.0.conv.weight"]),
        pw_out(weights["out.weight"]),
        block1(weights, "e.encoder.1"),
        blocks1(weights, "e.encoder.2"),
        block2(weights, "e.encoder.3"),
        blocks2(weights, "e.encoder.4"),
        blocks {
            Block6<BATCH_SIZE, CHANNELS>(weights, "e.encoder.5"),
            Block6<BATCH_SIZE, CHANNELS>(weights, "e.encoder.6"),
            Block6<BATCH_SIZE, CHANNELS>(weights, "e.encoder.7"),
        },
        blockc(weights, "e.encoder.8"),
        blockc2(weights, "e.encoder.9")
    {
        ib = (float*) aligned_alloc(ALIGN, CHANNELS/4*sizeof(float));
        ob = (float*) aligned_alloc(ALIGN, 5*sizeof(float));
        for (int i = 0; i < CHANNELS/4; i++) {
            ib[i] = weights["e.encoder.0.conv.0.conv.bias"][i];
        }
        for (int i = 0; i < 5; i++) {
            ob[i] = weights["out.bias"][i];
        }
        inp_im2col = (float*) aligned_alloc(ALIGN, (BATCH_SIZE)*9*4*sizeof(float));
        inp = (float*) aligned_alloc(ALIGN, (BATCH_SIZE+2*PAD)*CHANNELS*sizeof(float));
        out = (float*) aligned_alloc(ALIGN, (BATCH_SIZE+2*PAD)*CHANNELS*sizeof(float));
        buf2 = (float*) aligned_alloc(ALIGN, (BATCH_SIZE+2*PAD)*CHANNELS*sizeof(float));
        bufd2s = (float*) aligned_alloc(ALIGN, (BATCH_SIZE/3+2*PAD)*2*CHANNELS*sizeof(float));
        memset(inp_im2col, 0, (BATCH_SIZE)*9*4*sizeof(float));
        memset(inp, 0, (BATCH_SIZE+2*PAD)*CHANNELS*sizeof(float));
        memset(out, 0, (BATCH_SIZE+2*PAD)*CHANNELS*sizeof(float));
        memset(buf2, 0, (BATCH_SIZE+2*PAD)*CHANNELS*sizeof(float));
        memset(bufd2s, 0, (BATCH_SIZE/3+2*PAD)*2*CHANNELS*sizeof(float)); 
    }

    float* call_chunk(float* inpx) {
/*        for (int i = 0; i < BATCH_SIZE*4; i++)  {
            if (isnan(inpx[i])) {
                printf("wat inp %d\n", i);
            }
        }*/
        im2col(inpx, inp_im2col, BATCH_SIZE*4);
        for (int j = 0; j < BATCH_SIZE*4; j++) {
            memcpy(inp + (j+PAD*4)*CHANNELS/4, ib, CHANNELS/4*sizeof(float));
        }
/*        for (int i = 0; i < BATCH_SIZE*CHANNELS; i++) {
            if (isnan(inp_im2col[i])) {
                printf("wat %d\n", i);
            }
        }*/
        pw_in.run(inp_im2col, inp + PAD*CHANNELS);

/*        for (int i = 0; i < BATCH_SIZE*CHANNELS; i++) {
            inp[PAD*CHANNELS+i] = fasterswish(inp[PAD*CHANNELS+i]);
        }*/
        fasterswisharr(inp + PAD*CHANNELS, BATCH_SIZE*CHANNELS);


        block1.calc(inp + PAD*CHANNELS, out + PAD*CHANNELS, bufd2s + PAD*2*CHANNELS);
        blocks1.calc(out + PAD*CHANNELS, inp + PAD*CHANNELS, bufd2s + PAD*2*CHANNELS);
/*        for (int i = 0; i < BATCH_SIZE*CHANNELS; i+=2400) {
            printf("%f ", inp[PAD*CHANNELS + i]);
        }
        printf("\n");*/
        block2.calc(inp + PAD*CHANNELS, out + PAD*CHANNELS, bufd2s + PAD*2*CHANNELS);
        blocks2.calc(out + PAD*CHANNELS, inp + PAD*CHANNELS, bufd2s + PAD*2*CHANNELS);
        blocks[0].calc(inp + PAD*CHANNELS, out + PAD*CHANNELS, bufd2s + PAD*2*CHANNELS);
        blocks[1].calc(out + PAD*CHANNELS, inp + PAD*CHANNELS, bufd2s + PAD*2*CHANNELS);
        blocks[2].calc(inp + PAD*CHANNELS, out + PAD*CHANNELS, bufd2s + PAD*2*CHANNELS);

        blockc.calc(out + PAD*CHANNELS, inp + PAD*CHANNELS, buf2 + PAD*CHANNELS);
        blockc2.calc(inp + PAD*CHANNELS, out + PAD*CHANNELS, buf2 + PAD*CHANNELS);
        for (int i = 0; i < BATCH_SIZE; i++) {
            for (int j = 0; j < 5; j++) {
                inp[PAD*CHANNELS+i*5+j] = ob[j];
            }
        }
        
        pw_out.run(out + PAD*CHANNELS, inp + PAD*CHANNELS);
    
        return inp + PAD*CHANNELS;
    }

    py::array_t<float> call(py::array_t<float, py::array::c_style | py::array::forcecast> array) {
        float *ptr = (float*)array.request().ptr;
        auto result = py::array_t<float>((array.size()+3)/4*5);
        py::buffer_info result_buf = result.request();
        float* result_ptr = (float*) (result_buf.ptr);

        int out_pos = 0;
        for (int i = 0; i < array.size() - 8*STEPPING_PAD; i += BATCH_SIZE*4-8*STEPPING_PAD) {
            float *out;
            int pad_start, pad_end;
            if (i + BATCH_SIZE*4 < array.size()) {
//                printf("call %d\n", i);
                out = call_chunk(ptr+i);
                pad_start = i == 0 ? 0 : STEPPING_PAD;
                pad_end = STEPPING_PAD;
            } else {
                out = call_chunk(ptr+array.size()-BATCH_SIZE*4);
                pad_end = 0;
                if (i == 0) {
                    pad_start = 0;
                } else {
                    pad_start = (i - (array.size() - BATCH_SIZE*4))/4 + STEPPING_PAD;
                }
            }
/*            printf("%d %d %d copy from %d to %d as %d\n", i, pad_start, pad_end,
                   out_pos, out_pos + (BATCH_SIZE - pad_start - pad_end)*5,
                   (array.size()+3)/4*5);*/
            memcpy(result_ptr + out_pos, out + pad_start * 5, (BATCH_SIZE - pad_start - pad_end)*sizeof(float)*5);

            out_pos += (BATCH_SIZE - pad_start - pad_end) * 5;
        }
        return result;
    }
};

PYBIND11_MODULE(osprey24, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    py::class_<Caller>(m, "Caller")
        .def(py::init<>())
        .def("call", &Caller::call);
}

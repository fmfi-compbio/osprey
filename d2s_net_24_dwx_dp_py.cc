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
const int PAD = 32;
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

static inline void fastersigmoidarr(float *f, int l) {
/*    for (int i = 0; i < l; i++) {
        f[i] = 1 / (1 + expf(-f[i]));
    }
    return;*/
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
        __m256 r6 = _mm256_div_ps(one, r4);
       _mm256_store_ps(f + i, r6);
    }
}

static inline void fasterswisharr(float *f, int l) {
/*    for (int i = 0; i < l; i++) {
        f[i] = f[i] / (1 + expf(-f[i]));
    }
    return;*/
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

class DWX {
  public:
    int channels_;
    float *w3, *w7, *w15, *w31;
    DWX(int channels, map<string, vector<float>>&data, string prefix) : channels_(channels) {
        w3 = (float*) aligned_alloc(ALIGN, 2*channels/6*3*sizeof(float));
        w7 = (float*) aligned_alloc(ALIGN, 2*channels/6*7*sizeof(float));
        w15 = (float*) aligned_alloc(ALIGN, channels/6*15*sizeof(float));
        w31 = (float*) aligned_alloc(ALIGN, channels/6*31*sizeof(float));

        for (int i = 0; i < 2*channels/6*3; i++) {
            w3[i] = data[prefix+".c3.weight"][i];
        }
        for (int i = 0; i < 2*channels/6*7; i++) {
            w7[i] = data[prefix+".c7.weight"][i];
        }
        for (int i = 0; i < channels/6*15; i++) {
            w15[i] = data[prefix+".c15.weight"][i];
        }
        for (int i = 0; i < channels/6*31; i++) {
            w31[i] = data[prefix+".c31.weight"][i];
        }
    }

    void runb(float *data, float*out, int m) {
        for (int output_pos = 0; output_pos < m; output_pos++) {
            int s = 0;
            for (int c = 0; c < channels_*2/6; c++) {
                out[output_pos*channels_ + c + s] = 0;
                for (int j = 0; j < 3; j++) {
                    int i2 = output_pos - 1 + j;
                    out[output_pos*channels_ + c + s] += 
                        w3[j*channels_*2/6 + c] * data[i2*channels_ + c + s];
                }
            }
            s += channels_*2/6;
            for (int c = 0; c < channels_*2/6; c++) {
                out[output_pos*channels_ + c + s] = 0;
                for (int j = 0; j < 7; j++) {
                    int i2 = output_pos - 3 + j;
                    out[output_pos*channels_ + c + s] += 
                        w7[j*channels_*2/6 + c] * data[i2*channels_ + c + s];
                }
            }        
            s += channels_*2/6;
            for (int c = 0; c < channels_/6; c++) {
                out[output_pos*channels_ + c + s] = 0;
                for (int j = 0; j < 15; j++) {
                    int i2 = output_pos - 7 + j;
                    out[output_pos*channels_ + c + s] += 
                        w15[j*channels_/6 + c] * data[i2*channels_ + c + s];
                }
            }        
            s += channels_/6;
            for (int c = 0; c < channels_/6; c++) {
                out[output_pos*channels_ + c + s] = 0;
                for (int j = 0; j < 31; j++) {
                    int i2 = output_pos - 15 + j;
                    out[output_pos*channels_ + c + s] += 
                        w31[j*channels_/6 + c] * data[i2*channels_ + c + s];
                }
            }                
        }
    }

    void run(float *data, float *out, int m) {
        int BS = 64;
        for (int ib = 0; ib < m; ib += BS) {
            int s = 0;
            for (int k = 0; k < channels_*2/6; k+=8) {
                __m256 rs[] = {
                    _mm256_load_ps(w3 + 0*channels_*2/6+k),
                    _mm256_load_ps(w3 + 1*channels_*2/6+k),
                    _mm256_load_ps(w3 + 2*channels_*2/6+k)
                };
                if (false) {
                    for (int i = ib; i < min(ib+BS, m); i++) {
                        __m256 v = _mm256_set1_ps(0);
                        for (int j = 0; j < 3; j++) {
                            int i2 = i - 1 + j;
                            v = _mm256_fmadd_ps(_mm256_load_ps(data + (i2*channels_+k)),
                                                rs[j],
                                                v);
                        }
                        _mm256_store_ps(out + (i*channels_+k), v);
                    }
                } else  {
                    int saddr = ib - 1;
                    __m256 a = _mm256_load_ps(data + saddr*channels_+k+s);
                    __m256 b = _mm256_load_ps(data + (saddr+1)*channels_+k+s);
                    __m256 c = _mm256_load_ps(data + (saddr+2)*channels_+k+s);
                    for (int i = ib; i < min(ib+BS, m); i++) {
                        __m256 v = _mm256_set1_ps(0);
                        v = _mm256_fmadd_ps(a, rs[0], v);
                        v = _mm256_fmadd_ps(b, rs[1], v);
                        v = _mm256_fmadd_ps(c, rs[2], v);

                        _mm256_store_ps(out + (i*channels_+k+s), v);
                        a = b;
                        b = c;
                        c = _mm256_load_ps(data + (i+2) * channels_ + k + s);
                    }
                }
            }
            s += channels_*2/6;
            for (int k = 0; k < channels_*2/6; k+=8) {
                __m256 rs[] = {
                    _mm256_load_ps(w7 + 0*channels_*2/6+k),
                    _mm256_load_ps(w7 + 1*channels_*2/6+k),
                    _mm256_load_ps(w7 + 2*channels_*2/6+k),
                    _mm256_load_ps(w7 + 3*channels_*2/6+k),
                    _mm256_load_ps(w7 + 4*channels_*2/6+k),
                    _mm256_load_ps(w7 + 5*channels_*2/6+k),
                    _mm256_load_ps(w7 + 6*channels_*2/6+k)
                };
                if (true) {
                    for (int i = ib; i < min(ib+BS, m); i++) {
                        __m256 v = _mm256_set1_ps(0);
                        for (int j = 0; j < 7; j++) {
                            int i2 = i - 3 + j;
                            v = _mm256_fmadd_ps(_mm256_load_ps(data + (i2*channels_+k + s)),
                                                rs[j],
                                                v);
                        }
                        _mm256_store_ps(out + (i*channels_+k + s), v);
                    }
                }
            }
            s += channels_*2/6;
            for (int k = 0; k < channels_/6; k+=8) {
                {
                    __m256 rs[] = {
                        _mm256_load_ps(w15 + 0*channels_/6+k),
                        _mm256_load_ps(w15 + 1*channels_/6+k),
                        _mm256_load_ps(w15 + 2*channels_/6+k),
                        _mm256_load_ps(w15 + 3*channels_/6+k),
                        _mm256_load_ps(w15 + 4*channels_/6+k),
                        _mm256_load_ps(w15 + 5*channels_/6+k),
                        _mm256_load_ps(w15 + 6*channels_/6+k)
                    };
                    for (int i = ib; i < min(ib+BS, m); i++) {
                        __m256 v = _mm256_set1_ps(0);
                        for (int j = 0; j < 7; j++) {
                            int i2 = i - 7 + j;
                            v = _mm256_fmadd_ps(_mm256_load_ps(data + (i2*channels_+k + s)),
                                                rs[j],
                                                v);
                        }
                        _mm256_store_ps(out + (i*channels_ + k + s), v);
                    }
                }
                {
                    __m256 rs2[] = {
                        _mm256_load_ps(w15 + 7*channels_/6+k),
                        _mm256_load_ps(w15 + 8*channels_/6+k),
                        _mm256_load_ps(w15 + 9*channels_/6+k),
                        _mm256_load_ps(w15 + 10*channels_/6+k),
                        _mm256_load_ps(w15 + 11*channels_/6+k),
                        _mm256_load_ps(w15 + 12*channels_/6+k),
                        _mm256_load_ps(w15 + 13*channels_/6+k),
                        _mm256_load_ps(w15 + 14*channels_/6+k)
                    };
                    for (int i = ib; i < min(ib+BS, m); i++) {
                        __m256 v = _mm256_load_ps(out + (i*channels_+k+s));
                        for (int j = 0; j < 8; j++) {
                            int i2 = i + j;
                            v = _mm256_fmadd_ps(_mm256_load_ps(data + (i2*channels_+k + s)),
                                                rs2[j],
                                                v);
                        }
                        _mm256_store_ps(out + (i*channels_+k + s), v);
                    }
                }
            }
            s += channels_/6;
            for (int k = 0; k < channels_/6; k+=8) {
                {
                    __m256 rs[] = {
                        _mm256_load_ps(w31 + 0*channels_/6+k),
                        _mm256_load_ps(w31 + 1*channels_/6+k),
                        _mm256_load_ps(w31 + 2*channels_/6+k),
                        _mm256_load_ps(w31 + 3*channels_/6+k),
                        _mm256_load_ps(w31 + 4*channels_/6+k),
                        _mm256_load_ps(w31 + 5*channels_/6+k),
                        _mm256_load_ps(w31 + 6*channels_/6+k)
                    };
                    for (int i = ib; i < min(ib+BS, m); i++) {
                        __m256 v = _mm256_set1_ps(0);
                        for (int j = 0; j < 7; j++) {
                            int i2 = i - 15 + j;
                            v = _mm256_fmadd_ps(_mm256_load_ps(data + (i2*channels_+k + s)),
                                                rs[j],
                                                v);
                        }
                        _mm256_store_ps(out + (i*channels_ + k + s), v);
                    }
                }
                {
                    __m256 rs2[] = {
                        _mm256_load_ps(w31 + 7*channels_/6+k),
                        _mm256_load_ps(w31 + 8*channels_/6+k),
                        _mm256_load_ps(w31 + 9*channels_/6+k),
                        _mm256_load_ps(w31 + 10*channels_/6+k),
                        _mm256_load_ps(w31 + 11*channels_/6+k),
                        _mm256_load_ps(w31 + 12*channels_/6+k),
                        _mm256_load_ps(w31 + 13*channels_/6+k),
                        _mm256_load_ps(w31 + 14*channels_/6+k)
                    };
                    for (int i = ib; i < min(ib+BS, m); i++) {
                        __m256 v = _mm256_load_ps(out + (i*channels_+k+s));
                        for (int j = 0; j < 8; j++) {
                            int i2 = i - 8 + j;
                            v = _mm256_fmadd_ps(_mm256_load_ps(data + (i2*channels_+k + s)),
                                                rs2[j],
                                                v);
                        }
                        _mm256_store_ps(out + (i*channels_+k + s), v);
                    }
                }
                {
                    __m256 rs2[] = {
                        _mm256_load_ps(w31 + 15*channels_/6+k),
                        _mm256_load_ps(w31 + 16*channels_/6+k),
                        _mm256_load_ps(w31 + 17*channels_/6+k),
                        _mm256_load_ps(w31 + 18*channels_/6+k),
                        _mm256_load_ps(w31 + 19*channels_/6+k),
                        _mm256_load_ps(w31 + 20*channels_/6+k),
                        _mm256_load_ps(w31 + 21*channels_/6+k),
                        _mm256_load_ps(w31 + 22*channels_/6+k)
                    };
                    for (int i = ib; i < min(ib+BS, m); i++) {
                        __m256 v = _mm256_load_ps(out + (i*channels_+k+s));
                        for (int j = 0; j < 8; j++) {
                            int i2 = i + j;
                            v = _mm256_fmadd_ps(_mm256_load_ps(data + (i2*channels_+k + s)),
                                                rs2[j],
                                                v);
                        }
                        _mm256_store_ps(out + (i*channels_+k + s), v);
                    }
                }            
                {
                    __m256 rs2[] = {
                        _mm256_load_ps(w31 + 23*channels_/6+k),
                        _mm256_load_ps(w31 + 24*channels_/6+k),
                        _mm256_load_ps(w31 + 25*channels_/6+k),
                        _mm256_load_ps(w31 + 26*channels_/6+k),
                        _mm256_load_ps(w31 + 27*channels_/6+k),
                        _mm256_load_ps(w31 + 28*channels_/6+k),
                        _mm256_load_ps(w31 + 29*channels_/6+k),
                        _mm256_load_ps(w31 + 30*channels_/6+k)
                    };
                    for (int i = ib; i < min(ib+BS, m); i++) {
                        __m256 v = _mm256_load_ps(out + (i*channels_+k+s));
                        for (int j = 0; j < 8; j++) {
                            int i2 = i + j + 8;
                            v = _mm256_fmadd_ps(_mm256_load_ps(data + (i2*channels_+k + s)),
                                                rs2[j],
                                                v);
                        }
                        _mm256_store_ps(out + (i*channels_+k + s), v);
                    }
                }                       
            }
        }
    }
};

/*class DW11 {
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
};*/


void load_dwx_bias(map<string, vector<float>>& data, string prefix, float *out, int channels) {
    int s = 0; 
    for (int i = 0; i < 2*channels/6; i++) {
        out[i+s] = data[prefix+".c3.bias"][i];
    }
    s += 2*channels/6;
    for (int i = 0; i < 2*channels/6; i++) {
        out[i+s] = data[prefix+".c7.bias"][i];
    }
    s += 2*channels/6;
    for (int i = 0; i < channels/6; i++) {
        out[i+s] = data[prefix+".c15.bias"][i];
    }
    s += channels/6;
    for (int i = 0; i < channels/6; i++) {
        out[i+s] = data[prefix+".c31.bias"][i];
    }
}


template<int BatchSize, int Channels>
class Block4 {
  public:
    float *residualb,
          *pwb[4];
    

    PW<BatchSize/3, Channels, 2*Channels> in_pw;
    PW<BatchSize/3, 2*Channels, 2*Channels> pw[2];
    PW<BatchSize, Channels, Channels> residual;
    PW<BatchSize/3, Channels, Channels, 2*Channels, 3*Channels> expand[3];
    DWX dw[4];

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
        DWX(Channels*2, data, prefix+".conv.2"),
        DWX(Channels*2, data, prefix+".conv.6.depthwise"),
        DWX(Channels*2, data, prefix+".conv.10.depthwise"),
        DWX(Channels*2, data, prefix+".conv.14"),
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

        load_dwx_bias(data, prefix+".conv.2", pwb[0], 2*Channels);
        for (int i = 0; i < 2*Channels; i++) {
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

        memset(input + (BatchSize/3)*Channels, 0, (2*BatchSize/3)*Channels*sizeof(float));
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
    DWX dw[5];

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
        DWX(Channels*2, data, prefix+".conv.2"),
        DWX(Channels*2, data, prefix+".conv.6.depthwise"),
        DWX(Channels*2, data, prefix+".conv.10.depthwise"),
        DWX(Channels*2, data, prefix+".conv.14.depthwise"),
        DWX(Channels*2, data, prefix+".conv.18"),
    } {
        residualb = (float*) aligned_alloc(ALIGN, Channels*sizeof(float));
        for (int i = 0; i < 5; i++) {
            pwb[i] = (float*) aligned_alloc(ALIGN, 2*Channels*sizeof(float)); 
        }

        for (int i = 0; i < Channels; i++) {
            residualb[i] = data[prefix+".residual.0.conv.bias"][i];
            residualb[i] += data[prefix+".conv.19.convs.0.bias"][i];
        }


        load_dwx_bias(data, prefix+".conv.2", pwb[0], 2*Channels);
        for (int i = 0; i < 2*Channels; i++) {
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

        memset(input + (BatchSize/3)*Channels, 0, (2*BatchSize/3)*Channels*sizeof(float));
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
    DWX dw[6];

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
        DWX(Channels*2, data, prefix+".conv.2"),
        DWX(Channels*2, data, prefix+".conv.6.depthwise"),
        DWX(Channels*2, data, prefix+".conv.10.depthwise"),
        DWX(Channels*2, data, prefix+".conv.14.depthwise"),
        DWX(Channels*2, data, prefix+".conv.18.depthwise"),
        DWX(Channels*2, data, prefix+".conv.22"),
    }{
        residualb = (float*) aligned_alloc(ALIGN, Channels*sizeof(float));
        for (int i = 0; i < 6; i++) {
            pwb[i] = (float*) aligned_alloc(ALIGN, 2*Channels*sizeof(float)); 
        }

        for (int i = 0; i < Channels; i++) {
            residualb[i] = data[prefix+".residual.0.conv.bias"][i];
            residualb[i] += data[prefix+".conv.23.convs.0.bias"][i];
        }

        load_dwx_bias(data, prefix+".conv.2", pwb[0], 2*Channels);
        for (int i = 0; i < 2*Channels; i++) {
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

        memset(input + (BatchSize/3)*Channels, 0, (2*BatchSize/3)*Channels*sizeof(float));
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
    DWX dw;

    BlockC(map<string, vector<float>>& data, string prefix) :
        pw(data[prefix+".conv.0.pointwise.weight"]), dw(Channels, data, prefix+".conv.0.depthwise") {
        
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

template<int BatchSize, int Channels>
class BlockDP {
  public:
    PW<BatchSize, Channels, Channels*2> pw;
    PW<BatchSize, Channels, 2> predictor;
    
    float norm_mean;
    float *pwb, *predb;

    BlockDP(map<string, vector<float>>&data, string prefix) : 
        pw(data[prefix+".conv.0.conv.weight"]),
        predictor(data[prefix+".predictor.0.weight"]),
        norm_mean(data[prefix+".norm_mean"][0]) {    
        pwb = (float*) aligned_alloc(ALIGN, Channels*2*sizeof(float)); 
        predb = (float*) aligned_alloc(ALIGN, 8*sizeof(float)); 

        for (int i = 0; i < Channels*2; i++) {
            pwb[i] = data[prefix+".conv.0.conv.bias"][i];
        }
        for (int i = 0; i < 8; i++) {
            predb[i] = data[prefix+".predictor.0.bias"][i%2];
        }
    }

    void calc(float *input, float *outdata, float *outdp) {
        for (int i = 0; i < BatchSize; i++) {
            memcpy(outdata + i*Channels*2, pwb, Channels*2*sizeof(float));
        }
        pw.run(input, outdata);
        for (int i = 0; i < BatchSize/4; i++) {
            memcpy(outdp + i*8, predb, 8*sizeof(float));
        }
        predictor.run(input, outdp);
        fasterswisharr(outdata, BatchSize * Channels * 2);
        fastersigmoidarr(outdp, BatchSize * 2);
        for (int i = 1; i < BatchSize * 2; i+=2) {
            outdp[i] *= norm_mean;
        }
    }
};

class CallerDP {
  public:
    map<string, vector<float>> weights;
    PW<BATCH_SIZE*4, 9, CHANNELS/4> pw_in;

    Block4<BATCH_SIZE*4, CHANNELS/4> block1;
    BlockS<BATCH_SIZE*4, CHANNELS/4> blocks1;
    Block5<BATCH_SIZE*2, CHANNELS/2> block2;
//    BlockS<BATCH_SIZE*2, CHANNELS/2> blocks2;
    BlockDP<BATCH_SIZE*2, CHANNELS/2> blockdp;

    Block6<BATCH_SIZE, CHANNELS> blocks[3];

    BlockC<BATCH_SIZE, CHANNELS> blockc;
    BlockC2<BATCH_SIZE, CHANNELS> blockc2;
    float *ib, *inp_im2col, *inp, *out, *buf2, *bufd2s, *outdp;

    CallerDP() : weights(load_weight("weights/net24dp.txt")),
        pw_in(weights["s.e.encoder.0.conv.0.conv.weight"]),
        block1(weights, "s.e.encoder.1"),
        blocks1(weights, "s.e.encoder.2"),
        block2(weights, "s.e.encoder.3"),
        blockdp(weights, "s.e.encoder.4"),
        blocks {
            Block6<BATCH_SIZE, CHANNELS>(weights, "s.e.encoder.5"),
            Block6<BATCH_SIZE, CHANNELS>(weights, "s.e.encoder.6"),
            Block6<BATCH_SIZE, CHANNELS>(weights, "s.e.encoder.7"),
        },
        blockc(weights, "s.e.encoder.8"),
        blockc2(weights, "s.e.encoder.9")
    {
        ib = (float*) aligned_alloc(ALIGN, CHANNELS/4*sizeof(float));
        for (int i = 0; i < CHANNELS/4; i++) {
            ib[i] = weights["s.e.encoder.0.conv.0.conv.bias"][i];
        }
        inp_im2col = (float*) aligned_alloc(ALIGN, (BATCH_SIZE)*9*4*sizeof(float));
        inp = (float*) aligned_alloc(ALIGN, (BATCH_SIZE+2*PAD)*CHANNELS*sizeof(float));
        out = (float*) aligned_alloc(ALIGN, (BATCH_SIZE+2*PAD)*CHANNELS*sizeof(float));
        buf2 = (float*) aligned_alloc(ALIGN, (BATCH_SIZE+2*PAD)*CHANNELS*sizeof(float));
        bufd2s = (float*) aligned_alloc(ALIGN, (BATCH_SIZE/3+2*PAD)*2*CHANNELS*sizeof(float));
        outdp = (float*) aligned_alloc(ALIGN, (BATCH_SIZE)*CHANNELS*2*sizeof(float));
        memset(inp_im2col, 0, (BATCH_SIZE)*9*4*sizeof(float));
        memset(inp, 0, (BATCH_SIZE+2*PAD)*CHANNELS*sizeof(float));
        memset(out, 0, (BATCH_SIZE+2*PAD)*CHANNELS*sizeof(float));
        memset(buf2, 0, (BATCH_SIZE+2*PAD)*CHANNELS*sizeof(float));
        memset(bufd2s, 0, (BATCH_SIZE/3+2*PAD)*2*CHANNELS*sizeof(float)); 
    }

    void call_chunk_base(float* inpx) {
        im2col(inpx, inp_im2col, BATCH_SIZE*4);
        for (int j = 0; j < BATCH_SIZE*4; j++) {
            memcpy(inp + (j+PAD*4)*CHANNELS/4, ib, CHANNELS/4*sizeof(float));
        }
        pw_in.run(inp_im2col, inp + PAD*CHANNELS);

        fasterswisharr(inp + PAD*CHANNELS, BATCH_SIZE*CHANNELS);


        block1.calc(inp + PAD*CHANNELS, out + PAD*CHANNELS, bufd2s + PAD*2*CHANNELS);

        blocks1.calc(out + PAD*CHANNELS, inp + PAD*CHANNELS, bufd2s + PAD*2*CHANNELS);
        block2.calc(inp + PAD*CHANNELS, out + PAD*CHANNELS, bufd2s + PAD*2*CHANNELS);
        blockdp.calc(out + PAD*CHANNELS, outdp, bufd2s + PAD*2*CHANNELS);
    }

    void call_chunk_top() {
        blocks[0].calc(inp + PAD*CHANNELS, out + PAD*CHANNELS, bufd2s + PAD*2*CHANNELS);
        blocks[1].calc(out + PAD*CHANNELS, inp + PAD*CHANNELS, bufd2s + PAD*2*CHANNELS);
        blocks[2].calc(inp + PAD*CHANNELS, out + PAD*CHANNELS, bufd2s + PAD*2*CHANNELS);
        blockc.calc(out + PAD*CHANNELS, inp + PAD*CHANNELS, buf2 + PAD*CHANNELS);
        blockc2.calc(inp + PAD*CHANNELS, out + PAD*CHANNELS, buf2 + PAD*CHANNELS);
    }

    py::array_t<float> call(py::array_t<float, py::array::c_style | py::array::forcecast> array) {
        float *ptr = (float*)array.request().ptr;

        float* dyn_pool_res_data = (float*)aligned_alloc(ALIGN, (array.size()+1)/2*96*sizeof(float));
        float* dyn_pool_res_wm = (float*)aligned_alloc(ALIGN, (array.size()+1)/2*2*sizeof(float));

        int out_pos_data = 0;
        int out_pos_wm = 0;
        for (int i = 0; i < array.size() - 8*STEPPING_PAD; i += BATCH_SIZE*4-8*STEPPING_PAD) {
            int pad_start, pad_end;
            if (i + BATCH_SIZE*4 < array.size()) {
//                printf("call %d\n", i);
                call_chunk_base(ptr+i);
                pad_start = i == 0 ? 0 : 2*STEPPING_PAD;
                pad_end = 2*STEPPING_PAD;
            } else {
                call_chunk_base(ptr+array.size()-BATCH_SIZE*4);
                pad_end = 0;
                if (i == 0) {
                    pad_start = 0;
                } else {
                    pad_start = (i - (array.size() - BATCH_SIZE*4))/2 + 2*STEPPING_PAD;
                }
            }
//            printf("%d %d %d copy from %d to %d as %d\n", i, pad_start, pad_end,
//                   out_pos_data, out_pos_data + (BATCH_SIZE*2 - pad_start - pad_end)*96,
//                   (array.size()+1)/2*96);
            memcpy(dyn_pool_res_data + out_pos_data, outdp + pad_start * 96, (BATCH_SIZE*2 - pad_start - pad_end)*sizeof(float)*96);

            memcpy(dyn_pool_res_wm + out_pos_wm, bufd2s + 2*PAD*CHANNELS + pad_start * 2, (BATCH_SIZE*2 - pad_start - pad_end)*sizeof(float)*2);

            out_pos_data += (BATCH_SIZE*2 - pad_start - pad_end) * 96;
            out_pos_wm += (BATCH_SIZE*2 - pad_start - pad_end) * 2;
        }

        // Do cumsum
        float cur_cs = 0;
        for (int i = 1; i < (array.size()+1)/2*2; i+=2) {
            cur_cs += dyn_pool_res_wm[i];
            dyn_pool_res_wm[i] = cur_cs;
        }
        int pooled_size = cur_cs + 2;
        printf("pool size %d\n", pooled_size);
        float *pool_out = (float*) aligned_alloc(ALIGN, pooled_size*96*sizeof(float));
        memset(pool_out, 0, pooled_size*96*sizeof(float));

        for (int i = 0; i < (array.size()+1)/2; i++) {
            float w = dyn_pool_res_wm[i*2];
            float pos = dyn_pool_res_wm[i*2+1];
            int floor = pos;
            int ceil = pos+1;
            float floorw = (ceil - pos)*w;
            float ceilw = (pos - floor)*w;

            for (int j = 0; j < 96; j++) {
                pool_out[floor*96+j] += dyn_pool_res_data[i*96+j]*floorw;
                pool_out[ceil*96+j] += dyn_pool_res_data[i*96+j]*ceilw;
            }
        }

/*        for (int i = 0; i < 500; i+=50) {
            printf("%d %f\n", i, pool_out[i]);
        }*/

        if (pooled_size < BATCH_SIZE) {

            auto result = py::array_t<float>(0);
            return result;
        }
        auto result = py::array_t<float>(pooled_size*48);

        py::buffer_info result_buf = result.request();
        float* result_ptr = (float*) (result_buf.ptr);
//        printf("res ptr %llx\n", result_ptr);

        int out_pos = 0;
        for (int i = 0; i < pooled_size - 2*STEPPING_PAD; i+= BATCH_SIZE-2*STEPPING_PAD) {
            int pad_start, pad_end;
            if (i + BATCH_SIZE < pooled_size) {
//                printf("call %d\n", i);
                memcpy(inp + PAD*CHANNELS, pool_out + i*96, BATCH_SIZE*CHANNELS*sizeof(float));
                call_chunk_top();
                pad_start = i == 0 ? 0 : STEPPING_PAD;
                pad_end = STEPPING_PAD;
            } else {
                memcpy(inp + PAD*CHANNELS, pool_out + (pooled_size - BATCH_SIZE)*96, BATCH_SIZE*CHANNELS*sizeof(float));
                call_chunk_top();
                pad_end = 0;
                if (i == 0) {
                    pad_start = 0;
                } else {
                    pad_start = (i - (pooled_size - BATCH_SIZE)) + STEPPING_PAD;
                }
            }
//            printf("%d %d %d copy from %d to %d as %d\n", i, pad_start, pad_end,
//                   out_pos, out_pos + (BATCH_SIZE - pad_start - pad_end)*48,
//                   pooled_size*48);
            memcpy(result_ptr + out_pos, out + PAD*CHANNELS + pad_start * 48, (BATCH_SIZE - pad_start - pad_end)*sizeof(float)*48);
//            printf("copy done\n");

            out_pos += (BATCH_SIZE - pad_start - pad_end) * 48;
        }

        free(pool_out);
        free(dyn_pool_res_data);
        free(dyn_pool_res_wm);

        return result;
    }
};

PYBIND11_MODULE(osprey24dwxdp, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    py::class_<CallerDP>(m, "CallerDP")
        .def(py::init<>())
        .def("call", &CallerDP::call);
}

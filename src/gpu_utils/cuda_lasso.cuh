#ifndef CUDA_LASSO_H_
#define CUDA_LASSO_H_
#include "src/gpu_utils/cuda_ml_optimiser.h"

void cuda_lasso(int fsc143, int tv_iters, RFLOAT l_r, RFLOAT mu, RFLOAT tv_alpha, RFLOAT tv_beta, MultidimArray<RFLOAT> &Fconv,
        MultidimArray<RFLOAT> &Fweight, MultidimArray<Complex> &Ftest_conv, MultidimArray<RFLOAT> &Ftest_weight, MultidimArray<RFLOAT> &vol_out, MlDeviceBundle *devBundle, int data_dim, RFLOAT normfft, RFLOAT nrparts, bool do_nag = false, RFLOAT implicit_weight = 0.1, RFLOAT tv_eps=0.1, RFLOAT tv_epsp=0.1);

void cuda_lasso_o(int fsc143, int tv_iters, RFLOAT l_r, RFLOAT mu, RFLOAT tv_alpha, RFLOAT tv_beta, MultidimArray<RFLOAT> &Fconv,
        MultidimArray<RFLOAT> &Fweight, MultidimArray<RFLOAT> &vol_out, MlDeviceBundle *devBundle, int data_dim, RFLOAT normfft, RFLOAT nrparts, bool do_nag = false, RFLOAT implicit_weight = 0.1, RFLOAT tv_eps=0.1, RFLOAT tv_epsp=0.1);

#endif

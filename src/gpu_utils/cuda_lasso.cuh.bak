#ifndef CUDA_LASSO_H_
#define CUDA_LASSO_H_
#include "src/gpu_utils/cuda_ml_optimiser.h"

void cuda_lasso(int tv_iters, RFLOAT l_r, RFLOAT mu, RFLOAT tv_alpha, RFLOAT tv_beta, RFLOAT eps, MultidimArray<RFLOAT> &Fconv,
        MultidimArray<RFLOAT> &Fweight, MultidimArray<RFLOAT> &vol_out, MlDeviceBundle *devBundle, int data_dim, RFLOAT normfft, RFLOAT nrparts, bool do_nag = false, RFLOAT implicit_weight = 0.1);
#endif

#include "src/gpu_utils/cuda_lasso.cuh"
#include <signal.h>
#include <math.h>
#include "src/gpu_utils/cuda_utils_cub.cuh"
#include "src/gpu_utils/cuda_fft.h"
#include "src/gpu_utils/cuda_kernels/helper.cuh"
#include <stack>

void cuda_lasso(int tv_iters, RFLOAT l_r, RFLOAT mu, RFLOAT tv_alpha, RFLOAT tv_beta, RFLOAT eps, MultidimArray<RFLOAT> &Fconv,
        MultidimArray<RFLOAT> &Fweight, MultidimArray<RFLOAT> &vol_out, MlDeviceBundle *devBundle, int data_dim, RFLOAT normfft){
    //normfft = max(normfft, 1.);
    cudaSetDevice(devBundle->device_id);
    int img_size = vol_out.nzyxdim;
    CudaGlobalPtr<XFLOAT, false> img(img_size, devBundle->stream);
    CudaGlobalPtr<XFLOAT, false> yob(img_size, devBundle->stream);
    CudaGlobalPtr<XFLOAT, false> weight(Fweight.nzyxdim, devBundle->stream);
    CudaGlobalPtr<XFLOAT, false> grads(img_size, devBundle->stream);
    CudaGlobalPtr<XFLOAT, false> momentum(img_size, devBundle->stream);
    CudaFFTT<false> transformer(devBundle->stream, 0, data_dim);
    transformer.setSize(vol_out.xdim, vol_out.ydim, vol_out.zdim);
    img.device_alloc();
    yob.device_alloc();
    for(int i = 0; i < Fconv.getSize(); i++){
        //img[i] = 0.f;
        //transformer.fouriers[i].x = Fconv.data[i].real;
        //transformer.fouriers[i].y = Fconv.data[i].imag;
        yob[i] = Fconv.data[i];
    }
    //transformer.fouriers.cp_to_device();
    //transformer.fouriers.streamSync();
    //transformer.backward(yob);
    //img.cp_to_device();
    yob.cp_to_device();
    yob.streamSync();
    img.device_init(0.f);
    //img.streamSync();
    grads.device_alloc();
    momentum.device_alloc();
    momentum.device_init(0.f);
    //move Fweight from host to device
    weight.device_alloc();
    for(int i = 0; i < Fweight.nzyxdim; i++){
        weight[i] = Fweight.data[i];
    }
    weight.cp_to_device();
    weight.streamSync();
    RFLOAT tv_eps = 0.2;
    if(normfft > 1.)
        l_r /= normfft;
    int FBsize = (int) ceilf((float)transformer.fouriers.getSize()/(float)BLOCK_SIZE);
    int imgBsize = (int) ceilf((float)img.getSize()/(float)BLOCK_SIZE);
    //cuda_kernel_multi<<<imgBsize, BLOCK_SIZE, 0, yob.getStream()>>>(
    //            ~yob,
    //            (XFLOAT)1/((XFLOAT)normfft),
    //            yob.getSize());
    //yob.streamSync();
    //XFLOAT weight_norm = getSquareSumOnDevice(yob);
    //weight_norm = sqrt(weight_norm/img_size);
    //std::cout << "device: " << devBundle->device_id << " Mout: " << weight_norm << std::endl;
    std::cout << "start optimizing " << l_r << ", " << weight.getSize() << std::endl;

    for(int m_c = 0; m_c <= tv_iters; m_c++){
        //forward transform img
        img.cp_on_device(transformer.reals.d_ptr);
        transformer.reals.streamSync();
        transformer.forward();
        transformer.fouriers.streamSync();
        //multiply with weight and normalization factor
        
        cuda_kernel_complex_multi<<<FBsize, BLOCK_SIZE, 0, transformer.fouriers.getStream()>>>(
                (XFLOAT*)~transformer.fouriers,
                ~weight,
                (XFLOAT)1/((XFLOAT)transformer.reals.getSize()),
                transformer.fouriers.getSize());
        transformer.fouriers.streamSync();
        //inverse transform and put the transformation on grads
        transformer.backward(grads);
        //transformer.backward();
        //transformer.reals.cp_on_device(grads.d_ptr);
        //substract Mout
        cuda_kernel_substract<<<imgBsize, BLOCK_SIZE, 0, grads.getStream()>>>(
                ~grads,
                ~yob,
                grads.getSize());
        grads.streamSync();
        //get the norm of gradient
        //XFLOAT grads_norm = getSquareSumOnBlock(grads);
        //grads_norm = sqrt(grads_norm);
        //cuda_kernel_multi<<<imgBsize, BLOCK_SIZE, 0, grads.getStream()>>>(
        //        ~grads,
        //        (XFLOAT)1/((XFLOAT)normfft),
        //        grads.getSize());
        //grads.streamSync();
        //get graph gradient
        int Z = vol_out.zdim;
        int Y = vol_out.ydim;
        int X = vol_out.xdim;
        if(data_dim == 3){
            cuda_kernel_graph_grad<<<imgBsize, BLOCK_SIZE, 0, grads.getStream()>>>(
                    ~img,
                    ~grads,
                    Z,
                    Y,
                    X,
                    tv_beta,
                    eps,//eps of log approximation
                    tv_eps,//eps of l1 norm approximation
                    grads.getSize());
        } else {
            cuda_kernel_graph_grad<<<imgBsize, BLOCK_SIZE, 0, grads.getStream()>>>(
                    ~img,
                    ~grads,
                    Y,
                    X,
                    tv_beta,
                    tv_eps,
                    grads.getSize());

        }
        grads.streamSync();
        //update momentum
        //cuda_kernel_update_momentum<<<imgBsize, BLOCK_SIZE, 0, grads.getStream()>>>(
        //        ~grads,
        //        ~momentum,
        //        mu,
        //        l_r,
        //        grads.getSize());
        //grads.streamSync();
        //update momentum and soft thresholding
        cuda_kernel_soft_threshold<<<imgBsize, BLOCK_SIZE, 0, grads.getStream()>>>(
                ~img,
                ~momentum,
                ~grads,
                mu,
                l_r,
                tv_alpha,
                eps,
                grads.getSize());
        img.streamSync();
        if(m_c % 100 == 0){
            //std::cout << m_c << " " << grads_norm;
            RFLOAT grads_norm = getSquareSumOnBlock(img);
            RFLOAT resi_norm = getSquareSumOnBlock(grads);
            grads_norm = sqrt(grads_norm/img_size);
            resi_norm = sqrt(resi_norm/img_size);
            std::cout <<  m_c << " " << grads_norm << " " << resi_norm/grads_norm << std::endl;
        }
        l_r *= exp(-0.005);
    }
    //now copy image to host
    img.cp_to_host();
    //now set vol_out
    bool hasnan = false;
    for(int i = 0; i < img_size; i++){
        if(isnan(img[i])){
            vol_out.data[i] = 0.;
            hasnan = true;
        }
        else
            vol_out.data[i] = img[i];
    }
    if(hasnan) std::cout << "WARNING: find nan in reconstruction." << std::endl;
    transformer.clear();
    img.free();
    yob.free();
    weight.free();
    grads.free();
    momentum.free();
    //cudaDeviceReset();
}

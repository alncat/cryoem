#include "src/gpu_utils/cuda_lasso.cuh"
#include <iomanip>
#include <signal.h>
#include <queue>
#include "src/fftw.h"
#include <math.h>
#include "src/gpu_utils/cuda_utils_cub.cuh"
#include "src/gpu_utils/cuda_fft.h"
#include "src/gpu_utils/cuda_kernels/helper.cuh"

inline int mapToCompact(int k, int i, int j, int Z, int Y, int X, int ZZ, int YY, int XX){
    if(k >= ZZ/2) k += ZZ;
    if(i >= YY/2) i += YY;
    if(j >= XX/2) j += XX;
    return k*Y*X + i*X + j;
}

void cuda_lasso(int tv_iters, RFLOAT l_r, RFLOAT mu, RFLOAT tv_alpha, RFLOAT tv_beta, RFLOAT eps, MultidimArray<RFLOAT> &Fconv,
        MultidimArray<RFLOAT> &Fweight, MultidimArray<RFLOAT> &vol_out, MlDeviceBundle *devBundle, int data_dim, RFLOAT normalise, RFLOAT nrparts, bool do_nag, RFLOAT implicit_weight){
    //normfft = max(normfft, 1.);
    cudaSetDevice(devBundle->device_id);
    devBundle->setStream();
    std::cout <<" Device: " << devBundle->device_id <<", " << data_dim;
    std::cout << " " << vol_out.xdim << ", " << vol_out.ydim << ", " << vol_out.zdim << std::endl;
    //std::cout << "tv_iters: " << tv_iters << " ";
    int ZZ = vol_out.zdim >> 1;
    int YY = vol_out.ydim >> 1;
    int XX = vol_out.xdim >> 1;
    int Z = vol_out.zdim;
    int Y = vol_out.ydim;
    int X = vol_out.xdim;

    int img_size = vol_out.nzyxdim;
    int img_size_h = 0;

    if(data_dim == 3) img_size_h = img_size/8;
    else img_size_h = img_size/4;

    std::priority_queue<XFLOAT, std::vector<XFLOAT>, std::greater<XFLOAT>> pq;
    CudaUnifedPtr<XFLOAT> img(devBundle->stream, devBundle->device_id, img_size);
    //momentum term
    //CudaUnifedPtr<XFLOAT> momentum(devBundle->stream, devBundle->device_id, img_size);
    CudaUnifedPtr<XFLOAT> yob(devBundle->stream, devBundle->device_id, img_size);
    CudaUnifedPtr<XFLOAT> grads(devBundle->stream, devBundle->device_id, img_size);
    //CudaUnifedPtr<XFLOAT> vol(devBundle->stream, devBundle->device_id, img_size);
    CudaUnifedPtr<XFLOAT> sigma_norm(devBundle->stream, devBundle->device_id, 1);
    XFLOAT mem_size = vol_out.getSize()*sizeof(XFLOAT)/1024./1024.;
    //std::cout << "Mem size " << 4*mem_size << " " << 3*img_size_h*sizeof(XFLOAT)/1024./1024. << "MB "<< std::endl;
    //CudaFFTT<false> transformer(devBundle->stream, NULL, data_dim);
    CudaFFTU transformer(devBundle->stream, devBundle->device_id, data_dim);
    transformer.setSize(vol_out.xdim, vol_out.ydim, vol_out.zdim);
    CudaUnifedPtr<XFLOAT> weight(devBundle->stream, devBundle->device_id, Fweight.getSize());
    //img.device_alloc();
    img.setPtr(transformer.reals.ptr);
    //img.alloc();
    //momentum is used to calculate grads
    //momentum.setPtr(transformer.reals.ptr);
    yob.alloc();
    //vol.alloc();
    sigma_norm.alloc();
    XFLOAT lambda = implicit_weight*normalise;
    std::cout << "weight " << weight.getSize() << ", " << Fconv.getSize()<< std::endl;
    RFLOAT fconv_norm = 0.;
    XFLOAT yob_norm = 0.;
    int median_size = img_size_h/128;
    int sparse_count = 0;

    if(do_nag) {
        for(int i = 0; i < Fconv.nzyxdim; i++){
            //img[i] = 0.f;
            //transformer.fouriers[i].x = Fconv.data[i].real;
            //transformer.fouriers[i].y = Fconv.data[i].imag;
            yob[i] = Fconv.data[i];
            fconv_norm += Fconv.data[i]*Fconv.data[i];
            yob[i] += lambda*((XFLOAT)vol_out.data[i]);
            //vol[i] = lambda*((XFLOAT)vol_out.data[i]);
            //std::cout << yob[i] << std::endl;
        }
        fconv_norm = sqrt(fconv_norm/Fconv.nzyxdim);
        //for(int k = 0; k < ZZ; k++)
        //    for(int i = 0; i < YY; i++)
        //        for(int j = 0; j < XX; j++){
        //            int index = mapToCompact(k, i, j, Z, Y, X, ZZ, YY, XX);
        //            int my_index = k*YY*XX + i*XX + j;
        //            yob[my_index] = Fconv.data[index];
        //            yob[my_index] += lambda*((XFLOAT)vol_out.data[index]);
        //        }
        //for(int i = 0; i < img_size; i++){
        //    img[i] = vol_out.data[i];
        //    //momentum[i] = momentum_out.data[i];
        //}
        //momentum.device_init(0.f);
        //vol_out is of even size
        FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(vol_out){
            if((k < ZZ || k >= Z - ZZ) &&
               (i < YY || i >= Y - YY) &&
               (j < XX || j >= X - XX)){
                img[k*Y*X + i*X + j] = DIRECT_A3D_ELEM(vol_out, k, i, j);
                if(img[k*Y*X + i*X + j] > 1.e-3) {
                    sparse_count += 1;
                    yob_norm += abs(img[k*Y*X + i*X + j]);
                } else {
                    img[k*Y*X + i*X + j] = 0.;
                }
                if(pq.size() < median_size) pq.push(img[k*Y*X + i*X + j]);
                else if(pq.top() < img[k*Y*X + i*X + j]) {
                    pq.pop();
                    pq.push(img[k*Y*X + i*X + j]);
                }
            } else {
                img[k*Y*X + i*X + j] = 0.;
            }
            //momentum[k*Y*X + i*X + j] = img[k*Y*X + i*X + j];
        }
        //img.init();
    } else {
        for(int i = 0; i < Fconv.nzyxdim; i++){
            //img[i] = 0.f;
            //transformer.fouriers[i].x = Fconv.data[i].real;
            //transformer.fouriers[i].y = Fconv.data[i].imag;
            yob[i] = Fconv.data[i];
            //std::cout << yob[i] << std::endl;
        }
        img.init();
        //momentum.device_init(0.f);
    }

    //adjust median according to sparseness count
    if(sparse_count*0.5 < median_size){
        int new_median_size = sparse_count*0.5;
        for(int i = 0; i < median_size - new_median_size; i++){
            pq.pop();
        }
    }
    std::cout << "median: " << pq.top() << " pq_fraction: " << pq.size()/XFLOAT(img_size_h) << " sparseness: " << sparse_count/XFLOAT(img_size_h) << std::endl;
    //yob.cp_to_device();
    yob.set_read_only();
    yob.attach_to_stream();
    //vol.set_read_only();
    //vol.attach_to_stream();
    img.attach_to_stream();
    transformer.fouriers.attach_to_device();
    transformer.fouriers.attach_to_stream();
    //yob.attach_to_device();
    //transformer.fouriers.cp_to_device();
    //transformer.backward(yob);
    //img.cp_to_device();
    //yob.dump_device_to_file("yob"+devBundle->device_id);
    
    grads.alloc();
    grads.attach_to_stream();
    //move Fweight from host to device
    size_t free_byte ;
    size_t total_byte ;
    //cudaMemGetInfo( &free_byte, &total_byte ) ;
    //std::cout << "Free Mem " << free_byte/1024./1024. << " Total Mem " << total_byte/1024./1024. << std::endl;

    weight.alloc();
    //int xdim = Fconv.xdim/2 + 1;
    XFLOAT max_weight = 0;
    XFLOAT min_weight = normalise;
    FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Fweight){
        int kw = kp;
        int iw = ip;
        int jw = jp;
        if(kp < 0) kw += Fweight.zdim;
        if(ip < 0) iw += Fweight.ydim;
        int index = kw*Fweight.ydim*Fweight.xdim + iw*Fweight.xdim + jw;
        weight[index] = FFTW_ELEM(Fweight, kp, ip, jp);
        if(max_weight < weight[index]) max_weight = weight[index];
        if(min_weight > weight[index] && weight[index] > 1e-5) min_weight = weight[index];        
        //std::cout << weight[index] << " " << index << std::endl;
    }
    //for(int i = 0; i < Fweight.nzyxdim; i++){
    //    weight[i] = Fweight.data[i];
    //    tot_weight += std::abs(weight[i]);
    //}
    weight.set_read_only();
    weight.attach_to_stream();
    //synchronize before launch kernel
    img.streamSync();
    weight.cp_to_device();
    //std::cout << ~weight << " " << ~grads << " " << ~img << " " << ~yob << std::endl;
    yob_norm /= sparse_count;
    //eps = yob_norm*1.;
    eps = max(pq.top(), 0.1);
    XFLOAT tv_eps = 0.00005;
    XFLOAT tv_log_eps = eps/3;
    int FBsize = (int) ceilf((float)transformer.fouriers.getSize()/(float)BLOCK_SIZE);
    int imgBsize = (int) ceilf((float)img_size_h/(float)BLOCK_SIZE);
    int imgBFsize = (int) ceilf((float)img.getSize()/(float)BLOCK_SIZE);
    //cuda_kernel_multi<<<imgBsize, BLOCK_SIZE, 0, yob.getStream()>>>(
    //            ~yob,
    //            (XFLOAT)1/((XFLOAT)normfft),
    //            yob.getSize());
    XFLOAT weight_norm = getSquareSumOnBlock(weight);
    weight_norm = sqrt(weight_norm/weight.getSize());
    
    //RFLOAT yob_norm = getMaxOnDevice(yob);
    //yob_norm = sqrt(yob_norm/img_size);
    std::cout << "device: " << devBundle->device_id << " weight: " << weight_norm << std::endl;
    Fweight.printShape();
    //l_r *= 1./(weight_norm);
    //std::cout << "start optimizing " << "yob: " << yob_norm << ", " << "weight: " << weight_norm << ", l_r: " << l_r << ", " << std::endl;
    l_r = l_r/(max_weight + normalise);
    std::cout << "start optimizing " << "lr: " << l_r << " lambda: " << lambda << " avg weight : " << normalise << " max weight: " << max_weight << " min weight: " << min_weight << " condition number: " << max_weight/min_weight << std::endl;
    tv_alpha *= std::sqrt(normalise);
    tv_beta *= std::sqrt(normalise);
    //XFLOAT yob_norm = getSquareSumOnBlock(img);
    //XFLOAT yob_norm = 0;
    //for(int i = 0; i < img_size; i++){
    //    yob_norm += img[i]*img[i];
    //}
    std::cout << "tv_alpha: " << tv_alpha << ", tv_beta: " << tv_beta << " fconv_norm: " << fconv_norm << " img_norm: " << yob_norm <<" img_size " << img_size << " FBsize " << FBsize << std::endl;
    XFLOAT scale = (XFLOAT)1/((XFLOAT)transformer.reals.getSize());
    XFLOAT w = tv_alpha/eps*0.5;
    XFLOAT tk = 1.;
    for(int m_c = 0; m_c <= tv_iters; m_c++){
        //forward transform img/momentum
        transformer.forward();
        //multiply with weight and normalization factor
        //cuda_kernel_complex_multi<<<FBsize, BLOCK_SIZE, 0, transformer.fouriers.getStream()>>>(
        //        (XFLOAT*)~transformer.fouriers,
        //        ~weight,
        //        (XFLOAT)1/((XFLOAT)transformer.reals.getSize()),
        //        transformer.fouriers.getSize());
        int xdim = X/2 + 1;
        cuda_kernel_complex_multi<<<FBsize, BLOCK_SIZE, 0, transformer.fouriers.getStream()>>>(
                (XFLOAT*)~transformer.fouriers,
                ~weight,
                scale,
                w*scale,
                Z,
                Y,
                xdim,
                Fweight.zdim,
                Fweight.ydim,
                Fweight.xdim,
                transformer.fouriers.getSize());

        //inverse transform and put the transformation on grads
        transformer.backward(grads);
        //transformer.reals.cp_on_device(grads.d_ptr);
        //substract Mout
        //cuda_kernel_substract<<<imgBsize, BLOCK_SIZE, 0, grads.getStream()>>>(
        //        ~grads,
        //        ~yob,
        //        grads.getSize());
        //work on windowed grads
        //sigma_norm[0] = 0.;
        cuda_kernel_substract<<<imgBsize, BLOCK_SIZE, 0, grads.getStream()>>>(
                ~grads,
                ~yob,
                ~img,
                //~momentum,
                //~vol,
                (XFLOAT)lambda,
                //~sigma_norm,
                ZZ,
                YY,
                XX,
                img_size_h);
                //grads.getSize());
        //synchronize before returning sum
        //sigma_norm.streamSync();
        //sigma_norm[0] = sqrt(sigma_norm[0]/img_size_h);
        //get graph gradient
        if(data_dim == 3){
            //work on windowed grads
            cuda_kernel_graph_grad<<<imgBsize, BLOCK_SIZE, 0, grads.getStream()>>>(
                    ~img,
                    //~momentum,
                    ~grads,
                    Z,
                    Y,
                    X,
                    ZZ,
                    YY,
                    XX,
                    tv_beta,
                    tv_log_eps,//eps of log approximation
                    tv_eps,//eps of l1 norm approximation
                    //grads.getSize());
                    img_size_h);
        } else {
            cuda_kernel_graph_grad<<<imgBFsize, BLOCK_SIZE, 0, grads.getStream()>>>(
                    ~img,
                    ~grads,
                    Y,
                    X,
                    tv_beta,
                    tv_eps,
                    grads.getSize());

        }
        //update momentum
        //cuda_kernel_update_momentum<<<imgBsize, BLOCK_SIZE, 0, grads.getStream()>>>(
        //        ~grads,
        //        ~momentum,
        //        mu,
        //        l_r,
        //        grads.getSize());
        //update momentum
        XFLOAT tk1 = (1. + sqrt(1. + 4.*tk*tk))/2.;
        mu = (tk - 1)/tk1;
        tk = tk1;
        cuda_kernel_soft_threshold<<<imgBsize, BLOCK_SIZE, 0, grads.getStream()>>>(
                ~img,
                //~momentum,
                ~grads,
                //mu,
                l_r,
                tv_alpha,
                eps,
                X,
                Y,
                Z,
                XX,
                YY,
                ZZ,
                //grads.getSize());
                img_size_h);
        //update momentum and soft thresholding
        //cuda_kernel_soft_threshold<<<imgBsize, BLOCK_SIZE, 0, grads.getStream()>>>(
        //        ~img,
        //        ~momentum,
        //        ~grads,
        //        ZZ,
        //        YY,
        //        XX,
        //        mu,
        //        l_r,
        //        tv_alpha,
        //        eps,
        //        momentum.getSize());
        if(m_c % 100 == 0){
            std::cout << m_c << " ";
            //std::cout << sigma_norm[0] << " ";
            RFLOAT img_norm = getSquareSumOnBlock(img);
            RFLOAT resi_norm = getSquareSumOnBlock(grads);
            img_norm = sqrt(img_norm/img_size_h);
            resi_norm = sqrt(resi_norm/img_size_h);
            std::cout <<  resi_norm << " " << img_norm << " " << resi_norm/img_norm << std::endl;
        }
        //l_r *= exp(-0.0025);
    }
    //now copy image to host
    //now set vol_out and wait for stream to complete
    img.streamSync();
    bool hasnan = false;
    for(int i = 0; i < img_size; i++){
        if(isnan(img[i])){
            vol_out.data[i] = 0.;
            hasnan = true;
        }
        else
        {
            //consider upscale image
            vol_out.data[i] = (1. + w/img_size_h)*img[i];
        }
    }
    if(hasnan) std::cout << "WARNING: find nan in reconstruction." << std::endl;
    transformer.clear();
    devBundle->destroyStream();
}

//void cuda_lasso(int tv_iters, RFLOAT l_r, RFLOAT mu, RFLOAT tv_alpha, RFLOAT tv_beta, RFLOAT eps, MultidimArray<Complex> &Fconv,
//        MultidimArray<RFLOAT> &Fweight, MultidimArray<RFLOAT> &vol_out, MlDeviceBundle *devBundle, int data_dim, RFLOAT normfft, bool do_nag){
//    //normfft = max(normfft, 1.);
//    cudaSetDevice(devBundle->device_id);
//    devBundle->setStream();
//    std::cout <<" Device: " << devBundle->device_id <<", " << data_dim;
//    std::cout << " " << vol_out.xdim << ", " << vol_out.ydim << ", " << vol_out.zdim << std::endl;
//    int img_size = vol_out.nzyxdim;
//    CudaGlobalPtr<XFLOAT, false> img(img_size, devBundle->stream);
//    CudaGlobalPtr<XFLOAT, false> yob(img_size, devBundle->stream);
//    CudaGlobalPtr<XFLOAT, false> weight(Fweight.nzyxdim, devBundle->stream);
//    CudaGlobalPtr<XFLOAT, false> grads(img_size, devBundle->stream);
//    CudaGlobalPtr<XFLOAT, false> momentum(img_size, devBundle->stream);
//    CudaFFTT<false> transformer(devBundle->stream, NULL, data_dim);
//    transformer.setSize(vol_out.xdim, vol_out.ydim, vol_out.zdim);
//    img.device_alloc();
//    yob.device_alloc();
//    std::cout << ", " << img_size << ", " << Fconv.getSize()<< std::endl;
//    for(int i = 0; i < Fconv.nzyxdim; i++){
//        //img[i] = 0.f;
//        transformer.fouriers[i].x = Fconv.data[i].real;
//        transformer.fouriers[i].y = Fconv.data[i].imag;
//        //yob[i] = Fconv.data[i];
//        //std::cout << yob[i] << std::endl;
//    }
//    yob.cp_to_device();
//    if(do_nag) {
//        for(int i = 0; i < img_size; i++){
//            img[i] = vol_out.data[i];
//        }
//        img.cp_to_device();
//    } else 
//        img.device_init(0.f);
//    transformer.fouriers.cp_to_device();
//    transformer.fouriers.streamSync();
//    transformer.backward(yob);
//    //img.cp_to_device();
//    yob.streamSync();
//    img.streamSync();
//    //yob.dump_device_to_file("yob"+devBundle->device_id);
//    RFLOAT yob_norm = getSquareSumOnBlock(yob);
//    yob_norm = sqrt(yob_norm/img_size);
//    //img.streamSync();
//    grads.device_alloc();
//    momentum.device_alloc();
//    momentum.device_init(0.f);
//    //move Fweight from host to device
//    weight.device_alloc();
//    for(int i = 0; i < Fweight.nzyxdim; i++){
//        weight[i] = Fweight.data[i];
//    }
//    weight.cp_to_device();
//    weight.streamSync();
//    RFLOAT tv_eps = 0.2;
//    if(normfft > 1. && data_dim == 2)
//        l_r /= normfft;
//    int FBsize = (int) ceilf((float)transformer.fouriers.getSize()/(float)BLOCK_SIZE);
//    int imgBsize = (int) ceilf((float)img.getSize()/(float)BLOCK_SIZE);
//    cuda_kernel_multi<<<imgBsize, BLOCK_SIZE, 0, yob.getStream()>>>(
//                ~yob,
//                (XFLOAT)1/((XFLOAT)normfft),
//                yob.getSize());
//    yob.streamSync();
//    RFLOAT weight_norm = getSquareSumOnBlock(weight);
//    weight_norm = sqrt(weight_norm/weight.getSize());
//    //std::cout << "device: " << devBundle->device_id << " Mout: " << weight_norm << std::endl;
//    l_r *= 1./(weight_norm);
//    std::cout << "start optimizing " << "yob: " << yob_norm << ", " << "weight: " << weight_norm << ", l_r: " << l_r << ", " << std::endl;
//    tv_alpha *= yob_norm;
//    tv_beta *= yob_norm;
//    std::cout << "tv_alpha: " << tv_alpha << ", tv_beta: " << tv_beta << std::endl;
//    for(int m_c = 0; m_c <= tv_iters; m_c++){
//        //forward transform img
//        img.cp_on_device(transformer.reals.d_ptr);
//        transformer.reals.streamSync();
//        transformer.forward();
//        transformer.fouriers.streamSync();
//        //multiply with weight and normalization factor
//        
//        cuda_kernel_complex_multi<<<FBsize, BLOCK_SIZE, 0, transformer.fouriers.getStream()>>>(
//                (XFLOAT*)~transformer.fouriers,
//                ~weight,
//                (XFLOAT)1/((XFLOAT)transformer.reals.getSize()),
//                transformer.fouriers.getSize());
//        transformer.fouriers.streamSync();
//        //inverse transform and put the transformation on grads
//        transformer.backward(grads);
//        grads.streamSync();
//        //transformer.backward();
//        //transformer.reals.cp_on_device(grads.d_ptr);
//        //substract Mout
//        cuda_kernel_substract<<<imgBsize, BLOCK_SIZE, 0, grads.getStream()>>>(
//                ~grads,
//                ~yob,
//                grads.getSize());
//        grads.streamSync();
//        //get the norm of gradient
//        //XFLOAT grads_norm = getSquareSumOnBlock(grads);
//        //grads_norm = sqrt(grads_norm);
//        //cuda_kernel_multi<<<imgBsize, BLOCK_SIZE, 0, grads.getStream()>>>(
//        //        ~grads,
//        //        (XFLOAT)1/((XFLOAT)grads_norm),
//        //        grads.getSize());
//        //grads.streamSync();
//        //get graph gradient
//        int Z = vol_out.zdim;
//        int Y = vol_out.ydim;
//        int X = vol_out.xdim;
//        if(data_dim == 3){
//            cuda_kernel_graph_grad<<<imgBsize, BLOCK_SIZE, 0, grads.getStream()>>>(
//                    ~img,
//                    ~grads,
//                    Z,
//                    Y,
//                    X,
//                    tv_beta,
//                    eps,//eps of log approximation
//                    tv_eps,//eps of l1 norm approximation
//                    grads.getSize());
//        } else {
//            cuda_kernel_graph_grad<<<imgBsize, BLOCK_SIZE, 0, grads.getStream()>>>(
//                    ~img,
//                    ~grads,
//                    Y,
//                    X,
//                    tv_beta,
//                    tv_eps,
//                    grads.getSize());
//
//        }
//        grads.streamSync();
//        //update momentum
//        //cuda_kernel_update_momentum<<<imgBsize, BLOCK_SIZE, 0, grads.getStream()>>>(
//        //        ~grads,
//        //        ~momentum,
//        //        mu,
//        //        l_r,
//        //        grads.getSize());
//        //grads.streamSync();
//        //update momentum and soft thresholding
//        cuda_kernel_soft_threshold<<<imgBsize, BLOCK_SIZE, 0, grads.getStream()>>>(
//                ~img,
//                ~momentum,
//                ~grads,
//                mu,
//                l_r,
//                tv_alpha,
//                eps,
//                grads.getSize());
//        img.streamSync();
//        if(m_c % 20 == 0){
//            std::cout << m_c << " ";
//            RFLOAT grads_norm = getSquareSumOnBlock(img);
//            RFLOAT resi_norm = getSquareSumOnBlock(grads);
//            grads_norm = sqrt(grads_norm/img_size);
//            resi_norm = sqrt(resi_norm/img_size);
//            std::cout <<  " " << resi_norm << " " << grads_norm << " " << resi_norm/grads_norm << std::endl;
//        }
//        l_r *= exp(-0.005);
//    }
//    //now copy image to host
//    img.cp_to_host();
//    //now set vol_out
//    bool hasnan = false;
//    for(int i = 0; i < img_size; i++){
//        if(isnan(img[i])){
//            vol_out.data[i] = 0.;
//            hasnan = true;
//        }
//        else
//            vol_out.data[i] = img[i];
//    }
//    if(hasnan) std::cout << "WARNING: find nan in reconstruction." << std::endl;
//    transformer.clear();
//    img.free();
//    yob.free();
//    weight.free();
//    grads.free();
//    momentum.free();
//    //cudaDeviceReset();
//}
